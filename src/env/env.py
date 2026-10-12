import numpy as np
from src.utils.generators import (
    generate_bernoulli,
    generate_choice,
    generate_uniform,
)
from src.utils.get_config import get_system_config, get_agent_config
from src.env.env_classes import *
from src.utils.logger import get_logger
from src.env.env_classes import Waveguide

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()


class AntennaEnv:
    """
    Custom Gym environment for satellite operations.
    """

    def __init__(self):
        # Numerically harden config values (PyYAML may parse some as strings)
        self.tau = float(system_config["tau"])
        self.carrier_freq = float(system_config["fc"])
        self.bandwidth = float(system_config["B"])
        # Device tx power and noise are given in dBm -> convert to watts
        self.power = 10 ** (float(system_config["P"]) / 10) / 1000.0
        self.noise = 10 ** (float(system_config["sigma2"]) / 10) / 1000.0

        # Init the waveguide
        self.n_pinches = system_config["M"]
        l_wg = system_config["L"]
        h_wg = system_config["h"]
        # Switching overhead is normalized: tau_sw = rho_sw * tau (seconds)
        switch_cost = system_config["rho_sw"] * self.tau
        self.waveguide = Waveguide(self.n_pinches, l_wg, h_wg, switch_cost)

        # Init devices
        self.n_devices = system_config["K"]
        self.generation_prob = system_config["lambda"]
        service_width = system_config["W"]/2
        x_devices = generate_uniform(0, l_wg, self.n_devices)
        y_devices = generate_uniform(-service_width, service_width, self.n_devices)
        z_devices = np.zeros(self.n_devices)
        self.device_locations = np.stack([x_devices, y_devices, z_devices], axis=0)
        
        # State variables
        self.device_buffers = np.zeros(self.n_devices)  # bk(t) - post-arrival buffer state
        self.buffered_packet_ages = np.zeros(self.n_devices)  # δk(t) - buffered packet age
        self.bs_aois = np.zeros(self.n_devices)  # ∆k(t) - BS AoI

        # Total cost incurred
        self.total_cost = 0

        # Set time
        self.t_slot = 0
        self.t_sim = system_config["T_sim"]

    def reset(self):
        # Reset pinch configuration, AoIs, buffers, and total cost only
        self.waveguide.configure_pinch(-1)
        self.buffered_packet_ages = np.zeros(self.n_devices)
        self.device_buffers = np.zeros(self.n_devices)
        self.bs_aois = np.zeros(self.n_devices)
        self.total_cost = 0
        self.t_slot = 0
        # Apply slot-1 arrivals so the state observed right after reset is post-arrival
        self.handle_events()

    def _action_to_index(self, action):
        k, m = action
        return k * self.n_pinches + m

    def _index_to_action(self, index):
        k = index // self.n_pinches
        m = index % self.n_pinches
        return k, m

    def handle_events(self):
        """
        Run the packet-generation events advancing time to slot t.

        State s(t) is defined AFTER packet generation:
        - Ak(t) ~ Bernoulli(λ) for each device
        - bk(t) and δk(t) are updated post-arrival
        (δk = 0 for new arrivals, BS AoI is untouched)

        With the carry-forward design this is called at the end of step() (for
        the next slot) and inside reset(), so the state observed when the agent
        acts is always post-arrival.
        """
        # Start the new time slot
        self.t_slot += 1

        # Packet generation step: Ak(t) ~ Bernoulli(λ)
        new_arrivals = generate_bernoulli(self.generation_prob, self.n_devices)

        # Update buffered packet age (δk(t)) - reset to 0 for new arrivals
        self.buffered_packet_ages = np.where(new_arrivals == 1, 0, self.buffered_packet_ages)

        # Update device buffers (bk(t)) - set to 1 for new arrivals
        self.device_buffers = np.where(new_arrivals == 1, 1, self.device_buffers)

    def step(self, action):
        """
        Execute one time slot.

        Arrivals are applied in a carry-forward fashion: reset() and the tail
        of step() call handle_events(), so the observed/returned states are
        always post-arrival for the corresponding slot.
        """
        # Store previous states for updates
        prev_bs_aois = self.bs_aois.copy()
        prev_device_buffers = self.device_buffers.copy()
        prev_buffered_packet_ages = self.buffered_packet_ages.copy()

        # Bypass the action when every device buffer is empty.
        # If the PASS point is not configured yet (cur_pinch == -1), just pick a
        # random pinch to (re)configure; otherwise do nothing (no transmission).
        bypassed = int(np.all(self.device_buffers == 0))
        if bypassed:
            if self.waveguide.cur_pinch == -1:
                self.waveguide.configure_pinch(int(generate_choice(self.n_pinches)))
            k = -1
            m = self.waveguide.cur_pinch
        else:
            # Perform action
            k, m = self._index_to_action(action)

            # PASS reconfiguration: calculate switching time
            t_switch = self.waveguide.switch_pinch(m)
        
            # Calculate transmission time: τtx = τ - τsw
            tau = self.tau
            tau_tx = tau - t_switch

        # Initialize success indicator
        Yk = 0

        # Check if transmission is possible and device has packet
        if not bypassed and tau_tx > 0 and self.device_buffers[k] == 1:
            # Calculate SNR and success probability for device k and PASS point m
            device_loc = self.device_locations[:, k]
            pinch_loc = self.waveguide.pinches[:, m]
            
            # Free-space distance
            rk_m = np.linalg.norm(device_loc - pinch_loc)
            
            # Guided-wave distance (x-coordinate of pinch)
            l_m = pinch_loc[0]
            
            # Calculate power gain: Gk,m = β0 * (r0/rk,m)^α * exp(-αwg * l_m)
            fc = self.carrier_freq
            alpha = system_config["alpha"]
            alpha_wg = system_config["alpha_wg"]
            c = 3e8  # Speed of light
            r0 = 1.0
            beta0 = (c / (4 * np.pi * fc * r0)) ** 2
            power_gain = beta0 * (r0 / rk_m) ** alpha * np.exp(-alpha_wg * l_m)
            
            # Calculate SNR: γ = P * G / σ²
            P_linear = self.power
            sigma2 = self.noise
            snr = (P_linear * power_gain) / sigma2
            
            # Calculate SNR threshold: γth = 2^(D/(B*τtx)) - 1
            D = system_config["D"]
            B = self.bandwidth
            snr_threshold = 2 ** (D / (B * tau_tx)) - 1
            
            # Success probability under Rayleigh fading: psuc = exp(-γth / γ)
            psuc = np.exp(-snr_threshold / snr)
            
            # Determine if transmission succeeds
            Yk = generate_bernoulli(psuc)
        
        # Update BS AoI: ∆k(t+1) = ∆k(t) + 1 - Yk(t)[∆k(t) - δk(t)]
        for i in range(self.n_devices):
            if i == k and Yk == 1:
                # Successful delivery: AoI becomes the freshness of delivered packet
                freshness_gain = prev_bs_aois[i] - prev_buffered_packet_ages[i]
                self.bs_aois[i] = prev_bs_aois[i] + 1 - freshness_gain
            else:
                # Failed delivery or not scheduled: AoI increases by 1
                self.bs_aois[i] = prev_bs_aois[i] + 1
        
        # Update buffer: bk(t+1) = bk(t)[1 - Yk(t)], a packet is removed once
        # its transmission succeeds (arrivals were already applied in handle_events)
        for i in range(self.n_devices):
            if i == k and Yk == 1 and prev_device_buffers[i] == 1:
                self.device_buffers[i] = 0
        
        # Update buffered packet age: δk(t+1)
        for i in range(self.n_devices):
            if i == k and Yk == 1 and prev_device_buffers[i] == 1:
                # Packet delivered successfully: buffer empty -> age 0
                self.buffered_packet_ages[i] = 0
            elif prev_device_buffers[i] == 1:
                # Packet still buffered, not delivered: age increases by 1
                self.buffered_packet_ages[i] = prev_buffered_packet_ages[i] + 1
            else:
                # No packet in buffer: age stays 0
                self.buffered_packet_ages[i] = 0
        
        # Calculate cost based on weighted AoI and violation penalty
        wk = 1.0 / self.n_devices  # Uniform weight
        eta = system_config["eta"]
        
        # Weighted AoI cost
        cost_aoi = wk * np.sum(self.bs_aois)

        # Soft-margin violation penalty, normalized to a FRACTION of the AoI cost
        # (eta ~ 0.1..0.5). It measures how far each device's AoI exceeds the *mean*
        # (i.e. how far it lags the pack), so it actively discourages the
        # unbalanced/starvation failure mode. Continuous hinge (NOT a hard clip)
        # and, because sum(excess) <= sum(AoI), bounded by eta * cost_aoi.
        mean_aoi = float(np.mean(self.bs_aois))
        excess = np.maximum(self.bs_aois - mean_aoi, 0.0)
        aoi_total = float(np.sum(self.bs_aois))
        excess_fraction = float(np.sum(excess)) / aoi_total if aoi_total > 0 else 0.0
        cost_violation = eta * cost_aoi * excess_fraction

        # Total cost
        cost = cost_aoi + cost_violation
        self.total_cost += cost
        
        # Reward = -cost
        reward = -cost
        
        # Done flag
        done = self.t_sim == self.t_slot
        
        # Apply next slot's arrivals (carry-forward) so the returned next_state
        # is the post-arrival state the agent will observe when it acts next.
        if not done:
            self.handle_events()
        next_state = self._get_state()
        
        info = {
            "total_cost": self.total_cost,
            "bypassed": bool(bypassed),
        }    
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Get the current state for the agent.
        State s(t) = {∆k(t), bk(t), δk(t), m-(t)}
        """
        state = np.concatenate([
            self.bs_aois,                    # ∆k(t) - BS AoI for each device
            self.device_buffers,              # bk(t) - buffer state for each device
            self.buffered_packet_ages,        # δk(t) - buffered packet age for each device
            [self.waveguide.cur_pinch]        # m-(t) - inherited PASS configuration
        ])
        return state
