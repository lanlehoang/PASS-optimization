import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config, get_agent_config
from src.env.env_classes import *
from src.utils.logger import get_logger
from src.env.state_models import NeighbourState, EnvironmentState
from src.env.env_classes import Waveguide

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()


class AntennaEnv:
    """
    Custom Gym environment for satellite operations.
    """

    def __init__(self):
        # Init the waveguide
        self.n_pinches = system_config["M"]
        l_wg = system_config["L"]
        h_wg = system_config["h"]
        switch_cost = system_config["rho_sw"]
        self.waveguide = Waveguide(self.n_pinches, l_wg, h_wg, switch_cost) 

        # Init devices
        self.n_devices = system_config["K"]
        self.generation_prob = system_config["lambda"]
        service_width = system_config["W"]/2
        x_devices = np.random.uniform(0, l_wg, self.n_devices)
        y_devices = np.random.uniform(-service_width, service_width, self.n_devices)
        z_devices = np.zeros(self.n_devices)
        self.device_locations = np.stack([x_devices, y_devices, z_devices], axis=0)
        
        # State variables
        self.device_buffers = np.zeros(self.n_devices)  # bk(t) - post-arrival buffer state
        self.buffered_packet_ages = np.zeros(self.n_devices)  # δk(t) - buffered packet age
        self.bs_aois = np.zeros(self.n_devices)  # ∆k(t) - BS AoI

        # Total cost incurred
        self.total_cost = 0

    def reset(self):
        # Reset pinch configuration, AoIs, buffers, and total cost only
        self.waveguide.configure_pinch(0)
        self.buffered_packet_ages = np.zeros(self.n_devices)
        self.device_buffers = np.zeros(self.n_devices)
        self.bs_aois = np.zeros(self.n_devices)
        self.total_cost = 0

    def _action_to_index(self, action):
        k, m = action
        return k * self.n_pinches + m

    def _index_to_action(self, index):
        k = index // self.n_pinches
        m = index % self.n_pinches
        return k, m

    def step(self, action):
        """
        Execute one time slot.
        
        State s(t) is defined AFTER packet generation at the beginning of slot t:
        - ∆k(t): BS AoI (from previous successful delivery) - does NOT reset on new arrival
        - bk(t): post-arrival buffer state (after Ak(t) generation)
        - δk(t): buffered packet age (after generation, δk=0 for new arrivals)
        - m-(t): inherited PASS configuration
        
        Events in one time slot:
        1. Packet generation (Ak(t) ~ Bernoulli(λ))
        2. State s(t) = {∆k(t), bk(t), δk(t), m-(t)} ← agent observes THIS
        3. Select device k and PASS point m
        4. PASS reconfiguration (if needed)
        5. Transmission attempt
        6. Update states: AoI, buffer, packet age, PASS config
        7. Calculate cost and reward
        """
        # Store previous states for updates
        prev_bs_aois = self.bs_aois.copy()
        prev_device_buffers = self.device_buffers.copy()
        prev_buffered_packet_ages = self.buffered_packet_ages.copy()
        
        # Packet generation step: Ak(t) ~ Bernoulli(λ)
        rand_probs = np.random.uniform(0, 1, self.n_devices)
        new_arrivals = (rand_probs < self.generation_prob).astype(int)

        # Update buffered packet age (δk(t)) - reset to 0 for new arrivals
        self.buffered_packet_ages = np.where(new_arrivals == 1, 0, self.buffered_packet_ages)

        # Update device buffers (bk(t)) - set to 1 for new arrivals
        self.device_buffers = np.where(new_arrivals == 1, 1, self.device_buffers)

        # Perform action
        k, m = self._index_to_action(action)

        # PASS reconfiguration: calculate switching time
        t_switch = self.waveguide.switch_pinch(m)
        
        # Calculate transmission time: τtx = τ - τsw
        tau = system_config["tau"]
        tau_tx = tau - t_switch
        
        # Initialize success indicator
        Yk = 0
        
        # Check if transmission is possible and device has packet
        if tau_tx > 0 and self.device_buffers[k] == 1:
            # Calculate SNR and success probability for device k and PASS point m
            device_loc = self.device_locations[:, k]
            pinch_loc = self.waveguide.pinches[:, m]
            
            # Free-space distance
            rk_m = np.linalg.norm(device_loc - pinch_loc)
            
            # Guided-wave distance (x-coordinate of pinch)
            l_m = pinch_loc[0]
            
            # Calculate power gain: Gk,m = β0 * (r0/rk,m)^α * exp(-αwg * l_m)
            fc = system_config["fc"]
            alpha = system_config["alpha"]
            alpha_wg = system_config["alpha_wg"]
            c = 3e8  # Speed of light
            r0 = 1.0
            beta0 = (c / (4 * np.pi * fc * r0)) ** 2
            power_gain = beta0 * (r0 / rk_m) ** alpha * np.exp(-alpha_wg * l_m)
            
            # Calculate SNR: γ = P * G / σ²
            P_linear = system_config["P"]
            sigma2 = system_config["sigma2"]
            snr = (P_linear * power_gain) / sigma2
            
            # Calculate SNR threshold: γth = 2^(D/(B*τtx)) - 1
            D = system_config["D"]
            B = system_config["B"]
            snr_threshold = 2 ** (D / (B * tau_tx)) - 1
            
            # Success probability under Rayleigh fading: psuc = exp(-γth / γ)
            psuc = np.exp(-snr_threshold / snr)
            
            # Determine if transmission succeeds
            Yk = 1 if np.random.uniform(0, 1) < psuc else 0
        
        # Update BS AoI: ∆k(t+1) = ∆k(t) + 1 - Yk(t)[∆k(t) - δk(t)]
        for i in range(self.n_devices):
            if i == k and Yk == 1:
                # Successful delivery: AoI becomes the freshness of delivered packet
                freshness_gain = prev_bs_aois[i] - prev_buffered_packet_ages[i]
                self.bs_aois[i] = prev_bs_aois[i] + 1 - freshness_gain
            else:
                # Failed delivery or not scheduled: AoI increases by 1
                self.bs_aois[i] = prev_bs_aois[i] + 1
        
        # Update buffer: bk(t+1) = Ak(t+1) + [1 - Ak(t+1)]bk(t)[1 - Yk(t)]
        for i in range(self.n_devices):
            if new_arrivals[i] == 1:
                self.device_buffers[i] = 1
            elif prev_device_buffers[i] == 1 and Yk == 1 and i == k:
                self.device_buffers[i] = 0
        
        # Update buffered packet age: δk(t+1) = [1 - Ak(t+1)]bk(t)[1 - Yk(t)][δk(t) + 1]
        for i in range(self.n_devices):
            if new_arrivals[i] == 0 and prev_device_buffers[i] == 1 and not (Yk == 1 and i == k):
                self.buffered_packet_ages[i] = prev_buffered_packet_ages[i] + 1
            elif new_arrivals[i] == 0 and prev_device_buffers[i] == 0:
                self.buffered_packet_ages[i] = 0
        
        # Calculate cost based on weighted AoI and violation penalty
        wk = 1.0 / self.n_devices  # Uniform weight
        delta_max = system_config["delta_max"]
        eta = system_config["eta"]
        
        # Weighted AoI cost
        cost_aoi = wk * np.sum(self.bs_aois)
        
        # Violation penalty
        violations = np.sum(self.bs_aois > delta_max)
        cost_violation = eta * violations
        
        # Total cost
        cost = cost_aoi + cost_violation
        self.total_cost += cost
        
        # Reward = -cost
        reward = -cost
        
        # Get next state
        next_state = self._get_state()
        
        # Done flag (can be set based on episode length or other criteria)
        done = False        
        return next_state, reward, done, {}
    
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
