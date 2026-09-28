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
        self.waveguide = Waveguide(self.n_pinches, l_wg, h_wg) 

        # Init devices
        self.n_devices = system_config["K"]
        self.generation_prob = system_config["lambda"]
        service_width = system_config["W"]/2
        x_devices = np.random.uniform(0, l_wg, self.n_devices)
        y_devices = np.random.uniform(-service_width, service_width, self.n_devices)
        z_devices = np.zeros(self.n_devices)
        self.device_locations = np.stack([x_devices, y_devices, z_devices], axis=0)
        self.device_buffers = np.zeros(self.n_devices)
        self.device_aois = np.zeros(self.n_devices)
        self.bs_aois = np.zeros(self.n_devices)

        # Total cost incurred
        self.total_cost = 0

    def reset(self):
        # Reset pinch configuration, AoIs, buffers, and total cost only
        self.waveguide.configure_pinch(0)
        self.device_aois = np.zeros(self.n_devices)
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
        An action (k, m) is a tuple of the chosen device and the chosen pinch.
        """
        # Packet generation step
        rand_probs = np.random.uniform(0, 1, self.n_devices)
        mask = rand_probs < self.generation_prob

        # Update devices AoI
        self.device_aois[mask] = 0   # AoI of newly generated packets is 0

        # Update device buffers
        self.device_buffers[mask] = 1  # Buffer of newly generated packets is 1
