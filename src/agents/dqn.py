import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from src.utils.get_config import get_agent_config, get_system_config
from src.utils.logger import get_logger
from src.env.state_models import EnvironmentState
import pandas as pd
from src.utils.generators import generate_choice, generate_random

agent_config = get_agent_config()
system_config = get_system_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT_SHAPE = EnvironmentState.STATE_DIM
N_DEVICES, N_PINCHES = system_config["K"], system_config["M"]
N_ACTIONS = N_DEVICES * N_PINCHES

logger = get_logger(__name__)


def get_action_range(k):
    return k * N_PINCHES, (k + 1) * N_PINCHES


class QNetwork(nn.Module):
    def __init__(
        self,
        fc1_dims,
        fc2_dims,
        fc3_dims,
        dropout,
        lr,
    ):
        super().__init__()
        # Use a simple 3 hidden layer network
        self.fc = nn.Sequential(
            nn.Linear(ENVIRONMENT_SHAPE, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.LayerNorm(fc3_dims),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(fc3_dims, N_ACTIONS),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x):
        """
        x: (B, state_dim)
        Returns: (B, N_ACTIONS) Q-values
        """
        # Compute raw Q-values
        x = self.fc(x)

        # Get the invalid actions (devices with zero buffer)
        device_buffers = x[:, N_DEVICES: 2 * N_DEVICES]
        empty_buffers = device_buffers == 0  # (B, N_DEVICES)

        # For each device with zero buffer, ban all of its m corresponding actions


        return x

    def predict(self, states):
        """
        Predict Q-values Q(s,a) for all actions a (batched).
        Invalid actions padded with -inf.
        Args:
            states: (B,S) or (S,)
        Returns:
            q_all: (B,n_actions) or (n_actions,)
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        B = states.size(0)
        N = system_config["satellite"]["n_neighbours"]

        # Expand for all possible actions
        actions = torch.arange(N, device=states.device).repeat(B, 1)  # (B,N)
        states_rep = states.unsqueeze(1).repeat(1, N, 1)  # (B,N,S)
        sa_pairs = torch.cat((states_rep, actions.unsqueeze(2).float()), dim=2)  # (B,N,S+1)

        # Flatten to batch
        sa_pairs = sa_pairs.reshape(B * N, -1)
        with torch.no_grad():
            self.eval()
            q_vals = self.forward(sa_pairs).reshape(B, N)  # (B,N)

        # Mask invalid neighbours
        neighbours = states.reshape(B, N, NEIGHBOUR_SHAPE)
        mask = torch.any(neighbours != 0, dim=2)  # (B,N)
        q_vals[~mask] = -float("inf")

        return q_vals.squeeze(0) if B == 1 else q_vals

    def fit(self, states, actions, targets, epochs=1):
        """
        Train the network on (s,a) → target Q(s,a).
        Args:
            states: (B,S)
            actions: (B,) Long indices
            targets: (B,1) TD targets
        """
        avg_loss = 0.0
        self.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()
            # Build (s,a) input
            sa_pairs = torch.cat((states, actions.unsqueeze(1).float()), dim=1)  # (B,S+1)
            q_pred = self.forward(sa_pairs)  # (B,1)
            loss = self.loss_fn(q_pred, targets)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

        return avg_loss / epochs


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        # Input shape of the environment
        self.input_shape = input_shape
        self.state_memory = torch.zeros((self.mem_size, input_shape))
        self.new_state_memory = torch.zeros((self.mem_size, input_shape))
        self.action_memory = torch.zeros((self.mem_size,), dtype=torch.int64)
        self.reward_memory = torch.zeros(self.mem_size)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = torch.as_tensor(state, dtype=torch.float32)
        self.new_state_memory[index] = torch.as_tensor(new_state, dtype=torch.float32)
        self.reward_memory[index] = float(reward)
        self.terminal_memory[index] = 1.0 - float(int(done))
        # Store integer action directly
        self.action_memory[index] = int(action)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = generate_choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class DqnAgent:
    def __init__(self, input_dims=ENVIRONMENT_SHAPE, mem_size=2048, target_update_interval=10):
        logger.info(f"Initializing DQN Agent with device: {DEVICE}")
        n_actions = system_config["satellite"]["n_neighbours"]
        self.action_space = np.arange(n_actions)
        self.gamma = agent_config["train"]["gamma"]
        self.epsilon = agent_config["train"]["epsilon"]["init"]
        self.epsilon_dec = agent_config["train"]["epsilon"]["decay"]
        self.epsilon_min = agent_config["train"]["epsilon"]["min"]
        self.batch_size = agent_config["train"]["batch_size"]
        self.memory = ReplayBuffer(mem_size, input_dims)
        # Data samples for LLM prompt
        self.data_samples = {
            "dropped": [],
            "arrived": [],
            None: [],
        }  # Collect data samples for all 3 cases
        self.max_examples_per_type = 100
        self.lr = agent_config["train"]["lr"]

        self.q_eval = QNetwork(
            ds_input_dims=NEIGHBOUR_SHAPE,
            ds_fc1_dims=agent_config["dqn"]["deepset"]["fc1_dims"],
            ds_fc2_dims=agent_config["dqn"]["deepset"]["fc2_dims"],
            embed_dims=agent_config["dqn"]["deepset"]["embedding_dims"],
            final_fc_dims=agent_config["dqn"]["final_fc_dims"],
            dropout=agent_config["dqn"]["dropout"],
            lr=self.lr,
        ).to(DEVICE)

        self.q_target = QNetwork(
            ds_input_dims=NEIGHBOUR_SHAPE,
            ds_fc1_dims=agent_config["dqn"]["deepset"]["fc1_dims"],
            ds_fc2_dims=agent_config["dqn"]["deepset"]["fc2_dims"],
            embed_dims=agent_config["dqn"]["deepset"]["embedding_dims"],
            final_fc_dims=agent_config["dqn"]["final_fc_dims"],
            dropout=agent_config["dqn"]["dropout"],
            lr=self.lr,
        ).to(DEVICE)

        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.learn_step_counter = 0
        self.target_update_interval = target_update_interval

    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def _predict_q_values(self, state):
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_t = state.to(DEVICE, dtype=torch.float32)
        q_values = self.q_eval.predict(state_t)
        return q_values

    def choose_action(self, state):
        rand = generate_random()
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_t = state.to(DEVICE, dtype=torch.float32)

        if rand < self.epsilon:
            neighbour_states = state_t.cpu().numpy().reshape(-1, NEIGHBOUR_SHAPE)
            valid_actions = np.where(np.any(neighbour_states != 0, axis=1))[0]
            action = int(generate_choice(valid_actions))
        else:
            q_values = self._predict_q_values(state_t)
            action = int(torch.argmax(q_values).item())
        return action

    def choose_action_with_offset(self, state, q_offset: np.ndarray):
        """
        Choose action with Q-value offset computed by LLMs.
        q_offset: float value to add to Q-values before selecting action.
        """
        rand = generate_random()
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            state_t = state.to(DEVICE, dtype=torch.float32)

        if rand < self.epsilon:
            neighbour_states = state_t.cpu().numpy().reshape(-1, NEIGHBOUR_SHAPE)
            valid_actions = np.where(np.any(neighbour_states != 0, axis=1))[0]
            action = int(generate_choice(valid_actions))
        else:
            q_values = self._predict_q_values(state_t).cpu().numpy()
            q_values += q_offset
            action = int(np.argmax(q_values).item())
        return action

    def store_sample(self, state, action, reward, info):
        """
        Store different types of events (dropped, arrived, None) as examples for LLM prompt
        Information to store:
        - state: the environment state when the event occurred
        - q_values: the Q-values predicted at that state (when the agent is freezed)
        - action: the action taken
        - reward: the actual reward received
        """
        if len(self.data_samples[info]) < self.max_examples_per_type:
            q_values = self._predict_q_values(state).cpu().numpy().tolist()
            self.data_samples[info].append(
                {
                    "state": state,
                    "q_values": q_values,
                    "action": action,
                    "reward": reward,
                }
            )

    def learn(self):
        if self.memory.mem_counter >= self.batch_size:
            # 1. Sample batch
            states, actions_idx, rewards, new_states, not_done = self.memory.sample_buffer(self.batch_size)

            # 2. To device
            states = states.to(DEVICE).float()
            new_states = new_states.to(DEVICE).float()
            rewards = rewards.to(DEVICE).float().unsqueeze(1)
            not_done = not_done.to(DEVICE).float().unsqueeze(1)
            actions_idx = actions_idx.to(DEVICE).long()  # (B,)

            # 3. Target network for Q(s',a')
            with torch.no_grad():
                q_next = self.q_target.predict(new_states).max(dim=1, keepdim=True).values

            q_target = rewards + self.gamma * q_next * not_done

            # 4. Train eval net
            _ = self.q_eval.fit(states, actions_idx, q_target)

            # 5. Target network sync
            self.learn_step_counter = (self.learn_step_counter + 1) % self.target_update_interval
            if self.learn_step_counter == 0:
                self.update_target_network()

    def save_model(self, path):
        torch.save(self.q_eval.state_dict(), path)

    def load_model(self, path):
        self.q_eval.load_state_dict(torch.load(path))
        self.q_target.load_state_dict(torch.load(path))
        self.epsilon = self.epsilon_min  # Set epsilon to min for evaluation

    def write_samples(self, filepath):
        """
        Write data samples to an xlsx file for easier analysis.
        """
        states = []
        q_values = []
        actions = []
        rewards = []
        info_types = []

        for info_type, samples in self.data_samples.items():
            for sample in samples:
                states.append(sample["state"])
                q_values.append(sample["q_values"])
                actions.append(sample["action"])
                rewards.append(sample["reward"])
                info_types.append(info_type)

        df = pd.DataFrame(
            {
                "info_type": info_types,
                "state": states,
                "q_values": q_values,
                "action": actions,
                "reward": rewards,
            }
        )
        df.to_csv(filepath, index=False)
