import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.utils.get_config import get_agent_config, get_system_config
from src.utils.logger import get_logger
from src.env.state_models import EnvironmentState
from src.utils.generators import generate_choice, generate_random

agent_config = get_agent_config()
system_config = get_system_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT_SHAPE = EnvironmentState.STATE_DIM
N_DEVICES, N_PINCHES = system_config["K"], system_config["M"]
N_ACTIONS = N_DEVICES * N_PINCHES

logger = get_logger(__name__)


class QNetwork(nn.Module):
    def __init__(self, fc1_dims, fc2_dims, fc3_dims, dropout, lr):
        super().__init__()
        # Simple 3-hidden-layer MLP: state -> Q-value for every flat action index
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
        """x: (B, state_dim) -> raw Q-values (B, N_ACTIONS), no masking."""
        return self.fc(x)

    def invalid_mask(self, x):
        """(B, state_dim) -> bool (B, N_ACTIONS), True = infeasible.

        State layout: [AoI (K), buffer occupancy b (K), buffered age (K), pinch block].
        Action index = k * N_PINCHES + m, so device k owns [k*M, (k+1)*M).
        """
        empty = x[:, N_DEVICES: 2 * N_DEVICES] == 0
        return empty.repeat_interleave(N_PINCHES, dim=1)

    def predict(self, states):
        """Masked Q-values; invalid actions are -inf."""
        squeeze = states.dim() == 1
        if squeeze:
            states = states.unsqueeze(0)
        q = self.forward(states).masked_fill(self.invalid_mask(states), float("-inf"))
        return q.squeeze(0) if squeeze else q

    def fit(self, states, actions, targets, epochs=1):
        """
        Train on (s, a) -> target Q(s, a).
        Args:
            states:  (B, S)
            actions: (B,) Long flat action indices (always valid, so raw Q is used)
            targets: (B, 1) TD targets
        """
        avg_loss = 0.0
        self.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()
            q_pred = self.forward(states).gather(1, actions.unsqueeze(1))  # (B, 1)
            loss = self.loss_fn(q_pred, targets)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

        self.eval()  # keep dropout off for acting / target computation
        return avg_loss / epochs


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
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
        self.reward_memory[index] = float(reward)  # reward = NEGATIVE cost
        self.terminal_memory[index] = 1.0 - float(int(done))  # stored as not_done
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
        self.action_space = np.arange(N_ACTIONS)
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
        }
        self.max_examples_per_type = 100
        self.lr = agent_config["train"]["lr"]

        # NOTE: adjust these config keys to match your YAML
        net_kwargs = dict(
            fc1_dims=agent_config["dqn"]["fc1_dims"],
            fc2_dims=agent_config["dqn"]["fc2_dims"],
            fc3_dims=agent_config["dqn"]["fc3_dims"],
            dropout=agent_config["dqn"]["dropout"],
            lr=self.lr,
        )
        self.q_eval = QNetwork(**net_kwargs).to(DEVICE)
        self.q_target = QNetwork(**net_kwargs).to(DEVICE)

        self.q_target.load_state_dict(self.q_eval.state_dict())
        # Dropout off by default; fit() switches q_eval to train mode and back
        self.q_eval.eval()
        self.q_target.eval()

        self.learn_step_counter = 0
        self.target_update_interval = target_update_interval

    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def _to_tensor(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32, device=DEVICE)
        return state.to(DEVICE, dtype=torch.float32)

    @torch.no_grad()
    def _predict_q_values(self, state):
        return self.q_eval.predict(self._to_tensor(state))

    def _valid_actions(self, state_t):
        """Flat indices of feasible actions (devices with a non-empty buffer)."""
        mask = self.q_eval.invalid_mask(state_t.unsqueeze(0)).squeeze(0)
        return torch.where(~mask)[0].cpu().numpy()

    def choose_action(self, state):
        rand = generate_random()
        state_t = self._to_tensor(state)

        if rand < self.epsilon:
            valid_actions = self._valid_actions(state_t)
            # If nothing is feasible the env should idle instead of calling the agent
            action = int(generate_choice(valid_actions)) if len(valid_actions) > 0 else 0
        else:
            q_values = self._predict_q_values(state_t)
            action = int(torch.argmax(q_values).item())
        return action

    def choose_action_with_offset(self, state, q_offset: np.ndarray):
        """
        Choose action with a Q-value offset computed by the LLM heuristic.
        q_offset: array of shape (N_ACTIONS,) added to Q-values before argmax.
        Masked (-inf) entries stay -inf after the addition.
        """
        rand = generate_random()
        state_t = self._to_tensor(state)

        if rand < self.epsilon:
            valid_actions = self._valid_actions(state_t)
            action = int(generate_choice(valid_actions)) if len(valid_actions) > 0 else 0
        else:
            q_values = self._predict_q_values(state_t).cpu().numpy()
            q_values = q_values + q_offset
            action = int(np.argmax(q_values))
        return action

    def store_sample(self, state, action, reward, info):
        """
        Store events (dropped, arrived, None) as examples for the LLM prompt:
        state, predicted Q-values, action taken, reward received.
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

            # 3. TD target from the target network (masked max over feasible actions)
            with torch.no_grad():
                q_next = self.q_target.predict(new_states).max(dim=1, keepdim=True).values
                # Rows with no feasible action give -inf; bootstrap with 0 instead of NaN/-inf
                q_next = torch.where(torch.isinf(q_next), torch.zeros_like(q_next), q_next)
                q_target = rewards + self.gamma * q_next * not_done

            # 4. Train eval net
            self.q_eval.fit(states, actions_idx, q_target)

            # 5. Target network sync
            self.learn_step_counter = (self.learn_step_counter + 1) % self.target_update_interval
            if self.learn_step_counter == 0:
                self.update_target_network()

    def save_model(self, path):
        torch.save(self.q_eval.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=DEVICE)
        self.q_eval.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)
        self.q_eval.eval()
        self.q_target.eval()
        self.epsilon = self.epsilon_min  # evaluation: minimal exploration

    def write_samples(self, filepath):
        """Write data samples to a CSV file for easier analysis."""
        states, q_values, actions, rewards, info_types = [], [], [], [], []

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