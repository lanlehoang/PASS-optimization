"""
Evaluate all methods at the end
No training involved
"""

import pandas as pd
import numpy as np

from .env.env import AntennaEnv
from .utils.generators import RANDOM_SEED, set_global_seed
from .agents.dqn_heuristic import DqnHeuristicAgent
from .agents.dqn import DqnAgent
from .agents.greedy import GreedyAgent
from .utils.logger import get_logger
from pathlib import Path
import os

logger = get_logger(__name__)

BASE_PATH = Path(__file__).resolve().parent.parent
DQN_MODEL_PATH = BASE_PATH / "models" / "dqn_baseline_20251201.pth"
HEURISTIC_CONFIG_PATH = BASE_PATH / "eoh_results" / "20251201" / "pops_best" / "population_generation_4.json"


class MethodEvaluator:
    def __init__(self, n_episodes: int = 10, output_dir: str = "eval_results"):
        self.n_episodes = n_episodes
        self.output_dir = BASE_PATH / output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _to_csv(self, aois, dropped_ratios, filename):
        df = pd.DataFrame({"average_aoi": aois, "dropped_ratio": dropped_ratios})
        df.to_csv(f"{self.output_dir}/{filename}", index=False)
        logger.info(f"Saved evaluation results to {filename}")

    def evaluate_greedy_nearest(self):
        logger.info("Evaluating Greedy Nearest Method")
        set_global_seed(RANDOM_SEED)
        env = AntennaEnv()
        aois = []
        dropped_ratios = []

        for i in range(1, self.n_episodes + 1):
            episode_done = False
            env.reset()
            while not episode_done:
                env.handle_events()
                action = GreedyAgent.choose_nearest_to_destination(env.state)
                episode_done, episode_info = env.step(action)

            logger.info(f"Average AoI of episode {i}: {episode_info['average_aoi']:.3f}")
            aois.append(episode_info["average_aoi"])
            logger.info(f"Dropped ratio of episode {i}: {episode_info['dropped_ratio']:.3f}")
            dropped_ratios.append(episode_info["dropped_ratio"])

        aoi_mean = np.mean(aois).item()
        dropped_ratio_mean = np.mean(dropped_ratios).item()
        logger.info(f"Mean Average AoI over {self.n_episodes} episodes: {aoi_mean:.3f}")
        logger.info(f"Mean Dropped Ratio over {self.n_episodes} episodes: {dropped_ratio_mean:.3f}")
        self._to_csv(aois, dropped_ratios, "greedy_nearest_eval.csv")

    def evaluate_greedy_expected_time(self):
        logger.info("Evaluating Greedy Expected Time Method")
        set_global_seed(RANDOM_SEED)
        env = AntennaEnv()
        aois = []
        dropped_ratios = []

        for i in range(1, self.n_episodes + 1):
            episode_done = False
            env.reset()
            while not episode_done:
                env.handle_events()
                action = GreedyAgent.choose_best_expected_time(env.state)
                episode_done, episode_info = env.step(action)

            logger.info(f"Average AoI of episode {i}: {episode_info['average_aoi']:.3f}")
            aois.append(episode_info["average_aoi"])
            logger.info(f"Dropped ratio of episode {i}: {episode_info['dropped_ratio']:.3f}")
            dropped_ratios.append(episode_info["dropped_ratio"])

        aoi_mean = np.mean(aois).item()
        dropped_ratio_mean = np.mean(dropped_ratios).item()
        logger.info(f"Mean Average AoI over {self.n_episodes} episodes: {aoi_mean:.3f}")
        logger.info(f"Mean Dropped Ratio over {self.n_episodes} episodes: {dropped_ratio_mean:.3f}")
        self._to_csv(aois, dropped_ratios, "greedy_expected_time_eval.csv")

    def evaluate_dqn(self):
        logger.info("Evaluating DQN Heuristic Method")
        set_global_seed(RANDOM_SEED)
        env = AntennaEnv()

        # Load DQN agent
        agent = DqnAgent()
        agent.load_model(DQN_MODEL_PATH)

        aois = []
        dropped_ratios = []

        for i in range(1, self.n_episodes + 1):
            episode_done = False
            env.reset()
            while not episode_done:
                env.handle_events()
                action = agent.choose_action(env.state)
                episode_done, episode_info = env.step(action)

            logger.info(f"Average AoI of episode {i}: {episode_info['average_aoi']:.3f}")
            aois.append(episode_info["average_aoi"])
            logger.info(f"Dropped ratio of episode {i}: {episode_info['dropped_ratio']:.3f}")
            dropped_ratios.append(episode_info["dropped_ratio"])

        aoi_mean = np.mean(aois).item()
        dropped_ratio_mean = np.mean(dropped_ratios).item()
        logger.info(f"Mean Average AoI over {self.n_episodes} episodes: {aoi_mean:.3f}")
        logger.info(f"Mean Dropped Ratio over {self.n_episodes} episodes: {dropped_ratio_mean:.3f}")
        self._to_csv(aois, dropped_ratios, "dqn_eval.csv")

    def evaluate_dqn_heuristic(self):
        logger.info("Evaluating DQN Heuristic Method")
        set_global_seed(RANDOM_SEED)
        env = AntennaEnv()

        # Load DQN Heuristic agent
        agent = DqnHeuristicAgent(str(DQN_MODEL_PATH), str(HEURISTIC_CONFIG_PATH))

        aois = []
        dropped_ratios = []

        for i in range(1, self.n_episodes + 1):
            episode_done = False
            env.reset()
            while not episode_done:
                env.handle_events()
                action = agent.choose_action(env.state)
                episode_done, episode_info = env.step(action)

            logger.info(f"Average AoI of episode {i}: {episode_info['average_aoi']:.3f}")
            aois.append(episode_info["average_aoi"])
            logger.info(f"Dropped ratio of episode {i}: {episode_info['dropped_ratio']:.3f}")
            dropped_ratios.append(episode_info["dropped_ratio"])

        aoi_mean = np.mean(aois).item()
        dropped_ratio_mean = np.mean(dropped_ratios).item()
        logger.info(f"Mean Average AoI over {self.n_episodes} episodes: {aoi_mean:.3f}")
        logger.info(f"Mean Dropped Ratio over {self.n_episodes} episodes: {dropped_ratio_mean:.3f}")
        self._to_csv(aois, dropped_ratios, "dqn_heuristic_eval.csv")


if __name__ == "__main__":
    eval = MethodEvaluator(n_episodes=10, output_dir="eval_results/20251201")
    eval.evaluate_greedy_nearest()
    eval.evaluate_greedy_expected_time()
    eval.evaluate_dqn()
    eval.evaluate_dqn_heuristic()
