from src.env.env import AntennaEnv
from src.utils.logger import get_logger
from src.utils.get_config import get_agent_config, get_system_config
from src.utils.generators import RANDOM_SEED, set_global_seed
from src.agents.dqn import DqnAgent
import numpy as np
from src.env.state_models import EnvironmentState
from datetime import datetime
import os
from pathlib import Path
import pandas as pd

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()

ENVIRONMENT_SHAPE = EnvironmentState.STATE_DIM

# Define paths
BASE_DIR = Path.resolve(Path(__file__)).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

DMAX = system_config["satellite"]["d_max"]
SUFFIX = f"dmax_{DMAX}"  # To distinguish different system configurations


def main():
    logger.info(f"Initializing the environment with random seed {RANDOM_SEED}")
    set_global_seed(RANDOM_SEED)
    env = AntennaEnv()

    logger.info("Initializing the agent")
    agent = DqnAgent()
    steps = 0  # Count steps to decay epsilon and train the agent
    decay_interval = agent_config["train"]["epsilon"]["decay_interval"]

    aois = []  # AoI
    dropped_ratios = []

    logger.info("Training the agent")
    N_EPOCHS = agent_config["train"]["epochs"]
    TRAIN_INTERVAL = agent_config["train"].get("train_interval", 1)
    N_EPOCHS_FREEZE = agent_config["train"].get("epochs_freeze", 0)

    for i in range(1, N_EPOCHS + 1):
        logger.info(f"\n\n---Episode {i} begins---")
        logger.info(f"Epsilon: {agent.epsilon:.3}")
        env.reset()
        episode_done = False
        while not episode_done:
            env.handle_events()
            # Push all complete experiences into ReplayBuffer
            experiences = env.buffer.get_all_complete_experiences()
            for experience in experiences:
                state = experience["state"]
                action = experience["action"]
                reward = experience["reward"]
                next_state = experience["next_state"]
                info = experience.get("info")
                done = True if info else False  # Packet is done if info is not None
                agent.remember(state, action, reward, next_state, done)
                # Store samples only when in the freeze period
                if i >= N_EPOCHS - N_EPOCHS_FREEZE:
                    agent.store_sample(state, action, reward, info)

            # Train the agent only at specified intervals
            if i < N_EPOCHS - N_EPOCHS_FREEZE and i % TRAIN_INTERVAL == 0:
                agent.learn()
            action = agent.choose_action(env.state)
            episode_done, episode_info = env.step(action)

            # Epsilon decay
            steps = (steps + 1) % decay_interval
            if steps == 0:
                agent.decay_epsilon()

        logger.info(f"Average AoI of episode {i}: {episode_info['average_aoi']:.3f}")
        aois.append(episode_info["average_aoi"])
        logger.info(f"Dropped ratio of episode {i}: {episode_info['dropped_ratio']:.3f}")
        dropped_ratios.append(episode_info["dropped_ratio"])
        logger.info(f"---Episode {i} ends---")

    logger.info(f"AoI: {[aoi.item() for aoi in np.round(aois, 4)]}")
    logger.info(f"Dropped ratios: {[r.item() for r in np.round(dropped_ratios, 4)]}")
    logger.info("Saving the results")

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = f"{MODEL_DIR}/dqn_baseline_{SUFFIX}.pth"
    agent.save_model(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save the results
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_PATH = f"{DATA_DIR}/dqn_results_{SUFFIX}.csv"
    df = pd.DataFrame(
        {
            "average_aoi": [aoi.item() for aoi in np.round(aois, 4)],
            "dropped_ratio": [r.item() for r in np.round(dropped_ratios, 4)],
        }
    )
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Results saved to {DATA_PATH}")

    # Save data samples for LLM prompt
    SAMPLE_PATH = f"{DATA_DIR}/dqn_data_samples_{SUFFIX}.csv"
    agent.write_samples(SAMPLE_PATH)
    logger.info(f"Data samples saved to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
