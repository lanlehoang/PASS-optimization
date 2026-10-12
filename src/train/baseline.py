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

DMAX = system_config["delta_max"]
SUFFIX = f"dmax_{DMAX}"  # To distinguish different system configurations


def main():
    logger.info(f"Initializing the environment with random seed {RANDOM_SEED}")
    set_global_seed(RANDOM_SEED)
    env = AntennaEnv()

    logger.info("Initializing the agent")
    agent = DqnAgent()
    steps = 0  # Count steps to decay epsilon and train the agent
    decay_interval = agent_config["train"]["epsilon"]["decay_interval"]

    aois = []      # average AoI per episode
    costs = []     # total (undiscounted) cost per episode

    logger.info("Training the agent")
    N_EPOCHS = agent_config["train"]["epochs"]
    N_EPOCHS_FREEZE = agent_config["train"].get("epochs_freeze", 0)
    T_SIM = system_config["T_sim"]
    BATCH_SIZE = agent_config["train"]["batch_size"]
    LEARN_STEPS = agent_config["train"].get("train_steps", 1)

    # Save the best model found during training (the freezer may otherwise leave
    # a drifted late-episode copy as the final checkpoint).
    best_aoi = float("inf")

    for i in range(1, N_EPOCHS + 1):
        logger.info(f"\n\n---Episode {i} begins---")
        logger.info(f"Epsilon: {agent.epsilon:.3}")
        env.reset()
        state = env._get_state()  # post-arrival state of slot 1
        episode_done = False
        total_aoi = 0.0
        while not episode_done:
            # 1. Choose an action (state is post-arrival)
            action = agent.choose_action(state)

            # 2. Step the environment (carry-forward: also applies next slot's arrivals)
            next_state, reward, done, info = env.step(action)

            # 3. Remember the transition
            agent.remember(state, action, reward, next_state, done)

            # 4. Learn every episode once the replay buffer is warm (and not frozen)
            if i < N_EPOCHS - N_EPOCHS_FREEZE and agent.memory.mem_counter >= BATCH_SIZE:
                agent.learn(steps=LEARN_STEPS)

            # 5. Decay epsilon every decay_interval steps
            steps = (steps + 1) % decay_interval
            if steps == 0:
                agent.decay_epsilon()

            total_aoi += float(np.mean(env.bs_aois))
            episode_done = done
            state = next_state

        avg_aoi = total_aoi / T_SIM
        # Average cost per slot (normalized, not the raw cumulative sum over the
        # T_SIM slots which scales with episode length and "looks huge").
        avg_cost_per_slot = float(env.total_cost) / T_SIM
        logger.info(
            f"Average AoI of episode {i}: {avg_aoi:.3f} | avg cost/slot: {avg_cost_per_slot:.3f}"
        )
        aois.append(avg_aoi)
        costs.append(avg_cost_per_slot)
        logger.info(f"---Episode {i} ends---")

        # Keep the best-performing checkpoint so a drifted late-freeze copy is
        # not saved as the final model.
        if avg_aoi < best_aoi:
            best_aoi = avg_aoi
            os.makedirs(MODEL_DIR, exist_ok=True)
            agent.save_model(f"{MODEL_DIR}/dqn_baseline_{SUFFIX}.pth")

    logger.info(f"AoI: {[float(aoi) for aoi in np.round(aois, 4)]}")
    logger.info(f"Avg cost/slot (last 5 episodes): {[float(c) for c in np.round(costs[-5:], 4)]}")
    logger.info("Saving the results")

    # Note: the best model is saved in the training loop (best_aoi), so the
    # final checkpoint always reflects the best-performing weights.

    # Save the results
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_PATH = f"{DATA_DIR}/dqn_results_{SUFFIX}.csv"
    df = pd.DataFrame(
        {
            "average_aoi": [float(aoi) for aoi in np.round(aois, 4)],
            "avg_cost_per_slot": [float(cost) for cost in np.round(costs, 4)],
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
