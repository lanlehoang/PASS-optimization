import numpy as np
from src.agents.dqn import DqnAgent
from src.utils.get_config import get_system_config
from src.utils.generators import RANDOM_SEED
from src.utils.others import state_to_arrays
from src.env.env import AntennaEnv
from .prompts import GetPrompts
from src.utils.logger import get_logger

logger = get_logger(__name__)
N_NEIGHBOURS = get_system_config()["satellite"]["n_neighbours"]


class SatelliteRouting:
    def __init__(self, model_path) -> None:
        self.n_instance = 4

        # Early break from circuitous heuristics
        self.max_step_per_instance = 5000
        self.early_break_fitness = -1.0

        self.drop_threshold = 0.1  # Acceptable dropped ratio
        self._last_eval_metrics = None  # For logging purposes

        # Load pre-trained agent
        self.agent = DqnAgent()
        self.agent.load_model(model_path)
        self.prompts = GetPrompts()

    def evaluate(self, code_string):
        # Execute LLM-generated code ONCE before loop
        try:
            namespace = {"np": np, "numpy": np}
            exec(code_string, namespace)

            if "compute_offset" not in namespace:
                logger.error("compute_offset function not found in generated code")
                return self.early_break_fitness
            compute_offset = namespace["compute_offset"]

        except Exception as e:
            logger.error(f"Error executing generated code: {e}")
            return self.early_break_fitness

        # Initialize environment and run simulations
        np.random.seed(RANDOM_SEED)
        env = AntennaEnv()

        aois = []
        dropped_ratios = []

        for _ in range(self.n_instance):
            env.reset()
            episode_done = False
            steps = 0

            while not episode_done:
                env.handle_events()
                dis, arc, process, queue = state_to_arrays(env.state)

                # Call the offset function
                try:
                    q_offset = compute_offset(dis, arc, process, queue)

                    # Validate offset
                    if not isinstance(q_offset, np.ndarray):
                        q_offset = np.array(q_offset)

                    if q_offset.shape != (N_NEIGHBOURS,):
                        logger.error(f"offset shape {q_offset.shape} != expected ({N_NEIGHBOURS},)")
                        return self.early_break_fitness

                except Exception as e:
                    logger.error(f"Error calling compute_offset: {e}")
                    return self.early_break_fitness

                action = self.agent.choose_action_with_offset(env.state, q_offset)
                episode_done, episode_info = env.step(action)
                steps += 1

            if steps > self.max_step_per_instance:
                logger.warning("Max steps exceeded in an episode")
                return self.early_break_fitness

            # Get metrics
            aois.append(episode_info["average_aoi"])
            dropped_ratios.append(episode_info["dropped_ratio"])

        avg_aoi = np.mean(aois).item()
        avg_dropped_ratio = np.mean(dropped_ratios).item()

        fitness = (1 - avg_dropped_ratio) / avg_aoi
        logger.info(f"Avg AOI: {avg_aoi:.3f}, Avg Dropped Ratio: {avg_dropped_ratio:.3f}, Fitness: {fitness:.3f}")

        # Store metrics
        self._last_eval_metrics = {
            "avg_aoi": avg_aoi,
            "avg_dropped_ratio": avg_dropped_ratio,
        }
        return fitness


# Run a quick test
if __name__ == "__main__":
    from pathlib import Path
    import textwrap
    import time

    BASE_PATH = Path.resolve(Path(__file__)).parent.parent.parent.parent
    model_path = BASE_PATH / "models" / "dqn_baseline_20251201.pth"
    problem = SatelliteRouting(model_path)

    # Example code string
    test_code = textwrap.dedent(
        """
    import numpy as np
    def compute_offset(distances, arc_lengths, processing_rates, queue_lengths):
        v_light = 3 * 10**5  # km/s
        return -0.025 * queue_lengths + 0.005 * processing_rates - (arc_lengths + distances) / v_light
    """
    )

    start = time.time()
    fitness = problem.evaluate(test_code)
    end = time.time()

    print(f"Evaluation time: {end - start:.4f} seconds")
    print(f"Fitness of the test code: {fitness:.4f}")
