"""Run the GAdaOFUL simulation experiments reproducibly."""

from pathlib import Path
from typing import Iterable

import numpy as np

from src.algorithms import (
    run_adaoful,
    run_cw_oful,
    run_gadaoful,
    run_greedy,
    run_oful,
)
from src.config import DEFAULT_CONFIG, ExperimentConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "trajectories"


ALGORITHMS = (
    ("GADA", "GAdaOFUL", run_gadaoful),
    ("AdditiveC", "CW-OFUL", run_cw_oful),
    ("Greedy", "Greedy", run_greedy),
    ("OFUL", "OFUL", run_oful),
    ("ADA", "AdaOFUL", run_adaoful),
)


def save_trajectory(path: Path, trajectory: np.ndarray) -> None:
    """Save one cumulative-regret trajectory as a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory)


def run_all_experiments(
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: Iterable[int] | None = None,
) -> list[Path]:
    """Run all algorithms and repetitions and save their trajectories.

    Parameters
    ----------
    config
        Experiment configuration.
    output_dir
        Directory where trajectory files are written.
    seeds
        One random seed for each repetition. If omitted, the seeds are
        0, 1, ..., repeat - 1.

    Returns
    -------
    list[Path]
        Paths of all generated trajectory files.
    """
    output_dir = Path(output_dir)

    if seeds is None:
        seeds = tuple(range(config.repeat))
    else:
        seeds = tuple(seeds)

    if len(seeds) != config.repeat:
        raise ValueError(
            "The number of seeds must equal config.repeat."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(ALGORITHMS) * config.repeat
    completed_runs = 0
    generated_files = []

    for prefix, display_name, algorithm in ALGORITHMS:
        for repetition, seed in enumerate(seeds):
            completed_runs += 1

            print(
                f"[{completed_runs}/{total_runs}] "
                f"Running {display_name}, "
                f"repetition {repetition + 1}/{config.repeat} "
                f"(seed={seed})"
            )

            # The original analysis uses NumPy's global random-number
            # generator, including through SciPy distributions.
            # Resetting the seed before each repetition makes the
            # simulation deterministic.
            np.random.seed(seed)

            trajectory = np.asarray(
                algorithm(config),
                dtype=float,
            )

            if trajectory.shape != (config.T,):
                raise RuntimeError(
                    f"{display_name} returned a trajectory with "
                    f"shape {trajectory.shape}; expected ({config.T},)."
                )

            if not np.all(np.isfinite(trajectory)):
                raise RuntimeError(
                    f"{display_name} produced non-finite regret values."
                )

            output_path = (
                output_dir
                / (
                    f"{prefix}_lowerbound_"
                    f"{repetition}_{config.corruption}.txt"
                )
            )

            save_trajectory(
                output_path,
                trajectory,
            )

            generated_files.append(output_path)

            print(f"    Saved {output_path}")

    print(
        f"Completed {total_runs} runs. "
        f"Trajectories saved to {output_dir}."
    )

    return generated_files


def main() -> None:
    """Run the default reproduction experiment."""
    run_all_experiments()


if __name__ == "__main__":
    main()
