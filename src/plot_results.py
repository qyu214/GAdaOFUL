"""Aggregate simulation trajectories and generate the final regret figure."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DEFAULT_CONFIG, ExperimentConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "artifacts" / "trajectories"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "figures" / "cor+nonlin1.pdf"
)


SERIES = (
    ("OFUL", "OFUL", "black", ":"),
    ("Greedy", "Greedy", "blue", "--"),
    ("AdditiveC", "CW-OFUL", "green", "-."),
    ("GADA", "GAdaOFUL", "red", "-"),
    ("ADA", "AdaOFUL(nonlinear)", "orange", (0, (5, 1))),
)


def load_mean_trajectory(
    prefix: str,
    config: ExperimentConfig,
    input_dir: Path = DEFAULT_INPUT_DIR,
) -> np.ndarray:
    """Load repeated trajectories for one algorithm and return their mean."""
    input_dir = Path(input_dir)
    trajectories = []

    for repetition in range(config.repeat):
        path = (
            input_dir
            / (
                f"{prefix}_lowerbound_"
                f"{repetition}_{config.corruption}.txt"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Required trajectory file not found: {path}"
            )

        trajectory = np.loadtxt(path)

        if trajectory.shape != (config.T,):
            raise ValueError(
                f"{path} has shape {trajectory.shape}; "
                f"expected ({config.T},)."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                f"{path} contains non-finite values."
            )

        trajectories.append(trajectory)

    return np.mean(
        np.stack(trajectories, axis=0),
        axis=0,
    )


def plot_results(
    config: ExperimentConfig = DEFAULT_CONFIG,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Generate the cumulative-regret comparison figure."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    print(f"Loading trajectories from {input_dir}")

    means = {}

    for prefix, label, _, _ in SERIES:
        print(f"    Aggregating {label}")
        means[prefix] = load_mean_trajectory(
            prefix=prefix,
            config=config,
            input_dir=input_dir,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    # Match the plotting range used in the original notebook.
    x = np.arange(1, config.T)

    for prefix, label, color, linestyle in SERIES:
        ax.plot(
            x,
            means[prefix][: config.T - 1],
            color=color,
            label=label,
            linestyle=linestyle,
        )

    ax.legend(fontsize=10)
    ax.set_xlabel("Number of Steps", size=10)
    ax.set_ylabel("Regret", size=10)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved figure to {output_path}")

    return output_path


def main() -> None:
    """Generate the default reproduction figure."""
    plot_results()


if __name__ == "__main__":
    main()
