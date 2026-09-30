"""End-to-end tests for the reproduction pipeline."""

from src.algorithms import run_greedy
from src.config import ExperimentConfig
import src.plot_results as plot_module
import src.run_experiments as runner_module


def test_tiny_pipeline_generates_artifacts_and_figure(
    tmp_path,
    monkeypatch,
):
    """A small simulation should produce trajectories and a PDF figure."""
    config = ExperimentConfig(
        dim=2,
        sigma=1.0,
        corruption=1,
        T=4,
        repeat=2,
        actions=3,
        norm=1.0,
    )

    # Use one fast algorithm so the test checks the full pipeline
    # without running the expensive full reproduction experiment.
    monkeypatch.setattr(
        runner_module,
        "ALGORITHMS",
        (("Greedy", "Greedy", run_greedy),),
    )

    monkeypatch.setattr(
        plot_module,
        "SERIES",
        (("Greedy", "Greedy", "black", "--"),),
    )

    trajectory_dir = tmp_path / "trajectories"
    figure_path = tmp_path / "figures" / "regret.pdf"

    generated_files = runner_module.run_all_experiments(
        config=config,
        output_dir=trajectory_dir,
        seeds=[0, 1],
    )

    result_path = plot_module.plot_results(
        config=config,
        input_dir=trajectory_dir,
        output_path=figure_path,
    )

    assert len(generated_files) == 2

    for path in generated_files:
        assert path.exists()
        assert path.stat().st_size > 0

    assert result_path.exists()
    assert result_path.stat().st_size > 0
