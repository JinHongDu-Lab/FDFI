"""Run the draft manuscript simulation and its separate plotting script."""

from __future__ import annotations

import time

from .common import Mode, RunConfig, WorkflowBlocked, WorkflowResult, relative
from .simulation import DESIGN_STATUS, run_experiment
from .simulation_plot import create_figures


def preflight(mode: Mode = "full") -> dict[str, object]:
    """Allow a draft quick result but block formal use before author approval."""
    if mode == "quick":
        return {
            "ready": True,
            "detail": f"{DESIGN_STATUS}: quick meeting-preview run is allowed; not a formal manuscript result",
        }
    return {
        "ready": False,
        "detail": f"BLOCKED: {DESIGN_STATUS}; full settings require author confirmation",
    }


def run(mode: Mode, config: RunConfig) -> WorkflowResult:
    if mode == "full":
        raise WorkflowBlocked(
            f"{DESIGN_STATUS}: run quick for the meeting preview; confirm formal grids, "
            "repetitions, DGP, methods, and inference settings before a full run"
        )
    started = time.perf_counter()
    artifacts = run_experiment(
        mode=mode,
        results_dir=config.results_dir,
        tables_dir=config.tables_dir,
        metadata_dir=config.metadata_dir,
    )
    artifacts.extend(
        create_figures(
            config.tables_dir / "simulation_benchmark_summary.csv",
            config.tables_dir / "simulation_runtime_summary.csv",
            config.figures_dir,
        )
    )
    summary = config.tables_dir / "simulation_benchmark_summary.csv"
    return WorkflowResult(
        "manuscript simulation study",
        "SUCCESS",
        generated_files=[relative(path) for path in artifacts],
        key_results={
            "design_status": DESIGN_STATUS,
            "summary_csv": relative(summary),
            "formal_manuscript_result": False,
        },
        warnings=["Draft design for professor review; numerical results must not be cited as final."],
        seeds={"master_seed": 20260721},
        runtime_seconds=time.perf_counter() - started,
    )
