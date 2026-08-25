"""Run the reference Experiment 1 simulation and its plotting script."""

from __future__ import annotations

import time

import pandas as pd

from .common import Mode, RunConfig, WorkflowResult, relative
from .simulation import DESIGN_STATUS, run_experiment, runtime_settings
from .simulation_plot import create_figures


def preflight(mode: Mode = "full") -> dict[str, object]:
    """Report the confirmed reference specification used for both modes."""
    runtime = runtime_settings(mode)
    stage = {
        "quick": "quick is a smoke test",
        "review": "review uses the full grids with 10 benchmark repetitions and 1 runtime seed",
        "one_seed": "one_seed uses the full grids with 1 shared benchmark and runtime seed",
        "full": "full reproduces the manuscript protocol",
    }[mode]
    return {
        "ready": True,
        "detail": (
            f"{DESIGN_STATUS}: mode={mode}; {stage}; runtime "
            f"n={tuple(runtime['n_values'])}, seeds={tuple(runtime['seed_schedule'])}, "
            f"d(non-SHAP)={runtime['dimension_non_shap']}, "
            f"d(SHAP)={runtime['dimension_shap']}, timing=evaluate_only"
        ),
    }


def run(mode: Mode, config: RunConfig) -> WorkflowResult:
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
    type1_audit_path = config.tables_dir / "simulation_type1_error_audit.csv"
    type1_audit = pd.read_csv(type1_audit_path)
    incomplete = type1_audit.loc[type1_audit["status"] != "COMPLETE"]
    above_nominal = type1_audit.loc[
        (type1_audit["status"] == "COMPLETE") & type1_audit["above_nominal"]
    ]
    warnings = []
    if mode == "quick":
        warnings.append(
            "Quick mode is a computational smoke test and must not be cited as a formal result."
        )
    elif mode == "review":
        warnings.append(
            "Professor review stage: benchmarking uses 10 repetitions and runtime uses "
            "one seed; these results are preliminary and must not be cited as final."
        )
    elif mode == "one_seed":
        warnings.append(
            "Professor one-seed consistency check: all results use one shared seed; "
            "these results are preliminary and must not be cited as final."
        )
    if not incomplete.empty:
        warnings.append(
            f"Type-I-error audit has {len(incomplete)} incomplete method/configuration rows; "
            f"see {relative(type1_audit_path)} for missing or failed seeds."
        )
    if not above_nominal.empty:
        warnings.append(
            f"Empirical C3 Type-I error is at or above nominal alpha in "
            f"{len(above_nominal)} complete rows; these are reported for investigation "
            "and were not post-hoc adjusted."
        )
    return WorkflowResult(
        "manuscript simulation study",
        "SUCCESS",
        generated_files=[relative(path) for path in artifacts],
        key_results={
            "design_status": DESIGN_STATUS,
            "summary_csv": relative(summary),
            "type1_error_audit_csv": relative(type1_audit_path),
            "type1_error_configurations": int(len(type1_audit)),
            "type1_error_incomplete_configurations": int(len(incomplete)),
            "type1_error_above_nominal_configurations": int(len(above_nominal)),
            "type1_error_values_were_posthoc_adjusted": False,
            "formal_manuscript_result": mode == "full",
        },
        warnings=warnings,
        seeds={"master_seed": 20260721},
        runtime_seconds=time.perf_counter() - started,
    )
