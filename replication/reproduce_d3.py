#!/usr/bin/env python3
"""Reproduce the Appendix D.4 / Figure D3 computational-cost comparison.

This is a standalone counterpart to the runtime half of the manuscript
simulation study: it runs only the Experiment 2 runtime benchmark
(``run_runtime_experiment``) and renders the D3-style figure, without the
much more expensive Section 4.1 AUC/power benchmark that the other
``reproduce_*`` entry points bundle alongside it.

Sample-size grid (200, 400, 600, 800, 1000) matches Figure D3 exactly; the
repetition count is raised from the paper's 10 to 20 for a tighter estimate
of mean runtime per method.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fdfi-replication-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fdfi-replication-cache")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import common
from scripts.common import RUNS_DIR, RunConfig, relative
from scripts.simulation import runtime_settings, run_runtime_experiment
from scripts.simulation_plot import create_runtime_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the Figure D3 computational-cost comparison "
            "(Appendix D.4): CPI, LOCO, nLOCO, dLOCO, OT, EOT, FDFI, SHAP "
            "across n in (200, 400, 600, 800, 1000), 20 repetitions each."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the resolved configuration without running it",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="directory that receives the isolated timestamped D3 run",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "resume a specific existing run directory (e.g. "
            "d3-20260908T022122.041552Z) instead of starting a new one; "
            "already-completed n/seed/method cells are skipped"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = runtime_settings("d3")
    if args.check:
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    runs_dir = (args.runs_dir or RUNS_DIR).expanduser().resolve()
    config = RunConfig.create("d3", run_id=args.run_id, runs_dir=runs_dir)
    for directory in (
        config.figures_dir, config.tables_dir, config.results_dir,
        config.logs_dir, config.metadata_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"FDFI Figure D3 reproduction — run_id={config.run_id}")
    print(json.dumps(cfg, indent=2, sort_keys=True))

    started = time.perf_counter()
    artifacts = run_runtime_experiment(
        mode="d3",
        results_dir=config.results_dir,
        tables_dir=config.tables_dir,
        metadata_dir=config.metadata_dir,
    )
    figure_path = create_runtime_figure(
        config.tables_dir / "simulation_runtime_summary.csv", config.figures_dir
    )
    artifacts.append(figure_path)
    elapsed = time.perf_counter() - started

    environment_path = common._write_environment(config)
    artifacts.append(environment_path)

    print(f"Completed in {elapsed:.1f}s. Generated files:")
    for path in artifacts:
        print(f"  {relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
