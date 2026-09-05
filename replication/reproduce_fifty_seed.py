#!/usr/bin/env python3
"""Run the professor-approved 50-seed simulation validation."""

from __future__ import annotations

import argparse
import os
import sys
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
    os.environ[_name] = "1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import run_replication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the professor-approved 50-seed benchmark validation with one "
            "runtime seed."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the 50-seed configuration without running it",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="directory that receives the isolated timestamped 50-seed run",
    )
    parser.add_argument(
        "--source-feature-checkpoint",
        type=Path,
        default=None,
        help=(
            "optional compatible benchmark checkpoint; complete matching seeds "
            "are imported before missing seeds are computed"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    options = {}
    if args.runs_dir is not None:
        options["runs_dir"] = args.runs_dir.expanduser().resolve()
    if args.source_feature_checkpoint is not None:
        os.environ["FDFI_BENCHMARK_SOURCE_CHECKPOINT"] = str(
            args.source_feature_checkpoint.expanduser().resolve()
        )
    raise SystemExit(
        run_replication(
            mode="fifty_seed",
            check_only=args.check,
            workflow="simulation",
            **options,
        )
    )
