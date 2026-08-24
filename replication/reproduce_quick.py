#!/usr/bin/env python3
"""Run a reduced smoke test of the FDFI JSS replication workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fdfi-replication-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fdfi-replication-cache")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import run_replication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the FDFI JSS replication workflow."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check known files and inputs without running the analyses",
    )
    parser.add_argument(
        "--workflow", choices=("all", "sens50", "ctg", "simulation"), default="all",
        help="run/check one isolated smoke-test workflow",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "directory that receives timestamped run folders; defaults to "
            "this checkout's replication/runs"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    options = {}
    if args.runs_dir is not None:
        options["runs_dir"] = args.runs_dir.expanduser().resolve()
    raise SystemExit(
        run_replication(
            mode="quick",
            check_only=args.check,
            workflow=args.workflow,
            **options,
        )
    )
