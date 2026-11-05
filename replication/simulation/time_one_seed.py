#!/usr/bin/env python3
"""Time one simulation seed with the installed fdfi and estimate the full run.

Temporary helper. Put it in ``replication/simulation`` next to ``simulation.py``,
run it once, then delete it. It does not modify ``simulation.py``, ``results/``
or ``figures/``; all outputs go to a scratch folder.

Usage (from replication/simulation):
    python time_one_seed.py --smoke        # 1-2 minute wiring check (quick mode)
    python time_one_seed.py                # benchmark + runtime, one seed each
    python time_one_seed.py --part benchmark
    python time_one_seed.py --part runtime
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fdfi  # noqa: E402
import simulation as sim  # noqa: E402

# Number of seeds in the full runs used to scale the one-seed timing.
BENCHMARK_SEEDS_PAPER = 50     # Figures 1 and 8 (fifty_seed)
BENCHMARK_SEEDS_FULL = 100     # FULL_REPETITIONS in simulation.py
RUNTIME_SEEDS_PAPER = 16       # Figure 2 (D3_RUNTIME_REPETITIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("all", "benchmark", "runtime"), default="all")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the tiny quick mode only, to check that everything imports and runs",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="keep the scratch output folder instead of deleting it",
    )
    return parser.parse_args()


def fmt(seconds: float) -> str:
    return f"{seconds:,.0f} s = {seconds / 60:,.1f} min = {seconds / 3600:,.2f} h"


def main() -> int:
    args = parse_args()

    print(f"fdfi {fdfi.__version__} from {fdfi.__file__}")
    print(f"simulation.py from {sim.__file__}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}; "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    if not str(fdfi.__version__).startswith("0.2.1"):
        print("Expected fdfi 0.2.1. Run `pip install -e D:\\work\\hku\\FDFI` first.")
        return 2

    if args.smoke:
        bench_mode = runtime_mode = "quick"
    else:
        # Full benchmark grid with one seed.
        bench_mode = "one_seed"
        # Figure 2 grid (n = 200, ..., 1000) with one seed.
        runtime_mode = "d3"
        sim.D3_RUNTIME_REPETITIONS = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = HERE / f"_timing_{stamp}"
    dirs = {name: scratch / name for name in ("results", "tables", "metadata")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    print(f"Scratch output: {scratch}\n")

    timings: dict[str, float] = {}
    try:
        if args.part in ("all", "benchmark"):
            cfg = sim.settings(bench_mode)
            print(f"[benchmark] mode={bench_mode}, seeds={cfg['repetitions']}, "
                  f"n={tuple(cfg['n_values'])}, rho={tuple(cfg['rho_values'])}")
            start = time.perf_counter()
            sim.run_experiment(bench_mode, dirs["results"], dirs["tables"], dirs["metadata"])
            timings["benchmark"] = time.perf_counter() - start
            print(f"[benchmark] one seed: {fmt(timings['benchmark'])}\n")

        if args.part in ("all", "runtime"):
            cfg = sim.runtime_settings(runtime_mode)
            print(f"[runtime] mode={runtime_mode}, seeds={cfg['repetitions']}, "
                  f"n={tuple(cfg['n_values'])}")
            start = time.perf_counter()
            sim.run_runtime_experiment(
                runtime_mode, dirs["results"], dirs["tables"], dirs["metadata"]
            )
            timings["runtime"] = time.perf_counter() - start
            print(f"[runtime] one seed: {fmt(timings['runtime'])}\n")
    finally:
        if args.keep:
            print(f"Kept scratch output in {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    print("=" * 70)
    if args.smoke:
        print("Smoke test finished; timings are not representative.")
        return 0
    print("Estimated wall-clock time for the full runs (single process):")
    if "benchmark" in timings:
        t = timings["benchmark"]
        print(f"  benchmark x {BENCHMARK_SEEDS_PAPER} seeds (Figures 1, 8): "
              f"{fmt(BENCHMARK_SEEDS_PAPER * t)}")
        print(f"  benchmark x {BENCHMARK_SEEDS_FULL} seeds (full mode):     "
              f"{fmt(BENCHMARK_SEEDS_FULL * t)}")
    if "runtime" in timings:
        t = timings["runtime"]
        print(f"  runtime   x {RUNTIME_SEEDS_PAPER} seeds (Figure 2):        "
              f"{fmt(RUNTIME_SEEDS_PAPER * t)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
