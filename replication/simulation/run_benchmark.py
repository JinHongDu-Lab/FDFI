"""Simulation study of Section 3.5 and Appendix B: Figures 1, 2 and 8.

Benchmark (Figures 1 and 8, Appendix B.1-B.2): X ~ N(0, I_5 (x) Sigma_rho),
d = 50; n in {200, ..., 1000} at rho = 0.8 and rho in {0.4, 0.6, 0.8} at
n = 1000; random forest with 500 trees and min leaf 5; two-fold
cross-fitting; 50 draws; flow trained for 15,000 steps on 3,000 auxiliary
samples; alpha = 0.05; 50 seeds; 2,000 bootstrap draws.

Runtime (Figure 2, Appendix B.3): mixture 0.2 N(0, Sigma(0.8)) + 0.8 N(0, Sigma(0.2)),
n in {200, ..., 1000}, 16 repetitions, d = 50 (SHAP: d = 10, 100 coalition
draws), flow trained for 5,000 steps; only method evaluation is timed.

    python run_benchmark.py                      # Figures 1, 2, 8 from results/
    python run_benchmark.py --rerun              # rerun both studies into rerun/
    python run_benchmark.py --rerun benchmark    # Figures 1 and 8 only
    python run_benchmark.py --rerun runtime      # Figure 2 only
"""
import argparse
import os
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import pandas as pd

from simulation_plot import _benchmark_figure, _runtime_figure

HERE = Path(__file__).resolve().parent


def rerun_benchmark(results, work):
    import torchdiffeq, tqdm  # noqa: F401  without them FDFI rows silently fail
    import simulation

    runtime = simulation.run_runtime_experiment
    simulation.run_runtime_experiment = lambda *a, **k: []  # runtime is rerun separately
    try:
        simulation.run_experiment("fifty_seed", work, work, work)
    finally:
        simulation.run_runtime_experiment = runtime

    features = pd.read_csv(work / "simulation_feature_results.csv")
    failed = features[features["status"] != "SUCCESS"]
    if len(failed):
        raise SystemExit(f"{len(failed)} failed rows, e.g. {failed['error'].iloc[0]}")
    features.to_csv(results / "simulation_feature_results.csv", index=False)

    summary = pd.read_csv(work / "simulation_benchmark_summary.csv")
    for version in ("cpi", "scpi"):
        keep = summary["resampling_version"].isin(["baseline", version])
        summary[keep].to_csv(results / f"simulation_benchmark_{version}.csv", index=False)


def rerun_runtime(results, work):
    import simulation

    simulation.run_runtime_experiment("d3", work, work, work)
    for name in ("results", "summary"):
        pd.read_csv(work / f"simulation_runtime_{name}.csv").to_csv(
            results / f"runtime_{name}.csv", index=False)


def plot(results, figures):
    figures.mkdir(parents=True, exist_ok=True)
    for version, name in (("cpi", "fig1_benchmark_cpi.pdf"), ("scpi", "fig8_benchmark_scpi.pdf")):
        summary = results / f"simulation_benchmark_{version}.csv"
        if summary.exists():
            _benchmark_figure(pd.read_csv(summary), version, figures / name)
            print(f"wrote {figures / name}")
    summary = results / "runtime_summary.csv"
    if summary.exists():
        _runtime_figure(pd.read_csv(summary), figures / "fig2_runtime.pdf")
        print(f"wrote {figures / 'fig2_runtime.pdf'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", nargs="?", const="all", choices=("all", "benchmark", "runtime"))
    args = parser.parse_args()

    out = HERE / "rerun" if args.rerun else HERE
    results = out / "results"
    if args.rerun:
        results.mkdir(parents=True, exist_ok=True)
        if args.rerun in ("all", "benchmark"):
            work = out / "checkpoints" / "benchmark"
            work.mkdir(parents=True, exist_ok=True)
            rerun_benchmark(results, work)
        if args.rerun in ("all", "runtime"):
            work = out / "checkpoints" / "runtime"
            work.mkdir(parents=True, exist_ok=True)
            rerun_runtime(results, work)
    plot(results, out / "figures")


if __name__ == "__main__":
    main()
