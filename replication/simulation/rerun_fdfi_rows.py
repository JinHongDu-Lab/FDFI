"""Regenerate only the DFI-OT, DFI-EOT and FDFI rows of the simulation study.

The LOCO, CPI, nLOCO, dLOCO and SHAP baselines do not call fdfi, so their
saved rows in ``results/`` do not depend on the fdfi version. This script
reruns the three fdfi methods with the same design, seeds and inference as
``run_benchmark.py --rerun`` and merges them with the saved baseline rows.
It was used to update Figures 1, 2 and 8 to fdfi 0.2.1.

    python rerun_fdfi_rows.py benchmark --workers 50 --threads 4
    python rerun_fdfi_rows.py runtime
    python rerun_fdfi_rows.py merge          # write results/ and the figures

The benchmark runs seeds in parallel worker processes (one GPU is shared for
flow training). The runtime study runs sequentially and should be run on an
otherwise idle machine, because it records wall-clock times.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "rerun" / "fdfi_rows"
FDFI_RUNTIME_METHODS = ("OT", "EOT", "FDFI")


def _limit_threads(threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)


def _benchmark_shard(shard, seeds, threads):
    """Run the fdfi methods for one subset of the 50-seed schedule."""
    _limit_threads(threads)
    sys.path.insert(0, str(HERE))
    import torch
    import simulation
    from sklearn.ensemble import RandomForestRegressor

    torch.set_num_threads(threads)
    simulation.BASELINE_METHODS = ()
    simulation.run_runtime_experiment = lambda *a, **k: []

    def make_predictor(seed, n_estimators):
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=threads,
        )

    simulation._make_predictor = make_predictor
    settings = simulation.settings

    def shard_settings(mode):
        cfg = settings(mode)
        cfg["seed_schedule"] = [cfg["seed_schedule"][i] for i in seeds]
        cfg["repetitions"] = len(seeds)
        return cfg

    simulation.settings = shard_settings
    out = WORK / "benchmark" / f"shard_{shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    simulation.run_experiment("fifty_seed", out, out, out)
    return shard


def run_benchmark(workers, threads, n_seeds):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    shards = [list(range(n_seeds))[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(len(shards), mp_context=ctx) as pool:
        futures = [
            pool.submit(_benchmark_shard, i, s, threads) for i, s in enumerate(shards)
        ]
        for future in futures:
            print(f"shard {future.result()} done", flush=True)


def run_runtime(n_seeds):
    _limit_threads(1)
    sys.path.insert(0, str(HERE))
    import simulation

    simulation.RUNTIME_METHODS = FDFI_RUNTIME_METHODS
    if n_seeds:
        simulation.D3_RUNTIME_REPETITIONS = n_seeds
    out = WORK / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    simulation.run_runtime_experiment("d3", out, out, out)


def merge():
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import simulation
    from run_benchmark import plot

    results = HERE / "results"
    fdfi = {"DFI-OT", "DFI-EOT", "FDFI"}

    pattern = "shard_*/simulation_feature_results.csv"
    shard_files = sorted((WORK / "benchmark").glob(pattern))
    if shard_files:
        features = pd.concat([pd.read_csv(f) for f in shard_files], ignore_index=True)
        failed = features[features["status"] != "SUCCESS"]
        if len(failed):
            example = failed["error"].iloc[0]
            raise SystemExit(f"{len(failed)} failed rows, e.g. {example}")
        schedule = simulation.settings("fifty_seed")["seed_schedule"]
        index = {seed: i for i, seed in enumerate(schedule)}
        features["repetition"] = features["seed"].map(index)
        features.to_csv(WORK / "fdfi_feature_results.csv", index=False)
        summary = simulation.summarize(features, 2000)
        for version in ("cpi", "scpi"):
            path = results / f"simulation_benchmark_{version}.csv"
            saved = pd.read_csv(path)
            new = summary[summary["resampling_version"] == version]
            merged = pd.concat(
                [saved[~saved["method"].isin(fdfi)], new[saved.columns]],
                ignore_index=True,
            )
            merged.to_csv(path, index=False)
            print(f"wrote {path}")

    runtime_file = WORK / "runtime" / "simulation_runtime_results.csv"
    if runtime_file.exists():
        new = pd.read_csv(runtime_file)
        failed = new[new["status"] != "SUCCESS"]
        if len(failed):
            raise SystemExit(f"{len(failed)} failed runtime rows")
        path = results / "runtime_results.csv"
        saved = pd.read_csv(path)
        merged = pd.concat(
            [saved[~saved["method"].isin(FDFI_RUNTIME_METHODS)], new[saved.columns]],
            ignore_index=True,
        )
        merged.to_csv(path, index=False)
        simulation.summarize_runtime(merged).to_csv(
            results / "runtime_summary.csv", index=False
        )
        print(f"wrote {path}")

    plot(results, HERE / "figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("benchmark", "runtime", "merge"))
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of seeds (default: 50 benchmark, 16 runtime)")
    args = parser.parse_args()
    if args.step == "benchmark":
        run_benchmark(args.workers, args.threads, args.seeds or 50)
    elif args.step == "runtime":
        run_runtime(args.seeds)
    else:
        merge()


if __name__ == "__main__":
    main()
