"""JSS benchmarking simulation.

This single experiment generates the CPI-resampling benchmark, the
SCPI-resampling benchmark, and the computational-cost comparison requested in
Section 3.5 of the manuscript.  Formal settings remain explicitly marked as
provisional until the authors approve the final design.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from fdfi.explainers import EOTExplainer, FlowExplainer, OTExplainer


DESIGN_STATUS = "DRAFT_FOR_AUTHOR_CONFIRMATION"
MASTER_SEED = 20260721
ALPHA = 0.05
DIMENSION = 50
BLOCK_SIZE = 10
ACTIVE_FEATURES = (0, 1, 10, 11)
EFFECTS = (1.5, 1.0, 1.5, 1.0)
NOISE_SD = 1.0
FIXED_RHO = 0.8
FIXED_N = 1000
FULL_N_VALUES = (200, 400, 600, 800, 1000)
FULL_RHO_VALUES = (0.4, 0.6, 0.8)
FULL_REPETITIONS = 100
QUICK_N_VALUES = (80, 100, 120, 160, 200)
QUICK_RHO_VALUES = (0.2, 0.4, 0.6, 0.8)
QUICK_REPETITIONS = 2
FULL_NSAMPLES = 50
QUICK_NSAMPLES = 5
FULL_FLOW_STEPS = 5000
QUICK_FLOW_STEPS = 3
BASELINE_METHODS = ("LOCO", "CPI")
FDFI_FAMILIES = ("DFI-OT", "DFI-EOT", "FDFI")
RESAMPLING_VERSIONS = ("cpi", "scpi")

INFERENCE = {
    "alpha": ALPHA,
    "target": "X",
    "threshold_null": False,
    "multitest_method": None,
    "var_floor_c": 0.0,
    "var_floor_method": "fixed",
    "margin": 0.0,
    "margin_method": "fixed",
    "alternative": "greater",
}


def settings(mode: str) -> dict[str, object]:
    quick = mode == "quick"
    return {
        "design_status": DESIGN_STATUS,
        "benchmark_requirement": {
            "cpi_figure": ["LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI"],
            "scpi_figure": ["LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI"],
            "runtime_figure": "one two-panel computational-cost figure",
        },
        "mode": mode,
        "master_seed": MASTER_SEED,
        "alpha": ALPHA,
        "dimension": 10 if quick else DIMENSION,
        "block_size": 5 if quick else BLOCK_SIZE,
        "active_features": (0, 1, 5, 6) if quick else ACTIVE_FEATURES,
        "effects": EFFECTS,
        "response": "nonlinear regression with Gaussian noise",
        "noise_sd": NOISE_SD,
        "fixed_rho": FIXED_RHO,
        "fixed_n": 200 if quick else FIXED_N,
        "n_values": QUICK_N_VALUES if quick else FULL_N_VALUES,
        "rho_values": QUICK_RHO_VALUES if quick else FULL_RHO_VALUES,
        "repetitions": QUICK_REPETITIONS if quick else FULL_REPETITIONS,
        "bootstrap_draws": 200 if quick else 2000,
        "nsamples": QUICK_NSAMPLES if quick else FULL_NSAMPLES,
        "flow_steps": QUICK_FLOW_STEPS if quick else FULL_FLOW_STEPS,
        "baseline_methods": BASELINE_METHODS,
        "fdfi_families": FDFI_FAMILIES,
        "resampling_versions": RESAMPLING_VERSIONS,
        "runtime_definition": (
            "wall-clock seconds for each standalone method call; DFI/FDFI time "
            "includes transformation/flow construction or training and inference"
        ),
        "inference": INFERENCE,
        "predictor": {
            "class": "RandomForestRegressor",
            "n_estimators": 30 if quick else 300,
            "min_samples_leaf": 5,
            "test_size": 0.3,
        },
    }


def block_covariance(dimension: int, block_size: int, rho: float) -> np.ndarray:
    """Return a block-diagonal equicorrelation covariance matrix."""
    if dimension % block_size:
        raise ValueError("dimension must be divisible by block_size")
    if not (-1.0 / (block_size - 1) < rho < 1.0):
        raise ValueError("rho does not produce positive-definite blocks")
    covariance = np.eye(dimension)
    for start in range(0, dimension, block_size):
        stop = start + block_size
        covariance[start:stop, start:stop] = rho
        np.fill_diagonal(covariance[start:stop, start:stop], 1.0)
    return covariance


def generate_data(
    n: int,
    rho: float,
    seed: int,
    dimension: int,
    block_size: int,
    active_features: tuple[int, ...],
    effects: tuple[float, ...],
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate block-correlated Gaussian predictors and a nonlinear response."""
    rng = np.random.default_rng(seed)
    covariance = block_covariance(dimension, block_size, rho)
    X = rng.multivariate_normal(np.zeros(dimension), covariance, size=n)
    signal = (
        effects[0] * X[:, active_features[0]]
        + effects[1] * X[:, active_features[1]] ** 2
        + effects[2] * np.sin(X[:, active_features[2]])
        + effects[3] * X[:, active_features[2]] * X[:, active_features[3]]
    )
    y = signal + rng.normal(0.0, noise_sd, size=n)
    relevant = np.zeros(dimension, dtype=bool)
    relevant[list(active_features)] = True
    return X, y, relevant


def _feature_rows(
    method: str,
    resampling_version: str,
    score: np.ndarray,
    se: np.ndarray,
    relevant: np.ndarray,
    scenario: dict[str, object],
    repetition: int,
    seed: int,
    test_r2: float,
    runtime: float,
) -> list[dict[str, object]]:
    safe_se = np.maximum(np.asarray(se, dtype=float), np.finfo(float).eps)
    zscore = np.asarray(score, dtype=float) / safe_se
    pvalue = norm.sf(zscore)
    return [
        {
            **scenario,
            "repetition": repetition,
            "seed": seed,
            "method": method,
            "resampling_version": resampling_version,
            "feature": feature,
            "is_relevant": bool(relevant[feature]),
            "score": float(score[feature]),
            "se": float(safe_se[feature]),
            "zscore": float(zscore[feature]),
            "pvalue": float(pvalue[feature]),
            "reject_null": bool(pvalue[feature] < ALPHA),
            "test_r2": float(test_r2),
            "runtime_seconds": float(runtime),
            "status": "SUCCESS",
            "error": "",
        }
        for feature in range(len(relevant))
    ]


def _cpi_baseline(
    model: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    nsamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal CPI baseline using prediction averaging before squared loss."""
    prediction = model(X)
    base_loss = (y - prediction) ** 2
    n, d = X.shape
    ueifs = np.zeros((n, d))
    for feature in range(d):
        rng = np.random.default_rng(seed + feature)
        perturbed_predictions = []
        for _ in range(nsamples):
            changed = X.copy()
            changed[:, feature] = changed[rng.permutation(n), feature]
            perturbed_predictions.append(model(changed))
        predictions = np.asarray(perturbed_predictions)
        ueifs[:, feature] = (y - predictions.mean(axis=0)) ** 2 - base_loss
    return ueifs.mean(axis=0), ueifs.std(axis=0, ddof=1) / np.sqrt(n)


def _loco_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    full_prediction: np.ndarray,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LOCO loss differences from one reduced random forest per feature."""
    base_loss = (y_test - full_prediction) ** 2
    ueifs = np.zeros((len(y_test), X_test.shape[1]))
    for feature in range(X_test.shape[1]):
        reduced = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            random_state=seed + feature + 1,
            n_jobs=1,
        ).fit(np.delete(X_train, feature, axis=1), y_train)
        reduced_prediction = reduced.predict(np.delete(X_test, feature, axis=1))
        ueifs[:, feature] = (y_test - reduced_prediction) ** 2 - base_loss
    return ueifs.mean(axis=0), ueifs.std(axis=0, ddof=1) / np.sqrt(len(y_test))


def _fdfi_importance(
    family: str,
    resampling_version: str,
    model: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    nsamples: int,
    flow_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one standalone OT/EOT/Flow benchmark with CPI or SCPI scoring."""
    common = {
        "model": model,
        "data": X_train,
        "nsamples": nsamples,
        "random_state": seed,
        "method": resampling_version,
        "loss": "squared_error",
        "compute_diagnostics": False,
    }
    if family == "DFI-OT":
        explainer = OTExplainer(**common)
    elif family == "DFI-EOT":
        explainer = EOTExplainer(epsilon=0.001, **common)
    elif family == "FDFI":
        explainer = FlowExplainer(
            fit_flow=False,
            flow_training_seed=seed,
            num_steps=flow_steps,
            verbose=False,
            hidden_dim=32,
            time_embed_dim=16,
            num_blocks=1,
            use_bn=False,
            **common,
        )
        explainer.fit_flow(
            num_steps=flow_steps,
            verbose=False,
            batch_size=min(64, len(X_train)),
        )
    else:
        raise ValueError(f"unknown DFI/FDFI family: {family}")
    explainer(X_test, y=y_test)
    inference = explainer.conf_int(**INFERENCE)
    return np.asarray(inference["score"]), np.asarray(inference["se"])


def _scenarios(cfg: dict[str, object]) -> list[dict[str, object]]:
    scenarios = [
        {"sweep": "sample_size", "n": int(n), "rho": float(cfg["fixed_rho"])}
        for n in cfg["n_values"]
    ]
    scenarios.extend(
        {"sweep": "correlation", "n": int(cfg["fixed_n"]), "rho": float(rho)}
        for rho in cfg["rho_values"]
    )
    return scenarios


def _bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean over independent repetitions."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        mean = float(values.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(feature_results: pd.DataFrame, bootstrap_draws: int) -> pd.DataFrame:
    """Aggregate feature results into benchmark metrics and bootstrap intervals."""
    successful = feature_results.loc[feature_results["status"] == "SUCCESS"].copy()
    keys = ["sweep", "n", "rho", "method", "resampling_version"]
    rows: list[dict[str, object]] = []
    for group_index, (group_keys, frame) in enumerate(successful.groupby(keys, sort=False)):
        per_rep = []
        for repetition, rep_frame in frame.groupby("repetition"):
            active = rep_frame.loc[rep_frame["is_relevant"]]
            null = rep_frame.loc[~rep_frame["is_relevant"]]
            per_rep.append({
                "repetition": repetition,
                "test_r2": rep_frame["test_r2"].iloc[0],
                "power": active["reject_null"].mean(),
                "type1_error": null["reject_null"].mean(),
                "runtime_seconds": rep_frame["runtime_seconds"].iloc[0],
            })
        rep_metrics = pd.DataFrame(per_rep)
        row = dict(zip(keys, group_keys))
        row["successful_repetitions"] = len(rep_metrics)
        for metric in ("test_r2", "power", "type1_error", "runtime_seconds"):
            values = rep_metrics[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_interval(
                values, MASTER_SEED + group_index * 100 + len(metric), bootstrap_draws
            )
            output_name = "mean_runtime_seconds" if metric == "runtime_seconds" else metric
            row[output_name] = float(values.mean())
            row[f"{output_name}_ci_lower"] = lower
            row[f"{output_name}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _failure_row(
    scenario: dict[str, object], repetition: int, seed: int, method: str,
    version: str, test_r2: float, runtime: float, error: Exception,
) -> dict[str, object]:
    return {
        **scenario,
        "repetition": repetition,
        "seed": seed,
        "method": method,
        "resampling_version": version,
        "feature": -1,
        "is_relevant": False,
        "score": np.nan,
        "se": np.nan,
        "zscore": np.nan,
        "pvalue": np.nan,
        "reject_null": False,
        "test_r2": test_r2,
        "runtime_seconds": runtime,
        "status": "FAILED",
        "error": repr(error),
    }


def run_experiment(mode: str, results_dir: Path, tables_dir: Path, metadata_dir: Path) -> list[Path]:
    """Run both sweeps, both resampling versions, and write all source tables."""
    cfg = settings(mode)
    metadata_path = metadata_dir / "simulation_settings.json"
    metadata_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(_scenarios(cfg)):
        for repetition in range(int(cfg["repetitions"])):
            seed = MASTER_SEED + scenario_index * 10000 + repetition * 100
            X, y, relevant = generate_data(
                n=int(scenario["n"]), rho=float(scenario["rho"]), seed=seed,
                dimension=int(cfg["dimension"]), block_size=int(cfg["block_size"]),
                active_features=tuple(cfg["active_features"]), effects=tuple(cfg["effects"]),
                noise_sd=float(cfg["noise_sd"]),
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
            predictor = RandomForestRegressor(
                n_estimators=int(cfg["predictor"]["n_estimators"]),
                min_samples_leaf=int(cfg["predictor"]["min_samples_leaf"]),
                random_state=seed,
                n_jobs=1,
            ).fit(X_train, y_train)
            model = predictor.predict
            full_prediction = model(X_test)
            test_r2 = r2_score(y_test, full_prediction)

            for method in BASELINE_METHODS:
                started = time.perf_counter()
                try:
                    if method == "LOCO":
                        score, se = _loco_importance(
                            X_train, y_train, X_test, y_test, full_prediction,
                            seed, int(cfg["predictor"]["n_estimators"]),
                        )
                    else:
                        score, se = _cpi_baseline(
                            model, X_test, y_test, seed, int(cfg["nsamples"])
                        )
                    runtime = time.perf_counter() - started
                    rows.extend(_feature_rows(
                        method, "baseline", score, se, relevant, scenario,
                        repetition, seed, test_r2, runtime,
                    ))
                except Exception as exc:
                    rows.append(_failure_row(
                        scenario, repetition, seed, method, "baseline", test_r2,
                        time.perf_counter() - started, exc,
                    ))

            for family in FDFI_FAMILIES:
                for version in RESAMPLING_VERSIONS:
                    started = time.perf_counter()
                    try:
                        score, se = _fdfi_importance(
                            family, version, model, X_train, X_test, y_test, seed,
                            int(cfg["nsamples"]), int(cfg["flow_steps"]),
                        )
                        runtime = time.perf_counter() - started
                        rows.extend(_feature_rows(
                            family, version, score, se, relevant, scenario,
                            repetition, seed, test_r2, runtime,
                        ))
                    except Exception as exc:
                        rows.append(_failure_row(
                            scenario, repetition, seed, family, version, test_r2,
                            time.perf_counter() - started, exc,
                        ))

    feature_results = pd.DataFrame(rows)
    summary = summarize(feature_results, int(cfg["bootstrap_draws"]))
    runtime_results = feature_results.loc[
        feature_results["feature"].isin((-1, 0)),
        ["sweep", "n", "rho", "repetition", "seed", "method",
         "resampling_version", "runtime_seconds", "status", "error"],
    ].copy()
    runtime_summary = summary[[
        "sweep", "n", "rho", "method", "resampling_version",
        "mean_runtime_seconds", "mean_runtime_seconds_ci_lower",
        "mean_runtime_seconds_ci_upper", "successful_repetitions",
    ]].copy()

    result_path = results_dir / "simulation_feature_results.csv"
    runtime_result_path = results_dir / "simulation_runtime_results.csv"
    summary_path = tables_dir / "simulation_benchmark_summary.csv"
    runtime_summary_path = tables_dir / "simulation_runtime_summary.csv"
    feature_results.to_csv(result_path, index=False)
    runtime_results.to_csv(runtime_result_path, index=False)
    summary.to_csv(summary_path, index=False)
    runtime_summary.to_csv(runtime_summary_path, index=False)
    return [metadata_path, result_path, runtime_result_path, summary_path, runtime_summary_path]
