"""Experiment 1 benchmarks and the separate Experiment 2 runtime study.

The benchmark follows the published Gaussian block design and evaluates
Type-I error only on the independent null set C3.  Runtime is a separate
Gaussian-mixture experiment whose timer covers method evaluation only: shared
black-box fitting and OT/EOT/Flow construction or training are excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from fdfi.explainers import EOTExplainer, FlowExplainer, OTExplainer


DESIGN_STATUS = "REFERENCE_EXPERIMENT_1_SPECIFICATION"
MASTER_SEED = 20260721
ALPHA = 0.05
DIMENSION = 50
BLOCK_SIZE = 10
ACTIVE_FEATURES = (0, 1, 2, 3, 4)
NOISE_SD = 1.0
FIXED_RHO = 0.8
FIXED_N = 1000
FULL_N_VALUES = (200, 400, 600, 800, 1000)
FULL_RHO_VALUES = (0.4, 0.6, 0.8)
FULL_REPETITIONS = 100
REVIEW_REPETITIONS = 10
ONE_SEED_REPETITIONS = 1
QUICK_N_VALUES = (80, 100, 120, 160, 200)
QUICK_RHO_VALUES = (0.2, 0.4, 0.6, 0.8)
QUICK_REPETITIONS = 2
FULL_NSAMPLES = 50
QUICK_NSAMPLES = 5
FULL_FLOW_AUXILIARY_N = 3000
QUICK_FLOW_AUXILIARY_N = 200
FULL_FLOW_STEPS = 15000
QUICK_FLOW_STEPS = 3
N_FOLDS = 2
BASELINE_METHODS = ("LOCO", "CPI")
FDFI_FAMILIES = ("DFI-OT", "DFI-EOT", "FDFI")
RESAMPLING_VERSIONS = ("cpi", "scpi")
RUNTIME_METHODS = ("CPI", "LOCO", "nLOCO", "dLOCO", "OT", "EOT", "FDFI", "SHAP")
FULL_RUNTIME_N_VALUES = (200, 400, 600, 800, 1000, 2500)
QUICK_RUNTIME_N_VALUES = (40, 60)
FULL_RUNTIME_REPETITIONS = 10
REVIEW_RUNTIME_REPETITIONS = 1
QUICK_RUNTIME_REPETITIONS = 1
RUNTIME_DIMENSION = 50
SHAP_DIMENSION = 10
RUNTIME_RHO1 = 0.8
RUNTIME_RHO2 = 0.2
RUNTIME_MIX_WEIGHT = 0.2
RUNTIME_SHAP_MC = 100
RUNTIME_FLOW_STEPS = 5000

BENCHMARK_RESULT_COLUMNS = (
    "sweep", "n", "rho", "repetition", "seed", "method",
    "resampling_version", "feature", "is_relevant", "is_type1_null",
    "truth_group", "score", "se", "zscore", "pvalue", "reject_null",
    "margin", "margin_method", "test_r2", "runtime_seconds", "status", "error",
)

INFERENCE = {"alpha": ALPHA, "alternative": "greater"}


def settings(mode: str) -> dict[str, object]:
    quick = mode == "quick"
    review = mode == "review"
    one_seed = mode == "one_seed"
    repetitions = (
        QUICK_REPETITIONS
        if quick
        else ONE_SEED_REPETITIONS
        if one_seed
        else REVIEW_REPETITIONS if review else FULL_REPETITIONS
    )
    cfg = {
        "design_status": DESIGN_STATUS,
        "benchmark_requirement": {
            "cpi_figure": ["LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI"],
            "scpi_figure": ["LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI"],
            "runtime_figure": "one D3-style single-panel computational-cost figure",
        },
        "mode": mode,
        "execution_stage": (
            "quick_smoke_test"
            if quick
            else "professor_one_seed_consistency_check"
            if one_seed
            else "professor_review_stage_1" if review else "formal_manuscript"
        ),
        "master_seed": MASTER_SEED,
        "alpha": ALPHA,
        # The quick mode retains one complete 10-feature signal/correlated-null
        # block and one independent-null block so it exercises the same Type-I
        # error definition as the full design.
        "dimension": 20 if quick else DIMENSION,
        "block_size": BLOCK_SIZE,
        "active_features": ACTIVE_FEATURES,
        "truth_sets": {
            "active_C1": list(range(0, 5)),
            "correlated_null_C2": list(range(5, 10)),
            "independent_null_C3": list(range(10, 20 if quick else DIMENSION)),
        },
        "response": (
            "atan(X0 + X1) * I(X2 > 0) + sin(X3 * X4) * I(X2 < 0) "
            "+ Gaussian noise"
        ),
        "noise_sd": NOISE_SD,
        "fixed_rho": FIXED_RHO,
        "fixed_n": 200 if quick else FIXED_N,
        "n_values": QUICK_N_VALUES if quick else FULL_N_VALUES,
        "rho_values": QUICK_RHO_VALUES if quick else FULL_RHO_VALUES,
        "repetitions": repetitions,
        "seed_schedule": [MASTER_SEED + repetition * 100 for repetition in range(repetitions)],
        "bootstrap_draws": 200 if quick else 2000,
        "nsamples": QUICK_NSAMPLES if quick else FULL_NSAMPLES,
        "flow_steps": QUICK_FLOW_STEPS if quick else FULL_FLOW_STEPS,
        "flow_auxiliary_n": (
            QUICK_FLOW_AUXILIARY_N if quick else FULL_FLOW_AUXILIARY_N
        ),
        "n_folds": N_FOLDS,
        "baseline_methods": BASELINE_METHODS,
        "fdfi_families": FDFI_FAMILIES,
        "resampling_versions": RESAMPLING_VERSIONS,
        "runtime_experiment": runtime_settings(mode),
        "inference": INFERENCE,
        "predictor": {
            "class": "RandomForestRegressor",
            "n_estimators": 30 if quick else 500,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "cross_fitting": "KFold(n_splits=2, shuffle=True)",
        },
        "variance_floor": {
            "source": "original Experiment 1 reference estimator",
            "formula": (
                "c=min(sqrt(var_y), var_y, var_y^2, var_y^4)/d^2; "
                "se=sqrt(var_ueif+c)/(sqrt(n)*sqrt((K-1)/K))"
            ),
        },
        "checkpoint": (
            "one atomic feature-result CSV upsert after every unique "
            "sample-size x correlation x seed scenario"
        ),
    }
    return cfg


def runtime_settings(mode: str) -> dict[str, object]:
    """Return the independent Experiment 2 runtime contract."""
    quick = mode == "quick"
    review = mode == "review"
    one_seed = mode == "one_seed"
    repetitions = (
        QUICK_RUNTIME_REPETITIONS
        if quick
        else ONE_SEED_REPETITIONS
        if one_seed
        else REVIEW_RUNTIME_REPETITIONS if review else FULL_RUNTIME_REPETITIONS
    )
    return {
        "design": "Experiment 2 two-component Gaussian mixture",
        "mode": mode,
        "execution_stage": (
            "quick_smoke_test"
            if quick
            else "professor_one_seed_consistency_check"
            if one_seed
            else "professor_review_stage_1" if review else "formal_manuscript"
        ),
        "master_seed": MASTER_SEED,
        "n_values": QUICK_RUNTIME_N_VALUES if quick else FULL_RUNTIME_N_VALUES,
        "repetitions": repetitions,
        "seed_schedule": [
            MASTER_SEED + 900_000 + repetition * 100
            for repetition in range(repetitions)
        ],
        "methods": RUNTIME_METHODS,
        "dimension_non_shap": SHAP_DIMENSION if quick else RUNTIME_DIMENSION,
        "dimension_shap": SHAP_DIMENSION,
        "rho1": RUNTIME_RHO1,
        "rho2": RUNTIME_RHO2,
        "mix_weight": RUNTIME_MIX_WEIGHT,
        "nsamples": QUICK_NSAMPLES if quick else FULL_NSAMPLES,
        "shapley_mc_draws": 4 if quick else RUNTIME_SHAP_MC,
        "flow_steps": 1 if quick else RUNTIME_FLOW_STEPS,
        "flow_auxiliary_n": "match_runtime_n",
        "predictor": {
            "class": "RandomForestRegressor",
            "n_estimators": 3 if quick else 500,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "cross_fitting": "KFold(n_splits=2, shuffle=True)",
        },
        "timing_scope": {
            "included": (
                "method evaluation and method-specific submodel fits, including "
                "LOCO/nLOCO/dLOCO reduced or conditional models and SHAP coalition models"
            ),
            "excluded": (
                "data generation, shared black-box fitting, and OT/EOT/Flow "
                "map construction or Flow training"
            ),
        },
        "checkpoint": "one atomic CSV upsert after every sample-size x seed x method cell",
    }


def seed_everything(seed: int) -> None:
    """Set every available RNG used by the simulation to the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truth_group_labels(
    dimension: int,
    block_size: int,
    active_features: tuple[int, ...] = ACTIVE_FEATURES,
) -> np.ndarray:
    """Return explicit Experiment 1 C1/C2/C3 labels for every feature."""
    if dimension < 2 * block_size:
        raise ValueError("dimension must contain the signal block and at least one C3 block")
    labels = np.full(dimension, "independent_null_C3", dtype=object)
    labels[:block_size] = "correlated_null_C2"
    labels[list(active_features)] = "active_C1"
    return labels


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
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Experiment 1 data and explicit C1/C2/C3 truth labels."""
    if tuple(active_features) != ACTIVE_FEATURES:
        raise ValueError(f"active_features must be {ACTIVE_FEATURES}")
    rng = np.random.default_rng(seed)
    covariance = block_covariance(dimension, block_size, rho)
    X = rng.multivariate_normal(np.zeros(dimension), covariance, size=n)
    signal = (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] < 0)
    )
    y = signal + rng.normal(0.0, noise_sd, size=n)
    return X, y, truth_group_labels(dimension, block_size, active_features)


def generate_runtime_data(
    n: int,
    seed: int,
    dimension: int,
    rho1: float = RUNTIME_RHO1,
    rho2: float = RUNTIME_RHO2,
    mix_weight: float = RUNTIME_MIX_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the published Experiment 2 Gaussian-mixture runtime design."""
    rng = np.random.default_rng(seed)
    covariance_1 = block_covariance(dimension, BLOCK_SIZE, rho1)
    covariance_2 = block_covariance(dimension, BLOCK_SIZE, rho2)
    component = rng.binomial(n=1, p=mix_weight, size=n).astype(bool)
    X = np.empty((n, dimension), dtype=float)
    count_1 = int(component.sum())
    if count_1:
        X[component] = rng.multivariate_normal(
            np.zeros(dimension), covariance_1, size=count_1
        )
    if count_1 < n:
        X[~component] = rng.multivariate_normal(
            np.zeros(dimension), covariance_2, size=n - count_1
        )
    signal = (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] < 0)
    )
    return X, signal + rng.normal(0.0, NOISE_SD, size=n)


def _feature_rows(
    method: str,
    resampling_version: str,
    score: np.ndarray,
    se: np.ndarray,
    pvalue: np.ndarray,
    reject_null: np.ndarray,
    truth_groups: np.ndarray,
    scenario: dict[str, object],
    repetition: int,
    seed: int,
    test_r2: float,
    runtime: float,
    margin: float = 0.0,
    margin_method: str = "fixed",
) -> list[dict[str, object]]:
    safe_se = np.maximum(np.asarray(se, dtype=float), np.finfo(float).eps)
    zscore = (np.asarray(score, dtype=float) - margin) / safe_se
    return [
        {
            **scenario,
            "repetition": repetition,
            "seed": seed,
            "method": method,
            "resampling_version": resampling_version,
            "feature": feature,
            "is_relevant": truth_groups[feature] == "active_C1",
            "is_type1_null": truth_groups[feature] == "independent_null_C3",
            "truth_group": str(truth_groups[feature]),
            "score": float(score[feature]),
            "se": float(safe_se[feature]),
            "zscore": float(zscore[feature]),
            "pvalue": float(pvalue[feature]),
            "reject_null": bool(reject_null[feature]),
            "margin": float(margin),
            "margin_method": margin_method,
            "test_r2": float(test_r2),
            "runtime_seconds": float(runtime),
            "status": "SUCCESS",
            "error": "",
        }
        for feature in range(len(truth_groups))
    ]


def _make_predictor(seed: int, n_estimators: int) -> RandomForestRegressor:
    """Return a predictor configured for parallel fitting."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_predictor(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
) -> RandomForestRegressor:
    """Fit in parallel, then avoid joblib overhead on repeated small predicts."""
    predictor = _make_predictor(seed, n_estimators).fit(X, y)
    predictor.set_params(n_jobs=1)
    return predictor


def _fit_crossfit_predictors(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    n_folds: int = N_FOLDS,
) -> tuple[list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]], float]:
    """Fit shared predictors on the original two KFold training halves."""
    folds = []
    prediction = np.full(len(y), np.nan)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X):
        predictor = _fit_predictor(X[train_idx], y[train_idx], seed, n_estimators)
        prediction[test_idx] = predictor.predict(X[test_idx])
        folds.append((train_idx, test_idx, predictor))
    if np.isnan(prediction).any():
        raise RuntimeError("cross-fitting did not produce one prediction per observation")
    return folds, float(r2_score(y, prediction))


def _reference_inference(
    ueifs: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict[str, object]:
    """Apply the exact Experiment 1 threshold and variance-floor calculation."""
    ueifs = np.asarray(ueifs, dtype=float).copy()
    if ueifs.ndim != 2 or ueifs.shape[0] != len(y):
        raise ValueError("ueifs must have shape (len(y), d)")
    ueifs[:, np.nanmean(ueifs, axis=0) < 0] = 0.0
    score = np.nanmean(ueifs, axis=0)
    variance = np.nanvar(ueifs, axis=0, ddof=1)
    var_y = float(np.nanvar(y, ddof=1))
    floor_c = min(np.sqrt(var_y), var_y, var_y**2, var_y**4) / ueifs.shape[1] ** 2
    sigma = np.sqrt(variance + floor_c) + 1e-9
    effective_root_n = np.sqrt(len(y))
    if n_folds > 1:
        effective_root_n *= np.sqrt((n_folds - 1) / n_folds)
    se = sigma / effective_root_n
    zscore = score / se
    pvalue = norm.sf(zscore)
    return {
        "score": score,
        "se": se,
        "pvalue": pvalue,
        "reject_null": pvalue < ALPHA,
        "margin": 0.0,
        "margin_method": "fixed",
        "variance_floor_c": floor_c,
        "ueifs": ueifs,
    }


def _crossfit_loco(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]],
    seed: int,
    n_estimators: int,
) -> dict[str, object]:
    """Reproduce the reference LOCO estimator on the shared held-out folds."""
    ueifs = np.zeros_like(X, dtype=float)
    for train_idx, test_idx, full_model in folds:
        base_loss = (y[test_idx] - full_model.predict(X[test_idx])) ** 2
        for feature in range(X.shape[1]):
            reduced = _fit_predictor(
                np.delete(X[train_idx], feature, axis=1),
                y[train_idx],
                seed,
                n_estimators,
            )
            reduced_prediction = reduced.predict(
                np.delete(X[test_idx], feature, axis=1)
            )
            ueifs[test_idx, feature] = (
                (y[test_idx] - reduced_prediction) ** 2 - base_loss
            )
    return _reference_inference(ueifs, y)


def _crossfit_conditional_cpi(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]],
    seed: int,
    nsamples: int,
) -> dict[str, object]:
    """Reproduce the reference conditional-residual CPI baseline.

    The published estimator fits ``X_j | X_-j`` on each held-out fold before
    permuting its residuals, so this function deliberately retains that exact
    behaviour rather than substituting a stricter train-fold-only variant.
    """
    ueifs = np.zeros_like(X, dtype=float)
    for _, test_idx, full_model in folds:
        X_test, y_test = X[test_idx], y[test_idx]
        base_loss = (y_test - full_model.predict(X_test)) ** 2
        n_test, dimension = X_test.shape
        for feature in range(dimension):
            X_minus = np.delete(X_test, feature, axis=1)
            conditional = make_pipeline(
                StandardScaler(), LassoLarsIC(criterion="bic")
            ).fit(X_minus, X_test[:, feature])
            fitted = conditional.predict(X_minus)
            residual = X_test[:, feature] - fitted
            losses = np.zeros((nsamples, n_test))
            for draw in range(nsamples):
                changed = X_test.copy()
                changed[:, feature] = fitted + shuffle(
                    residual,
                    random_state=seed + feature * 1_000_003 + draw,
                )
                losses[draw] = (y_test - full_model.predict(changed)) ** 2
            ueifs[test_idx, feature] = 0.5 * (losses - base_loss).mean(axis=0)
    return _reference_inference(ueifs, y)


def _fit_reference_flow(
    X_auxiliary: np.ndarray,
    seed: int,
    flow_steps: int,
):
    """Fit the auxiliary Flow model using the published Experiment 1 settings."""
    trainer = FlowExplainer(
        model=lambda values: np.zeros(len(values)),
        data=X_auxiliary,
        fit_flow=False,
        flow_training_seed=seed,
        compute_diagnostics=False,
        verbose=False,
        hidden_dim=128,
        time_embed_dim=64,
        num_blocks=1,
        use_bn=False,
    )
    trainer.fit_flow(
        num_steps=flow_steps,
        verbose=False,
        batch_size=min(256, len(X_auxiliary)),
        lr=1e-3,
        show_plot=False,
    )
    return trainer.flow_model


def _crossfit_transformed(
    family: str,
    resampling_version: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]],
    seed: int,
    nsamples: int,
    flow_model=None,
) -> dict[str, object]:
    """Cross-fit an OT, EOT, or Flow explainer and retain held-out UEIFs."""
    ueifs = np.zeros_like(X, dtype=float)
    for fold_number, (train_idx, test_idx, predictor) in enumerate(folds):
        common = {
            "model": predictor.predict,
            "data": X[train_idx],
            "nsamples": nsamples,
            "random_state": seed + fold_number,
            "method": resampling_version,
            "loss": "squared_error",
            "compute_diagnostics": False,
        }
        if family == "DFI-OT":
            explainer = OTExplainer(**common)
        elif family == "DFI-EOT":
            # EOT is a JSS extension, not a method in the original Experiment 1.
            # Retain the existing manuscript draft epsilon until that extension
            # receives its own sensitivity analysis.
            explainer = EOTExplainer(epsilon=0.001, **common)
        elif family == "FDFI":
            if flow_model is None:
                raise ValueError("FDFI requires the shared auxiliary flow model")
            explainer = FlowExplainer(
                flow_model=flow_model,
                fit_flow=False,
                verbose=False,
                jacobian_mode="avg_sq",
                jacobian_n_samples=len(test_idx),
                **common,
            )
            # The reference estimator resamples latent values encoded from the
            # current training fold while reusing the auxiliary-trained flow.
            explainer.Z_full = explainer._encode_to_Z(X[train_idx])
        else:
            raise ValueError(f"unknown DFI/FDFI family: {family}")
        explainer(X[test_idx], y=y[test_idx])
        ueifs[test_idx] = np.asarray(explainer.ueifs_X)
    return _reference_inference(ueifs, y)


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
    """Aggregate benchmark metrics, using C3 only for Type-I error."""
    successful = feature_results.loc[feature_results["status"] == "SUCCESS"].copy()
    keys = ["sweep", "n", "rho", "method", "resampling_version"]
    rows: list[dict[str, object]] = []
    for group_index, (group_keys, frame) in enumerate(successful.groupby(keys, sort=False)):
        per_rep = []
        for repetition, rep_frame in frame.groupby("repetition"):
            active_c1 = rep_frame.loc[rep_frame["truth_group"] == "active_C1"]
            correlated_c2 = rep_frame.loc[
                rep_frame["truth_group"] == "correlated_null_C2"
            ]
            independent_c3 = rep_frame.loc[
                rep_frame["truth_group"] == "independent_null_C3"
            ]
            if active_c1.empty or correlated_c2.empty or independent_c3.empty:
                raise ValueError(
                    "each successful repetition must contain nonempty C1, C2, and C3 sets"
                )
            auc_frame = pd.concat([active_c1, independent_c3], ignore_index=True)
            c1_c2 = pd.concat([active_c1, correlated_c2], ignore_index=True)
            per_rep.append({
                "repetition": repetition,
                "test_r2": rep_frame["test_r2"].iloc[0],
                "auc": roc_auc_score(
                    auc_frame["is_relevant"].astype(int), auc_frame["score"]
                ),
                "power_c1": active_c1["reject_null"].mean(),
                "type1_error": independent_c3["reject_null"].mean(),
                "power_c1_c2": c1_c2["reject_null"].mean(),
                "type1_null_rejections": int(independent_c3["reject_null"].sum()),
                "type1_null_tests": len(independent_c3),
                "runtime_seconds": rep_frame["runtime_seconds"].iloc[0],
            })
        rep_metrics = pd.DataFrame(per_rep)
        row = dict(zip(keys, group_keys))
        row["successful_repetitions"] = len(rep_metrics)
        row["type1_null_rejections"] = int(rep_metrics["type1_null_rejections"].sum())
        row["type1_null_tests"] = int(rep_metrics["type1_null_tests"].sum())
        row["type1_error_above_nominal"] = bool(
            row["type1_null_rejections"] / row["type1_null_tests"] >= ALPHA
        )
        for metric in (
            "auc", "power_c1", "type1_error", "power_c1_c2",
            "test_r2", "runtime_seconds",
        ):
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


def audit_type1_error(
    feature_results: pd.DataFrame,
    expected_repetitions: int,
    type1_null_features: tuple[int, ...],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Audit completeness and report empirical C3 Type-I error transparently.

    The nominal level is a diagnostic reference, not a post-hoc acceptance
    target.  Values at or above alpha are flagged for investigation but are not
    manipulated and do not make an otherwise complete run computationally fail.
    """
    keys = ["sweep", "n", "rho", "method", "resampling_version"]
    expected_tests = expected_repetitions * len(type1_null_features)
    rows: list[dict[str, object]] = []
    for group_keys, frame in feature_results.groupby(keys, sort=False):
        successful = frame.loc[frame["status"] == "SUCCESS"]
        null = successful.loc[successful["is_type1_null"]]
        seed_counts = {
            str(int(seed)): int(seed_frame["reject_null"].sum())
            for seed, seed_frame in null.groupby("seed", sort=True)
        }
        failed_seeds = sorted({
            int(seed) for seed in frame.loc[frame["status"] != "SUCCESS", "seed"]
        })
        observed_tests = len(null)
        rejections = int(null["reject_null"].sum())
        type1_error = rejections / observed_tests if observed_tests else np.nan
        complete = observed_tests == expected_tests and not failed_seeds
        above_nominal = bool(complete and type1_error >= alpha)
        status = "COMPLETE" if complete else "INCOMPLETE"
        if above_nominal:
            investigation = (
                "PENDING: empirical C3 Type-I error is at or above the nominal "
                "level; inspect the implementation and per-seed raw rows without "
                "post-hoc adjustment."
            )
        elif not complete:
            investigation = (
                "PENDING: one or more expected method repetitions failed or are missing."
            )
        else:
            investigation = "Not required."
        row = dict(zip(keys, group_keys))
        row.update({
            "null_set": "C3 independent null features",
            "null_feature_indices": json.dumps(list(type1_null_features)),
            "expected_repetitions": expected_repetitions,
            "successful_repetitions": len(seed_counts),
            "expected_null_tests": expected_tests,
            "observed_null_tests": observed_tests,
            "null_rejections": rejections,
            "type1_error": type1_error,
            "nominal_alpha": alpha,
            "interpretation_rule": (
                f"report against nominal alpha={alpha}; do not force the empirical value"
            ),
            "above_nominal": above_nominal,
            "status": status,
            "rejection_counts_by_seed": json.dumps(seed_counts, sort_keys=True),
            "failed_seeds": json.dumps(failed_seeds),
            "investigation": investigation,
        })
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
        "is_type1_null": False,
        "truth_group": "not_available",
        "score": np.nan,
        "se": np.nan,
        "zscore": np.nan,
        "pvalue": np.nan,
        "reject_null": False,
        "margin": np.nan,
        "margin_method": "not_available",
        "test_r2": test_r2,
        "runtime_seconds": runtime,
        "status": "FAILED",
        "error": repr(error),
    }


def _runtime_loco_family(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]],
    seed: int,
    n_estimators: int,
) -> None:
    """Evaluate LOCO variants while counting their method-specific fits."""
    if method not in {"LOCO", "nLOCO", "dLOCO"}:
        raise ValueError(f"unknown LOCO runtime method: {method}")
    for fold_number, (train_idx, test_idx, full_model) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        base_loss = (y_test - full_model.predict(X_test)) ** 2
        for feature in range(X.shape[1]):
            X_train_minus = np.delete(X_train, feature, axis=1)
            X_test_minus = np.delete(X_test, feature, axis=1)
            reduced = _fit_predictor(
                X_train_minus,
                y_train,
                seed + fold_number * 10_000 + feature,
                n_estimators,
            )
            reduced_loss = (y_test - reduced.predict(X_test_minus)) ** 2
            if method == "LOCO":
                _ = float(np.mean(reduced_loss - base_loss))
                continue
            if method == "nLOCO":
                conditional = _fit_predictor(
                    X_train_minus,
                    X_train[:, feature],
                    seed + 100_000 + fold_number * 10_000 + feature,
                    n_estimators,
                )
                residual = X_test[:, feature] - conditional.predict(X_test_minus)
                cond_var = max(float(np.mean(residual**2)), 1e-8)
                _ = float(np.mean(reduced_loss - base_loss) / cond_var)
                continue

            # The public dLOCO estimator still fits the reduced models above,
            # then evaluates decorrelated prediction contrasts from the shared
            # full model.  Preserve that method-call cost here.
            rng = np.random.default_rng(seed + fold_number * 10_000 + feature)
            reference_ids = rng.choice(
                len(test_idx), size=min(1000, len(test_idx)), replace=False
            )
            original = full_model.predict(X_test[reference_ids])
            marginal = np.zeros(len(reference_ids), dtype=float)
            for batch_start in range(0, len(reference_ids), 32):
                batch_ids = reference_ids[batch_start:batch_start + 32]
                counterfactual = np.repeat(
                    X_test[batch_ids, None, :], len(test_idx), axis=1
                )
                counterfactual[:, :, feature] = X_test[:, feature]
                prediction = full_model.predict(
                    counterfactual.reshape(-1, X.shape[1])
                ).reshape(len(batch_ids), len(test_idx))
                marginal[batch_start:batch_start + len(batch_ids)] = prediction.mean(axis=1)
            variance = max(float(np.var(X_test[:, feature], ddof=1)), 1e-8)
            _ = float(np.mean((original - marginal) ** 2) / variance)


def _runtime_shapley(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    n_mc: int,
) -> None:
    """Evaluate the public weighted-coalition SHAP approximation at d=10."""
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        cache: dict[tuple[int, ...], object] = {}

        def fitted_model(features: tuple[int, ...]):
            if features in cache:
                return cache[features]
            if not features:
                model = float(y_train.mean())
            else:
                model = _fit_predictor(
                    X_train[:, features],
                    y_train,
                    seed + fold_number * 100_000 + sum(features) + len(features) * 1_003,
                    n_estimators,
                )
            cache[features] = model
            return model

        for feature in range(X.shape[1]):
            remaining = np.array([j for j in range(X.shape[1]) if j != feature])
            rng = np.random.default_rng(seed + fold_number * 10_000 + feature)
            contributions = []
            for _ in range(n_mc):
                size = int(rng.integers(0, X.shape[1]))
                subset = tuple(sorted(rng.choice(remaining, size=size, replace=False)))
                with_feature = tuple(sorted(subset + (feature,)))
                model_subset = fitted_model(subset)
                model_with = fitted_model(with_feature)
                if subset:
                    pred_subset = model_subset.predict(X_test[:, subset])
                else:
                    pred_subset = np.full(len(test_idx), model_subset)
                pred_with = model_with.predict(X_test[:, with_feature])
                contributions.append(float(np.mean((pred_with - pred_subset) ** 2)))
            _ = float(np.mean(contributions))


def _prepare_runtime_explainers(
    method: str,
    X: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray, RandomForestRegressor]],
    seed: int,
    cfg: dict[str, object],
) -> list[tuple[np.ndarray, object]]:
    """Build maps and train Flow outside the evaluate-only runtime timer."""
    if method not in {"OT", "EOT", "FDFI"}:
        raise ValueError(f"unknown transformed runtime method: {method}")
    flow_model = None
    if method == "FDFI":
        X_auxiliary, _ = generate_runtime_data(
            n=len(X),
            seed=seed + 7,
            dimension=X.shape[1],
            rho1=float(cfg["rho1"]),
            rho2=float(cfg["rho2"]),
            mix_weight=float(cfg["mix_weight"]),
        )
        flow_model = _fit_reference_flow(
            X_auxiliary, seed, int(cfg["flow_steps"])
        )

    prepared: list[tuple[np.ndarray, object]] = []
    for fold_number, (train_idx, test_idx, predictor) in enumerate(folds):
        common = {
            "model": predictor.predict,
            "data": X[train_idx],
            "nsamples": int(cfg["nsamples"]),
            "random_state": seed + fold_number,
            "method": "cpi",
            "loss": "squared_error",
            "compute_diagnostics": False,
        }
        if method == "OT":
            explainer = OTExplainer(**common)
        elif method == "EOT":
            explainer = EOTExplainer(epsilon=0.001, **common)
        else:
            explainer = FlowExplainer(
                flow_model=flow_model,
                fit_flow=False,
                verbose=False,
                jacobian_mode="avg_sq",
                jacobian_n_samples=len(test_idx),
                **common,
            )
            explainer.Z_full = explainer._encode_to_Z(X[train_idx])
        prepared.append((test_idx, explainer))
    return prepared


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace a CSV so interrupted experiment cells remain resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_or_validate_json_contract(payload: dict[str, object], path: Path) -> None:
    """Create an immutable run contract or reject an incompatible resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"existing run settings do not match the requested configuration: {path}"
            )
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _benchmark_scenario_complete(
    frame: pd.DataFrame,
    dimension: int,
) -> bool:
    """Return whether one persisted sweep/scenario contains every successful cell."""
    if frame.empty or len(frame) != dimension * (
        len(BASELINE_METHODS) + len(FDFI_FAMILIES) * len(RESAMPLING_VERSIONS)
    ):
        return False
    expected_features = set(range(dimension))
    cells = [(method, "baseline") for method in BASELINE_METHODS]
    cells.extend(
        (family, version)
        for family in FDFI_FAMILIES
        for version in RESAMPLING_VERSIONS
    )
    for method, version in cells:
        cell = frame.loc[
            (frame["method"] == method)
            & (frame["resampling_version"] == version)
        ]
        if (
            len(cell) != dimension
            or set(cell["status"]) != {"SUCCESS"}
            or set(cell["feature"].astype(int)) != expected_features
        ):
            return False
    return True


def _benchmark_mask(
    frame: pd.DataFrame,
    sweep: str,
    n: int,
    rho: float,
    seed: int,
) -> pd.Series:
    """Select one persisted benchmark sweep/scenario robustly across CSV reloads."""
    return (
        (frame["sweep"] == sweep)
        & (frame["n"] == n)
        & np.isclose(frame["rho"].astype(float), rho)
        & (frame["seed"] == seed)
    )


def summarize_runtime(runtime_results: pd.DataFrame) -> pd.DataFrame:
    """Return D3 plotting statistics: mean runtime and SD over seeds."""
    successful = runtime_results.loc[runtime_results["status"] == "SUCCESS"].copy()
    rows = []
    for (n, method), frame in successful.groupby(["n", "method"], sort=False):
        values = frame["runtime_seconds"].to_numpy(dtype=float)
        rows.append({
            "n": int(n),
            "method": method,
            "dimension": int(frame["dimension"].iloc[0]),
            "mean_runtime_seconds": float(values.mean()),
            "std_runtime_seconds": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "successful_repetitions": len(values),
        })
    return pd.DataFrame(rows, columns=[
        "n", "method", "dimension", "mean_runtime_seconds",
        "std_runtime_seconds", "successful_repetitions",
    ])


def run_runtime_experiment(
    mode: str,
    results_dir: Path,
    tables_dir: Path,
    metadata_dir: Path,
) -> list[Path]:
    """Run or resume every Experiment 2 sample-size x seed x method cell."""
    cfg = runtime_settings(mode)
    metadata_path = metadata_dir / "simulation_runtime_settings.json"
    _write_or_validate_json_contract(cfg, metadata_path)
    result_path = results_dir / "simulation_runtime_results.csv"
    summary_path = tables_dir / "simulation_runtime_summary.csv"
    columns = [
        "n", "repetition", "seed", "method", "dimension", "runtime_seconds",
        "status", "error", "timing_scope",
    ]
    if result_path.exists():
        runtime_results = pd.read_csv(result_path)
        missing = set(columns).difference(runtime_results.columns)
        if missing:
            raise ValueError(f"runtime checkpoint is missing columns: {sorted(missing)}")
    else:
        runtime_results = pd.DataFrame(columns=columns)

    for n in cfg["n_values"]:
        for repetition in range(int(cfg["repetitions"])):
            seed = int(cfg["seed_schedule"][repetition])
            completed = runtime_results.loc[
                (runtime_results["n"] == int(n))
                & (runtime_results["seed"] == seed)
                & (runtime_results["status"] == "SUCCESS"),
                "method",
            ]
            pending_methods = [
                method for method in cfg["methods"] if method not in set(completed)
            ]
            if not pending_methods:
                continue
            seed_everything(seed)
            dimension = int(cfg["dimension_non_shap"])
            X = y = folds = None
            if any(method != "SHAP" for method in pending_methods):
                X, y = generate_runtime_data(
                    int(n), seed, dimension, float(cfg["rho1"]),
                    float(cfg["rho2"]), float(cfg["mix_weight"]),
                )
                folds, _ = _fit_crossfit_predictors(
                    X, y, seed, int(cfg["predictor"]["n_estimators"]), N_FOLDS
                )

            for method in pending_methods:
                cell_dimension = int(cfg["dimension_shap"]) if method == "SHAP" else dimension
                key_mask = (
                    (runtime_results["n"] == int(n))
                    & (runtime_results["seed"] == seed)
                    & (runtime_results["method"] == method)
                )
                runtime_results = runtime_results.loc[~key_mask].copy()
                started = None
                try:
                    if method == "SHAP":
                        X_method, y_method = generate_runtime_data(
                            int(n), seed, cell_dimension, float(cfg["rho1"]),
                            float(cfg["rho2"]), float(cfg["mix_weight"]),
                        )
                        started = time.perf_counter()
                        _runtime_shapley(
                            X_method, y_method, seed,
                            int(cfg["predictor"]["n_estimators"]),
                            int(cfg["shapley_mc_draws"]),
                        )
                    elif method in {"LOCO", "nLOCO", "dLOCO"}:
                        started = time.perf_counter()
                        _runtime_loco_family(
                            method, X, y, folds, seed,
                            int(cfg["predictor"]["n_estimators"]),
                        )
                    elif method == "CPI":
                        started = time.perf_counter()
                        _crossfit_conditional_cpi(
                            X, y, folds, seed, int(cfg["nsamples"])
                        )
                    else:
                        prepared = _prepare_runtime_explainers(
                            method, X, folds, seed, cfg
                        )
                        started = time.perf_counter()
                        for test_idx, explainer in prepared:
                            explainer(X[test_idx], y=y[test_idx])
                    elapsed = time.perf_counter() - started
                    status, error = "SUCCESS", ""
                except Exception as exc:
                    elapsed = np.nan if started is None else time.perf_counter() - started
                    status, error = "FAILED", repr(exc)
                row = pd.DataFrame([{
                    "n": int(n), "repetition": repetition, "seed": seed,
                    "method": method, "dimension": cell_dimension,
                    "runtime_seconds": elapsed, "status": status, "error": error,
                    "timing_scope": "evaluate_only",
                }], columns=columns)
                if runtime_results.empty:
                    runtime_results = row
                else:
                    runtime_results = pd.concat(
                        [runtime_results, row], ignore_index=True
                    )
                _atomic_write_csv(runtime_results, result_path)

    runtime_summary = summarize_runtime(runtime_results)
    _atomic_write_csv(runtime_summary, summary_path)
    return [metadata_path, result_path, summary_path]


def run_experiment(mode: str, results_dir: Path, tables_dir: Path, metadata_dir: Path) -> list[Path]:
    """Run or resume both benchmark sweeps and the separate runtime study."""
    cfg = settings(mode)
    metadata_path = metadata_dir / "simulation_settings.json"
    _write_or_validate_json_contract(cfg, metadata_path)
    result_path = results_dir / "simulation_feature_results.csv"
    summary_path = tables_dir / "simulation_benchmark_summary.csv"
    type1_audit_path = tables_dir / "simulation_type1_error_audit.csv"
    bootstrap_metadata_path: Path | None = None
    source_checkpoint = os.environ.get("FDFI_ONE_SEED_SOURCE_CHECKPOINT")
    if mode == "one_seed" and source_checkpoint and not result_path.exists():
        source_path = Path(source_checkpoint).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"one-seed source checkpoint does not exist: {source_path}")
        source_results = pd.read_csv(source_path)
        missing = set(BENCHMARK_RESULT_COLUMNS).difference(source_results.columns)
        if missing:
            raise ValueError(
                f"one-seed source checkpoint is missing columns: {sorted(missing)}"
            )
        allowed = {
            (str(item["sweep"]), int(item["n"]), float(item["rho"]))
            for item in _scenarios(cfg)
        }
        first_seed = int(cfg["seed_schedule"][0])
        imported_groups: list[pd.DataFrame] = []
        for (sweep, n, rho, seed), group in source_results.groupby(
            ["sweep", "n", "rho", "seed"], sort=False
        ):
            key = (str(sweep), int(n), float(rho))
            if int(seed) != first_seed or key not in allowed:
                continue
            group = group.loc[:, list(BENCHMARK_RESULT_COLUMNS)].copy()
            if _benchmark_scenario_complete(group, int(cfg["dimension"])):
                imported_groups.append(group)
        imported = (
            pd.concat(imported_groups, ignore_index=True)
            if imported_groups
            else pd.DataFrame(columns=BENCHMARK_RESULT_COLUMNS)
        )
        _atomic_write_csv(imported, result_path)
        bootstrap_metadata_path = metadata_dir / "one_seed_checkpoint_import.json"
        _write_or_validate_json_contract(
            {
                "source_path": str(source_path),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "selected_seed": first_seed,
                "imported_rows": int(len(imported)),
                "imported_complete_scenarios": int(len(imported_groups)),
                "policy": "complete compatible first-seed scenarios only",
            },
            bootstrap_metadata_path,
        )
    if result_path.exists():
        feature_results = pd.read_csv(result_path)
        missing = set(BENCHMARK_RESULT_COLUMNS).difference(feature_results.columns)
        if missing:
            raise ValueError(
                f"benchmark checkpoint is missing columns: {sorted(missing)}"
            )
        feature_results = feature_results.loc[:, list(BENCHMARK_RESULT_COLUMNS)]
    else:
        feature_results = pd.DataFrame(columns=BENCHMARK_RESULT_COLUMNS)

    dimension = int(cfg["dimension"])
    scenario_cache: dict[tuple[int, float, int], pd.DataFrame] = {}
    if not feature_results.empty:
        for (n, rho, seed), group in feature_results.groupby(
            ["n", "rho", "seed"], sort=False
        ):
            for _, sweep_group in group.groupby("sweep", sort=False):
                if _benchmark_scenario_complete(sweep_group, dimension):
                    scenario_cache[(int(n), float(rho), int(seed))] = (
                        sweep_group.copy()
                    )
                    break

    for scenario in _scenarios(cfg):
        for repetition in range(int(cfg["repetitions"])):
            seed = int(cfg["seed_schedule"][repetition])
            cache_key = (int(scenario["n"]), float(scenario["rho"]), seed)
            target_mask = _benchmark_mask(
                feature_results,
                str(scenario["sweep"]),
                int(scenario["n"]),
                float(scenario["rho"]),
                seed,
            )
            target = feature_results.loc[target_mask]
            if _benchmark_scenario_complete(target, dimension):
                scenario_cache.setdefault(cache_key, target.copy())
                continue
            if cache_key in scenario_cache:
                copied = scenario_cache[cache_key].copy()
                copied["sweep"] = str(scenario["sweep"])
                copied["repetition"] = repetition
                remaining = feature_results.loc[~target_mask]
                feature_results = (
                    copied.reset_index(drop=True)
                    if remaining.empty
                    else pd.concat([remaining, copied], ignore_index=True)
                )
                _atomic_write_csv(feature_results, result_path)
                continue

            scenario_rows: list[dict[str, object]] = []
            seed_everything(seed)
            X, y, truth_groups = generate_data(
                n=int(scenario["n"]), rho=float(scenario["rho"]), seed=seed,
                dimension=int(cfg["dimension"]), block_size=int(cfg["block_size"]),
                active_features=tuple(cfg["active_features"]),
                noise_sd=float(cfg["noise_sd"]),
            )
            predictor_started = time.perf_counter()
            folds, test_r2 = _fit_crossfit_predictors(
                X,
                y,
                seed,
                int(cfg["predictor"]["n_estimators"]),
                int(cfg["n_folds"]),
            )
            predictor_runtime = time.perf_counter() - predictor_started

            for method in BASELINE_METHODS:
                started = time.perf_counter()
                try:
                    if method == "LOCO":
                        inference = _crossfit_loco(
                            X,
                            y,
                            folds,
                            seed,
                            int(cfg["predictor"]["n_estimators"]),
                        )
                    else:
                        inference = _crossfit_conditional_cpi(
                            X, y, folds, seed, int(cfg["nsamples"])
                        )
                    runtime = predictor_runtime + time.perf_counter() - started
                    scenario_rows.extend(_feature_rows(
                        method, "baseline",
                        np.asarray(inference["score"]),
                        np.asarray(inference["se"]),
                        np.asarray(inference["pvalue"]),
                        np.asarray(inference["reject_null"]),
                        truth_groups, scenario,
                        repetition, seed, test_r2, runtime,
                        margin=float(inference["margin"]),
                        margin_method=str(inference["margin_method"]),
                    ))
                except Exception as exc:
                    scenario_rows.append(_failure_row(
                        scenario, repetition, seed, method, "baseline", test_r2,
                        time.perf_counter() - started, exc,
                    ))

            flow_model = None
            flow_runtime = 0.0
            flow_error = None
            if "FDFI" in FDFI_FAMILIES:
                flow_started = time.perf_counter()
                try:
                    X_auxiliary, _, _ = generate_data(
                        n=int(cfg["flow_auxiliary_n"]),
                        rho=float(scenario["rho"]),
                        seed=seed,
                        dimension=int(cfg["dimension"]),
                        block_size=int(cfg["block_size"]),
                        active_features=tuple(cfg["active_features"]),
                        noise_sd=float(cfg["noise_sd"]),
                    )
                    flow_model = _fit_reference_flow(
                        X_auxiliary, seed, int(cfg["flow_steps"])
                    )
                except Exception as exc:
                    flow_error = exc
                flow_runtime = time.perf_counter() - flow_started

            for family in FDFI_FAMILIES:
                for version in RESAMPLING_VERSIONS:
                    started = time.perf_counter()
                    try:
                        if family == "FDFI" and flow_error is not None:
                            raise RuntimeError(
                                "auxiliary flow training failed"
                            ) from flow_error
                        inference = _crossfit_transformed(
                            family,
                            version,
                            X,
                            y,
                            folds,
                            seed,
                            int(cfg["nsamples"]),
                            flow_model=flow_model,
                        )
                        runtime = predictor_runtime + time.perf_counter() - started
                        if family == "FDFI":
                            runtime += flow_runtime
                        scenario_rows.extend(_feature_rows(
                            family, version,
                            np.asarray(inference["score"]),
                            np.asarray(inference["se"]),
                            np.asarray(inference["pvalue"]),
                            np.asarray(inference["reject_null"]),
                            truth_groups, scenario, repetition, seed,
                            test_r2, runtime, margin=float(inference["margin"]),
                            margin_method=str(inference["margin_method"]),
                        ))
                    except Exception as exc:
                        scenario_rows.append(_failure_row(
                            scenario, repetition, seed, family, version, test_r2,
                            time.perf_counter() - started, exc,
                        ))
            computed = pd.DataFrame(
                scenario_rows, columns=BENCHMARK_RESULT_COLUMNS
            )
            remaining = feature_results.loc[~target_mask]
            feature_results = (
                computed.reset_index(drop=True)
                if remaining.empty
                else pd.concat([remaining, computed], ignore_index=True)
            )
            _atomic_write_csv(feature_results, result_path)
            if _benchmark_scenario_complete(computed, dimension):
                scenario_cache[cache_key] = computed.copy()

    summary = summarize(feature_results, int(cfg["bootstrap_draws"]))
    type1_null_features = tuple(int(i) for i in cfg["truth_sets"]["independent_null_C3"])
    type1_audit = audit_type1_error(
        feature_results,
        expected_repetitions=int(cfg["repetitions"]),
        type1_null_features=type1_null_features,
    )
    _atomic_write_csv(feature_results, result_path)
    _atomic_write_csv(summary, summary_path)
    _atomic_write_csv(type1_audit, type1_audit_path)
    artifacts = [metadata_path, result_path, summary_path, type1_audit_path]
    if bootstrap_metadata_path is not None:
        artifacts.append(bootstrap_metadata_path)
    artifacts.extend(run_runtime_experiment(mode, results_dir, tables_dir, metadata_dir))
    return artifacts
