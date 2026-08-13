"""Standalone UCI CTG case study using :class:`FlowExplainer`."""

from __future__ import annotations

import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (CTG_NOTEBOOK, Mode, DATA_DIR, RunConfig, WorkflowBlocked,
                     WorkflowResult, relative, set_random_seeds, sha256_file)
from .fetch_ctg import SHA256 as CTG_SHA256

DATA_MODEL_SEED = 42
FLOW_SEED = 0
LOSS = "squared_error"
METHOD = "cpi"
FULL_NSAMPLES = 50
FULL_FLOW_STEPS = 5000
FLOW_BATCH_SIZE = 512
FLOW_LEARNING_RATE = 5e-4
FLOW_DEQUANTIZE_NOISE = 0.0
FLOW_SOLVER = {"method": "dopri5", "rtol": 1e-3, "atol": 1e-5}
DIAGNOSTICS_SOLVER = {"method": "dopri5", "rtol": 1e-6, "atol": 1e-8}
FEATURE_INFERENCE = {"alpha": .05, "target": "X", "alternative": "greater",
                     "multitest_method": "fdr_bh", "threshold_null": True,
                     "var_floor_c": .1, "var_floor_method": "mixture",
                     "var_floor_quantile": .95, "margin": 0.0,
                     "margin_method": "auto", "margin_quantile": .95, "verbose": False}
GROUP_INFERENCE = dict(FEATURE_INFERENCE)
NOTEBOOK_EXPECTED = {
    "observations": 2126,
    "features": 21,
    "feature_rejections": 18,
    "group_rejections": 1,
    "distribution_fidelity_mmd_display": .031184,
    "latent_independence_median_display": .092509,
}
CTG_PATH = DATA_DIR / "CTG.xls"
FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
            "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
            "Median", "Variance", "Tendency"]
CLINICAL_GROUPS = {
    "FHR baseline & variability": ["LB", "ASTV", "MSTV", "ALTV", "MLTV", "Variance"],
    "Accelerations & decelerations": ["AC", "FM", "UC", "DL", "DS", "DP"],
    "Histogram morphology": ["Width", "Min", "Max", "Nmax", "Nzeros", "Tendency"],
    "FHR central tendency": ["Mode", "Mean", "Median"],
}
GROUP_COLOUR = {
    "LB": "#4878a8", "ASTV": "#4878a8", "MSTV": "#4878a8", "ALTV": "#4878a8",
    "MLTV": "#4878a8", "Variance": "#4878a8", "AC": "#e07b54", "FM": "#e07b54",
    "UC": "#e07b54", "DL": "#e07b54", "DS": "#e07b54", "DP": "#e07b54",
    "Width": "#3a9e8c", "Min": "#3a9e8c", "Max": "#3a9e8c", "Nmax": "#3a9e8c",
    "Nzeros": "#3a9e8c", "Tendency": "#3a9e8c", "Mode": "#c9a227",
    "Mean": "#c9a227", "Median": "#c9a227",
}


def _load(path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    source = path or CTG_PATH
    if not source.is_file():
        raise WorkflowBlocked(f"required CTG input missing: {relative(source)}; run python replication/scripts/fetch_ctg.py")
    actual = sha256_file(source)
    if actual != CTG_SHA256:
        raise WorkflowBlocked(f"CTG checksum mismatch at {relative(source)}: expected {CTG_SHA256}, got {actual}")
    raw = pd.read_excel(source, sheet_name="Data", skiprows=1, header=0)
    return clean_ctg_frame(raw)


def clean_ctg_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the source notebook's fixed column selection and outcome rule."""
    frame = pd.concat([raw.iloc[:, 10:31].copy(), raw["NSP"].copy()], axis=1)
    frame.columns = FEATURES + ["NSP"]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame["NSP"] = frame["NSP"].astype(int)
    return frame[FEATURES], (frame["NSP"] != 1).astype(int).to_numpy()


def optional_summary_axis(values: np.ndarray) -> dict[str, float | str]:
    """Return the notebook's deterministic log/symlog axis decision."""
    finite = np.asarray(values)[np.isfinite(values)]
    negative_fraction = float(np.mean(finite < 0)) if finite.size else np.nan
    if negative_fraction <= 0.001:
        positive = finite[finite > 0]
        xmin = max(float(np.nanpercentile(positive, 1)), 1e-6) if positive.size else 1e-6
        xmax = float(np.nanpercentile(positive, 99.5)) * 1.2 if positive.size else 1.0
        return {"scale": "log", "xmin": xmin, "xmax": xmax,
                "negative_fraction": negative_fraction}
    linthresh = max(float(np.nanpercentile(np.abs(finite), 5)), 1e-6) if finite.size else 1e-6
    lo = float(np.nanpercentile(finite, .5)) if finite.size else -1e-6
    hi = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    return {"scale": "symlog", "linthresh": linthresh, "xmin": lo * 1.2,
            "xmax": hi * 1.2, "negative_fraction": negative_fraction}


def _intended_settings(mode: Mode) -> dict:
    notebook_path = CTG_NOTEBOOK
    flow_steps = FULL_FLOW_STEPS if mode == "full" else 3
    nsamples = FULL_NSAMPLES if mode == "full" else 1
    return {
        "configuration_status": "INTENDED",
        "intended": {
            "workflow": "CTG Flow case study", "mode": mode,
            "smoke_test_only": mode == "quick",
            "data": {"path": relative(CTG_PATH),
                     "sha256": sha256_file(CTG_PATH) if CTG_PATH.is_file() else None,
                     "sheet": "Data", "skiprows": 1, "raw_shape": [2129, 46],
                     "raw_feature_columns_zero_based": [10, 30],
                     "cleaning": "numeric coercion then complete-case drop",
                     "cleaned_shape": [2126, 21], "features": FEATURES,
                     "outcome": "Other = 1 when NSP != 1; Normal = 0 when NSP == 1",
                     "class_counts": {"Normal": 1655, "Other": 471}},
            "source_notebook": {"path": relative(notebook_path), "sha256": sha256_file(notebook_path)},
            "analysis_rule": "full data fitting and full data explanation; no train/test split",
            "preprocessing": "StandardScaler fitted on complete cleaned feature matrix",
            "random_forest": {"class": "RandomForestClassifier",
                "n_estimators": 300 if mode == "full" else 30, "max_depth": None,
                "min_samples_leaf": 3, "n_jobs": -1 if mode == "full" else 1,
                "random_state": DATA_MODEL_SEED},
            "flow_explainer": {"random_state": FLOW_SEED, "flow_training_seed": FLOW_SEED,
                "nsamples": nsamples, "sampling_method": "resample",
                "loss": LOSS, "method": METHOD,
                "configuration_source": "verified FDFI 0.0.10 notebook baseline",
                "architecture": {"hidden_dim": 64, "time_embed_dim": 32, "num_blocks": 1,
                    "use_bn": False, "sigma_min_implementation_default": .01,
                    "sigma_min_source": "fdfi.models.FlowMatchingModel.__init__"},
                "training": {"steps": flow_steps, "optimizer": "torch.optim.Adam",
                    "learning_rate": FLOW_LEARNING_RATE, "batch_size": FLOW_BATCH_SIZE,
                    "dequantize_noise": FLOW_DEQUANTIZE_NOISE},
                "ode_solver": FLOW_SOLVER, "diagnostics_solver": DIAGNOSTICS_SOLVER,
                "attribution_strategy": "single full-data call, matching the fixed notebook"},
            "feature_inference": FEATURE_INFERENCE,
            "group_inference": GROUP_INFERENCE,
            "groups": CLINICAL_GROUPS,
            "sorting": {"features": "descending score; stable first rank",
                        "groups": "descending score", "correlations": "descending absolute correlation; top 10"},
            "sampling_comparison": {"methods": ["resample", "permutation", "normal"],
                "observations": 20, "flow_configuration": "identical to main flow except sampling_method"},
        },
        "observed": None,
    }


def validate_ctg_artifacts(config: RunConfig) -> None:
    """Validate deterministic CTG table schemas before reporting success."""
    expected = {
        "ctg_class_distribution.csv": (2, ["class", "code", "count"]),
        "ctg_largest_correlations.csv": (10, ["feature_1", "feature_2", "correlation", "absolute_correlation"]),
        "ctg_classification_report.csv": (5, ["Unnamed: 0", "precision", "recall", "f1-score", "support"]),
        "ctg_feature_summary.csv": (21, ["feature", "score", "se", "ci_lower", "ci_upper",
                                                   "p_value", "p_value_adj", "reject_fdr", "ranking"]),
        "ctg_group_inference.csv": (4, ["group", "score", "se", "ci_lower", "ci_upper",
                                               "p_value", "p_value_adj", "reject_fdr"]),
        "ctg_sampling_methods.csv": (63, ["feature", "sampling_method", "phi_X"]),
    }
    for name, (rows, columns) in expected.items():
        frame = pd.read_csv(config.tables_dir / name)
        if frame.shape[0] != rows or list(frame.columns) != columns:
            raise ValueError(f"invalid CTG artifact {name}: shape={frame.shape}, columns={list(frame.columns)}")
        for column in frame.select_dtypes(include="object"):
            if frame[column].astype(str).str.match(r"^\s*[\[(].*[\])]\s*$").any():
                raise ValueError(f"object-array string found in {name}:{column}")
        numeric = frame.select_dtypes(include="number")
        if numeric.isna().any().any():
            raise ValueError(f"unexpected numeric NaN in {name}")
    diagnostics = pd.read_csv(config.tables_dir / "ctg_diagnostics.csv")
    if list(diagnostics.columns) != ["diagnostic", "value"] or diagnostics.empty:
        raise ValueError("invalid CTG diagnostics artifact")


def run(mode: Mode, config: RunConfig) -> WorkflowResult:
    started = datetime.now(timezone.utc)
    before = time.perf_counter()
    settings_path = config.metadata_dir / "ctg_settings.json"
    settings = _intended_settings(mode)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    set_random_seeds(DATA_MODEL_SEED)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from fdfi.explainers import FlowExplainer
    from fdfi.plots import correlation_heatmap, confidence_interval_plot, diagnostics_plot, summary_bar, summary_plot
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass

    X_df, y = _load()
    if X_df.shape != (2126, 21):
        raise ValueError(f"unexpected cleaned CTG dimensions: {X_df.shape}")
    X = StandardScaler().fit_transform(X_df)
    if mode == "quick":
        # Stratified deterministic subset, kept outside formal output paths.
        indices = np.r_[np.flatnonzero(y == 0)[:12], np.flatnonzero(y == 1)[:8]]
        X, y = X[indices], y[indices]
    model = RandomForestClassifier(n_estimators=300 if mode == "full" else 30,
                                   max_depth=None, min_samples_leaf=3,
                                   n_jobs=-1 if mode == "full" else 1,
                                   random_state=DATA_MODEL_SEED).fit(X, y)
    wrapper = lambda values: model.predict_proba(values)[:, 1]
    flow_steps = FULL_FLOW_STEPS if mode == "full" else 3
    nsamples = FULL_NSAMPLES if mode == "full" else 1
    explainer = FlowExplainer(wrapper, X, fit_flow=False, nsamples=nsamples, random_state=FLOW_SEED,
                              flow_training_seed=FLOW_SEED, num_steps=flow_steps,
                              loss=LOSS, method=METHOD,
                              verbose=False, hidden_dim=64, time_embed_dim=32,
                              num_blocks=1, use_bn=False, flow_solver_method="dopri5",
                              flow_solver_rtol=1e-3, flow_solver_atol=1e-5,
                              diagnostics_solver_method="dopri5",
                              diagnostics_solver_rtol=1e-6, diagnostics_solver_atol=1e-8)
    explainer.fit_flow(num_steps=flow_steps, verbose=False, batch_size=FLOW_BATCH_SIZE,
                       lr=FLOW_LEARNING_RATE, dequantize_noise=FLOW_DEQUANTIZE_NOISE)
    artifacts: list[str] = [relative(settings_path)]
    # Figure 1 uses unstandardized values, matching the source notebook.
    fig, _, _ = correlation_heatmap(X_df.to_numpy(), FEATURES,
                                    savepath=config.figures_dir / "fig1_correlation_matrix.pdf")
    plt.close(fig); artifacts.append(relative(config.figures_dir / "fig1_correlation_matrix.pdf"))
    diag_fig, _ = diagnostics_plot(explainer.diagnostics, feature_names=FEATURES,
                                   savepath=config.figures_dir / "fig4_flow_diagnostics.pdf")
    plt.close(diag_fig); artifacts.append(relative(config.figures_dir / "fig4_flow_diagnostics.pdf"))
    results = explainer(X)
    ci = explainer.conf_int(**FEATURE_INFERENCE)
    importance = pd.DataFrame({"feature": FEATURES, "score": ci["score"], "se": ci["se"],
                               "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
                               "p_value": ci["pvalue"], "p_value_adj": ci["pvalue_adj"],
                               "reject_fdr": ci["reject_null"]})
    importance["ranking"] = importance["score"].rank(ascending=False, method="first").astype(int)
    path = config.tables_dir / "ctg_feature_summary.csv"
    importance.sort_values("ranking").to_csv(path, index=False); artifacts.append(relative(path))
    fig, _, _ = summary_bar(ci["score"], ci["se"], FEATURES, group_colors=GROUP_COLOUR, max_display=21,
                            savepath=config.figures_dir / "fig2_fdfi_importance.pdf")
    plt.close(fig); artifacts.append(relative(config.figures_dir / "fig2_fdfi_importance.pdf"))
    order = np.argsort(-np.asarray(ci["score"]), kind="stable")
    ci_plot = dict(ci); ci_plot["ranking"] = np.empty_like(order); ci_plot["ranking"][order] = np.arange(1, 22)
    fig, _ = confidence_interval_plot(ci_plot, feature_names=FEATURES, max_display=20,
                                      savepath=config.figures_dir / "fig3_confidence_intervals.pdf")
    plt.close(fig); artifacts.append(relative(config.figures_dir / "fig3_confidence_intervals.pdf"))
    # Current manuscript/inventory includes this artifact, explicitly labelled optional.
    fig, _ = summary_plot(np.asarray(explainer.ueifs_X), features=X, feature_names=FEATURES,
                          max_display=21, show=False)
    axis = optional_summary_axis(np.asarray(explainer.ueifs_X))
    if axis["scale"] == "log":
        fig.axes[0].set_xscale("log")
        fig.axes[0].set_xlabel("Per-sample FDFI attribution value (log scale)")
    else:
        fig.axes[0].set_xscale("symlog", linthresh=float(axis["linthresh"]))
        fig.axes[0].set_xlabel("Per-sample FDFI attribution value (symlog scale)")
    fig.axes[0].set_xlim(float(axis["xmin"]), float(axis["xmax"]))
    optional_path = config.figures_dir / "fig_optional_summary_plot.pdf"
    fig.savefig(optional_path, dpi=150, bbox_inches="tight"); plt.close(fig); artifacts.append(relative(optional_path))

    counts = pd.Series(y).value_counts().sort_index()
    class_path = config.tables_dir / "ctg_class_distribution.csv"
    pd.DataFrame({"class": ["Normal", "Other"], "code": [0, 1],
                  "count": [int(counts.get(0, 0)), int(counts.get(1, 0))]}).to_csv(class_path, index=False)
    artifacts.append(relative(class_path))
    report_path = config.tables_dir / "ctg_classification_report.csv"
    pd.DataFrame(classification_report(y, model.predict(X), labels=[0, 1],
                                       target_names=["Normal", "Other"], output_dict=True)).T.to_csv(report_path)
    artifacts.append(relative(report_path))
    corr = X_df.corr(); pairs = []
    for i, left in enumerate(FEATURES):
        for right in FEATURES[i + 1:]:
            pairs.append((left, right, corr.loc[left, right], abs(corr.loc[left, right])))
    corr_path = config.tables_dir / "ctg_largest_correlations.csv"
    pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation", "absolute_correlation"]).sort_values(
        "absolute_correlation", ascending=False).head(10).to_csv(corr_path, index=False)
    artifacts.append(relative(corr_path))
    diag_path = config.tables_dir / "ctg_diagnostics.csv"
    pd.DataFrame([{"diagnostic": k, "value": v} for k, v in explainer.diagnostics.items() if np.isscalar(v)]).to_csv(diag_path, index=False)
    artifacts.append(relative(diag_path))
    group_indices = {name: [FEATURES.index(f) for f in members] for name, members in CLINICAL_GROUPS.items()}
    group_ci = explainer.conf_int(groups=group_indices, **GROUP_INFERENCE)
    group_path = config.tables_dir / "ctg_group_inference.csv"
    pd.DataFrame({"group": group_ci["groups"], "score": group_ci["score"], "se": group_ci["se"],
                  "ci_lower": group_ci["ci_lower"], "ci_upper": group_ci["ci_upper"],
                  "p_value": group_ci["pvalue"], "p_value_adj": group_ci["pvalue_adj"],
                  "reject_fdr": group_ci["reject_null"]}).sort_values("score", ascending=False).to_csv(group_path, index=False)
    artifacts.append(relative(group_path))
    # Sampling comparison retrains each flow exactly as in the notebook; reduce only smoke settings.
    comparison = []
    demo = X[:20]
    for method in ("resample", "permutation", "normal"):
        other = FlowExplainer(wrapper, X, fit_flow=False, sampling_method=method, nsamples=nsamples,
                              random_state=FLOW_SEED, flow_training_seed=FLOW_SEED,
                              loss=LOSS, method=METHOD,
                              num_steps=flow_steps, verbose=False, hidden_dim=64,
                              time_embed_dim=32, num_blocks=1, use_bn=False,
                              flow_solver_method=FLOW_SOLVER["method"],
                              flow_solver_rtol=FLOW_SOLVER["rtol"], flow_solver_atol=FLOW_SOLVER["atol"],
                              diagnostics_solver_method=DIAGNOSTICS_SOLVER["method"],
                              diagnostics_solver_rtol=DIAGNOSTICS_SOLVER["rtol"],
                              diagnostics_solver_atol=DIAGNOSTICS_SOLVER["atol"])
        other.fit_flow(num_steps=flow_steps, verbose=False, batch_size=FLOW_BATCH_SIZE,
                       lr=FLOW_LEARNING_RATE, dequantize_noise=FLOW_DEQUANTIZE_NOISE)
        values = other(demo)["phi_X"]
        comparison.extend({"feature": f, "sampling_method": method, "phi_X": value}
                          for f, value in zip(FEATURES, values))
    sampling_path = config.tables_dir / "ctg_sampling_methods.csv"
    pd.DataFrame(comparison).to_csv(sampling_path, index=False); artifacts.append(relative(sampling_path))
    key = {"observations": int(X.shape[0]), "features": 21,
           "full_data_accuracy": float(accuracy_score(y, model.predict(X))),
           "feature_rejections": int(np.sum(ci["reject_null"])),
           "group_rejections": int(np.sum(group_ci["reject_null"])),
           "distribution_fidelity_mmd": float(explainer.diagnostics["distribution_fidelity_mmd"]),
           "latent_independence_median": float(explainer.diagnostics["latent_independence_median"]),
           "flow_steps": flow_steps, "nsamples": nsamples,
           "data_model_seed": DATA_MODEL_SEED, "flow_seed": FLOW_SEED,
           "optional_summary_scale": axis["scale"],
           "optional_summary_negative_fraction": axis["negative_fraction"]}
    if mode == "full":
        for name in ("observations", "features", "feature_rejections", "group_rejections"):
            if key[name] != NOTEBOOK_EXPECTED[name]:
                raise ValueError(
                    f"CTG notebook-alignment failure for {name}: "
                    f"expected {NOTEBOOK_EXPECTED[name]}, got {key[name]}"
                )
        display_checks = {
            "distribution_fidelity_mmd": "distribution_fidelity_mmd_display",
            "latent_independence_median": "latent_independence_median_display",
        }
        for observed_name, expected_name in display_checks.items():
            if round(key[observed_name], 6) != NOTEBOOK_EXPECTED[expected_name]:
                raise ValueError(
                    f"CTG notebook-alignment failure for {observed_name}: "
                    f"expected {NOTEBOOK_EXPECTED[expected_name]:.6f} at notebook display "
                    f"precision, got {key[observed_name]:.6f}"
                )
    settings["configuration_status"] = "OBSERVED"
    settings["observed"] = {"analysis_observations": int(X.shape[0]), "features": int(X.shape[1]),
                            "class_counts": {"Normal": int(np.sum(y == 0)), "Other": int(np.sum(y == 1))},
                            "execution_device": str(getattr(explainer, "device", "CPU")),
                            "notebook_alignment": {
                                "expected": NOTEBOOK_EXPECTED,
                                "observed": {
                                    "observations": key["observations"],
                                    "features": key["features"],
                                    "feature_rejections": key["feature_rejections"],
                                    "group_rejections": key["group_rejections"],
                                    "distribution_fidelity_mmd_display":
                                        round(key["distribution_fidelity_mmd"], 6),
                                    "latent_independence_median_display":
                                        round(key["latent_independence_median"], 6),
                                },
                                "status": "PASS" if mode == "full" else "NOT_APPLICABLE_QUICK",
                            },
                            "diagnostics": {k: v for k, v in explainer.diagnostics.items() if np.isscalar(v)},
                            "status": "SUCCESS"}
    settings_path.write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
    validate_ctg_artifacts(config)
    ended = datetime.now(timezone.utc)
    return WorkflowResult(
        name="CTG Flow case study", status="SUCCESS", generated_files=artifacts, key_results=key,
        warnings=["No train/test split: current source notebook fits/explains all cleaned rows.",
                  "fig_optional_summary_plot.pdf is an optional inventory artifact."],
        seeds={"python": DATA_MODEL_SEED, "numpy": DATA_MODEL_SEED,
               "sklearn_random_forest": DATA_MODEL_SEED,
               "flow_random_state": FLOW_SEED, "flow_training_seed": FLOW_SEED},
        started_at=started.isoformat(), ended_at=ended.isoformat(),
        runtime_seconds=time.perf_counter() - before,
    )
