"""Standalone sens50 HIV case study using :class:`EOTExplainer`."""

from __future__ import annotations

import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (CASE_STUDY_DATA_DIR, EOT_NOTEBOOK, Mode, RunConfig,
                     WorkflowBlocked, WorkflowResult, relative, set_random_seeds,
                     sha256_file)

SEED = 0
LOSS = "squared_error"
METHOD = "cpi"
FEATURE_INFERENCE = {
    "alpha": .05,
    "var_floor_c": .1,
    "var_floor_method": "mixture",
    "var_floor_quantile": .95,
    "margin": 0.0,
    "margin_method": "auto",
    "margin_quantile": .95,
    "alternative": "two-sided",
    "verbose": False,
}
GROUP_INFERENCE = {
    "alpha": .05,
    "threshold_null": True,
    "var_floor_c": .1,
    "var_floor_method": "fixed",
    "margin": 0.0,
    "margin_method": "fixed",
    "alternative": "two-sided",
    "multitest_method": "bonferroni",
    "verbose": False,
}
NOTEBOOK_EXPECTED = {
    "phi_x_sum": 0.7396973904611562,
    "phi_z_sum": 0.7396975752007999,
    "x_feature_rejections": 6,
    "z_feature_rejections": 7,
    "x_group_rejections": 10,
    "z_group_rejections": 11,
}
DATA_PATH = CASE_STUDY_DATA_DIR / "sens50_processed_dataset.csv"
GROUP_PATH = CASE_STUDY_DATA_DIR / "feature_group.csv"
GROUP_MAPPING = pd.DataFrame({
    "group_num": range(1, 15),
    "group": ["vrc01", "cd4bs", "esa", "glyco", "covar", "pngs", "gp41",
              "pngs_novrc01", "subtype", "sequons", "geometry", "cysteines",
              "steric_bulk", "geog"],
    "description": ["VRC01 binding footprint", "CD4 binding sites",
                    "Sites with sufficient exposed surface area",
                    "Sites identified as important for glycosylation",
                    "Sites with residues that covary with\n"
                    "the VRC01 binding footprint",
                    "Sites associated with VRC01-specific\n"
                    "potential N-linked glycosylation (PNGS) effects",
                    "gp41 sites important for VRC01 binding",
                    "Sites for indicating N-linked glycosylation", "Majority virus subtypes",
                    "Region-specific counts of PNGS", "Viral geometry", "Cysteine counts",
                    "Steric bulk at critical locations", "Geographic confounders"],
})


def _frame(result: dict, names: list[str], name_column: str) -> pd.DataFrame:
    margin = np.asarray(result.get("margin", 0.0))
    scores = np.asarray(result["score"])
    z = np.asarray(result.get("zscore", (scores - margin) / np.maximum(result["se"], 1e-12)))
    frame = pd.DataFrame({name_column: names, "score": scores, "se": result["se"],
                          "zscore": z, "ci_lower": result["ci_lower"],
                          "ci_upper": result["ci_upper"], "p_value": result["pvalue"],
                          "p_value_adj": result.get("pvalue_adj", result["pvalue"]),
                          "reject_null": result["reject_null"]})
    if "ranking" in result:
        frame["ranking"] = np.asarray(result["ranking"], dtype=int)
    return frame


def select_quick_features(X_df: pd.DataFrame, groups: pd.DataFrame, budget: int = 60) -> list[str]:
    """Select at least one feature per group, then fill in source-column order."""
    selected: list[str] = []
    for group in groups.columns:
        candidates = groups.index[groups[group].astype(bool)].tolist()
        if not candidates:
            raise ValueError(f"biological group has no mapped feature: {group}")
        if candidates[0] not in selected:
            selected.append(candidates[0])
    for feature in X_df.columns:
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= budget:
            break
    subset_groups = groups.loc[selected]
    empty = subset_groups.columns[~subset_groups.astype(bool).any(axis=0)].tolist()
    if empty:
        raise ValueError(f"quick feature subset leaves empty groups: {empty}")
    return selected


def _plot_groups(frame: pd.DataFrame, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = frame.sort_values("score", ascending=True).merge(GROUP_MAPPING, on="group", how="left")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [3, 3, 3]})
    fig.suptitle("Sensitivity 50 (sens50) - EOT FDFI Group Importance", fontsize=22, y=1.04)
    colors = ["red" if reject else "gray" for reject in ordered["reject_null"]]
    positions = np.arange(len(ordered))
    axes[0].barh(positions, ordered["score"], xerr=ordered["se"],
                 color="steelblue", alpha=.7, error_kw={"capsize": 5})
    axes[0].set_yticks(positions); axes[0].set_yticklabels(ordered["group_num"], fontsize=12)
    axes[0].set_xlabel("Group Importance", fontsize=16)
    axes[0].set_ylabel("Group Number", fontsize=16)
    axes[0].set_title("Feature Group Importance (EOT FDFI)", fontsize=18)
    axes[0].grid(axis="x", alpha=.3)
    for position, (_, row) in enumerate(ordered.iterrows()):
        if row["reject_null"]:
            axes[0].text(row["score"] + row["se"] + .001, position, "*", va="center",
                         fontweight="bold", color="red")
    axes[1].barh(positions, ordered["zscore"], color=colors, alpha=.7)
    axes[1].set_yticks(positions); axes[1].set_yticklabels(ordered["group_num"], fontsize=12)
    axes[1].set_xlabel("Z-score", fontsize=16)
    axes[1].set_title("Feature Group Z-scores", fontsize=18)
    axes[1].grid(axis="x", alpha=.3)
    # This line is retained because the source notebook currently draws it.
    # Rejection colours still come from the Bonferroni-adjusted decision.
    axes[1].axvline(x=1.96, color="red", linestyle="--", linewidth=2,
                    alpha=.7, label="p=0.05")
    axes[1].legend(fontsize=12)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(-.5, len(ordered) - .5)
    for position, (_, row) in enumerate(ordered.iterrows()):
        axes[2].text(.02, position, f"{int(row['group_num']):2d}.", va="center",
                     ha="left", fontsize=14, fontweight="bold", color="black")
        axes[2].text(.09, position, row["description"], va="center", fontsize=11,
                     ha="left", color="red" if row["reject_null"] else "gray")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title("Feature Group Annotations", fontsize=18)
    for spine in axes[2].spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _write_notebook_summary_tables(
    explainer: object,
    group_dict: dict[str, list[int]],
    config: RunConfig,
) -> list[str]:
    """Save the public ``summary()`` output used by the source notebook."""
    feature_sections = []
    group_sections = []
    for target in ("X", "Z"):
        feature_summary = explainer.summary(
            target=target,
            print_output=False,
            **FEATURE_INFERENCE,
        )
        feature_sections.append(
            f"=== {target}-space feature summary ===\n{feature_summary}"
        )
        group_summary = explainer.summary(
            target=target,
            groups=group_dict,
            print_output=False,
            **GROUP_INFERENCE,
        )
        group_sections.append(
            f"=== {target}-space group summary ===\n{group_summary}"
        )

    feature_path = config.tables_dir / "sens50_eot_feature_summary.txt"
    group_path = config.tables_dir / "sens50_eot_group_summary.txt"
    feature_path.write_text("\n\n".join(feature_sections) + "\n", encoding="utf-8")
    group_path.write_text("\n\n".join(group_sections) + "\n", encoding="utf-8")
    return [relative(feature_path), relative(group_path)]


def run(mode: Mode, config: RunConfig) -> WorkflowResult:
    started_dt = datetime.now(timezone.utc)
    before = time.perf_counter()
    notebook_path = EOT_NOTEBOOK
    settings_path = config.metadata_dir / "sens50_settings.json"
    intended = {
        "workflow": "sens50 EOT case study", "mode": mode, "smoke_test_only": mode == "quick",
        "data": {"path": relative(DATA_PATH), "sha256": sha256_file(DATA_PATH) if DATA_PATH.is_file() else None,
                 "status": "processed", "expected_observations": 611,
                 "expected_predictors": 832, "outcome": "sens50"},
        "groups": {"path": relative(GROUP_PATH), "sha256": sha256_file(GROUP_PATH) if GROUP_PATH.is_file() else None,
                   "expected_count": 14, "coverage_required": True},
        "source_notebook": {"path": relative(notebook_path), "sha256": sha256_file(notebook_path)},
        "preprocessing": {"X": ["SimpleImputer(strategy=most_frequent)", "StandardScaler()"],
                          "y": "StandardScaler()"},
        "random_forest": {"class": "RandomForestRegressor", "n_estimators": 500 if mode == "full" else 30,
                          "max_depth": None, "min_samples_leaf": 5, "random_state": SEED, "n_jobs": -1},
        "explainer": {"class": "EOTExplainer", "nsamples": 50 if mode == "full" else 3,
                      "epsilon": .001, "auto_epsilon": False, "sampling_method": "resample",
                      "random_state": SEED, "loss": LOSS, "method": METHOD,
                      "regularize_implementation_default": 1e-6},
        "feature_inference": FEATURE_INFERENCE,
        "group_inference": GROUP_INFERENCE,
        "defaults_expanded": {"loss": LOSS, "method": METHOD,
                              "source": "verified FDFI 0.0.10 notebook baseline"},
        "seed": SEED,
    }
    settings = {"configuration_status": "INTENDED", "intended": intended, "observed": None}
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    for path in (DATA_PATH, GROUP_PATH):
        if not path.is_file():
            raise WorkflowBlocked(f"required sens50 input is missing: {relative(path)}")
    set_random_seeds(SEED)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from fdfi.explainers import EOTExplainer

    raw = pd.read_csv(DATA_PATH)
    if "sens50" not in raw:
        raise ValueError(f"sens50 response column missing from {relative(DATA_PATH)}")
    X_df = raw.drop(columns=["sens50"])
    y = raw["sens50"].astype(int).to_numpy()
    group_source = pd.read_csv(GROUP_PATH, index_col=0).set_index("feature")
    groups = pd.get_dummies(group_source["group"]).groupby(group_source.index).max().reindex(X_df.columns)
    if mode == "full" and (X_df.shape != (611, 832) or groups.shape[1] != 14):
        raise ValueError(f"unexpected sens50 dimensions: X={X_df.shape}, groups={groups.shape}")
    if groups.isna().any().any():
        missing = groups.index[groups.isna().any(axis=1)].tolist()[:10]
        raise ValueError(f"features missing biological group mapping: {missing}")

    if mode == "quick":
        selected = select_quick_features(X_df, groups, budget=60)
        X_df = X_df.loc[:, selected].iloc[:80]
        y = y[:80]
        groups = groups.loc[selected]
    preprocess = make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler())
    X = preprocess.fit_transform(X_df)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    rf = RandomForestRegressor(n_estimators=500 if mode == "full" else 30,
                               max_depth=None, min_samples_leaf=5,
                               random_state=SEED, n_jobs=-1)
    rf.fit(X, y_scaled)
    explainer = EOTExplainer(rf.predict, X, nsamples=50 if mode == "full" else 3,
                             epsilon=.001, sampling_method="resample", random_state=SEED,
                             loss=LOSS, method=METHOD)
    estimates = explainer(X, y=y_scaled)

    feature_frames: dict[str, pd.DataFrame] = {}
    artifacts: list[str] = []
    artifacts.append(relative(settings_path))
    for target in ("X", "Z"):
        result = explainer.conf_int(target=target, **FEATURE_INFERENCE)
        frame = _frame(result, list(X_df.columns), "feature")
        frame.insert(1, "space", target)
        path = config.tables_dir / f"sens50_eot_feature_inference_{target.lower()}.csv"
        frame.to_csv(path, index=False)
        artifacts.append(relative(path))
        feature_frames[target] = frame
    raw_frame = pd.DataFrame({"feature": X_df.columns,
                              "phi_X": estimates["phi_X"], "se_X": estimates["se_X"],
                              "phi_Z": estimates["phi_Z"], "se_Z": estimates["se_Z"]})
    raw_path = config.results_dir / "sens50_eot_raw_feature_results.csv"
    raw_frame.to_csv(raw_path, index=False)
    artifacts.append(relative(raw_path))

    group_frames = {}
    group_dict = {name: np.flatnonzero(groups[name].to_numpy()).tolist() for name in groups.columns}
    for target in ("X", "Z"):
        result = explainer.conf_int(target=target, groups=group_dict, **GROUP_INFERENCE)
        frame = _frame(result, list(result["groups"]), "group")
        frame.insert(1, "space", target)
        path = config.tables_dir / f"sens50_eot_group_summary_{target.lower()}.csv"
        frame.sort_values("ranking").to_csv(path, index=False)
        artifacts.append(relative(path))
        group_frames[target] = frame

    artifacts.extend(_write_notebook_summary_tables(explainer, group_dict, config))
    fig_path = config.figures_dir / "sens50_eot_group_importance.pdf"
    _plot_groups(group_frames["X"], fig_path)
    artifacts.append(relative(fig_path))
    mapping_path = config.tables_dir / "sens50_group_mapping.csv"
    GROUP_MAPPING.to_csv(mapping_path, index=False); artifacts.append(relative(mapping_path))
    key = {"observations": int(X.shape[0]), "predictors": int(X.shape[1]),
           "groups": len(group_dict), "rf_train_r2": float(rf.score(X, y_scaled)),
           "phi_x_sum": float(np.sum(estimates["phi_X"])),
           "phi_z_sum": float(np.sum(estimates["phi_Z"])),
           "x_feature_rejections": int(feature_frames["X"]["reject_null"].sum()),
           "z_feature_rejections": int(feature_frames["Z"]["reject_null"].sum()),
           "x_group_rejections": int(group_frames["X"]["reject_null"].sum()),
           "z_group_rejections": int(group_frames["Z"]["reject_null"].sum())}
    if mode == "full":
        for name in ("x_feature_rejections", "z_feature_rejections",
                     "x_group_rejections", "z_group_rejections"):
            if key[name] != NOTEBOOK_EXPECTED[name]:
                raise ValueError(
                    f"sens50 notebook-alignment failure for {name}: "
                    f"expected {NOTEBOOK_EXPECTED[name]}, got {key[name]}"
                )
        for name in ("phi_x_sum", "phi_z_sum"):
            if not np.isclose(key[name], NOTEBOOK_EXPECTED[name], rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"sens50 notebook-alignment failure for {name}: "
                    f"expected {NOTEBOOK_EXPECTED[name]:.16g}, got {key[name]:.16g}"
                )
    settings["configuration_status"] = "OBSERVED"
    settings["observed"] = {"observations": int(X.shape[0]), "predictors": int(X.shape[1]),
                            "groups": int(groups.shape[1]), "execution_device": "CPU",
                            "notebook_alignment": {
                                "expected": NOTEBOOK_EXPECTED,
                                "observed": {name: key[name] for name in NOTEBOOK_EXPECTED},
                                "status": "PASS" if mode == "full" else "NOT_APPLICABLE_QUICK",
                            },
                            "status": "SUCCESS"}
    settings_path.write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
    ended = datetime.now(timezone.utc)
    warnings = ["Quick results are smoke-test-only."] if mode == "quick" else []
    return WorkflowResult(
        name="sens50 EOT case study", status="SUCCESS", generated_files=artifacts,
        key_results=key, warnings=warnings,
        seeds={"python": SEED, "numpy": SEED, "sklearn": SEED, "eot": SEED},
        started_at=started_dt.isoformat(), ended_at=ended.isoformat(),
        runtime_seconds=time.perf_counter() - before,
    )
