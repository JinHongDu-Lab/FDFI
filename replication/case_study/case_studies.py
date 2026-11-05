"""Replication script for the case studies in
"fdfi: Disentangled Feature Importance in Python".

Case 1 (Section 4): EOTExplainer on the HIV sens50 data.
Case 2 (Section 5): FlowExplainer on the CTG data (Figures 3-7).

Run from the case_study/ folder:

    python case_studies.py

Figures are saved to figures/ and tables to outputs/.
"""

import os
import time
import warnings
from pathlib import Path

import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import fdfi
from fdfi.explainers import EOTExplainer, FlowExplainer
from fdfi.plots import (
    correlation_heatmap,
    diagnostics_plot,
    summary_bar,
    confidence_interval_plot,
    summary_plot,
)

warnings.filterwarnings("ignore")

# Software versions
print(f"fdfi {fdfi.__version__} from {fdfi.__file__}")
print(f"python {platform.python_version()}, numpy {np.__version__}, "
      f"pandas {pd.__version__}, scikit-learn {sklearn.__version__}")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Case 1 (Section 4): EOTExplainer on the HIV sens50 data
# =============================================================================

# Data and feature groups
DATA_DIR = Path("data")

DATASETS = {
    "sens50": {
        "file": DATA_DIR / "sens50_processed_dataset.csv",
        "target": "sens50",
    },
    "sens80": {
        "file": DATA_DIR / "sens80_processed_dataset.csv",
        "target": "sens80",
    },
    "ic50": {
        "file": DATA_DIR / "ic50.censored_processed_dataset.csv",
        "target": "ic50.censored",
    },
}

outcome = "sens50"

cfg = DATASETS[outcome]
filename = str(cfg["file"])

data = pd.read_csv(cfg["file"])
y = data[cfg["target"]].astype(int)
X = data.drop(columns=[cfg["target"]])

df_groups = pd.read_csv(DATA_DIR / "feature_group.csv", index_col=0).set_index("feature")
df_groups = pd.get_dummies(df_groups["group"]).groupby(df_groups.index).max().reindex(X.columns)

print(f"Dataset shape: {X.shape}")
print(f"Target variable: {outcome}")
print(f"Number of groups: {len(df_groups.columns)}")
print(f"Data loaded from: {filename}")

# Preprocessing
print("Data preparation...")

preprocess = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    StandardScaler(),
)

X_values = preprocess.fit_transform(X)
y_values = y.values.astype(float)

# Standardize the response
y_values = StandardScaler().fit_transform(
    y_values.reshape(-1, 1)).ravel()

print("Data preparation finished")
print(f"Feature matrix shape: {X_values.shape}")
print(f"y statistics: mean={y_values.mean():.3f}, std={y_values.std():.3f}")
print(f"Missing values - X: {np.isnan(X_values).sum()}, y: {np.isnan(y_values).sum()}")

# Prediction model
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    random_state=0,
    n_jobs=-1,
)
rf.fit(X_values, y_values)
model_fn = lambda X_: rf.predict(X_)
print(f"RandomForestRegressor fitted on full data")
print(f"Train R2: {rf.score(X_values, y_values):.3f}")

# EOTExplainer
start_time = time.time()

eot_explainer = EOTExplainer(
    model=model_fn,
    data=X_values,
    nsamples=50,
    epsilon=0.001,
    loss="squared_error",
    method="cpi",
    n_eta=10,
    sampling_method="resample",
    random_state=0,
)

eot_results = eot_explainer(X_values, y=y_values)

dfi_time = time.time() - start_time

print(f"EOT FDFI completed in {dfi_time:.2f} seconds")
print(f"DFI scores shape: {eot_results.phi_X.shape}")
print(eot_results.to_frame().head())

print("sum phi_X, sum phi_Z:", np.sum(eot_results.phi_X), np.sum(eot_results.phi_Z))

# Diagnostics (Appendix C.2)
eot_diag = eot_explainer.diagnostics
print(f"EOT MMD: {eot_diag['distribution_fidelity_mmd']:.6g}")
print(f"EOT median latent dCor: {eot_diag['latent_independence_median']:.6g}")

pd.DataFrame({
    "feature": X.columns,
    "phi_X": eot_results.phi_X, "se_X": eot_results.se_X,
    "phi_Z": eot_results.phi_Z, "se_Z": eot_results.se_Z,
}).to_csv(OUT_DIR / "case1_feature_scores.csv", index=False)

# Feature-level inference
ci_X = eot_explainer.conf_int(target="X")
ci_Z = eot_explainer.conf_int(target="Z")

print("=== X-space feature summary ===")
eot_explainer.summary(target="X")

print("\n=== Z-space feature summary ===")
eot_explainer.summary(target="Z")

# Group-level inference
group_ci = eot_explainer.conf_int(
    target="X",
    groups=df_groups,
    threshold_null=True,
    multitest_method="bonferroni",
)
res_df = (
    group_ci.to_frame()
    .reset_index()
    .rename(columns={"estimate": "importance", "pvalue": "p_value",
                     "pvalue_adj": "p_value_adj"})
)
res_df.to_csv(OUT_DIR / "case1_group_results_X.csv", index=False)

print("=== X-space group summary ===")
eot_explainer.summary(
    target="X",
    groups=df_groups,
    threshold_null=True,
    multitest_method="bonferroni",
)
print("\n=== Z-space group summary ===")
eot_explainer.summary(
    target="Z",
    groups=df_groups,
    threshold_null=True,
    multitest_method="bonferroni",
)

# Group labels
group_mapping = pd.DataFrame({
    "group_num": range(1, 15),
    "group": [
        "vrc01", "cd4bs", "esa", "glyco", "covar", "pngs", "gp41",
        "pngs_novrc01", "subtype", "sequons", "geometry", "cysteines",
        "steric_bulk", "geog"
    ],
    "description": [
        "VRC01 binding footprint",
        "CD4 binding sites",
        "Sites with sufficient exposed surface area",
        "Sites identified as important for glycosylation",
        "Sites with residues that covary with\nthe VRC01 binding footprint",
        "Sites associated with VRC01-specific\npotential N-linked glycosylation (PNGS) effects",
        "gp41 sites important for VRC01 binding",
        "Sites for indicating N-linked glycosylation",
        "Majority virus subtypes",
        "Region-specific counts of PNGS",
        "Viral geometry",
        "Cysteine counts",
        "Steric bulk at critical locations",
        "Geographic confounders",
    ],
})

# Group importance plot (supplementary)
DATASET_LABELS = {
    "sens50": "Sensitivity 50",
    "sens80": "Sensitivity 80",
    "ic50": "IC50 Censored",
}
dataset_label = DATASET_LABELS.get(outcome, outcome)

group_df_sorted = res_df.sort_values("importance", ascending=True).reset_index(drop=True)
group_df_sorted = group_df_sorted.merge(group_mapping, on="group", how="left")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 3, 3]})
fig.suptitle(
    f"{dataset_label} ({outcome}) - EOT FDFI Group Importance",
    fontsize=22,
    y=1.04,
)

group_numbers = list(group_df_sorted["group_num"])

ax1 = axes[0]
ax1.barh(
    range(len(group_df_sorted)),
    group_df_sorted["importance"],
    xerr=group_df_sorted["se"],
    error_kw=dict(capsize=5),
    alpha=0.7,
    color="steelblue",
)
ax1.set_yticks(range(len(group_df_sorted)))
ax1.set_yticklabels(group_numbers, fontsize=12)
ax1.set_xlabel("Group Importance", fontsize=16)
ax1.set_title("Feature Group Importance (EOT FDFI)", fontsize=18)
ax1.set_ylabel("Group Number", fontsize=16)
ax1.grid(axis="x", alpha=0.3)

for i, (_, row) in enumerate(group_df_sorted.iterrows()):
    if row["reject_null"]:
        ax1.text(
            row["importance"] + row["se"] + 0.001,
            i,
            "*",
            va="center",
            fontweight="bold",
            color="red",
        )

ax2 = axes[1]
colors = ["red" if reject_null else "gray" for reject_null in group_df_sorted["reject_null"]]
ax2.barh(range(len(group_df_sorted)), group_df_sorted["zscore"], alpha=0.7, color=colors)
ax2.set_yticks(range(len(group_df_sorted)))
ax2.set_yticklabels(group_numbers, fontsize=12)
ax2.set_xlabel("Z-score", fontsize=16)
ax2.set_title("Feature Group Z-scores", fontsize=18)
ax2.grid(axis="x", alpha=0.3)
ax2.axvline(x=1.96, color="red", linestyle="--", linewidth=2, alpha=0.7, label="p=0.05")
ax2.legend(fontsize=12)

ax3 = axes[2]
ax3.set_xlim(0, 1)
ax3.set_ylim(-0.5, len(group_df_sorted) - 0.5)

for i, (_, row) in enumerate(group_df_sorted.iterrows()):
    group_name = row["group"]
    display_name = row["description"] if pd.notna(row["description"]) else group_name
    text_color = "red" if row["reject_null"] else "gray"

    ax3.text(
        0.02,
        i,
        f"{int(row['group_num']):2d}.",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    ax3.text(
        0.09,
        i,
        display_name,
        va="center",
        ha="left",
        fontsize=12,
        color=text_color,
    )

ax3.set_yticks([])
ax3.set_xticks([])
ax3.set_title("Feature Group Annotations", fontsize=18)
for spine in ax3.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "case1_group_importance.pdf", bbox_inches="tight")
plt.close(fig)

# Ranked group results
print("=== Current ranked group-level results ===")
print(f"{'Group':<20} {'Importance':>12} {'SE':>10} {'Z-score':>10} {'Significant':>12}")
print("-" * 70)
for _, row in res_df.sort_values("zscore", ascending=False).iterrows():
    sig = "***" if row["reject_null"] else ""
    print(f"{row['group']:<20} {row['importance']:>12.4f} {row['se']:>10.4f} {row['zscore']:>10.2f} {sig:>12}")

n_sig = res_df["reject_null"].sum()
print(f"\nSignificant groups: {n_sig}/{len(res_df)} (Bonferroni alpha=0.05)")
print(f"Z-score range: {res_df['zscore'].min():.2f} to {res_df['zscore'].max():.2f}")


# =============================================================================
# Case 2 (Section 5): FlowExplainer on the CTG data
# =============================================================================

# Single-threaded PyTorch
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ctg_start = time.time()

# Data
CTG_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'
)

print('Downloading CTG dataset from UCI ...')
raw = pd.read_excel(CTG_URL, sheet_name='Data', skiprows=1, header=0)
print(f'  Raw shape (before cleaning): {raw.shape}')

X_raw = raw.iloc[:, 10:31].copy()
y_raw = raw['NSP'].copy()

data = pd.concat([X_raw, y_raw], axis=1)

feature_names = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
    'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
    'Median', 'Variance', 'Tendency'
]
data.columns = feature_names + ['NSP']

# Binary endpoint: Normal vs Other
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)
data['NSP'] = data['NSP'].astype(int)
data['Other'] = (data['NSP'] != 1).astype(int)

X = data[feature_names].values
y = data['Other'].values

print(f'  Cleaned shape : {data.shape}')
print(f'  Features ({len(feature_names)}): {feature_names}')
print('\n  Original target distribution (NSP):')
print(
    data['NSP']
    .value_counts()
    .sort_index()
    .rename({1: 'Normal (1)', 2: 'Suspect (2)', 3: 'Pathological (3)'})
)
print('\n  Binary target distribution:')
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .rename({0: 'Normal = 0', 1: 'Other = 1'})
)

# Figure 3
fig, ax, feature_names_reordered = correlation_heatmap(
    X, feature_names, show=False
)
fig.savefig(FIG_DIR / "fig3_ctg_correlation_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# Standardization
scaler = StandardScaler()
X_values = scaler.fit_transform(X)
y_values = y.astype(int)

X_train = X_values
X_explain = X_values
y_train = y_values

print(f'Analysis samples : {X_values.shape[0]}')
print(f'Analysis features: {X_values.shape[1]}')

# Prediction model
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model.fit(X_values, y_values)

y_pred = model.predict(X_values)
acc = accuracy_score(y_values, y_pred)
print(f'\nBlack-box full-data accuracy: {acc:.4f}')
print('\nClassification report (full data):')
print(classification_report(
    y_values,
    y_pred,
    labels=[0, 1],
    target_names=['Normal (0)', 'Other (1)'],
))

# FlowExplainer
def model_wrapper(X):
    """Return P(Other | X) for the fitted binary random forest."""
    return model.predict_proba(X)[:, 1]

explainer = FlowExplainer(
    model_wrapper,
    X_values,
    nsamples=50,
    loss='squared_error',
    method='cpi',
    random_state=0,
)

# Diagnostics (Figure 4)
diag = explainer.diagnostics

mmd_val = diag['distribution_fidelity_mmd']
dcor_val = diag['latent_independence_median']

print('=' * 55)
print('  FDFI Flow Diagnostics')
print('=' * 55)
print(f'  Distribution Fidelity   MMD        : {mmd_val:.6f}')
print(f'  Latent Independence     median dCor : {dcor_val:.6f}')
print('=' * 55)

mmd_status  = 'PASS' if mmd_val  < 0.05 else 'WARN - consider refitting flow'
dcor_status = 'PASS' if dcor_val < 0.10 else 'WARN - consider refitting flow'
print(f'\n  MMD  threshold < 0.05  ->  {mmd_status}')
print(f'  dCor threshold < 0.10  ->  {dcor_status}')

fig, axes = diagnostics_plot(
    explainer.diagnostics,
    feature_names=feature_names,
    show=False,
)
fig.savefig(FIG_DIR / "fig4_ctg_diagnostics.pdf", bbox_inches="tight")
plt.close(fig)

# Feature importance
print(f'Computing FDFI attributions for all {X_values.shape[0]} CTG instances ...')
print('Full-data binary distribution:')
print(pd.Series(y_values).value_counts().sort_index().rename({0: 'Normal = 0', 1: 'Other = 1'}))

results = explainer(X_values)

print(f"results.phi_X.shape: {results.phi_X.shape}")
print(f"results.se_X.shape: {results.se_X.shape}")
print(f"explainer.ueifs_X shape: {explainer.ueifs_X.shape}")

# Feature-level inference (Section 5.5)
ci = explainer.conf_int(
    alternative='greater',
    multitest_method='fdr_bh',
)

print('\n' + '=' * 80)
print('Feature Importance Results (Section 5.5)')
print('=' * 80)
print(
    ci.to_frame(feature_names=feature_names)[
        ['estimate', 'se', 'ci_lower', 'pvalue_adj', 'reject_null']
    ].round(4).to_string()
)

importance_df = (
    ci.to_frame(feature_names=feature_names)
    .reset_index()
    .rename(columns={'estimate': 'phi', 'reject_null': 'reject_fdr'})
    [['feature', 'phi', 'se', 'reject_fdr', 'pvalue_adj', 'ranking']]
    .sort_values('ranking')
)

print(f"\nSignificant features (FDR-adjusted): {len(ci.significant())}/{len(feature_names)}")
print(importance_df.head(10).to_string())
importance_df.to_csv(OUT_DIR / "case2_feature_results.csv", index=False)

# Figure 5, colored by clinical group
GROUP_COLOUR = {
    # FHR baseline & variability
    'LB':       '#4878a8',
    'ASTV':     '#4878a8',
    'MSTV':     '#4878a8',
    'ALTV':     '#4878a8',
    'MLTV':     '#4878a8',
    'Variance': '#4878a8',
    # Accelerations & decelerations
    'AC':       '#e07b54',
    'FM':       '#e07b54',
    'UC':       '#e07b54',
    'DL':       '#e07b54',
    'DS':       '#e07b54',
    'DP':       '#e07b54',
    # Histogram morphology
    'Width':    '#3a9e8c',
    'Min':      '#3a9e8c',
    'Max':      '#3a9e8c',
    'Nmax':     '#3a9e8c',
    'Nzeros':   '#3a9e8c',
    'Tendency': '#3a9e8c',
    # FHR central tendency
    'Mode':     '#c9a227',
    'Mean':     '#c9a227',
    'Median':   '#c9a227',
}

fig, ax, importance_df = summary_bar(
    ci.score,
    ci.se,
    feature_names,
    group_colors=GROUP_COLOUR,
    max_display=len(feature_names),
    show=False,
)
fig.savefig(FIG_DIR / "fig5_ctg_global_importance.pdf", bbox_inches="tight")
plt.close(fig)

# Group-level inference
clinical_groups = {
    'FHR baseline & variability': [
        feature_names.index(f) for f in ['LB', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Variance']
    ],
    'Accelerations & decelerations': [
        feature_names.index(f) for f in ['AC', 'FM', 'UC', 'DL', 'DS', 'DP']
    ],
    'Histogram morphology': [
        feature_names.index(f) for f in ['Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Tendency']
    ],
    'FHR central tendency': [
        feature_names.index(f) for f in ['Mode', 'Mean', 'Median']
    ],
}

clinical_group_ci = explainer.conf_int(
    alternative='greater',
    groups=clinical_groups,
    multitest_method='fdr_bh',
)

group_ci_df = (
    clinical_group_ci.to_frame()
    .reset_index()
    .rename(columns={'estimate': 'score', 'reject_null': 'reject_fdr'})
    [['group', 'score', 'se', 'ci_lower', 'pvalue', 'pvalue_adj', 'reject_fdr']]
    .sort_values('score', ascending=False)
    .reset_index(drop=True)
)

print(group_ci_df.to_string())
group_ci_df.to_csv(OUT_DIR / "case2_group_results.csv", index=False)

# Figure 6
fig, ax = confidence_interval_plot(
    ci,
    feature_names=feature_names,
    max_display=20,
    show=False,
)
fig.savefig(FIG_DIR / "fig6_ctg_confidence_intervals.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 7
per_sample_phi_X = np.asarray(explainer.ueifs_X)
finite_vals = per_sample_phi_X[np.isfinite(per_sample_phi_X)]
negative_fraction = float(np.mean(finite_vals < 0)) if finite_vals.size else np.nan

print('per_sample_phi_X shape:', per_sample_phi_X.shape)
print('min:', np.nanmin(per_sample_phi_X))
print('percentiles:', np.nanpercentile(per_sample_phi_X, [0, 1, 5, 50, 95, 99, 100]))
print('negative fraction:', negative_fraction)

fig, ax = summary_plot(
    per_sample_phi_X,
    features=X_values,
    feature_names=feature_names,
    max_display=len(feature_names),
    show=False,
)

if negative_fraction <= 0.001:
    positive_vals = finite_vals[finite_vals > 0]
    xmin = max(float(np.nanpercentile(positive_vals, 1)), 1e-6) if positive_vals.size else 1e-6
    xmax = float(np.nanpercentile(positive_vals, 99.5)) * 1.2 if positive_vals.size else 1.0
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Per-sample FDFI attribution value (log scale)')
    print(f'Figure 7 uses log x-axis: xmin={xmin:.3g}, xmax={xmax:.3g}')
else:
    linthresh = max(float(np.nanpercentile(np.abs(finite_vals), 5)), 1e-6) if finite_vals.size else 1e-6
    lo = float(np.nanpercentile(finite_vals, 0.5)) if finite_vals.size else -1e-6
    hi = float(np.nanpercentile(finite_vals, 99.5)) if finite_vals.size else 1.0
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_xlim(lo * 1.2, hi * 1.2)
    ax.set_xlabel('Per-sample FDFI attribution value (symlog scale)')
    print(f'Figure 7 uses symlog x-axis: linthresh={linthresh:.3g}, xlim=({lo * 1.2:.3g}, {hi * 1.2:.3g})')

fig.savefig(FIG_DIR / "fig7_ctg_per_sample_summary.pdf", bbox_inches="tight")
plt.close(fig)

# Sampling methods, on the first 20 observations
N_DEMO = 20
X_demo = X_explain[:N_DEMO]

sampling_methods = ['resample', 'permutation', 'normal']
results_by_method = {}

print('Comparing FDFI attributions across sampling strategies:')
print(f'(on subset of {N_DEMO} CTG instances)\n')

for method in sampling_methods:
    print(f'  Computing with sampling_method="{method}" ...')
    explainer_sm = FlowExplainer(
        model_wrapper,
        X_values,
        nsamples=50,
        loss='squared_error',
        method='cpi',
        sampling_method=method,
        random_state=0,
    )
    results_by_method[method] = explainer_sm(X_demo)

print('\n' + '=' * 80)
print('Feature Importance Comparison: Sampling Methods')
print('=' * 80)
print(f"{'Feature':>12} {'resample':>15} {'permutation':>15} {'normal':>15}")
print('-' * 80)

for i, fname in enumerate(feature_names):
    r = results_by_method['resample'].phi_X[i]
    p = results_by_method['permutation'].phi_X[i]
    n = results_by_method['normal'].phi_X[i]
    print(f"{fname:>12} {r:>15.6f} {p:>15.6f} {n:>15.6f}")

print('=' * 80)
print('\nInterpretation:')
print('  - Resample:    Uses background training data distribution (most conservative)')
print('  - Permutation: Permutes within the analysis set (minimal extrapolation)')
print('  - Normal:      Samples from standard Gaussian (maximum extrapolation)')

print(f"\nCase 2 completed in {time.time() - ctg_start:.2f} seconds")
