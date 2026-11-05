"""Replication script for the case studies in
"fdfi: Disentangled Feature Importance in Python".

Case 1 (Section 4): EOTExplainer on the HIV sens50 data.
Case 2 (Section 5): FlowExplainer on the CTG data (Figures 3-7).

Run from the case_study/ folder:

    python case_studies.py

Figures are saved to case_study/figures/ and key tables to case_study/outputs/.

Package: fdfi 0.2.1 from yuewen-dev@e6b9422 (installed with `pip install --no-deps -e .`).
Changes for 0.2.1 are marked with "[0.2.1]".
"""

# =============================================================================
# Imports
# =============================================================================
import os
import time
import warnings
from pathlib import Path

import platform

import numpy as np
import pandas as pd
import matplotlib
# [0.2.1] Non-interactive backend. The default Tk backend crashed the script
# ("main thread is not in main loop", "Tcl_AsyncDelete") during the CTG part.
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

# [0.2.1] Record the package location and library versions for the paper.
print(f"fdfi {fdfi.__version__} from {fdfi.__file__}")
print(f"python {platform.python_version()}, numpy {np.__version__}, "
      f"pandas {pd.__version__}, scikit-learn {sklearn.__version__}")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# ## Case 2 (Section 5): Flow FDFI on the Cardiotocography (CTG) data
# =============================================================================

# Keep numerical backends conservative during execution.
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

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ctg_start = time.time()  # [0.2.1] runtime of Case 2

# --- Load data and clinical features ------------------------------------------
# ── 2.1  Load from UCI repository ────────────────────────────────────────────
CTG_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'
)

print('Downloading CTG dataset from UCI ...')
raw = pd.read_excel(CTG_URL, sheet_name='Data', skiprows=1, header=0)
print(f'  Raw shape (before cleaning): {raw.shape}')

# ── 2.2  Extract feature matrix X and binary target vector y ─────────────────
X_raw = raw.iloc[:, 10:31].copy()
y_raw = raw['NSP'].copy()

data = pd.concat([X_raw, y_raw], axis=1)

feature_names = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
    'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
    'Median', 'Variance', 'Tendency'
]
data.columns = feature_names + ['NSP']

# ── 2.3  Data cleaning, type conversion, and binary endpoint ─────────────────
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

# --- Collinearity diagnostics (Figure 3) --------------------------------------
fig, ax, feature_names_reordered = correlation_heatmap(
    X, feature_names, show=False
)
fig.savefig(FIG_DIR / "fig3_ctg_correlation_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# --- Fit the black-box prediction model ---------------------------------------
# ── 4.1  Full-data standardization ────────────────────────────────────────────
scaler = StandardScaler()
X_values = scaler.fit_transform(X)
y_values = y.astype(int)

# Keep these aliases for downstream explanatory code.
X_train = X_values
X_explain = X_values
y_train = y_values

print(f'Analysis samples : {X_values.shape[0]}')
print(f'Analysis features: {X_values.shape[1]}')

# ── 4.2  Fit classifier on the full CTG dataset ──────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model.fit(X_values, y_values)

# ── 4.3  Full-data performance check ─────────────────────────────────────────
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

# --- Initialize FlowExplainer and compute Flow FDFI (Figure 4) ----------------
# ── 5.1  Initialise FlowExplainer ─────────────────────────────────────────────
# Return predicted probability of the binary target class Other = 1.
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

# ── 5.2  Display diagnostics ─────────────────────────────────────────────────
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

# --- Feature-level inference with conf_int() ----------------------------------
# ── 6.1  Compute full-data attributions with public batched API ──────────────
print(f'Computing FDFI attributions for all {X_values.shape[0]} CTG instances ...')
print('Full-data binary distribution:')
print(pd.Series(y_values).value_counts().sort_index().rename({0: 'Normal = 0', 1: 'Other = 1'}))

results = explainer(X_values)

# [0.2.1] Attribute access on the ExplanationResults object (comment 3).
print(f"results.phi_X.shape: {results.phi_X.shape}")
print(f"results.se_X.shape: {results.se_X.shape}")
print(f"explainer.ueifs_X shape: {explainer.ueifs_X.shape}")

# [0.2.1] ConfIntResults object; its ranking is used by Figure 6 (comment 4).
ci = explainer.conf_int(
    alternative='greater',
    multitest_method='fdr_bh',
)

# [0.2.1] Feature-level inference table with feature names (Section 5.5),
# replacing the string-slicing of summary() output.
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
importance_df.to_csv(OUT_DIR / "case2_feature_results.csv", index=False)  # [0.2.1]

# --- Global importance bar chart (Figure 5) -----------------------------------
# ── 6.3  Global importance bar chart with error bars ─────────────────────────

# Colour scheme by clinical feature group
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

fig, ax, importance_df_sorted = summary_bar(
    ci.score,
    ci.se,
    feature_names,
    group_colors=GROUP_COLOUR,
    max_display=len(feature_names),
    show=False,
)
fig.savefig(FIG_DIR / "fig5_ctg_global_importance.pdf", bbox_inches="tight")
plt.close(fig)

# --- Group-level inference ----------------------------------------------------
# ── Group-level clinical inference using fdfi conf_int API ────────────────────
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

group_ci = explainer.conf_int(
    alternative='greater',
    groups=clinical_groups,
    multitest_method='fdr_bh',
)

group_ci_df = (
    group_ci.to_frame()
    .reset_index()
    .rename(columns={'estimate': 'score', 'reject_null': 'reject_fdr'})
    [['group', 'score', 'se', 'ci_lower', 'pvalue', 'pvalue_adj', 'reject_fdr']]
    .sort_values('score', ascending=False)
    .reset_index(drop=True)
)

print(group_ci_df.to_string())
group_ci_df.to_csv(OUT_DIR / "case2_group_results.csv", index=False)  # [0.2.1]

# --- Feature-level uncertainty and visualization (Figures 6 and 7) ------------
# ── 6.4  Feature-level confidence intervals using the updated plotting API ────
fig, ax = confidence_interval_plot(
    ci,
    feature_names=feature_names,
    max_display=20,
    show=False,
)
fig.savefig(FIG_DIR / "fig6_ctg_confidence_intervals.pdf", bbox_inches="tight")
plt.close(fig)

# Optional per-sample attribution summary. The axis scale is chosen from the
# observed full-data attribution matrix and does not change the attribution values.
per_sample_phi_X = explainer.ueifs_X
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

# --- Sampling methods in latent space -----------------------------------------
# ── 7.1  Compare feature importance across sampling methods ──────────────────

# Use a small subset for quick demonstration while keeping the full-data model/background.
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

print(f"\nCase 2 completed in {time.time() - ctg_start:.2f} seconds")  # [0.2.1]
