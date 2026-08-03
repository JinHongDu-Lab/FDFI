# Changelog

## [Unreleased]
### Removed
- **`TreeExplainer`, `LinearExplainer`, and `KernelExplainer`**: these were placeholders whose `__call__` raised `NotImplementedError`, and they are gone from `fdfi.explainers` along with their API pages, user-guide sections, and tests. The three implemented variants (`OTExplainer`, `EOTExplainer`, `FlowExplainer`) are model-agnostic — they wrap any callable `f(X) -> y` — so tree ensembles, linear models, and arbitrary black boxes are already covered without a model-specific class. Code that imported these names will now raise `ImportError` instead of failing later at call time.

## [0.0.9] - 2026-07-13
### Added
- **Arbitrary loss functions**: importance can now be defined through any per-sample loss instead of only the squared-error (L2) residual difference. New `fdfi/losses.py` registry provides regression losses (`squared_error`/`l2`, `absolute_error`/`l1`, `huber`, `pinball`) and binary-classification losses (`log_loss`/`bce`, `brier`, `zero_one`), plus `resolve_loss()`/`available_losses()`. Custom callables `loss(y_true, y_pred)` are also accepted.
- **`loss` argument** on `OTExplainer`, `EOTExplainer`, `FlowExplainer`, and `Crossfitting` (default squared error → unchanged behaviour). Passing true labels `y` at call time uses the loss-difference (DFI) form; when `y` is omitted a label-free form is used that references the model's own prediction — the prediction shift for regression losses and a Bregman divergence (e.g. KL for log-loss) for proper scoring rules.
- **`method='cpi'|'scpi'`** now available on `OTExplainer` and `EOTExplainer` (previously only `FlowExplainer`), selecting the averaging order for the counterfactual prediction (CPI averages the prediction before the loss; SCPI averages the per-sample loss).
- New tests: `tests/test_losses.py` (registry/built-ins) and loss-integration tests in `tests/test_explainers.py` (L2 parity, regression/classification losses, CPI/SCPI, guards, cross-fitting).

### Changed
- `FlowExplainer` SCPI now follows the documented definition `E_b[L(Y, f(X̃_b))]` (for squared error, equal to CPI plus the prediction variance) rather than the raw prediction variance, making SCPI consistent across all explainers.
- Updated `docs/user_guide/concepts.rst`, `docs/user_guide/choosing_explainer.rst`, and `docs/api/explainers.rst` to document loss selection and the generalized CPI/SCPI formulas.

## [0.0.8] - 2026-06-30
### Added
- **One-sided confidence interval plots**: `confidence_interval_plot()` now detects `alternative='greater'` or `alternative='less'` in the `conf_int()` result dict and renders the open bound as a short stub with a native matplotlib limit-indicator caret (►/◄ via `xuplims`/`xlolims`), following the forest-plot truncation convention. Axis limits exclude the infinite bound; a corner annotation and one-sided hint are added to the default xlabel and title.
- New optional kwargs for `confidence_interval_plot()`: `stub_fraction` (default 0.06), `show_alternative_note` (default True), `note_fontsize` (default 8), `marker` (default 'o').
- Added 15 new tests in `TestCIPlotOneSided` covering smoke rendering, finite axis-limit checks, caret artist presence, backward compatibility, annotation toggle, label content, stub_fraction effect, unknown alternative validation, savepath, and max_display truncation.
- New tutorial section "Visualising One-Sided and Two-Sided Confidence Intervals" in `docs/tutorials/confidence_intervals.ipynb` with a 3-panel side-by-side comparison.

## [0.0.7] - 2026-05-26
### Added
- Working `fdfi.plots` visualizations for summary, waterfall, force, dependence, correlation heatmap, confidence interval, and diagnostics views.
- `summary_bar()` for sorted global FDFI bars with sanitized standard errors, optional group colors, and a returned sorted table.
- `correlation_heatmap()` with Pearson correlations, absolute-correlation hierarchical clustering, reordered feature names, and small-background warnings.
- Visualization tutorial and documentation examples covering `results["phi_X"]`, `results["se_X"]`, per-sample `ueifs_X`, `conf_int()` output, and explainer diagnostics.

### Changed
- Replaced placeholder plot tests with non-interactive Matplotlib `Agg` tests.
- Updated visualization documentation to remove stale "coming soon" language.

## [0.0.6] - 2026-05-17
### Added
- **`zscore` and `ranking` output fields**: `conf_int()` now always returns `zscore` (signed z-statistic `(score − margin) / se`) and `ranking` (integer rank by descending z-score, 1 = most important) for all targets and group modes.
- **`summary()` docstring**: full NumPy-style docstring added, documenting all forwarded `conf_int()` parameters including 0.0.5-introduced `groups` and `multitest_method`.
- **`Crossfitting` enhancements**: new `cv_kwargs` and `y_test` parameters; improved sklearn estimator detection; null-threshold logic in fold aggregation.
- **`FlowMatchingModel` improvements**: `dequantize_noise` parameter for binary data augmentation; `Jacobi_Batch` now uses a simpler sequential loop (one sample at a time via `Jacobi_N`).
- **Case study docs**: EOT FDFI case study notebook (`eot_case_study_sens50`) added to documentation under a new Case Studies section.
- **`docs/user_guide/concepts.rst`**: `conf_int()` return keys documented as a reference table.

### Fixed
- `Explainer` class docstring: corrected `from dfi import` → `from fdfi import`.
- `group_importance()` deprecation version tag corrected from `0.2.0` to `0.0.5`.
- `_format_summary()`: z-score now computed as `(score − margin) / se` (was `score / se`).
- `docs/conf.py` `release` field was stale at `0.0.4`; updated to match package version.
- `Crossfitting` docstring: RST bullet list in parameter description replaced with prose to fix Sphinx rendering error.
- `docs/tutorials/confidence_intervals.ipynb`: repaired two broken JSON cell boundaries.
- `MANIFEST.in`: exclude `docs/case_studies/data` and `docs/case_studies/results` from sdist.

## [0.0.5] - 2026-04-28
### Added
- **Multiple testing correction**: `conf_int()` and `summary()` now support a `multitest_method` parameter (mimicking `statsmodels`) for controlling FDR/FWER. Adjusted p-values are returned as `pvalue_adj`.
- **Unified Group Importance**: `conf_int()` now supports a `groups` argument to compute group-level feature importance with uncertainty, replacing the separate `group_importance()` logic (now deprecated).
  - Accepts groups as a `dict` of index lists, a 1-D label array, or a binary `pandas.DataFrame` indicator matrix (features may belong to multiple groups).
  - Optional null-feature thresholding (`threshold_null=True`) zeros out per-feature UEIFs with negative mean before aggregation.
- **Improved API Consistency**: Renamed `phi_hat` to `score` in `conf_int()` output for better clarity.
- Per-sample UEIFs (`ueifs_X`, `ueifs_Z`) are now stored as instance attributes after calling `OTExplainer`, `EOTExplainer`, and `FlowExplainer`, enabling downstream group aggregation.

## [0.0.4] - 2026-04-01
### Added
- **Crossfitting**: new cross-fitted DFI explainer for valid inference at small sample sizes. Wraps any explainer class (`OTExplainer`, `EOTExplainer`, `FlowExplainer`) and performs K-fold cross-fitting so that the disentanglement map is never evaluated on its own training data.
- Flexible `cv` parameter accepts an `int` (shorthand for `KFold`) or any scikit-learn splitter instance (`StratifiedKFold`, `ShuffleSplit`, `RepeatedKFold`, `GroupKFold`, custom, etc.).
- Optional `y` and `groups` parameters for stratified and group-aware splitters.
- Overlapping test set handling: splitters like `ShuffleSplit` and `RepeatedKFold` that assign samples to multiple test sets are handled by per-sample UEIF averaging.
- Ensemble prediction on new data: `cf(X_new)` averages importance from all fold explainers.
- `Crossfitting` inherits `conf_int()` and `summary()` from the base `Explainer` class.
- `Crossfitting` exported from `fdfi` top-level package.
- 17 new tests covering init, OT/EOT/Flow cross-fitting, all splitter types, conf_int, summary, and ensemble prediction.

## [0.0.3] - 2026-03-19
