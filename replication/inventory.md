# Manuscript result inventory

This map is based on files currently present in the Git working tree. The two
case-study workflows are now implemented. Selected full numerical runs passed
on 2026-07-27; a clean post-commit provenance run is still pending. The
computational sources of truth are:

1. `docs/case_studies/flow_case_study_ctg.ipynb`; and
2. `docs/case_studies/eot_case_study_sens50.ipynb`.

The replication scope consists of outputs produced by these two notebooks that
are synchronized into the manuscript. Other notebooks and exploratory outputs
are outside scope unless the manuscript later uses them. The manuscript master
TeX files remain in Overleaf and are still under revision, so a final audit
against those files is required before submission.

Status meanings:

- `READY`: inputs, executable source, parameters, and output are traceable;
- `PARTIAL`: some pieces are present, but the result is not reproducible end to
  end from a standalone script;
- `TODO`: scope or source is not yet known.

All generated candidates first live under `runs/<run_id>/`. Stable formal paths
are updated only after every required full stage succeeds. Existing PDFs in
`figures/` and legacy `quick_outputs/` are not evidence for a later run; only a
matching run manifest is authoritative.

## Computational figures

| Result | Current TeX reference | Current source | Input | Proposed output | Status | Gap / next action |
|---|---|---|---|---|---|---|
| CTG correlation heatmap | current Overleaf `img/` (master TeX pending) | implemented `reproduce_ctg.py`; `correlation_heatmap` | pinned checksum-valid CTG input | `figures/fig1_correlation_matrix.pdf` | PARTIAL | Full numeric run passed; final notebook rerun and PDF freeze pending. |
| CTG global FDFI importance | current Overleaf `img/` (master TeX pending) | implemented `reproduce_ctg.py`; `FlowExplainer`; `summary_bar` | cleaned and standardized CTG data | `figures/fig2_fdfi_importance.pdf` | PARTIAL | Full run matched 18/21 and top ranking; final notebook PDF freeze pending. |
| CTG one-sided confidence intervals | current Overleaf `img/` (master TeX pending) | implemented `reproduce_ctg.py` | CTG feature inference | `figures/fig3_confidence_intervals.pdf` | PARTIAL | Full inference passed; final notebook PDF freeze pending. |
| CTG flow diagnostics | current Overleaf `img/` (master TeX pending) | implemented `reproduce_ctg.py` | fitted full flow | `figures/fig4_flow_diagnostics.pdf` | PARTIAL | Full MMD/dCor matched notebook display precision; final notebook PDF freeze pending. |
| CTG per-sample attribution summary | current Overleaf `img/` (master TeX pending) | implemented `reproduce_ctg.py` | per-sample attributions | `figures/fig_optional_summary_plot.pdf` | PARTIAL | Full run selected the notebook log scale; final notebook PDF freeze pending. |
| sens50 three-panel group importance | current Overleaf audit pending | implemented `reproduce_sens50_eot.py`; source-notebook plot logic | notebook-aligned X-space group inference | `figures/sens50_eot_group_importance.pdf` | PARTIAL | Full values/counts passed; final notebook raster/PDF freeze pending. |
| Any additional main-text or appendix figure | final Overleaf sources | TODO | TODO | `figures/` | TODO | Audit the final Overleaf master and included TeX files. |

## Simulation status

`scripts/reproduce_simulation.py` is registered as the third required stage.
`scripts/simulation.py` and `scripts/simulation_plot.py` reproduce the original
Experiment 1 DGP, grids, two-fold inference, conditional CPI baseline,
variance-floor calculation, and 2-by-3 summaries. The public experiment and
estimator source supplied the previously missing executable specification, and
the author confirmed that CPI/SCPI mean the two package averaging orders.
`simulation_design_blockers.json` records the resolved specification and the
remaining EOT-specific extension note. Quick mode remains a smoke test; full
mode is the formal 100-repetition specification.

Both long-running studies are resumable. Experiment 1 atomically checkpoints
each unique sample-size x correlation x seed scenario; Experiment 2 atomically
checkpoints each sample-size x seed x method cell. A run directory is bound to
its stored settings contract so results from incompatible configurations cannot
be combined accidentally.

Current Section 3.5 simulation outputs are:

- `figures/simulation_benchmark_cpi.pdf`: LOCO, CPI, DFI-OT, DFI-EOT and FDFI,
  with the three package explainers using CPI scoring;
- `figures/simulation_benchmark_scpi.pdf`: the same legend and fixed baselines,
  with the three package explainers using SCPI scoring;
- `figures/simulation_runtime.pdf`: separate Experiment 2 evaluate-only runtime
  for CPI, LOCO, nLOCO, dLOCO, OT, EOT, FDFI, and SHAP on a log-second axis;
- feature-level, runtime, benchmark-summary and runtime-summary CSV sources.

## Computational tables and numerical results

| Result | Current TeX label | Current source | Proposed output | Status | Classification / next action |
|---|---|---|---|---|---|
| CTG class distribution | `tab:ctg-class` | implemented CTG cleaning in `reproduce_ctg.py` | `tables/ctg_class_distribution.csv` | READY | Full run confirms 1,655 Normal / 471 Other. |
| Largest CTG correlations | `tab:ctg-correlations` | implemented CTG correlation analysis | `tables/ctg_largest_correlations.csv` | PARTIAL | Implemented with deterministic ordering; fixed-input/full comparison pending. |
| CTG classification report | `tab:classification-report` | implemented full-data random-forest analysis | `tables/ctg_classification_report.csv` | READY | Full run confirms no-split accuracy 0.9868297272 with seed 42. |
| CTG flow diagnostics | `tab:ctg-diagnostics` | implemented same-flow `FlowExplainer` diagnostics | `tables/ctg_diagnostics.csv` | READY | Full run matches MMD 0.031184 and median dCor 0.092509 at notebook display precision. |
| CTG selected feature summary | `tab:feature-summary` | implemented CTG attribution results | `tables/ctg_feature_summary.csv` | READY | Full run confirms 18/21 and ASTV, ALTV, LB, AC as the top four. |
| CTG group inference | `tab:group-inference` | implemented CTG attribution/grouping analysis | `tables/ctg_group_inference.csv` | READY | Full run confirms 1/4, led by FHR baseline and variability. |
| CTG sampling-method comparison | `tab:sampling-methods` | implemented resample/transport comparison | `tables/ctg_sampling_methods.csv` | PARTIAL | Full 5,000-step execution passed; final row-by-row notebook table freeze remains pending. |
| sens50 EOT feature summaries | current Overleaf audit pending | implemented `reproduce_sens50_eot.py` | X/Z feature inference CSVs | READY | Full run confirms 6/832 and 7/832 plus notebook attribution totals. |
| sens50 EOT group summary | `tab:eot-group-summary` | implemented `reproduce_sens50_eot.py` | X/Z group summary CSVs | READY | Full run confirms thresholded Bonferroni counts 10/14 and 11/14. |
| sens50 biological group mapping | `tab:eot-group-mapping` | fixed `feature_group.csv` and notebook mapping | `tables/sens50_group_mapping.csv` | PARTIAL | Generated and checksummed; provenance/licence and Overleaf audit pending. |
| Main-text scalar values not in tables | final Overleaf sources | TODO | `tables/key_numerical_results.csv` or logged checks | TODO | Search the final manuscript for manually typed sample sizes, accuracy, diagnostics, p-values, and counts. |

## Descriptive tables that may remain in TeX

The following currently look like software/API documentation rather than
experimental outputs. They should be checked against the package API but do not
necessarily need numerical regeneration:

- `tab:flow-parameters`
- `tab:flow-explainer-api`
- `tab:diagnostic-thresholds`
- `tab:diagnostic-accessors`
- `tab:fdfi-components`
- `tab:fdfi-explainers`
- `tab:fdfi-shared-methods`
- `tab:fdfi-plot-diagnostics`
- `tab:fdfi-diagnostic-output`
- `tab:sens50-data-objects`
- `tab:sens50-model-fitting`
- `tab:sens50-eot-settings`
- `tab:sens50-workflow`

## Known inputs and preliminary parameters

### CTG

- Current external source:
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls`
- Current notebook random seed: `42`.
- Random-forest/data seed: `42`; notebook-equivalent `FlowExplainer` and flow
  training seed: default `0`.
- Current explainer: `FlowExplainer`.
- Current notebook setting observed: `nsamples=50`.
- Current workflow uses full-data fitting (no train/test split), 300 trees,
  `min_samples_leaf=3`, 5000 flow steps, one-sided feature inference, the
  notebook group-colour mapping, and its dynamic optional-plot axis rule.

### sens50 EOT

- Current processed data shape documented in the notebook: 611 observations,
  832 predictors, and 14 biological groups.
- Current predictive model settings observed: random forest with
  `n_estimators=500`, `min_samples_leaf=5`, `random_state=0`, and `n_jobs=-1`.
- Current explainer settings observed: `EOTExplainer`, `nsamples=50`,
  `epsilon=0.001`, and `random_state=0`.
- Formal inference is exactly the source-notebook inference: mixture-floor and
  automatic-margin feature summaries, plus thresholded fixed-floor
  Bonferroni group summaries. The expected counts are X/Z features 6/832 and
  7/832, and X/Z groups 10/14 and 11/14.
- The sens50 workflow intentionally does not export diagnostics, because the
  source notebook has no corresponding observed diagnostics output.

## Final Overleaf audit checklist

When the manuscript is ready, obtain the master TeX file plus every included
TeX file and then:

1. enumerate all `includegraphics` references;
2. enumerate all experimental table labels and values;
3. search prose for computed scalar values;
4. map each item to one script, fixed input set, and output;
5. add appendix and supplementary results;
6. compare regenerated artifacts with the reviewed manuscript artifacts; and
7. replace every `PARTIAL` and `TODO` needed by the paper with `READY`.
