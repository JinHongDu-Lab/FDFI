# FDFI JSS replication package

This directory is the reproducibility entry point for results reported in the
main JSS manuscript. It is intentionally separate from `docs/`, which contains
documentation and tutorials.

The case-study scaffold has been replaced by typed, deterministic standalone
workflows. The simulation in `scripts/simulation.py` reproduces the published
Experiment 1 specification, with CSV-only plotting in
`scripts/simulation_plot.py`. Its primary audit target is Type-I error on the
independent null set C3. Full mode uses the published grids, 100 repetitions,
500-tree Random Forest, two-fold cross-fitting, and 3000 auxiliary observations
with 15000 Flow-training steps. Quick mode is only a computational smoke test.

The simulation produces the three artifacts required by manuscript Section
3.5: a benchmark using CPI scoring for OT/EOT/Flow, the corresponding benchmark
using SCPI scoring, and a D3-style single-panel computational-cost figure. Both statistical
figures retain LOCO and CPI as fixed baselines and use the manuscript legend
`LOCO`, `CPI`, `DFI-OT`, `DFI-EOT`, and `FDFI`.
Each benchmark is a 2-by-3 figure reporting AUC on C1 plus C3, power on C1,
and Type-I error on C3. Rejection power on C1 plus C2 and predictive R-squared
remain CSV diagnostics and are not method-comparison panels.

The Gaussian-block DGP follows the earlier manuscript definition. C1 contains
features 0--4, which enter the nonlinear response; C2 contains features 5--9,
which share the first correlation block but do not enter the response; and C3
contains features 10--49, which lie in independent blocks. Type-I error is
computed only on C3 as the number of rejected C3 hypotheses divided by the
number tested, separately for every sweep value, method, and resampling version.
The quick smoke test retains this structure with features 10--19 as C3.

Every simulation run writes `simulation_type1_error_audit.csv`. It records the
null-feature indices, expected and observed test counts, raw rejection count,
empirical Type-I error, whether it is at or above nominal alpha, all per-seed
rejection counts, failed seeds, and a follow-up status. A missing repetition is
`INCOMPLETE`. An above-nominal result is reported for investigation; it is not
post-hoc adjusted or made a computational failure. Quick results exercise this
audit but never count as formal evidence. All methods use the published one-sided zero-margin test. The
implementation pools held-out UEIFs, zeros columns with negative mean, and
applies the published response-variance floor and two-fold effective-sample-size
correction; it does not use the package's automatic margin or mixture floor.

Runtime is a separate Experiment 2 Gaussian-mixture study, not a timing summary
derived from Experiment 1. It compares `CPI`, `LOCO`, `nLOCO`, `dLOCO`, `OT`,
`EOT`, `FDFI`, and `SHAP` at n=200, 400, 600, 800, 1000, and 2500 over ten
deterministic seeds. Non-SHAP methods use d=50 and SHAP follows the paper text
with d=10. The evaluate-only timer excludes data generation, shared black-box
fitting, and OT/EOT/Flow construction or training; method-specific LOCO-family
and SHAP submodel fits remain included. Each sample-size x seed x method cell is
atomically checkpointed so interrupted SHAP runs can resume.

Runtime quick mode deliberately uses only n=40 and 60, d=10, one seed, three
Random-Forest trees, five resamples, and one Flow-training step. It validates
the execution and plotting contracts only; its absolute timings, relative
method ordering, and visual trend must not be compared with Figure D3. Formal
runtime uses the full sample-size grid, ten seeds, d=50 except SHAP d=10, 500
trees, 50 resamples, and the Experiment 6 Flow preparation of 5000 steps on a
training sample matching the runtime sample size.

The baseline CPI reproduces the published residual conditional permutation
using `StandardScaler` and `LassoLarsIC(criterion="bic")`. For OT, EOT, and
Flow, `method="cpi"` and `method="scpi"` refer only to the two documented
averaging orders. The two versions reuse the same data, folds, Random Forest
specification, and auxiliary Flow fit.

Benchmark feature rows are atomically checkpointed after every unique
sample-size x correlation x seed scenario. Restarting the same run directory
skips complete scenarios, retries an interrupted or failed scenario, and
reuses the duplicated n=1000, rho=0.8 computation across the two sweeps. The
stored settings JSON is immutable: a resume with different settings is rejected
instead of mixing incompatible results. Benchmark summaries and the Type-I
audit are regenerated from the complete checkpoint after all scenarios finish.

## Scope and source of truth

Following the project meeting, this replication package is aligned primarily
with exactly two analysis notebooks:

1. `docs/case_studies/flow_case_study_ctg.ipynb` — the CTG analysis based on
   `FlowExplainer`; and
2. `docs/case_studies/eot_case_study_sens50.ipynb` — the `sens50` analysis
   based on `EOTExplainer`.

These notebooks are the computational source of truth. The purpose of
`reproduce.py` is to rerun their validated analysis logic in a standalone,
non-interactive form and regenerate the figures, tables, and numerical output
from them that are used in the manuscript. It is not intended to reproduce
every notebook in the repository or every exploratory display in these two
notebooks. The required scope is the intersection of (a) outputs produced by
these two notebooks and (b) outputs included in the submitted manuscript.

If a manuscript value cannot be traced to one of these two notebooks, or if a
notebook output differs from the manuscript, the discrepancy must be resolved
and documented rather than silently copied.

The replication code imports the package in the repository-level `fdfi/`
directory. The package source is not duplicated here.

## Directory layout

```text
replication/
├── README.md
├── environment.yml
├── reproduce.py
├── reproduce_quick.py
├── inventory.md
├── data/
│   └── README.md
├── figures/
├── tables/
├── results/
├── runs/
│   └── <run_id>/{figures,tables,results,logs,metadata}/
└── scripts/
    ├── common.py
    ├── reproduce_ctg.py
    ├── reproduce_sens50_eot.py
    ├── reproduce_simulation.py
    ├── simulation.py
    ├── simulation_plot.py
    └── fetch_ctg.py
```

`figures/` and `tables/` are output directories. The sens50 workflow reads the
canonical fixed inputs from `docs/case_studies/data/`; `data/checksums.sha256`
records their hashes without duplicating them. Their upstream acquisition,
licence, and redistribution permission remain explicitly unresolved. CTG is
not bundled because its redistribution permission is unresolved.

## Notebook-aligned environment

From the repository root:

```bash
conda env create -f replication/environment.yml
conda activate fdfi-replication
python -m pip install -e .
```

The exact pins in `environment.yml` reproduce the verified main@f86d696
Python 3.10.19 notebook baseline recorded on 2026-08-11. They include
`xlrd=2.0.2` for the legacy CTG `.xls` file and `psutil=7.2.2` for hardware
metadata. Every full
workflow, including an individually selected workflow, treats a pin mismatch
as a blocker. Only quick smoke tests downgrade a mismatch to a warning.

## Preflight check

The following commands inspect the known source files and inputs without
running an analysis:

```bash
python replication/reproduce.py --check
python replication/reproduce_quick.py --check
```

Every full command performs this strict check before creating a run directory
or fitting a model. Missing/checksum-invalid data, incompatible dependencies,
or an incomplete selected workflow therefore stop immediately and cannot alter
run or stable output directories.

The sens50 workflow reads only the canonical fixed inputs in
`docs/case_studies/data/` during formal execution. Preflight verifies their
pinned SHA-256 values, 611-by-833
schema, `sens50` outcome, complete coverage of 832 predictors and 14 groups.
CTG uses the checksum-valid fixed `replication/data/CTG.xls`. Preflight also
opens its `Data` sheet and verifies raw/cleaned dimensions, feature order,
outcome conversion, and the 1,655/471 class counts.

## Full and quick runs

The intended final commands are:

```bash
python replication/reproduce.py
python replication/reproduce_quick.py
```

Every execution writes only to a unique `replication/runs/<run_id>/` directory.
Its manifest lists files from that run only. Quick runs remain there permanently
labelled as smoke tests. A complete successful plain full run atomically
publishes its figures, tables, and raw results to the stable top-level output
directories; failed or blocked runs publish nothing. The old
`replication/quick_outputs/` directory is legacy output and is not consulted by
the new orchestrator or included in new manifests.

Development-only selection is available without weakening the plain command:

```bash
python replication/reproduce.py --workflow sens50
python replication/reproduce.py --workflow ctg
python replication/reproduce_quick.py --workflow sens50
```

Selected workflows still run in a unique staging directory and are never
published as a complete formal replication.

The verified UCI CTG spreadsheet can be acquired with:

```bash
python replication/scripts/fetch_ctg.py
```

The downloader verifies SHA-256
`d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421`,
computed from the file served by the notebook's UCI URL on 2026-07-21. Data
licensing/redistribution metadata remain to be reviewed before bundling it.

The sens50 formal inference is the source-notebook inference. Feature-level X/Z
summaries use a 0.1 mixture variance floor, the 0.95 floor quantile, automatic
margin selection at the 0.95 quantile, and two-sided tests with no multiplicity
argument. Group-level X/Z summaries use `threshold_null=True`, a fixed 0.1
variance floor, a fixed zero margin, two-sided tests, and Bonferroni correction.
The full workflow asserts the notebook rejection counts (6/832, 7/832, 10/14,
and 11/14) and attribution totals before reporting success.

The sens50 workflow writes both machine-readable CSV files and two
human-readable tables produced directly by the package's public `summary()`
method: `sens50_eot_feature_summary.txt` and
`sens50_eot_group_summary.txt`. These text files reproduce the format used in
the source notebook's feature- and group-summary cells.

Run status is `FAILED` when any required stage has an unexpected failure,
otherwise `BLOCKED` when a requirement is unavailable, and `SUCCESS` only when
all selected required stages succeed. Downstream stages are `SKIPPED` after a
required failure/block. Each run log tees complete terminal output, timestamps,
configuration, seeds, warnings, tracebacks, stage statuses, and final status.

Input dimensions and required output presence are validated, but a complete
reviewed schema/tolerance comparison against final manuscript values remains
pending. Every manifest artifact records size and SHA-256. PDFs are generated
from current numerical objects and are not required to be byte-identical because
PDF metadata can vary; stable names, plotting inputs, and figure logic are used.
The sens50 source notebook does not produce an observed diagnostics figure,
diagnostics table, latent dCor matrix, or strongest-dependency table. The
replication workflow therefore does not generate those artifacts. This keeps
the standalone reproduction limited to notebook-defined outputs.

Each case-study run writes a machine-readable settings file under its run
metadata directory. The run manifest records workflow selection, publication
attempt/success, stable published paths, input/notebook/workflow-script
checksums, and the complete settings objects. `formal_manuscript_run=true` is
possible only for a successful `full` + `all` run whose transactional
publication succeeds. Publication removes stale files only when they were
explicitly recorded as owned by the preceding replication publication;
unmanaged reference PDFs are preserved.

The full run will regenerate the manuscript figures, tables, and key numbers
derived from the two source notebooks. The quick run will use smaller
computational settings as a smoke test; its numerical values will not be
manuscript results.

## Completion criteria

Before submission, this package should satisfy all of the following:

- every computational result in the final Overleaf manuscript is listed in
  `inventory.md`;
- all required inputs are available without private local paths;
- all random seeds and analysis parameters are explicit;
- the full command recreates all listed figures and tables;
- the quick command completes in a few minutes;
- a clean full run completes in roughly one hour on an ordinary computer;
- key numerical output is compared with the fixed notebook expectations; and
- expected platform-dependent numerical tolerances are documented.
