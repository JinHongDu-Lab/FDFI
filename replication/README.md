# FDFI JSS replication package

This directory is the reproducibility entry point for results reported in the
main JSS manuscript. It is intentionally separate from `docs/`, which contains
documentation and tutorials.

The case-study scaffold has been replaced by typed, deterministic standalone
workflows. A draft simulation implementation now lives in `scripts/simulation.py`
with CSV-only plotting in `scripts/simulation_plot.py`. It can produce an
isolated quick meeting preview, but full/formal execution remains blocked until
the authors approve the DGP, grids, repetitions, method mapping, and inference
settings. The remaining decisions are recorded in
`simulation_design_blockers.json`.

The simulation produces the three artifacts required by manuscript Section
3.5: a benchmark using CPI scoring for OT/EOT/Flow, the corresponding benchmark
using SCPI scoring, and a two-panel computational-cost figure. Both statistical
figures retain LOCO and CPI as fixed baselines and use the manuscript legend
`LOCO`, `CPI`, `DFI-OT`, `DFI-EOT`, and `FDFI`.

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
