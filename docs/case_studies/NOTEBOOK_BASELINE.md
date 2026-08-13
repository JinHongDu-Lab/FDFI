# Case Study Notebook Baseline

## Scope and source

This record fixes the computational baseline for the two case-study notebooks at
the following Git commit:

- repository: `https://github.com/jaydu1/FDFI.git`
- branch used only for local validation:
  `codex/task-4-1-notebook-baseline-f86d696`
- source commit:
  `f86d69602849468c3bd8943b73322b38ae68943a`
- source tag: `0.0.10`
- remote verification: `refs/heads/main` resolved to the source commit immediately
  before the isolated worktree was created on 2026-08-12

The source notebooks were not edited or executed in place. Full executions were
written to the ignored `.venv/task41-validation/` directory. Cell sources and
notebook-level metadata in each executed copy were checked against the committed
source notebook and were identical. Only execution counts and outputs changed.

## Notebook identities

| Notebook | Git blob | SHA-256 |
|---|---|---|
| `docs/case_studies/eot_case_study_sens50.ipynb` | `353c53234cbbf20b5e0212ae58160cd95c3cd049` | `e306cb3c2c73134aaba6b07a29c4deba4f2a09f3bf340c6e6e0b94cc9592f753` |
| `docs/case_studies/flow_case_study_ctg.ipynb` | `875eb97485c7b1720702619b6801fb7439d82c73` | `1660ddbbb1ac7edfa6c1f5388f7d707ba7cd9d3e31d80fd044b2bd6e1b0ebace` |

Both committed notebooks retain their original metadata:

- kernelspec name: `python3`
- kernelspec display name: `.venv`
- `language_info.version`: `3.10.19`

For validation, an isolated `python3` kernelspec pointing to the worktree's
Python 3.10.19 environment was placed inside `.venv/share/jupyter`. The notebooks'
metadata was not changed.

## Code and configuration dependencies

The notebooks must be paired with the source commit above. Copying only the
notebooks to an older branch is not an equivalent baseline because FDFI 0.0.10
contains loss and CPI/SCPI behavior not present as a complete module set on the
audited `origin/yifan-dev` commit.

| File | Git blob | SHA-256 |
|---|---|---|
| `fdfi/__init__.py` | `aa1fb084e356e8f8faf558e9e10d25ead882dd0b` | `3c0d746a84db5a908f6c3f7f31e47eb6ae28b4d51681e23ec029de05c5394c2f` |
| `fdfi/explainers.py` | `e71622359c564e72fa6e0458dc80442331d69ad1` | `409c336941e234d5822b80d7b63883c50ad2765dcbfb12ce021d8f4dc7698e7c` |
| `fdfi/plots.py` | `75ae0fd32f90548c6c00998ee43cacebf78bba67` | `a03048f0daf56c9b0e0a00c65b79eecbe45bcb6f15651b20bbd6d70c7558757b` |
| `fdfi/losses.py` | `25363c9d11004c34ce6268aa357d9809cb7bc2e4` | `b7ca2be1d0a499cc12ca2ed5025c71aa7f31881a5b8da7695e25a1259d9048c5` |
| `fdfi/models.py` | `5aebbec430c44a2e3922f83a94585ddfe2831bca` | `114d96a1d4532f12feb8f6a3c5cea7e84f3c2c6c33dd147c523f6aa820bad29f` |
| `fdfi/utils.py` | `d2817de1a8f5829679ad72cd6f744a38bd58f534` | `32d7bae8bbe525ff393211e0b028437f97495983e6bca391471073bf819ad901` |
| `pyproject.toml` | `de1ec2e1b110b07b7fede8f27aaf274294c36565` | `93c4c877cb50064fb2e611f82f4c09c873425d8e6fe1e04e1b8c923ae19bf0b0` |
| `docs/requirements.txt` | `66f6f174f2e3ed043f026c7f8d5af38f652284fb` | `40f71a6634f0332fde8d84d9854549c9cadaf42913a12c447e19f0734d280c80` |

## Data dependencies

### EOT case study

The notebook must be run with `docs/case_studies` as its working directory. It
uses `data/sens50_processed_dataset.csv` and `data/feature_group.csv` for the
selected `sens50` outcome. The other declared outcome files are recorded because
changing the `outcome` parameter makes them active inputs.

| File | Git blob | SHA-256 |
|---|---|---|
| `data/feature_group.csv` | `023d43c64acd1d6daeda82c99ada8ecf49b1a989` | `fc519731d770ca4672029877fbcb884a83f859e687851d0390908c15bbc80e97` |
| `data/sens50_processed_dataset.csv` | `8b5d69f3efbbaea973e72e316bef856308fa5eef` | `a7831f7926706377ed9b81fd6c73f995f69f1d971374859828c9300b566a0287` |
| `data/sens80_processed_dataset.csv` | `cb68c86837cf3e58a1f2419e32698ce8769aff38` | `61136592d1bff8e691582c2378a7d4aa3fa4e464a66bf600c7304c61d93b9849` |
| `data/ic50.censored_processed_dataset.csv` | `47104c0a037f1a83359b5217c5d4d009217029cf` | `62308321d92c05bd0b9156f70faea66c770b21cac8cb1d65ee7dabaf4d98013f` |

### CTG case study

The notebook downloads the UCI Cardiotocography workbook at runtime:

`https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls`

A pre-run download on 2026-08-12 was 1,743,872 bytes with SHA-256
`d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421`.
The executed notebook loaded 2,129 raw rows and produced 2,126 clean rows with
21 analysis features. Future replications must verify the downloaded workbook
against this checksum or explicitly document a data revision.

## Randomness and analysis parameters

### EOT

- selected outcome: `sens50`
- Random Forest: 500 estimators, minimum leaf size 5, `random_state=0`
- EOT resamples: `nsamples=50`, `epsilon=0.001`, sampling method `resample`,
  `random_state=0`
- feature inference: two-sided, mixture variance floor and automatic margin
- group inference: `threshold_null=True`, fixed variance floor, fixed zero
  margin, two-sided alternative and Bonferroni adjustment

### CTG

- NumPy seed: `42`
- Random Forest: 300 estimators, minimum leaf size 3, `random_state=42`
- `FlowExplainer` defaults include `nsamples=50`, sampling method `resample`
  and flow training seed `0`
- feature and group inference: one-sided `greater` alternative with
  Benjamini-Hochberg FDR adjustment
- sampling-method demonstration: first 20 rows, methods `resample`,
  `permutation` and `normal`; each method trains its own 5,000-step flow

PyTorch and numerical libraries can still produce platform-level floating-point
variation. The checks below therefore fix both software versions and the exact
observed outputs rather than claiming byte-identical results on every platform.

## Validation environment

- operating system: macOS 15.3.1, arm64
- Python: 3.10.19
- FDFI: 0.0.10 from the source commit above
- NumPy: 2.2.6
- SciPy: 1.15.3
- pandas: 2.3.3
- scikit-learn: 1.7.2
- Matplotlib: 3.10.9
- seaborn: 0.13.2
- statsmodels: 0.14.6
- PyTorch: 2.13.0
- torchdiffeq: 0.2.5
- xlrd: 2.0.2
- nbformat: 5.11.0
- nbclient: 0.11.0
- nbconvert: 7.17.1

The full platform-specific package record is in
`notebook-baseline-py310-macos-arm64.txt`. `python -m pip check` reported no
broken requirements before execution.

## Full-execution commands

The environment was created in the isolated worktree and is ignored by Git:

```bash
conda create --prefix .venv python=3.10.19 pip -y
.venv/bin/python -m pip install -e ".[flow,docs]" \
  pandas nbformat nbclient nbconvert
.venv/bin/python -m ipykernel install \
  --prefix .venv --name python3 \
  --display-name "Python 3.10.19 (FDFI task 4.1)"
```

Each notebook was then run from `docs/case_studies` with the isolated kernelspec:

```bash
export JUPYTER_PATH="$PWD/../../.venv/share/jupyter"
export MPLCONFIGDIR="$PWD/../../.venv/task41-validation/mpl-cache"
export XDG_CACHE_HOME="$PWD/../../.venv/task41-validation/xdg-cache"

../../.venv/bin/python -m jupyter nbconvert \
  --execute --to notebook \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir ../../.venv/task41-validation \
  --output eot.executed.ipynb \
  eot_case_study_sens50.ipynb

../../.venv/bin/python -m jupyter nbconvert \
  --execute --to notebook \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir ../../.venv/task41-validation \
  --output flow-ctg.executed.ipynb \
  flow_case_study_ctg.ipynb
```

These are full notebook executions. No quick, smoke or reduced-data result was
used as a substitute.

## Observed validation results

### EOT case study

- execution: 2026-08-12 09:07:03--09:16:43 UTC
- wall time: 579.88 seconds; notebook-reported EOT computation: 542.70 seconds
- structure: 26 cells, including 12/12 executed code cells
- error outputs: none
- executed-copy SHA-256:
  `32d70d99af5bc28de6957436dad9c92516f94c34e8fc2099d913b51ab9f5e9e3`
- analysis matrix: 611 rows by 832 features; 14 feature groups
- `sum(phi_X)`: `0.7396973904610974`
- `sum(phi_Z)`: `0.7396975752007977`
- significant features: X-space 6/832; Z-space 7/832
- significant groups: X-space 10/14; Z-space 11/14

### CTG case study

- execution: 2026-08-12 09:17:31--09:20:24 UTC
- wall time: 172.46 seconds
- structure: 25 cells, including 10/10 executed code cells
- error outputs: none
- executed-copy SHA-256:
  `dc319560e55b4b4e125101eaad822c1bc537231631f84aacdf6bb17a3f00f967`
- analysis matrix: 2,126 rows by 21 features
- full-data Random Forest accuracy: `0.9868`
- distribution-fidelity MMD: `0.031184` (notebook threshold `< 0.05`: pass)
- median latent dCor: `0.092509` (notebook threshold `< 0.10`: pass)
- significant features: 18/21 after FDR adjustment
- significant groups: 1/4 after FDR adjustment
- all three sampling-method comparison blocks completed

## Integrity and handoff status

- Both committed source notebook files still match their recorded Git blobs and
  SHA-256 values after validation.
- Notebook kernel/environment metadata was preserved.
- Execution artifacts, caches and the Conda environment remain under ignored
  `.venv/` and are not part of the commit.
- No files from the dirty primary worktree, `replication/`, existing detached
  worktrees, stashes or the nested Overleaf repository are part of this baseline.
- This baseline must not be merged wholesale into `yifan-dev`. Integration needs
  a separate isolated branch and an explicit audit of the 0.0.10 API/loss changes.
