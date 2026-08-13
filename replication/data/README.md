# Fixed replication inputs

This directory records replication input provenance and checksums. The sens50
inputs remain in their canonical repository location under
`docs/case_studies/data/`; they are not duplicated here. The ignored CTG
workbook may be downloaded locally for validation. A successful local copy does
**not** establish permission to publish or redistribute a dataset.

## sens50 processed data

- File: `../../docs/case_studies/data/sens50_processed_dataset.csv`
- Role: predictor matrix and `sens50` outcome for the EOT case study.
- Repository source: `docs/case_studies/data/sens50_processed_dataset.csv`
- SHA-256: `a7831f7926706377ed9b81fd6c73f995f69f1d971374859828c9300b566a0287`
- Verification: the canonical file matches the pinned hash.
- Status: processed analysis data, not raw acquisition data.
- Dimensions: 611 rows, 833 columns: 832 predictors plus outcome `sens50`.
- Original acquisition source: **UNRESOLVED** from repository evidence.
- Processing history: **UNRESOLVED** beyond the fact that this is the processed
  file consumed by the source notebook.
- Licence: **UNRESOLVED**.
- Redistribution permission: **UNRESOLVED**. Inclusion for local replication
  must not be represented as evidence that public redistribution is allowed.

## sens50 feature groups

- File: `../../docs/case_studies/data/feature_group.csv`
- Role: map all 832 sens50 predictors to 14 biological groups.
- Repository source: `docs/case_studies/data/feature_group.csv`
- SHA-256: `fc519731d770ca4672029877fbcb884a83f859e687851d0390908c15bbc80e97`
- Verification: the canonical file matches the pinned hash.
- Status: processed feature annotation table.
- Structure: 1,155 mapping rows, 832 unique features, 14 unique groups; all
  predictors in the fixed sens50 data are covered.
- Original annotation source: **UNRESOLVED**.
- Licence and redistribution permission: **UNRESOLVED**.

## CTG

- File: `CTG.xls`
- Role: fixed raw input for the CTG / `FlowExplainer` case study.
- URL recorded in the source notebook:
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls`
- Download date: 2026-07-21.
- File size: 1,743,872 bytes.
- SHA-256: `d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421`.
- Status: raw UCI Excel workbook; the file is not manually edited or rewritten.
- Verification: downloaded to `CTG.xls.part`, hashed, and atomically renamed
  only after matching the pinned digest.
- Workbook sheets: `Description`, `Data`, `Raw Data`.
- Read rule: `pd.read_excel(path, sheet_name="Data", skiprows=1, header=0)`
  using `xlrd=2.0.2`.
- Read shape: 2,129 rows by 46 columns.
- Feature selection: zero-based columns `10:31`, in order: `LB`, `AC`, `FM`,
  `UC`, `DL`, `DS`, `DP`, `ASTV`, `MSTV`, `ALTV`, `MLTV`, `Width`, `Min`,
  `Max`, `Nmax`, `Nzeros`, `Mode`, `Mean`, `Median`, `Variance`, `Tendency`.
- Cleaning: numeric coercion followed by complete-case removal.
- Cleaned shape: 2,126 observations by 21 features.
- Outcome: `NSP == 1` gives Normal (`0`); `NSP != 1` gives Other (`1`).
- Class counts: Normal 1,655; Other 471.

The workbook's `Description` sheet supplies dataset field descriptions, but a
reliable licence grant, required citation, and redistribution permission have
not been established from repository evidence. They remain **UNRESOLVED**.
Successful local download does not establish permission to add the workbook to
a public Git repository; redistribution must be confirmed before staging or
committing this data file.
