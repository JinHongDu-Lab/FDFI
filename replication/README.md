# Replication materials for "fdfi: Disentangled Feature Importance in Python"


This folder contains the software and the replication materials for all
computational results in the manuscript.

## 1. Folder structure

```
fdfi-replication/
├── README.md
├── requirements.txt
├── environment.yml
├── fdfi-0.2.1.tar.gz
├── case_study/
│   ├── case_studies.py
│   ├── data/
│   │   ├── sens50_processed_dataset.csv
│   │   ├── feature_group.csv
│   │   └── CTG.xls
│   └── figures/
│       ├── case1_group_importance.pdf
│       ├── fig3_ctg_correlation_heatmap.pdf
│       ├── fig4_ctg_diagnostics.pdf
│       ├── fig5_ctg_global_importance.pdf
│       ├── fig6_ctg_confidence_intervals.pdf
│       └── fig7_ctg_per_sample_summary.pdf
└── simulation/
    ├── run_benchmark.py
    ├── simulation.py
    ├── simulation_plot.py
    ├── results/
    │   ├── simulation_benchmark_cpi.csv
    │   ├── simulation_benchmark_scpi.csv
    │   ├── runtime_results.csv
    │   └── runtime_summary.csv
    └── figures/
        ├── fig1_benchmark_cpi.pdf
        ├── fig2_runtime.pdf
        └── fig8_benchmark_scpi.pdf
```

**Top level**

- `fdfi-0.2.1.tar.gz`: source code of fdfi 0.2.1, the version used for the
  results in the manuscript, with the `get_*()` result interface used by these
  scripts. The result accessors do not change the numerical calculations.
- `requirements.txt`: the Python packages and versions used for the reported
  results.
- `environment.yml`: the same environment for conda.

**`case_study/`**

- `case_studies.py`: replication script for both case studies. Case study 1
  (Section 4) applies the EOT explainer to the HIV sens50 data. Case study 2
  (Section 5) applies the Flow explainer to the CTG data.
- `data/sens50_processed_dataset.csv`, `data/feature_group.csv`: data for Case study 1.
- `data/CTG.xls`: data for Case study 2.
- `figures/`: figures produced by `case_studies.py`. Files `fig3_*` to `fig7_*`
  are Figures 3–7 in the manuscript. `case1_group_importance.pdf` is a
  supplementary plot for Case study 1 and is not shown in the manuscript.

**`simulation/`**

- `run_benchmark.py`: main script. It draws Figures 1, 2 and 8 from the saved
  results, or reruns the simulations.
- `simulation.py`: data-generating processes, competing methods and
  performance measures.
- `simulation_plot.py`: plotting functions.
- `results/simulation_benchmark_cpi.csv`, `results/simulation_benchmark_scpi.csv`:
  benchmark summaries over 50 seeds (Figures 1 and 8).
- `results/runtime_results.csv`, `results/runtime_summary.csv`: evaluation
  times over 16 repetitions (Figure 2).
- `figures/`: Figures 1, 2 and 8 as reported in the manuscript.

## 2. Installation

The results were obtained with Python 3.10.19. Run the following commands in
this folder.

With pip (requires an existing Python 3.10 installation):

```bash
python3.10 -m venv .venv            # Windows: py -3.10 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


With conda:

```bash
conda env create -f environment.yml
conda activate fdfi-replication
```

Both files pin the same package versions. The path `./fdfi-0.2.1.tar.gz` in
`requirements.txt` is relative, so the command must be run in this folder.

Read explainer output with methods such as `results.get_phi_X()` and
`results.get_se_X()`. For inference, use `ci.get_score()`, `ci.get_ci()`
(the lower and upper bounds), `ci.get_pvalue()` and `ci.get_ranking()`.
Both result classes also provide `to_frame()` for a pandas table.

## 3. Replication

### 3.1 Case studies (Sections 4 and 5)

The script reads data with relative paths. Therefore, it must be run from
`case_study/`:

```bash
cd case_study
python case_studies.py
```

All numerical results are printed to the console. The figures are saved in
`case_study/figures/`.

The case studies were run on a laptop with an Intel Core i7-14650HX CPU under
Windows, with Python 3.10.19, NumPy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2.
The random forests use all CPU cores. Case study 2 uses one thread for BLAS,
OpenMP and PyTorch. The approximate running times are as follows.

| Part | Approximate time |
|---|---|
| Case study 1 (HIV sens50, 611 × 832, `EOTExplainer`) | about 50 minutes |
| Case study 2 (CTG, 2,126 × 21, `FlowExplainer`) | about 12 minutes |
| Total | about 1 hour |

The script reproduces the full manuscript results; it has no shorter
demonstration mode.

The script downloads `CTG.xls` from the UCI repository and thus needs an
internet connection. If the download fails, set `CTG_URL` in
`case_studies.py` to `data/CTG.xls`.

### 3.2 Simulation (Section 3.5 and Appendix B)

Run the following commands in `simulation/`:

```bash
cd simulation
python run_benchmark.py                     # Figures 1, 2 and 8 from results/
python run_benchmark.py --rerun             # rerun both simulations
python run_benchmark.py --rerun benchmark   # rerun Figures 1 and 8 only
python run_benchmark.py --rerun runtime     # rerun Figure 2 only
```

The commands work as follows.

- `python run_benchmark.py` reads the saved results in `results/` and draws
  Figures 1, 2 and 8 in `figures/`. This is the quick way to check the
  simulation figures without rerunning the simulations.
- `python run_benchmark.py --rerun benchmark` repeats the benchmark of
  Appendix B.1–B.2 with 50 seeds and draws Figures 1 and 8.
- `python run_benchmark.py --rerun runtime` repeats the runtime study of
  Appendix B.3 with 16 repetitions and draws Figure 2.
- `python run_benchmark.py --rerun` runs both studies.

**Approximate running times.** The simulations were run on a server with dual
Intel Xeon Platinum 8580 CPUs (240 logical cores), 1 TB of RAM and an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU (see Appendix B.3).
The flow-matching step of FDFI was run on the GPU, and the other methods were run on the CPUs.

| Command | Reproduces | Approximate time |
|---|---|---|
| `python run_benchmark.py` | Figures 1, 2 and 8 from the saved results | a few seconds |
| `python run_benchmark.py --rerun benchmark` | Figures 1 and 8 (50 seeds) | less than 40 hours |
| `python run_benchmark.py --rerun runtime` | Figure 2 (16 repetitions) | about 50 hours |
| `python run_benchmark.py --rerun` | Figures 1, 2 and 8 | about 90 hours |

*Benchmark (50 seeds).* Each scenario runs LOCO and CPI once, and DFI-OT,
DFI-EOT and FDFI with both CPI and SCPI resampling. The scenario n = 1000,
ρ = 0.8 is shared by the two sweeps and is computed once. The recorded time of
each method includes the shared random-forest fit, so the times below are upper
bounds.

| Scenario | Approximate time |
|---|---|
| n = 200, ρ = 0.8 | 2.1 h |
| n = 400, ρ = 0.8 | 3.5 h |
| n = 600, ρ = 0.8 | 5.0 h |
| n = 800, ρ = 0.8 | 6.0 h |
| n = 1000, ρ = 0.8 | 7.6 h |
| n = 1000, ρ = 0.4 | 7.5 h |
| n = 1000, ρ = 0.6 | 7.3 h |
| Total | less than 40 h |

*Runtime study (16 repetitions).* The times include method evaluation only.
SHAP is evaluated on 10 features, and the other methods on 50 features.

| n | LOCO | CPI | nLOCO | dLOCO | DFI-OT | DFI-EOT | FDFI | SHAP | Total |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 0.44 h | 0.39 h | 0.91 h | 0.53 h | 0.03 h | 0.03 h | 0.12 h | 6.42 h | 8.9 h |
| 400 | 0.45 h | 0.48 h | 0.94 h | 0.73 h | 0.06 h | 0.05 h | 0.21 h | 6.60 h | 9.5 h |
| 600 | 0.42 h | 0.57 h | 0.87 h | 1.03 h | 0.09 h | 0.09 h | 0.27 h | 6.44 h | 9.8 h |
| 800 | 0.42 h | 0.67 h | 0.91 h | 1.40 h | 0.12 h | 0.12 h | 0.34 h | 6.70 h | 10.7 h |
| 1000 | 0.43 h | 0.74 h | 0.88 h | 1.73 h | 0.16 h | 0.16 h | 0.43 h | 7.04 h | 11.6 h |
| Total | | | | | | | | | about 50 h |
