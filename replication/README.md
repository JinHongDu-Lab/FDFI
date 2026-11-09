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
│   ├── outputs/
│   │   ├── case1_feature_scores.csv
│   │   ├── case1_group_results_X.csv
│   │   ├── case2_feature_results.csv
│   │   └── case2_group_results.csv
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
- `data/CTG.xls`: data for Case study 2, the Cardiotocography data from the
  UCI Machine Learning Repository.
- `outputs/`: tables written by `case_studies.py`: per-feature scores and
  group results for Case study 1, and feature and group results for Case
  study 2.
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
`case_study/figures/` and the tables in `case_study/outputs/`.

The case studies were run on a laptop with an Intel Core i7-14650HX CPU.
Case study 1 (HIV sens50) took about 50 minutes,
and Case study 2 (CTG) took about 12 minutes,
for a total of about 1 hour.

The script reads all data from `case_study/data/` and needs no internet
connection.

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
- `--seeds N` reruns with the first `N` seeds instead of 50 (benchmark) or 16
  (runtime), for example `python run_benchmark.py --rerun --seeds 1` to check
  the setup. Rerun results and figures are written to `rerun/`.

**Approximate running times.** The simulations were run on a server with dual
Intel Xeon Platinum 8580 CPUs (240 logical cores), 1 TB of RAM, and an NVIDIA
RTX PRO 6000 Blackwell Workstation Edition GPU. FDFI flow matching used the GPU,
while the other methods used the CPUs. The running times are roughly estimated
as follows:

| Command | Reproduces | Approximate time |
|---|---|---|
| `python run_benchmark.py` | Figures 1, 2, and 8 from saved results | A few seconds |
| `python run_benchmark.py --rerun benchmark` | Figures 1 and 8 | About 40 hours |
| `python run_benchmark.py --rerun runtime` | Figure 2 | About 50 hours |
| `python run_benchmark.py --rerun` | Figures 1, 2, and 8 | About 90 hours |
