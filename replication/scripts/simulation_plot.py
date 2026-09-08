"""Generate the two 2x3 benchmarks and D3-style runtime figure from CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI")
RUNTIME_METHODS = ("CPI", "LOCO", "nLOCO", "dLOCO", "OT", "EOT", "FDFI", "SHAP")
COLORS = {
    "LOCO": "#4C78A8",
    "CPI": "#F2A541",
    "DFI-OT": "#8C6BB1",
    "DFI-EOT": "#D95F59",
    "FDFI": "#4E9F88",
}
LINESTYLES = {"LOCO": "-", "CPI": "--", "DFI-OT": "-.", "DFI-EOT": ":", "FDFI": "-"}
MARKERS = {"LOCO": "o", "CPI": "s", "DFI-OT": "^", "DFI-EOT": "D", "FDFI": "P"}


def _select_version(summary: pd.DataFrame, version: str) -> pd.DataFrame:
    return summary.loc[
        (summary["resampling_version"] == version)
        | (summary["resampling_version"] == "baseline")
    ].copy()


def _benchmark_figure(summary: pd.DataFrame, version: str, path: Path) -> None:
    data_for_version = _select_version(summary, version)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
    metrics = (
        ("auc", "AUC score"),
        ("power_c1", r"Power ($C_1$)"),
        ("type1_error", r"Type-I error ($C_3$)"),
    )
    for row, sweep in enumerate(("sample_size", "correlation")):
        subset = data_for_version.loc[data_for_version["sweep"] == sweep]
        x_field = "n" if sweep == "sample_size" else "rho"
        x_label = "Sample size n" if sweep == "sample_size" else r"Correlation $\rho$"
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            for method in METHODS:
                method_data = subset.loc[subset["method"] == method].sort_values(x_field)
                if method_data.empty:
                    continue
                x = method_data[x_field].to_numpy(dtype=float)
                y = method_data[metric].to_numpy(dtype=float)
                axis.plot(
                    x, y, marker=MARKERS[method], linestyle=LINESTYLES[method],
                    linewidth=2, markersize=5.5, color=COLORS[method], label=method,
                )
                axis.fill_between(
                    x,
                    method_data[f"{metric}_ci_lower"].to_numpy(dtype=float),
                    method_data[f"{metric}_ci_upper"].to_numpy(dtype=float),
                    color=COLORS[method], alpha=0.10,
                )
            if metric == "type1_error":
                axis.axhline(0.05, color="#333333", linestyle="--", linewidth=1.5, label=r"$\alpha=0.05$")
                upper = max(0.11, float(subset[metric].max()) * 1.15)
                axis.set_ylim(-0.005, upper)
            elif metric in {"power_c1", "auc"}:
                axis.set_ylim(-0.02, 1.02)
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ref_handles, ref_labels = axes[0, 2].get_legend_handles_labels()
    if r"$\alpha=0.05$" in ref_labels:
        reference = ref_labels.index(r"$\alpha=0.05$")
        handles.append(ref_handles[reference])
        labels.append(ref_labels[reference])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False)
    title_version = version.upper()
    fig.suptitle(f"Benchmarking with {title_version} resampling", y=0.925)
    fig.text(
        0.5, 0.895,
        "AUC uses C1 and C3 only; Type-I error uses independent null set C3; shaded bands are 95% bootstrap intervals.",
        ha="center", va="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _runtime_figure(runtime_summary: pd.DataFrame, path: Path) -> None:
    """Match reference D3: methods on x, sample size series, log-time y."""
    fig, axis = plt.subplots(figsize=(10.5, 3.6))
    sample_sizes = sorted(runtime_summary["n"].unique())
    palette = plt.get_cmap("tab10")(np.arange(len(sample_sizes)))
    method_x = np.arange(len(RUNTIME_METHODS), dtype=float)
    offsets = np.linspace(-0.3, 0.3, len(sample_sizes)) if len(sample_sizes) > 1 else [0.0]
    for index, (n, color, offset) in enumerate(zip(sample_sizes, palette, offsets)):
        subset = runtime_summary.loc[runtime_summary["n"] == n].set_index("method")
        subset = subset.reindex(RUNTIME_METHODS)
        y = subset["mean_runtime_seconds"].to_numpy(dtype=float)
        yerr = subset["std_runtime_seconds"].fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(y) & (y > 0)
        axis.errorbar(
            method_x[valid] + offset, y[valid], yerr=yerr[valid],
            color=color, marker="o", linestyle="none", linewidth=0.9,
            markersize=4.0, capsize=3, label=str(int(n)),
        )
    axis.set_xticks(method_x, RUNTIME_METHODS)
    axis.set_xlabel("Methods")
    axis.set_ylabel("Time (seconds)")
    axis.set_yscale("log")
    axis.grid(
        color="#D9D9D9", linestyle="--", linewidth=0.6,
        alpha=0.65, which="major", axis="y",
    )
    axis.set_axisbelow(True)
    axis.legend(
        title="Sample Size", loc="upper left",
        ncol=min(2, len(sample_sizes)), frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_runtime_figure(runtime_summary_path: Path, figures_dir: Path) -> Path:
    """Render the standalone D3-style computational-cost figure from its CSV."""
    runtime_summary = pd.read_csv(runtime_summary_path)
    runtime_required = {
        "n", "method", "mean_runtime_seconds", "std_runtime_seconds",
    }
    runtime_missing = runtime_required.difference(runtime_summary.columns)
    if runtime_missing:
        raise ValueError(
            f"simulation runtime summary is missing columns: {sorted(runtime_missing)}"
        )
    figures_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = figures_dir / "simulation_runtime.pdf"
    _runtime_figure(runtime_summary, runtime_path)
    return runtime_path


def create_figures(summary_path: Path, runtime_summary_path: Path, figures_dir: Path) -> list[Path]:
    summary = pd.read_csv(summary_path)
    required = {
        "sweep", "n", "rho", "method", "resampling_version",
        "auc", "power_c1", "type1_error",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"simulation benchmark summary is missing columns: {sorted(missing)}")
    cpi_path = figures_dir / "simulation_benchmark_cpi.pdf"
    scpi_path = figures_dir / "simulation_benchmark_scpi.pdf"
    _benchmark_figure(summary, "cpi", cpi_path)
    _benchmark_figure(summary, "scpi", scpi_path)
    runtime_path = create_runtime_figure(runtime_summary_path, figures_dir)
    return [cpi_path, scpi_path, runtime_path]
