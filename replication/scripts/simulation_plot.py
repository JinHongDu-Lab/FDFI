"""Generate the three Section 3.5 benchmark figures from saved CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI")
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
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    metrics = (
        ("test_r2", r"Predictive $R^2$"),
        ("power", "Power"),
        ("type1_error", "Type-I error"),
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
            elif metric == "power":
                axis.set_ylim(-0.02, 1.02)
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ref_handles, ref_labels = axes[0, 2].get_legend_handles_labels()
    if r"$\alpha=0.05$" in ref_labels:
        ref_index = ref_labels.index(r"$\alpha=0.05$")
        handles.append(ref_handles[ref_index]); labels.append(ref_labels[ref_index])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False)
    title_version = version.upper()
    fig.suptitle(f"Benchmarking with {title_version} resampling", y=0.925)
    fig.text(
        0.5, 0.895,
        "Lines show repetition means; shaded bands are 95% bootstrap intervals.",
        ha="center", va="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _runtime_figure(runtime_summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for axis, version in zip(axes, ("cpi", "scpi")):
        subset = _select_version(runtime_summary, version)
        subset = subset.loc[subset["sweep"] == "sample_size"]
        for method in METHODS:
            method_data = subset.loc[subset["method"] == method].sort_values("n")
            if method_data.empty:
                continue
            x = method_data["n"].to_numpy(dtype=float)
            y = method_data["mean_runtime_seconds"].to_numpy(dtype=float)
            axis.plot(
                x, y, marker=MARKERS[method], linestyle=LINESTYLES[method],
                linewidth=2, markersize=5.5, color=COLORS[method], label=method,
            )
            axis.fill_between(
                x,
                method_data["mean_runtime_seconds_ci_lower"].to_numpy(dtype=float),
                method_data["mean_runtime_seconds_ci_upper"].to_numpy(dtype=float),
                color=COLORS[method], alpha=0.10,
            )
        axis.set_title(f"{version.upper()} resampling")
        axis.set_xlabel("Sample size n")
        axis.set_yscale("log")
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Runtime (seconds, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False)
    fig.suptitle("Computational cost by sample size", y=0.895)
    fig.text(
        0.5, 0.825,
        "Standalone wall-clock time per method; DFI/FDFI includes transformation or flow fitting and inference.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_figures(summary_path: Path, runtime_summary_path: Path, figures_dir: Path) -> list[Path]:
    summary = pd.read_csv(summary_path)
    runtime_summary = pd.read_csv(runtime_summary_path)
    required = {
        "sweep", "n", "rho", "method", "resampling_version",
        "test_r2", "power", "type1_error",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"simulation benchmark summary is missing columns: {sorted(missing)}")
    cpi_path = figures_dir / "simulation_benchmark_cpi.pdf"
    scpi_path = figures_dir / "simulation_benchmark_scpi.pdf"
    runtime_path = figures_dir / "simulation_runtime.pdf"
    _benchmark_figure(summary, "cpi", cpi_path)
    _benchmark_figure(summary, "scpi", scpi_path)
    _runtime_figure(runtime_summary, runtime_path)
    return [cpi_path, scpi_path, runtime_path]
