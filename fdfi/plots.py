"""
Plotting utilities for Flow-Disentangled Feature Importance.

The functions in this module provide static Matplotlib visualizations for
global FDFI scores, per-sample UEIFs, confidence intervals, disentanglement
diagnostics, and feature-correlation structure.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster import hierarchy
from scipy.spatial import distance

ArrayLike = Union[np.ndarray, Sequence[float]]
FeatureRef = Union[int, str]

__all__ = [
    "summary_plot",
    "waterfall_plot",
    "force_plot",
    "dependence_plot",
    "correlation_heatmap",
    "summary_bar",
    "cv_scatter",
    "confidence_interval_plot",
    "diagnostics_plot",
]


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    """Return a finite-shape 1D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape {arr.shape}")
    return arr


def _as_2d(values: ArrayLike, name: str) -> np.ndarray:
    """Return a 2D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional; got shape {arr.shape}")
    return arr


def _feature_names(feature_names: Optional[Sequence[Any]], n_features: int) -> List[str]:
    """Validate or create feature names."""
    if feature_names is None:
        return [f"Feature {i}" for i in range(n_features)]
    names = [str(name) for name in feature_names]
    if len(names) != n_features:
        raise ValueError(
            f"feature_names has length {len(names)} but expected {n_features}"
        )
    return names


def _resolve_feature_index(
    feature: FeatureRef,
    feature_names: Optional[Sequence[Any]],
    n_features: int,
    name: str,
) -> int:
    """Resolve an integer or string feature reference to a column index."""
    if isinstance(feature, str):
        names = _feature_names(feature_names, n_features)
        if feature not in names:
            raise ValueError(f"{name}={feature!r} is not in feature_names")
        return names.index(feature)

    if isinstance(feature, (int, np.integer)):
        idx = int(feature)
        if idx < 0:
            idx += n_features
        if idx < 0 or idx >= n_features:
            raise ValueError(f"{name} index {feature} is out of bounds")
        return idx

    raise TypeError(f"{name} must be an integer index or feature name")


def _top_order(scores: np.ndarray, max_display: Optional[int]) -> np.ndarray:
    """Return stable descending order by absolute score."""
    if max_display is not None and max_display <= 0:
        raise ValueError("max_display must be positive")
    order = np.argsort(-np.abs(scores), kind="stable")
    if max_display is not None:
        order = order[: min(int(max_display), len(order))]
    return order


def _fig_ax(
    ax: Optional[Axes],
    figsize: Optional[Tuple[float, float]],
) -> Tuple[Figure, Axes]:
    """Create or reuse a Matplotlib figure/axes pair."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish_figure(
    fig: Figure,
    savepath: Optional[str],
    show: bool,
    *,
    tight_layout: bool = True,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> None:
    """Apply layout, optional save, and optional display."""
    if tight_layout:
        fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()


def _sanitize_se(se: Optional[ArrayLike], n_features: int) -> np.ndarray:
    """Return nonnegative finite standard errors suitable for error bars."""
    if se is None:
        return np.zeros(n_features, dtype=float)

    se_arr = _as_1d(se, "se_X")
    if se_arr.shape[0] != n_features:
        raise ValueError(f"se_X has length {se_arr.shape[0]} but expected {n_features}")

    se_arr = np.abs(se_arr.astype(float, copy=True))
    finite = np.isfinite(se_arr)
    max_finite = float(np.max(se_arr[finite])) if np.any(finite) else 0.0
    se_arr = np.where(np.isnan(se_arr), 0.0, se_arr)
    se_arr = np.where(np.isinf(se_arr), max_finite, se_arr)
    return se_arr


def _summary_dataframe(
    feature_names: Sequence[str],
    phi: np.ndarray,
    se: np.ndarray,
):
    """Build the sorted summary table, using pandas when available."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is installed via seaborn
        raise ImportError(
            "summary_bar returns a pandas DataFrame. Install pandas to use it."
        ) from exc

    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "phi": np.abs(phi),
                "se": se,
            }
        )
        .sort_values("phi", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def cv_scatter(
    phi_X: ArrayLike,
    se_X: ArrayLike,
    feature_names: Sequence[Any],
    y_cutoff: float = 3.0,
    group_colors: Optional[Mapping[str, Any]] = None,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """Plot coefficient of variation against global feature importance.

    This restores the public plotting helper previously published on
    ``yifan-dev``. High-CV features remain visible as capped hollow triangles
    rather than being silently dropped.

    Parameters
    ----------
    phi_X : array-like of shape (n_features,)
        Global feature-importance scores. Absolute values are plotted.
    se_X : array-like of shape (n_features,)
        Standard errors corresponding to ``phi_X``.
    feature_names : sequence of str
        Human-readable feature names.
    y_cutoff : float, default=3.0
        CV threshold above which markers are capped at the top of the plot.
    group_colors : mapping, optional
        Mapping from feature name to Matplotlib color.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options including ``figsize``, ``title_fontsize``,
        ``label_fontsize``, ``tick_fontsize``, ``dpi``, ``bbox_inches``,
        ``outlier_cap``, and ``show``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    cv_df_filtered : pandas.DataFrame
        Features whose CV is at most ``y_cutoff``.
    cv_df_outliers : pandas.DataFrame
        Features whose CV exceeds ``y_cutoff``.

    Notes
    -----
    CV is computed as ``se / (abs(phi) + 1e-8)``. The epsilon preserves
    zero-importance features with non-zero uncertainty as high-CV observations.

    Examples
    --------
    >>> phi = np.array([0.05, 0.0, 0.02])
    >>> se = np.array([0.005, 0.003, 0.002])
    >>> fig, ax, stable, noisy = cv_scatter(
    ...     phi, se, ["A", "B", "C"], show=False
    ... )
    """
    if y_cutoff <= 0:
        raise ValueError("y_cutoff must be positive")

    phi = _as_1d(phi_X, "phi_X")
    se = _sanitize_se(se_X, phi.shape[0])
    names = _feature_names(feature_names, phi.shape[0])

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas installed via seaborn
        raise ImportError(
            "cv_scatter returns pandas DataFrames. Install pandas to use it."
        ) from exc

    cv_df = pd.DataFrame(
        {
            "feature": names,
            "phi": np.abs(phi),
            "se": se,
        }
    )
    cv_df["cv"] = cv_df["se"] / (cv_df["phi"] + 1e-8)
    cv_df_filtered = (
        cv_df[cv_df["cv"] <= y_cutoff]
        .copy()
        .sort_values("phi", ascending=False, kind="mergesort")
    )
    cv_df_outliers = (
        cv_df[cv_df["cv"] > y_cutoff]
        .copy()
        .sort_values("phi", ascending=False, kind="mergesort")
    )

    fig, ax = _fig_ax(None, kwargs.get("figsize", (10.0, 4.0)))
    y_top = y_cutoff + 0.5
    outlier_cap = min(
        float(kwargs.get("outlier_cap", y_cutoff + 0.4)), y_top - 0.05
    )

    def colors(frame):
        if group_colors is None:
            return ["#4878a8"] * len(frame)
        return [group_colors.get(name, "#888888") for name in frame["feature"]]

    if not cv_df_filtered.empty:
        ax.scatter(
            cv_df_filtered["phi"],
            cv_df_filtered["cv"],
            c=colors(cv_df_filtered),
            s=85,
            marker="o",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    if not cv_df_outliers.empty:
        ax.scatter(
            cv_df_outliers["phi"],
            np.full(len(cv_df_outliers), outlier_cap),
            s=100,
            marker="v",
            facecolors="none",
            edgecolors=colors(cv_df_outliers),
            linewidths=1.2,
            zorder=4,
        )

    tick_fontsize = kwargs.get("tick_fontsize", 7.5)
    for _, row in cv_df_filtered.iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["phi"], row["cv"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=tick_fontsize,
        )
    for _, row in cv_df_outliers.iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["phi"], outlier_cap),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=tick_fontsize,
        )

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.1, label="CV = 1")
    ax.set_xlabel(
        r"Mean $|\phi_X|$ (importance)", fontsize=kwargs.get("label_fontsize", 10)
    )
    ax.set_ylabel(
        "Coefficient of Variation (se / phi)",
        fontsize=kwargs.get("label_fontsize", 10),
    )
    ax.set_ylim(-0.05, y_top)
    ax.set_title(
        "Attribution Reliability: Importance vs. Coefficient of Variation\n"
        f"(Features with CV > {y_cutoff:g} are capped at the top edge)",
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")
    _finish_figure(
        fig,
        savepath,
        kwargs.get("show", True),
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax, cv_df_filtered, cv_df_outliers


def _group_remaining(
    values: np.ndarray,
    names: Sequence[str],
    max_display: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select top waterfall/force features and optionally group the remainder."""
    if max_display <= 0:
        raise ValueError("max_display must be positive")

    n_features = values.shape[0]
    order = _top_order(values, None)
    if n_features <= max_display:
        top = order
        return values[top], [names[i] for i in top], top

    top_count = max_display - 1
    top = order[:top_count]
    rest = order[top_count:]
    grouped_values = np.concatenate([values[top], [values[rest].sum()]])
    grouped_names = [names[i] for i in top]
    grouped_names.append(f"{len(rest)} remaining features")
    grouped_order = np.concatenate([top, [-1]])
    return grouped_values, grouped_names, grouped_order


def _feature_value_label(name: str, value: Optional[float]) -> str:
    """Combine a feature name and displayed value for single-sample plots."""
    if value is None or not np.isfinite(value):
        return name
    return f"{name} = {value:.3g}"


def summary_bar(
    phi_X: ArrayLike,
    se_X: Optional[ArrayLike] = None,
    feature_names: Optional[Sequence[Any]] = None,
    group_colors: Optional[Mapping[str, Any]] = None,
    savepath: Optional[str] = None,
    max_display: Optional[int] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    **kwargs: Any,
):
    """
    Plot global FDFI feature importance as a sorted bar chart.

    Parameters
    ----------
    phi_X : array-like of shape (n_features,)
        Global FDFI scores such as ``results["phi_X"]`` or ``results["phi_Z"]``.
        Bar lengths use ``abs(phi_X)`` so signed attribution summaries are also
        supported.
    se_X : array-like of shape (n_features,), optional
        Standard errors such as ``results["se_X"]``. Missing values default to
        zero. NaN, inf, and negative entries are sanitized before plotting.
    feature_names : sequence of str, optional
        Feature names. Defaults to ``Feature 0``, ``Feature 1``, ...
    group_colors : mapping, optional
        Mapping from feature name to a Matplotlib color. Missing features use a
        neutral gray. When omitted, a colormap gradient is used.
    savepath : str, optional
        Path where the figure should be saved.
    max_display : int, optional
        Maximum number of features to show.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    **kwargs
        Styling options. Common keys include ``figsize``, ``title``, ``cmap``,
        ``capsize``, ``elinewidth``, ``dpi``, and ``bbox_inches``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    importance_df : pandas.DataFrame
        Sorted table with columns ``feature``, ``phi``, and ``se``.

    Examples
    --------
    >>> fig, ax, table = summary_bar(
    ...     results["phi_X"],
    ...     se_X=results["se_X"],
    ...     feature_names=feature_names,
    ...     show=False,
    ... )
    """
    phi = _as_1d(phi_X, "phi_X")
    names = _feature_names(feature_names, phi.shape[0])
    se = _sanitize_se(se_X, phi.shape[0])
    importance_df = _summary_dataframe(names, phi, se)

    if max_display is not None:
        if max_display <= 0:
            raise ValueError("max_display must be positive")
        plot_df = importance_df.head(int(max_display)).copy()
    else:
        plot_df = importance_df

    figsize = kwargs.get("figsize", (8.0, max(3.5, 0.34 * len(plot_df) + 1.5)))
    fig, ax = _fig_ax(ax, figsize)

    if group_colors is None:
        cmap = plt.get_cmap(kwargs.get("cmap", "viridis"))
        values = plot_df["phi"].to_numpy(dtype=float)
        if len(values) == 0 or float(values.max()) == float(values.min()):
            normalized = np.full(len(values), 0.55)
        else:
            normalized = (values - values.min()) / (values.max() - values.min())
        colors = [cmap(value) for value in normalized]
    else:
        colors = [group_colors.get(feature, "#888888") for feature in plot_df["feature"]]

    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["phi"],
        xerr=plot_df["se"],
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        capsize=kwargs.get("capsize", 4),
        error_kw={
            "elinewidth": kwargs.get("elinewidth", 1.2),
            "ecolor": kwargs.get("error_color", "#222222"),
            "capthick": kwargs.get("elinewidth", 1.2),
        },
        zorder=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=kwargs.get("tick_fontsize", 9))
    ax.invert_yaxis()
    ax.set_xlabel(kwargs.get("xlabel", "Mean absolute FDFI score"))
    ax.set_ylabel(kwargs.get("ylabel", "Feature"))
    ax.set_title(
        kwargs.get("title", "Global FDFI Feature Importance"),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax, importance_df


def summary_plot(
    shap_values: ArrayLike,
    features: Optional[ArrayLike] = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 20,
    show: bool = True,
    ax: Optional[Axes] = None,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Create a SHAP-like summary plot for FDFI attributions.

    Parameters
    ----------
    shap_values : array-like of shape (n_features,) or (n_samples, n_features)
        FDFI attribution values. Use aggregate arrays such as ``results["phi_X"]``
        for bar summaries, or per-sample arrays such as ``explainer.ueifs_X`` for
        beeswarm summaries.
    features : array-like of shape (n_samples, n_features), optional
        Feature values used for point colors in a 2D beeswarm plot.
    feature_names : sequence of str, optional
        Feature names.
    max_display : int, default=20
        Maximum number of features to display.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options. For 1D input, forwarded to :func:`summary_bar`.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes for 2D input.
    fig, ax, importance_df : tuple
        For 1D input, the return value from :func:`summary_bar`.

    Examples
    --------
    >>> summary_plot(explainer.ueifs_X, features=X_test, show=False)
    >>> summary_plot(results["phi_X"], se_X=results["se_X"], show=False)
    """
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 1:
        return summary_bar(
            values,
            se_X=kwargs.pop("se_X", None),
            feature_names=feature_names,
            group_colors=kwargs.pop("group_colors", None),
            savepath=savepath,
            max_display=max_display,
            ax=ax,
            show=show,
            **kwargs,
        )

    if values.ndim != 2:
        raise ValueError(f"shap_values must be 1D or 2D; got shape {values.shape}")

    n_samples, n_features = values.shape
    names = _feature_names(feature_names, n_features)
    if max_display <= 0:
        raise ValueError("max_display must be positive")

    feature_values = None
    if features is not None:
        feature_values = _as_2d(features, "features")
        if feature_values.shape != values.shape:
            raise ValueError(
                "features must have the same shape as shap_values; "
                f"got {feature_values.shape} and {values.shape}"
            )

    mean_abs = np.nanmean(np.abs(values), axis=0)
    order = _top_order(mean_abs, max_display)
    plot_names = [names[i] for i in order]

    figsize = kwargs.get("figsize", (8.5, max(3.5, 0.35 * len(order) + 1.6)))
    fig, ax = _fig_ax(ax, figsize)
    cmap = kwargs.get("cmap", "coolwarm")
    dot_size = kwargs.get("dot_size", 18)
    alpha = kwargs.get("alpha", 0.75)

    color_values = None
    scatter = None
    if feature_values is not None:
        color_values = feature_values[:, order].reshape(-1)
        finite_color = color_values[np.isfinite(color_values)]
        if finite_color.size:
            vmin, vmax = np.percentile(finite_color, [5, 95])
            if vmin == vmax:
                vmin, vmax = float(finite_color.min()), float(finite_color.max())
        else:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    for row, feature_index in enumerate(order):
        x = values[:, feature_index]
        finite = np.isfinite(x)
        if not np.any(finite):
            continue
        x = x[finite]
        offsets = ((np.arange(x.shape[0]) % 9) - 4) * 0.035
        y = np.full(x.shape[0], row, dtype=float) + offsets

        if feature_values is None:
            scatter = ax.scatter(
                x,
                y,
                s=dot_size,
                alpha=alpha,
                color=kwargs.get("color", "#1f77b4"),
                edgecolors="none",
                rasterized=kwargs.get("rasterized", False),
            )
        else:
            c = feature_values[:, feature_index][finite]
            scatter = ax.scatter(
                x,
                y,
                s=dot_size,
                alpha=alpha,
                c=c,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
                rasterized=kwargs.get("rasterized", False),
            )

    ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(plot_names, fontsize=kwargs.get("tick_fontsize", 9))
    ax.invert_yaxis()
    ax.set_xlabel(kwargs.get("xlabel", "FDFI attribution value"))
    ax.set_ylabel(kwargs.get("ylabel", "Feature"))
    ax.set_title(
        kwargs.get("title", "FDFI Per-sample Attribution Summary"),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)

    if feature_values is not None and scatter is not None and kwargs.get(
        "color_bar", True
    ):
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(kwargs.get("colorbar_label", "Feature value"))

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax


def waterfall_plot(
    shap_values: ArrayLike,
    features: Optional[ArrayLike] = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 10,
    base_value: float = 0.0,
    show: bool = True,
    ax: Optional[Axes] = None,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Create a single-explanation waterfall plot.

    Parameters
    ----------
    shap_values : array-like of shape (n_features,)
        Single-sample FDFI attributions, for example ``explainer.ueifs_X[0]``.
    features : array-like of shape (n_features,), optional
        Feature values for label annotations.
    feature_names : sequence of str, optional
        Feature names.
    max_display : int, default=10
        Maximum number of features to display. Extra features are summed into a
        final "remaining features" row.
    base_value : float, default=0.0
        Starting value for the additive explanation.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Examples
    --------
    >>> waterfall_plot(explainer.ueifs_X[0], feature_names=feature_names, show=False)
    """
    values = _as_1d(shap_values, "shap_values")
    n_features = values.shape[0]
    names = _feature_names(feature_names, n_features)

    feature_values = None
    if features is not None:
        feature_values = _as_1d(features, "features")
        if feature_values.shape[0] != n_features:
            raise ValueError(
                f"features has length {feature_values.shape[0]} but expected {n_features}"
            )

    grouped_values, grouped_names, grouped_order = _group_remaining(
        values, names, max_display
    )
    labels = []
    for label, original_index in zip(grouped_names, grouped_order):
        if original_index == -1 or feature_values is None:
            labels.append(label)
        else:
            labels.append(_feature_value_label(label, feature_values[original_index]))

    figsize = kwargs.get("figsize", (8.5, max(3.2, 0.45 * len(grouped_values) + 1.4)))
    fig, ax = _fig_ax(ax, figsize)

    positive_color = kwargs.get("positive_color", "#d62728")
    negative_color = kwargs.get("negative_color", "#1f77b4")
    running = float(base_value)
    y_pos = np.arange(len(grouped_values))

    for y, contribution in zip(y_pos, grouped_values):
        color = positive_color if contribution >= 0 else negative_color
        ax.barh(
            y,
            contribution,
            left=running,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        next_value = running + contribution
        ax.plot(
            [next_value, next_value],
            [y - 0.38, y + 0.38],
            color="#555555",
            linewidth=0.6,
            alpha=0.5,
        )
        running = next_value

    final_value = float(base_value + grouped_values.sum())
    ax.axvline(base_value, color="#333333", linestyle="--", linewidth=1.0)
    ax.axvline(final_value, color="#333333", linestyle="-", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=kwargs.get("tick_fontsize", 9))
    ax.invert_yaxis()
    ax.set_xlabel(kwargs.get("xlabel", "Model output"))
    ax.set_title(
        kwargs.get(
            "title",
            f"FDFI Waterfall (base={base_value:.3g}, final={final_value:.3g})",
        ),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax


def force_plot(
    base_value: float,
    shap_values: ArrayLike,
    features: Optional[ArrayLike] = None,
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 10,
    show: bool = True,
    ax: Optional[Axes] = None,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Create a static force-style contribution plot.

    Parameters
    ----------
    base_value : float
        Baseline model output.
    shap_values : array-like of shape (n_features,) or (n_samples, n_features)
        FDFI attributions. If a 2D array is supplied, values are averaged across
        samples before plotting.
    features : array-like, optional
        Feature values for single-sample label annotations.
    feature_names : sequence of str, optional
        Feature names.
    max_display : int, default=10
        Maximum number of features to display. Extra features are grouped.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Examples
    --------
    >>> force_plot(0.0, explainer.ueifs_X[0], feature_names=feature_names, show=False)
    """
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 2:
        values = np.nanmean(values, axis=0)
    elif values.ndim != 1:
        raise ValueError(f"shap_values must be 1D or 2D; got shape {values.shape}")

    n_features = values.shape[0]
    names = _feature_names(feature_names, n_features)
    feature_values = None
    if features is not None:
        feature_values = _as_1d(features, "features")
        if feature_values.shape[0] != n_features:
            raise ValueError(
                f"features has length {feature_values.shape[0]} but expected {n_features}"
            )

    grouped_values, grouped_names, grouped_order = _group_remaining(
        values, names, max_display
    )
    labels = []
    for label, original_index in zip(grouped_names, grouped_order):
        if original_index == -1 or feature_values is None:
            labels.append(label)
        else:
            labels.append(_feature_value_label(label, feature_values[original_index]))

    figsize = kwargs.get("figsize", (9.0, 2.8))
    fig, ax = _fig_ax(ax, figsize)
    positive_color = kwargs.get("positive_color", "#d62728")
    negative_color = kwargs.get("negative_color", "#1f77b4")
    height = kwargs.get("height", 0.45)

    pos_cursor = float(base_value)
    neg_cursor = float(base_value)
    for contribution, label in zip(grouped_values, labels):
        if contribution >= 0:
            left = pos_cursor
            pos_cursor += contribution
            y = 0.18
            color = positive_color
            va = "bottom"
            text_y = y + height / 2 + 0.04
        else:
            left = neg_cursor + contribution
            neg_cursor += contribution
            y = -0.18 - height
            color = negative_color
            va = "top"
            text_y = y - 0.04

        ax.barh(
            y,
            abs(contribution),
            left=left,
            height=height,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        if abs(contribution) > kwargs.get("label_min_width", 0.0):
            ax.text(
                left + abs(contribution) / 2,
                text_y,
                label,
                ha="center",
                va=va,
                fontsize=kwargs.get("label_fontsize", 8),
                rotation=kwargs.get("label_rotation", 0),
            )

    final_value = float(base_value + grouped_values.sum())
    ax.axvline(base_value, color="#333333", linestyle="--", linewidth=1.0)
    ax.axvline(final_value, color="#333333", linestyle="-", linewidth=1.0)
    ax.text(base_value, 0.78, "base", ha="center", fontsize=9)
    ax.text(final_value, 0.78, "final", ha="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel(kwargs.get("xlabel", "Model output"))
    ax.set_title(
        kwargs.get(
            "title", f"FDFI Force Plot (base={base_value:.3g}, final={final_value:.3g})"
        ),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_ylim(-0.9, 1.0)

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax


def dependence_plot(
    feature_idx: FeatureRef,
    shap_values: ArrayLike,
    features: ArrayLike,
    feature_names: Optional[Sequence[Any]] = None,
    interaction_index: Optional[FeatureRef] = None,
    show: bool = True,
    ax: Optional[Axes] = None,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Create a feature-dependence scatter plot.

    Parameters
    ----------
    feature_idx : int or str
        Feature index or feature name to plot.
    shap_values : array-like of shape (n_samples, n_features)
        Per-sample FDFI attributions such as ``explainer.ueifs_X``.
    features : array-like of shape (n_samples, n_features)
        Feature values for the same samples.
    feature_names : sequence of str, optional
        Feature names. Required when ``feature_idx`` or ``interaction_index`` is
        a string.
    interaction_index : int or str, optional
        Feature used to color points.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Examples
    --------
    >>> dependence_plot("age", explainer.ueifs_X, X_test,
    ...                 feature_names=feature_names, show=False)
    """
    values = _as_2d(shap_values, "shap_values")
    feature_values = _as_2d(features, "features")
    if feature_values.shape != values.shape:
        raise ValueError(
            "features must have the same shape as shap_values; "
            f"got {feature_values.shape} and {values.shape}"
        )

    n_samples, n_features = values.shape
    names = _feature_names(feature_names, n_features)
    idx = _resolve_feature_index(feature_idx, names, n_features, "feature_idx")

    color = kwargs.get("color", "#1f77b4")
    color_values = None
    color_label = None
    if interaction_index is not None:
        color_idx = _resolve_feature_index(
            interaction_index, names, n_features, "interaction_index"
        )
        color_values = feature_values[:, color_idx]
        color_label = names[color_idx]

    figsize = kwargs.get("figsize", (6.5, 4.8))
    fig, ax = _fig_ax(ax, figsize)

    if color_values is None:
        scatter = ax.scatter(
            feature_values[:, idx],
            values[:, idx],
            s=kwargs.get("dot_size", 28),
            alpha=kwargs.get("alpha", 0.75),
            color=color,
            edgecolors="none",
        )
    else:
        scatter = ax.scatter(
            feature_values[:, idx],
            values[:, idx],
            s=kwargs.get("dot_size", 28),
            alpha=kwargs.get("alpha", 0.75),
            c=color_values,
            cmap=kwargs.get("cmap", "viridis"),
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(color_label)

    ax.axhline(0.0, color="#555555", linewidth=0.8, zorder=0)
    ax.set_xlabel(kwargs.get("xlabel", names[idx]))
    ax.set_ylabel(kwargs.get("ylabel", f"FDFI attribution for {names[idx]}"))
    ax.set_title(
        kwargs.get("title", f"FDFI Dependence: {names[idx]}"),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(linestyle="--", alpha=0.25)

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax


def correlation_heatmap(
    X_background: ArrayLike,
    feature_names: Optional[Sequence[Any]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    ax: Optional[Axes] = None,
    **kwargs: Any,
):
    """
    Plot a clustered Pearson correlation heatmap for background features.

    Parameters
    ----------
    X_background : array-like of shape (n_samples, n_features)
        Background or training feature matrix used to estimate correlation
        structure.
    feature_names : sequence of str, optional
        Feature names. Defaults to ``Feature 0``, ``Feature 1``, ...
    savepath : str, optional
        Path where the figure should be saved.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    **kwargs
        Styling options including ``figsize``, ``cmap``, ``vmin``, ``vmax``,
        ``fontsize``, ``dpi``, and ``bbox_inches``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.
    feature_names_reordered : list of str
        Feature names in clustered order.

    Examples
    --------
    >>> correlation_heatmap(X_background, feature_names, show=False)
    """
    X = _as_2d(X_background, "X_background")
    n_samples, n_features = X.shape
    names = _feature_names(feature_names, n_features)

    if n_samples < kwargs.get("sample_warning_threshold", 50):
        warnings.warn(
            f"X_background has only {n_samples} samples; correlation estimates "
            "may be unstable. Use a representative background set when possible.",
            UserWarning,
            stacklevel=2,
        )

    corr = np.corrcoef(X, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    if n_features > 1:
        dist_matrix = 1.0 - np.abs(corr)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)
        linkage_matrix = hierarchy.linkage(
            distance.squareform(dist_matrix, checks=False), method="average"
        )
        leaf_order = hierarchy.dendrogram(linkage_matrix, no_plot=True)["leaves"]
    else:
        leaf_order = [0]

    corr_reordered = corr[np.ix_(leaf_order, leaf_order)]
    names_reordered = [names[i] for i in leaf_order]

    figsize = kwargs.get("figsize", (8.5, 7.0))
    fig, ax = _fig_ax(ax, figsize)
    im = ax.imshow(
        corr_reordered,
        cmap=kwargs.get("cmap", "RdBu_r"),
        vmin=kwargs.get("vmin", -1),
        vmax=kwargs.get("vmax", 1),
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(kwargs.get("colorbar_label", "Pearson r"))

    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    fontsize = kwargs.get("fontsize", 9)
    ax.set_xticklabels(names_reordered, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(names_reordered, fontsize=fontsize)
    ax.set_title(
        kwargs.get(
            "title",
            "Pearson Correlation Matrix (clustered by absolute correlation)",
        ),
        fontsize=kwargs.get("title_fontsize", 11),
    )

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax, names_reordered


def confidence_interval_plot(
    ci_results: Mapping[str, Any],
    feature_names: Optional[Sequence[Any]] = None,
    max_display: int = 20,
    ax: Optional[Axes] = None,
    show: bool = True,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Plot FDFI confidence intervals from ``conf_int()`` output.

    For two-sided CIs both error bar arms are drawn with flat caps.  For
    one-sided CIs (``alternative='greater'`` or ``'less'``) the open arm is
    rendered as a short stub whose cap is replaced by an outward-pointing caret
    (► or ◄), following the forest-plot truncation convention and using
    Matplotlib's native ``xuplims`` / ``xlolims`` limit-indicator support.

    Parameters
    ----------
    ci_results : mapping
        Dictionary returned by ``explainer.conf_int()``. Required keys are
        ``score``, ``ci_lower``, and ``ci_upper``. ``reject_null``,
        ``ranking``, and ``alternative`` are used when present.
    feature_names : sequence of str, optional
        Feature names. If ``ci_results`` contains ``groups``, those group names
        are used by default.
    max_display : int, default=20
        Maximum number of rows to display.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options.  One-sided-specific keys:

        ``stub_fraction`` : float, default 0.06
            Fraction of the axis width used for the open-arm stub.
        ``show_alternative_note`` : bool, default True
            Show a corner annotation describing the open bound.
        ``note_fontsize`` : int, default 8
            Font size for the corner annotation.
        ``marker`` : str, default ``'o'``
            Marker style for point estimates.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Examples
    --------
    >>> ci = explainer.conf_int(alpha=0.05, target="X")
    >>> confidence_interval_plot(ci, feature_names=feature_names, show=False)

    >>> ci_one = explainer.conf_int(alpha=0.05, alternative="greater")
    >>> confidence_interval_plot(ci_one, feature_names=feature_names, show=False)
    """
    required = ("score", "ci_lower", "ci_upper")
    missing = [key for key in required if key not in ci_results]
    if missing:
        raise ValueError(f"ci_results is missing required keys: {missing}")
    if max_display <= 0:
        raise ValueError("max_display must be positive")

    # -- Alternative hypothesis ------------------------------------------
    alternative = ci_results.get("alternative", "two-sided")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"ci_results['alternative'] must be 'two-sided', 'greater', or 'less';"
            f" got {alternative!r}"
        )

    score = _as_1d(ci_results["score"], "ci_results['score']")
    lower = _as_1d(ci_results["ci_lower"], "ci_results['ci_lower']")
    upper = _as_1d(ci_results["ci_upper"], "ci_results['ci_upper']")
    n_features = score.shape[0]
    if lower.shape[0] != n_features or upper.shape[0] != n_features:
        raise ValueError("score, ci_lower, and ci_upper must have the same length")

    if feature_names is None and "groups" in ci_results:
        names = _feature_names(ci_results["groups"], n_features)
    else:
        names = _feature_names(feature_names, n_features)

    if "ranking" in ci_results:
        ranking = _as_1d(ci_results["ranking"], "ci_results['ranking']")
        if ranking.shape[0] != n_features:
            raise ValueError("ranking must have the same length as score")
        order = np.argsort(ranking, kind="stable")
    else:
        order = np.argsort(-np.abs(score), kind="stable")
    order = order[: min(max_display, n_features)]

    reject = np.asarray(ci_results.get("reject_null", np.zeros(n_features, dtype=bool)))
    if reject.shape[0] != n_features:
        raise ValueError("reject_null must have the same length as score")

    plot_score = score[order]
    plot_lower = lower[order]
    plot_upper = upper[order]
    plot_names = [names[i] for i in order]
    plot_reject = reject[order].astype(bool)

    # -- Axis limits: exclude the infinite bound for one-sided CIs -------
    finite_scores = plot_score[np.isfinite(plot_score)]
    if alternative == "greater":
        finite_values = np.concatenate(
            [finite_scores, plot_lower[np.isfinite(plot_lower)]]
        )
    elif alternative == "less":
        finite_values = np.concatenate(
            [finite_scores, plot_upper[np.isfinite(plot_upper)]]
        )
    else:
        finite_values = np.concatenate(
            [
                finite_scores,
                plot_lower[np.isfinite(plot_lower)],
                plot_upper[np.isfinite(plot_upper)],
            ]
        )

    if finite_values.size == 0:
        finite_min, finite_max = -1.0, 1.0
    else:
        finite_min = float(finite_values.min())
        finite_max = float(finite_values.max())
        if finite_min == finite_max:
            finite_min -= 1.0
            finite_max += 1.0

    pad = 0.08 * (finite_max - finite_min)
    clip_min = finite_min - pad
    clip_max = finite_max + pad

    # Stub length for the open arm; also widen axis so the caret fits
    stub_fraction = kwargs.get("stub_fraction", 0.06)
    stub = stub_fraction * (clip_max - clip_min)
    if alternative == "greater":
        clip_max += stub
    elif alternative == "less":
        clip_min -= stub

    # -- Build xerr (2 x n): [left_arm, right_arm] -----------------------
    if alternative == "greater":
        left_bound = np.where(np.isfinite(plot_lower), plot_lower, plot_score - stub)
        left_arm = np.maximum(plot_score - left_bound, 0.0)
        right_arm = np.full(len(order), stub)
        xerr = np.vstack([left_arm, right_arm])
    elif alternative == "less":
        right_bound = np.where(np.isfinite(plot_upper), plot_upper, plot_score + stub)
        right_arm = np.maximum(right_bound - plot_score, 0.0)
        left_arm = np.full(len(order), stub)
        xerr = np.vstack([left_arm, right_arm])
    else:
        lower_plot = np.where(np.isfinite(plot_lower), plot_lower, clip_min)
        upper_plot = np.where(np.isfinite(plot_upper), plot_upper, clip_max)
        xerr = np.vstack([plot_score - lower_plot, upper_plot - plot_score])
        xerr = np.maximum(xerr, 0.0)

    # -- Draw ------------------------------------------------------------
    figsize = kwargs.get("figsize", (8.0, max(3.5, 0.34 * len(order) + 1.5)))
    fig, ax = _fig_ax(ax, figsize)
    y_pos = np.arange(len(order))
    colors = np.where(plot_reject, kwargs.get("significant_color", "#d62728"), "#777777")
    marker = kwargs.get("marker", "o")

    for y, value, err, color in zip(y_pos, plot_score, xerr.T, colors):
        eb_kwargs: Dict[str, Any] = dict(
            fmt=marker,
            color=kwargs.get("interval_color", "#333333"),
            ecolor=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=kwargs.get("markersize", 5),
            capsize=kwargs.get("capsize", 3),
            linewidth=kwargs.get("linewidth", 1.2),
            zorder=3,
        )
        if alternative == "greater":
            eb_kwargs["xuplims"] = True
        elif alternative == "less":
            eb_kwargs["xlolims"] = True
        ax.errorbar(value, y, xerr=err.reshape(2, 1), **eb_kwargs)

    margin = ci_results.get("margin", 0.0)
    if np.ndim(margin) == 0:
        ax.axvline(float(margin), color="#555555", linestyle="--", linewidth=0.9)
    ax.axvline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names, fontsize=kwargs.get("tick_fontsize", 9))
    ax.invert_yaxis()

    # Default labels reflect the alternative
    if alternative == "greater":
        default_xlabel = (
            "FDFI score with confidence interval  [H\u2081: \u03c6 > margin, one-sided]"
        )
        default_title = "FDFI Confidence Intervals  (one-sided)"
    elif alternative == "less":
        default_xlabel = (
            "FDFI score with confidence interval  [H\u2081: \u03c6 < margin, one-sided]"
        )
        default_title = "FDFI Confidence Intervals  (one-sided)"
    else:
        default_xlabel = "FDFI score with confidence interval"
        default_title = "FDFI Confidence Intervals"

    ax.set_xlabel(kwargs.get("xlabel", default_xlabel))
    ax.set_title(
        kwargs.get("title", default_title),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_xlim(clip_min, clip_max)

    # Corner annotation for one-sided plots
    if alternative != "two-sided" and kwargs.get("show_alternative_note", True):
        note = (
            "\u25ba upper bound is +\u221e"
            if alternative == "greater"
            else "\u25c4 lower bound is \u2212\u221e"
        )
        ax.text(
            0.98,
            0.02,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=kwargs.get("note_fontsize", 8),
            color="#555555",
            style="italic",
        )

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, ax


def diagnostics_plot(
    diagnostics: Mapping[str, Any],
    feature_names: Optional[Sequence[Any]] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    savepath: Optional[str] = None,
    **kwargs: Any,
):
    """
    Plot FDFI disentanglement diagnostics.

    Parameters
    ----------
    diagnostics : mapping
        Diagnostics dictionary from an explainer. Existing keys include
        ``latent_independence_median``, ``distribution_fidelity_mmd``,
        ``latent_independence_label``, ``distribution_fidelity_label``, and
        optionally ``latent_independence_dcor``.
    feature_names : sequence of str, optional
        Feature names for the latent dCor matrix when displayed.
    ax : matplotlib.axes.Axes, optional
        Existing axes for the scalar diagnostics bar plot. When omitted and a
        dCor matrix is available, a two-panel figure is created.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.
    savepath : str, optional
        Path where the figure should be saved.
    **kwargs
        Styling options.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax_or_axes : matplotlib.axes.Axes or numpy.ndarray
        The axes object(s).

    Examples
    --------
    >>> diagnostics_plot(explainer.diagnostics, show=False)
    """
    metric_keys = [
        ("latent_independence_median", "Median dCor", "latent_independence_label"),
        ("distribution_fidelity_mmd", "MMD", "distribution_fidelity_label"),
    ]
    available = [
        (key, label, label_key)
        for key, label, label_key in metric_keys
        if key in diagnostics
    ]
    if not available:
        raise ValueError(
            "diagnostics must contain at least one of "
            "'latent_independence_median' or 'distribution_fidelity_mmd'"
        )

    dcor = diagnostics.get("latent_independence_dcor")
    include_matrix = kwargs.get("include_matrix", ax is None and dcor is not None)
    if ax is None and include_matrix:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=kwargs.get("figsize", (10.0, 4.0)),
            gridspec_kw={"width_ratios": [1.0, 1.25]},
        )
        metric_ax = axes[0]
        matrix_ax = axes[1]
        returned_axes: Union[Axes, np.ndarray] = axes
    else:
        fig, metric_ax = _fig_ax(ax, kwargs.get("figsize", (5.8, 3.8)))
        matrix_ax = None
        returned_axes = metric_ax

    label_colors = {
        "GOOD": kwargs.get("good_color", "#2ca02c"),
        "MODERATE": kwargs.get("moderate_color", "#ffbf00"),
        "POOR": kwargs.get("poor_color", "#d62728"),
    }
    labels = [label for _, label, _ in available]
    values = np.array([float(diagnostics[key]) for key, _, _ in available])
    quality = [str(diagnostics.get(label_key, "")).upper() for _, _, label_key in available]
    colors = [label_colors.get(label, "#777777") for label in quality]

    x_pos = np.arange(len(values))
    metric_ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.7)
    metric_ax.set_xticks(x_pos)
    metric_ax.set_xticklabels(labels, rotation=20, ha="right")
    metric_ax.set_ylabel(kwargs.get("ylabel", "Diagnostic value"))
    metric_ax.set_title(
        kwargs.get("title", "FDFI Diagnostics"),
        fontsize=kwargs.get("title_fontsize", 11),
    )
    metric_ax.grid(axis="y", linestyle="--", alpha=0.25)
    for x, value, label in zip(x_pos, values, quality):
        text = f"{value:.3g}"
        if label:
            text = f"{text}\n{label.title()}"
        metric_ax.text(x, value, text, ha="center", va="bottom", fontsize=8)

    if matrix_ax is not None:
        dcor_arr = _as_2d(dcor, "diagnostics['latent_independence_dcor']")
        if dcor_arr.shape[0] != dcor_arr.shape[1]:
            raise ValueError("latent_independence_dcor must be a square matrix")
        dcor_names = _feature_names(feature_names, dcor_arr.shape[0])
        im = matrix_ax.imshow(dcor_arr, cmap=kwargs.get("matrix_cmap", "magma_r"))
        matrix_ax.set_title("Latent dCor Matrix", fontsize=kwargs.get("title_fontsize", 11))
        matrix_ax.set_xticks(np.arange(dcor_arr.shape[1]))
        matrix_ax.set_yticks(np.arange(dcor_arr.shape[0]))
        matrix_ax.set_xticklabels(
            dcor_names, rotation=45, ha="right", fontsize=kwargs.get("matrix_fontsize", 8)
        )
        matrix_ax.set_yticklabels(dcor_names, fontsize=kwargs.get("matrix_fontsize", 8))
        fig.colorbar(im, ax=matrix_ax, fraction=0.046, pad=0.04)

    _finish_figure(
        fig,
        savepath,
        show,
        dpi=kwargs.get("dpi", 150),
        bbox_inches=kwargs.get("bbox_inches", "tight"),
    )
    return fig, returned_axes
