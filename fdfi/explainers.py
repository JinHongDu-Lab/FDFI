"""
Explainer classes for DFI.

This module contains the main explainer classes for computing
disentangled feature importance.
"""

import numpy as np
from typing import Optional, Union, Callable, Any, Tuple
from scipy.spatial.distance import cdist, pdist
from scipy import stats
from .utils import (
    TwoComponentMixture,
    detect_feature_types,
    gower_cost_matrix,
    compute_latent_independence,
    compute_mmd,
)
from .losses import resolve_loss



class Explainer:
    """
    Base class for DFI explainers.
    
    This class provides the interface for computing feature importance
    using disentangled methods, similar to SHAP explainers.
    It also provides post-hoc confidence intervals via `conf_int()` and
    formatted summaries via `summary()`.
    
    Parameters
    ----------
    model : callable
        The model to explain. Should be a function that takes a numpy array
        and returns predictions.
    data : numpy.ndarray, optional
        Background data to use for explanations.
    **kwargs : dict
        Additional parameters for the explainer.  Notable keys:

        - ``loss`` (str or callable, default ``None`` → squared error): the loss
          ``L(y_true, y_pred)`` used to define importance.  String keys such as
          ``'l1'``, ``'huber'``, ``'pinball'``, ``'log_loss'``, ``'brier'`` are
          accepted (see :func:`fdfi.losses.available_losses`), or pass any
          callable returning the per-sample loss.  Passing true labels ``y`` at
          call time uses the loss-difference (DFI) form; otherwise a label-free
          divergence from the model's own prediction is used.
    
    Attributes
    ----------
    model : callable
        The model being explained.
    data : numpy.ndarray or None
        Background data for explanations.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fdfi import Explainer
    >>> 
    >>> # Define a simple model
    >>> def model(x):
    ...     return x.sum(axis=1)
    >>> 
    >>> # Create an explainer
    >>> explainer = Explainer(model)
    >>> 
    >>> # Compute explanations (when implemented)
    >>> # explanations = explainer(X_test)
    """
    
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: Optional[np.ndarray] = None,
        **kwargs: Any
    ):
        """Initialize the Explainer."""
        self.model = model
        self.data = data
        self.kwargs = kwargs
        self._last_results = None
        self._last_n = None
        self._var_floor_mixture = None
        self._margin_mixture = None
        self._var_floor_value = None
        self.ueifs_X = None
        self.ueifs_Z = None
        self.verbose = kwargs.get("verbose", False)
        # Loss used to define feature importance. ``None`` → squared error,
        # which reduces the DFI score to the classic difference of L2 residuals.
        self.loss = kwargs.get("loss", None)
        self._loss = resolve_loss(self.loss)
        self.diagnostics = None
        self.compute_diagnostics = kwargs.get("compute_diagnostics", True)
        self.diagnostics_subset_max_samples = kwargs.get(
            "diagnostics_subset_max_samples", 1000
        )
        self.latent_independence_thresholds = kwargs.get(
            "latent_independence_thresholds", (0.1, 0.25)
        )
        self.distribution_fidelity_thresholds = kwargs.get(
            "distribution_fidelity_thresholds", (0.05, 0.15)
        )

        fit_flow = kwargs.get("fit_flow", True)
        self.flow_engine = None
        if data is not None and fit_flow:
            try:
                from .models import FlowMatchingModel
            except ImportError as exc:
                raise ImportError(
                    "Flow matching requires torch; install it or pass fit_flow=False."
                ) from exc
            self.flow_engine = FlowMatchingModel(
                X=data,
                dim=data.shape[1],
                device=kwargs.get("device")
            )
            print("Training flow model for disentanglement...")
            self.flow_engine.fit(num_steps=kwargs.get("num_steps", 5000))

    def _cache_results(self, results: dict, n: int) -> None:
        self._last_results = results
        self._last_n = n

    def _ueif_from_counterfactuals(
        self,
        y_pred: np.ndarray,
        y_tilde_all: np.ndarray,
        method: str = "cpi",
        y_true: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Per-feature uncentered EIF from counterfactual predictions.

        Generalises the L2 residual-difference importance to an arbitrary loss
        ``L = self._loss``.  Two averaging orders (formalised in the FDFI docs)
        are selected via ``method``:

        - ``'cpi'`` (Conditional Permutation Importance): average the
          counterfactual *prediction* over the Monte-Carlo replicates first,
          then apply the loss — ``L(r, E_b[ŷ_b])``.
        - ``'scpi'`` (Sobol-CPI): apply the loss to *each* replicate first, then
          average — ``E_b[L(r, ŷ_b)]``.

        When ``y_true`` is provided the score is a difference of losses (the DFI
        / LOCO form, centred near zero for null features)::

            UEIF = agg_b L(y_true, ŷ_b) - L(y_true, ŷ)

        Otherwise it is the label-free form that uses the model's own prediction
        ``ŷ`` as the reference and subtracts the self-loss floor::

            UEIF = agg_b L(ŷ, ŷ_b) - L(ŷ, ŷ)

        For the squared error (and other losses with ``L(a, a) = 0``) this is the
        prediction shift ``(ŷ - ŷ_b)²``.  For a proper scoring rule such as
        log-loss or Brier it is the associated Bregman divergence between the
        baseline and counterfactual predictions (e.g. ``KL(ŷ ‖ ŷ_b)`` for
        log-loss), which is non-negative and ~0 for null features.  Non-proper /
        discontinuous losses (e.g. ``zero_one``) are not meaningful label-free
        and should be used with ``y_true``.

        Parameters
        ----------
        y_pred : ndarray of shape (n,)
            Model predictions on the observed inputs (the baseline ``ŷ``).
        y_tilde_all : ndarray of shape (B, n)
            Counterfactual predictions for the ``B`` Monte-Carlo replicates of
            the perturbed feature.
        method : {'cpi', 'scpi'}, default='cpi'
            Averaging order (see above).
        y_true : ndarray of shape (n,), optional
            True outcomes.  When omitted, the label-free divergence form above is
            used.

        Returns
        -------
        ndarray of shape (n,)
            Per-sample uncentered EIF for the feature.
        """
        loss = self._loss
        method = (method or "cpi").lower()
        if method not in ("cpi", "scpi"):
            raise ValueError(f"method must be 'cpi' or 'scpi', got {method!r}")

        if y_true is None:
            # Label-free: reference is the model's own prediction; subtract the
            # self-loss floor L(ŷ, ŷ) so null features stay near zero. This is 0
            # for regression losses and the predictive entropy for proper
            # scoring rules (giving a Bregman divergence).
            ref = np.asarray(y_pred)
            base = loss(ref, ref)
        else:
            ref = np.asarray(y_true)
            base = loss(ref, y_pred)

        if method == "cpi":
            agg = loss(ref, y_tilde_all.mean(axis=0))
        else:  # scpi
            agg = loss(ref[None, :], y_tilde_all).mean(axis=0)

        return agg - base


    def _adjust_se(
        self,
        se_raw: np.ndarray,
        var_floor_c: float = 0.1,
        var_floor_method: str = "mixture",
        var_floor_quantile: float = 0.95,
    ) -> np.ndarray:
        if self._last_n is None:
            return se_raw

        if var_floor_method == "mixture":
            self._var_floor_mixture = TwoComponentMixture().fit(se_raw)
            floor = self._var_floor_mixture.quantile(var_floor_quantile, "smaller")
            self._var_floor_value = floor
        else:
            if var_floor_c <= 0:
                return se_raw
            floor = var_floor_c / np.sqrt(self._last_n)
            self._var_floor_value = floor

        return np.sqrt(se_raw**2 + floor**2)

    # Minimum number of features for which GMM-based margin is reliable.
    _MARGIN_GMM_MIN_D = 30

    def conf_int(
        self,
        alpha: float = 0.05,
        target: str = "X",
        groups: Optional[Union[dict, np.ndarray, Any]] = None,
        threshold_null: bool = True,
        multitest_method: Optional[str] = None,
        var_floor_c: float = 0.1,
        var_floor_method: str = "mixture",
        var_floor_quantile: float = 0.95,
        margin: float = 0.0,
        margin_method: str = "auto",
        margin_quantile: float = 0.95,
        alternative: str = "two-sided",
        verbose: bool = False,
    ) -> dict:
        """
        Compute confidence intervals and significance statistics for feature importance.

        If `groups` is provided, computes importance and uncertainty at the group level.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.
        target : str, default='X'
            Which space to use: 'X' (original) or 'Z' (latent).
        groups : dict, numpy.ndarray, or pandas.DataFrame, optional
            Group assignment for features. Accepts:
            - ``dict``: ``{group_name: [feature_indices]}``
            - ``numpy.ndarray``: 1-D array of length *d* with group labels.
            - ``pandas.DataFrame``: binary indicator matrix (features x groups).
        threshold_null : bool, default=True
            Zero out per-feature uncentered UEIFs with negative mean before summing.
        multitest_method : str, optional
            Multiple testing correction method. Supports methods from 
            ``statsmodels.stats.multitest.multipletests``, e.g., 'bonferroni', 
            'holm', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by'.
        var_floor_c : float, default=0.1
            Constant for the variance floor.
        var_floor_method : str, default='mixture'
            Method for variance floor calculation ('mixture' or 'fixed').
        var_floor_quantile : float, default=0.95
            Quantile for the 'mixture' variance floor method.
        margin : float, default=0.0
            Hypothesized margin for null hypothesis.
        margin_method : str, default='auto'
            Method to estimate the margin ('auto', 'mixture', 'gap', or 'fixed').
        margin_quantile : float, default=0.95
            Quantile for the 'mixture' margin method.
        alternative : str, default='two-sided'
            Alternative hypothesis ('two-sided', 'greater', or 'less').
        verbose : bool, default=False
            Whether to print debug information.

        Returns
        -------
        dict
            Dictionary with the following keys (each an array of length *d* or *G*):

            - ``'score'``: estimated feature importance (mean UEIF).
            - ``'se'``: standard error of the mean UEIF (after variance floor).
            - ``'zscore'``: signed z-statistic ``(score - margin) / se``.
            - ``'ranking'``: integer rank by descending z-score (1 = most important).
            - ``'ci_lower'``: lower confidence interval bound.
            - ``'ci_upper'``: upper confidence interval bound.
            - ``'reject_null'``: boolean array, True where null is rejected.
            - ``'pvalue'``: two-sided or one-sided p-value.
            - ``'margin'``: null hypothesis margin used.
            - ``'margin_method'``: method used to select the margin.
            - ``'alternative'``: alternative hypothesis string.

            Additional keys added when applicable:

            - ``'groups'``: list of group names (when ``groups`` is provided).
            - ``'pvalue_adj'``: multiple-testing-adjusted p-values (when
              ``multitest_method`` is provided).
        """
        if groups is not None:
            if target == "Z":
                ueifs = self.ueifs_Z
            else:
                ueifs = self.ueifs_X

            if ueifs is None:
                raise ValueError(
                    "Per-sample UEIFs not available. Run the explainer first."
                )

            n = ueifs.shape[0]
            d = ueifs.shape[1]
            group_dict = self._normalize_groups(groups, d)

            group_names = []
            phi_hat_list = []
            se_raw_list = []

            for name, indices in group_dict.items():
                ueifs_g = ueifs[:, indices].copy()
                if threshold_null:
                    feature_means = ueifs_g.mean(axis=0)
                    ueifs_g[:, feature_means < 0] = 0
                grouped_ueifs = ueifs_g.sum(axis=1)

                phi_hat_list.append(grouped_ueifs.mean())
                # Standard error of the mean
                se_raw_list.append(grouped_ueifs.std(ddof=1) / np.sqrt(n))
                group_names.append(name)

            phi_hat = np.array(phi_hat_list)
            se_raw = np.array(se_raw_list)
            self._last_n = n
        else:
            if self._last_results is None:
                raise ValueError("Run the explainer first to compute scores.")

            if target == "Z":
                phi_hat = self._last_results["phi_Z"]
                se_raw = self._last_results["se_Z"]
            else:
                phi_hat = self._last_results["phi_X"]
                se_raw = self._last_results["se_X"]

        se_adj = self._adjust_se(
            se_raw,
            var_floor_c=var_floor_c,
            var_floor_method=var_floor_method,
            var_floor_quantile=var_floor_quantile,
        )

        d = len(phi_hat)
        margin, margin_method_used = self._compute_margin(
            phi_hat,
            margin,
            margin_method,
            margin_quantile,
            d,
            verbose,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            if alternative == "greater":
                z = stats.norm.ppf(1 - alpha)
                ci_lower = phi_hat - z * se_adj
                ci_upper = np.full_like(phi_hat, np.inf)
                reject_null = ci_lower > margin
                z_scores = (phi_hat - margin) / se_adj
                pvalues = 1 - stats.norm.cdf(z_scores)
            elif alternative == "less":
                z = stats.norm.ppf(1 - alpha)
                ci_lower = np.full_like(phi_hat, -np.inf)
                ci_upper = phi_hat + z * se_adj
                reject_null = ci_upper < margin
                z_scores = (phi_hat - margin) / se_adj
                pvalues = stats.norm.cdf(z_scores)
            elif alternative == "two-sided":
                z = stats.norm.ppf(1 - alpha / 2)
                ci_lower = phi_hat - z * se_adj
                ci_upper = phi_hat + z * se_adj
                reject_null = (ci_lower > margin) | (ci_upper < margin)
                z_scores = np.abs(phi_hat - margin) / se_adj
                pvalues = 2 * (1 - stats.norm.cdf(z_scores))
            else:
                raise ValueError(
                    "alternative must be 'greater', 'less', or 'two-sided'"
                )

            pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)

        # Signed z-score: (score - margin) / se  (always signed, regardless of alternative)
        signed_z = (phi_hat - margin) / se_adj
        signed_z = np.where(np.isfinite(signed_z), signed_z, 0.0)

        # Ranking: rank 1 = highest z-score (most important).
        # Use a stable descending sort so tied z-scores receive a deterministic,
        # reproducible order based on their original position.
        _order = np.argsort(-signed_z, kind="stable")
        ranking = np.empty(len(signed_z), dtype=int)
        ranking[_order] = np.arange(1, len(signed_z) + 1)

        out = {
            "score": phi_hat,
            "se": se_adj,
            "zscore": signed_z,
            "ranking": ranking,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "reject_null": reject_null,
            "pvalue": pvalues,
            "margin": margin,
            "margin_method": margin_method_used,
            "alternative": alternative,
        }

        if multitest_method is not None:
            try:
                from statsmodels.stats.multitest import multipletests
            except ImportError as exc:
                raise ImportError(
                    "Multiple testing correction requires statsmodels. "
                    "Install it with `pip install statsmodels`."
                ) from exc

            reject, pvals_corrected, _, _ = multipletests(
                pvalues, alpha=alpha, method=multitest_method
            )
            out["reject_null"] = reject
            out["pvalue_adj"] = pvals_corrected
            out["multitest_method"] = multitest_method

        if groups is not None:
            out["groups"] = group_names
        return out

    # ------------------------------------------------------------------
    # Margin estimation helpers
    # ------------------------------------------------------------------

    def _compute_margin(
        self,
        phi_hat: np.ndarray,
        margin: float,
        margin_method: str,
        margin_quantile: float,
        d: int,
        verbose: bool,
    ) -> tuple:
        """Return (margin_value, method_used_string)."""

        if margin_method == "fixed":
            self._margin_mixture = None
            if verbose:
                print(f"[margin] method=fixed, margin={margin:.4f}")
            return margin, "fixed"

        if margin_method == "auto":
            if d < self._MARGIN_GMM_MIN_D:
                method = "gap"
                reason = f"d={d} < {self._MARGIN_GMM_MIN_D}"
            else:
                method = "mixture"
                reason = f"d={d} >= {self._MARGIN_GMM_MIN_D}"
            if verbose:
                print(f"[margin] method=auto → {method} ({reason})")
        else:
            method = margin_method

        if method == "gap":
            margin = self._gap_margin(phi_hat, verbose)
            self._margin_mixture = None
            return margin, "gap"
        elif method == "mixture":
            self._margin_mixture = TwoComponentMixture().fit(phi_hat)
            margin = max(
                self._margin_mixture.quantile(margin_quantile, "smaller"), 0
            )
            if verbose:
                print(
                    f"[margin] mixture: means={self._margin_mixture.means_.round(4)}, "
                    f"weights={self._margin_mixture.weights_.round(3)}, "
                    f"margin={margin:.4f}"
                )
            return margin, "mixture"
        else:
            raise ValueError(
                f"margin_method must be 'auto', 'mixture', 'gap', or 'fixed', "
                f"got '{margin_method}'"
            )

    @staticmethod
    def _gap_margin(phi_hat: np.ndarray, verbose: bool = False) -> float:
        """Largest-gap margin: cluster null vs signal by the biggest
        multiplicative jump (log-scale gap).

        Uses log-transformed phi values so that a jump from 0.03 → 0.57
        (~19×) dominates over a jump from 3.6 → 6.2 (~1.7×).
        The margin is set to the value at the top of the lower cluster.
        """
        vals = np.sort(phi_hat)
        if len(vals) < 2:
            return 0.0
        # Work in log-space; shift by a small fraction to handle zeros
        floor = max(vals[vals > 0].min() * 1e-2, 1e-12) if np.any(vals > 0) else 1e-12
        log_vals = np.log(np.maximum(vals, floor))
        log_gaps = np.diff(log_vals)
        k = int(np.argmax(log_gaps))      # index of the lower-side element
        margin = float(vals[k])           # top of the null cluster
        if verbose:
            print(
                f"[margin] gap: sorted phi range [{vals[0]:.4f}, {vals[-1]:.4f}], "
                f"largest log-gap between rank {k} ({vals[k]:.4f}) and {k+1} "
                f"({vals[k+1]:.4f}), ratio={vals[k+1]/(vals[k]+1e-15):.1f}x, "
                f"margin={margin:.4f}"
            )
        return margin

    def summary(self, alpha: float = 0.05, print_output: bool = True, **kwargs) -> str:
        """
        Print and return a formatted feature importance summary table.

        Computes confidence intervals via :meth:`conf_int` and formats the
        results as a human-readable table.  Supports both individual-feature
        and group-level summaries, as well as multiple-testing correction.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level passed to :meth:`conf_int`.
        print_output : bool, default=True
            If ``True``, print the table to stdout.
        **kwargs
            All keyword arguments are forwarded to :meth:`conf_int`.  Common
            options include:

            - ``target`` (``'X'`` or ``'Z'``) — which feature space to report.
            - ``groups`` — dict, 1-D array, or binary DataFrame for group-level
              summaries (new in 0.0.5).
            - ``multitest_method`` — e.g. ``'bonferroni'``, ``'fdr_bh'`` for
              multiple-testing correction (new in 0.0.5).
            - ``threshold_null`` — zero out negative-mean UEIFs before group
              aggregation (new in 0.0.5).
            - ``var_floor_method``, ``var_floor_c``, ``var_floor_quantile``
            - ``margin``, ``margin_method``, ``margin_quantile``
            - ``alternative`` (``'two-sided'``, ``'greater'``, ``'less'``)
            - ``verbose``

        Returns
        -------
        str
            The formatted summary string (same text that is printed when
            ``print_output=True``).

        Examples
        --------
        Individual-feature summary::

            explainer(X_test, y=y_test)
            explainer.summary(alpha=0.05, target="X")

        Group-level summary with Bonferroni correction::

            explainer.summary(
                alpha=0.05,
                target="X",
                groups=df_groups,
                threshold_null=True,
                multitest_method="bonferroni",
            )
        """
        results = self.conf_int(alpha=alpha, **kwargs)
        return self._format_summary(results, alpha, print_output)

    # ------------------------------------------------------------------
    # Group importance
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_groups(groups, d: int) -> dict:
        """Normalize group input to ``{group_name: ndarray of feature indices}``."""
        if isinstance(groups, dict):
            return {k: np.asarray(v, dtype=int) for k, v in groups.items()}
        if isinstance(groups, np.ndarray) and groups.ndim == 1:
            unique_labels = np.unique(groups)
            return {label: np.where(groups == label)[0] for label in unique_labels}
        # pandas DataFrame (binary indicator matrix, features × groups)
        if hasattr(groups, "iloc"):
            return {
                col: np.where(groups[col].values > 0)[0]
                for col in groups.columns
            }
        raise TypeError(
            f"groups must be dict, 1D array, or DataFrame, got {type(groups)}"
        )

    def group_importance(
        self,
        groups: Union[dict, np.ndarray, Any],
        target: str = "X",
        threshold_null: bool = True,
        se_adjustment: float = 0.1,
        alpha: float = 0.05,
    ) -> dict:
        """Compute group-level feature importance with uncertainty.

        .. deprecated:: 0.0.5
            Use :meth:`conf_int` with the ``groups`` argument instead.

        Parameters
        ----------
        groups : dict, numpy.ndarray, or pandas.DataFrame
            Group assignment for features. Accepts:

            - ``dict``: ``{group_name: [feature_indices]}``
            - ``numpy.ndarray``: 1-D array of length *d* with group labels.
            - ``pandas.DataFrame``: binary indicator matrix (features × groups).
        target : str, default='X'
            Which space to aggregate: ``'X'`` or ``'Z'``.
        threshold_null : bool, default=True
            Zero out per-feature UEIFs with negative mean before summing.
        se_adjustment : float, default=0.1
            Finite-sample SE correction constant. Set to 0.0 to disable.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        dict
            ``'groups'``, ``'importance'``, ``'se'``, ``'zscore'``, ``'pvalue'``
            — each an array of length *G* (number of groups).
        """
        import warnings

        warnings.warn(
            "group_importance() is deprecated and will be removed in a future version. "
            "Use conf_int(groups=...) instead.",
            FutureWarning,
            stacklevel=2,
        )

        # Map group_importance parameters to conf_int parameters
        results = self.conf_int(
            alpha=alpha,
            target=target,
            groups=groups,
            threshold_null=threshold_null,
            var_floor_c=se_adjustment,
            var_floor_method="fixed",
        )

        return {
            "groups": np.array(results["groups"]),
            "importance": results["score"],
            "se": results["se"],
            "zscore": results["zscore"],
            "pvalue": results["pvalue"],
        }


    def _format_summary(self, results: dict, alpha: float, print_output: bool = True) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("Feature Importance Results")
        lines.append("=" * 78)
        lines.append(f"Method: {self.__class__.__name__}")
        lines.append(f"Number of units: {len(results['score'])}")
        lines.append(f"Significance level: {alpha}")
        lines.append(f"Alternative: {results['alternative']}")
        
        multitest_method = results.get("multitest_method")
        if multitest_method:
            lines.append(f"Multiple testing: {multitest_method}")
            
        margin_method_str = results.get("margin_method", "")
        if margin_method_str:
            lines.append(f"Margin method: {margin_method_str}")
        if results["margin"] > 0:
            lines.append(f"Practical margin: {results['margin']:.4f}")
        lines.append("-" * 78)

        has_groups = "groups" in results
        unit_label = "Group" if has_groups else "Feature"
        pval_label = "Adj P-val" if multitest_method else "P-value"
        header = (
            f"{unit_label:>15} {'Estimate':>10} {'Std Err':>10} "
            f"{'CI Lower':>10} {'CI Upper':>10} {pval_label:>10} {'Sig':>5}"
        )
        lines.append(header)
        lines.append("-" * 78)

        has_pvalue_adj = "pvalue_adj" in results
        for i in range(len(results["score"])):
            ci_lower = results["ci_lower"][i]
            ci_upper = results["ci_upper"][i]
            ci_upper_str = (
                f"{ci_upper:>10.4f}" if np.isfinite(ci_upper) else f"{'inf':>10}"
            )
            ci_lower_str = (
                f"{ci_lower:>10.4f}" if np.isfinite(ci_lower) else f"{'-inf':>10}"
            )
            
            pval = results["pvalue_adj"][i] if has_pvalue_adj else results["pvalue"][i]
            sig = (
                "***"
                if pval < 0.01
                else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            )
            name = str(results["groups"][i]) if has_groups else str(i)
            row = (
                f"{name:>15} {results['score'][i]:>10.4f} "
                f"{results['se'][i]:>10.4f} {ci_lower_str} {ci_upper_str} "
                f"{pval:>10.4f} {sig:>5}"
            )
            lines.append(row)

        lines.append("=" * 78)
        n_sig = np.sum(results["reject_null"])
        lines.append(f"Significant units: {n_sig} / {len(results['score'])}")
        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")
        lines.append("=" * 78)

        output = "\n".join(lines)
        if print_output:
            print(output)
        return output

    def _log(self, message: str, level: str = "INFO") -> None:
        """Consistent, verbosity-aware logging with unified style."""
        v = getattr(self, "verbose", False)
        if v is False:
            return
        if v in ("all", True, "final"):
            print(f"[FDFI][{level.upper()}] {message}")

    @staticmethod
    def _qualitative_score(
        value: float,
        thresholds: Tuple[float, float],
        lower_is_better: bool = True,
    ) -> str:
        """
        Return GOOD / MODERATE / POOR label based on thresholds.

        thresholds: (good_cutoff, moderate_cutoff); behavior flips if lower_is_better=False.
        """
        good, moderate = thresholds
        if lower_is_better:
            if value < good:
                return "GOOD"
            if value < moderate:
                return "MODERATE"
            return "POOR"
        if value > moderate:
            return "POOR"
        if value > good:
            return "MODERATE"
        return "GOOD"

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent-space samples to original feature space.

        Subclasses with disentanglement maps (OT/EOT/Flow) should implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement latent decoding."
        )

    def _get_diagnostics_sources(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return default (X_orig, Z_full) used for diagnostics."""
        return self.data, getattr(self, "Z_full", None)

    def _compute_diagnostics(
        self,
        X_orig: Optional[np.ndarray] = None,
        Z_full: Optional[np.ndarray] = None,
        report_title: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compute and store generic disentanglement diagnostics.

        Diagnostics:
        - Latent independence: median distance correlation across latent dims.
        - Distribution fidelity: Maximum Mean Discrepancy (MMD) between
          original data and reconstructions.
        """
        if not self.compute_diagnostics:
            return None

        if X_orig is None or Z_full is None:
            default_X, default_Z = self._get_diagnostics_sources()
            if X_orig is None:
                X_orig = default_X
            if Z_full is None:
                Z_full = default_Z

        if X_orig is None or Z_full is None:
            return None

        X_arr = np.asarray(X_orig)
        Z_arr = np.asarray(Z_full)
        if X_arr.ndim != 2 or Z_arr.ndim != 2:
            raise ValueError("X_orig and Z_full must both be 2D arrays.")

        n = min(X_arr.shape[0], Z_arr.shape[0])
        if n == 0:
            return None
        X_use = X_arr[:n]
        Z_use = Z_arr[:n]

        subset_size = None
        if (
            self.diagnostics_subset_max_samples is not None
            and n > self.diagnostics_subset_max_samples
        ):
            subset_size = int(self.diagnostics_subset_max_samples)

        dcor_matrix, median_dcor = compute_latent_independence(
            Z_use, subset_size=subset_size
        )
        dcor_label = self._qualitative_score(
            float(median_dcor),
            thresholds=self.latent_independence_thresholds,
            lower_is_better=True,
        )

        X_hat = self._decode_from_Z(Z_use)
        mmd_score = compute_mmd(X_use, X_hat, subset_size=subset_size)
        mmd_label = self._qualitative_score(
            float(mmd_score),
            thresholds=self.distribution_fidelity_thresholds,
            lower_is_better=True,
        )

        self.diagnostics = {
            "latent_independence_dcor": dcor_matrix,
            "latent_independence_median": float(median_dcor),
            "distribution_fidelity_mmd": float(mmd_score),
            "latent_independence_label": dcor_label,
            "distribution_fidelity_label": mmd_label,
        }

        title = report_title if report_title is not None else self.__class__.__name__
        self._log(f"{title} Diagnostics", level="diag")
        self._log(
            f"Latent independence (median dCor): {median_dcor:.6f} [{dcor_label}]  "
            "-> lower is better",
            level="diag",
        )
        self._log(
            f"Distribution fidelity (MMD):       {mmd_score:.6f} [{mmd_label}]  "
            "-> lower is better",
            level="diag",
        )
        return self.diagnostics

    def diagnose(
        self,
        X_orig: Optional[np.ndarray] = None,
        Z_full: Optional[np.ndarray] = None,
        report_title: Optional[str] = None,
    ) -> dict:
        """
        Compute (or recompute) disentanglement diagnostics.

        Evaluates latent independence via pairwise distance correlation (dCor)
        and distribution fidelity via Maximum Mean Discrepancy (MMD).  Called
        automatically during ``__init__`` when ``compute_diagnostics=True``.
        Use this method to recompute diagnostics on a custom subset or after
        calling :meth:`set_flow`.

        Parameters
        ----------
        X_orig : np.ndarray of shape (n_samples, n_features), optional
            Original-space data to use for MMD fidelity check.  When *None*
            the background data stored during ``__init__`` is used.
        Z_full : np.ndarray of shape (n_samples, n_features), optional
            Pre-encoded latent representations.  When *None* the background
            latent data stored during ``__init__`` is used.
        report_title : str, optional
            Label shown in verbose logging output.

        Returns
        -------
        diagnostics : dict
            Dictionary with keys:

            ``latent_independence_dcor`` : np.ndarray
                Pairwise dCor matrix of shape ``(d, d)``.
            ``latent_independence_median`` : float
                Median off-diagonal dCor (lower = more independent).
            ``latent_independence_label`` : str
                Qualitative label ``'GOOD'``, ``'MODERATE'``, or ``'POOR'``.
            ``distribution_fidelity_mmd`` : float
                MMD between original and reconstructed distributions.
            ``distribution_fidelity_label`` : str
                Qualitative label ``'GOOD'``, ``'MODERATE'``, or ``'POOR'``.

        Raises
        ------
        ValueError
            If diagnostics are unavailable (e.g. ``compute_diagnostics=False``
            was set and no latent data is accessible).

        Examples
        --------
        >>> diag = explainer.diagnose()
        >>> print(diag["latent_independence_label"])  # 'GOOD' / 'MODERATE' / 'POOR'
        >>> print(diag["distribution_fidelity_mmd"])
        """
        diagnostics = self._compute_diagnostics(
            X_orig=X_orig,
            Z_full=Z_full,
            report_title=report_title,
        )
        if diagnostics is None:
            raise ValueError(
                "Diagnostics unavailable. Ensure diagnostics are enabled and latent data exists."
            )
        return diagnostics
    
    def __call__(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute feature importance for the given input.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain. Shape (n_samples, n_features).
        **kwargs : dict
            Additional parameters for explanation.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values. Shape (n_samples, n_features).
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        n_samples, n_features = X.shape
        attributions = np.zeros((n_samples, n_features))
        
        raise NotImplementedError(
            "Explainer.__call__ must be implemented by subclasses"
        )
    
    def shap_values(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute SHAP-like values (alias for __call__).
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values.
        """
        return self(X, **kwargs)


class OTExplainer(Explainer):
    """
    Optimal-transport DFI explainer using Gaussian transport.

    Computes Disentangled Feature Importance (DFI) by mapping observed
    features to an uncorrelated (whitened) latent space via a Gaussian
    optimal-transport linear map, computing per-sample UEIFs in that space,
    and projecting back to the original feature space via the Jacobian.

    This is the recommended starting point for most use cases with
    continuous data. For non-Gaussian or mixed-type data, prefer
    :class:`EOTExplainer`. For rigorous inference with a small sample,
    consider wrapping this class with :class:`Crossfitting`.

    Parameters
    ----------
    model : callable
        Prediction function with signature ``f(X) -> np.ndarray`` where ``X``
        has shape ``(n_samples, n_features)``.
    data : np.ndarray of shape (n_background, n_features)
        Background data used to estimate the Gaussian transport map
        (mean and covariance). Larger backgrounds give more stable estimates;
        100–500 samples is typically sufficient.
    nsamples : int, default=50
        Number of Monte Carlo resamples per feature used to estimate the
        marginal-replacement expectation.
    sampling_method : {'resample', 'permutation', 'normal'}, default='resample'
        Strategy for drawing replacement values for each feature:

        * ``'resample'``  – draw with replacement from the background latent
          distribution (recommended).
        * ``'permutation'`` – permute the test-set latent values.
        * ``'normal'`` – draw i.i.d. standard normal samples.
    random_state : int, default=0
        Seed for the random number generator used in resampling.
    method : {'cpi', 'scpi'}, default='cpi'
        Averaging order for the counterfactual predictions:

        * ``'cpi'`` – average the prediction over resamples first, then apply
          the loss (``L(r, E_b[ŷ_b])``).
        * ``'scpi'`` – apply the loss per resample first, then average
          (``E_b[L(r, ŷ_b)]``; for squared error this equals CPI plus the
          prediction variance).
    verbose : bool, default=False
        Print progress messages during setup and inference.
    compute_diagnostics : bool, default=True
        Compute latent-independence (dCor) and distribution-fidelity (MMD)
        diagnostics during initialisation.
    diagnostics_subset_max_samples : int, default=1000
        Maximum number of background samples used for the dCor computation.
    latent_independence_thresholds : tuple of float, default=(0.1, 0.25)
        ``(good, poor)`` thresholds for the median off-diagonal dCor.  Values
        below the first threshold receive label ``'GOOD'``.
    distribution_fidelity_thresholds : tuple of float, default=(0.05, 0.15)
        ``(good, poor)`` thresholds for the MMD.  Values below the first
        threshold receive label ``'GOOD'``.
    **kwargs
        Additional keyword arguments forwarded to :class:`Explainer`.
        Useful keys include ``regularize`` (float, default ``1e-6``) which
        clips small eigenvalues of the covariance before computing the
        Cholesky factor.

    Attributes
    ----------
    mean : np.ndarray of shape (1, n_features)
        Background mean used for centring.
    L : np.ndarray of shape (n_features, n_features)
        Square-root of the background covariance (Cholesky-like factor);
        used as the decoder ``Z → X``.
    L_inv : np.ndarray of shape (n_features, n_features)
        Inverse of ``L``; used as the encoder ``X → Z``.
    Z_full : np.ndarray of shape (n_background, n_features)
        Background data projected into the latent space.
    ueifs_X : np.ndarray of shape (n_test, n_features)
        Per-sample UEIFs in the original X-space after calling the explainer.
    ueifs_Z : np.ndarray of shape (n_test, n_features)
        Per-sample UEIFs in the latent Z-space after calling the explainer.
    diagnostics : dict
        Disentanglement quality metrics; see :meth:`diagnose`.

    Examples
    --------
    >>> import numpy as np
    >>> from fdfi.explainers import OTExplainer
    >>> from fdfi.plots import summary_bar
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X_bg  = rng.standard_normal((200, 6))
    >>> X_test = rng.standard_normal((50, 6))
    >>> def model(X): return X[:, 0] + 2 * X[:, 1]
    >>>
    >>> explainer = OTExplainer(model, data=X_bg, nsamples=50)
    >>> results = explainer(X_test)
    >>> print(results["phi_X"])        # global importance, X-space
    >>> print(results["phi_Z"])        # global importance, Z-space
    >>>
    >>> ci = explainer.conf_int(alpha=0.05, alternative="greater")
    >>> summary_bar(results["phi_X"], results["se_X"], show=False)
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        sampling_method: str = "resample",
        random_state: int = 0,
        method: str = "cpi",
        **kwargs: Any
    ):
        """Initialize the OTExplainer."""
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.nsamples = nsamples
        self.regularize = kwargs.get("regularize", 1e-6)
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.method = method

       
        self.mean = np.mean(data, axis=0, keepdims=True)
        
        self.cov = np.cov(data, rowvar=False, ddof=0)
        self.cov = (self.cov + self.cov.T) / 2  

      
        eigenvals, eigenvecs = np.linalg.eigh(self.cov)
        eigenvals = np.maximum(eigenvals, self.regularize) 

        
        self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
       
        self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T

        self.Z_full = (data - self.mean) @ self.L_inv
        self._compute_diagnostics(report_title="OTExplainer")

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
       
        n, d = X.shape
        Z = (X - self.mean) @ self.L_inv
        
    
        y_pred = self.model(X)
        
       
        # get per-sample UEIFs in latent space (n_samples, n_features)
        ueifs_Z = self._phi_Z(Z, y_pred, y_true=y)

        # Jacobian sensitivity matrix (constant for linear mapping)
        H = self.L ** 2
        # Map latent-space UEIFs back to original X-space per-sample
        ueifs_X = ueifs_Z @ H.T

        # Store per-sample UEIFs for group_importance()
        self.ueifs_X = ueifs_X
        self.ueifs_Z = ueifs_Z

        # Aggregate to get mean importance and uncertainty (std) across samples
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)

        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return results

    def _phi_Z(
        self,
        Z: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sample uncentered EIF in Z-space via counterfactual resampling.

        For each feature ``j`` the j-th latent coordinate is replaced by
        independent draws and importance is scored through ``self._loss`` using
        the CPI/SCPI averaging order given by ``self.method``.  With the default
        squared-error loss and no ``y_true`` this reduces to the L2 prediction
        shift ``(ŷ - E_b[ŷ_b])²``; with ``y_true`` it becomes the DFI residual
        difference ``L(y, ŷ_b) - L(y, ŷ)``.
        """
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        
        for j in range(d):

            rng = np.random.default_rng(self.random_state + j)
            
            # Z_tilde: (nsamples, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            if self.sampling_method == "resample":
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = Z_tilde_flat @ self.L + self.mean
            
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde_all = y_tilde_flat.reshape(self.nsamples, n)
            
            ueifs_Z[:, j] = self._ueif_from_counterfactuals(
                y_pred, y_tilde_all, method=self.method, y_true=y_true
            )

        # Return per-sample UEIFs in latent space (no aggregation here)
        return ueifs_Z

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Decode Z to X using the Gaussian OT linear map."""
        return Z @ self.L + self.mean


class EOTExplainer(Explainer):
    """
    Entropic optimal-transport DFI explainer using semicontinuous transport
    and population backward attribution.

    Uses the population EOT coupling between the empirical source and
    continuous N(0, I) target.  The forward map is analytical:

        Z = c_ε · X_whitened,   c_ε = √(1 + ε) / (1 + ε/2)

    Backward attribution uses the best linear projection:

        E[X_whitened | Z] = M_w · Z

    where M_w = E_π[ZZ^T]^{-1} E_π[ZX_w^T] is computed analytically
    from the semicontinuous coupling moments.  This gives the weight
    matrix  W = L @ M_w  used for the decomposition:

        φ_X_j = Σ_k W[j,k]² · φ_Z_k

    Feature importance is measured via the uncentered efficient influence
    function (UEIF):

        UEIF_{i,j} = (Y_i - ŷ_{-j,i})²

    where ŷ_{-j} averages predictions over counterfactual resamples of
    feature j.

    Parameters
    ----------
    model : callable
        The model to explain.  Takes (n, d) array, returns (n,) predictions.
    data : numpy.ndarray
        Background data for whitening and resampling.  Shape (n, d).
    nsamples : int, default=50
        Number of Monte Carlo samples per feature for counterfactual
        resampling.
    epsilon : float, default=0.1
        EOT regularization parameter.  Smaller ε → closer to exact OT;
        larger ε → more Gaussian shrinkage.
    auto_epsilon : bool, default=False
        If True, set ε from a median-distance heuristic in whitened space.
    sampling_method : str, default='resample'
        How to draw counterfactual Z_j values:
        - 'resample': sample from the background Z pool
        - 'permutation': permute within the test set
        - 'normal': sample from N(0, 1)
    random_state : int, default=0
        Random seed for reproducibility.
    method : {'cpi', 'scpi'}, default='cpi'
        Averaging order for counterfactual predictions (CPI averages the
        prediction before the loss; SCPI averages the per-resample loss).
    **kwargs : dict
        Extra arguments forwarded to the base Explainer.
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        epsilon: float = 0.1,
        auto_epsilon: bool = False,
        sampling_method: str = "resample",
        random_state: int = 0,
        method: str = "cpi",
        **kwargs: Any
    ):
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.auto_epsilon = auto_epsilon
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.method = method
        self.regularize = kwargs.get("regularize", 1e-6)

        # ── Gaussian whitening ───────────────────────────────────────────
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.cov(data - self.mean, rowvar=False, ddof=0)
        self.cov = (self.cov + self.cov.T) / 2

        eigenvals, eigenvecs = np.linalg.eigh(self.cov)
        eigenvals = np.maximum(eigenvals, self.regularize)

        self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
        self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T

        X_centered = data - self.mean
        X_whitened = X_centered @ self.L_inv

        if self.auto_epsilon:
            self.epsilon = self._auto_epsilon(X_centered)

        # ── Semicontinuous population forward map ────────────────────────
        # Analytical scaling for Gaussian source → N(0,I) target:
        #   c_ε = √(1 + ε) / (1 + ε/2)
        # At ε=0 this gives c=1 (identity); as ε→∞ it approaches 0.
        eps = self.epsilon
        if eps == 0:
            self.s_fwd = 1.0
        else:
            self.s_fwd = np.sqrt(1.0 + eps) / (1.0 + eps / 2.0)
        self.L_z = self.s_fwd * self.L

        # ── Population backward attribution weights ──────────────────────
        # Under the semicontinuous coupling Z|X ~ N(c_ε·X_w, σ²·I)
        # where σ² = 1 - c_ε²  (variance complement for unit marginals).
        # E_π[ZX_w^T] = c_ε · Σ̂_w
        # E_π[ZZ^T]   = σ² · I + c_ε² · Σ̂_w
        # M_w = E_π[ZZ^T]^{-1} E_π[ZX_w^T]  (best linear projection)
        n, d = X_whitened.shape
        Sigma_hat = (X_whitened.T @ X_whitened) / n
        c = self.s_fwd
        sigma_sq = 1.0 - c ** 2
        Ezz = sigma_sq * np.eye(d) + c ** 2 * Sigma_hat
        Ezx = c * Sigma_hat
        M_w = np.linalg.solve(Ezz, Ezx)
        self.W = self.L @ M_w

        # ── Resampling pool ──────────────────────────────────────────────
        self.Z_full = self.s_fwd * X_whitened
        self._compute_diagnostics(report_title="EOTExplainer")

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> dict:
        n, d = X.shape
        Z = self.s_fwd * (X - self.mean) @ self.L_inv
        y_pred = self.model(X)

        ueifs_Z = self._phi_Z(Z, y_pred, y_true=y)
        ueifs_X = ueifs_Z @ (self.W ** 2).T

        # Store per-sample UEIFs for group_importance()
        self.ueifs_X = ueifs_X
        self.ueifs_Z = ueifs_Z

        ddof = 1 if n > 1 else 0
        results = {
            "phi_X": np.mean(ueifs_X, axis=0),
            "std_X": np.std(ueifs_X, axis=0),
            "se_X": np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n),
            "phi_Z": np.mean(ueifs_Z, axis=0),
            "std_Z": np.std(ueifs_Z, axis=0),
            "se_Z": np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n),
        }
        self._cache_results(results, n)
        return results

    def _phi_Z(self, Z: np.ndarray, y_pred: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute per-sample uncentered UEIF in Z-space via counterfactual resampling.

        For each feature j, replaces Z_j with independent draws and scores the
        result through ``self._loss`` using the CPI/SCPI averaging order in
        ``self.method``:

            DFI / loss-difference form (y_true provided):
                UEIF_{i,j} = agg_b L(y_i, ŷ_{b,i}) - L(y_i, ŷ_i)
            prediction-shift form (default, y_true=None; squared error only):
                UEIF_{i,j} = agg_b L(ŷ_i, ŷ_{b,i})

        where ``agg_b`` averages the prediction first (CPI) or the loss first
        (SCPI).  With the default squared-error loss the DFI form reduces to
        ``(y_i - ȳ_{-j,i})² - (y_i - ŷ_i)²`` (Williamson & Feng, 2023), which is
        centered near zero for null features.

        Parameters
        ----------
        Z : (n, d) array
            EOT-mapped whitened data.
        y_pred : (n,) array
            Model predictions on the original X.
        y_true : (n,) array or None
            True outcome values.  When provided, uses the DFI loss-difference form
            and enables proper null-feature thresholding; when omitted, the
            label-free divergence form is used.
        """
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        L_z = self.L_z

        for j in range(d):
            rng = np.random.default_rng(self.random_state + j)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))

            if self.sampling_method == "resample":
                idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[idx, j]
            elif self.sampling_method == "permutation":
                perm = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_flat = Z_tilde.reshape(-1, d)
            X_tilde = Z_flat @ L_z + self.mean
            y_tilde_all = self.model(X_tilde).reshape(self.nsamples, n)
            ueifs_Z[:, j] = self._ueif_from_counterfactuals(
                y_pred, y_tilde_all, method=self.method, y_true=y_true
            )

        return ueifs_Z

    def _auto_epsilon(self, X_centered: np.ndarray) -> float:
        """
        Auto-tune epsilon from latent geometry.

        Estimates pairwise distances in Gaussian-whitened latent space and
        uses a conservative shrinkage factor to avoid over-smoothing.
        """
        if X_centered.shape[0] < 2:
            return self.epsilon

        Z_ref = X_centered @ self.L_inv
        n = min(1000, Z_ref.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(Z_ref.shape[0], size=n, replace=False)
        Z_sub = Z_ref[idx]

        sq_dists = pdist(Z_sub, metric="sqeuclidean")
        if sq_dists.size == 0:
            return self.epsilon

        median_sq_dist = np.median(sq_dists)
        eps = 0.25 * (median_sq_dist / Z_ref.shape[1])
        return max(eps, 1e-3)

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Decode Z to X using the population coupling: E[X | Z] ≈ Z @ L_z + mean."""
        return np.asarray(Z) @ self.L_z + self.mean


DFIExplainer = OTExplainer


class FlowExplainer(Explainer):
    """
    Flow-based DFI explainer using normalizing flows.
    
    Implements CPI (Conditional Permutation Importance) and 
    SCPI (Sobol-CPI) methods. Both measure feature importance in Z-space:

    - CPI: Squared difference after averaging predictions: (Y - E[f(X_tilde)])^2
    - SCPI: Conditional variance of predictions: Var[f(X_tilde)]
    
    For L2 loss with independent (disentangled) features, CPI and SCPI give
    similar results. SCPI is related to the Sobol total-order sensitivity index.
    
    Z-space importance is transformed to X-space using the Jacobian of the flow
    ``phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]`` where H = dX/dZ is the Jacobian
    of the decoder transformation.
    
    Parameters
    ----------
    model : callable
        The model to explain. Should take (n, d) array and return (n,) predictions.
    data : numpy.ndarray
        Background data for fitting flow and resampling. Shape (n, d).
    flow_model : object, optional
        Pre-trained flow model. If None, will create default FlowMatchingModel.
    fit_flow : bool, default=True
        Whether to fit flow model during initialization.
    nsamples : int, default=50
        Number of Monte Carlo samples per feature.
    sampling_method : str, default='resample'
        Method for generating counterfactual Z values:
        - 'resample': Sample from encoded background data
        - 'permutation': Permute within test set
        - 'normal': Sample from standard normal
        - 'condperm': Conditional permutation (regress Z_j | Z_{-j})
    permuter : object, optional
        Regressor for conditional permutation method. Defaults to LinearRegression.
    method : str, default='cpi'
        Which importance method to use:
        - 'cpi': Conditional Permutation Importance - average predictions first
        - 'scpi': Sobol-CPI - average squared differences
        - 'both': Compute both CPI and SCPI
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool or str, default='final'
        Controls training output:
        - True or 'all': Show full progress bar
        - 'final': Only print final step status (default)
        - False: Silent
    compute_diagnostics : bool, default=True
        Whether to compute disentanglement diagnostics at setup time.
    flow_solver_rtol : float, default=1e-3
        Relative tolerance for default ODE integration in flow encode/decode.
    flow_solver_atol : float, default=1e-5
        Absolute tolerance for default ODE integration in flow encode/decode.
    diagnostics_solver_rtol : float, default=1e-6
        Relative tolerance for diagnostics round-trip integration.
    diagnostics_solver_atol : float, default=1e-8
        Absolute tolerance for diagnostics round-trip integration.
    **kwargs : dict
        Additional arguments passed to FlowMatchingModel if creating default.
    
    Attributes
    ----------
    flow_model : object
        The fitted normalizing flow model.
    Z_full : numpy.ndarray
        Encoded background data in latent space.
    method : str
        The importance method being used ('cpi', 'scpi', or 'both').
    
    Examples
    --------
    >>> import numpy as np
    >>> from fdfi.explainers import FlowExplainer
    >>> 
    >>> # Define a simple model
    >>> def model(x):
    ...     return x[:, 0] + 2 * x[:, 1]
    >>> 
    >>> # Create background data
    >>> X_train = np.random.randn(200, 5)
    >>> X_test = np.random.randn(50, 5)
    >>> 
    >>> # CPI only (default)
    >>> explainer = FlowExplainer(model, X_train, method='cpi')
    >>> results = explainer(X_test)
    >>> 
    >>> # SCPI (Sobol-CPI - different averaging order)
    >>> explainer = FlowExplainer(model, X_train, method='scpi')
    >>> results = explainer(X_test)
    """
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        flow_model: Optional[Any] = None,
        fit_flow: bool = True,
        nsamples: int = 50,
        sampling_method: str = "resample",
        permuter: Optional[Any] = None,
        method: str = "cpi",
        random_state: Optional[int] = None,
        verbose: Union[bool, str] = "final",
        compute_diagnostics: bool = True,
        **kwargs: Any
    ):
        """Initialize the FlowExplainer."""
        # Don't fit flow in base class
        super().__init__(
            model,
            data,
            fit_flow=False,
            verbose=verbose,
            compute_diagnostics=compute_diagnostics,
            **kwargs,
        )
        
        self.nsamples = nsamples
        self.sampling_method = sampling_method
        self.random_state = random_state if random_state is not None else 0
        self.method = method.lower()
        self.verbose = verbose
        self.kwargs = kwargs
        self.flow_solver_method = kwargs.get("flow_solver_method", "dopri5")
        self.flow_solver_rtol = kwargs.get("flow_solver_rtol", 1e-3)
        self.flow_solver_atol = kwargs.get("flow_solver_atol", 1e-5)
        self.diagnostics_solver_method = kwargs.get(
            "diagnostics_solver_method", self.flow_solver_method
        )
        self.diagnostics_solver_rtol = kwargs.get("diagnostics_solver_rtol", 1e-6)
        self.diagnostics_solver_atol = kwargs.get("diagnostics_solver_atol", 1e-8)
        self.flow_training_seed = kwargs.get("flow_training_seed", self.random_state)
        
        if self.method not in ("cpi", "scpi", "both"):
            raise ValueError(f"method must be 'cpi', 'scpi', or 'both', got '{method}'")
        
        # Set up permuter for condperm method
        if permuter is None:
            from sklearn.linear_model import LinearRegression
            self.permuter = LinearRegression()
        else:
            self.permuter = permuter
        
        # Flow model setup
        self.flow_model = flow_model
        self.Z_full = None
        
        if flow_model is not None:
            # Use provided flow model.
            # When compute_diagnostics=False (e.g. inside Crossfitting._crossfit),
            # skip both the full-data encoding and the expensive high-precision
            # diagnostic encoding — Z_full will be set externally by _crossfit.
            if self.compute_diagnostics:
                self._encode_background()
                if self.Z_full is not None:
                    self._refresh_flow_diagnostics()
        elif fit_flow and data is not None:
            # Create and fit default flow model
            self.fit_flow(num_steps=kwargs.get("num_steps", 5000), verbose=self.verbose)
    
    def fit_flow(
        self,
        X: Optional[np.ndarray] = None,
        num_steps: int = 5000,
        verbose: Union[bool, str] = None,
        **kwargs
    ) -> "FlowExplainer":
        """
        Fit the flow model on data.
        
        Can be called after initialization with fit_flow=False,
        or to refit on new data.
        
        Parameters
        ----------
        X : numpy.ndarray, optional
            Data to fit on. If None, uses self.data.
        num_steps : int, default=5000
            Number of training steps.
        verbose : bool or str, optional
            Controls training output. If None, uses self.verbose.
            - True or 'all': Show full progress bar
            - 'final': Only print final step status (default)
            - False: Silent
        **kwargs
            Additional arguments passed to flow_model.fit().
        
        Returns
        -------
        self
            For method chaining.
        """
        if X is not None:
            self.data = X
        
        if self.data is None:
            raise ValueError("No data provided for flow fitting.")
        
        try:
            from .models import FlowMatchingModel
        except ImportError as exc:
            raise ImportError(
                "Flow matching requires torch; install it with: pip install torch torchdiffeq"
            ) from exc

        import torch
        if self.flow_training_seed is not None:
            torch.manual_seed(int(self.flow_training_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.flow_training_seed))
            np.random.seed(int(self.flow_training_seed))
        
        d = self.data.shape[1]
        self.flow_model = FlowMatchingModel(
            X=self.data,
            dim=d,
            device=self.kwargs.get("device"),
            hidden_dim=self.kwargs.get("hidden_dim", 64),
            time_embed_dim=self.kwargs.get("time_embed_dim", 32),
            num_blocks=self.kwargs.get("num_blocks", 1),
            use_bn=self.kwargs.get("use_bn", False),
        )
        
        _verbose = verbose if verbose is not None else getattr(self, 'verbose', 'final')
        self.verbose = _verbose
        if _verbose is True or _verbose == 'all' or _verbose == 'final':
            self._log("Training flow model...", level="info")
        _dequantize_noise = kwargs.pop("dequantize_noise", self.kwargs.get("dequantize_noise", 0.0))
        self.flow_model.fit(num_steps=num_steps, verbose=_verbose,
                            dequantize_noise=_dequantize_noise, **kwargs)
        
        # Encode background data
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        
        return self
    
    def set_flow(self, flow_model: Any) -> "FlowExplainer":
        """
        Set a user-provided flow model.
        
        The flow model must have a sample_batch(x, t_span) method where:
        - t_span=(1, 0) encodes X to Z
        - t_span=(0, 1) decodes Z to X
        
        Parameters
        ----------
        flow_model : object
            A flow model with sample_batch(x, t_span) method.
        
        Returns
        -------
        self
            For method chaining.
        """
        self.flow_model = flow_model
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        return self
    
    def _encode_background(self) -> None:
        """Encode background data to Z space."""
        if self.data is not None and self.flow_model is not None:
            self.Z_full = self._encode_to_Z(self.data)
    
    def _encode_to_Z(
        self,
        X: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform X to latent space Z."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(
                X,
                t_span=(1, 0),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(Z, torch.Tensor):
                Z = Z.cpu().numpy()
        return Z
    
    def _decode_to_X(
        self,
        Z: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform Z back to X space."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(
                Z,
                t_span=(0, 1),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(X_hat, torch.Tensor):
                X_hat = X_hat.cpu().numpy()
        return X_hat

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode Z to X for diagnostics using tighter ODE tolerances.

        This keeps reconstruction-fidelity diagnostics focused on model fit
        rather than numerical integration error.
        """
        return self._decode_to_X(
            Z,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )

    def _refresh_flow_diagnostics(self) -> None:
        """Compute diagnostics from high-precision X -> Z -> X round-trip."""
        if self.data is None or self.flow_model is None:
            return
        Z_diag = self._encode_to_Z(
            self.data,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )
        self._compute_diagnostics(
            X_orig=self.data,
            Z_full=Z_diag,
            report_title="Flow Model",
        )
    
    def _compute_jacobian(self, Z: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """
        Compute the Jacobian dX/dZ averaged over samples.
        
        For the flow decoder T: Z -> X, computes H = dX/dZ at each sample point
        and returns the average H matrix. This is used to transform Z-space
        importance to X-space importance via:
            phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]
        
        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        batch_size : int, default=50
            Batch size for Jacobian computation.
        
        Returns
        -------
        H : numpy.ndarray
            Average Jacobian matrix of shape (d, d), where H[l, k] = dX_l/dZ_k.
        """
        import torch
        from torch.autograd.functional import jacobian
        
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        n, d = Z.shape
        
        # Get device from flow model
        if hasattr(self.flow_model, 'device'):
            device = self.flow_model.device
        elif hasattr(self.flow_model, 'model'):
            device = next(self.flow_model.model.parameters()).device
        else:
            device = torch.device('cpu')
        
        # Define decoder function for jacobian computation
        def decoder_fn(z_single):
            """Decode a single Z vector to X."""
            z_batch = z_single.unsqueeze(0)  # (1, d)
            x_batch = self.flow_model.sample_batch(
                z_batch,
                t_span=(0, 1),
                rtol=self.flow_solver_rtol,
                atol=self.flow_solver_atol,
                method=self.flow_solver_method,
            )
            if not isinstance(x_batch, torch.Tensor):
                x_batch = torch.tensor(x_batch, dtype=torch.float32, device=device)
            return x_batch.squeeze(0)  # (d,)
        
        # Compute Jacobian for a subset of samples (for efficiency)
        n_samples = min(n, batch_size)
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
        
        H_sum = np.zeros((d, d))
        
        for idx in indices:
            z_single = torch.tensor(Z[idx], dtype=torch.float32, device=device)
            # jacobian returns (output_dim, input_dim) = (d, d)
            jac = jacobian(decoder_fn, z_single)
            H_sum += jac.cpu().numpy()
        
        H_avg = H_sum / n_samples
        return H_avg
    
    def _phi_Z(self, Z: np.ndarray, y: np.ndarray, y_true: Optional[np.ndarray] = None) -> tuple:
        """
        Compute CPI and SCPI importance in Z-space through ``self._loss``.

        Both scores use the counterfactual predictions for feature ``j`` but
        differ in the averaging order:

        - CPI: average the prediction first, then apply the loss
          ``L(r, E_b[Ỹ])``.
        - SCPI: apply the loss per Monte-Carlo replicate first, then average
          ``E_b[L(r, Ỹ)]``.

        Here ``r`` is ``y_true`` when supplied (loss-difference / DFI form) or
        the baseline prediction ``y`` otherwise (prediction-shift form, defined
        only for the squared-error loss).

        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        y : numpy.ndarray
            Baseline model predictions, shape (n,).
        y_true : numpy.ndarray, optional
            True outcomes, shape (n,).  When provided, uses the DFI loss-difference
            form; when omitted, the label-free divergence form is used.

        Returns
        -------
        ueifs_cpi : numpy.ndarray
            CPI per-sample importance, shape (n, d).
        ueifs_scpi : numpy.ndarray
            SCPI per-sample importance, shape (n, d).
        """
        n, d = Z.shape
        ueifs_cpi = np.zeros((n, d))
        ueifs_scpi = np.zeros((n, d))
        
        for j in range(d):
            rng = np.random.default_rng(self.random_state + j)
            
            # Create B copies of Z: shape (B, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            
            # Replace j-th component based on sampling method
            if self.sampling_method == "resample":
                if self.Z_full is None:
                    raise ValueError("Z_full is None. Call fit_flow() first.")
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            elif self.sampling_method == "condperm":
                # Conditional permutation using permuter
                from sklearn.base import clone
                from sklearn.utils import shuffle
                Z_minus_j = np.delete(Z, j, axis=1)
                z_j = Z[:, j]
                rg = clone(self.permuter)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)
                eps_z = z_j - z_j_hat
                for b in range(self.nsamples):
                    eps_perm = shuffle(eps_z, random_state=self.random_state + j * 1000 + b)
                    Z_tilde[b, :, j] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            # Decode to X space
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = self._decode_to_X(Z_tilde_flat)
            
            # Evaluate model: shape (nsamples, n)
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n)
            
            # CPI: average the prediction first, then apply the loss.
            #   φ_j^CPI = L(r, E_b[Ỹ])   with r = y_true or the baseline prediction
            ueifs_cpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="cpi", y_true=y_true
            )
            
            # SCPI (Sobol-CPI): apply the loss per replicate, then average.
            #   φ_j^SCPI = E_b[L(r, Ỹ)]
            # For squared error this equals CPI + Var_b(Ỹ), matching the
            # documented SCPI = E_b[(Y - f(X̃_b))²].
            ueifs_scpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="scpi", y_true=y_true
            )
        
        return ueifs_cpi, ueifs_scpi
    
    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> dict:
        """
        Compute feature importance.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain. Shape (n_samples, n_features).
        y : numpy.ndarray, optional
            True outcomes, shape (n_samples,). When provided, uses the DFI
            loss-difference form; when omitted, the label-free divergence form
            is used.
        **kwargs : dict
            Additional parameters (unused, for API compatibility).
        
        Returns
        -------
        dict
            Dictionary containing:
            - phi_Z: Z-space importance (d,) - CPI or SCPI depending on method
            - std_Z: Standard deviation (d,)
            - se_Z: Standard error (d,)
            - phi_X: X-space importance (d,) - transformed via Jacobian
            - std_X: Standard deviation (d,)
            - se_X: Standard error (d,)
            When method='both', also includes phi_Z_scpi, std_Z_scpi, se_Z_scpi.
        """
        n, d = X.shape
        
        # Encode to Z space
        Z = self._encode_to_Z(X)
        
        # Get model predictions on decoded X (for consistency with flow)
        X_hat = self._decode_to_X(Z)
        y_pred = self.model(X_hat)
        
        # Compute both CPI and SCPI in Z-space (per-sample UEIFs)
        ueifs_cpi, ueifs_scpi = self._phi_Z(Z, y_pred, y_true=y)
        
        # Transform per-sample UEIFs from Z-space to X-space via Jacobian H = dX/dZ
        # ueifs_X[i, l] = sum_k H[l,k]^2 * ueifs_Z[i, k]
        jacobian_mode = self.kwargs.get("jacobian_mode", "average")
        n_jac = self.kwargs.get("jacobian_n_samples", 100)
        if jacobian_mode == "per_sample":
            # Per-sample Jacobians: H_batch[i] = dX/dZ at Z[i]
            H_batch = self.flow_model.Jacobi_Batch(Z)   # (n, d, d)
            H_sq_batch = H_batch ** 2
            ueifs_cpi_X  = np.einsum("ilk,ik->il", H_sq_batch, ueifs_cpi)
            ueifs_scpi_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_scpi)
        elif jacobian_mode == "avg_sq":
            # Average of squared Jacobians: E[H_i^2] — correct for binary data.
            # Avoids the cancellation in E[H_i]^2 for sign-flipping features.
            n_est = min(n, n_jac)
            H_batch = self.flow_model.Jacobi_Batch(Z[:n_est])  # (n_est, d, d)
            H_sq_avg = (H_batch ** 2).mean(axis=0)              # (d, d)
            ueifs_cpi_X  = ueifs_cpi  @ H_sq_avg.T
            ueifs_scpi_X = ueifs_scpi @ H_sq_avg.T
        else:  # "average" — existing behaviour, backward-compatible
            H = self._compute_jacobian(Z)
            H_sq = H ** 2
            ueifs_cpi_X  = ueifs_cpi  @ H_sq.T
            ueifs_scpi_X = ueifs_scpi @ H_sq.T
        
        # Store per-sample UEIFs for group_importance()
        if self.method == "scpi":
            self.ueifs_Z = ueifs_scpi
            self.ueifs_X = ueifs_scpi_X
        else:
            self.ueifs_Z = ueifs_cpi
            self.ueifs_X = ueifs_cpi_X
        
        # Aggregate statistics
        ddof = 1 if n > 1 else 0
        results = {}
        
        if self.method == "cpi":
            # Z-space (CPI)
            phi_Z = np.mean(ueifs_cpi, axis=0)
            std_Z = np.std(ueifs_cpi, axis=0)
            se_Z = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_cpi_X, axis=0)
            std_X = np.std(ueifs_cpi_X, axis=0)
            se_X = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        elif self.method == "scpi":
            # Z-space (SCPI)
            phi_Z = np.mean(ueifs_scpi, axis=0)
            std_Z = np.std(ueifs_scpi, axis=0)
            se_Z = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_scpi_X, axis=0)
            std_X = np.std(ueifs_scpi_X, axis=0)
            se_X = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        else:  # method == "both"
            # CPI (Z-space and X-space)
            phi_Z_cpi = np.mean(ueifs_cpi, axis=0)
            std_Z_cpi = np.std(ueifs_cpi, axis=0)
            se_Z_cpi = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_cpi = np.mean(ueifs_cpi_X, axis=0)
            std_X_cpi = np.std(ueifs_cpi_X, axis=0)
            se_X_cpi = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            # SCPI (Z-space and X-space)
            phi_Z_scpi = np.mean(ueifs_scpi, axis=0)
            std_Z_scpi = np.std(ueifs_scpi, axis=0)
            se_Z_scpi = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_scpi = np.mean(ueifs_scpi_X, axis=0)
            std_X_scpi = np.std(ueifs_scpi_X, axis=0)
            se_X_scpi = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                # Default to CPI for phi_Z/phi_X
                "phi_Z": phi_Z_cpi,
                "std_Z": std_Z_cpi,
                "se_Z": se_Z_cpi,
                "phi_X": phi_X_cpi,
                "std_X": std_X_cpi,
                "se_X": se_X_cpi,
                # SCPI with suffix
                "phi_Z_scpi": phi_Z_scpi,
                "std_Z_scpi": std_Z_scpi,
                "se_Z_scpi": se_Z_scpi,
                "phi_X_scpi": phi_X_scpi,
                "std_X_scpi": std_X_scpi,
                "se_X_scpi": se_X_scpi,
            })
        
        self._cache_results(results, n)
        return results

    def explain_batches(
        self,
        X: np.ndarray,
        batch_size: int = 50,
        y: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict:
        """Compute feature importance in row batches and aggregate all rows.

        This preserves the public batching API introduced on ``yifan-dev``
        while using the 0.0.10 loss semantics.  ``y`` is sliced with ``X`` and
        passed to :meth:`_phi_Z` as ``y_true`` so each batch uses the same
        label-aware loss-difference form as :meth:`__call__`.  With ``y=None``
        the label-free CPI/SCPI form is used.

        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain, shape ``(n_samples, n_features)``.
        batch_size : int, default=50
            Number of rows processed per batch.
        y : numpy.ndarray, optional
            True outcomes, shape ``(n_samples,)``.  Required by losses whose
            intended interpretation is label-aware.
        **kwargs : dict
            Reserved for API compatibility.

        Returns
        -------
        dict
            The same aggregate result dictionary as :meth:`__call__`.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one row.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError("y must be a 1D array with one value per row of X.")

        n, _ = X.shape
        Z = self._encode_to_Z(X)
        X_hat = self._decode_to_X(Z)
        y_pred = self.model(X_hat)

        ueifs_cpi_batches = []
        ueifs_scpi_batches = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            y_true_batch = None if y is None else y[start:stop]
            ueifs_cpi_batch, ueifs_scpi_batch = self._phi_Z(
                Z[start:stop],
                y_pred[start:stop],
                y_true=y_true_batch,
            )
            ueifs_cpi_batches.append(ueifs_cpi_batch)
            ueifs_scpi_batches.append(ueifs_scpi_batch)

        ueifs_cpi = np.vstack(ueifs_cpi_batches)
        ueifs_scpi = np.vstack(ueifs_scpi_batches)

        jacobian_mode = self.kwargs.get("jacobian_mode", "average")
        n_jac = self.kwargs.get("jacobian_n_samples", 100)
        if jacobian_mode == "per_sample":
            H_batches = []
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                H_batches.append(self.flow_model.Jacobi_Batch(Z[start:stop]))
            H_sq_batch = np.concatenate(H_batches, axis=0) ** 2
            ueifs_cpi_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_cpi)
            ueifs_scpi_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_scpi)
        elif jacobian_mode == "avg_sq":
            n_est = min(n, n_jac)
            H_batch = self.flow_model.Jacobi_Batch(Z[:n_est])
            H_sq_avg = (H_batch ** 2).mean(axis=0)
            ueifs_cpi_X = ueifs_cpi @ H_sq_avg.T
            ueifs_scpi_X = ueifs_scpi @ H_sq_avg.T
        else:
            H = self._compute_jacobian(Z)
            H_sq = H ** 2
            ueifs_cpi_X = ueifs_cpi @ H_sq.T
            ueifs_scpi_X = ueifs_scpi @ H_sq.T

        if self.method == "scpi":
            self.ueifs_Z = ueifs_scpi
            self.ueifs_X = ueifs_scpi_X
        else:
            self.ueifs_Z = ueifs_cpi
            self.ueifs_X = ueifs_cpi_X

        ddof = 1 if n > 1 else 0

        def aggregate(ueifs_Z: np.ndarray, ueifs_X: np.ndarray) -> dict:
            return {
                "phi_Z": ueifs_Z.mean(axis=0),
                "std_Z": ueifs_Z.std(axis=0),
                "se_Z": ueifs_Z.std(axis=0, ddof=ddof) / np.sqrt(n),
                "phi_X": ueifs_X.mean(axis=0),
                "std_X": ueifs_X.std(axis=0),
                "se_X": ueifs_X.std(axis=0, ddof=ddof) / np.sqrt(n),
            }

        if self.method == "cpi":
            results = aggregate(ueifs_cpi, ueifs_cpi_X)
        elif self.method == "scpi":
            results = aggregate(ueifs_scpi, ueifs_scpi_X)
        else:
            results = aggregate(ueifs_cpi, ueifs_cpi_X)
            scpi_results = aggregate(ueifs_scpi, ueifs_scpi_X)
            results.update(
                {
                    "phi_Z_scpi": scpi_results["phi_Z"],
                    "std_Z_scpi": scpi_results["std_Z"],
                    "se_Z_scpi": scpi_results["se_Z"],
                    "phi_X_scpi": scpi_results["phi_X"],
                    "std_X_scpi": scpi_results["std_X"],
                    "se_X_scpi": scpi_results["se_X"],
                }
            )

        self._cache_results(results, n)
        return results


class Crossfitting(Explainer):
    """
    Cross-fitted DFI explainer for valid inference at small sample sizes.

    Wraps any Explainer subclass and performs cross-fitting using a
    scikit-learn cross-validation splitter.  The disentanglement map is
    fitted on the training portion of each split and importance is
    evaluated on the held-out portion.  Final estimates are the
    ensemble average of cross-fitted predictors.

    Parameters
    ----------
    model : callable
        The model to explain.  Takes (n, d) array, returns (n,) predictions.
    data : numpy.ndarray
        Full dataset.  Shape (n, d).
    explainer_class : type, default=OTExplainer
        The explainer class to instantiate per split.  Must be a subclass
        of Explainer (e.g., OTExplainer, EOTExplainer, FlowExplainer).
    cv : int or sklearn cross-validation splitter, default=5
        Controls how data is split for cross-fitting.
        Pass an ``int`` for ``KFold(n_splits=cv, shuffle=True)``,
        or any scikit-learn splitter instance (e.g. ``KFold``,
        ``StratifiedKFold``, ``ShuffleSplit``, ``RepeatedKFold``,
        ``GroupKFold``).  Any object implementing
        ``.split(X, y, groups)`` is accepted.
    y : array-like of shape (n,), optional
        Target / response variable.  Required only when using a stratified
        splitter so that fold assignment preserves class distribution.
    groups : array-like of shape (n,), optional
        Group labels for group-aware splitters (``GroupKFold``, etc.).
    random_state : int or None, default=None
        Random seed for the default ``KFold`` splitter (when *cv* is int)
        and passed to child explainers.
    **kwargs : dict
        Additional keyword arguments forwarded to each split's explainer
        constructor (e.g., nsamples, epsilon, sampling_method, num_steps).

    Attributes
    ----------
    cv_ : sklearn splitter instance
        The resolved cross-validation splitter.
    fold_explainers : list[Explainer]
        The fitted explainer instances (one per split).
    fold_indices : list[tuple[numpy.ndarray, numpy.ndarray]]
        ``(train_idx, test_idx)`` for each split.
    ueifs_X : numpy.ndarray or None
        Per-sample X-space UEIFs, shape (n, d), after calling with
        ``X=None``.
    ueifs_Z : numpy.ndarray or None
        Per-sample Z-space UEIFs, shape (n, d), after calling with
        ``X=None``.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        explainer_class: type = OTExplainer,
        cv: Union[int, Any] = 5,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        cv_kwargs: Optional[dict] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        # Remove fit_flow from kwargs before passing to super() and child explainers
        # so it doesn't conflict with the hardcoded fit_flow=False below.
        kwargs.pop("fit_flow", None)
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.explainer_class = explainer_class
        self.y = np.asarray(y) if y is not None else None
        self.groups = np.asarray(groups) if groups is not None else None
        # cv_kwargs: extra keyword arguments forwarded to cv.split().
        # Overrides the defaults (y=self.y, groups=self.groups).
        # Use case: pass discrete class labels to StratifiedKFold when y
        # is continuous (e.g. standardized), e.g. cv_kwargs={"y": y_binary}.
        self.cv_kwargs = cv_kwargs or {}
        self.random_state = random_state
        self.cf_kwargs = kwargs

        self.cv_ = self._resolve_cv(cv)
        self.fold_explainers: list = []
        self.fold_indices: list = []
        self.ueifs_X: Optional[np.ndarray] = None
        self.ueifs_Z: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # CV resolution
    # ------------------------------------------------------------------

    def _resolve_cv(self, cv: Union[int, Any]) -> Any:
        """Resolve *cv* parameter to a scikit-learn splitter instance."""
        if isinstance(cv, int):
            from sklearn.model_selection import KFold
            return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        # Accept any object with a .split() method
        if not hasattr(cv, "split"):
            raise TypeError(
                f"cv must be an int or an object with a .split() method, "
                f"got {type(cv)}"
            )
        return cv

    # ------------------------------------------------------------------
    # Per-sample UEIF extraction (dispatch by explainer type)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_persample_ueifs(
        explainer: "Explainer",
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-sample UEIFs in both Z- and X-space for *X_test*
        using the fitted *explainer*.

        Parameters
        ----------
        y_test : (n_test,) array or None
            True outcomes for X_test.  When provided to EOTExplainer, uses
            the DFI formula ``(y - ȳ_{-j})² - (y - ŷ)²`` for proper
            null-feature thresholding.

        Returns
        -------
        ueifs_X : numpy.ndarray, shape (n_test, d)
        ueifs_Z : numpy.ndarray, shape (n_test, d)
        """
        y_pred = explainer.model(X_test)

        if isinstance(explainer, FlowExplainer):
            Z = explainer._encode_to_Z(X_test)
            X_hat = explainer._decode_to_X(Z)
            y_pred = explainer.model(X_hat)
            ueifs_cpi, ueifs_scpi = explainer._phi_Z(Z, y_pred, y_true=y_test)
            jacobian_mode = explainer.kwargs.get("jacobian_mode", "average")
            n_jac = explainer.kwargs.get("jacobian_n_samples", 100)
            n_test = Z.shape[0]
            if jacobian_mode == "per_sample":
                H_batch = explainer.flow_model.Jacobi_Batch(Z)   # (n, d, d)
                H_sq_batch = H_batch ** 2
                if explainer.method == "scpi":
                    ueifs_Z = ueifs_scpi
                    ueifs_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_scpi)
                else:
                    ueifs_Z = ueifs_cpi
                    ueifs_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_cpi)
            elif jacobian_mode == "avg_sq":
                n_est = min(n_test, n_jac)
                H_batch = explainer.flow_model.Jacobi_Batch(Z[:n_est])  # (n_est, d, d)
                H_sq_avg = (H_batch ** 2).mean(axis=0)                   # (d, d)
                if explainer.method == "scpi":
                    ueifs_Z = ueifs_scpi
                else:
                    ueifs_Z = ueifs_cpi
                ueifs_X = ueifs_Z @ H_sq_avg.T
            else:
                H = explainer._compute_jacobian(Z)
                H_sq = H ** 2
                if explainer.method == "scpi":
                    ueifs_Z = ueifs_scpi
                else:
                    ueifs_Z = ueifs_cpi
                ueifs_X = ueifs_Z @ H_sq.T
        elif isinstance(explainer, EOTExplainer):
            Z = explainer.s_fwd * (X_test - explainer.mean) @ explainer.L_inv
            ueifs_Z = explainer._phi_Z(Z, y_pred, y_true=y_test)
            ueifs_X = ueifs_Z @ (explainer.W ** 2).T
        elif isinstance(explainer, OTExplainer):
            Z = (X_test - explainer.mean) @ explainer.L_inv
            ueifs_Z = explainer._phi_Z(Z, y_pred, y_true=y_test)
            H = explainer.L ** 2
            ueifs_X = ueifs_Z @ H.T
        else:
            # Fallback: call the explainer and replicate aggregated scores
            results = explainer(X_test)
            n = X_test.shape[0]
            ueifs_X = np.tile(results["phi_X"], (n, 1))
            ueifs_Z = np.tile(results["phi_Z"], (n, 1))

        return ueifs_X, ueifs_Z

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Compute cross-fitted feature importance.

        If *X* is ``None``, performs full cross-fitting on ``self.data``:
        each split's test set is the held-out portion of the data.

        If *X* is provided, uses the ensemble of fitted fold explainers
        to compute importance on *X* and averages the results.

        Parameters
        ----------
        X : numpy.ndarray or None
            If None, cross-fit on ``self.data`` (recommended for valid
            inference).  If provided, shape (m, d), ensemble-predict on
            new data.

        Returns
        -------
        dict
            Same format as OTExplainer / FlowExplainer:
            ``phi_X, std_X, se_X, phi_Z, std_Z, se_Z``.
        """
        if X is not None:
            return self._ensemble_predict(X)
        return self._crossfit()

    # ------------------------------------------------------------------
    # Internal: cross-fit on self.data
    # ------------------------------------------------------------------

    def _crossfit(self) -> dict:
        n, d = self.data.shape
        self.fold_explainers = []
        self.fold_indices = []

        # Detect whether the model is a sklearn-style estimator that should
        # be cloned and refitted per fold (true cross-fitting of the predictor).
        _is_estimator = hasattr(self.model, "fit") and hasattr(self.model, "predict")

        # Collect per-sample UEIFs; support overlapping test sets
        ueif_counts = np.zeros(n, dtype=int)
        ueifs_X_accum = np.zeros((n, d))
        ueifs_Z_accum = np.zeros((n, d))

        _split_kw = {"y": self.y, "groups": self.groups}
        _split_kw.update(self.cv_kwargs)
        for train_idx, test_idx in self.cv_.split(self.data, **_split_kw):
            self.fold_indices.append((train_idx, test_idx))

            if _is_estimator:
                from sklearn.base import clone
                if self.y is None:
                    raise ValueError(
                        "y must be provided to Crossfitting when model is a "
                        "sklearn estimator so it can be refitted per fold."
                    )
                fold_clf = clone(self.model)
                fold_clf.fit(self.data[train_idx], self.y[train_idx])
                if hasattr(fold_clf, "predict_proba") and hasattr(fold_clf, "classes_"):
                    if len(fold_clf.classes_) == 2:
                        fold_model = lambda X_, clf=fold_clf: clf.predict_proba(X_)[:, 1]
                    else:
                        fold_model = lambda X_, clf=fold_clf: clf.predict(X_)
                else:
                    fold_model = lambda X_, clf=fold_clf: clf.predict(X_)
            else:
                fold_model = self.model

            # Build fold explainer on FULL data so that covariance / whitening
            # uses all n observations (matches refit_cov=False in the reference).
            # This avoids rank-deficiency when d > n_train (half the data).
            # Diagnostics are suppressed because ODE encoding of all n samples
            # (including normal and tight-tolerance variants) would be
            # immediately overwritten when we restrict Z_full below.
            fold_kwargs = {"compute_diagnostics": False}
            fold_kwargs.update(self.cf_kwargs)
            fold_exp = self.explainer_class(
                model=fold_model,
                data=self.data,
                random_state=self.random_state,
                **fold_kwargs,
            )
            # Restrict resampling pool to training fold only.
            if hasattr(fold_exp, "s_fwd"):
                # EOT: analytical forward map (scale * whitening)
                fold_exp.Z_full = (
                    fold_exp.s_fwd
                    * (self.data[train_idx] - fold_exp.mean)
                    @ fold_exp.L_inv
                )
            elif hasattr(fold_exp, "_encode_to_Z"):
                # FlowExplainer: encode training-fold samples through the flow.
                # The flow itself was trained on all data (full n); only the
                # resampling pool is restricted to in-fold rows so that
                # counterfactual draws respect the cross-fitting split.
                # Z_full may be None here (skipped _encode_background above)
                # — that is expected; we set it now.
                fold_exp.Z_full = fold_exp._encode_to_Z(self.data[train_idx])
            elif hasattr(fold_exp, "Z_full") and fold_exp.Z_full is not None:
                # OT / generic linear map (mean + L_inv attributes)
                fold_exp.Z_full = (
                    (self.data[train_idx] - fold_exp.mean) @ fold_exp.L_inv
                )
            self.fold_explainers.append(fold_exp)

            X_test = self.data[test_idx]
            y_test = self.y[test_idx] if self.y is not None else None
            ueifs_X_fold, ueifs_Z_fold = self._get_persample_ueifs(fold_exp, X_test, y_test=y_test)

            # Accumulate (handles overlapping test sets by averaging later)
            for local_i, global_i in enumerate(test_idx):
                ueifs_X_accum[global_i] += ueifs_X_fold[local_i]
                ueifs_Z_accum[global_i] += ueifs_Z_fold[local_i]
                ueif_counts[global_i] += 1

        # Average for samples that appeared in multiple test sets
        seen = ueif_counts > 0
        ueifs_X_accum[seen] /= ueif_counts[seen, None]
        ueifs_Z_accum[seen] /= ueif_counts[seen, None]

        # Keep only the samples that were actually evaluated
        self.ueifs_X = ueifs_X_accum[seen]
        self.ueifs_Z = ueifs_Z_accum[seen]
        n_eff = int(seen.sum())

        # Null-threshold for _last_results: zero out features whose mean UEIF is
        # negative (matches the DFI paper: features with E[UEIF] < 0 have no
        # evidence of importance). Applied to a *copy* so that self.ueifs_X /
        # self.ueifs_Z remain unthresholded — conf_int(groups=...) can then apply
        # its own threshold_null flag on the raw per-sample arrays.
        ueifs_X_thresh = self.ueifs_X.copy()
        ueifs_Z_thresh = self.ueifs_Z.copy()
        for arr in (ueifs_X_thresh, ueifs_Z_thresh):
            null_mask = arr.mean(axis=0) < 0
            arr[:, null_mask] = 0.0

        # SE with n_folds correction to match the DFI paper:
        #   sqn_eff = sqrt(n) * sqrt((n_folds-1)/n_folds)
        # For n_folds=2 this gives sqn_eff = sqrt(n/2), which is sqrt(2)× larger
        # than the naive sqrt(n), resulting in more conservative inference.
        n_folds = getattr(self.cv_, "n_splits", 1)
        sqn_eff = np.sqrt(n_eff)
        if n_folds > 1:
            sqn_eff *= np.sqrt((n_folds - 1) / n_folds)

        ddof = 1 if n_eff > 1 else 0
        results = {
            "phi_X": ueifs_X_thresh.mean(axis=0),
            "std_X": ueifs_X_thresh.std(axis=0),
            "se_X": ueifs_X_thresh.std(axis=0, ddof=ddof) / sqn_eff,
            "phi_Z": ueifs_Z_thresh.mean(axis=0),
            "std_Z": ueifs_Z_thresh.std(axis=0),
            "se_Z": ueifs_Z_thresh.std(axis=0, ddof=ddof) / sqn_eff,
        }
        self._cache_results(results, n_eff)
        return results

    # ------------------------------------------------------------------
    # Internal: ensemble prediction on new data
    # ------------------------------------------------------------------

    def _ensemble_predict(self, X: np.ndarray) -> dict:
        if not self.fold_explainers:
            # No fold explainers yet — run cross-fitting first
            self._crossfit()

        n = X.shape[0]
        phi_X_list, phi_Z_list = [], []
        se_X_list, se_Z_list = [], []

        for fold_exp in self.fold_explainers:
            r = fold_exp(X)
            phi_X_list.append(r["phi_X"])
            phi_Z_list.append(r["phi_Z"])
            se_X_list.append(r["se_X"])
            se_Z_list.append(r["se_Z"])

        K = len(self.fold_explainers)
        phi_X = np.mean(phi_X_list, axis=0)
        phi_Z = np.mean(phi_Z_list, axis=0)

        # Pooled SE: sqrt( mean of se_k^2 ) — accounts for within-fold variance
        se_X = np.sqrt(np.mean(np.array(se_X_list) ** 2, axis=0))
        se_Z = np.sqrt(np.mean(np.array(se_Z_list) ** 2, axis=0))

        std_X = se_X * np.sqrt(n)
        std_Z = se_Z * np.sqrt(n)

        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return results
