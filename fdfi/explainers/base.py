"""
Base Explainer class for DFI.

Provides the interface for computing feature importance using disentangled
methods, plus shared inference (conf_int/summary/group_importance) and
diagnostics logic used by all Explainer subclasses.
"""

import numpy as np
from typing import Optional, Union, Callable, Any, Tuple
from scipy import stats
from ..utils import (
    TwoComponentMixture,
    compute_latent_independence,
    compute_mmd,
)
from ..losses import resolve_loss


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
                from ..models import FlowMatchingModel
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

        - ``'cpi'`` (the normalized FDFI convention): apply the loss to each
          replicate, average the loss differences, and multiply by one half —
          ``0.5 * E_b[L(r, ŷ_b) - L(r, ŷ)]``.
        - ``'scpi'`` (Sobol-CPI): average the counterfactual predictions first,
          then apply the loss — ``L(r, E_b[ŷ_b]) - L(r, ŷ)``.

        The factor one half makes CPI target the same population quantity as
        SCPI under squared loss, an exact disentangling map, and a Bayes
        predictor.  It is the normalization used by the FDFI paper; conventional
        (unnormalized) CPI is twice this value.

        When ``y_true`` is provided the score is a difference of losses (the DFI
        / LOCO form, centred near zero for null features)::

            CPI  = 0.5 * E_b[L(y_true, ŷ_b) - L(y_true, ŷ)]
            SCPI = L(y_true, E_b[ŷ_b]) - L(y_true, ŷ)

        Otherwise it is the label-free form that uses the model's own prediction
        ``ŷ`` as the reference and subtracts the self-loss floor::

            CPI  = 0.5 * E_b[L(ŷ, ŷ_b) - L(ŷ, ŷ)]
            SCPI = L(ŷ, E_b[ŷ_b]) - L(ŷ, ŷ)

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
            perturbed = loss(ref[None, :], y_tilde_all).mean(axis=0)
            return 0.5 * (perturbed - base)

        # Sobol-CPI estimates the restricted prediction by averaging the
        # counterfactual predictions before evaluating the loss.
        return loss(ref, y_tilde_all.mean(axis=0)) - base


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
