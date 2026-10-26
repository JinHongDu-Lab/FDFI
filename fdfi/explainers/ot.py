"""
Optimal-transport (Gaussian) DFI explainer.
"""

import numpy as np
from typing import Optional, Callable, Any, Tuple
from .base import Explainer
from ..results import ExplanationResults


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
        Seed for the counterfactual draws.  Each call draws from a new stream
        spawned from it, so repeated calls (e.g. on chunks of a test set) are
        independent and the sequence of calls is reproducible.
    method : {'cpi', 'scpi'}, default='cpi'
        Counterfactual resampling estimator:

        * ``'cpi'`` – normalized CPI from the FDFI paper: apply the loss to
          every resample, average the loss differences, and multiply by
          one half (``0.5 * E_b[L(r, ŷ_b) - L(r, ŷ)]``).
        * ``'scpi'`` – Sobol-CPI: average counterfactual predictions first,
          then apply the loss (``L(r, E_b[ŷ_b]) - L(r, ŷ)``).

        With squared-error loss, exact latent independence, and infinitely
        many resamples, the two population scores agree. The finite-resample
        SCPI plug-in is generally not identical to CPI.
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
    >>> print(results.phi_X)        # global importance, X-space
    >>> print(results.phi_Z)        # global importance, Z-space
    >>>
    >>> ci = explainer.conf_int(alpha=0.05, alternative="greater")
    >>> summary_bar(results.phi_X, results.se_X, show=False)
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        sampling_method: str = "resample",
        random_state: Optional[int] = 0,
        method: str = "cpi",
        **kwargs: Any
    ):
        """Initialize the OTExplainer."""
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.nsamples = nsamples
        self.regularize = kwargs.get("regularize", 1e-6)
        self.sampling_method = sampling_method
        self._set_random_state(random_state)
        self.method = method

        self._whiten(data)
        self._set_pool(data)
        self._compute_diagnostics(report_title="OTExplainer")

    def _set_pool(self, X: np.ndarray) -> None:
        self.Z_full = (X - self.mean) @ self.L_inv

    def __call__(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any
    ) -> ExplanationResults:
        """Compute feature importance for new observations.

        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape (n_samples, n_features).
        y : numpy.ndarray, optional
            True outcomes. When omitted, use the label-free divergence form.
        **kwargs : dict
            Additional parameters for API compatibility.

        Returns
        -------
        ExplanationResults
            Dict-compatible importance and uncertainty estimates in X- and
            Z-space, with attribute access, ``importance()``, ``to_frame()``
            and ``top_features()``.
        """
        return self._explain(X, y)

    def _ueifs(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample UEIFs ``(ueifs_X, ueifs_Z)`` for ``X``; shared with
        :class:`Crossfitting`."""
        Z = (X - self.mean) @ self.L_inv
        y_pred = np.asarray(self.model(X), dtype=float)
        ueifs_Z = self._phi_Z(Z, y_pred, self._call_key(), y_true=y)
        return ueifs_Z @ (self.L ** 2).T, ueifs_Z

    def _phi_Z(
        self,
        Z: np.ndarray,
        y_pred: np.ndarray,
        key: int,
        y_true: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sample uncentered EIF in Z-space via counterfactual resampling.

        For each feature ``j`` the j-th latent coordinate is replaced by
        independent draws and importance is scored through ``self._loss`` using
        the CPI/SCPI definition given by ``self.method``. CPI is one half the
        average per-resample loss difference; SCPI applies the loss after
        averaging the counterfactual predictions. With the default squared-
        error loss and no ``y_true``, the reference is the baseline prediction.
        Feature ``j`` draws from the stream ``[key, j]`` (see :meth:`_call_key`).
        """
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        
        for j in range(d):

            rng = np.random.default_rng([key, j])
            
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



DFIExplainer = OTExplainer
