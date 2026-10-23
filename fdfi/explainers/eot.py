"""
Entropic optimal-transport (EOT) DFI explainer.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typing import Optional, Callable, Any, Tuple
from scipy.spatial.distance import pdist
from .base import Explainer
from ..results import ExplanationResults


def gaussian_eot_scale(epsilon: float) -> float:
    """Cross-covariance scale of the Gaussian EOT coupling.

    For source and target both N(0, I), quadratic cost ``||x - z||^2`` and
    entropic penalty ``epsilon * KL``, the optimal coupling is Gaussian with
    ``Z | X = x ~ N(s x, (1 - s^2) I)`` where ``s`` is the positive root of
    ``2 s^2 + epsilon s - 2 = 0``:

        s = (sqrt(epsilon^2 + 16) - epsilon) / 4.

    The conditional variance ``1 - s^2`` is what makes the marginal of ``Z``
    equal to ``N(0, I)``.  ``epsilon = 0`` gives the identity map (``s = 1``).
    """
    if epsilon <= 0:
        return 1.0
    return float((np.sqrt(epsilon ** 2 + 16.0) - epsilon) / 4.0)


class EOTExplainer(Explainer):
    """
    Entropic optimal-transport DFI explainer.

    The covariates are whitened with the background covariance,
    ``X_w = L^{-1} (X - mean)``, and coupled with the reference law ``N(0, I)``
    by entropic OT with quadratic cost and regularization ``epsilon``.
    Treating ``X_w`` as Gaussian, the coupling has the closed form

        Z | X_w ~ N(s X_w, (1 - s^2) I),   X_w | Z ~ N(s Z, (1 - s^2) I),
        s = (sqrt(eps^2 + 16) - eps) / 4,

    so that ``Z ~ N(0, I)`` has independent coordinates (see
    :func:`gaussian_eot_scale`).

    Latent features are drawn from the forward conditional, and the latent
    regression ``eta(z) = E[f(X) | Z = z]`` is the average of the model over
    ``n_eta`` draws from the backward conditional.  Counterfactuals replace
    ``Z_j`` by draws from its marginal and are scored by ``method``:

    - ``'cpi'``: one half the average per-replicate loss difference, each
      replicate evaluated with ``n_eta`` coupling draws.
    - ``'loco'``: integrated LOCO (squared error only).  Each replicate uses a
      single joint draw of ``Z_j`` and the coupling noise, so a feature costs
      ``nsamples`` model evaluations per point instead of
      ``nsamples * n_eta``.

    For the squared error both scores remove the Monte Carlo bias of the
    averaged baseline and counterfactual predictions (see
    :meth:`Explainer._ueif_from_counterfactuals`).  Each coupling draw's noise
    is shared across features, which leaves every feature's score and the
    per-sample standard errors unchanged in distribution and avoids
    recomputing the ``d x d`` noise transform per feature.

    Importance is attributed to ``X`` through the barycentric projection
    ``E[X | Z = z] = mean + s L z``, giving ``phi_X_l = sum_j W[l, j]^2 phi_Z_j``
    with ``W = s L``.  For standardized features
    ``sum_l phi_X_l = s^2 sum_j phi_Z_j``; the totals agree only at
    ``epsilon = 0``.

    Parameters
    ----------
    model : callable
        The model to explain.  Takes (n, d) array, returns (n,) predictions.
    data : numpy.ndarray
        Background data for whitening and resampling.  Shape (n, d).
    nsamples : int, default=50
        Number of counterfactual replicates per feature.
    epsilon : float, default=0.1
        EOT regularization parameter on the whitened scale.  ``epsilon = 0``
        gives the Gaussian OT map.
    auto_epsilon : bool, default=False
        If True, set ε from a median-distance heuristic in whitened space.
    sampling_method : str, default='resample'
        How to draw counterfactual Z_j values:
        - 'resample': sample from the background Z pool
        - 'permutation': permute within the test set
        - 'normal': sample from N(0, 1)
    random_state : int, default=0
        Random seed for reproducibility.
    method : {'cpi', 'loco'}, default='cpi'
        Importance estimator (see above).
    n_eta : int, default=10
        Coupling draws for the Monte Carlo average ``eta``; at least 2 so that
        its Monte Carlo variance can be estimated.
    n_jobs : int, default=1
        Number of features scored in parallel threads; -1 uses all CPUs.
        Each thread holds a few arrays of ``nsamples * n_test * d`` floats,
        and ``model`` must be safe to call from several threads (scikit-learn
        ``predict`` is).  Results do not depend on ``n_jobs``.
    **kwargs : dict
        Extra arguments forwarded to the base Explainer.
    """
    #: Attributes shown in the printed representation.
    _REPR_PARAMS = Explainer._REPR_PARAMS + ("n_eta",)

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        epsilon: float = 0.1,
        auto_epsilon: bool = False,
        sampling_method: str = "resample",
        random_state: Optional[int] = 0,
        method: str = "cpi",
        n_eta: int = 10,
        n_jobs: int = 1,
        **kwargs: Any
    ):
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.method = str(method).lower()
        if self.method not in ("cpi", "loco"):
            raise ValueError(f"method must be 'cpi' or 'loco', got {method!r}")
        if n_eta < 2:
            raise ValueError(f"n_eta must be at least 2, got {n_eta}")
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.auto_epsilon = auto_epsilon
        self.sampling_method = sampling_method
        self.random_state = random_state if random_state is not None else 0
        self.n_eta = n_eta
        self.n_jobs = n_jobs
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

        # ── Gaussian EOT coupling ────────────────────────────────────────
        self.s_fwd = gaussian_eot_scale(self.epsilon)
        self.sigma_fwd = float(np.sqrt(1.0 - self.s_fwd ** 2))
        # Jacobian of the barycentric projection E[X | Z = z] = z @ W + mean.
        self.W = self.s_fwd * self.L

        # ── Latent pool for resampling and diagnostics ───────────────────
        self.Z_full = self._latent(X_whitened, np.random.default_rng([self.random_state, 11]))
        self._compute_diagnostics(report_title="EOTExplainer")

    def _latent(self, X_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw latent features from the forward conditional ``Z | X_w``."""
        Z = self.s_fwd * X_w
        if self.sigma_fwd > 0:
            Z = Z + self.sigma_fwd * rng.standard_normal(X_w.shape)
        return Z

    def _eta(
        self, Z: np.ndarray, rng: np.random.Generator, n_draws: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo latent regression ``eta(z) = E[f(X) | Z = z]``.

        Returns the average of the model over ``n_draws`` backward draws
        ``X_w = s z + sqrt(1 - s^2) xi`` and the Monte Carlo variance of that
        average (zero for a single draw or a deterministic coupling).
        """
        if self.sigma_fwd == 0:
            out = np.asarray(self.model(Z @ self.L + self.mean), dtype=float)
            return out, np.zeros_like(out)
        draws = np.empty((n_draws, Z.shape[0]))
        for k in range(n_draws):
            X_w = self.s_fwd * Z + self.sigma_fwd * rng.standard_normal(Z.shape)
            draws[k] = np.asarray(self.model(X_w @ self.L + self.mean), dtype=float)
        mean = draws.mean(axis=0)
        var = draws.var(axis=0, ddof=1) / n_draws if n_draws > 1 else np.zeros_like(mean)
        return mean, var

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
        n = X.shape[0]
        ueifs_X, ueifs_Z = self._ueifs(X, y)

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
        return ExplanationResults(results, explainer_name=type(self).__name__, n=n)

    def _ueifs(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample UEIFs ``(ueifs_X, ueifs_Z)`` for ``X``.

        Draws the latent features, evaluates the Monte Carlo baseline
        ``eta(Z)`` and scores the counterfactuals.  Shared with
        :class:`Crossfitting`.
        """
        rng = np.random.default_rng([self.random_state, 12])
        Z = self._latent((X - self.mean) @ self.L_inv, rng)
        y_pred, base_var = self._eta(Z, rng, self.n_eta)
        ueifs_Z = self._phi_Z(Z, y_pred, y_true=y, base_var=base_var)
        return ueifs_Z @ (self.W ** 2).T, ueifs_Z

    def _phi_Z(
        self,
        Z: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        base_var: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sample uncentered UEIF in latent space.

        For each feature ``j`` the latent coordinate is replaced by
        independent draws, ``eta`` is evaluated at the counterfactuals, and the
        result is scored by ``self.method`` (see
        :meth:`Explainer._ueif_from_counterfactuals`).

        Parameters
        ----------
        Z : (n, d) array
            Latent features.
        y_pred : (n,) array
            Baseline predictions ``eta(Z)``.
        y_true : (n,) array or None
            Outcomes.  When provided, the loss-difference (DFI) form is used;
            when omitted, the label-free divergence form.
        base_var : (n,) array or None
            Monte Carlo variance of ``y_pred``; None when ``y_pred`` is exact.
        """
        n, d = Z.shape
        B = self.nsamples
        # A deterministic coupling needs a single draw.
        n_draws = self.n_eta if self.method == "cpi" and self.sigma_fwd > 0 else 1

        def shift(j: int) -> np.ndarray:
            """``Z~_j - Z_j`` for the B replicates, flattened; seeded per feature."""
            rng = np.random.default_rng([self.random_state, 13, j])
            if self.sampling_method == "resample":
                new = self.Z_full[rng.choice(self.Z_full.shape[0], size=(B, n)), j]
            elif self.sampling_method == "permutation":
                new = Z[np.array([rng.permutation(n) for _ in range(B)]), j]
            elif self.sampling_method == "normal":
                new = rng.normal(0.0, 1.0, size=(B, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            return (new - Z[None, :, j]).ravel()

        # A counterfactual differs from the observed latent point only in Z_j,
        # so its model input is mean + Z W + (Z~_j - Z_j) W[j] + sigma xi L.
        # The first two terms are shared by all features, and each coupling
        # draw's noise is shared across features (it stays independent across
        # replicates and points), so the d x d product runs once per draw.
        X_obs = np.tile(Z @ self.W + self.mean, (B, 1))
        sums = np.zeros((d, B * n))
        n_jobs = (os.cpu_count() or 1) if self.n_jobs == -1 else max(int(self.n_jobs), 1)
        with ThreadPoolExecutor(max_workers=min(n_jobs, d)) as pool:
            for k in range(n_draws):
                X_k = X_obs
                if self.sigma_fwd > 0:
                    rng = np.random.default_rng([self.random_state, 14, k])
                    X_k = X_obs + self.sigma_fwd * rng.standard_normal(X_obs.shape) @ self.L

                def predict(j: int, X_k: np.ndarray = X_k) -> np.ndarray:
                    X = X_k + np.outer(shift(j), self.W[j])
                    return np.asarray(self.model(X), dtype=float).reshape(-1)

                for j, f in enumerate(pool.map(predict, range(d))):
                    sums[j] += f

        return np.column_stack([
            self._ueif_from_counterfactuals(
                y_pred,
                (sums[j] / n_draws).reshape(B, n),
                method=self.method,
                y_true=y_true,
                base_var=base_var,
            )
            for j in range(d)
        ])

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
        """Decode Z to X with the barycentric projection E[X | Z] = Z @ W + mean."""
        return np.asarray(Z) @ self.W + self.mean
