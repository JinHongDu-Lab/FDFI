"""
Entropic optimal-transport (EOT) DFI explainer.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
from typing import Optional, Callable, Any, Tuple
from scipy.spatial.distance import pdist
from .base import Explainer
from ..losses import brier, squared_error
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
    regression ``eta(z) = E[f(X) | Z = z]`` is a Monte Carlo average of the
    model over draws from the backward conditional.  Counterfactuals replace
    ``Z_j`` by draws from its marginal and are scored by ``method``:

    - ``'cpi'``: one half the average per-replicate loss difference.  The
      baseline averages ``n_eta`` coupling draws.  For the squared error each
      replicate uses two draws, whose difference estimates its Monte Carlo
      variance, and the Monte Carlo bias is removed (see
      :meth:`Explainer._ueif_from_counterfactuals`); a feature costs
      ``2 * nsamples`` model evaluations per point.  Other losses use
      ``n_eta`` draws per replicate, so that the noise of the baseline and the
      replicates matches.
    - ``'scpi'``: the loss at the averaged counterfactual prediction minus the
      baseline loss.  Each replicate uses one joint draw of ``Z_j`` and the
      coupling noise, and the baseline averages the model over the same
      coupling draws without replacing ``Z_j``, so their noise cancels in the
      score up to an ``O(1/nsamples)`` bias.  A feature costs ``nsamples``
      model evaluations per point.

    Each coupling draw's noise
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
        If True, set ε from a median-distance heuristic in whitened space.  It
        depends on ``data`` only, not on ``random_state``.
    sampling_method : str, default='resample'
        How to draw counterfactual Z_j values:
        - 'resample': sample from the background Z pool
        - 'permutation': permute within the test set
        - 'normal': sample from N(0, 1)
    random_state : int, default=0
        Seed for the latent pool and the Monte Carlo draws.  Each call draws
        from a new stream spawned from it, so repeated calls (e.g. on chunks of
        a test set) are independent and the sequence of calls is reproducible.
    method : {'cpi', 'scpi'}, default='cpi'
        Importance estimator (see above).
    n_eta : int, default=10
        Coupling draws for the baseline ``eta`` of ``'cpi'``; at least 2 so
        that its Monte Carlo variance can be estimated.
    n_jobs : int, default=1
        Number of features scored in parallel threads; -1 uses all CPUs.
        Each thread holds a few arrays of ``nsamples * n_test * d`` floats,
        and ``model`` must be safe to call from several threads (scikit-learn
        ``predict`` is).  With ``n_jobs=1`` the model runs on the calling
        thread.  Results do not depend on ``n_jobs``.
    **kwargs : dict
        Extra arguments forwarded to the base Explainer.
    """
    #: Attributes shown in the printed representation.
    _REPR_PARAMS = Explainer._REPR_PARAMS + ("epsilon", "auto_epsilon", "n_eta")

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
        if self.method not in ("cpi", "scpi"):
            raise ValueError(f"method must be 'cpi' or 'scpi', got {method!r}")
        if n_eta < 2:
            raise ValueError(f"n_eta must be at least 2, got {n_eta}")
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.auto_epsilon = auto_epsilon
        self.sampling_method = sampling_method
        self._set_random_state(random_state)
        self.n_eta = n_eta
        self.n_jobs = n_jobs
        self.regularize = kwargs.get("regularize", 1e-6)

        # ── Gaussian whitening ───────────────────────────────────────────
        self._whiten(data)
        if self.auto_epsilon:
            self.epsilon = self._auto_epsilon(data - self.mean)

        # ── Gaussian EOT coupling ────────────────────────────────────────
        self.s_fwd = gaussian_eot_scale(self.epsilon)
        self.sigma_fwd = float(np.sqrt(1.0 - self.s_fwd ** 2))
        # Jacobian of the barycentric projection E[X | Z = z] = z @ W + mean.
        self.W = self.s_fwd * self.L

        # ── Latent pool for resampling and diagnostics ───────────────────
        self._set_pool(data)
        self._compute_diagnostics(report_title="EOTExplainer")

    def _set_pool(self, X: np.ndarray) -> None:
        rng = np.random.default_rng([self.random_state, 11])
        self.Z_full = self._latent((X - self.mean) @ self.L_inv, rng)

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
        if n_draws == 1:
            return mean, np.zeros_like(mean)
        return mean, draws.var(axis=0, ddof=1) / n_draws

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
        """Per-sample UEIFs ``(ueifs_X, ueifs_Z)`` for ``X``.

        Draws the latent features, evaluates the Monte Carlo baseline
        ``eta(Z)`` and scores the counterfactuals.  Shared with
        :class:`Crossfitting`.
        """
        key = self._call_key()
        Z = self._latent((X - self.mean) @ self.L_inv, np.random.default_rng([key, 12]))
        ueifs_Z = self._phi_Z(Z, key, y_true=y)
        return ueifs_Z @ (self.W ** 2).T, ueifs_Z

    def _phi_Z(
        self,
        Z: np.ndarray,
        key: int,
        y_true: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sample uncentered UEIF in latent space.

        For each feature ``j`` the latent coordinate is replaced by
        independent draws, the model is evaluated at the counterfactuals, and
        the result is scored by ``self.method`` (see
        :meth:`Explainer._ueif_from_counterfactuals`).

        - ``'cpi'``: the baseline is ``eta(Z)`` from ``n_eta`` coupling draws.
          Each replicate uses two coupling draws for the squared error, whose
          difference estimates and removes its Monte Carlo variance, and
          ``n_eta`` draws for other losses, matching the baseline.
        - ``'scpi'``: each replicate uses one coupling draw, and the baseline
          averages the model over the same draws without replacing ``Z_j``,
          so their Monte Carlo noise cancels in the score.

        Parameters
        ----------
        Z : (n, d) array
            Latent features.
        key : int
            Seed of this call's Monte Carlo streams (see :meth:`_call_key`).
        y_true : (n,) array or None
            Outcomes.  When provided, the loss-difference (DFI) form is used;
            when omitted, the label-free divergence form.
        """
        n, d = Z.shape
        B = self.nsamples
        noisy = self.sigma_fwd > 0
        coupled = self.method == "scpi" and noisy
        y_pred = base_var = None
        if not coupled:
            rng = np.random.default_rng([key, 15])
            y_pred, base_var = self._eta(Z, rng, self.n_eta)
        # A deterministic coupling needs a single draw.
        n_draws = 1
        if self.method == "cpi" and noisy:
            n_draws = 2 if self._loss in (squared_error, brier) else self.n_eta

        def shift(j: int) -> np.ndarray:
            """``Z~_j - Z_j`` for the B replicates, flattened; seeded per feature."""
            rng = np.random.default_rng([key, 13, j])
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
        # Monte Carlo variance of each replicate's two-draw mean, (f1 - f2)^2 / 4,
        # averaged over replicates.
        rep_var = np.zeros((d, n)) if n_draws == 2 else None
        n_jobs = int(self.n_jobs)
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        n_jobs = min(max(n_jobs, 1), d)
        with ThreadPoolExecutor(n_jobs) if n_jobs > 1 else nullcontext() as pool:
            pmap = map if pool is None else pool.map
            for k in range(n_draws):
                X_k = X_obs
                if noisy:
                    draw = np.random.default_rng([key, 14, k])
                    xi = draw.standard_normal(X_obs.shape)
                    X_k = X_obs + self.sigma_fwd * xi @ self.L
                if coupled:
                    y_pred = np.asarray(self.model(X_k), dtype=float)
                    y_pred = y_pred.reshape(B, n).mean(axis=0)

                def predict(j: int, X_k: np.ndarray = X_k) -> np.ndarray:
                    X = X_k + np.outer(shift(j), self.W[j])
                    return np.asarray(self.model(X), dtype=float).reshape(-1)

                for j, f in enumerate(pmap(predict, range(d))):
                    if k == 1 and rep_var is not None:
                        rep_var[j] = ((sums[j] - f) ** 2).reshape(B, n).mean(axis=0) / 4
                    sums[j] += f

        return np.column_stack(
            [
                self._ueif_from_counterfactuals(
                    y_pred,
                    (sums[j] / n_draws).reshape(B, n),
                    method=self.method,
                    y_true=y_true,
                    base_var=base_var,
                    rep_var=None if rep_var is None else rep_var[j],
                )
                for j in range(d)
            ]
        )

    def _auto_epsilon(self, X_centered: np.ndarray) -> float:
        """
        Auto-tune epsilon from latent geometry.

        Estimates pairwise distances in Gaussian-whitened latent space and
        uses a conservative shrinkage factor to avoid over-smoothing.  The
        subsample of at most 1000 rows has a fixed seed, so epsilon depends on
        the data only and explainers that differ in ``random_state`` (such as
        :class:`Crossfitting` folds) share it.
        """
        if X_centered.shape[0] < 2:
            return self.epsilon

        Z_ref = X_centered @ self.L_inv
        n = min(1000, Z_ref.shape[0])
        rng = np.random.default_rng(0)
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
