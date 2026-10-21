"""
Entropic optimal-transport (EOT) DFI explainer.
"""

import numpy as np
from typing import Optional, Callable, Any
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
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


def fit_semicontinuous_potentials(
    X_w: np.ndarray,
    epsilon: float,
    n_iter: int = 300,
    tol: float = 1e-6,
    random_state: int = 0,
    mc_batch: int = 2048,
    step_size: float = 1.0,
) -> np.ndarray:
    """Source Schrödinger potentials for an empirical source and N(0, I) target.

    Solves the semi-dual EOT problem between ``(1/n) sum_i delta_{x_i}`` and
    ``N(0, I)`` with cost ``||x - z||^2`` by a stochastic fixed-point
    iteration on the source-marginal constraints
    ``E_Z[p_i(Z)] = 1/n`` with ``p_i(z) ∝ exp((f_i - ||x_i - z||^2) / epsilon)``.
    Each step uses fresh Monte Carlo draws from the target, so the step size
    decays as ``t^{-0.6}`` and the returned potentials are the average of the
    iterates over the second half of the run (Polyak averaging).  The loop
    stops early when an exponential moving average of the constraint
    violation falls below ``tol``.

    Returns the potentials ``f`` (shape ``(n,)``), gauge-fixed to mean zero.
    """
    n, d = X_w.shape
    if epsilon <= 0:
        return np.zeros(n)
    rng = np.random.default_rng([random_state, 21])
    f = np.zeros(n)
    f_avg = np.zeros(n)
    n_avg = 0
    target_mass = 1.0 / n
    x_sq = np.sum(X_w ** 2, axis=1, keepdims=True)
    ema = None
    for t in range(n_iter):
        z = rng.standard_normal((mc_batch, d))
        cost = x_sq + np.sum(z ** 2, axis=1)[None, :] - 2.0 * X_w @ z.T
        logits = (f[:, None] - np.maximum(cost, 0.0)) / epsilon
        logits -= logsumexp(logits, axis=0, keepdims=True)
        row_mass = np.exp(logits).mean(axis=1)
        grad = target_mass - row_mass
        lr = step_size / (1.0 + t) ** 0.6
        f += lr * epsilon * grad * n
        f -= f.mean()
        if t >= n_iter // 2:
            f_avg += f
            n_avg += 1
        viol = np.max(np.abs(grad)) * n
        ema = viol if ema is None else 0.9 * ema + 0.1 * viol
        if ema < tol and n_avg > 0:
            break
    return f_avg / n_avg if n_avg > 0 else f


class EOTExplainer(Explainer):
    """
    Entropic optimal-transport DFI explainer.

    The covariates are whitened with the background covariance and coupled
    with the reference law ``N(0, I)`` by entropic optimal transport with
    quadratic cost and regularization ``epsilon``.

    ``source="gaussian"`` (default) treats the whitened source as Gaussian, so
    the coupling has the closed form

        Z | X_w = x ~ N(s_eps x, (1 - s_eps^2) I),
        s_eps = (sqrt(eps^2 + 16) - eps) / 4,

    see :func:`gaussian_eot_scale`.  ``source="empirical"`` fits the
    semi-discrete Sinkhorn potentials of the empirical background source and
    uses the resulting discrete backward posterior.

    Latent features are drawn from the fitted forward conditional
    (``latent="draw"``), which matches the construction analysed in the DFI
    paper; ``latent="barycentric"`` uses the deterministic mean
    ``s_eps X_w`` instead, which has latent variance ``s_eps^2 < 1`` and is
    kept only as an approximation.

    The latent regression function ``eta(z) = E[Y | Z = z]`` is estimated by
    averaging the model under the backward kernel (``eta="mc"`` with ``n_eta``
    Monte Carlo draws for the Gaussian source, or the softmax-weighted average
    over background points for the empirical source); ``eta="plugin"``
    evaluates the model at the backward barycentric mean, which coincides
    with the kernel average only for linear models.

    Backward attribution uses the best linear projection
    ``E[X_w | Z] ≈ M_w Z`` computed from the coupling moments, giving the
    weight matrix ``W = L @ M_w`` used in ``phi_X_j = sum_k W[j,k]^2 phi_Z_k``.

    Feature importance is measured from counterfactual resamples in latent
    space: ``method="loco"`` (default) is the integrated LOCO estimator with
    the Monte Carlo variance correction, ``"scpi"`` the same without the
    correction, and ``"cpi"`` one half the average per-resample loss
    difference, whose regression bias is first order.

    Parameters
    ----------
    model : callable
        The model to explain.  Takes (n, d) array, returns (n,) predictions.
    data : numpy.ndarray
        Background data for whitening, potentials, and resampling.  Shape (n, d).
    nsamples : int, default=50
        Number of Monte Carlo samples per feature for counterfactual
        resampling.
    epsilon : float, default=0.1
        EOT regularization parameter.
    auto_epsilon : bool, default=False
        If True, set ``epsilon`` from a median-distance heuristic in whitened space.
    sampling_method : str, default='resample'
        How to draw counterfactual ``Z_j`` values: 'resample' from the
        background latent pool, 'permutation' within the test set, or
        'normal' from N(0, 1).
    random_state : int, default=0
        Random seed.
    method : {'loco', 'scpi', 'cpi'}, default='loco'
        Importance estimator applied to the counterfactual predictions.
    latent : {'draw', 'barycentric'}, default='draw'
        Latent construction.
    eta : {'mc', 'plugin'}, default='mc'
        Latent regression estimator (Gaussian source only; the empirical source
        always uses the kernel average over background points).
    n_eta : int, default=10
        Monte Carlo draws for ``eta="mc"``.
    source : {'gaussian', 'empirical'}, default='gaussian'
        Source law used for the coupling.
    n_pool : int, default=2000
        Size of the N(0, I) target pool for the empirical source.
    n_eot_iter, eot_tol : int, float
        Iterations and tolerance for the potential fixed-point iteration.
    **kwargs : dict
        Extra arguments forwarded to the base Explainer.
    """
    #: Attributes shown in the printed representation.
    _REPR_PARAMS = Explainer._REPR_PARAMS + ("latent", "eta", "n_eta", "source")

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        epsilon: float = 0.1,
        auto_epsilon: bool = False,
        sampling_method: str = "resample",
        random_state: Optional[int] = 0,
        method: str = "loco",
        latent: str = "draw",
        eta: str = "mc",
        n_eta: int = 10,
        source: str = "gaussian",
        n_pool: int = 2000,
        n_eot_iter: int = 300,
        eot_tol: float = 1e-6,
        **kwargs: Any
    ):
        super().__init__(model, data, fit_flow=False, **kwargs)
        if latent not in ("draw", "barycentric"):
            raise ValueError(f"latent must be 'draw' or 'barycentric', got {latent!r}")
        if eta not in ("mc", "plugin"):
            raise ValueError(f"eta must be 'mc' or 'plugin', got {eta!r}")
        if source not in ("gaussian", "empirical"):
            raise ValueError(f"source must be 'gaussian' or 'empirical', got {source!r}")
        if str(method).lower() not in ("loco", "scpi", "cpi"):
            raise ValueError(f"method must be 'loco', 'scpi', or 'cpi', got {method!r}")
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.auto_epsilon = auto_epsilon
        self.sampling_method = sampling_method
        self.random_state = random_state if random_state is not None else 0
        self.method = method
        self.latent = latent
        self.eta = eta
        self.n_eta = n_eta
        self.source = source
        self.n_pool = n_pool
        self.n_eot_iter = n_eot_iter
        self.eot_tol = eot_tol
        self.regularize = kwargs.get("regularize", 1e-6)
        self._rng = np.random.default_rng([self.random_state, 11])

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
        self.X_w_bg = X_whitened

        if self.auto_epsilon:
            self.epsilon = self._auto_epsilon(X_centered)

        # ── Gaussian EOT closed form ─────────────────────────────────────
        self.s_fwd = gaussian_eot_scale(self.epsilon)
        self.sigma_fwd = float(np.sqrt(max(1.0 - self.s_fwd ** 2, 0.0)))
        # Backward barycentric mean of the Gaussian coupling: E[X_c | Z = z] = z @ L_z.
        self.L_z = self.s_fwd * self.L

        n, d = X_whitened.shape
        if self.source == "empirical" and self.epsilon > 0:
            self.f_potential = fit_semicontinuous_potentials(
                X_whitened, self.epsilon, n_iter=self.n_eot_iter,
                tol=self.eot_tol, random_state=self.random_state,
            )
            self.z_pool = self._rng.standard_normal((max(self.n_pool, n), d))
            self.g_pool = self._target_potential(self.z_pool)
            self.mu_bg = np.asarray(self.model(data), dtype=float)
            # Coupling moments over the pool for the backward projection,
            # accumulated in chunks so that no (n x m) matrix is stored.
            EZ_given_x = np.vstack([P @ self.z_pool for P in self._forward_probs_chunks(X_whitened)])
            Ezx = (EZ_given_x.T @ X_whitened) / n          # E[Z X_w^T]
            Ezz = np.eye(d)                                # target marginal
        else:
            self.f_potential = None
            Sigma_hat = (X_whitened.T @ X_whitened) / n
            c = self.s_fwd
            Ezz = (1.0 - c ** 2) * np.eye(d) + c ** 2 * Sigma_hat
            Ezx = c * Sigma_hat
        M_w = np.linalg.solve(Ezz, Ezx)
        self.W = self.L @ M_w

        # ── Latent pool for resampling and diagnostics ───────────────────
        self.Z_full = self._latent(X_whitened, self._rng)
        self._compute_diagnostics(report_title="EOTExplainer")

    # ------------------------------------------------------------------ #
    # Coupling helpers
    # ------------------------------------------------------------------ #
    _chunk_elements = int(2e7)

    @staticmethod
    def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise squared distances between the rows of ``a`` and ``b``."""
        d2 = (np.sum(a ** 2, axis=1)[:, None] + np.sum(b ** 2, axis=1)[None, :]
              - 2.0 * a @ b.T)
        return np.maximum(d2, 0.0)

    def _row_chunks(self, n_rows: int, n_cols: int):
        step = max(1, self._chunk_elements // max(n_cols, 1))
        for a in range(0, n_rows, step):
            yield a, min(a + step, n_rows)

    def _target_potential(self, z: np.ndarray) -> np.ndarray:
        """g(z) = -eps log mean_i exp((f_i - ||x_i - z||^2) / eps), computed in chunks of z."""
        out = np.empty(z.shape[0])
        n_bg = self.X_w_bg.shape[0]
        for a, b in self._row_chunks(z.shape[0], n_bg):
            logits = (self.f_potential[None, :] - self._sq_dist(z[a:b], self.X_w_bg)) / self.epsilon
            out[a:b] = -self.epsilon * (logsumexp(logits, axis=1) - np.log(n_bg))
        return out

    def _forward_probs_chunks(self, x_w: np.ndarray):
        """Yield row-chunks of the forward conditional over the target pool (rows sum to one)."""
        m = self.z_pool.shape[0]
        for a, b in self._row_chunks(x_w.shape[0], m):
            logits = (self.g_pool[None, :] - self._sq_dist(x_w[a:b], self.z_pool)) / self.epsilon
            logits -= logsumexp(logits, axis=1, keepdims=True)
            yield np.exp(logits)

    def _forward_probs(self, x_w: np.ndarray) -> np.ndarray:
        """Forward conditional over the target pool, rows sum to one (materialized)."""
        return np.vstack(list(self._forward_probs_chunks(x_w)))

    def _latent(self, x_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Latent features for whitened inputs."""
        if self.source == "empirical" and self.epsilon > 0:
            out = np.empty((x_w.shape[0], self.z_pool.shape[1]))
            start = 0
            for P in self._forward_probs_chunks(x_w):
                b = start + P.shape[0]
                if self.latent == "barycentric":
                    out[start:b] = P @ self.z_pool
                else:
                    cum = np.cumsum(P, axis=1)
                    u = rng.random((P.shape[0], 1))
                    idx = np.minimum((cum < u).sum(axis=1), self.z_pool.shape[0] - 1)
                    out[start:b] = self.z_pool[idx]
                start = b
            return out
        Z = self.s_fwd * x_w
        if self.latent == "draw" and self.sigma_fwd > 0:
            Z = Z + self.sigma_fwd * rng.standard_normal(x_w.shape)
        return Z

    def _eta(
        self,
        Z: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        return_var: bool = False,
        n_draws: Optional[int] = None,
    ):
        """Estimated latent regression function eta(z) = E[Y | Z = z].

        ``n_draws`` overrides ``self.n_eta`` (the number of coupling draws used by
        ``eta="mc"``); the counterfactual evaluation in :meth:`_phi_Z` uses one
        draw per replicate.

        With ``return_var=True`` also returns the Monte Carlo variance of the
        estimate at each point (``Var_xi[model(...)] / n_eta`` for ``eta="mc"``;
        zero for the deterministic variants), which is used to debias the
        squared baseline residual in the integrated LOCO estimator.
        """
        Z = np.asarray(Z)
        if self.source == "empirical" and self.epsilon > 0:
            out = np.empty(Z.shape[0])
            n_bg = self.X_w_bg.shape[0]
            for a, b in self._row_chunks(Z.shape[0], n_bg):
                logits = (self.f_potential[None, :] - self._sq_dist(Z[a:b], self.X_w_bg)) / self.epsilon
                logits -= logsumexp(logits, axis=1, keepdims=True)
                out[a:b] = np.exp(logits) @ self.mu_bg
            return (out, np.zeros_like(out)) if return_var else out
        if self.eta == "plugin" or self.sigma_fwd == 0:
            out = np.asarray(self.model(Z @ self.L_z + self.mean), dtype=float)
            return (out, np.zeros_like(out)) if return_var else out
        rng = self._rng if rng is None else rng
        n_draws = self.n_eta if n_draws is None else n_draws
        draws = np.empty((n_draws, Z.shape[0]))
        for k in range(n_draws):
            xi = rng.standard_normal(Z.shape)
            X_w = self.s_fwd * Z + self.sigma_fwd * xi
            draws[k] = np.asarray(self.model(X_w @ self.L + self.mean), dtype=float)
        mean = draws.mean(axis=0)
        if not return_var:
            return mean
        var = draws.var(axis=0, ddof=1) / n_draws if n_draws > 1 else np.zeros_like(mean)
        return mean, var

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #
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
        n, d = X.shape
        rng = np.random.default_rng([self.random_state, 12])
        X_w = (X - self.mean) @ self.L_inv
        Z = self._latent(X_w, rng)
        y_pred, base_var = self._eta(Z, rng, return_var=True)

        ueifs_Z = self._phi_Z(Z, y_pred, y_true=y, base_var=base_var)
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
        return ExplanationResults(results, explainer_name=type(self).__name__, n=n)

    def _phi_Z(
        self,
        Z: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        base_var: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-sample uncentered influence function in latent space.

        For each feature ``j`` the latent coordinate is replaced by independent
        draws from the reference marginal and the fitted latent regression
        ``eta`` is evaluated at the counterfactuals; the result is scored by
        ``self.method`` (see :meth:`Explainer._ueif_from_counterfactuals`).

        Parameters
        ----------
        Z : (n, d) array
            Latent features.
        y_pred : (n,) array
            Fitted latent regression ``eta(Z)`` at the observed latent features.
        y_true : (n,) array or None
            Outcomes.  When provided, the loss-difference (DFI) form is used.
        base_var : (n,) array or None
            Monte Carlo variance of ``y_pred`` (see :meth:`_eta`), subtracted
            from the squared baseline residual by ``method="loco"``.
        """
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        if base_var is None:
            _, base_var = self._eta(Z, np.random.default_rng([self.random_state, 14]), return_var=True)

        for j in range(d):
            rng = np.random.default_rng([self.random_state, 13, j])
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
            # "loco" and "scpi" only use the average over replicates, so each
            # replicate gets a single coupling draw: the joint draws of (Z_j, xi)
            # estimate the same integrated regression with nsamples evaluations
            # per point instead of nsamples * n_eta, and the Monte Carlo
            # correction in _ueif_from_counterfactuals accounts for the noise.
            # "cpi" scores each replicate separately and keeps n_eta draws.
            n_draws = None if self.method == "cpi" else 1
            y_tilde_all = self._eta(Z_flat, rng, n_draws=n_draws).reshape(self.nsamples, n)
            ueifs_Z[:, j] = self._ueif_from_counterfactuals(
                y_pred, y_tilde_all, method=self.method, y_true=y_true, base_var=base_var
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
        """Decode Z to X using the backward barycentric mean E[X | Z] = Z @ L_z + mean."""
        return np.asarray(Z) @ self.L_z + self.mean
