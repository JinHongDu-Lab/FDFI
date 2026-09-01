"""
Entropic optimal-transport (EOT) DFI explainer.
"""

import numpy as np
from typing import Optional, Callable, Any
from scipy.spatial.distance import pdist
from .base import Explainer


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

    Feature importance is measured from counterfactual resamples in latent
    space. CPI uses one half the average per-resample loss difference; SCPI
    applies the loss after averaging the counterfactual predictions.

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
        Resampling estimator. CPI averages per-resample loss differences and
        multiplies by one half; SCPI averages predictions before applying the
        loss. These match the definitions in the FDFI paper.
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

        CPI uses ``0.5 * E_b[L(y_i, ŷ_{b,i}) - L(y_i, ŷ_i)]``. SCPI uses
        ``L(y_i, E_b[ŷ_{b,i}]) - L(y_i, ŷ_i)``. With squared-error loss,
        exact latent independence, and infinitely many resamples, their
        population targets agree.

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
