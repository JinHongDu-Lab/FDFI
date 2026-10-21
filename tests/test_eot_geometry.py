"""
Tests for the Gaussian EOT geometry, latent draws, and the integrated LOCO
estimator in :mod:`fdfi.explainers.eot`.
"""

import numpy as np
import pytest

from fdfi.explainers import EOTExplainer
from fdfi.explainers.eot import (
    fit_semicontinuous_potentials,
    gaussian_eot_scale,
)


class TestGaussianEOTScale:
    @pytest.mark.parametrize("eps", [1e-3, 0.1, 1.0, 5.0])
    def test_solves_quadratic_and_lies_in_unit_interval(self, eps):
        s = gaussian_eot_scale(eps)
        assert 0.0 < s < 1.0
        assert abs(2 * s ** 2 + eps * s - 2) < 1e-12

    def test_limits(self):
        assert gaussian_eot_scale(0.0) == 1.0
        assert gaussian_eot_scale(1e-9) == pytest.approx(1.0, abs=1e-6)
        assert gaussian_eot_scale(1e6) < 1e-5

    def test_minimizes_gaussian_eot_objective(self):
        eps = 1.0
        s_grid = np.linspace(0.01, 0.999, 20000)
        objective = 2 * (1 - s_grid) - (eps / 2) * np.log(1 - s_grid ** 2)
        assert abs(s_grid[np.argmin(objective)] - gaussian_eot_scale(eps)) < 1e-3


class TestLatentDraws:
    def test_latent_draws_have_unit_covariance(self):
        rng = np.random.default_rng(0)
        d = 4
        A = rng.standard_normal((d, d))
        X = rng.standard_normal((20000, d)) @ A.T
        explainer = EOTExplainer(lambda x: x[:, 0], X, epsilon=1.0, nsamples=2, random_state=1)
        cov = np.cov(explainer.Z_full, rowvar=False)
        assert np.allclose(cov, np.eye(d), atol=0.05)

    def test_barycentric_latent_has_shrunken_variance(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20000, 3))
        explainer = EOTExplainer(lambda x: x[:, 0], X, epsilon=1.0, latent="barycentric", nsamples=2)
        var = np.var(explainer.Z_full, axis=0)
        assert np.allclose(var, explainer.s_fwd ** 2, atol=0.05)
        assert explainer.s_fwd ** 2 < 0.7

    def test_eta_mc_matches_closed_form_for_linear_model(self):
        rng = np.random.default_rng(0)
        d = 3
        X = rng.standard_normal((5000, d))
        beta = np.array([2.0, -1.0, 0.5])
        explainer = EOTExplainer(lambda x: x @ beta, X, epsilon=1.0, n_eta=400, nsamples=2)
        Z = rng.standard_normal((50, d))
        eta = explainer._eta(Z, np.random.default_rng(3))
        expected = explainer.s_fwd * (Z @ explainer.L) @ beta + float((explainer.mean @ beta)[0])
        assert np.mean(np.abs(eta - expected)) < 0.1


class TestIntegratedLOCO:
    def test_loco_scoring_removes_monte_carlo_bias(self):
        rng = np.random.default_rng(0)
        explainer = EOTExplainer(lambda x: x[:, 0], rng.standard_normal((50, 2)), nsamples=2)
        n, B = 200000, 5
        y = rng.standard_normal(n)
        y_pred = np.zeros(n)
        # counterfactual predictions with mean 0 and variance 1 per sample
        y_tilde = rng.standard_normal((B, n))
        loco = explainer._ueif_from_counterfactuals(y_pred, y_tilde, method="loco", y_true=y)
        scpi = explainer._ueif_from_counterfactuals(y_pred, y_tilde, method="scpi", y_true=y)
        # true population value: E[(Y - 0)^2] - E[(Y - 0)^2] = 0
        assert abs(loco.mean()) < 0.02
        assert abs(scpi.mean() - 1.0 / B) < 0.02

    def test_loco_recovers_latent_importance_for_linear_gaussian_model(self):
        rng = np.random.default_rng(1)
        d = 3
        Sigma = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])
        beta = np.array([3.0, 0.0, 1.0])
        X_bg = rng.multivariate_normal(np.zeros(d), Sigma, size=4000)
        X = rng.multivariate_normal(np.zeros(d), Sigma, size=4000)
        y = X @ beta + rng.standard_normal(X.shape[0])
        eps = 0.5
        explainer = EOTExplainer(lambda x: x @ beta, X_bg, epsilon=eps, nsamples=30, n_eta=20, random_state=2)
        res = explainer(X, y)
        s = gaussian_eot_scale(eps)
        w, V = np.linalg.eigh(Sigma)
        B = V @ np.diag(np.sqrt(w)) @ V.T
        # eta(z) = s * beta^T B z, so phi_{Z_j} = s^2 (B beta)_j^2
        expected = s ** 2 * (B @ beta) ** 2
        assert np.allclose(res["phi_Z"], expected, rtol=0.2, atol=0.3)
        # with weights (s L_{lj})^2 the attributed total is s^2 times the latent total
        assert res["phi_X"].sum() == pytest.approx(s ** 2 * res["phi_Z"].sum(), rel=0.05)

    def test_cpi_has_first_order_bias_under_perturbed_regression(self):
        rng = np.random.default_rng(4)
        d = 2
        X_bg = rng.standard_normal((3000, d))
        X = rng.standard_normal((3000, d))
        y = 2.0 * X[:, 0] + rng.standard_normal(X.shape[0])
        t = 0.3
        # eta_hat = eta + t * Delta_1 for the latent coordinate 1, with eps = 0 (identity map)
        model_true = lambda x: 2.0 * x[:, 0]
        model_biased = lambda x: (1 + t) * 2.0 * x[:, 0]
        kw = dict(epsilon=0.0, nsamples=40, random_state=5)
        cpi_true = EOTExplainer(model_true, X_bg, method="cpi", **kw)(X, y)["phi_Z"][0]
        cpi_bias = EOTExplainer(model_biased, X_bg, method="cpi", **kw)(X, y)["phi_Z"][0]
        loco_bias = EOTExplainer(model_biased, X_bg, method="loco", **kw)(X, y)["phi_Z"][0]
        phi = 4.0
        assert cpi_true == pytest.approx(phi, abs=0.4)
        assert cpi_bias - phi == pytest.approx(t * phi, abs=0.4)
        assert loco_bias - phi == pytest.approx(-t ** 2 * phi, abs=0.4)


class TestEmpiricalSource:
    def test_source_marginal_constraint(self):
        rng = np.random.default_rng(0)
        X_w = rng.standard_normal((60, 2))
        eps = 1.0
        f = fit_semicontinuous_potentials(X_w, eps, n_iter=400, random_state=0)
        z = rng.standard_normal((40000, 2))
        cost = np.sum(X_w ** 2, axis=1)[:, None] + np.sum(z ** 2, axis=1)[None, :] - 2 * X_w @ z.T
        logits = (f[:, None] - cost) / eps
        logits -= logits.max(axis=0, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=0, keepdims=True)
        row_mass = p.mean(axis=1)
        assert np.allclose(row_mass, 1.0 / X_w.shape[0], atol=0.004)

    def test_empirical_source_runs_and_matches_gaussian_for_gaussian_data(self):
        rng = np.random.default_rng(0)
        d = 2
        X_bg = rng.standard_normal((300, d))
        X = rng.standard_normal((300, d))
        beta = np.array([2.0, 0.0])
        y = X @ beta + 0.1 * rng.standard_normal(300)
        emp = EOTExplainer(lambda x: x @ beta, X_bg, epsilon=1.0, source="empirical", nsamples=20, random_state=0)
        gau = EOTExplainer(lambda x: x @ beta, X_bg, epsilon=1.0, nsamples=20, n_eta=20, random_state=0)
        r_emp = emp(X, y)
        r_gau = gau(X, y)
        assert np.all(np.isfinite(r_emp["phi_Z"]))
        assert r_emp["phi_Z"][0] > 3 * max(r_emp["phi_Z"][1], 1e-3)
        assert r_emp["phi_Z"][0] == pytest.approx(r_gau["phi_Z"][0], rel=0.35)
