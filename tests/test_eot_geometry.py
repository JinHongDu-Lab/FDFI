"""
Tests for the Gaussian EOT coupling, latent draws, and the Monte Carlo
correction of the CPI score in :mod:`fdfi.explainers.eot`.
"""

import numpy as np
import pytest

from fdfi.explainers import EOTExplainer
from fdfi.explainers.eot import gaussian_eot_scale


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


class TestCoupling:
    def test_latent_draws_have_unit_covariance(self):
        rng = np.random.default_rng(0)
        d = 4
        A = rng.standard_normal((d, d))
        X = rng.standard_normal((20000, d)) @ A.T
        explainer = EOTExplainer(lambda x: x[:, 0], X, epsilon=1.0, nsamples=2, random_state=1)
        cov = np.cov(explainer.Z_full, rowvar=False)
        assert np.allclose(cov, np.eye(d), atol=0.05)

    def test_eta_matches_closed_form_for_linear_model(self):
        rng = np.random.default_rng(0)
        d = 3
        X = rng.standard_normal((5000, d))
        beta = np.array([2.0, -1.0, 0.5])
        explainer = EOTExplainer(lambda x: x @ beta, X, epsilon=1.0, nsamples=2)
        Z = rng.standard_normal((50, d))
        eta, var = explainer._eta(Z, np.random.default_rng(3), 400)
        expected = explainer.s_fwd * (Z @ explainer.L) @ beta + float((explainer.mean @ beta)[0])
        assert np.mean(np.abs(eta - expected)) < 0.1
        # Var_xi[f] / n_draws = (1 - s^2) |L beta|^2 / n_draws
        expected_var = (1 - explainer.s_fwd ** 2) * np.sum((explainer.L @ beta) ** 2) / 400
        assert np.mean(var) == pytest.approx(expected_var, rel=0.1)

    @pytest.mark.parametrize("method", ["cpi", "scpi"])
    def test_results_do_not_depend_on_n_jobs(self, method):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 5))
        y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + rng.standard_normal(300)
        model = lambda x: np.sin(x[:, 0]) + x[:, 1] * x[:, 2]
        kw = dict(epsilon=1.0, nsamples=10, method=method, compute_diagnostics=False)
        r1 = EOTExplainer(model, X, n_jobs=1, **kw)(X, y)
        r4 = EOTExplainer(model, X, n_jobs=4, **kw)(X, y)
        assert np.array_equal(r1["phi_Z"], r4["phi_Z"])
        assert np.array_equal(r1["se_X"], r4["se_X"])

    def test_model_runs_on_calling_thread_with_one_job(self):
        import threading

        X = np.random.default_rng(0).standard_normal((40, 3))
        threads = set()

        def model(x):
            threads.add(threading.get_ident())
            return x[:, 0]

        EOTExplainer(model, X, nsamples=2, compute_diagnostics=False)(X)
        assert threads == {threading.get_ident()}

    def test_options_are_validated(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        assert EOTExplainer(lambda x: x[:, 0], X, method="SCPI").method == "scpi"
        with pytest.raises(ValueError):
            EOTExplainer(lambda x: x[:, 0], X, method="loco")
        with pytest.raises(ValueError):
            EOTExplainer(lambda x: x[:, 0], X, n_eta=1)
        ex = EOTExplainer(lambda x: x[:, 0], X, method="scpi", loss="l1", nsamples=2)
        res = ex(X)
        assert np.all(np.isfinite(res["phi_Z"]))


class TestMonteCarloCorrections:
    """Null latent features score zero although eta is a Monte Carlo average."""

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4000, 3))
        y = X[:, 0] + rng.standard_normal(X.shape[0])
        return X, y

    @pytest.mark.parametrize("method", ["cpi", "scpi"])
    @pytest.mark.parametrize("labelled", [True, False])
    def test_null_features_are_unbiased(self, data, method, labelled):
        X, y = data
        explainer = EOTExplainer(
            lambda x: x[:, 0], X, epsilon=1.0, nsamples=20, n_eta=10,
            method=method, random_state=1, compute_diagnostics=False,
        )
        res = explainer(X, y if labelled else None)
        # Without the corrections (CPI) or the shared coupling draws (SCPI)
        # the null features are off by a multiple of (1 - s^2) / n_eta = 0.04.
        assert np.all(np.abs(res["phi_Z"][1:]) < 0.01)
        s2 = explainer.s_fwd ** 2
        assert res["phi_Z"][0] == pytest.approx(s2, abs=0.06)

    @pytest.mark.parametrize("labelled", [True, False])
    def test_cpi_removes_replicate_monte_carlo_variance(self, labelled):
        rng = np.random.default_rng(0)
        explainer = EOTExplainer(lambda x: x[:, 0], rng.standard_normal((50, 2)), nsamples=2)
        n, B = 200000, 5
        y = rng.standard_normal(n) if labelled else None
        y_tilde = rng.standard_normal((B, n))      # null feature, variance 1
        cpi = explainer._ueif_from_counterfactuals(
            np.zeros(n), y_tilde, method="cpi", y_true=y, rep_var=np.ones(n)
        )
        assert abs(cpi.mean()) < 0.01

    def test_scpi_keeps_counterfactual_monte_carlo_bias(self):
        rng = np.random.default_rng(0)
        explainer = EOTExplainer(lambda x: x[:, 0], rng.standard_normal((50, 2)), nsamples=2)
        n, B = 200000, 5
        y = rng.standard_normal(n)
        y_pred = np.zeros(n)
        y_tilde = rng.standard_normal((B, n))
        scpi = explainer._ueif_from_counterfactuals(y_pred, y_tilde, method="scpi", y_true=y)
        assert abs(scpi.mean() - 1.0 / B) < 0.02


class TestAttribution:
    def test_linear_gaussian_model(self):
        rng = np.random.default_rng(1)
        d = 3
        Sigma = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])
        beta = np.array([3.0, 0.0, 1.0])
        X_bg = rng.multivariate_normal(np.zeros(d), Sigma, size=4000)
        X = rng.multivariate_normal(np.zeros(d), Sigma, size=4000)
        y = X @ beta + rng.standard_normal(X.shape[0])
        eps = 0.5
        explainer = EOTExplainer(
            lambda x: x @ beta, X_bg, epsilon=eps, nsamples=30, random_state=2,
        )
        res = explainer(X, y)
        s = gaussian_eot_scale(eps)
        w, V = np.linalg.eigh(Sigma)
        B = V @ np.diag(np.sqrt(w)) @ V.T
        # eta(z) = s beta^T B z, so phi_{Z_j} = s^2 (B beta)_j^2
        assert np.allclose(res["phi_Z"], s ** 2 * (B @ beta) ** 2, rtol=0.2, atol=0.3)
        # With weights (s B_lj)^2 the attributed total is s^2 times the latent
        # total for standardized features.
        assert res["phi_X"].sum() == pytest.approx(s ** 2 * res["phi_Z"].sum(), rel=0.05)


class TestOrthogonalizedCPI:
    def test_cpi_is_orthogonal_under_misspecification(self):
        """With outcomes and the squared error, CPI carries the orthogonalizing
        term and estimates b_hat_j (2 b_j - b_hat_j) for a linear model fitted
        with coefficients b_hat on independent Gaussian X.  SCPI needs no
        orthogonalizing term and exceeds it by the Monte Carlo variance
        b_hat_j^2 / B."""
        rng = np.random.default_rng(1)
        n, d = 6000, 4
        X = rng.standard_normal((n, d))
        b = np.array([1.0, 0.5, 0.0, 2.0])
        y = X @ b + rng.standard_normal(n)
        b_hat = b + np.array([0.3, -0.3, 0.3, 0.0])
        phi = {}
        B = 50
        for m in ("cpi", "scpi"):
            ex = EOTExplainer(lambda Xo: Xo @ b_hat, X, epsilon=0.0, nsamples=B,
                              random_state=3, method=m, compute_diagnostics=False)
            phi[m] = np.asarray(ex(X, y)["phi_Z"])
        target = b_hat * (2 * b - b_hat)
        assert np.allclose(phi["cpi"], target, rtol=0.05, atol=0.05)
        assert np.allclose(phi["scpi"], phi["cpi"] + b_hat ** 2 / B, atol=0.03)


class TestCrossfitting:
    def test_crossfitting_matches_explainer_construction(self):
        from fdfi.explainers import Crossfitting

        rng = np.random.default_rng(2)
        X = rng.standard_normal((600, 3))
        y = X[:, 0] + rng.standard_normal(600)
        cf = Crossfitting(
            lambda x: x[:, 0], X, y=y, explainer_class=EOTExplainer, cv=2,
            epsilon=1.0, nsamples=20,
        )
        res = cf()
        s2 = gaussian_eot_scale(1.0) ** 2
        assert abs(res["phi_Z"][0] - s2) < 3 * res["se_Z"][0]
        assert np.all(np.abs(res["phi_Z"][1:]) < 0.1)

    def test_folds_share_auto_epsilon(self):
        from fdfi.explainers import Crossfitting

        X = np.random.default_rng(3).standard_normal((1200, 2))
        cf = Crossfitting(
            lambda x: x[:, 0], X, explainer_class=EOTExplainer, cv=2,
            auto_epsilon=True, nsamples=2,
        )
        cf()
        eps = [fold.epsilon for fold in cf.fold_explainers]
        seeds = [fold.random_state for fold in cf.fold_explainers]
        assert eps[0] == eps[1]
        assert seeds[0] != seeds[1]
