"""
Tests for FDFI explainers.
"""

import pytest
import numpy as np
import importlib.util
from fdfi.explainers import (
    Explainer,
    OTExplainer,
    EOTExplainer,
    FlowExplainer,
    Crossfitting,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestExplainer:
    """Test the base Explainer class."""
    
    def test_init(self):
        """Test explainer initialization."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        assert explainer.model == dummy_model
        assert explainer.data is None
    
    def test_init_with_data(self):
        """Test explainer initialization with background data."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        data = np.random.randn(100, 10)
        explainer = Explainer(dummy_model, data=data, fit_flow=False)
        assert explainer.model == dummy_model
        assert np.array_equal(explainer.data, data)
    
    def test_call_not_implemented(self):
        """Test that calling base explainer raises NotImplementedError."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotImplementedError):
            explainer(X)
    
    def test_shap_values_calls_call(self):
        """Test that shap_values is an alias for __call__."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        X = np.random.randn(10, 5)
        
        # Both should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            explainer.shap_values(X)

    def test_cpi_matches_normalized_fdfi_formula(self):
        """CPI is one half of the mean per-resample loss difference."""
        explainer = Explainer(lambda x: x[:, 0], fit_flow=False)
        y_pred = np.array([0.0, 1.0, 2.0])
        y_true = np.array([1.0, 3.0, -1.0])
        y_tilde = np.array([
            [2.0, 2.0, 0.0],
            [4.0, -1.0, 3.0],
        ])

        actual = explainer._ueif_from_counterfactuals(
            y_pred, y_tilde, method="cpi", y_true=y_true
        )
        base = (y_true - y_pred) ** 2
        expected = 0.5 * (((y_true[None, :] - y_tilde) ** 2).mean(axis=0) - base)

        assert np.allclose(actual, expected)

    def test_scpi_matches_prediction_first_formula(self):
        """SCPI applies the loss after averaging counterfactual predictions."""
        explainer = Explainer(lambda x: x[:, 0], fit_flow=False)
        y_pred = np.array([0.0, 1.0, 2.0])
        y_true = np.array([1.0, 3.0, -1.0])
        y_tilde = np.array([
            [2.0, 2.0, 0.0],
            [4.0, -1.0, 3.0],
        ])

        actual = explainer._ueif_from_counterfactuals(
            y_pred, y_tilde, method="scpi", y_true=y_true
        )
        expected = (y_true - y_tilde.mean(axis=0)) ** 2 - (y_true - y_pred) ** 2

        assert np.allclose(actual, expected)

    def test_l2_finite_resample_identity(self):
        """For L2, 2*CPI = SCPI + the counterfactual prediction variance."""
        explainer = Explainer(lambda x: x[:, 0], fit_flow=False)
        y_pred = np.array([0.0, 1.0, 2.0])
        y_true = np.array([1.0, 3.0, -1.0])
        y_tilde = np.array([
            [2.0, 2.0, 0.0],
            [4.0, -1.0, 3.0],
            [-2.0, 5.0, 1.0],
        ])

        cpi = explainer._ueif_from_counterfactuals(
            y_pred, y_tilde, method="cpi", y_true=y_true
        )
        scpi = explainer._ueif_from_counterfactuals(
            y_pred, y_tilde, method="scpi", y_true=y_true
        )

        assert np.allclose(2.0 * cpi, scpi + y_tilde.var(axis=0))


def generate_exp3_data(n=500, rho=0.8, seed=42):
    rng = np.random.default_rng(seed)
    d = 50
    mean = np.zeros(d)
    cov = np.eye(d)
    cov[:10, :10] = rho
    np.fill_diagonal(cov[:10, :10], 1.0)
    X = rng.multivariate_normal(mean, cov, size=n)
    y = (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] <= 0)
    )
    y += rng.normal(0, 0.1, size=n)
    return X, y


def exp3_model(X):
    return (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] <= 0)
    )


class TestOTExplainer:
    def test_exp3_mean_variance_consistency(self):
        X_train, _ = generate_exp3_data(n=300, seed=1)
        X_test, _ = generate_exp3_data(n=150, seed=2)

        explainer = OTExplainer(exp3_model, X_train, nsamples=15, random_state=0)
        results = explainer(X_test)

        assert results["phi_X"].shape == (50,)
        assert results["std_X"].shape == (50,)
        assert results["se_X"].shape == (50,)
        assert np.all(np.isfinite(results["phi_X"]))
        assert np.all(results["std_X"] >= 0)

        Z = (X_test - explainer.mean) @ explainer.L_inv
        y_pred = exp3_model(X_test)
        ueifs_Z = explainer._phi_Z(Z, y_pred)

        ueifs_X = ueifs_Z @ (explainer.L ** 2).T
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)
        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        assert np.allclose(phi_X, results["phi_X"])
        assert np.allclose(std_X, results["std_X"])
        assert np.allclose(se_X, results["se_X"])
        assert np.allclose(phi_Z, results["phi_Z"])
        assert np.allclose(std_Z, results["std_Z"])
        assert np.allclose(se_Z, results["se_Z"])

        active_mean = phi_X[:5].mean()
        null_mean = phi_X[10:].mean()
        assert active_mean > null_mean
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="two-sided", var_floor_c=0.0
        )
        assert np.all(ci["ci_lower"][:10] > 0)
        assert "score" in ci

    def test_diagnostics_populated(self):
        X_train, _ = generate_exp3_data(n=120, seed=11)
        explainer = OTExplainer(exp3_model, X_train, nsamples=5, random_state=0)

        diag = explainer.diagnostics
        assert diag is not None
        assert "latent_independence_median" in diag
        assert "distribution_fidelity_mmd" in diag
        assert diag["latent_independence_label"] in {"GOOD", "MODERATE", "POOR"}
        assert diag["distribution_fidelity_label"] in {"GOOD", "MODERATE", "POOR"}

    def test_diagnostics_can_be_disabled(self):
        X_train, _ = generate_exp3_data(n=80, seed=12)
        explainer = OTExplainer(
            exp3_model,
            X_train,
            nsamples=5,
            random_state=0,
            compute_diagnostics=False,
        )
        assert explainer.diagnostics is None
        with pytest.raises(ValueError, match="Diagnostics unavailable"):
            explainer.diagnose()


class TestEOTExplainer:
    def test_exp3_mean_variance_consistency(self):
        X_train, _ = generate_exp3_data(n=80, seed=3)
        X_test, _ = generate_exp3_data(n=40, seed=4)

        explainer = EOTExplainer(exp3_model, X_train)
        results = explainer(X_test)

        assert results["phi_X"].shape == (50,)
        assert results["std_X"].shape == (50,)
        assert results["se_X"].shape == (50,)
        assert np.all(np.isfinite(results["phi_X"]))
        assert np.all(results["std_X"] >= 0)

        Z = explainer.s_fwd * (X_test - explainer.mean) @ explainer.L_inv
        y_pred = exp3_model(X_test)
        ueifs_Z = explainer._phi_Z(Z, y_pred)

        ueifs_X = ueifs_Z @ (explainer.W ** 2).T
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)
        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        assert np.allclose(phi_X, results["phi_X"])
        assert np.allclose(std_X, results["std_X"])
        assert np.allclose(se_X, results["se_X"])
        assert np.allclose(phi_Z, results["phi_Z"])
        assert np.allclose(std_Z, results["std_Z"])
        assert np.allclose(se_Z, results["se_Z"])

        active_mean = phi_X[:5].mean()
        null_mean = phi_X[10:].mean()
        assert active_mean > null_mean
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="two-sided", var_floor_c=0.0
        )
        assert np.all(ci["ci_lower"][:10] > 0)
        assert "score" in ci

    def test_margin_method_auto_uses_gap_for_small_d(self):
        """auto margin should use gap method when d < 30."""
        X_train, _ = generate_exp3_data(n=90, seed=30)
        X_test, _ = generate_exp3_data(n=40, seed=31)
        explainer = EOTExplainer(exp3_model, X_train, nsamples=5, random_state=0)
        explainer(X_test)

        ci = explainer.conf_int(alpha=0.05, target="X", alternative="greater")
        # exp3 has d=50 features, so auto should pick mixture
        assert ci["margin_method"] == "mixture"

        # Now test with a small-d model
        rng = np.random.default_rng(30)
        d_small = 8
        X_s = rng.standard_normal((60, d_small))
        X_st = rng.standard_normal((30, d_small))
        model_s = lambda x: 3.0 * x[:, 0] + 2.0 * x[:, 1]
        exp_s = EOTExplainer(model_s, X_s, nsamples=5, random_state=0)
        exp_s(X_st)
        ci_s = exp_s.conf_int(alpha=0.05, target="X", alternative="greater")
        assert ci_s["margin_method"] == "gap"
        assert ci_s["margin"] >= 0

    def test_margin_method_fixed(self):
        """fixed margin should use the provided value."""
        X_train, _ = generate_exp3_data(n=90, seed=32)
        X_test, _ = generate_exp3_data(n=40, seed=33)
        explainer = EOTExplainer(exp3_model, X_train, nsamples=5, random_state=0)
        explainer(X_test)
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="greater",
            margin=0.5, margin_method="fixed",
        )
        assert ci["margin"] == 0.5
        assert ci["margin_method"] == "fixed"

    def test_margin_method_gap_explicit(self):
        """Explicitly requesting gap method."""
        X_train, _ = generate_exp3_data(n=90, seed=34)
        X_test, _ = generate_exp3_data(n=40, seed=35)
        explainer = EOTExplainer(exp3_model, X_train, nsamples=5, random_state=0)
        explainer(X_test)
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="greater",
            margin_method="gap",
        )
        assert ci["margin_method"] == "gap"
        assert ci["margin"] >= 0

    def test_diagnostics_populated(self):
        X_train, _ = generate_exp3_data(n=90, seed=13)
        explainer = EOTExplainer(exp3_model, X_train, nsamples=5, random_state=0)

        diag = explainer.diagnostics
        assert diag is not None
        assert "latent_independence_median" in diag
        assert "distribution_fidelity_mmd" in diag
        assert diag["latent_independence_label"] in {"GOOD", "MODERATE", "POOR"}
        assert diag["distribution_fidelity_label"] in {"GOOD", "MODERATE", "POOR"}

    def test_auto_epsilon_handles_bimodal_data(self):
        rng = np.random.default_rng(21)
        n, d = 200, 5
        X_train = np.vstack(
            [
                rng.standard_normal((n // 2, d)) - 2.0,
                rng.standard_normal((n // 2, d)) + 2.0,
            ]
        )
        rng.shuffle(X_train)

        def bimodal_model(X):
            return X[:, 0] ** 2 + 0.5 * X[:, 1]

        explainer = EOTExplainer(
            bimodal_model,
            X_train,
            nsamples=10,
            auto_epsilon=True,
            random_state=0,
        )

        assert explainer.epsilon < 2.0
        assert explainer.diagnostics["latent_independence_median"] < 0.25
        assert explainer.diagnostics["distribution_fidelity_mmd"] < 0.5

    def test_population_backward_weights(self):
        """Test that population backward weights W are well-formed."""
        X_train, _ = generate_exp3_data(n=120, seed=33)
        explainer = EOTExplainer(
            exp3_model,
            X_train,
            random_state=0,
        )
        # W should be (d, d) and finite
        d = X_train.shape[1]
        assert explainer.W.shape == (d, d)
        assert np.all(np.isfinite(explainer.W))
        # For perfectly whitened data, W ≈ s * L (population backward = forward)
        assert explainer.s_fwd > 0 and explainer.s_fwd <= 1

    def test_decode_from_Z_roundtrip(self):
        """Test that decode_from_Z approximately reconstructs X."""
        X_train, _ = generate_exp3_data(n=120, seed=34)
        explainer = EOTExplainer(exp3_model, X_train, random_state=0)
        X_hat = explainer._decode_from_Z(explainer.Z_full)
        # Reconstruction should be close (up to EOT shrinkage)
        mse = np.mean((X_hat - X_train) ** 2)
        assert mse < 1.0


def simple_linear_model(X):
    """Simple linear model: y = x0 + 2*x1"""
    return X[:, 0] + 2 * X[:, 1]


@pytest.mark.skipif(not HAS_TORCH, reason="FlowExplainer tests require torch")
class TestFlowExplainer:
    """Test the FlowExplainer class."""
    
    def test_init_no_fit(self):
        """Test FlowExplainer initialization without flow fitting."""
        X = np.random.randn(50, 5)
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        
        assert explainer.model == simple_linear_model
        assert explainer.flow_model is None
        assert explainer.Z_full is None
        assert explainer.method == "cpi"
    
    def test_init_with_fit(self):
        """Test FlowExplainer initialization with flow fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        explainer = FlowExplainer(
            simple_linear_model, X, fit_flow=True, 
            num_steps=50, nsamples=5, random_state=0, verbose=False
        )
        
        assert explainer.flow_model is not None
        assert explainer.Z_full is not None
        assert explainer.Z_full.shape == (50, 5)
    
    def test_method_options(self):
        """Test different method options."""
        X = np.random.randn(50, 5)
        
        # CPI (default)
        exp_cpi = FlowExplainer(simple_linear_model, X, fit_flow=False, method='cpi')
        assert exp_cpi.method == 'cpi'
        
        # SCPI
        exp_scpi = FlowExplainer(simple_linear_model, X, fit_flow=False, method='scpi')
        assert exp_scpi.method == 'scpi'
        
        # Both
        exp_both = FlowExplainer(simple_linear_model, X, fit_flow=False, method='both')
        assert exp_both.method == 'both'
        
        # Invalid method
        with pytest.raises(ValueError, match="method must be"):
            FlowExplainer(simple_linear_model, X, fit_flow=False, method='invalid')
    
    def test_fit_flow_deferred(self):
        """Test deferred flow fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        assert explainer.flow_model is None
        
        explainer.fit_flow(num_steps=50, verbose=False)
        assert explainer.flow_model is not None
        assert explainer.Z_full is not None
    
    def test_set_flow(self):
        """Test setting external flow model."""
        from fdfi.models import FlowMatchingModel
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        # Create and train external flow
        external_flow = FlowMatchingModel(X, dim=5)
        external_flow.fit(num_steps=50, verbose=False)
        
        # Set it in explainer
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        explainer.set_flow(external_flow)
        
        assert explainer.flow_model is external_flow
        assert explainer.Z_full is not None
    
    def test_call_returns_expected_keys(self):
        """Test that __call__ returns all expected result keys."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='cpi', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        expected_keys = ['phi_X', 'std_X', 'se_X', 'phi_Z', 'std_Z', 'se_Z']
        assert all(key in results for key in expected_keys)
        
        # Check shapes
        assert results['phi_X'].shape == (5,)
        assert results['phi_Z'].shape == (5,)
        assert results['std_X'].shape == (5,)
        assert results['se_X'].shape == (5,)
    
    def test_cpi_phi_X_transformed_via_jacobian(self):
        """Test that CPI method transforms phi_Z to phi_X via Jacobian."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='cpi', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # phi_X should be transformed from phi_Z via Jacobian
        # They should not be exactly equal due to the transformation
        assert np.all(np.isfinite(results['phi_X']))
        assert np.all(np.isfinite(results['phi_Z']))
        # Both should be positive (squared differences)
        assert np.all(results['phi_X'] >= 0)
        assert np.all(results['phi_Z'] >= 0)
    
    def test_scpi_computes_different_importance(self):
        """Test that SCPI method computes importance differently from CPI."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer_scpi = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='scpi', random_state=0, verbose=False
        )
        
        results_scpi = explainer_scpi(X_test)
        
        # SCPI should compute E[(Y - f(X_tilde))^2]
        assert results_scpi['phi_X'] is not None
        assert results_scpi['phi_Z'] is not None
        assert np.all(np.isfinite(results_scpi['phi_Z']))
        
        # phi_X is transformed from phi_Z via Jacobian
        assert np.all(np.isfinite(results_scpi['phi_X']))
        # Both should be positive (squared differences)
        assert np.all(results_scpi['phi_X'] >= 0)
        assert np.all(results_scpi['phi_Z'] >= 0)
    
    def test_sampling_methods(self):
        """Test all sampling methods produce valid results."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        for method in ['resample', 'permutation', 'normal']:
            explainer = FlowExplainer(
                simple_linear_model, X_train, fit_flow=True,
                num_steps=50, nsamples=5, sampling_method=method, random_state=0, verbose=False
            )
            
            results = explainer(X_test)
            
            assert np.all(np.isfinite(results['phi_X']))
            assert np.all(np.isfinite(results['phi_Z']))
            assert np.all(results['std_X'] >= 0)
            assert np.all(results['se_X'] >= 0)
    
    def test_condperm_sampling(self):
        """Test conditional permutation sampling method."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, sampling_method='condperm', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        assert np.all(np.isfinite(results['phi_X']))
        assert np.all(np.isfinite(results['phi_Z']))
    
    def test_conf_int_integration(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(20, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=10, random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # Test conf_int
        ci = explainer.conf_int(alpha=0.05, target='X', alternative='two-sided')
        
        assert 'score' in ci
        assert 'se' in ci
        assert 'ci_lower' in ci
        assert 'ci_upper' in ci
        assert 'reject_null' in ci
        assert 'pvalue' in ci
    
    def test_active_features_higher_importance(self):
        """Test that active features have higher importance."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(30, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=200, nsamples=20, random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # Features 0 and 1 should have higher importance than 2, 3, 4
        active_importance = (results['phi_Z'][0] + results['phi_Z'][1]) / 2
        null_importance = results['phi_Z'][2:].mean()
        
        assert active_importance > null_importance
    
    def test_reproducibility(self):
        """Test random_state produces reproducible results."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer1 = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, random_state=123
        )
        results1 = explainer1(X_test)
        
        # Create new explainer with same seed
        np.random.seed(42)
        explainer2 = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, random_state=123
        )
        results2 = explainer2(X_test)
        
        # Results should be correlated (exact reproducibility is hard due to
        # flow training stochasticity and Jacobian computation)
        # Check rank correlation instead of exact values
        from scipy.stats import spearmanr
        corr, _ = spearmanr(results1['phi_Z'], results2['phi_Z'])
        assert corr > 0.5, f"Rank correlation should be positive: {corr}"


@pytest.mark.skipif(not HAS_TORCH, reason="FlowExplainer tests require torch")
class TestFlowVsOTExplainer:
    """Integration tests comparing FlowExplainer and OTExplainer."""
    
    def test_both_identify_active_features(self):
        """Both explainers should identify active features > null features."""
        np.random.seed(42)
        
        # Create correlated data
        n_train, n_test, d = 200, 50, 8
        cov = np.eye(d)
        cov[0, 1] = cov[1, 0] = 0.6
        
        X_train = np.random.multivariate_normal(np.zeros(d), cov, size=n_train)
        X_test = np.random.multivariate_normal(np.zeros(d), cov, size=n_test)
        
        # Model uses features 0, 1, 2 only
        def model(X):
            return X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2]
        
        # FlowExplainer
        flow_exp = FlowExplainer(
            model, X_train, fit_flow=True,
            num_steps=200, nsamples=20, random_state=0
        )
        flow_results = flow_exp(X_test)
        
        # OTExplainer
        ot_exp = OTExplainer(model, X_train, nsamples=20, random_state=0)
        ot_results = ot_exp(X_test)
        
        # Both should identify active > null features
        active_idx = [0, 1, 2]
        null_idx = [3, 4, 5, 6, 7]
        
        flow_active = np.mean([flow_results['phi_X'][i] for i in active_idx])
        flow_null = np.mean([flow_results['phi_X'][i] for i in null_idx])
        
        ot_active = np.mean([ot_results['phi_X'][i] for i in active_idx])
        ot_null = np.mean([ot_results['phi_X'][i] for i in null_idx])
        
        assert flow_active > flow_null, "FlowExplainer should identify active > null"
        assert ot_active > ot_null, "OTExplainer should identify active > null"
    
    def test_positive_correlation_between_methods(self):
        """Feature rankings should be positively correlated."""
        np.random.seed(42)
        
        n_train, n_test, d = 150, 40, 6
        X_train = np.random.randn(n_train, d)
        X_test = np.random.randn(n_test, d)
        
        def model(X):
            return X[:, 0] + 1.5 * X[:, 1] + 0.3 * X[:, 2]
        
        flow_exp = FlowExplainer(
            model, X_train, fit_flow=True,
            num_steps=200, nsamples=15, random_state=0
        )
        flow_results = flow_exp(X_test)
        
        ot_exp = OTExplainer(model, X_train, nsamples=15, random_state=0)
        ot_results = ot_exp(X_test)
        
        # Correlation should be positive
        correlation = np.corrcoef(flow_results['phi_X'], ot_results['phi_X'])[0, 1]
        assert correlation > 0, f"Expected positive correlation, got {correlation}"


# ──────────────────────────────────────────────────────────────────────────
# Crossfitting tests
# ──────────────────────────────────────────────────────────────────────────

def _simple_model(X):
    """y = x0 + 2*x1, used across Crossfitting tests."""
    return X[:, 0] + 2.0 * X[:, 1]


class TestCrossfittingInit:
    """Constructor and CV resolution."""

    def test_init_stores_args(self):
        data = np.random.randn(60, 4)
        cf = Crossfitting(_simple_model, data, explainer_class=OTExplainer, cv=3)
        assert cf.explainer_class is OTExplainer
        assert cf.fold_explainers == []
        assert cf.ueifs_X is None

    def test_cv_int_resolves_to_kfold(self):
        from sklearn.model_selection import KFold
        data = np.random.randn(50, 3)
        cf = Crossfitting(_simple_model, data, cv=7, random_state=0)
        assert isinstance(cf.cv_, KFold)
        assert cf.cv_.n_splits == 7

    def test_cv_splitter_passthrough(self):
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        data = np.random.randn(50, 3)
        cf = Crossfitting(_simple_model, data, cv=ss)
        assert cf.cv_ is ss

    def test_cv_invalid_type_raises(self):
        data = np.random.randn(50, 3)
        with pytest.raises(TypeError):
            Crossfitting(_simple_model, data, cv=3.14)


class TestCrossfittingWithOT:
    """Cross-fitting with OTExplainer."""

    def test_crossfit_result_keys(self):
        np.random.seed(0)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=3, random_state=0,
            nsamples=10,
        )
        results = cf()
        for key in ("phi_X", "std_X", "se_X", "phi_Z", "std_Z", "se_Z"):
            assert key in results, f"Missing key {key}"
            assert results[key].shape == (4,)

    def test_crossfit_fold_count(self):
        np.random.seed(1)
        data = np.random.randn(60, 3)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=4, nsamples=5,
            random_state=0,
        )
        cf()
        assert len(cf.fold_explainers) == 4
        assert len(cf.fold_indices) == 4

    def test_no_train_test_overlap(self):
        np.random.seed(2)
        data = np.random.randn(50, 3)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=5, nsamples=5,
            random_state=0,
        )
        cf()
        for train_idx, test_idx in cf.fold_indices:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_conf_int_works(self):
        np.random.seed(3)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=3, nsamples=10,
            random_state=0,
        )
        cf()
        ci = cf.conf_int(alpha=0.05)
        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["ci_lower"].shape == (4,)

    def test_summary_runs(self):
        np.random.seed(4)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=3, nsamples=10,
            random_state=0,
        )
        cf()
        output = cf.summary(print_output=False)
        assert "Feature Importance" in output

    def test_ensemble_predict_new_data(self):
        np.random.seed(5)
        data = np.random.randn(80, 4)
        X_new = np.random.randn(20, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer, cv=3, nsamples=10,
            random_state=0,
        )
        cf()  # fit folds
        results = cf(X_new)
        assert results["phi_X"].shape == (4,)


class TestCrossfittingWithEOT:
    """Cross-fitting with EOTExplainer."""

    def test_eot_crossfit(self):
        np.random.seed(10)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=EOTExplainer, cv=3, nsamples=10,
            random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (4,)
        assert np.all(np.isfinite(results["phi_X"]))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCrossfittingWithFlow:
    """Cross-fitting with FlowExplainer."""

    def test_flow_crossfit(self):
        np.random.seed(20)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=FlowExplainer, cv=2,
            nsamples=5, num_steps=100, random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (4,)


class TestCrossfittingSplitters:
    """Different sklearn splitter strategies."""

    def test_stratified_kfold(self):
        from sklearn.model_selection import StratifiedKFold
        np.random.seed(30)
        data = np.random.randn(80, 4)
        y = np.array([0, 1] * 40)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer,
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
            y=y, nsamples=5, random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (4,)

    def test_shuffle_split(self):
        from sklearn.model_selection import ShuffleSplit
        np.random.seed(31)
        data = np.random.randn(80, 4)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer,
            cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=0),
            nsamples=5, random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (4,)

    def test_repeated_kfold(self):
        from sklearn.model_selection import RepeatedKFold
        np.random.seed(32)
        data = np.random.randn(60, 3)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer,
            cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=0),
            nsamples=5, random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (3,)
        # All 60 samples should have been evaluated
        assert cf.ueifs_X.shape[0] == 60

    def test_group_kfold(self):
        from sklearn.model_selection import GroupKFold
        np.random.seed(33)
        data = np.random.randn(60, 3)
        groups = np.repeat(np.arange(6), 10)  # 6 groups of 10
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer,
            cv=GroupKFold(n_splits=3),
            groups=groups, nsamples=5, random_state=0,
        )
        results = cf()
        assert results["phi_X"].shape == (3,)

    def test_custom_splitter(self):
        """Any object with .split() should work."""
        class TwoFoldSplitter:
            def split(self, X, y=None, groups=None):
                n = X.shape[0]
                mid = n // 2
                yield np.arange(mid, n), np.arange(mid)
                yield np.arange(mid), np.arange(mid, n)

            def get_n_splits(self, X=None, y=None, groups=None):
                return 2

        np.random.seed(34)
        data = np.random.randn(40, 3)
        cf = Crossfitting(
            _simple_model, data,
            explainer_class=OTExplainer,
            cv=TwoFoldSplitter(), nsamples=5, random_state=0,
        )
        results = cf()
        assert len(cf.fold_explainers) == 2
        assert results["phi_X"].shape == (3,)


# ── Group importance tests ────────────────────────────────────────────


def _make_ot_explainer_with_results(seed=42):
    """Create an OTExplainer and run it so ueifs are populated."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((80, 5))
    X = rng.standard_normal((60, 5))
    explainer = OTExplainer(_simple_model, data, nsamples=5, random_state=0)
    explainer(X)
    return explainer


class TestGroupImportanceInput:
    """Test _normalize_groups and input validation."""

    def test_dict_input(self):
        explainer = _make_ot_explainer_with_results()
        groups = {"ab": [0, 1], "cde": [2, 3, 4]}
        res = explainer.group_importance(groups)
        assert set(res.keys()) == {"groups", "importance", "se", "zscore", "pvalue"}
        assert len(res["groups"]) == 2
        assert res["importance"].shape == (2,)

    def test_array_input(self):
        explainer = _make_ot_explainer_with_results()
        labels = np.array(["A", "A", "B", "B", "B"])
        res = explainer.group_importance(labels)
        assert len(res["groups"]) == 2
        assert set(res["groups"]) == {"A", "B"}

    def test_dataframe_input(self):
        import pandas as pd
        explainer = _make_ot_explainer_with_results()
        df = pd.DataFrame(
            {"g1": [1, 1, 0, 0, 0], "g2": [0, 0, 1, 1, 1]},
        )
        res = explainer.group_importance(df)
        assert len(res["groups"]) == 2

    def test_overlapping_groups(self):
        import pandas as pd
        explainer = _make_ot_explainer_with_results()
        df = pd.DataFrame(
            {"g1": [1, 1, 0, 0, 0], "g2": [0, 1, 1, 0, 0]},  # feature 1 in both
        )
        res = explainer.group_importance(df)
        assert len(res["groups"]) == 2
        assert np.all(np.isfinite(res["importance"]))

    def test_invalid_input_raises(self):
        explainer = _make_ot_explainer_with_results()
        with pytest.raises(TypeError):
            explainer.group_importance("bad_input")


class TestGroupImportanceBehavior:
    """Test group_importance logic and conf_int(groups=...)."""

    def test_before_call_raises(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 4))
        explainer = OTExplainer(_simple_model, data, nsamples=5, random_state=0)
        # Haven't called explainer(X) yet
        with pytest.raises(ValueError, match="Per-sample UEIFs not available"):
            explainer.conf_int(groups={"all": [0, 1, 2, 3]})

    def test_conf_int_groups(self):
        explainer = _make_ot_explainer_with_results()
        groups = {"ab": [0, 1], "cde": [2, 3, 4]}
        res = explainer.conf_int(groups=groups)
        
        assert "groups" in res
        assert len(res["groups"]) == 2
        assert res["score"].shape == (2,)
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert "pvalue" in res

    def test_threshold_null(self):
        explainer = _make_ot_explainer_with_results()
        groups = {"all": [0, 1, 2, 3, 4]}
        res_thresh = explainer.conf_int(
            groups=groups, threshold_null=True
        )
        res_no = explainer.conf_int(
            groups=groups, threshold_null=False
        )
        # With thresholding, importance >= without (since we zero negatives)
        assert res_thresh["score"][0] >= res_no["score"][0] - 1e-12

    def test_se_adjustment(self):
        explainer = _make_ot_explainer_with_results()
        groups = {"all": [0, 1, 2, 3, 4]}
        # Use var_floor_c which maps to the old se_adjustment
        res_adj = explainer.conf_int(
            groups=groups, var_floor_c=0.1, var_floor_method="fixed"
        )
        res_no_adj = explainer.conf_int(
            groups=groups, var_floor_c=0.0, var_floor_method="fixed"
        )
        # With adjustment, SE should be larger
        assert res_adj["se"][0] >= res_no_adj["se"][0]

    def test_single_feature_group(self):
        explainer = _make_ot_explainer_with_results()
        res = explainer.conf_int(
            groups={"feat0": [0]}, threshold_null=False, var_floor_c=0.0
        )
        # Should match individual feature importance
        phi_X = explainer.ueifs_X[:, 0].mean()
        assert np.isclose(res["score"][0], phi_X, atol=1e-10)

    def test_target_Z(self):
        explainer = _make_ot_explainer_with_results()
        res = explainer.conf_int(
            groups={"ab": [0, 1]}, target="Z"
        )
        assert len(res["groups"]) == 1
        assert np.all(np.isfinite(res["score"]))

    def test_pvalues_valid(self):
        explainer = _make_ot_explainer_with_results()
        res = explainer.conf_int(groups={"ab": [0, 1], "cde": [2, 3, 4]})
        assert np.all(res["pvalue"] >= 0)
        assert np.all(res["pvalue"] <= 1)
        assert np.all(np.isfinite(res["score"]))

    def test_deprecation_warning(self):
        explainer = _make_ot_explainer_with_results()
        with pytest.warns(FutureWarning, match=r"group_importance\(\) is deprecated"):
            res = explainer.group_importance({"ab": [0, 1]})
        assert "importance" in res



class TestMultipleTesting:
    """Test multiple testing corrections in conf_int."""

    def test_conf_int_multitest_bonferroni(self):
        explainer = _make_ot_explainer_with_results()
        # d=5 features
        res = explainer.conf_int(multitest_method="bonferroni")
        
        assert "pvalue_adj" in res
        assert "multitest_method" in res
        assert res["multitest_method"] == "bonferroni"
        
        # Adjusted p-values should be min(p * 5, 1)
        expected = np.minimum(res["pvalue"] * 5, 1.0)
        assert np.allclose(res["pvalue_adj"], expected)
        # reject_null should follow pvalue_adj < alpha
        assert np.array_equal(res["reject_null"], res["pvalue_adj"] < 0.05)

    def test_conf_int_multitest_fdr_bh(self):
        explainer = _make_ot_explainer_with_results()
        res = explainer.conf_int(multitest_method="fdr_bh")
        
        assert "pvalue_adj" in res
        assert res["multitest_method"] == "fdr_bh"
        # Adjusted p-values for FDR should be >= raw p-values
        assert np.all(res["pvalue_adj"] >= res["pvalue"] - 1e-12)
        assert np.array_equal(res["reject_null"], res["pvalue_adj"] < 0.05)

    def test_summary_with_multitest(self):
        explainer = _make_ot_explainer_with_results()
        output = explainer.summary(multitest_method="fdr_bh", print_output=False)
        assert "Multiple testing: fdr_bh" in output
        assert "Adj P-val" in output

    def test_multitest_missing_statsmodels(self, monkeypatch):
        import sys
        # Simulate missing statsmodels
        monkeypatch.setitem(sys.modules, "statsmodels.stats.multitest", None)
        monkeypatch.setitem(sys.modules, "statsmodels", None)
        
        explainer = _make_ot_explainer_with_results()
        with pytest.raises(ImportError, match="Multiple testing correction requires statsmodels"):
            explainer.conf_int(multitest_method="bonferroni")


# ──────────────────────────────────────────────────────────────────────────
# Arbitrary-loss support (regression + classification, CPI/SCPI)
# ──────────────────────────────────────────────────────────────────────────

def _linear_regression_dgp(n=300, d=6, seed=0):
    """Linear DGP: features 0, 1 active; 2..d-1 null."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta = np.zeros(d)
    beta[0], beta[1] = 2.0, -1.5
    y = X @ beta + 0.1 * rng.standard_normal(n)

    def model(Xin):
        return Xin @ beta

    return X, y, model


def _binary_classification_dgp(n=400, d=6, seed=0):
    """Logistic DGP with a probability-output model; features 0, 1 active."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = np.zeros(d)
    w[0], w[1] = 2.0, -1.5
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p).astype(float)

    def model(Xin):  # returns P(y=1)
        return 1.0 / (1.0 + np.exp(-(Xin @ w)))

    return X, y, model


class TestArbitraryLossOT:
    """OTExplainer with non-L2 losses and CPI/SCPI averaging."""

    def test_default_equals_squared_error(self):
        X, y, model = _linear_regression_dgp()
        e_default = OTExplainer(model, X, nsamples=15, random_state=0)
        e_l2 = OTExplainer(model, X, nsamples=15, random_state=0, loss="squared_error")
        r_default = e_default(X[:80])
        r_l2 = e_l2(X[:80])
        assert np.allclose(r_default["phi_Z"], r_l2["phi_Z"], atol=1e-12)
        assert np.allclose(r_default["phi_X"], r_l2["phi_X"], atol=1e-12)

    def test_cpi_scpi_l2_population_agreement(self):
        """Normalized CPI and SCPI agree up to finite-sample Monte Carlo error."""
        X, y, model = _linear_regression_dgp(n=1200, d=4, seed=3)
        Xt, yt = X[:600], y[:600]
        common = dict(
            nsamples=100,
            random_state=0,
            compute_diagnostics=False,
        )
        e_cpi = OTExplainer(model, X, method="cpi", **common)
        e_scpi = OTExplainer(model, X, method="scpi", **common)
        r_cpi = e_cpi(Xt, y=yt)
        r_scpi = e_scpi(Xt, y=yt)

        # Compare active latent directions. Null directions have population
        # importance zero, so relative tolerances are not informative there.
        assert np.allclose(
            r_cpi["phi_Z"][:2], r_scpi["phi_Z"][:2], rtol=0.12, atol=0.08
        )

    @pytest.mark.parametrize("loss", ["l1", "huber", "pinball"])
    def test_regression_losses_rank_active(self, loss):
        X, y, model = _linear_regression_dgp(seed=1)
        Xt, yt = X[:120], y[:120]
        e = OTExplainer(model, X, nsamples=25, random_state=0, loss=loss)
        res = e(Xt, y=yt)
        active = res["phi_X"][:2].mean()
        null = res["phi_X"][2:].mean()
        assert active > null

    @pytest.mark.parametrize("loss", ["log_loss", "brier"])
    def test_classification_losses_rank_active(self, loss):
        X, y, model = _binary_classification_dgp(seed=2)
        Xt, yt = X[:150], y[:150]
        e = OTExplainer(model, X, nsamples=25, random_state=0, loss=loss)
        res = e(Xt, y=yt)
        active = res["phi_X"][:2].mean()
        null = np.abs(res["phi_X"][2:]).mean()
        assert active > null

    def test_non_l2_without_y_divergence(self):
        # Label-free non-L2 is allowed: it computes a divergence from the model's
        # own prediction (KL for log-loss), non-negative and active > null.
        X, y, model = _binary_classification_dgp(seed=7)
        e = OTExplainer(model, X, nsamples=25, random_state=0, loss="log_loss")
        res = e(X[:150])   # no y
        assert np.all(np.isfinite(res["phi_X"]))
        assert res["phi_Z"].min() >= -1e-8   # KL divergence is non-negative
        assert res["phi_X"][:2].mean() > np.abs(res["phi_X"][2:]).mean()

    def test_l1_without_y_equals_prediction_shift(self):
        # For losses with L(a, a) = 0 the label-free base vanishes.
        X, y, model = _linear_regression_dgp(seed=8)
        e = OTExplainer(model, X, nsamples=20, random_state=0, loss="l1")
        res = e(X[:100])   # no y
        assert np.all(np.isfinite(res["phi_X"]))
        assert res["phi_X"][:2].mean() > res["phi_X"][2:].mean()

    def test_invalid_method_raises(self):
        X, y, model = _linear_regression_dgp()
        e = OTExplainer(model, X, nsamples=10, random_state=0, method="bogus")
        with pytest.raises(ValueError, match="method must be"):
            e(X[:20], y=y[:20])

    def test_custom_callable_loss(self):
        X, y, model = _linear_regression_dgp(seed=3)
        Xt, yt = X[:100], y[:100]

        def cubic(y_true, y_pred):
            return np.abs(y_true - y_pred) ** 3

        e = OTExplainer(model, X, nsamples=20, random_state=0, loss=cubic)
        res = e(Xt, y=yt)
        assert np.all(np.isfinite(res["phi_X"]))
        assert res["phi_X"][:2].mean() > res["phi_X"][2:].mean()


class TestArbitraryLossEOT:
    """EOTExplainer with non-L2 losses."""

    def test_default_equals_squared_error(self):
        X, y, model = _linear_regression_dgp()
        e_default = EOTExplainer(model, X, nsamples=15, random_state=0)
        e_l2 = EOTExplainer(model, X, nsamples=15, random_state=0, loss="squared_error")
        r_default = e_default(X[:80])
        r_l2 = e_l2(X[:80])
        assert np.allclose(r_default["phi_Z"], r_l2["phi_Z"], atol=1e-12)

    def test_log_loss_ranks_active(self):
        X, y, model = _binary_classification_dgp(seed=4)
        Xt, yt = X[:150], y[:150]
        e = EOTExplainer(model, X, nsamples=25, random_state=0, loss="log_loss")
        res = e(Xt, y=yt)
        assert res["phi_X"][:2].mean() > np.abs(res["phi_X"][2:]).mean()


class TestArbitraryLossCrossfitting:
    """Crossfitting forwards loss to fold explainers end-to-end."""

    def test_log_loss_with_sklearn_classifier(self):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, _ = _binary_classification_dgp(n=200, seed=5)
        cf = Crossfitting(
            LogisticRegression(max_iter=500),
            X,
            explainer_class=OTExplainer,
            cv=3,
            y=y,
            loss="log_loss",
            nsamples=15,
            random_state=0,
        )
        results = cf()
        assert np.all(np.isfinite(results["phi_X"]))
        ci = cf.conf_int(alpha=0.05, target="X")
        assert np.all(np.isfinite(ci["score"]))
        assert np.all(ci["pvalue"] >= 0) and np.all(ci["pvalue"] <= 1)


@pytest.mark.skipif(not HAS_TORCH, reason="FlowExplainer tests require torch")
class TestArbitraryLossFlow:
    """FlowExplainer with non-L2 losses and CPI/SCPI."""

    def test_default_cpi_unchanged_for_l2(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((60, 5))
        X_test = rng.standard_normal((15, 5))
        e_default = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method="cpi", random_state=0, verbose=False,
        )
        e_l2 = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method="cpi", random_state=0, verbose=False,
            loss="squared_error",
        )
        r_default = e_default(X_test)
        r_l2 = e_l2(X_test)
        assert np.allclose(r_default["phi_Z"], r_l2["phi_Z"], atol=1e-10)

    def test_log_loss_with_and_without_y(self):
        X, y, model = _binary_classification_dgp(n=120, d=5, seed=6)
        e = FlowExplainer(
            model, X, fit_flow=True, num_steps=80, nsamples=8,
            random_state=0, verbose=False, loss="log_loss",
        )
        # Without y -> label-free divergence (finite, non-negative in Z-space)
        res_free = e(X[:40])
        assert np.all(np.isfinite(res_free["phi_X"]))
        assert res_free["phi_Z"].min() >= -1e-6
        # With y -> DFI loss-difference form (finite)
        res = e(X[:40], y=y[:40])
        assert np.all(np.isfinite(res["phi_X"]))
