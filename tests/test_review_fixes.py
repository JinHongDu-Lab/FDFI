"""Regression tests for the fixes from the PR #10 review."""

import numpy as np
import pytest

from fdfi.explainers import Crossfitting, EOTExplainer, OTExplainer
from fdfi.explainers.base import Explainer
from fdfi.results import ConfIntResults


def _linear_data(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 1] += 0.5 * X[:, 0]
    y = 2 * X[:, 0] + X[:, 1] + rng.normal(size=n)
    return X, y


def _model(X):
    return 2 * X[:, 0] + X[:, 1]


def test_results_do_not_share_arrays_with_explainer():
    X, y = _linear_data()
    explainer = OTExplainer(_model, X, nsamples=5, random_state=0)
    results = explainer(X, y=y)
    before = explainer.conf_int(target="X").get_score().copy()
    with pytest.raises(ValueError):
        results.get_phi_X()[0] = 999.0
    np.array(results.get_phi_X())[0] = 999.0  # a writable copy is independent
    np.testing.assert_array_equal(explainer.conf_int(target="X").get_score(), before)
    ci = explainer.conf_int(target="X")
    with pytest.raises(ValueError):
        ci.get_score()[0] = 999.0


@pytest.mark.parametrize("cls", [OTExplainer, EOTExplainer])
def test_single_feature_gaussian_explainers(cls):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 1))
    explainer = cls(lambda X_: 3 * X_[:, 0], X, nsamples=5, random_state=0)
    results = explainer(X)
    assert results.get_phi_X().shape == (1,)
    assert results.get_phi_X()[0] > 0


def test_single_observation_has_no_standard_error():
    X, y = _linear_data()
    explainer = OTExplainer(_model, X, nsamples=5, random_state=0)
    results = explainer(X[:1], y=y[:1])
    assert np.all(np.isnan(results.get_se_X()))
    with pytest.raises(ValueError, match="at least two"):
        explainer.conf_int()


def test_gap_margin_ignores_nan_scores():
    margin = Explainer._gap_margin(np.array([np.nan, 0.001, 0.002, 1.0]))
    assert margin == pytest.approx(0.002)


def test_ot_rejects_unknown_method_at_construction():
    X, _ = _linear_data()
    with pytest.raises(ValueError, match="method"):
        OTExplainer(_model, X, method="loco")


def test_flow_rejects_unknown_jacobian_mode():
    pytest.importorskip("torch")
    from fdfi.explainers import FlowExplainer

    X, _ = _linear_data(n=40)
    with pytest.raises(ValueError, match="jacobian_mode"):
        FlowExplainer(_model, X, fit_flow=False, jacobian_mode="avg_sq ")


def test_confidence_interval_plot_rejects_dict():
    from fdfi.plots import confidence_interval_plot

    with pytest.raises(TypeError, match="ConfIntResults"):
        confidence_interval_plot({"score": np.zeros(2)})


def test_crossfit_group_se_matches_feature_se():
    X, y = _linear_data(n=200)
    cf = Crossfitting(_model, X, explainer_class=OTExplainer, cv=2, y=y,
                      nsamples=5, random_state=0)
    cf()
    fixed = dict(var_floor_method="fixed", var_floor_c=0, margin_method="fixed",
                 margin=0, threshold_null=False)
    groups = {f"g{j}": [j] for j in range(X.shape[1])}
    ci_groups = cf.conf_int(target="X", groups=groups, **fixed)
    feature_se = cf.ueifs_X.std(axis=0, ddof=1) / np.sqrt(len(X) / 2)
    np.testing.assert_allclose(ci_groups.get_se(), feature_se)


def test_crossfit_groups_unavailable_after_new_data():
    X, y = _linear_data(n=200)
    cf = Crossfitting(_model, X, explainer_class=OTExplainer, cv=2, y=y,
                      nsamples=5, random_state=0)
    cf()
    cf(X[:50], y=y[:50])
    assert isinstance(cf.conf_int(target="X"), ConfIntResults)
    with pytest.raises(ValueError, match="not available"):
        cf.conf_int(target="X", groups={"a": [0, 1]})
