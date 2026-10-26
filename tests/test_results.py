"""Tests for result classes, printing methods and summary() output (JSS revision)."""

import numpy as np
import pytest

from fdfi.explainers import Crossfitting, EOTExplainer, Explainer, OTExplainer
from fdfi.results import ConfIntResults, ExplanationResults


def _model(X):
    return X[:, 0] + 2.0 * X[:, 1]


@pytest.fixture
def fitted_ot():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 5))
    explainer = OTExplainer(_model, data=X[:100], nsamples=10)
    results = explainer(X[100:])
    return explainer, results


class TestExplanationResults:
    def test_type_and_backward_compat(self, fitted_ot):
        _, results = fitted_ot
        assert isinstance(results, ExplanationResults)
        assert isinstance(results, dict)
        np.testing.assert_array_equal(results["phi_X"], results.phi_X)

    def test_methods(self, fitted_ot):
        _, results = fitted_ot
        assert results.n == 50
        assert results.n_features == 5
        np.testing.assert_array_equal(results.importance("Z"), results["phi_Z"])
        top = results.top_features(2)
        assert set(top) == {0, 1}
        with pytest.raises(ValueError):
            results.importance("W")

    def test_to_frame(self, fitted_ot):
        pytest.importorskip("pandas")
        _, results = fitted_ot
        df = results.to_frame(feature_names=list("abcde"), sort=True)
        assert list(df.columns) == ["estimate", "se", "std"]
        assert df.shape == (5, 3)
        assert df.index[0] == "b"

    def test_repr(self, fitted_ot):
        _, results = fitted_ot
        text = repr(results)
        assert "ExplanationResults" in text and "OTExplainer" in text
        assert str(results) == text

    def test_missing_attribute(self, fitted_ot):
        _, results = fitted_ot
        with pytest.raises(AttributeError):
            results.not_a_key

    def test_all_explainers_return_results_class(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((120, 4))
        eot = EOTExplainer(_model, data=X[:80], nsamples=10)
        assert isinstance(eot(X[80:]), ExplanationResults)
        cf = Crossfitting(_model, X, explainer_class=OTExplainer, cv=3,
                          nsamples=10, random_state=0)
        assert isinstance(cf(), ExplanationResults)
        assert isinstance(cf(X[:20]), ExplanationResults)


class TestConfIntResults:
    def test_type_and_backward_compat(self, fitted_ot):
        explainer, _ = fitted_ot
        ci = explainer.conf_int(alpha=0.05)
        assert isinstance(ci, ConfIntResults)
        assert isinstance(ci, dict)
        np.testing.assert_array_equal(ci["pvalue"], ci.pvalue)

    def test_significant_and_frame(self, fitted_ot):
        explainer, _ = fitted_ot
        ci = explainer.conf_int(alpha=0.05)
        expected = list(np.flatnonzero(ci["reject_null"]))
        assert ci.significant() == expected
        pytest.importorskip("pandas")
        df = ci.to_frame()
        assert df.shape[0] == 5
        assert "ci_lower" in df.columns and "pvalue" in df.columns

    def test_groups_and_multitest(self, fitted_ot):
        explainer, _ = fitted_ot
        ci = explainer.conf_int(groups={"signal": [0, 1], "noise": [2, 3, 4]},
                                multitest_method="bonferroni")
        assert ci.names == ["signal", "noise"]
        assert "bonferroni" in repr(ci)

    def test_repr(self, fitted_ot):
        explainer, _ = fitted_ot
        ci = explainer.conf_int(alpha=0.05)
        assert repr(ci).startswith("ConfIntResults")
        assert str(ci) == repr(ci)


class TestExplainerRepr:
    def test_repr_informative(self, fitted_ot):
        explainer, _ = fitted_ot
        text = repr(explainer)
        assert "object at 0x" not in text
        assert text.startswith("OTExplainer")
        assert "100 samples x 5 features" in text
        assert "computed on 50 samples" in text
        assert str(explainer) == text

    def test_repr_before_call_and_without_data(self):
        rng = np.random.default_rng(2)
        explainer = OTExplainer(_model, data=rng.standard_normal((60, 3)), nsamples=5)
        assert "not yet applied" in repr(explainer)
        bare = Explainer(_model)
        assert "Background data: none" in repr(bare)

    def test_crossfitting_repr(self):
        rng = np.random.default_rng(3)
        cf = Crossfitting(_model, rng.standard_normal((60, 4)),
                          explainer_class=OTExplainer, cv=3, nsamples=5)
        text = repr(cf)
        assert text.startswith("Crossfitting")
        assert "OTExplainer on 3 folds" in text


class TestSummaryOutput:
    def test_summary_prints_once_and_returns_none(self, fitted_ot, capsys):
        explainer, _ = fitted_ot
        out = explainer.summary(alpha=0.05)
        assert out is None
        printed = capsys.readouterr().out
        assert printed.count("Feature Importance Results") == 1

    def test_summary_returns_string_when_not_printing(self, fitted_ot, capsys):
        explainer, _ = fitted_ot
        text = explainer.summary(alpha=0.05, print_output=False)
        assert isinstance(text, str) and "Feature Importance Results" in text
        assert capsys.readouterr().out == ""


class TestResultEdgeCases:
    def test_lazy_pandas_error(self, fitted_ot, monkeypatch):
        import sys

        explainer, results = fitted_ot
        ci = explainer.conf_int()
        monkeypatch.setitem(sys.modules, "pandas", None)
        for result in (results, ci):
            with pytest.raises(ImportError, match="pip install pandas"):
                result.to_frame()
        assert results.top_features(1) == [1]
        assert isinstance(ci.significant(), list)

    def test_frames_validate_names_and_target(self, fitted_ot):
        pytest.importorskip("pandas")
        explainer, results = fitted_ot
        ci = explainer.conf_int(target="Z")
        for result in (results, ci):
            with pytest.raises(ValueError, match="feature_names"):
                result.to_frame(feature_names=["only_one"])
        with pytest.raises(ValueError, match="target"):
            results.to_frame(target="W")
        frame = results.to_frame(target="Z")
        np.testing.assert_array_equal(frame["estimate"], results.phi_Z)
        np.testing.assert_array_equal(frame["se"], results.se_Z)
        assert frame.index.name == "feature"
        assert ci.target == "Z" and ci.alpha == 0.05

    def test_group_names_and_adjusted_pvalues_in_frame(self, fitted_ot):
        pytest.importorskip("pandas")
        explainer, _ = fitted_ot
        ci = explainer.conf_int(
            groups={"signal": [0, 1], "noise": [2, 3, 4]},
            multitest_method="bonferroni",
        )
        frame = ci.to_frame()
        assert frame.index.name == "group"
        assert list(frame.index) == ["signal", "noise"]
        assert ci.significant() == list(frame.index[frame["reject_null"]])
        np.testing.assert_array_equal(frame["pvalue_adj"], ci.pvalue_adj)
        assert list(ci.to_frame(feature_names=["A", "B"]).index) == ["A", "B"]

    def test_wrapper_preserves_arrays_and_plain_cache(self, fitted_ot):
        explainer, results = fitted_ot
        assert type(explainer._last_results) is dict
        for key, value in explainer._last_results.items():
            assert results[key] is value
        assert "phi_X" in dir(results)
        assert "explainer_name" not in results
        assert "n" not in results

    def test_top_features_stable_ties(self):
        results = ExplanationResults({"phi_X": np.array([1., 2., 2., 0.])})
        assert results.top_features(2) == [1, 2]
        assert results.top_features(0) == []
        assert results.top_features(10) == [1, 2, 0, 3]

    @pytest.mark.parametrize("cls", [ExplanationResults, ConfIntResults])
    def test_repr_survives_missing_and_malformed_fields(self, cls):
        for result in (cls({}), cls.__new__(cls), cls({"phi_X": None})):
            assert repr(result).startswith(cls.__name__)
            assert str(result) == repr(result)

    def test_str_dispatches_to_subclass_repr(self):
        class CustomExplainer(Explainer):
            def __repr__(self):
                return "custom explainer"

        class CustomResults(ExplanationResults):
            def __repr__(self):
                return "custom results"

        assert str(CustomExplainer(_model)) == "custom explainer"
        assert str(CustomResults({})) == "custom results"

    @pytest.mark.parametrize("cls", [Explainer, OTExplainer, EOTExplainer,
                                      Crossfitting])
    def test_explainer_repr_before_initialization(self, cls):
        obj = cls.__new__(cls)
        assert repr(obj).startswith(cls.__name__)
        assert str(obj) == repr(obj)

    def test_flow_result_preserves_extra_fields(self):
        pytest.importorskip("torch")
        from fdfi.explainers import FlowExplainer

        rng = np.random.default_rng(4)
        explainer = FlowExplainer(
            _model, rng.standard_normal((30, 3)), method="both",
            nsamples=3, num_steps=5, compute_diagnostics=False,
        )
        results = explainer(rng.standard_normal((5, 3)))
        assert isinstance(results, ExplanationResults)
        assert results.n == 5 and results.explainer_name == "FlowExplainer"
        assert "Flow model: fitted" in repr(explainer)
        assert "phi_X_scpi" in repr(results)
        assert results.phi_X_scpi is explainer._last_results["phi_X_scpi"]
        unfitted = FlowExplainer(_model, explainer.data, fit_flow=False)
        assert "not fitted (call fit_flow())" in repr(unfitted)
        assert repr(FlowExplainer.__new__(FlowExplainer)).startswith("FlowExplainer")
