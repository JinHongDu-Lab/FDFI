"""
Result containers returned by fdfi explainers.

Values are read through explicit ``get_*()`` methods. The containers also
provide convenience methods and an informative printed summary.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "to_frame() returns a pandas DataFrame. Install pandas to use it "
            "(pip install pandas)."
        ) from exc
    return pd


class _ResultBase:
    """Result container with a readable printed form."""

    def _repr_getters(self) -> str:
        methods = [
            f"get_{field}()" for field in self._data
            if isinstance(field, str) and callable(getattr(self, f"get_{field}", None))
        ]
        if {"ci_lower", "ci_upper"}.issubset(self._data) and callable(
            getattr(self, "get_ci", None)
        ):
            methods.append("get_ci()")
        return "  Getters: " + ", ".join(methods)

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_pretty_(self, p, cycle):  # IPython / Jupyter display
        p.text(self.__repr__())


class ExplanationResults(_ResultBase):
    """
    Feature importance estimates returned by calling an explainer,
    e.g. ``results = explainer(X_test)``.

    Read importance, standard error and per-sample standard deviation with
    ``get_phi_X()``, ``get_se_X()`` and ``get_std_X()`` in X-space, or the
    corresponding ``*_Z()`` methods in latent space. For FlowExplainer with
    ``method='both'``, ``get_*_scpi()`` returns the additional SCPI fields.
    Getters return the stored arrays without copying or recomputing them;
    a getter for an unavailable field raises ``KeyError``.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        explainer_name: str = "",
        n: Optional[int] = None,
    ):
        self._data = dict(data)
        self.explainer_name = explainer_name
        self.n = n

    def get_phi_X(self) -> np.ndarray:
        """Return original-space importance estimates."""
        return self._data["phi_X"]

    def get_phi_Z(self) -> np.ndarray:
        """Return latent-space importance estimates."""
        return self._data["phi_Z"]

    def get_se_X(self) -> np.ndarray:
        """Return original-space standard errors."""
        return self._data["se_X"]

    def get_se_Z(self) -> np.ndarray:
        """Return latent-space standard errors."""
        return self._data["se_Z"]

    def get_std_X(self) -> np.ndarray:
        """Return original-space per-sample standard deviations."""
        return self._data["std_X"]

    def get_std_Z(self) -> np.ndarray:
        """Return latent-space per-sample standard deviations."""
        return self._data["std_Z"]

    def get_phi_X_scpi(self) -> np.ndarray:
        """Return original-space SCPI importance for method='both'."""
        return self._data["phi_X_scpi"]

    def get_phi_Z_scpi(self) -> np.ndarray:
        """Return latent-space SCPI importance for method='both'."""
        return self._data["phi_Z_scpi"]

    def get_se_X_scpi(self) -> np.ndarray:
        """Return original-space SCPI standard errors for method='both'."""
        return self._data["se_X_scpi"]

    def get_se_Z_scpi(self) -> np.ndarray:
        """Return latent-space SCPI standard errors for method='both'."""
        return self._data["se_Z_scpi"]

    def get_std_X_scpi(self) -> np.ndarray:
        """Return original-space SCPI standard deviations for method='both'."""
        return self._data["std_X_scpi"]

    def get_std_Z_scpi(self) -> np.ndarray:
        """Return latent-space SCPI standard deviations for method='both'."""
        return self._data["std_Z_scpi"]

    @property
    def n_features(self) -> int:
        return len(self.get_phi_X())

    def _check_target(self, target: str) -> str:
        if target not in ("X", "Z"):
            raise ValueError("target must be 'X' or 'Z'")
        return target

    def importance(self, target: str = "X") -> np.ndarray:
        """Return the importance vector in X-space (default) or Z-space."""
        return self.get_phi_X() if self._check_target(target) == "X" else self.get_phi_Z()

    def top_features(self, k: int = 5, target: str = "X") -> List[int]:
        """Indices of the ``k`` features with the largest estimated importance."""
        phi = self.importance(target)
        return [int(j) for j in np.argsort(-phi, kind="stable")[:k]]

    def to_frame(
        self,
        target: str = "X",
        feature_names: Optional[Sequence[str]] = None,
        sort: bool = False,
    ):
        """
        Return importance estimates as a :class:`pandas.DataFrame` with
        columns ``estimate``, ``se`` and ``std``, one row per feature.
        """
        pd = _require_pandas()
        t = self._check_target(target)
        phi = self.importance(t)
        se = self.get_se_X() if t == "X" else self.get_se_Z()
        std = self.get_std_X() if t == "X" else self.get_std_Z()
        if feature_names is None:
            feature_names = list(range(len(phi)))
        elif len(feature_names) != len(phi):
            raise ValueError("feature_names must have one entry per feature")
        df = pd.DataFrame(
            {"estimate": phi, "se": se, "std": std},
            index=pd.Index(list(feature_names), name="feature"),
        )
        if sort:
            df = df.sort_values("estimate", ascending=False)
        return df

    def __repr__(self) -> str:
        try:
            head = "ExplanationResults"
            meta = []
            if getattr(self, "explainer_name", ""):
                meta.append(self.explainer_name)
            if getattr(self, "n", None) is not None:
                meta.append(f"n = {self.n} samples")
            if "phi_X" in self._data:
                meta.append(f"d = {self.n_features} features")
            lines = [head + (f" ({', '.join(meta)})" if meta else "")]
            lines.append(self._repr_getters())
            if "phi_X" in self._data and "se_X" in self._data:
                k = min(5, self.n_features)
                lines.append(f"  Top {k} features (X-space):")
                for j in self.top_features(k):
                    lines.append(
                        f"    feature {j}: {self.get_phi_X()[j]:.4f} "
                        f"(se {self.get_se_X()[j]:.4f})"
                    )
            lines.append(
                "  Use .to_frame() for the full table; explainer.conf_int() or "
                "explainer.summary() for inference."
            )
            return "\n".join(lines)
        except Exception:
            text = f"{type(self).__name__} ({len(getattr(self, '_data', {}))} fields)"
            if hasattr(self, "_data"):
                text += "\n" + self._repr_getters()
            return text


class ConfIntResults(_ResultBase):
    """
    Inference results returned by :meth:`Explainer.conf_int`.

    Read each field through its ``get_*()`` method. ``get_ci()`` returns the
    ``(lower, upper)`` arrays. Arrays have one entry per feature, or per group
    when ``groups`` was given. ``get_groups()``, ``get_pvalue_adj()`` and
    ``get_multitest_method()`` return ``None`` when not applicable. Other
    getters raise ``KeyError`` for unavailable fields. Stored values are
    returned without copying or recomputing them.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        alpha: Optional[float] = None,
        target: Optional[str] = None,
    ):
        self._data = dict(data)
        self.alpha = alpha
        self.target = target

    def get_score(self) -> np.ndarray:
        """Return estimated importance per feature or group."""
        return self._data["score"]

    def get_se(self) -> np.ndarray:
        """Return standard errors after variance-floor adjustment."""
        return self._data["se"]

    def get_ci(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return confidence interval bounds as ``(lower, upper)``."""
        return self.get_ci_lower(), self.get_ci_upper()

    def get_ci_lower(self) -> np.ndarray:
        """Return lower confidence interval bounds."""
        return self._data["ci_lower"]

    def get_ci_upper(self) -> np.ndarray:
        """Return upper confidence interval bounds."""
        return self._data["ci_upper"]

    def get_pvalue(self) -> np.ndarray:
        """Return unadjusted p-values."""
        return self._data["pvalue"]

    def get_ranking(self) -> np.ndarray:
        """Return ranks by descending z-score (1 is most important)."""
        return self._data["ranking"]

    def get_zscore(self) -> np.ndarray:
        """Return signed z-statistics."""
        return self._data["zscore"]

    def get_pvalue_adj(self) -> Optional[np.ndarray]:
        """Return adjusted p-values, or None without multiple testing."""
        return self._data.get("pvalue_adj")

    def get_reject_null(self) -> np.ndarray:
        """Return the boolean null-rejection decisions."""
        return self._data["reject_null"]

    def get_groups(self) -> Optional[list]:
        """Return group names, or None for feature-level inference."""
        return self._data.get("groups")

    def get_margin(self) -> float:
        """Return the null-hypothesis margin."""
        return self._data["margin"]

    def get_margin_method(self) -> str:
        """Return the method used to select the margin."""
        return self._data["margin_method"]

    def get_alternative(self) -> str:
        """Return the alternative hypothesis."""
        return self._data["alternative"]

    def get_multitest_method(self) -> Optional[str]:
        """Return the multiple-testing method, or None if not requested."""
        return self._data.get("multitest_method")

    @property
    def names(self) -> list:
        """Feature indices, or group names for group-level results."""
        if self.get_groups() is not None:
            return list(self.get_groups())
        return list(range(len(self.get_score())))

    def significant(self) -> list:
        """Names (feature indices or group names) of units with ``reject_null``."""
        names = self.names
        return [names[i] for i in np.flatnonzero(np.asarray(self.get_reject_null()))]

    def to_frame(self, feature_names: Optional[Sequence[str]] = None):
        """Return the inference table as a :class:`pandas.DataFrame`."""
        pd = _require_pandas()
        names = self.names if feature_names is None else list(feature_names)
        if len(names) != len(self.get_score()):
            raise ValueError("feature_names must have one entry per unit")
        cols = {
            "estimate": self.get_score(),
            "se": self.get_se(),
            "ci_lower": self.get_ci_lower(),
            "ci_upper": self.get_ci_upper(),
            "zscore": self.get_zscore(),
            "pvalue": self.get_pvalue(),
        }
        if self.get_pvalue_adj() is not None:
            cols["pvalue_adj"] = self.get_pvalue_adj()
        cols["reject_null"] = self.get_reject_null()
        cols["ranking"] = self.get_ranking()
        label = "group" if self.get_groups() is not None else "feature"
        return pd.DataFrame(cols, index=pd.Index(names, name=label))

    def __repr__(self) -> str:
        try:
            n_units = len(self.get_score())
            unit = "groups" if self.get_groups() is not None else "features"
            n_sig = int(np.sum(np.asarray(self.get_reject_null())))
            meta = []
            if self.target is not None:
                meta.append(f"target {self.target}")
            if self.alpha is not None:
                meta.append(f"alpha = {self.alpha}")
            meta.append(str(self.get_alternative()))
            lines = [f"ConfIntResults ({', '.join(m for m in meta if m)})"]
            lines.append(self._repr_getters())
            lines.append(f"  Units: {n_units} {unit} | significant: {n_sig}")
            margin = self.get_margin()
            margin_method = self.get_margin_method()
            margin_str = f"{float(margin):.4f}"
            if margin_method:
                margin_str += f" ({margin_method})"
            lines.append(
                f"  Practical margin: {margin_str} | multiple testing: "
                f"{self.get_multitest_method() or 'none'}"
            )
            lines.append(
                "  Use .to_frame() for the full table, .significant() for the "
                "significant units, or explainer.summary() for a formatted report."
            )
            return "\n".join(lines)
        except Exception:
            text = f"{type(self).__name__} ({len(getattr(self, '_data', {}))} fields)"
            if hasattr(self, "_data"):
                text += "\n" + self._repr_getters()
            return text
