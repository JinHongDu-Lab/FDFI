"""
Result containers returned by fdfi explainers.

Both classes subclass :class:`dict`, so existing code that indexes results
with keys (e.g. ``results["phi_X"]``) keeps working, while users also get
attribute access, convenience methods and an informative printed summary.
"""

from typing import Any, Dict, List, Optional, Sequence

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


class _ResultBase(dict):
    """Dict subclass with attribute access and a readable printed form."""

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute or key '{name}'"
            ) from None

    def __dir__(self):
        keys = {k for k in self.keys() if isinstance(k, str)}
        return sorted(set(super().__dir__()) | keys)

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_pretty_(self, p, cycle):  # IPython / Jupyter display
        p.text(self.__repr__())


class ExplanationResults(_ResultBase):
    """
    Feature importance estimates returned by calling an explainer,
    e.g. ``results = explainer(X_test)``.

    Keys / attributes
    -----------------
    phi_X, se_X, std_X : numpy.ndarray
        Importance, standard error and per-sample standard deviation of the
        UEIFs in the original feature (X) space.
    phi_Z, se_Z, std_Z : numpy.ndarray
        The same quantities in the disentangled latent (Z) space.

    Some explainers add further keys (e.g. ``phi_X_scpi``); all keys are
    listed in the printed summary.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        explainer_name: str = "",
        n: Optional[int] = None,
    ):
        super().__init__(data)
        self.explainer_name = explainer_name
        self.n = n

    @property
    def n_features(self) -> int:
        return len(self["phi_X"])

    def _check_target(self, target: str) -> str:
        if target not in ("X", "Z"):
            raise ValueError("target must be 'X' or 'Z'")
        return target

    def importance(self, target: str = "X") -> np.ndarray:
        """Return the importance vector in X-space (default) or Z-space."""
        return self[f"phi_{self._check_target(target)}"]

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
        phi = self[f"phi_{t}"]
        if feature_names is None:
            feature_names = list(range(len(phi)))
        elif len(feature_names) != len(phi):
            raise ValueError("feature_names must have one entry per feature")
        df = pd.DataFrame(
            {"estimate": phi, "se": self[f"se_{t}"], "std": self[f"std_{t}"]},
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
            if "phi_X" in self:
                meta.append(f"d = {self.n_features} features")
            lines = [head + (f" ({', '.join(meta)})" if meta else "")]
            lines.append("  Fields: " + ", ".join(str(k) for k in self.keys()))
            if "phi_X" in self and "se_X" in self:
                k = min(5, self.n_features)
                lines.append(f"  Top {k} features (X-space):")
                for j in self.top_features(k):
                    lines.append(
                        f"    feature {j}: {self['phi_X'][j]:.4f} "
                        f"(se {self['se_X'][j]:.4f})"
                    )
            lines.append(
                "  Use .to_frame() for the full table; explainer.conf_int() or "
                "explainer.summary() for inference."
            )
            return "\n".join(lines)
        except Exception:
            return f"{type(self).__name__} ({len(self)} fields)"


class ConfIntResults(_ResultBase):
    """
    Inference results returned by :meth:`Explainer.conf_int`.

    Keys / attributes
    -----------------
    score, se, zscore, ci_lower, ci_upper, pvalue, reject_null, ranking :
        One entry per feature (or per group when ``groups`` was given).
    pvalue_adj, multitest_method :
        Present when a multiple-testing correction was requested.
    groups :
        Group names, present for group-level inference.
    margin, margin_method, alternative :
        Settings used for the test.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        alpha: Optional[float] = None,
        target: Optional[str] = None,
    ):
        super().__init__(data)
        self.alpha = alpha
        self.target = target

    @property
    def names(self) -> list:
        """Feature indices, or group names for group-level results."""
        if "groups" in self:
            return list(self["groups"])
        return list(range(len(self["score"])))

    def significant(self) -> list:
        """Names (feature indices or group names) of units with ``reject_null``."""
        names = self.names
        return [names[i] for i in np.flatnonzero(np.asarray(self["reject_null"]))]

    def to_frame(self, feature_names: Optional[Sequence[str]] = None):
        """Return the inference table as a :class:`pandas.DataFrame`."""
        pd = _require_pandas()
        names = self.names if feature_names is None else list(feature_names)
        if len(names) != len(self["score"]):
            raise ValueError("feature_names must have one entry per unit")
        cols = {
            "estimate": self["score"],
            "se": self["se"],
            "ci_lower": self["ci_lower"],
            "ci_upper": self["ci_upper"],
            "zscore": self["zscore"],
            "pvalue": self["pvalue"],
        }
        if "pvalue_adj" in self:
            cols["pvalue_adj"] = self["pvalue_adj"]
        cols["reject_null"] = self["reject_null"]
        cols["ranking"] = self["ranking"]
        label = "group" if "groups" in self else "feature"
        return pd.DataFrame(cols, index=pd.Index(names, name=label))

    def __repr__(self) -> str:
        try:
            n_units = len(self["score"])
            unit = "groups" if "groups" in self else "features"
            n_sig = int(np.sum(np.asarray(self["reject_null"])))
            meta = []
            if self.target is not None:
                meta.append(f"target {self.target}")
            if self.alpha is not None:
                meta.append(f"alpha = {self.alpha}")
            meta.append(str(self.get("alternative", "")))
            lines = [f"ConfIntResults ({', '.join(m for m in meta if m)})"]
            lines.append(f"  Units: {n_units} {unit} | significant: {n_sig}")
            margin = self.get("margin", 0.0)
            margin_method = self.get("margin_method")
            margin_str = f"{float(margin):.4f}"
            if margin_method:
                margin_str += f" ({margin_method})"
            lines.append(
                f"  Practical margin: {margin_str} | multiple testing: "
                f"{self.get('multitest_method') or 'none'}"
            )
            lines.append(
                "  Use .to_frame() for the full table, .significant() for the "
                "significant units, or explainer.summary() for a formatted report."
            )
            return "\n".join(lines)
        except Exception:
            return f"{type(self).__name__} ({len(self)} fields)"
