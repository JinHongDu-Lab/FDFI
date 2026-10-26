"""
Cross-fitted DFI explainer for valid inference at small sample sizes.
"""

import numpy as np
from typing import Optional, Union, Callable, Any
from .base import Explainer
from ..results import ExplanationResults
from .ot import OTExplainer


class Crossfitting(Explainer):
    """
    Cross-fitted DFI explainer for valid inference at small sample sizes.

    Wraps any Explainer subclass and performs cross-fitting using a
    scikit-learn cross-validation splitter.  The disentanglement map is
    fitted on the training portion of each split and importance is
    evaluated on the held-out portion.  Final estimates are the
    ensemble average of cross-fitted predictors.

    Parameters
    ----------
    model : callable
        The model to explain.  Takes (n, d) array, returns (n,) predictions.
    data : numpy.ndarray
        Full dataset.  Shape (n, d).
    explainer_class : type, default=OTExplainer
        The explainer class to instantiate per split.  Must be a subclass
        of Explainer (e.g., OTExplainer, EOTExplainer, FlowExplainer).
    cv : int or sklearn cross-validation splitter, default=5
        Controls how data is split for cross-fitting.
        Pass an ``int`` for ``KFold(n_splits=cv, shuffle=True)``,
        or any scikit-learn splitter instance (e.g. ``KFold``,
        ``StratifiedKFold``, ``ShuffleSplit``, ``RepeatedKFold``,
        ``GroupKFold``).  Any object implementing
        ``.split(X, y, groups)`` is accepted.
    y : array-like of shape (n,), optional
        Target / response variable.  Required only when using a stratified
        splitter so that fold assignment preserves class distribution.
    groups : array-like of shape (n,), optional
        Group labels for group-aware splitters (``GroupKFold``, etc.).
    random_state : int or None, default=0
        Seed for the default ``KFold`` splitter (when *cv* is int) and for the
        fold explainers, each of which gets its own seed spawned from it.
        ``None`` draws fresh entropy, so results are not reproducible.
    **kwargs : dict
        Additional keyword arguments forwarded to each split's explainer
        constructor (e.g., nsamples, epsilon, sampling_method, num_steps).

    Attributes
    ----------
    cv_ : sklearn splitter instance
        The resolved cross-validation splitter.
    fold_explainers : list[Explainer]
        The fitted explainer instances (one per split).
    fold_indices : list[tuple[numpy.ndarray, numpy.ndarray]]
        ``(train_idx, test_idx)`` for each split.
    ueifs_X : numpy.ndarray or None
        Per-sample X-space UEIFs, shape (n, d), after calling with
        ``X=None``.
    ueifs_Z : numpy.ndarray or None
        Per-sample Z-space UEIFs, shape (n, d), after calling with
        ``X=None``.
    """

    def _repr_lines(self) -> list:
        lines = super()._repr_lines()
        n_folds = getattr(getattr(self, "cv_", None), "n_splits", "?")
        explainer_class = getattr(self, "explainer_class", None)
        base = getattr(explainer_class, "__name__", str(explainer_class))
        lines.insert(2, f"  Cross-fitting: {base} on {n_folds} folds")
        return lines

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        explainer_class: type = OTExplainer,
        cv: Union[int, Any] = 5,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        cv_kwargs: Optional[dict] = None,
        random_state: Optional[int] = 0,
        **kwargs: Any,
    ):
        # Remove fit_flow from kwargs before passing to super() and child explainers
        # so it doesn't conflict with the hardcoded fit_flow=False below.
        kwargs.pop("fit_flow", None)
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.explainer_class = explainer_class
        self.y = np.asarray(y) if y is not None else None
        self.groups = np.asarray(groups) if groups is not None else None
        # cv_kwargs: extra keyword arguments forwarded to cv.split().
        # Overrides the defaults (y=self.y, groups=self.groups).
        # Use case: pass discrete class labels to StratifiedKFold when y
        # is continuous (e.g. standardized), e.g. cv_kwargs={"y": y_binary}.
        self.cv_kwargs = cv_kwargs or {}
        self.random_state = random_state
        self.cf_kwargs = kwargs

        self.cv_ = self._resolve_cv(cv)
        self.fold_explainers: list = []
        self.fold_indices: list = []
        self.ueifs_X: Optional[np.ndarray] = None
        self.ueifs_Z: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # CV resolution
    # ------------------------------------------------------------------

    def _resolve_cv(self, cv: Union[int, Any]) -> Any:
        """Resolve *cv* parameter to a scikit-learn splitter instance."""
        if isinstance(cv, int):
            from sklearn.model_selection import KFold
            return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        # Accept any object with a .split() method
        if not hasattr(cv, "split"):
            raise TypeError(
                f"cv must be an int or an object with a .split() method, "
                f"got {type(cv)}"
            )
        return cv

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> ExplanationResults:
        """
        Compute cross-fitted feature importance.

        If *X* is ``None``, performs full cross-fitting on ``self.data``:
        each split's test set is the held-out portion of the data.

        If *X* is provided, uses the ensemble of fitted fold explainers
        to compute importance on *X* and averages the results.

        Parameters
        ----------
        X : numpy.ndarray or None
            If None, cross-fit on ``self.data`` (recommended for valid
            inference).  If provided, shape (m, d), ensemble-predict on
            new data.

        Returns
        -------
        ExplanationResults
            Dict-compatible result with attribute access, ``importance()``,
            ``to_frame()`` and ``top_features()``. Same fields as OT / Flow:
            ``phi_X, std_X, se_X, phi_Z, std_Z, se_Z``.
        """
        if X is not None:
            return self._ensemble_predict(X)
        return self._crossfit()

    # ------------------------------------------------------------------
    # Internal: cross-fit on self.data
    # ------------------------------------------------------------------

    def _crossfit(self) -> ExplanationResults:
        """Cross-fit and return dict-compatible ExplanationResults.

        Returns
        -------
        ExplanationResults
            Aggregated importance and uncertainty for evaluated samples.
        """
        n, d = self.data.shape
        self.fold_explainers = []
        self.fold_indices = []

        # Detect whether the model is a sklearn-style estimator that should
        # be cloned and refitted per fold (true cross-fitting of the predictor).
        _is_estimator = hasattr(self.model, "fit") and hasattr(self.model, "predict")

        # Collect per-sample UEIFs; support overlapping test sets
        ueif_counts = np.zeros(n, dtype=int)
        ueifs_X_accum = np.zeros((n, d))
        ueifs_Z_accum = np.zeros((n, d))

        _split_kw = {"y": self.y, "groups": self.groups}
        _split_kw.update(self.cv_kwargs)
        # Each fold draws its own counterfactuals and coupling noise; a shared
        # seed would give rows in different folds identical draws.
        seeds = np.random.SeedSequence(self.random_state)
        for train_idx, test_idx in self.cv_.split(self.data, **_split_kw):
            self.fold_indices.append((train_idx, test_idx))

            if _is_estimator:
                from sklearn.base import clone
                if self.y is None:
                    raise ValueError(
                        "y must be provided to Crossfitting when model is a "
                        "sklearn estimator so it can be refitted per fold."
                    )
                fold_clf = clone(self.model)
                fold_clf.fit(self.data[train_idx], self.y[train_idx])
                if hasattr(fold_clf, "predict_proba") and hasattr(fold_clf, "classes_"):
                    if len(fold_clf.classes_) == 2:
                        fold_model = lambda X_, clf=fold_clf: clf.predict_proba(X_)[:, 1]
                    else:
                        fold_model = lambda X_, clf=fold_clf: clf.predict(X_)
                else:
                    fold_model = lambda X_, clf=fold_clf: clf.predict(X_)
            else:
                fold_model = self.model

            # Build fold explainer on FULL data so that covariance / whitening
            # uses all n observations (matches refit_cov=False in the reference).
            # This avoids rank-deficiency when d > n_train (half the data).
            # Diagnostics are suppressed because the latent pool of all n
            # samples is immediately replaced by the training fold's below.
            fold_kwargs = {"compute_diagnostics": False}
            fold_kwargs.update(self.cf_kwargs)
            fold_exp = self.explainer_class(
                model=fold_model,
                data=self.data,
                random_state=int(seeds.spawn(1)[0].generate_state(1)[0]),
                **fold_kwargs,
            )
            # Counterfactuals resample from the training fold only.
            fold_exp._set_pool(self.data[train_idx])
            self.fold_explainers.append(fold_exp)

            X_test = self.data[test_idx]
            y_test = self.y[test_idx] if self.y is not None else None
            ueifs_X_fold, ueifs_Z_fold = fold_exp._ueifs(X_test, y_test)

            # Accumulate (handles overlapping test sets by averaging later)
            for local_i, global_i in enumerate(test_idx):
                ueifs_X_accum[global_i] += ueifs_X_fold[local_i]
                ueifs_Z_accum[global_i] += ueifs_Z_fold[local_i]
                ueif_counts[global_i] += 1

        # Average for samples that appeared in multiple test sets
        seen = ueif_counts > 0
        ueifs_X_accum[seen] /= ueif_counts[seen, None]
        ueifs_Z_accum[seen] /= ueif_counts[seen, None]

        # Keep only the samples that were actually evaluated
        self.ueifs_X = ueifs_X_accum[seen]
        self.ueifs_Z = ueifs_Z_accum[seen]
        n_eff = int(seen.sum())

        # Null-threshold for _last_results: zero out features whose mean UEIF is
        # negative (matches the DFI paper: features with E[UEIF] < 0 have no
        # evidence of importance). Applied to a *copy* so that self.ueifs_X /
        # self.ueifs_Z remain unthresholded — conf_int(groups=...) can then apply
        # its own threshold_null flag on the raw per-sample arrays.
        ueifs_X_thresh = self.ueifs_X.copy()
        ueifs_Z_thresh = self.ueifs_Z.copy()
        for arr in (ueifs_X_thresh, ueifs_Z_thresh):
            null_mask = arr.mean(axis=0) < 0
            arr[:, null_mask] = 0.0

        # SE with n_folds correction to match the DFI paper:
        #   sqn_eff = sqrt(n) * sqrt((n_folds-1)/n_folds)
        # For n_folds=2 this gives sqn_eff = sqrt(n/2), which is sqrt(2)× larger
        # than the naive sqrt(n), resulting in more conservative inference.
        n_folds = getattr(self.cv_, "n_splits", 1)
        sqn_eff = np.sqrt(n_eff)
        if n_folds > 1:
            sqn_eff *= np.sqrt((n_folds - 1) / n_folds)

        ddof = 1 if n_eff > 1 else 0
        results = {
            "phi_X": ueifs_X_thresh.mean(axis=0),
            "std_X": ueifs_X_thresh.std(axis=0),
            "se_X": ueifs_X_thresh.std(axis=0, ddof=ddof) / sqn_eff,
            "phi_Z": ueifs_Z_thresh.mean(axis=0),
            "std_Z": ueifs_Z_thresh.std(axis=0),
            "se_Z": ueifs_Z_thresh.std(axis=0, ddof=ddof) / sqn_eff,
        }
        self._cache_results(results, n_eff)
        return ExplanationResults(
            results, explainer_name=type(self).__name__, n=n_eff
        )

    # ------------------------------------------------------------------
    # Internal: ensemble prediction on new data
    # ------------------------------------------------------------------

    def _ensemble_predict(self, X: np.ndarray) -> ExplanationResults:
        """Predict with the fitted folds.

        Returns
        -------
        ExplanationResults
            Dict-compatible importance and uncertainty on the new data.
        """
        if not self.fold_explainers:
            # No fold explainers yet — run cross-fitting first
            self._crossfit()

        n = X.shape[0]
        phi_X_list, phi_Z_list = [], []
        se_X_list, se_Z_list = [], []

        for fold_exp in self.fold_explainers:
            r = fold_exp(X)
            phi_X_list.append(r["phi_X"])
            phi_Z_list.append(r["phi_Z"])
            se_X_list.append(r["se_X"])
            se_Z_list.append(r["se_Z"])

        K = len(self.fold_explainers)
        phi_X = np.mean(phi_X_list, axis=0)
        phi_Z = np.mean(phi_Z_list, axis=0)

        # Pooled SE: sqrt( mean of se_k^2 ) — accounts for within-fold variance
        se_X = np.sqrt(np.mean(np.array(se_X_list) ** 2, axis=0))
        se_Z = np.sqrt(np.mean(np.array(se_Z_list) ** 2, axis=0))

        std_X = se_X * np.sqrt(n)
        std_Z = se_Z * np.sqrt(n)

        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return ExplanationResults(
            results, explainer_name=type(self).__name__, n=n
        )
