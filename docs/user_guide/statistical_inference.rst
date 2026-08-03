Statistical Inference with FDFI
================================

FDFI provides CLT-based confidence intervals and hypothesis tests for feature
importances.  This page explains how inference works, the options available,
and how to use them correctly.

.. contents:: On this page
   :local:
   :depth: 2

----

How FDFI Inference Works
--------------------------

The unit-effect independent features (UEIFs) are defined per test sample.
Under regularity conditions, their sample mean ``phi`` is asymptotically
normal:

.. math::

   \frac{\hat{\phi}_j - \phi_j^*}{\hat{\sigma}_j / \sqrt{n}}
   \;\xrightarrow{d}\; \mathcal{N}(0,1)

where ``n`` is the number of test samples and ``hat{sigma}_j`` is the
estimated standard deviation of the per-sample UEIFs for feature *j*.  This
gives the standard error ``se_j = hat{sigma}_j / sqrt(n)`` used in confidence
intervals.

.. note::

   The central-limit-theorem approximation works best when ``n >= 30``.  With
   very small test sets the coverage of the reported CIs will be approximate.

----

Variance Floor
--------------

When many features have near-zero importance, their raw SEs can be
spuriously small, causing inflated z-scores.  FDFI addresses this by fitting
a :class:`~fdfi.utils.TwoComponentMixture` to the distribution of raw standard
errors and using the upper quantile of the *smaller* (noise) component as a
**variance floor**.  All SEs below the floor are clamped to the floor before
computing z-statistics.

This prevents noise-level importances from appearing statistically significant
and is especially useful when the number of features is large.

**Controlling the floor**:

.. code-block:: python

   ci = explainer.conf_int(
       alpha=0.05,
       variance_floor_q=0.90,     # quantile of the noise component (default 0.90)
       variance_floor_method="mixture",
   )

----

The Margin Parameter
--------------------

By default, ``conf_int()`` tests the null hypothesis ``phi_j = 0``.  A
positive ``margin`` shifts the null to ``phi_j = margin``, testing whether
feature importance is *practically significant* (not merely nonzero):

.. code-block:: python

   ci = explainer.conf_int(alpha=0.05, margin=0.05)

Three margin strategies are available:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``margin_method``
     - Behaviour
   * - ``'fixed'``
     - Use the provided ``margin`` value directly.
   * - ``'gap'``
     - Set the margin to the midpoint of the largest gap in the sorted point
       estimates.  Useful for separating important from unimportant features
       automatically.
   * - ``'mixture'``
     - Fit a :class:`~fdfi.utils.TwoComponentMixture` to the point estimates;
       use the upper quantile of the smaller component as the margin.

----

One-sided vs. Two-sided Tests
-------------------------------

The ``alternative`` parameter controls the direction of the test:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Value
     - Hypothesis
     - When to use
   * - ``'two-sided'``
     - ``phi != margin``
     - Default.  Detect both positive and negative importance, or features
       that may suppress the target.
   * - ``'greater'``
     - ``phi > margin``
     - You are only interested in *positive* contributions.  The CI has an
       open upper bound (+∞); the plot renders it with a ► caret.
   * - ``'less'``
     - ``phi < margin``
     - You are only interested in *negative* contributions.  The CI has an
       open lower bound (−∞); the plot renders it with a ◄ caret.

.. code-block:: python

   # One-sided test (greater)
   ci = explainer.conf_int(alpha=0.05, alternative="greater")

   from fdfi.plots import confidence_interval_plot
   confidence_interval_plot(ci, feature_names=feature_names)

For a side-by-side comparison see the
:doc:`../tutorials/confidence_intervals` tutorial.

----

Multiple-Testing Correction
-----------------------------

When testing many features simultaneously, the family-wise type-I error rate
is inflated.  Pass ``multitest_method`` to apply one of the
:func:`statsmodels.stats.multitest.multipletests` methods:

.. code-block:: python

   ci = explainer.conf_int(
       alpha=0.05,
       alternative="greater",
       multitest_method="fdr_bh",   # Benjamini-Hochberg FDR control
   )

   # Inspect adjusted p-values and rejection flags
   import pandas as pd
   pd.DataFrame({
       "feature":    feature_names,
       "score":      ci["score"].round(4),
       "pvalue_adj": ci["pvalue_adj"].round(4),
       "reject":     ci["reject_null"],
   })

Commonly used methods:

* ``'fdr_bh'`` — Benjamini-Hochberg (FDR control; recommended for feature selection).
* ``'bonferroni'`` — Bonferroni correction (family-wise error rate; conservative).
* ``'holm'`` — Holm-Bonferroni (family-wise, less conservative).

----

Cross-fitting
-------------

When the test set is small or the model is highly non-linear, the empirical
cross-covariance used for the Gaussian OT map can over-fit to the background
sample, inducing finite-sample bias in the attributions.  Cross-fitting
addresses this by alternating which half of the data is used to fit the OT
map vs. evaluate the UEIFs:

Cross-fitting is provided by the :class:`~fdfi.explainers.Crossfitting` wrapper,
which accepts any explainer class:

.. code-block:: python

   from fdfi.explainers import Crossfitting, OTExplainer

   cf = Crossfitting(
       model.predict,
       data=X_background,
       explainer_class=OTExplainer,   # or EOTExplainer / FlowExplainer
       cv=5,                          # 5-fold KFold, or any sklearn splitter
       nsamples=100,
       random_state=42,
   )
   results = cf()                     # cross-fit on X_background
   ci = cf.conf_int(alpha=0.05)

Cross-fitting is especially helpful when ``len(X_background)`` is small
(< 200) and the number of features is large.  See
:doc:`choosing_explainer` for the full list of ``Crossfitting`` options.

----

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import GradientBoostingRegressor
   from fdfi.explainers import OTExplainer
   from fdfi.plots import summary_bar, confidence_interval_plot

   rng = np.random.default_rng(42)
   n, d = 300, 8
   X = rng.standard_normal((n, d))
   y = X[:, 0] * 2 + X[:, 1] - X[:, 3] * 0.5 + rng.standard_normal(n) * 0.5

   model = GradientBoostingRegressor(n_estimators=100, random_state=42)
   model.fit(X[:200], y[:200])

   feature_names = [f"F{i}" for i in range(d)]
   explainer = OTExplainer(model.predict, data=X[:200], nsamples=80)
   results = explainer(X[200:])

   # Bar chart with error bars
   summary_bar(results["phi_X"], results["se_X"], feature_names=feature_names)

   # One-sided confidence intervals (greater) with FDR correction
   ci = explainer.conf_int(
       alpha=0.05,
       alternative="greater",
       multitest_method="fdr_bh",
   )
   confidence_interval_plot(ci, feature_names=feature_names)

.. seealso::

   :doc:`interpreting_results` — reference table for all output dictionary
   keys.

   :doc:`../tutorials/confidence_intervals` — worked tutorial with visual
   comparisons of one-sided and two-sided CIs.
