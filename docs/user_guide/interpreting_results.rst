Interpreting Results
====================

After calling an explainer, FDFI returns two kinds of objects: the
``ExplanationResults`` from ``explainer(X_test)`` and the ``ConfIntResults``
from ``explainer.conf_int()``. This page explains what every getter
means and when to prefer X-space versus Z-space attribution.

.. contents:: On this page
   :local:
   :depth: 2

----

The ExplanationResults Object
-----------------------------

Calling any working explainer returns an object with the following getters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Getter
     - Shape
     - Description
   * - ``get_phi_X()``
     - ``(d,)``
     - Mean UEIF (uncentered efficient influence function) in the **original
       feature space**.  Summarises how much each original feature contributes
       to prediction variance across the test set.
   * - ``get_se_X()``
     - ``(d,)``
     - Standard error of ``phi_X`` (across test samples).  Use this for error
       bars in :func:`~fdfi.plots.summary_bar` and confidence intervals in
       :meth:`~fdfi.explainers.Explainer.conf_int`.
   * - ``get_std_X()``
     - ``(d,)``
     - Standard deviation of per-sample UEIFs in X-space (``se_X * sqrt(n)``).
   * - ``get_phi_Z()``
     - ``(d,)``
     - Mean UEIF in the **latent (disentangled) space**.  Because features are
       approximately independent in Z-space, each ``phi_Z[j]`` measures the
       *intrinsic* contribution of latent dimension *j* without correlation
       confounding.
   * - ``get_se_Z()``
     - ``(d,)``
     - Standard error of ``phi_Z``.
   * - ``get_std_Z()``
     - ``(d,)``
     - Standard deviation of per-sample UEIFs in Z-space.

Per-Sample UEIFs
~~~~~~~~~~~~~~~~

After calling ``explainer(X_test)`` the explainer also stores per-sample
attributions as instance attributes:

* ``explainer.ueifs_X``  — shape ``(n_test, d)``
* ``explainer.ueifs_Z``  — shape ``(n_test, d)``

These are used by :func:`~fdfi.plots.summary_plot` (beeswarm),
:func:`~fdfi.plots.waterfall_plot`, :func:`~fdfi.plots.force_plot`,
:func:`~fdfi.plots.dependence_plot`, and
:meth:`~fdfi.explainers.Explainer.conf_int` with ``groups=``.

.. code-block:: python

   results = explainer(X_test)

   # Global bar chart
   from fdfi.plots import summary_bar, summary_plot
   summary_bar(results.get_phi_X(), results.get_se_X(), feature_names=feature_names)

   # Per-sample beeswarm
   summary_plot(explainer.ueifs_X, features=X_test, feature_names=feature_names)

X-space vs. Z-space
~~~~~~~~~~~~~~~~~~~~

Both spaces answer different questions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - When to prefer it
   * - ``get_phi_X()`` / ``ueifs_X``
     - Importance of the *original observed* features.  Accounts for the
       correlation structure (correlated features share credit via the Jacobian).
       Use this for feature selection, model debugging, and reporting to
       stakeholders.
   * - ``get_phi_Z()`` / ``ueifs_Z``
     - Importance of the *latent independent* dimensions.  Each dimension's
       contribution is orthogonal to all others.  Use this for theoretical
       analysis or when you specifically want correlation-free attribution.

----

The ``conf_int()`` Object
------------------------

:meth:`~fdfi.explainers.Explainer.conf_int` returns a ``ConfIntResults`` with the
following getters (arrays have length ``d`` for feature-level output, or length
``G`` for group-level output):

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Getter
     - Type
     - Description
   * - ``get_score()``
     - ``ndarray (d,)``
     - Estimated feature importance — the mean UEIF (after optional null
       thresholding).  This is the point estimate used as the centre of the CI.
   * - ``get_se()``
     - ``ndarray (d,)``
     - Standard error after variance-floor regularisation.  The floor prevents
       near-zero SEs from inflating z-scores for unimportant features.
   * - ``get_zscore()``
     - ``ndarray (d,)``
     - Signed z-statistic: ``(score - margin) / se``.  Positive means feature
       importance exceeds the margin.
   * - ``get_ranking()``
     - ``ndarray[int] (d,)``
     - Rank by descending z-score; ``1`` = most important.
   * - ``get_ci_lower()``
     - ``ndarray (d,)``
     - Lower confidence-interval bound.  ``-inf`` when
       ``alternative='less'``.
   * - ``get_ci_upper()``
     - ``ndarray (d,)``
     - Upper confidence-interval bound.  ``+inf`` when
       ``alternative='greater'``.
   * - ``get_reject_null()``
     - ``ndarray[bool] (d,)``
     - ``True`` where the null hypothesis is rejected at significance level
       ``alpha``.
   * - ``get_pvalue()``
     - ``ndarray (d,)``
     - One- or two-sided p-value depending on ``alternative``.
   * - ``get_pvalue_adj()``
     - ``ndarray (d,)``
     - Multiple-testing-adjusted p-value.  Returns None unless
       ``multitest_method`` was specified (e.g. ``'fdr_bh'``).
   * - ``get_margin()``
     - ``float``
     - The null-hypothesis threshold used.  ``0.0`` by default (test against
       zero importance).
   * - ``get_margin_method()``
     - ``str``
     - How the margin was selected: ``'fixed'``, ``'gap'``, or ``'mixture'``.
   * - ``get_alternative()``
     - ``str``
     - ``'two-sided'``, ``'greater'``, or ``'less'``.
   * - ``get_groups()``
     - ``list[str]``
     - Group names.  Returns None unless ``groups=`` was passed to
       ``conf_int()``.

.. code-block:: python

   ci = explainer.conf_int(alpha=0.05, alternative="greater")
   lower, upper = ci.get_ci()

   import numpy as np
   significant = np.where(ci.get_reject_null())[0]
   print("Significant features:", significant)
   print("z-scores:", ci.get_zscore()[significant].round(2))
   print("p-values:", ci.get_pvalue()[significant].round(4))

Reading Diagnostics
--------------------

After initialisation, disentangled explainers (``OTExplainer``,
``EOTExplainer``, ``FlowExplainer``) populate ``explainer.diagnostics``.  Call
:meth:`~fdfi.explainers.Explainer.diagnose` to recompute on a custom subset.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Getter
     - Meaning
   * - ``latent_independence_median``
     - Median off-diagonal dCor.  Lower is better (features more independent).
   * - ``latent_independence_label``
     - ``'GOOD'`` < 0.10, ``'MODERATE'`` < 0.25, otherwise ``'POOR'``.
   * - ``distribution_fidelity_mmd``
     - MMD between background and reconstructed distributions.
   * - ``distribution_fidelity_label``
     - ``'GOOD'`` < 0.05, ``'MODERATE'`` < 0.15, otherwise ``'POOR'``.

A ``'POOR'`` disentanglement label means the latent space is not well
decorrelated; X-space and Z-space attributions may both be unreliable.  Try
:class:`~fdfi.explainers.EOTExplainer` or
:class:`~fdfi.explainers.FlowExplainer` for potentially better disentanglement.

.. seealso::

   :doc:`statistical_inference` — detailed guidance on confidence intervals,
   one-sided tests, variance floors, and FDR correction.

   :doc:`choosing_explainer` — when to use each explainer class.
