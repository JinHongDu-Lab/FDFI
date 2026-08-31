Explainers
==========

Overview
--------

The ``fdfi.explainers`` module provides classes for computing flow-disentangled
feature importance.  Three explainer variants are implemented —
``OTExplainer`` (Gaussian OT), ``EOTExplainer`` (entropic OT), and
``FlowExplainer`` (normalizing flows) — plus the ``Crossfitting`` wrapper for
cross-fitted inference.  All three are model-agnostic: they wrap any callable
``f(X) -> y``, so they apply equally to tree ensembles, linear models, and
arbitrary black boxes.

Shared Methods
--------------

The following methods are available on all working explainer classes
(``OTExplainer``, ``EOTExplainer``, ``FlowExplainer``, ``Crossfitting``):

``__call__(X_test)``
    Compute per-sample UEIFs and aggregate them to ``phi_X`` / ``phi_Z`` plus
    standard errors.  Returns a ``dict``; also stores ``explainer.ueifs_X``
    and ``explainer.ueifs_Z``.

``conf_int(alpha=0.05, alternative='two-sided', margin=0.0, multitest_method=None, ...)``
    Compute confidence intervals and p-values from the stored UEIFs.  See
    :doc:`../user_guide/statistical_inference` for detailed usage.

``summary(feature_names=None)``
    Print a tabular summary of feature importances, standard errors, and
    significance.

``diagnose(X_orig=None, Z_full=None, report_title='')``
    (Re-)compute latent independence (dCor) and distribution fidelity (MMD)
    diagnostics.

Loss Functions
--------------

The importance score is defined through a per-sample loss ``L(y_true, y_pred)``.
All working explainers accept a ``loss`` argument (default: squared error) and a
``method`` argument (``'cpi'`` or ``'scpi'``) selecting the resampling
functional defined below.

- **Regression losses:** ``'squared_error'`` (``'l2'``/``'mse'``),
  ``'absolute_error'`` (``'l1'``/``'mae'``), ``'huber'``, ``'pinball'``.
- **Classification losses:** ``'log_loss'`` (``'bce'``/``'cross_entropy'``),
  ``'brier'``, ``'zero_one'`` — the model must output a probability ``P(y=1)``.
- **Custom:** any callable ``loss(y_true, y_pred)`` returning the per-sample loss.

Passing true labels ``y`` at call time (``explainer(X_test, y=y_test)``) uses the
loss-difference (DFI) form. If ``y`` is omitted, a label-free form referencing the
model's own prediction is used (prediction shift for regression losses; a Bregman
divergence such as KL for proper scoring rules). Non-proper losses like
``'zero_one'`` should be used with ``y``.

.. code-block:: python

   from fdfi.explainers import OTExplainer

   # Binary classification with log loss (model must output probabilities)
   def prob_model(X):
       return clf.predict_proba(X)[:, 1]

   explainer = OTExplainer(prob_model, X_background, loss="log_loss")
   results = explainer(X_test, y=y_test)

.. automodule:: fdfi.losses
   :members: resolve_loss, available_losses, squared_error, absolute_error, huber, pinball, log_loss, brier, zero_one

Base Explainer
--------------

.. autoclass:: fdfi.explainers.Explainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Gaussian Optimal Transport (OTExplainer)
----------------------------------------

The ``OTExplainer`` implements Gaussian optimal-transport DFI (Disentangled 
Feature Importance) without cross-fitting. This is the recommended starting 
point for most use cases.

.. autoclass:: fdfi.explainers.OTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example:**

.. code-block:: python

   import numpy as np
   from fdfi.explainers import OTExplainer

   # Create model and data
   def model(X):
       return X[:, 0] + 2 * X[:, 1]

   X_background = np.random.randn(100, 10)
   X_test = np.random.randn(10, 10)

   # Create explainer and compute importance
   explainer = OTExplainer(model, data=X_background, nsamples=50)
   results = explainer(X_test)

   print("Feature importance (X-space):", results["phi_X"])
   print("Standard errors:", results["se_X"])

   # Compute confidence intervals with FDR control (Benjamini-Hochberg)
   ci = explainer.conf_int(multitest_method='fdr_bh', alpha=0.05)
   print("Significant features after FDR control:", np.where(ci["reject_null"])[0])
   print("Adjusted p-values:", ci["pvalue_adj"])

Entropic Optimal Transport (EOTExplainer)
-----------------------------------------

The ``EOTExplainer`` uses entropic optimal transport with Sinkhorn iterations. 
It supports adaptive epsilon, stochastic transport sampling, and both Gaussian 
and empirical transport targets.

.. autoclass:: fdfi.explainers.EOTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example with advanced options:**

.. code-block:: python

   from fdfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       X_background,
       auto_epsilon=True,           # Adaptive regularization
       stochastic_transport=True,   # Sample from transport kernel
       n_transport_samples=10,      # Number of transport samples
       target="gaussian",           # or "empirical"
   )
   results = explainer(X_test)

Shared Disentanglement Diagnostics
----------------------------------

``OTExplainer``, ``EOTExplainer``, and ``FlowExplainer`` expose a shared
diagnostics interface via:

- ``explainer.diagnostics`` (computed at setup by default)
- ``explainer.diagnose(...)`` (recompute manually)

The diagnostics dictionary contains:

- ``latent_independence_dcor`` (pairwise dCor matrix)
- ``latent_independence_median`` and ``latent_independence_label``
- ``distribution_fidelity_mmd`` and ``distribution_fidelity_label``

.. code-block:: python

   diag = explainer.diagnostics
   # or: diag = explainer.diagnose()
   print(diag["latent_independence_median"], diag["latent_independence_label"])
   print(diag["distribution_fidelity_mmd"], diag["distribution_fidelity_label"])

Flow-Based DFI (FlowExplainer)
------------------------------

The ``FlowExplainer`` implements Flow-Disentangled Feature Importance using 
normalizing flows. It supports both CPI (Conditional Permutation Importance) 
and SCPI (Sobol-CPI), using the definitions in the FDFI paper:

- **CPI**: Average per-resample loss differences and multiply by one half:
  :math:`\tfrac12\mathbb{E}_b[L(Y,f(\tilde X_b))-L(Y,f(X))]`.
- **SCPI**: Average the counterfactual predictions before applying the loss:
  :math:`L(Y,\mathbb{E}_b[f(\tilde X_b)])-L(Y,f(X))`.

Both use the configurable ``loss`` (default squared error). For the same finite
Monte Carlo draws and squared-error loss,
:math:`2\widehat\phi^{CPI}=\widehat\phi^{SCPI}+\mathrm{Var}_b[f(\tilde X_b)]`.
With an exact disentangling map and a Bayes predictor, CPI and infinite-resample
SCPI estimate the same population importance. Conventional CPI in the broader
CPI literature omits the one-half normalization and is therefore twice the
package's CPI.

.. autoclass:: fdfi.explainers.FlowExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example with CPI (default):**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       X_background,
       fit_flow=True,           # Fit normalizing flow during init
       method='cpi',            # CPI (default)
       num_steps=200,           # Flow training iterations
       nsamples=50,             # Monte Carlo samples
       random_state=42,
   )
   results = explainer(X_test)

   print("Z-space importance (CPI):", results["phi_Z"])
   print("Confidence intervals:")
   ci = explainer.conf_int(alpha=0.05, target="Z")

**Example with SCPI (Sobol-CPI):**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       X_background,
       fit_flow=True,
       method='scpi',           # SCPI (Sobol-CPI)
       num_steps=200,
       nsamples=50,
   )
   results = explainer(X_test)

   print("Importance (SCPI):", results["phi_Z"])

**Using external flow models:**

.. code-block:: python

   from fdfi.explainers import FlowExplainer
   from fdfi.models import FlowMatchingModel

   # Train flow externally
   flow = FlowMatchingModel(X_background, dim=X_background.shape[1])
   flow.fit(num_steps=500, verbose='final')

   # Use in explainer
   explainer = FlowExplainer(model.predict, X_background, fit_flow=False)
   explainer.set_flow(flow)
   results = explainer(X_test)

DFIExplainer Alias
------------------

``DFIExplainer`` is an alias for ``OTExplainer`` for backward compatibility:

.. data:: fdfi.explainers.DFIExplainer

   Alias for :class:`OTExplainer`.

Cross-Fitting (Crossfitting)
----------------------------

The ``Crossfitting`` class wraps any of the above explainers and performs
K-fold cross-fitting so that the disentanglement map is never evaluated on
its own training data.  This yields valid standard errors and confidence
intervals even when the sample size is small.

.. autoclass:: fdfi.explainers.Crossfitting
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example — cross-fitted OTExplainer (default KFold):**

.. code-block:: python

   from fdfi.explainers import Crossfitting, OTExplainer

   cf = Crossfitting(
       model.predict,
       data=X_background,
       explainer_class=OTExplainer,
       cv=5,                    # 5-fold KFold (default)
       nsamples=50,
       random_state=42,
   )
   results = cf()               # cross-fit on X_background
   ci = cf.conf_int(alpha=0.05)
   cf.summary()

**Example — using a custom sklearn splitter:**

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold, ShuffleSplit
   from fdfi.explainers import Crossfitting, EOTExplainer

   # Stratified K-Fold (preserves class balance)
   cf = Crossfitting(
       model.predict, X_background,
       explainer_class=EOTExplainer,
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
       y=y_train,               # required for stratification
       nsamples=50,
   )
   results = cf()

   # ShuffleSplit (random train/test splits)
   cf = Crossfitting(
       model.predict, X_background,
       explainer_class=OTExplainer,
       cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
   )
   results = cf()

**Ensemble prediction on new data:**

.. code-block:: python

   # After cross-fitting, predict on unseen data
   results_new = cf(X_test)     # averages across all fold explainers

Group Importance
----------------

All explainer classes support group-level feature importance via the
``groups`` argument in ``conf_int()``. After running an explainer (so that
per-sample UEIFs are available), call ``conf_int(groups=...)`` to obtain
group-level importance, standard errors, and p-values.

Groups can be specified as:

- A ``dict`` mapping group names to lists of feature indices.
- A 1-D ``numpy`` array of group labels (one per feature).
- A binary ``pandas.DataFrame`` (features × groups) — features may belong
  to multiple groups.

**Example — dict input:**

.. code-block:: python

   from fdfi.explainers import OTExplainer

   explainer = OTExplainer(model.predict, X_background, nsamples=50)
   explainer(X_test)

   groups = {"signal": [0, 1, 2], "noise": [3, 4, 5, 6, 7, 8, 9]}
   res = explainer.conf_int(groups=groups)

   for name, imp, se, p in zip(
       res["groups"], res["score"], res["se"], res["pvalue"]
   ):
       print(f"{name}: importance={imp:.4f}  se={se:.4f}  p={p:.4f}")

**Example — pandas DataFrame (overlapping groups):**

.. code-block:: python

   import pandas as pd

   # Features can belong to multiple groups
   df_groups = pd.DataFrame({
       "group_A": [1, 1, 0, 0, 0],
       "group_B": [0, 1, 1, 0, 0],   # feature 1 in both A and B
       "group_C": [0, 0, 0, 1, 1],
   })
   res = explainer.conf_int(groups=df_groups)

**Example — with Crossfitting:**

.. code-block:: python

   from fdfi.explainers import Crossfitting, OTExplainer

   cf = Crossfitting(model.predict, X_background, cv=5, nsamples=50)
   cf()  # cross-fit first
   res = cf.conf_int(groups={"signal": [0, 1, 2], "noise": [3, 4]})
