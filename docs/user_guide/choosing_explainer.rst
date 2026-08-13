Choosing an Explainer
=====================

DFI provides several explainer classes for different use cases. This guide 
helps you choose the right one.

Quick Decision Guide
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Situation
     - Recommended
     - Notes
   * - General use, continuous data
     - ``OTExplainer``
     - Fast, stable, good default
   * - Non-Gaussian data
     - ``EOTExplainer``
     - Adaptive epsilon, more flexible
   * - Complex multimodal data
     - ``FlowExplainer``
     - Learns data distribution via normalizing flow
   * - Small sample / valid inference
     - ``Crossfitting``
     - Wraps any explainer with K-fold cross-fitting
   * - Mixed data types
     - ``EOTExplainer`` with Gower
     - Use ``cost_metric="gower"``
   * - Any black-box model
     - ``OTExplainer``
     - Model-agnostic; needs only a ``f(X) -> y`` callable

.. note::

   All three variants are model-agnostic — they wrap any callable
   ``f(X) -> y`` — so there is no separate explainer to pick for tree
   ensembles, linear models, or arbitrary black boxes. Choose based on the
   *data* distribution, not the model type.

OTExplainer (Gaussian OT)
-------------------------

**Best for:** Continuous data that is approximately Gaussian

**Pros:**

- Fast closed-form computation
- Stable and reliable
- Good starting point for most problems

**Cons:**

- Assumes Gaussian structure
- May be suboptimal for heavy-tailed or multimodal data

**Example:**

.. code-block:: python

   from fdfi.explainers import OTExplainer

   explainer = OTExplainer(
       model.predict,
       data=X_background,
       nsamples=50,            # Monte Carlo samples per feature
       sampling_method="resample",  # or "permutation", "normal"
   )
   results = explainer(X_test)

EOTExplainer (Entropic OT)
--------------------------

**Best for:** Non-Gaussian, multimodal, or mixed-type data

**Pros:**

- Relaxes Gaussian assumption
- Adaptive regularization (``auto_epsilon=True``)
- Supports categorical features via Gower distance
- Stochastic transport for variance reduction

**Cons:**

- Slower than Gaussian OT
- More hyperparameters to tune

**Key options:**

.. code-block:: python

   from fdfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       data=X_background,
       # Regularization
       auto_epsilon=True,      # Auto-tune from median distance
       epsilon=0.1,            # Manual epsilon (if auto_epsilon=False)
       
       # Transport target
       target="gaussian",      # or "empirical"
       
       # Stochastic transport
       stochastic_transport=True,
       n_transport_samples=10,
       
       # Cost function for mixed data
       cost_metric="sqeuclidean",  # or "gower", "auto"
   )

FlowExplainer (Flow-Based DFI)
------------------------------

**Best for:** Complex, non-Gaussian data where normalizing flows can capture the
underlying distribution structure

**Pros:**

- Handles complex, multimodal distributions
- Maps data to Gaussian latent space via learned normalizing flow
- Supports both CPI and SCPI (Sobol-CPI) methods with different averaging orders
- Flexible flow training and pre-trained model support

**Cons:**

- Requires PyTorch and torchdiffeq dependencies
- Flow training can be slow for large datasets

**Key options:**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       data=X_background,
       
       # Flow fitting
       fit_flow=True,          # Fit flow during init (or fit later)
       num_steps=200,          # Flow training iterations
       
       # Method selection
       method='cpi',           # 'cpi', 'scpi', or 'both'
       
       # Counterfactual sampling
       nsamples=50,            # Monte Carlo samples per feature
       sampling_method='resample',  # 'resample', 'permutation', 'normal', 'condperm'
       
       # Reproducibility
       random_state=42,
   )
   
   results = explainer(X_test)

**Understanding CPI vs SCPI:**

- **CPI (Conditional Permutation Importance)**: Average the counterfactual
  prediction first, then apply the loss:
  
  .. math::
  
     \phi_j^{CPI} = L\big(Y, E_b[f(\tilde{X}_b^{(j)})]\big) - L\big(Y, f(X)\big)
  
- **SCPI (Sobol-CPI)**: Apply the loss per Monte Carlo sample first, then
  average:
  
  .. math::
  
     \phi_j^{SCPI} = E_b\big[L\big(Y, f(\tilde{X}_b^{(j)})\big)\big] - L\big(Y, f(X)\big)

The ``method`` argument (``'cpi'`` or ``'scpi'``) is available on ``OTExplainer``,
``EOTExplainer``, and ``FlowExplainer``.

**Choosing a loss:**

The importance score is defined through a loss ``L(y_true, y_pred)``. By default
this is the squared error, so the score is the classic difference of L2
residuals. The ``loss`` argument (orthogonal to ``method``) selects another
loss:

- **Regression:** ``'squared_error'`` (``'l2'``/``'mse'``), ``'absolute_error'``
  (``'l1'``/``'mae'``), ``'huber'``, ``'pinball'`` (``'quantile'``).
- **Binary classification:** ``'log_loss'`` (``'bce'``/``'cross_entropy'``),
  ``'brier'``, ``'zero_one'`` — the model must output a probability ``P(y=1)``.
- **Custom:** any callable ``loss(y_true, y_pred)`` returning the per-sample loss.

Passing the true labels ``y`` at call time uses the loss-difference (DFI) form,
which is preferred for interpretability (null features ~0):

.. code-block:: python

   explainer = OTExplainer(model, X_background, loss="log_loss")
   results = explainer(X_test, y=y_test)   # DFI loss-difference form

If ``y`` is omitted, a label-free form referencing the model's own prediction is
used instead: :math:`\operatorname{agg}_b L(\hat{Y}, f(\tilde{X}_b)) - L(\hat{Y}, \hat{Y})`.
This is the prediction shift for regression losses and a Bregman divergence
(e.g. KL for log-loss) for proper scoring rules. Non-proper losses such as
``'zero_one'`` should always be used with ``y``.

**External flow models:**

.. code-block:: python

   from fdfi.models import FlowMatchingModel

   # Train flow externally with custom settings
   flow = FlowMatchingModel(X_background, dim=X_background.shape[1])
   flow.fit(num_steps=500, verbose='final')

   # Use pre-trained flow in explainer
   explainer = FlowExplainer(model.predict, X_background, fit_flow=False)
   explainer.set_flow(flow)

Shared Diagnostics (OT / EOT / Flow)
------------------------------------

All disentangled explainers expose a shared ``diagnostics`` payload:

- ``latent_independence_median`` with qualitative label
- ``distribution_fidelity_mmd`` with qualitative label

Lower is better for both metrics. Labels use the same thresholds across
explainers:

- ``GOOD``: dCor < 0.10, MMD < 0.05
- ``MODERATE``: dCor < 0.25, MMD < 0.15
- ``POOR``: otherwise

.. code-block:: python

   explainer = OTExplainer(model.predict, X_background)
   diag = explainer.diagnostics
   print(diag["latent_independence_median"], diag["latent_independence_label"])
   print(diag["distribution_fidelity_mmd"], diag["distribution_fidelity_label"])

Crossfitting (Cross-Fitted Inference)
-------------------------------------

**Best for:** Small-to-moderate sample sizes where valid confidence intervals
are critical

**Pros:**

- Eliminates overfitting bias in the disentanglement map
- Yields valid standard errors and CIs even at small *n*
- Works with any explainer class (``OTExplainer``, ``EOTExplainer``,
  ``FlowExplainer``)
- Supports any scikit-learn cross-validation splitter (``KFold``,
  ``StratifiedKFold``, ``ShuffleSplit``, ``RepeatedKFold``, ``GroupKFold``,
  etc.)

**Cons:**

- K× slower than a single explainer (fits one per fold)
- For ``FlowExplainer`` folds, this means K separate flow trainings

**Key options:**

.. code-block:: python

   from fdfi.explainers import Crossfitting, OTExplainer
   from sklearn.model_selection import RepeatedKFold

   # Default: 5-fold KFold
   cf = Crossfitting(
       model.predict,
       data=X_background,
       explainer_class=OTExplainer,
       cv=5,
       nsamples=50,
       random_state=42,
   )
   results = cf()          # cross-fit on X_background
   ci = cf.conf_int(alpha=0.05)
   cf.summary()

   # RepeatedKFold for lower-variance estimates
   cf = Crossfitting(
       model.predict, X_background,
       explainer_class=OTExplainer,
       cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=0),
       nsamples=50,
   )
   results = cf()

Hyperparameter Guidelines
-------------------------

nsamples
~~~~~~~~

Number of Monte Carlo samples for counterfactual estimation.

- **Low (10-30)**: Fast but high variance
- **Medium (50-100)**: Good balance (recommended)
- **High (200+)**: Low variance but slow

sampling_method
~~~~~~~~~~~~~~~

How to generate counterfactual feature values:

- ``"resample"``: Sample from background data (default, preserves marginal)
- ``"permutation"``: Permute within test set (no new values)
- ``"normal"``: Sample from standard normal (strong Gaussian assumption)

epsilon (EOTExplainer)
~~~~~~~~~~~~~~~~~~~~~~

Entropic regularization strength:

- **Small (0.01)**: Sharp transport, may be unstable
- **Medium (0.1)**: Good balance
- **Large (1.0+)**: Smooth transport, loses structure
- **auto_epsilon=True**: Recommended, auto-tunes from data

target (EOTExplainer)
~~~~~~~~~~~~~~~~~~~~~

Transport target distribution:

- ``"gaussian"``: Standard normal target (default)
- ``"empirical"``: Permuted data as target

Statistical Inference: One-sided Tests and FDR
----------------------------------------------

All working explainers support :meth:`~fdfi.explainers.Explainer.conf_int`
with optional one-sided tests and multiple-testing correction.  Key options:

.. code-block:: python

   # Two-sided test (default)
   ci = explainer.conf_int(alpha=0.05)

   # One-sided (only care about positive importance)
   ci = explainer.conf_int(alpha=0.05, alternative="greater")

   # With Benjamini-Hochberg FDR correction
   ci = explainer.conf_int(alpha=0.05, alternative="greater", multitest_method="fdr_bh")

For full guidance see :doc:`statistical_inference`.

Computing Confidence Intervals
------------------------------

All explainers support post-hoc confidence intervals:

.. code-block:: python

   results = explainer(X_test)
   ci = explainer.conf_int(
       alpha=0.05,
       alternative="two-sided",     # or "greater", "less"
       margin=0.0,                  # Practical significance threshold
   )

   print("Significant features:", np.where(ci["reject_null"])[0])
