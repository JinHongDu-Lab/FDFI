Frequently Asked Questions
==========================

General Questions
-----------------

What does DFI stand for?
~~~~~~~~~~~~~~~~~~~~~~~~

**D**\ isentangled **F**\ eature **I**\ mportance. It's a framework for
computing feature importance by mapping correlated features to a latent space
where they are independent, measuring importance there, and mapping the result
back.  Three variants are implemented, differing only in how that map is built:
Gaussian OT (:class:`~fdfi.explainers.OTExplainer`), entropic OT
(:class:`~fdfi.explainers.EOTExplainer`), and normalizing flows
(:class:`~fdfi.explainers.FlowExplainer`).  See :doc:`choosing_explainer`.

How is DFI different from SHAP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SHAP uses Shapley values from game theory, computing the average marginal 
contribution of each feature across all possible feature orderings. DFI uses 
**optimal transport** to create counterfactual distributions, specifically:

1. Transform to an uncorrelated representation (Z-space)
2. Replace each feature with an independent sample
3. Measure the change in model output

Both measure feature importance but use different mathematical frameworks.

When should I use DFI vs SHAP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use DFI when:

- You have continuous or mixed-type data
- You want built-in confidence intervals and p-values
- You prefer an OT-based interpretation

Use SHAP when:

- You need exact Shapley values
- You're using tree models (TreeSHAP is very fast)
- You want the game-theoretic interpretation

Explainer Questions
-------------------

Which explainer should I start with?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with **OTExplainer**. It's fast, stable, and works well for most 
continuous data problems.

.. code-block:: python

   from fdfi.explainers import OTExplainer
   explainer = OTExplainer(model.predict, data=X_train, nsamples=50)
   results = explainer(X_test)

Why does my explainer raise NotImplementedError?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some explainers (TreeExplainer, LinearExplainer, KernelExplainer) are 
placeholder implementations. Use **OTExplainer** or **EOTExplainer** for 
working implementations.

How do I handle categorical features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use EOTExplainer with Gower distance:

.. code-block:: python

   from fdfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       data=X_train,
       cost_metric="gower",  # or "auto"
       categorical_threshold=10,  # Features with ≤10 unique values = categorical
   )

You can also manually specify feature types:

.. code-block:: python

   import numpy as np
   feature_types = np.array(["continuous", "binary", "categorical", "continuous"])
   
   explainer = EOTExplainer(
       model.predict,
       data=X_train,
       cost_metric="gower",
       feature_types=feature_types,
   )

Statistical Inference Questions
-------------------------------

How do I get confidence intervals?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``conf_int()`` method after computing importance:

.. code-block:: python

   results = explainer(X_test)
   ci = explainer.conf_int(alpha=0.05, alternative="two-sided")

   print("Estimates:", ci["score"])
   print("CI Lower:", ci["ci_lower"])
   print("CI Upper:", ci["ci_upper"])
   print("P-values:", ci["pvalue"])

What is the variance floor?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **variance floor** is a minimum standard error applied to prevent
confidence intervals from being too narrow for features with very small
variance. This improves coverage and statistical stability.  A
:class:`~fdfi.utils.TwoComponentMixture` is fitted to the raw standard errors;
the upper quantile of the *smaller* (noise) component is used as the floor.

.. code-block:: python

   ci = explainer.conf_int(
       variance_floor_q=0.90,       # quantile of the noise component (default)
   )

What is the practical margin?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **margin** is a threshold for practical significance. Instead of testing
:math:`H_0: \phi_j = 0`, you can test :math:`H_0: \phi_j \leq \delta` where
:math:`\delta` is a meaningful effect size.

.. code-block:: python

   ci = explainer.conf_int(
       margin=0.01,  # Only significant if importance > 0.01
       alternative="greater",
   )

How do I run a one-sided test?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``alternative='greater'`` (or ``'less'``) to :meth:`~fdfi.explainers.Explainer.conf_int`.
The resulting CI has an open bound (``+inf`` or ``-inf``), and
:func:`~fdfi.plots.confidence_interval_plot` renders it automatically with a
caret stub.

.. code-block:: python

   ci = explainer.conf_int(alpha=0.05, alternative="greater")
   from fdfi.plots import confidence_interval_plot
   confidence_interval_plot(ci, feature_names=feature_names)

See :doc:`statistical_inference` for a full discussion of one-sided tests,
variance floors, and FDR correction.

How do I correct for multiple testing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``multitest_method`` to :meth:`~fdfi.explainers.Explainer.conf_int`:

.. code-block:: python

   ci = explainer.conf_int(alpha=0.05, multitest_method="fdr_bh")
   # ci["pvalue_adj"] contains Benjamini-Hochberg adjusted p-values
   # ci["reject_null"] reflects the adjusted decision

Performance Questions
---------------------

How can I speed up explanation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduce nsamples** (but increases variance):

   .. code-block:: python

      explainer = OTExplainer(model, data=X, nsamples=20)

2. **Use subset of background data**:

   .. code-block:: python

      from fdfi.utils import sample_background
      X_bg = sample_background(X_train, n_samples=100)
      explainer = OTExplainer(model, data=X_bg)

3. **Use a cheaper explainer**: ``OTExplainer`` has a closed-form transport map
   and is substantially faster than ``EOTExplainer`` (Sinkhorn iterations) or
   ``FlowExplainer`` (flow training).

Why is EOTExplainer slower than OTExplainer?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EOTExplainer uses **Sinkhorn iterations** to solve the entropic OT problem, 
which requires O(n²) cost matrix computation and iterative optimization. 
OTExplainer uses a closed-form Gaussian solution.

Troubleshooting
---------------

I get "Flow matching requires torch" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only ``FlowExplainer`` requires PyTorch.  Install the flow dependencies:

.. code-block:: bash

   pip install -e ".[flow]"

Or switch to ``OTExplainer`` / ``EOTExplainer``, which never import torch.

My confidence intervals are all negative to positive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This happens when the standard errors are large relative to the estimates. 
Try:

1. Increase ``nsamples`` for lower variance
2. Use more test samples
3. Your features may genuinely have low importance

I get NaN or infinite values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check for:

1. NaN values in your data
2. Model returning NaN for some inputs
3. Very large or small feature values (consider standardizing)
