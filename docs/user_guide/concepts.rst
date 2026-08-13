Core Concepts
=============

This page introduces the key concepts behind DFI (Disentangled Feature 
Importance) and FDFI (Flow-DFI), and how they relate to other interpretability methods.

What is Feature Importance?
---------------------------

Feature importance quantifies **how much each input feature contributes** to a 
model's predictions. Given a model :math:`f(x)` and an input 
:math:`x = (x_1, ..., x_d)`, we want to compute attributions 
:math:`\phi = (\phi_1, ..., \phi_d)` where :math:`\phi_j` represents the 
importance of feature :math:`j`.

SHAP and Shapley Values
-----------------------

SHAP (SHapley Additive exPlanations) computes feature importance using 
**Shapley values** from cooperative game theory. For a prediction :math:`f(x)`, 
SHAP values satisfy:

1. **Efficiency**: :math:`\sum_j \phi_j = f(x) - E[f(X)]`
2. **Symmetry**: Features with equal contributions get equal attributions
3. **Null**: Features that don't affect the output get zero attribution
4. **Linearity**: Attributions combine linearly for ensemble models

DFI is inspired by SHAP but uses **optimal transport** methods to compute 
feature importance.

Disentangled Feature Importance (DFI)
-------------------------------------

DFI introduces **disentangled feature importance** using optimal transport 
to create counterfactual distributions. The key insight is:

   *To measure the importance of feature j, compare the model output when 
   feature j comes from the data distribution vs. when it's replaced by 
   an independent sample.*

Mathematically, let :math:`Z = L^{-1}(X - \mu)` be the whitened
(disentangled) representation where features are uncorrelated. Importance is
measured by the **uncentered efficient influence function** (UEIF), which for
feature :math:`j` is:

.. math::

   \text{UEIF}_j(x) = \left( f(x) - E[f(\tilde{X}^{(j)})] \right)^2

where :math:`\tilde{X}^{(j)}` has feature :math:`j` replaced with an 
independent sample from the marginal distribution.

Gaussian vs Entropic OT
-----------------------

DFI provides two main approaches:

**Gaussian OT (OTExplainer)**

- Assumes data is approximately Gaussian
- Uses closed-form Gaussian optimal transport: :math:`Z = L^{-1}(X - \mu)`
- Fast and stable
- Best for continuous, roughly normal data

**Entropic OT (EOTExplainer)**

- Relaxes Gaussian assumption using Sinkhorn algorithm
- Adaptive regularization via median distance heuristic
- Supports mixed data types (continuous + categorical)
- Better for non-Gaussian or mixed-type data

Flow-Disentangled Feature Importance (Flow-DFI)
-----------------------------------------------

**FlowExplainer** uses **normalizing flows** to learn a flexible, data-driven
transformation between the original feature space X and a disentangled latent
space Z where features are approximately independent.

**CPI (Conditional Permutation Importance)**

Averages the counterfactual prediction first, then applies the loss:

.. math::

   \phi_{Z,j}^{CPI} = L\!\big(Y,\; \mathbb{E}_b[f(\tilde{X}_b^{(j)})]\big)
   \; - \; L\!\big(Y, f(X)\big)

where :math:`\tilde{X}_b^{(j)} = T^{-1}(\tilde{Z}_b^{(j)})` and 
:math:`\tilde{Z}_b^{(j)}` has the j-th component replaced with sample b, and
:math:`L` is a per-sample loss.

**SCPI (Sobol-CPI)**

Applies the loss to each Monte Carlo sample first, then averages:

.. math::

   \phi_{Z,j}^{SCPI} = \mathbb{E}_b\!\big[L\!\big(Y, f(\tilde{X}_b^{(j)})\big)\big]
   \; - \; L\!\big(Y, f(X)\big)

The key difference from CPI is the **order of averaging**; the two coincide for
a linear loss and differ by a Jensen gap otherwise. For the squared-error loss,
:math:`\phi^{SCPI} = \phi^{CPI} + \mathrm{Var}_b[f(\tilde{X}_b)]`, recovering the
Sobol total-order sensitivity index.

**Choosing a loss**

By default :math:`L` is the squared error, so the score reduces to the classic
difference of L2 residuals. Any regression loss (``'l1'``, ``'huber'``,
``'pinball'``) or binary-classification loss (``'log_loss'``, ``'brier'``,
``'zero_one'``) can be selected via the ``loss`` argument, or a custom callable
``loss(y_true, y_pred)`` supplied directly.

If the true labels ``y`` are passed at call time
(``explainer(X_test, y=y_test)``), the score is the loss-difference (DFI / LOCO)
form, which is centred near zero for null features. If ``y`` is omitted, a
label-free form is used that references the model's own prediction and subtracts
the self-loss floor:

.. math::

   \phi_{Z,j} = \operatorname*{agg}_b L(\hat{Y}, f(\tilde{X}_b^{(j)}))
   \; - \; L(\hat{Y}, \hat{Y}), \qquad \hat{Y} = f(X).

For losses with :math:`L(a, a) = 0` (squared error, L1, Huber, pinball) this is
the prediction shift under that loss. For a proper scoring rule (log-loss,
Brier) it is the associated Bregman divergence between the baseline and
counterfactual predictions (e.g. :math:`\mathrm{KL}(\hat{Y}\,\|\,f(\tilde{X}_b))`
for log-loss), which is non-negative and ~0 for null features. Non-proper or
discontinuous losses (e.g. ``'zero_one'``) are only meaningful with ``y``.

**Jacobian Transformation to X-space**

Both CPI and SCPI compute importance in the disentangled Z-space. To attribute
importance to the original features :math:`X_l`, we use the **Jacobian** of the
decoder transformation :math:`T^{-1}: Z \to X`:

.. math::

   \phi_{X,l} = \sum_{k=1}^{d} H_{lk}^2 \cdot \phi_{Z,k}

where :math:`H = \frac{\partial X}{\partial Z}` is the Jacobian matrix evaluated
at the data points. This correctly accounts for how changes in each latent 
dimension Z_k affect each original feature X_l.

For linear transformations (as in OTExplainer), this reduces to :math:`\phi_X = H^T H \phi_Z`
where :math:`H = L` is the Cholesky factor. For normalizing flows, the Jacobian
varies with position and is computed via automatic differentiation.

**When to Use FlowExplainer**

- Complex non-linear dependencies between features
- Non-Gaussian data distributions
- When OT assumptions are too restrictive
- When you have sufficient data (>500 samples) to train the flow
- When feature correlations are complex and non-linear

**Trade-offs vs. OTExplainer / EOTExplainer**

- Requires PyTorch; flow training adds up-front computation (``num_steps=200``
  recommended as a starting point).
- The Jacobian used for X-space attribution varies across samples and is
  computed via automatic differentiation — more expensive than the fixed
  Cholesky map in ``OTExplainer``.
- Pre-trained flow models can be reused across experiments via
  ``explainer.set_flow(flow)``.

Shared Disentanglement Diagnostics
----------------------------------

All disentangled explainers (OT, EOT, Flow) report two common diagnostics:

- **Latent independence**: median pairwise distance correlation in latent space.
- **Distribution fidelity**: MMD between original data and reconstructed data.

Both are "lower is better" metrics and are reported with qualitative labels
(``GOOD``, ``MODERATE``, ``POOR``) using shared thresholds.

Relationship to Other Methods
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Comparison to DFI
   * - **SHAP KernelExplainer**
     - Model-agnostic like DFI, but uses sampling-based Shapley estimation. 
       DFI uses OT-based counterfactuals.
   * - **SHAP TreeExplainer**
     - Exact Shapley for trees. DFI works with any model.
   * - **LIME**
     - Local linear approximation. DFI considers global distribution.
   * - **Permutation Importance**
     - Breaks feature dependencies. DFI preserves correlation structure.
   * - **Integrated Gradients**
     - Requires gradients. DFI is gradient-free.

Confidence Intervals
--------------------

A key advantage of DFI is **built-in uncertainty quantification**. The 
``conf_int()`` method provides:

- Standard errors computed across samples
- Confidence intervals using normal approximation
- P-values for testing :math:`H_0: \phi_j = 0` or :math:`H_0: \phi_j \leq \delta`
- Variance floor methods for stable inference with small effects
- **Multiple Testing Correction**: control of False Discovery Rate (FDR) or
  Family-Wise Error Rate (FWER) via ``multitest_method``.

The returned dictionary includes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``score``
     - Estimated feature importance (mean UEIF).
   * - ``se``
     - Standard error of the mean UEIF (after variance floor).
   * - ``zscore``
     - Signed z-statistic: ``(score − margin) / se``.
   * - ``ranking``
     - Integer rank by descending z-score (1 = most important).
   * - ``ci_lower`` / ``ci_upper``
     - Confidence interval bounds.
   * - ``reject_null``
     - Boolean array indicating rejected null hypotheses.
   * - ``pvalue``
     - One- or two-sided p-value.
   * - ``pvalue_adj``
     - Multiple-testing-adjusted p-values (present when ``multitest_method`` is set).
   * - ``groups``
     - List of group names (present when ``groups`` argument is provided).

This enables **statistical feature selection**: identify features that are 
significantly different from zero or a practical threshold.

Multiple Testing Correction
---------------------------

When testing hundreds of features simultaneously, the probability of obtaining
false positives (Type I errors) increases. ``conf_int()`` supports multiple
testing corrections using the ``statsmodels`` library.

By setting ``multitest_method``, you can choose from various correction methods:

- **FWER Control**: ``'bonferroni'``, ``'holm'``, ``'sidak'``, etc.
- **FDR Control**: ``'fdr_bh'`` (Benjamini-Hochberg), ``'fdr_by'`` (Benjamini-Yekutieli).

When a correction is applied, ``conf_int()`` returns adjusted p-values as
``pvalue_adj``, and the ``reject_null`` decision is updated to reflect the
specified ``alpha`` (e.g., FDR < 0.05).

Group-Level Importance
----------------------

In many applications features naturally belong to **groups** (e.g., genomic
regions, sensor categories, feature families).  The ``conf_int()`` method
supports a ``groups`` argument that aggregates per-sample UEIFs across features
within each group and reports group-level importance with proper uncertainty.

Given a group :math:`S_g \subseteq \{1, \ldots, d\}`, the group importance is:

.. math::

   \phi_g = \frac{1}{n} \sum_{i=1}^{n} \sum_{j \in S_g} \text{UEIF}_{ij}

The standard error is computed from the per-sample grouped UEIFs
:math:`u_i = \sum_{j \in S_g} \text{UEIF}_{ij}`:

.. math::

   \text{SE}_g = \frac{\sigma(u)}{\sqrt{n}} + \frac{c}{\sqrt{n}\; z_{1-\alpha/2}}

where :math:`c` is a small finite-sample correction constant (default 0.1)
that prevents anti-conservative z-scores when the raw SE is very small.

An optional **null-thresholding** step (``threshold_null=True``) zeros out per-feature UEIFs with
negative mean before aggregation, preventing estimation noise from
artificially deflating group importance.

Previously, this was handled by a separate ``group_importance()`` method,
which is now deprecated in favor of the more flexible ``conf_int(groups=...)``
API.

Further Reading
---------------

- `SHAP paper <https://arxiv.org/abs/1705.07874>`_: Lundberg & Lee (2017)
- `Optimal Transport <https://arxiv.org/abs/1803.00567>`_: Peyré & Cuturi (2019)
- `Entropic OT <https://arxiv.org/abs/1306.0895>`_: Cuturi (2013)
