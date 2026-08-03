FDFI Documentation
==================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

**FDFI** (Flow-Disentangled Feature Importance) is a Python library for computing
feature importance using disentangled methods, inspired by SHAP. This package
implements both OT-based DFI and flow-based FDFI methods.

Key Features
------------

- **Three explainer variants**, all model-agnostic and sharing one API:
  :class:`~fdfi.explainers.OTExplainer` (Gaussian OT — the fast default),
  :class:`~fdfi.explainers.EOTExplainer` (entropic OT — non-Gaussian and
  mixed-type data), and :class:`~fdfi.explainers.FlowExplainer` (normalizing
  flows — complex non-linear dependence)
- **Cross-fitted inference**: :class:`~fdfi.explainers.Crossfitting` wraps any
  variant for valid standard errors at small sample sizes
- **Shared Diagnostics**: Latent independence (dCor) and distribution fidelity (MMD) checks for OT/EOT/Flow
- **Visualization**: Summary, waterfall, force, dependence, CI, and diagnostics plots
- **Statistical Inference**: Confidence intervals, one-sided tests, FDR correction, and group-level importance
- **Easy to Use**: Simple API; background data replaces the ``predict_proba`` baseline
- **Extensible**: Built with modularity for future enhancements

.. note::

   Recent additions include configurable loss functions (any regression or
   binary-classification loss, or a custom callable) and one-sided confidence
   intervals in :func:`~fdfi.plots.confidence_interval_plot`.  See the
   :doc:`changelog` for the full history and
   :doc:`tutorials/confidence_intervals` for worked examples.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from fdfi.explainers import OTExplainer
   from fdfi.plots import summary_bar

   rng = np.random.default_rng(0)

   def model(X):
       return X.sum(axis=1)

   X_background = rng.standard_normal((100, 10))
   explainer = OTExplainer(model, data=X_background, nsamples=50)

   X_test = rng.standard_normal((50, 10))
   results = explainer(X_test)

   # Confidence intervals (one-sided, with FDR correction)
   ci = explainer.conf_int(alpha=0.05, alternative="greater", multitest_method="fdr_bh")

   feature_names = [f"X{i}" for i in range(X_background.shape[1])]
   summary_bar(results["phi_X"], results["se_X"], feature_names, show=False)

Installation
------------

.. code-block:: bash

   git clone https://github.com/jinhongdu-lab/FDFI.git
   cd FDFI
   pip install -e .

   # With optional dependencies
   pip install -e ".[flow]"    # For flow matching models
   pip install -e ".[dev]"     # For development

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/concepts
   user_guide/choosing_explainer
   user_guide/interpreting_results
   user_guide/statistical_inference
   user_guide/faq

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Case Studies

   case_studies/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
