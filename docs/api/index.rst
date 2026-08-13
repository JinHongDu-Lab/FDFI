API Reference
=============

This section contains the complete API reference for FDFI, auto-generated from 
the source code docstrings.

.. toctree::
   :maxdepth: 2

   explainers
   plots
   utils

Overview
--------

FDFI provides the following main modules:

**Explainers** (:mod:`fdfi.explainers`)
   Core explainer classes for computing feature importance. Includes the base
   ``Explainer`` class, optimal transport-based explainers (``OTExplainer``,
   ``EOTExplainer``), flow-based DFI (``FlowExplainer``), and cross-fitted
   inference (``Crossfitting``).

**Plotting** (:mod:`fdfi.plots`)
   Visualization functions for feature importance including summary plots, 
   waterfall plots, force plots, and dependence plots.

**Utilities** (:mod:`fdfi.utils`)
   Helper functions for input validation, data sampling, feature name 
   management, and statistical utilities like the ``TwoComponentMixture`` 
   class for variance floor estimation.
