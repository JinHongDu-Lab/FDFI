Utilities
=========

Overview
--------

The ``fdfi.utils`` module provides helper functions and classes used across
the FDFI package.  The most commonly useful symbols for end users are
:class:`~fdfi.utils.TwoComponentMixture` (for understanding variance-floor and
margin estimation), :func:`~fdfi.utils.compute_latent_independence`, and
:func:`~fdfi.utils.compute_mmd`.  The remaining helpers are used internally by
the explainer classes.

Statistical Utilities
---------------------

TwoComponentMixture
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fdfi.utils.TwoComponentMixture
   :members:

Latent Independence (dCor)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fdfi.utils.compute_latent_independence

Maximum Mean Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fdfi.utils.compute_mmd

Feature Type Detection
----------------------

.. autofunction:: fdfi.utils.detect_feature_types

Gower Distance
--------------

.. autofunction:: fdfi.utils.gower_cost_matrix

Computes the Gower distance matrix for mixed-type data (continuous, binary,
and categorical features).

Internal Helpers
----------------

The following functions are used internally by the explainer classes.  They
are documented here for completeness but are not part of the stable public API.

.. autofunction:: fdfi.utils.validate_input

.. autofunction:: fdfi.utils.sample_background

.. autofunction:: fdfi.utils.get_feature_names

.. autofunction:: fdfi.utils.convert_to_link

The following link functions are supported:

- ``"identity"``: No transformation (default)
- ``"logit"``: Logit transformation for probability outputs

.. autofunction:: fdfi.utils.check_additivity
