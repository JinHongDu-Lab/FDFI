Case Studies
============

Two end-to-end analyses on real datasets, one per disentanglement approach. Both
follow the same workflow — load data, fit a black-box model, initialize an
explainer, check the disentanglement diagnostics, then report feature-level and
group-level inference — so the sections line up even though the applications
differ.

.. list-table::
   :header-rows: 1
   :widths: 26 22 26 26

   * - Case study
     - Explainer
     - Data
     - Why this explainer
   * - :doc:`Case Study 1 <eot_case_study_sens50>`
     - ``EOTExplainer``
     - HIV-1 VRC01 neutralization: 611 viral sequences, 832 binary
       sequence-derived predictors, 14 biological feature groups
     - High-dimensional and mostly binary, where an analytical whitening map is
       better behaved than a learned decoder
   * - :doc:`Case Study 2 <flow_case_study_ctg>`
     - ``FlowExplainer``
     - UCI Cardiotocography: 2,126 fetal monitoring records, 21 continuous
       features, 4 clinical feature groups
     - Continuous features that are collinear by construction — FHR mean, mode,
       and median all measure the same signal

Case Study 1 works in the original feature space, where the biological groups are
defined and the entropic-OT decoder is analytical. Case Study 2 reports
X-space attributions obtained by projecting through the learned flow Jacobian, and
leads with a collinearity diagnostic that motivates disentanglement in the first
place.

.. note::

   Both notebooks ship with their outputs stored and are rendered without being
   re-executed, so they can be read as-is. Re-running Case Study 1 takes a few
   minutes; Case Study 2 downloads the CTG spreadsheet from the UCI repository
   (network access plus the ``xlrd`` package) and trains several normalizing
   flows, so it takes considerably longer.

.. toctree::
   :maxdepth: 1

   eot_case_study_sens50
   flow_case_study_ctg
   NOTEBOOK_BASELINE
