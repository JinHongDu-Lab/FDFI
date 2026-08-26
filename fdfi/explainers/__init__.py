"""
Explainer classes for DFI.

This subpackage contains the main explainer classes for computing
disentangled feature importance:

- :class:`Explainer` -- shared base class (inference, diagnostics, summaries).
- :class:`OTExplainer` -- Gaussian optimal-transport explainer.
- :class:`EOTExplainer` -- entropic optimal-transport explainer.
- :class:`FlowExplainer` -- normalizing-flow explainer.
- :class:`Crossfitting` -- cross-fitted wrapper around any of the above.
"""

from .base import Explainer
from .ot import OTExplainer, DFIExplainer
from .eot import EOTExplainer
from .flow import FlowExplainer
from .crossfitting import Crossfitting

__all__ = [
    "Explainer",
    "OTExplainer",
    "EOTExplainer",
    "FlowExplainer",
    "DFIExplainer",
    "Crossfitting",
]
