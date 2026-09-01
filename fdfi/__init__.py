"""
FDFI: Flow-Disentangled Feature Importance

A Python library for computing feature importance using disentangled methods.
Includes both DFI (Disentangled Feature Importance) and FDFI (Flow-DFI) methods.
"""

from importlib import metadata as importlib_metadata
from pathlib import Path
import re


def _version_from_pyproject() -> str:
    """Read package version from pyproject.toml when running from source tree."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not parse version from pyproject.toml")
    return match.group(1)


def _resolve_version() -> str:
    """
    Resolve version without hardcoding.

    Priority:
    1) pyproject.toml (source/dev workflow, always aligned with project metadata)
    2) installed distribution metadata (wheel/sdist runtime)
    """
    try:
        return _version_from_pyproject()
    except (FileNotFoundError, ValueError):
        pass

    try:
        return importlib_metadata.version("fdfi")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()
__author__ = "FDFI Team"

from .explainers import Crossfitting
from .losses import (
    resolve_loss,
    available_losses,
    squared_error,
    absolute_error,
    huber,
    pinball,
    log_loss,
    brier,
    zero_one,
)

__all__ = [
    "__version__",
    "Crossfitting",
    "resolve_loss",
    "available_losses",
    "squared_error",
    "absolute_error",
    "huber",
    "pinball",
    "log_loss",
    "brier",
    "zero_one",
]
