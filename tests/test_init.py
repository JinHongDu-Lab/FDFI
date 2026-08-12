"""
Tests for FDFI package initialization.
"""

import re
from pathlib import Path

import fdfi


def _version_from_pyproject() -> str:
    """Read package version from pyproject.toml."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    if not match:
        raise AssertionError("Could not parse version from pyproject.toml")
    return match.group(1)


def test_version():
    """Test that version is defined."""
    assert hasattr(fdfi, "__version__")
    assert isinstance(fdfi.__version__, str)
    assert fdfi.__version__ == _version_from_pyproject()


def test_package_imports():
    """Test that package can be imported."""
    # Just check that the package imports without errors
    assert fdfi is not None


def test_yifan_convenience_exports_are_preserved():
    """The 0.0.10 integration keeps the public yifan-dev import surface."""
    assert fdfi.FlowExplainer is fdfi.explainers.FlowExplainer
    assert fdfi.plots is not None
    assert fdfi.utils is not None
    assert fdfi.models is not None
    assert fdfi.resolve_loss("squared_error") is not None
