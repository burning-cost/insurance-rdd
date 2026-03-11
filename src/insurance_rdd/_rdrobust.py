"""Import guard for rdrobust and companion packages.

All four packages (rdrobust, rddensity, rdlocrand, rdmulti) ship separate
Python packages on PyPI. They are authored by Cattaneo, Titiunik, and
collaborators — the same group that published the original CCT methodology.
This module checks availability and gives actionable error messages.
"""

from __future__ import annotations


def _require(package: str, install_name: str | None = None) -> None:
    """Raise ImportError with install instructions if package is missing."""
    try:
        __import__(package)
    except ImportError:
        install = install_name or package
        raise ImportError(
            f"Package '{package}' is required but not installed. "
            f"Install it with: pip install {install}"
        ) from None


def check_rdrobust() -> None:
    """Verify rdrobust is available."""
    _require("rdrobust")


def check_rddensity() -> None:
    """Verify rddensity is available."""
    _require("rddensity")


def check_rdlocrand() -> None:
    """Verify rdlocrand is available."""
    _require("rdlocrand")


def check_rdmulti() -> None:
    """Verify rdmulti is available."""
    _require("rdmulti")


def check_geopandas() -> None:
    """Verify geopandas and shapely are available (optional geo dependency)."""
    _require("geopandas", "geopandas>=0.14")
    _require("shapely", "shapely>=2.0")


def import_rdrobust():
    """Import and return rdrobust module, with helpful error on failure."""
    check_rdrobust()
    import rdrobust
    return rdrobust


def import_rddensity():
    """Import and return rddensity module, with helpful error on failure."""
    check_rddensity()
    import rddensity
    return rddensity


def import_rdlocrand():
    """Import and return rdlocrand module, with helpful error on failure."""
    check_rdlocrand()
    import rdlocrand
    return rdlocrand


def import_rdmulti():
    """Import and return rdmulti module, with helpful error on failure."""
    check_rdmulti()
    import rdmulti
    return rdmulti
