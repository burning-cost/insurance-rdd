"""Tests for the rdrobust import guard module."""

from __future__ import annotations

import pytest

from insurance_rdd._rdrobust import (
    check_rdrobust,
    check_rddensity,
    check_rdlocrand,
    check_rdmulti,
    import_rdrobust,
    import_rddensity,
    import_rdlocrand,
    import_rdmulti,
)


class TestImportGuards:
    def test_check_rdrobust_does_not_raise(self):
        """rdrobust is a required dependency — should always be available."""
        check_rdrobust()

    def test_check_rddensity_does_not_raise(self):
        check_rddensity()

    def test_check_rdlocrand_does_not_raise(self):
        check_rdlocrand()

    def test_check_rdmulti_does_not_raise(self):
        check_rdmulti()

    def test_import_rdrobust_returns_module(self):
        rdrobust = import_rdrobust()
        assert hasattr(rdrobust, "rdrobust")

    def test_import_rddensity_returns_module(self):
        rddensity = import_rddensity()
        assert hasattr(rddensity, "rddensity")

    def test_import_rdlocrand_returns_module(self):
        rdlocrand = import_rdlocrand()
        assert rdlocrand is not None

    def test_import_rdmulti_returns_module(self):
        rdmulti = import_rdmulti()
        assert rdmulti is not None

    def test_missing_package_raises_import_error(self, monkeypatch):
        """Simulate a missing package and verify ImportError with install instructions."""
        import insurance_rdd._rdrobust as guard
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fake_nonexistent_pkg":
                raise ImportError("No module named fake_nonexistent_pkg")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pip install"):
            guard._require("fake_nonexistent_pkg")
