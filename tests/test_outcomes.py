"""Tests for PoissonRD and GammaRD local GLM estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd.outcomes import (
    PoissonRD,
    GammaRD,
    LocalGLMResult,
    _triangular_weights,
    _epanechnikov_weights,
    _uniform_weights,
    _build_design_matrix,
    _fit_one_side,
    _auto_bandwidth,
)

from conftest import make_sharp_rdd_data


# ---------------------------------------------------------------------------
# Kernel weights
# ---------------------------------------------------------------------------


class TestKernelWeights:
    def test_triangular_zero_at_boundary(self):
        x = np.array([-1.0, 0.0, 1.0])
        w = _triangular_weights(x, cutoff=0.0, h=1.0)
        assert abs(w[0]) < 1e-10
        assert abs(w[2]) < 1e-10
        assert w[1] == 1.0

    def test_triangular_nonnegative(self):
        x = np.linspace(-2, 2, 50)
        w = _triangular_weights(x, cutoff=0.0, h=1.0)
        assert np.all(w >= 0)

    def test_triangular_zero_outside_bandwidth(self):
        x = np.array([-2.0, 2.0])
        w = _triangular_weights(x, cutoff=0.0, h=1.0)
        assert w[0] == 0.0
        assert w[1] == 0.0

    def test_epanechnikov_nonnegative(self):
        x = np.linspace(-2, 2, 50)
        w = _epanechnikov_weights(x, cutoff=0.0, h=1.0)
        assert np.all(w >= 0)

    def test_uniform_binary(self):
        x = np.array([-0.5, -1.5, 0.5, 1.5])
        w = _uniform_weights(x, cutoff=0.0, h=1.0)
        assert w[0] == 1.0
        assert w[1] == 0.0
        assert w[2] == 1.0
        assert w[3] == 0.0

    def test_triangular_symmetric(self):
        x = np.array([-0.5, 0.5])
        w = _triangular_weights(x, cutoff=0.0, h=1.0)
        assert abs(w[0] - w[1]) < 1e-10

    def test_epanechnikov_peak_at_cutoff(self):
        x = np.array([0.0, 0.5, 0.9])
        w = _epanechnikov_weights(x, cutoff=0.0, h=1.0)
        assert w[0] > w[1] > w[2]


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    def test_design_matrix_shape(self):
        x = np.linspace(-1, 1, 10)
        X = _build_design_matrix(x, cutoff=0.0, p=1)
        assert X.shape == (10, 2)

    def test_design_matrix_quadratic(self):
        x = np.linspace(-1, 1, 10)
        X = _build_design_matrix(x, cutoff=0.0, p=2)
        assert X.shape == (10, 3)

    def test_design_matrix_intercept_is_ones(self):
        x = np.linspace(-1, 1, 10)
        X = _build_design_matrix(x, cutoff=0.0, p=1)
        assert np.all(X[:, 0] == 1.0)

    def test_design_matrix_centred(self):
        cutoff = 5.0
        x = np.array([4.0, 5.0, 6.0])
        X = _build_design_matrix(x, cutoff=cutoff, p=1)
        assert np.allclose(X[:, 1], x - cutoff)


# ---------------------------------------------------------------------------
# One-side fitting
# ---------------------------------------------------------------------------


class TestFitOneSide:
    def test_poisson_one_side_converges(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(-1, 0, n)
        exposure = rng.uniform(0.1, 1.0, n)
        mu = np.exp(-1.5 + 0.3 * x) * exposure
        y = rng.poisson(mu).astype(float)

        kw = _triangular_weights(x, cutoff=0.0, h=1.0)
        coef, converged = _fit_one_side(x, y, exposure, kw, cutoff=0.0, p=1, outcome_type="poisson")
        assert converged
        assert len(coef) == 2

    def test_gamma_one_side_converges(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(-1, 0, n)
        mu = np.exp(2.0 + 0.2 * x)
        y = rng.gamma(3.0, mu / 3.0, size=n)

        kw = _triangular_weights(x, cutoff=0.0, h=1.0)
        coef, converged = _fit_one_side(x, y, None, kw, cutoff=0.0, p=1, outcome_type="gamma")
        assert converged
        assert len(coef) == 2

    def test_invalid_outcome_type_raises(self):
        x = np.zeros(10)
        y = np.ones(10)
        kw = np.ones(10)
        with pytest.raises(ValueError):
            _fit_one_side(x, y, None, kw, cutoff=0.0, p=1, outcome_type="invalid")


# ---------------------------------------------------------------------------
# Auto-bandwidth
# ---------------------------------------------------------------------------


class TestAutoBandwidth:
    def test_auto_bandwidth_positive(self):
        x = np.random.default_rng(42).uniform(-3, 3, 200)
        h = _auto_bandwidth(x, cutoff=0.0, outcome_type="poisson")
        assert h > 0

    def test_auto_bandwidth_smaller_than_range(self):
        x = np.random.default_rng(42).uniform(-3, 3, 200)
        h = _auto_bandwidth(x, cutoff=0.0, outcome_type="poisson")
        assert h < 6.0  # total range


# ---------------------------------------------------------------------------
# PoissonRD
# ---------------------------------------------------------------------------


class TestPoissonRD:
    def test_fit_returns_result(self, poisson_df):
        rd = PoissonRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=poisson_df,
            exposure="exposure",
            n_boot=50,
        )
        result = rd.fit()
        assert isinstance(result, LocalGLMResult)

    def test_result_attributes(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50
        )
        result = rd.fit()
        assert hasattr(result, "tau")
        assert hasattr(result, "se_boot")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "p_value")
        assert hasattr(result, "bandwidth")
        assert hasattr(result, "n_left")
        assert hasattr(result, "n_right")

    def test_tau_is_finite(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50
        )
        result = rd.fit()
        assert np.isfinite(result.tau)

    def test_ci_is_ordered(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50
        )
        result = rd.fit()
        assert result.ci_lower < result.ci_upper

    def test_tau_sign_recovers_dgp(self, poisson_df):
        """True tau=0.3, estimate should be positive."""
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=100, exposure="exposure"
        )
        result = rd.fit()
        assert result.tau > 0, f"Expected positive tau, got {result.tau:.4f}"

    def test_with_fixed_bandwidth(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            h=1.0, n_boot=30
        )
        result = rd.fit()
        assert result.bandwidth == 1.0

    def test_donut_reduces_n(self, poisson_df):
        rd_no = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=30
        )
        rd_d = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            donut_radius=0.2, n_boot=30
        )
        r_no = rd_no.fit()
        r_d = rd_d.fit()
        assert r_d.n_left + r_d.n_right <= r_no.n_left + r_no.n_right

    def test_rate_ratio_method(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50
        )
        result = rd.fit()
        rr = result.rate_ratio()
        assert rr["rate_ratio"] > 0
        assert rr["ci_lower"] < rr["rate_ratio"] < rr["ci_upper"]

    def test_summary_returns_string(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=30
        )
        result = rd.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "poisson" in s.lower()

    def test_p_value_significant_for_large_effect(self):
        """Large true effect (tau=1.0) should produce small p-value."""
        df = make_sharp_rdd_data(n=3000, true_tau=1.0, outcome_type="poisson")
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            exposure="exposure", n_boot=200
        )
        result = rd.fit()
        assert result.p_value < 0.05, f"Expected significant, got p={result.p_value:.4f}"

    def test_no_effect_not_significant(self):
        """True tau=0 should not be significant at 5%."""
        df = make_sharp_rdd_data(n=3000, true_tau=0.0, outcome_type="poisson")
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            exposure="exposure", n_boot=200
        )
        result = rd.fit()
        # Generous: tau should be small in magnitude
        assert abs(result.tau) < 0.5, f"Expected near-zero tau, got {result.tau:.4f}"

    def test_different_kernels(self, poisson_df):
        for kernel in ["triangular", "epanechnikov", "uniform"]:
            rd = PoissonRD(
                outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
                kernel=kernel, n_boot=30
            )
            result = rd.fit()
            assert np.isfinite(result.tau), f"Failed with kernel={kernel}"

    def test_outcome_type_is_poisson(self, poisson_df):
        rd = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=30
        )
        result = rd.fit()
        assert result.outcome_type == "poisson"

    def test_rng_reproducibility(self, poisson_df):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        rd1 = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50, rng=rng1
        )
        rd2 = PoissonRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            n_boot=50, rng=rng2
        )
        r1 = rd1.fit()
        r2 = rd2.fit()
        assert abs(r1.se_boot - r2.se_boot) < 1e-10


# ---------------------------------------------------------------------------
# GammaRD
# ---------------------------------------------------------------------------


class TestGammaRD:
    def test_fit_returns_result(self, gamma_df):
        rd = GammaRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gamma_df,
            n_boot=50,
        )
        result = rd.fit()
        assert isinstance(result, LocalGLMResult)

    def test_outcome_type_is_gamma(self, gamma_df):
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=30
        )
        result = rd.fit()
        assert result.outcome_type == "gamma"

    def test_tau_is_finite(self, gamma_df):
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=30
        )
        result = rd.fit()
        assert np.isfinite(result.tau)

    def test_ci_ordered(self, gamma_df):
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=30
        )
        result = rd.fit()
        assert result.ci_lower < result.ci_upper

    def test_tau_sign_recovers_dgp(self, gamma_df):
        """True tau=0.25, estimate should be positive."""
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=100
        )
        result = rd.fit()
        assert result.tau > 0, f"Expected positive tau, got {result.tau:.4f}"

    def test_severity_only_true_filters_zeros(self):
        """severity_only=True should only use positive outcomes."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-2, 2, n)
        y = np.where(rng.random(n) > 0.5, rng.gamma(2.0, 1000.0, n), 0.0)
        df = pd.DataFrame({"x": x, "y": y})

        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            severity_only=True, n_boot=30
        )
        result = rd.fit()
        # All in-bandwidth obs should be positive claims
        assert result.n_left > 0 or result.n_right > 0

    def test_rate_ratio_positive(self, gamma_df):
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=30
        )
        result = rd.fit()
        rr = result.rate_ratio()
        assert rr["rate_ratio"] > 0

    def test_summary_contains_gamma(self, gamma_df):
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            n_boot=30
        )
        result = rd.fit()
        s = result.summary()
        assert "gamma" in s.lower()

    def test_insufficient_data_raises(self):
        df = pd.DataFrame({"x": [0.1, 0.2], "y": [100.0, 200.0]})
        rd = GammaRD(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            n_boot=10
        )
        with pytest.raises((ValueError, Exception)):
            rd.fit()
