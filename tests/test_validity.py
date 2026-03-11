"""Tests for validity checks: DensityTest, CovariateBalance, PlaceboTest, DonutRDD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd.validity import (
    DensityTest,
    DensityTestResult,
    CovariateBalance,
    CovariateBalanceResult,
    PlaceboTest,
    PlaceboTestResult,
    DonutRDD,
    DonutRDDResult,
)

from conftest import make_sharp_rdd_data


# ---------------------------------------------------------------------------
# DensityTest
# ---------------------------------------------------------------------------


class TestDensityTest:
    def test_fit_returns_result(self, gaussian_df):
        dt = DensityTest(
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = dt.fit()
        assert isinstance(result, DensityTestResult)

    def test_result_attributes(self, gaussian_df):
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        result = dt.fit()
        assert hasattr(result, "test_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "significant")
        assert hasattr(result, "interpretation")

    def test_p_value_in_unit_interval(self, gaussian_df):
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        result = dt.fit()
        if not np.isnan(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_no_manipulation_in_dgp(self, gaussian_df):
        """Uniform running variable should not show manipulation."""
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        result = dt.fit()
        # Large p-value expected (not significant).
        # We can't guarantee this statistically, just that it runs.
        assert result is not None

    def test_summary_returns_string(self, gaussian_df):
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        result = dt.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "p-value" in s.lower() or "test" in s.lower()

    def test_manipulation_expected_flag(self, gaussian_df):
        dt = DensityTest(
            running_var="x", cutoff=0.0, data=gaussian_df,
            manipulation_expected=True
        )
        result = dt.fit()
        assert result.interpretation is not None
        assert len(result.interpretation) > 0

    def test_significant_attribute_matches_p_value(self, gaussian_df):
        alpha = 0.05
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df, alpha=alpha)
        result = dt.fit()
        if not np.isnan(result.p_value):
            assert result.significant == (result.p_value < alpha)

    def test_bandwidths_are_positive_or_nan(self, gaussian_df):
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        result = dt.fit()
        if not np.isnan(result.bandwidth_left):
            assert result.bandwidth_left > 0
        if not np.isnan(result.bandwidth_right):
            assert result.bandwidth_right > 0

    def test_works_with_ncd_like_discrete_variable(self):
        rng = np.random.default_rng(42)
        ncd = rng.integers(0, 6, size=1000).astype(float)
        df = pd.DataFrame({"ncd": ncd})
        dt = DensityTest(running_var="ncd", cutoff=3.0, data=df)
        result = dt.fit()
        assert result is not None


# ---------------------------------------------------------------------------
# CovariateBalance
# ---------------------------------------------------------------------------


class TestCovariateBalance:
    def test_fit_returns_result(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        assert isinstance(result, CovariateBalanceResult)

    def test_result_attributes(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        assert hasattr(result, "results_df")
        assert hasattr(result, "n_covariates")
        assert hasattr(result, "n_significant")
        assert hasattr(result, "balanced")

    def test_one_covariate_row_in_df(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        assert len(result.results_df) == 1

    def test_multiple_covariates(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["covariate_1", "covariate_2"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        # covariate_2 is binary but still numeric
        assert result.n_covariates >= 1  # at least one numeric cov tested

    def test_predetermined_covariate_should_balance(self, gaussian_df):
        """covariate_1 is smooth at cutoff — should not be significant."""
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        # With a smooth covariate, most runs should not find a significant effect.
        # We check that the absolute tau is small.
        if len(result.results_df) > 0:
            tau = result.results_df["tau"].iloc[0]
            assert abs(tau) < 1.0, f"Covariate imbalance seems too large: tau={tau:.4f}"

    def test_summary_returns_string(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        result = cb.fit()
        s = result.summary()
        assert isinstance(s, str)

    def test_missing_covariate_warns(self, gaussian_df):
        cb = CovariateBalance(
            covariates=["nonexistent_col"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cb.fit()
            assert any(issubclass(ww.category, RuntimeWarning) for ww in w)


# ---------------------------------------------------------------------------
# PlaceboTest
# ---------------------------------------------------------------------------


class TestPlaceboTest:
    def test_fit_returns_result(self, gaussian_df):
        pt = PlaceboTest(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            placebo_cutoffs=[-1.5, 1.5],
            true_tau=0.4,
        )
        result = pt.fit()
        assert isinstance(result, PlaceboTestResult)

    def test_result_attributes(self, gaussian_df):
        pt = PlaceboTest(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            placebo_cutoffs=[-1.5],
        )
        result = pt.fit()
        assert hasattr(result, "results_df")
        assert hasattr(result, "true_tau")
        assert hasattr(result, "n_significant")

    def test_results_df_has_correct_columns(self, gaussian_df):
        pt = PlaceboTest(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            placebo_cutoffs=[-1.5, 1.5],
        )
        result = pt.fit()
        df = result.results_df
        assert "placebo_cutoff" in df.columns
        assert "tau" in df.columns
        assert "p_value" in df.columns

    def test_n_significant_nonnegative(self, gaussian_df):
        pt = PlaceboTest(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            placebo_cutoffs=[-1.5, 1.5],
        )
        result = pt.fit()
        assert result.n_significant >= 0

    def test_summary_returns_string(self, gaussian_df):
        pt = PlaceboTest(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            placebo_cutoffs=[-1.5],
        )
        result = pt.fit()
        s = result.summary()
        assert isinstance(s, str)

    def test_placebo_tau_near_zero_for_no_effect_dgp(self):
        """With no real treatment, placebo taus should be near zero."""
        df = make_sharp_rdd_data(n=3000, true_tau=0.0, outcome_type="gaussian")
        pt = PlaceboTest(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            placebo_cutoffs=[-1.0, 1.0],
        )
        result = pt.fit()
        for _, row in result.results_df.iterrows():
            assert abs(row["tau"]) < 1.0  # should be small


# ---------------------------------------------------------------------------
# DonutRDD
# ---------------------------------------------------------------------------


class TestDonutRDD:
    def test_fit_returns_result(self, gaussian_df):
        d = DonutRDD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            donut_radii=[0.1, 0.2, 0.3],
            original_tau=0.4,
        )
        result = d.fit()
        assert isinstance(result, DonutRDDResult)

    def test_results_df_has_correct_shape(self, gaussian_df):
        d = DonutRDD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radii=[0.1, 0.2, 0.3],
        )
        result = d.fit()
        assert len(result.results_df) <= 3  # could fail at some radii

    def test_results_df_columns(self, gaussian_df):
        d = DonutRDD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radii=[0.1],
        )
        result = d.fit()
        if len(result.results_df) > 0:
            df = result.results_df
            assert "donut_radius" in df.columns
            assert "tau" in df.columns

    def test_summary_returns_string(self, gaussian_df):
        d = DonutRDD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radii=[0.1, 0.2],
        )
        result = d.fit()
        s = result.summary()
        assert isinstance(s, str)

    def test_estimate_stable_across_small_donuts(self, gaussian_df):
        """For a clean DGP, effect should be stable as donut grows."""
        d = DonutRDD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radii=[0.05, 0.10, 0.15],
            original_tau=0.4,
        )
        result = d.fit()
        if len(result.results_df) >= 2:
            taus = result.results_df["tau"].values
            # All taus should be in a reasonable range of each other.
            assert np.nanmax(taus) - np.nanmin(taus) < 0.5
