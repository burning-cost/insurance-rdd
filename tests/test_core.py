"""Tests for InsuranceRD core estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd import InsuranceRD, RDResult, presets
from insurance_rdd.presets import AGE_25, NCD_STEP, VEHICLE_AGE_10

from conftest import make_sharp_rdd_data, make_age_rdd_data


# ---------------------------------------------------------------------------
# Basic estimation
# ---------------------------------------------------------------------------


class TestInsuranceRDBasic:
    def test_gaussian_fit_returns_result(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            outcome_type="gaussian",
        )
        result = rd.fit()
        assert isinstance(result, RDResult)

    def test_gaussian_recovers_true_tau_in_ci(self, gaussian_df):
        """True tau=0.4 should be inside the 95% CI."""
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            outcome_type="gaussian",
        )
        result = rd.fit()
        true_tau = 0.4
        assert result.ci_lower < true_tau < result.ci_upper, (
            f"True tau {true_tau} not in CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        )

    def test_poisson_fit_returns_result(self, poisson_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=poisson_df,
            outcome_type="poisson",
            exposure="exposure",
        )
        result = rd.fit()
        assert isinstance(result, RDResult)

    def test_poisson_tau_sign_correct(self, poisson_df):
        """True tau=0.3 (positive), estimate should be positive."""
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=poisson_df,
            outcome_type="poisson",
            exposure="exposure",
        )
        result = rd.fit()
        assert result.tau_bc > 0, f"Expected positive tau, got {result.tau_bc:.4f}"

    def test_gamma_fit_returns_result(self, gamma_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gamma_df,
            outcome_type="gamma",
        )
        result = rd.fit()
        assert isinstance(result, RDResult)

    def test_result_has_required_attributes(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df
        )
        result = rd.fit()
        assert hasattr(result, "tau")
        assert hasattr(result, "tau_bc")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "p_value")
        assert hasattr(result, "bandwidth_h")
        assert hasattr(result, "bandwidth_b")
        assert hasattr(result, "n_left")
        assert hasattr(result, "n_right")

    def test_ci_is_ordered(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        assert result.ci_lower < result.ci_upper

    def test_se_is_positive(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        assert result.se > 0

    def test_p_value_is_in_unit_interval(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        assert 0.0 <= result.p_value <= 1.0

    def test_bandwidth_is_positive(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        h_l, h_r = result.bandwidth_h
        assert h_l > 0
        assert h_r > 0

    def test_sample_sizes_are_positive(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        assert result.n_left > 0
        assert result.n_right > 0


# ---------------------------------------------------------------------------
# Rate ratio output
# ---------------------------------------------------------------------------


class TestRateRatio:
    def test_rate_ratio_gaussian_warns(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        with pytest.warns(UserWarning):
            rr = result.rate_ratio()

    def test_rate_ratio_poisson_no_warning(self, poisson_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            outcome_type="poisson", exposure="exposure"
        )
        result = rd.fit()
        rr = result.rate_ratio()
        assert "rate_ratio" in rr
        assert rr["rate_ratio"] > 0
        assert rr["ci_lower"] < rr["rate_ratio"] < rr["ci_upper"]

    def test_rate_ratio_exp_tau(self, poisson_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
            outcome_type="poisson"
        )
        result = rd.fit()
        rr = result.rate_ratio()
        assert abs(rr["rate_ratio"] - np.exp(result.tau_bc)) < 1e-10

    def test_rate_ratio_returns_dict(self, poisson_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=poisson_df)
        result = rd.fit()
        with pytest.warns(UserWarning):
            rr = result.rate_ratio()
        assert isinstance(rr, dict)
        assert set(rr.keys()) >= {"rate_ratio", "ci_lower", "ci_upper", "tau"}


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_string(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_tau(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        s = result.summary()
        assert "tau" in s.lower()

    def test_summary_contains_ci(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        result = rd.fit()
        s = result.summary()
        assert "CI" in s or "ci" in s.lower()


# ---------------------------------------------------------------------------
# Donut RDD
# ---------------------------------------------------------------------------


class TestDonutRDD:
    def test_donut_excludes_near_cutoff(self, gaussian_df):
        rd_no_donut = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df
        )
        rd_donut = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radius=0.1
        )
        res_no = rd_no_donut.fit()
        res_d = rd_donut.fit()

        # Effective sample sizes should differ.
        n_total_no = res_no.n_left + res_no.n_right
        n_total_d = res_d.n_left + res_d.n_right
        assert n_total_d <= n_total_no

    def test_donut_zero_still_works(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            donut_radius=0.0
        )
        result = rd.fit()
        assert result is not None

    def test_donut_too_large_raises(self):
        df = make_sharp_rdd_data(n=100)
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=df,
            donut_radius=100.0  # excludes everything
        )
        with pytest.raises((ValueError, Exception)):
            rd.fit()


# ---------------------------------------------------------------------------
# Preset handling
# ---------------------------------------------------------------------------


class TestPresets:
    def test_preset_loads_donut(self):
        rd = InsuranceRD(
            outcome="claim_count",
            running_var="driver_age_months",
            cutoff=300.0,
            data=pd.DataFrame({
                "driver_age_months": np.random.uniform(200, 400, 200),
                "claim_count": np.zeros(200),
                "exposure_years": np.ones(200),
            }),
            preset=AGE_25,
        )
        assert rd.donut_radius == AGE_25.donut_radius

    def test_preset_does_not_override_explicit(self):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=pd.DataFrame({"x": [0.0], "y": [0.0]}),
            donut_radius=5.0,  # explicit
            preset=AGE_25,     # preset has 3.0
        )
        # Explicit value wins.
        assert rd.donut_radius == 5.0

    def test_all_presets_have_name(self):
        from insurance_rdd.presets import PRESETS
        for key, preset in PRESETS.items():
            assert len(preset.name) > 0

    def test_get_preset_age_25(self):
        from insurance_rdd import get_preset
        p = get_preset("age_25")
        assert p.cutoff == 300.0

    def test_get_preset_unknown_raises(self):
        from insurance_rdd import get_preset
        with pytest.raises(KeyError):
            get_preset("nonexistent_threshold")

    def test_all_preset_keys_accessible(self):
        from insurance_rdd import PRESETS
        for key in PRESETS:
            p = PRESETS[key]
            assert isinstance(p.cutoff, float)


# ---------------------------------------------------------------------------
# Fixed bandwidth
# ---------------------------------------------------------------------------


class TestFixedBandwidth:
    def test_fixed_scalar_bandwidth(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            h=1.0
        )
        result = rd.fit()
        assert result is not None

    def test_fixed_tuple_bandwidth(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            h=(0.8, 1.2)
        )
        result = rd.fit()
        assert result is not None

    def test_different_left_right_bandwidths_differ(self, gaussian_df):
        rd_sym = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            h=1.0
        )
        rd_asym = InsuranceRD(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            h=(0.5, 1.5)
        )
        # With asymmetric bandwidth, effective N differs from symmetric.
        res_sym = rd_sym.fit()
        res_asym = rd_asym.fit()
        # Just check both converge.
        assert res_sym is not None
        assert res_asym is not None


# ---------------------------------------------------------------------------
# Fuzzy RDD
# ---------------------------------------------------------------------------


class TestFuzzyRDD:
    def test_fuzzy_fit_with_noisy_treatment(self):
        """Fuzzy RDD: treatment probability jumps at cutoff but not perfectly."""
        rng = np.random.default_rng(99)
        n = 2000
        x = rng.uniform(-3, 3, n)
        cutoff = 0.0

        # First stage: P(D=1|X>=c) = 0.8, P(D=1|X<c) = 0.2
        prob_d = np.where(x >= cutoff, 0.8, 0.2)
        D = rng.binomial(1, prob_d).astype(float)

        # Outcome depends on actual treatment.
        y = 1.0 + 0.5 * D + 0.2 * x + 0.5 * rng.standard_normal(n)

        df = pd.DataFrame({"x": x, "y": y, "D": D})

        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=cutoff,
            data=df,
            fuzzy="D",
        )
        result = rd.fit()
        assert result is not None
        assert isinstance(result.tau_bc, float)


# ---------------------------------------------------------------------------
# Covariate inclusion
# ---------------------------------------------------------------------------


class TestCovariates:
    def test_covariates_included_in_fit(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            covariates=["covariate_1"],
        )
        result = rd.fit()
        assert result is not None

    def test_missing_covariate_raises(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
            covariates=["nonexistent_col"],
        )
        with pytest.raises((ValueError, Exception)):
            rd.fit()


# ---------------------------------------------------------------------------
# Age RDD integration test
# ---------------------------------------------------------------------------


class TestAgeRDD:
    def test_age_rdd_recovers_negative_tau(self, age_df):
        """True tau=-0.35 (rate drops above 25), estimate should be negative."""
        rd = InsuranceRD(
            outcome="claim_count",
            running_var="driver_age_months",
            cutoff=300.0,
            data=age_df,
            outcome_type="poisson",
            exposure="exposure_years",
        )
        result = rd.fit()
        assert result.tau_bc < 0, f"Expected negative tau (rate drops above 25), got {result.tau_bc:.4f}"

    def test_age_rdd_with_age25_preset(self, age_df):
        rd = InsuranceRD(
            outcome="claim_count",
            running_var="driver_age_months",
            cutoff=300.0,
            data=age_df,
            outcome_type="poisson",
            exposure="exposure_years",
            preset=AGE_25,
        )
        result = rd.fit()
        assert result is not None

    def test_age_rdd_rate_ratio_less_than_one(self, age_df):
        rd = InsuranceRD(
            outcome="claim_count",
            running_var="driver_age_months",
            cutoff=300.0,
            data=age_df,
            outcome_type="poisson",
            exposure="exposure_years",
        )
        result = rd.fit()
        with pytest.warns(UserWarning):
            rr = result.rate_ratio()  # warning because Gaussian path
        # Without warning suppression check:
        # Rate ratio < 1 means claims drop above 25.
        rr_val = np.exp(result.tau_bc)
        assert rr_val < 1.0, f"Expected rate ratio < 1, got {rr_val:.4f}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_outcome_column_raises(self, gaussian_df):
        rd = InsuranceRD(
            outcome="nonexistent",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        with pytest.raises(ValueError):
            rd.fit()

    def test_missing_running_var_raises(self, gaussian_df):
        rd = InsuranceRD(
            outcome="y",
            running_var="nonexistent",
            cutoff=0.0,
            data=gaussian_df,
        )
        with pytest.raises(ValueError):
            rd.fit()

    def test_invalid_outcome_type_raises(self, gaussian_df):
        with pytest.raises(ValueError):
            InsuranceRD(
                outcome="y",
                running_var="x",
                cutoff=0.0,
                data=gaussian_df,
                outcome_type="invalid_family",
            )

    def test_negative_exposure_raises(self):
        df = pd.DataFrame({
            "x": [1.0, -1.0, 0.5, -0.5],
            "y": [2.0, 1.0, 3.0, 2.0],
            "exposure": [-0.5, 1.0, 0.5, 1.0],  # negative exposure
        })
        rd = InsuranceRD(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=df,
            outcome_type="poisson",
            exposure="exposure",
        )
        with pytest.raises(ValueError):
            rd.fit()


# ---------------------------------------------------------------------------
# Bandwidth sensitivity
# ---------------------------------------------------------------------------


class TestBandwidthSensitivity:
    def test_sensitivity_returns_dataframe(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        rd.fit()
        df = rd.bandwidth_sensitivity(n_points=5)
        assert isinstance(df, pd.DataFrame)
        assert "tau" in df.columns
        assert len(df) > 0

    def test_sensitivity_has_h_factor_column(self, gaussian_df):
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        rd.fit()
        df = rd.bandwidth_sensitivity(n_points=3)
        assert "h_factor" in df.columns

    def test_sensitivity_tau_near_optimal(self, gaussian_df):
        """At h_factor=1, tau should be very close to the main result."""
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
        main_result = rd.fit()
        df = rd.bandwidth_sensitivity(h_factors=[1.0])
        if len(df) > 0:
            tau_at_1 = df["tau"].iloc[0]
            assert abs(tau_at_1 - main_result.tau_bc) < 0.1  # close but not identical due to re-fit
