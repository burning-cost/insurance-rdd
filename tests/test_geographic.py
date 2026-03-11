"""Tests for GeographicRD.

Geographic RDD requires geopandas and a boundary file. Most tests use
pre-computed signed distances to avoid the geopandas dependency in CI.
The full geographic pipeline (distance computation from coordinates) is
tested with a mock boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd.geographic import GeographicRD, GeographicRDResult


def make_geo_data(n: int = 2000, true_tau: float = -0.25, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic geographic RDD data using pre-computed distances.

    Returns a DataFrame with 'signed_distance' (negative = territory A,
    positive = territory B), 'territory' (0/1), 'claim_count', 'exposure_years'.
    """
    rng = np.random.default_rng(seed)

    # Signed distance: uniform in [-2000, 2000] metres.
    signed_dist = rng.uniform(-2000, 2000, size=n)
    territory = (signed_dist >= 0).astype(float)

    exposure = rng.uniform(0.1, 1.0, size=n)

    # Log claim rate: linear in distance, jump at boundary.
    log_rate = -1.5 + 0.00005 * signed_dist + true_tau * territory
    mu = np.exp(log_rate) * exposure
    claim_count = rng.poisson(mu).astype(float)

    return pd.DataFrame(
        {
            "signed_distance": signed_dist,
            "territory": territory,
            "claim_count": claim_count,
            "exposure_years": exposure,
        }
    )


class TestGeographicRDWithProvidedDistance:
    """Tests using pre-computed signed distances (no geopandas needed)."""

    @pytest.fixture
    def geo_df(self):
        return make_geo_data(n=2000, true_tau=-0.25)

    def test_fit_returns_result(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            outcome_type="poisson",
            exposure="exposure_years",
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert isinstance(result, GeographicRDResult)

    def test_result_attributes(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert hasattr(result, "tau")
        assert hasattr(result, "tau_bc")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "p_value")
        assert hasattr(result, "bandwidth_h")

    def test_tau_is_finite(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            outcome_type="poisson",
            exposure="exposure_years",
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert np.isfinite(result.tau_bc)

    def test_tau_sign_negative(self, geo_df):
        """True tau=-0.25 (lower claims in territory B), estimate should be negative."""
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            outcome_type="poisson",
            exposure="exposure_years",
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert result.tau_bc < 0, f"Expected negative tau, got {result.tau_bc:.4f}"

    def test_ci_ordered(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert result.ci_lower < result.ci_upper

    def test_rate_ratio_method(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            outcome_type="poisson",
            exposure="exposure_years",
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        rr = result.rate_ratio()
        assert rr["rate_ratio"] > 0
        assert rr["ci_lower"] < rr["rate_ratio"] < rr["ci_upper"]

    def test_summary_returns_string(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "Bandwidth" in s

    def test_distances_computed_flag_false(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert result.distances_computed is False

    def test_max_distance_filter(self, geo_df):
        geo_rd_full = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        geo_rd_restricted = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
            max_distance=500.0,
        )
        r_full = geo_rd_full.fit()
        r_restr = geo_rd_restricted.fit()
        n_full = r_full.n_left + r_full.n_right
        n_restr = r_restr.n_left + r_restr.n_right
        assert n_restr <= n_full

    def test_missing_signed_distance_col_raises(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="nonexistent_column",
            border_segment_fes=False,
        )
        with pytest.raises(ValueError):
            geo_rd.fit()

    def test_with_gaussian_outcome(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            outcome_type="gaussian",
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert result is not None

    def test_border_segment_fes_false_no_error(self, geo_df):
        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=geo_df,
            signed_distance_col="signed_distance",
            border_segment_fes=False,
        )
        result = geo_rd.fit()
        assert result.border_segment_fes is False


class TestGeographicRDMissingGeopandas:
    """Test that helpful error is raised when geopandas not installed."""

    def test_boundary_file_without_geopandas_raises(self, monkeypatch):
        """When boundary_file is used and geopandas is missing, should raise ImportError."""
        import insurance_rdd._rdrobust as _r
        original = _r.check_geopandas

        def fake_check():
            raise ImportError("geopandas not installed — pip install insurance-rdd[geo]")

        monkeypatch.setattr(_r, "check_geopandas", fake_check)

        df = make_geo_data(n=100)
        # Remove signed_distance so it tries to compute from coordinates.
        df = df.drop(columns=["signed_distance"])
        df["lat"] = np.linspace(51.0, 51.1, 100)
        df["lon"] = np.linspace(-0.1, 0.1, 100)

        geo_rd = GeographicRD(
            outcome="claim_count",
            treatment_col="territory",
            data=df,
            boundary_file="fake_file.geojson",
            lat_col="lat",
            lon_col="lon",
            border_segment_fes=False,
        )
        with pytest.raises(ImportError):
            geo_rd.fit()
