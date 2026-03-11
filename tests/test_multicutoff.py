"""Tests for MultiCutoffRD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd import MultiCutoffRD, MultiCutoffRDResult

from conftest import make_multicutoff_data


class TestMultiCutoffRDInit:
    def test_requires_at_least_two_cutoffs(self, multicutoff_df):
        with pytest.raises(ValueError):
            MultiCutoffRD(
                outcome="claim_count",
                running_var="x",
                cutoffs=[1.0],
                data=multicutoff_df,
            )

    def test_cutoffs_sorted(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[3.0, 1.0, 2.0],
            data=multicutoff_df,
        )
        assert mc.cutoffs == [1.0, 2.0, 3.0]


class TestMultiCutoffRDFit:
    def test_fit_returns_result(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
            outcome_type="poisson",
            exposure="exposure_years",
        )
        result = mc.fit()
        assert isinstance(result, MultiCutoffRDResult)

    def test_result_has_cutoff_effects(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert len(result.cutoff_effects) == 3

    def test_pooled_tau_is_finite(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert np.isfinite(result.pooled_tau)

    def test_pooled_se_positive(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert result.pooled_se > 0

    def test_pooled_ci_ordered(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert result.pooled_ci_lower < result.pooled_ci_upper

    def test_pooled_rate_ratio_is_exp_pooled_tau(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert abs(result.pooled_rate_ratio - np.exp(result.pooled_tau)) < 1e-10

    def test_weights_sum_to_one(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert abs(sum(result.weights) - 1.0) < 1e-10

    def test_cutoff_effects_df_has_correct_shape(self, multicutoff_df):
        cutoffs = [1.0, 2.0, 3.0]
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=cutoffs,
            data=multicutoff_df,
        )
        result = mc.fit()
        df = result.cutoff_effects_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(cutoffs)
        assert "cutoff" in df.columns
        assert "tau" in df.columns

    def test_summary_returns_string(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "Pooled" in s

    def test_pooled_effect_dict(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        pe = result.pooled_effect()
        assert "pooled_tau" in pe
        assert "rate_ratio" in pe
        assert "heterogeneity_p" in pe

    def test_heterogeneity_p_in_unit_interval(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert 0.0 <= result.heterogeneity_p <= 1.0

    def test_pooled_tau_recovers_negative_sign(self, multicutoff_df):
        """True taus are all negative (claims decrease with NCD level)."""
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0, 4.0, 5.0],
            data=multicutoff_df,
            outcome_type="poisson",
            exposure="exposure_years",
        )
        result = mc.fit()
        assert result.pooled_tau < 0, (
            f"Expected negative pooled tau (effect is rate reduction), got {result.pooled_tau:.4f}"
        )

    def test_five_cutoffs(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0, 4.0, 5.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        assert len(result.cutoff_effects) == 5

    def test_per_cutoff_taus_are_finite(self, multicutoff_df):
        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        for e in result.cutoff_effects:
            assert np.isfinite(e.tau), f"Non-finite tau at cutoff {e.cutoff}"

    def test_donut_radius_propagated(self, multicutoff_df):
        """With donut radius, effective N should be smaller."""
        mc_no = MultiCutoffRD(
            outcome="claim_count", running_var="x", cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df
        )
        mc_d = MultiCutoffRD(
            outcome="claim_count", running_var="x", cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df, donut_radius=0.1
        )
        r_no = mc_no.fit()
        r_d = mc_d.fit()
        total_no = sum(e.n_left + e.n_right for e in r_no.cutoff_effects)
        total_d = sum(e.n_left + e.n_right for e in r_d.cutoff_effects)
        assert total_d <= total_no
