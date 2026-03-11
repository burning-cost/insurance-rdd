"""Tests for insurance threshold presets."""

from __future__ import annotations

import pytest

from insurance_rdd.presets import (
    ThresholdPreset,
    AGE_25,
    AGE_25_YEARS,
    NCD_STEP,
    NCD_MAX,
    VEHICLE_AGE_10,
    VEHICLE_AGE_15,
    TENURE_3,
    SUM_INSURED_100K,
    PRESETS,
    get_preset,
)


class TestThresholdPreset:
    def test_age_25_cutoff(self):
        assert AGE_25.cutoff == 300.0

    def test_age_25_donut(self):
        assert AGE_25.donut_radius == 3.0

    def test_age_25_not_discrete(self):
        assert AGE_25.discrete is False

    def test_age_25_years_is_discrete(self):
        assert AGE_25_YEARS.discrete is True

    def test_ncd_step_discrete(self):
        assert NCD_STEP.discrete is True

    def test_ncd_max_cutoff(self):
        assert NCD_MAX.cutoff == 5.0

    def test_vehicle_age_10_cutoff(self):
        assert VEHICLE_AGE_10.cutoff == 120.0

    def test_vehicle_age_15_cutoff(self):
        assert VEHICLE_AGE_15.cutoff == 180.0

    def test_tenure_3_cutoff(self):
        assert TENURE_3.cutoff == 36.0

    def test_sum_insured_100k_cutoff(self):
        assert SUM_INSURED_100K.cutoff == 100_000.0

    def test_all_presets_have_fca_context(self):
        for key, preset in PRESETS.items():
            assert len(preset.fca_context) > 0, f"Preset '{key}' missing fca_context"

    def test_all_presets_have_notes(self):
        for key, preset in PRESETS.items():
            assert len(preset.notes) > 0, f"Preset '{key}' missing notes"

    def test_all_presets_are_frozen(self):
        for key, preset in PRESETS.items():
            with pytest.raises(Exception):
                object.__setattr__(preset, "cutoff", 999.0)

    def test_preset_kernel_is_valid(self):
        valid_kernels = {"triangular", "epanechnikov", "uniform"}
        for key, preset in PRESETS.items():
            assert preset.kernel in valid_kernels, f"Preset '{key}' has invalid kernel '{preset.kernel}'"

    def test_preset_outcome_type_is_valid(self):
        valid_types = {"gaussian", "poisson", "gamma", "tweedie"}
        for key, preset in PRESETS.items():
            assert preset.outcome_type in valid_types

    def test_preset_donut_radius_nonnegative(self):
        for key, preset in PRESETS.items():
            assert preset.donut_radius >= 0.0

    def test_ncd_step_kernel_uniform(self):
        """Discrete running variables should use uniform kernel."""
        assert NCD_STEP.kernel == "uniform"

    def test_sum_insured_gamma_outcome(self):
        """Severity analysis at sum insured boundary uses Gamma."""
        assert SUM_INSURED_100K.outcome_type == "gamma"

    def test_age_25_placebo_cutoffs_populated(self):
        assert len(AGE_25.placebo_cutoffs) > 0

    def test_vehicle_age_10_placebo_cutoffs_populated(self):
        assert len(VEHICLE_AGE_10.placebo_cutoffs) > 0

    def test_get_preset_valid(self):
        p = get_preset("age_25")
        assert p is AGE_25

    def test_get_preset_case_insensitive(self):
        p = get_preset("AGE_25")
        assert p is AGE_25

    def test_get_preset_hyphen_normalised(self):
        p = get_preset("age-25")
        assert p is AGE_25

    def test_get_preset_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_preset("unknown_threshold")

    def test_presets_dict_contains_all_keys(self):
        expected = {
            "age_25", "age_25_years", "ncd_step", "ncd_max",
            "vehicle_age_10", "vehicle_age_15", "tenure_3", "sum_insured_100k"
        }
        assert set(PRESETS.keys()) == expected

    def test_preset_description_nonempty(self):
        for key, preset in PRESETS.items():
            assert len(preset.description) > 0

    def test_ncd_max_notes_mention_manipulation(self):
        assert "manipulat" in NCD_MAX.notes.lower()

    def test_age_25_fca_context_mentions_consumer_duty(self):
        assert "Consumer Duty" in AGE_25.fca_context
