"""Tests for power, required sample size, and minimum detectable effect.

The important tests here are round-trips: a design sized to 80% power must
actually achieve it, and an effect at the MDE must sit right at the power
target. Those catch sign errors and unit mistakes that monotonicity checks miss.
"""

from __future__ import annotations

import pytest

from experimentlens.stats.effects import cohens_h
from experimentlens.stats.power import (
    calculate_power,
    design_summary,
    effect_size,
    minimum_detectable_effect,
    post_hoc_power,
    required_sample_size,
)


class TestEffectSize:
    def test_identical_rates_give_zero(self) -> None:
        assert effect_size(0.12, 0.12) == pytest.approx(0.0)

    def test_matches_cohens_h_magnitude(self) -> None:
        assert effect_size(0.10, 0.13) == pytest.approx(abs(cohens_h(0.10, 0.13)))

    def test_baseline_at_boundary_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_rate"):
            effect_size(0.0, 0.1)

    def test_baseline_at_one_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_rate"):
            effect_size(1.0, 0.9)

    def test_treatment_rate_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="treatment_rate"):
            effect_size(0.5, 1.5)


class TestPower:
    def test_power_increases_with_sample_size(self) -> None:
        powers = [calculate_power(0.12, 0.15, n).power for n in (500, 1_000, 5_000)]
        assert powers == sorted(powers)

    def test_power_increases_with_effect_size(self) -> None:
        powers = [calculate_power(0.12, rate, 2_000).power for rate in (0.121, 0.13, 0.15, 0.20)]
        assert powers == sorted(powers)

    def test_power_is_bounded(self) -> None:
        for n in (10, 1_000, 100_000):
            assert 0.0 <= calculate_power(0.12, 0.15, n).power <= 1.0

    def test_no_effect_gives_no_power(self) -> None:
        result = calculate_power(0.12, 0.12, 100_000)
        assert result.power == pytest.approx(0.05, abs=0.02)

    def test_target_is_reported(self) -> None:
        result = calculate_power(0.12, 0.13, 1_000, target_power=0.9)
        assert result.target_power == 0.9
        assert result.meets_target == (result.power >= 0.9)

    def test_relative_effect_is_computed(self) -> None:
        result = calculate_power(0.10, 0.12, 1_000)
        assert result.relative_effect == pytest.approx(0.20)
        assert result.absolute_effect == pytest.approx(0.02)

    def test_zero_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm must be positive"):
            calculate_power(0.12, 0.13, 0)

    def test_summary_is_renderable(self) -> None:
        assert "power=" in calculate_power(0.12, 0.13, 1000).summary()


class TestRequiredSampleSize:
    def test_larger_effect_needs_fewer_users(self) -> None:
        sizes = [
            required_sample_size(0.12, rate).required_n_per_arm
            for rate in (0.121, 0.125, 0.14, 0.20)
        ]
        assert sizes == sorted(sizes, reverse=True)

    def test_higher_power_needs_more_users(self) -> None:
        sizes = [
            required_sample_size(0.12, 0.14, power=p).required_n_per_arm for p in (0.8, 0.9, 0.95)
        ]
        assert sizes == sorted(sizes)

    def test_lower_alpha_needs_more_users(self) -> None:
        sizes = [
            required_sample_size(0.12, 0.14, alpha=a).required_n_per_arm for a in (0.10, 0.05, 0.01)
        ]
        assert sizes == sorted(sizes)

    def test_size_is_positive_integer(self) -> None:
        result = required_sample_size(0.12, 0.13)
        assert isinstance(result.required_n_per_arm, int)
        assert result.required_n_per_arm > 0
        assert result.total_n == 2 * result.required_n_per_arm

    def test_round_trip_the_sized_design_reaches_target_power(self) -> None:
        """The headline guarantee: sized for 80% power, achieves at least 80%."""
        required = required_sample_size(0.12, 0.13, power=0.80)
        achieved = calculate_power(0.12, 0.13, required.required_n_per_arm)

        assert achieved.power >= 0.80
        assert achieved.power < 0.85

    def test_one_fewer_users_falls_below_target(self) -> None:
        """Confirms the solved size is not gratuitously large."""
        required = required_sample_size(0.12, 0.13, power=0.80)
        achieved = calculate_power(0.12, 0.13, required.required_n_per_arm - 1)
        assert achieved.power < 0.80

    def test_zero_effect_raises(self) -> None:
        with pytest.raises(ValueError, match="no finite sample size"):
            required_sample_size(0.12, 0.12)

    def test_invalid_power_raises(self) -> None:
        with pytest.raises(ValueError, match="power must lie"):
            required_sample_size(0.12, 0.13, power=1.5)

    def test_summary_is_renderable(self) -> None:
        assert "required n=" in required_sample_size(0.12, 0.13).summary()


class TestMinimumDetectableEffect:
    def test_mde_decreases_with_sample_size(self) -> None:
        mdes = [minimum_detectable_effect(0.12, n).mde_absolute for n in (1_000, 10_000, 100_000)]
        assert mdes == sorted(mdes, reverse=True)

    def test_mde_increases_with_baseline_rate(self) -> None:
        low = minimum_detectable_effect(0.02, 10_000).mde_absolute
        high = minimum_detectable_effect(0.50, 10_000).mde_absolute
        assert high > low

    def test_detectable_rate_is_above_baseline(self) -> None:
        result = minimum_detectable_effect(0.12, 10_000)
        assert result.detectable_treatment_rate > result.baseline_rate

    def test_round_trip_effect_at_mde_hits_target_power(self) -> None:
        mde = minimum_detectable_effect(0.12, 10_000, power=0.80)
        achieved = calculate_power(0.12, mde.detectable_treatment_rate, 10_000)

        assert achieved.power == pytest.approx(0.80, abs=0.02)

    def test_relative_mde_matches_absolute(self) -> None:
        mde = minimum_detectable_effect(0.12, 10_000)
        assert mde.mde_relative == pytest.approx(mde.mde_absolute / mde.baseline_rate)

    def test_higher_power_target_raises_mde(self) -> None:
        """A harder detection target requires a larger effect to qualify."""
        low = minimum_detectable_effect(0.12, 10_000, power=0.8).mde_absolute
        high = minimum_detectable_effect(0.12, 10_000, power=0.9).mde_absolute
        assert high > low

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="baseline_rate"):
            minimum_detectable_effect(0.0, 1000)
        with pytest.raises(ValueError, match="n_per_arm must be positive"):
            minimum_detectable_effect(0.12, 0)

    def test_summary_is_renderable(self) -> None:
        assert "MDE at n=" in minimum_detectable_effect(0.12, 10_000).summary()


class TestDesignSummary:
    def test_reports_all_design_quantities(self) -> None:
        summary = design_summary(0.12, 143_345, 0.1186)

        for key in (
            "achieved_power",
            "mde_absolute",
            "mde_relative",
            "observed_n_per_arm",
            "required_n_per_arm",
            "shortfall_per_arm",
        ):
            assert key in summary

    def test_shortfall_is_non_negative(self) -> None:
        summary = design_summary(0.12, 10, 0.14)
        assert summary["shortfall_per_arm"] >= 0

    def test_shortfall_is_zero_when_adequately_sized(self) -> None:
        required = required_sample_size(0.12, 0.13, power=0.8).required_n_per_arm
        summary = design_summary(0.12, required, 0.13)
        assert summary["shortfall_per_arm"] == 0.0


def test_post_hoc_power_matches_calculate_power() -> None:
    a = post_hoc_power(0.12, 0.14, 5_000)
    b = calculate_power(0.12, 0.14, 5_000)
    assert a.power == pytest.approx(b.power)
