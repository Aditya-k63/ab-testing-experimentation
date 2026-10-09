"""Tests for effect sizes and confidence intervals.

Cohen's h is cross-validated against `statsmodels.proportion_effectsize` rather
than against a hard-coded number, so the test keeps its meaning if the upstream
convention is ever restated.
"""

from __future__ import annotations

import math

import pytest
from statsmodels.stats.proportion import proportion_effectsize

from experimentlens.stats.effects import (
    absolute_effect,
    cohens_h,
    effect_summary,
    relative_effect,
    summarise_effects,
)
from experimentlens.stats.frequentist import two_proportion_z_test


class TestCohensH:
    def test_identical_rates_give_zero(self) -> None:
        assert cohens_h(0.12, 0.12) == pytest.approx(0.0)

    def test_sign_follows_direction(self) -> None:
        assert cohens_h(0.10, 0.12) > 0
        assert cohens_h(0.12, 0.10) < 0

    def test_matches_statsmodels_in_magnitude(self) -> None:
        """statsmodels uses the opposite sign convention; magnitude must agree.

        Power calculations depend on |h| only, so that is what is cross-checked
        against the upstream definition.
        """
        for p0, p1 in ((0.12, 0.125), (0.05, 0.09), (0.40, 0.30)):
            assert abs(cohens_h(p0, p1)) == pytest.approx(
                abs(proportion_effectsize(p0, p1)), abs=1e-12
            )

    def test_symmetric_under_argument_swap(self) -> None:
        assert cohens_h(0.10, 0.14) == pytest.approx(-cohens_h(0.14, 0.10))

    def test_known_value_at_half(self) -> None:
        """h(0.5, 0.6) is a standard worked example."""
        expected = 2 * (math.asin(math.sqrt(0.6)) - math.asin(math.sqrt(0.5)))
        assert cohens_h(0.5, 0.6) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"p1|p2"):
            cohens_h(bad, 0.5)

    def test_boundary_rates_are_allowed(self) -> None:
        assert math.isfinite(cohens_h(0.0, 1.0))


class TestAbsoluteEffect:
    def test_difference_matches_rates(self) -> None:
        effect = absolute_effect(0.10, 0.13, 1000, 1000)
        assert effect.difference == pytest.approx(0.03)

    def test_interval_contains_zero_when_rates_match(self) -> None:
        effect = absolute_effect(0.12, 0.12, 10_000, 10_000)
        assert effect.ci_lower < 0 < effect.ci_upper
        assert not effect.excludes_zero

    def test_interval_excludes_zero_for_large_effect(self) -> None:
        effect = absolute_effect(0.10, 0.20, 10_000, 10_000)
        assert effect.excludes_zero
        assert effect.ci_lower > 0

    def test_interval_is_symmetric_about_the_estimate(self) -> None:
        effect = absolute_effect(0.10, 0.13, 1000, 1000)
        midpoint = (effect.ci_lower + effect.ci_upper) / 2
        assert midpoint == pytest.approx(effect.difference)

    def test_wider_interval_at_higher_confidence(self) -> None:
        narrow = absolute_effect(0.10, 0.13, 1000, 1000, 0.90)
        wide = absolute_effect(0.10, 0.13, 1000, 1000, 0.99)
        assert wide.ci_width > narrow.ci_width

    def test_narrower_interval_with_more_data(self) -> None:
        small = absolute_effect(0.10, 0.13, 100, 100)
        large = absolute_effect(0.10, 0.13, 100_000, 100_000)
        assert large.ci_width < small.ci_width

    def test_zero_sample_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            absolute_effect(0.10, 0.13, 0, 100)

    def test_invalid_confidence_level_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            absolute_effect(0.10, 0.13, 100, 100, 1.5)


class TestRelativeEffect:
    def test_relative_lift_is_ratio_minus_one(self) -> None:
        effect = relative_effect(0.10, 0.12, 1000, 1000)
        assert effect.relative_difference == pytest.approx(0.20)

    def test_interval_contains_zero_under_the_null(self) -> None:
        effect = relative_effect(0.12, 0.12, 10_000, 10_000)
        assert effect.ci_lower < 0 < effect.ci_upper

    def test_interval_excludes_zero_for_large_effect(self) -> None:
        effect = relative_effect(0.10, 0.20, 10_000, 10_000)
        assert effect.excludes_zero
        assert effect.ci_lower > 0

    def test_negative_lift_is_reported_correctly(self) -> None:
        """A small negative lift at this n is correctly reported as inconclusive."""
        effect = relative_effect(0.12, 0.118, 100_000, 100_000)

        assert effect.relative_difference < 0
        assert effect.ci_lower < 0 < effect.ci_upper
        assert not effect.excludes_zero

    def test_large_negative_lift_excludes_zero(self) -> None:
        effect = relative_effect(0.20, 0.10, 100_000, 100_000)

        assert effect.relative_difference < 0
        assert effect.ci_upper < 0

    def test_clears_checks_the_lower_bound(self) -> None:
        effect = relative_effect(0.10, 0.20, 10_000, 10_000)
        assert effect.clears(0.01)
        assert not effect.clears(0.90)

    def test_zero_control_rate_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="absolute difference"):
            relative_effect(0.0, 0.1, 100, 100)

    def test_zero_treatment_rate_is_handled(self) -> None:
        effect = relative_effect(0.10, 0.0, 1000, 1000)
        assert effect.relative_difference == 0.0


class TestEffectSummary:
    def test_summary_carries_all_three_quantities(self) -> None:
        summary = effect_summary(1000, 10_000, 1300, 10_000)

        assert summary.absolute_difference == pytest.approx(0.03)
        assert summary.relative_difference == pytest.approx(0.03 / 0.10)
        assert summary.cohens_h > 0

    def test_summary_from_existing_test_matches(self) -> None:
        test = two_proportion_z_test(1000, 10_000, 1300, 10_000)
        summary = summarise_effects(test)

        assert summary.control_rate == pytest.approx(test.control_rate)
        assert summary.cohens_h == pytest.approx(cohens_h(test.control_rate, test.treatment_rate))

    def test_absolute_ci_excludes_zero_while_relative_excludes_threshold(self) -> None:
        summary = effect_summary(1000, 10_000, 1300, 10_000)
        assert summary.absolute.excludes_zero
        assert summary.relative.excludes_zero

    def test_components_are_consistent(self) -> None:
        summary = effect_summary(1200, 10_000, 1320, 10_000)
        assert summary.absolute_difference == pytest.approx(
            summary.treatment_rate - summary.control_rate
        )

    def test_summary_is_renderable(self) -> None:
        text = effect_summary(100, 1000, 150, 1000).summary()
        assert "control=" in text
        assert "absolute" in text
        assert "h=" in text
