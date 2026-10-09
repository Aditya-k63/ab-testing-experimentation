"""Tests for the two-proportion z-test.

Known-answer properties are used rather than snapshots: identical rates must
give z=0, the statistic must carry the sign of the observed difference, and the
pooled and unpooled standard errors must agree exactly under the null and
diverge away from it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import make_frame
from experimentlens.stats.frequentist import (
    InvalidProportionError,
    two_proportion_z_test,
    two_proportion_z_test_from_frame,
)


class TestNullCase:
    def test_identical_rates_give_zero_statistic(self) -> None:
        result = two_proportion_z_test(100, 1000, 100, 1000)
        assert result.z_statistic == pytest.approx(0.0)
        assert result.p_value_two_sided == pytest.approx(1.0)

    def test_pooled_and_unpooled_agree_under_the_null(self) -> None:
        result = two_proportion_z_test(120, 1000, 120, 1000)
        assert result.pooled_standard_error == pytest.approx(result.unpooled_standard_error)

    def test_nominal_alpha_is_not_significant_at_the_null(self) -> None:
        result = two_proportion_z_test(120, 1000, 120, 1000)
        assert not result.significant


class TestDirectionAndSignificance:
    def test_positive_difference_gives_positive_z(self) -> None:
        result = two_proportion_z_test(100, 1000, 150, 1000)
        assert result.absolute_difference == pytest.approx(0.05)
        assert result.z_statistic > 0

    def test_negative_difference_gives_negative_z(self) -> None:
        result = two_proportion_z_test(150, 1000, 100, 1000)
        assert result.absolute_difference == pytest.approx(-0.05)
        assert result.z_statistic < 0

    def test_large_effect_is_significant(self) -> None:
        result = two_proportion_z_test(100, 10_000, 200, 10_000)
        assert result.significant
        assert result.p_value_two_sided < 0.05

    def test_tiny_effect_is_not_significant(self) -> None:
        result = two_proportion_z_test(10_000, 100_000, 10_050, 100_000)
        assert not result.significant

    def test_pooled_and_unpooled_diverge_when_rates_differ(self) -> None:
        result = two_proportion_z_test(100, 1000, 150, 1000)
        assert result.pooled_standard_error != pytest.approx(result.unpooled_standard_error)

    def test_pooled_se_uses_the_null_rate(self) -> None:
        result = two_proportion_z_test(100, 1000, 150, 1000)
        expected = math.sqrt(result.pooled_rate * (1 - result.pooled_rate) * (1 / 1000 + 1 / 1000))
        assert result.pooled_standard_error == pytest.approx(expected)

    def test_monotonic_in_effect_size(self) -> None:
        p_values = [
            two_proportion_z_test(100, 10_000, 120, 10_000).p_value_two_sided,
            two_proportion_z_test(100, 10_000, 150, 10_000).p_value_two_sided,
            two_proportion_z_test(100, 10_000, 300, 10_000).p_value_two_sided,
        ]
        assert p_values == sorted(p_values, reverse=True)


class TestAlternatives:
    def test_two_sided_p_is_double_the_one_sided_tail(self) -> None:
        result = two_proportion_z_test(100, 1000, 150, 1000)
        assert result.p_value_two_sided == pytest.approx(2 * result.p_value_one_sided)

    def test_tails_sum_to_one(self) -> None:
        result = two_proportion_z_test(100, 1000, 150, 1000)
        assert result.p_value_one_sided + result.p_value_lower_sided == pytest.approx(1.0)

    def test_larger_selects_the_positive_tail(self) -> None:
        positive = two_proportion_z_test(100, 1000, 150, 1000, alternative="larger")
        negative = two_proportion_z_test(150, 1000, 100, 1000, alternative="larger")

        assert positive.p_value < 0.5
        assert negative.p_value == 1.0

    def test_smaller_selects_the_negative_tail(self) -> None:
        positive = two_proportion_z_test(100, 1000, 150, 1000, alternative="smaller")
        negative = two_proportion_z_test(150, 1000, 100, 1000, alternative="smaller")

        assert negative.p_value < 0.5
        assert positive.p_value == 1.0

    def test_unknown_alternative_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown alternative"):
            two_proportion_z_test(100, 1000, 150, 1000, alternative="sideways")  # type: ignore[arg-type]

    def test_alpha_controls_significance(self) -> None:
        lenient = two_proportion_z_test(1000, 20_000, 1100, 20_000, alpha=0.10)
        strict = two_proportion_z_test(1000, 20_000, 1100, 20_000, alpha=0.001)

        assert lenient.significant
        assert not strict.significant


class TestValidation:
    def test_zero_trials_raises(self) -> None:
        with pytest.raises(InvalidProportionError, match="must be positive"):
            two_proportion_z_test(0, 0, 10, 100)

    def test_negative_trials_raises(self) -> None:
        with pytest.raises(InvalidProportionError, match="must be positive"):
            two_proportion_z_test(10, -5, 10, 100)

    def test_successes_above_trials_raises(self) -> None:
        with pytest.raises(InvalidProportionError, match="0 <= successes"):
            two_proportion_z_test(200, 100, 10, 100)

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(InvalidProportionError, match="0 <= successes"):
            two_proportion_z_test(-1, 100, 10, 100)


class TestDegenerateCases:
    def test_all_zero_outcomes_gives_no_evidence(self) -> None:
        result = two_proportion_z_test(0, 1000, 0, 1000)
        assert result.z_statistic == 0.0
        assert result.p_value_two_sided == 1.0

    def test_all_one_outcomes_gives_no_evidence(self) -> None:
        result = two_proportion_z_test(1000, 1000, 1000, 1000)
        assert result.z_statistic == 0.0
        assert result.p_value_two_sided == 1.0

    def test_zero_versus_some_conversions_is_certain(self) -> None:
        """Zero baseline conversions is decisive, not a degenerate case.

        The pooled rate is small but positive, so the test is well defined and
        returns a very small p-value rather than an infinite statistic.
        """
        result = two_proportion_z_test(0, 1000, 20, 1000)

        assert result.significant
        assert result.p_value_two_sided < 1e-5
        assert result.z_statistic > 0


class TestFromFrame:
    def test_counts_derived_from_arms(self) -> None:
        frame = make_frame(
            n_control=1000, n_treatment=1000, control_conversions=100, treatment_conversions=150
        )
        result = two_proportion_z_test_from_frame(frame)

        assert result.control_successes == 100
        assert result.control_trials == 1000
        assert result.treatment_successes == 150
        assert result.treatment_rate == pytest.approx(0.15)

    def test_unknown_arms_are_ignored(self) -> None:
        frame = make_frame(n_control=100, n_treatment=100)
        frame = frame.assign(group=lambda d: d["group"].mask(d.index == 0, "holdout"))
        result = two_proportion_z_test_from_frame(frame)

        assert result.control_trials == 99
        assert result.treatment_trials == 100

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError):
            two_proportion_z_test_from_frame({"group": []})  # type: ignore[arg-type]

    def test_missing_column_raises(self) -> None:
        with pytest.raises(KeyError, match="converted"):
            two_proportion_z_test_from_frame(pd.DataFrame({"group": ["control", "treatment"]}))


def test_result_is_immutable() -> None:
    result = two_proportion_z_test(100, 1000, 150, 1000)
    with pytest.raises(AttributeError):
        result.z_statistic = 0.0  # type: ignore[misc]


def test_summary_is_renderable() -> None:
    assert "control=" in two_proportion_z_test(100, 1000, 150, 1000).summary()
