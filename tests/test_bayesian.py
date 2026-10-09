"""Tests for the Bayesian analysis.

The key test is that the sampled posterior agrees with the closed-form
conjugate answer. Because the Beta-Binomial model is conjugate, the exact
probability is available in closed form, so a disagreement indicates a sampler
or wiring fault rather than approximation error.

Fast settings are used throughout; the posterior means a Beta with hundreds of
thousands of pseudo-counts is dominated by the data, so the sampling budget
needed to pin it down is small.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_frame
from experimentlens.stats.bayesian import (
    BayesianABResult,
    analytic_probability_treatment_better,
    bayesian_ab_test,
    bayesian_ab_test_from_frame,
    highest_density_interval,
)

FAST = {"draws": 300, "tune": 200, "chains": 2, "seed": 42}


class TestHighestDensityInterval:
    def test_interval_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.5, 0.1, size=20_000)
        lower, upper = highest_density_interval(samples, 0.94)
        assert lower < samples.mean() < upper

    def test_wider_mass_gives_wider_interval(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, size=20_000)
        assert (
            highest_density_interval(samples, 0.99)[1] - highest_density_interval(samples, 0.99)[0]
        ) > (
            highest_density_interval(samples, 0.50)[1] - highest_density_interval(samples, 0.50)[0]
        )

    def test_interval_is_tighter_than_full_range(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, size=20_000)
        lower, upper = highest_density_interval(samples, 0.94)
        assert (lower, upper) != (float(samples.min()), float(samples.max()))

    def test_posterior_mass_inside_interval_is_at_least_requested(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, size=50_000)
        lower, upper = highest_density_interval(samples, 0.94)
        assert np.mean((samples >= lower) & (samples <= upper)) >= 0.93

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sample"):
            highest_density_interval(np.array([]))

    def test_invalid_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="hdi_mass"):
            highest_density_interval(np.array([0.1, 0.2]), 1.5)

    def test_single_observation_returns_degenerate_interval(self) -> None:
        lower, upper = highest_density_interval(np.array([0.3]), 0.94)
        assert lower == pytest.approx(0.3)
        assert upper == pytest.approx(0.3)


class TestSamplerAgreement:
    def test_sampled_probability_matches_closed_form(self) -> None:
        """The core correctness check for the whole module."""
        result = bayesian_ab_test(1000, 10_000, 1200, 10_000, **FAST)
        exact = analytic_probability_treatment_better(
            result.treatment.alpha,
            result.treatment.beta,
            result.control.alpha,
            result.control.beta,
        )
        assert result.probability_treatment_better == pytest.approx(exact, abs=0.05)

    def test_posterior_mean_matches_observed_rate_at_scale(self) -> None:
        result = bayesian_ab_test(17_215, 143_293, 17_024, 143_398, **FAST)
        assert result.control.mean == pytest.approx(17_215 / 143_293, abs=0.0005)
        assert result.treatment.mean == pytest.approx(17_024 / 143_398, abs=0.0005)

    def test_posterior_parameters_are_conjugate_updates(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.control.alpha == pytest.approx(101.0)
        assert result.control.beta == pytest.approx(901.0)
        assert result.treatment.alpha == pytest.approx(151.0)
        assert result.treatment.beta == pytest.approx(851.0)


class TestProbabilities:
    def test_probability_is_bounded(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert 0.0 <= result.probability_treatment_better <= 1.0
        assert 0.0 <= result.probability_clears_threshold <= 1.0
        assert 0.0 <= result.probability_harm <= 1.0

    def test_clearly_better_treatment_gives_high_probability(self) -> None:
        result = bayesian_ab_test(100, 1000, 300, 1000, **FAST)
        assert result.probability_treatment_better > 0.99

    def test_clearly_worse_treatment_gives_low_probability(self) -> None:
        result = bayesian_ab_test(300, 1000, 100, 1000, **FAST)
        assert result.probability_treatment_better < 0.01

    def test_identical_arms_give_roughly_half(self) -> None:
        result = bayesian_ab_test(120, 1000, 120, 1000, **FAST)
        assert result.probability_treatment_better == pytest.approx(0.5, abs=0.06)

    def test_impossible_threshold_gives_zero_chance(self) -> None:
        result = bayesian_ab_test(100, 1000, 120, 1000, min_practical_lift_rel=5.0, **FAST)
        assert result.probability_clears_threshold == 0.0

    def test_threshold_below_effect_gives_high_chance(self) -> None:
        result = bayesian_ab_test(100, 1000, 300, 1000, min_practical_lift_rel=0.01, **FAST)
        assert result.probability_clears_threshold > 0.99

    def test_harm_and_better_are_complementary(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.probability_treatment_better + result.probability_harm <= 1.0


class TestIntervals:
    def test_absolute_hdi_brackets_the_mean(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.absolute_hdi_lower < result.absolute_lift_mean < result.absolute_hdi_upper

    def test_relative_hdi_brackets_the_mean(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.relative_hdi_lower < result.relative_lift_mean < result.relative_hdi_upper

    def test_large_effect_gives_interval_away_from_zero(self) -> None:
        result = bayesian_ab_test(100, 10_000, 300, 10_000, **FAST)
        assert result.absolute_hdi_lower > 0

    def test_arm_hdis_bracket_their_means(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        for arm in (result.control, result.treatment):
            assert arm.hdi_lower < arm.mean < arm.hdi_upper

    def test_absolute_lift_matches_rate_difference(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.absolute_lift_mean == pytest.approx(
            result.treatment.mean - result.control.mean
        )


class TestEconomics:
    def test_expected_revenue_is_negative_when_treatment_worse(self) -> None:
        result = bayesian_ab_test(120, 1000, 100, 1000, **FAST)
        assert result.expected_revenue_delta < 0
        assert not result.ship_preferred

    def test_expected_revenue_is_positive_when_treatment_better(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.expected_revenue_delta > 0
        assert result.ship_preferred

    def test_expected_loss_is_never_negative(self) -> None:
        for conversions in (80, 100, 150, 300):
            result = bayesian_ab_test(conversions, 1000, 120, 1000, **FAST)
            assert result.expected_loss_ship >= 0.0

    def test_revenue_per_conversion_scales_the_delta(self) -> None:
        low = bayesian_ab_test(100, 1000, 150, 1000, revenue_per_conversion=10.0, **FAST)
        high = bayesian_ab_test(100, 1000, 150, 1000, revenue_per_conversion=100.0, **FAST)
        assert high.expected_revenue_delta > low.expected_revenue_delta

    def test_ship_preferred_tracks_the_sign_of_the_delta(self) -> None:
        result = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert result.ship_preferred == (result.expected_revenue_delta > 0.0)


class TestReproducibility:
    def test_same_seed_gives_identical_result(self) -> None:
        a = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        b = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
        assert a.probability_treatment_better == b.probability_treatment_better

    def test_different_seed_gives_close_result(self) -> None:
        a = bayesian_ab_test(100, 1000, 150, 1000, draws=300, tune=200, chains=2, seed=1)
        b = bayesian_ab_test(100, 1000, 150, 1000, draws=300, tune=200, chains=2, seed=2)
        assert a.probability_treatment_better == pytest.approx(
            b.probability_treatment_better, abs=0.02
        )

    def test_seed_is_recorded(self) -> None:
        assert bayesian_ab_test(100, 1000, 150, 1000, **FAST).seed == 42


class TestValidation:
    def test_zero_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="control_trials must be positive"):
            bayesian_ab_test(0, 0, 10, 100, **FAST)

    def test_successes_above_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="0 <= successes"):
            bayesian_ab_test(200, 100, 10, 100, **FAST)

    def test_invalid_hdi_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="hdi_mass"):
            bayesian_ab_test(100, 1000, 150, 1000, hdi_mass=2.0, **FAST)


class TestFromFrame:
    def test_counts_derived_from_arms(self) -> None:
        frame = make_frame(
            n_control=2000, n_treatment=2000, control_conversions=200, treatment_conversions=300
        )
        result = bayesian_ab_test_from_frame(frame, **FAST)

        assert result.control.successes == 200
        assert result.control.trials == 2000
        assert result.treatment.successes == 300

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError):
            bayesian_ab_test_from_frame({"group": []}, **FAST)

    def test_missing_column_raises(self) -> None:
        import pandas as pd

        with pytest.raises(KeyError, match="converted"):
            bayesian_ab_test_from_frame(pd.DataFrame({"group": ["control"]}), **FAST)


def test_result_is_immutable() -> None:
    result: BayesianABResult = bayesian_ab_test(100, 1000, 150, 1000, **FAST)
    with pytest.raises(AttributeError):
        result.probability_treatment_better = 1.0  # type: ignore[misc]


def test_summary_is_renderable() -> None:
    assert "P(B>A)" in bayesian_ab_test(100, 1000, 150, 1000, **FAST).summary()
