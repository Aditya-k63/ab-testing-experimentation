"""Tests for simulation-based calibration.

These tests are the ones that would catch a broken statistical method that
still returns plausible numbers. They assert behaviour that must hold across
many simulated experiments, not a single stored value.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from experimentlens.simulation import (
    power_curve,
    pvalue_null_distribution,
    simulate_power,
    simulate_type_i_error,
    type_i_error_table,
)

SIMS = 5_000


class TestTypeICalibration:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_empirical_alpha_tracks_nominal(self, alpha: float) -> None:
        result = simulate_type_i_error(n_per_arm=10_000, alpha=alpha, n_simulations=SIMS, seed=1)
        assert result.is_calibrated
        assert abs(result.empirical_alpha - alpha) < 0.015

    def test_null_pvalues_are_approximately_uniform(self) -> None:
        p_values = pvalue_null_distribution(n_per_arm=10_000, n_simulations=20_000, seed=3)
        ks = stats.kstest(p_values, "uniform")
        assert ks.pvalue > 0.01

    def test_null_pvalues_are_not_skewed_low(self) -> None:
        """A test that rejects too often shows up as a mean below 0.5."""
        p_values = pvalue_null_distribution(n_per_arm=10_000, n_simulations=20_000, seed=4)
        assert p_values.mean() == pytest.approx(0.5, abs=0.02)

    def test_calibration_table_reports_every_level(self) -> None:
        table = type_i_error_table(n_simulations=SIMS)
        assert len(table) == 3
        assert set(table["nominal_alpha"]) == {0.01, 0.05, 0.10}
        assert table["calibrated"].all()

    def test_custom_alpha_levels_are_honoured(self) -> None:
        table = type_i_error_table([0.2, 0.3], n_simulations=SIMS)
        assert set(table["nominal_alpha"]) == {0.2, 0.3}

    def test_same_seed_is_reproducible(self) -> None:
        a = simulate_type_i_error(n_simulations=2_000, seed=11)
        b = simulate_type_i_error(n_simulations=2_000, seed=11)
        assert a.empirical_alpha == b.empirical_alpha

    def test_different_seed_changes_the_draw(self) -> None:
        a = simulate_type_i_error(n_simulations=2_000, seed=1)
        b = simulate_type_i_error(n_simulations=2_000, seed=2)
        assert a.empirical_alpha != b.empirical_alpha

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm must be positive"):
            simulate_type_i_error(n_per_arm=0)
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            simulate_type_i_error(n_simulations=0)
        with pytest.raises(ValueError, match="true_rate must lie"):
            simulate_type_i_error(true_rate=1.0)


class TestPowerConsistency:
    @pytest.mark.parametrize("n", [1_000, 5_000, 20_000])
    def test_empirical_power_matches_analytic_two_sided(self, n: int) -> None:
        """Closes the loop between the closed form and real sampling behaviour."""
        result = simulate_power(n, 0.12, 0.13, alternative="two-sided", n_simulations=SIMS, seed=5)
        assert result.is_consistent

    @pytest.mark.parametrize("n", [1_000, 5_000, 20_000])
    def test_empirical_power_matches_analytic_one_sided(self, n: int) -> None:
        result = simulate_power(n, 0.12, 0.13, alternative="larger", n_simulations=SIMS, seed=6)
        assert result.is_consistent

    def test_one_sided_has_more_power_than_two_sided(self) -> None:
        two = simulate_power(2_000, 0.12, 0.13, alternative="two-sided", n_simulations=SIMS, seed=7)
        one = simulate_power(2_000, 0.12, 0.13, alternative="larger", n_simulations=SIMS, seed=7)
        assert one.empirical_power > two.empirical_power

    def test_power_increases_with_sample_size(self) -> None:
        powers = [
            simulate_power(n, 0.12, 0.13, n_simulations=SIMS, seed=8).empirical_power
            for n in (500, 2_000, 10_000, 50_000)
        ]
        assert powers == sorted(powers)

    def test_power_increases_with_effect_size(self) -> None:
        powers = [
            simulate_power(2_000, 0.12, rate, n_simulations=SIMS, seed=9).empirical_power
            for rate in (0.1205, 0.121, 0.125, 0.15)
        ]
        assert powers == sorted(powers)

    def test_null_effect_gives_nominal_rate(self) -> None:
        result = simulate_power(10_000, 0.12, 0.12, n_simulations=SIMS, seed=10)
        assert result.empirical_power == pytest.approx(0.05, abs=0.015)

    def test_power_is_bounded(self) -> None:
        for n in (100, 1_000, 100_000):
            result = simulate_power(n, 0.12, 0.13, n_simulations=1_000, seed=12)
            assert 0.0 <= result.empirical_power <= 1.0

    def test_summary_is_renderable(self) -> None:
        assert "empirical=" in simulate_power(1_000, n_simulations=500).summary()


class TestPowerCurve:
    def test_grid_has_expected_shape(self) -> None:
        curve = power_curve(
            treatment_rates=[0.13, 0.15],
            sample_sizes=[1_000, 5_000],
            n_simulations=1_000,
        )
        assert len(curve) == 4
        assert set(curve["n_per_arm"]) == {1_000, 5_000}

    def test_columns_are_present(self) -> None:
        curve = power_curve(n_simulations=500)
        for column in (
            "n_per_arm",
            "treatment_rate",
            "relative_effect",
            "empirical_power",
            "analytic_power",
        ):
            assert column in curve.columns

    def test_power_rises_across_the_size_grid(self) -> None:
        curve = power_curve(
            treatment_rates=[0.13],
            sample_sizes=[1_000, 5_000, 20_000, 50_000],
            n_simulations=SIMS,
            seed=13,
        )
        assert list(curve["empirical_power"]) == sorted(curve["empirical_power"])

    def test_relative_effect_is_derived(self) -> None:
        curve = power_curve(
            baseline_rate=0.10, treatment_rates=[0.11], sample_sizes=[1_000], n_simulations=500
        )
        assert curve.iloc[0]["relative_effect"] == pytest.approx(0.1)

    def test_defaults_produce_a_usable_grid(self) -> None:
        curve = power_curve(n_simulations=200)
        assert len(curve) > 10
        assert isinstance(curve, pd.DataFrame)
