"""Tests for the decision engine.

The load-bearing test is `test_significance_alone_never_ships`: a p-value small
enough to be conclusive must still be unable to produce SHIP on its own. The
rest of the suite covers each gate, the three outcomes, and the validity
override.
"""

from __future__ import annotations

import pytest

from experimentlens.config import Thresholds
from experimentlens.stats.decision import (
    DecisionInputs,
    build_inputs,
    decide,
    outcomes,
)

BASE = Thresholds()


class _FakeSrm:
    p_value = 0.9
    passed = True


class _FakeValidity:
    """Minimal stand-in exposing only what the engine reads."""

    def __init__(self, *, valid: bool = True, reasons: tuple[str, ...] = ()) -> None:
        self._valid = valid
        self._reasons = reasons or ("synthetic validity failure",)
        self.srm = _FakeSrm()
        self.blocking_reasons = () if valid else self._reasons

    @property
    def is_valid(self) -> bool:
        return self._valid


def make_inputs(**overrides: float) -> DecisionInputs:
    """Inputs describing a clear, shippable win, overridable per test."""
    defaults: dict[str, float] = {
        "control_rate": 0.100,
        "treatment_rate": 0.115,
        "absolute_difference": 0.015,
        "relative_difference": 0.15,
        "relative_ci_lower": 0.11,
        "relative_ci_upper": 0.19,
        "p_value_two_sided": 0.0001,
        "p_value_one_sided": 0.00005,
        "probability_treatment_better": 0.999,
        "probability_clears_threshold": 0.995,
        "relative_hdi_lower": 0.10,
        "relative_hdi_upper": 0.20,
        "expected_revenue_delta": 50_000.0,
        "achieved_power": 0.95,
        "mde_relative": 0.02,
        "required_n_per_arm": 10_000,
        "observed_n_per_arm": 40_000,
    }
    defaults.update(overrides)
    return build_inputs(**defaults)  # type: ignore[arg-type]


class TestCoreSafeguard:
    def test_significance_alone_never_ships(self) -> None:
        """A conclusive p-value with a sub-threshold lift must not ship.

        Realistic borderline: a +0.4% relative effect that is just significant
        at this sample size, with a credible interval wide enough to still
        contain the worthwhile range. Statistically suggestive, commercially
        below the bar, and uncertain enough to call inconclusive.
        """
        result = decide(
            make_inputs(
                treatment_rate=0.104,
                absolute_difference=0.004,
                relative_difference=0.004,
                relative_ci_lower=-0.010,
                relative_ci_upper=0.018,
                p_value_two_sided=0.044,
                p_value_one_sided=0.022,
                probability_treatment_better=0.68,
                probability_clears_threshold=0.30,
                relative_hdi_lower=-0.010,
                relative_hdi_upper=0.018,
                expected_revenue_delta=12_000.0,
            ),
            _FakeValidity(),
            BASE,
        )

        assert result.outcome == "INCONCLUSIVE"
        assert result.outcome != "SHIP"

    def test_tiny_but_precise_effect_is_rejected_not_shipped(self) -> None:
        """A negligible effect measured to high precision is rejected, not shipped.

        Distinct from the inconclusive case: here the interval is tight enough
        to rule out anything worthwhile, which is a firmer answer than 'not
        proven'. Both outcomes refuse to ship, for different reasons.
        """
        result = decide(
            make_inputs(
                treatment_rate=0.10005,
                absolute_difference=0.00005,
                relative_difference=0.0005,
                relative_ci_lower=0.0004,
                relative_ci_upper=0.0006,
                p_value_two_sided=0.0003,
                p_value_one_sided=0.00015,
                probability_treatment_better=0.995,
                probability_clears_threshold=0.0,
                relative_hdi_lower=0.0004,
                relative_hdi_upper=0.0006,
                expected_revenue_delta=180.0,
            ),
            _FakeValidity(),
            BASE,
        )

        assert result.outcome == "DONT_SHIP"
        assert result.outcome != "SHIP"

    def test_p_value_is_not_a_decision_criterion(self) -> None:
        """Changing only the p-value must not change the outcome."""
        strong = decide(
            make_inputs(p_value_two_sided=1e-30, p_value_one_sided=1e-30),
            _FakeValidity(),
            BASE,
        )
        weak = decide(
            make_inputs(p_value_two_sided=0.99, p_value_one_sided=0.995),
            _FakeValidity(),
            BASE,
        )
        assert strong.outcome == weak.outcome == "SHIP"

    def test_very_high_significance_cannot_rescue_a_trivial_effect(self) -> None:
        for p in (1e-5, 1e-10, 1e-50):
            result = decide(
                make_inputs(
                    relative_difference=0.0005,
                    probability_clears_threshold=0.0,
                    relative_hdi_upper=0.0006,
                    relative_hdi_lower=0.0004,
                    relative_ci_lower=0.0004,
                    relative_ci_upper=0.0006,
                    p_value_two_sided=p,
                    p_value_one_sided=p / 2,
                ),
                _FakeValidity(),
                BASE,
            )
            assert result.outcome != "SHIP"

    def test_trivially_small_effect_cannot_ship_at_any_alpha(self) -> None:
        for alpha in (0.001, 0.05, 0.20):
            th = Thresholds(alpha=alpha)
            result = decide(
                make_inputs(
                    relative_difference=0.001,
                    probability_clears_threshold=0.01,
                    relative_hdi_upper=0.002,
                    relative_hdi_lower=0.0005,
                    relative_ci_lower=0.0005,
                    relative_ci_upper=0.0015,
                ),
                _FakeValidity(),
                th,
            )
            assert result.outcome != "SHIP"


class TestShip:
    def test_clear_win_ships(self) -> None:
        result = decide(make_inputs(), _FakeValidity(), BASE)
        assert result.outcome == "SHIP"
        assert result.label == "SHIP"

    def test_ship_requires_positive_revenue(self) -> None:
        result = decide(
            make_inputs(expected_revenue_delta=-1.0, relative_hdi_upper=-0.02),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome != "SHIP"

    def test_ship_requires_positive_hdi_lower_bound(self) -> None:
        result = decide(
            make_inputs(relative_hdi_lower=-0.01, relative_hdi_upper=-0.01),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome != "SHIP"

    def test_ship_requires_clearing_the_threshold(self) -> None:
        result = decide(
            make_inputs(probability_clears_threshold=0.5),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome != "SHIP"

    def test_strict_threshold_still_ships_a_large_effect(self) -> None:
        th = Thresholds(min_practical_lift_rel=0.10, min_posterior_probability=0.95)
        result = decide(make_inputs(relative_difference=0.20), _FakeValidity(), th)
        assert result.outcome == "SHIP"


class TestDontShip:
    def test_interval_below_threshold_blocks_shipping(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=-0.012,
                relative_hdi_lower=-0.030,
                relative_hdi_upper=0.006,
                probability_clears_threshold=0.012,
                probability_treatment_better=0.12,
                expected_revenue_delta=-5_188.0,
                relative_ci_lower=-0.031,
                relative_ci_upper=0.008,
                p_value_two_sided=0.232,
                p_value_one_sided=0.884,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "DONT_SHIP"
        assert result.label == "DON'T SHIP"

    def test_low_probability_of_being_better_blocks_shipping(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=-0.20,
                probability_treatment_better=0.01,
                relative_hdi_lower=-0.30,
                relative_hdi_upper=-0.10,
                probability_clears_threshold=0.0,
                expected_revenue_delta=-200_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "DONT_SHIP"

    def test_negative_revenue_blocks_shipping(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=-0.05,
                relative_hdi_lower=-0.09,
                relative_hdi_upper=-0.01,
                probability_clears_threshold=0.0,
                probability_treatment_better=0.005,
                expected_revenue_delta=-90_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "DONT_SHIP"

    def test_dont_ship_distinguishes_inconclusive_from_negative(self) -> None:
        """The reasoning must not rest on the p-value alone."""
        result = decide(
            make_inputs(
                relative_difference=-0.012,
                relative_hdi_lower=-0.030,
                relative_hdi_upper=0.006,
                probability_clears_threshold=0.012,
                probability_treatment_better=0.12,
                expected_revenue_delta=-5_188.0,
                p_value_two_sided=0.232,
            ),
            _FakeValidity(),
            BASE,
        )
        joined = " ".join(result.reasoning)
        assert "credible interval" in joined or "credible" in joined


class TestInconclusive:
    def test_wide_interval_overlapping_threshold_is_inconclusive(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=0.004,
                relative_ci_lower=-0.02,
                relative_ci_upper=0.03,
                relative_hdi_lower=-0.02,
                relative_hdi_upper=0.05,
                probability_treatment_better=0.62,
                probability_clears_threshold=0.30,
                expected_revenue_delta=900.0,
                p_value_two_sided=0.31,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "INCONCLUSIVE"
        assert result.label == "INCONCLUSIVE"

    def test_inconclusive_recommends_more_data(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=0.004,
                relative_ci_lower=-0.02,
                relative_ci_upper=0.03,
                relative_hdi_lower=-0.02,
                relative_hdi_upper=0.05,
                probability_treatment_better=0.62,
                probability_clears_threshold=0.30,
                expected_revenue_delta=900.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert "longer" in result.recommended_action or "more data" in (
            result.recommended_action.lower()
        )

    def test_inconclusive_names_the_unresolved_criteria(self) -> None:
        result = decide(
            make_inputs(
                relative_difference=0.004,
                relative_ci_lower=-0.02,
                relative_ci_upper=0.03,
                relative_hdi_lower=-0.02,
                relative_hdi_upper=0.05,
                probability_treatment_better=0.62,
                probability_clears_threshold=0.30,
                expected_revenue_delta=900.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.reasoning
        assert any("Unresolved" in r for r in result.reasoning)


class TestValidityGate:
    def test_invalid_experiment_forces_inconclusive(self) -> None:
        """A huge, significant win still cannot ship on a broken experiment."""
        result = decide(
            make_inputs(),
            _FakeValidity(valid=False, reasons=("SRM failed (p=1e-9)",)),
            BASE,
        )
        assert result.outcome == "INCONCLUSIVE"

    def test_invalid_experiment_names_the_failure(self) -> None:
        result = decide(
            make_inputs(),
            _FakeValidity(valid=False, reasons=("SRM failed (p=1e-9)",)),
            BASE,
        )
        assert "SRM failed" in " ".join(result.reasoning)
        assert result.blocking_reasons

    def test_invalid_experiment_advises_against_shipping(self) -> None:
        result = decide(make_inputs(), _FakeValidity(valid=False), BASE)
        assert "Do not ship" in result.recommended_action


class TestDesignReporting:
    def test_weak_design_is_reported_not_enforced(self) -> None:
        """An inadequate design must not flip an otherwise clear decision."""
        result = decide(
            make_inputs(
                mde_relative=0.25,
                required_n_per_arm=600_000,
                observed_n_per_arm=143_000,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "SHIP"
        assert result.design_notes
        assert any("minimum detectable effect" in n for n in result.design_notes)

    def test_shortfall_is_reported(self) -> None:
        result = decide(
            make_inputs(required_n_per_arm=600_000, observed_n_per_arm=143_000),
            _FakeValidity(),
            BASE,
        )
        assert any("shortfall" in n for n in result.design_notes)

    def test_sound_design_reports_the_passing_check(self) -> None:
        result = decide(
            make_inputs(mde_relative=0.005, required_n_per_arm=1_000, observed_n_per_arm=40_000),
            _FakeValidity(),
            BASE,
        )
        assert any("at or below" in n for n in result.design_notes)


class TestFrequentistBayesianDisagreement:
    """The hard cases: the two frameworks pointing different ways.

    Both normally agree, because they answer nearly the same question. Where
    they diverge the engine must follow the business evidence rather than
    whichever significance statistic happens to look better.
    """

    def test_significant_but_bayesian_clears_nothing_is_not_a_ship(self) -> None:
        """p < 0.05, yet the posterior says the worthwhile range is out of reach."""
        result = decide(
            make_inputs(
                p_value_two_sided=0.031,
                p_value_one_sided=0.016,
                probability_treatment_better=0.75,
                probability_clears_threshold=0.18,
                relative_difference=0.002,
                relative_hdi_lower=-0.004,
                relative_hdi_upper=0.009,
                relative_ci_lower=-0.005,
                relative_ci_upper=0.009,
                expected_revenue_delta=6_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome != "SHIP"

    def test_bayesian_strong_but_frequentist_not_significant_can_ship(self) -> None:
        """A large effect on a modest sample can fail the frequentist test yet
        still clear the practical bar on posterior evidence."""
        result = decide(
            make_inputs(
                p_value_two_sided=0.31,
                p_value_one_sided=0.16,
                probability_treatment_better=0.997,
                probability_clears_threshold=0.96,
                relative_difference=0.12,
                relative_hdi_lower=0.04,
                relative_hdi_upper=0.20,
                expected_revenue_delta=60_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "SHIP"

    def test_posterior_effect_far_beyond_p_value_does_not_force_a_ship(self) -> None:
        """Huge relative lift but with negative expected revenue must not ship."""
        result = decide(
            make_inputs(
                relative_difference=5.0,
                probability_treatment_better=1.0,
                probability_clears_threshold=1.0,
                relative_hdi_lower=1.0,
                relative_hdi_upper=9.0,
                p_value_two_sided=1e-12,
                expected_revenue_delta=-1.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome != "SHIP"

    def test_significant_harm_is_never_inconclusive(self) -> None:
        """When the credible interval sits wholly below the threshold, the
        answer is a definite refusal rather than indecision."""
        result = decide(
            make_inputs(
                p_value_two_sided=0.0001,
                p_value_one_sided=0.9999,
                relative_difference=-0.30,
                relative_hdi_lower=-0.40,
                relative_hdi_upper=-0.22,
                relative_ci_lower=-0.39,
                relative_ci_upper=-0.21,
                probability_treatment_better=0.0,
                probability_clears_threshold=0.0,
                expected_revenue_delta=-400_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "DONT_SHIP"

    def test_mixed_signals_land_on_inconclusive_not_arbitrary_choice(self) -> None:
        """Positive effect, but the posterior cannot rule out the null."""
        result = decide(
            make_inputs(
                p_value_two_sided=0.02,
                p_value_one_sided=0.01,
                probability_treatment_better=0.58,
                probability_clears_threshold=0.25,
                relative_difference=0.004,
                relative_hdi_lower=-0.010,
                relative_hdi_upper=0.020,
                relative_ci_lower=-0.012,
                relative_ci_upper=0.021,
                expected_revenue_delta=2_000.0,
            ),
            _FakeValidity(),
            BASE,
        )
        assert result.outcome == "INCONCLUSIVE"


class TestEvidenceAndRendering:
    def test_every_outcome_has_an_evidence_checklist(self) -> None:
        result = decide(make_inputs(), _FakeValidity(), BASE)
        assert len(result.evidence) == 9
        assert all(c.detail for c in result.evidence)

    def test_passed_and_failed_partition_the_checks(self) -> None:
        result = decide(make_inputs(), _FakeValidity(), BASE)
        assert len(result.passed_checks) + len(result.failed_checks) == len(result.evidence)

    def test_frame_rendering_has_expected_columns(self) -> None:
        frame = decide(make_inputs(), _FakeValidity(), BASE).to_frame()
        assert list(frame.columns) == ["criterion", "passed", "detail"]

    def test_summary_is_renderable(self) -> None:
        text = decide(make_inputs(), _FakeValidity(), BASE).summary()
        assert "DECISION: SHIP" in text
        assert "Evidence:" in text

    def test_thresholds_are_injected_not_read_from_state(self) -> None:
        """The same inputs give different outcomes under different thresholds."""
        inputs = make_inputs(relative_difference=0.05)
        strict = decide(inputs, _FakeValidity(), Thresholds(min_practical_lift_rel=0.10))
        loose = decide(inputs, _FakeValidity(), Thresholds(min_practical_lift_rel=0.01))

        assert strict.outcome != "SHIP"
        assert loose.outcome == "SHIP"


def test_outcomes_are_the_three_declared_states() -> None:
    assert set(outcomes()) == {"SHIP", "DONT_SHIP", "INCONCLUSIVE"}


def test_result_is_immutable() -> None:
    result = decide(make_inputs(), _FakeValidity(), BASE)
    with pytest.raises(AttributeError):
        result.outcome = "SHIP"  # type: ignore[misc]
