"""Tests for experiment validity checks (SRM, contamination, arm statistics).

The SRM tests use known-answer properties: a perfectly balanced split must
pass, a deliberately skewed split must fail, and the chi-square statistic must
match the closed-form value for a two-cell goodness-of-fit test.
"""

from __future__ import annotations

import math

import pytest

from conftest import make_frame
from experimentlens.stats.validity import (
    arm_conversion_rates,
    arm_counts,
    arm_summary_rows,
    assess_validity,
    contamination_mask,
    contamination_rate,
    count_contaminated,
    expected_counts_for,
    sample_ratio_mismatch,
    sample_ratio_mismatch_from_frame,
)


class TestSrm:
    def test_perfectly_balanced_split_passes(self) -> None:
        result = sample_ratio_mismatch({"control": 50_000, "treatment": 50_000})

        assert result.passed
        assert result.observed_ratio == pytest.approx(0.5)
        assert result.chi2_statistic == pytest.approx(0.0, abs=1e-9)

    def test_empty_population_passes_without_dividing_by_zero(self) -> None:
        result = sample_ratio_mismatch({"control": 0, "treatment": 0})

        assert result.n_total == 0
        assert result.passed
        assert result.p_value == 1.0
        assert result.observed_ratio == 0.0

    def test_large_skew_fails(self) -> None:
        result = sample_ratio_mismatch({"control": 20_000, "treatment": 80_000})

        assert not result.passed
        assert result.observed_ratio == pytest.approx(0.8)

    def test_chi2_matches_closed_form(self) -> None:
        """For a two-cell GOF, chi2 = (n1 - E1)^2 / E1 + (n2 - E2)^2 / E2."""
        control, treatment = 147_202, 147_276
        total = control + treatment
        e_control, e_treatment = total / 2, total / 2

        expected = (control - e_control) ** 2 / e_control + (
            treatment - e_treatment
        ) ** 2 / e_treatment

        result = sample_ratio_mismatch({"control": control, "treatment": treatment})
        assert result.chi2_statistic == pytest.approx(expected)
        assert result.degrees_of_freedom == 1

    def test_expected_counts_follow_declared_ratio(self) -> None:
        expected = expected_counts_for({"control": 60, "treatment": 40}, 0.5)
        assert expected["treatment"] == pytest.approx(50.0)
        assert expected["control"] == pytest.approx(50.0)

    def test_uneven_ratio_is_honoured(self) -> None:
        result = sample_ratio_mismatch({"control": 50, "treatment": 50}, expected_ratio=0.9)
        assert not result.passed

    def test_share_difference_is_signed(self) -> None:
        result = sample_ratio_mismatch({"control": 45, "treatment": 55})
        assert result.share_difference == pytest.approx(0.05)

    def test_lower_alpha_flags_fewer_imbalances(self) -> None:
        """A low alpha requires stronger evidence before declaring failure.

        This split sits at p~0.01: it clears the 0.001 SRM threshold but not a
        conventional 0.05, which is exactly why the SRM alpha is set low.
        """
        strict = sample_ratio_mismatch({"control": 149_294, "treatment": 150_706}, alpha=0.001)
        lenient = sample_ratio_mismatch({"control": 149_294, "treatment": 150_706}, alpha=0.05)

        assert strict.p_value == pytest.approx(0.00996, abs=1e-4)
        assert strict.passed
        assert not lenient.passed

    def test_summary_mentions_verdict(self) -> None:
        summary = sample_ratio_mismatch({"control": 50_000, "treatment": 50_000}).summary()
        assert "SRM PASS" in summary

    def test_from_frame_counts_arms(self) -> None:
        result = sample_ratio_mismatch_from_frame(make_frame(n_control=100, n_treatment=100))
        assert result.observed_counts == {"control": 100, "treatment": 100}
        assert result.passed

    def test_from_frame_requires_group_column(self) -> None:
        with pytest.raises(KeyError, match="group"):
            sample_ratio_mismatch_from_frame(make_frame().drop(columns=["group"]))


class TestContamination:
    def test_clean_frame_has_no_contamination(self) -> None:
        frame = make_frame()
        assert count_contaminated(frame) == 0
        assert contamination_rate(frame) == 0.0

    def test_mismatched_arm_page_is_contaminated(self) -> None:
        frame = make_frame(control_page="new_page")
        assert count_contaminated(frame) == 100

    def test_mask_marks_only_mismatched_rows(self) -> None:
        frame = make_frame(n_control=10, n_treatment=10)
        mask = contamination_mask(frame)

        assert mask.sum() == 0
        assert (~mask).all()

    def test_mask_length_matches_frame(self) -> None:
        frame = make_frame(n_control=7, n_treatment=11)
        assert len(contamination_mask(frame)) == len(frame)

    def test_empty_frame_rate_is_zero(self) -> None:
        assert contamination_rate(make_frame().iloc[0:0]) == 0.0

    def test_missing_page_column_raises(self) -> None:
        with pytest.raises(KeyError, match="landing_page"):
            contamination_mask(make_frame().drop(columns=["landing_page"]))


class TestArmStatistics:
    def test_arm_counts_reports_both_arms(self) -> None:
        frame = make_frame(n_control=30, n_treatment=40)
        assert arm_counts(frame) == {"control": 30, "treatment": 40}

    def test_conversion_rates_match_design(self) -> None:
        frame = make_frame(
            n_control=100, n_treatment=100, control_conversions=10, treatment_conversions=20
        )
        rates = arm_conversion_rates(frame)

        assert rates["control"] == pytest.approx(0.10)
        assert rates["treatment"] == pytest.approx(0.20)

    def test_conversion_rates_are_proportions(self) -> None:
        frame = make_frame(n_control=50, n_treatment=50, control_conversions=0)
        rates = arm_conversion_rates(frame)

        assert 0.0 <= rates["control"] <= 1.0
        assert 0.0 <= rates["treatment"] <= 1.0

    def test_summary_rows_have_consistent_conversion_counts(self) -> None:
        frame = make_frame(
            n_control=100, n_treatment=100, control_conversions=10, treatment_conversions=20
        )
        rows = {r["arm"]: r for r in arm_summary_rows(frame)}

        assert rows["control"]["conversions"] == 10
        assert rows["treatment"]["conversions"] == 20
        for row in rows.values():
            assert row["conversions"] <= row["users"]

    def test_undisclosed_arms_are_excluded_from_rates(self) -> None:
        frame = make_frame(n_control=10, n_treatment=10)
        frame = frame.assign(group=lambda d: d["group"].mask(d.index == 0, "holdout"))

        assert set(arm_conversion_rates(frame)) == {"control", "treatment"}


class TestValidityReport:
    def test_valid_experiment_passes(self) -> None:
        frame = make_frame(n_control=5000, n_treatment=5000)
        report = assess_validity(frame, min_sample_per_arm=1000)

        assert report.is_valid
        assert report.blocking_reasons == ()
        assert report.srm.passed
        assert report.sufficient_sample

    def test_srm_failure_blocks_validity(self) -> None:
        frame = make_frame(n_control=3000, n_treatment=7000)
        report = assess_validity(frame, min_sample_per_arm=1000)

        assert not report.is_valid
        assert any("SRM failed" in r for r in report.blocking_reasons)

    def test_small_sample_blocks_validity(self) -> None:
        frame = make_frame(n_control=10, n_treatment=10)
        report = assess_validity(frame, min_sample_per_arm=1000)

        assert not report.is_valid
        assert any("Insufficient sample" in r for r in report.blocking_reasons)

    def test_srm_counts_override_frame_counts(self) -> None:
        """SRM must be computable on a different population than the frame."""
        frame = make_frame(n_control=50, n_treatment=50)
        report = assess_validity(
            frame,
            srm_counts={"control": 20_000, "treatment": 80_000},
            min_sample_per_arm=1,
        )

        assert not report.srm.passed
        assert report.srm.observed_ratio == pytest.approx(0.8)

    def test_advisory_notes_flag_duplicate_rate(self) -> None:
        frame = make_frame(n_control=2000, n_treatment=2000)
        report = assess_validity(frame, duplicate_rate=0.013)

        assert report.is_valid
        assert any("repeat occurrences" in n for n in report.advisory_notes)

    def test_passing_srm_is_qualified_as_insufficient(self) -> None:
        """A SRM pass must never be presented as proof randomisation held."""
        report = assess_validity(make_frame(n_control=2000, n_treatment=2000))

        assert any("not sufficient" in n for n in report.advisory_notes)

    def test_summary_reports_verdict(self) -> None:
        report = assess_validity(make_frame(n_control=2000, n_treatment=2000))
        assert "Experiment validity: VALID" in report.summary()

    def test_report_is_immutable(self) -> None:
        report = assess_validity(make_frame())
        with pytest.raises(AttributeError):
            report.n_rows = 0  # type: ignore[misc]


def test_srm_chi2_is_non_negative() -> None:
    for control in (10, 100, 1000):
        for treatment in (10, 120, 1400):
            result = sample_ratio_mismatch({"control": control, "treatment": treatment})
            assert result.chi2_statistic >= 0.0
            assert not math.isnan(result.p_value)
            assert 0.0 <= result.p_value <= 1.0
