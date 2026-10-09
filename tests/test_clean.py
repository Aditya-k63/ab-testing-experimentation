"""Tests for the cleaning pipeline and its population ledger.

The central invariant is that cleaning is fully auditable: the recorded steps
must reconstruct the final population exactly, the raw snapshot must be
untouched, and SRM must be computed before contamination removal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_frame
from experimentlens.data.clean import (
    CleaningError,
    CleaningStep,
    clean_experiment,
    summarise,
)
from experimentlens.data.loader import RawDataset, RawSnapshot
from experimentlens.data.schema import validate_schema


def _wrap(frame: pd.DataFrame) -> RawDataset:
    """Build a RawDataset around a synthetic frame."""
    snapshot = RawSnapshot(
        n_rows=len(frame),
        n_unique_users=int(frame["user_id"].nunique()),
        group_counts=frame["group"].value_counts().astype(int).to_dict(),
        page_counts=frame["landing_page"].value_counts().astype(int).to_dict(),
    )
    return RawDataset(
        path=Path("synthetic.csv"),
        frame=frame,
        schema=validate_schema(frame),
        snapshot=snapshot,
    )


class TestCleaningStep:
    def test_valid_step_constructs(self) -> None:
        step = CleaningStep(
            name="s",
            reason="r",
            rows_before=100,
            rows_removed=10,
            rows_after=90,
        )
        assert step.removed_share == pytest.approx(0.10)

    def test_non_reconciling_step_is_rejected(self) -> None:
        with pytest.raises(CleaningError, match="does not reconcile"):
            CleaningStep(name="s", reason="r", rows_before=100, rows_removed=10, rows_after=80)

    def test_negative_counts_are_rejected(self) -> None:
        with pytest.raises(CleaningError, match="negative"):
            CleaningStep(name="s", reason="r", rows_before=100, rows_removed=-5, rows_after=105)

    def test_zero_rows_before_gives_zero_share(self) -> None:
        step = CleaningStep(name="s", reason="r", rows_before=0, rows_removed=0, rows_after=0)
        assert step.removed_share == 0.0

    def test_step_is_immutable(self) -> None:
        step = CleaningStep(name="s", reason="r", rows_before=10, rows_removed=1, rows_after=9)
        with pytest.raises(AttributeError):
            step.rows_after = 0  # type: ignore[misc]


class TestCleanPipeline:
    def test_clean_frame_has_no_contamination(self) -> None:
        frame = make_frame(n_control=500, n_treatment=500, control_page="new_page")
        result = clean_experiment(_wrap(frame))

        assert result.validity.contamination_rate == 0.0

    def test_contaminated_rows_are_removed(self) -> None:
        frame = make_frame(n_control=400, n_treatment=400)
        contaminated_count = 400
        frame = frame.assign(
            landing_page=lambda d: d["landing_page"].mask(d["group"].eq("control"), "new_page")
        )
        result = clean_experiment(_wrap(frame))

        assert result.n_rows_raw == 800
        assert result.n_rows_clean == 400
        assert result.steps[-1].rows_removed == contaminated_count

    def test_step_chain_reconciles(self) -> None:
        frame = make_frame(n_control=300, n_treatment=300)
        result = clean_experiment(_wrap(frame))

        result.verify_reconciliation()
        running = result.n_rows_raw
        for step in result.steps:
            assert step.rows_before == running
            running = step.rows_after
        assert running == result.n_rows_clean

    def test_total_removed_matches_raw_minus_clean(self) -> None:
        frame = make_frame(n_control=200, n_treatment=200, control_page="new_page")
        result = clean_experiment(_wrap(frame))

        assert result.total_removed == result.n_rows_raw - result.n_rows_clean

    def test_duplicate_rows_are_resolved(self) -> None:
        frame = make_frame(n_control=100, n_treatment=100)
        duplicated = pd.concat([frame, frame.head(10)], ignore_index=True)
        result = clean_experiment(_wrap(duplicated))

        assert result.steps[0].name == "resolve_duplicate_users"
        assert result.steps[0].rows_removed == 10

    def test_clean_frame_is_one_row_per_user(self) -> None:
        frame = make_frame(n_control=100, n_treatment=100)
        duplicated = pd.concat([frame, frame.head(20)], ignore_index=True)
        result = clean_experiment(_wrap(duplicated))

        assert result.frame["user_id"].is_unique

    def test_raw_input_is_not_mutated(self) -> None:
        frame = make_frame(n_control=200, n_treatment=200, control_page="new_page")
        raw = _wrap(frame)
        before = raw.frame.copy(deep=True)

        clean_experiment(raw)

        pd.testing.assert_frame_equal(raw.frame, before)

    def test_reconciliation_table_starts_and_ends_correctly(self) -> None:
        frame = make_frame(n_control=200, n_treatment=200)
        table = clean_experiment(_wrap(frame)).reconciliation()

        assert table.iloc[0]["step"] == "raw_load"
        assert table.iloc[-1]["step"] == "analysis_population"
        assert int(table.iloc[0]["after"]) == 400
        assert int(table.iloc[-1]["after"]) == table.iloc[-1].pipe(
            lambda r: int(r["before"]) - int(r["removed"])
        )


class TestSrmOrdering:
    def test_srm_uses_post_dedup_counts_before_contamination_filter(self) -> None:
        """SRM must see contaminated rows, otherwise filtering hides the defect."""
        frame = make_frame(n_control=10_000, n_treatment=10_000)
        frame = frame.assign(
            landing_page=lambda d: d["landing_page"].mask(d["group"].eq("control"), "new_page")
        )
        result = clean_experiment(_wrap(frame))

        assert result.srm_arm_counts == {"control": 10_000, "treatment": 10_000}
        assert result.validity.srm.n_total == 20_000
        assert result.srm_population_names == ("post_dedup",)

    def test_dedup_all_population_keeps_conflicting_users(self) -> None:
        """`dedup_all` is the one-row-per-user basis before any exclusion."""
        base = make_frame(n_control=10, n_treatment=10)
        conflicted = pd.DataFrame(
            [
                {
                    "user_id": 999,
                    "timestamp": "2017-01-01",
                    "group": "control",
                    "landing_page": "old_page",
                    "converted": 0,
                },
                {
                    "user_id": 999,
                    "timestamp": "2017-01-02",
                    "group": "treatment",
                    "landing_page": "new_page",
                    "converted": 1,
                },
            ]
        )
        frame = pd.concat([base, conflicted], ignore_index=True)
        result = clean_experiment(_wrap(frame))

        assert result.srm_populations["post_dedup"]["control"] == 10
        assert result.srm_populations["dedup_all"]["control"] == 11

    def test_srm_comparison_reports_every_population(self) -> None:
        frame = make_frame(n_control=10_000, n_treatment=10_000)
        comparison = clean_experiment(_wrap(frame)).srm_comparison()

        assert set(comparison["population"]) == {
            "raw",
            "dedup_all",
            "post_dedup",
            "final",
        }
        assert bool(comparison["used_for_decision"].sum()) is True
        assert bool(comparison["used_for_decision"].sum()) == 1

    def test_srm_comparison_totals_match_the_pipeline(self) -> None:
        frame = make_frame(n_control=5_000, n_treatment=5_000)
        duplicated = pd.concat([frame, frame.head(50)], ignore_index=True)
        comparison = clean_experiment(_wrap(duplicated)).srm_comparison()

        totals = dict(zip(comparison["population"], comparison["total"], strict=True))
        assert totals["raw"] == 10_050
        assert totals["dedup_all"] == 10_000
        assert totals["post_dedup"] == 10_000

    def test_final_option_would_reflect_cleaned_population(self) -> None:
        frame = make_frame(n_control=10_000, n_treatment=10_000)
        frame = frame.assign(
            landing_page=lambda d: d["landing_page"].mask(d["group"].eq("control"), "new_page")
        )

        post_dedup = clean_experiment(_wrap(frame), srm_on=("post_dedup",))
        final = clean_experiment(_wrap(frame), srm_on=("final",))

        assert post_dedup.srm_arm_counts == {"control": 10_000, "treatment": 10_000}
        assert final.srm_arm_counts == {"control": 0, "treatment": 10_000}

    def test_unknown_srm_population_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown srm_on"):
            clean_experiment(_wrap(make_frame()), srm_on=("nonsense",))

    def test_raw_option_counts_raw_rows(self) -> None:
        frame = make_frame(n_control=100, n_treatment=100)
        duplicated = pd.concat([frame, frame.head(30)], ignore_index=True)
        result = clean_experiment(_wrap(duplicated), srm_on=("raw",))

        assert result.srm_arm_counts["control"] > 100


class TestContaminationAccounting:
    """The contamination rate must state which population it divides by."""

    def test_contamination_rate_is_over_the_examined_population(self) -> None:
        frame = make_frame(n_control=400, n_treatment=400)
        frame = frame.assign(
            landing_page=lambda d: d["landing_page"].mask(d["group"].eq("control"), "new_page")
        )
        result = clean_experiment(_wrap(frame))

        assert result.contamination_rate == pytest.approx(0.5)
        assert result.contamination_rate_over_raw == pytest.approx(0.5)

    def test_rates_differ_when_duplicates_are_also_removed(self) -> None:
        """The two denominators must be genuinely different quantities."""
        base = make_frame(n_control=400, n_treatment=400)
        frame = pd.concat([base, base.head(80)], ignore_index=True)
        frame = frame.assign(
            landing_page=lambda d: d["landing_page"].mask(d["group"].eq("control"), "new_page")
        )
        result = clean_experiment(_wrap(frame))

        step = result.contamination_step
        assert step is not None
        removed = step.rows_removed
        assert result.contamination_rate == pytest.approx(removed / step.rows_before)
        assert result.contamination_rate_over_raw == pytest.approx(removed / result.n_rows_raw)
        assert result.contamination_rate > result.contamination_rate_over_raw
        assert step.rows_before == 800
        assert result.n_rows_raw == 880

    def test_contamination_step_is_found_by_name(self) -> None:
        step = clean_experiment(_wrap(make_frame(n_control=10, n_treatment=10))).contamination_step
        assert step is not None
        assert step.name == "remove_contaminated_rows"

    def test_rates_are_zero_without_contamination(self) -> None:
        result = clean_experiment(_wrap(make_frame(n_control=10, n_treatment=10)))
        assert result.contamination_rate == 0.0
        assert result.contamination_rate_over_raw == 0.0

    def test_old_misleading_name_no_longer_exists(self) -> None:
        result = clean_experiment(_wrap(make_frame()))
        assert not hasattr(result, "contamination_rate_raw")


class TestSummarise:
    def test_summary_reports_ledger_and_validity(self) -> None:
        text = summarise(clean_experiment(_wrap(make_frame(n_control=200, n_treatment=200))))

        assert "RAW SNAPSHOT" in text
        assert "CLEANING LEDGER" in text
        assert "VALIDITY" in text


class TestRealDataset:
    def test_real_dataset_pipeline_runs(self, loaded_raw: RawDataset) -> None:
        result = clean_experiment(loaded_raw)
        result.verify_reconciliation()

        assert result.n_rows_clean > 0
        assert result.validity.contamination_rate == 0.0
        assert result.frame["user_id"].is_unique
