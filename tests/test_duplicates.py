"""Tests for duplicate-user classification and resolution.

Each of the four documented dispositions is exercised with a hand-built frame
whose expected class is obvious by inspection, plus property checks on the
population counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_frame
from experimentlens.data.duplicates import (
    classify_duplicates,
    resolve_duplicates,
)


def _row(user_id: int, group: str, page: str, converted: int, ts: str) -> dict[str, object]:
    return {
        "user_id": user_id,
        "timestamp": ts,
        "group": group,
        "landing_page": page,
        "converted": converted,
    }


class TestClassification:
    def test_single_occurrence_users_are_not_duplicates(self) -> None:
        frame = make_frame(n_control=5, n_treatment=5)
        classification = classify_duplicates(frame)

        assert classification.counts["SINGLE"] == 10
        assert classification.n_repeated_users == 0
        assert classification.n_excluded_users == 0

    def test_identical_rows_are_redundant(self) -> None:
        frame = pd.DataFrame([_row(1, "control", "old_page", 0, "2017-01-01 00:00:00")] * 3)
        classification = classify_duplicates(frame)

        assert classification.counts["REDUNDANT"] == 1
        assert classification.n_excluded_users == 0
        assert classification.n_collapsed_users == 1

    def test_different_timestamps_same_assignment_is_timestamp_split(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 1, "2017-01-01 00:00:00"),
                _row(1, "control", "old_page", 1, "2017-01-05 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)

        assert classification.counts["TIMESTAMP_SPLIT"] == 1
        assert classification.n_excluded_users == 0

    def test_group_conflict_is_detected(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "treatment", "new_page", 1, "2017-01-02 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)

        assert classification.counts["GROUP_CONFLICT"] == 1
        assert classification.n_excluded_users == 1

    def test_page_conflict_counts_as_assignment_conflict(self) -> None:
        """A control user shown the treatment page has no valid assignment."""
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "control", "new_page", 0, "2017-01-02 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)

        assert classification.counts["GROUP_CONFLICT"] == 1

    def test_outcome_instability_is_detected(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "treatment", "new_page", 0, "2017-01-01 00:00:00"),
                _row(1, "treatment", "new_page", 1, "2017-01-02 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)

        assert classification.counts["OUTCOME_INSTALLED"] == 1
        assert classification.n_excluded_users == 1

    def test_assignment_conflict_outranks_outcome_conflict(self) -> None:
        """A user in both arms is a randomisation failure, not an outcome quirk."""
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "treatment", "new_page", 1, "2017-01-02 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)
        assert classification.counts["GROUP_CONFLICT"] == 1
        assert classification.counts["OUTCOME_INSTALLED"] == 0

    def test_outcome_conflict_outranks_timestamp_split(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "control", "old_page", 1, "2017-01-02 00:00:00"),
            ]
        )
        classification = classify_duplicates(frame)
        assert classification.counts["OUTCOME_INSTALLED"] == 1
        assert classification.counts["TIMESTAMP_SPLIT"] == 0

    def test_missing_column_raises(self) -> None:
        frame = make_frame().drop(columns=["group"])
        with pytest.raises(KeyError, match="group"):
            classify_duplicates(frame)


class TestCounts:
    def test_counts_include_every_class(self) -> None:
        classification = classify_duplicates(make_frame())
        assert set(classification.counts) == {
            "SINGLE",
            "REDUNDANT",
            "TIMESTAMP_SPLIT",
            "GROUP_CONFLICT",
            "OUTCOME_INSTALLED",
        }

    def test_duplicate_rate_uses_rows_not_users(self) -> None:
        frame = pd.DataFrame([_row(1, "control", "old_page", 0, "2017-01-01")] * 2)
        classification = classify_duplicates(frame)

        assert classification.n_rows_in == 2
        assert classification.n_users == 1
        assert classification.duplicate_rate == pytest.approx(0.5)

    def test_empty_frame_has_zero_duplicate_rate(self) -> None:
        classification = classify_duplicates(make_frame().iloc[0:0])
        assert classification.duplicate_rate == 0.0

    def test_summary_is_renderable(self) -> None:
        assert "rows=" in classify_duplicates(make_frame()).summary()


class TestConflictPolicy:
    def _conflicted(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "treatment", "new_page", 1, "2017-01-02 00:00:00"),
                _row(2, "control", "old_page", 0, "2017-01-01 00:00:00"),
            ]
        )

    def test_default_policy_excludes_conflicting_user(self) -> None:
        resolution = resolve_duplicates(self._conflicted())
        assert resolution.frame["user_id"].tolist() == [2]

    def test_keep_consistent_row_retains_the_valid_record(self) -> None:
        resolution = resolve_duplicates(self._conflicted(), conflict_policy="keep_consistent_row")

        assert sorted(resolution.frame["user_id"].tolist()) == [1, 2]

    def test_keep_consistent_row_drops_mismatched_record(self) -> None:
        resolution = resolve_duplicates(self._conflicted(), conflict_policy="keep_consistent_row")
        row = resolution.frame[resolution.frame["user_id"] == 1].iloc[0]

        assert row["group"] == "control"
        assert row["landing_page"] == "old_page"

    def test_both_policies_yield_one_row_per_user(self) -> None:
        frame = self._conflicted()
        for policy in ("exclude_user", "keep_consistent_row"):
            resolution = resolve_duplicates(frame, conflict_policy=policy)  # type: ignore[arg-type]
            assert resolution.frame["user_id"].is_unique

    def test_outcome_conflicts_are_dropped_under_both_policies(self) -> None:
        """There is no consistent row to fall back on for an unstable outcome."""
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "control", "old_page", 1, "2017-01-02 00:00:00"),
            ]
        )
        for policy in ("exclude_user", "keep_consistent_row"):
            resolution = resolve_duplicates(frame, conflict_policy=policy)  # type: ignore[arg-type]
            assert resolution.frame.empty

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown conflict_policy"):
            resolve_duplicates(self._conflicted(), conflict_policy="whatever")  # type: ignore[arg-type]

    def test_keep_consistent_row_removes_fewer_rows(self) -> None:
        frame = self._conflicted()
        strict = resolve_duplicates(frame, conflict_policy="exclude_user")
        lenient = resolve_duplicates(frame, conflict_policy="keep_consistent_row")

        assert lenient.n_rows_removed < strict.n_rows_removed


class TestResolution:
    def test_redundant_rows_collapse_to_one(self) -> None:
        frame = pd.DataFrame([_row(1, "control", "old_page", 0, "2017-01-01")] * 4)
        resolution = resolve_duplicates(frame)

        assert len(resolution.frame) == 1
        assert resolution.n_rows_removed == 3
        assert resolution.n_users_excluded == 0

    def test_conflicting_users_are_dropped_entirely(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01 00:00:00"),
                _row(1, "treatment", "new_page", 1, "2017-01-02 00:00:00"),
                _row(2, "control", "old_page", 0, "2017-01-01 00:00:00"),
            ]
        )
        resolution = resolve_duplicates(frame)

        assert resolution.frame["user_id"].tolist() == [2]
        assert resolution.n_users_excluded == 1
        assert resolution.n_rows_removed == 2

    def test_timestamp_split_keeps_first_exposure(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-09 00:00:00"),
                _row(1, "control", "old_page", 0, "2017-01-02 00:00:00"),
            ]
        )
        resolution = resolve_duplicates(frame)

        kept = resolution.frame.iloc[0]
        assert pd.Timestamp(kept["timestamp"]) == pd.Timestamp("2017-01-02 00:00:00")

    def test_resolution_is_one_row_per_user(self) -> None:
        frame = pd.DataFrame(
            [_row(u, "control", "old_page", 0, "2017-01-01") for u in range(3)]
            + [_row(3, "control", "old_page", 0, "2017-01-01")] * 2
        )
        resolution = resolve_duplicates(frame)

        assert resolution.frame["user_id"].is_unique

    def test_resolution_does_not_mutate_input(self) -> None:
        frame = pd.DataFrame([_row(1, "control", "old_page", 0, "2017-01-01")] * 3)
        before = frame.copy(deep=True)
        resolve_duplicates(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test_row_arithmetic_reconciles(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, "control", "old_page", 0, "2017-01-01"),
                _row(1, "control", "old_page", 0, "2017-01-02"),
                _row(2, "control", "old_page", 1, "2017-01-01"),
                _row(3, "treatment", "new_page", 1, "2017-01-01"),
                _row(3, "control", "old_page", 0, "2017-01-02"),
            ]
        )
        resolution = resolve_duplicates(frame)

        assert resolution.classification.n_rows_in - resolution.n_rows_removed == (
            resolution.n_rows_out
        )
