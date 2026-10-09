"""Tests for the schema contract (`experimentlens.data.schema`).

These verify the behaviour agreed in design: required columns are mandatory,
optional columns are reported but never demanded, unknown columns are
informational rather than fatal, and malformed values are caught before any
statistic is computed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_frame
from experimentlens.config import REQUIRED_COLUMNS
from experimentlens.data.schema import SchemaContractError, validate_schema


class TestRequiredColumns:
    def test_valid_frame_passes(self, valid_frame: pd.DataFrame) -> None:
        report = validate_schema(valid_frame)
        assert report.ok
        assert report.missing_required == ()
        assert report.errors == ()

    def test_exactly_the_required_columns_are_present(self, minimal_frame: pd.DataFrame) -> None:
        report = validate_schema(minimal_frame)
        assert report.missing_required == ()
        assert set(REQUIRED_COLUMNS).issubset(report.columns)

    def test_missing_required_column_fails_contract(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.drop(columns=["converted"])
        report = validate_schema(broken)

        assert not report.ok
        assert "converted" in report.missing_required
        assert any(i.code == "missing_required_column" for i in report.errors)

    def test_missing_timestamp_fails_contract(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.drop(columns=["timestamp"])
        report = validate_schema(broken)
        assert not report.ok
        assert "timestamp" in report.missing_required

    def test_missing_required_column_suppresses_value_checks(
        self, valid_frame: pd.DataFrame
    ) -> None:
        """Value-level checks must not fire on absent columns."""
        broken = valid_frame.drop(columns=["group"])
        report = validate_schema(broken)

        codes = {i.code for i in report.errors}
        assert "missing_required_column" in codes
        assert "unexpected_group_values" not in codes

    def test_raise_for_contract_raises_with_detail(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.drop(columns=["landing_page"])
        report = validate_schema(broken)

        with pytest.raises(SchemaContractError) as excinfo:
            report.raise_for_contract()
        assert "landing_page" in str(excinfo.value)

    def test_raise_for_contract_is_silent_when_valid(self, valid_frame: pd.DataFrame) -> None:
        validate_schema(valid_frame).raise_for_contract()


class TestOptionalColumns:
    def test_optional_column_absent_is_fine(self, minimal_frame: pd.DataFrame) -> None:
        report = validate_schema(minimal_frame)
        assert report.ok
        assert report.found_optional == ()

    def test_optional_column_present_is_reported(self, frame_with_country: pd.DataFrame) -> None:
        report = validate_schema(frame_with_country)
        assert report.ok
        assert "country" in report.found_optional
        assert "country" not in report.unexpected_columns

    def test_optional_column_never_appears_in_missing_required(
        self, minimal_frame: pd.DataFrame
    ) -> None:
        report = validate_schema(minimal_frame)
        assert "country" not in report.missing_required


class TestUnexpectedColumns:
    def test_extra_column_is_informational_not_fatal(self, valid_frame: pd.DataFrame) -> None:
        augmented = valid_frame.assign(some_new_metric=1.0)
        report = validate_schema(augmented)

        assert report.ok
        assert "some_new_metric" in report.unexpected_columns
        assert any(i.code == "unexpected_columns" for i in report.infos)

    def test_extra_column_is_not_reported_as_missing(self, valid_frame: pd.DataFrame) -> None:
        augmented = valid_frame.assign(extra=1)
        report = validate_schema(augmented)
        assert "extra" not in report.missing_required


class TestValueLevelValidation:
    def test_non_binary_outcome_is_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "converted"] = 7
        report = validate_schema(broken)

        assert not report.ok
        assert any(i.code == "outcome_not_binary" for i in report.errors)

    def test_boolean_outcome_is_accepted(self, valid_frame: pd.DataFrame) -> None:
        ok_frame = valid_frame.copy()
        ok_frame["converted"] = ok_frame["converted"].astype(bool)
        assert validate_schema(ok_frame).ok

    def test_float_zero_one_outcome_is_accepted(self, valid_frame: pd.DataFrame) -> None:
        ok_frame = valid_frame.copy()
        ok_frame["converted"] = ok_frame["converted"].astype(float)
        assert validate_schema(ok_frame).ok

    def test_unknown_group_is_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "group"] = "holdout"
        report = validate_schema(broken)

        assert not report.ok
        assert any(i.code == "unexpected_group_values" for i in report.errors)

    def test_unknown_page_is_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "landing_page"] = "some_other_page"
        report = validate_schema(broken)

        assert not report.ok
        assert any(i.code == "unexpected_page_values" for i in report.errors)

    def test_null_user_id_is_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "user_id"] = None
        report = validate_schema(broken)

        assert not report.ok
        assert any(i.code == "user_id_missing_values" for i in report.errors)

    def test_null_outcome_is_a_warning_not_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "converted"] = None
        report = validate_schema(broken)

        assert report.ok
        assert any(i.code == "outcome_missing_values" for i in report.warnings)

    def test_unparseable_timestamp_is_an_error(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken["timestamp"] = "not-a-timestamp"
        report = validate_schema(broken)

        assert not report.ok
        assert any(i.code == "timestamp_unparseable" for i in report.errors)

    def test_string_user_id_is_a_warning(self, valid_frame: pd.DataFrame) -> None:
        ok_frame = valid_frame.copy()
        ok_frame["user_id"] = ok_frame["user_id"].astype(str)
        report = validate_schema(ok_frame)

        assert report.ok
        assert any(i.code == "user_id_not_numeric" for i in report.warnings)


class TestReportShape:
    def test_row_and_column_counts(self, valid_frame: pd.DataFrame) -> None:
        report = validate_schema(valid_frame)
        assert report.n_rows == len(valid_frame)
        assert report.n_columns == len(valid_frame.columns)

    def test_dtypes_are_reported_as_strings(self, valid_frame: pd.DataFrame) -> None:
        report = validate_schema(valid_frame)
        assert all(isinstance(v, str) for v in report.dtypes.values())

    def test_null_counts_are_reported(self, valid_frame: pd.DataFrame) -> None:
        broken = valid_frame.copy()
        broken.loc[broken.index[0], "country"] = None
        augmented = broken.assign(country=broken["country"])
        report = validate_schema(augmented)

        assert "country" in report.missing_values
        assert report.missing_value_total >= 1

    def test_summary_mentions_counts(self, valid_frame: pd.DataFrame) -> None:
        summary = validate_schema(valid_frame).summary()
        assert "rows=200" in summary

    def test_empty_frame_with_columns_is_schema_valid(self) -> None:
        empty = make_frame().iloc[0:0]
        report = validate_schema(empty)
        assert report.ok
        assert report.n_rows == 0

    def test_report_is_immutable(self, valid_frame: pd.DataFrame) -> None:
        report = validate_schema(valid_frame)
        with pytest.raises(AttributeError):
            report.n_rows = 0  # type: ignore[misc]
