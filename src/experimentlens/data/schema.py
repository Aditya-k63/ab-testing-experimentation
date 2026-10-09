"""Schema validation for the raw experiment table.

The contract is explicit rather than assumed: required columns must be
present and well-formed, optional columns are reported but never demanded, and
any extra column is surfaced as informational so a newer dataset variant does
not silently pass or silently fail.

`validate_schema` never raises. It returns a `SchemaReport` so that callers can
decide whether a failed contract is fatal, and the dashboard can render the
detail instead of collapsing into a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from experimentlens.config import (
    BINARY_OUTCOME_COLUMN,
    EXPECTED_GROUPS,
    EXPECTED_PAGES,
    GROUP_COLUMN,
    PAGE_COLUMN,
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMN,
    USER_ID_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SchemaSeverity = Literal["error", "warning", "info"]

_SEVERITY_ORDER: dict[SchemaSeverity, int] = {"error": 0, "warning": 1, "info": 2}


class SchemaContractError(ValueError):
    """Raised when a caller requires the schema contract to hold and it does not."""


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    """A single validation finding."""

    code: str
    severity: SchemaSeverity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class SchemaReport:
    """Structured outcome of validating a raw experiment table."""

    n_rows: int
    n_columns: int
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    missing_required: tuple[str, ...] = ()
    found_optional: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    issues: tuple[SchemaIssue, ...] = ()
    missing_values: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[SchemaIssue, ...]:
        """Issues that make the table unusable for analysis."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[SchemaIssue, ...]:
        """Issues that degrade trust but do not block analysis."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def infos(self) -> tuple[SchemaIssue, ...]:
        """Neutral observations about the table."""
        return tuple(i for i in self.issues if i.severity == "info")

    @property
    def ok(self) -> bool:
        """True when the contract holds: every required column is present and
        no blocking issue was found."""
        return not self.errors

    @property
    def missing_value_total(self) -> int:
        """Total number of null cells across all columns."""
        return sum(self.missing_values.values())

    def summary(self) -> str:
        """Human-readable rendering used by tests and CLI output."""
        header = (
            f"rows={self.n_rows} columns={self.n_columns} "
            f"required_ok={not self.missing_required} errors={len(self.errors)}"
        )
        if not self.issues:
            return f"{header}\nno issues"
        ordered = sorted(
            self.issues, key=lambda i: (_SEVERITY_ORDER[i.severity], i.code)
        )
        body = "\n".join(f"  {issue}" for issue in ordered)
        return f"{header}\n{body}"

    def raise_for_contract(self) -> None:
        """Raise `SchemaContractError` if the contract does not hold."""
        if self.ok:
            return
        details = "\n".join(f"  {issue}" for issue in self.errors)
        raise SchemaContractError(
            f"Raw experiment data failed schema validation:\n{details}\n\n"
            f"Required columns: {list(REQUIRED_COLUMNS)}"
        )


def _check_required_columns(columns: Sequence[str]) -> list[SchemaIssue]:
    present = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if not missing:
        return []
    return [
        SchemaIssue(
            code="missing_required_column",
            severity="error",
            message=(
                f"Required column(s) {missing} are absent. Present columns: {list(columns)}"
            ),
        )
    ]


def _check_binary_outcome(frame: pd.DataFrame) -> list[SchemaIssue]:
    if BINARY_OUTCOME_COLUMN not in frame.columns:
        return []
    series = frame[BINARY_OUTCOME_COLUMN]
    if series.isna().any():
        return [
            SchemaIssue(
                code="outcome_missing_values",
                severity="warning",
                message=(
                    f"'{BINARY_OUTCOME_COLUMN}' contains "
                    f"{int(series.isna().sum())} null value(s); "
                    "these rows must be handled before computing rates."
                ),
            )
        ]
    observed = set(pd.unique(series.dropna()))
    valid = {0, 1, 0.0, 1.0, True, False}
    unexpected = sorted(
        (v for v in observed if v not in valid), key=lambda v: str(type(v).__name__)
    )
    if unexpected:
        return [
            SchemaIssue(
                code="outcome_not_binary",
                severity="error",
                message=(
                    f"'{BINARY_OUTCOME_COLUMN}' must be binary but contains "
                    f"unexpected value(s): {unexpected}"
                ),
            )
        ]
    return []


def _check_categorical_domains(frame: pd.DataFrame) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    if GROUP_COLUMN in frame.columns:
        unknown = sorted(set(frame[GROUP_COLUMN].dropna().unique()) - set(EXPECTED_GROUPS))
        if unknown:
            issues.append(
                SchemaIssue(
                    code="unexpected_group_values",
                    severity="error",
                    message=(
                        f"'{GROUP_COLUMN}' contains unknown arm(s) {unknown}; "
                        f"expected a subset of {list(EXPECTED_GROUPS)}"
                    ),
                )
            )
    if PAGE_COLUMN in frame.columns:
        unknown = sorted(set(frame[PAGE_COLUMN].dropna().unique()) - set(EXPECTED_PAGES))
        if unknown:
            issues.append(
                SchemaIssue(
                    code="unexpected_page_values",
                    severity="error",
                    message=(
                        f"'{PAGE_COLUMN}' contains unknown page(s) {unknown}; "
                        f"expected a subset of {list(EXPECTED_PAGES)}"
                    ),
                )
            )
    return issues


def _check_timestamp(frame: pd.DataFrame) -> list[SchemaIssue]:
    if TIMESTAMP_COLUMN not in frame.columns:
        return []
    try:
        parsed = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise", format="mixed")
    except (ValueError, TypeError) as exc:
        return [
            SchemaIssue(
                code="timestamp_unparseable",
                severity="error",
                message=f"'{TIMESTAMP_COLUMN}' could not be parsed as datetime: {exc}",
            )
        ]
    if parsed.isna().any():
        return [
            SchemaIssue(
                code="timestamp_missing_values",
                severity="warning",
                message=(
                    f"'{TIMESTAMP_COLUMN}' contains {int(parsed.isna().sum())} "
                    "unparseable value(s)."
                ),
            )
        ]
    return []


def _check_user_id(frame: pd.DataFrame) -> list[SchemaIssue]:
    if USER_ID_COLUMN not in frame.columns:
        return []
    series = frame[USER_ID_COLUMN]
    nulls = int(series.isna().sum())
    if nulls:
        return [
            SchemaIssue(
                code="user_id_missing_values",
                severity="error",
                message=f"'{USER_ID_COLUMN}' contains {nulls} null value(s); ids must be present.",
            )
        ]
    if pd.api.types.is_numeric_dtype(series):
        return []
    return [
        SchemaIssue(
            code="user_id_not_numeric",
            severity="warning",
            message=(
                f"'{USER_ID_COLUMN}' has dtype {series.dtype!r}, not numeric. "
                "Duplicate detection still works but is slower."
            ),
        )
    ]


def validate_schema(frame: pd.DataFrame) -> SchemaReport:
    """Validate a raw experiment table against the declared column contract.

    Required columns must exist and hold well-formed values. Optional columns
    are reported when present. Unknown extra columns are informational and do
    not fail the contract, so a superset variant of the dataset still loads.
    """
    columns = tuple(str(c) for c in frame.columns)
    missing_required = tuple(c for c in REQUIRED_COLUMNS if c not in columns)
    optional = tuple(c for c in columns if c in _optional_column_names())
    unexpected = tuple(c for c in columns if c not in REQUIRED_COLUMNS and c not in optional)

    issues: list[SchemaIssue] = []
    issues.extend(_check_required_columns(columns))

    if not missing_required:
        issues.extend(_check_binary_outcome(frame))
        issues.extend(_check_categorical_domains(frame))
        issues.extend(_check_timestamp(frame))
        issues.extend(_check_user_id(frame))

    if unexpected:
        issues.append(
            SchemaIssue(
                code="unexpected_columns",
                severity="info",
                message=(
                    f"Ignored extra column(s) {list(unexpected)}; they are not used by the "
                    "primary analysis."
                ),
            )
        )

    null_counts = {
        str(name): int(count)
        for name, count in frame.isna().sum().items()
        if int(count) > 0
    }
    if null_counts:
        issues.append(
            SchemaIssue(
                code="null_values_present",
                severity="warning",
                message=(
                    f"Null cells found in {sorted(null_counts)} "
                    f"(total {sum(null_counts.values())})."
                ),
            )
        )

    return SchemaReport(
        n_rows=int(len(frame)),
        n_columns=int(len(frame.columns)),
        columns=columns,
        dtypes={str(k): str(v) for k, v in frame.dtypes.items()},
        missing_required=missing_required,
        found_optional=optional,
        unexpected_columns=unexpected,
        issues=tuple(issues),
        missing_values=null_counts,
    )


def _optional_column_names() -> tuple[str, ...]:
    from experimentlens.config import OPTIONAL_COLUMNS

    return OPTIONAL_COLUMNS