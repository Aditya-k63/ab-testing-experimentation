"""Raw data ingestion.

Reads the experiment CSV exactly as supplied and records an immutable
`RawSnapshot` of the pre-cleaning population. The snapshot is taken here, at the
edge of the pipeline, so that every later cleaning step can be reconciled
against the original data rather than against a moving target.

No cleaning happens in this module. Duplicates, contamination and SRM are all
downstream concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from experimentlens.config import (
    GROUP_COLUMN,
    MISSING_DATA_MESSAGE,
    RAW_DATA_SEARCH_PATHS,
    USER_ID_COLUMN,
)
from experimentlens.data.schema import SchemaReport, validate_schema

if TYPE_CHECKING:
    from collections.abc import Mapping


class RawDataNotFoundError(FileNotFoundError):
    """Raised when the raw experiment CSV cannot be located."""


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    """Immutable record of the experiment population as loaded.

    Captured before any filtering so that cleaning decisions remain auditable.
    """

    n_rows: int
    n_unique_users: int
    group_counts: Mapping[str, int]
    page_counts: Mapping[str, int]

    @property
    def n_duplicate_rows(self) -> int:
        """Rows beyond the first occurrence of each user id."""
        return self.n_rows - self.n_unique_users

    @property
    def has_duplicates(self) -> bool:
        """Whether any user id appears more than once."""
        return self.n_duplicate_rows > 0

    def summary(self) -> str:
        """Compact rendering used in reports and tests."""
        groups = ", ".join(f"{k}={v}" for k, v in sorted(self.group_counts.items()))
        return (
            f"rows={self.n_rows} unique_users={self.n_unique_users} "
            f"duplicate_rows={self.n_duplicate_rows} groups=[{groups}]"
        )


@dataclass(frozen=True, slots=True)
class RawDataset:
    """A loaded, validated but not yet cleaned experiment table."""

    path: Path
    frame: pd.DataFrame
    schema: SchemaReport
    snapshot: RawSnapshot

    @property
    def ok(self) -> bool:
        """Whether the schema contract holds for this dataset."""
        return self.schema.ok


def resolve_raw_path(explicit: Path | None = None) -> Path:
    """Locate the raw experiment CSV.

    An explicit path wins. Otherwise the configured search paths are tried in
    order, which allows the dataset to live in either `data/raw/` or the
    `dataset/` folder without a code change.
    """
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise RawDataNotFoundError(
                f"No file at the supplied path: {candidate}\n\n{MISSING_DATA_MESSAGE}"
            )
        return candidate

    for candidate in RAW_DATA_SEARCH_PATHS:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"    {p}" for p in RAW_DATA_SEARCH_PATHS)
    raise RawDataNotFoundError(
        f"{MISSING_DATA_MESSAGE}\nSearched:\n{searched}"
    )


def _build_snapshot(frame: pd.DataFrame) -> RawSnapshot:
    user_ids = frame[USER_ID_COLUMN]
    group_counts = (
        frame[GROUP_COLUMN].value_counts().to_dict() if GROUP_COLUMN in frame.columns else {}
    )
    page_counts = (
        frame["landing_page"].value_counts().to_dict()
        if "landing_page" in frame.columns
        else {}
    )
    return RawSnapshot(
        n_rows=int(len(frame)),
        n_unique_users=int(user_ids.nunique()) if USER_ID_COLUMN in frame.columns else 0,
        group_counts={str(k): int(v) for k, v in group_counts.items()},
        page_counts={str(k): int(v) for k, v in page_counts.items()},
    )


def load_raw_data(
    path: Path | None = None,
    *,
    raise_on_schema_error: bool = False,
) -> RawDataset:
    """Load and validate the raw experiment table without altering it.

    Args:
        path: Optional explicit path to the CSV. Resolved against the configured
            search paths when omitted.
        raise_on_schema_error: When True a failed schema contract raises
            `SchemaContractError`. When False the caller inspects
            `RawDataset.schema` instead.

    Returns:
        The loaded frame, its schema report, and the pre-cleaning snapshot.
    """
    resolved = resolve_raw_path(path)
    frame = pd.read_csv(resolved)
    frame.columns = [str(c).strip() for c in frame.columns]

    schema = validate_schema(frame)
    if raise_on_schema_error:
        schema.raise_for_contract()

    return RawDataset(
        path=resolved,
        frame=frame,
        schema=schema,
        snapshot=_build_snapshot(frame),
    )


def describe_source(path: Path | None = None) -> pd.DataFrame:
    """Return the raw table with no validation, for quick schema inspection."""
    return pd.read_csv(resolve_raw_path(path), nrows=5)