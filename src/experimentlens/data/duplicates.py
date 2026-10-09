"""Duplicate user classification and resolution.

A repeated `user_id` is not a cosmetic problem. It breaks the independence
assumption of the binomial model, and depending on how it is handled it can
bias the arm counts or the conversion rate in either direction. Every repeated
user is therefore classified before any rate is computed, and the resolution
policy is a consequence of the class rather than a blanket "keep first".

Classes, in the order they are tested:

``GROUP_CONFLICT``
    The same user appears under more than one ``(group, landing_page)``
    assignment. Randomisation failed for this user, so there is no valid arm to
    attribute the outcome to. The user is excluded.

``OUTCOME_INSTALLED``
    The same user has conflicting ``converted`` values under a single
    assignment. The outcome is not stable for this unit, which means the
    estimand is not well defined for it. The user is excluded.

``TIMESTAMP_SPLIT``
    Consistent assignment and outcome, but multiple exposures at different
    times. Collapsed to the first exposure (first-touch principle).

``REDUNDANT``
    Byte-identical repeated rows. Collapsed to a single record so the user is
    not counted twice.

``SINGLE``
    The user appears once. Not a duplicate; carried through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from experimentlens.config import (
    BINARY_OUTCOME_COLUMN,
    GROUP_COLUMN,
    PAGE_COLUMN,
    TIMESTAMP_COLUMN,
    USER_ID_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

DuplicateClass = Literal[
    "SINGLE",
    "REDUNDANT",
    "TIMESTAMP_SPLIT",
    "GROUP_CONFLICT",
    "OUTCOME_INSTALLED",
]

ConflictPolicy = Literal["exclude_user", "keep_consistent_row"]
"""How to treat a user recorded under conflicting ``(group, landing_page)``
assignments.

``exclude_user``
    Drop every row for that user. The conservative choice: the user appears to
    have been exposed to both experiences, so the outcome cannot be attributed
    to a single arm.

``keep_consistent_row``
    Retain only the row whose group agrees with the landing page, discarding
    the mismatched record. Appropriate when the duplicate is a logging or join
    artefact rather than genuine dual exposure, and it preserves the valid
    observation. Users are still dropped under this policy when no consistent
    row exists.
"""

DUPLICATE_CLASSES: tuple[DuplicateClass, ...] = (
    "GROUP_CONFLICT",
    "OUTCOME_INSTALLED",
    "TIMESTAMP_SPLIT",
    "REDUNDANT",
)

EXCLUDING_CLASSES: frozenset[DuplicateClass] = frozenset(
    {"GROUP_CONFLICT", "OUTCOME_INSTALLED"}
)
"""Classes whose users are dropped entirely rather than collapsed."""

COLLAPSING_CLASSES: frozenset[DuplicateClass] = frozenset({"REDUNDANT", "TIMESTAMP_SPLIT"})
"""Classes whose repeated rows are collapsed to one first exposure."""

_ASSIGNMENT_COLUMNS: tuple[str, ...] = (GROUP_COLUMN, PAGE_COLUMN)


@dataclass(frozen=True, slots=True)
class DuplicateClassification:
    """Per-user duplicate verdicts for a raw experiment table."""

    table: pd.DataFrame
    """One row per `user_id`, with columns ``n_rows``, ``n_assignments``,
    ``n_outcomes``, ``n_timestamps`` and ``duplicate_class``."""

    n_rows_in: int

    @property
    def n_users(self) -> int:
        """Distinct users observed."""
        return int(len(self.table))

    @property
    def counts(self) -> Mapping[DuplicateClass, int]:
        """User counts per class, every class present including zeros."""
        observed = self.table["duplicate_class"].value_counts().to_dict()
        all_classes: tuple[DuplicateClass, ...] = ("SINGLE", *DUPLICATE_CLASSES)
        return {cls: int(observed.get(cls, 0)) for cls in all_classes}

    @property
    def n_repeated_users(self) -> int:
        """Users appearing more than once."""
        return int(self.n_users - self.counts["SINGLE"])

    @property
    def n_excluded_users(self) -> int:
        """Users that will be dropped for conflicting assignment or outcome."""
        return sum(self.counts[cls] for cls in EXCLUDING_CLASSES)

    @property
    def n_collapsed_users(self) -> int:
        """Users whose repeated rows will be collapsed to one exposure."""
        return sum(self.counts[cls] for cls in COLLAPSING_CLASSES)

    @property
    def duplicate_rate(self) -> float:
        """Share of rows that are repeat occurrences of an existing user."""
        if self.n_rows_in == 0:
            return 0.0
        return (self.n_rows_in - self.n_users) / self.n_rows_in

    def users_of(self, *classes: DuplicateClass) -> pd.Index:
        """User ids belonging to the given classes."""
        mask = self.table["duplicate_class"].isin(list(classes))
        return self.table.index[mask]

    def conflicting_users(self) -> pd.Index:
        """User ids excluded due to conflicting assignment or outcome."""
        return self.users_of("GROUP_CONFLICT", "OUTCOME_INSTALLED")

    def outcome_conflicted_users(self) -> pd.Index:
        """User ids with an unstable ``converted`` value.

        These are always dropped: no policy can recover a consistent row when
        the recorded outcomes themselves disagree.
        """
        return self.users_of("OUTCOME_INSTALLED")

    def summary(self) -> str:
        """Compact rendering used in reports."""
        parts = ", ".join(f"{k}={v}" for k, v in self.counts.items())
        return (
            f"rows={self.n_rows_in} users={self.n_users} "
            f"repeated={self.n_repeated_users} "
            f"excluded={self.n_excluded_users} ({parts})"
        )


@dataclass(frozen=True, slots=True)
class DuplicateResolution:
    """Outcome of applying a duplicate policy to a raw table."""

    frame: pd.DataFrame
    """User-level table, one row per user, in first-exposure order."""

    classification: DuplicateClassification
    n_rows_removed: int
    excluded_user_ids: tuple[object, ...] = field(default=())

    @property
    def n_rows_out(self) -> int:
        """Rows remaining after resolution."""
        return int(len(self.frame))

    @property
    def n_users_excluded(self) -> int:
        """Distinct users dropped."""
        return int(len(self.excluded_user_ids))

    @property
    def duplicate_rate_raw(self) -> float:
        """Share of raw rows that were repeat occurrences."""
        return self.classification.duplicate_rate


def classify_duplicates(frame: pd.DataFrame) -> DuplicateClassification:
    """Classify every user by their duplicate disposition.

    Classification is deliberately separate from resolution so that the
    population-level effect of each policy can be inspected before any row is
    dropped.
    """
    missing = [
        col
        for col in (*_ASSIGNMENT_COLUMNS, BINARY_OUTCOME_COLUMN, TIMESTAMP_COLUMN)
        if col not in frame.columns
    ]
    if USER_ID_COLUMN not in frame.columns or missing:
        raise KeyError(
            "Cannot classify duplicates: frame is missing required column(s) "
            f"{[USER_ID_COLUMN, *missing]}"
        )

    grouped = frame.groupby(USER_ID_COLUMN, sort=True)

    n_rows = grouped.size()
    n_assignments = grouped[list(_ASSIGNMENT_COLUMNS)].nunique(dropna=False).max(axis=1)
    n_outcomes = grouped[BINARY_OUTCOME_COLUMN].nunique(dropna=False)
    n_timestamps = grouped[TIMESTAMP_COLUMN].nunique(dropna=False)

    table = pd.DataFrame(
        {
            "n_rows": n_rows,
            "n_assignments": n_assignments,
            "n_outcomes": n_outcomes,
            "n_timestamps": n_timestamps,
        }
    )

    repeated = table["n_rows"] > 1
    conflicted = repeated & (table["n_assignments"] > 1)
    outcome_conflicted = repeated & ~conflicted & (table["n_outcomes"] > 1)
    split = repeated & ~conflicted & ~outcome_conflicted & (table["n_timestamps"] > 1)

    table["duplicate_class"] = "SINGLE"
    table.loc[repeated, "duplicate_class"] = "REDUNDANT"
    table.loc[split, "duplicate_class"] = "TIMESTAMP_SPLIT"
    table.loc[outcome_conflicted, "duplicate_class"] = "OUTCOME_INSTALLED"
    table.loc[conflicted, "duplicate_class"] = "GROUP_CONFLICT"

    table.index.name = USER_ID_COLUMN
    table["duplicate_class"] = table["duplicate_class"].astype("string")

    return DuplicateClassification(table=table, n_rows_in=int(len(frame)))


def _drop_inconsistent_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose group disagrees with the landing page shown."""
    from experimentlens.stats.validity import contamination_mask

    return frame[~contamination_mask(frame)]


def resolve_duplicates(
    frame: pd.DataFrame,
    *,
    conflict_policy: ConflictPolicy = "exclude_user",
) -> DuplicateResolution:
    """Apply the duplicate policy and return the user-level table.

    Users with conflicting assignment are handled by `conflict_policy`. Users
    with conflicting *outcome* are always excluded, because there is no
    consistent row to fall back on. All other repeated users are collapsed to
    their earliest exposure, which is both the first-touch principle and a
    deterministic tie-break when timestamps match.

    The result holds at most one row per user under either policy.
    """
    if conflict_policy not in ("exclude_user", "keep_consistent_row"):
        raise ValueError(
            f"Unknown conflict_policy {conflict_policy!r}; expected one of "
            "'exclude_user', 'keep_consistent_row'"
        )

    classification = classify_duplicates(frame)

    ordered = frame.sort_values(
        [TIMESTAMP_COLUMN, USER_ID_COLUMN], kind="stable", ignore_index=True
    )

    if conflict_policy == "keep_consistent_row":
        drop_users = classification.outcome_conflicted_users()
        ordered = _drop_inconsistent_rows(ordered)
        ordered = ordered.drop_duplicates(subset=[USER_ID_COLUMN], keep="first")
    else:
        drop_users = classification.conflicting_users()
        ordered = ordered.drop_duplicates(subset=[USER_ID_COLUMN], keep="first")

    resolved = ordered[~ordered[USER_ID_COLUMN].isin(drop_users)]
    resolved = resolved.drop_duplicates(subset=[USER_ID_COLUMN], keep="first")
    resolved = resolved.reset_index(drop=True)

    return DuplicateResolution(
        frame=resolved,
        classification=classification,
        n_rows_removed=int(len(frame)) - int(len(resolved)),
        excluded_user_ids=tuple(drop_users.tolist()),
    )