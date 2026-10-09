"""Experiment cleaning with an auditable population ledger.

Every transformation that removes rows is recorded as a `CleaningStep` carrying
its before/after counts, so the population can always be reconciled back to the
raw snapshot. Nothing is dropped silently, and the arithmetic is checked as the
steps are constructed rather than being verified later by hand.

The pipeline is:

1. Resolve duplicate users (collapse repeats, exclude conflicts).
2. Compute SRM on the user-level population, *before* contamination removal.
3. Remove group/landing-page contaminated rows.

The ordering matters. Filtering contamination rebalances the arms, so running
SRM afterwards would partly test the cleaning step rather than the
randomisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from experimentlens.config import (
    CONTROL,
    EXPECTED_RATIO,
    GROUP_COLUMN,
    Thresholds,
    TREATMENT,
    USER_ID_COLUMN,
)
from experimentlens.data.duplicates import (
    ConflictPolicy,
    DuplicateClassification,
    resolve_duplicates,
)
from experimentlens.data.loader import RawDataset, RawSnapshot
from experimentlens.stats.validity import (
    ValidityReport,
    assess_validity,
    contamination_mask,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class CleaningError(RuntimeError):
    """Raised when a cleaning step's arithmetic does not reconcile."""


SRM_POPULATION_ORDER: tuple[str, ...] = ("raw", "dedup_all", "post_dedup", "final")
"""Candidate populations for the SRM check, ordered from least to most cleaned."""


@dataclass(frozen=True, slots=True)
class CleaningStep:
    """One recorded exclusion from the experiment population."""

    name: str
    reason: str
    rows_before: int
    rows_removed: int
    rows_after: int

    def __post_init__(self) -> None:
        if self.rows_before - self.rows_removed != self.rows_after:
            raise CleaningError(
                f"Cleaning step '{self.name}' does not reconcile: "
                f"{self.rows_before} - {self.rows_removed} != {self.rows_after}"
            )
        if min(self.rows_before, self.rows_removed, self.rows_after) < 0:
            raise CleaningError(
                f"Cleaning step '{self.name}' has a negative count: "
                f"before={self.rows_before} removed={self.rows_removed} "
                f"after={self.rows_after}"
            )

    @property
    def removed_share(self) -> float:
        """Share of the incoming population removed by this step."""
        if self.rows_before == 0:
            return 0.0
        return self.rows_removed / self.rows_before

    def to_row(self) -> dict[str, object]:
        """Render as a reconciliation-table row."""
        return {
            "step": self.name,
            "reason": self.reason,
            "before": self.rows_before,
            "removed": self.rows_removed,
            "after": self.rows_after,
            "removed_pct": f"{self.removed_share:.4%}",
        }


@dataclass(frozen=True, slots=True)
class CleanResult:
    """A cleaned experiment population plus its full audit trail."""

    frame: pd.DataFrame
    snapshot: RawSnapshot
    steps: tuple[CleaningStep, ...]
    duplicates: DuplicateClassification
    validity: ValidityReport
    srm_arm_counts: dict[str, int]
    thresholds: Thresholds
    srm_populations: dict[str, dict[str, int]] = field(default_factory=dict)
    srm_population_names: tuple[str, ...] = ("post_dedup",)
    excluded_user_ids: tuple[object, ...] = field(default=())

    @property
    def n_rows_raw(self) -> int:
        """Rows in the raw file."""
        return self.snapshot.n_rows

    @property
    def n_rows_clean(self) -> int:
        """Rows in the analysis population."""
        return int(len(self.frame))

    @property
    def total_removed(self) -> int:
        """Rows removed across every step."""
        return sum(step.rows_removed for step in self.steps)

    @property
    def duplicate_rate(self) -> float:
        """Share of raw rows that were repeat occurrences."""
        return self.duplicates.duplicate_rate

    @property
    def contamination_step(self) -> CleaningStep | None:
        """The step that removed contaminated rows, if one ran."""
        for step in self.steps:
            if step.name == "remove_contaminated_rows":
                return step
        return None

    @property
    def contamination_rate(self) -> float:
        """Share of the population *examined by the contamination filter*.

        The denominator is the post-duplicate population, not the raw file.
        Naming this over the raw count would misstate it, so the counts are
        exposed alongside via `contamination_step` instead.
        """
        step = self.contamination_step
        return step.removed_share if step is not None else 0.0

    @property
    def contamination_rate_over_raw(self) -> float:
        """Share of the *raw* population that was contaminated.

        Equivalent in numerator to `contamination_rate` but expressed against
        the raw row count, which is the figure worth quoting when describing
        the dataset as a whole.
        """
        step = self.contamination_step
        if step is None or self.n_rows_raw == 0:
            return 0.0
        return step.rows_removed / self.n_rows_raw

    def srm_comparison(
        self, expected_ratio: float = EXPECTED_RATIO
    ) -> pd.DataFrame:
        """Run SRM across every candidate population.

        Reporting all of them, not only the one that drives the decision, makes
        the choice of basis auditable: a reader can see whether the conclusion
        depends on which population is chosen. It should not, and confirming
        that it does not is the point.
        """
        from experimentlens.stats.validity import sample_ratio_mismatch

        labels = {
            "raw": "raw rows (repeat users counted twice)",
            "dedup_all": "one row per user, all users",
            "post_dedup": "one row per user, conflicts excluded",
            "final": "cleaned analysis population",
        }
        rows: list[dict[str, object]] = []
        for name in SRM_POPULATION_ORDER:
            counts = self.srm_populations.get(name)
            if not counts:
                continue
            result = sample_ratio_mismatch(counts, expected_ratio, self.thresholds.srm_alpha)
            rows.append(
                {
                    "population": name,
                    "description": labels.get(name, name),
                    "control": result.observed_counts[CONTROL],
                    "treatment": result.observed_counts[TREATMENT],
                    "total": result.n_total,
                    "chi2": result.chi2_statistic,
                    "p_value": result.p_value,
                    "verdict": "PASS" if result.passed else "FAIL",
                    "used_for_decision": name in self.srm_population_names,
                }
            )
        return pd.DataFrame(rows)

    def reconciliation(self) -> pd.DataFrame:
        """Population reconciliation table from raw to analysis."""
        rows: list[dict[str, object]] = [
            {
                "step": "raw_load",
                "reason": "Rows as read from the source file.",
                "before": 0,
                "removed": 0,
                "after": self.n_rows_raw,
                "removed_pct": "0.0000%",
            }
        ]
        rows.extend(step.to_row() for step in self.steps)
        rows.append(
            {
                "step": "analysis_population",
                "reason": "Population used for the conversion analysis.",
                "before": self.n_rows_raw,
                "removed": self.total_removed,
                "after": self.n_rows_clean,
                "removed_pct": (
                    f"{(self.total_removed / self.n_rows_raw):.4%}"
                    if self.n_rows_raw
                    else "0.0000%"
                ),
            }
        )
        return pd.DataFrame(rows)

    def verify_reconciliation(self) -> None:
        """Assert the step chain reconstructs the final population.

        Guards the invariant that the sum of removals equals the difference
        between the raw snapshot and the cleaned frame.
        """
        if self.n_rows_raw - self.total_removed != self.n_rows_clean:
            raise CleaningError(
                "Cleaning ledger does not reconcile with the frames: "
                f"raw={self.n_rows_raw} removed={self.total_removed} "
                f"clean={self.n_rows_clean}"
            )
        running = self.n_rows_raw
        for step in self.steps:
            if step.rows_before != running:
                raise CleaningError(
                    f"Step '{step.name}' expects {step.rows_before} rows "
                    f"but the running population is {running}"
                )
            running = step.rows_after


def _srm_counts_from_raw(snapshot: RawSnapshot) -> dict[str, int]:
    """User-level arm counts for SRM, from the raw snapshot."""
    return {
        CONTROL: int(snapshot.group_counts.get(CONTROL, 0)),
        TREATMENT: int(snapshot.group_counts.get(TREATMENT, 0)),
    }


def clean_experiment(
    raw: RawDataset,
    *,
    thresholds: Thresholds | None = None,
    expected_ratio: float = EXPECTED_RATIO,
    srm_on: Sequence[str] = ("post_dedup",),
    conflict_policy: ConflictPolicy = "exclude_user",
) -> CleanResult:
    """Clean a raw dataset and record every exclusion.

    Args:
        raw: The loaded, schema-validated dataset. Its frame is never mutated.
        thresholds: Decision thresholds; defaults to `config.DEFAULT_THRESHOLDS`.
        expected_ratio: Declared treatment share for the SRM null.
        srm_on: Which population SRM is computed on. Four are available:

            ``raw``
                Raw rows as loaded. Shown for completeness but not the right
                basis for the test: repeated users are counted more than once,
                so this measures the de-duplication pattern as much as the
                randomisation.
            ``dedup_all``
                One row per user, every user retained. True randomisation
                unit, before conflicting users are dropped.
            ``post_dedup`` (default)
                One row per user with conflicting assignments excluded, but
                contaminated rows still present. This is the defensible
                default: filtering contamination rebalances the arms and would
                mask the very defect SRM exists to detect.
            ``final``
                The cleaned analysis population.

    Returns:
        A `CleanResult` with the cleaned frame, the step ledger, the
        classification, and the validity verdict.
    """
    th = thresholds if thresholds is not None else Thresholds()
    raw_rows = int(len(raw.frame))

    resolution = resolve_duplicates(raw.frame, conflict_policy=conflict_policy)
    after_duplicates = resolution.frame

    duplicate_step = CleaningStep(
        name="resolve_duplicate_users",
        reason=(
            f"Policy '{conflict_policy}'. Collapsed repeat exposures to one record "
            "per user; excluded users with conflicting assignment "
            f"({resolution.classification.counts['GROUP_CONFLICT']}) or conflicting "
            f"outcome ({resolution.classification.counts['OUTCOME_INSTALLED']})."
        ),
        rows_before=raw_rows,
        rows_removed=resolution.n_rows_removed,
        rows_after=after_duplicates.shape[0],
    )

    mask = contamination_mask(after_duplicates)
    cleaned = after_duplicates[~mask].reset_index(drop=True)

    def _counts_of(source: pd.DataFrame) -> dict[str, int]:
        return {
            CONTROL: int((source[GROUP_COLUMN] == CONTROL).sum()),
            TREATMENT: int((source[GROUP_COLUMN] == TREATMENT).sum()),
        }

    dedup_all = (
        raw.frame.sort_values(
            ["timestamp", USER_ID_COLUMN], kind="stable", ignore_index=True
        )
        .drop_duplicates(subset=[USER_ID_COLUMN], keep="first")
        .reset_index(drop=True)
    )

    srm_populations: dict[str, dict[str, int]] = {
        "raw": _srm_counts_from_raw(raw.snapshot),
        "dedup_all": _counts_of(dedup_all),
        "post_dedup": _counts_of(after_duplicates),
        "final": _counts_of(cleaned),
    }
    unknown = set(srm_on) - set(srm_populations)
    if unknown:
        raise ValueError(
            f"Unknown srm_on option(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(srm_populations)}"
        )
    srm_counts = srm_populations[srm_on[0]]

    contamination_step = CleaningStep(
        name="remove_contaminated_rows",
        reason=(
            "Excluded rows whose group disagrees with the landing page shown "
            "(treatment/old_page or control/new_page), which would otherwise "
            "dilute the measured effect."
        ),
        rows_before=after_duplicates.shape[0],
        rows_removed=int(mask.sum()),
        rows_after=cleaned.shape[0],
    )

    steps = (duplicate_step, contamination_step)

    validity = assess_validity(
        cleaned,
        expected_ratio=expected_ratio,
        srm_alpha=th.srm_alpha,
        min_sample_per_arm=th.min_sample_per_arm,
        duplicate_rate=resolution.duplicate_rate_raw,
        srm_counts=srm_counts,
    )

    result = CleanResult(
        frame=cleaned,
        snapshot=raw.snapshot,
        steps=steps,
        duplicates=resolution.classification,
        validity=validity,
        srm_arm_counts=srm_counts,
        thresholds=th,
        srm_populations=srm_populations,
        srm_population_names=(srm_on[0],),
        excluded_user_ids=resolution.excluded_user_ids,
    )
    result.verify_reconciliation()
    return result


def summarise(result: CleanResult) -> str:
    """Multi-line rendering of the cleaning ledger and validity verdict."""
    lines: list[str] = [
        "RAW SNAPSHOT",
        f"  {result.snapshot.summary()}",
        "",
        "DUPLICATE CLASSIFICATION",
        f"  {result.duplicates.summary()}",
        "",
        "CLEANING LEDGER",
    ]
    for step in result.steps:
        lines.append(
            f"  {step.name}: {step.rows_before} -> {step.rows_after} "
            f"(-{step.rows_removed}, {step.removed_share:.4%})"
        )
    lines.append(f"  analysis population: {result.n_rows_clean}")
    lines.append("")
    lines.append("VALIDITY")
    lines.append(f"  {result.validity.srm.summary()}")
    for reason in result.validity.blocking_reasons:
        lines.append(f"  BLOCKING: {reason}")
    for note in result.validity.advisory_notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)