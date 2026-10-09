"""Experiment validity checks.

These answer "can this experiment be believed at all?" rather than "what was
the effect?". They gate the decision engine: a favourable p-value is irrelevant
if the split was not random or the arms were not observed as assigned.

SRM is deliberately computed on the user-level population *before*
contaminated rows are removed. Removing rows whose group disagrees with their
landing page would quietly rebalance the arms and hide the very defect SRM is
meant to detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from experimentlens.config import (
    BINARY_OUTCOME_COLUMN,
    CONTROL,
    CONTROL_PAGE,
    EXPECTED_GROUPS,
    EXPECTED_RATIO,
    GROUP_COLUMN,
    PAGE_COLUMN,
    TREATMENT,
    TREATMENT_PAGE,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPECTED_PAGE_BY_GROUP: Mapping[str, str] = {
    CONTROL: CONTROL_PAGE,
    TREATMENT: TREATMENT_PAGE,
}


@dataclass(frozen=True, slots=True)
class SrmResult:
    """Outcome of a sample ratio mismatch check."""

    observed_counts: Mapping[str, int]
    expected_counts: Mapping[str, float]
    expected_ratio: float
    chi2_statistic: float
    p_value: float
    degrees_of_freedom: int
    n_total: int
    alpha: float

    @property
    def passed(self) -> bool:
        """True when the observed split is compatible with the declared ratio.

        A pass is necessary but not sufficient evidence that randomisation held;
        it can only detect imbalances large enough to be visible in the counts.
        """
        return self.p_value > self.alpha

    @property
    def observed_ratio(self) -> float:
        """Observed share of the final arm in the population."""
        treatment_count = self.observed_counts.get(TREATMENT, 0)
        if self.n_total == 0:
            return 0.0
        return treatment_count / self.n_total

    @property
    def share_difference(self) -> float:
        """Signed deviation of the observed treatment share from expectation."""
        return self.observed_ratio - self.expected_ratio

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"SRM {verdict}: observed treatment share {self.observed_ratio:.4f} "
            f"vs expected {self.expected_ratio:.4f} "
            f"(chi2={self.chi2_statistic:.4f}, p={self.p_value:.4g}, n={self.n_total})"
        )


@dataclass(frozen=True, slots=True)
class ValidityReport:
    """Aggregated experiment-validity verdict for a population."""

    srm: SrmResult
    n_rows: int
    n_per_arm: Mapping[str, int]
    contamination_rate: float
    duplicate_rate: float
    min_sample_required: int

    @property
    def sufficient_sample(self) -> bool:
        """Whether both arms reach the configured floor."""
        return all(count >= self.min_sample_required for count in self.n_per_arm.values())

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        """Reasons the experiment cannot be trusted, in reporting order."""
        reasons: list[str] = []
        if not self.srm.passed:
            reasons.append(
                f"SRM failed (p={self.srm.p_value:.3g} <= {self.srm.alpha:g}): the arm "
                "split is not compatible with the declared allocation ratio."
            )
        if not self.sufficient_sample:
            reasons.append(
                f"Insufficient sample per arm: {dict(self.n_per_arm)} < {self.min_sample_required}."
            )
        return tuple(reasons)

    @property
    def advisory_notes(self) -> tuple[str, ...]:
        """Observations that reduce confidence without blocking analysis."""
        notes: list[str] = []
        if self.srm.passed:
            notes.append(
                "SRM passed, which is necessary but not sufficient evidence that "
                "randomisation held."
            )
        if self.duplicate_rate > 0:
            notes.append(
                f"{self.duplicate_rate:.2%} of raw rows were repeat occurrences of an "
                "existing user; these were resolved before analysis."
            )
        if self.contamination_rate > 0:
            notes.append(
                f"{self.contamination_rate:.2%} of the raw population had a "
                "group/landing-page mismatch and were excluded."
            )
        return tuple(notes)

    @property
    def is_valid(self) -> bool:
        """True when no blocking validity issue was found."""
        return not self.blocking_reasons

    def summary(self) -> str:
        verdict = "VALID" if self.is_valid else "INVALID"
        return f"Experiment validity: {verdict}\n  {self.srm.summary()}"


def expected_counts_for(
    counts: Mapping[str, int],
    expected_ratio: float = EXPECTED_RATIO,
) -> dict[str, float]:
    """Expected arm sizes under a declared allocation.

    `expected_ratio` is the share of traffic assigned to the treatment arm. The
    complement is assigned to control, so with two arms the test has a single
    degree of freedom.
    """
    total = float(sum(counts.values()))
    return {
        CONTROL: total * (1.0 - expected_ratio),
        TREATMENT: total * expected_ratio,
    }


def sample_ratio_mismatch(
    counts: Mapping[str, int],
    expected_ratio: float = EXPECTED_RATIO,
    alpha: float = 0.001,
) -> SrmResult:
    """Test whether the observed arm split matches the declared allocation.

    Uses a chi-square goodness-of-fit test against the expected binomial split.
    With two arms the expected count per cell is ``n * ratio`` and
    ``n * (1 - ratio)``, giving one degree of freedom.

    The default alpha is far below the outcome test's 0.05, and that direction
    matters. At these sample sizes a sub-percent allocation drift already
    produces a tiny p-value, so a conventional threshold would fire on ordinary
    sampling noise and train the reader to ignore it. Requiring strong evidence
    means a failure only appears when the split is grossly wrong, which is the
    situation where the experiment genuinely cannot be trusted. Note the
    direction of the effect: a lower alpha makes this check *less* likely to
    declare a failure, not more.
    """
    control = int(counts.get(CONTROL, 0))
    treatment = int(counts.get(TREATMENT, 0))
    total = control + treatment

    observed = {CONTROL: control, TREATMENT: treatment}
    expected = expected_counts_for(observed, expected_ratio)

    if total == 0:
        return SrmResult(
            observed_counts=observed,
            expected_counts=expected,
            expected_ratio=expected_ratio,
            chi2_statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=1,
            n_total=0,
            alpha=alpha,
        )

    chi2, p_value = stats.chisquare(
        f_obs=[control, treatment],
        f_exp=[expected[CONTROL], expected[TREATMENT]],
    )

    return SrmResult(
        observed_counts=observed,
        expected_counts=expected,
        expected_ratio=expected_ratio,
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=1,
        n_total=total,
        alpha=alpha,
    )


def sample_ratio_mismatch_from_frame(
    frame: pd.DataFrame,
    expected_ratio: float = EXPECTED_RATIO,
    alpha: float = 0.001,
) -> SrmResult:
    """Compute SRM directly from an experiment table's arm counts."""
    if GROUP_COLUMN not in frame.columns:
        raise KeyError(f"SRM requires the '{GROUP_COLUMN}' column")
    counts = {str(k): int(v) for k, v in frame[GROUP_COLUMN].value_counts().to_dict().items()}
    return sample_ratio_mismatch(counts, expected_ratio, alpha)


def contamination_mask(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows whose group disagrees with the landing page shown.

    A control user viewing the treatment page has been contaminated: the
    recorded arm is not the arm actually experienced, so attributing the
    outcome to it would dilute the effect in an unquantifiable direction.
    """
    required = (GROUP_COLUMN, PAGE_COLUMN)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Contamination check requires column(s) {missing}")

    expected_pages = frame[GROUP_COLUMN].map(EXPECTED_PAGE_BY_GROUP)
    return frame[PAGE_COLUMN].ne(expected_pages)


def count_contaminated(frame: pd.DataFrame) -> int:
    """Number of rows with a group/landing-page mismatch."""
    return int(contamination_mask(frame).sum())


def contamination_rate(frame: pd.DataFrame) -> float:
    """Share of rows with a group/landing-page mismatch."""
    if len(frame) == 0:
        return 0.0
    return count_contaminated(frame) / len(frame)


def arm_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Row counts per arm, restricted to the two declared arms."""
    if GROUP_COLUMN not in frame.columns:
        raise KeyError(f"Arm counts require the '{GROUP_COLUMN}' column")
    counts = frame[GROUP_COLUMN].value_counts().to_dict()
    return {arm: int(counts.get(arm, 0)) for arm in EXPECTED_GROUPS}


def arm_conversion_rates(frame: pd.DataFrame) -> dict[str, float]:
    """Conversion rate per arm, computed only over the declared arms."""
    required = (GROUP_COLUMN, BINARY_OUTCOME_COLUMN)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Conversion rates require column(s) {missing}")

    subset = frame[frame[GROUP_COLUMN].isin(EXPECTED_GROUPS)]
    grouped = subset.groupby(GROUP_COLUMN)[BINARY_OUTCOME_COLUMN]
    totals = grouped.size()
    successes = grouped.sum()
    rates = np.divide(
        successes.to_numpy(dtype=float),
        totals.to_numpy(dtype=float),
        out=np.full(len(totals), np.nan, dtype=float),
        where=totals.to_numpy() > 0,
    )
    return {str(arm): float(rate) for arm, rate in zip(totals.index, rates, strict=True)}


def assess_validity(
    frame: pd.DataFrame,
    *,
    expected_ratio: float = EXPECTED_RATIO,
    srm_alpha: float = 0.001,
    min_sample_per_arm: int = 1000,
    duplicate_rate: float = 0.0,
    srm_counts: Mapping[str, int] | None = None,
) -> ValidityReport:
    """Build the full validity verdict for a population.

    Args:
        frame: The population the effect will be estimated on.
        expected_ratio: Declared treatment share for the SRM null.
        srm_alpha: Significance level for SRM.
        min_sample_per_arm: Floor below which the arms are too small.
        duplicate_rate: Raw duplicate rate, carried through for reporting.
        srm_counts: Arm counts to use for SRM. Defaults to the counts of
            `frame`; pass the pre-cleaning user-level counts to keep SRM
            independent of contamination removal.
    """
    counts = arm_counts(frame)
    srm = (
        sample_ratio_mismatch(srm_counts, expected_ratio, srm_alpha)
        if srm_counts is not None
        else sample_ratio_mismatch(counts, expected_ratio, srm_alpha)
    )

    return ValidityReport(
        srm=srm,
        n_rows=len(frame),
        n_per_arm=counts,
        contamination_rate=contamination_rate(frame),
        duplicate_rate=duplicate_rate,
        min_sample_required=min_sample_per_arm,
    )


def arm_summary_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Per-arm descriptive rows used by the dashboard and EDA notebook."""
    counts = arm_counts(frame)
    subset = frame[frame[GROUP_COLUMN].isin(EXPECTED_GROUPS)]
    conversions = subset.groupby(GROUP_COLUMN)[BINARY_OUTCOME_COLUMN].sum().astype(int).to_dict()
    rates = arm_conversion_rates(frame)
    return [
        {
            "arm": arm,
            "users": counts.get(arm, 0),
            "conversions": int(conversions.get(arm, 0)),
            "conversion_rate": rates.get(arm, float("nan")),
        }
        for arm in EXPECTED_GROUPS
    ]
