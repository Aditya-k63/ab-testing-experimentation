"""Two-proportion z-test for a binary outcome.

This is the primary inferential test for the experiment: the outcome is
`converted` in {0, 1}, so the comparison of interest is between two binomial
proportions rather than between two means.

Two standard errors appear in the literature and they are not
interchangeable:

* the **pooled** standard error, used for the test statistic, which assumes a
  common rate under the null;
* the **unpooled** standard error, used for the confidence interval, which
  does not assume equality.

Conflating them yields intervals that are too narrow and p-values that
disagree with the interval. Both are computed and exposed separately here so
the discrepancy cannot hide.

For the large-sample count data in this project the Wald test is appropriate;
with per-arm counts in the hundreds of thousands the approximation is safe. For
small counts a Fisher exact test would be preferred, and that limitation is
recorded rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy import stats

Alternative = Literal["two-sided", "larger", "smaller"]


class InvalidProportionError(ValueError):
    """Raised when counts are not a valid pair of binomial totals."""


@dataclass(frozen=True, slots=True)
class TwoProportionTest:
    """Result of comparing two conversion rates."""

    control_successes: int
    control_trials: int
    treatment_successes: int
    treatment_trials: int
    control_rate: float
    treatment_rate: float
    absolute_difference: float
    pooled_rate: float
    pooled_standard_error: float
    unpooled_standard_error: float
    z_statistic: float
    p_value_one_sided: float
    p_value_lower_sided: float
    p_value_two_sided: float
    alternative: Alternative
    alpha: float

    @property
    def relative_difference(self) -> float:
        """Relative lift of treatment over control."""
        if self.control_rate == 0:
            return float("nan")
        return self.absolute_difference / self.control_rate

    @property
    def p_value(self) -> float:
        """P-value matching the requested alternative."""
        if self.alternative == "two-sided":
            return self.p_value_two_sided
        if self.alternative == "smaller":
            return self.p_value_lower_sided if self.absolute_difference < 0 else 1.0
        return self.p_value_one_sided if self.absolute_difference > 0 else 1.0

    @property
    def significant(self) -> bool:
        """Whether the p-value clears alpha.

        Statistical significance only. It says nothing about whether the effect
        is large enough to matter, which is what the decision engine judges.
        """
        return self.p_value < self.alpha

    def summary(self) -> str:
        """Human-readable rendering."""
        direction = "B > A" if self.alternative == "larger" else self.alternative
        return (
            f"control={self.control_rate:.5f} treatment={self.treatment_rate:.5f} "
            f"diff={self.absolute_difference:+.5f} z={self.z_statistic:+.4f} "
            f"p={self.p_value:.6g} ({direction})"
        )


def _validate(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
) -> None:
    for name, successes, trials in (
        ("control", control_successes, control_trials),
        ("treatment", treatment_successes, treatment_trials),
    ):
        if trials <= 0:
            raise InvalidProportionError(f"{name}_trials must be positive, got {trials}")
        if not 0 <= successes <= trials:
            raise InvalidProportionError(
                f"{name}_successes must satisfy 0 <= successes <= trials, "
                f"got {successes} with {trials} trials"
            )


def two_proportion_z_test(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TwoProportionTest:
    """Compare two conversion rates with a Wald z-test on the difference.

    Null hypothesis: the two conversion probabilities are equal.

    Args:
        control_successes: Conversions in the control arm.
        control_trials: Total users in the control arm.
        treatment_successes: Conversions in the treatment arm.
        treatment_trials: Total users in the treatment arm.
        alternative: Direction of the test. `larger` tests whether treatment
            exceeds control, which matches the usual shipping hypothesis.
        alpha: Significance level.

    Returns:
        A `TwoProportionTest` carrying both standard errors and both p-values.
    """
    if alternative not in ("two-sided", "larger", "smaller"):
        raise ValueError(
            f"Unknown alternative {alternative!r}; expected 'two-sided', 'larger' or 'smaller'"
        )
    _validate(control_successes, control_trials, treatment_successes, treatment_trials)

    control_rate = control_successes / control_trials
    treatment_rate = treatment_successes / treatment_trials
    difference = treatment_rate - control_rate

    pooled_rate = (control_successes + treatment_successes) / (control_trials + treatment_trials)
    pooled_variance = (
        pooled_rate * (1.0 - pooled_rate) * (1.0 / control_trials + 1.0 / treatment_trials)
    )
    unpooled_variance = (
        control_rate * (1.0 - control_rate) / control_trials
        + treatment_rate * (1.0 - treatment_rate) / treatment_trials
    )

    pooled_se = pooled_variance**0.5
    unpooled_se = unpooled_variance**0.5

    if pooled_se == 0.0:
        z_statistic = 0.0
        p_two = 1.0
        p_upper = 1.0
        p_lower = 1.0
    else:
        z_statistic = difference / pooled_se
        p_two = float(2.0 * stats.norm.sf(abs(z_statistic)))
        p_upper = float(stats.norm.sf(z_statistic))
        p_lower = float(stats.norm.cdf(z_statistic))

    return TwoProportionTest(
        control_successes=control_successes,
        control_trials=control_trials,
        treatment_successes=treatment_successes,
        treatment_trials=treatment_trials,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_difference=difference,
        pooled_rate=pooled_rate,
        pooled_standard_error=pooled_se,
        unpooled_standard_error=unpooled_se,
        z_statistic=z_statistic,
        p_value_one_sided=p_upper,
        p_value_lower_sided=p_lower,
        p_value_two_sided=p_two,
        alternative=alternative,
        alpha=alpha,
    )


def two_proportion_z_test_from_frame(
    frame: object,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TwoProportionTest:
    """Run the test on an experiment table, deriving counts from the arms."""
    import pandas as pd

    from experimentlens.config import BINARY_OUTCOME_COLUMN, EXPECTED_GROUPS, GROUP_COLUMN

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = [c for c in (GROUP_COLUMN, BINARY_OUTCOME_COLUMN) if c not in frame.columns]
    if missing:
        raise KeyError(f"Two-proportion test requires column(s) {missing}")

    control_rows = frame[frame[GROUP_COLUMN] == EXPECTED_GROUPS[0]]
    treatment_rows = frame[frame[GROUP_COLUMN] == EXPECTED_GROUPS[1]]

    return two_proportion_z_test(
        int(control_rows[BINARY_OUTCOME_COLUMN].sum()),
        len(control_rows),
        int(treatment_rows[BINARY_OUTCOME_COLUMN].sum()),
        len(treatment_rows),
        alternative=alternative,
        alpha=alpha,
    )
