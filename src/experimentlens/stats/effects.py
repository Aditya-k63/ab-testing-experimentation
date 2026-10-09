"""Effect sizes and confidence intervals for conversion differences.

Three quantities are reported, because they answer different questions and a
decision needs all of them:

``absolute_difference``
    Percentage-point change. Interpretable in the units the business reports,
    and the quantity whose confidence interval drives the decision.

``relative_difference``
    Percentage change over baseline. Scale-free, and the axis on which the
    minimum practically worthwhile lift is defined.

``cohens_h``
    Standardised effect size. Needed to reason about power and to compare
    against other experiments; uninterpretable on its own.

The confidence interval for the absolute difference uses the **unpooled**
standard error. Pooling assumes the null is true and produces an interval that
is too narrow, which is exactly the mistake that makes a marginal result look
decisive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from experimentlens.stats.frequentist import TwoProportionTest, two_proportion_z_test


@dataclass(frozen=True, slots=True)
class AbsoluteEffect:
    """Absolute difference in conversion rates with a Wald confidence interval."""

    difference: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    @property
    def ci_width(self) -> float:
        """Full width of the interval."""
        return self.ci_upper - self.ci_lower

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of no effect."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"{self.difference:+.5f} "
            f"[{self.ci_lower:+.5f}, {self.ci_upper:+.5f}] "
            f"at {self.confidence_level:.0%}"
        )


@dataclass(frozen=True, slots=True)
class RelativeEffect:
    """Relative lift with a log-method (Katz) confidence interval."""

    relative_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of no effect."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0

    def clears(self, threshold: float) -> bool:
        """Whether the whole interval sits above the practical threshold."""
        return self.ci_lower > threshold

    def summary(self) -> str:
        """Human-readable rendering."""
        return f"{self.relative_difference:+.5f} [{self.ci_lower:+.5f}, {self.ci_upper:+.5f}]"


@dataclass(frozen=True, slots=True)
class EffectSummary:
    """Absolute effect, relative effect and standardised size together."""

    control_rate: float
    treatment_rate: float
    absolute: AbsoluteEffect
    relative: RelativeEffect
    cohens_h: float

    @property
    def absolute_difference(self) -> float:
        """Percentage-point change in conversion rate."""
        return self.absolute.difference

    @property
    def relative_difference(self) -> float:
        """Fractional change in conversion rate relative to control."""
        return self.relative.relative_difference

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"control={self.control_rate:.5f} treatment={self.treatment_rate:.5f} | "
            f"absolute {self.absolute.summary()} | "
            f"relative {self.relative.summary()} | h={self.cohens_h:+.5f}"
        )


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h standardised difference between two proportions.

    Defined as ``2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))``. It is the arcsine
    transform of the difference, so unlike the raw difference it is not bounded
    by the scale of the underlying rates.
    """
    for name, value in (("p1", p1), ("p2", p2)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return 2.0 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))


def absolute_effect(
    control_rate: float,
    treatment_rate: float,
    control_n: int,
    treatment_n: int,
    confidence_level: float = 0.95,
) -> AbsoluteEffect:
    """Wald interval for the difference in proportions, using unpooled variance."""
    for n in (control_n, treatment_n):
        if n <= 0:
            raise ValueError(f"Sample sizes must be positive, got {control_n}, {treatment_n}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")

    difference = treatment_rate - control_rate
    variance = (
        control_rate * (1.0 - control_rate) / control_n
        + treatment_rate * (1.0 - treatment_rate) / treatment_n
    )
    standard_error = math.sqrt(variance)
    z_critical = float(stats.norm.ppf(0.5 + confidence_level / 2.0))

    return AbsoluteEffect(
        difference=difference,
        standard_error=standard_error,
        ci_lower=difference - z_critical * standard_error,
        ci_upper=difference + z_critical * standard_error,
        confidence_level=confidence_level,
    )


def relative_effect(
    control_rate: float,
    treatment_rate: float,
    control_n: int,
    treatment_n: int,
    confidence_level: float = 0.95,
) -> RelativeEffect:
    """Katz log-method interval for the relative risk.

    The interval is built on the log scale, which keeps it inside the positive
    reals and behaves sensibly away from the null. It requires strictly
    positive control and treatment rates; a zero rate makes the ratio undefined
    and is reported as such rather than patched.
    """
    if control_rate <= 0.0:
        raise ValueError(
            f"Relative lift is undefined when the control rate is {control_rate}; "
            "use the absolute difference instead."
        )
    if treatment_rate <= 0.0:
        return RelativeEffect(
            relative_difference=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
        )

    ratio = treatment_rate / control_rate
    log_ratio = math.log(ratio)
    variance = (1.0 - treatment_rate) / (treatment_n * treatment_rate) + (1.0 - control_rate) / (
        control_n * control_rate
    )
    standard_error = math.sqrt(variance)
    z_critical = float(stats.norm.ppf(0.5 + confidence_level / 2.0))

    lower = math.exp(log_ratio - z_critical * standard_error)
    upper = math.exp(log_ratio + z_critical * standard_error)

    return RelativeEffect(
        relative_difference=ratio - 1.0,
        ci_lower=lower - 1.0,
        ci_upper=upper - 1.0,
        confidence_level=confidence_level,
    )


def effect_summary(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    confidence_level: float = 0.95,
) -> EffectSummary:
    """Compute absolute effect, relative effect and Cohen's h in one call."""
    test = two_proportion_z_test(
        control_successes,
        control_trials,
        treatment_successes,
        treatment_trials,
    )
    return summarise_effects(test, confidence_level=confidence_level)


def summarise_effects(
    test: TwoProportionTest,
    confidence_level: float = 0.95,
) -> EffectSummary:
    """Derive the full effect summary from an existing test result."""
    relative = relative_effect(
        test.control_rate,
        test.treatment_rate,
        test.control_trials,
        test.treatment_trials,
        confidence_level,
    )
    return EffectSummary(
        control_rate=test.control_rate,
        treatment_rate=test.treatment_rate,
        absolute=absolute_effect(
            test.control_rate,
            test.treatment_rate,
            test.control_trials,
            test.treatment_trials,
            confidence_level,
        ),
        relative=relative,
        cohens_h=cohens_h(test.control_rate, test.treatment_rate),
    )
