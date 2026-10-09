"""Experiment design: power, minimum detectable effect, and sample size.

These are planning tools, run *before* an experiment is read. They answer the
question the p-value cannot: given a baseline rate and a fixed sample size,
how large an effect would this experiment have had a fair chance of detecting?

Note the tail convention: these functions default to a **one-sided** test,
matching the pre-registered hypothesis that treatment beats control. See
`DEFAULT_ALTERNATIVE_NOTE` - the p-value reported elsewhere is two-sided, and
the two are not directly comparable.

The underlying machinery is Cohen's h, the arcsine-transformed difference
between two proportions. It is the same statistic `statsmodels` uses for
proportion comparisons, so results here are directly comparable with standard
practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

Alternative = Literal["two-sided", "larger"]

DEFAULT_ALTERNATIVE_NOTE = (
    "Power and MDE default to a one-sided ('larger') test, matching the "
    "pre-registered hypothesis that treatment beats control. The reported "
    "p-value is two-sided, so the two figures are not directly comparable: a "
    "one-sided test is the more powerful design for a directional hypothesis. "
    "Pass alternative='two-sided' to power the test actually being reported."
)


@dataclass(frozen=True, slots=True)
class PowerResult:
    """Achieved power for a proposed design."""

    baseline_rate: float
    treatment_rate: float
    n_per_arm: int
    effect_size_h: float
    power: float
    alpha: float
    target_power: float
    alternative: Alternative

    @property
    def absolute_effect(self) -> float:
        """Percentage-point effect the design is powered against."""
        return self.treatment_rate - self.baseline_rate

    @property
    def relative_effect(self) -> float:
        """Fractional effect the design is powered against."""
        if self.baseline_rate == 0:
            return float("nan")
        return (self.treatment_rate - self.baseline_rate) / self.baseline_rate

    @property
    def meets_target(self) -> bool:
        """Whether achieved power reaches the declared target."""
        return self.power >= self.target_power

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"n={self.n_per_arm}/arm baseline={self.baseline_rate:.4f} "
            f"treatment={self.treatment_rate:.4f} h={self.effect_size_h:+.5f} "
            f"power={self.power:.4f} (target {self.target_power:.2f})"
        )


@dataclass(frozen=True, slots=True)
class SampleSizeResult:
    """Sample size required to reach a target power."""

    baseline_rate: float
    treatment_rate: float
    required_n_per_arm: int
    total_n: int
    effect_size_h: float
    power: float
    alpha: float
    alternative: Alternative

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"required n={self.required_n_per_arm}/arm (total {self.total_n}) "
            f"for baseline={self.baseline_rate:.4f} "
            f"treatment={self.treatment_rate:.4f} at power={self.power:.2f}"
        )


@dataclass(frozen=True, slots=True)
class MdeResult:
    """Smallest effect the design could detect at a target power."""

    baseline_rate: float
    detectable_treatment_rate: float
    mde_absolute: float
    mde_relative: float
    n_per_arm: int
    power: float
    alpha: float
    alternative: Alternative

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"MDE at n={self.n_per_arm}/arm: {self.mde_absolute:+.5f} absolute "
            f"({self.mde_relative:+.4%} relative) for baseline "
            f"{self.baseline_rate:.4f}"
        )


def _validate_baseline(baseline_rate: float) -> None:
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must lie strictly between 0 and 1, got {baseline_rate}")


def effect_size(baseline_rate: float, treatment_rate: float) -> float:
    """Cohen's h for two conversion rates.

    Thin wrapper over `statsmodels.proportion_effectsize` so the package has a
    single definition of the effect size used for power.
    """
    _validate_baseline(baseline_rate)
    if not 0.0 <= treatment_rate <= 1.0:
        raise ValueError(f"treatment_rate must lie in [0, 1], got {treatment_rate}")
    return float(abs(proportion_effectsize(baseline_rate, treatment_rate)))


def calculate_power(
    baseline_rate: float,
    treatment_rate: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    alternative: Alternative = "larger",
) -> PowerResult:
    """Power achieved by a design with a fixed sample size."""
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")

    h = effect_size(baseline_rate, treatment_rate)
    power = NormalIndPower().power(
        effect_size=h,
        nobs1=n_per_arm,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
    )
    return PowerResult(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        n_per_arm=n_per_arm,
        effect_size_h=h,
        power=float(power),
        alpha=alpha,
        target_power=target_power,
        alternative=alternative,
    )


def required_sample_size(
    baseline_rate: float,
    treatment_rate: float,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "larger",
) -> SampleSizeResult:
    """Sample size per arm needed to detect a given effect."""
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie strictly between 0 and 1, got {power}")

    h = effect_size(baseline_rate, treatment_rate)
    if h == 0.0:
        raise ValueError(
            "baseline_rate and treatment_rate are equal; no finite sample size "
            "can detect an effect of zero."
        )

    n = NormalIndPower().solve_power(
        effect_size=h,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
    )
    n_rounded = math.ceil(n)
    return SampleSizeResult(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        required_n_per_arm=n_rounded,
        total_n=n_rounded * 2,
        effect_size_h=h,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_arm: int,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "larger",
) -> MdeResult:
    """Smallest effect the design can detect at the given sample size."""
    _validate_baseline(baseline_rate)
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie strictly between 0 and 1, got {power}")

    h = NormalIndPower().solve_power(
        nobs1=n_per_arm,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
    )
    h = float(abs(h))

    baseline_angle = math.asin(math.sqrt(baseline_rate))
    detectable = math.sin(baseline_angle + h / 2.0) ** 2

    return MdeResult(
        baseline_rate=baseline_rate,
        detectable_treatment_rate=detectable,
        mde_absolute=detectable - baseline_rate,
        mde_relative=(detectable - baseline_rate) / baseline_rate,
        n_per_arm=n_per_arm,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def post_hoc_power(
    baseline_rate: float,
    observed_treatment_rate: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    alternative: Alternative = "larger",
) -> PowerResult:
    """Power of the design that was actually run.

    Reported for completeness and for comparison with the planned design. It is
    not a substitute for pre-registration: observed-effect power computed after
    seeing the data is itself a function of the estimate, and is therefore
    optimistic.
    """
    return calculate_power(
        baseline_rate,
        observed_treatment_rate,
        n_per_arm,
        alpha=alpha,
        target_power=target_power,
        alternative=alternative,
    )


def design_summary(
    baseline_rate: float,
    n_per_arm: int,
    observed_treatment_rate: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    alternative: Alternative = "larger",
) -> dict[str, float]:
    """Combined design report: achieved power, MDE, and sample requirement."""
    achieved = post_hoc_power(
        baseline_rate,
        observed_treatment_rate,
        n_per_arm,
        alpha=alpha,
        target_power=target_power,
        alternative=alternative,
    )
    mde = minimum_detectable_effect(
        baseline_rate,
        n_per_arm,
        power=target_power,
        alpha=alpha,
        alternative=alternative,
    )
    required = required_sample_size(
        baseline_rate,
        observed_treatment_rate,
        power=target_power,
        alpha=alpha,
        alternative=alternative,
    )
    return {
        "achieved_power": achieved.power,
        "target_power": target_power,
        "observed_effect_h": achieved.effect_size_h,
        "mde_absolute": mde.mde_absolute,
        "mde_relative": mde.mde_relative,
        "observed_n_per_arm": float(n_per_arm),
        "required_n_per_arm": float(required.required_n_per_arm),
        "shortfall_per_arm": float(max(0, required.required_n_per_arm - n_per_arm)),
    }
