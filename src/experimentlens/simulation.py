"""Simulation-based validation of the test itself.

Calling `scipy` and getting a p-value shows the code runs. It does not show the
test is *calibrated*. These simulations check the two properties that make a
test trustworthy:

**Type I error.** Under a true null, the fraction of runs that declare
significance should approach alpha. A test that rejects far more often than its
nominal rate is not conservative enough to interpret; one that rarely rejects
is wasting the experiment.

**Power.** Under a known true effect, the fraction of runs that detect it should
match the analytic power calculated in `stats.power`. A gap between the two
means at least one of them is wrong.

Every routine takes an explicit seed and uses a local generator, so results are
reproducible and independent of test execution order. No module-level RNG state
is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from experimentlens.config import SEED

if TYPE_CHECKING:
    from collections.abc import Sequence

Alternative = Literal["two-sided", "larger"]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Empirical versus nominal rejection rate under a true null."""

    nominal_alpha: float
    empirical_alpha: float
    n_simulations: int
    n_per_arm: int
    true_rate: float
    seed: int

    @property
    def absolute_error(self) -> float:
        """Distance from the nominal rate."""
        return abs(self.empirical_alpha - self.nominal_alpha)

    @property
    def is_calibrated(self) -> bool:
        """Whether empirical alpha sits inside the Monte Carlo tolerance.

        The tolerance is three standard errors of a binomial proportion, so
        occasional borderline results are expected rather than a defect.
        """
        standard_error = np.sqrt(
            max(self.nominal_alpha * (1.0 - self.nominal_alpha), 1e-12) / self.n_simulations
        )
        return bool(self.absolute_error <= 3.0 * standard_error)

    def summary(self) -> str:
        """Human-readable rendering."""
        verdict = "calibrated" if self.is_calibrated else "MISCALIBRATED"
        return (
            f"nominal={self.nominal_alpha:.3f} "
            f"empirical={self.empirical_alpha:.4f} "
            f"error={self.absolute_error:.4f} ({verdict}, "
            f"n={self.n_simulations:,})"
        )


@dataclass(frozen=True, slots=True)
class PowerSimulationResult:
    """Empirical power for a design."""

    n_per_arm: int
    baseline_rate: float
    treatment_rate: float
    empirical_power: float
    analytic_power: float
    n_simulations: int
    alpha: float
    seed: int

    @property
    def absolute_error(self) -> float:
        """Gap between empirical and analytic power."""
        return abs(self.empirical_power - self.analytic_power)

    @property
    def is_consistent(self) -> bool:
        """Whether empirical power tracks the analytic value.

        Tolerances are deliberately wider at small empirical power, where a
        binomial proportion is less well behaved.
        """
        tolerance = max(0.02, 2.5 * np.sqrt(0.25 / self.n_simulations))
        return bool(self.absolute_error <= tolerance)

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"n={self.n_per_arm}/arm empirical={self.empirical_power:.4f} "
            f"analytic={self.analytic_power:.4f} error={self.absolute_error:.4f}"
        )


def _two_proportion_pvalues(
    control_successes: np.ndarray,
    control_n: int,
    treatment_successes: np.ndarray,
    treatment_n: int,
    alternative: Alternative = "two-sided",
) -> np.ndarray:
    """Vectorised p-values for arrays of success counts.

    The tail chosen must match the alternative used by the analytic power
    calculation being compared against, otherwise the two are answering
    different questions and the disagreement looks like an error in the test
    rather than in the comparison.
    """
    control_rate = control_successes / control_n
    treatment_rate = treatment_successes / treatment_n
    pooled = (control_successes + treatment_successes) / (control_n + treatment_n)

    variance = pooled * (1.0 - pooled) * (1.0 / control_n + 1.0 / treatment_n)
    standard_error = np.sqrt(variance)

    difference = treatment_rate - control_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        z_statistic = np.where(standard_error > 0, difference / standard_error, 0.0)

    if alternative == "larger":
        return stats.norm.sf(z_statistic)
    return 2.0 * stats.norm.sf(np.abs(z_statistic))


def simulate_type_i_error(
    n_per_arm: int = 10_000,
    true_rate: float = 0.12,
    *,
    alpha: float = 0.05,
    n_simulations: int = 10_000,
    seed: int = SEED,
) -> CalibrationResult:
    """Empirical rejection rate when both arms truly share one rate.

    `true_rate` is used for both arms, so there is no real effect to detect and
    the nominal rate of false positives is the correct expectation.
    """
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")
    if not 0.0 < true_rate < 1.0:
        raise ValueError(f"true_rate must lie in (0, 1), got {true_rate}")

    rng = np.random.default_rng(seed)
    control_successes = rng.binomial(n_per_arm, true_rate, size=n_simulations)
    treatment_successes = rng.binomial(n_per_arm, true_rate, size=n_simulations)

    p_values = _two_proportion_pvalues(control_successes, n_per_arm, treatment_successes, n_per_arm)
    empirical = float(np.mean(p_values < alpha))

    return CalibrationResult(
        nominal_alpha=alpha,
        empirical_alpha=empirical,
        n_simulations=n_simulations,
        n_per_arm=n_per_arm,
        true_rate=true_rate,
        seed=seed,
    )


def simulate_power(
    n_per_arm: int = 10_000,
    baseline_rate: float = 0.12,
    treatment_rate: float = 0.13,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    n_simulations: int = 10_000,
    seed: int = SEED,
) -> PowerSimulationResult:
    """Empirical probability of detecting a known true effect.

    The analytic value from `stats.power.calculate_power` is computed alongside
    using the same `alternative`, so the two are directly comparable; their
    agreement is what validates the closed-form power calculation against actual
    sampling behaviour.
    """
    from experimentlens.stats.power import calculate_power

    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must lie in (0, 1), got {baseline_rate}")

    rng = np.random.default_rng(seed)
    control_successes = rng.binomial(n_per_arm, baseline_rate, size=n_simulations)
    treatment_successes = rng.binomial(n_per_arm, treatment_rate, size=n_simulations)

    p_values = _two_proportion_pvalues(
        control_successes, n_per_arm, treatment_successes, n_per_arm, alternative
    )
    empirical = float(np.mean(p_values < alpha))
    analytic = calculate_power(
        baseline_rate, treatment_rate, n_per_arm, alpha=alpha, alternative=alternative
    ).power

    return PowerSimulationResult(
        n_per_arm=n_per_arm,
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        empirical_power=empirical,
        analytic_power=analytic,
        n_simulations=n_simulations,
        alpha=alpha,
        seed=seed,
    )


def type_i_error_table(
    alphas: Sequence[float] | None = None,
    *,
    n_per_arm: int = 10_000,
    true_rate: float = 0.12,
    n_simulations: int = 10_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Empirical versus nominal rejection rate across a range of alphas.

    A calibrated test tracks the diagonal: every nominal rate should be
    reproduced to within Monte Carlo error.
    """
    levels = tuple(alphas) if alphas is not None else (0.01, 0.05, 0.10)
    rows: list[dict[str, object]] = []
    for index, level in enumerate(levels):
        result = simulate_type_i_error(
            n_per_arm,
            true_rate,
            alpha=level,
            n_simulations=n_simulations,
            seed=seed + index,
        )
        rows.append(
            {
                "nominal_alpha": level,
                "empirical_alpha": result.empirical_alpha,
                "absolute_error": result.absolute_error,
                "calibrated": result.is_calibrated,
                "n_simulations": result.n_simulations,
            }
        )
    return pd.DataFrame(rows)


def power_curve(
    baseline_rate: float = 0.12,
    treatment_rates: Sequence[float] | None = None,
    sample_sizes: Sequence[int] | None = None,
    *,
    alpha: float = 0.05,
    n_simulations: int = 2_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Power across a grid of sample sizes and true treatment rates.

    The grid is what a planning conversation actually needs: not one power
    number but the surface of achievable power over design choices.
    """
    rates = (
        tuple(treatment_rates)
        if treatment_rates is not None
        else tuple(baseline_rate * (1 + r) for r in (0.005, 0.01, 0.02, 0.05))
    )
    sizes = (
        tuple(sample_sizes)
        if sample_sizes is not None
        else (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)
    )

    rows: list[dict[str, object]] = []
    for size_index, size in enumerate(sizes):
        for rate_index, rate in enumerate(rates):
            result = simulate_power(
                int(size),
                baseline_rate,
                float(rate),
                alpha=alpha,
                n_simulations=n_simulations,
                seed=seed + 100 * size_index + rate_index,
            )
            rows.append(
                {
                    "n_per_arm": int(size),
                    "treatment_rate": float(rate),
                    "relative_effect": (float(rate) - baseline_rate) / baseline_rate,
                    "empirical_power": result.empirical_power,
                    "analytic_power": result.analytic_power,
                }
            )
    return pd.DataFrame(rows)


def pvalue_null_distribution(
    n_per_arm: int = 10_000,
    true_rate: float = 0.12,
    *,
    n_simulations: int = 10_000,
    seed: int = SEED,
) -> np.ndarray:
    """P-values simulated under the null, for inspecting uniformity.

    Under a calibrated test these should be approximately uniform on [0, 1].
    Comparing the observed distribution against that reference is a stronger
    check than comparing a single summary statistic.
    """
    rng = np.random.default_rng(seed)
    control_successes = rng.binomial(n_per_arm, true_rate, size=n_simulations)
    treatment_successes = rng.binomial(n_per_arm, true_rate, size=n_simulations)
    return _two_proportion_pvalues(control_successes, n_per_arm, treatment_successes, n_per_arm)
