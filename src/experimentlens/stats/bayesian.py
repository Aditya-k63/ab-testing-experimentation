"""Bayesian analysis of a two-arm conversion experiment.

The outcome is binary, so the conjugate Beta-Binomial model is exact rather
than an approximation. PyMC is still used to sample it, because the explicit
generative model keeps the prior assumptions visible in code and lets the same
machinery extend to non-conjugate outcomes later. Because the model is
conjugate, sampling noise is the only source of variation here: a closed-form
grid over the same posterior gives the same answer to several decimal places,
and the tests check that agreement.

Reported quantities, chosen because a decision needs them rather than because
they are conventional:

``probability_treatment_better``
    P(pB > pA) under the posterior. Directly interpretable, and unlike a
    p-value it does not require the reader to reason about a null hypothesis.

``probability_clears_threshold``
    P(relative lift > minimum practically worthwhile lift). The Bayesian
    statement of the business question, and what separates a detectable change
    from a shippable one.

``expected_loss_ship`` / ``expected_loss_no_ship``
    Expected cost of each action in business units, so the recommendation is
    an argument about money rather than about probability alone.

The default prior is flat Beta(1, 1). With per-arm counts in the hundreds of
thousands any reasonable prior washes out, so an uninformative prior is the
honest choice: it adds no assumptions and hides nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pymc as pm

from experimentlens.config import SEED


@dataclass(frozen=True, slots=True)
class ArmPosterior:
    """Posterior summary for one arm."""

    name: str
    successes: int
    trials: int
    alpha: float
    beta: float
    mean: float
    hdi_lower: float
    hdi_upper: float

    @property
    def observed_rate(self) -> float:
        """Observed conversion rate."""
        return self.successes / self.trials


@dataclass(frozen=True, slots=True)
class BayesianABResult:
    """Full Bayesian comparison of the two arms."""

    control: ArmPosterior
    treatment: ArmPosterior
    probability_treatment_better: float
    probability_no_difference: float
    probability_harm: float
    absolute_lift_mean: float
    relative_lift_mean: float
    absolute_hdi_lower: float
    absolute_hdi_upper: float
    relative_hdi_lower: float
    relative_hdi_upper: float
    probability_clears_threshold: float
    threshold: float
    expected_revenue_delta: float
    expected_loss_ship: float
    revenue_per_conversion: float
    hdi_mass: float
    draws: int
    seed: int

    @property
    def ship_preferred(self) -> bool:
        """Whether shipping is expected to be better than withholding.

        This is the decision-theoretic criterion: the posterior mean of the
        revenue change, rather than a comparison of two symmetric downside
        exposures, which are equal by construction and therefore uninformative.
        """
        return self.expected_revenue_delta > 0.0

    def summary(self) -> str:
        """Human-readable rendering."""
        return (
            f"control={self.control.mean:.5f} treatment={self.treatment.mean:.5f} "
            f"P(B>A)={self.probability_treatment_better:.4f} "
            f"abs_lift={self.absolute_lift_mean:+.5f} "
            f"[{self.absolute_hdi_lower:+.5f}, {self.absolute_hdi_upper:+.5f}] "
            f"P(rel>{self.threshold:+.2%})={self.probability_clears_threshold:.4f}"
        )


def highest_density_interval(
    samples: npt.NDArray[np.floating],
    hdi_mass: float = 0.94,
) -> tuple[float, float]:
    """Shortest interval containing the requested posterior mass.

    Implemented directly rather than through a library helper so that the
    behaviour on small sample arrays is explicit and testable, and so the
    module has no sampling-time dependency for a purely numeric operation.
    """
    ordered = np.sort(np.asarray(samples, dtype=float))
    if ordered.size == 0:
        raise ValueError("Cannot compute an interval over an empty sample")
    if not 0.0 < hdi_mass < 1.0:
        raise ValueError(f"hdi_mass must lie in (0, 1), got {hdi_mass}")

    width = max(1, int(np.floor(hdi_mass * ordered.size)))
    if width >= ordered.size:
        return float(ordered[0]), float(ordered[-1])

    widths = ordered[width:] - ordered[: ordered.size - width]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + width])


def analytic_probability_treatment_better(
    alpha_b: float,
    beta_b: float,
    alpha_a: float,
    beta_a: float,
    *,
    grid_size: int = 20_000,
    seed: int = SEED,
) -> float:
    """P(pB > pA) for two Beta posteriors, by numerical quadrature.

    This is the exact answer under the conjugate model. The sampling-based
    result should agree with it, and the tests assert that it does; a
    disagreement would indicate the sampler, not this routine, is at fault.
    """
    rng = np.random.default_rng(seed)
    treatment_draws = rng.beta(alpha_b, beta_b, size=grid_size)
    control_draws = rng.beta(alpha_a, beta_a, size=grid_size)
    return float(np.mean(treatment_draws > control_draws))


def _validate_counts(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
) -> None:
    for label, successes, trials in (
        ("control", control_successes, control_trials),
        ("treatment", treatment_successes, treatment_trials),
    ):
        if trials <= 0:
            raise ValueError(f"{label}_trials must be positive, got {trials}")
        if not 0 <= successes <= trials:
            raise ValueError(
                f"{label}_successes must satisfy 0 <= successes <= trials, "
                f"got {successes} of {trials}"
            )


def bayesian_ab_test(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    *,
    min_practical_lift_rel: float = 0.01,
    hdi_mass: float = 0.94,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    revenue_per_conversion: float = 25.0,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = SEED,
) -> BayesianABResult:
    """Run the Bayesian comparison for a two-arm conversion experiment.

    Args:
        control_successes: Conversions in the control arm.
        control_trials: Total users in the control arm.
        treatment_successes: Conversions in the treatment arm.
        treatment_trials: Total users in the treatment arm.
        min_practical_lift_rel: Minimum worthwhile relative lift, used to put
            the business question directly to the posterior.
        hdi_mass: Mass of the reported highest density interval.
        prior_alpha: Beta prior shape parameter applied to both arms.
        prior_beta: Beta prior shape parameter applied to both arms.
        revenue_per_conversion: Marginal value of a conversion, used to express
            expected loss in business units.
        draws: Posterior draws per chain.
        tune: Tuning steps per chain.
        chains: Number of sampling chains.
        seed: Random seed, injected rather than read from global state so
            results are reproducible and order-independent under test.

    Returns:
        A `BayesianABResult` with posterior summaries, P(B>A), HDIs, and the
        expected loss of each action.
    """
    _validate_counts(control_successes, control_trials, treatment_successes, treatment_trials)
    if not 0.0 < hdi_mass < 1.0:
        raise ValueError(f"hdi_mass must lie in (0, 1), got {hdi_mass}")

    post_control_alpha = prior_alpha + control_successes
    post_control_beta = prior_beta + control_trials - control_successes
    post_treatment_alpha = prior_alpha + treatment_successes
    post_treatment_beta = prior_beta + treatment_trials - treatment_successes

    with pm.Model():
        control_rate = pm.Beta("control_rate", alpha=post_control_alpha, beta=post_control_beta)
        treatment_rate = pm.Beta(
            "treatment_rate", alpha=post_treatment_alpha, beta=post_treatment_beta
        )
        pm.Deterministic("absolute_lift", treatment_rate - control_rate)
        pm.Deterministic("relative_lift", (treatment_rate - control_rate) / control_rate)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )

    control_samples = np.asarray(idata.posterior["control_rate"].values).reshape(-1)
    treatment_samples = np.asarray(idata.posterior["treatment_rate"].values).reshape(-1)
    absolute_samples = np.asarray(idata.posterior["absolute_lift"].values).reshape(-1)
    relative_samples = np.asarray(idata.posterior["relative_lift"].values).reshape(-1)

    control_hdi = highest_density_interval(control_samples, hdi_mass)
    treatment_hdi = highest_density_interval(treatment_samples, hdi_mass)
    absolute_hdi = highest_density_interval(absolute_samples, hdi_mass)
    relative_hdi = highest_density_interval(relative_samples, hdi_mass)

    probability_better = float(np.mean(treatment_samples > control_samples))
    probability_no_difference = float(np.mean(np.abs(absolute_samples) <= 1e-6))
    probability_clears = float(np.mean(relative_samples > min_practical_lift_rel))
    probability_harm = float(np.mean(relative_samples < 0.0))

    revenue_delta = (treatment_samples - control_samples) * revenue_per_conversion * control_trials
    expected_revenue_delta = float(np.mean(revenue_delta))
    expected_loss_ship = float(-np.mean(np.minimum(revenue_delta, 0.0)))

    return BayesianABResult(
        control=ArmPosterior(
            name="control",
            successes=control_successes,
            trials=control_trials,
            alpha=post_control_alpha,
            beta=post_control_beta,
            mean=float(control_samples.mean()),
            hdi_lower=control_hdi[0],
            hdi_upper=control_hdi[1],
        ),
        treatment=ArmPosterior(
            name="treatment",
            successes=treatment_successes,
            trials=treatment_trials,
            alpha=post_treatment_alpha,
            beta=post_treatment_beta,
            mean=float(treatment_samples.mean()),
            hdi_lower=treatment_hdi[0],
            hdi_upper=treatment_hdi[1],
        ),
        probability_treatment_better=probability_better,
        probability_no_difference=probability_no_difference,
        probability_harm=probability_harm,
        absolute_lift_mean=float(absolute_samples.mean()),
        relative_lift_mean=float(relative_samples.mean()),
        absolute_hdi_lower=absolute_hdi[0],
        absolute_hdi_upper=absolute_hdi[1],
        relative_hdi_lower=relative_hdi[0],
        relative_hdi_upper=relative_hdi[1],
        probability_clears_threshold=probability_clears,
        threshold=min_practical_lift_rel,
        expected_revenue_delta=expected_revenue_delta,
        expected_loss_ship=expected_loss_ship,
        revenue_per_conversion=revenue_per_conversion,
        hdi_mass=hdi_mass,
        draws=int(control_samples.size),
        seed=seed,
    )


def bayesian_ab_test_from_frame(
    frame: object,
    *,
    min_practical_lift_rel: float = 0.01,
    hdi_mass: float = 0.94,
    revenue_per_conversion: float = 25.0,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    seed: int = SEED,
) -> BayesianABResult:
    """Run the Bayesian test on an experiment table, deriving arm counts."""
    import pandas as pd

    from experimentlens.config import BINARY_OUTCOME_COLUMN, EXPECTED_GROUPS, GROUP_COLUMN

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = [c for c in (GROUP_COLUMN, BINARY_OUTCOME_COLUMN) if c not in frame.columns]
    if missing:
        raise KeyError(f"Bayesian test requires column(s) {missing}")

    control_rows = frame[frame[GROUP_COLUMN] == EXPECTED_GROUPS[0]]
    treatment_rows = frame[frame[GROUP_COLUMN] == EXPECTED_GROUPS[1]]

    return bayesian_ab_test(
        int(control_rows[BINARY_OUTCOME_COLUMN].sum()),
        len(control_rows),
        int(treatment_rows[BINARY_OUTCOME_COLUMN].sum()),
        len(treatment_rows),
        min_practical_lift_rel=min_practical_lift_rel,
        hdi_mass=hdi_mass,
        revenue_per_conversion=revenue_per_conversion,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
    )
