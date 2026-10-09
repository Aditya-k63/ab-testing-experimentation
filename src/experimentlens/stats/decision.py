"""Business decision engine.

The single rule this module exists to enforce: **a p-value below 0.05 is never
sufficient to ship.** Significance says the data would be unlikely under a null
of no effect. It says nothing about whether the effect is large enough to pay
for the change, and it says nothing about whether the experiment was measuring
the right thing. Both of those are what actually determine the decision.

The engine is a pure function over three inputs - observed quantities, the
validity verdict, and an injected `Thresholds` object - so every branch is
unit-testable and the dashboard never decides anything itself.

Three outcomes, not two:

``SHIP``
    Validity checks pass and the evidence supports an improvement worth having.

``DONT_SHIP``
    The evidence supports no improvement, or a harmful one.

``INCONCLUSIVE``
    Uncertainty remains too high to defend either call. This is a real answer,
    not a failure, and it comes with the sample size that would resolve it.

On statistical versus practical significance: the two are separated by
construction. `min_practical_lift_rel` is a business quantity, alpha is a
statistical one, and no configuration of the thresholds lets a p-value stand in
for the other.

On the asymmetry between the two gates: shipping requires **every** criterion
to pass, while refusing requires **any one**. That is deliberate and it is
worth stating plainly rather than leaving to be inferred from the code.

The asymmetry is conservative in the safe direction. Refusing a change is
reversible - the experiment can be rerun - whereas shipping a regression is
not. It also matches the asymmetry of the evidence: SHIP is a claim that a
change is *worthwhile*, which ought to require more than any single optimistic
reading supports, while DON'T_SHIP is a refusal to act, which only needs one
sound reason.

Each refusing criterion stands on its own merits without corroboration:

* the credible interval lies below the practical threshold - nothing worth
  shipping exists anywhere in the credible range;
* expected revenue is negative - the action has negative expected value, which
  is sufficient on its own;
* the posterior probability of an improvement is at or below the declared
  threshold - the change is more likely to be neutral-to-worse than better.

The alternative policy - requiring two refusing criteria before refusing - was
considered and rejected. It makes the engine *more* willing to ship, which is
the wrong direction for a change nobody is obliged to make, and altering the
rules to loosen a live verdict would be exactly the wrong reason to touch them.

On power: observed-effect power is *not* used to gate the outcome. It is a
function of the estimate it is meant to inform, so letting it decide would be
circular. What is reported instead is whether the design could have resolved
the business question at all - if the minimum detectable effect exceeds the
practical threshold, the experiment was never capable of settling it, and that
is stated as a design finding.

On correlated evidence: the criteria below are *not* independent
confirmations. Every one of them is a different summary of the same underlying
2x2 table of conversion counts. They are grouped into frequentist, Bayesian and
business views precisely so they are not mistaken for three separate sources of
evidence, and the rendered output says so. Two summaries agreeing tells you the
result is robust to how you summarise it; it does not multiply the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from experimentlens.config import Thresholds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from experimentlens.data.clean import CleanResult
    from experimentlens.stats.validity import ValidityReport

DecisionOutcome = Literal["SHIP", "DONT_SHIP", "INCONCLUSIVE"]

EvidenceView = Literal["frequentist", "bayesian", "business", "validity"]

VIEW_TITLES: dict[str, str] = {
    "validity": "Data validity",
    "frequentist": "Frequentist view",
    "bayesian": "Bayesian view",
    "business": "Business view",
}

VIEW_ORDER: tuple[EvidenceView, ...] = (
    "validity",
    "frequentist",
    "bayesian",
    "business",
)

CORRELATED_EVIDENCE_NOTE = (
    "These are three summaries of the same 2x2 table of conversion counts, not "
    "three independent experiments. Agreement between them shows the conclusion "
    "is robust to how the evidence is summarised; it does not multiply the "
    "evidence. A single dataset cannot supply independent corroboration."
)

_OUTCOME_LABELS: dict[str, str] = {
    "SHIP": "SHIP",
    "DONT_SHIP": "DON'T SHIP",
    "INCONCLUSIVE": "INCONCLUSIVE",
}


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """One named criterion evaluated by the engine.

    `view` groups the checks by the lens that produced them. Every check is a
    summary of the same experiment, so the grouping exists to stop them reading
    as independent confirmations rather than to imply they are.
    """

    name: str
    passed: bool
    detail: str
    view: EvidenceView = "business"

    def to_row(self) -> dict[str, object]:
        """Render as a table row."""
        return {
            "view": VIEW_TITLES.get(self.view, self.view),
            "criterion": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """Observed quantities the decision depends on.

    Deliberately a plain bundle of scalars rather than a wrapper around the
    analysis objects, so decision tests need no sampler and no dataframe.
    """

    control_rate: float
    treatment_rate: float
    absolute_difference: float
    relative_difference: float
    relative_ci_lower: float
    relative_ci_upper: float
    p_value_two_sided: float
    p_value_one_sided: float
    probability_treatment_better: float
    probability_clears_threshold: float
    relative_hdi_lower: float
    relative_hdi_upper: float
    expected_revenue_delta: float
    achieved_power: float = 0.0
    mde_relative: float = 0.0
    required_n_per_arm: int = 0
    observed_n_per_arm: int = 0
    control_posterior_alpha: float = 1.0
    control_posterior_beta: float = 1.0
    treatment_posterior_alpha: float = 1.0
    treatment_posterior_beta: float = 1.0


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """A decision, its justification, and the evidence behind it."""

    outcome: DecisionOutcome
    headline: str
    reasoning: tuple[str, ...]
    evidence: tuple[EvidenceCheck, ...]
    recommended_action: str
    blocking_reasons: tuple[str, ...] = ()
    design_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Human-readable outcome label."""
        return _OUTCOME_LABELS[self.outcome]

    @property
    def passed_checks(self) -> tuple[EvidenceCheck, ...]:
        """Criteria that were satisfied."""
        return tuple(c for c in self.evidence if c.passed)

    @property
    def failed_checks(self) -> tuple[EvidenceCheck, ...]:
        """Criteria that were not satisfied."""
        return tuple(c for c in self.evidence if not c.passed)

    def to_frame(self) -> object:
        """Evidence checklist as a dataframe, for the dashboard."""
        import pandas as pd

        return pd.DataFrame([c.to_row() for c in self.evidence])

    def evidence_by_view(self) -> dict[str, tuple[EvidenceCheck, ...]]:
        """Group the evidence by lens, in presentation order."""
        grouped: dict[str, list[EvidenceCheck]] = {view: [] for view in VIEW_ORDER}
        for check in self.evidence:
            grouped.setdefault(check.view, []).append(check)
        return {view: tuple(checks) for view, checks in grouped.items() if checks}

    def summary(self) -> str:
        """Multi-line rendering, grouped by lens."""
        lines = [f"DECISION: {self.label}", f"  {self.headline}"]
        lines.extend(f"  - {reason}" for reason in self.reasoning)
        for view, checks in self.evidence_by_view().items():
            lines.append(f"  {VIEW_TITLES.get(view, view)}:")
            lines.extend(f"    [{'x' if c.passed else ' '}] {c.name}: {c.detail}" for c in checks)
        lines.append(f"  Note: {CORRELATED_EVIDENCE_NOTE}")
        if self.design_notes:
            lines.append("  Design:")
            lines.extend(f"    - {note}" for note in self.design_notes)
        if self.notes:
            lines.append("  Refusal triggered by:")
            lines.extend(f"    - {note}" for note in self.notes)
        lines.append(f"  Next: {self.recommended_action}")
        return "\n".join(lines)


def _design_notes(inputs: DecisionInputs, thresholds: Thresholds) -> tuple[EvidenceCheck, ...]:
    """Assess the experiment design against the business question.

    Reported, never enforced. A design can be inadequate and the evidence can
    still be clear; conflating the two would let a retrospective power number
    override the data.
    """
    notes: list[EvidenceCheck] = []

    if inputs.mde_relative > thresholds.min_practical_lift_rel:
        notes.append(
            EvidenceCheck(
                name="design could resolve the business question",
                passed=False,
                detail=(
                    f"minimum detectable effect is {inputs.mde_relative:+.2%} relative, "
                    f"above the {thresholds.min_practical_lift_rel:+.2%} practical "
                    "threshold; this design was not able to settle the question in "
                    "advance"
                ),
            )
        )
    else:
        notes.append(
            EvidenceCheck(
                name="design could resolve the business question",
                passed=True,
                detail=(
                    f"minimum detectable effect {inputs.mde_relative:+.2%} is at or below "
                    f"the {thresholds.min_practical_lift_rel:+.2%} threshold"
                ),
            )
        )

    shortfall = inputs.required_n_per_arm - inputs.observed_n_per_arm
    if shortfall > 0:
        notes.append(
            EvidenceCheck(
                name="sample sufficient for the observed effect",
                passed=False,
                detail=(
                    f"{inputs.observed_n_per_arm:,} per arm observed; "
                    f"{inputs.required_n_per_arm:,} would be needed for 80% power "
                    f"against an effect of this size (shortfall {shortfall:,})"
                ),
            )
        )

    return tuple(notes)


def decide(
    inputs: DecisionInputs,
    validity: ValidityReport,
    thresholds: Thresholds | None = None,
) -> DecisionResult:
    """Turn evidence into a shipping recommendation.

    The order of the gates matters. Validity is checked first because no amount
    of statistical evidence rescues an experiment whose randomisation is in
    doubt. Practical significance is checked before statistical significance
    because a significant result on a trivial effect is not a reason to ship.
    """
    th = thresholds if thresholds is not None else Thresholds()

    if not validity.is_valid:
        reasons = validity.blocking_reasons
        return DecisionResult(
            outcome="INCONCLUSIVE",
            headline=(
                "The experiment cannot be trusted, so no shipping recommendation is defensible."
            ),
            reasoning=tuple(f"Validity check failed: {reason}" for reason in reasons),
            evidence=(),
            recommended_action=(
                "Resolve the validity failure and re-run before considering the "
                "result. Do not ship on the strength of a broken experiment."
            ),
            blocking_reasons=reasons,
        )

    evidence: list[EvidenceCheck] = [
        EvidenceCheck(
            name="validity checks",
            passed=True,
            detail=(f"SRM passed (p={validity.srm.p_value:.3f}); sample sizes adequate"),
            view="validity",
        ),
        EvidenceCheck(
            name="effect clears the practical threshold",
            passed=inputs.relative_difference >= th.min_practical_lift_rel,
            detail=(
                f"relative lift {inputs.relative_difference:+.2%} vs minimum worthwhile "
                f"{th.min_practical_lift_rel:+.2%}"
            ),
            view="business",
        ),
        EvidenceCheck(
            name="posterior supports clearing the threshold",
            passed=inputs.probability_clears_threshold >= th.min_posterior_probability,
            detail=(
                f"P(relative lift > {th.min_practical_lift_rel:+.2%}) = "
                f"{inputs.probability_clears_threshold:.2%}, threshold "
                f"{th.min_posterior_probability:.0%}"
            ),
            view="bayesian",
        ),
        EvidenceCheck(
            name="credible interval excludes harm",
            passed=inputs.relative_hdi_lower > 0.0,
            detail=(
                f"{th.hdi_mass:.0%} HDI on relative lift is "
                f"[{inputs.relative_hdi_lower:+.2%}, {inputs.relative_hdi_upper:+.2%}]"
            ),
            view="bayesian",
        ),
        EvidenceCheck(
            name="credible interval excludes the worthwhile range",
            passed=inputs.relative_hdi_upper < th.min_practical_lift_rel,
            detail=(
                f"even the optimistic end ({inputs.relative_hdi_upper:+.2%}) is below the "
                f"{th.min_practical_lift_rel:+.2%} threshold"
            ),
            view="bayesian",
        ),
        EvidenceCheck(
            name="expected revenue effect is positive",
            passed=inputs.expected_revenue_delta > 0.0,
            detail=f"posterior mean revenue change {inputs.expected_revenue_delta:+,.0f}",
            view="business",
        ),
        EvidenceCheck(
            name="expected revenue effect is negative",
            passed=inputs.expected_revenue_delta < 0.0,
            detail=f"posterior mean revenue change {inputs.expected_revenue_delta:+,.0f}",
            view="business",
        ),
        EvidenceCheck(
            name="posterior favours control",
            passed=inputs.probability_treatment_better <= th.dont_ship_posterior_threshold,
            detail=(
                f"P(treatment > control) = {inputs.probability_treatment_better:.2%}, "
                f"at or below the {th.dont_ship_posterior_threshold:.0%} threshold for "
                "treating the change as no better than control"
            ),
            view="bayesian",
        ),
        EvidenceCheck(
            name="frequentist interval excludes zero",
            passed=inputs.relative_ci_lower > 0.0,
            detail=(
                f"95% CI on relative lift is "
                f"[{inputs.relative_ci_lower:+.2%}, {inputs.relative_ci_upper:+.2%}]"
            ),
            view="frequentist",
        ),
    ]

    by_name = {c.name: c for c in evidence}
    ship_checks = [
        by_name["effect clears the practical threshold"],
        by_name["posterior supports clearing the threshold"],
        by_name["credible interval excludes harm"],
        by_name["expected revenue effect is positive"],
    ]
    dont_ship_checks = [
        by_name["credible interval excludes the worthwhile range"],
        by_name["expected revenue effect is negative"],
        by_name["posterior favours control"],
    ]

    design = _design_notes(inputs, th)
    policy_note = (
        f"Refusal requires any one of {len(dont_ship_checks)} criteria; shipping "
        f"requires all {len(ship_checks)}. The asymmetry is deliberate - refusing is "
        "reversible, shipping a regression is not."
    )

    if all(c.passed for c in ship_checks):
        return DecisionResult(
            outcome="SHIP",
            headline=(
                f"Treatment improves conversion by {inputs.relative_difference:+.2%} "
                f"relative, and the evidence supports it being worthwhile."
            ),
            reasoning=(
                "Every shipping criterion is satisfied: the observed lift exceeds the "
                "minimum worthwhile lift, the posterior puts a high probability on "
                "clearing that threshold, the credible interval excludes harm, and the "
                "expected revenue effect is positive.",
                f"Two-sided p = {inputs.p_value_two_sided:.4f} is reported for "
                "reference but did not drive this decision; power figures elsewhere in "
                "this report are one-sided, matching the pre-registered hypothesis "
                "that treatment beats control.",
            ),
            evidence=tuple(evidence),
            recommended_action=(
                "Ship the treatment, and record the effect size and interval so the "
                "result can be compared against future experiments."
            ),
            design_notes=tuple(c.detail for c in design),
            notes=(policy_note,),
        )

    triggered = [c for c in dont_ship_checks if c.passed]
    if triggered:
        return DecisionResult(
            outcome="DONT_SHIP",
            headline=(
                f"Evidence does not support shipping: the best case is below the "
                f"{th.min_practical_lift_rel:+.2%} minimum worthwhile lift."
            ),
            reasoning=(
                (
                    f"The credible interval reaches no higher than "
                    f"{inputs.relative_hdi_upper:+.2%} relative, so even the optimistic "
                    f"reading is worth less than the {th.min_practical_lift_rel:+.2%} "
                    "needed to justify the change."
                    if by_name["credible interval excludes the worthwhile range"].passed
                    else (
                        "The posterior mean revenue effect is negative, so shipping has "
                        "negative expected value."
                        if by_name["expected revenue effect is negative"].passed
                        else "The posterior gives only a "
                        f"{inputs.probability_treatment_better:.1%} chance that treatment "
                        "converts better than control."
                    )
                ),
                f"Refused on {len(triggered)} of {len(dont_ship_checks)} possible "
                f"grounds. Note the distinction: a non-significant p-value on its own "
                "would only mean 'not proven'; the recommendation rests on the "
                "credible interval excluding the worthwhile range, which is a stronger "
                "statement.",
            ),
            evidence=tuple(evidence),
            recommended_action=(
                "Do not ship. If a lift of this size matters commercially, the honest "
                "next step is a redesign sized against the practical threshold rather "
                "than a re-run of this one."
            ),
            design_notes=tuple(c.detail for c in design),
            notes=(
                policy_note,
                *(f"Triggered criterion: {c.name} - {c.detail}" for c in triggered),
            ),
        )

    failing = [c for c in ship_checks if not c.passed]
    return DecisionResult(
        outcome="INCONCLUSIVE",
        headline=(
            "The evidence is too uncertain to justify either shipping or rejecting the change."
        ),
        reasoning=tuple(f"Unresolved: {c.name} - {c.detail}" for c in failing),
        evidence=tuple(evidence),
        recommended_action=(
            f"Run the experiment longer. Approximately {inputs.required_n_per_arm:,} "
            f"users per arm are needed for 80% power against an effect of this size; "
            f"{inputs.observed_n_per_arm:,} were observed."
        )
        if inputs.required_n_per_arm > inputs.observed_n_per_arm
        else "Collect more data before revisiting this decision.",
        design_notes=tuple(c.detail for c in design),
    )


def build_inputs(
    *,
    control_rate: float,
    treatment_rate: float,
    absolute_difference: float,
    relative_difference: float,
    relative_ci_lower: float,
    relative_ci_upper: float,
    p_value_two_sided: float,
    p_value_one_sided: float,
    probability_treatment_better: float,
    probability_clears_threshold: float,
    relative_hdi_lower: float,
    relative_hdi_upper: float,
    expected_revenue_delta: float,
    achieved_power: float = 0.0,
    mde_relative: float = 0.0,
    required_n_per_arm: int = 0,
    observed_n_per_arm: int = 0,
    control_posterior_alpha: float = 1.0,
    control_posterior_beta: float = 1.0,
    treatment_posterior_alpha: float = 1.0,
    treatment_posterior_beta: float = 1.0,
) -> DecisionInputs:
    """Construct `DecisionInputs` from named arguments."""
    return DecisionInputs(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
        relative_ci_lower=relative_ci_lower,
        relative_ci_upper=relative_ci_upper,
        p_value_two_sided=p_value_two_sided,
        p_value_one_sided=p_value_one_sided,
        probability_treatment_better=probability_treatment_better,
        probability_clears_threshold=probability_clears_threshold,
        relative_hdi_lower=relative_hdi_lower,
        relative_hdi_upper=relative_hdi_upper,
        expected_revenue_delta=expected_revenue_delta,
        achieved_power=achieved_power,
        mde_relative=mde_relative,
        required_n_per_arm=required_n_per_arm,
        observed_n_per_arm=observed_n_per_arm,
        control_posterior_alpha=control_posterior_alpha,
        control_posterior_beta=control_posterior_beta,
        treatment_posterior_alpha=treatment_posterior_alpha,
        treatment_posterior_beta=treatment_posterior_beta,
    )


def summarise_result(result: DecisionResult) -> str:
    """Convenience wrapper used by the CLI and notebook."""
    return result.summary()


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Everything the decision engine needs, assembled from one experiment."""

    inputs: DecisionInputs
    validity: ValidityReport
    thresholds: Thresholds = field(default_factory=Thresholds)

    def decide(self) -> DecisionResult:
        """Evaluate the decision rules for this bundle."""
        return decide(self.inputs, self.validity, self.thresholds)


def outcomes() -> Sequence[DecisionOutcome]:
    """The three possible outcomes, for tests and documentation."""
    return ("SHIP", "DONT_SHIP", "INCONCLUSIVE")


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """A complete analysis: effects, inference, design and the decision."""

    bundle: AnalysisBundle
    decision: DecisionResult

    def summary(self) -> str:
        """Multi-line rendering of the whole report."""
        inputs = self.bundle.inputs
        lines = [
            f"control   {inputs.control_rate:.5f}",
            f"treatment {inputs.treatment_rate:.5f}",
            f"absolute  {inputs.absolute_difference:+.5f} "
            f"({inputs.absolute_difference * 100:+.4f} pp)",
            f"relative  {inputs.relative_difference:+.5f} ({inputs.relative_difference:+.2%})",
            f"p (2-sided) {inputs.p_value_two_sided:.6f}",
            f"P(B>A)    {inputs.probability_treatment_better:.4f}",
            "",
            self.decision.summary(),
        ]
        return "\n".join(lines)


def _design_for_null_effect(baseline_rate: float, n_per_arm: int, alpha: float) -> dict[str, float]:
    """Design figures for the case where the observed effect is exactly zero.

    No finite sample size can be powered against an effect of zero, so the
    requirement is reported as zero rather than allowed to abort the analysis.
    A null result is a legitimate outcome and must reach the decision engine,
    not crash the pipeline on its way there.
    """
    from experimentlens.stats.power import minimum_detectable_effect

    mde = minimum_detectable_effect(baseline_rate, n_per_arm, alpha=alpha)
    return {
        "achieved_power": alpha,
        "target_power": 0.80,
        "observed_effect_h": 0.0,
        "mde_absolute": mde.mde_absolute,
        "mde_relative": mde.mde_relative,
        "observed_n_per_arm": float(n_per_arm),
        "required_n_per_arm": 0.0,
        "shortfall_per_arm": 0.0,
    }


def analyse_and_decide(
    clean_result: CleanResult,
    *,
    thresholds: Thresholds | None = None,
    min_practical_lift_rel: float | None = None,
    revenue_per_conversion: float | None = None,
    seed: int = 42,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
) -> ExperimentReport:
    """Run the full pipeline on a cleaned experiment and return the decision.

    This is the composition root: it wires the validated population to the
    frequentist engine, the Bayesian engine, the design calculations and the
    decision rules. Heavy dependencies are imported here rather than at module
    scope so that testing the decision rules in isolation does not pay the cost
    of importing the sampler.

    Args:
        clean_result: Output of `data.clean.clean_experiment`. Raw or
            unvalidated data is rejected here rather than analysed by mistake.
    """
    from experimentlens.stats.bayesian import bayesian_ab_test_from_frame
    from experimentlens.stats.effects import summarise_effects
    from experimentlens.stats.frequentist import two_proportion_z_test_from_frame
    from experimentlens.stats.power import design_summary

    if not hasattr(clean_result, "validity") or not hasattr(clean_result, "frame"):
        raise TypeError(
            "analyse_and_decide expects a CleanResult from clean_experiment(), "
            f"got {type(clean_result).__name__}. Clean the experiment first so that "
            "duplicate and contamination decisions are recorded before inference."
        )

    th = thresholds if thresholds is not None else Thresholds()
    mpwl = (
        min_practical_lift_rel if min_practical_lift_rel is not None else th.min_practical_lift_rel
    )
    revenue = revenue_per_conversion if revenue_per_conversion is not None else th.revenue_per_user

    frame = clean_result.frame
    validity = clean_result.validity

    test = two_proportion_z_test_from_frame(frame, alternative="two-sided", alpha=th.alpha)
    effects = summarise_effects(test)
    bayesian = bayesian_ab_test_from_frame(
        frame,
        min_practical_lift_rel=mpwl,
        hdi_mass=th.hdi_mass,
        revenue_per_conversion=revenue,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
    )
    try:
        design = design_summary(
            test.control_rate,
            min(test.control_trials, test.treatment_trials),
            test.treatment_rate,
            alpha=th.alpha,
            target_power=0.80,
        )
    except ValueError:
        design = _design_for_null_effect(
            test.control_rate, min(test.control_trials, test.treatment_trials), th.alpha
        )

    inputs = build_inputs(
        control_rate=test.control_rate,
        treatment_rate=test.treatment_rate,
        absolute_difference=effects.absolute_difference,
        relative_difference=effects.relative_difference,
        relative_ci_lower=effects.relative.ci_lower,
        relative_ci_upper=effects.relative.ci_upper,
        p_value_two_sided=test.p_value_two_sided,
        p_value_one_sided=test.p_value_one_sided,
        probability_treatment_better=bayesian.probability_treatment_better,
        probability_clears_threshold=bayesian.probability_clears_threshold,
        relative_hdi_lower=bayesian.relative_hdi_lower,
        relative_hdi_upper=bayesian.relative_hdi_upper,
        expected_revenue_delta=bayesian.expected_revenue_delta,
        achieved_power=design["achieved_power"],
        mde_relative=design["mde_relative"],
        required_n_per_arm=int(design["required_n_per_arm"]),
        observed_n_per_arm=min(test.control_trials, test.treatment_trials),
        control_posterior_alpha=bayesian.control.alpha,
        control_posterior_beta=bayesian.control.beta,
        treatment_posterior_alpha=bayesian.treatment.alpha,
        treatment_posterior_beta=bayesian.treatment.beta,
    )

    bundle = AnalysisBundle(inputs=inputs, validity=validity, thresholds=th)
    return ExperimentReport(bundle=bundle, decision=bundle.decide())
