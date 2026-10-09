"""ExperimentLens dashboard.

A presentation layer only. Every number, interval, posterior and decision shown
here is produced by a tested function in `experimentlens` and rendered as-is.
This module contains no statistical computation: if a value appears on screen,
a function elsewhere calculated and tested it.

That boundary is enforced by `tests/test_architecture.py`, which fails the build
if this file imports scipy or statsmodels directly, or if the package imports
streamlit.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")

from experimentlens import viz  # noqa: E402
from experimentlens.config import Thresholds  # noqa: E402
from experimentlens.data.clean import CleanResult, clean_experiment, summarise  # noqa: E402
from experimentlens.data.duplicates import ConflictPolicy  # noqa: E402
from experimentlens.data.loader import (  # noqa: E402
    RawDataNotFoundError,
    RawDataset,
    RawSnapshot,
    load_raw_data,
)
from experimentlens.data.schema import validate_schema  # noqa: E402
from experimentlens.stats.decision import (  # noqa: E402
    CORRELATED_EVIDENCE_NOTE,
    VIEW_TITLES,
    DecisionResult,
    ExperimentReport,
    analyse_and_decide,
)
from experimentlens.stats.validity import arm_conversion_rates  # noqa: E402

st.set_page_config(page_title="ExperimentLens", layout="wide")

SAMPLER = {"draws": 1000, "tune": 500, "chains": 4}


@dataclass(frozen=True, slots=True)
class AnalysisView:
    """Everything the dashboard renders, already computed."""

    cleaned: CleanResult
    report: ExperimentReport
    rates: dict[str, float]


@st.cache_data(show_spinner=False)
def run_analysis(
    frame: pd.DataFrame,
    thresholds: Thresholds,
    policy: ConflictPolicy,
    seed: int,
) -> AnalysisView:
    """Run the full pipeline and collect everything the dashboard renders."""
    snapshot = RawSnapshot(
        n_rows=len(frame),
        n_unique_users=int(frame["user_id"].nunique()),
        group_counts={str(k): int(v) for k, v in frame["group"].value_counts().items()},
        page_counts={str(k): int(v) for k, v in frame["landing_page"].value_counts().items()},
    )
    raw = RawDataset(
        path=Path("source"),
        frame=frame,
        schema=validate_schema(frame),
        snapshot=snapshot,
    )
    cleaned = clean_experiment(raw=raw, thresholds=thresholds, conflict_policy=policy)
    report = analyse_and_decide(cleaned, thresholds=thresholds, seed=seed, **SAMPLER)
    return AnalysisView(cleaned=cleaned, report=report, rates=arm_conversion_rates(cleaned.frame))


def sidebar() -> tuple[Thresholds, ConflictPolicy, int]:
    """Render controls and return the resolved configuration."""
    st.sidebar.title("Configuration")
    selection = st.sidebar.selectbox(
        "Conflicting-assignment policy",
        options=["exclude_user", "keep_consistent_row"],
        help=(
            "What to do with a user recorded under more than one group/page "
            "assignment. 'exclude_user' is the conservative choice."
        ),
    )
    policy: ConflictPolicy = cast(ConflictPolicy, selection)
    practical = st.sidebar.slider(
        "Minimum worthwhile lift (relative)",
        min_value=0.0,
        max_value=0.10,
        value=0.01,
        step=0.005,
        format="%.3f",
    )
    alpha = st.sidebar.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)
    hdi = st.sidebar.selectbox("Credible interval mass", [0.90, 0.94, 0.95, 0.99], index=1)
    revenue = st.sidebar.number_input("Revenue per conversion", min_value=0.0, value=25.0, step=1.0)
    seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)

    thresholds = Thresholds(
        alpha=float(alpha),
        min_practical_lift_rel=practical,
        hdi_mass=float(hdi),
        revenue_per_user=float(revenue),
    )
    return thresholds, policy, int(seed)


def render_header(report: ExperimentReport, colour: str) -> None:
    """Render the decision banner."""
    decision: DecisionResult = report.decision
    inputs = report.bundle.inputs

    st.markdown(
        f"<div style='background:{colour};padding:18px;border-radius:8px'>"
        f"<h2 style='margin:0'>{decision.label}</h2>"
        f"<p style='margin:6px 0 0;font-size:17px'>{decision.headline}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    first, second, third = st.columns(3)
    first.metric(
        "Relative lift",
        f"{inputs.relative_difference:+.2%}",
        delta=f"{inputs.absolute_difference * 100:+.3f} pp",
        delta_color="off",
    )
    second.metric("P(treatment > control)", f"{inputs.probability_treatment_better:.1%}")
    third.metric("Outcome", decision.label)


def render_data_quality(cleaned: CleanResult) -> None:
    """Render the data-quality and validity section."""
    st.subheader("1. Data quality")
    validity = cleaned.validity
    duplicates = cleaned.duplicates

    left, right = st.columns(2)

    with left:
        st.markdown("**Schema**")
        st.write(f"`{cleaned.frame.shape[0]:,}` rows x `{cleaned.frame.shape[1]}` columns")
        st.markdown("**Duplicate classification**")
        counts = duplicates.counts
        st.dataframe(
            {"class": list(counts), "users": list(counts.values())},
            hide_index=True,
            width="stretch",
        )

    with right:
        st.markdown("**Sample ratio mismatch**")
        if validity.srm.passed:
            st.success(validity.srm.summary())
        else:
            st.error(validity.srm.summary())
        st.caption("A pass is necessary but not sufficient evidence that randomisation held.")
        st.markdown("**Blocking issues**")
        if validity.blocking_reasons:
            for reason in validity.blocking_reasons:
                st.error(reason)
        else:
            st.info("None")

    st.markdown("**Sample ratio mismatch across candidate populations**")
    st.caption(
        "The SRM basis is a choice, so every candidate is shown rather than only "
        "the one that drives the decision. Raw rows double-count repeated users, "
        "so they are not the right basis; the default is one row per user with "
        "conflicting assignments excluded but contaminated rows still present."
    )
    comparison = cleaned.srm_comparison()
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "control": st.column_config.NumberColumn(format="%d"),
            "treatment": st.column_config.NumberColumn(format="%d"),
            "total": st.column_config.NumberColumn(format="%d"),
            "chi2": st.column_config.NumberColumn(format="%.5f"),
            "p_value": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    verdicts = set(comparison["verdict"])
    if verdicts == {"PASS"}:
        st.success(
            "All four populations pass, so the conclusion does not depend on which "
            "basis is chosen. A pass remains necessary but not sufficient evidence "
            "that randomisation held."
        )
    else:
        st.error("SRM does not pass on every population; investigate before deciding.")

    st.markdown("**Population reconciliation**")
    reconciliation = cleaned.reconciliation()
    st.dataframe(reconciliation, hide_index=True, width="stretch")
    st.plotly_chart(viz.cleaning_waterfall_figure(reconciliation), width="stretch")
    st.caption(f"Raw snapshot: {cleaned.snapshot.summary()}")


def render_effect(view: AnalysisView) -> None:
    """Render the effect size section."""
    st.subheader("2. Effect")
    inputs = view.report.bundle.inputs
    thresholds = view.report.bundle.thresholds

    st.plotly_chart(viz.conversion_rate_figure(view.rates), width="stretch")
    st.plotly_chart(
        viz.lift_interval_figure(
            inputs.absolute_difference,
            inputs.relative_ci_lower,
            inputs.relative_ci_upper,
            threshold=0.0,
            practical_lift=thresholds.min_practical_lift_rel,
        ),
        width="stretch",
    )


def render_inference(report: ExperimentReport) -> None:
    """Render the frequentist and Bayesian sections."""
    inputs = report.bundle.inputs
    thresholds = report.bundle.thresholds

    st.subheader("3. Frequentist")
    first, second = st.columns(2)
    first.metric("p-value (two-sided)", f"{inputs.p_value_two_sided:.4f}")
    second.metric("Significant at alpha", str(inputs.p_value_two_sided < thresholds.alpha))
    st.caption(
        f"95% CI on relative lift: [{inputs.relative_ci_lower:+.2%}, "
        f"{inputs.relative_ci_upper:+.2%}]"
    )
    st.info(
        "Statistical significance is reported here but never decides the outcome. "
        "A p-value below alpha says the data would be unlikely under no effect; it "
        "says nothing about whether the effect is worth having."
    )
    st.caption(
        "This p-value is two-sided. The power and MDE figures in section 5 are "
        "one-sided, matching the pre-registered hypothesis that treatment beats "
        "control; a one-sided test is the more powerful design for that hypothesis, "
        "so the two are not directly comparable."
    )

    st.subheader("4. Bayesian")
    posterior_column, lift_column = st.columns(2)
    with posterior_column:
        figure: go.Figure = viz.posterior_overlap_figure(
            inputs.control_posterior_alpha,
            inputs.control_posterior_beta,
            inputs.treatment_posterior_alpha,
            inputs.treatment_posterior_beta,
        )
        st.plotly_chart(figure, width="stretch")
    with lift_column:
        st.plotly_chart(
            viz.posterior_lift_figure(
                inputs.relative_hdi_lower,
                inputs.relative_hdi_upper,
                inputs.relative_difference,
                thresholds.min_practical_lift_rel,
            ),
            width="stretch",
        )

    metrics = st.columns(4)
    metrics[0].metric("P(treatment > control)", f"{inputs.probability_treatment_better:.2%}")
    metrics[1].metric(
        f"P(lift > {thresholds.min_practical_lift_rel:.0%})",
        f"{inputs.probability_clears_threshold:.2%}",
    )
    metrics[2].metric(
        "Modelled revenue change",
        f"{inputs.expected_revenue_delta:+,.0f}",
        help=(
            "Posterior mean of (pB - pA) x control users x revenue per conversion. "
            "A modelled quantity, not observed revenue; it scales linearly with the "
            "revenue assumption."
        ),
    )
    metrics[3].metric(
        f"{thresholds.hdi_mass:.0%} HDI upper (relative)", f"{inputs.relative_hdi_upper:+.2%}"
    )


def render_design(report: ExperimentReport) -> None:
    """Render the experiment-design section."""
    st.subheader("5. Experiment design")
    inputs = report.bundle.inputs
    thresholds = report.bundle.thresholds

    metrics = st.columns(4)
    metrics[0].metric("Achieved power", f"{inputs.achieved_power:.1%}")
    metrics[1].metric("Minimum detectable effect", f"{inputs.mde_relative:+.2%}")
    metrics[2].metric("Users observed per arm", f"{inputs.observed_n_per_arm:,}")
    metrics[3].metric("Needed for 80% power", f"{inputs.required_n_per_arm:,}")

    if inputs.mde_relative > thresholds.min_practical_lift_rel:
        st.warning(
            f"The minimum detectable effect ({inputs.mde_relative:+.2%}) exceeds the "
            f"practical threshold ({thresholds.min_practical_lift_rel:+.2%}). This "
            "experiment was never able to settle the question it was run to answer. "
            "Reported as a design finding; it does not override the evidence."
        )
    st.caption(
        "Power computed against the observed effect is a function of that estimate, "
        "so it is reported for context and never used to gate the decision."
    )


def render_decision(decision: DecisionResult) -> None:
    """Render the evidence by lens, then the recommendation."""
    st.subheader("6. Decision")
    st.caption(CORRELATED_EVIDENCE_NOTE)

    for view, checks in decision.evidence_by_view().items():
        st.markdown(f"**{VIEW_TITLES.get(view, view)}**")
        for check in checks:
            mark = "PASS" if check.passed else "fail"
            st.write(f":{'white_check_mark' if check.passed else 'x'}: {mark} — **{check.name}**  ")
            st.caption(check.detail)

    st.markdown("**Reasoning**")
    for reason in decision.reasoning:
        st.markdown(f"- {reason}")

    if decision.design_notes:
        st.markdown("**Design notes**")
        for note in decision.design_notes:
            st.markdown(f"- {note}")

    if decision.notes:
        st.markdown("**Policy and triggers**")
        for note in decision.notes:
            st.caption(note)

    st.markdown("**Recommended action**")
    st.info(decision.recommended_action)


def main() -> None:
    """Entry point."""
    thresholds, policy, seed = sidebar()

    st.title("ExperimentLens")
    st.caption(
        "E-commerce A/B testing and experimentation analytics. A p-value below 0.05 "
        "is never sufficient to ship."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data")
    upload = st.sidebar.file_uploader("Upload experiment CSV", type=["csv"])

    if upload is not None:
        frame = pd.read_csv(upload)
        frame.columns = [str(c).strip() for c in frame.columns]
        st.sidebar.caption(f"Using uploaded file ({len(frame):,} rows)")
    else:
        try:
            frame = load_raw_data().frame
        except RawDataNotFoundError:
            st.error(
                "No experiment data found. Upload a CSV, or place `ab_data.csv` in `data/raw/`."
            )
            st.stop()

    if frame.empty:
        st.warning("The dataset is empty.")
        st.stop()

    with st.spinner("Running analysis..."):
        view = run_analysis(frame, thresholds, policy, seed)

    render_header(view.report, viz.decision_banner_colour(view.report.decision.outcome))
    render_data_quality(view.cleaned)
    render_effect(view)
    render_inference(view.report)
    render_design(view.report)
    render_decision(view.report.decision)

    with st.expander("Raw pipeline log"):
        st.code(summarise(view.cleaned), language="text")


if __name__ == "__main__":
    main()
