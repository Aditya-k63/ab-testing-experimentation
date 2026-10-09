"""Plotly figures for the dashboard.

Every function here renders quantities that have already been computed by the
statistical modules. No figure recomputes a rate, an interval or a posterior,
so a chart can never disagree with the number printed beside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy import stats

if TYPE_CHECKING:
    import pandas as pd

ARM_COLOURS: dict[str, str] = {"control": "#4C78A8", "treatment": "#F58518"}
ARM_FILLS: dict[str, str] = {
    "control": "rgba(76, 120, 168, 0.22)",
    "treatment": "rgba(245, 133, 24, 0.22)",
}
_OUTCOME_COLOURS: dict[str, str] = {
    "SHIP": "#2E7D32",
    "DONT_SHIP": "#C62828",
    "INCONCLUSIVE": "#F9A825",
}


def conversion_rate_figure(rates: dict[str, float]) -> go.Figure:
    """Bar chart of conversion rate by arm, with the difference annotated."""
    arms = list(rates)
    values = [rates[a] for a in arms]
    colours = [ARM_COLOURS.get(a, "#888888") for a in arms]

    figure = go.Figure(
        go.Bar(
            x=arms,
            y=values,
            marker_color=colours,
            text=[f"{v:.4%}" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4%}<extra></extra>",
        )
    )
    if "control" in arms and "treatment" in arms:
        difference = rates["treatment"] - rates["control"]
        figure.add_annotation(
            x="treatment",
            y=max(values),
            text=f"{difference:+.4%} relative",
            showarrow=False,
            yshift=28,
            font={"size": 13},
        )
    figure.update_layout(
        title="Conversion rate by arm",
        yaxis_tickformat=".2%",
        height=380,
        margin={"t": 60, "b": 40},
    )
    return figure


def lift_interval_figure(
    difference: float,
    ci_lower: float,
    ci_upper: float,
    threshold: float = 0.0,
    practical_lift: float | None = None,
) -> go.Figure:
    """Confidence interval for the absolute lift, against a reference line."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[ci_lower, ci_upper],
            y=[0, 0],
            mode="lines",
            line={"color": "#4C78A8", "width": 10},
            name="95% CI",
            hovertemplate="%{x:.4%}<extra>95% CI</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[difference],
            y=[0],
            mode="markers",
            marker={"color": "#F58518", "size": 16, "line": {"width": 2, "color": "white"}},
            name="estimate",
            hovertemplate="%{x:.4%}<extra>estimate</extra>",
        )
    )
    figure.add_vline(x=threshold, line_dash="dash", line_color="#888888")
    figure.add_annotation(
        x=threshold, y=1, text="no effect", showarrow=False, yshift=16, font={"size": 11}
    )
    if practical_lift is not None:
        figure.add_vline(x=practical_lift, line_dash="dot", line_color="#2E7D32")
        figure.add_annotation(
            x=practical_lift,
            y=-1,
            text="worthwhile",
            showarrow=False,
            yshift=-16,
            font={"size": 11},
        )

    figure.update_layout(
        title="Absolute lift with 95% confidence interval",
        xaxis_tickformat=".2%",
        height=300,
        margin={"t": 60, "b": 60},
        yaxis={"showticklabels": False, "range": [-2, 2]},
    )
    return figure


def _beta_moments(alpha: float, beta: float) -> tuple[float, float]:
    """Posterior mean and standard deviation for a Beta distribution."""
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total**2 * (total + 1.0))
    return mean, float(np.sqrt(variance))


def posterior_plot_range(
    arms: list[tuple[float, float]], *, margin_sd: float = 3.0
) -> tuple[float, float]:
    """Choose an x-axis that actually shows the posteriors.

    With per-arm counts in the hundreds of thousands the posterior standard
    deviation is around 0.0009, so a fixed 0-to-1 axis leaves the density as a
    near-invisible spike. Centring on the posterior means and covering three
    standard deviations either side puts the mass across the whole plot, which
    is the only way the overlap between arms is readable at all.
    """
    centres: list[float] = []
    spread: list[float] = []
    for alpha, beta in arms:
        mean, sd = _beta_moments(alpha, beta)
        centres.append(mean)
        spread.append(max(sd * margin_sd, 1e-6))

    lower = min(c - s for c, s in zip(centres, spread, strict=True))
    upper = max(c + s for c, s in zip(centres, spread, strict=True))
    return max(0.0, lower), min(1.0, upper)


def posterior_overlap_figure(
    control_alpha: float,
    control_beta: float,
    treatment_alpha: float,
    treatment_beta: float,
) -> go.Figure:
    """Overlaid Beta posteriors for both arms.

    Drawn from the closed-form posterior parameters rather than from samples, so
    the figure is smooth and costs nothing to render.
    """
    low, high = posterior_plot_range(
        [(control_alpha, control_beta), (treatment_alpha, treatment_beta)]
    )
    grid = np.linspace(low, high, 600)

    figure = go.Figure()
    for name, alpha, beta in (
        ("control", control_alpha, control_beta),
        ("treatment", treatment_alpha, treatment_beta),
    ):
        mean = _beta_moments(alpha, beta)[0]
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=stats.beta.pdf(grid, alpha, beta),
                mode="lines",
                fill="tozeroy",
                fillcolor=ARM_FILLS[name],
                line={"color": ARM_COLOURS[name], "width": 2},
                name=name,
                hovertemplate="%{x:.4f}<extra>%{fullData.name}</extra>",
            )
        )
        figure.add_vline(x=mean, line_dash="dot", line_color=ARM_COLOURS[name], opacity=0.55)

    figure.update_layout(
        title="Posterior conversion rate by arm (zoomed to the mass)",
        xaxis_title="conversion rate",
        yaxis_title="density",
        xaxis_tickformat=".4f",
        xaxis={"range": [low, high]},
        height=380,
        margin={"t": 60, "b": 40},
        legend={"title": {"text": "posterior mean (dotted)"}},
    )
    return figure


def posterior_lift_figure(
    hdi_lower: float,
    hdi_upper: float,
    mean: float,
    practical_lift: float,
) -> go.Figure:
    """Credible interval for the relative lift against the practical threshold."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[hdi_lower, hdi_upper],
            y=[0, 0],
            mode="lines",
            line={"color": "#7B4FBF", "width": 10},
            name="94% HDI",
            hovertemplate="%{x:+.2%}<extra>94% HDI</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[mean],
            y=[0],
            mode="markers",
            marker={"color": "#7B4FBF", "size": 16, "line": {"width": 2, "color": "white"}},
            name="posterior mean",
            hovertemplate="%{x:+.2%}<extra>posterior mean</extra>",
        )
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="#888888")
    figure.add_vline(x=practical_lift, line_dash="dot", line_color="#2E7D32")
    figure.add_annotation(
        x=practical_lift, y=1, text="minimum worthwhile", showarrow=False, yshift=16
    )
    figure.update_layout(
        title="Relative lift with 94% credible interval",
        xaxis_tickformat="+.1%",
        height=300,
        margin={"t": 60, "b": 40},
        yaxis={"showticklabels": False, "range": [-2, 2]},
    )
    return figure


def power_curve_figure(curve: pd.DataFrame) -> go.Figure:
    """Power against sample size, one line per true effect size."""
    figure = go.Figure()
    for relative, group in curve.groupby("relative_effect"):
        ordered = group.sort_values("n_per_arm")
        figure.add_trace(
            go.Scatter(
                x=ordered["n_per_arm"],
                y=ordered["analytic_power"],
                mode="lines+markers",
                name=f"{relative:+.2%} relative",
                hovertemplate="n=%{x:,}<br>power=%{y:.3f}<extra></extra>",
            )
        )
    figure.add_hline(y=0.80, line_dash="dash", line_color="#888888")
    figure.update_layout(
        title="Power against sample size",
        xaxis_title="users per arm",
        yaxis_title="power",
        yaxis_range=[0, 1],
        height=400,
        margin={"t": 60, "b": 50},
    )
    return figure


def calibration_figure(p_values: np.ndarray, alpha: float = 0.05) -> go.Figure:
    """Null p-value distribution against the uniform reference."""
    histogram = np.histogram(p_values, bins=20, range=(0.0, 1.0))
    centres = (histogram[1][:-1] + histogram[1][1:]) / 2
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=centres,
            y=histogram[0],
            marker_color="#4C78A8",
            name="simulated under null",
            hovertemplate="p in [%{x:.2f}]<br>%{y} runs<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[len(p_values) / 20, len(p_values) / 20],
            mode="lines",
            line={"color": "#C62828", "dash": "dash"},
            name="uniform expectation",
        )
    )
    figure.add_vline(x=alpha, line_dash="dot", line_color="#F9A825")
    figure.update_layout(
        title="Null p-values should be uniform",
        xaxis_title="p-value",
        yaxis_title="count",
        height=340,
        margin={"t": 60, "b": 40},
    )
    return figure


SUMMARY_ROWS: frozenset[str] = frozenset({"analysis_population"})
"""Reconciliation rows that restate the total rather than forming a stage."""


def cleaning_waterfall_figure(reconciliation: pd.DataFrame) -> go.Figure:
    """Population through the cleaning pipeline, stage by stage.

    A Plotly waterfall treats ``measure="relative"`` entries as *deltas* and adds
    them to a running total. The reconciliation table stores cumulative
    ``after`` counts, so feeding those straight in double-counts every stage -
    this chart once reported 1.15 million users from a 294k dataset. Only the
    first row is drawn as an absolute level; the rest are drawn as the negative
    of what each step removed, and the closing summary row is excluded because
    it restates the total rather than describing a transformation.
    """
    stages = reconciliation[~reconciliation["step"].isin(SUMMARY_ROWS)]
    if stages.empty:
        raise ValueError("Reconciliation table contains no cleaning stages to plot")

    opening = int(stages["after"].iloc[0])
    removals = [-int(value) for value in stages["removed"].iloc[1:]]

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(removals),
            x=list(stages["step"]),
            y=[opening, *removals],
            decreasing={"marker": {"color": "#C62828"}},
            increasing={"marker": {"color": "#2E7D32"}},
            connector={"line": {"color": "#888888"}},
            text=[f"{opening:,}", *[f"-{abs(v):,}" for v in removals]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,d} users<extra></extra>",
        )
    )
    figure.update_layout(
        title="Population through the cleaning pipeline",
        yaxis_title="users",
        yaxis_rangemode="tozero",
        height=360,
        margin={"t": 60, "b": 90},
    )
    return figure


def decision_banner_colour(outcome: str) -> str:
    """Colour associated with a decision outcome."""
    return _OUTCOME_COLOURS.get(outcome, "#757575")
