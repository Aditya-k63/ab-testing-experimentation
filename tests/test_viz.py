"""Tests for the plotting layer.

Charts are a presentation concern, but a figure that silently returns an empty
trace or an invalid colour still ships a wrong picture. These tests check that
each figure is populated and structurally sound, using synthetic values so the
suite stays fast and does not need the real dataset.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from experimentlens import viz


def _points(figure: go.Figure) -> int:
    return sum(len(trace.y) if trace.y is not None else 0 for trace in figure.data)


class TestConversionRateFigure:
    def test_renders_one_bar_per_arm(self) -> None:
        figure = viz.conversion_rate_figure({"control": 0.12, "treatment": 0.13})
        assert len(figure.data) == 1
        assert len(figure.data[0].x) == 2

    def test_annotates_the_relative_difference(self) -> None:
        figure = viz.conversion_rate_figure({"control": 0.10, "treatment": 0.15})
        assert any("relative" in str(a.text) for a in figure.layout.annotations or [])

    def test_y_axis_is_percentage_formatted(self) -> None:
        figure = viz.conversion_rate_figure({"control": 0.12, "treatment": 0.13})
        assert figure.layout.yaxis.tickformat == ".2%"

    def test_handles_single_arm(self) -> None:
        figure = viz.conversion_rate_figure({"control": 0.12})
        assert _points(figure) == 1


class TestLiftIntervalFigure:
    def test_has_interval_and_estimate_traces(self) -> None:
        figure = viz.lift_interval_figure(0.01, -0.01, 0.03)
        assert len(figure.data) == 2
        assert _points(figure) == 3

    def test_draws_reference_lines(self) -> None:
        figure = viz.lift_interval_figure(0.01, -0.01, 0.03, practical_lift=0.02)
        shapes = figure.layout.shapes
        assert shapes is not None
        assert len(shapes) >= 2

    def test_without_practical_lift_draws_fewer_lines(self) -> None:
        figure = viz.lift_interval_figure(0.01, -0.01, 0.03)
        shapes = figure.layout.shapes
        assert shapes is None or len(shapes) == 1


class TestPosteriorFigures:
    def test_overlap_draws_two_densities(self) -> None:
        figure = viz.posterior_overlap_figure(15_000, 110_000, 15_500, 110_000)
        assert len(figure.data) == 2
        assert _points(figure) == 1200

    def test_fill_colours_are_valid(self) -> None:
        """Plotly rejects 8-digit hex, so this guards a real past failure."""
        figure = viz.posterior_overlap_figure(15_000, 110_000, 15_500, 110_000)
        for trace in figure.data:
            assert str(trace.fillcolor).startswith(("rgba", "rgb", "#"))

    def test_lift_interval_renders(self) -> None:
        figure = viz.posterior_lift_figure(-0.02, 0.03, 0.005, 0.01)
        assert len(figure.data) == 2
        assert figure.layout.xaxis.tickformat == "+.1%"

    def test_lift_interval_annotates_threshold(self) -> None:
        figure = viz.posterior_lift_figure(-0.02, 0.03, 0.005, 0.01)
        assert figure.layout.annotations


class TestPowerCurveFigure:
    def test_one_trace_per_effect_size(self) -> None:
        curve = pd.DataFrame(
            {
                "n_per_arm": [1_000, 5_000, 1_000, 5_000],
                "relative_effect": [0.01, 0.01, 0.02, 0.02],
                "analytic_power": [0.1, 0.3, 0.2, 0.6],
            }
        )
        figure = viz.power_curve_figure(curve)
        assert len(figure.data) == 2

    def test_draws_the_target_reference_line(self) -> None:
        curve = pd.DataFrame(
            {
                "n_per_arm": [1_000],
                "relative_effect": [0.01],
                "analytic_power": [0.5],
            }
        )
        figure = viz.power_curve_figure(curve)
        assert len(figure.layout.shapes) >= 1


class TestCalibrationFigure:
    def test_histogram_matches_sample_count(self) -> None:
        import numpy as np

        p_values = np.linspace(0.0, 1.0, 10_000)
        figure = viz.calibration_figure(p_values)
        assert sum(figure.data[0].y) == pytest.approx(10_000)

    def test_includes_the_uniform_reference(self) -> None:
        import numpy as np

        figure = viz.calibration_figure(np.random.default_rng(0).uniform(size=1_000))
        assert len(figure.data) == 2


class TestCleaningWaterfallFigure:
    def test_one_bar_per_stage(self) -> None:
        reconciliation = pd.DataFrame(
            {
                "step": ["raw_load", "resolve_duplicate_users", "analysis_population"],
                "after": [100, 90, 90],
            }
        )
        figure = viz.cleaning_waterfall_figure(reconciliation)
        assert len(figure.data[0].x) == 3

    def test_first_measure_is_absolute(self) -> None:
        reconciliation = pd.DataFrame({"step": ["raw_load", "clean"], "after": [100, 90]})
        figure = viz.cleaning_waterfall_figure(reconciliation)
        assert figure.data[0].measure[0] == "absolute"


class TestPosteriorPlotRange:
    """The axis must show the posterior mass, not a needle on a blank line."""

    def test_range_centres_on_the_posterior_means(self) -> None:
        low, high = viz.posterior_plot_range([(17_216, 126_078), (17_025, 126_374)])
        centre = (low + high) / 2
        assert 0.119 < centre < 0.121

    def test_range_covers_three_standard_deviations(self) -> None:
        low, high = viz.posterior_plot_range([(17_216, 126_078)])
        mean = 17_216 / (17_216 + 126_078)
        total = 17_216 + 126_078
        sd = (17_216 * 126_078 / (total**2 * (total + 1))) ** 0.5
        assert low == pytest.approx(mean - 3 * sd, abs=1e-9)
        assert high == pytest.approx(mean + 3 * sd, abs=1e-9)

    def test_range_is_far_narrower_than_a_fixed_axis(self) -> None:
        low, high = viz.posterior_plot_range([(17_216, 126_078), (17_025, 126_374)])
        assert (high - low) < 0.02

    def test_range_stays_within_the_unit_interval(self) -> None:
        low, high = viz.posterior_plot_range([(1, 1), (2, 2)])
        assert 0.0 <= low < high <= 1.0

    def test_wider_posterior_gives_wider_axis(self) -> None:
        narrow = viz.posterior_plot_range([(17_216, 126_078)])
        wide = viz.posterior_plot_range([(17_216, 12_6078 - 100_000)])
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_axis_is_set_explicitly_on_the_figure(self) -> None:
        """Relying on autoscale would let a filled trace re-widen the axis."""
        figure = viz.posterior_overlap_figure(17_216, 126_078, 17_025, 126_374)
        assert figure.layout.xaxis.range is not None
        low, high = figure.layout.xaxis.range
        assert (high - low) < 0.02

    def test_posterior_means_are_drawn(self) -> None:
        figure = viz.posterior_overlap_figure(17_216, 126_078, 17_025, 126_374)
        assert len(figure.layout.shapes) == 2


class TestDecisionColours:
    @pytest.mark.parametrize("outcome", ["SHIP", "DONT_SHIP", "INCONCLUSIVE"])
    def test_every_outcome_has_a_colour(self, outcome: str) -> None:
        colour = viz.decision_banner_colour(outcome)
        assert colour.startswith("#")
        assert len(colour) == 7

    def test_unknown_outcome_falls_back(self) -> None:
        assert viz.decision_banner_colour("MAYBE") == "#757575"

    def test_ship_and_dont_ship_differ(self) -> None:
        assert viz.decision_banner_colour("SHIP") != viz.decision_banner_colour("DONT_SHIP")
