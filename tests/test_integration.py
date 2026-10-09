"""Integration tests for the end-to-end pipeline.

These exercise the composition root - load, validate, clean, analyse, decide -
against synthetic frames, and confirm the numbers that travel between stages are
internally consistent. The real dataset path is covered separately and skips
when the CSV is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_frame
from experimentlens.config import Thresholds
from experimentlens.data.clean import clean_experiment
from experimentlens.data.loader import RawDataset, RawSnapshot
from experimentlens.data.schema import validate_schema
from experimentlens.stats.decision import analyse_and_decide

FAST = {"draws": 200, "tune": 150, "chains": 2, "seed": 42}


def _wrap(frame: pd.DataFrame) -> RawDataset:
    snapshot = RawSnapshot(
        n_rows=len(frame),
        n_unique_users=int(frame["user_id"].nunique()),
        group_counts={str(k): int(v) for k, v in frame["group"].value_counts().items()},
        page_counts={str(k): int(v) for k, v in frame["landing_page"].value_counts().items()},
    )
    return RawDataset(
        path=Path("synthetic.csv"),
        frame=frame,
        schema=validate_schema(frame),
        snapshot=snapshot,
    )


class TestEndToEnd:
    def test_clear_win_produces_ship(self) -> None:
        frame = make_frame(
            n_control=20_000,
            n_treatment=20_000,
            control_conversions=2_000,
            treatment_conversions=2_600,
        )
        report = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST)
        assert report.decision.outcome == "SHIP"

    def test_harmful_change_produces_dont_ship(self) -> None:
        frame = make_frame(
            n_control=20_000,
            n_treatment=20_000,
            control_conversions=2_400,
            treatment_conversions=2_000,
        )
        report = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST)
        assert report.decision.outcome == "DONT_SHIP"

    def test_identical_arms_are_not_shipped(self) -> None:
        frame = make_frame(
            n_control=20_000,
            n_treatment=20_000,
            control_conversions=2_400,
            treatment_conversions=2_400,
        )
        report = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST)
        assert report.decision.outcome != "SHIP"

    def test_mrpwl_controls_the_outcome(self) -> None:
        """A 7.5% lift should ship against a 1% bar and fail a 10% bar."""
        frame = make_frame(
            n_control=20_000,
            n_treatment=20_000,
            control_conversions=2_000,
            treatment_conversions=2_150,
        )
        permissive = analyse_and_decide(
            clean_experiment(_wrap(frame)), min_practical_lift_rel=0.01, **FAST
        )
        strict = analyse_and_decide(
            clean_experiment(_wrap(frame)), min_practical_lift_rel=0.10, **FAST
        )
        assert permissive.decision.outcome == "SHIP"
        assert strict.decision.outcome != "SHIP"

    def test_summary_renders_every_section(self) -> None:
        frame = make_frame(n_control=5_000, n_treatment=5_000)
        report = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST)
        text = report.summary()

        assert "control" in text
        assert "DECISION:" in text

    def test_posterior_parameters_are_populated(self) -> None:
        """The dashboard plots Beta densities; unpopulated shapes would break it."""
        frame = make_frame(n_control=5_000, n_treatment=5_000, treatment_conversions=600)
        inputs = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST).bundle.inputs

        assert inputs.control_posterior_alpha > 1.0
        assert inputs.control_posterior_beta > 1.0
        assert inputs.treatment_posterior_alpha > 1.0
        assert inputs.treatment_posterior_beta > 1.0

    def test_rates_agree_between_arms(self) -> None:
        frame = make_frame(n_control=10_000, n_treatment=10_000, treatment_conversions=1_300)
        inputs = analyse_and_decide(clean_experiment(_wrap(frame)), **FAST).bundle.inputs

        assert inputs.relative_difference == pytest.approx(
            (inputs.treatment_rate - inputs.control_rate) / inputs.control_rate
        )

    def test_thresholds_object_is_respected(self) -> None:
        frame = make_frame(n_control=10_000, n_treatment=10_000)
        thresholds = Thresholds(min_practical_lift_rel=0.5)
        report = analyse_and_decide(clean_experiment(_wrap(frame)), thresholds=thresholds, **FAST)
        assert report.bundle.thresholds.min_practical_lift_rel == 0.5


class TestRealDatasetPipeline:
    def test_full_pipeline_reaches_a_decision(self, loaded_raw: RawDataset) -> None:
        cleaned = clean_experiment(loaded_raw)
        report = analyse_and_decide(cleaned, **FAST)

        assert report.decision.outcome in {"SHIP", "DONT_SHIP", "INCONCLUSIVE"}
        assert report.decision.reasoning or report.decision.blocking_reasons

    def test_population_is_reconciled(self, loaded_raw: RawDataset) -> None:
        cleaned = clean_experiment(loaded_raw)
        cleaned.verify_reconciliation()
        assert cleaned.n_rows_raw - cleaned.total_removed == cleaned.n_rows_clean

    def test_reported_lift_matches_arm_rates(self, loaded_raw: RawDataset) -> None:
        from experimentlens.stats.validity import arm_conversion_rates

        cleaned = clean_experiment(loaded_raw)
        report = analyse_and_decide(cleaned, **FAST)
        rates = arm_conversion_rates(cleaned.frame)
        inputs = report.bundle.inputs

        assert inputs.control_rate == pytest.approx(rates["control"])
        assert inputs.treatment_rate == pytest.approx(rates["treatment"])

    def test_is_deterministic(self, loaded_raw: RawDataset) -> None:
        """Same seed, same answer - the reproducibility rule."""
        cleaned = clean_experiment(loaded_raw)
        first = analyse_and_decide(cleaned, **FAST).decision.outcome
        second = analyse_and_decide(cleaned, **FAST).decision.outcome
        assert first == second
