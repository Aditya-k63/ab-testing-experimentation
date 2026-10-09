"""Shared pytest fixtures.

Synthetic fixtures are used throughout so the suite never depends on the Kaggle
CSV being present locally, and so known-answer tests are deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def make_frame(
    *,
    n_control: int = 100,
    n_treatment: int = 100,
    control_conversions: int = 10,
    treatment_conversions: int = 20,
    control_page: str = "old_page",
    treatment_page: str = "new_page",
    user_id_offset: int = 0,
    start: str = "2017-01-01 00:00:00",
    include_country: bool = False,
) -> pd.DataFrame:
    """Build a minimal but contract-compliant experiment table.

    Conversions are assigned to the leading rows of each arm so the resulting
    per-arm rates are exactly ``control_conversions / n_control`` and
    ``treatment_conversions / n_treatment``.
    """
    rows: list[dict[str, object]] = []

    for i in range(n_control):
        rows.append(
            {
                "user_id": user_id_offset + i,
                "timestamp": start,
                "group": "control",
                "landing_page": control_page,
                "converted": 1 if i < control_conversions else 0,
            }
        )

    for i in range(n_treatment):
        rows.append(
            {
                "user_id": user_id_offset + n_control + i,
                "timestamp": start,
                "group": "treatment",
                "landing_page": treatment_page,
                "converted": 1 if i < treatment_conversions else 0,
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    if include_country:
        frame["country"] = "US"

    return frame


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """A frame that satisfies the column contract cleanly."""
    return make_frame()


@pytest.fixture
def minimal_frame() -> pd.DataFrame:
    """A frame with exactly the required columns and no optional extras."""
    return make_frame(include_country=False)


@pytest.fixture
def frame_with_country() -> pd.DataFrame:
    """A frame carrying the optional `country` column."""
    return make_frame(include_country=True)


@pytest.fixture
def frame_factory() -> Callable[..., pd.DataFrame]:
    """Expose `make_frame` to tests that need parameterised variations."""
    return make_frame


@pytest.fixture(scope="session")
def raw_dataset_path() -> Path | None:
    """Resolve the raw CSV if it happens to be present, else None."""
    from experimentlens.config import RAW_DATA_SEARCH_PATHS

    for candidate in RAW_DATA_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture
def loaded_raw(raw_dataset_path: Path | None) -> Iterator[object]:
    """Load the real dataset when available, otherwise skip."""
    if raw_dataset_path is None:
        pytest.skip("Raw ab_data.csv not present; skipping integration test.")

    from experimentlens.data.loader import load_raw_data

    yield load_raw_data(raw_dataset_path)
