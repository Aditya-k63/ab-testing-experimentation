"""Central configuration for ExperimentLens.

Single source of truth for filesystem layout, the raw-data column contract,
and every tunable decision threshold. Nothing in the package reads these from
the environment or from widget state, so an analysis run is fully determined
by this module plus the data.

Assumptions declared here are mirrored in the README assumptions table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__version__: Final[str] = "0.1.0"

SEED: Final[int] = 42
"""Master seed. Simulation and Bayesian routines also take an explicit seed
argument; this is the project default and the value recorded in the README."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing ``pyproject.toml``.

    Resolving the root this way keeps the package importable and functional no
    matter the process working directory (notebooks, pytest, and Streamlit all
    launch from different places).
    """
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root: no pyproject.toml found above "
        f"{current}. Run from inside the experimentlens checkout."
    )


PROJECT_ROOT: Final[Path] = find_project_root()

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

RAW_FILENAME: Final[str] = "ab_data.csv"

RAW_DATA_SEARCH_PATHS: Final[tuple[Path, ...]] = (
    RAW_DATA_DIR / RAW_FILENAME,
    PROJECT_ROOT / "dataset" / RAW_FILENAME,
)

MISSING_DATA_MESSAGE: Final[str] = (
    "Raw experiment data not found.\n\n"
    "Download the Kaggle dataset 'A/B Testing' (Marketing campaign results) and "
    f"save the CSV as:\n    {RAW_DATA_DIR / RAW_FILENAME}\n\n"
    "See the 'Getting the data' section of the README for the full procedure."
)


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "timestamp",
    "group",
    "landing_page",
    "converted",
)
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("country",)
"""Optional columns power supplementary diagnostics only and are never required
by the primary analysis path."""

CONTROL: Final[str] = "control"
TREATMENT: Final[str] = "treatment"
CONTROL_PAGE: Final[str] = "old_page"
TREATMENT_PAGE: Final[str] = "new_page"

EXPECTED_GROUPS: Final[tuple[str, ...]] = (CONTROL, TREATMENT)
EXPECTED_PAGES: Final[tuple[str, ...]] = (CONTROL_PAGE, TREATMENT_PAGE)

BINARY_OUTCOME_COLUMN: Final[str] = "converted"
TIMESTAMP_COLUMN: Final[str] = "timestamp"
USER_ID_COLUMN: Final[str] = "user_id"
GROUP_COLUMN: Final[str] = "group"
PAGE_COLUMN: Final[str] = "landing_page"


# ---------------------------------------------------------------------------
# Experiment design & decision thresholds
# ---------------------------------------------------------------------------

EXPECTED_RATIO: Final[float] = 0.5
"""Expected share of the traffic allocated to the treatment arm. The raw
dataset is a 50/50 control/treatment split; declared explicitly so the SRM test
has a stated null hypothesis rather than an inferred one."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Decision thresholds injected into the decision engine.

    The engine never reads these from the UI, which keeps every branch of the
    decision logic unit-testable and the dashboard a pure presentation layer.
    """

    alpha: float = 0.05
    """Two-sided significance level for the frequentist test."""

    srm_alpha: float = 0.001
    """Significance level for the sample ratio mismatch check. Set far below the
    outcome test's alpha so that only a gross allocation imbalance is flagged;
    at large n, a conventional threshold would fire on ordinary drift."""

    min_practical_lift_rel: float = 0.01
    """Minimum practically worthwhile relative lift (1%). The axis that
    separates statistical significance from business significance."""

    hdi_mass: float = 0.94
    """Mass of the highest density interval reported for posterior summaries."""

    min_posterior_probability: float = 0.80
    """Posterior probability that the relative lift clears the practical
    threshold, required before the engine will recommend SHIP."""

    dont_ship_posterior_threshold: float = 0.20
    """Posterior probability of B beating A at or below which the engine treats
    the effect as absent or harmful."""

    revenue_per_user: float = 25.0
    """Marginal value used to express expected loss in business units. Replace
    with a figure appropriate to the business under analysis."""

    max_contamination_rate: float = 0.02
    """Above this share of rows with group/landing-page mismatch, the validity
    gate fails and the decision is INCONCLUSIVE regardless of statistics."""

    min_sample_per_arm: int = 1000
    """Floor below which the experiment is treated as underpowered by default."""

    min_effect_for_power_curve: float = 0.0
    """Lower bound of the relative-lift axis used when plotting power curves."""

    max_effect_for_power_curve: float = 0.05
    """Upper bound of the relative-lift axis used when plotting power curves."""

    extras: dict[str, float] = field(default_factory=dict)
    """Reserved for ad-hoc threshold overrides without widening the contract."""


DEFAULT_THRESHOLDS: Final[Thresholds] = Thresholds()
