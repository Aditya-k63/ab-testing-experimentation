"""Import the Streamlit app as a module, to prove it loads outside Streamlit.

Used by CI as a smoke test and by the test suite, so a broken import fails
locally rather than only on GitHub.

The module must be registered in `sys.modules` before execution. `@dataclass`
resolves annotations through `sys.modules[cls.__module__]`, so a bare
`module_from_spec` followed by `exec_module` raises `AttributeError: 'NoneType'
object has no attribute '__dict__'` on any dataclass declared at module scope.
That is an artefact of the loader, not a fault in the app.

Run directly with `python scripts/import_dashboard.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
MODULE_NAME = "streamlit_app"


def load_dashboard_module() -> object:
    """Import `app/streamlit_app.py` and return the module."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {APP_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Import the dashboard and report what it exposed."""
    module = load_dashboard_module()
    for name in ("main", "run_analysis", "render_decision"):
        if not hasattr(module, name):
            raise AttributeError(f"Dashboard module is missing {name!r}")
    print(f"dashboard imported cleanly from {APP_PATH}")
    print("render entry points: main, run_analysis, render_decision")


if __name__ == "__main__":
    main()
