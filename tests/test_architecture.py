"""Architectural tests.

These enforce the project's second rule: *test before visualizing*. All
statistical and decision logic lives in tested modules under
`src/experimentlens/`; the Streamlit app is a presentation layer.

The rules are mechanical rather than stylistic. They fail the build when
violated, so the boundary cannot erode quietly as the project grows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "experimentlens"
APP_ROOT = PROJECT_ROOT / "app"

FORBIDDEN_IN_PACKAGE = frozenset({"streamlit", "matplotlib.pyplot", "plotly.express", "seaborn"})
FORBIDDEN_IN_APP = frozenset({"scipy", "statsmodels", "pymc", "numpy", "arviz"})


def python_files(root: Path) -> list[Path]:
    """Every Python file under a root, excluding caches."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )


def imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file, including nested ones."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


class TestPackageIsFrameworkFree:
    def test_package_sources_exist(self) -> None:
        assert python_files(PACKAGE_ROOT), "No package sources found"

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_IN_PACKAGE))
    def test_package_does_not_import_ui_frameworks(self, forbidden: str) -> None:
        offenders = [
            f"{path.relative_to(PROJECT_ROOT)}: {module}"
            for path in python_files(PACKAGE_ROOT)
            for module in imported_modules(path)
            if module == forbidden or module.startswith(f"{forbidden}.")
        ]
        assert not offenders, (
            f"UI framework '{forbidden}' imported inside the package: {offenders}. "
            "The package must stay importable without a presentation layer."
        )

    def test_package_does_not_import_the_app(self) -> None:
        for path in python_files(PACKAGE_ROOT):
            modules = imported_modules(path)
            assert "app" not in modules
            assert "streamlit_app" not in modules


class TestAppIsThin:
    def test_app_sources_exist(self) -> None:
        assert python_files(APP_ROOT), "No app sources found"

    def test_app_does_not_import_scientific_libraries_directly(self) -> None:
        """Statistical libraries must be reached through the package, not imported.

        Direct imports would let the dashboard grow its own untested copy of the
        maths, which is exactly the failure mode this rule exists to prevent.
        """
        offenders = [
            f"{path.relative_to(PROJECT_ROOT)}: {module}"
            for path in python_files(APP_ROOT)
            for module in imported_modules(path)
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_IN_APP
            )
        ]
        assert not offenders, (
            f"Scientific libraries imported directly in the app: {offenders}. "
            "Use the tested functions in experimentlens instead."
        )

    def test_app_imports_the_package(self) -> None:
        modules = {module for path in python_files(APP_ROOT) for module in imported_modules(path)}
        assert any(m.startswith("experimentlens") for m in modules)

    def test_dashboard_module_loads(self) -> None:
        """The dashboard must import outside Streamlit.

        Guards a class of failure pytest never sees: a bad import in `app/` is
        invisible until someone runs `streamlit run`. The helper registers the
        module in `sys.modules` because `@dataclass` resolves annotations through
        it, and omitting that raises an AttributeError that looks like a fault
        in the app when it is really a fault in the loader.
        """
        import sys

        scripts = str(PROJECT_ROOT / "scripts")
        sys.path.insert(0, scripts)
        try:
            from import_dashboard import load_dashboard_module

            module = load_dashboard_module()
        finally:
            sys.path.remove(scripts)

        for name in ("main", "run_analysis", "render_decision"):
            assert hasattr(module, name), f"dashboard module is missing {name!r}"

    def test_dashboard_import_helper_exists(self) -> None:
        assert (PROJECT_ROOT / "scripts" / "import_dashboard.py").is_file()


class TestDecisionLogicIsTested:
    def test_decision_module_has_a_matching_test_file(self) -> None:
        test_file = PROJECT_ROOT / "tests" / "test_decision.py"
        assert test_file.is_file()
        assert (PACKAGE_ROOT / "stats" / "decision.py").is_file()

    def test_decision_thresholds_are_injected(self) -> None:
        """The engine must take thresholds as an argument, not read globals.

        If the engine read module-level state the decision tests could not
        exercise different business thresholds without mutating global state.
        """
        source = (PACKAGE_ROOT / "stats" / "decision.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        decide_node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide"
        )
        parameters = {arg.arg for arg in decide_node.args.args}
        assert "thresholds" in parameters
