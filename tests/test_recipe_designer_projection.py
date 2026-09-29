"""The recipe designer projects a recipe's disk the way the engine does for the run the page starts.

Breakage these prevent: once the render choice became a required argument
of disk_budget.projected_run_bytes, the recipe designer still called it
without one, so every projection it makes (the disk figure of each recipe
row, the output intervals it picks per card, the words of each row's disk
basis) raised TypeError and no seed recipe could be designed or rebuilt.
No test called the designer, so the suite stayed green.  A designer that
answered the choice with "no pictures" would instead fit rows to a disk
budget the run then overruns, because the page starts every recipe with
the standard picture set.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import disk_budget, runplan
from gpuwm.first_products import DEFAULT_RENDER_PRODUCTS

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


@pytest.fixture
def design(tmp_path, monkeypatch):
    """tools/wiki_seed/design_recipes.py, with its work folder under tmp_path."""
    monkeypatch.setenv("RECIPE_WORK", str(tmp_path / "work"))
    # The designer puts its engine tree first on sys.path; keep that local.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "design_recipes_under_test", ROOT / "tools" / "wiki_seed" / "design_recipes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_rows():
    for path in sorted((ROOT / "gpuwm/gui/seed/recipes").glob("*.json")):
        for recipe in json.loads(path.read_text(encoding="utf-8"))["recipes"]:
            for row in recipe["cards"]:
                if row.get("fits"):
                    yield path.name, row


def _fit(row, restart_interval_s=0.0):
    """The part of a designer fit that the projection reads, rebuilt from a shipped row."""
    return {"domains": [{"nx": d["nx"], "ny": d["ny"], "nz": d["nz"]} for d in row["domains"]],
            "restart_interval_s": restart_interval_s, "fetch": None, "chain": None}


def test_the_designer_projects_every_shipped_row_with_its_pictures(design):
    rows = list(_shipped_rows())
    assert rows
    for name, row in rows:
        out = row["output"]
        p = design.projection(_fit(row), float(row["length_h"]) * 3600,
                              out["history_interval_s"], out["nest_history_interval_s"])
        assert p["picture_bytes"] > 0, (name, row["card_gb"])
        # The row's own disk basis names the pictures and the history the
        # designer projected when the row was built.
        assert f"the rendered pictures, {p['picture_bytes'] / GIB:.1f} GiB" in row["disk_basis"], \
            (name, row["card_gb"])
        assert f"the history files, {p['history_bytes'] / GIB:.1f} GiB" in row["disk_basis"], \
            (name, row["card_gb"])


def _page_plan(route, run_dir):
    """The run-plan the page writes when a recipe row is pressed (gpuwm/gui/api.py)."""
    if route == "experiment":
        # An ERA5 row and the storm-following layout: the page names the set.
        return SimpleNamespace(route="experiment", config_intent=None, run_dir=run_dir,
                               run_options={"render_products": DEFAULT_RENDER_PRODUCTS})
    return SimpleNamespace(route="prepared", config_intent={}, run_dir=run_dir, run_options={})


def test_saved_recipe_totals_match_the_current_picture_allowance(design):
    """The event-page disk check reads these totals before run-plan starts."""
    for name, row in _shipped_rows():
        out = row["output"]
        p = design.projection(_fit(row, restart_interval_s=3600), row["length_h"] * 3600,
                              out["history_interval_s"], out["nest_history_interval_s"])
        total = p["total_bytes"] + row["download"]["bytes"] + row["download"]["preparation_bytes"]
        # Both the saved total and its picture-only update round to 0.1 GiB.
        assert abs(row["disk_gib"] - total / GIB) <= 0.1, (name, row["card_gb"])


@pytest.mark.parametrize("route", ["prepared", "experiment"])
def test_the_designer_and_the_engine_charge_the_same_pictures(design, tmp_path, route):
    plan = _page_plan(route, tmp_path)
    plan.run_options["keep_checkpoints"] = design.KEEP_CHECKPOINTS
    # The chain comes from the engine's resolve answer, as fit_domain reads it.
    fit = {"domains": [{"nx": 170, "ny": 170, "nz": 49}, {"nx": 336, "ny": 336, "nz": 49}],
           "restart_interval_s": 3600.0, "fetch": None,
           "chain": runplan._preparation_chain(plan, {})}
    run_seconds, root_s, nest_s = 6 * 3600.0, 3600.0, 900.0
    designed = design.projection(fit, run_seconds, root_s, nest_s)
    exp = SimpleNamespace(run_seconds=run_seconds, restart_interval_s=3600.0, domains=[
        SimpleNamespace(grid_id=i + 1, history_interval_s=root_s if i == 0 else nest_s,
                        run=SimpleNamespace(**d))
        for i, d in enumerate(fit["domains"])])
    engine = runplan._disk_projection(plan, exp, raw={}, data=None, fetch_arguments=None)
    assert designed["picture_bytes"] == engine["picture_bytes"] == sum(
        disk_budget.projected_picture_bytes(d.run.nx, d.run.ny, run_seconds,
                                           d.history_interval_s) for d in exp.domains)
    assert designed["total_bytes"] == engine["total_bytes"]


def _calls_without_render(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "projected_run_bytes":
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        if "render" not in keywords and None not in keywords:
            yield node.lineno


def test_every_caller_in_the_engine_and_its_tools_passes_the_render_choice():
    # A caller no test reaches only fails when someone runs it, as the
    # recipe designer did; this reads every call site instead.
    pattern = re.compile(r"\bprojected_run_bytes\s*\(")
    # Build output, vendored crates and the designer's scratch hold no caller.
    skip = {".git", "__pycache__", "node_modules", "target", "vendor", "work"}
    missing, callers = [], set()
    for folder in ("gpuwm", "tools"):
        for directory, subdirs, files in os.walk(ROOT / folder):
            subdirs[:] = [name for name in subdirs if name not in skip]
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = Path(directory) / file
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if not pattern.search(text):
                    continue
                name = path.relative_to(ROOT).as_posix()
                callers.add(name)
                missing.extend(f"{name}:{line}" for line in _calls_without_render(path))
    assert {"gpuwm/runplan.py", "tools/wiki_seed/design_recipes.py"} <= callers
    assert missing == []
