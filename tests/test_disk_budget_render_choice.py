"""Disk admission charges for pictures only when the run's route draws them.

Breakage these prevent: a run with ``render_products = "none"``, or an
authored experiment config whose route draws nothing by default, was
charged 150 MB of pictures per history frame and refused for a disk it
would never have filled (a 24 hour one-domain run was projected at 3.6 GiB
of which 3.5 GiB were pictures).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm import disk_budget, runplan


def _projection(tmp_path, *, products, route="experiment", intent=None):
    plan = SimpleNamespace(
        run_options={"render_products": products, "keep_checkpoints": 1},
        route=route, config_intent=intent, run_dir=tmp_path)
    exp = SimpleNamespace(
        run_seconds=24 * 3600.0, restart_interval_s=3600.0,
        domains=[SimpleNamespace(grid_id=1, history_interval_s=3600.0,
                                 run=SimpleNamespace(nx=32, ny=32, nz=49))])
    return runplan._disk_projection(plan, exp, raw={}, data=None,
                                    fetch_arguments=None)


@pytest.mark.parametrize("route", ["experiment", "prepared"])
@pytest.mark.parametrize("none", ["none", " None "])
def test_a_run_that_draws_no_pictures_is_not_charged_for_them(tmp_path, route, none):
    drawn = _projection(tmp_path, products="all", route=route)
    skipped = _projection(tmp_path, products=none, route=route)
    assert skipped["picture_bytes"] == 0
    assert all(row["picture_bytes"] == 0 for row in skipped["domains"])
    assert drawn["total_bytes"] - skipped["total_bytes"] == drawn["picture_bytes"]
    # Free space that holds everything but the pictures: the run that
    # draws none fits, the run that draws them is still refused.
    free = drawn["total_bytes"] - drawn["picture_bytes"] + 1
    assert disk_budget.disk_refusal(skipped, free) is None
    assert disk_budget.disk_refusal(drawn, free) is not None


def test_an_authored_experiment_config_draws_nothing_by_default(tmp_path):
    assert _projection(tmp_path, products=None)["picture_bytes"] == 0


@pytest.mark.parametrize("route,intent", [("experiment", {}), ("prepared", None)])
def test_the_defaults_that_draw_keep_the_picture_charge(tmp_path, route, intent):
    result = _projection(tmp_path, products=None, route=route, intent=intent)
    assert result["picture_bytes"] == disk_budget.projected_picture_bytes(32, 32, 24 * 3600, 3600)


@pytest.mark.parametrize("products,route,intent,draws", [
    (None, "experiment", None, False),
    (None, "experiment", {}, True),
    ("none", "experiment", {}, False),
    ("t2", "experiment", None, True),
    (None, "prepared", None, True),
    ("none", "prepared", None, False),
])
def test_the_charge_and_the_experiment_route_answer_from_one_rule(
        products, route, intent, draws):
    plan = SimpleNamespace(run_options={"render_products": products},
                           route=route, config_intent=intent)
    assert runplan._plan_draws_pictures(plan) is draws
    if route == "experiment":
        # What the route itself renders with (None or "none" draws nothing).
        rendered = runplan._experiment_render_products(plan)
        assert (rendered is not None
                and str(rendered).strip().lower() != "none") is draws


def test_the_projection_cannot_be_asked_without_the_render_choice():
    exp = SimpleNamespace(run_seconds=3600.0, restart_interval_s=0.0, domains=[])
    with pytest.raises(TypeError):
        disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None, chain=None)
