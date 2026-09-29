"""Statics that differ beyond one-ulp rounding are refused."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_relocation import Placement, plan_relocation
from gpuwm.ingest.relocation_init import overlap_statics_mismatches


WITNESSES = json.loads((Path(__file__).parent / "fixtures" /
                       "relocation_static_identity.json").read_text())["fields"]


def _plan():
    return plan_relocation(
        placement_from=Placement(2, 4, 4),
        placement_to=Placement(2, 4, 4, 1),
        parent_grid_ratio=3, child_nx=6, child_ny=6)


@pytest.mark.parametrize("name", WITNESSES)
def test_measured_cross_library_climatology_is_refused(name):
    row = WITNESSES[name]
    prepared = np.array(row["prepared"])
    rebuilt = np.array(row["rebuilt"])
    assert not np.array_equal(prepared, rebuilt)
    before = prepared.tobytes(), rebuilt.tobytes()
    verdict = overlap_statics_mismatches(
        {name: prepared}, {name: rebuilt}, _plan())
    assert not verdict["pass"], verdict
    assert name in verdict["mismatched_fields"]
    assert before == (prepared.tobytes(), rebuilt.tobytes())


@pytest.mark.parametrize("name", WITNESSES)
def test_a_shifted_cell_is_refused(name):
    prepared = np.broadcast_to(np.arange(6., dtype=float), (6, 6)).copy()
    for offset in (0.002, 1., 100.):
        rebuilt = prepared.copy()
        rebuilt[2, 2] += offset
        verdict = overlap_statics_mismatches(
            {name: prepared}, {name: rebuilt}, _plan())
        assert not verdict["pass"], verdict
        assert verdict["mismatched_fields"] == {name: 1}


@pytest.mark.parametrize("name", ["HGT_M", "LANDMASK", "LANDUSEF", "LU_INDEX"])
def test_terrain_and_categories_keep_their_existing_bound(name):
    prepared = np.broadcast_to(np.arange(6., dtype=float), (6, 6)).copy()
    rebuilt = prepared.copy()
    rebuilt[2, 2] += 0.0005
    assert not overlap_statics_mismatches(
        {name: prepared}, {name: rebuilt}, _plan())["pass"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_climatology_is_refused(value):
    prepared = np.ones((6, 6))
    rebuilt = prepared.copy()
    rebuilt[2, 2] = value
    assert not overlap_statics_mismatches(
        {"GREENFRAC": prepared}, {"GREENFRAC": rebuilt}, _plan())["pass"]


def test_flat_climatology_keeps_the_one_ulp_bound():
    prepared = np.ones((6, 6))
    rebuilt = np.nextafter(np.nextafter(prepared, np.inf), np.inf)
    assert not overlap_statics_mismatches(
        {"GREENFRAC": prepared}, {"GREENFRAC": rebuilt}, _plan())["pass"]


def test_preparer_reports_only_the_strict_overlap_check(monkeypatch):
    from test_relocation_real_init import _preparer_fixture

    preparer, node, new_dc, initialized = _preparer_fixture(monkeypatch)
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())
    verdict = preparer.last_receipt["overlap_statics"]
    assert verdict["pass"] and verdict["within_one_ulp"] == {}
    assert "within_grid_tolerance" not in verdict
