"""Ozone on the offline child route, where there is no resident parent.

``gpuwm downscale`` derives a child's physics from its parent verbatim, and
the shipped desktop profile for an HRRR parent resolves to legacy RRTMG on
both streams with ``o3input = 2``, which is also the RunConfig default. That
pairing used to be refused on this route, so a default downscale of a default
forecast could not run at all.

WHAT WRF DOES, which is what settles it. Under ``o3input = 2`` the EM_CORE
build guards BOTH halves of the chain on ``id == 1``: ``oznini`` interpolates
the packaged climatology to a domain's own XLAT
(``phys/module_physics_init.F:2203-2212``) and
``ozn_time_int``/``ozn_p_int`` run on that domain's own columns
(``phys/module_radiation_driver.F:1801-1823``). Only a NEST is handed the
root's field, through the parent-to-child forcing stream the Registry
declares on the variable (``rdf=(p2c)``,
``Registry/Registry.EM_COMMON:1264``).

An offline child is not a resident nest. It is configured as a WRF root,
``specified = true`` and ``nested = false`` with lateral boundaries read from
a file, and the route stamps ``parent_id = 0``. So WRF's own answer for it is
the climatology on its own grid, and that is what this route now evaluates.
What the refusal was right about is the REPORTING: the field must not be
recorded as a root's. It is recorded as ``child-grid-climatology``.

CPU-side by construction: nothing here imports cupy at module scope, in a
fixture or in a non-``test_`` helper, so conftest's AST auto-marker leaves the
module unmarked and it runs on a machine with no card.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import offline_child_run
from gpuwm.config import RunConfig
from gpuwm.core import cam_ozone
from gpuwm.core.cam_ozone import (
    OZONE_ROUTINGS, ROUTING_CHILD_GRID_CLIMATOLOGY,
    ROUTING_PARENT_INTERPOLATED, ROUTING_ROOT_CLIMATOLOGY,
    ROUTING_WRAPPER_O3DATA, resolve_ozone_routing)

START = datetime(1974, 4, 3, 18)


def _legacy_child_config(**kwargs):
    """A child the default downscale of a default forecast would produce.

    Legacy RRTMG on both streams with the default ``o3input``, and no
    surface physics, so the route reaches its radiation construction
    without needing a child-grid surface source.
    """
    settings = dict(
        nx=6, ny=6, nz=8, dx=1000.0, dy=1000.0, ztop=9000.0, dt=5.0,
        run_seconds=300.0, hybrid_opt=2, etac=0.2, moist=True, mp_physics=8,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=1, ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant="rrtmg_legacy", o3input=2, use_mp_re=1,
        swrad_scat=1.0, sf_surface_physics=0, sf_sfclay_physics=0,
        bl_pbl_physics=0, cu_physics=0)
    settings.update(kwargs)
    return RunConfig(**settings)


def _initial(cfg):
    latitude = np.linspace(34.0, 36.0, cfg.ny)[:, None] * np.ones((1, cfg.nx))
    longitude = np.linspace(-98.0, -96.0, cfg.nx)[None, :] * np.ones((cfg.ny, 1))
    return SimpleNamespace(
        fields={"XLAT": latitude, "XLONG": longitude},
        receipt={"p_top": 10000.0}, valid_time=START)


def _capture(monkeypatch):
    """Run the route's physics attachment with both allocators replaced."""
    from gpuwm.core import physics, radiation_composition
    seen = {}

    def make_radiation(cfg, start_time, latitude, longitude, **kwargs):
        seen["radiation_kwargs"] = kwargs
        seen["latitude"] = np.asarray(latitude)
        return SimpleNamespace(ozone_routing=kwargs.get("ozone_routing"),
                               spectrum_adapters=None)

    def initialize_physics(child, cfg, **kwargs):
        seen["physics_kwargs"] = kwargs
        return SimpleNamespace(radiation_callable=kwargs.get("radiation"),
                               fields={})

    monkeypatch.setattr(radiation_composition, "make_radiation",
                        make_radiation)
    monkeypatch.setattr(physics, "initialize_physics", initialize_physics)
    return seen


# ---------------------------------------------------------------------------
# The route, called
# ---------------------------------------------------------------------------

def test_the_default_pairing_builds_radiation_instead_of_refusing(monkeypatch):
    """The reproduced defect: this raised for every default downscale."""
    cfg = _legacy_child_config()
    seen = _capture(monkeypatch)
    driver = offline_child_run._initialize_child_physics(
        None, cfg, _initial(cfg), None, START)
    assert driver is not None
    assert seen["physics_kwargs"]["radiation"] is not None


def test_the_route_declares_the_climatology_is_on_the_child_grid(monkeypatch):
    cfg = _legacy_child_config()
    seen = _capture(monkeypatch)
    offline_child_run._initialize_child_physics(
        None, cfg, _initial(cfg), None, START)
    assert (seen["radiation_kwargs"]["ozone_routing"]
            == ROUTING_CHILD_GRID_CLIMATOLOGY)
    assert "ozone_parent" not in seen["radiation_kwargs"]


def test_the_climatology_is_evaluated_on_the_childs_own_latitudes(monkeypatch):
    """Not the parent's: the grid handed to the adapter is the child's."""
    cfg = _legacy_child_config()
    seen = _capture(monkeypatch)
    initial = _initial(cfg)
    offline_child_run._initialize_child_physics(None, cfg, initial, None, START)
    assert np.array_equal(seen["latitude"], initial.fields["XLAT"])


def test_o3input_zero_still_declares_the_wrapper_profile(monkeypatch):
    """The exit the refusal used to name stays exactly where it was."""
    cfg = _legacy_child_config(o3input=0, use_mp_re=0)
    seen = _capture(monkeypatch)
    offline_child_run._initialize_child_physics(
        None, cfg, _initial(cfg), None, START)
    assert seen["radiation_kwargs"]["ozone_routing"] is None


def test_the_refusal_sentence_is_gone_from_the_route():
    """Retired with the defect, not left as unreachable text a grep finds."""
    import inspect

    source = inspect.getsource(offline_child_run._initialize_child_physics)
    assert "needs ozone interpolated from the parent domain" not in source


# ---------------------------------------------------------------------------
# The report says where the ozone came from
# ---------------------------------------------------------------------------

def test_the_report_reads_the_routing_off_the_adapter():
    adapter = type("RRTMGLegacyRadiation", (), {
        "__module__": "gpuwm.core.rrtmg_legacy",
        "ozone_routing": ROUTING_CHILD_GRID_CLIMATOLOGY,
        "spectrum_adapters": None})()
    driver = SimpleNamespace(radiation_callable=adapter)
    assert (offline_child_run._child_ozone_routing(driver)
            == ROUTING_CHILD_GRID_CLIMATOLOGY)


def test_a_child_without_legacy_radiation_reports_no_routing():
    """The third state: not a missing key, and not a guessed name."""
    assert offline_child_run._child_ozone_routing(
        SimpleNamespace(radiation_callable=None)) is None
    assert offline_child_run._child_ozone_routing(
        SimpleNamespace(radiation_callable=SimpleNamespace(
            spectrum_adapters=None))) is None


def test_the_report_carries_the_routing_key():
    """The key is written by the one report builder, not by a caller."""
    import inspect

    source = inspect.getsource(offline_child_run._run)
    assert '"child_ozone_routing": ozone_routing,' in source


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("o3input,has_parent,expected", [
    (0, False, ROUTING_WRAPPER_O3DATA),
    (2, False, ROUTING_ROOT_CLIMATOLOGY),
    (2, True, ROUTING_PARENT_INTERPOLATED),
])
def test_an_undeclared_routing_is_derived_from_the_construction(
        o3input, has_parent, expected):
    assert resolve_ozone_routing(
        None, o3input=o3input, has_parent=has_parent) == expected


def test_the_offline_name_is_admitted_where_the_arithmetic_is_the_same():
    assert resolve_ozone_routing(
        ROUTING_CHILD_GRID_CLIMATOLOGY, o3input=2,
        has_parent=False) == ROUTING_CHILD_GRID_CLIMATOLOGY


def test_a_routing_that_contradicts_the_construction_is_refused():
    """A wrong name sends a reader after a parent field that never existed."""
    with pytest.raises(ValueError, match="contradicts how this adapter"):
        resolve_ozone_routing(ROUTING_PARENT_INTERPOLATED, o3input=2,
                              has_parent=False)
    with pytest.raises(ValueError, match="contradicts how this adapter"):
        resolve_ozone_routing(ROUTING_CHILD_GRID_CLIMATOLOGY, o3input=2,
                              has_parent=True)
    with pytest.raises(ValueError, match="contradicts how this adapter"):
        resolve_ozone_routing(ROUTING_ROOT_CLIMATOLOGY, o3input=0,
                              has_parent=False)


def test_the_refusal_names_a_way_out_a_caller_can_take():
    with pytest.raises(ValueError) as error:
        resolve_ozone_routing(ROUTING_PARENT_INTERPOLATED, o3input=2,
                              has_parent=False)
    assert "ozone_routing=None" in str(error.value)


def test_an_unknown_routing_name_is_refused_with_the_vocabulary():
    with pytest.raises(ValueError, match="unknown ozone routing"):
        resolve_ozone_routing("from-the-parent", o3input=2, has_parent=False)


def test_the_producer_modes_are_spelled_from_the_vocabulary():
    """One owner for the names, so a report and a producer cannot drift."""
    assert ROUTING_ROOT_CLIMATOLOGY in OZONE_ROUTINGS
    owner = cam_ozone.CamOzoneState(
        START, np.full((2, 3), 35.0), np.full((2, 3), -97.0),
        ROUTING_ROOT_CLIMATOLOGY)
    assert owner.mode == ROUTING_ROOT_CLIMATOLOGY
    with pytest.raises(ValueError, match="unknown CAM ozone producer mode"):
        cam_ozone.CamOzoneState(
            START, np.full((2, 3), 35.0), np.full((2, 3), -97.0),
            ROUTING_CHILD_GRID_CLIMATOLOGY)


# ---------------------------------------------------------------------------
# The field itself: WRF's pipeline, on the child's grid
# ---------------------------------------------------------------------------

def test_the_child_grid_field_is_the_wrf_pipeline_evaluated_directly():
    """The chain the child runs is oznini plus ozn_time_int plus ozn_p_int.

    Evaluated through the CAM producer on a child-shaped grid and compared
    against ``gpuwm.ingest.wrf_ozone`` called by hand on the same columns,
    so the producer's field on a child grid is pinned to the ported
    pipeline rather than to whatever the producer happens to do.  The
    field a downscaled run actually radiates with comes from the legacy
    adapter's own root branch (``RRTMGLegacyRadiation``), which this CPU
    test does not construct; that branch's retained field was captured on
    the card against the same direct evaluation and matched it exactly
    (``proof/child-ozone-276/``), and pinning it through the adapter
    itself needs a card.
    """
    from gpuwm.ingest import wrf_ozone

    nz, ny, nx = 8, 5, 4
    latitude = np.linspace(34.0, 36.0, ny)[:, None] * np.ones((1, nx))
    longitude = np.full((ny, nx), -97.0)
    pressure = np.broadcast_to(
        np.linspace(95000.0, 12000.0, nz, dtype=np.float32)[:, None, None],
        (nz, ny, nx)).copy()

    owner = cam_ozone.CamOzoneState(START, latitude, longitude,
                                    ROUTING_ROOT_CLIMATOLOGY)
    produced = owner.evaluate(pressure, 0.0)

    climatology = wrf_ozone.load_ozone_climatology()
    julday = START.timetuple().tm_yday
    julian = np.float32((julday - 1) + START.hour / 24.0)
    columns = np.ascontiguousarray(
        pressure.transpose(1, 2, 0).reshape(-1, nz))
    latitudes = wrf_ozone.interp_ozone_to_latitudes(
        np.asarray(latitude, np.float32).reshape(-1), climatology)
    timed = wrf_ozone.ozn_time_int(julday, julian, latitudes)
    expected = wrf_ozone.ozn_p_int(columns, climatology.plev, timed)
    expected = np.ascontiguousarray(
        expected.reshape(ny, nx, nz).transpose(2, 0, 1))

    assert np.array_equal(produced, expected)
    assert produced.min() > 0.0


def test_the_field_follows_the_childs_own_latitudes():
    """A column at 34N and one at 46N do not get the same profile."""
    ny, nx, nz = 2, 1, 8
    latitude = np.asarray([[34.0], [46.0]], dtype=np.float64)
    pressure = np.broadcast_to(
        np.linspace(95000.0, 12000.0, nz, dtype=np.float32)[:, None, None],
        (nz, ny, nx)).copy()
    owner = cam_ozone.CamOzoneState(
        START, latitude, np.full((ny, nx), -97.0), ROUTING_ROOT_CLIMATOLOGY)
    field = owner.evaluate(pressure, 0.0)
    assert not np.allclose(field[:, 0, 0], field[:, 1, 0])
