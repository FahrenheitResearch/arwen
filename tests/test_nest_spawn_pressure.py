"""The pressure spawn/retire trigger: a tropical cyclone can OPEN a nest.

Rotation and echo are maxima; a cyclone is a MINIMUM, and until this
lane the spawn and retire evaluators only knew how to look for a large
number.  A tropical cyclone therefore had a follow side and no birth
side: ``[relocation.follow]`` would ride a vortex all day, but nothing
could decide to open the nest in the first place.

The instrument rule applies throughout.  Every firing test has a control
that does NOT fire on the same code path, both pressure forms are proven
by their sign (a decoy of the WRONG extremum sits in every field), and
the two maximum triggers are pinned against values recorded from the
implementation this lane replaced -- if rerouting the evaluator through
``storm_tracking.locate_signal`` moved a uh or reflectivity placement by
so much as a cell, these fail.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.core.nest_lifecycle as nl
import gpuwm.core.nest_spawn as ns
from gpuwm.core.nest_lifecycle import RetireConfig, RetirementWatch
from gpuwm.core.nest_spawn import (SPAWN_KEYS, SpawnConfig, SpawnController,
                                   SpawnWatch, build_spawn_config)
from gpuwm.core.storm_tracking import (DEFAULT_LEVEL_HPA, NestFootprint,
                                       SEA_LEVEL_HPA)

PARENT_NY, PARENT_NX = 100, 120
RATIO, CHILD_N = 3, 45
PARENT_DX_M = 12000.0


# ---------------------------------------------------------------------------
# Synthetic fields, with a decoy of the wrong sign in every one
# ---------------------------------------------------------------------------

def _blob(cj, ci, amp, width=6.0):
    jj, ii = np.mgrid[0:PARENT_NY, 0:PARENT_NX]
    r2 = (jj - cj) ** 2 + (ii - ci) ** 2
    return amp * np.exp(-r2 / (2.0 * width ** 2))


def _bowl(cj, ci, depth=60.0, base=1500.0, width=6.0, ridge=None):
    """A geopotential-height low, with an optional RIDGE decoy.

    The ridge is what proves the sign: it is a larger number than
    anything in the bowl, so an evaluator that still looks for a maximum
    puts the nest on it.
    """
    plane = np.full((PARENT_NY, PARENT_NX), float(base))
    plane -= _blob(cj, ci, float(depth), width)
    if ridge is not None:
        rj, ri, amp = ridge
        plane += _blob(rj, ri, float(amp), width)
    return plane


def _declared(i=40, j=40, grid_id=2, dx_m=PARENT_DX_M):
    return NestFootprint(grid_id=grid_id, i_parent_start=i, j_parent_start=j,
                         child_nx=CHILD_N, child_ny=CHILD_N,
                         parent_grid_ratio=RATIO, parent_dx_m=dx_m)


class _PlaneSource:
    """Stands in for the plane builder, and RECORDS what it was asked for.

    The level is half the contract: a spawn watch that reads the right
    field on the wrong surface is exactly as wrong as one that reads the
    wrong field, and the only way to see the difference is to look at
    the argument.
    """

    def __init__(self, plane):
        self.plane = np.asarray(plane, dtype=np.float64)
        self.calls: list[dict] = []

    def __call__(self, state, field, *, uh_slot=None, level_hpa=None,
                 window=None):
        self.calls.append({"field": field, "level_hpa": level_hpa})
        return self.plane


def _patch_spawn_plane(monkeypatch, plane):
    source = _PlaneSource(plane)
    monkeypatch.setattr(ns, "signal_plane", source)
    return source


def _patch_retire_plane(monkeypatch, plane):
    source = _PlaneSource(plane)
    monkeypatch.setattr(nl, "_plane_from_state", source)
    return source


def _pressure_spawn(**over):
    base = dict(trigger="pressure", threshold=30.0, earliest_s=0.0,
                latest_s=3600.0)
    base.update(over)
    return SpawnConfig(**base)


def _watch(config, **over):
    return SpawnWatch(config, _declared(**over), keepout_cells=10)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

def test_pressure_joins_both_trigger_vocabularies():
    assert "pressure" in ns.SPAWN_TRIGGERS
    assert "pressure" in nl.RETIRE_TRIGGERS
    assert {"level_hpa", "radius_km"} <= SPAWN_KEYS
    assert "level_hpa" in nl.RETIRE_KEYS


def test_an_absent_level_is_the_850_hpa_surface_and_says_so():
    cfg = _pressure_spawn()
    assert cfg.level_hpa == (DEFAULT_LEVEL_HPA,) == (850.0,)
    record = cfg.to_json()
    assert record["level_hpa"] == [850.0]
    assert record["level_hpa_source"] == "default"
    assert record["threshold_units"] == "m above search-box minimum"
    named = _pressure_spawn(level_hpa=850.0)
    assert named.to_json()["level_hpa_source"] == "config"


def test_level_zero_is_the_mslp_form_with_an_absolute_ceiling():
    cfg = _pressure_spawn(level_hpa=0, threshold=1004.0)
    assert cfg.level_hpa is None
    record = cfg.to_json()
    assert record["level_hpa"] == [0.0]
    assert record["threshold_units"] == "hPa (absolute MSLP ceiling)"


def test_the_two_threshold_bands_are_disjoint_and_refuse_the_other():
    """An hPa ceiling under the defaulted surface is a units error, and a
    metres deficit under the sea-level form is the same error mirrored.
    Both refuse by name, and both messages carry the way out."""
    with pytest.raises(ValueError) as err:
        _pressure_spawn(threshold=1004.0)
    assert "level_hpa = 0" in str(err.value) and "850" in str(err.value)
    with pytest.raises(ValueError, match="1100"):
        _pressure_spawn(level_hpa=0, threshold=30.0)
    # ...and both ways out are ways out.
    assert _pressure_spawn(level_hpa=0, threshold=1004.0).level_hpa is None
    assert _pressure_spawn(threshold=30.0).level_hpa == (850.0,)


def test_a_non_pressure_trigger_refuses_the_pressure_keys():
    common = dict(threshold=60.0, earliest_s=0.0, latest_s=600.0)
    with pytest.raises(ValueError, match="refuses level_hpa"):
        SpawnConfig(trigger="uh", level_hpa=850.0, **common)
    with pytest.raises(ValueError, match="refuses level_hpa"):
        SpawnConfig(trigger="uh", level_hpa=0, **common)
    with pytest.raises(ValueError, match="refuses radius_km"):
        SpawnConfig(trigger="reflectivity", radius_km=50.0, **common)
    with pytest.raises(ValueError, match="refuses"):
        SpawnConfig(trigger="time", at_s=60.0, level_hpa=850.0)


def test_the_spawn_table_loads_the_pressure_form_and_still_refuses_typos():
    table = {"trigger": "pressure", "threshold": 30.0, "earliest_s": 0.0,
             "latest_s": 3600.0, "level_hpa": 700.0, "radius_km": 60.0}
    cfg = build_spawn_config(dict(table), "case.toml", grid_id=4)
    assert cfg.level_hpa == (700.0,) and cfg.radius_km == 60.0
    with pytest.raises(ValueError, match="level_hp"):
        build_spawn_config({**table, "level_hp": 700.0}, "case.toml",
                           grid_id=4)


def test_a_spawn_tracks_one_surface_because_it_claims_one_storm():
    with pytest.raises(ValueError, match="one surface"):
        _pressure_spawn(level_hpa=[850.0, 700.0])


# ---------------------------------------------------------------------------
# Firing: the isobaric form
# ---------------------------------------------------------------------------

def test_a_deepening_low_fires_a_pressure_spawn_on_its_centre(monkeypatch):
    """The bowl is at (i 70, j 55) and the RIDGE decoy is elsewhere and
    taller than anything in the field, so a maximum-seeking evaluator
    lands on the ridge."""
    plane = _bowl(55.0, 70.0, depth=60.0, ridge=(20.0, 25.0, 90.0))
    source = _patch_spawn_plane(monkeypatch, plane)
    watch = _watch(_pressure_spawn())
    event = watch.evaluate(SimpleNamespace(), 900.0)
    assert event is not None
    assert source.calls[0] == {"field": "pressure", "level_hpa": 850.0}
    receipt = event.receipt
    assert receipt["centroid_parent_ij"] == pytest.approx([70.0, 55.0],
                                                          abs=0.3)
    # The declared footprint spans (45 - 1)/3 parent cells, centred:
    # 70 - 44/6 + 1 = 63.67 -> 64, and 55 - 44/6 + 1 = 48.67 -> 49.
    assert event.position == (64, 49)
    assert receipt["extremum_kind"] == "minimum"
    assert receipt["extremum_units"] == "m"
    assert receipt["max_value"] == pytest.approx(1440.0, abs=1.0)


def test_a_flat_box_is_a_no_signal_not_a_spawn_on_its_own_centre(monkeypatch):
    """Under a RELATIVE threshold every cell clears `box minimum +
    threshold` when the box is flat, and the centroid of every cell in a
    box is the box's own centre -- which is a spawn on nothing.  The
    span test is what keeps that from firing."""
    _patch_spawn_plane(monkeypatch, np.full((PARENT_NY, PARENT_NX), 1500.0))
    watch = _watch(_pressure_spawn())
    assert watch.evaluate(SimpleNamespace(), 900.0) is None
    row = watch.receipts[-1]
    assert row["decision"] == "no-signal"
    assert row["signal_span"] == pytest.approx(0.0, abs=1e-9)


def test_a_shallow_low_does_not_clear_a_deeper_threshold(monkeypatch):
    """The control for the test above: the same code path, a real bowl,
    and a threshold the bowl cannot reach."""
    _patch_spawn_plane(monkeypatch, _bowl(55.0, 70.0, depth=12.0))
    assert _watch(_pressure_spawn(threshold=30.0)).evaluate(
        SimpleNamespace(), 900.0) is None
    assert _watch(_pressure_spawn(threshold=10.0)).evaluate(
        SimpleNamespace(), 900.0) is not None


# ---------------------------------------------------------------------------
# Firing: the sea-level form
# ---------------------------------------------------------------------------

def test_the_mslp_form_fires_on_an_absolute_hpa_ceiling(monkeypatch):
    plane = _bowl(55.0, 70.0, depth=25.0, base=1010.0,
                  ridge=(20.0, 25.0, 30.0))
    source = _patch_spawn_plane(monkeypatch, plane)
    watch = _watch(_pressure_spawn(level_hpa=0, threshold=1000.0))
    event = watch.evaluate(SimpleNamespace(), 900.0)
    assert event is not None
    assert source.calls[0] == {"field": "pressure", "level_hpa": None}
    assert event.position == (64, 49)
    assert event.receipt["extremum_units"] == "hPa"
    assert event.receipt["max_value"] == pytest.approx(985.0, abs=0.5)


def test_the_mslp_form_holds_while_the_field_stays_above_the_ceiling(
        monkeypatch):
    """A ceiling nothing reaches finds nothing -- which the RELATIVE form
    structurally cannot do, and is why the absolute form exists."""
    _patch_spawn_plane(monkeypatch, _bowl(55.0, 70.0, depth=4.0,
                                          base=1010.0))
    watch = _watch(_pressure_spawn(level_hpa=0, threshold=1000.0))
    assert watch.evaluate(SimpleNamespace(), 900.0) is None
    assert watch.receipts[-1]["decision"] == "no-signal"


# ---------------------------------------------------------------------------
# Two storms, two slots
# ---------------------------------------------------------------------------

def test_two_lows_fire_two_slots_and_the_exclusion_keeps_them_apart(
        monkeypatch):
    """The deeper low claims the first slot; the second slot must not be
    able to see it, or both nests are born on one storm."""
    plane = _bowl(55.0, 70.0, depth=60.0)
    plane -= _blob(20.0, 25.0, 40.0)
    _patch_spawn_plane(monkeypatch, plane)
    controller = SpawnController(
        {2: _watch(_pressure_spawn(), grid_id=2),
         3: _watch(_pressure_spawn(), grid_id=3)}, {2: 1, 3: 1})
    events = controller.evaluate_all({1: SimpleNamespace()}, 900.0)
    assert [e.grid_id for e in events] == [2, 3]
    assert events[0].position == (64, 49)
    assert events[1].position == (19, 14)
    assert events[1].receipt["excluded_active_footprints"]


# ---------------------------------------------------------------------------
# Retirement: a decaying cyclone is a RISING minimum
# ---------------------------------------------------------------------------

def _child_cfg(grid_id=4):
    return SimpleNamespace(grid_id=grid_id, i_parent_start=60,
                           j_parent_start=45, parent_grid_ratio=RATIO,
                           run=SimpleNamespace(nx=CHILD_N, ny=CHILD_N,
                                               dx=PARENT_DX_M / RATIO))


def test_a_filling_cyclone_retires_on_the_mslp_form(monkeypatch):
    deep = _bowl(55.0, 70.0, depth=25.0, base=1010.0)
    filled = _bowl(55.0, 70.0, depth=4.0, base=1010.0)
    holder = {"plane": deep}
    monkeypatch.setattr(nl, "_plane_from_state",
                        lambda *a, **k: holder["plane"])
    watch = RetirementWatch(RetireConfig(
        trigger="pressure", threshold=1004.0, level_hpa=0,
        sustained_s=600.0, min_lifetime_s=0.0))
    assert watch.evaluate(SimpleNamespace(), _child_cfg(), t=0.0,
                          born_t=0.0)["decision"] == "hold:signal"
    holder["plane"] = filled
    assert not watch.evaluate(SimpleNamespace(), _child_cfg(), t=300.0,
                              born_t=0.0)["retire"]
    row = watch.evaluate(SimpleNamespace(), _child_cfg(), t=900.0,
                         born_t=0.0)
    assert row["retire"] and row["extremum_kind"] == "minimum"
    # ...and a re-deepening low un-retires: the timer is continuous.
    holder["plane"] = deep
    assert not watch.evaluate(SimpleNamespace(), _child_cfg(), t=1200.0,
                              born_t=0.0)["retire"]
    assert watch.quiet_since is None


def test_a_decaying_vortex_retires_on_the_isobaric_depth(monkeypatch):
    """Under ``level_hpa`` the threshold is METRES, and what decays is
    the DEPTH of the height field under the footprint -- an absolute
    height would drift with the season and retire the nest on the
    airmass."""
    source = _patch_retire_plane(monkeypatch,
                                 _bowl(55.0, 70.0, depth=60.0))
    watch = RetirementWatch(RetireConfig(
        trigger="pressure", threshold=25.0, level_hpa=700.0,
        sustained_s=0.0, min_lifetime_s=0.0))
    row = watch.evaluate(SimpleNamespace(), _child_cfg(), t=600.0,
                         born_t=0.0)
    assert source.calls[0]["level_hpa"] == 700.0
    assert not row["retire"] and row["extremum_kind"] == "depth"
    flat = _patch_retire_plane(monkeypatch, _bowl(55.0, 70.0, depth=5.0))
    assert flat is not None
    assert watch.evaluate(SimpleNamespace(), _child_cfg(), t=1200.0,
                          born_t=0.0)["retire"]


def test_retire_refuses_the_pressure_keys_on_a_maximum_trigger():
    with pytest.raises(ValueError, match="refuses level_hpa"):
        RetireConfig(trigger="uh", threshold=60.0, level_hpa=850.0)
    with pytest.raises(ValueError, match="refuses level_hpa"):
        RetireConfig(trigger="time", at_s=600.0, level_hpa=0)


def test_the_retire_table_loads_the_pressure_form_and_refuses_a_string():
    cfg = nl.build_retire_config(
        {"trigger": "pressure", "threshold": 1004.0, "level_hpa": 0,
         "sustained_s": 3600.0, "min_lifetime_s": 7200.0},
        "case.toml", grid_id=4)
    assert cfg.level_hpa is None and cfg.level is None
    assert cfg.to_json()["threshold_units"] == "hPa (absolute MSLP ceiling)"
    with pytest.raises(ValueError, match="must be a number in hPa"):
        nl.build_retire_config(
            {"trigger": "pressure", "threshold": 30.0, "level_hpa": "850"},
            "case.toml", grid_id=4)


# ---------------------------------------------------------------------------
# The whole config door
# ---------------------------------------------------------------------------

TC_TOML = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 7200.0
restart_interval_s = 0.0

[shared]
nz = 8
ztop = 12000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 3600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
spawn = { trigger = "pressure", threshold = 30.0, level_hpa = 850.0, \
radius_km = 60.0, earliest_s = 0.0, latest_s = 5400.0 }
retire = { trigger = "pressure", threshold = 20.0, level_hpa = 850.0, \
sustained_s = 1800.0, min_lifetime_s = 1800.0 }
rearm = { max_firings = 3, cooldown_s = 1800.0 }
"""


def test_the_config_door_accepts_a_pressure_spawn_and_retire(tmp_path):
    from gpuwm.experiment import load_experiment

    path = tmp_path / "tc.toml"
    path.write_text(TC_TOML, encoding="utf-8")
    exp = load_experiment(path)
    child = exp.domain(2)
    assert child.spawn.trigger == "pressure"
    assert child.spawn.level_hpa == (850.0,) and child.spawn.radius_km == 60.0
    assert child.retire.trigger == "pressure"
    assert child.retire.level_hpa == (850.0,)
    assert child.rearm.max_firings == 3


def test_the_shipped_cyclone_config_is_the_shape_it_documents():
    """A capability with no shipped config is not reachable.  This is the
    front door for the pressure trigger, and it is loaded through the
    real loader rather than described in prose."""
    import pathlib

    from gpuwm.experiment import dormant_domain_ids, load_experiment

    path = (pathlib.Path(__file__).resolve().parents[1] / "configs"
            / "cyclone_nest_slots_12km.toml")
    exp = load_experiment(path)
    # THREE slots, which is the point: a basin has more than one storm.
    assert dormant_domain_ids(exp) == (2, 3, 4)
    for gid in (2, 3, 4):
        dc = exp.domain(gid)
        assert dc.spawn.trigger == "pressure"
        assert dc.spawn.level_hpa == (850.0,)
        assert dc.retire.trigger == "pressure"
        assert dc.retire.level_hpa == (850.0,)
        assert dc.rearm is not None
        # Per-domain follow, on the same surface the slot was born on.
        assert dc.follow.tracker.field == "pressure"
        assert dc.follow.tracker.level_hpa == (850.0,)
    # ...and the leg walk goes straight to the window rather than asking
    # once an hour until it opens.
    from gpuwm.core.spawn_runner import SpawnRunner
    from gpuwm.runtime import _spawn_leg_seconds, spawn_leg_boundary

    runner = SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None)
    earliest = float(exp.domain(2).spawn.earliest_s)
    assert runner.next_decision_time(0.0) == pytest.approx(earliest)
    assert spawn_leg_boundary(
        runner, 0.0, leg=_spawn_leg_seconds(exp),
        total=exp.run_seconds) == pytest.approx(earliest)


# ---------------------------------------------------------------------------
# The goldens: the maximum triggers did not move
# ---------------------------------------------------------------------------

def _uh_state(plane):
    class _Stub:
        def existing_scratch(self, slot):
            return plane if slot == "uh_spawn_window" else None
    return _Stub()


def _refl_state(plane):
    class _Stub:
        def existing_scratch(self, slot):
            return plane if slot == "refl_10cm" else None
    return _Stub()


def _uh_plane():
    return _blob(55.0, 70.0, 180.0, width=3.0) + _blob(20.0, 25.0, 90.0,
                                                       width=2.5)


def test_a_uh_spawn_lands_exactly_where_it_landed_before_the_reroute():
    """RECORDED from the implementation this lane replaced.  A pressure
    trigger is a new branch; rerouting uh through the same locate call
    must not move a single cell of the answer."""
    watch = _watch(SpawnConfig(trigger="uh", threshold=60.0,
                               earliest_s=600.0, latest_s=3000.0),
                   dx_m=None)
    event = watch.evaluate(_uh_state(_uh_plane()), 900.0)
    assert event is not None
    assert event.position == (64, 49)
    receipt = event.receipt
    assert receipt["cells_above_threshold"] == 61
    assert receipt["max_value"] == 180.0
    assert receipt["centroid_parent_ij"] == [70.0, 55.0]
    assert receipt["peak_parent_ij"] == [70, 55]
    assert receipt["local_window"] == [[47, 64], [62, 79]]
    assert receipt["search_box"] == [[0, 100], [0, 120]]
    assert receipt["clamped_to_keepout"] is False


def test_two_uh_slots_split_two_storms_exactly_as_they_did():
    state = _uh_state(_uh_plane())
    watches = {gid: _watch(SpawnConfig(trigger="uh", threshold=60.0,
                                       earliest_s=600.0, latest_s=3000.0),
                           grid_id=gid, dx_m=None)
               for gid in (2, 3)}
    events = SpawnController(watches, {2: 1, 3: 1}).evaluate_all(
        {1: state}, 900.0)
    assert [(e.grid_id, e.position) for e in events] == [
        (2, (64, 49)), (3, (19, 14))]
    assert [e.receipt["cells_above_threshold"] for e in events] == [61, 21]
    assert [e.receipt["max_value"] for e in events] == [180.0, 90.0]
    assert [e.receipt["centroid_parent_ij"] for e in events] == [
        [70.0, 55.0], [25.0, 20.0]]


def test_a_reflectivity_spawn_lands_exactly_where_it_landed_before():
    watch = _watch(SpawnConfig(trigger="reflectivity", threshold=45.0,
                               earliest_s=0.0, latest_s=3000.0), dx_m=None)
    event = watch.evaluate(_refl_state(_blob(30.0, 90.0, 62.0, width=4.0)),
                           600.0)
    assert event is not None and event.position == (84, 24)
    receipt = event.receipt
    assert receipt["cells_above_threshold"] == 37
    assert receipt["max_value"] == 62.0
    assert receipt["centroid_parent_ij"] == [90.0, 30.0]
    assert receipt["peak_parent_ij"] == [90, 30]
    assert receipt["local_window"] == [[22, 39], [82, 99]]


def test_a_maximum_trigger_no_signal_receipt_keeps_its_shape():
    watch = _watch(SpawnConfig(trigger="uh", threshold=1.0e6,
                               earliest_s=0.0, latest_s=3000.0), dx_m=None)
    assert watch.evaluate(_uh_state(_uh_plane()), 600.0) is None
    assert sorted(watch.receipts[-1]) == [
        "contract", "decision", "excluded_active_footprints", "field",
        "grid_id", "search_box", "search_box_source", "t", "threshold"]


def test_a_maximum_retire_receipt_keeps_its_shape(monkeypatch):
    class _Stub:
        def existing_scratch(self, slot):
            return np.full((PARENT_NY, PARENT_NX), 20.0)
    watch = RetirementWatch(RetireConfig(
        trigger="uh", threshold=60.0, sustained_s=900.0,
        min_lifetime_s=0.0))
    row = watch.evaluate(_Stub(), _child_cfg(), t=0.0, born_t=0.0)
    assert row["decision"] == "hold:quiet"
    assert row["max_value"] == 20.0
    assert row["field"] == "uh" and row["threshold"] == 60.0
    assert row["quiet_for_s"] == 0.0 and row["sustained_s"] == 900.0


def test_the_uh_and_reflectivity_keys_are_still_refused_by_name():
    assert SEA_LEVEL_HPA == 0.0
    with pytest.raises(ValueError, match="refuses at_s"):
        SpawnConfig(trigger="pressure", threshold=30.0, at_s=60.0)
    with pytest.raises(ValueError, match="requires a finite"):
        SpawnConfig(trigger="pressure", earliest_s=0.0, latest_s=60.0)
