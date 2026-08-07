"""CPU contracts for spawn-at-trigger nests: config, reservation, trigger.

The instrument rule applies throughout (the storm-tracking test's
posture): triggers are tested against synthetic features with KNOWN
positions in both directions, every suppression has a control that
fires without it, and the reservation claims are measured against the
estimator rather than asserted about it.
"""

from __future__ import annotations

import textwrap
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_spawn import (SPAWN_KEYS, SpawnConfig, SpawnController,
                                   SpawnEvent, SpawnRefusal, SpawnWatch,
                                   build_spawn_config)
from gpuwm.core.storm_tracking import FollowConfig, NestFootprint
from gpuwm.experiment import (active_experiment, dormant_domain_ids,
                              load_experiment, pre_spawn_experiment,
                              refuse_unrouted_spawn,
                              validate_spawn_placement)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PARENT_NY, PARENT_NX = 100, 120
RATIO = 3
CHILD_N = 45  # spans 15 parent cells (loader span nx // ratio)

#: Loader fixture: root 100x80 at 12 km / 60 s; the {d02} hole carries
#: the spawn table under test (the test_experiment.py BASE idiom).
BASE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 3600.0
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
{d02}
"""

FIELD_SPAWN = ('spawn = { trigger = "uh", threshold = 60.0, '
               'earliest_s = 600.0, latest_s = 3000.0 }')
TIME_SPAWN = 'spawn = { trigger = "time", at_s = 120.0 }'


def _write(tmp_path, d02="", text=None):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(text if text is not None
                                    else BASE.format(d02=d02)))
    return path


class StubState:
    """LOUD stub of the one surface the watch reads: named scratch planes."""

    def __init__(self, **planes):
        self._planes = {k: np.asarray(v) for k, v in planes.items()}

    def existing_scratch(self, slot):
        return self._planes.get(slot)

    def __getattr__(self, name):
        raise AttributeError(
            f"StubState exposes only existing_scratch(); {name!r} was read")


def _bump(cj, ci, amp, sigma=2.0, shape=(PARENT_NY, PARENT_NX)):
    jj, ii = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-(((ii - ci) ** 2 + (jj - cj) ** 2)
                          / (2.0 * sigma ** 2)))


def _state(uh=None):
    planes = np.zeros((PARENT_NY, PARENT_NX)) if uh is None else uh
    # The SPAWN consumer's own window (gpuwm.core.uh_diag), not the
    # history-reset diagnostic and not the relocation runner's slot.
    return StubState(uh_spawn_window=planes)


def _declared(i=40, j=40, grid_id=2):
    return NestFootprint(grid_id=grid_id, i_parent_start=i,
                         j_parent_start=j, child_nx=CHILD_N,
                         child_ny=CHILD_N, parent_grid_ratio=RATIO)


def _spawn(**over):
    base = dict(trigger="uh", threshold=50.0, earliest_s=0.0,
                latest_s=100000.0)
    base.update(over)
    return SpawnConfig(**base)


def _watch(config=None, declared=None, keepout=10, follow=None):
    return SpawnWatch(config or _spawn(), declared or _declared(),
                      keepout_cells=keepout, follow=follow)


def _expected_start(center, span=(CHILD_N - 1) / RATIO):
    """The leg-1 centering convention inverted: start = c - span/2 + 1."""
    return int(np.floor(center - span / 2.0 + 1.0 + 0.5))


# ---------------------------------------------------------------------------
# SpawnConfig: governance
# ---------------------------------------------------------------------------

def test_spawn_config_echoes_every_value():
    cfg = _spawn(search_box=(10, 12, 90, 80))
    document = cfg.to_json()
    assert document["trigger"] == "uh"
    assert document["threshold"] == 50.0
    assert document["earliest_s"] == 0.0
    assert document["latest_s"] == 100000.0
    assert document["search_box"] == [10, 12, 90, 80]
    timed = SpawnConfig(trigger="time", at_s=120.0).to_json()
    assert timed == {"contract": "gpuwm-nest-spawn.v1",
                     "trigger": "time", "at_s": 120.0}


def test_trigger_vocabulary_is_closed():
    with pytest.raises(ValueError, match="must be one of"):
        SpawnConfig(trigger="vorticity", threshold=1.0,
                    earliest_s=0.0, latest_s=1.0)


def test_time_trigger_requires_at_s_and_refuses_field_keys():
    with pytest.raises(ValueError, match="requires at_s"):
        SpawnConfig(trigger="time")
    with pytest.raises(ValueError, match="refuses"):
        SpawnConfig(trigger="time", at_s=60.0, threshold=50.0)


def test_field_trigger_requires_threshold_and_window():
    with pytest.raises(ValueError, match="threshold"):
        SpawnConfig(trigger="uh", earliest_s=0.0, latest_s=10.0)
    with pytest.raises(ValueError, match="earliest_s"):
        SpawnConfig(trigger="uh", threshold=50.0, latest_s=10.0)
    with pytest.raises(ValueError, match="latest_s"):
        SpawnConfig(trigger="reflectivity", threshold=40.0, earliest_s=0.0)
    with pytest.raises(ValueError, match="empty window"):
        SpawnConfig(trigger="uh", threshold=50.0, earliest_s=10.0,
                    latest_s=10.0)
    with pytest.raises(ValueError, match="refuses at_s"):
        SpawnConfig(trigger="uh", threshold=50.0, earliest_s=0.0,
                    latest_s=10.0, at_s=5.0)


def test_search_box_must_be_an_ordered_one_based_box():
    with pytest.raises(ValueError, match="ordered 1-based"):
        _spawn(search_box=(0, 1, 10, 10))
    with pytest.raises(ValueError, match="ordered 1-based"):
        _spawn(search_box=(20, 1, 10, 10))


def test_build_spawn_config_refuses_unknown_keys_with_a_suggestion():
    with pytest.raises(ValueError, match="treshold.*did you mean"):
        build_spawn_config({"trigger": "uh", "treshold": 50.0},
                           "test.toml", grid_id=2)
    assert "threshold" in SPAWN_KEYS


# ---------------------------------------------------------------------------
# Loader surface
# ---------------------------------------------------------------------------

def test_field_spawn_toml_loads_and_binds(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    spawn = exp.domain(2).spawn
    assert spawn is not None and spawn.trigger == "uh"
    assert spawn.threshold == 60.0
    assert dormant_domain_ids(exp) == (2,)


def test_time_spawn_toml_loads(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=TIME_SPAWN))
    assert exp.domain(2).spawn.at_s == 120.0


def test_spawn_on_the_root_refuses(tmp_path):
    text = BASE.format(d02="").replace(
        "time_step = 60", "time_step = 60\n"
        'spawn = { trigger = "time", at_s = 120.0 }')
    with pytest.raises(ValueError, match="root.*no parent to be placed"):
        load_experiment(_write(tmp_path, text=text))


def test_spawn_beside_start_time_refuses(tmp_path):
    d02 = TIME_SPAWN + "\nstart_time = 1974-04-03T12:30:00"
    with pytest.raises(ValueError, match="activation time belongs to its "
                                         "trigger"):
        load_experiment(_write(tmp_path, d02=d02))


def test_at_s_off_the_parent_step_lattice_refuses(tmp_path):
    with pytest.raises(ValueError, match="whole number of parent"):
        load_experiment(_write(
            tmp_path, d02='spawn = { trigger = "time", at_s = 61.0 }'))


def test_at_s_past_the_run_end_refuses(tmp_path):
    with pytest.raises(ValueError, match="never spawn"):
        load_experiment(_write(
            tmp_path, d02='spawn = { trigger = "time", at_s = 3600.0 }'))


def test_window_opening_past_the_run_end_refuses(tmp_path):
    d02 = ('spawn = { trigger = "uh", threshold = 60.0, '
           'earliest_s = 3600.0, latest_s = 7200.0 }')
    with pytest.raises(ValueError, match="window can never open"):
        load_experiment(_write(tmp_path, d02=d02))


def test_a_child_under_a_dormant_parent_refuses(tmp_path):
    text = BASE.format(d02=FIELD_SPAWN) + textwrap.dedent("""
        [[domain]]
        grid_id = 3
        parent_id = 2
        i_parent_start = 20
        j_parent_start = 20
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        e_we = 31
        e_sn = 31
        history_interval_s = 900.0
        """)
    with pytest.raises(ValueError, match="cascading activation"):
        load_experiment(_write(tmp_path, text=text))


def test_two_dormant_nests_load_as_first_class(tmp_path):
    """Multiple dormant [[domain]] blocks, each with its own trigger."""
    text = BASE.format(d02=FIELD_SPAWN) + textwrap.dedent("""
        [[domain]]
        grid_id = 3
        parent_id = 1
        i_parent_start = 62
        j_parent_start = 42
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        e_we = 61
        e_sn = 61
        history_interval_s = 900.0
        spawn = { trigger = "reflectivity", threshold = 45.0, earliest_s = 0.0, latest_s = 3000.0 }
        """)
    exp = load_experiment(_write(tmp_path, text=text))
    assert dormant_domain_ids(exp) == (2, 3)
    assert exp.domain(2).spawn.trigger == "uh"
    assert exp.domain(3).spawn.trigger == "reflectivity"


# ---------------------------------------------------------------------------
# Active-tree views (the leg runner's schedule-surgery seam)
# ---------------------------------------------------------------------------

def test_pre_spawn_view_removes_dormant_and_is_identity_without_any(
        tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    pre = pre_spawn_experiment(exp)
    assert [dc.grid_id for dc in pre.domains] == [1]
    plain = load_experiment(_write(tmp_path, d02=""))
    assert pre_spawn_experiment(plain) is plain


def test_active_experiment_places_the_fired_nest_and_keeps_spawn(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    act = active_experiment(exp, {2: (55, 33)})
    d02 = act.domain(2)
    assert (d02.i_parent_start, d02.j_parent_start) == (55, 33)
    # The declaration is KEPT: the activated identity binds the spawn.
    assert d02.spawn is not None


def test_active_experiment_refuses_an_undeclared_grid(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    with pytest.raises(ValueError, match="not declared dormant"):
        active_experiment(exp, {3: (55, 33)})


def test_spawned_placement_re_passes_the_clearance_rule(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    with pytest.raises(ValueError, match="parent-row clearance"):
        active_experiment(exp, {2: (3, 30)})
    # Control: the loader-admitted declared placement passes.
    validate_spawn_placement(exp, 2, 40, 30)


def test_refuse_unrouted_spawn_fires_only_with_a_dormant_nest(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    with pytest.raises(ValueError, match="does not implement "
                                         "spawn-triggered"):
        refuse_unrouted_spawn(exp, "prepared single-domain")
    refuse_unrouted_spawn(
        load_experiment(_write(tmp_path, d02="")), "prepared single-domain")


# ---------------------------------------------------------------------------
# Identity: absent spawn is byte-inert, a declared spawn binds
# ---------------------------------------------------------------------------

def test_absent_spawn_is_outside_the_restart_identity(tmp_path):
    from gpuwm.core.model import restart_identity_payload

    exp = load_experiment(_write(tmp_path, d02=""))
    payload = restart_identity_payload(exp)
    for domain in payload["domains"]:
        assert "spawn" not in domain


def test_a_declared_spawn_binds_the_identity_and_fingerprint(tmp_path):
    from gpuwm.core.model import (experiment_fingerprint,
                                  restart_identity_payload)

    plain = load_experiment(_write(tmp_path, d02=""))
    dormant = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    assert (restart_identity_payload(plain)
            != restart_identity_payload(dormant))
    catalog = SimpleNamespace(run_provenance={})
    assert (experiment_fingerprint(plain, catalog)
            != experiment_fingerprint(dormant, catalog))


def test_prepared_cache_tolerates_absent_spawn_and_binds_a_real_one(
        tmp_path):
    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, prepared_domain_config_identity,
        undelayed_identity_defaults)

    exp = load_experiment(_write(tmp_path, d02=""))
    live = prepared_domain_config_identity(exp.domain(2))
    cached = {key: value for key, value in live.items() if key != "spawn"}
    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=undelayed_identity_defaults(exp))
    assert "spawn" in tolerated and not differing
    # A really-dormant live domain is refused against the old header.
    dormant = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    live_dormant = prepared_domain_config_identity(dormant.domain(2))
    _tolerated, differing = compare_prepared_domain_config(
        cached, live_dormant, not_in_use=undelayed_identity_defaults(exp))
    assert "spawn" in differing


# ---------------------------------------------------------------------------
# Reservation accounting (decision 1, measured)
# ---------------------------------------------------------------------------

def test_dormant_nest_is_priced_exactly_like_a_live_one(tmp_path):
    from gpuwm.core.preflight import estimate_experiment

    live = load_experiment(_write(tmp_path, d02=""))
    dormant = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    est_live = estimate_experiment(live)
    est_dormant = estimate_experiment(dormant)
    # The spawn declaration changes NOTHING about the plan: reservation
    # is "as if it existed", to the byte.
    assert est_dormant.resident_bytes == est_live.resident_bytes
    assert est_dormant.alloc_estimate_bytes == est_live.alloc_estimate_bytes


def test_reservation_exceeds_the_pre_spawn_plan_and_sums_over_nests(
        tmp_path):
    from gpuwm.core.preflight import estimate_experiment

    one = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    two_text = BASE.format(d02=FIELD_SPAWN) + textwrap.dedent("""
        [[domain]]
        grid_id = 3
        parent_id = 1
        i_parent_start = 62
        j_parent_start = 42
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        e_we = 61
        e_sn = 61
        history_interval_s = 900.0
        spawn = { trigger = "reflectivity", threshold = 45.0, earliest_s = 0.0, latest_s = 3000.0 }
        """)
    two = load_experiment(_write(tmp_path, text=two_text))
    root_only = estimate_experiment(pre_spawn_experiment(one))
    with_one = estimate_experiment(one)
    with_two = estimate_experiment(two)
    assert with_one.alloc_estimate_bytes > root_only.alloc_estimate_bytes
    assert with_two.alloc_estimate_bytes > with_one.alloc_estimate_bytes


def test_preflight_refuses_on_the_reservation_sum(tmp_path):
    """A budget that fits the pre-spawn tree but not the reservation must
    refuse: the dormant nest is part of the plan, not a future problem."""
    from gpuwm.core.preflight import estimate_experiment, estimate_phases

    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    pre = estimate_experiment(pre_spawn_experiment(exp))
    phases = estimate_phases(exp, source=None)
    budget = (pre.peak_envelope_bytes + phases.peak_envelope_bytes) // 2
    assert pre.peak_envelope_bytes <= budget  # sanity: pre-spawn fits
    assert not phases.fits(budget)
    assert "EXCEEDS" in phases.verdict(budget)


def test_check_advisories_name_each_dormant_reservation(tmp_path):
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    lines = [line for line in check_advisories(exp)
             if "DORMANT spawn-triggered" in line]
    assert len(lines) == 1
    assert "d02" in lines[0]
    assert "GiB" in lines[0]
    assert "never fires" in lines[0]
    plain = load_experiment(_write(tmp_path, d02=""))
    assert not [line for line in check_advisories(plain)
                if "DORMANT" in line]


# ---------------------------------------------------------------------------
# Sibling ground truth: two children of one parent are supported
# ---------------------------------------------------------------------------

def test_two_children_of_one_parent_are_admitted_and_scheduled(tmp_path):
    """The truth two spawned siblings will land on: the loader admits
    two children of one root and the schedule gives each its FORCE and
    ratio-3 substeps.  Spawn builds on supported ground."""
    from gpuwm.core.clock import build_schedule, resolve_clock

    text = BASE.format(d02="") + textwrap.dedent("""
        [[domain]]
        grid_id = 3
        parent_id = 1
        i_parent_start = 62
        j_parent_start = 42
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        e_we = 61
        e_sn = 61
        history_interval_s = 900.0
        """)
    exp = load_experiment(_write(tmp_path, text=text))
    assert [dc.grid_id for dc in exp.children_of(1)] == [2, 3]
    schedule = build_schedule(exp, resolve_clock(exp, lbc_interval_s=3600))
    counts: dict[tuple, int] = {}
    for op in schedule.interior_period:
        key = (op.kind, op.grid_id)
        counts[key] = counts.get(key, 0) + 1
    assert counts[("STEP", 2)] == 3 and counts[("STEP", 3)] == 3
    assert counts[("FORCE", 2)] == 1 and counts[("FORCE", 3)] == 1


# ---------------------------------------------------------------------------
# Trigger firing on synthetic parent fields
# ---------------------------------------------------------------------------

def test_uh_trigger_fires_on_a_known_bump_and_centers_the_footprint():
    watch = _watch()
    event = watch.evaluate(_state(_bump(50.0, 60.0, 200.0)), 0.0)
    assert isinstance(event, SpawnEvent)
    assert event.i_parent_start == _expected_start(60.0)
    assert event.j_parent_start == _expected_start(50.0)
    assert event.receipt["decision"] == "fired"
    assert event.receipt["placement_source"] == "centroid"
    assert watch.fired


def test_east_and_west_storms_give_mirrored_placements():
    """The instrument rule: a sign flip must move the answer."""
    east = _watch().evaluate(_state(_bump(50.0, 80.0, 200.0)), 0.0)
    west = _watch().evaluate(_state(_bump(50.0, 40.0, 200.0)), 0.0)
    assert east.i_parent_start == _expected_start(80.0)
    assert west.i_parent_start == _expected_start(40.0)
    assert east.i_parent_start - west.i_parent_start == 40


def test_no_signal_below_threshold():
    watch = _watch()
    assert watch.evaluate(_state(_bump(50.0, 60.0, 30.0)), 0.0) is None
    assert watch.receipts[-1]["decision"] == "no-signal"
    assert not watch.fired


def test_trigger_waits_before_earliest_and_fires_inside_the_window():
    watch = _watch(_spawn(earliest_s=600.0, latest_s=1200.0))
    plane = _bump(50.0, 60.0, 200.0)
    assert watch.evaluate(_state(plane), 0.0) is None
    assert watch.receipts[-1]["decision"] == "waiting:before-window"
    # Control: the same state inside the window fires.
    assert watch.evaluate(_state(plane), 600.0) is not None


def test_window_closes_after_latest_and_never_reopens():
    watch = _watch(_spawn(earliest_s=0.0, latest_s=100.0))
    plane = _bump(50.0, 60.0, 200.0)
    assert watch.evaluate(_state(plane), 101.0) is None
    assert watch.closed
    assert watch.receipts[-1]["decision"] == "window-closed"
    before = len(watch.receipts)
    assert watch.evaluate(_state(plane), 200.0) is None
    assert len(watch.receipts) == before  # closed is silent thereafter
    # Control: the identical watch inside the window fires.
    assert _watch(_spawn(earliest_s=0.0, latest_s=100.0)).evaluate(
        _state(plane), 100.0) is not None


def test_manual_time_trigger_fires_at_the_declared_placement():
    watch = _watch(SpawnConfig(trigger="time", at_s=120.0),
                   _declared(i=40, j=40))
    assert watch.evaluate(None, 60.0) is None
    assert watch.receipts[-1]["decision"] == "waiting:before-at_s"
    event = watch.evaluate(None, 120.0)
    assert event.position == (40, 40)
    assert event.receipt["placement_source"] == "declared"
    assert watch.evaluate(None, 180.0) is None  # one-shot


def test_placement_is_clamped_to_the_clearance_keepout():
    edge = _watch().evaluate(_state(_bump(50.0, 2.0, 200.0)), 0.0)
    assert edge is not None
    assert edge.i_parent_start == 11  # keepout 10 -> minimum start 11
    assert edge.receipt["clamped_to_keepout"] is True
    # Control: an interior storm is not clamped.
    interior = _watch().evaluate(_state(_bump(50.0, 60.0, 200.0)), 0.0)
    assert interior.receipt["clamped_to_keepout"] is False


def test_search_box_excludes_an_outside_stronger_storm():
    plane = _bump(50.0, 60.0, 100.0) + _bump(10.0, 10.0, 500.0)
    boxed = _watch(_spawn(search_box=(50, 40, 71, 61)))
    event = boxed.evaluate(_state(plane), 0.0)
    assert abs(event.i_parent_start - _expected_start(60.0)) <= 1
    assert event.receipt["search_box_source"] == "explicit"
    # Control: with the whole parent open the intruder MUST win.
    wide = _watch().evaluate(_state(plane), 0.0)
    assert wide.i_parent_start == 11  # clamped toward the corner storm
    assert wide.receipt["peak_parent_ij"][0] < 20


def test_default_search_box_inherits_the_follow_margin():
    follow = FollowConfig(field="uh", threshold=50.0,
                          fallback_threshold=40.0, search_margin_cells=5,
                          min_shift_cells=2, max_shift_cells=6,
                          cooldown_seconds=0.0)
    watch = _watch(follow=follow)
    # Storm far outside declared footprint (i 40..55) + 5: invisible.
    assert watch.evaluate(_state(_bump(50.0, 90.0, 200.0)), 0.0) is None
    assert watch.receipts[-1]["search_box_source"] == \
        "declared-footprint+follow-margin"
    # Control: inside the margin box it fires.
    assert _watch(follow=follow).evaluate(
        _state(_bump(48.0, 50.0, 200.0)), 0.0) is not None


def test_two_storms_in_one_box_pick_the_stronger_not_the_midpoint():
    plane = _bump(50.0, 40.0, 150.0) + _bump(50.0, 90.0, 400.0)
    event = _watch().evaluate(_state(plane), 0.0)
    assert abs(event.i_parent_start - _expected_start(90.0)) <= 1
    # The failure mode this design kills: the box-wide centroid between
    # the storms (~i 65) must NOT be the birth position.
    assert abs(event.i_parent_start - _expected_start(65.0)) > 5


def test_exclusion_ignores_signal_inside_an_active_footprint():
    plane = _bump(50.0, 60.0, 200.0)
    active = NestFootprint(grid_id=9, i_parent_start=54, j_parent_start=44,
                           child_nx=CHILD_N, child_ny=CHILD_N,
                           parent_grid_ratio=RATIO)
    watch = _watch()
    assert watch.evaluate(_state(plane), 0.0,
                          exclude_footprints=(active,)) is None
    assert watch.receipts[-1]["decision"] == "no-signal"
    assert watch.receipts[-1]["excluded_active_footprints"]
    # Control: without the exclusion the same storm fires.
    assert _watch().evaluate(_state(plane), 0.0) is not None


def test_missing_signal_plane_refuses_loudly():
    from gpuwm.core.storm_tracking import TrackerRefusal

    with pytest.raises(TrackerRefusal, match="uh_spawn_window"):
        _watch().evaluate(StubState(), 0.0)


def test_a_nest_too_large_to_place_refuses_rather_than_clamping():
    big = NestFootprint(grid_id=2, i_parent_start=40, j_parent_start=40,
                        child_nx=330, child_ny=330, parent_grid_ratio=RATIO)
    watch = SpawnWatch(_spawn(), big, keepout_cells=10)
    with pytest.raises(SpawnRefusal, match="too large to spawn"):
        watch.evaluate(_state(_bump(50.0, 60.0, 200.0)), 0.0)


# ---------------------------------------------------------------------------
# SpawnController: many nests, one boundary
# ---------------------------------------------------------------------------

def _controller(*watch_specs):
    watches = {}
    parent_of = {}
    for grid_id, watch in watch_specs:
        watches[grid_id] = watch
        parent_of[grid_id] = 1
    return SpawnController(watches, parent_of)


def test_controller_from_experiment_builds_keepout_from_the_loader_rule(
        tmp_path):
    exp = load_experiment(_write(tmp_path, d02=FIELD_SPAWN))
    controller = SpawnController.from_experiment(exp)
    assert controller is not None
    assert controller.pending == (2,)
    watch = controller.watches[2]
    assert watch.keepout_cells == exp.spec_bdy_width + exp.blend_width
    plain = load_experiment(_write(tmp_path, d02=""))
    assert SpawnController.from_experiment(plain) is None


def test_two_dormant_nests_fire_on_two_distinct_storms():
    """The coordinator's scenario: both triggers cross threshold at ONE
    boundary with TWO storms on the plane; the first watch claims the
    stronger storm and its fresh footprint excludes it from the second,
    which lands on the other storm."""
    plane = _bump(50.0, 90.0, 400.0) + _bump(50.0, 40.0, 150.0)
    controller = _controller(
        (2, _watch(declared=_declared(grid_id=2))),
        (3, _watch(declared=_declared(grid_id=3))))
    events = controller.evaluate_all({1: _state(plane)}, 0.0)
    assert [event.grid_id for event in events] == [2, 3]
    first, second = events
    assert abs(first.i_parent_start - _expected_start(90.0)) <= 1
    assert abs(second.i_parent_start - _expected_start(40.0)) <= 1
    assert first.position != second.position
    assert controller.pending == ()


def test_one_storm_feeds_only_one_nest():
    """The negative control: with a single storm the second watch sees
    only masked ground and holds."""
    plane = _bump(50.0, 60.0, 400.0)
    controller = _controller(
        (2, _watch(declared=_declared(grid_id=2))),
        (3, _watch(declared=_declared(grid_id=3))))
    events = controller.evaluate_all({1: _state(plane)}, 0.0)
    assert [event.grid_id for event in events] == [2]
    assert controller.pending == (3,)
    assert controller.watches[3].receipts[-1]["decision"] == "no-signal"


def test_controller_refuses_an_unknown_grid_and_a_missing_parent_state():
    controller = _controller((2, _watch(declared=_declared(grid_id=2))))
    with pytest.raises(SpawnRefusal, match="not a declared dormant"):
        controller.evaluate(7, _state(), 0.0)
    with pytest.raises(SpawnRefusal, match="no parent state"):
        controller.evaluate_all({4: _state()}, 0.0)


def test_receipts_drain_in_grid_id_order():
    controller = _controller(
        (3, _watch(declared=_declared(grid_id=3))),
        (2, _watch(declared=_declared(grid_id=2))))
    controller.evaluate_all({1: _state()}, 0.0)
    drained = controller.drain_receipts()
    assert drained  # configured + no-signal entries
    assert controller.drain_receipts() == []
