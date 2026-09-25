"""A cycling leg boundary is a restart: measured, array for array.

The DA driver rebuilds every trajectory's model at every leg and joins
one leg to the next.  The join used to be a host copy of the serialised
atmosphere and a clock placed by arithmetic, which carried nothing the
physics driver owns -- soil, surface, precipitation accumulators, held
tendencies, the radiation carriers -- so all of it restarted from the
prepared background at every analysis.  The join is now the restart
owner's own checkpoint set.

These cells hold that as a measurement on the card, on the same
hand-built tree the driver assembles (the synthetic parent and nest of
``tests/test_da_nested_forecast_gpu.py``): the checkpoint a continuous
run writes at its end has to equal, member for member and byte for
byte, the checkpoint a run split into two legs and joined through the
owner writes at the same instant.  The retired join is kept beside it
as the red arm, so the comparison cannot pass by comparing nothing.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from types import MappingProxyType

import numpy as np
import pytest
from conftest import requires_gpu
from test_da_nested_forecast_gpu import (_IDEALISED_GLW, _build_parent_state,
                                         _geometry, _synthetic_land, _terrain,
                                         _parent_run)

from gpuwm.da import nested_forecast as nf
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)

PARENT_DT = 15.0
#: Eight parent steps: four per leg, so a leg has a whole physics
#: cadence of its own and the join lands mid-run rather than at t=0.
LEG_SECONDS = 60.0
RUN_SECONDS = 2 * LEG_SECONDS
START = datetime(2024, 5, 21, 21)


def _experiment_to(stop_seconds: float) -> ExperimentConfig:
    """The fixture's experiment, running to ``stop_seconds``."""
    run = dataclasses.replace(_parent_run(), run_seconds=float(stop_seconds),
                              output_interval_s=float(stop_seconds))
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(stop_seconds), run=run,
        time_step=int(PARENT_DT))
    return ExperimentConfig(
        name="cycle_join", start_time=START, run_seconds=float(stop_seconds),
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0,
                                    -97.0),
        restart_interval_s=0.0, domains=(root,))


def _wire_parent(exp, *, child_dc=None):
    """The parent model exactly as the driver wires one, child not built.

    Returns the pieces the driver's own ``wire`` returns, so a cell can
    build, restore or perturb in the order the driver does.
    """
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.model import DomainNode
    from gpuwm.core.physics import initialize_physics
    from gpuwm.static.lambert import grids_from_projection_config

    run = exp.root.run
    state, _coord = _build_parent_state(run)
    static, surface, identity = _synthetic_land(run, _terrain(run))
    grid = grids_from_projection_config(exp)[0]
    lat, lon = grid.latlon_mass()
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=surface.fields["SNOW"],
        xice=surface.fields["SEAICE"], valid_time=exp.start_time,
        cen_lat=float(getattr(grid, "cen_lat", grid.ref_lat)),
        mminlu=identity["MMINLU"], iswater=identity["ISWATER"],
        islake=identity["ISLAKE"], isice=identity["ISICE"],
        fractional_seaice=True,
        soil_temperature=surface.fields["TSLB"])
    driver = initialize_physics(
        state, run, landuse=landuse, tsk=surface.fields["TSK"],
        soil_temperature=surface.fields["TSLB"],
        soil_moisture=surface.fields["SMOIS"],
        liquid_moisture=surface.fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=100.0 * static["GREENFRAC"][4],
        tmn=surface.fields["TMN"], xice=surface.fields["SEAICE"],
        snow=surface.fields["SNOW"], snow_depth=surface.fields["SNOWH"],
        sst=surface.fields["TSK"],
        radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon, glw=_IDEALISED_GLW)
    nested_exp = exp
    live_born = ()
    if child_dc is not None:
        nested_exp = nf.nested_experiment(exp, child_dc)
        live_born = (child_dc.grid_id,)
    tick = resolve_clock(nested_exp, live_born_children=live_born)
    schedule = build_schedule(nested_exp, tick)
    clocks = tick.clocks()
    root = DomainNode(exp.root, grid, state, clocks[1], None, [], None)
    return {"root": root, "driver": driver, "clocks": clocks,
            "schedule": schedule, "static": static, "surface": surface,
            "identity": identity, "child_dc": child_dc}


def _build_child(wired):
    child_dc = wired["child_dc"]
    clock = wired["clocks"][child_dc.grid_id]
    nf.place_newborn_clock(clock)
    node, driver, _receipt = nf.build_nested_child(
        wired["root"], child_dc, static=wired["static"],
        surface=wired["surface"], landuse_identity=wired["identity"],
        valid_time=START, clock=clock, parent_driver=wired["driver"],
        constant_glw_wm2=_IDEALISED_GLW)
    return node


def _assemble(wired, *, child_node=None, fingerprint="cycle-join"):
    from gpuwm.core.model import ExperimentState, ModelRuntimeStatus

    nodes = {1: wired["root"]}
    if child_node is not None:
        nodes[child_node.cfg.grid_id] = child_node
    model = ExperimentState(wired["root"], MappingProxyType(nodes),
                            wired["schedule"], None, fingerprint)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({})
    return model


def _run(model):
    import cupy as cp

    from gpuwm.core.model import execute_experiment

    execute_experiment(model, history_handler=None, progress_callback=None,
                       validate_state=True, skip_feedback_path=True,
                       pool_trim_per_period=True)
    cp.cuda.Stream.null.synchronize()


def _checkpoint(model, directory, seconds: float):
    from tools.da_cycle_prepared import write_leg_restart

    return write_leg_restart(model, directory,
                             valid_time=START + timedelta(seconds=seconds))


def _members(root_member) -> dict:
    """``{grid_id: {key: bytes}}`` of every array in a checkpoint set."""
    from gpuwm.io.restart import tree_restart_members

    out = {}
    for gid, path in tree_restart_members(root_member).items():
        with np.load(path, allow_pickle=False) as data:
            out[gid] = {key: data[key].tobytes() for key in data.files
                        if key != "__gpuwm_restart_header__"}
    return out


def _headers(root_member) -> dict:
    from gpuwm.io.restart import read_restart_header, tree_restart_members

    return {gid: read_restart_header(path)
            for gid, path in tree_restart_members(root_member).items()}


def _differing(expected: dict, actual: dict) -> dict:
    """Per domain, the keys whose bytes differ or that only one side has."""
    out = {}
    for gid in sorted(set(expected) | set(actual)):
        left = expected.get(gid, {})
        right = actual.get(gid, {})
        names = sorted(key for key in set(left) | set(right)
                       if left.get(key) != right.get(key))
        if names:
            out[gid] = names
    return out


def _continuous(tmp_path, *, nested: bool):
    """One model, the whole run, one checkpoint at the end."""
    exp = _experiment_to(RUN_SECONDS)
    child_dc = nf.nest_domain_config(exp, _geometry()) if nested else None
    if child_dc is not None:
        child_dc = nf.child_born_at(child_dc, exp, 0.0)
    wired = _wire_parent(exp, child_dc=child_dc)
    child = _build_child(wired) if nested else None
    model = _assemble(wired, child_node=child)
    _run(model)
    return _checkpoint(model, tmp_path / "continuous", RUN_SECONDS)


def _two_legs_through_the_owner(tmp_path, *, nested: bool):
    """Leg 1 to the boundary, a checkpoint, a fresh model restored, leg 2."""
    from tools.da_cycle_prepared import restore_leg_restart

    first = _experiment_to(LEG_SECONDS)
    child_dc = nf.nest_domain_config(first, _geometry()) if nested else None
    if child_dc is not None:
        child_dc = nf.child_born_at(child_dc, first, 0.0)
    wired = _wire_parent(first, child_dc=child_dc)
    child = _build_child(wired) if nested else None
    model = _assemble(wired, child_node=child)
    _run(model)
    boundary = _checkpoint(model, tmp_path / "leg1", LEG_SECONDS)
    del model, wired, child

    second = _experiment_to(RUN_SECONDS)
    child_dc = nf.nest_domain_config(second, _geometry()) if nested else None
    if child_dc is not None:
        child_dc = nf.child_born_at(child_dc, second, 0.0)
    wired = _wire_parent(second, child_dc=child_dc)
    child = _build_child(wired) if nested else None
    model = _assemble(wired, child_node=child)
    info = restore_leg_restart(model, boundary, expected_seconds=LEG_SECONDS)
    assert info.elapsed_ticks / info.tick_den == LEG_SECONDS
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(model.nodes_by_grid_id)
    _run(model)
    return _checkpoint(model, tmp_path / "leg2", RUN_SECONDS)


def _two_legs_through_the_retired_join(tmp_path):
    """The join this tree used to make: the atmosphere and a placed clock."""
    import cupy as cp

    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    first = _experiment_to(LEG_SECONDS)
    wired = _wire_parent(first)
    model = _assemble(wired)
    _run(model)
    state = wired["root"].state
    handoff = {field: getattr(state, field).get()
               for field in STATE_SERIALIZED_ATTRS
               if getattr(state, field, None) is not None}
    del model, wired

    second = _experiment_to(RUN_SECONDS)
    wired = _wire_parent(second)
    state = wired["root"].state
    steps = int(round(LEG_SECONDS / PARENT_DT))
    clock = wired["root"].clock
    clock.ticks = steps * clock.spec.step_ticks
    clock.step_count = steps
    value = np.float32(0.0)
    for _ in range(steps):
        value = np.float32(value + clock.spec.dt_fp32)
    clock.dtbc_fp32 = value
    for field, host in handoff.items():
        getattr(state, field)[...] = cp.asarray(
            host, dtype=getattr(state, field).dtype)
    model = _assemble(wired)
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset({1})
    _run(model)
    return _checkpoint(model, tmp_path / "retired", RUN_SECONDS)


def _clock_fields(header: dict) -> dict:
    return {key: header[key] for key in
            ("elapsed_ticks", "tick_den", "elapsed_seconds", "dtbc_fp32_bits",
             "domain_start_ticks", "domain_lifecycle")}


def _two_legs_staged(tmp_path):
    """The two legs through the owner, staged where the driver stages them.

    Leg 1 writes its set under the stage, leg 2 restores it, consumes it
    and writes its own, and the generation copies that final set out; what
    the stage holds after that is what a run would leave behind.
    """
    from tools import da_ensemble_state as ens_state
    from tools.da_cycle_prepared import StagedRestarts, restore_leg_restart

    out = tmp_path / "out"
    stage = StagedRestarts(out / "stage", default=True)
    first = _experiment_to(LEG_SECONDS)
    wired = _wire_parent(first)
    model = _assemble(wired)
    _run(model)
    boundary = _checkpoint(model, stage.directory(1, "control"), LEG_SECONDS)
    assert stage.holds(boundary)
    del model, wired

    second = _experiment_to(RUN_SECONDS)
    wired = _wire_parent(second)
    model = _assemble(wired)
    restore_leg_restart(model, boundary, expected_seconds=LEG_SECONDS)
    assert stage.consume(boundary) is True
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(model.nodes_by_grid_id)
    _run(model)
    final = _checkpoint(model, stage.directory(2, "control"), RUN_SECONDS)
    generation = tmp_path / "generation"
    copied = ens_state.copy_restart_set(
        final, ens_state.restart_dir(generation, "control"))
    return stage, final, generation, copied


@requires_gpu
@pytest.mark.gpu
def test_two_legs_joined_through_the_restart_owner_are_one_continuous_run(
        tmp_path):
    """Every array of the checkpoint set, byte for byte, and both clocks."""
    whole = _continuous(tmp_path, nested=False)
    legs = _two_legs_through_the_owner(tmp_path, nested=False)
    expected, actual = _members(whole), _members(legs)
    assert set(expected) == set(actual) == {1}
    assert len(expected[1]) > 0
    assert set(expected[1]) == set(actual[1])
    assert _differing(expected, actual) == {}
    whole_header, legs_header = _headers(whole)[1], _headers(legs)[1]
    assert _clock_fields(whole_header) == _clock_fields(legs_header)
    assert legs_header["elapsed_seconds"] == RUN_SECONDS


@requires_gpu
@pytest.mark.gpu
def test_the_leg_loop_leaves_the_stage_empty_at_exit(tmp_path):
    """Real sets on the stage: the last leg's is removed at exit and the
    generation's copy of it is the whole set, byte for byte."""
    from gpuwm.io.restart import tree_restart_members
    from tools import da_ensemble_state as ens_state

    stage, final, generation, copied = _two_legs_staged(tmp_path)
    staged_bytes = {gid: path.read_bytes()
                    for gid, path in tree_restart_members(final).items()}
    assert stage.inventory()["restart_sets"] == 1
    receipt = stage.clear()
    assert receipt["restart_sets_removed"] == 1
    assert receipt["restart_bytes_removed"] == sum(
        len(data) for data in staged_bytes.values())
    assert receipt["stage_root_removed"] is True
    assert not (tmp_path / "out" / "stage").exists()
    root = (ens_state.restart_dir(generation, "control")
            / copied["restart_root"])
    assert {gid: path.read_bytes()
            for gid, path in tree_restart_members(root).items()} == staged_bytes


@requires_gpu
@pytest.mark.gpu
def test_the_retired_atmosphere_only_join_is_not_a_continuous_run(tmp_path):
    """The red arm, kept: the fix above is measured against this.

    A join that carries the serialised atmosphere alone leaves the
    physics driver's own state -- soil, surface, accumulators, held
    tendencies -- at the prepared background's values, and the arrays a
    continuous run wrote differ from it.  If this ever passes with an
    empty difference, the comparison above has stopped measuring
    anything.
    """
    whole = _continuous(tmp_path, nested=False)
    retired = _two_legs_through_the_retired_join(tmp_path)
    differing = _differing(_members(whole), _members(retired))
    assert differing, "the retired join reproduced a continuous run"
    names = differing[1]
    # The driver's own state is what the retired join lost.
    assert any(key.startswith("fields/") for key in names), names
    assert any(key.startswith("driver/") or key.startswith("scratch/")
               for key in names), names


@requires_gpu
@pytest.mark.gpu
def test_a_carried_nest_crosses_the_boundary_inside_the_same_set(tmp_path):
    """Parent and child, restored by one call, equal to the continuous run."""
    whole = _continuous(tmp_path, nested=True)
    legs = _two_legs_through_the_owner(tmp_path, nested=True)
    expected, actual = _members(whole), _members(legs)
    assert set(expected) == set(actual) == {1, 2}
    for gid in (1, 2):
        assert len(expected[gid]) > 0
        assert set(expected[gid]) == set(actual[gid])
    assert _differing(expected, actual) == {}
    for gid in (1, 2):
        assert _clock_fields(_headers(whole)[gid]) == _clock_fields(
            _headers(legs)[gid])


@requires_gpu
@pytest.mark.gpu
def test_a_nest_born_on_the_second_leg_activates_there_and_is_restored_after(
        tmp_path):
    """A child born mid-cycle counts its own steps and rides the next set.

    Leg 1 runs the parent alone.  Leg 2 restores it, is given a child
    born at the boundary (built from the restored parent, its clock
    placed at its birth), and integrates.  Leg 3 restores parent and
    child from leg 2's set.  What is measured: the child's activation
    epoch is the boundary, its ITIMESTEP counts from there, its
    checkpoint header says so, and the restore of leg 3 finds the
    calendar it wrote.
    """
    from gpuwm.io.restart import read_restart_header, tree_restart_members
    from tools.da_cycle_prepared import (restart_child_birth_seconds,
                                         restore_leg_restart)

    first = _experiment_to(LEG_SECONDS)
    wired = _wire_parent(first)
    model = _assemble(wired)
    _run(model)
    boundary = _checkpoint(model, tmp_path / "leg1", LEG_SECONDS)
    del model, wired

    second = _experiment_to(RUN_SECONDS)
    child_dc = nf.child_born_at(nf.nest_domain_config(second, _geometry()),
                                second, LEG_SECONDS)
    wired = _wire_parent(second, child_dc=child_dc)
    model = _assemble(wired)
    restore_leg_restart(model, boundary, expected_seconds=LEG_SECONDS)
    child = _build_child(wired)
    assert child.clock.ticks == wired["root"].clock.ticks
    assert child.clock.step_count == 0
    assert child.state.elapsed_seconds == LEG_SECONDS
    assert child.state.domain_start_offset == LEG_SECONDS
    model = _assemble(wired, child_node=child)
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(model.nodes_by_grid_id)
    _run(model)
    ratio = child_dc.parent_grid_ratio
    assert child.clock.step_count == int(LEG_SECONDS / PARENT_DT) * ratio
    assert child.state.domain_start_offset == LEG_SECONDS
    second_set = _checkpoint(model, tmp_path / "leg2", RUN_SECONDS)
    members = tree_restart_members(second_set)
    assert set(members) == {1, 2}
    child_header = read_restart_header(members[2])
    assert child_header["domain_start_ticks"] == int(
        LEG_SECONDS * child_header["tick_den"])
    assert child_header["domain_lifecycle"] == "STARTED"
    assert restart_child_birth_seconds(
        second_set, grid_id=2, start_time=START) == LEG_SECONDS
    del model, wired, child

    third = _experiment_to(RUN_SECONDS + LEG_SECONDS)
    child_dc = nf.child_born_at(nf.nest_domain_config(third, _geometry()),
                                third, LEG_SECONDS)
    wired = _wire_parent(third, child_dc=child_dc)
    child = _build_child(wired)
    model = _assemble(wired, child_node=child)
    info = restore_leg_restart(model, second_set, expected_seconds=RUN_SECONDS)
    assert info.elapsed_ticks / info.tick_den == RUN_SECONDS
    assert child.clock.step_count == int(LEG_SECONDS / PARENT_DT) * ratio
    assert child.state.elapsed_seconds == RUN_SECONDS
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(model.nodes_by_grid_id)
    _run(model)
    assert child.clock.step_count == 2 * int(LEG_SECONDS / PARENT_DT) * ratio


@requires_gpu
@pytest.mark.gpu
def test_a_child_built_with_a_clock_off_its_birth_is_refused(tmp_path):
    exp = _experiment_to(LEG_SECONDS)
    child_dc = nf.child_born_at(nf.nest_domain_config(exp, _geometry()),
                                exp, 0.0)
    wired = _wire_parent(exp, child_dc=child_dc)
    clock = wired["clocks"][child_dc.grid_id]
    clock.ticks = clock.spec.step_ticks
    with pytest.raises(nf.NestedForecastRefusal, match="birth"):
        nf.build_nested_child(
            wired["root"], child_dc, static=wired["static"],
            surface=wired["surface"], landuse_identity=wired["identity"],
            valid_time=START, clock=clock, parent_driver=wired["driver"],
            constant_glw_wm2=_IDEALISED_GLW)
