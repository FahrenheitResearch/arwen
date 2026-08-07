"""CPU contracts for the host-staged relocation transplant.

The claims, each tested in both directions where a direction exists:

1. EQUIVALENCE -- a relocation staged through host memory produces a
   child bitwise identical to the in-device transplant, field for field,
   because D2H and H2D are byte transports.  Checked for a real move and
   for the null move (the calibration point).
2. RESIDENCY ORDER -- under host staging the outgoing child's arrays are
   FREED before the incoming child's initializer runs, which is the
   structural half of the "peak stays at ~one child" claim (the measured
   half is the GPU test's pool samples).  Proved with weakrefs that must
   be dead inside the initializer, plus a negative control: the in-device
   staging keeps them alive at the same instant.
3. STATICS SURVIVE -- static fields are not in the transplant inventory,
   so an initializer that rebuilds footprint-parametric statics keeps
   them: after the move the child's statics are the initializer's,
   bitwise, and on the overlap they equal the outgoing child's (same
   source, same ground -- the 2026-08-06 statics-on-move requirement's
   overlap claim, asserted rather than assumed).
4. PROVENANCE -- every receipt names the initializer and its statics
   provenance; a custom initializer that states none refuses.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core.nest_relocation import (
    PARENT_INTERPOLATED_STATICS_FALLBACK, HostStateSnapshot,
    RelocationRefusal, relocate_child, release_state_arrays,
    snapshot_state_to_host, transplant_overlap, plan_relocation, Placement)

NZ = 2

#: The state inventory the CPU fakes carry (a representative slice of the
#: serialized contract: every stagger, a 2-D field, a scalar tracer).
_FIELDS = ("u", "v", "w", "thp", "php", "mup", "qv")
_SEEDS = ("u0", "v0", "w0", "thp0", "php0", "mup0", "qv0")


def _scaffold():
    from gpuwm.verify.cases.nest_ideal_r3 import load_scaffold

    return load_scaffold()


def _shapes(nx, ny):
    return {
        "u": (NZ, ny, nx + 1), "v": (NZ, ny + 1, nx), "w": (NZ + 1, ny, nx),
        "thp": (NZ, ny, nx), "php": (NZ + 1, ny, nx), "mup": (ny, nx),
        "qv": (NZ, ny, nx),
    }


def _ramp_state(nx, ny, *, offset=0.0, statics=None, with_seeds=False):
    state = SimpleNamespace()
    for index, (name, shape) in enumerate(_shapes(nx, ny).items()):
        count = int(np.prod(shape))
        base = np.arange(count, dtype=np.float32).reshape(shape)
        setattr(state, name,
                base + np.float32(offset) + np.float32(1000 * index))
    if with_seeds:
        for name in _SEEDS:
            shape = _shapes(nx, ny)[name[:-1]]
            setattr(state, name, np.zeros(shape, dtype=np.float32))
    if statics is not None:
        state.pb = statics
    return state


def _footprint_statics(parent_plane, dc):
    """Footprint-parametric statics: nearest-donor pickup off one source.

    The same absolute ground always samples the same source cell, which
    is exactly the property a real per-footprint static rebuild (30s
    baseline / [static.highres]) has, and the property the overlap claim
    rests on.
    """
    ratio = int(dc.parent_grid_ratio)
    nx, ny = int(dc.run.nx), int(dc.run.ny)
    ci = (dc.i_parent_start - 1) + np.arange(nx) // ratio
    cj = (dc.j_parent_start - 1) + np.arange(ny) // ratio
    return parent_plane[np.ix_(cj, ci)].astype(np.float32)


def _cpu_tree():
    exp = _scaffold()
    root_dc, child_dc = exp.domains
    pnx, pny = int(root_dc.run.nx), int(root_dc.run.ny)
    parent_plane = np.arange(pny * pnx, dtype=np.float32).reshape(pny, pnx)
    # The parent carries a (tiny) slice of the serialized contract so the
    # DEFAULT state digest (live_state_sha256) can witness it unchanged
    # when a caller does not inject one -- the runner tests rely on that.
    parent = SimpleNamespace(
        cfg=root_dc,
        state=SimpleNamespace(
            u=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            thp=np.ones((2, 3, 3), dtype=np.float32)),
        clock=SimpleNamespace(ticks=0), children=[])
    child_state = _ramp_state(
        child_dc.run.nx, child_dc.run.ny,
        statics=_footprint_statics(parent_plane, child_dc))
    child = SimpleNamespace(
        cfg=child_dc, state=child_state, grid="old-grid", parent=parent,
        coupler=SimpleNamespace(
            relocate=lambda: {"rolling_tables": "INVALID"}),
        clock=SimpleNamespace(ticks=0), children=[])
    return parent_plane, parent, child


def _initializer(parent_plane, *, on_call=None):
    """A deterministic CPU stand-in for the cold-start rebuild."""

    def initialize(new_dc, parent_node, **_kwargs):
        if on_call is not None:
            on_call()
        state = _ramp_state(
            new_dc.run.nx, new_dc.run.ny, offset=5.0e5,
            statics=_footprint_statics(parent_plane, new_dc),
            with_seeds=True)
        return SimpleNamespace(state=state, grid="new-grid")

    return initialize


def _relocate(child, parent_plane, *, staging, di=1, on_call=None,
              on_before_release=None):
    return relocate_child(
        child,
        i_parent_start=int(child.cfg.i_parent_start) + di,
        j_parent_start=int(child.cfg.j_parent_start),
        initializer=_initializer(parent_plane, on_call=on_call),
        static_provenance="footprint-parametric synthetic statics (test)",
        state_digest=lambda _s: "digest",
        staging=staging, on_before_release=on_before_release)


# ---------------------------------------------------------------------------
# Snapshot semantics
# ---------------------------------------------------------------------------

def test_snapshot_is_bitwise_and_deep():
    state = _ramp_state(12, 12)
    snapshot = snapshot_state_to_host(state, _FIELDS)
    assert isinstance(snapshot, HostStateSnapshot)
    assert snapshot.field_names == tuple(sorted(_FIELDS))
    assert snapshot.nbytes == sum(
        getattr(state, name).nbytes for name in _FIELDS)
    for name in _FIELDS:
        assert np.array_equal(
            getattr(snapshot, name).view(np.uint32),
            getattr(state, name).view(np.uint32)), name
    # Deep: mutating the source cannot reach the snapshot.
    state.thp[...] = -1.0
    assert not np.array_equal(snapshot.thp, state.thp)
    # Absent fields are absent attributes, the transplant's skip contract.
    assert getattr(snapshot, "qh", None) is None


def test_release_drops_every_array_and_counts_them():
    state = _ramp_state(12, 12)
    state.physics = SimpleNamespace(name="driver")
    receipt = release_state_arrays(state)
    assert receipt["host_arrays"] == len(_FIELDS)
    assert receipt["owned_device_arrays"] == 0
    assert receipt["owned_device_bytes"] == 0
    assert all(getattr(state, name) is None for name in _FIELDS)
    assert state.physics is None


def test_transplant_from_snapshot_equals_transplant_from_live_source():
    exp = _scaffold()
    child_dc = exp.domains[1]
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10,
                                 j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=12,
                               j_parent_start=10, generation=1),
        parent_grid_ratio=child_dc.parent_grid_ratio,
        child_nx=24, child_ny=24)
    source = _ramp_state(24, 24)
    live_target = _ramp_state(24, 24, offset=7777.0)
    staged_target = _ramp_state(24, 24, offset=7777.0)
    transplant_overlap(source_state=source, target_state=live_target,
                       plan=plan, attrs=_FIELDS)
    snapshot = snapshot_state_to_host(source, _FIELDS)
    transplant_overlap(source_state=snapshot, target_state=staged_target,
                       plan=plan, attrs=_FIELDS)
    for name in _FIELDS:
        assert np.array_equal(
            getattr(live_target, name).view(np.uint32),
            getattr(staged_target, name).view(np.uint32)), name


# ---------------------------------------------------------------------------
# The full primitive, host-staged, on CPU
# ---------------------------------------------------------------------------

def test_host_and_device_staging_produce_bitwise_identical_children():
    results = {}
    for staging in ("device", "host"):
        parent_plane, _parent, child = _cpu_tree()
        receipt = _relocate(child, parent_plane, staging=staging, di=1)
        results[staging] = (child.state, receipt)
    device_state, device_receipt = results["device"]
    host_state, host_receipt = results["host"]
    for name in _FIELDS + ("pb",) + _SEEDS:
        assert np.array_equal(
            getattr(device_state, name).view(np.uint32),
            getattr(host_state, name).view(np.uint32)), name
    assert device_receipt["transplant"] == host_receipt["transplant"]
    assert device_receipt["plan"] == host_receipt["plan"]
    assert host_receipt["staging"]["mode"] == "host"
    assert host_receipt["staging"]["snapshot_bytes"] > 0
    assert host_receipt["staging"]["released"]["host_arrays"] > 0
    assert device_receipt["staging"]["mode"] == "device"


def test_null_move_is_the_identity_under_host_staging():
    parent_plane, _parent, child = _cpu_tree()
    before = {name: getattr(child.state, name).copy() for name in _FIELDS}
    receipt = _relocate(child, parent_plane, staging="host", di=0)
    assert receipt["plan"]["null_move"] is True
    for name in _FIELDS:
        assert np.array_equal(
            getattr(child.state, name).view(np.uint32),
            before[name].view(np.uint32)), name


def test_host_staging_frees_the_outgoing_child_before_the_rebuild():
    parent_plane, _parent, child = _cpu_tree()
    refs = [weakref.ref(getattr(child.state, name)) for name in _FIELDS]
    events = []

    def rebuild_probe():
        gc.collect()
        events.append(("rebuild", [ref() is None for ref in refs]))

    _relocate(child, parent_plane, staging="host", di=1,
              on_call=rebuild_probe,
              on_before_release=lambda: events.append(("before_release",)))
    assert events[0] == ("before_release",)
    kind, dead = events[1]
    assert kind == "rebuild"
    assert all(dead), (
        "outgoing child arrays were still alive when the incoming child "
        "allocated; peak residency would be two children")


def test_device_staging_keeps_the_outgoing_child_alive_at_the_rebuild():
    """The negative control: the instrument can tell the stagings apart."""
    parent_plane, _parent, child = _cpu_tree()
    refs = [weakref.ref(getattr(child.state, name)) for name in _FIELDS]
    alive_at_rebuild = []

    def rebuild_probe():
        gc.collect()
        alive_at_rebuild.append(all(ref() is not None for ref in refs))

    _relocate(child, parent_plane, staging="device", di=1,
              on_call=rebuild_probe)
    assert alive_at_rebuild == [True]


def test_on_before_release_never_fires_on_the_device_path():
    parent_plane, _parent, child = _cpu_tree()
    fired = []
    _relocate(child, parent_plane, staging="device", di=1,
              on_before_release=lambda: fired.append(True))
    assert fired == []


def test_unknown_staging_refuses():
    parent_plane, _parent, child = _cpu_tree()
    with pytest.raises(RelocationRefusal, match="staging"):
        _relocate(child, parent_plane, staging="pinned", di=1)


def test_relocating_a_child_with_children_refuses():
    parent_plane, _parent, child = _cpu_tree()
    child.children = [SimpleNamespace()]
    with pytest.raises(RelocationRefusal, match="LEAF"):
        _relocate(child, parent_plane, staging="host", di=1)


# ---------------------------------------------------------------------------
# Statics on relocation (the 2026-08-06 requirement's overlap claim)
# ---------------------------------------------------------------------------

def test_footprint_rebuilt_statics_survive_and_match_on_the_overlap():
    parent_plane, _parent, child = _cpu_tree()
    old_pb = child.state.pb.copy()
    old_dc = child.cfg
    receipt = _relocate(child, parent_plane, staging="host", di=2)
    new_dc = replace(old_dc, i_parent_start=old_dc.i_parent_start + 2)
    rebuilt = _footprint_statics(parent_plane, new_dc)
    # The transplant must not have written the statics: they are the
    # initializer's own, everywhere -- including the overlap.
    assert np.array_equal(child.state.pb.view(np.uint32),
                          rebuilt.view(np.uint32))
    # And on the overlap the rebuilt statics equal the OUTGOING child's:
    # same source, same ground.  This is the claim that makes a real
    # per-footprint static rebuild compatible with the bitwise transplant.
    shift = 2 * int(old_dc.parent_grid_ratio)
    keep = old_pb.shape[-1] - shift
    assert np.array_equal(
        child.state.pb[:, :keep].view(np.uint32),
        old_pb[:, shift:].view(np.uint32))
    assert receipt["donor_alignment"]["pass"]


def test_receipt_records_the_statics_provenance():
    parent_plane, _parent, child = _cpu_tree()
    receipt = _relocate(child, parent_plane, staging="host", di=1)
    rebuild = receipt["child_rebuild"]
    assert rebuild["static_fields"] == (
        "footprint-parametric synthetic statics (test)")
    assert "initialize" in rebuild["initializer"]


def test_custom_initializer_without_provenance_refuses():
    parent_plane, _parent, child = _cpu_tree()
    with pytest.raises(RelocationRefusal, match="static_provenance"):
        relocate_child(
            child,
            i_parent_start=int(child.cfg.i_parent_start) + 1,
            j_parent_start=int(child.cfg.j_parent_start),
            initializer=_initializer(parent_plane),
            state_digest=lambda _s: "digest", staging="host")


def test_default_initializer_records_the_named_fallback():
    """The default path's provenance is the explicit fallback text, so a
    receipt can never be silent about parent-interpolated statics.  Since
    leg 3 the text also points at the landed real-data initializer
    instead of naming a follow-up."""
    assert "parent-interpolated" in PARENT_INTERPOLATED_STATICS_FALLBACK
    assert "relocation_init" in PARENT_INTERPOLATED_STATICS_FALLBACK
    assert "follow-up" not in PARENT_INTERPOLATED_STATICS_FALLBACK


# ---------------------------------------------------------------------------
# On the card: the measured memory claim, and bitwise equivalence
# ---------------------------------------------------------------------------

def _integrated_tree(leg_seconds=30.0):
    from gpuwm.core.model import execute_experiment
    from gpuwm.static.lambert import grids_from_projection_config
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree
    from gpuwm.verify.cases.nest_ideal_r3 import _build_root
    from gpuwm.verify.cases.nest_relocate import short_scaffold

    exp = short_scaffold(run_seconds=float(leg_seconds))
    model = assemble_idealized_tree(
        exp, _build_root(exp), grids=grids_from_projection_config(exp))
    execute_experiment(model)
    return exp, model


def _gpu_relocate(exp, model, *, staging, move_cells=2):
    from gpuwm.core.nest_relocation import relocatable_attrs
    from gpuwm.verify.cases.nest_relocate import _child_preparer

    child = model.node(exp.domains[1].grid_id)
    receipt = relocate_child(
        child,
        i_parent_start=int(child.cfg.i_parent_start) + int(move_cells),
        j_parent_start=int(child.cfg.j_parent_start),
        on_child_built=_child_preparer(exp.start_time),
        staging=staging)
    fields = {}
    for name in relocatable_attrs():
        value = getattr(child.state, name, None)
        if value is not None:
            fields[name] = np.ascontiguousarray(value.get())
    return receipt, fields


@requires_gpu
def test_gpu_host_staging_is_bitwise_equivalent_and_keeps_peak_at_one_child():
    """The leg-2 claim, measured on the card.

    Two identical integrated trees; one relocation each, staged in-device
    and through pinned host memory.  (a) Every serialized field of the
    two relocated children is bitwise identical.  (b) The host-staged
    receipt's live-pool samples show the rebuild peaking within a small
    margin of steady state, while the in-device arm -- the negative
    control -- shows the transient second child.
    """
    pytest.importorskip("cupy")
    exp_a, model_a = _integrated_tree()
    receipt_a, fields_a = _gpu_relocate(exp_a, model_a, staging="device")
    del model_a
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    exp_b, model_b = _integrated_tree()
    receipt_b, fields_b = _gpu_relocate(exp_b, model_b, staging="host")

    # (a) bitwise equivalence, field for field.
    assert sorted(fields_a) == sorted(fields_b)
    for name, expected in fields_a.items():
        actual = fields_b[name]
        assert actual.dtype == expected.dtype and (
            actual.shape == expected.shape), name
        assert np.array_equal(
            actual.view(np.uint8), expected.view(np.uint8)), name

    # (b) the measured memory claim.
    child_bytes = receipt_b["staging"]["released"]["owned_device_bytes"]
    assert child_bytes > 0
    samples_b = receipt_b["staging"]["device_pool_used_bytes"]
    steady = samples_b["steady_state_before"]
    released = samples_b["after_release"]
    rebuilt = samples_b["after_rebuild_and_transplant"]
    # The release really freed the outgoing child...
    assert steady - released >= 0.5 * child_bytes
    # ...and the rebuild peaked within a small margin of steady state.
    assert rebuilt - steady <= 0.25 * child_bytes, (
        f"host-staged rebuild grew the live pool by {rebuilt - steady} "
        f"bytes against a child of {child_bytes}")
    # Negative control: the in-device transplant holds two children.
    samples_a = receipt_a["staging"]["device_pool_used_bytes"]
    device_growth = (samples_a["after_rebuild_and_transplant"]
                     - samples_a["steady_state_before"])
    assert device_growth >= 0.5 * child_bytes, (
        "the in-device arm no longer doubles residency; the negative "
        "control lost its treatment")


@requires_gpu
def test_gpu_scheduled_move_executes_inside_a_run(tmp_path):
    """The runner end-to-end on the card: a [[relocation.move]] row fires
    at its cycle boundary inside execute_experiment, the SAME model keeps
    integrating on the rebuilt coupler tables, and the receipts land."""
    pytest.importorskip("cupy")
    import json

    from gpuwm.core.model import execute_experiment
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
    from gpuwm.static.lambert import grids_from_projection_config
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree
    from gpuwm.verify.cases.nest_ideal_r3 import _build_root
    from gpuwm.verify.cases.nest_relocate import (_child_preparer,
                                                  short_scaffold)

    exp = short_scaffold(run_seconds=60.0)
    exp = replace(exp, relocation=RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.5,
        moves=(ScheduledRelocationMove(30.0, 2, 0),)))
    model = assemble_idealized_tree(
        exp, _build_root(exp), grids=grids_from_projection_config(exp))
    child = model.node(2)
    start_i = int(child.cfg.i_parent_start)
    fingerprint_before = model.experiment_fingerprint
    runner = RelocationRunner.from_experiment(
        exp, schedule=model.schedule,
        on_child_built=_child_preparer(exp.start_time),
        receipts_path=tmp_path / "relocation.json")

    execution = execute_experiment(model, relocation_runner=runner)

    assert execution.steps > 0
    assert runner.moves_executed == 1
    assert child.cfg.i_parent_start == start_i + 2
    assert model.experiment_fingerprint != fingerprint_before
    document = json.loads(
        (tmp_path / "relocation.json").read_text(encoding="utf-8"))
    events = [row["event"] for row in document["receipts"]]
    assert events.count("relocated") == 1
    moved = next(row for row in document["receipts"]
                 if row["event"] == "relocated")
    assert moved["elapsed_seconds"] == 30.0
    assert moved["staging"]["mode"] == "host"
    assert moved["donor_alignment_pass"] is True
    assert moved["parent_bitwise_unchanged"] is True
