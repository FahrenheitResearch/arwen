"""CPU contracts for the streamed-child coupling corridor.

The mirrored shape of the nest-stream lane: RESIDENT parent, TILE-STREAMED
child, one run, coupled every parent step.  These units drive the REAL
seams on host stand-ins -- ``refresh_from_store``'s host-mirror branch,
the real windowed tile attachment, the real launch-time generation reload
-- so what is under test is the code that ships, not a transcription of
it.  The stepped bit-identity half is ``tilestream/test_streamed_child.py``
on the GPU shard (the roles-flipped twin of
``tilestream/test_nest_executor.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import nest_stream
from gpuwm.core import streaming
from gpuwm.core.clock import DomainClock, DomainTicks
from gpuwm.core.nest import NestCoupler
from gpuwm.ingest.lateral_bc import (_active_device_interval,
                                     attach_nest_boundaries)


# ---------------------------------------------------------------------------
# fixtures: tests/test_nest_coupler.py's family, inlined.  That module
# imports the dycore (whose module scope imports cupy) for an unrelated
# predicate, and THIS module is on the battery's CPU-hermetic stage-1
# list, so it must stay importable where cupy is not installed at all.
# ---------------------------------------------------------------------------

def _run(nx, ny, *, nested, grid_id):
    return RunConfig(nx=nx, ny=ny, nz=2, dx=1000.0, dy=1000.0,
                     ztop=10000.0, dt=3.0, run_seconds=9.0,
                     nested=nested, specified=not nested, grid_id=grid_id,
                     spec_bdy_width=5, spec_zone=1, relax_zone=4)


def _clock(grid_id, parent_id, *, step_ticks, dt, advanced=False):
    spec = DomainTicks(
        grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
        step_ticks=step_ticks, dt_fp32=np.float32(dt), history_ticks=100,
        restart_ticks=None, radt_ticks=None, stepra=None, cudt_ticks=None,
        stepcu=None, bldt_ticks=None, stepbl=None)
    clock = DomainClock(spec, tick_den=1, run_ticks=1000)
    if advanced:
        clock.advance()
    return clock


class _State:
    def __init__(self, run):
        nz, ny, nx = run.nz, run.ny, run.nx
        self.mub2d = np.arange(ny * nx, dtype=np.float32
                               ).reshape(ny, nx) / 9 + 5
        self.mup = np.full((ny, nx), np.float32(0.25))
        self.u = np.full((nz, ny, nx + 1), np.float32(2.0))
        self.v = np.full((nz, ny + 1, nx), np.float32(-1.5))
        self.w = np.full((nz + 1, ny, nx), np.float32(0.75))
        self.thp = np.full((nz, ny, nx), np.float32(1.25))
        self.php = np.full((nz + 1, ny, nx), np.float32(3.0))
        self.thb = np.array([300.0, 302.0], dtype=np.float32)
        self.c1h = np.array([0.8, 0.6], dtype=np.float32)
        self.c2h = np.array([1.0, 2.0], dtype=np.float32)
        self.c1f = np.array([1.0, 0.7, 0.4], dtype=np.float32)
        self.c2f = np.array([0.0, 1.5, 3.0], dtype=np.float32)
        self.msft = np.full((ny, nx), np.float32(1.25))
        self.msfu = np.full((ny, nx + 1), np.float32(1.5))
        self.msfv = np.full((ny + 1, nx), np.float32(1.75))
        self.has_msf = True
        self._scratch = {}
        self.lateral_boundaries = None

    def scratch(self, shape, slot, dtype=None):
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = np.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result


def _nodes():
    prun = _run(16, 16, nested=False, grid_id=1)
    crun = _run(30, 30, nested=True, grid_id=2)
    pcfg = SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                           i_parent_start=1, j_parent_start=1, run=prun)
    ccfg = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                           i_parent_start=4, j_parent_start=4, run=crun)
    parent = SimpleNamespace(
        cfg=pcfg, state=_State(prun), parent=None,
        clock=_clock(1, 0, step_ticks=3, dt=9.0, advanced=True))
    child_clock = _clock(2, 1, step_ticks=1, dt=3.0)
    child_clock.prepare_step()
    child = SimpleNamespace(cfg=ccfg, state=_State(crun), parent=parent,
                            clock=child_clock)
    return parent, child


def _publish(state, arrays):
    """Give ``state`` a streaming store, the way ``streaming.attach`` does."""
    setattr(state, streaming._STORE_ATTR,
            {f"state/{name}": value for name, value in arrays.items()})


# ---------------------------------------------------------------------------
# the frame rule
# ---------------------------------------------------------------------------

def test_frame_windows_are_the_four_strips_of_the_one_slice_rule():
    """West/east full-height, south/north full-width, corners twice.

    The overlap is the superset rule at work: an overlapping copy is
    idempotent, a missing one is a stale cell inside the zone the reader
    uses.  Each window is then sliced by ``window_slices`` -- the same +1
    widening and clamp every consumer shares -- so the frame rule adds no
    second slice arithmetic.
    """
    windows = streaming.frame_windows(20, 12, 3)
    assert windows == ((0, 20, 0, 3), (0, 20, 9, 12),
                       (0, 3, 0, 12), (17, 20, 0, 12))
    covered = np.zeros((20, 12), dtype=bool)
    for window in windows:
        sl = streaming.window_slices((20, 12), window)
        covered[sl[1:]] = True
    # Every cell within 3 of an edge is covered (4 on the low sides: the
    # +1 slice widening extends a window's END, so the low-side strips
    # reach one deeper); the deep interior is not -- the window is real,
    # not a whole-field pull with a frame name.
    frame = np.zeros((20, 12), dtype=bool)
    frame[:4, :] = frame[-3:, :] = frame[:, :4] = frame[:, -3:] = True
    assert covered[frame].all()
    assert not covered[5:15, 5:7].any()
    with pytest.raises(ValueError):
        streaming.frame_windows(20, 12, 0)


def test_child_frame_windows_carry_the_boundary_width_plus_the_halo():
    """``spec_bdy_width + 8``, degenerating safely on a tiny child."""
    run = SimpleNamespace(nx=30, ny=30, spec_bdy_width=5)
    assert nest_stream.NEST_FORCE_FRAME_HALO_CHILD_CELLS == 8
    windows = nest_stream.child_frame_windows(run)
    assert windows[0] == (0, 30, 0, 13)
    # A child smaller than two frames degenerates to full coverage, which
    # is the superset rule doing exactly what it says.
    tiny = SimpleNamespace(nx=10, ny=10, spec_bdy_width=5)
    for j0, j1, i0, i1 in nest_stream.child_frame_windows(tiny):
        assert 0 <= j0 and j1 <= 10 and 0 <= i0 and i1 <= 10


# ---------------------------------------------------------------------------
# the coupler's child-side corridor
# ---------------------------------------------------------------------------

def _force_probe(monkeypatch, coupler):
    """Run ``force`` with the device kernels stubbed; return the child
    values ``couple_nest_field`` saw, keyed by kind, at a frame cell and
    at the deep-interior cell.
    """
    import gpuwm.core.nest as nest_mod

    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    monkeypatch.setattr(nest_mod, "bdy_interp1", lambda *a, **k: k["out"])
    monkeypatch.setattr(nest_mod, "attach_nest_boundaries",
                        lambda *a, **k: None)
    seen = {}

    def observe(state, kind, out):
        field = {"mu": state.mup[None], "t": state.thp,
                 "ph": state.php}.get(kind, getattr(state, kind, None))
        if field is not None:
            seen[kind] = (float(field[0, 0, 0]), float(field[0, 15, 15]))
        return out

    monkeypatch.setattr(nest_mod, "couple_nest_field", observe)
    coupler.force(coupler.child_node)
    return seen


def test_force_reads_the_streamed_childs_store_frame_and_only_the_frame(
        monkeypatch):
    """The FORCE corridor, child side, on the REAL store seam.

    The child's store carries a theta that differs from the frozen state
    everywhere.  ``force`` must couple the STORE's number at a boundary
    cell (the value tables are built from it) and must NOT have pulled the
    deep interior -- the window is narrowing the read, which is what the
    O(perimeter) claim rests on.  Both directions, so a frame that
    quietly became a whole-field pull fails here rather than passing as a
    slower green.
    """
    _parent, child = _nodes()
    coupler = NestCoupler(child)
    frozen = float(child.state.thp[0, 0, 0])
    swept = child.state.thp + np.float32(11.0)
    _publish(child.state, {"thp": swept, "mup": child.state.mup.copy()})

    seen = _force_probe(monkeypatch, coupler)
    boundary, interior = seen["t"]
    assert boundary == pytest.approx(float(swept[0, 0, 0]))
    assert interior == pytest.approx(frozen), (
        "the deep interior moved: the frame pull is reading the whole "
        "child, so the corridor's O(perimeter) claim is not being "
        "exercised by anything")
    assert coupler.force_sync_bytes > 0
    # The receipt is bounded by the frame arithmetic: four strips of
    # width spec_bdy_width + 8 (+1 slice widening), value + mup, per kind.
    run = child.cfg.run
    width = run.spec_bdy_width + nest_stream.NEST_FORCE_FRAME_HALO_CHILD_CELLS
    strip_cells = 2 * (run.ny + run.nx) * (width + 1)
    kinds = 16          # superset of this dry fixture's inventory
    bound = kinds * strip_cells * (run.nz + 2) * 4 * 2
    assert coupler.force_sync_bytes <= bound


def test_force_with_the_store_unpublished_couples_the_frozen_child(
        monkeypatch):
    """The negative control: the instrument must see a stale coupler."""
    _parent, child = _nodes()
    coupler = NestCoupler(child)
    frozen = float(child.state.thp[0, 0, 0])
    seen = _force_probe(monkeypatch, coupler)
    assert seen["t"][0] == pytest.approx(frozen)
    assert coupler.force_sync_bytes == 0


def test_feedback_reads_the_whole_child_field_not_the_frame(monkeypatch):
    """The restriction reads the whole child interior; its pull is honest.

    ``_coupled_child_field`` without ``frame=True`` must land the store's
    INTERIOR values on the state -- a frame pull here would restrict
    attach-time air from every interior donor cell, silently.
    """
    _parent, child = _nodes()
    coupler = NestCoupler(child)
    swept = child.state.thp + np.float32(3.0)
    _publish(child.state, {"thp": swept, "mup": child.state.mup.copy()})
    monkeypatch.setattr("gpuwm.core.nest.couple_nest_field",
                        lambda state, kind, out: out)
    coupler._coupled_child_field("t")
    assert child.state.thp[0, 15, 15] == pytest.approx(
        float(swept[0, 15, 15]))


# ---------------------------------------------------------------------------
# the per-tile windowed rolling attachment
# ---------------------------------------------------------------------------

def _rolling_fields(state, width=5, seed=7):
    """A full-perimeter rolling table set for ``attach_nest_boundaries``."""
    run_shapes = {
        "u": state.u.shape, "v": state.v.shape, "w": state.w.shape,
        "theta": state.thp.shape, "phi": state.php.shape,
        "mu": (1, *state.mup.shape),
    }
    rng = np.random.default_rng(seed)
    fields = {}
    for name, (nz, ny, nx) in run_shapes.items():
        sides = {}
        for side in ("west", "east", "south", "north"):
            shape = ((nz, ny, width) if side in ("west", "east")
                     else (nz, width, nx))
            sides[side] = (
                rng.standard_normal(shape).astype(np.float32),
                rng.standard_normal(shape).astype(np.float32))
        fields[name] = sides
    return fields


def _attached_child(width=5):
    _parent, child = _nodes()
    fields = _rolling_fields(child.state, width)
    attach_nest_boundaries(child.state, fields, clock=child.clock,
                           spec_bdy_width=width, spec_zone=1, relax_zone=4)
    return child


def _tspec(child, ci0, cj0, cnx, cny):
    run = child.cfg.run
    return SimpleNamespace(ci0=ci0, cj0=cj0, cnx=cnx, cny=cny,
                           nx=run.nx, ny=run.ny,
                           periodic_x=False, periodic_y=False,
                           index=0, ty=0, tx=0)


def _tile_state(child, cnx, cny):
    from dataclasses import replace

    return _State(replace(child.cfg.run, nx=cnx, ny=cny))


def test_tile_attachment_windows_owned_sides_and_zeros_the_seams():
    """Owned sides are tangential slices of the domain tables, contiguous;
    seam sides are inert zeros -- deliberately not the domain's data, so a
    seam that reached a tile interior would show in the bit-identity gate.
    """
    child = _attached_child()
    spec = _tspec(child, ci0=0, cj0=6, cnx=12, cny=12)   # west edge tile
    tile = _tile_state(child, 12, 12)
    nest_stream.attach_streaming_nest_boundaries(
        tile, child.state, spec, child.clock)

    source = child.state._lateral_boundary_device.intervals[0].fields
    mine = tile._lateral_boundary_device.intervals[0].fields
    # The attach itself copies nothing (generation 0); the LAUNCH does.
    interval, dtbc, dt, spec_exp = _active_device_interval(
        tile, child.cfg.run)
    theta = interval.fields["theta"]
    expect_west = source["theta"].west.value[:, 6:18, :]
    assert np.array_equal(theta.west.value, expect_west)
    assert theta.west.value.flags["C_CONTIGUOUS"], (
        "the packed table is not contiguous; a raw CUDA kernel handed "
        "this would read the wrong bytes silently")
    # u is x-staggered: its west table gains no tangential face, its
    # south table gains one on x -- but this tile owns neither of those
    # seams, which must be zeros.
    assert theta.east.value.shape == (2, 12, 5)
    assert not theta.east.value.any()
    assert not theta.north.tendency.any()
    u_south = interval.fields["u"].south.value
    assert u_south.shape == (2, 5, 13)
    assert not u_south.any()
    assert mine["mu"].west.value.shape == (1, 12, 5)
    assert dt == child.clock.spec.dt_fp32


def test_launch_time_reload_tracks_the_rolling_generation():
    """A FORCE moves the generation; the NEXT launch re-copies.  A buffer
    that never changes tiles (its hook fired once) is therefore
    structurally unable to serve a previous interval's forcing.
    """
    child = _attached_child()
    spec = _tspec(child, ci0=0, cj0=0, cnx=12, cny=12)
    tile = _tile_state(child, 12, 12)
    nest_stream.attach_streaming_nest_boundaries(
        tile, child.state, spec, child.clock)
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    first = interval.fields["theta"].west.value.copy()
    assert tile._lateral_boundary_device.rolling_generation == \
        child.state._lateral_boundary_device.rolling_generation

    # FORCE N+1: new tables, generation bumps, launch re-copies.
    attach_nest_boundaries(child.state,
                           _rolling_fields(child.state, seed=8),
                           clock=child.clock, spec_bdy_width=5,
                           spec_zone=1, relax_zone=4)
    assert child.state._lateral_boundary_device.rolling_generation == 2
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    second = interval.fields["theta"].west.value
    assert not np.array_equal(first, second)
    assert tile._lateral_boundary_device.external_reload_count == 2

    # Same generation, no copy: the reload is a comparison, not a tax.
    before = tile._lateral_boundary_device.external_reload_count
    _active_device_interval(tile, child.cfg.run)
    assert tile._lateral_boundary_device.external_reload_count == before


def test_the_stale_tables_control_disarms_exactly_the_copy(monkeypatch):
    """The negative control the GPU gate uses, proven able to fire here.

    With ``_copy_owned_sides`` disarmed the buffer keeps serving FORCE-1
    tables while the generation claims currency -- the exact silent
    failure the launch-time reload exists to prevent, made visible on
    demand so the gate's PASS rows are not vacuous.
    """
    child = _attached_child()
    spec = _tspec(child, ci0=0, cj0=0, cnx=12, cny=12)
    tile = _tile_state(child, 12, 12)
    nest_stream.attach_streaming_nest_boundaries(
        tile, child.state, spec, child.clock)
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    first = interval.fields["theta"].west.value.copy()

    monkeypatch.setattr(nest_stream, "_copy_owned_sides",
                        lambda specs, source: 0)
    attach_nest_boundaries(child.state,
                           _rolling_fields(child.state, seed=9),
                           clock=child.clock, spec_bdy_width=5,
                           spec_zone=1, relax_zone=4)
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    assert np.array_equal(interval.fields["theta"].west.value, first), (
        "the disarmed control still refreshed the tables; the GPU gate's "
        "stale-tables leg would be measuring nothing")


def test_tile_attachment_refuses_before_the_first_force():
    """Ordering is an executor guarantee AND a named refusal, not a hope."""
    _parent, child = _nodes()
    spec = _tspec(child, ci0=0, cj0=0, cnx=12, cny=12)
    tile = _tile_state(child, 12, 12)
    with pytest.raises(RuntimeError, match="before NestCoupler.force"):
        nest_stream.attach_streaming_nest_boundaries(
            tile, child.state, spec, child.clock)


def test_the_hook_is_the_attachment_bound_to_the_node():
    child = _attached_child()
    hook = nest_stream.make_nest_tile_hook(child)
    spec = _tspec(child, ci0=18, cj0=18, cnx=12, cny=12)  # east/north tile
    tile = _tile_state(child, 12, 12)
    hook(tile, spec, 3, stream=None)
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    source = child.state._lateral_boundary_device.intervals[0].fields
    assert np.array_equal(
        interval.fields["phi"].east.value,
        source["phi"].east.value[:, 18:30, :])
    assert not interval.fields["phi"].west.value.any()


def test_retarget_preserves_the_weights_cache_and_forces_a_recopy():
    """A buffer moving to a new tile keeps its Davies weights and cannot
    keep the old tile's forcing: generation resets to 0, below any real
    generation, so the next launch copies unconditionally.
    """
    child = _attached_child()
    spec_a = _tspec(child, ci0=0, cj0=0, cnx=12, cny=12)
    tile = _tile_state(child, 12, 12)
    nest_stream.attach_streaming_nest_boundaries(
        tile, child.state, spec_a, child.clock)
    _active_device_interval(tile, child.cfg.run)
    tile._lateral_boundary_device.weights["probe"] = ("f", "g")

    spec_b = _tspec(child, ci0=6, cj0=6, cnx=12, cny=12)  # interior tile
    nest_stream.attach_streaming_nest_boundaries(
        tile, child.state, spec_b, child.clock)
    mine = tile._lateral_boundary_device
    assert mine.weights.get("probe") == ("f", "g")
    assert mine.rolling_generation == 0
    interval, *_rest = _active_device_interval(tile, child.cfg.run)
    # An interior tile owns nothing: every side inert.
    assert not interval.fields["theta"].west.value.any()
    assert not interval.fields["v"].north.value.any()


# ---------------------------------------------------------------------------
# roads and budgets
# ---------------------------------------------------------------------------

def test_attach_refuses_tables_and_hook_together():
    with pytest.raises(streaming.StreamingRefused, match="one of these"):
        streaming.attach(
            object(), SimpleNamespace(nx=4, ny=4, nz=2),
            streaming.StreamingDecision(
                True, "pinned", 2, 2, 2, 4, "host", "ring"),
            tile_state_factory=lambda cfg: None,
            boundary_tables=[object()], tile_hook=lambda *a: None)


def test_corridor_claim_prices_the_slots_and_the_streamed_packing():
    """Resident child: rolling + geometry only (the two full-field slots
    alias dead RK backings already inside its resident price).  Streamed
    child: everything, plus the per-buffer packed table windows.
    """
    _parent, child = _nodes()
    node = SimpleNamespace(cfg=child.cfg, parent=child.parent)
    resident = nest_stream.corridor_claim_bytes(node, decision=None)
    streamed = nest_stream.corridor_claim_bytes(
        node, decision=streaming.StreamingDecision(
            True, "test", 12, 12, 2, 4, "host", "ring"))
    from gpuwm.core.preflight import nest_slot_shapes
    shapes = nest_slot_shapes(child.cfg, child.cfg.run.spec_bdy_width,
                              child.parent.cfg)
    field_slots = 4 * (int(np.prod(shapes["nest_parent_field"]))
                       + int(np.prod(shapes["nest_child_field"])))
    assert resident > 0
    assert streamed > resident
    assert streamed - resident >= field_slots


def test_the_tree_walk_prices_each_domain_against_what_is_left(monkeypatch):
    """Per-domain road assignment: the second decision sees the budget the
    first one left, and every decision's receipt names its road, claim and
    the corridor's.
    """
    from tilestream.autoplan import Machine

    parent, child = _nodes()
    parent.parent = None
    model = SimpleNamespace(walk_parent_first=lambda: iter([parent, child]))
    machine = Machine(vram_bytes=10 * 2**30, host_bytes=32 * 2**30,
                      vram_headroom=0.0)

    seen = []
    resident_claim = 6 * 2**30

    def fake_decide(cfg, options, machine=None):
        seen.append(machine)
        return streaming.StreamingDecision(
            False, "scripted", resident_bytes=resident_claim)

    monkeypatch.setattr(streaming, "decide", fake_decide)
    # The walk is under test, not the stepper: binding a resident stepper
    # imports the dycore, whose module scope needs cupy, and this module
    # is CPU-hermetic.
    monkeypatch.setattr(streaming, "make_stepper",
                        lambda *a, **k: object())
    decisions = {}
    options = streaming.StreamingOptions(mode="auto")
    out = streaming.steppers_for_tree(model, options, machine=machine,
                                      decisions=decisions)
    assert out == {}
    corridor = nest_stream.corridor_claim_bytes(
        child, decision=decisions[2])
    # The claim a domain leaves behind is MARGINAL: the scripted 6 GiB is a
    # whole-process price and carries the CUDA context and the rung's
    # tables, which the tree pays ONCE and the second domain must not be
    # charged for again.  ``budget_spent_before_bytes`` moves with it.
    overhead = streaming._process_overhead_bytes(parent)
    marginal = resident_claim - overhead
    assert overhead > 0
    assert seen[0].vram_bytes == machine.vram_budget_bytes
    assert seen[1].vram_bytes == machine.vram_budget_bytes - marginal
    assert seen[1].vram_headroom == 0.0
    assert decisions[1].detail["road"] == "resident"
    assert decisions[1].detail["claim_bytes"] == marginal
    assert decisions[1].detail["corridor_claim_bytes"] == 0
    assert decisions[2].detail["budget_spent_before_bytes"] == marginal
    assert decisions[2].detail["corridor_claim_bytes"] == corridor > 0


def test_a_pinned_tiling_walk_records_roads_and_probes_no_card(monkeypatch):
    """The bit-exactness gates pin their tiling; the walk must not detect
    a machine for them, and the roads receipt still lands.
    """
    from tilestream import autoplan

    parent, child = _nodes()
    parent.parent = None
    model = SimpleNamespace(walk_parent_first=lambda: iter([parent, child]))
    monkeypatch.setattr(
        autoplan.Machine, "detect",
        classmethod(lambda cls, **k: pytest.fail(
            "a pinned tiling consulted the card")))

    def fake_decide(cfg, options, machine=None):
        return streaming.StreamingDecision(False, "scripted")

    monkeypatch.setattr(streaming, "decide", fake_decide)
    monkeypatch.setattr(streaming, "make_stepper",
                        lambda *a, **k: object())
    decisions = {}
    options = streaming.StreamingOptions(mode="on", tile_nx=12, tile_ny=12)
    streaming.steppers_for_tree(model, options, decisions=decisions)
    assert decisions[1].detail["road"] == "resident"
    assert decisions[2].detail["budget_spent_before_bytes"] == 0
