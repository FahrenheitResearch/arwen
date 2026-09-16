"""A cross-scheme microphysics nest edge off a STREAMED parent, on the CPU.

This lived behind a refusal: ``_coupled_parent_field`` raised rather than
run, on the argument that ``launch_microphysics_edge_field`` reads
parent species the coupler could not enumerate and would therefore map the
parent's air AT ATTACH TIME into the edge.  The species list was never
missing -- ``microphysics_transition`` already declares it as the kernel's
own input set -- so the fix is the windowed pull the non-transition arm two
lines below already does, over that list instead of over one field plus
``mup``.  These are the CPU contracts for it.

Why not ``tests/test_nest_coupler.py``: that module is uncollectable
without cupy on a CPU box (its ``from gpuwm.core.dycore import
_boundary_forced`` pulls ``gpuwm/core/dycore.py``, which imports cupy at
module scope), so a test placed there would never run in the CPU gate.
``gpuwm.core.nest``, ``gpuwm.core.microphysics_transition`` and
``gpuwm.core.streaming`` all import cleanly without it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import nest as nest_mod
from gpuwm.core import streaming
from gpuwm.core.clock import DomainClock, DomainTicks
from gpuwm.core.microphysics_transition import (
    MP8_TO_MP18_POLICY, MicrophysicsTransitionContract)
from gpuwm.core.nest import NestCoupler


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
    """A plain-numpy stand-in carrying what the edge arm touches."""

    def __init__(self, run):
        nz, ny, nx = run.nz, run.ny, run.nx
        self.mub2d = np.full((ny, nx), np.float32(5.0))
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
        self.p = np.full((nz, ny, nx), np.float32(9.0e4))
        self.alt = np.full((nz, ny, nx), np.float32(0.9))
        for index, name in enumerate(
                ("qv", "qc", "qr", "qi", "qs", "qg"), start=1):
            setattr(self, name, np.full(
                (nz, ny, nx), np.float32(index / 100.0)))
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
        cfg=pcfg, state=_State(prun),
        clock=_clock(1, 0, step_ticks=3, dt=9.0, advanced=True))
    child_clock = _clock(2, 1, step_ticks=1, dt=3.0)
    child_clock.prepare_step()
    child = SimpleNamespace(cfg=ccfg, state=_State(crun), parent=parent,
                            clock=child_clock)
    return parent, child


def _mixed_coupler(child):
    coupler = NestCoupler(child)
    coupler.microphysics_transition = MicrophysicsTransitionContract(
        source_mp_physics=8, target_mp_physics=18,
        policy_id=MP8_TO_MP18_POLICY, mixed=True)
    return coupler


def _publish(state, arrays):
    """Bind a store the way ``streaming.attach`` does, keyed by member name."""
    setattr(state, streaming._STORE_ATTR,
            {f"state/{name}": value for name, value in arrays.items()})


def _record_launcher(monkeypatch):
    seen = []

    def _fake(contract, parent_state, field_name, *, out, coupled):
        seen.append(float(parent_state.qv[0, 0, 0]))
        return out

    # The RATIFIED spelling, which is what ``_coupled_parent_field`` calls.
    # The ``..._parent_field`` alias is kept for the call sites that have
    # not followed yet; nothing in ``gpuwm/core/nest.py`` is one of them,
    # so a lane that drops the alias cannot break this.
    monkeypatch.setattr(nest_mod, "launch_microphysics_edge_field", _fake)
    return seen


def test_a_streamed_parent_feeds_a_cross_scheme_edge_from_its_store(
        monkeypatch):
    """The store's number reaches the launcher, not the attach-time array.

    RED before the fix with ``RuntimeError: a cross-scheme microphysics
    nest edge (...) off a STREAMED parent is unimplemented``.  The number
    is the point: a run that merely stopped raising, and mapped the frozen
    attachment array anyway, is the wrong answer the refusal was written
    against and would still pass a bare "does not raise".
    """
    parent, child = _nodes()
    coupler = _mixed_coupler(child)
    seen = _record_launcher(monkeypatch)

    resident = float(parent.state.qv[0, 0, 0])
    swept = parent.state.qv.copy() + np.float32(5.0)
    # (0, 0) is INSIDE this geometry's clamped footprint window -- child
    # origin 4, span 10, halo 8 on a 16^2 parent covers the whole parent --
    # so the store's number has to arrive there under a WINDOWED pull; the
    # window-vs-whole-field distinction is controlled separately below.
    _publish(parent.state, {"qv": swept, "mup": parent.state.mup.copy()})

    out = coupler._coupled_parent_field("qv")

    assert out.shape == (2, 16, 16)
    assert seen == [pytest.approx(resident + 5.0)], (
        "the cross-scheme edge read the parent's attach-time air; the "
        "windowed pull did not land")
    assert coupler.force_sync_bytes > 0, (
        "the transition arm moved no bytes through the store seam; the "
        "receipt is dead and the pull cannot be audited")


def test_the_pull_is_the_kernels_own_declared_plane_set():
    """One list, two readers: no second membership table to drift.

    ``edge_parent_planes()`` is what the tile-streamed donor cuts and what
    the coupler pulls.  A plane added to the launcher and to only one of
    those is the mp=9 windowed-edge defect in a second place.
    """
    import inspect

    from gpuwm.core import microphysics_transition as mt
    from gpuwm.core.microphysics_transition import edge_parent_planes

    assert edge_parent_planes() is mt._WINDOWED_EDGE_PLANES
    source = inspect.getsource(NestCoupler._coupled_parent_field)
    assert "edge_parent_planes()" in source, (
        "the transition arm's pull no longer goes through the kernel's "
        "declared plane set")
    assert "unimplemented" not in source


def test_a_resident_parent_still_reads_its_own_arrays_and_moves_no_bytes(
        monkeypatch):
    """Control (i): the resident trajectory is untouched by the fix.

    No store is published, so ``refresh_from_store`` is a no-op and the
    launcher must see the resident value with a zero receipt.  Without
    this, a fix that pulled unconditionally, or that read the store even
    when there is none, would pass the test above.
    """
    parent, child = _nodes()
    coupler = _mixed_coupler(child)
    seen = _record_launcher(monkeypatch)

    resident = float(parent.state.qv[0, 0, 0])
    coupler._coupled_parent_field("qv")

    assert seen == [pytest.approx(resident)]
    assert coupler.force_sync_bytes == 0, (
        "a resident parent paid store traffic it has no store for")


def test_the_pull_is_windowed_and_still_covers_the_childs_footprint(
        monkeypatch):
    """Control (ii): the probe cell sits inside the clamped window.

    Mirrors the reasoning at tests/test_nest_coupler.py's windowed FORCE
    contract.  The store value is planted at a cell the child's footprint
    actually reaches, so a whole-field-vs-window mistake in either
    direction is caught: a pull that skipped the window would miss it, and
    a window computed somewhere else would not contain it.
    """
    parent, child = _nodes()
    coupler = _mixed_coupler(child)
    seen = _record_launcher(monkeypatch)

    window = nest_mod.parent_footprint_window(child.cfg)
    probe = streaming.window_slices(parent.state.qv.shape, window)
    _, jsl, isl = probe
    assert jsl.start <= 0 < jsl.stop and isl.start <= 0 < isl.stop, (
        "this fixture's probe cell fell outside the clamped footprint "
        "window; the control no longer controls anything")

    swept = parent.state.qv.copy()
    swept[:, jsl, isl] += np.float32(5.0)
    _publish(parent.state, {"qv": swept})

    coupler._coupled_parent_field("qv")
    assert seen == [pytest.approx(float(swept[0, 0, 0]))]


def test_a_plane_the_store_does_not_carry_is_skipped_not_refused(
        monkeypatch):
    """An mp8 parent carries no qh/qir/qib and the pull says so quietly.

    ``refresh_from_store`` skips a plane the store does not hold, which is
    why the coupler can name the whole declared set without writing down a
    per-scheme membership table beside the one the transition already has.
    """
    from gpuwm.core.microphysics_transition import edge_parent_planes

    parent, child = _nodes()
    coupler = _mixed_coupler(child)
    seen = _record_launcher(monkeypatch)

    assert not hasattr(parent.state, "qh")
    assert "qh" in edge_parent_planes()
    _publish(parent.state, {"qv": parent.state.qv.copy() + np.float32(5.0)})

    coupler._coupled_parent_field("qv")
    assert seen == [pytest.approx(0.01 + 5.0)]
