# tests/test_uh_lifecycle.py
"""UP_HELI_MAX lifecycle: allocation, emission, reset, restart, inertness.

WRF semantics under test (authorities cited in gpuwm/core/uh_diag.py):
the accumulator exists exactly when nwp_diagnostics = 1, is folded every
step, rides in every history frame, is zeroed after each frame ("reset at
history interval"), is carried through restarts (Registry IO "rh02"), and
-- the load-bearing property -- is provably trajectory-inert: a run with
the knob on is bitwise identical to the same run with it off in every
model field, differing only by the UP_HELI_MAX plane itself.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.uh_diag import reset_up_heli_max, update_up_heli_max
from gpuwm.io import restart

from test_restart import _cfg, _fill_setup, _shim_state

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CPU tier
# ---------------------------------------------------------------------------

def test_state_allocates_the_accumulator_eagerly_under_the_knob(monkeypatch):
    on = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    assert on.existing_scratch("up_heli_max") is not None
    assert on.existing_scratch("up_heli_max").shape == (4, 6)
    off = _shim_state(_cfg(), monkeypatch)
    assert off.existing_scratch("up_heli_max") is None


def test_reset_zeroes_and_tolerates_an_absent_slot(monkeypatch):
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state.existing_scratch("up_heli_max")[...] = 123.0
    reset_up_heli_max(state)
    assert (state.existing_scratch("up_heli_max") == 0.0).all()
    # No slot (knob off) is a clean no-op, not an error.
    reset_up_heli_max(_shim_state(_cfg(), monkeypatch))


def test_wrfout_meta_and_shared_field_builder_carry_up_heli_max(monkeypatch):
    from gpuwm.io.wrfout import _VAR_META, _live_state_history_fields

    # WRF Registry.EM_COMMON:2083 attributes, verbatim.
    assert _VAR_META["UP_HELI_MAX"] == ("MAX UPDRAFT HELICITY", "m2 s-2")
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    buf = state.existing_scratch("up_heli_max")
    buf[...] = 7.5
    fields = _live_state_history_fields(state)
    assert fields["UP_HELI_MAX"] is buf     # live view, snapshot at write
    off = _shim_state(_cfg(), monkeypatch)
    assert "UP_HELI_MAX" not in _live_state_history_fields(off)


def test_tree_history_handler_snapshots_then_resets(monkeypatch):
    """Reset-on-write: the production tree handler zeroes the accumulator
    AFTER the frame submit observed the accumulated values."""
    from gpuwm.runtime import _submit_tree_history_frame

    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state.existing_scratch("up_heli_max")[...] = 42.0
    seen = {}

    class _Writers:
        def submit(self, node, ticks, *, refl_field=None):
            seen["at_submit"] = node.state.existing_scratch(
                "up_heli_max").copy()

    node = SimpleNamespace(state=state)
    _submit_tree_history_frame(_Writers(), node, 0)
    assert (seen["at_submit"] == 42.0).all()
    assert (state.existing_scratch("up_heli_max") == 0.0).all()


def test_restart_roundtrip_carries_the_running_max(monkeypatch, tmp_path):
    cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    pattern = np.arange(24, dtype=np.float32).reshape(4, 6) * np.float32(1.5)
    source.existing_scratch("up_heli_max")[...] = pattern
    path = restart.write_restart(tmp_path / "uh.npz", source, cfg)

    with np.load(path, allow_pickle=False) as archive:
        assert "scratch/up_heli_max" in archive.files

    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(
        fresh.existing_scratch("up_heli_max"), pattern)


def test_old_checkpoint_without_the_slot_restores_zeroed_with_a_note(
        monkeypatch, tmp_path, capsys):
    """A checkpoint predating the diagnostic restores under
    nwp_diagnostics=1 with the accumulator zeroed -- a note, NOT a
    refusal.  The header written by the old build also lacked the config
    key entirely, which the diagnostic-field echo exemption covers."""
    import json

    old_cfg = _cfg()
    source = _shim_state(old_cfg, monkeypatch)
    _fill_setup(source)
    path = restart.write_restart(tmp_path / "old.npz", source, old_cfg)

    # Age the header: strip the config key a pre-knob build never wrote.
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
        assert "scratch/up_heli_max" not in archive.files
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    del header["config"]["nwp_diagnostics"]
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    aged = tmp_path / "pre-knob.npz"
    with aged.open("wb") as stream:
        np.savez(stream, **payload)

    live_cfg = _cfg(nwp_diagnostics=1)
    live = _shim_state(live_cfg, monkeypatch)
    _fill_setup(live)
    live.existing_scratch("up_heli_max")[...] = 99.0   # must be wiped
    restart.restore_restart(aged, live, live_cfg)
    assert (live.existing_scratch("up_heli_max") == 0.0).all()
    out = capsys.readouterr().out
    assert "note:" in out and "up_heli_max" in out
    assert "zero-initialized" in out


def test_diagnostics_off_resume_drops_the_stored_accumulator_with_a_note(
        monkeypatch, tmp_path, capsys):
    on_cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(on_cfg, monkeypatch)
    _fill_setup(source)
    source.existing_scratch("up_heli_max")[...] = 3.25
    path = restart.write_restart(tmp_path / "on.npz", source, on_cfg)

    off_cfg = _cfg()
    live = _shim_state(off_cfg, monkeypatch)
    _fill_setup(live)
    restart.restore_restart(path, live, off_cfg)
    assert live.existing_scratch("up_heli_max") is None
    out = capsys.readouterr().out
    assert "note:" in out and "dropped" in out


def test_launcher_refuses_unsupported_geometry_and_tiny_grids(monkeypatch):
    # Periodic idealized geometry: WRF's periodic cal_helicity branch is
    # not transcribed; refusing beats guessing.
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    with pytest.raises(NotImplementedError, match="specified/nested"):
        update_up_heli_max(state, _cfg(nwp_diagnostics=1))
    # Stencil floor.
    tiny = _cfg(nwp_diagnostics=1, specified=True, nx=3, ny=8, nz=8)
    with pytest.raises(ValueError, match="nx, ny >= 4"):
        update_up_heli_max(_shim_state(tiny, monkeypatch), tiny)


def _mesocyclone_winds(ny, nx, nz):
    """Coherent cyclonic rotation around (4.5, 3.5) plus westerly shear."""
    xs = np.arange(nx + 1, dtype=np.float32) - 0.5
    ys = np.arange(ny, dtype=np.float32)
    r2u = (xs[None, :] - 4.5)**2 + (ys[:, None] - 3.5)**2
    u = (8.0 - 12.0 * np.exp(-r2u / 6.0) * (ys[:, None] - 3.5)
         )[None].repeat(nz, axis=0).astype(np.float32)
    xs = np.arange(nx, dtype=np.float32)
    ys = np.arange(ny + 1, dtype=np.float32) - 0.5
    r2v = (xs[None, :] - 4.5)**2 + (ys[:, None] - 3.5)**2
    v = (2.0 + 12.0 * np.exp(-r2v / 6.0) * (xs[None, :] - 4.5)
         )[None].repeat(nz, axis=0).astype(np.float32)
    return u, v


def _loaded_numpy_state(monkeypatch, **overrides):
    from gpuwm.core.grid import make_base_state, make_vertical_coord

    cfg = _cfg(nx=9, ny=8, nz=24, ztop=12000.0, specified=True,
               nwp_diagnostics=1, **overrides)
    state = _shim_state(cfg, monkeypatch)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state.load_base(vc, base)
    return state, cfg


def test_numpy_state_path_accumulates_and_matches_the_mirror(monkeypatch):
    """update_up_heli_max on a NumPy state == the parity-gated mirror, and
    the running max holds when a later step weakens."""
    from gpuwm.core.uh_diag import mirror_up_heli_max_step_np

    state, cfg = _loaded_numpy_state(monkeypatch)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    z_w = (np.asarray(state.phb, dtype=np.float32)
           / np.float32(9.81))[:, None, None]
    blob = np.exp(-(((np.arange(nx) - 4.5)[None, :])**2
                    + ((np.arange(ny) - 3.5)[:, None])**2) / 4.0)
    wz = np.exp(-(((z_w - 3500.0) / 2000.0)**2))
    strong = (18.0 * blob[None, :, :] * wz).astype(np.float32)
    state.u[...], state.v[...] = _mesocyclone_winds(ny, nx, nz)

    mirror_max = np.zeros((ny, nx), dtype=np.float32)

    def mirror_step():
        mirror_up_heli_max_step_np(
            state.u, state.v, state.w, state.php, state.phb,
            state.msfu, state.msfv, state.ht,
            state.dn, state.dnw, state.fnm, state.fnp,
            state.cf1, state.cf2, state.cf3,
            np.float32(1.0) / np.float32(cfg.dx),
            np.float32(1.0) / np.float32(cfg.dy), mirror_max)

    state.w[...] = strong
    update_up_heli_max(state, cfg)
    mirror_step()
    first = state.existing_scratch("up_heli_max").copy()
    np.testing.assert_array_equal(first, mirror_max)
    assert first.max() > 10.0, "fixture updraft produced no UH"

    state.w[...] = np.float32(0.4) * strong     # weaker step: max holds
    update_up_heli_max(state, cfg)
    mirror_step()
    second = state.existing_scratch("up_heli_max")
    np.testing.assert_array_equal(second, mirror_max)
    np.testing.assert_array_equal(second, first)

    # Reset starts a new window: the weak step alone now sets a smaller max.
    reset_up_heli_max(state)
    update_up_heli_max(state, cfg)
    third = state.existing_scratch("up_heli_max")
    assert third.max() < first.max()
    assert third.max() > 0.0


def test_up_heli_max_has_no_trajectory_or_restart_reader():
    """The accumulator can feed output only, never the model.

    Grep pin: the exact repository files allowed to mention the slot (or
    the update/reset entry points whose names embed it), plus an AST pin
    that no module outside the owner reads the slot through the scratch
    APIs except the sanctioned allocation/emission/restore sites.
    """
    gpuwm = REPO / "gpuwm"
    hits = {
        path.relative_to(REPO).as_posix()
        for path in gpuwm.rglob("*.py")
        if "up_heli_max" in path.read_text(encoding="utf-8")
    }
    assert hits == {
        "gpuwm/core/uh_diag.py",      # owner: update/reset/mirror
        "gpuwm/core/state.py",        # eager allocation under the knob
        "gpuwm/core/dycore.py",       # calls update_up_heli_max only
        "gpuwm/core/preflight.py",    # pricing registry + lifetime audit
        "gpuwm/runtime.py",           # calls reset_up_heli_max only
        "gpuwm/offline_child_run.py",  # downscale child reset-on-write
        # `gpuwm go`'s runner: reset-on-write, same as the others.  This
        # entry is an ADDITION and the reason is worth the line: for as
        # long as it was absent, this assertion was the thing certifying
        # that the one production runner which never reset the window was
        # behaving correctly.  A list of who may MENTION a slot cannot
        # notice who forgot to USE it, which is what
        # test_every_history_publisher_resets_the_window is for.
        "gpuwm/prepared_single_domain_forecast.py",
        "gpuwm/io/wrfout.py",         # emission
        "gpuwm/io/restart.py",        # serialization + tolerant restore
    }

    # Outside the owner and the sanctioned sites, no scratch-API access to
    # the slot at all; in dycore/runtime specifically, only the two entry
    # points may appear (no direct slot access).
    sanctioned_scratch = {"gpuwm/core/uh_diag.py", "gpuwm/core/state.py",
                          "gpuwm/io/wrfout.py", "gpuwm/io/restart.py"}
    for path in gpuwm.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("scratch", "existing_scratch")
                    and any(isinstance(arg, ast.Constant)
                            and arg.value == "up_heli_max"
                            for arg in node.args)):
                assert rel in sanctioned_scratch, \
                    f"unsanctioned slot access in {rel}"


#: The one delegate a runner may reset THROUGH rather than directly.
#: It is `gpuwm/runtime.py::_submit_tree_history_frame`, which submits
#: the frame and zeroes the window in the same two lines.
_RESET_DELEGATES = ("_submit_tree_history_frame",)


def _history_publishing_modules() -> dict[str, str]:
    """Every gpuwm module that constructs a `PerDomainWrfoutWriters`.

    DERIVED, not listed.  A hand-maintained roster of runners is the
    same instrument as the allowlist above and fails the same way: a new
    runner is simply not on it, so its absence reads as compliance.  The
    question "who publishes durable history frames" already has a
    mechanical answer in the tree -- who builds the writer that publishes
    them -- so the test asks the tree instead of a list.
    """

    found = {}
    for path in (REPO / "gpuwm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None)
            if name == "PerDomainWrfoutWriters":
                found[path.relative_to(REPO).as_posix()] = text
                break
    return found


def test_every_history_publisher_resets_the_window():
    """A runner that writes history frames must restart the UH window.

    WRF zeroes the nwp_diagnostics running maxima every history interval
    (module_diag_nwp.F:246-269).  A runner that publishes frames and
    never resets does not merely lose the reset: every frame after the
    first reports the maximum since MODEL START, so UP_HELI_MAX is a
    published, headline severe-weather field carrying a wrong number
    that grows monotonically worse with run length.

    `gpuwm.prepared_single_domain_forecast` -- the runner behind `gpuwm
    go`, i.e. the one an ordinary run uses -- did exactly that, while
    the tree runner beside it and the two case integrators did it right.
    The gap survived because the only structural test in this file asked
    which files may MENTION the slot, and a file that never mentions it
    passes that question by doing nothing.
    """

    modules = _history_publishing_modules()
    # Non-vacuity: the derivation has to actually find the runners.
    assert "gpuwm/prepared_single_domain_forecast.py" in modules
    assert "gpuwm/prepared_domain_tree_forecast.py" in modules
    assert len(modules) >= 2

    delinquent = [
        name for name, text in modules.items()
        if "reset_up_heli_max" not in text
        and not any(delegate in text for delegate in _RESET_DELEGATES)]
    assert delinquent == [], (
        "history-publishing runner(s) never restart the UP_HELI_MAX "
        f"window: {delinquent}.  Call "
        "gpuwm.core.uh_diag.reset_up_heli_max on the domain state right "
        "after the frame is submitted, as every other publisher does.")


# ---------------------------------------------------------------------------
# GPU tier
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_trajectory_inertness_bitwise(tmp_path):
    """THE load-bearing gate: knob on vs off over real dycore steps leaves
    every model field bitwise identical, THROUGH to actual wrfout files --
    every variable common to both files is byte-identical and the on-file
    differs exactly by carrying UP_HELI_MAX."""
    import cupy as cp
    from netCDF4 import Dataset

    from gpuwm.core.dycore import step
    from gpuwm.io.wrfout import (WrfoutWriter, _live_state_history_fields,
                                 state_frame)

    from test_mp_spec_zone_ring import _square_moist_state

    s_off, cfg_off = _square_moist_state(mp=1, open_x=True, open_y=True)
    s_on, cfg_on = _square_moist_state(mp=1, open_x=True, open_y=True,
                                       nwp_diagnostics=1)
    for _ in range(6):
        step(s_off, cfg_off)
        step(s_on, cfg_on)

    mutated = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
               "qv", "qc", "qr", "h_diabatic")
    for name in mutated:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s_on, name)), cp.asnumpy(getattr(s_off, name)),
            err_msg=f"nwp_diagnostics=1 moved model field {name}")
    # Non-vacuous: the trajectories actually evolved.
    assert float(cp.abs(s_off.w).max()) > 0.0

    on_fields = _live_state_history_fields(s_on)
    off_fields = _live_state_history_fields(s_off)
    assert set(on_fields) - set(off_fields) == {"UP_HELI_MAX"}
    # Precipitation accumulators (scratch) are equally untouched.
    np.testing.assert_array_equal(
        cp.asnumpy(s_on.scratch((cfg_on.ny, cfg_on.nx), "mp_rainnc")),
        cp.asnumpy(s_off.scratch((cfg_off.ny, cfg_off.nx), "mp_rainnc")))

    # File-level proof, the task's own wording: write real wrfouts from
    # both states and byte-compare every common variable.
    paths = {}
    for label, state, cfg in (("off", s_off, cfg_off), ("on", s_on, cfg_on)):
        frame = state_frame(state, include_diagnostic_pressure=True)
        path = tmp_path / f"wrfout_{label}.nc"
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, title="uh-inertness",
                          field_schema=frame) as writer:
            writer.write_frame("1974-01-01_00:00:00", frame)
        paths[label] = path
    with Dataset(paths["off"]) as off_nc, Dataset(paths["on"]) as on_nc:
        off_vars = set(off_nc.variables)
        on_vars = set(on_nc.variables)
        assert on_vars - off_vars == {"UP_HELI_MAX"}
        assert off_vars - on_vars == set()
        for name in sorted(off_vars):
            if name == "Times":
                continue
            np.testing.assert_array_equal(
                on_nc.variables[name][...], off_nc.variables[name][...],
                err_msg=f"wrfout variable {name} differs across the knob")


@pytest.mark.gpu
@requires_gpu
def test_gpu_device_path_matches_the_numpy_state_path():
    """One update on a device state == the same update on the NumPy state
    (the NumPy path is itself pinned to the oracle by the parity gate)."""
    import cupy as cp
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DomainState

    cfg = _cfg(nx=9, ny=8, nz=24, ztop=12000.0, specified=True,
               nwp_diagnostics=1)
    host_state = DomainState(cfg, array_module=np)
    device_state = DomainState(cfg)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    host_state.load_base(vc, base)
    device_state.load_base(vc, base)

    u, v = _mesocyclone_winds(cfg.ny, cfg.nx, cfg.nz)
    z_w = (np.asarray(host_state.phb, dtype=np.float32)
           / np.float32(9.81))[:, None, None]
    w = (16.0 * np.exp(-(((z_w - 3500.0) / 2000.0)**2))
         * np.ones((1, cfg.ny, cfg.nx))).astype(np.float32)
    for state, xp in ((host_state, np), (device_state, cp)):
        state.u[...] = xp.asarray(u)
        state.v[...] = xp.asarray(v)
        state.w[...] = xp.asarray(w)
        update_up_heli_max(state, cfg)
    got = cp.asnumpy(device_state.existing_scratch("up_heli_max"))
    want = host_state.existing_scratch("up_heli_max")
    assert want.max() > 0.0
    np.testing.assert_array_equal(got, want)
