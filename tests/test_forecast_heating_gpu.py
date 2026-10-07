"""[radar_heating] on a CUDA device: the box acceptance for the door hook.

Run on the box, never on a desktop card::

    pytest -q tests/test_radar_tten_lane6.py tests/test_radar_tten_grid.py \\
        tests/test_forecast_heating_gpu.py

(a) the hook's attach gives theta byte-identical to a direct
    ``build_forcing`` + ``attach`` on the same state, over 30 model steps;
(b) the multi-card road (gather, build on one card, scatter slabs) gives
    every rank exactly the one-card slots for its window, and one forced
    microphysics call gives the one-card theta at every column of every
    slab, seams and the specified ring included.  Ranks go on cards 0 and 1
    when two exist, else both on card 0;
(b2) with ``GPUWM_HEAT_BOX_DOOR`` set (see :func:`_door_case`), the real
    door runs the same case on the card counts the case names (2 and 4 for
    HRRR's full grid, which one card cannot hold) and every history
    variable matches byte for byte;
(c) a manifest taken while the forcing is attached is legal and carries
    none of it, a resume inside the forced period refuses, and a resume
    after it matches the uninterrupted run byte for byte;
(d) after ``active_minutes`` every step is a pass-through (calls counted,
    theta equals the unforced continuation).

The synthetic cases use the small moist Kessler state of
``tests/test_radar_tten_forcing.py`` and a synthetic prepared root whose
grid identity the windows are written against, so the strict reader and
the identity check are on the path exactly as in a real run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RadarHeatingConfig
from gpuwm.da import forecast_heating as fh
from gpuwm.da import radar_tten

pytestmark = [pytest.mark.gpu, requires_gpu]

START = datetime(2026, 10, 1, 18, 0, 0)
NX, NY, NZ = 40, 20, 30


def _state(**kwargs):
    """The Kessler state with specified lateral boundaries ATTACHED.

    ``specified=True`` puts the specified ring on the path, as the real arms'
    RAP boundaries do, and a specified state steps only with boundaries
    attached (``apply_state_lateral_boundaries``).  A real run's door has
    them already: ``restore_prepared_cache`` attaches the prepared tables
    before the heating hook attaches, and the hook never touches them.  The
    fixture does the same, with the state's own frame held over the hour.
    """
    from test_radar_tten_forcing import _moist_state
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)
    state, cfg = _moist_state(nx=NX, ny=NY, nz=NZ, mp=1, dt=30.0, specified=True, **kwargs)
    attach_lateral_boundaries(state, build_state_lateral_boundaries(
        [state, state], [0.0, 3600.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone))
    return state, cfg


def _prepared_root(tmp_path):
    from test_forecast_heating import _prepared_root as build
    return build(tmp_path, nz=NZ, ny=NY, nx=NX)


def _windows(root_dir, identity, heating):
    """NOAA-convention windows: a storm that moves east, clear air beside
    it, no coverage elsewhere."""
    rng = np.random.default_rng(11)
    for n, end_minutes in enumerate(heating.window_end_minutes()):
        end = START + timedelta(minutes=end_minutes)
        directory = Path(root_dir) / end.strftime("%Y%m%dT%H%MZ")
        directory.mkdir(parents=True, exist_ok=True)
        field = np.full((NZ, NY, NX), -99999.0, dtype="<f4")
        field[:, 2:18, 2:38] = -99.0
        i0 = 6 + 4 * n
        field[0:14, 5:15, i0:i0 + 10] = rng.uniform(20.0, 55.0, (14, 10, 10))
        field.tofile(directory / "ref.f32")
        raw = (directory / "ref.f32").read_bytes()
        (directory / "ref.json").write_text(json.dumps({
            "schema": "gpuwm-obs.radar-tten-ref.v1", "status": "READY",
            "shape": [NZ, NY, NX],
            "data": {"file": "ref.f32", "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest()},
            "grid": {"identity_sha256": identity},
            "window": {"end": end.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "source": {"lead_class": "forecast", "id": "synthetic"}}))


def _case(tmp_path, *, window_minutes=1.0, active_minutes=2.0):
    root = _prepared_root(tmp_path)
    heating = RadarHeatingConfig(windows=str(tmp_path / "windows"),
                                 window_minutes=window_minutes,
                                 active_minutes=active_minutes)
    _windows(tmp_path / "windows", fh.prepared_grid_identity(root), heating)
    return root, heating


def _node(state, cfg):
    return SimpleNamespace(state=state, cfg=SimpleNamespace(run=cfg, grid_id=1))


def _bytes(array):
    import cupy as cp
    return cp.asnumpy(array).tobytes()


# ---------------------------------------------------------------------- (a)

def test_a_the_hook_matches_a_direct_attach_byte_for_byte(tmp_path):
    from gpuwm.core.dycore import step

    root, heating = _case(tmp_path, window_minutes=5.0, active_minutes=15.0)
    hooked, cfg = _state()
    direct, _ = _state()
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.attach(SimpleNamespace(), _node(hooked, cfg))
    assert hook.road == "resident"

    import cupy as cp
    refs = [cp.asarray(radar_tten.read_window_host(
        row["path"], identity_sha256=hook.identity)[0]) for row in hook.windows]
    forcing = radar_tten.build_forcing(
        direct, refs, heating.window_end_minutes(), fh.radar_config(heating),
        mp_tend_lim=heating.mp_tend_lim, active_minutes=heating.active_minutes)
    radar_tten.attach(direct, forcing, cfg)
    for _ in range(30):
        step(hooked, cfg)
        step(direct, cfg)
    hook.detach()
    radar_tten.detach(direct)
    assert _bytes(hooked.thp) == _bytes(direct.thp)
    receipt = hook.receipt()
    assert receipt["forcing"]["calls_by_slot"] == forcing.calls_by_slot
    assert receipt["applied_calls"] == 30
    hook.require_applied()


# ---------------------------------------------------------------------- (b)

def _ranks(full, halo=4):
    """Two x-slabs of ``full`` with halos, as resident rank states."""
    import cupy as cp

    devices = [0, 1] if cp.cuda.runtime.getDeviceCount() > 1 else [0, 0]
    half = NX // 2
    specs = [
        SimpleNamespace(index=(0, 0), j0=0, j1=NY, i0=0, i1=half, cj0=0, ci0=0,
                        cny=NY, cnx=half + halo, periodic_x=False, periodic_y=False),
        SimpleNamespace(index=(0, 1), j0=0, j1=NY, i0=half, i1=NX, cj0=0,
                        ci0=half - halo, cny=NY, cnx=NX - half + halo,
                        periodic_x=False, periodic_y=False),
    ]
    tiles = []
    for spec, dev in zip(specs, devices):
        with cp.cuda.Device(dev):
            def cut(array):
                if array.ndim == 1:
                    return cp.asarray(cp.asnumpy(array))
                return cp.asarray(np.ascontiguousarray(cp.asnumpy(
                    array[:, spec.cj0:spec.cj0 + spec.cny,
                          spec.ci0:spec.ci0 + spec.cnx])))
            tiles.append(SimpleNamespace(**{name: cut(getattr(full, name)) for name in (
                "thb", "thp", "p", "phb", "php", "qv")}))
    return SimpleNamespace(ranked=True, specs=specs, devices=devices, tiles=tiles)


def test_b_one_card_and_two_card_heating_are_byte_identical(tmp_path):
    import cupy as cp

    root, heating = _case(tmp_path, window_minutes=5.0, active_minutes=15.0)
    resident, cfg = _state()
    one = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    one.attach(SimpleNamespace(), _node(resident, cfg))
    (_, whole, _), = one._attached

    shadow, _ = _state()
    run = _ranks(shadow)
    run.sub_cfgs = [cfg, cfg]
    two = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    two.attach(SimpleNamespace(), _node(shadow, cfg),
               steppers={1: SimpleNamespace(ranked=True, tiled_run=run)})
    assert two.road == "ranks"
    for tile, forcing, rank in two._attached:
        spec = run.specs[rank]
        assert forcing.extent == (spec.cj0, spec.ci0, NY, NX)
        for mine, theirs in zip(forcing.slots, whole.slots):
            expected = cp.asnumpy(theirs)[:, spec.cj0:spec.cj0 + spec.cny,
                                          spec.ci0:spec.ci0 + spec.cnx]
            assert cp.asnumpy(mine).tobytes() == np.ascontiguousarray(expected).tobytes()

    # One forced microphysics call: the same "scheme" increment everywhere,
    # then the tendency where covered, with the ring at the DOMAIN's edge.
    increment = np.random.default_rng(2).normal(0.0, 0.05, resident.thp.shape).astype(np.float32)
    whole.before_microphysics(resident, cfg, cfg.dt)
    resident.thp += cp.asarray(increment)
    whole.after_microphysics(resident, cfg, cfg.dt, ring_width=1)
    reference = cp.asnumpy(resident.thp)
    for tile, forcing, rank in two._attached:
        spec = run.specs[rank]
        window = (slice(None), slice(spec.cj0, spec.cj0 + spec.cny),
                  slice(spec.ci0, spec.ci0 + spec.cnx))
        with cp.cuda.Device(run.devices[rank]):
            forcing.before_microphysics(tile, cfg, cfg.dt)
            tile.thp += cp.asarray(np.ascontiguousarray(increment[window]))
            forcing.after_microphysics(tile, cfg, cfg.dt, ring_width=1)
            got = cp.asnumpy(tile.thp)
        assert got.tobytes() == np.ascontiguousarray(reference[window]).tobytes(), rank
    two.detach()
    one.detach()
    assert all(not hasattr(t, radar_tten.SLAB_ATTRIBUTE) for t in run.tiles)


def _door_case():
    """``GPUWM_HEAT_BOX_DOOR``: a JSON file on the box naming a prepared
    HRRR-physics case, a heating table and the card counts to compare::

        {"argv": [... the door's arguments without --outdir/--devices-table,
                  with --run-seconds ...],
         "radar_heating": {...}, "cards": [2, 4], "out": "/work/nh/gpu-test"}

    Keep the run short and the forced period shorter than it (for example
    ``window_minutes = 5``, ``active_minutes = 10`` and ``--run-seconds
    1200``), so both the forced steps and the pass-through are compared.
    One card cannot hold HRRR's full grid (about 119 GiB), so ``cards`` is
    required rather than defaulting to 1 and 2.
    """
    path = os.environ.get("GPUWM_HEAT_BOX_DOOR")
    if not path:
        pytest.skip("GPUWM_HEAT_BOX_DOOR names no prepared box case")
    case = json.loads(Path(path).read_text())
    # Without --run-seconds the door runs the prepared length (18 h for the
    # campaign's preparations) once per card count: hours of rented cards
    # for a byte comparison that minutes settle.
    if "--run-seconds" not in case.get("argv", ()):
        pytest.fail(f"{path}: argv carries no --run-seconds; the door would "
                    "run the whole prepared forecast once per card count")
    cards = [int(c) for c in case.get("cards", ())]
    if len(set(cards)) < 2 or min(cards, default=0) < 1:
        pytest.fail(f"{path}: cards {case.get('cards')!r} must name at least "
                    "two different card counts to compare")
    case["cards"] = cards
    return case


def test_b2_the_door_on_one_and_two_cards_writes_the_same_history():
    import cupy as cp
    from gpuwm import netcdf_bridge

    case = _door_case()
    if cp.cuda.runtime.getDeviceCount() < max(case["cards"]):
        pytest.skip(f"needs {max(case['cards'])} cards")
    outs = []
    for count in case["cards"]:
        out = Path(case["out"]) / f"cards{count}"
        devices = {"count": count, "ids": list(range(count)), "transport": "auto"}
        command = [sys.executable, "-m", "gpuwm.prepared_single_domain_forecast",
                   *case["argv"], "--outdir", str(out),
                   "--devices-table", json.dumps(devices),
                   "--radar-heating-table", json.dumps(case["radar_heating"])]
        # A hung door would hold the cards until the box's dead-man fires.
        subprocess.run(command, check=True,
                       timeout=float(case.get("timeout_seconds", 3600)))
        outs.append(out)
    for count, out in zip(case["cards"], outs):
        report = json.loads((out / "report.json").read_text())
        heating = report["radar_heating"]
        assert heating["road"] == ("resident" if count == 1 else "ranks"), count
        assert heating["ranks_agree"] is True
        assert heating["forcing"]["calls_after_active"] > 0, (
            "the run must outlast the forced period so the pass-through is compared")
    # The door lands its history under <outdir>/wrfout/ (the folder that
    # outlives the run directory); globbing <outdir> itself found nothing
    # and failed the gate after a comparison that passes (box S, 2026-10-06).
    frames = sorted(p.name for p in (outs[0] / "wrfout").glob("wrfout_d01_*"))
    for out in outs[1:]:
        assert frames and frames == sorted(
            p.name for p in (out / "wrfout").glob("wrfout_d01_*"))
        for name in frames:
            with netcdf_bridge.open_dataset(outs[0] / "wrfout" / name) as a, \
                    netcdf_bridge.open_dataset(out / "wrfout" / name) as b:
                assert set(a.variables) == set(b.variables), name
                for variable in a.variables:
                    left = np.asarray(a.variables[variable][...])
                    right = np.asarray(b.variables[variable][...])
                    assert left.tobytes() == right.tobytes(), (out.name, name, variable)


# ------------------------------------------------------------------ (c), (d)

def test_c_a_checkpoint_inside_the_forced_period_is_legal_and_carries_no_forcing(tmp_path):
    from gpuwm.io import restart

    root, heating = _case(tmp_path)
    state, cfg = _state()
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.attach(SimpleNamespace(), _node(state, cfg))
    assert getattr(state, radar_tten.STATE_ATTRIBUTE) is not None
    manifest = restart.state_manifest(state)
    assert not any("radar_tten" in str(key) for key in manifest)
    hook.detach()
    with pytest.raises(fh.ForecastHeatingRefused, match="inside the forced period"):
        fh.ForecastHeating(heating, start_time=START, prepared_root=root,
                           restored_seconds=60.0)


def test_c_d_a_resume_after_the_forced_period_matches_and_steps_pass_through(tmp_path):
    from gpuwm.core.dycore import step

    root, heating = _case(tmp_path, window_minutes=1.0, active_minutes=2.0)
    # The uninterrupted run: 4 forced steps (2 min at 30 s), then 6 free.
    straight, cfg = _state()
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.attach(SimpleNamespace(), _node(straight, cfg))
    for _ in range(10):
        step(straight, cfg)
    hook.detach()
    receipt = hook.receipt()
    assert sum(receipt["forcing"]["calls_by_slot"]) == 4
    assert receipt["forcing"]["calls_after_active"] == 6

    # The same run stopped at the end of the forced period and resumed:
    # the resume attaches nothing and lands on the same bytes.  The state
    # carries over in place; that a checkpoint restores a state to the bit
    # is the restart writer's own contract, and the door-level resume is
    # the box case's (b2) to run end to end.
    first, _ = _state()
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.attach(SimpleNamespace(), _node(first, cfg))
    for _ in range(4):
        step(first, cfg)
    hook.detach()
    resumed = first
    after = fh.ForecastHeating(heating, start_time=START, prepared_root=root,
                               restored_seconds=120.0)
    model = SimpleNamespace()
    after.attach(model, _node(resumed, cfg))
    assert getattr(model, fh.ROUTE_ATTRIBUTE) is after
    assert getattr(resumed, radar_tten.STATE_ATTRIBUTE, None) is None
    for _ in range(6):
        step(resumed, cfg)
    assert _bytes(resumed.thp) == _bytes(straight.thp)

    # (d) the free steps of the uninterrupted run are the unforced
    # continuation of its forced state.
    free, _ = _state()
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.attach(SimpleNamespace(), _node(free, cfg))
    for _ in range(4):
        step(free, cfg)
    hook.detach()
    for _ in range(6):
        step(free, cfg)
    assert _bytes(free.thp) == _bytes(straight.thp)
