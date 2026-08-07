"""Shin-Hong runtime wiring (bl_pbl_physics=11): the Phase-D driver half.

Conformance against the byte-frozen WRF v4.6.1 module lives in
tests/test_shinhong_wrf461_parity.py; this file covers what the DRIVER
does with the scheme -- dispatch, the e_sgs publication seam, the
registered stable-limb non-regression, and the off-path inertness pin
the gray-zone expectation note cites
(docs/superpowers/specs/2026-08-03-shinhong-grayzone-expectation.md).

CPU where possible; the two tests that need a real integration are
gpu-marked and import cupy inside their bodies only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.certify.compile_platform import (
    compile_platform_fingerprint,
    nvrtc_build,
)

#: WRF's shinhonginit cold-start SGS TKE, epsq2l/2 (the oracle driver's
#: documented value, tools/shinhong_wrf461_oracle/run_bl_shinhong.F90:414).
#: state.py fills state.e_sgs with this exact float32 under scheme 11;
#: the cross-check test below gates the pair against the authority.
SHINHONG_TKE_COLD_START = np.float32(0.005)


def _grid(nz=36, ztop=6000.0):
    """The tests/test_ysu.py hydrostatic synthetic column, verbatim."""
    z_ifc = np.linspace(0.0, ztop, nz + 1)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    p_ifc = 100000.0 * np.exp(-z_ifc / 8200.0)
    p = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    return z, dz, p, p_ifc, exner


# ---------------------------------------------------------------------------
# (a) Stable-limb non-regression (the expectation note's registered item:
#     "same idiom as tests/test_ysu.py::test_stable_boundary_layer_wind_
#     decays: 3 h, hfx < 0, wind KE must decay, hpbl bounded").
# ---------------------------------------------------------------------------

def test_stable_boundary_layer_wind_decays_under_shinhong():
    """Stable surface drag and local diffusion reduce low-level KE.

    The exact grid and forcing shape of the YSU stable test (nz=36 to
    6 km, theta 296 + 0.009z, hfx = -35 W m-2, br = 0.14, dt = 30 s,
    3 simulated hours), run through the float32 CPU authority at a
    mesoscale spacing where the partition functions sit at their
    YSU-family limit.  corf is a midlatitude 1e-4 s-1 -- the scheme
    reads it through f = max(corf, eps1) for the QNSE mixing length --
    and the TKE carrier starts at WRF's shinhonginit cold-start value
    and is fed back each step exactly as _run_shinhong feeds
    state.e_sgs.  Measured at time of writing: KE ratio 0.807, hpbl
    327-395 m, every field finite, TKE floored at epsq2l/2 everywhere.
    """
    from gpuwm.verify.shinhong_ref import np_shinhong_column

    nz = 36
    z, dz, p, p_ifc, exner = _grid(nz)
    theta = 296.0 + 0.009 * z
    u = 11.0 + 0.0015 * z
    v = 2.0 - 0.0003 * z
    qv = 0.010 * np.exp(-z / 2400.0)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    tke = np.full(nz, SHINHONG_TKE_COLD_START)
    dt = 30.0
    ke0 = np.mean(u[:4] ** 2 + v[:4] ** 2)
    hpbl = []
    for _ in range(3 * 120):
        out = np_shinhong_column(
            u, v, theta * exner, qv, qc, qi, p, exner, p_ifc, dz, tke,
            psfc=100000.0, znt=0.10, ust=0.30, hfx=-35.0, qfx=0.0,
            wspd=max(float(np.hypot(u[0], v[0])), 0.1), br=0.14,
            psim=np.log(max(z[0] / 0.10, 1.01)),
            psih=np.log(max(z[0] / 0.01, 1.01)),
            xland=1.0, corf=1.0e-4, u10=float(u[0]), v10=float(v[0]),
            dt=dt, dx=3000.0, dy=3000.0, tke_diag=1)
        hpbl.append(float(out["hpbl"]))
        theta = theta + dt * out["rthblten"]
        u = u + dt * out["rublten"]
        v = v + dt * out["rvblten"]
        qv = qv + dt * out["rqvblten"]
        tke = out["tke"]
    ke1 = np.mean(u[:4] ** 2 + v[:4] ** 2)
    assert ke1 < 0.92 * ke0
    assert all(0.0 < h < 1000.0 for h in hpbl)
    for field in (theta, u, v, qv, tke):
        assert np.isfinite(field).all()
    # The scheme's own floor held through three hours of feedback --
    # the same floor the e_sgs publication test asserts on device.
    assert (tke >= SHINHONG_TKE_COLD_START).all()


# ---------------------------------------------------------------------------
# (b) OFF-path bitwise inertness.
# ---------------------------------------------------------------------------

def _micro_run_config():
    from gpuwm.config import RunConfig, validate_run_config

    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=16, dx=500.0, dy=500.0, ztop=2000.0,
        dt=3.0, run_seconds=60.0, time_step_sound=4,
        moist=True, mp_physics=0,
        bl_pbl_physics=1,              # YSU: the OFF path under test
        sf_sfclay_physics=91,
        sf_surface_physics=0,
        km_opt=4, c_s=0.25,
        bldt=0.0, radt=0.0, cu_physics=0))


@pytest.mark.gpu
@requires_gpu
def test_off_path_bl1_micro_run_state_hash_is_pinned():
    """Scheme-11 code is selector-gated: a bl=1 run's bytes are pinned.

    20 steps of a tiny YSU dry CBL, SHA-256 over the prognostic bytes.
    The pin below was recorded WITH the scheme-11 runtime code present
    in the tree but unselected (lane/shinhong-port, Phase D wiring, RTX
    5090 -- the card the suite's device goldens and the cross-card
    determinism receipt sanction byte pins on).  Any future leak of
    scheme-11 code into the off path -- an allocation in a shared
    branch, an unconditional launch, a dispatch fall-through -- moves
    this hash; two in-process runs of the same lane tree agreeing would
    prove nothing (vacuous), which is why the value is a literal and
    not a second run.  The pinned-golden mechanism is
    tests/sase_goldens.py's: numbers recorded at a stated commit, moved
    only with a rationale.
    """
    import cupy as cp

    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import init_at_rest

    cfg = _micro_run_config()
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    rng = np.random.default_rng(20260803)
    field = np.full(tuple(state.u.shape), 2.0, dtype=np.float64)
    field += 0.01 * rng.standard_normal(field.shape)
    field[:, :, -1] = field[:, :, 0]                    # periodic seam
    state.u[...] = cp.asarray(field, dtype=state.u.dtype)
    initialize_physics(state, cfg, landmask=1.0, tsk=305.0)
    run_steps(state, cfg, 20)

    # Structural half of the proof: the e_sgs allocation gate did not
    # leak into a non-selecting configuration's object graph.
    assert not hasattr(state, "e_sgs")

    digest = hashlib.sha256()
    for name in ("u", "v", "w", "thp", "php", "mup", "qv", "qc"):
        digest.update(name.encode())
        digest.update(cp.asnumpy(getattr(state, name)).tobytes())
    # A byte pin is a pin on the COMPILED image, so it is keyed by the NVRTC
    # build the same way the Shin-Hong ULP table is -- see
    # tests/test_shinhong_wrf461_parity.py's GPU_BASELINE_MAX_ULP_BY_NVRTC_-
    # BUILD for the full account of the 2026-08-04 compiler swap.  The 13.0.48
    # row is the original: recorded 2026-08-03 on the RTX 5090 at the Phase-D
    # wiring tip, two independent processes producing identical bytes (the
    # dual-run comparison that doubles as the no-ECC corruption screen).  The
    # 12.9.86 row was measured 2026-08-06 on the same card, deterministic over
    # two runs and identical across four checkouts of this tree.
    #
    # NOTE this run selects bl_pbl_physics=1 (YSU), NOT Shin-Hong: the swap
    # moved YSU's compiled bytes too.  It did not move YSU's ULP table, which
    # is a max over a coarser quantity; bytes are the finer instrument and
    # they saw it.
    pinned = {
        "13.0.48":
            "76450502591b84d602b8711d7414d62a3feed17a951212748319878742ed6394",
        "12.9.86":
            "ee18e6fbf4d7d0a4a9d6dd5508d2ee78180446192188d0b958fc54ebdcec20e8",
    }
    build = nvrtc_build()
    assert build in pinned, (
        "no recorded run-state hash for the NVRTC build compiling these"
        f" kernels.\n  kernel compiler: NVRTC {build}\n"
        f"  builds recorded: {sorted(pinned)}\n"
        f"  fingerprint:     {compile_platform_fingerprint()}\n"
        "Byte pins are pins on the compiled image; an unmeasured compiler has"
        " no pin to be compared against.  Record this build's bytes with the"
        " attribution, or compile with a recorded one.")
    assert digest.hexdigest() == pinned[build], (
        "the off-path bl=1 run's bytes moved under a compiler this pin has"
        f" already been measured under (NVRTC {build}).  That is a leak of"
        " scheme-11 code into the off path, or a change to the shared"
        " dycore -- not the 2026-08-04 NVRTC swap.")


# ---------------------------------------------------------------------------
# (c) Dispatch reachability, both directions.
# ---------------------------------------------------------------------------

def test_selector_11_resolves_to_run_shinhong_and_unrouted_still_raises():
    from gpuwm.core.physics import (PhysicsDriver,
                                    UnroutedPhysicsSelectorError,
                                    resolve_physics_slot)

    assert resolve_physics_slot("bl_pbl_physics", 11) == "_run_shinhong"
    # The runner the row names must actually exist on the driver class:
    # a dispatch row pointing at a typo would fail only at the first
    # due surface step of a real run.
    assert callable(getattr(PhysicsDriver, "_run_shinhong"))
    # Negative control, provably firing: a real WRF PBL value with no
    # gpuwm runner (12 is GBM) still refuses instead of substituting.
    with pytest.raises(UnroutedPhysicsSelectorError) as caught:
        resolve_physics_slot("bl_pbl_physics", 12)
    assert "refusing to substitute" in str(caught.value)
    assert "PHYSICS_SLOT_DISPATCH" in str(caught.value)


def test_driver_compute_dispatches_11_to_run_shinhong(monkeypatch):
    """bl_pbl_physics=11 reaches _run_shinhong and nothing else's runner.

    The tests/test_physics_dispatch.py CPU harness (no device): the
    module's ``cp`` is rebound to numpy and the driver is assembled
    with object.__new__, so this covers the compute() dispatch seam --
    the place a truthiness branch would silently run YSU for a
    Shin-Hong request, which is the file-founding failure mode.
    """
    from types import SimpleNamespace

    import gpuwm.core.physics as physics

    monkeypatch.setattr(physics, "cp", np)
    atmosphere = {
        "p_interface": np.array(
            [[[100000.0, 90000.0]], [[95000.0, 85000.0]]], np.float32),
    }
    monkeypatch.setattr(
        physics, "_prepare_atmosphere", lambda state: atmosphere)
    state = SimpleNamespace(elapsed_seconds=0.0)
    cfg = SimpleNamespace(
        dt=60.0, bldt=0.0, ra_physics=0,
        sf_sfclay_physics=91, sf_surface_physics=0,
        bl_pbl_physics=11, cu_physics=0)
    driver = object.__new__(physics.PhysicsDriver)
    driver.state = state
    driver.fields = {"psfc": np.full((1, 2), -123.0, np.float32)}
    driver.surface_enabled = True
    driver.stepbl = 1
    driver.radt_minutes = 12.0
    driver.cudt_minutes = 5.0
    driver.call_counts = {
        "radiation": 0, "sfclay": 0, "noah": 0, "ysu": 0,
        "cumulus": 0, "cumulus_history": 0,
    }
    driver.tendencies = object()
    driver._compose_tendencies = lambda cfg_arg: None
    seen = {}
    driver._run_sfclay = lambda *_: None
    driver._run_noah = lambda *_: pytest.fail("no LSM is selected")
    driver._run_ysu = lambda *_: pytest.fail(
        "Shin-Hong (bl_pbl_physics=11) must never run YSU")
    driver._run_mynn_pbl = lambda *_: pytest.fail(
        "Shin-Hong (bl_pbl_physics=11) must never run MYNN")
    driver._run_sase = lambda *_: pytest.fail(
        "Shin-Hong (bl_pbl_physics=11) must never run SASE")
    driver._run_shinhong = lambda atmosphere_arg, cfg_arg: seen.update(
        option=cfg_arg.bl_pbl_physics)

    driver.compute(state, cfg)

    assert seen == {"option": 11}
    assert driver.call_counts["ysu"] == 1   # the shared PBL-call counter


# ---------------------------------------------------------------------------
# (d) e_sgs publication after a real scheme-11 step.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_shinhong_step_publishes_its_tke_into_e_sgs():
    """After scheme-11 steps, e_sgs is finite, floored, and MOVED.

    Both halves are load-bearing: floor-respected-everywhere alone
    would pass a buffer stuck at the cold-start fill, and
    changed-somewhere alone would pass a dead pointer aliased onto
    garbage -- together they prove the driver launched the scheme and
    wrote its validated TKE back into the array the D1 instrument
    scores.
    """
    import cupy as cp

    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import init_at_rest

    cfg = validate_run_config(RunConfig(
        nx=8, ny=8, nz=16, dx=500.0, dy=500.0, ztop=2000.0,
        dt=3.0, run_seconds=60.0, time_step_sound=4,
        moist=True, mp_physics=0,
        bl_pbl_physics=11,
        sf_sfclay_physics=91,
        sf_surface_physics=0,
        km_opt=4, c_s=0.25,
        bldt=0.0, radt=0.0, cu_physics=0))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    # A light wind so the surface layer produces a nonzero ust -- the
    # QNSE mixing length divides by it (rlambda = f/(blckdr*ust)).
    state.u[...] = cp.float32(2.0)
    initialize_physics(state, cfg, landmask=1.0, tsk=305.0)

    before = cp.asnumpy(state.e_sgs).copy()
    np.testing.assert_array_equal(before, SHINHONG_TKE_COLD_START)

    run_steps(state, cfg, 2)

    after = cp.asnumpy(state.e_sgs)
    assert np.isfinite(after).all()
    # WRF's floor, epsq2l/2: the scheme's prodq2/vdifq chain never
    # publishes below it where it ran, and it ran in every column of
    # this all-land convective micro-grid.
    assert (after >= SHINHONG_TKE_COLD_START).all()
    # ... and the field genuinely moved off its cold-start fill.
    assert (after != before).any()
    assert float(after.max()) > float(SHINHONG_TKE_COLD_START)
    # The driver's published diagnostics moved with it.
    pblh = cp.asnumpy(state.physics.fields["pblh"])
    kpbl = cp.asnumpy(state.physics.fields["kpbl"])
    assert np.isfinite(pblh).all() and (pblh > 0.0).all()
    assert (kpbl >= 1).all()


# ---------------------------------------------------------------------------
# (e) Cross-module constants that must not drift.
# ---------------------------------------------------------------------------

def test_preflight_roster_and_cold_start_match_the_authorities():
    """The pins other modules carry as literals, gated to their sources.

    * preflight's _SHINHONG_3D/_SHINHONG_2D price exactly the launcher's
      per-call output roster (preflight must stay importable without
      CuPy, so it cannot import the launcher module itself);
    * state.py's cold-start fill is epsq2l/2 against the authority's
      EPSQ2L (state.py must not depend on the verification tree, the
      sase_limits precedent, so it carries the literal).
    """
    from gpuwm.core import preflight as pf
    from gpuwm.core.shinhong import (_SHINHONG_2D_OUTPUTS,
                                     _SHINHONG_3D_FLOAT_OUTPUTS)
    from gpuwm.verify.shinhong_ref import EPSQ2L

    assert pf._SHINHONG_3D == _SHINHONG_3D_FLOAT_OUTPUTS
    assert pf._SHINHONG_2D == _SHINHONG_2D_OUTPUTS
    assert SHINHONG_TKE_COLD_START == np.float32(0.5) * EPSQ2L

    import dataclasses

    cfg = _micro_run_config()
    assert pf.shinhong_output_transient_shapes(cfg) == {}   # bl=1: none
    shapes = pf.shinhong_output_transient_shapes(
        dataclasses.replace(cfg, bl_pbl_physics=11))
    assert len(shapes) == 13
    assert shapes["shinhong_output/du"] == (cfg.nz, cfg.ny, cfg.nx)
    assert shapes["shinhong_output/kpbl"] == (cfg.ny, cfg.nx)
