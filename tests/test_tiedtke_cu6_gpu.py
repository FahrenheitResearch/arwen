"""cu6 on the device: the engine route, and many j rows in one launch.

Not the parity harness.  Word-for-word agreement with WRF's Fortran is
tools/tiedtke_wrf461_oracle/check.py --gpu, which needs the built oracle.
These run on any card:

* a RunConfig asking for cu_physics=6 gets the original Tiedtke adapter
  through initialize_physics and PhysicsDriver.compute, and its rates and
  RAINC are finite;
* the adapter walks many complete j rows per launch.  F:960 is a row-wide
  early return, so a launch of several rows must give every word that one
  launch per row gives -- including rows with no active column (they take
  the early return) beside rows that continue.  The breakage this
  prevents: deciding the predicate over the whole launch would continue
  inactive columns of a quiet row whenever another row in the same launch
  convects, which changes their tendencies (WRF returns them early).
  Both first-launch guesses (continue, early return) are held to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")

RD, RV, G, CP = 287.0, 461.6, 9.81, 7.0 * 287.0 / 2.0


def _columns(families, nz=30):
    """Bottom-up synthetic columns, one per family entry.

    'convective': warm, moist, steep lapse rate with low-level moisture
    convergence and surface evaporation; 'stable': cold, dry, weak lapse
    rate with no forcing.  The shapes follow the oracle fixture's first and
    third families (tools/tiedtke_wrf461_oracle/oracle.F90).
    """
    nc = len(families)
    dzz = 16000.0 / nz
    out = {name: np.zeros((nz, nc)) for name in (
        "u3d", "v3d", "t3d", "qv3d", "qc3d", "qi3d", "pi3d", "rho3d",
        "qvften", "qvpblten", "dz8w", "pcps")}
    out["w"] = np.zeros((nz + 1, nc))
    out["p8w"] = np.zeros((nz + 1, nc))
    out["xland"] = np.ones(nc)
    out["qfx"] = np.zeros(nc)
    out["znu"] = 1.0 - (np.arange(1, nz + 1) - 0.5) / nz
    for c, family in enumerate(families):
        variant = c % 12
        variation = 0.7 + 0.06 * variant
        convective = family == "convective"
        ts = (304.0 if convective else 286.0) + (variant - 5) * 0.25
        lapse = 0.007 if convective else 0.003
        out["qfx"][c] = 0.00008 * variation if convective else 0.0
        out["p8w"][0, c] = 100000.0
        for k in range(nz + 1):
            h = k * dzz
            out["w"][k, c] = 0.05 * variation * np.sin(np.pi * h / 16000.0) if convective else 0.0
        for k in range(nz):
            zh = (k + 0.5) * dzz
            t = max(205.0, ts - lapse * min(zh, 10000.0)) + max(0.0, zh - 10000.0) * 0.003
            rh = (0.75 + 0.01 * variant if zh <= 10000.0 else 0.20) if convective else 0.35
            p_guess = out["p8w"][k, c] * np.exp(-0.5 * dzz / 7800.0)
            es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
            qv = rh * 0.622 * es / max(p_guess - es, 1.0)
            out["t3d"][k, c] = t
            out["qv3d"][k, c] = qv
            tv = t * (1.0 + (RV / RD - 1.0) * qv)
            out["p8w"][k + 1, c] = out["p8w"][k, c] * np.exp(-G * dzz / (RD * tv))
            p = 0.5 * (out["p8w"][k, c] + out["p8w"][k + 1, c])
            out["pcps"][k, c] = p
            out["pi3d"][k, c] = (p / 100000.0) ** (RD / CP)
            out["rho3d"][k, c] = p / (RD * tv)
            out["dz8w"][k, c] = dzz
            out["u3d"][k, c] = 5.0 + 0.001 * zh
            out["v3d"][k, c] = -2.0 + 0.0003 * zh
            if convective and zh < 3000.0:
                out["qvften"][k, c] = 1.0e-7 * variation
                out["qvpblten"][k, c] = 2.0e-8 * variation
    return {name: cp.asarray(np.ascontiguousarray(a, dtype=np.float32)) for name, a in out.items()}


def _outputs(nz, nc):
    from gpuwm.core.tiedtke import LEVEL_OUTPUTS, SURFACE_OUTPUTS
    out = {n: cp.full((nz, nc), -7.25, dtype=cp.float32) for n in LEVEL_OUTPUTS}
    out.update({n: cp.full(nc, -7.25, dtype=cp.float32) for n in SURFACE_OUTPUTS})
    out["cu_act_flag"] = cp.zeros(nc, dtype=cp.int32)
    return out


def _columns_of(inputs, lo, hi):
    return {n: a if n == "znu" else cp.ascontiguousarray(a[lo:hi] if a.ndim == 1 else a[:, lo:hi])
            for n, a in inputs.items()}


@pytest.mark.gpu
@requires_gpu
def test_many_rows_in_one_launch_give_one_launch_per_row_word_for_word():
    from gpuwm.core.tiedtke import column_step, workspace_shapes

    width = 12
    rows = (["convective"] * width, ["stable"] * width,
            ["convective"] * 6 + ["stable"] * 6, ["stable"] * width,
            ["convective"] * width)
    families = [f for row in rows for f in row]
    inputs = _columns(families)
    nz, nc = inputs["t3d"].shape
    for cfg in (dict(dt=60.0, itimestep=2, stepcu=1, f_qc=True, f_qi=True),
                dict(dt=20.0, itimestep=2, stepcu=3, f_qc=False, f_qi=False)):
        separate = {n: [] for n in _outputs(nz, 1)}
        row_active = []
        for r in range(len(rows)):
            lo, hi = r * width, (r + 1) * width
            ws = tuple(cp.zeros(s, dtype=d) for s, d in zip(
                workspace_shapes(nz, width), (cp.float32, cp.int32, cp.bool_), strict=True))
            out = column_step(_columns_of(inputs, lo, hi), **cfg, outputs=_outputs(nz, width),
                              workspace=ws)
            row_active.append(bool(cp.any(ws[1][-1] != 0)))
            for n, a in out.items():
                separate[n].append(cp.asnumpy(a))
        # The case is only a test of the predicate if it holds both kinds of row.
        assert row_active == [True, False, True, False, True], row_active
        # The first launch guesses one path for every column (continuation
        # or the early return); the guess moves cost only, never a word.
        for first in (1, 0):
            together = column_step(inputs, **cfg, outputs=_outputs(nz, nc), row_width=width,
                                   guess=first)
            for n, parts in separate.items():
                expected = np.concatenate(parts, axis=-1)
                actual = cp.asnumpy(together[n])
                assert np.array_equal(expected.view(np.uint32 if expected.dtype == np.float32 else np.int32),
                                      actual.view(np.uint32 if actual.dtype == np.float32 else np.int32)), (n, cfg, first)
        # The quiet rows took WRF's early return: no rain, no heating.
        for r in (1, 3):
            lo, hi = r * width, (r + 1) * width
            assert np.all(cp.asnumpy(together["raincv"][lo:hi]) == 0.0)
        assert float(cp.asnumpy(together["raincv"][:width]).max()) > 0.0


@pytest.mark.gpu
@requires_gpu
def test_row_width_must_divide_the_launch():
    from gpuwm.core.tiedtke import column_step
    inputs = _columns(["convective"] * 10, nz=8)
    with pytest.raises(ValueError, match="must divide"):
        column_step(inputs, dt=60.0, row_width=4)


def _state_for(cfg):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z) / 2200.0))


@pytest.mark.gpu
@requires_gpu
def test_cu_physics_6_routes_to_original_tiedtke_and_steps():
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.tiedtke import Tiedtke

    cfg = validate_run_config(RunConfig(
        nx=8, ny=5, nz=20, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=6,
        bl_pbl_physics=1, sf_sfclay_physics=91, bldt=0.0,
        cu_physics=6, cudt_minutes=0.0))
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=300.0)
    assert isinstance(driver.cumulus_callable, Tiedtke)
    tendencies = driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 1
    state.elapsed_seconds = cfg.dt
    driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 2
    shape = state.p.shape
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"):
        rate = driver.cu_rates[name]
        assert rate.shape == shape
        assert bool(cp.isfinite(rate).all()), name
    assert driver.rainc.shape == shape[1:]
    assert bool(cp.isfinite(driver.rainc).all())
    assert bool((driver.rainc >= 0).all())
    assert bool(cp.isfinite(tendencies.rtheta).all())
    assert bool(cp.isfinite(tendencies.rqv).all())
    # The adapter holds no array a checkpoint would have to classify (a
    # carried device scalar raised RestartManifestError at the end of a
    # real 12 km run).
    from gpuwm.io import restart
    restart._callable_state_check(driver.cumulus_callable, restart.CUMULUS_CALLABLE_ARRAYS,
                                  restart.CUMULUS_CALLABLE_CONTAINERS, "cumulus")
