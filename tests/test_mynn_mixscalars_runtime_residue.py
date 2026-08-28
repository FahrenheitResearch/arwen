"""W4 Stage-C residue pin: the runtime mixscalars GPU-vs-CPU qn residue,
root-caused and bounded.

The Stage-B key-on probe (``tools/mynn_pbl_wrf461_oracle/
probe_mynn_mixscalars_runtime.py``) measured a GPU-vs-CPU residue on the
five ``rqn*blten`` outputs of the full driver at ``bl_mynn_mixscalars=1``
(worst measured: ``rqnwfablten`` 21,846 ULP / rel ~1.4e-3, ``rqnifablten``
5,461, ``rqncblten`` 247 at subnormal magnitude, ``rqniblten`` and
``rqnbcablten`` exactly 0).  This lane (mf-close3, Stage 1) bisected it:

1. **Not the plume terms.** On the final PBL call of a 20-step coupled
   run, every ``s_awqn*`` interface exported by the sibling device DMP
   unit is BIT-EQUAL to the CPU ``_dmp_mf_column`` replay on the sampled
   columns, and replaying the CPU flux chain with the GPU's exported
   ``s_awqn*`` forced into the solve leaves the residue byte-for-byte
   unchanged.

2. **Not the qn solve consumption either.** ``mynn_tendencies_default``
   (CPU) evaluated on the GPU tendencies unit's EXACT captured inputs
   reproduces every compared device output at ULP 0 -- including all five
   ``dqn*``.  The mixscalars lane's own code (the qn solves, the flux
   arm, the sibling DMP chain) is bitwise.

3. **The residue enters through the solve's consumed inputs** -- dfh/dfm
   (hence khdz) from ``mynn_turbulence_default_interfaces`` and vt/vq/
   cldfra/sgm from ``mynn_condensation_default_columns`` (first upstream
   divergence: ``psig_shcu`` 1 ULP from ``mynn_pblh_scale_columns``).
   Those are three of the four pre-discipline kernels already recorded in
   ``tests/test_mynn_pbl_driver_gpu.py``: plain C operators, so NVRTC
   contracts ``a*b+c`` into FMAs and CuPy's ``-ftz=true`` flushes
   subnormals -- a device-order difference inside the byte-frozen
   ``gpuwm/core/kernels/mynn_pbl.cu`` (pin ``b53ab90e...``), not editable
   without breaking the freeze.  The residue is NOT qn-specific: the same
   replay shows ``rqvblten``/``rthblten``/``rvblten`` moving too
   (cancellation residues, same mechanism, same fixture).

So this file pins all three facts.  The envelope in (3) is a measured
bound on a mechanism proven by (1)+(2), not an unexplained tolerance:
if either exactness assertion ever fails, the envelope no longer has its
attribution and this test must not be widened to hide that.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

try:
    import cupy as cp
except Exception:  # pragma: no cover - the marker skips
    cp = None

from gpuwm.config import RunConfig, validate_run_config

QN_SPECIES = ("qnc", "qni", "qnwfa", "qnifa", "qnbca")
SAMPLE = (0, 7, 13, 19, 25, 31, 37, 43)
STEPS = 20

#: Measured 2026-08-26, RTX 3080 (sm_86): worst 21,846 ULP (rqnwfablten,
#: rel ~1.4e-3), 5,461 (rqnifablten), 247 (rqncblten, at 6.7e-31), 0
#: (rqniblten, rqnbcablten).  Bound one power-of-two bin above the
#: measurement; the mechanism carrying it is pinned exactly below.
QN_RESIDUE_ULP_ENVELOPE = 32768


def _run_capture():
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    import gpuwm.core.mynn_pbl_gpu as gpu_mod
    import gpuwm.core.mynn_pbl_runtime as runtime_mod

    cfg = RunConfig(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=28, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0, bl_mynn_mixscalars=1)
    validate_run_config(cfg)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)
    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -2:] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 301.0)
    tsk[landmask == 0.0] = 297.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(landmask, 10, 17), isltyp=np.where(landmask, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_mynn_pbl"

    capture: dict[str, object] = {}
    orig_dmp = gpu_mod.mynn_dmp_mf_cuda
    orig_tend = gpu_mod.mynn_tendencies_default_cuda
    orig_drv = runtime_mod.mynn_bl_driver_cuda

    def dmp_wrap(values, **kw):
        result = orig_dmp(values, **kw)
        capture["gpu_dmp"] = {
            f.name: cp.asnumpy(getattr(result, f.name))
            for f in dataclasses.fields(result)}
        return result

    def tend_wrap(values, **kw):
        capture["tend_in"] = {k: cp.asnumpy(cp.asarray(v))
                              for k, v in values.items()}
        capture["tend_kw"] = {k: v for k, v in kw.items() if k != "scratch"}
        result = orig_tend(values, **kw)
        capture["tend_out"] = {
            f.name: cp.asnumpy(getattr(result, f.name))
            for f in dataclasses.fields(result)}
        return result

    def drv_wrap(values, **kw):
        capture["values"] = {k: cp.asnumpy(cp.asarray(v))
                             for k, v in values.items()}
        capture["kwargs"] = {k: v for k, v in kw.items() if k != "scratch"}
        out = orig_drv(values, **kw)
        capture["out"] = {k: cp.asnumpy(v) for k, v in out.items()}
        return out

    gpu_mod.mynn_dmp_mf_cuda = dmp_wrap
    gpu_mod.mynn_tendencies_default_cuda = tend_wrap
    runtime_mod.mynn_bl_driver_cuda = drv_wrap
    try:
        for _ in range(STEPS):
            step(state, cfg)
    finally:
        gpu_mod.mynn_dmp_mf_cuda = orig_dmp
        gpu_mod.mynn_tendencies_default_cuda = orig_tend
        runtime_mod.mynn_bl_driver_cuda = orig_drv
    return capture


@pytest.fixture(scope="module")
def runtime_capture():
    if cp is None:
        pytest.skip("no CUDA GPU / cupy")
    return _run_capture()


def _ulp(a, b):
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    return np.abs(monotone_fp32_key(np.asarray(a, np.float32))
                  - monotone_fp32_key(np.asarray(b, np.float32)))


@requires_gpu
def test_qn_solve_is_bitwise_on_the_gpu_solve_inputs(runtime_capture):
    """Fact (2): the CPU qn solves on the device unit's exact captured
    inputs reproduce every device tendency output at ULP 0 -- the
    mixscalars solve consumption itself is bitwise."""
    from gpuwm.core.mynn_pbl import mynn_tendencies_default

    capture = runtime_capture
    rows = np.asarray(SAMPLE, dtype=np.intp)
    ncol = capture["tend_in"]["dz"].shape[0]
    tin = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
               and v.shape[0] == ncol else v)
           for k, v in capture["tend_in"].items()}
    cpu_out = mynn_tendencies_default(tin, **capture["tend_kw"])
    for name in sorted(cpu_out):
        if name not in capture["tend_out"]:
            continue
        gpu = capture["tend_out"][name]
        gpu = gpu[rows] if (getattr(gpu, "ndim", 0) >= 1
                            and gpu.shape[0] == ncol) else gpu
        worst = int(_ulp(gpu, cpu_out[name]).max(initial=0))
        assert worst == 0, (
            f"{name}: {worst} ULP on identical inputs -- the tendencies "
            "unit itself diverged; the Stage-1 attribution is void")


@requires_gpu
def test_qn_flux_exports_are_bitwise_and_residue_is_bounded(runtime_capture):
    """Facts (1) and (3): the sibling DMP ``s_awqn*`` exports are
    bit-equal to the CPU replay, and the end-to-end qn residue those two
    exactness facts leave to the pre-discipline kernels stays inside the
    measured envelope."""
    from gpuwm.core.mynn_pbl import mynn_bl_driver
    import gpuwm.core.mynn_pbl as cpu_mod

    capture = runtime_capture
    rows = np.asarray(SAMPLE, dtype=np.intp)
    ncol = capture["gpu_dmp"]["ktop"].size
    values = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
                  and v.shape[0] == ncol else v)
              for k, v in capture["values"].items()}
    kwargs = dict(capture["kwargs"])
    kwargs.pop("flag_qs", None)
    base_kw = dict(initflag=kwargs.pop("initflag"),
                   delt=kwargs.pop("delt"),
                   flag_qs=capture["kwargs"].get("flag_qs", False),
                   **kwargs)

    cpu_plumes: dict[str, np.ndarray] = {}
    orig_cpu_dmp = cpu_mod.mynn_dmp_mf

    def cpu_dmp_capture(v, **kw):
        out = orig_cpu_dmp(v, **kw)
        cpu_plumes.update({k: np.array(val) for k, val in out.items()})
        return out

    cpu_mod.mynn_dmp_mf = cpu_dmp_capture
    try:
        cpu0 = mynn_bl_driver(dict(values), **base_kw)
    finally:
        cpu_mod.mynn_dmp_mf = orig_cpu_dmp

    # Fact (1): every s_awqn* interface bit-equal on the sampled columns.
    for name in QN_SPECIES:
        key = f"s_aw{name}"
        gpu = capture["gpu_dmp"][key][rows]
        worst = int(_ulp(gpu, cpu_plumes[key]).max(initial=0))
        assert worst == 0, (
            f"{key}: {worst} ULP -- the sibling DMP flux chain diverged; "
            "the Stage-1 attribution is void")

    # Fact (3): the bounded envelope, with its two structural zeros kept
    # exact -- a species whose column is zero must stay exactly zero.
    worst_all = {}
    for name in QN_SPECIES:
        key = f"r{name}blten"
        worst_all[key] = int(
            _ulp(capture["out"][key][rows], cpu0[key]).max(initial=0))
    assert worst_all["rqniblten"] == 0, worst_all
    assert worst_all["rqnbcablten"] == 0, worst_all
    worst = max(worst_all.values())
    assert worst <= QN_RESIDUE_ULP_ENVELOPE, (
        f"qn runtime residue {worst_all} exceeded the measured envelope "
        f"{QN_RESIDUE_ULP_ENVELOPE}; do not widen this bound without "
        "re-running the Stage-1 bisect and re-attributing the mechanism")
