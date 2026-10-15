"""Purdue Lin (mp_physics = 2) against byte-unmodified WRF v4.6.1.

Every expected value in the fixture was returned by WRF's own ``lin_et_al``
(phys/module_mp_lin.F at d66e442), compiled at -O0 by gfortran 13.3 against
glibc 2.39 and driven column by column exactly as module_microphysics_driver.F
drives mp_physics = 2: tools/lin_wrf461_oracle/build.sh rebuilds it.

The gate is bitwise.  Under ``GPUWM_WRF_EXACT=1`` (no FMA contraction, no
flush-to-zero, IEEE division and square root) every word of every output
field -- the six species, theta, RAINNC/RAINNCV, SNOWNC/SNOWNCV,
GRAUPELNC/GRAUPELNCV and SR -- must equal WRF's, signed zeros included.  The
planted-bit control nudges one constant by one ULP and requires the same
comparison to fail, so the gate is shown to be able to fail.  Default
arithmetic is measured and reported per field; it is not a parity claim.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = Path(__file__).parent / "data" / "oracles" / "lin" / \
    "lin-wrf461-columns.npz"
PINNED_WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
STATE_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "qg")
SURFACE_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                  "graupelncv", "sr")
REGIMES = {"deep_convective", "stratiform", "cold_dry", "warm_rain_only",
           "supersaturated", "near_zero_subnormal", "high_terrain",
           "phase_boundaries"}
STRICT = os.environ.get("GPUWM_WRF_EXACT") == "1"


def _fixture():
    return np.load(FIXTURE, allow_pickle=False)


def _expected(fx) -> dict[str, np.ndarray]:
    return {name: fx[f"out_{name}"] for name in STATE_FIELDS + SURFACE_FIELDS}


def run_fixture_columns(fx, kernel=None) -> dict[str, np.ndarray]:
    """Run the CUDA scheme over every fixture column, one launch per dt.

    Columns are laid side by side as an (nz, 1, ncol) domain.  ``kernel``
    replaces the production kernel (the planted-bit control).
    """
    import cupy as cp
    from gpuwm.core.lin import launch_lin

    nz, ncol = fx["in_th"].shape
    out = {name: np.empty((nz, ncol), np.float32) for name in STATE_FIELDS}
    out.update({name: np.empty(ncol, np.float32) for name in SURFACE_FIELDS})
    dts = fx["in_dt"]
    for dt in np.unique(dts):
        cols = np.flatnonzero(dts == dt)

        def vol(name):
            return cp.asarray(np.ascontiguousarray(
                fx[f"in_{name}"][:, cols].reshape(nz, 1, cols.size)))

        def surf(name):
            return cp.asarray(np.ascontiguousarray(
                fx[f"in_{name}"][cols].reshape(1, cols.size)))

        state = {name: vol(name) for name in STATE_FIELDS}
        zeros = cp.zeros((1, cols.size), cp.float32)
        surface = {
            "rainnc": surf("rainnc"), "rainncv": zeros.copy(),
            "snownc": surf("snownc"), "snowncv": zeros.copy(),
            "graupelnc": surf("graupelnc"), "graupelncv": zeros.copy(),
            "sr": zeros.copy(),
        }
        launch_lin(state["th"], state["qv"], state["qc"], state["qr"],
                   state["qi"], state["qs"], state["qg"], vol("rho"),
                   vol("pii"), vol("p"), vol("z"), vol("dz8w"), surf("ht"),
                   surface["rainnc"], surface["rainncv"], surface["snownc"],
                   surface["snowncv"], surface["graupelnc"],
                   surface["graupelncv"], surface["sr"], float(dt),
                   kernel=kernel)
        for name in STATE_FIELDS:
            out[name][:, cols] = cp.asnumpy(state[name]).reshape(nz, -1)
        for name in SURFACE_FIELDS:
            out[name][cols] = cp.asnumpy(surface[name]).reshape(-1)
    return out


def field_report(got, want) -> dict[str, dict[str, int]]:
    from gpuwm.core.fp32_ulp import fp32_ulp_distance

    report = {}
    for name in STATE_FIELDS + SURFACE_FIELDS:
        g, w = got[name], want[name]
        bits_differ = int(np.count_nonzero(
            g.view(np.uint32) != w.view(np.uint32)))
        report[name] = {"max_ulp": int(fp32_ulp_distance(g, w).max()),
                        "max_absolute_difference":float(np.max(np.abs(g-w))),
                        "words_differing": bits_differ,
                        "words": int(w.size),
                        "got_sha256":hashlib.sha256(g.tobytes()).hexdigest(),
                        "wrf_sha256":hashlib.sha256(w.tobytes()).hexdigest()}
    return report


def test_fixture_is_wrf_output_over_every_regime():
    fx = _fixture()
    nz, ncol = fx["in_th"].shape
    assert ncol >= 64
    assert set(fx["regimes"].tolist()) == REGIMES
    provenance = json.loads(str(fx["provenance"]))
    assert provenance["wrf_commit"] == PINNED_WRF_COMMIT
    assert "module_mp_lin.F" in provenance["sha256"]
    for name in STATE_FIELDS + SURFACE_FIELDS:
        assert np.isfinite(fx[f"out_{name}"]).all(), name
    # The near-zero regime really carries signed zeros and subnormals.
    q = fx["in_qc"][:, fx["regimes"] == "near_zero_subnormal"]
    bits = q.view(np.uint32)
    assert (bits == 0x80000000).any()
    tiny = np.abs(q[q != 0])
    assert (tiny < np.finfo(np.float32).tiny).any()


@requires_gpu
def test_lin_matches_wrf_bitwise_under_strict_arithmetic():
    if not STRICT:
        pytest.skip("requires a GPUWM_WRF_EXACT=1 process")
    fx = _fixture()
    got = run_fixture_columns(fx)
    report = field_report(got, _expected(fx))
    receipt = os.environ.get("GPUWM_LIN_PARITY_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps(
            {"mode": "strict", "fields": report}, indent=1) + "\n",
            encoding="utf-8")
    bad = {k: v for k, v in report.items() if v["words_differing"]}
    assert not bad, bad


@requires_gpu
def test_planted_one_ulp_constant_breaks_parity():
    """One ULP on the rain intercept xnor must be caught."""
    if not STRICT:
        pytest.skip("requires a GPUWM_WRF_EXACT=1 process")
    import cupy as cp
    from gpuwm.core.kernels import module_options, module_source

    src = module_source("lin")
    original = "#define LIN_XNOR 8.0e6f"
    assert src.count(original) == 1
    planted = float(np.nextafter(np.float32(8.0e6), np.float32(np.inf)))
    src = src.replace(original, f"#define LIN_XNOR {planted!r}f")
    mod = cp.RawModule(code=src, options=module_options("lin"))
    fx = _fixture()
    got = run_fixture_columns(fx, kernel=mod.get_function("lin_column"))
    report = field_report(got, _expected(fx))
    receipt=os.environ.get("GPUWM_LIN_PLANT_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps({"constant":"LIN_XNOR", "before":8000000.0, "after":planted, "fields":report},indent=1)+"\n")
    with pytest.raises(AssertionError):
        assert not {k:v for k,v in report.items() if v["words_differing"]}, report


@requires_gpu
def test_lin_default_arithmetic_is_measured_per_field():
    """Default arithmetic (contraction and FTZ allowed) is reported, not
    gated: the receipt carries the per-field ULP distances."""
    if STRICT:
        pytest.skip("requires a default-arithmetic process")
    fx = _fixture()
    got = run_fixture_columns(fx)
    report = field_report(got, _expected(fx))
    receipt = os.environ.get("GPUWM_LIN_PARITY_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps(
            {"mode": "default", "fields": report}, indent=1) + "\n",
            encoding="utf-8")
    for name in STATE_FIELDS + SURFACE_FIELDS:
        assert np.isfinite(got[name]).all(), name


@requires_gpu
def test_lin_radar_diagnostic_against_native_wrf():
    import cupy as cp
    from gpuwm.core.lin import launch_refl10cm_lin
    fx=_fixture(); got=run_fixture_columns(fx)
    nz,ncol=fx['in_th'].shape
    def vol(a): return cp.asarray(np.ascontiguousarray(a.reshape(nz,1,ncol)))
    refl=cp.zeros((nz,1,ncol),cp.float32)
    launch_refl10cm_lin(vol(got['qv']),vol(got['qr']),vol(got['qs']),vol(got['qg']),
                       vol(got['th']*fx['in_pii']),vol(fx['in_p']),refl)
    actual=cp.asnumpy(refl).reshape(nz,ncol); want=fx['out_refl_10cm']
    error=float(np.max(np.abs(actual-want)))
    receipt=os.environ.get('GPUWM_LIN_RADAR_RECEIPT')
    if receipt: Path(receipt).write_text(json.dumps({'maximum_absolute_dbz':error})+'\n')
    assert np.isfinite(actual).all()
    assert error < 0.003, error


@requires_gpu
def test_lin_dispatch_updates_a_real_domain_state():
    import cupy as cp
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import init_at_rest
    from gpuwm.core import microphysics
    cfg=validate_run_config(RunConfig(nx=8,ny=8,nz=40,dx=12000.,dy=12000.,
        ztop=12000.,dt=10.,run_seconds=600.,moist=True,mp_physics=2,
        ra_lw_physics=0,ra_sw_physics=0))
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:np.full_like(z,300.),p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_at_rest(cfg,coord,base)
    state.qv.fill(0.005)
    for name,value in [('qc',2e-4),('qr',3e-4),('qi',5e-5),('qs',1e-4),('qg',1e-4)]:
        getattr(state,name).fill(value)
    from gpuwm.core.diagnostics import update_diagnostics
    update_diagnostics(state,cfg.hypsometric_opt)
    from gpuwm.core.acoustic import prepare_moist_cq
    cq=prepare_moist_cq(state,cfg)
    assert cq[-1] is True
    assert all(bool(cp.all(cp.isfinite(a))) for a in cq[:-1])
    before=state.qc.copy()
    result=microphysics.apply(state,cfg,10.,refl_10cm_due=False)
    assert result is not None
    assert result.rainncv.shape == (8,8)
    assert bool(cp.any(state.qc != before))
    for name in ('qv','qc','qr','qi','qs','qg','thp','h_diabatic'):
        assert bool(cp.all(cp.isfinite(getattr(state,name)))), name
    assert bool(cp.any(state.h_diabatic != 0))

    from gpuwm.io.restart import classify_scratch_slot
    assert all(classify_scratch_slot(slot) for slot in state._scratch)
