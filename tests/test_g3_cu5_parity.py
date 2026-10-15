"""Grell 3D (cu_physics = 5) against WRF v4.6.1, word for word.

The fixtures are tools/g3_wrf461_oracle runs of the byte-unmodified
phys/module_cu_g3.F: seven 16 x 16 x 40 domains (448 graded band columns:
deep land and ocean convection, shallow-only, capped, dry, high terrain,
moist-neutral maritime and ensemble edge cases, under both shallow arms,
both cugd_avedx regimes, closure choices 0/5/7/13, day and night caps and
with and without prognostic ice), each handed whole to G3DRV and then to
conv_grell_spread3d.  Every array either call writes is graded over the
whole domain: the band is where the physics is, and outside it the port
must leave exactly WRF's initial values.

ARMS.  Under ``GPUWM_WRF_EXACT=1`` (the strict arm) every word must match:
0 ULP.  Under the default arm the same kernel source is compiled with
multiply-add contraction and FTZ disabled through the shared loader. Both arms
require identical output words. The former contraction tolerance is retired.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)

from gpuwm import wrf_exact  # noqa: E402
from gpuwm.core import g3  # noqa: E402

FIX = Path(__file__).resolve().parent / "data" / "oracles" / "g3"
CONFIGS = sorted(p.stem for p in FIX.glob("g3-*.npz"))

#: Both arms now pin operation order. The default contraction arm failed
#: the former 2 percent tolerance, including closure trigger differences.
DEFAULT_REL_TOLERANCE = 0.0


def _fixture(name: str) -> dict:
    with np.load(FIX / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def _lev(a):
    """(ny, nz, nx) Fortran-order fixture array -> engine (nz, ny, nx)."""
    return np.ascontiguousarray(np.transpose(a, (1, 0, 2)))


def run_fixture(fx: dict, module=None) -> dict:
    lev_in = np.stack([_lev(fx[f"in_{k}"]) for k in g3.LEV_IN]).astype(np.float32)
    w = _lev(fx["in_w"])
    sfc = np.stack([fx["in_p8w"][:, 0, :], fx["in_ht"], fx["in_xland"],
                    fx["in_gsw"]]).astype(np.float32)
    apr = cp.asarray(np.stack([fx[f"in_{k}_in"] for k in g3.APR]).astype(np.float32))
    step = g3.run_g3(
        lev_in=cp.asarray(lev_in), w=cp.asarray(w), sfc_in=cp.asarray(sfc),
        kpbl=cp.asarray(fx["in_kpbl"]), apr=apr, dt=float(fx["in_dt"]),
        dx=float(fx["in_dx"]), ishallow=int(fx["in_ishallow"]),
        avedx=int(fx["in_avedx"]), ichoice=int(fx["in_ichoice"]),
        fqi=int(fx["in_fqi"]), module=module,
        periodic_x=bool(fx.get("in_periodic_x",0)),
        periodic_y=bool(fx.get("in_periodic_y",0)),
        **{k: int(fx[f"in_{k}"]) for k in ("maxiens", "maxens", "maxens2", "maxens3", "ensdim") if f"in_{k}" in fx})
    out = {}
    lev = step.lev.get()
    for k, name in enumerate(g3.LEV_OUT):
        out[name] = np.transpose(lev[k], (1, 0, 2))
    sfc = step.sfc.get()
    for k, name in enumerate(g3.SFC_OUT):
        out[name] = sfc[k]
    apr_o = step.apr.get()
    for k, name in enumerate(g3.APR):
        out[name] = apr_o[k]
    isfc = step.isfc.get()
    for k, name in enumerate(g3.ISFC_OUT):
        out[name] = isfc[k]
    out["xf_ens"] = step.xf_ens.get()
    out["pr_ens"] = step.pr_ens.get()
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"):
        out[name] = np.transpose(getattr(step, name).get(), (1, 0, 2))
    out["raincv"] = step.raincv.get()
    out["pratec"] = step.pratec.get()
    out["cu_act_flag"] = step.cu_act_flag.get()
    return out


#: Oracle fields the port does not reproduce, with why.  cu_act_flag is
#: G3DRV setting .true. everywhere (no scheme or driver reads it for G3).
NOT_GRADED = set()


def compare(fx: dict, out: dict) -> dict:
    """Per field: (differing words, max |diff|, max |wrf|)."""
    rows = {}
    for key in fx:
        if not key.startswith("out_"):
            continue
        name = key[4:]
        if name in NOT_GRADED:
            continue
        want = fx[key]
        got = out[name].reshape(want.shape)
        if want.dtype.kind == "i":
            nd = int(np.sum(want != got))
            rows[name] = (nd, float(np.max(np.abs(want - got))) if nd else 0.0,
                          float(np.max(np.abs(want))))
        else:
            nd = int(np.sum(want.view(np.int32) != got.astype(np.float32).view(np.int32)))
            diff = np.abs(want.astype(np.float64) - got.astype(np.float64))
            rows[name] = (nd, float(diff.max()), float(np.abs(want).max()))
    return rows


@pytest.mark.parametrize("name", CONFIGS)
def test_g3_matches_wrf(name):
    fx = _fixture(name)
    rows = compare(fx, run_fixture(fx))
    bad = {k: v for k, v in rows.items() if v[0]}
    arm = "strict" if wrf_exact.ENABLED else "default"
    assert not bad, f"{name}: {arm} arm differs from WRF: {bad}"


def test_fixture_covers_the_regimes():
    """The fixture triggers what it claims to: deep and shallow cloud, both
    shallow arms, both avedx regimes, nonzero closure choices, ice on/off."""
    deep = shallow = 0
    seen = set()
    for name in CONFIGS:
        fx = _fixture(name)
        b = (slice(4, -4), slice(4, -4))
        deep += int(np.sum(fx["out_raincv"][b] > 0))
        shallow += int(np.sum(fx["out_xmb_shallow"][b] > 0))
        seen.add((int(fx["in_ishallow"]), int(fx["in_avedx"]),
                  int(fx["in_ichoice"]), int(fx["in_fqi"])))
    assert len(CONFIGS) >= 7 and deep >= 150 and shallow >= 150
    assert {s[0] for s in seen} == {0, 1} and {s[1] for s in seen} == {1, 3}
    assert {s[3] for s in seen} == {0, 1} and len({s[2] for s in seen}) >= 3


#: The planted defect: closure members 10-12 divide the cloud work function
#: by 60*40 s (module_cu_g3.F:3278-3288); the plant moves the 40 by one ULP.
PLANT_FROM = "a_ave / (60.f * 40.f)"
PLANT_TO = "a_ave / (60.f * 0x1.400002p+5f)"


def test_a_one_bit_closure_constant_change_is_caught():
    """The check is sharp: one ULP in one closure constant fails it."""
    from gpuwm.core.kernels import module_source, module_options, compile_module_no_ftz

    src = module_source("g3")
    assert src.count(PLANT_FROM) == 1
    planted = compile_module_no_ftz(
        "g3", src.replace(PLANT_FROM, PLANT_TO), module_options("g3"))
    caught = 0
    for name in CONFIGS:
        fx = _fixture(name)
        if int(fx.get("in_ensdim", 144)) != 144:
            continue  # the planted module is compiled at the default tier
        clean = run_fixture(fx)
        bent = run_fixture(fx, module=planted)
        moved = compare({f"out_{k}": v for k, v in clean.items()
                         if f"out_{k}" in fx}, bent)
        caught += sum(1 for v in moved.values() if v[0])
        if wrf_exact.ENABLED:
            # against WRF itself: the planted build is no longer 0 ULP
            caught += sum(1 for v in compare(fx, bent).values() if v[0])
    assert caught > 0, "a one-ULP closure constant change went unnoticed"


def test_closure_choice_16_keeps_wrfs_copy_order():
    """WRF copies the initially zero member 16 before filling it."""
    fx = _fixture(CONFIGS[0])
    out = run_fixture(dict(fx, in_ichoice=np.int32(16)))
    assert not np.any(out["raincv"])
    assert not np.any(out["xf_ens"])


def test_workspace_slot_count_matches_the_kernel():
    from gpuwm.core.kernels import load_module

    assert g3.g3_slot_count(load_module("g3")) == g3.G3_SLOT_COUNT
