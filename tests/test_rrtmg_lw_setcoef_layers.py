"""rlw_setcoef_layers (one thread per column and layer) stores exactly the
words of the serial rlw_setcoef (one thread per column), on the same inputs.

The batched LW engine launches the per-layer kernel; the serial kernel keeps
the single-column entry points and their WRF oracle gates.  The per-layer
kernel is derived from the serial kernel's text (rrtmg_lw._setcoef_layers_
source), so these gates hold the derivation to its claim: every output slot
and laytrop are uint32-identical, dual-run, on decks that cross the
tropopause switch, sit on the Planck-table clamps and carry the non-positive
pressure guard.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from gpuwm.core import rrtmg_lw as lw

cp = pytest.importorskip("cupy")
try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                     # pragma: no cover
    _HAS_GPU = False

DUAL_RUNS = 2


def test_derived_kernel_keeps_every_arithmetic_statement():
    """The derivation changes the index prologue, the loop header and the
    laytrop count only: the ordered list of RLW_* arithmetic statements is
    the serial kernel's, statement for statement."""
    src = lw._gpu_source()
    serial = src[src.index('void rlw_setcoef('):]
    serial = serial[:serial.index('extern "C" __global__ void rlw_setcoef_layers(')]
    layers = src[src.index('void rlw_setcoef_layers('):]
    layers = layers[:layers.index("#undef PLLEV")]
    stmt = r"RLW_\w+\([^;]+"
    assert re.findall(stmt, serial) == re.findall(stmt, layers)
    assert "atomicAdd(&laytrop_v[col], 1);" in layers
    assert "for (int lay = 1; lay <= nl; ++lay)" not in layers
    assert src.count('void rlw_setcoef_layers(') == 1


def _deck(ncol, nl, seed):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    psfc = rng.uniform(600.0, 1050.0, ncol)
    ptop = rng.uniform(0.005, 60.0, ncol)
    frac = np.sort(rng.uniform(0.0, 1.0, (ncol, nl)), axis=1)
    pavel = (psfc[:, None] - frac * (psfc - ptop)[:, None]).astype(f32)
    # The tropopause switch plog <= 4.56 sits at pavel = exp(4.56) ~ 95.6 mb:
    # put some layers exactly on and around it.
    pavel[0::5, nl // 2] = f32(np.exp(np.float32(4.56)))
    pavel[1::5, nl // 2] = np.nextafter(f32(np.exp(np.float32(4.56))), f32(0))
    # The non-positive pressure guard (kernel clamps to 1e-6 for plog only).
    if ncol > 3:
        pavel[3, -1] = f32(0.0)
    tavel = rng.uniform(150.0, 345.0, (ncol, nl)).astype(f32)
    tavel[0::7, 0] = f32(158.5)      # below the Planck table: index clamp
    tavel[1::7, 0] = f32(345.5)      # above it
    tz = rng.uniform(150.0, 345.0, (ncol, nl + 1)).astype(f32)
    tz[2::7, 0] = f32(100.0)
    tbound = rng.uniform(150.0, 345.0, ncol).astype(f32)
    tbound[3::7] = f32(400.0)
    semiss = rng.uniform(0.85, 1.0, (ncol, lw.NBNDLW)).astype(f32)
    coldry = rng.uniform(1.0e21, 4.0e23, (ncol, nl)).astype(f32)
    wkl = np.zeros((ncol, lw.MXMOL, nl), f32)
    for slot, (lo, hi) in enumerate(((1e-7, 3e-2), (3e-4, 5e-4), (1e-8, 1e-5),
                                     (2e-7, 4e-7), (0.0, 0.0), (1e-6, 2e-6),
                                     (0.2, 0.22))):
        wkl[:, slot] = (rng.uniform(lo, hi, (ncol, nl)) * coldry).astype(f32)
    wkl[4::9, 1] = f32(0.0)          # the colco2 == 0 fallback
    wkl[5::9, 2] = f32(0.0)          # colo3
    wbroad = (coldry * rng.uniform(0.7, 0.8, (ncol, nl))).astype(f32)
    return dict(pavel=pavel, tavel=tavel, tz=tz, tbound=tbound,
                semiss=semiss, coldry=coldry, wkl=wkl, wbroad=wbroad)


def _run(kernel, deck, C, per_layer):
    i32 = np.int32
    ncol, nl = deck["pavel"].shape
    K = lw._lw_dev_consts(cp, C)
    d = {k: cp.asarray(v) for k, v in deck.items()}
    fs = cp.zeros((len(lw.GPU_FSLOTS), ncol, nl), cp.float32)
    isv = cp.zeros((len(lw.GPU_ISLOTS), ncol, nl), cp.int32)
    laytrop = cp.zeros(ncol, cp.int32)
    planklay = cp.zeros((ncol, nl, lw.NBNDLW), cp.float32)
    planklev = cp.zeros((ncol, nl + 1, lw.NBNDLW), cp.float32)
    plankbnd = cp.zeros((ncol, lw.NBNDLW), cp.float32)
    fslot = {n: i for i, n in enumerate(lw.GPU_FSLOTS)}
    islot = {n: i for i, n in enumerate(lw.GPU_ISLOTS)}
    F = lambda n: fs[fslot[n]]
    I = lambda n: isv[islot[n]]
    if per_layer:
        launch = (((ncol * nl + 127) // 128,), (128,))
    else:
        launch = (((ncol + 63) // 64,), (64,))
    lw._gpu_kernel(kernel)(*launch, (
        i32(ncol), i32(nl), i32(1),
        d["pavel"], d["tavel"], d["tz"], d["tbound"], d["semiss"],
        d["coldry"], d["wkl"], d["wbroad"],
        K["totplnk"], K["totplk16"], K["preflog"], K["tref"], K["chi"],
        laytrop, I("jp"), I("jt"), I("jt1"), planklay, planklev, plankbnd,
        F("colh2o"), F("colco2"), F("colo3"), F("coln2o"), F("colco"),
        F("colch4"), F("colo2"), F("colbrd"),
        F("fac00"), F("fac01"), F("fac10"), F("fac11"),
        F("rat_h2oco2"), F("rat_h2oco2_1"), F("rat_h2oo3"), F("rat_h2oo3_1"),
        F("rat_h2on2o"), F("rat_h2on2o_1"), F("rat_h2och4"), F("rat_h2och4_1"),
        F("rat_n2oco2"), F("rat_n2oco2_1"), F("rat_o3co2"), F("rat_o3co2_1"),
        F("selffac"), F("selffrac"), I("indself"),
        F("forfac"), F("forfrac"), I("indfor"),
        F("minorfrac"), F("scaleminor"), F("scaleminorn2"), I("indminor")))
    return {"fs": cp.asnumpy(fs), "isv": cp.asnumpy(isv),
            "laytrop": cp.asnumpy(laytrop), "planklay": cp.asnumpy(planklay),
            "planklev": cp.asnumpy(planklev), "plankbnd": cp.asnumpy(plankbnd)}


@pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU")
@pytest.mark.parametrize("ncol,nl", [(1, 4), (7, 62), (1536, 62),
                                     (3001, 74)])
def test_per_layer_setcoef_equals_serial_setcoef(ncol, nl):
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    lw.gpu_preflight()
    C = _lw_coeffs()
    deck = _deck(ncol, nl, seed=ncol * 1000 + nl)
    want = _run("rlw_setcoef", deck, C, per_layer=False)
    assert want["laytrop"].min() >= 0
    for run in range(DUAL_RUNS):
        got = _run("rlw_setcoef_layers", deck, C, per_layer=True)
        for key, ref in want.items():
            diff = int(np.count_nonzero(
                got[key].view(np.uint32) != ref.view(np.uint32)))
            assert diff == 0, f"{key}: {diff} of {ref.size} words (run {run})"
