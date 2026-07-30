"""Device acceptance gate for the Noah-MP PHENOLOGY / PRECIP_HEAT kernels.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixtures.  The CUDA transcription has to
reproduce glibc's ``expf`` and ``powf`` itself -- CUDA's device versions are
different functions -- so this test is also the only check that the
``__constant__`` tables in ``noahmp_vegprecip.cu`` survived ptxas.

Skipped where there is no device.  Run it on the machine that owns the GPU.
"""

from __future__ import annotations

import csv
import os
import struct

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core.kernels import load_module  # noqa: E402

_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gpuwm", "data", "noahmp", "oracle",
)
PHEN_CSV = os.path.join(_DATA, "noahmp-vegprecip-phenology.csv")
PRCP_CSV = os.path.join(_DATA, "noahmp-vegprecip-precip_heat.csv")

PHEN_OUT = ("lai", "sai", "elai", "esai", "igs", "fb")
PRCP_OUT = ("canliq", "canice", "qintr", "qdripr", "qthror", "qints",
            "qdrips", "qthros", "pahv", "pahg", "pahb", "qrain", "qsnow",
            "snowhin", "fwet", "cmc")


def _uh(s: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(s, 16)), dtype=np.float32)[0]


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fcol(rows, name):
    return cp.asarray(np.array([_uh(r[name]) for r in rows], dtype=np.float32))


def _icol(rows, name):
    return cp.asarray(np.array([int(r[name]) for r in rows], dtype=np.int32))


def _want(rows, names):
    return {n: np.array([_uh(r[n]) for r in rows], dtype=np.float32)
            for n in names}


def _assert_bit_exact(got: np.ndarray, want: np.ndarray, rows, label):
    gb = got.view(np.uint32)
    wb = want.view(np.uint32)
    bad = np.flatnonzero(gb != wb)
    assert bad.size == 0, (
        f"{label}: {bad.size}/{gb.size} lanes differ; first at "
        f"{rows[int(bad[0])]['case']}: got 0x{gb[int(bad[0])]:08X} "
        f"want 0x{wb[int(bad[0])]:08X}"
    )


def test_phenology_kernel_is_bit_exact():
    rows = _rows(PHEN_CSV)
    n = len(rows)
    assert all(int(r["dveg"]) == 4 and int(r["croptype"]) == 0 for r in rows), (
        "the kernel is written for dveg=4 / croptype=0 only"
    )

    laim = cp.asarray(np.array(
        [[_uh(r[f"laim{i:02d}"]) for i in range(1, 13)] for r in rows],
        dtype=np.float32).ravel())
    saim = cp.asarray(np.array(
        [[_uh(r[f"saim{i:02d}"]) for i in range(1, 13)] for r in rows],
        dtype=np.float32).ravel())

    out = {k: cp.zeros(n, dtype=cp.float32) for k in PHEN_OUT}
    fn = load_module("noahmp_vegprecip").get_function("noahmp_phenology")
    fn((1,), (max(n, 1),), (
        np.int32(n),
        _icol(rows, "vegtyp"), _icol(rows, "yearlen"),
        _icol(rows, "iswater"), _icol(rows, "isbarren"), _icol(rows, "isice"),
        _icol(rows, "urban_flag"),
        _fcol(rows, "snowh"), _fcol(rows, "tv"), _fcol(rows, "lat"),
        _fcol(rows, "julian"),
        _fcol(rows, "hvt"), _fcol(rows, "hvb"), _fcol(rows, "tmin"),
        laim, saim,
        out["lai"], out["sai"], out["elai"], out["esai"], out["igs"], out["fb"],
    ))

    want = _want(rows, PHEN_OUT)
    for name in PHEN_OUT:
        _assert_bit_exact(cp.asnumpy(out[name]), want[name], rows,
                          f"PHENOLOGY.{name}")


def test_precip_heat_kernel_is_bit_exact():
    rows = _rows(PRCP_CSV)
    n = len(rows)

    out = {k: cp.zeros(n, dtype=cp.float32) for k in PRCP_OUT}
    fn = load_module("noahmp_vegprecip").get_function("noahmp_precip_heat")
    fn((1,), (max(n, 1),), (
        np.int32(n),
        _icol(rows, "ist"),
        _fcol(rows, "dt"), _fcol(rows, "uu"), _fcol(rows, "vv"),
        _fcol(rows, "elai"), _fcol(rows, "esai"), _fcol(rows, "fveg"),
        _fcol(rows, "bdfall"), _fcol(rows, "rain"), _fcol(rows, "snow"),
        _fcol(rows, "fp"), _fcol(rows, "canliq_in"), _fcol(rows, "canice_in"),
        _fcol(rows, "tv"), _fcol(rows, "sfctmp"), _fcol(rows, "tg"),
        _fcol(rows, "ch2op"),
        out["canliq"], out["canice"], out["qintr"], out["qdripr"],
        out["qthror"], out["qints"], out["qdrips"], out["qthros"],
        out["pahv"], out["pahg"], out["pahb"], out["qrain"], out["qsnow"],
        out["snowhin"], out["fwet"], out["cmc"],
    ))

    want = _want(rows, PRCP_OUT)
    for name in PRCP_OUT:
        _assert_bit_exact(cp.asnumpy(out[name]), want[name], rows,
                          f"PRECIP_HEAT.{name}")


def test_device_libm_differs_from_cuda_libm():
    """The kernel must be using its own glibc transcription, not CUDA's.

    If someone replaces ``vp_glibc_expf`` with the device ``expf``, the leaf
    gate above breaks -- but only on the handful of arguments where they
    disagree, which is fragile evidence.  This probes the two directly over a
    dense sweep and asserts they are distinguishable, so the dependency is
    documented by a measurement rather than by a comment.
    """
    src = r'''
extern "C" __global__ void vp_probe(int n, const float* x, float* a, float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = vp_glibc_expf(x[i]);
    b[i] = expf(x[i]);
}
'''
    mod = load_module("noahmp_vegprecip")
    # reuse the compiled module for the leaf kernels; compile the probe
    # separately against the same source text
    from pathlib import Path
    from gpuwm.core.kernels import _preamble, _KDIR
    code = _preamble() + (_KDIR / "noahmp_vegprecip.cu").read_text() + src
    probe = cp.RawModule(code=code, options=("-std=c++17",))
    probe.compile()
    fn = probe.get_function("vp_probe")

    n = 100000
    x = cp.asarray(np.linspace(-40.0, 0.0, n, dtype=np.float32))
    a = cp.zeros(n, dtype=cp.float32)
    b = cp.zeros(n, dtype=cp.float32)
    fn((n // 256 + 1,), (256,), (np.int32(n), x, a, b))
    differ = int(np.count_nonzero(
        cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)))
    assert differ > 0, (
        "the glibc transcription is indistinguishable from CUDA's expf over "
        "this sweep, so the gate cannot detect a swap"
    )
    print(f"vp_glibc_expf vs CUDA expf: {differ}/{n} arguments differ")
