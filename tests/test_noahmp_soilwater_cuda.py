"""Device acceptance gate for the Noah-MP soil-water kernels.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixture.  Nothing is relaxed for the GPU --
the gate is ``max_ulp == 0`` and no tolerance is applied anywhere.

Three things this has to prove, not one:

1. Each of the five kernels in ``noahmp_soilwater.cu`` reproduces every row of
   ``noahmp-soilwater.csv`` bit for bit.
2. ``glibc_expf`` and ``glibc_powf``, which the group is built on, are bitwise
   right on the device over the sweeps in ``noahmp-soilwater-libm.csv`` -- the
   *live glibc 2.39 symbols'* own output, since gfortran lowers ``REAL(4) EXP``
   to ``expf@plt`` and ``**`` with a real exponent to ``powf@plt``.  A
   mis-folded ``__constant__`` entry could otherwise hide inside a leaf that
   never selects it.
3. The gate can fail.  A deliberately perturbed constant must be rejected, or
   the two checks above are unfalsifiable.

Everything compared here comes from the oracle, not from the CPU
transcription, so a shared mistake in the two ports cannot pass this file.
"""

from __future__ import annotations

import csv
import os
import struct
import sys
from collections import defaultdict

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.kernels import get_kernel, load_module  # noqa: E402

_DATA = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
FIXTURE = os.path.join(_DATA, "noahmp-soilwater.csv")
PROBE = os.path.join(_DATA, "noahmp-soilwater-libm.csv")

NSOIL = 4
NSNOW = 3

# Must match the layout documented at the top of noahmp_soilwater.cu.
P_STRIDE = 24
IN_STRIDE = 40
OUT_STRIDE = 32

PAR = {"smcmax": 0, "smcwlt": 4, "bexp": 8, "dksat": 12, "dwsat": 16}
PAR_SCALAR = {"kdt": 20, "frzx": 21, "slope": 22, "ch2op": 23}


def _f32(hexbits: str) -> np.float32:
    return np.uint32(int(hexbits, 16)).view(np.float32)


def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "oracle fixture is empty"
    table = defaultdict(dict)
    cases = defaultdict(list)
    for r in rows:
        key = (r["field"], int(r["index"]))
        v = int(r["value"]) if r["dtype"] == "int" else _f32(r["bits"])
        table[(r["leaf"], r["case"], r["stage"])][key] = v
        if r["stage"] == "input" and r["case"] not in cases[r["leaf"]]:
            cases[r["leaf"]].append(r["case"])
    return table, {k: sorted(v) for k, v in cases.items()}


def _pack_params(table, leaf, cases):
    par = np.zeros((len(cases), P_STRIDE), dtype=np.float32)
    for i, c in enumerate(cases):
        p = table[(leaf, c, "param")]
        for name, base in PAR.items():
            for k in range(NSOIL):
                par[i, base + k] = p[(name, k + 1)]
        for name, slot in PAR_SCALAR.items():
            par[i, slot] = p[(name, 0)]
    return par


def _vec(d, name, lo=1, hi=NSOIL):
    return [d[(name, k)] for k in range(lo, hi + 1)]


def _launch(func, args, n):
    threads = 128
    blocks = (n + threads - 1) // threads
    func((blocks,), (threads,), args)
    cp.cuda.runtime.deviceSynchronize()


def _compare(leaf, cases, table, out, layout):
    """layout: list of (slot, field, index).  Every entry must match exactly."""
    bad = []
    for i, c in enumerate(cases):
        want = table[(leaf, c, "output")]
        for slot, field, index in layout:
            got = np.float32(out[i, slot])
            if _bits(got) != _bits(want[(field, index)]):
                bad.append(f"{c}/{field}[{index}] gpu={_bits(got)} "
                           f"oracle={_bits(want[(field, index)])}")
    assert not bad, f"{leaf}: " + "; ".join(bad[:12])


# ---------------------------------------------------------------------------

def test_canwater_kernel_matches_oracle(fixture):
    table, cases = fixture
    cs = cases["canwater"]
    par = _pack_params(table, "canwater", cs)
    fin = np.zeros((len(cs), IN_STRIDE), dtype=np.float32)
    iin = np.zeros(len(cs), dtype=np.int32)
    for i, c in enumerate(cs):
        x = table[("canwater", c, "input")]
        for k, name in enumerate(["dt", "fcev", "fctr", "elai", "esai",
                                  "fveg", "bdfall", "canliq", "canice", "tv"]):
            fin[i, k] = x[(name, 0)]
        iin[i] = x[("frozen_canopy", 0)]
    d_par, d_in, d_ii = cp.asarray(par), cp.asarray(fin), cp.asarray(iin)
    d_out = cp.zeros((len(cs), OUT_STRIDE), dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_canwater"),
            (d_par, d_in, d_ii, d_out, np.int32(len(cs))), len(cs))
    out = cp.asnumpy(d_out)
    names = ["canliq", "canice", "tv", "cmc", "ecan", "etran", "fwet",
             "qsubc", "qfroc", "qfrzc", "qmeltc", "qevac", "qdewc"]
    _compare("canwater", cs, table, out,
             [(k, n, 0) for k, n in enumerate(names)])


def test_infil_kernel_matches_oracle(fixture):
    table, cases = fixture
    cs = cases["infil"]
    par = _pack_params(table, "infil", cs)
    fin = np.zeros((len(cs), IN_STRIDE), dtype=np.float32)
    for i, c in enumerate(cs):
        x = table[("infil", c, "input")]
        fin[i, 0] = x[("dt", 0)]
        fin[i, 1:5] = _vec(x, "zsoil")
        fin[i, 5:9] = _vec(x, "sh2o")
        fin[i, 9:13] = _vec(x, "sice")
        fin[i, 13] = x[("sicemax", 0)]
        fin[i, 14] = x[("qinsur", 0)]
        fin[i, 15] = x[("pddum", 0)]
        fin[i, 16] = x[("runsrf", 0)]
    d_out = cp.zeros((len(cs), OUT_STRIDE), dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_infil"),
            (cp.asarray(par), cp.asarray(fin), d_out, np.int32(len(cs))),
            len(cs))
    _compare("infil", cs, table, cp.asnumpy(d_out),
             [(0, "pddum", 0), (1, "runsrf", 0)])


def test_srt_kernel_matches_oracle(fixture):
    table, cases = fixture
    cs = cases["srt"]
    par = _pack_params(table, "srt", cs)
    fin = np.zeros((len(cs), IN_STRIDE), dtype=np.float32)
    for i, c in enumerate(cs):
        x = table[("srt", c, "input")]
        fin[i, 0] = x[("pddum", 0)]
        fin[i, 1:5] = _vec(x, "zsoil")
        fin[i, 5:9] = _vec(x, "etrani")
        fin[i, 9] = x[("qseva", 0)]
        fin[i, 10:14] = _vec(x, "smc")
        fin[i, 14:18] = _vec(x, "fcr")
    d_out = cp.zeros((len(cs), OUT_STRIDE), dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_srt"),
            (cp.asarray(par), cp.asarray(fin), d_out, np.int32(len(cs))),
            len(cs))
    layout = []
    for b, name in enumerate(["rhstt", "ai", "bi", "ci", "wcnd"]):
        layout += [(b * NSOIL + k, name, k + 1) for k in range(NSOIL)]
    layout.append((5 * NSOIL, "qdrain", 0))
    _compare("srt", cs, table, cp.asnumpy(d_out), layout)


def test_sstep_kernel_matches_oracle(fixture):
    table, cases = fixture
    cs = cases["sstep"]
    par = _pack_params(table, "sstep", cs)
    fin = np.zeros((len(cs), IN_STRIDE), dtype=np.float32)
    for i, c in enumerate(cs):
        x = table[("sstep", c, "input")]
        fin[i, 0] = x[("dt", 0)]
        fin[i, 1:5] = [x[("dzsnso", k)] for k in range(1, NSOIL + 1)]
        fin[i, 5:9] = _vec(x, "sh2o")
        fin[i, 9:13] = _vec(x, "ai")
        fin[i, 13:17] = _vec(x, "bi")
        fin[i, 17:21] = _vec(x, "ci")
        fin[i, 21:25] = _vec(x, "rhstt")
        fin[i, 25:29] = _vec(x, "sice")
    d_out = cp.zeros((len(cs), OUT_STRIDE), dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_sstep"),
            (cp.asarray(par), cp.asarray(fin), d_out, np.int32(len(cs))),
            len(cs))
    layout = []
    for b, name in enumerate(["sh2o", "smc", "ai", "bi", "ci", "rhstt"]):
        layout += [(b * NSOIL + k, name, k + 1) for k in range(NSOIL)]
    layout.append((6 * NSOIL, "wplus", 0))
    _compare("sstep", cs, table, cp.asnumpy(d_out), layout)


def test_soilwater_kernel_matches_oracle(fixture):
    table, cases = fixture
    cs = cases["soilwater"]
    par = _pack_params(table, "soilwater", cs)
    fin = np.zeros((len(cs), IN_STRIDE), dtype=np.float32)
    iin = np.zeros(len(cs), dtype=np.int32)
    for i, c in enumerate(cs):
        x = table[("soilwater", c, "input")]
        p = table[("soilwater", c, "param")]
        fin[i, 0] = x[("dt", 0)]
        fin[i, 1:5] = _vec(x, "zsoil")
        fin[i, 5:9] = [x[("dzsnso", k)] for k in range(1, NSOIL + 1)]
        fin[i, 9] = x[("qinsur", 0)]
        fin[i, 10] = x[("qseva", 0)]
        fin[i, 11:15] = _vec(x, "sice")
        fin[i, 15:19] = _vec(x, "sh2o")
        fin[i, 19:23] = _vec(x, "smc")
        fin[i, 23] = x[("runsub", 0)]
        fin[i, 24:28] = _vec(x, "etrani")
        iin[i] = p[("urban_flag", 0)]
    d_out = cp.zeros((len(cs), OUT_STRIDE), dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_soilwater"),
            (cp.asarray(par), cp.asarray(fin), cp.asarray(iin), d_out,
             np.int32(len(cs))), len(cs))
    layout = []
    for b, name in enumerate(["sh2o", "smc", "wcnd"]):
        layout += [(b * NSOIL + k, name, k + 1) for k in range(NSOIL)]
    layout += [(3 * NSOIL, "runsrf", 0), (3 * NSOIL + 1, "qdrain", 0),
               (3 * NSOIL + 2, "runsub", 0), (3 * NSOIL + 3, "fcrmax", 0)]
    _compare("soilwater", cs, table, cp.asnumpy(d_out), layout)


# ---------------------------------------------------------------------------
# libm sweeps, wider than any leaf reaches
# ---------------------------------------------------------------------------

def _probe_rows():
    with open(PROBE, newline="") as fh:
        return {r["name"].strip(): r["bits"].upper() for r in csv.DictReader(fh)}


def test_device_expf_matches_glibc_over_the_fcr_domain():
    rows = _probe_rows()
    x = np.array([np.float32(-4.0 * (1.0 - np.float32(i) / np.float32(256.0)))
                  for i in range(257)], dtype=np.float32)
    d_y = cp.zeros(x.size, dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_expf_sweep"),
            (cp.asarray(x), d_y, np.int32(x.size)), x.size)
    y = cp.asnumpy(d_y)
    bad = [i for i in range(257) if _bits(y[i]) != rows[f"expf_fcr_{i}"]]
    assert not bad, f"{len(bad)} of 257 expf points differ, first {bad[:8]}"


def test_device_powf_matches_glibc_over_every_probed_domain():
    rows = _probe_rows()
    xs, es, keys = [], [], []
    for i in range(257):
        xs.append(np.float32(np.float32(i) / np.float32(256.0)))
        es.append(np.float32(0.667))
        keys.append(f"powf_fwet_{i}")
    for i in range(129):
        x = np.float32(np.float32(0.01)
                       + np.float32(np.float32(1.0 - 0.01)
                                    * np.float32(np.float32(i)
                                                 / np.float32(128.0))))
        xs += [x, x]
        es += [np.float32(6.74), np.float32(12.48)]
        keys += [f"powf_wdf_{i}", f"powf_wcnd_{i}"]
    x = np.asarray(xs, dtype=np.float32)
    e = np.asarray(es, dtype=np.float32)
    d_y = cp.zeros(x.size, dtype=cp.float32)
    _launch(get_kernel("noahmp_soilwater", "k_powf_sweep"),
            (cp.asarray(x), cp.asarray(e), d_y, np.int32(x.size)), x.size)
    y = cp.asnumpy(d_y)
    bad = []
    for i, key in enumerate(keys):
        # glibc powf(0, y) is 0 and the transcription's fast path returns NaN
        # outside its ported domain; the fixture's first fwet point is x == 0.
        if x[i] == np.float32(0.0):
            continue
        if _bits(y[i]) != rows[key]:
            bad.append(key)
    assert not bad, f"{len(bad)} of {len(keys)} powf points differ: {bad[:8]}"


def test_the_device_gate_can_fail():
    """Perturb one __constant__ entry and require the comparison to reject it.

    Without this the two tests above are unfalsifiable: a kernel that silently
    returned the host's own inputs would pass them.
    """
    src = open(os.path.join(_ROOT, "gpuwm", "core", "kernels",
                            "noahmp_soilwater.cu")).read()
    # K_A, SOILWATER's frozen-fraction decay constant: 4.0 -> 4.001.
    # A one-ULP nudge is not enough -- FCRMAX quantises it away -- so this
    # uses the same relative 1e-3 the CPU mutation study uses.
    assert "0x40800000u, /* 0 K_A" in src
    mutated = src.replace("0x40800000u, /* 0 K_A", "0x40800831u, /* 0 K_A", 1)
    assert mutated != src

    from gpuwm.core.kernels import _preamble
    mod = cp.RawModule(code=_preamble() + mutated, options=("-std=c++17",))
    mod.compile()

    with open(FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    table = defaultdict(dict)
    for r in rows:
        v = int(r["value"]) if r["dtype"] == "int" else _f32(r["bits"])
        table[(r["leaf"], r["case"], r["stage"])][(r["field"],
                                                   int(r["index"]))] = v
    cs = ["slw_frozen"]
    par = _pack_params(table, "soilwater", cs)
    fin = np.zeros((1, IN_STRIDE), dtype=np.float32)
    x = table[("soilwater", "slw_frozen", "input")]
    fin[0, 0] = x[("dt", 0)]
    fin[0, 1:5] = _vec(x, "zsoil")
    fin[0, 5:9] = [x[("dzsnso", k)] for k in range(1, NSOIL + 1)]
    fin[0, 9] = x[("qinsur", 0)]
    fin[0, 10] = x[("qseva", 0)]
    fin[0, 11:15] = _vec(x, "sice")
    fin[0, 15:19] = _vec(x, "sh2o")
    fin[0, 19:23] = _vec(x, "smc")
    fin[0, 23] = x[("runsub", 0)]
    fin[0, 24:28] = _vec(x, "etrani")
    d_out = cp.zeros((1, OUT_STRIDE), dtype=cp.float32)
    _launch(mod.get_function("k_soilwater"),
            (cp.asarray(par), cp.asarray(fin), cp.zeros(1, dtype=cp.int32),
             d_out, np.int32(1)), 1)
    out = cp.asnumpy(d_out)
    want = table[("soilwater", "slw_frozen", "output")]
    assert _bits(out[0, 3 * NSOIL + 3]) != _bits(want[("fcrmax", 0)]), (
        "perturbing K_A did not move FCRMAX; the device gate cannot fail")
