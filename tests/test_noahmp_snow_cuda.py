"""Device acceptance gate for the Noah-MP snow-layer kernels.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixture.  Nothing is relaxed for the GPU --
the gate is ``max_ulp == 0`` and no tolerance is applied anywhere.

Two things this has to prove, not one:

1. Each of the seven kernels in ``noahmp_snow.cu`` reproduces every row of
   ``noahmp-snow.csv`` bit for bit, including the integer layer count.
2. ``glibc_expf``, which COMPACT is built on, is bitwise right on the device
   over a sweep far wider than the eleven COMPACT cases reach, checked against
   ``noahmp-snow-expf.csv`` -- the *live glibc 2.39 expf's* own output, since
   gfortran lowers REAL(4) EXP to ``expf@plt``.  A mis-folded ``__constant__``
   table entry could otherwise hide inside a leaf that never selects it.

Everything compared here comes from the oracle, not from the CPU
transcription, so a shared mistake in the two ports cannot pass this file.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.kernels import get_kernel  # noqa: E402

_DATA = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
FIXTURE = os.path.join(_DATA, "noahmp-snow.csv")
EXPF_FIXTURE = os.path.join(_DATA, "noahmp-snow-expf.csv")

NSNOW = 3
NSOIL = 4
NFULL = NSNOW + NSOIL

# Must match the layout documented at the top of noahmp_snow.cu.
ST = {"snowh": 0, "sneqv": 1, "snice": 2, "snliq": 5, "stc": 8,
      "zsnso": 15, "dzsnso": 22, "sh2o": 29, "sice": 33}
NSTATE = 37
IN_STRIDE = 53
OUT_STRIDE = 41

_SPANS = (("snice", -NSNOW + 1, 0), ("snliq", -NSNOW + 1, 0),
          ("stc", -NSNOW + 1, NSOIL), ("zsnso", -NSNOW + 1, NSOIL),
          ("dzsnso", -NSNOW + 1, NSOIL), ("sh2o", 1, NSOIL), ("sice", 1, NSOIL))

# Per-leaf scalar inputs, in the order the kernel reads them from slot NSTATE.
SCALARS = {
    "snowfall": ["dt", "qsnow", "snowhin", "sfctmp"],
    "compact": ["dt"],
    "combine": ["ponding1", "ponding2"],
    "divide": [],
    "snowh2o": ["dt", "qsnfro", "qsnsub", "qrain", "ssi", "snow_ret_fac",
                "ponding1", "ponding2"],
    "snowwater": ["dt", "sfctmp", "snowhin", "qsnow", "qsnfro", "qsnsub",
                  "qrain", "ssi", "snow_ret_fac"],
}
# Per-leaf extra outputs, in the order the kernel writes them from slot NSTATE.
EXTRAS = {
    "snowfall": [], "compact": [], "divide": [],
    "combine": ["ponding1", "ponding2"],
    "snowh2o": ["qsnbot", "ponding1", "ponding2"],
    "snowwater": ["qsnbot", "snoflow", "ponding1", "ponding2"],
}


def _bits(hexbits: str) -> np.uint32:
    return np.uint32(int(hexbits, 16))


def _f32(hexbits: str) -> np.float32:
    return _bits(hexbits).view(np.float32)


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "oracle fixture is empty"
    data: dict = {}
    order: list = []
    for r in rows:
        key = (r["leaf"], r["case"])
        if key not in data:
            data[key] = {}
            order.append(key)
        stage = data[key].setdefault(r["stage"], {})
        val = int(r["value"]) if r["dtype"] == "int" else _f32(r["bits"])
        stage[(r["field"], int(r["index"]))] = val
    return data, order


def _pack_state(dst: np.ndarray, stage: dict) -> None:
    dst[ST["snowh"]] = stage[("snowh", 0)]
    dst[ST["sneqv"]] = stage[("sneqv", 0)]
    for name, lo, hi in _SPANS:
        for k, j in enumerate(range(lo, hi + 1)):
            dst[ST[name] + k] = stage[(name, j)]


def _build(leaf: str, cases: list, data: dict):
    n = len(cases)
    fin = np.zeros((n, IN_STRIDE), dtype=np.float32)
    iin = np.zeros((n, 4), dtype=np.int32)
    for r, key in enumerate(cases):
        inp = data[key]["input"]
        if leaf == "combo":
            for k, name in enumerate(["dz", "wliq", "wice", "t",
                                      "dz2", "wliq2", "wice2", "t2"]):
                fin[r, k] = inp[(name, 0)]
            continue
        _pack_state(fin[r], inp)
        iin[r, 0] = inp[("isnow", 0)]
        for k, name in enumerate(SCALARS[leaf]):
            fin[r, NSTATE + k] = inp[(name, 0)]
        if leaf in ("compact", "snowwater"):
            for k, j in enumerate(range(-NSNOW + 1, 1)):
                iin[r, 1 + k] = inp[("imelt", j)]
        if leaf == "compact":
            for k, j in enumerate(range(-NSNOW + 1, 1)):
                fin[r, NSTATE + 1 + k] = inp[("ficeold", j)]
        if leaf == "snowwater":
            for k, j in enumerate(range(-NSNOW + 1, 1)):
                fin[r, NSTATE + 9 + k] = inp[("ficeold", j)]
            for k, j in enumerate(range(1, NSOIL + 1)):
                fin[r, NSTATE + 12 + k] = inp[("zsoil", j)]
    return fin, iin


def _launch(leaf: str, fin: np.ndarray, iin: np.ndarray):
    n = fin.shape[0]
    d_in = cp.asarray(fin.ravel())
    d_iin = cp.asarray(iin.ravel())
    d_out = cp.zeros(n * OUT_STRIDE, dtype=cp.float32)
    d_iout = cp.zeros(n, dtype=cp.int32)
    kern = get_kernel("noahmp_snow", f"noahmp_snow_{leaf}")
    block = 64
    grid = (n + block - 1) // block
    if leaf == "combo":
        kern((grid,), (block,), (d_in, d_out, np.int32(n)))
    else:
        kern((grid,), (block,), (d_in, d_iin, d_out, d_iout, np.int32(n)))
    return (cp.asnumpy(d_out).reshape(n, OUT_STRIDE),
            cp.asnumpy(d_iout))


def _assert_bits(got: np.float32, want: np.float32, label: str) -> None:
    gb = np.float32(got).view(np.uint32)
    wb = np.float32(want).view(np.uint32)
    if gb != wb:
        raise AssertionError(
            f"{label}: got {np.float32(got)!r} (0x{gb:08x}) "
            f"want {np.float32(want)!r} (0x{wb:08x})")


@pytest.mark.parametrize("leaf", ["combo", "snowfall", "compact", "combine",
                                  "divide", "snowh2o", "snowwater"])
def test_kernel_is_bit_exact(fixture, leaf):
    data, order = fixture
    cases = [k for k in order if k[0] == leaf]
    assert cases, f"fixture has no cases for {leaf}"

    fin, iin = _build(leaf, cases, data)
    fout, iout = _launch(leaf, fin, iin)

    for r, key in enumerate(cases):
        out = data[key]["output"]
        checked = set()
        if leaf == "combo":
            for k, name in enumerate(["dz", "wliq", "wice", "t"]):
                _assert_bits(fout[r, k], out[(name, 0)], f"{leaf}/{key[1]}:{name}")
                checked.add((name, 0))
        else:
            assert iout[r] == out[("isnow", 0)], (
                f"{leaf}/{key[1]}: isnow {iout[r]} != {out[('isnow', 0)]}")
            checked.add(("isnow", 0))
            for name in ("snowh", "sneqv"):
                _assert_bits(fout[r, ST[name]], out[(name, 0)],
                             f"{leaf}/{key[1]}:{name}")
                checked.add((name, 0))
            for name, lo, hi in _SPANS:
                for k, j in enumerate(range(lo, hi + 1)):
                    _assert_bits(fout[r, ST[name] + k], out[(name, j)],
                                 f"{leaf}/{key[1]}:{name}[{j}]")
                    checked.add((name, j))
            for k, name in enumerate(EXTRAS[leaf]):
                _assert_bits(fout[r, NSTATE + k], out[(name, 0)],
                             f"{leaf}/{key[1]}:{name}")
                checked.add((name, 0))
        missing = set(out) - checked
        assert not missing, (
            f"{leaf}/{key[1]}: fixture outputs never compared: {sorted(missing)}")


def test_glibc_expf_on_device_matches_live_glibc():
    """The device transcription must equal glibc 2.39 expf over the whole
    domain COMPACT can reach, not merely at the points COMPACT visits."""
    with open(EXPF_FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 30000, f"expf sweep is too small: {len(rows)}"

    x = np.array([_f32(r["x_bits"]) for r in rows], dtype=np.float32)
    want = np.array([_bits(r["y_bits"]) for r in rows], dtype=np.uint32)

    d_x = cp.asarray(x)
    d_y = cp.zeros(x.size, dtype=cp.float32)
    kern = get_kernel("noahmp_snow", "noahmp_snow_expf_probe")
    block = 256
    kern(((x.size + block - 1) // block,), (block,), (d_x, d_y, np.int32(x.size)))
    got = cp.asnumpy(d_y).view(np.uint32)

    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"glibc_expf differs from live glibc on {bad.size}/{x.size} samples; "
        f"first at x={x[bad[0]]!r}: got 0x{got[bad[0]]:08x} "
        f"want 0x{want[bad[0]]:08x}"
    )


def test_expf_sweep_selects_every_table_entry():
    """The sweep must actually exercise all 32 __exp2f_data entries.

    Otherwise a mis-transcribed entry could survive a passing sweep, which is
    the exact failure this fixture exists to prevent.
    """
    with open(EXPF_FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    x = np.array([_f32(r["x_bits"]) for r in rows], dtype=np.float64)
    invln2_scaled = np.float64.fromhex("0x1.71547652b82fep+5") \
        if hasattr(np.float64, "fromhex") else float.fromhex("0x1.71547652b82fep+5")
    shift = float.fromhex("0x1.8p+52")
    z = invln2_scaled * x
    kd = z + shift
    ki = kd.view(np.uint64) if isinstance(kd, np.ndarray) else None
    idx = np.unique((ki & np.uint64(31)))
    assert idx.size == 32, f"sweep selects only {idx.size}/32 table entries"


def test_device_negative_controls():
    """The device gate must be able to fail, on both of its load-bearing parts.

    1. CUDA's own ``expf`` must be seen to disagree with glibc's over the same
       sweep.  If it agreed everywhere, ``glibc_expf`` passing would say
       nothing about whether the transcription is right.
    2. Associating the SNEQV accumulation as ``SNEQV + (SNICE + SNLIQ)``
       instead of Fortran's ``(SNEQV + SNICE) + SNLIQ`` must produce a
       different float32.  That is the error this gate caught during the port,
       and it is a one-ULP error, so it also calibrates the gate's resolution.
    """
    with open(EXPF_FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    x = np.array([_f32(r["x_bits"]) for r in rows], dtype=np.float32)
    want = np.array([_bits(r["y_bits"]) for r in rows], dtype=np.uint32)

    d_x = cp.asarray(x)
    d_y = cp.zeros(x.size, dtype=cp.float32)
    block = 256
    get_kernel("noahmp_snow", "noahmp_snow_expf_native_probe")(
        ((x.size + block - 1) // block,), (block,), (d_x, d_y, np.int32(x.size)))
    native = cp.asnumpy(d_y).view(np.uint32)
    ndiff = int(np.count_nonzero(native != want))
    assert ndiff > 0, (
        "CUDA expf agrees with glibc expf on every sample of the sweep, so "
        "this gate cannot distinguish the two functions"
    )

    # 2. the association control, driven by the real fixture states
    with open(FIXTURE, newline="") as fh:
        frows = list(csv.DictReader(fh))
    data: dict = {}
    order: list = []
    for r in frows:
        if r["stage"] != "input":
            continue
        key = (r["leaf"], r["case"])
        if key not in data:
            data[key] = {}
            order.append(key)
        if r["dtype"] != "int":
            data[key][(r["field"], int(r["index"]))] = _f32(r["bits"])

    cases = [k for k in order if k[0] == "combine"]
    fin = np.zeros((len(cases), IN_STRIDE), dtype=np.float32)
    for r, key in enumerate(cases):
        for name in ("snice", "snliq"):
            for k, j in enumerate(range(-NSNOW + 1, 1)):
                fin[r, ST[name] + k] = data[key][(name, j)]
    d_in = cp.asarray(fin.ravel())
    d_out = cp.zeros(len(cases) * OUT_STRIDE, dtype=cp.float32)
    get_kernel("noahmp_snow", "noahmp_snow_sneqv_misassociated")(
        ((len(cases) + 63) // 64,), (64,), (d_in, d_out, np.int32(len(cases))))
    res = cp.asnumpy(d_out).reshape(len(cases), OUT_STRIDE)
    left = res[:, 0].view(np.uint32)
    pair = res[:, 1].view(np.uint32)
    assert int(np.count_nonzero(left != pair)) > 0, (
        "no fixture case distinguishes the two SNEQV summation orders, so the "
        "gate could not have caught the association error it did catch"
    )
