"""Shin-Hong's csfac tanhf against WRF v4.6.1 on 1,134 columns built to reach it.

csfac = 0.5*(tanh(uwstx)+3) with uwstx = -80*|ust/wstar - 0.5| + 14
(module_bl_shinhong.F:846-847) scales cslen, and cslen feeds the partition
functions pu, pq and pthnl (:919-947, :1115), so it reaches du, dv, dtheta,
dqv and, through the TKE chain, tke and el.  WRF calls the C library's
tanhf there.  On the reference host (W1: gfortran 13.3.0, glibc 2.39, the
host the wide column oracle and every other libm word of the kernel are
pinned to) that tanhf is a float32 expm1f expression, not a correctly
rounded function: it is not the once-rounded double tanh on 118,674,314
float32 arguments in [-10, 14], and csfac is a different word on 4,098,664
of them (``tanh-receipt.txt``, ``tools/shinhong_column_oracle/
tanh_receipt.c``).

The kernel used to spell it ``(float)tanh((double)x)``.  The 903-column wide
oracle never caught that: 238 of its columns reach the unsaturated range,
14 of those land on an argument where tanhf moves, and none on one where
csfac moves.  This fixture is the wide oracle's 567 columns with a non-zero
wstar, each with ust re-chosen twice so that uwstx lands on an argument
where csfac moves (``refute_columns.py tanh``).  wstar does not depend on
ust (wstar3 = govrth*bfx0*hpbl on the first-guess hpbl, :768), so every
other input word is the wide oracle's own.  Before the fix the strict build
differed from WRF on du (105 lanes, up to 58,255 ULP), dv (140 lanes, up to
131,072 ULP), tke (193 lanes, 22 ULP) and el (518 lanes, 45 ULP); 0.4% of
4,000 randomly drawn columns (``refute_columns.py fuzz``) hit it unaided.
The kernel now evaluates WOOF's own float32 tanhf (``sh_tanhf``, the twin of
mynn_pbl.cu's ``mynn_tanhf``) and the strict build is WRF's word on every
lane of this fixture.

Measured 2026-10-07 on W1 cards 0 and 4 (RTX PRO 6000 Blackwell Server
Edition, NVRTC 12.9.86, cupy 14.2.0), strict (GPUWM_WRF_EXACT=1) and default
arithmetic, one process each.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify.shinhong_column_oracle import (
    LEVEL_FIELDS, OOB_FIELDS, SHINHONG_COLUMNS_DIR, SURFACE_FIELDS, compare,
    load_columns, wrf_passthrough_violations)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "shinhong_column_oracle"
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "shinhong.cu"
TANH_DIR = ROOT / "tests" / "data" / "oracles" / "shinhong_columns_tanh"
F32 = np.float32

#: kpbl == kte with tke_diag = 1 (WRF reads q2xk(kte+1), :1557): tke/el
#: excluded there, as in the wide oracle.
OOB_COLUMNS = tuple(range(1036, 1048))

#: Default arithmetic only (CuPy's -ftz=true): (differing lanes, lanes where
#: WRF wrote a subnormal and the kernel a zero, worst ULP over the rest).
DEFAULT_FLUSH = {"dv": (26, 22, 1), "dqc": (1048, 1028, 6),
                 "dqi": (843, 843, 0)}

#: (x, the reference host's tanhf(x), the once-rounded double tanh(x)),
#: float32 words, recorded on W1 (glibc 2.39) at uwstx arguments of this
#: fixture plus two arguments where the two agree.
HOST_TANHF_WORDS = (
    (0xBF987710, 0xBF54B7D6, 0xBF54B7D5),
    (0xBF87ECC0, 0xBF49512A, 0xBF49512B),
    (0xBF45E760, 0xBF26119E, 0xBF26119D),
    (0xBF365680, 0xBF1CB1F6, 0xBF1CB1F5),
    (0xBF1D0A20, 0xBF0BEA1D, 0xBF0BEA1E),
    (0xBF115CE0, 0xBF0385B5, 0xBF0385B6),
    (0xBF03BDE0, 0xBEF2730C, 0xBEF2730B),
    (0xBEB5C180, 0xBEAE7CB5, 0xBEAE7CB4),
    (0xBE832900, 0xBE805D4C, 0xBE805D4B),
    (0x3F0ABE10, 0x3EFD2B73, 0x3EFD2B75),
    (0x3F166AA0, 0x3F07344A, 0x3F07344B),
    (0x3F79A8C0, 0x3F404125, 0x3F404126),
    (0x3FD65788, 0x3F6EA166, 0x3F6EA165),
    (0x3FD75070, 0x3F6EE232, 0x3F6EE233),
    (0x3F400000, 0x3F22991F, 0x3F22991F),
    (0xC0500000, 0xBF7F3B3D, 0xBF7F3B3D),
)


def _f(word: int) -> np.float32:
    return np.uint32(word).view(np.float32)


def _w(x) -> int:
    return int(np.asarray(x, np.float32).view(np.uint32))


@functools.cache
def _columns(ref: str = "wrf-out.bin"):
    return load_columns(TANH_DIR, outputs=ref)


def _gunzip(directory: Path, name: str) -> bytes:
    return gzip.decompress((directory / f"{name}.gz").read_bytes())


def _uwstx(ust, wstar) -> np.float32:
    uwst = F32(abs(F32(F32(ust) / F32(wstar)) - F32(0.5)))
    return F32(F32(F32(-80.0) * uwst) + F32(14.0))


def _csfac(t) -> np.float32:
    return F32(F32(0.5) * F32(F32(t) + F32(3.0)))


# --------------------------------------------------------------------------
# CPU: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_bytes_are_the_ones_the_oracle_run_recorded():
    recorded = {}
    for line in (TANH_DIR / "oracle-sha256sums.txt").read_text(
            encoding="ascii").splitlines():
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    assert recorded["module_bl_shinhong.F"] == (
        "99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56")
    for name in ("columns-in.bin", "wrf-out.bin"):
        assert hashlib.sha256(_gunzip(TANH_DIR, name)).hexdigest() == (
            recorded[name]), name
    for name in ("refute_columns.py", "run_shinhong_columns.F90"):
        assert hashlib.sha256((TOOL / name).read_bytes()).hexdigest() == (
            recorded[name]), f"{name} changed since the fixture was recorded"


def test_the_generator_reproduces_the_probe_columns():
    """The probes are a function of the wide oracle and refute_columns.py,
    choosing ust with WOOF's own float32 tanhf (on W1 the C library's tanhf
    chose the same bytes)."""
    spec = importlib.util.spec_from_file_location(
        "shinhong_refute_columns", TOOL / "refute_columns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data, labels = mod.tanh_records(
        _gunzip(SHINHONG_COLUMNS_DIR, "columns-in.bin"),
        _gunzip(SHINHONG_COLUMNS_DIR, "wrf-out.bin"))
    assert len(labels) == 1134
    assert data == _gunzip(TANH_DIR, "columns-in.bin")


def test_the_woof_tanhf_has_the_reference_host_words():
    from gpuwm.core.mynn_pbl import _tanhf

    for x, host, _rounded in HOST_TANHF_WORDS:
        assert _w(_tanhf(_f(x))) == host, hex(x)


def test_every_probe_column_lands_where_csfac_moves():
    """Non-vacuity: on every column the once-rounded double tanh gives a
    different csfac word than the reference host's tanhf, at the uwstx WRF
    itself formed (wstar is WRF's own output word, untouched by ust)."""
    from gpuwm.core.mynn_pbl import _tanhf

    cols = _columns()
    assert len(cols) == 1134
    for c in cols:
        wstar = c.wrf["wstar"][0]
        assert wstar > 0, c.index
        x = _uwstx(c.scalars["ust"], wstar)
        assert -10.0 < x < 14.0
        host = _csfac(_tanhf(x))
        rounded = _csfac(F32(math.tanh(float(x))))
        assert _w(host) != _w(rounded), (c.index, float(x))
    assert tuple(c.index for c in cols if c.oob_tke) == OOB_COLUMNS
    assert wrf_passthrough_violations(cols) == []


def test_the_kernel_forms_csfac_with_its_float32_tanhf():
    src = "\n".join(line.split("//", 1)[0] for line in
                    KERNEL.read_text(encoding="utf-8").splitlines())
    assert "csfac = 0.5f * (sh_tanhf(uwstx) + 3.0f);" in src
    strict_arm=src.split("#ifdef GPUWM_WRF_EXACT",1)[1].split("#else",1)[0]
    assert "tanh((double)uwstx)" not in strict_arm
    assert "__double2float_rn(tanh((double)uwstx))" in src
    assert "__device__ real sh_tanhf(real x)" in src


# --------------------------------------------------------------------------
# GPU: the kernel against every word, and its tanhf against its twin.
# --------------------------------------------------------------------------

from conftest import requires_gpu  # noqa: E402


def _strict() -> bool:
    from gpuwm import wrf_exact

    return wrf_exact.ENABLED


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_the_kernel_is_wrf_word_for_word_on_the_tanh_probes():
    import cupy  # noqa: F401

    from gpuwm.verify.shinhong_column_oracle import run_port

    cols = _columns()
    port = run_port(cols)
    result = compare(cols, port)
    assert result["kpbl_mismatch"] == []
    assert result["oob_excluded"] == 2 * len(OOB_COLUMNS)
    flush = {} if _strict() else DEFAULT_FLUSH
    tiny = float(np.finfo(np.float32).tiny)
    for name, st in result["fields"].items():
        assert st.nan_lanes == 0, (name, st.as_dict())
        assert (st.differing, st.ftz_lanes, st.max_ulp_outside_ftz) == (
            flush.get(name, (0, 0, 0))), (name, st.as_dict())
        assert st.max_abs < tiny, (name, st.as_dict())
    if not _strict():
        return
    checked = 0
    for c in cols:
        got = port[c.index]
        for name, wname in {**LEVEL_FIELDS, **SURFACE_FIELDS}.items():
            if c.oob_tke and name in OOB_FIELDS:
                continue
            g = np.ascontiguousarray(got[name], np.float32).reshape(-1)
            assert np.array_equal(g.view(np.uint32),
                                  c.wrf[wname].reshape(-1).view(np.uint32)), (
                c.index, name)
            checked += g.size
    # seven level fields on 52,640 lanes, tke and el on 52,220 (the OOB
    # columns' 420 excluded), hpbl/wstar/delta on 1,134 columns
    assert checked == 7 * 52_640 + 2 * 52_220 + 3 * 1134


def _tanh_sweep() -> np.ndarray:
    """Every 29th float32 bit pattern over the normal arguments of
    [-10, 14] (about 36 million), plus +/-0 and the pinned words."""
    pos = np.arange(0x00800000, 0x41600001, 29, dtype=np.uint32)
    neg = np.arange(0x80800000, 0xC1200001, 29, dtype=np.uint32)
    pinned = np.asarray([w for w, _h, _r in HOST_TANHF_WORDS], np.uint32)
    words = np.concatenate([pos, neg, pinned,
                            np.asarray([0, 0x80000000], np.uint32)])
    return words.view(np.float32)


@pytest.mark.gpu
@requires_gpu
def test_sh_tanhf_is_its_mynn_twin_and_the_host_word():
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    x = _tanh_sweep()
    n = x.size
    dx = cp.asarray(x)
    sh = cp.empty(n, dtype=cp.float32)
    threads = 256
    get_kernel("shinhong", "shinhong_tanhf_probe")(
        ((n + threads - 1) // threads,), (threads,), (dx, sh, np.int64(n)))
    twin = cp.empty(3 * n, dtype=cp.float32)
    get_kernel("mynn_pbl", "mynn_glibc_libm_probe2")(
        ((n + threads - 1) // threads,), (threads,), (dx, twin, np.int32(n)))
    sh_h = cp.asnumpy(sh)
    twin_h = cp.asnumpy(twin).reshape(-1, 3)[:, 0]
    differ = np.flatnonzero(sh_h.view(np.uint32) != twin_h.view(np.uint32))
    assert differ.size == 0, (differ[:8], x[differ[:8]])
    for i, (_x, host, _r) in enumerate(HOST_TANHF_WORDS):
        assert _w(sh_h[n - 2 - len(HOST_TANHF_WORDS) + i]) == host
    # the old spelling is a different function on this sweep
    rounded = np.tanh(x.astype(np.float64)).astype(np.float32)
    assert int((rounded.view(np.uint32) != sh_h.view(np.uint32)).sum()) > 0
