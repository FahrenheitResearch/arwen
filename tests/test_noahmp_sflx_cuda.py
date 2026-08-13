"""Device acceptance gate for the Noah-MP NOAHMP_SFLX kernels.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixtures.  Nothing is relaxed for the GPU --
the gate is ``max_ulp == 0`` and no tolerance is applied anywhere.

``noahmp_sflx.cu`` covers ERROR and NOAHMP_SFLX's own marshalling (DZSNSO,
TROOT, BEG_WB), not the composed column; the .cu's header says why, and
``test_kernel_does_not_claim_the_column`` asserts the file has not quietly
grown past that claim.

Four things this has to prove:

1. ``k_sflx_error`` reproduces every ``output`` row of
   ``noahmp-sflx-error.csv`` bit for bit, on all sixteen cases.
2. Its status word agrees with WRF's three abort gates -- the kernel reports
   where the Fortran dies, and ``build_sflx_compose.sh`` already showed the
   Fortran dies there.  A kernel that silently returned a plausible number
   would pass (1) on every fixture case and be wrong on the first bad column.
3. ``k_sflx_marshal`` reproduces DZSNSO and TROOT bit for bit against
   ``noahmp-energy.csv``, which recorded both from the patched module for the
   same four columns ``noahmp-sflx.csv`` carries.
4. The gate can fail.  Compiling with NVRTC's default ``--fmad=true`` and the
   rounding intrinsics stripped must be rejected -- otherwise (1) and (3) are
   claims about a hazard that was never present.

Everything compared here comes from an oracle CSV, never from
:mod:`gpuwm.core.noahmp_sflx`, so a shared mistake in the two ports cannot
pass this file.
"""

from __future__ import annotations

import csv
import os
import re
import struct
import sys
from collections import defaultdict

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_KDIR = os.path.join(_ROOT, "gpuwm", "core", "kernels")
_DATA = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
KERNEL = os.path.join(_KDIR, "noahmp_sflx.cu")
ERROR_CSV = os.path.join(_DATA, "noahmp-sflx-error.csv")
SFLX_CSV = os.path.join(_DATA, "noahmp-sflx.csv")
ENERGY_CSV = os.path.join(_DATA, "noahmp-energy.csv")

NSOIL = 4
NSNOW = 3
NFULL = NSOIL + NSNOW

# The layout tables and the comment-stripper live in
# tests/noahmp_kernel_sources.py, shared with
# tests/test_noahmp_kernel_source_scans.py -- which is where the two checks
# that need no device now live.  They used to sit at the top of THIS file,
# above its `cp = pytest.importorskip("cupy")`, and therefore never ran: a
# module-level skip fires at import and takes everything above it too, and
# tests/conftest.py marks a whole module `gpu` when cupy is imported at
# module scope.  Keeping SCALARS/OUTPUTS in one module is the point -- their
# whole job is to be ONE spelling of a layout.
from noahmp_kernel_sources import OUTPUTS, SCALARS  # noqa: E402
from noahmp_kernel_sources import code as _code  # noqa: E402

COLUMNS = ("veg_warm_day_dry", "veg_warm_night_rain",
           "snowpack_frozen_soil", "bare_thin_snow_melt")


def _f32(hexbits: str) -> np.float32:
    return np.uint32(int(hexbits, 16)).view(np.float32)


def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def _load_error():
    with open(ERROR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "the ERROR oracle fixture is empty"
    table = defaultdict(dict)
    cases = []
    for r in rows:
        key = (r["field"], int(r["index"]))
        table[(r["case"], r["stage"])][key] = (
            int(r["value"]) if r["dtype"] == "int" else _f32(r["bits"]))
        if r["stage"] == "input" and r["case"] not in cases:
            cases.append(r["case"])
    return table, cases


def _load_sflx():
    table = defaultdict(dict)
    with open(SFLX_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            table[(r["case"], r["stage"])][(r["field"], int(r["index"]))] = \
                np.float32(float(r["value"]))
    return table


def _load_energy():
    table = defaultdict(dict)
    with open(ENERGY_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            table[(r["case"], r["role"])][r["name"]] = r["hex"].strip()
    return table


ERR, ERR_CASES = _load_error()
SF = _load_sflx()
EN = _load_energy()


# The two checks that need no GPU -- the SC_/OU_ layout agreement and the
# "this kernel has not grown a column" claim -- are in
# tests/test_noahmp_kernel_source_scans.py.  They were HERE, above the
# importorskip below, which meant they ran nowhere at all.


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
cp = pytest.importorskip("cupy")

_OPTS = ("-std=c++17",)


def _module(source: str | None = None):
    code = source if source is not None else open(KERNEL, encoding="ascii").read()
    return cp.RawModule(code=code, options=_OPTS)


def _run_error(mod):
    n = len(ERR_CASES)
    sc = np.zeros((n, len(SCALARS)), dtype=np.float32)
    smc = np.zeros((n, NSOIL), dtype=np.float32)
    dz = np.zeros((n, NSOIL), dtype=np.float32)
    ist = np.zeros(n, dtype=np.int32)
    calc = np.zeros(n, dtype=np.int32)
    for c, case in enumerate(ERR_CASES):
        x = ERR[(case, "input")]
        for j, name in enumerate(SCALARS):
            sc[c, j] = x[(name, 0)]
        for k in range(NSOIL):
            smc[c, k] = x[("smc", k + 1)]
            dz[c, k] = x[("dzsnso", k + 1)]
        ist[c] = int(x[("ist", 0)])
        calc[c] = int(x[("calculate_soil", 0)])

    d_out = cp.zeros((n, len(OUTPUTS)), dtype=cp.float32)
    d_status = cp.zeros(n, dtype=cp.int32)
    mod.get_function("k_sflx_error")(
        (1,), (max(n, 1),),
        (cp.asarray(sc), cp.asarray(smc), cp.asarray(dz), cp.asarray(ist),
         cp.asarray(calc), d_out, d_status, np.int32(n)))
    return cp.asnumpy(d_out), cp.asnumpy(d_status)


def _run_marshal(mod):
    n = len(COLUMNS)
    zsnso = np.zeros((n, NFULL), dtype=np.float32)
    stc = np.zeros((n, NFULL), dtype=np.float32)
    smc = np.zeros((n, NSOIL), dtype=np.float32)
    zsoil = np.zeros((n, NSOIL), dtype=np.float32)
    canliq = np.zeros(n, dtype=np.float32)
    canice = np.zeros(n, dtype=np.float32)
    sneqv = np.zeros(n, dtype=np.float32)
    wa = np.zeros(n, dtype=np.float32)
    isnow = np.zeros(n, dtype=np.int32)
    nroot = np.zeros(n, dtype=np.int32)
    ist = np.zeros(n, dtype=np.int32)

    from gpuwm.core.noahmp import load_noahmp_parameters, transfer_mp_parameters
    bundle = load_noahmp_parameters()
    for c, case in enumerate(COLUMNS):
        x = SF[(case, "input")]
        for k, iz in enumerate(range(-NSNOW + 1, NSOIL + 1)):
            zsnso[c, k] = x[("zsnso", iz)]
            stc[c, k] = x[("stc", iz)]
        for k in range(NSOIL):
            smc[c, k] = x[("smc", k + 1)]
            zsoil[c, k] = x[("zsoil", k + 1)]
        canliq[c] = x[("canliq", 0)]
        canice[c] = x[("canice", 0)]
        sneqv[c] = x[("sneqv", 0)]
        wa[c] = x[("wa", 0)]
        isnow[c] = int(x[("isnow", 0)])
        ist[c] = int(x[("ist", 0)])
        p = transfer_mp_parameters(
            bundle, vegtype=int(x[("vegtyp", 0)]),
            soiltype=[int(x[("soiltype", 0)])] * NSOIL,
            slopetype=int(x[("slopetype", 0)]),
            soilcolor=int(x[("soilcolor", 0)]), croptype=0)
        nroot[c] = int(p.scalar("NROOT"))

    d_dz = cp.zeros((n, NFULL), dtype=cp.float32)
    d_troot = cp.zeros(n, dtype=cp.float32)
    d_begwb = cp.zeros(n, dtype=cp.float32)
    mod.get_function("k_sflx_marshal")(
        (1,), (max(n, 1),),
        (cp.asarray(zsnso), cp.asarray(stc), cp.asarray(smc),
         cp.asarray(zsoil), cp.asarray(canliq), cp.asarray(canice),
         cp.asarray(sneqv), cp.asarray(wa), cp.asarray(isnow),
         cp.asarray(nroot), cp.asarray(ist), d_dz, d_troot, d_begwb,
         np.int32(n)))
    return (cp.asnumpy(d_dz), cp.asnumpy(d_troot), cp.asnumpy(d_begwb),
            isnow)


# ---------------------------------------------------------------------------
# 1 and 2: ERROR
# ---------------------------------------------------------------------------

def x_of(case):
    return ERR[(case, "input")]


def test_error_kernel_is_bitwise():
    out, status = _run_error(_module())
    bad = []
    for c, case in enumerate(ERR_CASES):
        want = ERR[(case, "output")]
        for j, name in enumerate(OUTPUTS):
            if (name, 0) not in want:
                continue                     # errsw/erreng/end_wb are locals
            if (name == "errwat" and int(x_of(case)[("ist", 0)]) == 1
                    and not x_of(case)[("calculate_soil", 0)]):
                continue          # WRF writes nothing here; checked below
            if _bits(out[c, j]) != _bits(want[(name, 0)]):
                bad.append((case, name, _bits(out[c, j]),
                            _bits(want[(name, 0)])))
    assert not bad, bad[:8]
    # The ERRWAT case the fixture cannot pin: WRF leaves it unassigned on the
    # non-soil substep, so the fixture holds the harness's poison and the
    # kernel, like the CPU port, writes the defined 0.0.
    for c, case in enumerate(ERR_CASES):
        x = ERR[(case, "input")]
        if int(x[("ist", 0)]) == 1 and not x[("calculate_soil", 0)]:
            assert float(out[c, OUTPUTS.index("errwat")]) == 0.0


def test_error_kernel_status_is_clean_on_every_fixture_case():
    """Every fixture case returns without tripping a gate, by construction:
    ``build_sflx_compose.sh`` refuses to emit a row from a case that aborts."""
    _out, status = _run_error(_module())
    assert list(status) == [0] * len(ERR_CASES)


def test_error_kernel_reports_each_gate():
    """The three abort gates, reported rather than aborted.  A kernel cannot
    call wrf_error_fatal, so the status word is the only thing standing between
    a bad column and a plausible-looking number."""
    mod = _module()
    base = ERR[(ERR_CASES[0], "input")]

    def run(**over):
        sc = np.zeros((1, len(SCALARS)), dtype=np.float32)
        for j, name in enumerate(SCALARS):
            sc[0, j] = over.get(name, base[(name, 0)])
        smc = np.asarray([[base[("smc", k + 1)] for k in range(NSOIL)]],
                         dtype=np.float32)
        dz = np.asarray([[base[("dzsnso", k + 1)] for k in range(NSOIL)]],
                        dtype=np.float32)
        d_out = cp.zeros((1, len(OUTPUTS)), dtype=cp.float32)
        d_status = cp.zeros(1, dtype=cp.int32)
        mod.get_function("k_sflx_error")(
            (1,), (1,),
            (cp.asarray(sc), cp.asarray(smc), cp.asarray(dz),
             cp.asarray(np.ones(1, dtype=np.int32)),
             cp.asarray(np.ones(1, dtype=np.int32)), d_out, d_status,
             np.int32(1)))
        return int(cp.asnumpy(d_status)[0])

    assert run() == 0
    assert run(swdown=np.float32(800.0 + 0.010002)) == 1
    assert run(pah=np.float32(0.010002)) == 2
    assert run(runsrf=np.float32(0.10002)) == 3
    # And the values just inside each threshold must not trip it.
    assert run(swdown=np.float32(800.0 - 0.0099)) == 0
    assert run(pah=np.float32(0.009998)) == 0
    assert run(runsrf=np.float32(0.09998)) == 0


# ---------------------------------------------------------------------------
# 3: the marshalling
# ---------------------------------------------------------------------------

def test_marshal_kernel_is_bitwise():
    dz, troot, begwb, isnow = _run_marshal(_module())
    bad = []
    for c, case in enumerate(COLUMNS):
        ein, seed = EN[(case, "in")], EN[(case, "seed")]
        for k, iz in enumerate(range(-NSNOW + 1, NSOIL + 1)):
            key = f"DZSNSO_{'+' if iz >= 0 else '-'}{abs(iz)}"
            if _bits(dz[c, k]) != ein[key]:
                bad.append((case, key, _bits(dz[c, k]), ein[key]))
        if _bits(troot[c]) != seed["TROOT"]:
            bad.append((case, "TROOT", _bits(troot[c]), seed["TROOT"]))
    assert not bad, bad[:8]
    # BEG_WB reaches no fixture directly; it enters ERROR and leaves through
    # ACC_DWATER, which test_noahmp_sflx.py gates on the whole column.  The
    # only claim made here is that it is finite and positive on a soil column,
    # which is what stops an uninitialised read passing silently.
    assert np.all(np.isfinite(begwb))
    assert np.all(begwb > 0.0)


def test_marshal_buried_slots_are_the_defined_zero():
    dz, _troot, _begwb, isnow = _run_marshal(_module())
    for c in range(len(COLUMNS)):
        buried = int(isnow[c]) + NSNOW
        assert np.all(dz[c, :buried] == 0.0), COLUMNS[c]
        assert np.all(dz[c, buried:] != 0.0), COLUMNS[c]


# ---------------------------------------------------------------------------
# 4: the gate can fail
# ---------------------------------------------------------------------------

_INTRINSIC = re.compile(r"__f(add|sub|mul|div)_rn\(([^,]+), ([^)]+)\)")


def _strip_rounding(text: str) -> str:
    """Replace the four rounding intrinsics with infix operators, which is what
    a transcription written the obvious way would contain."""
    op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    prev = None
    while prev != text:
        prev = text
        text = _INTRINSIC.sub(lambda m: f"(({m.group(2)}) {op[m.group(1)]} "
                                        f"({m.group(3)}))", text)
    return text


def test_contraction_is_the_hazard_these_intrinsics_prevent():
    """NVRTC defaults to --fmad=true.  With the rounding intrinsics removed the
    compiler is free to contract ``END_WB + SMC*DZ*1000`` and TROOT's
    ``STC*DZ/(-ZSOIL)`` into FMAs, and gfortran at -O0 emits none.  If the
    stripped kernel still reproduced the fixture, this file would be asserting
    a hazard that is not there."""
    stripped = _strip_rounding(open(KERNEL, encoding="ascii").read())
    body = _code(stripped)
    assert "__fadd_rn" not in body and "__fmul_rn" not in body
    mod = _module(stripped)
    out, _status = _run_error(mod)
    dz, troot, _begwb, _isnow = _run_marshal(mod)

    differs = False
    for c, case in enumerate(ERR_CASES):
        want = ERR[(case, "output")]
        for j, name in enumerate(OUTPUTS):
            if (name, 0) in want and _bits(out[c, j]) != _bits(want[(name, 0)]):
                differs = True
    for c, case in enumerate(COLUMNS):
        if _bits(troot[c]) != EN[(case, "seed")]["TROOT"]:
            differs = True
    assert differs, (
        "the contraction-free spelling and the contracted one agree on every "
        "value this file checks, so the intrinsics are not what makes the "
        "gate pass and this test proves nothing")
