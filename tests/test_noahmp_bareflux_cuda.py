"""Device acceptance gate for the Noah-MP BARE_FLUX kernel.

Same bar as the CPU side: bitwise identity with the unmodified WRF v4.6.1
module over the committed oracle fixtures.  Nothing is relaxed for the GPU.

Three things this has to prove, not just one:

1. ``noahmp_bare_flux`` reproduces every row of ``noahmp-bareflux.csv`` bit
   for bit.
2. The glibc kernels the leaf is built on -- ``glibc_logf``, ``glibc_atanf``,
   ``glibc_powf`` -- are bitwise right on the device over a sweep far wider
   than the leaf reaches, checked against ``noahmp-bareflux-libm.csv``, which
   is the *live glibc 2.39 symbol's* own output.  A mis-folded table entry
   could otherwise hide inside a leaf that never touches it.
3. The ESAT coefficient table survived ptxas, checked against
   ``noahmp-bareflux-esat.csv``, which is the Fortran ESAT itself over 12007
   temperatures.  ptxas 12.8's constant folder does not honour
   round-to-nearest-even on literal arrays and ESAT is exactly such an array,
   which is why it lives in ``__constant__`` memory; this is what shows the
   precaution worked.

Every reference here is an oracle, not the CPU transcription, so this file
imports nothing that needs Python 3.13 -- it runs on the GPU box as shipped.
"""

from __future__ import annotations

import csv
import os
import struct
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "noahmp_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwm.core.kernels import load_module                    # noqa: E402
from gen_bareflux_cases import (                              # noqa: E402
    ARRAY_FIELDS, INT_FIELDS, NSNOW, NSOIL, OUT_FIELDS, REAL_FIELDS,
)

_DATA = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
FIXTURE = os.path.join(_DATA, "noahmp-bareflux.csv")
LIBM_FIXTURE = os.path.join(_DATA, "noahmp-bareflux-libm.csv")
ESAT_FIXTURE = os.path.join(_DATA, "noahmp-bareflux-esat.csv")

_ARRAY_COLS = [f"{name}{k}" for name in ARRAY_FIELDS
               for k in range(-NSNOW + 1, NSOIL + 1)]
# Must match the input layout documented in noahmp_bareflux.cu.
_IN_ORDER = REAL_FIELDS + _ARRAY_COLS


def _uh(word: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(word, 16)), dtype=np.float32)[0]


def _read(path):
    with open(path, newline="") as handle:
        lines = [ln for ln in handle if not ln.startswith("#")]
    return list(csv.DictReader(lines))


@pytest.fixture(scope="module")
def rows():
    return _read(FIXTURE)


@pytest.fixture(scope="module")
def module():
    return load_module("noahmp_bareflux")


def _assert_bit_exact(got: np.ndarray, want: np.ndarray, labels, what: str):
    gb = np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
    wb = np.ascontiguousarray(want, dtype=np.float32).view(np.uint32)
    bad = np.flatnonzero(gb != wb)
    assert bad.size == 0, (
        f"{what}: {bad.size}/{gb.size} lanes differ; first at "
        f"{labels[int(bad[0])]}: got 0x{gb[int(bad[0])]:08X} "
        f"want 0x{wb[int(bad[0])]:08X}"
    )


def test_bare_flux_kernel_bitwise(rows, module):
    n = len(rows)
    assert n >= 27
    flat = np.empty((n, len(_IN_ORDER)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(_IN_ORDER):
            flat[i, j] = _uh(row[name])
    ints = np.empty((n, len(INT_FIELDS)), dtype=np.int32)
    for i, row in enumerate(rows):
        for j, name in enumerate(INT_FIELDS):
            ints[i, j] = int(row[name])

    d_in = cp.asarray(flat.ravel())
    d_int = cp.asarray(ints.ravel())
    d_out = cp.zeros(n * len(OUT_FIELDS), dtype=cp.float32)

    fn = module.get_function("noahmp_bare_flux")
    fn(((n + 63) // 64,), (64,), (d_in, d_int, d_out, np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, len(OUT_FIELDS))

    labels = [row["case"] for row in rows]
    for j, name in enumerate(OUT_FIELDS):
        want = np.array([_uh(row[name]) for row in rows], dtype=np.float32)
        _assert_bit_exact(got[:, j].copy(), want, labels, f"bare_flux.{name}")


def test_device_libm_matches_live_glibc(module):
    ref = _read(LIBM_FIXTURE)
    assert len(ref) >= 2000
    x = np.array([_uh(r["x"]) for r in ref], dtype=np.float32)
    n = x.size

    d_x = cp.asarray(x)
    d_out = cp.zeros(3 * n, dtype=cp.float32)
    fn = module.get_function("noahmp_bareflux_libm_probe")
    fn(((n + 255) // 256,), (256,), (d_x, d_out, np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, 3)

    labels = [f"x=0x{r['x']}" for r in ref]
    for j, name in enumerate(("logf", "atanf", "powf025")):
        want = np.array([_uh(r[name]) for r in ref], dtype=np.float32)
        _assert_bit_exact(got[:, j].copy(), want, labels, f"glibc_{name}")


def test_device_esat_matches_the_fortran(module):
    ref = _read(ESAT_FIXTURE)
    assert len(ref) >= 10000
    t = np.array([_uh(r["t"]) for r in ref], dtype=np.float32)
    n = t.size

    d_t = cp.asarray(t)
    d_out = cp.zeros(4 * n, dtype=cp.float32)
    fn = module.get_function("noahmp_bareflux_esat_probe")
    fn(((n + 255) // 256,), (256,), (d_t, d_out, np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, 4)

    labels = [f"T=0x{r['t']}" for r in ref]
    for j, name in enumerate(("esw", "esi", "desw", "desi")):
        want = np.array([_uh(r[name]) for r in ref], dtype=np.float32)
        _assert_bit_exact(got[:, j].copy(), want, labels, f"esat.{name}")


def _compile_source(src: str, options=("-std=c++17",)):
    from gpuwm.core.kernels import _preamble

    mod = cp.RawModule(code=_preamble() + src, options=tuple(options))
    mod.compile()
    return mod


def _kernel_source() -> str:
    from gpuwm.core.kernels import _KDIR

    return (_KDIR / "noahmp_bareflux.cu").read_text()


def _swap_libm(name: str, cuda_name: str) -> str:
    """Point every call site of glibc_<name> at CUDA's own <cuda_name>.

    The definition has to be renamed to something that does NOT contain the
    substring being replaced, or the substitution silently renames the
    definition too and the mutant becomes a no-op -- which is exactly what
    happened the first time this control was written, and why it appeared to
    pass.
    """
    src = _kernel_source()
    defn = f"__device__ float glibc_{name}("
    assert defn in src, f"definition of glibc_{name} vanished"
    src = src.replace(defn, f"__device__ float dead_{name}_impl(")
    assert f"glibc_{name}(" in src, f"no call sites of glibc_{name} remain"
    return src.replace(f"glibc_{name}(", f"{cuda_name}(")


def _mutate(old: str, new: str) -> str:
    src = _kernel_source()
    assert old in src, f"mutation target {old!r} vanished from the kernel"
    return src.replace(old, new)


def _fixture_rejects(mod, rows) -> bool:
    n = len(rows)
    flat = np.array([[_uh(row[name]) for name in _IN_ORDER] for row in rows],
                    dtype=np.float32)
    ints = np.array([[int(row[name]) for name in INT_FIELDS] for row in rows],
                    dtype=np.int32)
    d_out = cp.zeros(n * len(OUT_FIELDS), dtype=cp.float32)
    mod.get_function("noahmp_bare_flux")(
        ((n + 63) // 64,), (64,),
        (cp.asarray(flat.ravel()), cp.asarray(ints.ravel()), d_out,
         np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, len(OUT_FIELDS))
    want = np.array([[_uh(row[name]) for name in OUT_FIELDS] for row in rows],
                    dtype=np.float32)
    return bool((got.view(np.uint32) != want.view(np.uint32)).any())


def _probe_rejects(mod) -> bool:
    ref = _read(LIBM_FIXTURE)
    x = np.array([_uh(r["x"]) for r in ref], dtype=np.float32)
    n = x.size
    d_out = cp.zeros(3 * n, dtype=cp.float32)
    mod.get_function("noahmp_bareflux_libm_probe")(
        ((n + 255) // 256,), (256,), (cp.asarray(x), d_out, np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, 3)
    want = np.array([[_uh(r[c]) for c in ("logf", "atanf", "powf025")]
                     for r in ref], dtype=np.float32)
    return bool((got.view(np.uint32) != want.view(np.uint32)).any())


def test_device_negative_controls_libm():
    """CUDA's own logf/atanf/powf must be rejected by the glibc reference.

    This is why ``noahmp-bareflux-libm.csv`` exists as a separate fixture:
    CUDA's device libm is a different set of functions from glibc's and the
    leaf alone would not reliably see the difference.
    """
    for name, cuda_name in (("logf", "logf"), ("atanf", "atanf"),
                            ("powf", "powf")):
        mod = _compile_source(_swap_libm(name, cuda_name))
        assert _probe_rejects(mod), (
            f"device libm negative control went undetected: glibc_{name} -> "
            f"CUDA {cuda_name}"
        )


def test_device_negative_controls_rounding(rows):
    """Un-pinning the rounding intrinsics must be caught by the case fixture.

    ``-fmad=true`` is nvcc's default, so once both the multiply and the add
    are bare operators the compiler is free to fuse a site the Fortran
    rounded twice.  Un-pinning the multiply alone is not enough -- every
    multiply here feeds a still-pinned __fadd_rn/__fsub_rn, so there is
    nothing to fuse with; that is a fact about the kernel's shape, not a
    weakness of the fixture, and the combined control below is the one that
    actually tests the claim.
    """
    combined = _kernel_source()
    for old, new in (
        ("#define FADD(a, b) __fadd_rn((a), (b))",
         "#define FADD(a, b) ((a) + (b))"),
        ("#define FSUB(a, b) __fsub_rn((a), (b))",
         "#define FSUB(a, b) ((a) - (b))"),
        ("#define FMUL(a, b) __fmul_rn((a), (b))",
         "#define FMUL(a, b) ((a) * (b))"),
    ):
        assert old in combined, f"macro {old!r} vanished from the kernel"
        combined = combined.replace(old, new)
    assert _fixture_rejects(_compile_source(combined), rows), (
        "un-pinning FADD/FSUB/FMUL together went undetected"
    )

    # The double-precision FMA sites are a different story and the honest
    # answer is measured, not asserted.  Replacing __fma_rn(a,b,c) with a bare
    # a*b+c and compiling with --fmad=false (so nvcc cannot contract it back)
    # changes NO float32 result: a sweep of 120,000,000 random float32 inputs
    # through the libm probe on the RTX 5090 found zero differing lanes in
    # logf, atanf or powf.  glibc's double-precision kernels carry roughly
    # 2^-33 relative error against a 2^-24 float32 ULP, so the final rounding
    # absorbs the difference.  __fma_rn is kept anyway -- it makes the result
    # independent of the contraction flag, which is asserted below -- but it
    # is not claimed to be a negative control this fixture can fail.
    unpinned_dfma = _compile_source(
        _mutate("#define DFMA(a, b, c) __fma_rn((a), (b), (c))",
                "#define DFMA(a, b, c) ((a) * (b) + (c))"),
        ("-std=c++17", "--fmad=false"))
    assert not _fixture_rejects(unpinned_dfma, rows)
    assert not _probe_rejects(unpinned_dfma)

    # What __fma_rn does buy: the kernel's answer does not depend on nvcc's
    # contraction flag.  That is the property worth having and it is testable.
    flagged = _compile_source(_kernel_source(), ("-std=c++17", "--fmad=false"))
    assert not _fixture_rejects(flagged, rows),         "the kernel's result depends on nvcc's contraction flag"
    assert not _probe_rejects(flagged),         "the libm kernels' results depend on nvcc's contraction flag"


def test_kernel_source_still_pins_rounding_and_constants():
    """A refactor that drops a rounding intrinsic or inlines a table must fail.

    Both are silent at runtime on most inputs, so the source is asserted
    directly rather than trusted.
    """
    path = os.path.join(_ROOT, "gpuwm", "core", "kernels",
                        "noahmp_bareflux.cu")
    with open(path, encoding="utf-8") as handle:
        src = handle.read()
    for needle in ("__fadd_rn", "__fsub_rn", "__fmul_rn", "__fdiv_rn",
                   "__fsqrt_rn", "__fma_rn",
                   "__constant__ unsigned int C_ESAT",
                   "__constant__ unsigned int C_ATAN",
                   "__constant__ unsigned int C_F32"):
        assert needle in src, f"noahmp_bareflux.cu lost {needle}"
    # No bare float arithmetic operator may appear in the numeric bodies.
    # Array subscripts and the __global__ launchers are pure integer index
    # arithmetic, so they are masked out first rather than whitelisted by
    # accident.
    import re

    body = src[src.index("__device__ float glibc_atanf"):]
    body = re.sub(r'extern "C" __global__.*?\n\}\n', "", body, flags=re.S)
    offenders = []
    for line in body.splitlines():
        code = line.split("//", 1)[0]
        code = re.sub(r"\[[^\]]*\]", "[]", code)      # mask index arithmetic
        stripped = code.lstrip()
        if stripped.startswith(("int ", "unsigned ", "const int ",
                                "const float *", "#define", "#undef")):
            continue
        if "for (" in code:
            continue
        if any(op in code for op in (" * ", " / ", " + ", " - ")):
            offenders.append(line.strip())
    assert not offenders, (
        "un-pinned arithmetic in the kernel body: " + repr(offenders[:5])
    )


def _physical_kwargs(row: dict) -> dict:
    """The physical BARE_FLUX keyword call for one fixture row."""
    from gen_bareflux_cases import unhexf

    kw = {name: int(row[name]) for name in INT_FIELDS if name != "iurban"}
    kw["urban_flag"] = bool(int(row["iurban"]))
    for name in REAL_FIELDS:
        kw[name] = unhexf(row[name])
    for name in ARRAY_FIELDS:
        kw[name] = [unhexf(row[f"{name}{k}"])
                    for k in range(-NSNOW + 1, NSOIL + 1)]
    kw["nsnow"] = NSNOW
    kw["nsoil"] = NSOIL
    kw["opt_sfc"] = int(row["opt_sfc"])
    kw["opt_stc"] = int(row["opt_stc"])
    return kw


def test_the_runtime_batch_reproduces_the_fixture(rows):
    """The gate on what the forecast actually calls.

    ``test_bare_flux_kernel_bitwise`` above hand-builds the flat row the kernel
    indexes.  The runtime does not: it packs a physical Python keyword call
    through :func:`gpuwm.core.noahmp_bareflux_gpu.evaluate_bare_flux_calls` and
    unpacks a :class:`BareFluxOut`.  Everything between those two -- the slot
    order, the layer origin, the ``urban_flag`` to ``iurban`` conversion and the
    output field order -- is only exercised here.  All of it must be bitwise.
    """
    from gen_bareflux_cases import unhexf
    from gpuwm.core.noahmp_bareflux_gpu import evaluate_bare_flux_calls

    calls = [((), _physical_kwargs(row)) for row in rows]
    got = evaluate_bare_flux_calls(calls)
    assert len(got) == len(rows)

    # BareFluxOut names the three INOUTs without the fixture's ``_out`` suffix.
    attribute = {name: name[:-4] if name.endswith("_out") else name
                 for name in OUT_FIELDS}
    labels = [row["case"] for row in rows]
    for name in OUT_FIELDS:
        want = np.array([unhexf(row[name]) for row in rows], dtype=np.float32)
        have = np.array([getattr(out, attribute[name]) for out in got],
                        dtype=np.float32)
        _assert_bit_exact(have, want, labels, f"batch.{name}")


#: Measured, not assumed: a one-ULP nudge of each live scalar on one fixture
#: column, and which of the thirteen outputs moved.  This is what chose the
#: perturbation below, and the first attempt -- a one-ULP nudge of the entry
#: ``TGB`` -- is in here because it proved *nothing*: five Newton iterations
#: converge and the seed is washed out.  ``CM``, ``CH`` and ``QSFC`` are inert
#: for a different reason: under opt_sfc=1 the kernel never reads them.
_ONE_ULP_REACH = {
    "sfctmp": 10,   # cm ch tauxb tauyb irb shb evb ghb q2b ehb2
    "rhsur": 7,     # tgb qsfc shb evb ghb t2mb q2b
    "ur": 4,
    "rhoair": 4,
    "eair": 2,
    "psfc": 2,
    "lwdn": 1,
    "uu": 1,
    "vv": 1,
    "rsurf": 1,
    "gamma": 1,
    "tgb": 0,       # the Newton seed; converged away
    "cm": 0,
    "ch": 0,
    "qsfc": 0,
    "dt": 0, "sag": 0, "thair": 0, "qair": 0, "zlvl": 0, "z0m": 0,
    "emg": 0, "lathea": 0, "q2": 0, "dx": 0, "dz8w": 0, "sfcprs": 0,
}


def test_the_runtime_batch_gate_can_fail(rows):
    """Perturb one input by a single ULP and the batch gate must reject it.

    The batch is a repacking, so the way it fails quietly is by reading the
    wrong slot for a column, not by returning nonsense.  A one-ULP nudge of a
    live input on a single column is the smallest thing that must still be
    visible through it -- but *which* input is a measurement, not a guess: see
    :data:`_ONE_ULP_REACH`.  ``SFCTMP`` moves ten of the thirteen outputs and
    is the honest choice; the entry ``TGB`` moves none of them.
    """
    import struct

    from gen_bareflux_cases import unhexf
    from gpuwm.core.noahmp_bareflux_gpu import evaluate_bare_flux_calls

    calls = [((), _physical_kwargs(row)) for row in rows]
    bits = struct.unpack("<I", struct.pack("<f", calls[0][1]["sfctmp"]))[0]
    calls[0][1]["sfctmp"] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    got = evaluate_bare_flux_calls(calls)

    attribute = {name: name[:-4] if name.endswith("_out") else name
                 for name in OUT_FIELDS}
    moved = [name for name in OUT_FIELDS
             if np.float32(getattr(got[0], attribute[name])).tobytes()
             != np.float32(unhexf(rows[0][name])).tobytes()]
    assert len(moved) >= 8, (
        "a one-ULP SFCTMP perturbation moved only "
        f"{moved}; the batch round trip is not sensitive to its inputs")

    # ...and the columns behind it are untouched, which is the pairing claim.
    for index in range(1, len(rows)):
        for name in OUT_FIELDS:
            assert (np.float32(getattr(got[index], attribute[name])).tobytes()
                    == np.float32(unhexf(rows[index][name])).tobytes()), (
                f"perturbing column 0 moved column {index} {name}")


def test_the_newton_seed_really_is_unobservable_at_one_ulp(rows):
    """Record the measurement that rejected the first falsification attempt.

    This is not a property anyone should assume.  It is the reason the gate
    above uses SFCTMP, and if a future change to NITERB or to the convergence
    makes the entry TGB observable again, this test says so rather than leaving
    a silently weaker falsification in place.
    """
    import struct

    from gpuwm.core.noahmp_bareflux_gpu import evaluate_bare_flux_calls

    base = evaluate_bare_flux_calls(
        [((), _physical_kwargs(row)) for row in rows])
    calls = [((), _physical_kwargs(row)) for row in rows]
    for name in ("tgb", "cm", "ch", "qsfc"):
        word = struct.unpack("<I", struct.pack("<f", calls[0][1][name]))[0]
        calls[0][1][name] = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    got = evaluate_bare_flux_calls(calls)
    attribute = {name: name[:-4] if name.endswith("_out") else name
                 for name in OUT_FIELDS}
    moved = [name for name in OUT_FIELDS
             if np.float32(getattr(got[0], attribute[name])).tobytes()
             != np.float32(getattr(base[0], attribute[name])).tobytes()]
    assert not moved, (
        "the entry TGB/CM/CH/QSFC are now observable at one ULP "
        f"({moved}); the falsification in this file may be re-pointed at them")
