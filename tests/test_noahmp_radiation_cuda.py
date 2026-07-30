"""max_ulp-0 gate for the CUDA half of the Noah-MP radiation leaves.

Same fixtures as ``tests/test_noahmp_radiation.py``, same bar: bitwise
identity against the WRF-4.6.1 oracle.  Skipped when there is no CuPy or no
device; it is meant to be run on the rented RTX 5090, never on the
workstation GPU.

The module is compiled here rather than through ``gpuwm.core.kernels`` so the
compile options can be pinned locally (``--fmad=false``) without editing a
file other lanes share.  Every arithmetic site in the kernel already uses an
explicit ``*_rn`` intrinsic, so ``--fmad=false`` is belt-and-braces: if it
ever mattered, some site would be missing its intrinsic.
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import assert_bit_exact, max_ulp

cp = pytest.importorskip("cupy")

KDIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
ORACLE = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "noahmp" / "oracle"

F = np.float32


def _f(h: str) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])


def _rows(leaf: str):
    with (ORACLE / f"noahmp-radiation-{leaf}.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _vec(row, base: str, n: int = 2):
    return [_f(row[f"{base}{i}"]) for i in range(1, n + 1)]


@pytest.fixture(scope="module")
def module():
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    src = (KDIR / "noahmp_radiation.cu").read_text()
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"))
    mod.compile()
    return mod


def _launch(mod, name, fin, fout_cols, iin=None):
    n = fin.shape[0]
    d_in = cp.asarray(fin.ravel(), dtype=cp.float32)
    d_out = cp.empty(n * fout_cols, dtype=cp.float32)
    args = [d_in]
    if iin is not None:
        args.append(cp.asarray(np.asarray(iin, dtype=np.int32).ravel()))
    args += [d_out, np.int32(n)]
    threads = 128
    blocks = (n + threads - 1) // threads
    mod.get_function(name)((blocks,), (threads,), tuple(args))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(d_out).reshape(n, fout_cols)


def _cols(rows, names):
    return np.array([[_f(r[c]) for c in names] for r in rows], dtype=np.float32)


# --------------------------------------------------------------------------
def test_cuda_snow_age(module):
    rows = _rows("snow_age")
    fin = _cols(rows, ["tau0", "grain_growth", "extra_growth", "dirt_soot",
                       "swemx", "dt", "tg", "sneqvo", "sneqv", "tauss_in"])
    got = _launch(module, "noahmp_rad_snow_age", fin, 2)
    want = _cols(rows, ["tauss_out", "fage"])
    assert_bit_exact(got, want, "cuda snow_age")
    assert max_ulp(got, want) == 0


def test_cuda_snowalb_class(module):
    rows = _rows("snowalb_class")
    fin = _cols(rows, ["swemx", "qsnow", "dt", "albold"])
    got = _launch(module, "noahmp_rad_snowalb_class", fin, 5)
    want = _cols(rows, ["alb", "albsnd1", "albsnd2", "albsni1", "albsni2"])
    assert_bit_exact(got, want, "cuda snowalb_class")
    assert max_ulp(got, want) == 0


def test_cuda_groundalb(module):
    rows = _rows("groundalb")
    fin = _cols(rows, ["albsat1", "albsat2", "albdry1", "albdry2",
                       "alblak1", "alblak2", "fsno", "smc1",
                       "albsnd1", "albsnd2", "albsni1", "albsni2",
                       "cosz", "tg"])
    ist = [int(r["ist"]) for r in rows]
    got = _launch(module, "noahmp_rad_groundalb", fin, 4, iin=ist)
    want = _cols(rows, ["albgrd1", "albgrd2", "albgri1", "albgri2"])
    assert_bit_exact(got, want, "cuda groundalb")
    assert max_ulp(got, want) == 0


_SURRAD_IN = (["mpe", "fsun", "fsha", "elai", "vai", "laisun", "laisha"]
              + [f"{b}{i}" for b in ("solad", "solai", "fabd", "fabi", "ftdd",
                                     "ftid", "ftii", "albgrd", "albgri",
                                     "albd", "albi", "frevd", "frevi",
                                     "fregd", "fregi") for i in (1, 2)])
_SURRAD_OUT = ["parsun", "parsha", "sav", "sag", "fsa", "fsr", "fsrv", "fsrg"]


def test_cuda_surrad(module):
    rows = _rows("surrad")
    fin = _cols(rows, _SURRAD_IN)
    assert fin.shape[1] == 37
    got = _launch(module, "noahmp_rad_surrad", fin, 8)
    want = _cols(rows, _SURRAD_OUT)
    assert_bit_exact(got, want, "cuda surrad")
    assert max_ulp(got, want) == 0


_TS_IN = ["xl", "omegas1", "omegas2", "betads", "betais",
          "cosz", "vai", "fwet", "t", "fveg",
          "albgrd1", "albgrd2", "albgri1", "albgri2",
          "rho1", "rho2", "tau1", "tau2",
          "fab_in1", "fab_in2", "fre_in1", "fre_in2",
          "ftd_in1", "ftd_in2", "fti_in1", "fti_in2",
          "gdir_in", "frev_in1", "frev_in2", "freg_in1", "freg_in2",
          "bgap_in", "wgap_in"]
_TS_OUT = ["fab1", "fab2", "fre1", "fre2", "ftd1", "ftd2", "fti1", "fti2",
           "gdir", "frev1", "frev2", "freg1", "freg2", "bgap", "wgap"]


def test_cuda_twostream(module):
    rows = _rows("twostream")
    fin = _cols(rows, _TS_IN)
    assert fin.shape[1] == 33
    ibic = [[int(r["ib"]), int(r["ic"])] for r in rows]
    got = _launch(module, "noahmp_rad_twostream", fin, 15, iin=ibic)
    want = _cols(rows, _TS_OUT)
    assert_bit_exact(got, want, "cuda twostream")
    assert max_ulp(got, want) == 0


_ALB_IN = (["tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx",
            "albsat1", "albsat2", "albdry1", "albdry2", "alblak1", "alblak2",
            "rhol1", "rhol2", "rhos1", "rhos2", "taul1", "taul2",
            "taus1", "taus2", "xl", "omegas1", "omegas2", "betads", "betais",
            "dt", "cosz", "elai", "esai", "tg", "tv", "snowh", "fsno", "fwet",
            "sneqvo", "sneqv", "qsnow", "fveg",
            "smc1", "smc2", "smc3", "smc4", "albold_in", "tauss_in", "fage_in",
            "frevd_in1", "frevd_in2", "frevi_in1", "frevi_in2",
            "fregd_in1", "fregd_in2", "fregi_in1", "fregi_in2"])
_ALB_OUT = (["fage", "albold", "tauss", "fsun", "bgap", "wgap"]
            + [f"{b}{i}" for b in ("albgrd", "albgri", "albd", "albi",
                                   "fabd", "fabi", "ftdd", "ftid", "ftii",
                                   "frevd", "frevi", "fregd", "fregi",
                                   "albsnd", "albsni") for i in (1, 2)])


def test_cuda_albedo(module):
    rows = _rows("albedo")
    fin = _cols(rows, _ALB_IN)
    assert fin.shape[1] == 52
    ist = [int(r["ist"]) for r in rows]
    got = _launch(module, "noahmp_rad_albedo", fin, 36, iin=ist)
    want = _cols(rows, _ALB_OUT)
    assert_bit_exact(got, want, "cuda albedo")
    assert max_ulp(got, want) == 0


def test_cuda_matches_cpu_transcription(module):
    """CUDA and the CPU transcription agree bitwise, not merely both-vs-oracle.

    A shared misreading of the Fortran would still pass the oracle gates only
    if both halves made the *same* mistake, which the oracle already rules
    out; this asserts the two ports have not diverged on anything the oracle
    happens not to observe.

    This used to skip on the GPU host: the transcription this lane was written
    against called :func:`math.fma`, which is CPython 3.13+, and the rented
    box runs 3.12, so the one comparison that needs *both* halves on the same
    machine was the one that could not run there.  The merged
    ``gpuwm.core.noahmp_libm`` contracts nothing and needs no ``math.fma``, so
    the leg now runs where the GPU is.
    """
    from gpuwm.core.noahmp_radiation import twostream

    rows = _rows("twostream")
    fin = _cols(rows, _TS_IN)
    ibic = [[int(r["ib"]), int(r["ic"])] for r in rows]
    gpu = _launch(module, "noahmp_rad_twostream", fin, 15, iin=ibic)

    cpu = []
    for r in rows:
        fab, fre, ftd, fti, gdir, frev, freg, bgap, wgap = twostream(
            xl=_f(r["xl"]), omegas=_vec(r, "omegas"),
            betads=_f(r["betads"]), betais=_f(r["betais"]),
            ib=int(r["ib"]), ic=int(r["ic"]), cosz=_f(r["cosz"]),
            vai=_f(r["vai"]), fwet=_f(r["fwet"]), t=_f(r["t"]),
            albgrd=_vec(r, "albgrd"), albgri=_vec(r, "albgri"),
            rho=_vec(r, "rho"), tau=_vec(r, "tau"), fveg=_f(r["fveg"]),
            fab=_vec(r, "fab_in"), fre=_vec(r, "fre_in"),
            ftd=_vec(r, "ftd_in"), fti=_vec(r, "fti_in"),
            gdir=_f(r["gdir_in"]),
            frev=_vec(r, "frev_in"), freg=_vec(r, "freg_in"),
            bgap=_f(r["bgap_in"]), wgap=_f(r["wgap_in"]))
        cpu.append([fab[0], fab[1], fre[0], fre[1], ftd[0], ftd[1],
                    fti[0], fti[1], gdir, frev[0], frev[1], freg[0], freg[1],
                    bgap, wgap])
    assert_bit_exact(gpu, np.array(cpu, dtype=np.float32), "cuda vs cpu")


# ==========================================================================
# The shipped compilation, RADIATION as one call, and the powf domain guard
# ==========================================================================
# Everything above compiles the kernel with --fmad=false.  The runtime does
# not: gpuwm.core.kernels.get_kernel uses -std=c++17 and nothing else, and this
# project's standing rule is that -fmad=false is a diagnostic and never a
# shipped fix.  So "every arithmetic site already carries its own rounding
# intrinsic" has to be a measurement of the *shipped* translation unit rather
# than a comment about the source.

_LEAF_ORACLES = (
    ("snow_age", "noahmp_rad_snow_age",
     ["tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx", "dt",
      "tg", "sneqvo", "sneqv", "tauss_in"],
     ["tauss_out", "fage"], None),
    ("snowalb_class", "noahmp_rad_snowalb_class",
     ["swemx", "qsnow", "dt", "albold"],
     ["alb", "albsnd1", "albsnd2", "albsni1", "albsni2"], None),
    ("groundalb", "noahmp_rad_groundalb",
     ["albsat1", "albsat2", "albdry1", "albdry2", "alblak1", "alblak2",
      "fsno", "smc1", "albsnd1", "albsnd2", "albsni1", "albsni2", "cosz",
      "tg"],
     ["albgrd1", "albgrd2", "albgri1", "albgri2"], "ist"),
    ("surrad", "noahmp_rad_surrad", _SURRAD_IN, _SURRAD_OUT, None),
    ("twostream", "noahmp_rad_twostream", _TS_IN, _TS_OUT, "ibic"),
    ("albedo", "noahmp_rad_albedo", _ALB_IN, _ALB_OUT, "ist"),
)


@pytest.fixture(scope="module")
def shipped_module():
    """The exact module the runtime loads: preamble + source, no extra flags."""
    from gpuwm.core.kernels import load_module

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    return load_module("noahmp_radiation")


@pytest.mark.parametrize("leaf,kernel,cols_in,cols_out,ints", _LEAF_ORACLES)
def test_the_shipped_compilation_needs_no_fmad_false(
        shipped_module, leaf, kernel, cols_in, cols_out, ints):
    """Each radiation leaf is max_ulp 0 without --fmad=false, or it is not.

    If this ever fails the fix is the missing rounding intrinsic at the
    contracted site, found by diffing the two compilations' PTX, and not the
    flag.
    """
    rows = _rows(leaf)
    fin = _cols(rows, cols_in)
    if ints == "ist":
        iin = [int(r["ist"]) for r in rows]
    elif ints == "ibic":
        iin = [[int(r["ib"]), int(r["ic"])] for r in rows]
    else:
        iin = None
    got = _launch(shipped_module, kernel, fin, len(cols_out), iin=iin)
    want = _cols(rows, cols_out)
    assert max_ulp(got, want) == 0, leaf
    assert_bit_exact(got, want, f"shipped {leaf}")


def test_the_radiation_composition_is_albedo_then_surrad(shipped_module):
    """d_radiation composes the two oracle-matched leaves and nothing else.

    Driven from the ALBEDO fixture rows with SOLAD/SOLAI attached, the composed
    kernel must reproduce (a) ALBEDO's own outputs bit for bit out of its new
    output slots, and (b) SURRAD evaluated on the host from those same ALBEDO
    outputs through :2787-2790's five assignments.  (b) is what gates the slot
    wiring between the two, which is the only new code in the composition.
    """
    from gpuwm.core.noahmp_energy import _RADIATION_MPE
    from gpuwm.core.noahmp_libm import f32
    from gpuwm.core.noahmp_radiation import surrad
    from gpuwm.core.noahmp_radiation_gpu import N_INPUT, N_OUTPUT

    rows = _rows("albedo")
    alb_in = _cols(rows, _ALB_IN)
    ist = [int(r["ist"]) for r in rows]
    n = alb_in.shape[0]

    # Physically plausible, distinguishable direct and diffuse forcing: the two
    # bands and the two streams must all carry different numbers, or a
    # transposed slot would be invisible.
    solad = np.tile(np.array([[310.0, 250.0]], np.float32), (n, 1))
    solai = np.tile(np.array([[95.0, 60.0]], np.float32), (n, 1))
    fin = np.concatenate([alb_in, solad, solai], axis=1).astype(np.float32)
    assert fin.shape[1] == N_INPUT

    got = _launch(shipped_module, "noahmp_radiation_batch", fin, N_OUTPUT,
                  iin=ist)
    alb_out = _launch(shipped_module, "noahmp_rad_albedo", alb_in, 36, iin=ist)

    # (a) ALBEDO's carried outputs survive the composition untouched.
    for slot, alb_slot, name in ((0, 3, "fsun"), (11, 32, "albsnd1"),
                                 (12, 33, "albsnd2"), (13, 34, "albsni1"),
                                 (14, 35, "albsni2"), (15, 4, "bgap"),
                                 (16, 5, "wgap"), (17, 1, "albold"),
                                 (18, 2, "tauss")):
        assert_bit_exact(got[:, slot], alb_out[:, alb_slot], name)

    # (b) the five assignments and SURRAD, evaluated on the host.
    want = []
    for k, row in enumerate(rows):
        elai = _f(row["elai"])
        esai = _f(row["esai"])
        fsun = np.float32(alb_out[k, 3])
        fsha = f32(f32(1.0) - fsun)
        laisun = f32(elai * fsun)
        laisha = f32(elai * fsha)
        vai = f32(elai + esai)

        def pair(base, _row=alb_out[k]):
            return (np.float32(_row[base]), np.float32(_row[base + 1]))

        parsun, parsha, sav, sag, fsa, fsr, fsrv, fsrg = surrad(
            _RADIATION_MPE, fsun, fsha, elai, vai, laisun, laisha,
            solad[k], solai[k], pair(14), pair(16),
            pair(18), pair(20), pair(22),
            pair(6), pair(8), pair(10), pair(12),
            pair(24), pair(26), pair(28), pair(30))
        want.append([fsun, laisun, laisha, parsun, parsha, sav, sag, fsa,
                     fsr, fsrv, fsrg])
    assert_bit_exact(got[:, 0:11], np.array(want, dtype=np.float32),
                     "d_radiation vs host ALBEDO+SURRAD")


def _probe_libm(module, x, y):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    n = x.size
    out = cp.empty(3 * n, dtype=cp.float32)
    threads = 128
    module.get_function("noahmp_rad_libm_probe")(
        ((n + threads - 1) // threads,), (threads,),
        (cp.asarray(x), cp.asarray(y), out, np.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(out).reshape(n, 3)


#: The bases outside [FLT_MIN, inf) that this file's narrow glibc_powf refuses.
#: -ftz=true is appended by CuPy unconditionally, so the subnormals are here
#: for the same reason they are in every other sweep in this project: three
#: real bugs so far have come from a flushed subnormal.
_GUARDED_BASES = np.array([
    0.0, -0.0, 1.4e-45, -1.4e-45, 5.877472e-39, 1.1754942e-38,
    -1.0, -0.5, -2.0, np.inf, -np.inf, np.nan,
], dtype=np.float32)


def test_the_powf_domain_guard_fires(shipped_module):
    """The guard is observed refusing, and the divergence is recorded.

    A guard nothing has ever driven proves nothing.  This drives it, and also
    records what the authoritative host transcription answers instead, so the
    narrowing is a measured difference rather than an assertion that it cannot
    matter.
    """
    from gpuwm.core.noahmp_libm import powf as host_powf

    y = np.full(_GUARDED_BASES.size, np.float32(0.17), dtype=np.float32)
    got = _probe_libm(shipped_module, _GUARDED_BASES, y)[:, 2]
    assert np.isnan(got).all(), (
        "the narrow powf accepted a base outside [FLT_MIN, inf): "
        f"{_GUARDED_BASES[~np.isnan(got)]}")

    defined = [float(base) for base in _GUARDED_BASES
               if not np.isnan(np.float32(
                   host_powf(np.float32(base), np.float32(0.17))))]
    assert defined, (
        "the host libm answers NaN for every guarded base too, so the guard "
        "narrows nothing and this test is not measuring what it claims")

    # y == 0 is the other half of the same guard.
    zero_exponent = _probe_libm(
        shipped_module, np.array([0.5, 2.0, 1.0], np.float32),
        np.zeros(3, np.float32))[:, 2]
    assert np.isnan(zero_exponent).all(), zero_exponent


def test_radiations_only_powf_site_cannot_reach_the_guard(shipped_module):
    """RADIATION's answer does not depend on the guard, and that is measured.

    glibc_powf appears once on this file's live path -- GROUNDALB's
    MAX(0.01, COSZ) ** 0.17 -- and WRF's own MAX is what holds the base at or
    above 0.01.  Sweeping COSZ across the guarded neighbourhood, including both
    zeros, the subnormals, the negatives and NaN, must therefore produce no NaN
    at all.
    """
    rows = _rows("groundalb")
    template = _cols(rows, ["albsat1", "albsat2", "albdry1", "albdry2",
                            "alblak1", "alblak2", "fsno", "smc1",
                            "albsnd1", "albsnd2", "albsni1", "albsni2",
                            "cosz", "tg"])[:1]
    ist = int(rows[0]["ist"])

    sweep = np.concatenate([
        _GUARDED_BASES,
        np.array([0.00999999, 0.01, 0.0100001, 1.0], np.float32)])
    fin = np.repeat(template, sweep.size, axis=0)
    fin[:, 12] = sweep
    got = _launch(shipped_module, "noahmp_rad_groundalb", fin, 4,
                  iin=[ist] * sweep.size)
    bad = sweep[np.isnan(got).any(axis=1)]
    assert bad.size == 0, (
        f"GROUNDALB produced NaN for COSZ = {bad}; the powf domain guard is "
        "reachable on the live path after all and has to be settled, not noted")


def test_the_device_powf_matches_the_host_libm_on_the_live_domain(
        shipped_module):
    """Inside the domain the guard admits, the two transcriptions agree.

    This is the half of the nine-copy question RADIATION actually depends on:
    not whether the copies agree at zero, but whether they agree where
    GROUNDALB evaluates them.
    """
    from gpuwm.core.noahmp_libm import powf as host_powf

    base = np.concatenate([
        np.array([0.01, 0.010000001, 0.5, 1.0, 0.9999999], np.float32),
        np.linspace(0.01, 1.0, 4093, dtype=np.float32),
    ])
    y = np.full(base.size, np.float32(0.17), dtype=np.float32)
    got = _probe_libm(shipped_module, base, y)[:, 2]
    want = np.array([host_powf(b, np.float32(0.17)) for b in base],
                    dtype=np.float32)
    assert max_ulp(got, want) == 0
    assert_bit_exact(got, want, "radiation glibc_powf vs noahmp_libm.powf")
