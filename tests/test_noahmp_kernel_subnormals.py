"""Every Noah-MP device libm, swept across the band where its results are subnormal.

``__double2float_rn`` flushes a subnormal result to zero on this toolchain
(sm_120, CUDA 13.0), and ``--ftz=false`` does not change it -- cupy appends
``-ftz=true`` after it, and the last occurrence wins.  glibc 2.39's ``expf`` and
``powf`` do return subnormals, gfortran at ``-O0`` leaves MXCSR's FTZ and DAZ
clear, and :mod:`gpuwm.core.noahmp_libm` -- the CPython authority, verified
against the live glibc over 1,106,247,680 inputs -- returns them too.  So every
device transcription that ended in ``__double2float_rn`` disagreed with the
authority across the whole band where ``expf`` underflows into the subnormals,
returning ``+0.0`` for every argument in it.

No fixture in the tree reaches that band, which is why every leaf oracle stayed
green through the whole defect.  **A passing oracle is therefore not evidence
about this fix at all**; only a direct sweep is, and that is what this file is.
``tests/test_noahmp_slab_libm.py`` does the same job for ``noahmp_leaves.cu``
and ``noahmp_fluxprep.cu``; this file covers six of the seven others.  The
seventh, ``noahmp_vegeflux.cu``, still flushes -- it spells the conversion as a
plain ``(float)`` cast, which is the same instruction under another name -- and
:func:`test_vegeflux_still_flushes_the_band_and_is_not_this_fixs_site` measures
that rather than leaving it to be rediscovered.

Each sweep is paired with a **negative control that reconstructs the defect**:
the same source text with ``return nmp_d2f_rn(`` substituted back to ``return
__double2float_rn(``, compiled and swept, and required to *differ*.  Without it
"the band is bitwise" would be consistent with the band being empty, with the
probe not reaching the transcription, or with the compiled conversion having
quietly been fixed and the helper being dead weight.

TWO KERNELS SHIP NO PROBE THAT REACHES THEIR OWN LIBM, and that is recorded
here rather than papered over:

* ``noahmp_bareflux.cu``'s ``noahmp_bareflux_libm_probe`` evaluates
  ``glibc_powf(x, K_QUARTER)``.  A fourth root of any normal float is at least
  ``2**-31.5``, so that probe cannot produce a subnormal whatever it is fed.
* ``noahmp_vegprecip.cu`` exposes only ``noahmp_phenology`` and
  ``noahmp_precip_heat``; neither returns a raw ``vp_glibc_expf`` /
  ``vp_glibc_powf`` value.

For those two the sweep drives an entry point appended to the shipped source at
compile time (:data:`_ADDED_PROBES`).  That is still the file's own device code
-- the same text NVRTC is handed on the real path, plus a way in -- but it is
not a shipped probe, and :func:`test_two_kernels_ship_no_probe_that_reaches_the_band`
pins the distinction so nobody reads this file as claiming otherwise.
"""

from __future__ import annotations

import functools
import re

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.core import noahmp_libm as scalar
from gpuwm.core.noahmp_kernel_sources import KERNEL_DIR, translation_unit_source

#: The largest binary32 that is not normal.  Everything strictly below this in
#: magnitude, and not zero, is the range the compiled conversion flushes.
SMALLEST_NORMAL = np.float32(1.1754944e-38)

#: ``expf`` returns a binary32 subnormal for exactly this closed interval of
#: FP32 arguments, and ``+0.0`` below it.  Same constant as
#: ``tests/test_noahmp_slab_libm.py``; the two must not drift.
SUBNORMAL_EXPF_BAND = (-103.61632918473205, -87.33654475125263)

#: ``powf`` has no single argument band -- the result depends on both operands
#: -- so two bases are swept instead, one a power of two (``y*log2(x)`` is the
#: exponent itself, exercising every subnormal binade end to end) and one not
#: (``log2_inline``'s polynomial is on the path).  The exponent ranges stop
#: short of ``y*log2(x) == -150``, where glibc's own underflow shim takes over
#: and the question stops being about the conversion.
POWF_BASES = ((2.0, -149.5, -126.5), (3.0, -94.4, -79.6))


def _float32_ladder(lo: float, hi: float) -> np.ndarray:
    """Every FP32 value between ``lo`` and ``hi`` inclusive, not a sample.

    Both ends are negative here, and for negative floats a larger magnitude is
    a larger *unsigned* bit pattern; walking the int32 view between the two
    endpoints therefore enumerates the ladder exactly once.
    """
    lo_bits = np.float32(lo).view(np.int32)
    hi_bits = np.float32(hi).view(np.int32)
    bits = np.arange(min(lo_bits, hi_bits), max(lo_bits, hi_bits) + 1,
                     dtype=np.int32)
    return bits.view(np.float32)


@functools.lru_cache(maxsize=1)
def expf_band() -> tuple[np.ndarray, np.ndarray]:
    """``(argument, glibc answer)`` over the whole subnormal ``expf`` band."""
    x = _float32_ladder(*SUBNORMAL_EXPF_BAND)
    want = np.array([scalar.expf(v) for v in x], dtype=np.float32)
    assert x.size > 2_000_000, f"the band collapsed to {x.size} arguments"
    assert (want != 0.0).all() and (np.abs(want) < SMALLEST_NORMAL).all(), (
        "this band is supposed to be exactly the subnormal results; it is not")
    return x, want


@functools.lru_cache(maxsize=1)
def powf_band() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(base, exponent, glibc answer)`` over the subnormal ``powf`` band."""
    bases: list[np.ndarray] = []
    exponents: list[np.ndarray] = []
    for base, lo, hi in POWF_BASES:
        ladder = _float32_ladder(lo, hi)
        exponents.append(ladder)
        bases.append(np.full(ladder.size, base, dtype=np.float32))
    base = np.concatenate(bases)
    exponent = np.concatenate(exponents)
    want = np.array([scalar.powf(b, e) for b, e in zip(base, exponent)],
                    dtype=np.float32)
    assert base.size > 2_000_000, f"the band collapsed to {base.size} pairs"
    assert (want != 0.0).all() and (np.abs(want) < SMALLEST_NORMAL).all(), (
        "this band is supposed to be exactly the subnormal results; it is not")
    return base, exponent, want


# ---------------------------------------------------------------------------
# how each kernel's own transcription is reached from the host
# ---------------------------------------------------------------------------

#: Entry points appended to the shipped source for the two files that expose
#: none.  They add a way in; they change no arithmetic.
_ADDED_PROBES: dict[str, str] = {
    "noahmp_bareflux": """
extern "C" __global__ void nmp_subnormal_powf_probe(const float *x,
                                                    const float *e,
                                                    float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = glibc_powf(x[i], e[i]);
}
""",
    "noahmp_vegprecip": """
extern "C" __global__ void nmp_subnormal_expf_probe(const float *x,
                                                    float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = vp_glibc_expf(x[i]);
}

extern "C" __global__ void nmp_subnormal_powf_probe(const float *x,
                                                    const float *e,
                                                    float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = vp_glibc_powf(x[i], e[i]);
}
""",
}

#: ``unit -> (kernel, output stride, output slot)`` for the ``expf`` sweep.
#: ``noahmp_bareflux.cu`` has no ``expf`` at all -- SFCDIF1 needs ``powf``,
#: ``logf`` and ``atanf`` and nothing else -- so it is absent by fact, not by
#: omission.
EXPF_PROBES: dict[str, tuple[str, int, int]] = {
    "noahmp_radiation": ("noahmp_rad_libm_probe", 3, 0),
    "noahmp_snow": ("noahmp_snow_expf_probe", 1, 0),
    "noahmp_soilwater": ("k_expf_sweep", 1, 0),
    "noahmp_vegprecip": ("nmp_subnormal_expf_probe", 1, 0),
    "noahmp_water": ("k_water_expf", 1, 0),
}

#: ``unit -> (kernel, output stride, output slot)`` for the ``powf`` sweep.
#: ``noahmp_snow.cu`` transcribes ``expf`` only.
POWF_PROBES: dict[str, tuple[str, int, int]] = {
    "noahmp_bareflux": ("nmp_subnormal_powf_probe", 1, 0),
    "noahmp_radiation": ("noahmp_rad_libm_probe", 3, 2),
    "noahmp_soilwater": ("k_powf_sweep", 1, 0),
    "noahmp_vegprecip": ("nmp_subnormal_powf_probe", 1, 0),
    "noahmp_water": ("k_water_powf", 1, 0),
}

#: The units whose sweep goes through an appended entry point rather than one
#: the file ships.  Named, because "verified" and "verified through a probe we
#: had to add" are different claims.
SWEPT_THROUGH_ADDED_PROBE = frozenset(_ADDED_PROBES)


@functools.lru_cache(maxsize=None)
def _module(unit: str, flushing: bool):
    """Compile ``unit``; ``flushing`` reconstructs the pre-fix source.

    The substitution touches call sites only -- ``nmp_d2f_rn``'s own definition
    ends in ``return __double2float_rn(y);`` already and does not match -- so
    the mutated unit is byte-for-byte the file as it stood before the fix,
    plus any appended probe.
    """
    import cupy as cp

    source = translation_unit_source(unit) + _ADDED_PROBES.get(unit, "")
    if flushing:
        mutated = source.replace("return nmp_d2f_rn(",
                                 "return __double2float_rn(")
        assert mutated != source, (
            f"{unit} has no nmp_d2f_rn call site to undo, so this control "
            f"would be comparing the fixed kernel against itself")
        source = mutated
    module = cp.RawModule(code=source, options=("-std=c++17",))
    module.compile()
    return module


def _run(unit: str, kernel: str, stride: int, slot: int, args, count: int,
         *, flushing: bool) -> np.ndarray:
    import cupy as cp

    out = cp.empty(stride * count, dtype=cp.float32)
    threads = 256
    blocks = (count + threads - 1) // threads
    function = _module(unit, flushing).get_function(kernel)
    function((blocks,), (threads,), (*args, out, np.int32(count)))
    return cp.asnumpy(out)[slot::stride]


def _device_expf(unit: str, x: np.ndarray, *, flushing: bool) -> np.ndarray:
    import cupy as cp

    kernel, stride, slot = EXPF_PROBES[unit]
    device_x = cp.asarray(x)
    # noahmp_rad_libm_probe takes a second operand for its powf slot; the expf
    # slot ignores it.
    args = (device_x, device_x) if stride == 3 else (device_x,)
    return _run(unit, kernel, stride, slot, args, x.size, flushing=flushing)


def _device_powf(unit: str, base: np.ndarray, exponent: np.ndarray,
                 *, flushing: bool) -> np.ndarray:
    import cupy as cp

    kernel, stride, slot = POWF_PROBES[unit]
    args = (cp.asarray(base), cp.asarray(exponent))
    return _run(unit, kernel, stride, slot, args, base.size, flushing=flushing)


def _differing(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.argwhere(got.view(np.int32) != want.view(np.int32)).ravel()


# ---------------------------------------------------------------------------
# the sweeps
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("unit", sorted(EXPF_PROBES))
def test_expf_is_bitwise_across_the_whole_subnormal_band(unit):
    """Every FP32 argument whose ``expf`` is subnormal, not a sample of them."""
    x, want = expf_band()
    got = _device_expf(unit, x, flushing=False)
    bad = _differing(got, want)
    assert bad.size == 0, (
        f"{unit}: {bad.size}/{x.size} subnormal expf results differ; first is "
        f"expf({x[bad[0]]!r}): device {got[bad[0]]!r} vs glibc {want[bad[0]]!r}")


@requires_gpu
@pytest.mark.parametrize("unit", sorted(POWF_PROBES))
def test_powf_is_bitwise_across_the_subnormal_band(unit):
    """Both bases, every exponent whose ``powf`` lands in the subnormals."""
    base, exponent, want = powf_band()
    got = _device_powf(unit, base, exponent, flushing=False)
    bad = _differing(got, want)
    assert bad.size == 0, (
        f"{unit}: {bad.size}/{base.size} subnormal powf results differ; first "
        f"is powf({base[bad[0]]!r}, {exponent[bad[0]]!r}): device "
        f"{got[bad[0]]!r} vs glibc {want[bad[0]]!r}")


# ---------------------------------------------------------------------------
# the negative controls: the same sweep against the defect, reconstructed
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("unit", sorted(EXPF_PROBES))
def test_the_pre_fix_expf_flushed_the_entire_band_to_zero(unit):
    """What the sweep above would have measured before ``nmp_d2f_rn``.

    Every argument, not merely some: the compiled conversion does not round
    towards the subnormals at all, it flushes.  If this ever passes with a
    small count, the toolchain has changed and the claim needs re-deriving; if
    it fails outright, ``__double2float_rn`` has been fixed and the helper can
    be retired.
    """
    x, want = expf_band()
    got = _device_expf(unit, x, flushing=True)
    bad = _differing(got, want)
    assert bad.size == x.size, (
        f"{unit}: the pre-fix form differed on {bad.size}/{x.size} arguments, "
        f"not all of them")
    assert (got == 0.0).all(), "the pre-fix form returned something other than zero"


@requires_gpu
@pytest.mark.parametrize("unit", sorted(POWF_PROBES))
def test_the_pre_fix_powf_flushed_the_entire_band_to_zero(unit):
    """The ``powf`` half of the control above."""
    base, exponent, want = powf_band()
    got = _device_powf(unit, base, exponent, flushing=True)
    bad = _differing(got, want)
    assert bad.size == base.size, (
        f"{unit}: the pre-fix form differed on {bad.size}/{base.size} pairs, "
        f"not all of them")
    assert (got == 0.0).all(), "the pre-fix form returned something other than zero"


# ---------------------------------------------------------------------------
# what this file does NOT cover through a shipped probe
# ---------------------------------------------------------------------------

def test_two_kernels_ship_no_probe_that_reaches_the_band():
    """The finding, pinned so it is not mistaken for full shipped coverage.

    If either kernel later grows a real probe over its own libm, this test
    fails and the entry should come out of :data:`_ADDED_PROBES` -- which is
    the outcome to want.
    """
    bareflux = (KERNEL_DIR / "noahmp_bareflux.cu").read_text(encoding="ascii")
    assert "glibc_powf(x[i], K_QUARTER)" in bareflux, (
        "noahmp_bareflux_libm_probe no longer pins the exponent at 1/4")
    # A fourth root of the smallest normal is 2**-31.5, so no argument to that
    # probe can produce a subnormal.
    assert scalar.powf(np.float32(np.finfo(np.float32).tiny),
                       np.float32(0.25)) > SMALLEST_NORMAL

    vegprecip = (KERNEL_DIR / "noahmp_vegprecip.cu").read_text(encoding="ascii")
    shipped = set(re.findall(r'extern "C" __global__ void (\w+)', vegprecip))
    assert shipped == {"noahmp_phenology", "noahmp_precip_heat"}, shipped

    assert SWEPT_THROUGH_ADDED_PROBE == {"noahmp_bareflux", "noahmp_vegprecip"}


def _carriers() -> set[str]:
    """Every Noah-MP kernel with its own 32-entry ``exp2f`` table.

    That table is what makes a file a separate transcription rather than a
    consumer of one, and it is the criterion
    ``tests/test_noahmp_kernel_symbol_duplication.py`` is really about.
    """
    return {
        path.stem
        for path in sorted(KERNEL_DIR.glob("noahmp*.cu"))
        if re.search(r"exp2f_tab\s*\[", path.read_text(encoding="ascii"),
                     re.I)
    }


def test_every_kernel_carrying_its_own_transcription_is_accounted_for():
    """No file with its own libm may sit outside this file without a reason.

    ``noahmp_leaves.cu`` and ``noahmp_fluxprep.cu`` are swept by
    ``tests/test_noahmp_slab_libm.py``; ``noahmp_vegeflux.cu`` is the subject
    of the test below.  Nine carriers, and every one of them named.
    """
    carriers = _carriers()
    assert len(carriers) == 9, sorted(carriers)
    swept_here = set(EXPF_PROBES) | set(POWF_PROBES)
    swept_elsewhere = {"noahmp_leaves", "noahmp_fluxprep"}
    still_flushing = {"noahmp_vegeflux"}
    unaccounted = carriers - swept_here - swept_elsewhere - still_flushing
    assert unaccounted == set(), sorted(unaccounted)


@requires_gpu
def test_vegeflux_still_flushes_the_band_and_is_not_this_fixs_site():
    """A finding, recorded rather than filed away.

    ``noahmp_vegeflux.cu`` converts its ``expf``/``powf`` result with a plain
    ``(float)`` cast (``:117``, ``:153``, ``:229``) rather than
    ``__double2float_rn``.  Different spelling, same instruction: it flushes
    the whole subnormal band to zero exactly as the other eight did.  It is
    also the tree's oldest libm generation -- literal float constants where
    the others use ``__constant__`` tables, and an overflow threshold one ULP
    from theirs -- so the right repair there is retirement onto ``r_exp`` /
    ``r_pow``, not a tenth copy of ``nmp_d2f_rn``.  That is another lane's
    file and another lane's call.

    Measured, not asserted: when VEGE_FLUX stops flushing this fails, and the
    exemption in the test above has to be re-argued rather than inherited.
    """
    import cupy as cp

    source = translation_unit_source("noahmp_vegeflux") + """
extern "C" __global__ void nmp_subnormal_expf_probe(const float *x,
                                                    float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = glibc_expf(x[i]);
}
"""
    module = cp.RawModule(code=source, options=("-std=c++17",))
    module.compile()
    x = np.array([-88.0, -90.0, -95.0, -100.0, -103.0], dtype=np.float32)
    want = np.array([scalar.expf(v) for v in x], dtype=np.float32)
    assert (want != 0.0).all() and (np.abs(want) < SMALLEST_NORMAL).all()
    out = cp.empty(x.size, dtype=cp.float32)
    module.get_function("nmp_subnormal_expf_probe")(
        (1,), (32,), (cp.asarray(x), out, np.int32(x.size)))
    got = cp.asnumpy(out)
    assert (got == 0.0).all(), (
        "noahmp_vegeflux.cu no longer flushes the subnormal band; the "
        "exemption in test_every_kernel_carrying_its_own_transcription_is_"
        f"accounted_for is stale.  Got {got!r}")
