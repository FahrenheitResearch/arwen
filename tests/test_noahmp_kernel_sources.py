"""Every Noah-MP kernel compiles, and the per-thread local frames are priced.

Two facts about the Noah-MP kernel directory are not visible from the directory
listing and have already cost another lane a measurement:

* ``noahmp_driver.cu``, ``noahmp_energy.cu`` and ``noahmp_thermal.cu`` are
  fragments.  Each borrows the single device transcription of glibc's powf /
  expf / logf that lives in ``noahmp_leaves.cu``, so each is compiled after it.
  A sweep that compiles every ``.cu`` alone reports them broken, and anything
  it then computes about ``sf_surface_physics = 4`` is computed from kernels
  that never compiled.
* ``noahmp_fluxprep.cu`` is the opposite: it carries its own libm tables, so
  prepending ``noahmp_leaves.cu`` to *it* is a duplicate definition.

So neither "compile each file alone" nor "compile everything after the libm" is
right, and :data:`gpuwm.core.noahmp_kernel_sources.NOAHMP_TRANSLATION_UNITS` is
the table that says which is which.  This file pins it both ways: every listed
unit compiles, and every composed unit still genuinely fails alone -- without
the second half the table could quietly become decorative.

NVRTC is a host compiler.  Everything here runs without a CUDA device.
"""

from __future__ import annotations

import re

import pytest

from gpuwm.core.noahmp_kernel_sources import (COMPOSED_UNITS,
                                              NOAHMP_TRANSLATION_UNITS,
                                              kernel_files,
                                              translation_unit_source)

cp = pytest.importorskip("cupy")

#: Blackwell.  The frames measured below are a property of the source and the
#: compiler, not of which card is plugged in, and pinning one arch keeps the
#: numbers comparable between runs.
ARCH = "120"


def _compile(source: str) -> bytes:
    from cupy.cuda.compiler import compile_using_nvrtc

    out = compile_using_nvrtc(source, options=("-std=c++17",), arch=ARCH)
    ptx = out[0] if isinstance(out, tuple) else out
    return ptx if isinstance(ptx, bytes) else str(ptx).encode()


def test_every_kernel_file_is_declared():
    assert set(kernel_files()) == set(NOAHMP_TRANSLATION_UNITS), (
        "a Noah-MP .cu appeared or disappeared without the translation-unit "
        "table moving with it")


@pytest.mark.parametrize("unit", sorted(NOAHMP_TRANSLATION_UNITS))
def test_the_declared_translation_unit_compiles(unit):
    _compile(translation_unit_source(unit))


@pytest.mark.parametrize("unit", COMPOSED_UNITS)
def test_a_composed_unit_still_fails_on_its_own(unit):
    """Without this, the table above could list a composition nobody needs.

    ``r_pow`` is the identifier that goes missing, and it goes missing because
    there is exactly one transcription of glibc's powf in this tree.  If this
    ever passes, either the file became self-contained -- in which case it has
    a second copy of a constant table and that is the real problem -- or the
    entry belongs in the single-part half of the table.
    """
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.noahmp_kernel_sources import KERNEL_DIR

    # The file by itself, exactly as a naive directory sweep would take it.
    lone = _preamble() + (KERNEL_DIR / f"{unit}.cu").read_text(encoding="ascii")
    with pytest.raises(Exception) as excinfo:
        _compile(lone)
    assert "r_pow" in str(excinfo.value), str(excinfo.value)[:400]


def test_prepending_the_libm_to_a_self_contained_unit_is_an_error():
    """The other half of why a blanket rule cannot work.

    ``noahmp_fluxprep.cu`` carries its own copy of the tables, so it is not a
    fragment and must not be composed.  Recording the failure here is what
    stops "just prepend noahmp_leaves.cu to everything" from being tried again.
    """
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.noahmp_kernel_sources import KERNEL_DIR, LIBM_UNIT

    assert NOAHMP_TRANSLATION_UNITS["noahmp_fluxprep"] == ("noahmp_fluxprep",)
    source = (_preamble()
              + (KERNEL_DIR / f"{LIBM_UNIT}.cu").read_text(encoding="ascii")
              + (KERNEL_DIR / "noahmp_fluxprep.cu").read_text(encoding="ascii"))
    with pytest.raises(Exception) as excinfo:
        _compile(source)
    assert "already been defined" in str(excinfo.value), \
        str(excinfo.value)[:400]


# ---------------------------------------------------------------------------
# per-thread local frames
# ---------------------------------------------------------------------------
# A kernel whose per-thread arrays are dimensioned on a compile-time maximum
# rather than the run's actual layer count makes the driver reserve a backing
# store for the device's whole resident-thread capacity, on first launch, for
# the process lifetime -- and it is invisible to the CuPy pool.  Another lane
# measured that law exactly on this card:
#
#     reservation_bytes = (max_local_size_bytes - 1024) * 1536 * 170
#
# so 24 KiB per thread is 5.7 GiB of reservation.  Noah-MP's arrays are all
# dimensioned NMP_NLAY = NSNOW + NSOIL = 7, a physical constant of the pinned
# identity rather than a compile-time ceiling, so the frames should be small.
# "Should be" is not a measurement, and this is the measurement.

#: The largest per-thread frame any Noah-MP kernel may declare, in bytes.
#: 4 KiB is roughly 1,000 floats -- far above anything a seven-layer column
#: needs and far below the 24 KiB that cost 5.7 GiB elsewhere in this project.
#: If a conversion ever needs more than this, the reservation it implies has to
#: be priced in the same commit, not discovered in a forecast.
MAX_LOCAL_FRAME_BYTES = 4096

#: Blackwell, measured by the lane that derived the law above.
_RESIDENT_THREADS = 1536 * 170

_ENTRY = re.compile(r'extern\s+"C"\s+__global__\s+(?:void\s+)?'
                    r'([A-Za-z_]\w*)\s*\(', re.S)
_ENTRY_SPLIT = re.compile(r'extern\s+"C"\s+__global__\s+void\s*\n?\s*'
                          r'([A-Za-z_]\w*)\s*\(')


def _entry_points(unit: str) -> list[str]:
    text = translation_unit_source(unit, preamble=False)
    names = set(_ENTRY.findall(text)) | set(_ENTRY_SPLIT.findall(text))
    names.discard("void")
    return sorted(names)


@pytest.mark.parametrize("unit", sorted(NOAHMP_TRANSLATION_UNITS))
def test_the_per_thread_local_frame_is_small(unit, capsys):
    """``local_size_bytes`` per kernel, read from the loaded module.

    This is the driver's own number, not a guess off the PTX: NVRTC leaves the
    decision to ptxas, so the only honest place to read it is the function
    attribute after the module is loaded.
    """
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")

    module = cp.RawModule(code=translation_unit_source(unit),
                          options=("-std=c++17",))
    module.compile()
    names = _entry_points(unit)
    assert names, f"{unit} declares no entry point"
    rows = []
    for name in names:
        frame = int(module.get_function(name).local_size_bytes)
        rows.append((frame, name))
    rows.sort(reverse=True)
    with capsys.disabled():
        print(f"\n{unit:22s} largest frame {rows[0][0]:6d} B "
              f"({rows[0][1]})  => "
              f"{max(rows[0][0] - 1024, 0) * _RESIDENT_THREADS / 2**30:6.2f} "
              f"GiB reserved on first launch")
    assert rows[0][0] <= MAX_LOCAL_FRAME_BYTES, (
        f"{unit}:{rows[0][1]} declares a {rows[0][0]}-byte per-thread frame, "
        f"which reserves "
        f"{(rows[0][0] - 1024) * _RESIDENT_THREADS / 2**30:.2f} GiB of "
        "non-pool device memory for the process lifetime on first launch")
