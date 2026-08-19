"""Which Noah-MP ``.cu`` files are translation units, and which are fragments.

Three of the Noah-MP kernel sources do not compile on their own::

    noahmp_driver.cu
    noahmp_energy.cu
    noahmp_thermal.cu      NVRTC: identifier "r_pow" is undefined

That is deliberate and it is the whole reason ``r_pow`` has that name.  glibc
2.39's ``powf``, ``expf`` and ``logf`` are transcribed **once** in this tree, in
``noahmp_leaves.cu``, because two copies of a 32-entry constant table can drift
and only one of them would be audited against ``glibc-libm-fp32.csv``.  The
three files above borrow that single copy, so each is compiled *after*
``noahmp_leaves.cu`` -- which is what :func:`gpuwm.core.noahmp_thermal_gpu
.thermal_source` and :func:`gpuwm.core.noahmp_energy_gpu.energy_source` already
do, and why those kernels compile and launch perfectly well on the real path.

The reverse is equally true and equally easy to get wrong: ``noahmp_fluxprep.cu``
carries its own libm tables, so prepending ``noahmp_leaves.cu`` to *it* is a
duplicate-definition error.  "Compile each file alone" and "compile everything
after the libm" are therefore both wrong, and neither one is discoverable from
the directory listing.

This module states the answer once, in a form a tool can read.  A whole-project
sweep -- a VRAM census, a static-analysis pass, a compile-warning gate -- should
build its source text from :func:`translation_unit_source` rather than from
``Path.read_text`` on each ``.cu``, or it will price ``sf_surface_physics = 4``
off three kernels that never compiled.

``tests/test_noahmp_kernel_sources.py`` pins every entry, and pins that the
three composed ones really do fail alone, so this table cannot quietly become
decorative.
"""

from __future__ import annotations

from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: The one device transcription of glibc's powf/expf/logf.  Its ``r_pow``,
#: ``r_exp`` and ``r_log`` are what the composed units below are missing.
LIBM_UNIT = "noahmp_leaves"

#: Translation unit name -> the ordered ``.cu`` stems it is compiled from.
#: The key is what a caller asks for; the value is the concatenation order.
NOAHMP_TRANSLATION_UNITS: dict[str, tuple[str, ...]] = {
    "noahmp_bareflux": ("noahmp_bareflux",),
    "noahmp_driver": (LIBM_UNIT, "noahmp_driver"),
    "noahmp_energy": (LIBM_UNIT, "noahmp_energy"),
    "noahmp_fluxprep": ("noahmp_fluxprep",),
    # NOAHMP_GLACIER borrows r_pow/r_exp/r_log (and the AD/SU/MU/DV
    # macros plus NMP_* constants) from the libm unit and carries only its
    # own glibc atanf copy, like noahmp_bareflux does.
    "noahmp_glacier": (LIBM_UNIT, "noahmp_glacier"),
    "noahmp_leaves": ("noahmp_leaves",),
    # The slab composition's elementwise transcendentals borrow r_pow/r_exp/
    # r_log from LIBM_UNIT and nmpe_tanhf from noahmp_energy, so it is the one
    # unit built from three parts.
    "noahmp_libm_slab": (LIBM_UNIT, "noahmp_energy", "noahmp_libm_slab"),
    "noahmp_radiation": ("noahmp_radiation",),
    "noahmp_sflx": ("noahmp_sflx",),
    "noahmp_snow": ("noahmp_snow",),
    "noahmp_soilwater": ("noahmp_soilwater",),
    "noahmp_thermal": (LIBM_UNIT, "noahmp_thermal"),
    "noahmp_vegeflux": ("noahmp_vegeflux",),
    "noahmp_vegprecip": ("noahmp_vegprecip",),
    "noahmp_water": ("noahmp_water",),
}

#: The units that are *fragments*: they need :data:`LIBM_UNIT` in front and
#: fail NVRTC without it.  Named separately because "this file cannot be
#: compiled alone" is the fact a sweep trips over, and a test asserts each of
#: these really does still fail alone rather than having quietly become
#: self-contained.
COMPOSED_UNITS = tuple(
    name for name, parts in NOAHMP_TRANSLATION_UNITS.items() if len(parts) > 1)


def translation_unit_source(name: str, *, preamble: bool = True) -> str:
    """The exact source text NVRTC should be handed for ``name``.

    ``preamble`` prepends :func:`gpuwm.core.kernels._preamble`, which is what
    ``load_module`` does; pass ``False`` only when composing this into a larger
    unit that already carries it.
    """
    try:
        parts = NOAHMP_TRANSLATION_UNITS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a Noah-MP translation unit; the units are "
            f"{sorted(NOAHMP_TRANSLATION_UNITS)}") from None
    text = "".join(
        (KERNEL_DIR / f"{part}.cu").read_text(encoding="ascii")
        for part in parts)
    if not preamble:
        return text
    from gpuwm.core.kernels import _preamble

    return _preamble() + text


def kernel_files() -> tuple[str, ...]:
    """Every ``noahmp*.cu`` stem present on disk, sorted."""
    return tuple(sorted(path.stem for path in KERNEL_DIR.glob("noahmp*.cu")))


__all__ = [
    "COMPOSED_UNITS",
    "KERNEL_DIR",
    "LIBM_UNIT",
    "NOAHMP_TRANSLATION_UNITS",
    "kernel_files",
    "translation_unit_source",
]
