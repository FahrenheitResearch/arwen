"""Authoritative Noah-MP runtime translation units and compilation identities.

Five source files are fragments rather than standalone translation units:
``noahmp_driver``, ``noahmp_energy``, ``noahmp_thermal``, ``noahmp_glacier``
and ``noahmp_libm_slab``. They borrow the r_pow/r_exp/r_log helpers from
``noahmp_leaves``; the slab also borrows helpers from ``noahmp_energy``.
Some standalone units carry independent libm transcriptions. In particular,
prepending leaves to fluxprep would introduce duplicate definitions. Nothing
here changes CUDA math or consolidates those transcriptions.

Runtime factories and memory measurement share :func:`runtime_unit`, which
binds ordered parts, the common preamble, exact compiler options and all global
exports (including inherited helpers). VEGE_FLUX deliberately uses C++14 and no
common preamble; the other production units use C++17 with the preamble.

:func:`translation_unit_source` remains the generic compilation-check helper
for existing tools, not a memory-measurement identity. Its optional preamble
switch must not erase the distinct runtime VEGE_FLUX contract.

The CPU provenance tests check factory inputs and source drift. The existing
GPU compile tests check that the five composed fragments fail alone.  The
per-thread local frames these units compile to are recorded per compile
platform in :mod:`gpuwm.core.kernel_frame_recordings`
(``NOAHMP_COMPOSED_FRAME_RECORDINGS``) and priced by
:mod:`gpuwm.core.noahmp_frame_provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: The shared helper transcription for composed units. Its ``r_pow``,
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
    prefix = ""
    if preamble:
        from gpuwm.core.kernels import _preamble
        prefix = _preamble()
    return _assemble(_component_texts(parts), prefix)


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


# Runtime identity is deliberately separate from a standalone .cu stem.  In
# particular, load_module("noahmp_driver") must STILL fail: that is a fragment,
# not the unit the driver factory compiles.  The old standalone census and its
# measurements retain their meanings.
DEFAULT_OPTIONS = ("-std=c++17",)


def pricing_key(name: str) -> str:
    """The memory-inventory key for the runtime unit (never a raw fragment)."""
    parts = NOAHMP_TRANSLATION_UNITS[name]
    if len(parts) > 1:
        return name + "_composed"
    if name == "noahmp_vegeflux":
        # Its runtime uses C++14 WITHOUT the common preamble.  A generic
        # load_module(name) C++17 measurement does not describe that image.
        return name + "_runtime"
    return name


NOAHMP_PRICING_MODULES = tuple(pricing_key(name)
                              for name in NOAHMP_TRANSLATION_UNITS)


@dataclass(frozen=True)
class RuntimeUnit:
    """A snapshot of the *actual* RawModule inputs, constructed without CuPy.

    The source SHA hashes the exact UTF-8 string passed to RawModule, including
    the common header/defines when present.  The identity additionally binds
    ordered components, compiler options and every exported kernel, including
    helper exports inherited by a composed unit.  No byte measurements live
    here: ``identity()["identity_sha256"]`` is what a frame recording row
    binds itself to, so a row read from a different source or option tuple
    stops matching and the platform is refused until re-read.
    """

    name: str
    source: str
    parts: tuple[tuple[str, str], ...]
    preamble_sha256: str
    options: tuple[str, ...]
    exports: tuple[str, ...]

    @property
    def key(self) -> str:
        return pricing_key(self.name)

    def identity(self) -> dict:
        payload = {
            "name": self.name,
            "pricing_key": self.key,
            "parts": [{"file": name + ".cu", "sha256": digest}
                      for name, digest in self.parts],
            "preamble_sha256": self.preamble_sha256,
            "source_sha256": _sha(self.source),
            "backend": "nvrtc",
            "options": list(self.options),
            "name_expressions": None,
            "exports": list(self.exports),
        }
        payload["identity_sha256"] = _sha(json.dumps(
            payload, sort_keys=True, separators=(",", ":")))
        return payload


_COMMENT = re.compile(r'/\*.*?\*/|//[^\n]*', re.S)
_EXPORT = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_]\w*)\s*\(')


def exported_kernels(source: str) -> tuple[str, ...]:
    """Enumerate all exports; refuse unfamiliar declarations instead of omitting.

    The shipped Noah-MP sources use explicit extern-C global functions.  This
    is a checked source contract, not a general CUDA parser: a new macro,
    template or linkage-block spelling must extend this parser and its tests.
    """
    text = _COMMENT.sub("", source)
    names = _EXPORT.findall(text)
    if (not names or len(names) != len(set(names))
            or len(names) != len(re.findall(r"\b__global__\b", text))):
        raise ValueError("Noah-MP exports are empty, duplicated or use an "
                         "unrecognised __global__ declaration")
    return tuple(sorted(names))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component_texts(parts):
    return tuple((part, (KERNEL_DIR / f"{part}.cu").read_text(encoding="ascii"))
                 for part in parts)


def _assemble(texts, prefix):
    return prefix + "".join(text for _, text in texts)


def runtime_unit(name: str) -> RuntimeUnit:
    """Snapshot the production composition and options, including VEGE_FLUX.

    All custom runtime factories and the compile-only measurement harness use
    this function.  Ordinary standalone units have byte-identical inputs to
    kernels.load_module; the CPU factory-interception tests enforce that too.
    """
    parts = NOAHMP_TRANSLATION_UNITS[name]
    # Read each component ONCE so its recorded digest and compiled text cannot
    # describe different reads of a concurrently edited file.
    texts = _component_texts(parts)
    if name == "noahmp_vegeflux":
        prefix = ""
        options = ("-std=c++14",)
    else:
        from gpuwm.core.kernels import _preamble
        prefix = _preamble()
        options = DEFAULT_OPTIONS
    source = _assemble(texts, prefix)
    return RuntimeUnit(
        name=name, source=source,
        parts=tuple((part, _sha(text)) for part, text in texts),
        preamble_sha256=_sha(prefix), options=options,
        exports=exported_kernels(source))


def compile_runtime_unit(name: str, *, module_key: str,
                         options: tuple[str, ...] | None = None,
                         source: str | None = None, log_stream=None):
    """Compile and load one Noah-MP runtime unit; no launch, no constant upload.

    THE one ``cp.RawModule`` site for Noah-MP.  Every runtime factory
    (driver, energy, thermal, glacier, the libm slab, the generic loader's
    standalone units) and the frame measurement in
    :func:`gpuwm.core.noahmp_frame_provenance.measure_live` compile through
    here, so the source string and option tuple a forecast hands NVRTC are
    the ones the recorded frames were read from -- by construction rather
    than by a second assembler agreeing with the first.

    ``source`` / ``options`` exist for the negative controls the parity
    suites compile (a perturbed copy, ``-fmad=false``); they are recorded
    under the caller's own ``module_key`` and are not the production unit.
    """
    import cupy as cp
    from gpuwm.certify.kernel_manifest import record_module

    unit = runtime_unit(name)
    code = unit.source if source is None else source
    opts = unit.options if options is None else tuple(options)
    module = cp.RawModule(code=code, options=opts, backend="nvrtc",
                          name_expressions=None)
    if log_stream is None:
        module.compile()
    else:
        module.compile(log_stream=log_stream)
    record_module(module_key, source=code, options=opts, module=module)
    return module


__all__ += ["DEFAULT_OPTIONS", "NOAHMP_PRICING_MODULES", "RuntimeUnit",
            "compile_runtime_unit", "exported_kernels", "pricing_key",
            "runtime_unit"]
