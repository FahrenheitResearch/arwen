from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from gpuwm.core.constants import CUDA_DEFINES

_KDIR = Path(__file__).parent

# Every read of a .cu/.cuh in this directory is UTF-8, named explicitly.
#
# These sources are the input to nvrtc AND to the pinned hash the kernel
# manifest records (certify/kernel_manifest.py::record_module digests exactly
# the string module_source returns).  Path.read_text() with no encoding
# decodes with the host's locale, which is cp1252 on a stock Windows box, and
# cp1252 does not fail on a UTF-8 em dash -- it silently turns the three bytes
# into three characters.  acoustic.cu, advection.cu and coriolis_map.cu all
# carry U+2014 in comments, so the same checkout produced two different
# source_sha256 values depending on the host, with nothing raising to say so.
# The Thompson translation units happen to be pure ASCII today, which is luck,
# not a property anyone is maintaining.
_ENCODING = "utf-8"

# Explicit, closed allow-list of modules that receive an extra device header
# prepended between the preamble and their own source.  There is no #include
# path under cupy.RawModule, so this is how the six aerosol-aware Thompson
# (mp_physics=28) translation units share one set of __device__ helpers.
#
# The mechanism is deliberately inert for everything else: a module absent
# from this dict contributes the empty string and therefore assembles a
# BYTE-IDENTICAL source to what it assembled before this hook existed.  That
# is what keeps gpuwm/core/kernels/thompson.cu's compiled source string -- and
# so its PTX, register allocation and FP contraction -- unchanged by
# construction rather than by measurement.  tests/test_kernel_loader_inert.py
# proves it for every .cu file in this directory.
#
# This table must stay a literal name -> filenames mapping.  Do not give it
# filesystem probing, globbing, or any implicit fallback.
_EXTRA_HEADERS: dict[str, tuple[str, ...]] = {
    "thompson_aerosol_probe": ("thompson_aerosol_common.cuh",),
    "thompson_aerosol_state": ("thompson_aerosol_common.cuh",),
    "thompson_aerosol_sat": ("thompson_aerosol_common.cuh",),
    "thompson_aerosol_cold": ("thompson_aerosol_common.cuh",),
    "thompson_aerosol_warm": ("thompson_aerosol_common.cuh",),
    "thompson_aerosol_sed": ("thompson_aerosol_common.cuh",),
    # The LW solver derives the Planck sources itself instead of loading
    # what rrtmgp_planck_sources wrote; the helpers it needs live in a
    # header whose every FP op is pinned (HOWTO 13.6j).  rrtmgp_gas is
    # deliberately NOT listed -- it keeps its own helpers verbatim so its
    # assembled source and PTX stay byte-identical.
    "rrtmgp_rte": ("rrtmgp_planck_common.cuh",),
}

#: Read-only view for tests and freeze receipts.
EXTRA_HEADERS = MappingProxyType(_EXTRA_HEADERS)


def _preamble() -> str:
    lines = [f"#define {k} {float(v)!r}f" for k, v in CUDA_DEFINES.items()]
    lines.append((_KDIR / "common.cuh").read_text(encoding=_ENCODING))
    return "\n".join(lines) + "\n"


def _extra_header_text(name: str) -> str:
    """Return the allow-listed headers for ``name``, or ``''`` for any other.

    The empty-string return for an unlisted module is the whole point: it
    makes the assembled source byte-identical to the pre-hook string.
    """
    return "".join((_KDIR / header).read_text(encoding=_ENCODING)
                   for header in _EXTRA_HEADERS.get(name, ()))


def module_source(name: str) -> str:
    """The exact source string :func:`load_module` hands to nvrtc."""
    return (_preamble() + _extra_header_text(name)
            + (_KDIR / f"{name}.cu").read_text(encoding=_ENCODING))


@lru_cache(maxsize=None)
def load_module(name: str):
    import cupy as cp
    src = module_source(name)
    mod = cp.RawModule(code=src, options=("-std=c++17",), name_expressions=None)
    mod.compile()
    from gpuwm.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}",
                  source=src, options=("-std=c++17",), module=mod)
    return mod


@lru_cache(maxsize=None)
def load_module_int_defines(
        name: str, defines: tuple[tuple[str, int], ...]):
    """Compile one kernel source with a small, identity-bound integer tier.

    This keeps compile-time local-array bounds specialized without enlarging
    the common kernel.  Only uppercase C-preprocessor identifiers and positive
    integer values are accepted; callers cannot inject arbitrary source text.
    """
    import re
    import cupy as cp

    normalized = tuple((str(key), int(value)) for key, value in defines)
    if normalized != defines:
        raise TypeError("kernel integer defines must be canonical (str, int) pairs")
    for key, value in normalized:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            raise ValueError(f"invalid CUDA preprocessor identifier {key!r}")
        if isinstance(value, bool) or value < 1:
            raise ValueError(
                f"CUDA integer define {key} must be a positive integer")
    prefix = "\n".join(f"#define {key} {value}" for key, value in normalized)
    src = module_source_int_defines(name, normalized, prefix=prefix)
    mod = cp.RawModule(code=src, options=("-std=c++17",),
                       name_expressions=None)
    mod.compile()
    from gpuwm.certify.kernel_manifest import record_module
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    record_module(f"{MODULE_KEY_ROOT}:{name}[{tier}]",
                  source=src, options=("-std=c++17",), module=mod)
    return mod


#: Manifest namespace for the translation units the loaders above compile.
#: Declared below them because the FTZ receipt pins their RawModule lines.
MODULE_KEY_ROOT = "gpuwm.core.kernels"


def module_source_int_defines(
        name: str, defines: tuple[tuple[str, int], ...],
        *, prefix: str | None = None) -> str:
    """The exact source :func:`load_module_int_defines` hands to nvrtc.

    The allow-listed header, when present, goes immediately after the
    preamble, exactly as in :func:`module_source`; for every module absent
    from ``_EXTRA_HEADERS`` the inserted text is empty and the string is
    byte-identical to the pre-hook assembly.
    """
    if prefix is None:
        prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
    return (_preamble() + _extra_header_text(name) + prefix + "\n"
            + (_KDIR / f"{name}.cu").read_text(encoding=_ENCODING))


@lru_cache(maxsize=None)
def get_kernel(name: str, func: str):
    """Return one stable CuPy function wrapper per raw-kernel symbol."""
    return load_module(name).get_function(func)


@lru_cache(maxsize=None)
def get_kernel_int_defines(
        name: str, func: str, defines: tuple[tuple[str, int], ...]):
    """Return a cached kernel compiled with validated integer definitions."""
    return load_module_int_defines(name, defines).get_function(func)
