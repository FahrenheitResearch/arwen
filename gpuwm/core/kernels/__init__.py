from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from gpuwm.core.constants import CUDA_DEFINES

_KDIR = Path(__file__).parent


def _preamble() -> str:
    lines = [f"#define {k} {float(v)!r}f" for k, v in CUDA_DEFINES.items()]
    lines.append((_KDIR / "common.cuh").read_text())
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def load_module(name: str):
    import cupy as cp
    src = _preamble() + (_KDIR / f"{name}.cu").read_text()
    mod = cp.RawModule(code=src, options=("-std=c++17",), name_expressions=None)
    mod.compile()
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
    src = _preamble() + prefix + "\n" + (_KDIR / f"{name}.cu").read_text()
    mod = cp.RawModule(code=src, options=("-std=c++17",),
                       name_expressions=None)
    mod.compile()
    return mod


@lru_cache(maxsize=None)
def get_kernel(name: str, func: str):
    """Return one stable CuPy function wrapper per raw-kernel symbol."""
    return load_module(name).get_function(func)


@lru_cache(maxsize=None)
def get_kernel_int_defines(
        name: str, func: str, defines: tuple[tuple[str, int], ...]):
    """Return a cached kernel compiled with validated integer definitions."""
    return load_module_int_defines(name, defines).get_function(func)
