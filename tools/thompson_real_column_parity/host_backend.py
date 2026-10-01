"""Run gpuwm's CUDA Thompson kernels on the host, through the production
adapter, with no GPU anywhere.

Two pieces, both local to the calling process:

1. ``HostModule`` compiles ``gpuwm.core.kernels.module_source(name)`` -- the
   exact string nvrtc receives -- as plain C++ behind ``cuda_host_shim.h``,
   with one generated launcher per ``extern "C" __global__`` kernel, and
   loads the shared object with ctypes.
2. ``install()`` puts a NumPy-backed ``cupy`` module in ``sys.modules`` and
   points ``gpuwm.core.kernels.get_kernel`` (and its integer-define sibling)
   at the host builds, so every launcher in ``gpuwm.core.thompson*`` and the
   adapter ``gpuwm.core.microphysics_aerosol._apply_thompson_aerosol`` run
   unmodified: same launch order, same arguments, same scratch protocol.

Call ``install()`` BEFORE importing any ``gpuwm.core`` module that imports
cupy.  It refuses to run in a process where the real cupy is already
imported, because mixing device and host arrays is not something this
backend can make safe.

The build is g++ with ``-O2 -std=c++17 -ffp-contract=off
-fno-unsafe-math-optimizations -fno-tree-vectorize``: IEEE per-operation
rounding, no fused multiply-add, no reassociation, no libmvec.  See
``cuda_host_shim.h`` for what that grades and what it cannot see.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHIM = HERE / "cuda_host_shim.h"

CXX_FLAGS = ("-O2", "-std=c++17", "-ffp-contract=off",
             "-fno-unsafe-math-optimizations", "-fno-tree-vectorize",
             "-shared", "-fPIC")

#: Environment override for the build cache directory.
BUILD_DIR_ENV = "GPUWM_HOST_KERNEL_BUILD_DIR"

_KERNEL_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(')

_SCALAR_CTYPES = {
    "float": ctypes.c_float,
    "real": ctypes.c_float,
    "double": ctypes.c_double,
    "int": ctypes.c_int,
    "int32_t": ctypes.c_int32,
    "unsigned": ctypes.c_uint,
    "unsigned int": ctypes.c_uint,
    "uint32_t": ctypes.c_uint32,
    "long long": ctypes.c_longlong,
    "int64_t": ctypes.c_int64,
    "unsigned long long": ctypes.c_ulonglong,
    "uint64_t": ctypes.c_uint64,
    "size_t": ctypes.c_size_t,
    "bool": ctypes.c_bool,
    "char": ctypes.c_char,
    "unsigned char": ctypes.c_ubyte,
    "uint8_t": ctypes.c_uint8,
}

_POINTER_DTYPES = {
    "float": np.float32,
    "real": np.float32,
    "double": np.float64,
    "int": np.int32,
    "int32_t": np.int32,
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "uint32_t": np.uint32,
    "long long": np.int64,
    "int64_t": np.int64,
    "unsigned long long": np.uint64,
    "uint64_t": np.uint64,
    "bool": np.bool_,
    "char": np.int8,
    "unsigned char": np.uint8,
    "uint8_t": np.uint8,
}


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


class Param:
    """One kernel parameter: its declaration text, name and base type."""

    __slots__ = ("text", "name", "base", "pointer")

    def __init__(self, text: str):
        self.text = " ".join(text.split())
        tokens = re.findall(r"[A-Za-z_]\w*|\*", text)
        self.pointer = "*" in tokens
        idents = [t for t in tokens if t != "*"
                  and t not in ("const", "__restrict__", "volatile",
                                "restrict")]
        if len(idents) < 2:
            raise ValueError(f"cannot parse kernel parameter {text!r}")
        self.name = idents[-1]
        self.base = " ".join(idents[:-1])
        if self.pointer:
            if self.base not in _POINTER_DTYPES:
                raise ValueError(f"unsupported pointer type in {text!r}")
        elif self.base not in _SCALAR_CTYPES:
            raise ValueError(f"unsupported scalar type in {text!r}")


def _preprocess(source: str) -> str:
    """Macro-expand a source string with the host compiler's preprocessor.

    Some kernels spell their parameter lists as macros, so signatures are
    read after expansion.  The shim is NOT included here: it defines
    ``__global__`` away, and the regex below needs it.
    """
    done = subprocess.run(
        [compiler(), "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=source, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError("preprocessing failed: " + done.stderr[-4000:])
    return done.stdout


def parse_kernels(source: str) -> dict[str, list[Param]]:
    """``{kernel name: [Param, ...]}`` for every extern "C" __global__."""
    text = _strip_comments(_preprocess(source))
    out: dict[str, list[Param]] = {}
    for match in _KERNEL_RE.finditer(text):
        name = match.group(1)
        depth, j = 1, match.end()
        while depth:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        body = text[match.end():j - 1]
        params = [Param(p) for p in _split_top_level(body)]
        if name in out:
            raise ValueError(f"kernel {name} defined twice")
        out[name] = params
    return out


def _launcher(name: str, params: list[Param]) -> str:
    decl = ", ".join(p.text for p in params)
    args = ", ".join(p.name for p in params)
    head = ("unsigned gx, unsigned gy, unsigned gz, "
            "unsigned bx, unsigned by, unsigned bz")
    if decl:
        head += ", " + decl
    return (
        f'extern "C" void hostlaunch__{name}({head})\n'
        "{\n"
        "    gridDim.x = gx; gridDim.y = gy; gridDim.z = gz;\n"
        "    blockDim.x = bx; blockDim.y = by; blockDim.z = bz;\n"
        "    for (unsigned kz = 0; kz < gz; ++kz)\n"
        "    for (unsigned ky = 0; ky < gy; ++ky)\n"
        "    for (unsigned kx = 0; kx < gx; ++kx) {\n"
        "        blockIdx.x = kx; blockIdx.y = ky; blockIdx.z = kz;\n"
        "        for (unsigned tz = 0; tz < bz; ++tz)\n"
        "        for (unsigned ty = 0; ty < by; ++ty)\n"
        "        for (unsigned tx = 0; tx < bx; ++tx) {\n"
        "            threadIdx.x = tx; threadIdx.y = ty; threadIdx.z = tz;\n"
        f"            {name}({args});\n"
        "        }\n"
        "    }\n"
        "}\n")


def host_translation_unit(module_text: str, *, rates: bool = False) -> str:
    """The C++ the host compiles for one nvrtc source string."""
    kernels = parse_kernels(module_text)
    parts = []
    if rates:
        parts.append("#define GPUWM_HOST_RATES 1\n")
    parts.append(f'#include "{SHIM.name}"\n')
    parts.append('#line 1 "nvrtc-source"\n')
    parts.append(module_text)
    parts.append("\n// ---- generated host launchers ----\n")
    for name, params in kernels.items():
        parts.append(_launcher(name, params))
    return "".join(parts)


def _build_dir() -> Path:
    env = os.environ.get(BUILD_DIR_ENV)
    path = Path(env) if env else Path.home() / ".cache" / "gpuwm-host-kernels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def compiler() -> str:
    """The C++ compiler: ``$CXX``, else g++, else c++."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise RuntimeError("no C++ compiler on PATH (set CXX)")
    return cxx


class HostModule:
    """A host-built translation unit and its kernels."""

    def __init__(self, label: str, text: str, *, rates: bool = False):
        self.label = label
        self.kernels = parse_kernels(text)
        unit = host_translation_unit(text, rates=rates)
        cxx = compiler()
        key = "\0".join((unit, " ".join(CXX_FLAGS), cxx,
                         SHIM.read_text(encoding="utf-8")))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        build = _build_dir()
        stem = f"{re.sub(r'[^A-Za-z0-9_]+', '_', label)}-{digest}"
        so = build / f"{stem}.so"
        if not so.exists():
            # The source is written under this process's own name: several
            # harness processes sharing one cache used to write the same
            # .cpp at once, and one compiled another's half-written file into
            # a library missing its later launchers, which the cache then
            # kept.
            cpp = build / f"{stem}.{os.getpid()}.cpp"
            cpp.write_text(unit, encoding="utf-8", newline="\n")
            tmp = build / f"{stem}.{os.getpid()}.tmp.so"
            done = subprocess.run(
                [cxx, *CXX_FLAGS, "-I", str(HERE), str(cpp), "-o", str(tmp),
                 "-lm"], capture_output=True, text=True, check=False)
            cpp.unlink(missing_ok=True)
            if done.returncode != 0:
                raise RuntimeError(
                    f"host build of {label} failed:\n{done.stderr[-6000:]}")
            os.replace(tmp, so)
        self.path = so
        self.lib = ctypes.CDLL(str(so))
        self.rates = rates
        self._functions: dict[str, HostKernel] = {}

    def get_function(self, name: str) -> "HostKernel":
        fn = self._functions.get(name)
        if fn is None:
            if name not in self.kernels:
                raise AttributeError(f"{self.label} has no kernel {name!r}")
            fn = HostKernel(self, name, self.kernels[name])
            self._functions[name] = fn
        return fn


class HostKernel:
    """Callable with CuPy's RawKernel signature: ``k(grid, block, args)``."""

    def __init__(self, module: HostModule, name: str, params: list[Param]):
        self.module = module
        self.name = name
        self.params = params
        fn = getattr(module.lib, f"hostlaunch__{name}")
        fn.restype = None
        argtypes = [ctypes.c_uint] * 6
        for p in params:
            argtypes.append(ctypes.c_void_p if p.pointer
                            else _SCALAR_CTYPES[p.base])
        fn.argtypes = argtypes
        self._fn = fn

    def __call__(self, grid, block, args, shared_mem=0, stream=None,
                 **_ignored):
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        block = tuple(int(b) for b in block) + (1,) * (3 - len(block))
        if len(args) != len(self.params):
            raise TypeError(
                f"{self.name} takes {len(self.params)} arguments, "
                f"got {len(args)}")
        converted = []
        for p, value in zip(self.params, args):
            if p.pointer:
                if value is None:
                    # CuPy passes None as a null pointer; the kernels test
                    # optional buffers against nullptr.
                    converted.append(None)
                    continue
                if not isinstance(value, np.ndarray):
                    raise TypeError(
                        f"{self.name}: {p.name} must be an ndarray, got "
                        f"{type(value).__name__}")
                want = np.dtype(_POINTER_DTYPES[p.base])
                if value.dtype != want:
                    raise TypeError(
                        f"{self.name}: {p.name} is {value.dtype}, the "
                        f"kernel reads {want}")
                if not (value.flags.c_contiguous
                        or value.flags.f_contiguous):
                    raise ValueError(
                        f"{self.name}: {p.name} is not contiguous")
                converted.append(value.ctypes.data)
            else:
                ctype = _SCALAR_CTYPES[p.base]
                if isinstance(value, np.ndarray):
                    if value.size != 1:
                        raise TypeError(
                            f"{self.name}: scalar {p.name} got an array")
                    value = value.reshape(()).item()
                if ctype in (ctypes.c_float, ctypes.c_double):
                    converted.append(float(value))
                elif ctype is ctypes.c_bool:
                    converted.append(bool(value))
                else:
                    converted.append(int(value))
        self._fn(*grid, *block, *converted)


# ---------------------------------------------------------------------------
# The NumPy-backed cupy stand-in and the loader patch.
# ---------------------------------------------------------------------------

_INSTALLED: dict[str, object] = {}
_TRANSFORMS: dict[str, object] = {}
_RATES = {"on": False}


def _fake_cupy() -> types.ModuleType:
    mod = types.ModuleType("cupy")

    def asnumpy(value, order="K", **_kw):
        return np.asarray(value, order=order)

    def asarray(value, dtype=None, order=None, **_kw):
        return np.asarray(value, dtype=dtype, order=order)

    def get_array_module(*_args):
        return np

    class _NoRawModule:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "the host backend compiles kernels itself; cupy.RawModule "
                "is not available")

    def __getattr__(name):
        return getattr(np, name)

    # ``cupy.cuda.Stream.null.synchronize()`` is how callers fence a launch;
    # on the host every launch has returned by the time the call does.
    # There is deliberately no ``Device``: table loaders read its absence as
    # "no device id", which is the truth here.
    cuda = types.SimpleNamespace(Stream=types.SimpleNamespace(
        null=types.SimpleNamespace(synchronize=lambda: None)))

    mod.__dict__.update(cuda=cuda,
        __gpuwm_host_backend__=True, ndarray=np.ndarray, asnumpy=asnumpy,
        asarray=asarray, get_array_module=get_array_module,
        RawModule=_NoRawModule, RawKernel=_NoRawModule,
        __getattr__=__getattr__)
    return mod


def set_transform(module_name: str, transform) -> None:
    """Rewrite one module's source before the host build (instrumentation).

    ``transform(text) -> text``; ``None`` removes it.  Clears the caches.
    """
    if transform is None:
        _TRANSFORMS.pop(module_name, None)
    else:
        _TRANSFORMS[module_name] = transform
    _module_cached.cache_clear()
    _module_int_cached.cache_clear()


def set_rates(on: bool) -> None:
    """Build with ``GPUWM_HOST_RATES`` so instrumented copies can report."""
    _RATES["on"] = bool(on)
    _module_cached.cache_clear()
    _module_int_cached.cache_clear()


@lru_cache(maxsize=None)
def _module_cached(name: str) -> HostModule:
    from gpuwm.core import kernels as K
    text = K.module_source(name)
    transform = _TRANSFORMS.get(name)
    if transform is not None:
        text = transform(text)
    suffix = "-rates" if _RATES["on"] else ""
    return HostModule(name + suffix, text, rates=_RATES["on"])


@lru_cache(maxsize=None)
def _module_int_cached(name: str, defines: tuple) -> HostModule:
    from gpuwm.core import kernels as K
    text = K.module_source_int_defines(name, defines)
    transform = _TRANSFORMS.get(name)
    if transform is not None:
        text = transform(text)
    tier = "_".join(f"{k}{v}" for k, v in defines)
    return HostModule(f"{name}-{tier}", text, rates=_RATES["on"])


def build_module(name: str) -> HostModule:
    """The host build of one kernel module (cached per process)."""
    return _module_cached(name)


def host_get_kernel(name: str, func: str) -> HostKernel:
    """Drop-in for ``gpuwm.core.kernels.get_kernel``."""
    return _module_cached(name).get_function(func)


def host_get_kernel_int_defines(name: str, func: str,
                                defines: tuple) -> HostKernel:
    """Drop-in for ``gpuwm.core.kernels.get_kernel_int_defines``."""
    return _module_int_cached(name, tuple(defines)).get_function(func)


def install() -> None:
    """Make ``import cupy`` and ``get_kernel`` resolve to the host backend."""
    if _INSTALLED:
        return
    existing = sys.modules.get("cupy")
    if existing is not None and not getattr(
            existing, "__gpuwm_host_backend__", False):
        raise RuntimeError(
            "the real cupy is already imported in this process; the host "
            "backend must be installed before any gpuwm.core import")
    sys.modules["cupy"] = _fake_cupy()
    from gpuwm.core import kernels as K
    originals = {
        "get_kernel": K.get_kernel,
        "get_kernel_int_defines": K.get_kernel_int_defines,
        "load_module": K.load_module,
        "load_module_int_defines": K.load_module_int_defines,
    }
    K.get_kernel = host_get_kernel
    K.get_kernel_int_defines = host_get_kernel_int_defines
    K.load_module = _module_cached
    K.load_module_int_defines = _module_int_cached
    _INSTALLED.update(originals)
    replacements = {id(v): getattr(K, k) for k, v in originals.items()}
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("gpuwm") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if id(value) in replacements:
                setattr(mod, attr, replacements[id(value)])
    # The level-parallel fallout kernels need a block barrier, which the
    # serial host launchers cannot honour, and are compiled for the device
    # only; the column kernels they match bit for bit run here instead.
    from gpuwm.core import thompson
    thompson.LEVEL_PARALLEL_FALLOUT = False


__all__ = [
    "CXX_FLAGS", "HostKernel", "HostModule", "build_module", "compiler",
    "host_get_kernel", "host_get_kernel_int_defines",
    "host_translation_unit", "install", "parse_kernels", "set_rates",
    "set_transform",
]
