"""Member-batched momentum coupling and sequential Omega column scans.

The numerical bodies come from the existing dycore ElementwiseKernel factories.
N=1 calls those factories with their original argument views and sizes. N>1
embeds each factory's operation verbatim in a raw CUDA entry, with independent
member pointers and scalar thread indices. These are flux-building components,
not an implemented batched forecast executor or a forecast identity verdict.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from gpuwm.core.device_cache import cuda_cache
from gpuwm.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, batch_grid, generate_batch_source,
    pack_pointer_strides, validate_pointer_arguments,
)

_OPTIONS = ("-std=c++17", "-fmad=false")
_TPB = 128
_FLOAT32 = np.dtype("float32")
_INT32_MAX = np.iinfo(np.int32).max


def _original(kind, has_msf, reciprocal=False):
    from gpuwm.core.dycore import _couple_momentum_kernel, _omega_column_kernel

    if kind == "omega":
        return _omega_column_kernel(has_msf)
    if kind == "momentum":
        return _couple_momentum_kernel(has_msf, reciprocal)
    raise ValueError(f"unknown batched flux component {kind!r}")


def _checked_operation(kind, has_msf, reciprocal=False):
    """Read the installed CuPy factory contract instead of copying its math."""
    kernel = _original(kind, has_msf, reciprocal)
    if kind == "omega":
        expected = [(name, "T", True, True)
                    for name in ("ru", "rv", "dnw", "c1h")]
        if has_msf:
            expected.append(("msft", "T", True, True))
        expected += [("rdx", "T", False, True), ("rdy", "T", False, True)]
        expected += [(name, "int", False, True) for name in ("nz", "ny", "nx")]
        expected.append(("ww", "T", True, False))
    else:
        expected = [("wind", "T", False, True)]
        expected += [(name, "T", True, True) for name in ("c1h", "c2h", "muface")]
        if has_msf:
            expected.append(("msf", "T", True, True))
        expected += [("ncol", "int", False, True), ("flux", "T", False, False)]
    try:
        actual = [(p.name, p.ctype, bool(p.raw), bool(p.is_const))
                  for p in kernel.in_params + kernel.out_params]
        operation = kernel.operation
        preamble = kernel.preamble
    except AttributeError as error:
        raise RuntimeError(
            "CuPy does not expose the operation/parameter contract needed "
            "to preserve the original flux arithmetic") from error
    if actual != expected or preamble or not isinstance(operation, str) or not operation:
        raise RuntimeError(
            f"{kind} ElementwiseKernel contract changed; the member adapter "
            "cannot infer new parameter ownership or numerical dependencies")
    if "_ind" in operation:
        raise RuntimeError(
            f"{kind} operation uses a CuPy indexer that a scalar member index cannot replace")
    return operation


@lru_cache(maxsize=None)
def _raw_source(kind, has_msf, reciprocal=False):
    """Build a typed pointer entry containing the factory operation verbatim."""
    operation = _checked_operation(kind, has_msf, reciprocal)
    if kind == "omega":
        parameters = ["const float* __restrict__ ru", "const float* __restrict__ rv",
                      "const float* __restrict__ dnw", "const float* __restrict__ c1h"]
        pointers = [PointerSpec("ru", "member"), PointerSpec("rv", "member"),
                    PointerSpec("dnw", "shared"), PointerSpec("c1h", "shared")]
        names = ["ru", "rv", "dnw", "c1h"]
        if has_msf:
            parameters.append("const float* __restrict__ msft")
            pointers.append(PointerSpec("msft", "shared"))
            names.append("msft")
        parameters += ["float rdx", "float rdy", "int nz", "int ny", "int nx",
                       "float* __restrict__ ww", "int scalar_size"]
        pointers.append(PointerSpec("ww", "member"))
        names += ["rdx", "rdy", "nz", "ny", "nx", "ww", "scalar_size"]
        entry = "omega_columns"
        locals_text = ""
    else:
        parameters = ["const float* __restrict__ wind_values",
                      "const float* __restrict__ c1h", "const float* __restrict__ c2h",
                      "const float* __restrict__ muface"]
        pointers = [PointerSpec("wind_values", "member"), PointerSpec("c1h", "shared"),
                    PointerSpec("c2h", "shared"), PointerSpec("muface", "member")]
        names = ["wind_values", "c1h", "c2h", "muface"]
        if has_msf:
            parameters.append("const float* __restrict__ msf")
            pointers.append(PointerSpec("msf", "shared"))
            names.append("msf")
        parameters += ["int ncol", "float* __restrict__ flux_values", "int scalar_size"]
        pointers.append(PointerSpec("flux_values", "member"))
        names += ["ncol", "flux_values", "scalar_size"]
        entry = "couple_momentum"
        locals_text = "const T wind = wind_values[i];\nT& flux = flux_values[i];\n"
    source = ("typedef float T;\nextern \"C\" __global__ void " + entry + "("
              + ", ".join(parameters) + ") {\n"
              + "const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
              + "if (i >= scalar_size) return;\n" + locals_text
              + operation + "\n}\n")
    spec = KernelSpec("ensemble_fluxes", entry, tuple(pointers), options=_OPTIONS)
    return source, spec, tuple(names)


@cuda_cache(maxsize=None)
def _compiled(kind, members, has_msf, reciprocal=False):
    import cupy as cp
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.core.kernels import _compile_observed

    source, spec, _ = _raw_source(kind, has_msf, reciprocal)
    source = generate_batch_source(source, spec, members)
    key = (f"gpuwm.ensemble.batch_fluxes:{kind}[members={members},"
           f"map={int(has_msf)},reciprocal={int(reciprocal)}]")
    # CuPy's compiler hook applies the same FTZ/strict process setting as the
    # original ElementwiseKernel. Record the exact source/options handed in.
    module = cp.RawModule(code=source, options=_OPTIONS)
    _compile_observed(module, key)
    record_module(key, source=source, options=_OPTIONS, module=module)
    return module.get_function(spec.entry)


def _cuda_array(array, name, ndim):
    if not hasattr(array, "__cuda_array_interface__"):
        raise TypeError(f"{name} needs a CUDA array backing")
    if array.dtype != _FLOAT32 or not array.flags.c_contiguous:
        raise ValueError(f"{name} needs contiguous float32 storage")
    if array.ndim != ndim or not array.nbytes or any(n < 1 for n in array.shape):
        raise ValueError(f"{name} needs nonempty {ndim}-D storage")
    return array


def _member(array, name, ndim):
    _cuda_array(array, name, ndim)
    members = int(array.shape[0])
    stride = int(array.strides[0])
    if members * stride != array.nbytes:
        raise ValueError(f"{name} must expose every member slab and its exact byte stride")
    return members, stride


def _shared(array, name, shape):
    _cuda_array(array, name, len(shape))
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} must have shared shape {tuple(shape)}")


def _shape(array, name, expected, ndim):
    members, stride = _member(array, name, ndim)
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f"{name} must have member/C-grid shape {tuple(expected)}")
    return members, stride


def _separate(output, inputs):
    start = int(output.__cuda_array_interface__["data"][0])
    end = start + int(output.nbytes)
    for name, value in inputs:
        other = int(value.__cuda_array_interface__["data"][0])
        if start < other + int(value.nbytes) and other < end:
            raise ValueError(
                f"flux output overlaps {name}; its writes would change "
                "member inputs or shared coefficients during the operation")


def _scalar_index_limit(words):
    if words > _INT32_MAX:
        raise ValueError("one member's flux grid exceeds the original kernel's int32 indexing")


def _wrf_reciprocal(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{name} must be one real spatial spacing")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        rounded = np.float32(value)
        result = np.float32(1.0) / rounded
    if not np.isfinite(rounded) or rounded <= 0.0 or not np.isfinite(result):
        raise ValueError(f"{name} cannot produce the original kernel's finite float32 reciprocal")
    return result


def _logical(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _raw_launch(kind, members, has_msf, reciprocal, scalar_size, args, strides):
    _, spec, names = _raw_source(kind, has_msf, reciprocal)
    descriptor = pack_pointer_strides(spec, strides)
    validate_pointer_arguments(spec, members, args, strides, names)
    grid = batch_grid(((scalar_size + _TPB - 1) // _TPB,), members)

    def launch():
        _compiled(kind, members, has_msf, reciprocal)(
            grid, (_TPB,), tuple(args) + (descriptor,))

    return launch


def prepare_omega_columns(ru, rv, ww, dnw, c1h, *, dx, dy,
                          has_msf=False, msft=None):
    """Bind one member-batched surface-up Omega scan with its original folds.

    RU/RV/WW are full 4-D C-grid member backings. DNW/C1H and the optional
    mass-point map factor are shared. The two vertical passes remain serial
    within each member/column; no reduction or scan ever mixes members.
    The original output explicitly writes positive zero at surface and lid.
    """
    members, ru_stride = _member(ru, "ru", 4)
    nz, ny, nxp1 = map(int, ru.shape[1:])
    nx = nxp1 - 1
    if nx < 1:
        raise ValueError("ru needs at least one mass column and its staggered face")
    _, rv_stride = _shape(rv, "rv", (members, nz, ny + 1, nx), 4)
    _, ww_stride = _shape(ww, "ww", (members, nz + 1, ny, nx), 4)
    _shared(dnw, "dnw", (nz,))
    _shared(c1h, "c1h", (nz,))
    has_msf = _logical(has_msf, "has_msf")
    inputs = [("ru", ru), ("rv", rv), ("dnw", dnw), ("c1h", c1h)]
    if has_msf:
        _shared(msft, "msft", (ny, nx))
        inputs.append(("msft", msft))
    _separate(ww, inputs)
    _scalar_index_limit(max(ru_stride, rv_stride, ww_stride) // _FLOAT32.itemsize)
    rdx, rdy = _wrf_reciprocal(dx, "dx"), _wrf_reciprocal(dy, "dy")
    dimensions = (rdx, rdy, np.int32(nz), np.int32(ny), np.int32(nx))
    if members == 1:
        original = _original("omega", has_msf)
        args = [ru[0].reshape(-1), rv[0].reshape(-1), dnw, c1h]
        if has_msf:
            args.append(msft.reshape(-1))
        args += list(dimensions) + [ww[0].reshape(-1)]

        def launch():
            original(*args, size=ny * nx)

        return launch
    args = [ru, rv, dnw, c1h]
    strides = {"ru": ru_stride, "rv": rv_stride, "ww": ww_stride,
               "dnw": 0, "c1h": 0}
    if has_msf:
        args.append(msft)
        strides["msft"] = 0
    args += list(dimensions) + [ww, np.int32(ny * nx)]
    return _raw_launch("omega", members, has_msf, False, ny * nx, tuple(args), strides)


def prepare_couple_momentum(wind, muface, flux, c1h, c2h, *,
                            has_msf=False, msf=None, reciprocal=False):
    """Bind one coupled-momentum operation over independent member slabs.

    Wind/flux are full 4-D member fields; face dry mass is a full 3-D member
    surface field. C1H/C2H and the optional 2-D map factor are shared. The
    reciprocal flag selects the original v-map arithmetic when appropriate;
    it does not invent a new division order or choose a physics/clock policy.
    """
    members, wind_stride = _member(wind, "wind", 4)
    nz, ny, nx = map(int, wind.shape[1:])
    _, flux_stride = _shape(flux, "flux", tuple(wind.shape), 4)
    _, mass_stride = _shape(muface, "muface", (members, ny, nx), 3)
    _shared(c1h, "c1h", (nz,))
    _shared(c2h, "c2h", (nz,))
    has_msf = _logical(has_msf, "has_msf")
    reciprocal = _logical(reciprocal, "reciprocal")
    inputs = [("wind", wind), ("muface", muface), ("c1h", c1h), ("c2h", c2h)]
    if has_msf:
        _shared(msf, "msf", (ny, nx))
        inputs.append(("msf", msf))
    _separate(flux, inputs)
    scalar_size, ncol = nz * ny * nx, ny * nx
    _scalar_index_limit(scalar_size)
    if members == 1:
        original = _original("momentum", has_msf, reciprocal)
        args = [wind[0], c1h, c2h, muface[0].reshape(-1)]
        if has_msf:
            args.append(msf.reshape(-1))
        args += [np.int32(ncol), flux[0]]

        def launch():
            original(*args)

        return launch
    args = [wind, c1h, c2h, muface]
    strides = {"wind_values": wind_stride, "muface": mass_stride,
               "flux_values": flux_stride, "c1h": 0, "c2h": 0}
    if has_msf:
        args.append(msf)
        strides["msf"] = 0
    args += [np.int32(ncol), flux, np.int32(scalar_size)]
    return _raw_launch("momentum", members, has_msf, reciprocal,
                       scalar_size, tuple(args), strides)
