"""WRF's historical RUC hydraulic-conductivity parameter perturbation.

The two operations are from WRF V3.9.1 ``module_sf_ruclsm.F:6358-6365``.
WRF 4.6.1 and 4.7.1 retain the arguments but omit this operator. Only the
hydraulic operator is restored; the separately removed optics block is not.
The surrounding deterministic RUC physics remains the current port.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

OPERATOR_ID = "wrf-v3.9.1-ruc-hydraulic-spp-v1"
RUC_SPP_VALUES = (0, 1)
REFERENCE_SHA256 = "834833a79a7a57a4e436a038f3019ce2b5a998141cc5bb9cd39cf2d83ec4d786"
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.ruc_spp:ruc_spp"


@lru_cache(maxsize=None)
def _hydraulic_module(device):
    """Keep WRF's subnormal hydraulic values through direct NVRTC.

    RawModule appends FTZ=true after supplied options. The ordinary loader
    therefore flushed real small conductivities and their diagnostics even
    though the kernel uses explicit round-to-nearest operations.
    """
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile

    source = (Path(__file__).parent / "kernels/ruc_spp.cu").read_text(encoding="utf-8")
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        binary, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "ruc_spp.cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return module


def validate_spp_mode(spp_lsm: int) -> bool:
    if type(spp_lsm) is not int or spp_lsm not in RUC_SPP_VALUES:
        raise ValueError("RUC spp_lsm must be integer 0 or 1")
    return spp_lsm == 1


def pattern_inputs(spp_lsm, pattern, field_sf, shape, *, arrays=np):
    """Admit a read-only pattern and optional mutable diagnostic buffer.

    Disabled calls do not inspect arrays, allocate memory or launch kernels.
    A pattern below -1 would reverse hydraulic transport, so it is refused
    rather than silently clipped to a different perturbation distribution.
    """
    if not validate_spp_mode(spp_lsm):
        return None, None
    if pattern is None:
        raise ValueError("spp_lsm=1 requires pattern_spp_lsm for hydraulic conductivity")
    pattern = arrays.asarray(pattern)
    if pattern.dtype != np.dtype("float32") or tuple(pattern.shape) != tuple(shape):
        raise ValueError(f"pattern_spp_lsm must be float32 with soil-first shape {tuple(shape)}")
    if not bool(arrays.all(arrays.isfinite(pattern))):
        raise ValueError("pattern_spp_lsm must be finite before soil moisture transport")
    if bool(arrays.any(pattern < arrays.float32(-1.0))):
        raise ValueError("pattern_spp_lsm below -1 would make hydraulic conductivity negative")
    if field_sf is not None:
        if (not isinstance(field_sf, arrays.ndarray)
                or getattr(field_sf, "dtype", None) != np.dtype("float32")
                or tuple(getattr(field_sf, "shape", ())) != tuple(shape)):
            raise ValueError(f"field_sf must be a writable float32 buffer with shape {tuple(shape)}")
        if not field_sf.flags.c_contiguous:
            raise ValueError("field_sf must be contiguous so diagnostic writes reach its owner")
        if hasattr(field_sf.flags, "writeable") and not field_sf.flags.writeable:
            raise ValueError("field_sf must be writable")
        if bool(arrays.shares_memory(pattern, field_sf)):
            raise ValueError("field_sf must not alias the read-only SPP pattern")
    return arrays.ascontiguousarray(pattern), field_sf


def hydraulic_spp(hydro, pattern, field_sf=None):
    """Apply the two WRF float32 operations to host reference arrays."""
    if isinstance(pattern, np.ndarray) and np.shares_memory(hydro, pattern):
        raise ValueError("hydraulic conductivity must not alias the read-only SPP pattern")
    pattern, field_sf = pattern_inputs(1, pattern, field_sf, hydro.shape)
    if np.shares_memory(hydro, pattern):
        raise ValueError("hydraulic conductivity must not alias the read-only SPP pattern")
    if field_sf is not None and np.shares_memory(hydro, field_sf):
        raise ValueError("field_sf must not alias hydraulic conductivity")
    if field_sf is not None:
        field_sf[...] = np.multiply(hydro, pattern, dtype=np.float32)
    hydro[...] = np.multiply(hydro, np.add(np.float32(1.0), pattern,
                                          dtype=np.float32), dtype=np.float32)


def hydraulic_spp_device(hydro, pattern, field_sf=None):
    """Apply WRF's operator in one pointwise GPU kernel, without host fields."""
    import cupy as cp

    if hydro.dtype != np.dtype("float32") or not hydro.flags.c_contiguous:
        raise ValueError("hydraulic conductivity must be contiguous float32")
    if isinstance(pattern, cp.ndarray) and cp.shares_memory(hydro, pattern):
        raise ValueError("hydraulic conductivity must not alias the read-only SPP pattern")
    pattern, field_sf = pattern_inputs(1, pattern, field_sf, hydro.shape, arrays=cp)
    if cp.shares_memory(hydro, pattern):
        raise ValueError("hydraulic conductivity must not alias the read-only SPP pattern")
    if field_sf is not None and cp.shares_memory(hydro, field_sf):
        raise ValueError("field_sf must not alias hydraulic conductivity")
    count = int(hydro.size)
    if not count:
        return
    _hydraulic_module(cp.cuda.Device().id).get_function("ruc_spp_hydraulic")(
        ((count + 127) // 128,), (128,),
        (hydro, pattern, field_sf if field_sf is not None else np.uint64(0),
         np.int32(count)))


__all__ = ["OPERATOR_ID", "REFERENCE_SHA256", "validate_spp_mode", "pattern_inputs",
           "hydraulic_spp", "hydraulic_spp_device"]
