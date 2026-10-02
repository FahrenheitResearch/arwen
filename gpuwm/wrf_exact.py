"""Opt-in strict arithmetic for WRF verification, selected before import.

The default process does not import CuPy or alter its compiler. CuPy adds
FTZ after caller options, so verification normalizes options at the final
NVRTC program boundary as well as before forming the compiled-image cache key.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
from pathlib import Path
import threading

ENABLED = os.environ.get("GPUWM_WRF_EXACT", "0") == "1"
DIAGNOSTICS_ENABLED = (ENABLED and
    os.environ.get("GPUWM_WRF_EXACT_DIAGNOSTICS", "0") == "1")
ADVECTION_ENABLED = (ENABLED and
    os.environ.get("GPUWM_WRF_EXACT_ADVECTION", "0") == "1")
DIFFUSION_ENABLED = (ENABLED and
    os.environ.get("GPUWM_WRF_EXACT_DIFFUSION", "0") == "1")
BIGSTEP_ENABLED = (ENABLED and
    os.environ.get("GPUWM_WRF_EXACT_BIGSTEP", "0") == "1")
STRICT_OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true",
                  "--prec-sqrt=true", "-DGPUWM_WRF_EXACT=1")
for _selected, _macro in (
    (DIAGNOSTICS_ENABLED, "GPUWM_WRF_EXACT_D_DIAGNOSTICS"),
    (ADVECTION_ENABLED, "GPUWM_WRF_EXACT_C_ADVECTION"),
    (DIFFUSION_ENABLED, "GPUWM_WRF_EXACT_C_DIFFUSION"),
    (BIGSTEP_ENABLED, "GPUWM_WRF_EXACT_C_BIGSTEP"),
):
    if _selected:
        STRICT_OPTIONS += (f"-D{_macro}=1",)
_OVERRIDDEN = {"fmad", "ftz", "prec-div", "prec-sqrt", "use_fast_math",
               "DGPUWM_WRF_EXACT", "DGPUWM_WRF_EXACT_D_DIAGNOSTICS",
               "DGPUWM_WRF_EXACT_C_ADVECTION", "DGPUWM_WRF_EXACT_C_DIFFUSION",
               "DGPUWM_WRF_EXACT_C_BIGSTEP"}
_RECORDS: list[dict] = []
_LOCK = threading.Lock()


def effective_options(options) -> tuple[str, ...]:
    """Replace conflicting arithmetic switches, preserving every other option."""
    kept = []
    for option in options:
        key = option.lstrip("-").split("=", 1)[0]
        if key not in _OVERRIDDEN:
            kept.append(option)
    return tuple(kept) + STRICT_OPTIONS


def _record(kind, source, options):
    with _LOCK:
        _RECORDS.append({"kind": kind,
                         "source_sha256": hashlib.sha256(
                             source.encode("utf-8")).hexdigest(),
                         "options": list(options)})


def compile_receipt() -> dict:
    """Requests include cache hits; NVRTC entries identify actual compilations."""
    with _LOCK:
        return {"enabled": ENABLED, "strict_options": list(STRICT_OPTIONS),
                "records": list(_RECORDS)}


def _write_receipt():
    target = os.environ.get("GPUWM_WRF_EXACT_COMPILE_RECEIPT")
    receipt = compile_receipt()
    if target and receipt["records"]:
        # A CLI parent can exit after its forecast subprocess. It compiled
        # nothing and must not overwrite that subprocess's actual receipt.
        Path(target).write_text(json.dumps(receipt, indent=2) + "\n",
                                encoding="utf-8")


def install(compiler=None):
    """Install once, only in a process explicitly opting into verification.

    Normalized cache options prevent loading a default compiled image. The
    program hook also covers direct NVRTC loads, preprocessing, RawKernel,
    RawModule and CuPy-generated elementwise/reduction kernels.
    """
    if not ENABLED:
        return
    if compiler is None:
        from cupy.cuda import compiler
    if getattr(compiler, "_gpuwm_wrf_exact", False):
        return
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        options = effective_options(options)
        _record("cache_request", source, options)
        return original_cache(source, options, *args, **kwargs)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        options = effective_options(options)
        _record("nvrtc_compile", self.src, options)
        return original_program(self, options, log_stream)

    compiler._compile_with_cache_cuda = cached
    compiler._NVRTCProgram.compile = program
    compiler._gpuwm_wrf_exact = True
    atexit.register(_write_receipt)
