"""Retain subnormals for marked translation units, including CuPy's final options.

CuPy adds -ftz=true after RawModule options. Normalize both the cache key
and the NVRTC invocation, scoped by a source marker so other units keep
their existing arithmetic. Strict verification can wrap this hook too.
"""
from __future__ import annotations

import functools
from contextvars import ContextVar

MARKER = "#define GPUWM_RETAIN_SUBNORMALS 1\n"
_ACTIVE = ContextVar("gpuwm_retain_subnormals_compile", default=False)


def effective_options(source, options):
    if MARKER not in source and not _ACTIVE.get():
        return tuple(options)
    return tuple(o for o in options
                 if o.lstrip("-").split("=", 1)[0] != "ftz") + ("--ftz=false",)


def install(compiler=None):
    if compiler is None:
        from cupy.cuda import compiler
    if getattr(compiler, "_gpuwm_retain_subnormals", False):
        return
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        # CuPy preprocesses an empty program before compiling this source.
        # It carries the same options but no marker, so retain the policy
        # for all NVRTC calls in this cache request, isolated per context.
        token = _ACTIVE.set(MARKER in source)
        try:
            return original_cache(source, effective_options(source, options),
                                  *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        return original_program(self, effective_options(self.src, options), log_stream)

    compiler._compile_with_cache_cuda = cached
    compiler._NVRTCProgram.compile = program
    compiler._gpuwm_retain_subnormals = True
