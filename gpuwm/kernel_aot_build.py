"""Build the shared source-bound kernel pack from real engine compiler requests.

Set ``GPUWM_AOT_RECORD=recipes.json`` for a real forecast or warm-kernels
worker under each compiler family and numerics mode. Build in that family's
pinned environment, then stage kernel-pack-cuda12.json beside cuda13.
This module uses CPU compiler processes only, never a CUDA context.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import lzma
import multiprocessing
import os
import time

from gpuwm.kernel_aot import (ARCHITECTURES, Pack, compiler_identity, local_options,
                              recipe, source_hash)
from gpuwm.kernel_pack import read, write


def cached_program_filename(row, options, target):
    """Reproduce CuPy's program name, including headers used only in its key.

    CUB embeds assertion filenames in its images. A placeholder filename
    produces different bytes despite identical source and arithmetic flags.
    """
    from gpuwm.kernel_plan import _cupy_entry, _cupy_extra_sources
    metadata = row.get("compile_request") or {}
    expected = metadata.get("extra_source")
    extra = None
    if expected is not None:
        candidates = [value for value in _cupy_extra_sources()
                      if value is not None and "sha256:" + source_hash(value) == expected]
        if not candidates:
            raise ValueError("compiler recipe headers differ from the installed CuPy headers")
        extra = candidates[0]
    return _cupy_entry(dict(source=row["source"], options=options,
                            name_expressions=metadata.get("name_expressions")), target, extra) + ".cu"


def compile_entry(row, arch):
    from cupy.cuda import compiler
    from gpuwm.certify.compile_platform import nvrtc_banner
    from gpuwm.kernel_compile_worker import compile_request

    target = str(arch).removeprefix("sm_")
    options = tuple(local_options(row["options"]))
    cached = row.get("compiler_route") not in ("direct-nvrtc", "raw-nvrtc")
    if cached:
        options += ("-ftz=true",)
    if (row.get("compiler_route") != "raw-nvrtc"
            and compiler._get_bool_env_variable("CUPY_CUDA_COMPILE_WITH_DEBUG", False)):
        options += ("--device-debug", "--generate-line-info")
    metadata = row.get("compile_request") or {}
    filename = metadata.get("filename", "kern.cu")
    if cached:
        # CuPy's memoized selector queries the current device even with an
        # explicit arch. The pack builder must also name its program without
        # creating a context, before entering compile_request's own selector.
        selector = compiler._get_arch_for_options_for_nvrtc
        def explicit(selected=None):
            if str(selected) != target:
                raise ValueError("AOT filename target changed")
            return "-arch=sm_" + target, "cubin"
        compiler._get_arch_for_options_for_nvrtc = explicit
        try:
            filename = cached_program_filename(row, options, target)
        finally:
            compiler._get_arch_for_options_for_nvrtc = selector
    arguments = dict(source=row["source"], options=options, arch=target, filename=filename,
                     name_expressions=metadata.get("name_expressions"),
                     cache_in_memory=False, method=None)
    (blob, mapping), timing = compile_request(arguments, target, compiler_identity())
    if not isinstance(blob, bytes) or blob[:4] != b"\x7fELF":
        raise ValueError("AOT worker did not produce an ELF cubin")
    shipped = {key: value for key, value in row.items() if key != "source"}
    return {**shipped, "arch": arch, "cubin_sha256": hashlib.sha256(blob).hexdigest(),
            "cubin_xz_base64": base64.b64encode(lzma.compress(blob)).decode("ascii"),
            "name_mapping": mapping, "nvrtc_banner": nvrtc_banner(),
            "effective_compile": {"options": list(options), "filename": filename},
            "compile_seconds": timing["seconds"]}


def _compile_job(job):
    return compile_entry(*job)


def build(output, recipes, *, architectures=ARCHITECTURES, workers=None, base_pack=None):
    identity = compiler_identity()
    rows = {}
    for catalog in recipes:
        # Recipes are compiler-bound: include layouts from CUDA 13 cannot
        # silently stand in for CUDA 12. Record both families separately.
        for entry in read(catalog, identity):
            row = recipe(entry)
            key = (row["module_key"], tuple(row["options"]),
                   str(row.get("compile_request")))
            rows.setdefault(key, row)
    if not rows:
        raise ValueError("the compiler recipe inventory is empty")
    selected = Pack(base_pack, identity=identity) if base_pack else None
    if selected is not None and selected.reason:
        raise ValueError(f"base pack cannot be reused: {selected.reason}")
    retained = [] if selected is None else list(selected.rows)
    jobs = []
    reused = 0
    for row in rows.values():
        for arch in architectures:
            existing = None
            if selected is not None:
                existing, _ = selected.select(row["module_key"], row["source"], row["options"],
                                              arch, metadata=row.get("compile_request"))
            if existing is None:
                jobs.append((row, arch))
            else:
                if selected.image(existing) is None:
                    raise ValueError("base pack image failed its digest")
                reused += 1
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    count = min(len(jobs), workers or max(1, cores))
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    compiled = []
    started = time.perf_counter()
    if jobs:
        with concurrent.futures.ProcessPoolExecutor(
                count, mp_context=multiprocessing.get_context("spawn")) as pool:
            compiled = list(pool.map(_compile_job, jobs, chunksize=1))
    write(output, identity, [*retained, *compiled])
    return {"modules": len(rows), "entries": len(retained) + len(compiled),
            "compiled": len(compiled), "reused": reused, "architectures": list(architectures),
            "workers": count, "wall_seconds": time.perf_counter() - started}
