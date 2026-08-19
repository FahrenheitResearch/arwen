"""Batched CUDA execution of the NOAHMP_GLACIER column.

The arithmetic lives in ``gpuwm/core/kernels/noahmp_glacier.cu`` (one
thread per glacier column, compiled after ``noahmp_leaves.cu`` for the
single glibc 2.39 device transcription -- see
``gpuwm/core/noahmp_kernel_sources.py``).  This module is the
runtime-facing half: it packs the per-column input dictionary the
runtime gathers, launches one kernel over all glacier columns, and
unpacks the result columns.

The paired host authority is :func:`evaluate_glacier_columns_on_host`,
which answers the identical input dictionary through
:func:`gpuwm.core.noahmp_glacier.noahmp_glacier`, the CPython
transcription -- the same two-implementation discipline every other
Noah-MP leaf carries, and what the GPU parity test compares bit for bit.

The Fortran aborts (``wrf_error_fatal``) in four places; the kernel
cannot, so each column reports an error code and this wrapper raises
:class:`gpuwm.core.noahmp_glacier.GlacierBalanceError` naming the first
offending column and the per-code counts.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from gpuwm.core.noahmp_glacier import GlacierBalanceError, noahmp_glacier

__all__ = [
    "GLACIER_INPUT_SCALARS",
    "GLACIER_OUTPUT_SCALARS",
    "evaluate_glacier_columns",
    "evaluate_glacier_columns_on_host",
    "pack_glacier_columns",
]

NSNOW = 3
NSOIL = 4
NLAYER = NSNOW + NSOIL
THREADS = 64

#: Per-column float input slots 0..20, in kernel order.
GLACIER_INPUT_SCALARS = (
    "cosz", "sfctmp", "sfcprs", "uu", "vv", "q2", "soldn", "prcp", "lwdn",
    "tbot", "zlvl", "qsnow", "sneqvo", "albold", "cm", "ch", "sneqv",
    "snowh", "tg", "tauss", "qsfc",
)

#: (name, width) of the packed float array slots following the scalars.
GLACIER_INPUT_ARRAYS = (
    ("ficeold", NSNOW), ("smc", NSOIL), ("sh2o", NSOIL),
    ("zsnso", NLAYER), ("stc", NLAYER), ("snice", NSNOW),
    ("snliq", NSNOW),
)

N_INPUT = len(GLACIER_INPUT_SCALARS) + sum(w for _, w in
                                           GLACIER_INPUT_ARRAYS)

#: Per-column float output slots 0..32, in kernel order.
GLACIER_OUTPUT_SCALARS = (
    "tg", "tauss", "qsfc", "qsnow", "sneqvo", "albold", "cm", "ch",
    "sneqv", "snowh", "fsa", "fsr", "fira", "fsh", "fgev", "ssoil",
    "trad", "edir", "runsrf", "runsub", "sag", "albedo", "qsnbot",
    "ponding", "ponding1", "ponding2", "t2m", "q2e", "emissi", "fpice",
    "ch2b", "qmelt", "eflxb",
)

GLACIER_OUTPUT_ARRAYS = (
    ("smc", NSOIL), ("sh2o", NSOIL), ("stc", NLAYER), ("zsnso", NLAYER),
    ("hcpct", NLAYER), ("snice", NSNOW), ("snliq", NSNOW),
)

N_OUTPUT = len(GLACIER_OUTPUT_SCALARS) + sum(w for _, w in
                                             GLACIER_OUTPUT_ARRAYS)

_ERROR_NAMES = {
    1: "emitted longwave <= 0 (module_sf_noahmp_glacier.F:500)",
    2: "ERRSW radiation budget (:3009-3016)",
    3: "ERRENG energy budget (:3018-3025)",
    4: "ERRWAT water budget (:3031-3045)",
}


def _require(inputs: dict, count: int) -> None:
    for name in GLACIER_INPUT_SCALARS:
        value = np.asarray(inputs[name])
        if value.shape != (count,):
            raise ValueError(
                f"glacier input {name!r} must have shape ({count},), got "
                f"{value.shape}")
    for name, width in GLACIER_INPUT_ARRAYS:
        value = np.asarray(inputs[name])
        if value.shape != (count, width):
            raise ValueError(
                f"glacier input {name!r} must have shape ({count}, {width}),"
                f" got {value.shape}")
    isnow = np.asarray(inputs["isnow"])
    if isnow.shape != (count,):
        raise ValueError(
            f"glacier input 'isnow' must have shape ({count},), got "
            f"{isnow.shape}")
    zsoil = np.asarray(inputs["zsoil"], dtype=np.float32)
    if zsoil.shape != (NSOIL,):
        raise ValueError(
            f"glacier input 'zsoil' must have shape ({NSOIL},), got "
            f"{zsoil.shape}")


def pack_glacier_columns(inputs: dict, count: int) -> np.ndarray:
    """The ``float32[count, 52]`` row block the kernel reads."""
    _require(inputs, count)
    packed = np.empty((count, N_INPUT), dtype=np.float32)
    base = 0
    for name in GLACIER_INPUT_SCALARS:
        packed[:, base] = np.asarray(inputs[name], dtype=np.float32)
        base += 1
    for name, width in GLACIER_INPUT_ARRAYS:
        packed[:, base:base + width] = np.asarray(inputs[name],
                                                  dtype=np.float32)
        base += width
    assert base == N_INPUT
    return packed


def _raise_on_errors(errors: np.ndarray) -> None:
    if not errors.any():
        return
    bad = np.nonzero(errors)[0]
    first = int(bad[0])
    code = int(errors[first])
    counts = {int(c): int((errors == c).sum())
              for c in np.unique(errors[bad])}
    raise GlacierBalanceError(
        f"NOAHMP_GLACIER batch: {bad.size} of {errors.size} columns "
        f"failed a WRF balance gate; first is batch column {first} with "
        f"{_ERROR_NAMES.get(code, f'code {code}')}; per-code counts "
        f"{counts}")


@lru_cache(maxsize=None)
def _module(options: tuple[str, ...]):
    """Compile (once per option set) the COMPOSED glacier translation
    unit -- noahmp_leaves.cu (the single glibc device transcription)
    followed by noahmp_glacier.cu, exactly as
    gpuwm/core/noahmp_kernel_sources.py declares it.  ``get_kernel``
    loads single files and must not be handed a fragment."""
    import cupy as cp

    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.core.noahmp_kernel_sources import translation_unit_source

    source = translation_unit_source("noahmp_glacier")
    module = cp.RawModule(code=source, options=options)
    module.compile()
    record_module("gpuwm.core.noahmp_glacier_gpu:glacier",
                  source=source, options=options, module=module)
    return module


def glacier_module(options: tuple[str, ...] = ("-std=c++17",)):
    """The compiled composed glacier unit (cached per option set)."""
    return _module(tuple(options))


def evaluate_glacier_columns(inputs: dict, count: int,
                             dt: float) -> dict[str, np.ndarray]:
    """Advance ``count`` glacier columns in one CUDA launch."""
    import cupy as cp

    packed = pack_glacier_columns(inputs, count)
    isnow = np.ascontiguousarray(np.asarray(inputs["isnow"],
                                            dtype=np.int32))
    zsoil = np.ascontiguousarray(np.asarray(inputs["zsoil"],
                                            dtype=np.float32))
    device_out = cp.empty(count * N_OUTPUT, dtype=cp.float32)
    device_io = cp.empty(count * 2, dtype=cp.int32)
    kernel = glacier_module().get_function("noahmp_glacier_column")
    blocks = (count + THREADS - 1) // THREADS
    kernel(
        (blocks,), (THREADS,),
        (cp.asarray(packed.ravel()), cp.asarray(isnow), cp.asarray(zsoil),
         np.float32(dt), device_out, device_io, np.int32(count)),
    )
    host = cp.asnumpy(device_out).reshape(count, N_OUTPUT)
    host_io = cp.asnumpy(device_io).reshape(count, 2)
    _raise_on_errors(host_io[:, 1])
    out: dict[str, np.ndarray] = {}
    base = 0
    for name in GLACIER_OUTPUT_SCALARS:
        out[name] = host[:, base].copy()
        base += 1
    for name, width in GLACIER_OUTPUT_ARRAYS:
        out[name] = host[:, base:base + width].copy()
        base += width
    assert base == N_OUTPUT
    out["isnow"] = host_io[:, 0].copy()
    return out


def evaluate_glacier_columns_on_host(inputs: dict, count: int,
                                     dt: float) -> dict[str, np.ndarray]:
    """The paired host authority: the identical batch through the CPython
    transcription, one column at a time."""
    _require(inputs, count)
    zsoil = np.asarray(inputs["zsoil"], dtype=np.float32)
    out: dict[str, np.ndarray] = {
        name: np.empty(count, dtype=np.float32)
        for name in GLACIER_OUTPUT_SCALARS}
    for name, width in GLACIER_OUTPUT_ARRAYS:
        out[name] = np.empty((count, width), dtype=np.float32)
    out["isnow"] = np.empty(count, dtype=np.int32)
    for i in range(count):
        result = noahmp_glacier(
            cosz=inputs["cosz"][i], nsnow=NSNOW, nsoil=NSOIL, dt=dt,
            sfctmp=inputs["sfctmp"][i], sfcprs=inputs["sfcprs"][i],
            uu=inputs["uu"][i], vv=inputs["vv"][i], q2=inputs["q2"][i],
            soldn=inputs["soldn"][i], prcp=inputs["prcp"][i],
            lwdn=inputs["lwdn"][i], tbot=inputs["tbot"][i],
            zlvl=inputs["zlvl"][i], ficeold=inputs["ficeold"][i],
            zsoil=zsoil, qsnow=inputs["qsnow"][i],
            sneqvo=inputs["sneqvo"][i], albold=inputs["albold"][i],
            cm=inputs["cm"][i], ch=inputs["ch"][i],
            isnow=int(inputs["isnow"][i]), sneqv=inputs["sneqv"][i],
            smc=inputs["smc"][i], zsnso=inputs["zsnso"][i],
            snowh=inputs["snowh"][i], snice=inputs["snice"][i],
            snliq=inputs["snliq"][i], tg=inputs["tg"][i],
            stc=inputs["stc"][i], sh2o=inputs["sh2o"][i],
            tauss=inputs["tauss"][i], qsfc=inputs["qsfc"][i])
        for name in GLACIER_OUTPUT_SCALARS:
            out[name][i] = getattr(result, name)
        for name, _w in GLACIER_OUTPUT_ARRAYS:
            out[name][i] = getattr(result, name)
        out["isnow"][i] = result.isnow
    return out
