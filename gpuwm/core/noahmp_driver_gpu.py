"""CUDA wrappers for the Noah-MP driver cold start.

One thread per column.  The device layout is the flat slot layout documented
at the top of ``gpuwm/core/kernels/noahmp_driver.cu``, and
:func:`pack_snow_init` / :func:`pack_noahmp_init` are the only place it is
constructed, so the test and any future caller cannot disagree about it.

These wrappers are a validation surface for the port, not a runtime path:
Noah-MP is not dispatchable and ``sf_surface_physics=4`` stays blocked.

``noahmp_driver.cu`` is compiled **after** ``noahmp_leaves.cu``.  The
supercooled-liquid guess at 2095-2096 needs glibc 2.39's ``powf`` on the
device, and there must be exactly one transcription of it: two copies could
drift and only one of them would be audited against ``glibc-libm-fp32.csv``.
Composing the two sources rather than duplicating the tables keeps that single
copy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from gpuwm.core.kernels import _preamble

_KDIR = Path(__file__).resolve().parent / "kernels"

# The glibc transcendental block this group borrows lives in noahmp_leaves.cu.
_LIBM_SOURCE = _KDIR / "noahmp_leaves.cu"
_DRIVER_SOURCE = _KDIR / "noahmp_driver.cu"

__all__ = [
    "NSNOW",
    "NSOIL_MAX",
    "NLAY_MAX",
    "SI_IX_STRIDE",
    "SI_IN_STRIDE",
    "SI_OUT_STRIDE",
    "NI_IX_STRIDE",
    "NI_IN_STRIDE",
    "NI_OUT_STRIDE",
    "SI_IN",
    "SI_OUT",
    "NI_IX",
    "NI_IN",
    "NI_OUT",
    "driver_source",
    "driver_module",
    "run_snow_init",
    "run_noahmp_init",
]

NSNOW = 3
NSOIL_MAX = 9
NLAY_MAX = NSNOW + NSOIL_MAX

_THREADS = 64

# --- SNOW_INIT slots -------------------------------------------------------
SI_IX_STRIDE = 2
SI_IX = {"nsnow": 0, "nsoil": 1}

SI_IN = {
    "swe": 0,
    "tgxy": 1,
    "snodep": 2,
    "zsoil": 3,
    "zsnsoxy": 3 + NSOIL_MAX,
    "tsnoxy": 3 + NSOIL_MAX + NLAY_MAX,
    "snicexy": 3 + NSOIL_MAX + NLAY_MAX + NSNOW,
    "snliqxy": 3 + NSOIL_MAX + NLAY_MAX + 2 * NSNOW,
}
SI_IN_STRIDE = 3 + NSOIL_MAX + NLAY_MAX + 3 * NSNOW

SI_OUT = {
    "isnowxy": 0,
    "zsnsoxy": 1,
    "tsnoxy": 1 + NLAY_MAX,
    "snicexy": 1 + NLAY_MAX + NSNOW,
    "snliqxy": 1 + NLAY_MAX + 2 * NSNOW,
}
SI_OUT_STRIDE = 1 + NLAY_MAX + 3 * NSNOW

# --- NOAHMP_INIT slots -----------------------------------------------------
NI_IX = {
    "nsoil": 0, "nsnow": 1, "fndsnowh": 2, "vegtyp": 3, "cropcat": 4,
    "sf_urban_physics": 5, "isice": 6, "isurban": 7, "iswater": 8,
    "isbarren": 9, "lcz": 10,
}
NI_IX_STRIDE = 10 + 11

NI_IN = {
    "xice": 0, "tsk": 1, "lai": 2, "bexp": 3, "smcmax": 4, "psisat": 5,
    "sla": 6, "sla_natural": 7, "snow": 8, "snowh": 9,
    "dzs": 10,
    "tslb": 10 + NSOIL_MAX,
    "smois": 10 + 2 * NSOIL_MAX,
    "zsnsoxy": 10 + 3 * NSOIL_MAX,
    "tsnoxy": 10 + 3 * NSOIL_MAX + NLAY_MAX,
    "snicexy": 10 + 3 * NSOIL_MAX + NLAY_MAX + NSNOW,
    "snliqxy": 10 + 3 * NSOIL_MAX + NLAY_MAX + 2 * NSNOW,
}
NI_IN_STRIDE = 10 + 3 * NSOIL_MAX + NLAY_MAX + 3 * NSNOW

_NI_SCALAR_OUT = (
    "snow", "snowh", "canwat", "tv", "tg", "canice", "canliq", "eah", "tah",
    "cm", "ch", "fwet", "sneqvo", "albold", "qsnow", "qrain", "wslake",
    "zwt", "wa", "wt", "lai", "xsai", "lfmass", "rtmass", "stmass", "wood",
    "stblcp", "fastcp", "grain", "gdd", "t2mv", "t2mb", "chstar", "qtdrain",
    "isnow", "cropcat",
)
NI_OUT = {name: i for i, name in enumerate(_NI_SCALAR_OUT)}
_base = len(_NI_SCALAR_OUT)
NI_OUT.update({
    "tslb": _base,
    "smois": _base + NSOIL_MAX,
    "sh2o": _base + 2 * NSOIL_MAX,
    "zsoil": _base + 3 * NSOIL_MAX,
    "zsnso": _base + 4 * NSOIL_MAX,
    "tsno": _base + 4 * NSOIL_MAX + NLAY_MAX,
    "snice": _base + 4 * NSOIL_MAX + NLAY_MAX + NSNOW,
    "snliq": _base + 4 * NSOIL_MAX + NLAY_MAX + 2 * NSNOW,
})
NI_OUT_STRIDE = _base + 4 * NSOIL_MAX + NLAY_MAX + 3 * NSNOW


def driver_source() -> str:
    """The exact translation unit the driver kernels are compiled from."""
    return (_preamble()
            + _LIBM_SOURCE.read_text(encoding="ascii")
            + _DRIVER_SOURCE.read_text(encoding="ascii"))


@lru_cache(maxsize=None)
def _module(options: tuple[str, ...], source: str | None):
    import cupy as cp

    module = cp.RawModule(code=source if source is not None else driver_source(),
                          options=options)
    module.compile()
    return module


def driver_module(options: tuple[str, ...] = ("-std=c++17",),
                  source: str | None = None):
    """Compile (once per option set) the composed driver translation unit.

    ``source`` exists so a negative control can compile a perturbed copy and
    show the gate can fail; leave it ``None`` for the real thing.
    """
    return _module(tuple(options), source)


def _launch(name, host_x, host_ix, out_stride, module):
    import cupy as cp

    ncase = host_x.shape[0]
    device_x = cp.asarray(np.ascontiguousarray(host_x, dtype=np.float32))
    device_ix = cp.asarray(np.ascontiguousarray(host_ix, dtype=np.int32))
    device_y = cp.zeros((ncase, out_stride), dtype=cp.float32)
    kernel = (module or driver_module()).get_function(name)
    blocks = (ncase + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (device_x, device_ix, device_y, np.int32(ncase)))
    return device_y


def run_snow_init(x, ix, module=None):
    """Run ``SNOW_INIT`` over a batch of columns.

    ``x`` is ``(ncase, SI_IN_STRIDE)`` FP32 and ``ix`` is
    ``(ncase, SI_IX_STRIDE)`` int32.  Returns ``(ncase, SI_OUT_STRIDE)`` FP32.
    """
    host_x = np.asarray(x, dtype=np.float32)
    host_ix = np.asarray(ix, dtype=np.int32)
    if host_x.ndim != 2 or host_x.shape[1] != SI_IN_STRIDE:
        raise ValueError(f"x must be (ncase, {SI_IN_STRIDE}), got "
                         f"{host_x.shape}")
    if host_ix.shape != (host_x.shape[0], SI_IX_STRIDE):
        raise ValueError(f"ix must be ({host_x.shape[0]}, {SI_IX_STRIDE}), "
                         f"got {host_ix.shape}")
    return _launch("noahmp_driver_snow_init", host_x, host_ix, SI_OUT_STRIDE,
                   module)


def run_noahmp_init(x, ix, module=None):
    """Run ``NOAHMP_INIT`` over a batch of columns.

    ``x`` is ``(ncase, NI_IN_STRIDE)`` FP32 and ``ix`` is
    ``(ncase, NI_IX_STRIDE)`` int32.  Returns ``(ncase, NI_OUT_STRIDE)`` FP32.
    """
    host_x = np.asarray(x, dtype=np.float32)
    host_ix = np.asarray(ix, dtype=np.int32)
    if host_x.ndim != 2 or host_x.shape[1] != NI_IN_STRIDE:
        raise ValueError(f"x must be (ncase, {NI_IN_STRIDE}), got "
                         f"{host_x.shape}")
    if host_ix.shape != (host_x.shape[0], NI_IX_STRIDE):
        raise ValueError(f"ix must be ({host_x.shape[0]}, {NI_IX_STRIDE}), "
                         f"got {host_ix.shape}")
    return _launch("noahmp_driver_noahmp_init", host_x, host_ix,
                   NI_OUT_STRIDE, module)
