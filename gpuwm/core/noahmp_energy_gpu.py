"""CUDA wrapper for the Noah-MP ENERGY assembly.

One thread per column.  The device layout is the flat slot layout declared in
``kernels/noahmp_energy.cu``; the host packs it straight out of
``gpuwm/data/noahmp/oracle/noahmp-energy.csv``, so the fixture is replayed slot
for slot with no repacking on either side.

Scope, stated once so it is not overclaimed elsewhere: this kernel is
**ENERGY's own arithmetic**, not the whole column.  The six subsystems ENERGY
composes each have a device port in a sibling ``.cu`` already, and running them
inside this kernel is blocked by three things in other lanes' files -- three
copies of the device glibc libm with the same symbol names, ``TSNOSOI``/
``PHASECHANGE`` exposed only as ``__global__`` entry points, and per-lane flat
fixture packings in place of physical argument lists.  Their results are
therefore fed in from the same pinned fixture.  See the kernel's header comment.

``noahmp_energy.cu`` is compiled **after** ``noahmp_leaves.cu``: ENERGY needs
glibc 2.39's ``powf`` and ``expf`` on the device, and there must be exactly one
transcription of those in the tree.  ``expm1f``/``tanhf`` are new here because
ENERGY's ``FSNO`` is the only ``TANH`` in Noah-MP.

This is a validation surface, not a runtime path: Noah-MP is not dispatchable
and ``sf_surface_physics=4`` stays blocked.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from gpuwm.core.kernels import _preamble

_KDIR = Path(__file__).resolve().parent / "kernels"
_LIBM_SOURCE = _KDIR / "noahmp_leaves.cu"
_ENERGY_SOURCE = _KDIR / "noahmp_energy.cu"

NSOIL = 4
N_IN = 67
N_INT = 2
N_OUT = 27
_THREADS = 64

#: Input slot names, in device order.  Mirrors the ``E_*`` defines in the .cu.
IN_SLOTS = (
    "UU", "VV", "ELAI", "ESAI", "SNOWH", "SNEQV", "TG", "TV", "SFCPRS",
    "LWDN", "FVEG", "ZREF", "DT", "ACC_SSOIL",
    "MFSNO", "SCFFAC", "Z0SNO", "Z0MVT", "HVT", "EG", "SNOW_EMIS",
) + tuple(f"SH2O_{k}" for k in range(1, NSOIL + 1)) \
  + tuple(f"SMCWLT_{k}" for k in range(1, NSOIL + 1)) \
  + tuple(f"SMCREF_{k}" for k in range(1, NSOIL + 1)) \
  + tuple(f"DZSNSO_{k}" for k in range(1, NSOIL + 1)) \
  + tuple(f"ZSOIL_{k}" for k in range(1, NSOIL + 1)) \
  + ("IRC", "IRG", "IRB", "SHC", "SHG", "SHB", "EVC", "EVG", "EVB", "TR",
     "GHV", "GHB", "TGV", "TGB", "T2MV", "T2MB", "CHV", "CHB", "EAH",
     "QSFC", "Q2V", "Q2B", "PAHV", "PAHG", "PAHB", "TV_POST")

#: Output slot names, in device order.  Mirrors the ``O_*`` defines.
OUT_SLOTS = (
    ("FSNO", "Z0WRF", "BTRAN")
    + tuple(f"BTRANI_{k}" for k in range(1, NSOIL + 1))
    + ("LATHEAV", "LATHEAG", "FROZEN_CANOPY", "FROZEN_GROUND",
       "FIRA", "FSH", "FGEV", "SSOIL", "FCEV", "FCTR", "PAH", "TG", "T2M",
       "TS", "CH", "Q1", "Q2E", "EMISSI", "TRAD", "ACC_SSOIL")
)

assert len(IN_SLOTS) == N_IN, (len(IN_SLOTS), N_IN)
assert len(OUT_SLOTS) == N_OUT, (len(OUT_SLOTS), N_OUT)


def energy_source() -> str:
    """The exact translation unit the ENERGY kernels are compiled from."""
    return (_preamble()
            + _LIBM_SOURCE.read_text(encoding="ascii")
            + _ENERGY_SOURCE.read_text(encoding="ascii"))


@lru_cache(maxsize=None)
def _module(options: tuple[str, ...]):
    import cupy as cp

    module = cp.RawModule(code=energy_source(), options=options)
    module.compile()
    return module


def energy_module(options: tuple[str, ...] = ("-std=c++17",)):
    """Compile (once per option set) the composed ENERGY translation unit."""
    return _module(tuple(options))


def evaluate_energy(x, ix):
    """Run the ENERGY assembly kernel over a batch of columns.

    ``x`` is ``(ncolumn, N_IN)`` FP32, ``ix`` is ``(ncolumn, N_INT)`` int32.
    Returns a host array of shape ``(ncolumn, N_OUT)``, FP32.
    """
    import cupy as cp

    host_x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    host_ix = np.ascontiguousarray(np.asarray(ix, dtype=np.int32))
    if host_x.ndim != 2 or host_x.shape[1] != N_IN:
        raise ValueError(f"expected x with shape (n, {N_IN}), got {host_x.shape}")
    ncolumn = host_x.shape[0]
    if host_ix.shape != (ncolumn, N_INT):
        raise ValueError(
            f"expected ix with shape ({ncolumn}, {N_INT}), got {host_ix.shape}")

    device_y = cp.zeros((ncolumn, N_OUT), dtype=cp.float32)
    kernel = energy_module().get_function("noahmp_energy_assembly")
    blocks = (ncolumn + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (cp.asarray(host_x), cp.asarray(host_ix), device_y,
            np.int32(ncolumn)))
    return cp.asnumpy(device_y)


def evaluate_tanhf(x):
    """Device ``tanhf``/``expm1f`` over a batch, for the libm parity probe."""
    import cupy as cp

    host_x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
    n = host_x.size
    tanh_out = cp.zeros(n, dtype=cp.float32)
    expm1_out = cp.zeros(n, dtype=cp.float32)
    kernel = energy_module().get_function("noahmp_energy_tanhf_probe")
    blocks = (n + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (cp.asarray(host_x), tanh_out, expm1_out, np.int32(n)))
    return cp.asnumpy(tanh_out), cp.asnumpy(expm1_out)
