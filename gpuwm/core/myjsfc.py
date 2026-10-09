"""MYJ (Eta similarity) surface layer, ``sf_sfclay_physics = 2``.

CUDA launcher around :mod:`gpuwm.core.kernels.myjsfc`, the device port of
the byte-frozen WRF v4.6.1 ``phys/module_sf_myjsfc.F`` (the float32 CPU
authority ``gpuwm.verify.myj_ref.np_myjsfc_column`` is its tolerance
twin).  Both interpolate the SAME similarity tables: they are built once on
the host by :func:`gpuwm.core.myjsfc_tables.build_psi_tables` (a
transcription of ``MYJSFCINIT``, phys/module_sf_myjsfc.F:1174-1299) and
uploaded here.

The scheme is the Eta surface layer WRF pairs with the MYJ PBL and with
nothing else (phys/module_physics_init.F:3770-3772 fatals a MYJ PBL whose
surface layer did not set ``isfc = 2``).  gpuwm enforces the same pairing in
:func:`gpuwm.config.validate_myj_pairing`.

Conformance status: BIT-IDENTICAL to WRF v4.6.1's MYJSFC and MYJSFCINIT
on the 224-column oracle (tools/myjsfc_wrf461_oracle,
tests/test_myjsfc_wrf461_parity.py) under the strict build and under
default arithmetic; registry maturity stays implemented-unverified because
no matched forecast trajectory exists.  ``surface["ht"]`` is WRF's HT:
MYJSFC's interface heights start at the terrain height (ZINT(KTE+1)=HT,
:165), PBLH and ZSL are differences of them, and the float32 words depend
on it.
"""

from __future__ import annotations

from functools import lru_cache
from gpuwm.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.myjsfc_tables import build_psi_tables
# WRF INOUT surface state and pure outputs, in kernel argument order.  The
# tables live in the runtime-free inventory module so the VRAM estimator
# prices the same names this launcher binds without importing cupy.
from gpuwm.core.physics_inventory import (MYJ_SFCLAY_INOUT,
                                          MYJ_SFCLAY_OUTPUTS)
from gpuwm.core.state import DTYPE

_TPB = 128

#: Read-only column inputs, in kernel argument order.
_COLUMN_INPUTS = ("dz", "tke")
#: Read-only surface inputs, in kernel argument order.
_SURFACE_INPUTS = ("u1", "v1", "t1", "th1", "qv1", "qc1", "p1", "psfc",
                   "tsk", "xland", "mavail", "z0base", "ht")


@cuda_cache(maxsize=None, ready=True)
def _device_tables():
    """Upload the MYJSFCINIT tables once per process.

    The scalars ride along as Python floats; the kernel takes them by
    value.  ``lru_cache`` keeps the four 10001-word arrays resident for the
    life of the process rather than re-uploading 160 kB every call.
    """
    tables = build_psi_tables()
    arrays = tuple(cp.asarray(tables[name], dtype=DTYPE)
                   for name in ("psim1", "psih1", "psim2", "psih2"))
    scalars = tuple(DTYPE(tables[name]) for name in (
        "dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2", "ztmax2",
        "fh01", "fh02"))
    return arrays, scalars


def launch_myj_sfclay(columns, surface, state, outputs, *,
                      itimestep: int) -> None:
    """Run one MYJ surface-layer step in place.

    ``columns`` supplies the full bottom-up ``(nz, ny, nx)`` ``dz`` and
    ``tke`` arrays the PBLH scan needs (module_sf_myjsfc.F:263-277);
    ``surface`` the lowest-model-level and static fields; ``state`` the
    nine WRF INOUT values; ``outputs`` the pure outputs.  Every array is
    contiguous float32 with the same ``(ny, nx)`` surface shape.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    if nz < 2:
        raise ValueError("the MYJ surface layer needs at least two levels: "
                         "its PBLH scan walks the TKE column")
    for name in _COLUMN_INPUTS:
        array = columns[name]
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                f"launch_myj_sfclay needs a contiguous float32 "
                f"({nz}, {shape[0]}, {shape[1]}) {name!r} column")
    named = (
        [columns[name] for name in _COLUMN_INPUTS]
        + [surface[name] for name in _SURFACE_INPUTS]
        + [state[name] for name in MYJ_SFCLAY_INOUT]
        + [outputs[name] for name in MYJ_SFCLAY_OUTPUTS])
    for array in named[len(_COLUMN_INPUTS):]:
        if (array.shape != shape or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError("launch_myj_sfclay requires same-shape "
                             "contiguous float32 surface arrays")
    arrays, scalars = _device_tables()
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("myjsfc", "myjsfc_column")
    kernel((blocks,), (_TPB,), tuple(named) + arrays + scalars
           + (np.int32(itimestep), np.int32(nz), np.int32(n)))
