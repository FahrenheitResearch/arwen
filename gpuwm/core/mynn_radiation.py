"""WRF v4.6.1 MYNN boundary-layer cloud merge for radiation.

``module_radiation_driver.F:1403-1429`` diagnoses grid-scale cloud fraction
first, then applies ``icloud_bl``.  After the first model timestep it replaces
the diagnosed fraction with the carried MYNN ``CLDFRA_BL`` field.  On every
timestep it adds ``QC_BL`` or ``QI_BL`` only where the corresponding
grid-scale condensate is below WRF's threshold and ``CLDFRA_BL > 0.001``.

Radiation runs before PBL in WRF and gpuwm, so the fields presented here are
the previous interval's carried MYNN state.  This module does not advance or
re-diagnose them.
"""

from __future__ import annotations

import numpy as np


MYNN_PBL_PHYSICS = 5


def mynn_bl_cloud_active(bl_pbl_physics: int, icloud_bl: int) -> bool:
    """Return the exact WRF gate for the MYNN radiation merge."""

    return int(bl_pbl_physics) == MYNN_PBL_PHYSICS and int(icloud_bl) > 0


def wrf_itimestep(elapsed_seconds: float, dt: float) -> int:
    """Recover WRF's one-based model timestep from gpuwm's carried clock."""

    return int(np.floor(float(elapsed_seconds) / float(dt) + 0.5)) + 1


def merge_mynn_bl_clouds(
    qc,
    qi,
    cldfra,
    *,
    qc_bl=None,
    qi_bl=None,
    cldfra_bl=None,
    bl_pbl_physics: int,
    icloud_bl: int,
    itimestep: int,
):
    """Apply ``module_radiation_driver.F:1403-1429`` in place.

    ``cldfra`` may be ``None`` for Dudhia, which consumes condensate mass but
    has no cloud-fraction input.  The non-MYNN return happens before inspecting
    any MYNN field and returns the original objects without a write; all
    non-MYNN radiation configurations are therefore byte-identical at this
    seam.
    """

    if not mynn_bl_cloud_active(bl_pbl_physics, icloud_bl):
        return qc, qi, cldfra
    if type(itimestep) is not int or itimestep < 1:
        raise ValueError("MYNN radiation itimestep must be a positive int")
    if qc_bl is None or qi_bl is None or cldfra_bl is None:
        raise ValueError(
            "MYNN radiation coupling requires QC_BL, QI_BL, and CLDFRA_BL")

    arrays = (qc, qi, qc_bl, qi_bl, cldfra_bl)
    shape = qc.shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("MYNN radiation cloud fields must share one shape")
    if cldfra is not None and cldfra.shape != shape:
        raise ValueError("MYNN and grid-scale cloud fractions must share shape")

    if any(hasattr(array, "__cuda_array_interface__") for array in arrays):
        import cupy as xp
    else:
        xp = np
    real = qc.dtype.type

    # WRF applies the mass merge on timestep one too.  Only the fraction
    # replacement is delayed until the carried field has a previous interval.
    cloudy_bl = cldfra_bl > real(0.001)
    if cldfra is not None and itimestep != 1:
        cldfra[...] = cldfra_bl
    qc[...] = xp.where(
        (qc < real(1.0e-6)) & cloudy_bl, qc + qc_bl, qc)
    qi[...] = xp.where(
        (qi < real(1.0e-8)) & cloudy_bl, qi + qi_bl, qi)
    return qc, qi, cldfra


__all__ = [
    "MYNN_PBL_PHYSICS",
    "merge_mynn_bl_clouds",
    "mynn_bl_cloud_active",
    "wrf_itimestep",
]
