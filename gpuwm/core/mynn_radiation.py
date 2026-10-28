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
#: WRF's merge thresholds (module_radiation_driver.F:1403-1429): MYNN water
#: is added where the grid-scale condensate is below these and the carried
#: MYNN fraction is above the last.
MERGE_QC_BELOW = 1.0e-6
MERGE_QI_BELOW = 1.0e-8
MERGE_CLDFRA_BL_ABOVE = 0.001


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
    bl_mynn_version: str = "wrf_461",
    temperature=None,
):
    """Apply ``module_radiation_driver.F:1403-1429`` in place.

    ``bl_mynn_version = "gsd_41"`` applies that generation's merge instead
    (:func:`_merge_gsd41`, which needs the layer ``temperature``), so every
    radiation coupling hands its scheme the cloud the legacy RRTMG adapter
    hands its own (kernels/rrtmg_legacy_adapter.cu ``rla_mynn_gsd41``).

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

    if bl_mynn_version == "gsd_41":
        return _merge_gsd41(qc, qi, cldfra, qc_bl=qc_bl, cldfra_bl=cldfra_bl,
                            temperature=temperature, itimestep=itimestep, xp=xp)
    # WRF applies the mass merge on timestep one too.  Only the fraction
    # replacement is delayed until the carried field has a previous interval.
    cloudy_bl = cldfra_bl > real(MERGE_CLDFRA_BL_ABOVE)
    if cldfra is not None and itimestep != 1:
        cldfra[...] = cldfra_bl
    qc[...] = xp.where(
        (qc < real(MERGE_QC_BELOW)) & cloudy_bl, qc + qc_bl, qc)
    qi[...] = xp.where(
        (qi < real(MERGE_QI_BELOW)) & cloudy_bl, qi + qi_bl, qi)
    return qc, qi, cldfra


def _gsd41_split(qc, qi, qc_bl, cldfra_bl, temperature, xp):
    """Where the GSD MYNN v4.1 merge adds water, and how much of each phase.

    NOAA-EMC WRF 3.9 module_radiation_driver.F:1256-1303: in-cloud QC_BL
    times CLDFRA_BL goes in only where neither resolved phase is present
    (qc < 1e-6 and qi < 1e-8) and CLDFRA_BL > 0.001, split liquid/ice by
    MIN(1, MAX(0, (T-254)/15)).  Each product is rounded to FP32 in the
    adapter kernel's order: ``(QC_BL*liq)*CLDFRA_BL``.
    """

    if temperature is None:
        raise ValueError("the gsd_41 MYNN radiation merge needs the layer temperature")
    if temperature.shape != qc.shape:
        raise ValueError("the gsd_41 MYNN radiation merge needs temperature on the cloud fields' shape")
    real = qc.dtype.type
    where = ((qc < real(MERGE_QC_BELOW)) & (qi < real(MERGE_QI_BELOW))
             & (cldfra_bl > real(MERGE_CLDFRA_BL_ABOVE)))
    liquid = xp.clip((temperature.astype(qc.dtype, copy=False) - real(254.0)) / real(15.0),
                     real(0.0), real(1.0))
    add_l = xp.where(where, (qc_bl * liquid) * cldfra_bl, real(0.0))
    add_i = xp.where(where, (qc_bl * (real(1.0) - liquid)) * cldfra_bl, real(0.0))
    return where, add_l, add_i


def _merge_gsd41(qc, qi, cldfra, *, qc_bl, cldfra_bl, temperature, itimestep, xp):
    """The GSD MYNN v4.1 merge (``rla_mynn_gsd41``) on any radiation coupling.

    On the first model step the fraction takes CLDFRA_BL only where it
    exceeds 0.001; after it, everywhere.  The mass goes in by
    :func:`_gsd41_split`.  Ported from the legacy RRTMG adapter so that
    radiation choice no longer decides whether a gsd_41 run's subgrid cloud
    is 1/CLDFRA_BL times too thick (U-E4).
    """

    real = qc.dtype.type
    where, add_l, add_i = _gsd41_split(qc, qi, qc_bl, cldfra_bl, temperature, xp)
    if cldfra is not None:
        if itimestep == 1:
            cldfra[...] = xp.where(cldfra_bl > real(MERGE_CLDFRA_BL_ABOVE), cldfra_bl, cldfra)
        else:
            cldfra[...] = cldfra_bl
    qc[...] = xp.where(where, qc + add_l, qc)
    qi[...] = xp.where(where, qi + add_i, qi)
    return qc, qi, cldfra


def mynn_bl_cloud_supplied(
    qc,
    qi,
    *,
    qc_bl=None,
    qi_bl=None,
    cldfra_bl=None,
    bl_pbl_physics: int,
    icloud_bl: int,
    bl_mynn_version: str = "wrf_461",
    temperature=None,
):
    """Where :func:`merge_mynn_bl_clouds` adds MYNN water: ``(liquid, ice)``.

    Under ``bl_mynn_version = "gsd_41"`` the layers are the ones that
    generation's merge gives nonzero liquid or ice (:func:`_gsd41_split`).

    Each is a boolean array of the layers the merge gives nonzero QC_BL or
    QI_BL, or ``None`` when the merge is off.  Call it on the grid-scale
    ``qc``/``qi`` BEFORE the merge, which writes them in place.

    The merge adds MYNN's water wherever the grid-scale condensate is below
    WRF's threshold, 1e-6 kg/kg liquid and 1e-8 ice.  A microphysics scheme
    sizes only its own cloud, so below that threshold its radius is either
    its no-cloud background (2.49 um liquid for Thompson, 2.51 um for NSSL)
    or the size it gives a trace of its own condensate (Thompson sizes
    liquid from 1e-12 kg m-3 and clamps it at 2.51 um, and sizes trace ice
    from that ice alone, often above WRF's 5 um bound).  Neither is a size
    for MYNN's water, and any grid-scale condensate beside it is below the
    merge threshold.  The RRTMG couplings therefore
    size these layers the way WRF sizes a cloudy layer the scheme left
    unsized (:func:`gpuwm.core.rrtmgp.cloudy_background_radii`).
    """

    if not mynn_bl_cloud_active(bl_pbl_physics, icloud_bl):
        return None, None
    if qc_bl is None or qi_bl is None or cldfra_bl is None:
        raise ValueError(
            "MYNN radiation coupling requires QC_BL, QI_BL, and CLDFRA_BL")
    arrays = (qc, qi, qc_bl, qi_bl, cldfra_bl)
    if any(array.shape != qc.shape for array in arrays[1:]):
        raise ValueError("MYNN radiation cloud fields must share one shape")
    real = qc.dtype.type
    if bl_mynn_version == "gsd_41":
        if any(hasattr(array, "__cuda_array_interface__") for array in arrays):
            import cupy as xp
        else:
            xp = np
        where, add_l, add_i = _gsd41_split(qc, qi, qc_bl, cldfra_bl, temperature, xp)
        return where & (add_l > real(0.0)), where & (add_i > real(0.0))
    cloudy_bl = cldfra_bl > real(MERGE_CLDFRA_BL_ABOVE)
    liquid = (qc < real(MERGE_QC_BELOW)) & cloudy_bl & (qc_bl > real(0.0))
    ice = (qi < real(MERGE_QI_BELOW)) & cloudy_bl & (qi_bl > real(0.0))
    return liquid, ice


__all__ = [
    "MERGE_CLDFRA_BL_ABOVE",
    "MERGE_QC_BELOW",
    "MERGE_QI_BELOW",
    "MYNN_PBL_PHYSICS",
    "merge_mynn_bl_clouds",
    "mynn_bl_cloud_active",
    "mynn_bl_cloud_supplied",
    "wrf_itimestep",
]
