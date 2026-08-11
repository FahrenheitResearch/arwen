"""WDM6 double-moment warm-rain microphysics (WRF ``mp_physics=16``).

WDM6 is WSM6's ice half carried unchanged beside a DOUBLE-MOMENT warm rain:
cloud droplet number ``nc``, rain number ``nr`` and a prognostic CCN
reservoir ``nn`` are transported and predicted, rain's PSD becomes a gamma
with shape ``mu = 1`` and a diagnosed intercept, and autoconversion,
accretion, self-collection, evaporation and freezing all move number as
well as mass.

The numerical authority is the byte-frozen WRF v4.6.1
``phys/module_mp_wdm6.F`` in the 1974 reference bundle; the CUDA kernel
(``gpuwm/core/kernels/wdm6.cu``) is transcribed from it line by line and
carries the file:line citations.  The host coefficient block is
``gpuwm/core/wdm6_constants.py``, which imports WSM6's ``_rgmma`` and
``rimed_ice_constants`` rather than copying them, because ``wdm6init``'s
``hail_opt`` arm (:2096-2108) sets exactly the five constants
``mp_wsm6_init`` sets.

WHAT THE ADAPTER OWES THE SCHEME THAT WSM6'S DOES NOT

* ``xland``.  WDM6 picks its autoconversion threshold per column from the
  land/sea mask -- ``qc0`` (maritime, ``xncr0 = 5e7``) where ``slmsk == 2``
  and ``qc1`` (continental, ``xncr1 = 5e8``) elsewhere
  (module_mp_wdm6.F:607-614; the driver's own comment at
  module_microphysics_driver.F:2495 spells the convention "1: land,
  2: water").  gpuwm keeps XLAND on the physics driver's surface field
  dict, so :func:`apply` reads it from there and REFUSES rather than
  substituting a mask: guessing "all land" would silently move the
  continental threshold onto an ocean domain, which is a factor of ten in
  the autoconversion trigger.
* ``nn`` initialisation.  WRF fills the whole CCN array with the namelist
  ``ccn_conc`` on the first time step (:220-227), and ``ccn0`` reaches
  nothing else in the module -- ``wdm62D`` and ``wdm6init`` both declare it
  ``intent(in)`` and never read it.  gpuwm performs that fill once at state
  allocation from ``cfg.wdm6_ccn_conc`` (``gpuwm/core/state.py``), which is
  the NSSL ``qnn`` precedent and is exactly equivalent for a cold start.

EVIDENCE.  This is an implemented-unverified integration port.  No oracle
comparison against the WRF Fortran has been run: the campaign that produced
Shin-Hong's and Grell-Freitas's ULP numbers is the declared next stage for
this scheme too.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.core.wdm6_constants import (WDM6_DEEP_KMAX, WDM6_NUMBER_SPECIES,
                                       WDM6_SHALLOW_KMAX,
                                       WDM6_VERTICAL_LEVEL_BOUNDS,
                                       wdm6_level_tier)

_COLUMN_TPB = 32
# The ladder itself lives in wdm6_constants (a CuPy-free leaf) so the memory
# preflight can price a WDM6 configuration on a host with no device; these
# names are the adapter's local spelling of it.
_SHALLOW_KMAX = WDM6_SHALLOW_KMAX
_KMAX = WDM6_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WDM6_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wdm6_level_tier


def launch_wdm6(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr,
                rho, pii, pressure, dz, xland,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0) -> None:
    """Launch one WDM6 call over contiguous FP32 ``(nz,ny,nx)`` fields."""
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WDM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg,
        "nn": nn, "nc": nc, "nr": nr,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {
            "xland": xland,
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel = (get_kernel("wdm6", "wdm6_column")
              if capacity == _SHALLOW_KMAX else
              get_kernel_int_defines(
                  "wdm6", "wdm6_column", (("WDM6_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
         rho, pressure, pii, dz, xland,
         rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
         effc, effi, effs, DTYPE(dt), np.int32(hail_opt),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def column_land_mask(state: DomainState) -> cp.ndarray:
    """WRF XLAND for the microphysics call, or a refusal naming the gap.

    Separated from :func:`apply` so the requirement is testable on its own
    and so the refusal text lives beside the reason.  WDM6 is the first
    gpuwm microphysics scheme whose PROCESS RATES read a surface field, and
    a fabricated mask is not a degraded answer -- it is the wrong
    autoconversion threshold everywhere it is wrong.
    """
    driver = getattr(state, "physics", None)
    fields = getattr(driver, "fields", None)
    xland = None if fields is None else fields.get("xland")
    if xland is None:
        raise ValueError(
            "mp_physics=16 (WDM6) needs WRF's XLAND land/sea mask for its "
            "per-column autoconversion threshold (module_mp_wdm6.F:607-614) "
            "and the physics driver carries none; refusing to substitute an "
            "all-land or all-water mask. Construct the domain through "
            "gpuwm.core.physics.initialize_physics, which builds XLAND from "
            "the landmask for every configuration.")
    return xland


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WDM6, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nn", "nc", "nr", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=16 state lacks WDM6 fields: "
                         + ", ".join(missing))
    xland = column_land_mask(state)
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "wdm6_theta")
    rho = state.scratch((nz, ny, nx), "wdm6_rho")
    pii = state.scratch((nz, ny, nx), "wdm6_pii")
    dz = state.scratch((nz, ny, nx), "wdm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wdm6_z8w")
    theta[...] = thb + state.thp
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")

    from gpuwm.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    launch_wdm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        state.nn, state.nc, state.nr,
        rho, pii, state.p, dz, cp.ascontiguousarray(xland),
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wdm6_hail_opt)
    if refl_10cm_due:
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


__all__ = ["WDM6_NUMBER_SPECIES", "apply", "column_land_mask", "launch_wdm6"]
