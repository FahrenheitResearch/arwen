"""Morrison two-moment microphysics (WRF ``mp_physics=10``).

The CUDA pipeline stages independent level work around one sedimentation
thread per atmospheric column, which carries WRF's internal substeps.  The
public launcher accepts contiguous FP32 ``(nz, ny, nx)`` fields and updates
potential temperature, six water categories, and their moments in place.
:func:`apply` prepares WRF's microphysics inputs from
:class:`~gpuwm.core.state.DomainState`.

Transcription authority: local WRF v4.6.1
``phys/module_mp_morr_two_moment.F``: wrapper/API lines 563-925, column
scheme lines 929-4062, saturation helper lines 4066-4149.  Float64 mirror:
``gpuwm.verify.npref.np_morrison_column``.
"""

from __future__ import annotations

import numpy as np

import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel
from gpuwm.core.morrison_constants import rimed_ice_constants
from gpuwm.core.state import DTYPE, DomainState

_CELL_TPB = 64
_COLUMN_TPB = 32
_SHALLOW_KMAX = 64
_KMAX = 256


def launch_morrison(theta, qv, qc, qr, qi, qs, qg,
                    nc, nr, ni, ns, ng, rho, pii, pressure, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt: float,
                    *, effc=None, effr=None, effi=None, effs=None,
                    qrcuten=None, qscuten=None, qicuten=None,
                    morr_rimed_ice: int = 1,
                    _rhoa_scratch=None, _ice_to_snow_scratch=None) -> None:
    """Launch one WRF Morrison call over an FP32 column batch.

    Atmospheric and hydrometeor arrays are ``(nz, ny, nx)``.  Accumulated,
    per-call, and frozen-fraction precipitation fields are ``(ny, nx)``.
    All prognostic arguments through ``ng`` and all precipitation fields
    are updated in place.  ``rho`` is read-only and retained in the
    WRF-compatible API; WRF's Morrison implementation documents it as
    unused and diagnoses its own density from pressure and temperature
    (source line 589/1325).  The private scratch hook lets the state adapter
    reuse its persistent named density buffer without changing this public
    contract.  ``morr_rimed_ice`` follows WRF's scalar Registry option:
    1 = hail (default), 0 = graupel (Registry.EM_COMMON:2663-2666;
    module_mp_morr_two_moment.F:337-411).
    ``qrcuten``/``qscuten``/``qicuten`` are the optional all-or-none raw KF
    mass rates used by WRF to seed rain/snow/ice number moments before process
    calculations (module_mp_morr_two_moment.F:1327-1343).
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"Morrison fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds MORR_KMAX={_KMAX}")
    if nz < 2:
        raise ValueError("Morrison requires nz >= 2")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    rimed = rimed_ice_constants(morr_rimed_ice)
    for name, value in (
            ("qv", qv), ("qc", qc), ("qr", qr), ("qi", qi),
            ("qs", qs), ("qg", qg), ("nc", nc), ("nr", nr),
            ("ni", ni), ("ns", ns), ("ng", ng), ("rho", rho),
            ("pii", pii), ("pressure", pressure), ("dz", dz)):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    cu_values = (qrcuten, qscuten, qicuten)
    has_cu_tendencies = any(value is not None for value in cu_values)
    if has_cu_tendencies and not all(value is not None for value in cu_values):
        raise ValueError("qrcuten, qscuten, and qicuten must be supplied "
                         "together")
    if has_cu_tendencies:
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu_values):
            if (value.shape != shape or value.dtype != DTYPE
                    or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be contiguous float32 with "
                                 f"shape {shape}")
        qrcu, qscu, qicu = cu_values
    else:
        # The kernel gates every read with has_cu_tendencies.  Reusing a valid
        # pointer avoids allocating a full-volume zero placeholder.
        qrcu = qscu = qicu = rho
    surface_shape = (ny, nx)
    for name, value in (
            ("rainnc", rainnc), ("rainncv", rainncv),
            ("snownc", snownc), ("snowncv", snowncv),
            ("graupelnc", graupelnc), ("graupelncv", graupelncv),
            ("sr", sr)):
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, "
                             f"got {value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    if _rhoa_scratch is None:
        rhoa_scratch = cp.empty_like(rho)
    else:
        rhoa_scratch = _rhoa_scratch
        if (rhoa_scratch.shape != shape or rhoa_scratch.dtype != DTYPE
                or not rhoa_scratch.flags.c_contiguous):
            raise ValueError("_rhoa_scratch must be contiguous float32 with "
                             f"shape {shape}")
    if _ice_to_snow_scratch is None:
        ice_to_snow_scratch = cp.empty_like(rho)
    else:
        ice_to_snow_scratch = _ice_to_snow_scratch
        if (ice_to_snow_scratch.shape != shape
                or ice_to_snow_scratch.dtype != DTYPE
                or not ice_to_snow_scratch.flags.c_contiguous):
            raise ValueError("_ice_to_snow_scratch must be contiguous float32 "
                             f"with shape {shape}")

    effective = {"effc": effc, "effr": effr, "effi": effi, "effs": effs}
    for name, value in effective.items():
        if value is None:
            effective[name] = cp.empty_like(theta)
        elif (value.shape != shape or value.dtype != DTYPE
              or not value.flags.c_contiguous):
            raise ValueError(f"{name} must be contiguous float32 with "
                             f"shape {shape}")

    ncell = nz * ny * nx
    ncol = ny * nx
    cell_blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    process = get_kernel("morrison", "morrison_process_levels")
    sediment_name = ("morrison_sediment_64" if nz <= _SHALLOW_KMAX
                     else "morrison_sediment_256")
    sediment = get_kernel("morrison", sediment_name)
    finalize = get_kernel("morrison", "morrison_finalize_levels")

    process((cell_blocks,), (_CELL_TPB,),
            (theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
             qrcu, qscu, qicu, rhoa_scratch, pii, pressure,
             ice_to_snow_scratch,
             effective["effc"], effective["effi"], effective["effs"],
             DTYPE(rimed.ag), DTYPE(rimed.bg), DTYPE(rimed.rhog),
             DTYPE(dt), np.int32(has_cu_tendencies), np.int32(ncell)))
    sediment((column_blocks,), (_COLUMN_TPB,),
             (qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
              effective["effs"], theta, pii, pressure, rhoa_scratch, dz,
              rainnc, rainncv, snownc, snowncv,
              graupelnc, graupelncv, sr, DTYPE(dt),
              DTYPE(rimed.ag), DTYPE(rimed.bg), DTYPE(rimed.rhog),
              np.int32(nz), np.int32(ny), np.int32(nx)))
    finalize((cell_blocks,), (_CELL_TPB,),
             (theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
              rhoa_scratch, pii, pressure, ice_to_snow_scratch,
              effective["effc"], effective["effi"], effective["effs"],
              effective["effr"], DTYPE(rimed.rhog), np.int32(ncell)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields and apply Morrison to ``state`` in place.

    On a history step, reflectivity is evaluated after the scheme call from
    its post-call temperature and moments plus the unchanged prepared
    pressure, matching module_mp_morr_two_moment.F:913-914.
    """
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=10 state lacks Morrison fields: "
                         + ", ".join(missing))

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "morr_theta")
    rho = state.scratch((nz, ny, nx), "morr_rho")
    pii = state.scratch((nz, ny, nx), "morr_pii")
    dz = state.scratch((nz, ny, nx), "morr_dz")
    ice_to_snow = state.scratch((nz, ny, nx), "morr_ice_to_snow")
    z8w = state.scratch((nz + 1, ny, nx), "morr_z8w")
    theta[...] = thb + state.thp
    # launch_morrison's process stage diagnoses and overwrites every rho
    # element before sedimentation; Morrison never consumes the wrapper rho.
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
    # Lazy import (matches microphysics.apply's lazy Morrison dispatch):
    # the WRF moist_physics_prep_em/finish_em bracket around the scheme.
    from gpuwm.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    cu_rates = (getattr(getattr(state, "physics", None), "cu_rates", None)
                if getattr(cfg, "cu_physics", 0) else None)
    launch_morrison(theta, state.qv, state.qc, state.qr,
                    state.qi, state.qs, state.qg,
                    state.nc, state.nr, state.ni, state.ns, state.ng,
                    rho, pii, state.p, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt,
                    effc=state.effc, effr=state.effr,
                    effi=state.effi, effs=state.effs,
                    qrcuten=(None if cu_rates is None else
                              cu_rates["rqrcuten"]),
                    qscuten=(None if cu_rates is None else
                              cu_rates["rqscuten"]),
                    qicuten=(None if cu_rates is None else
                              cu_rates["rqicuten"]),
                    morr_rimed_ice=cfg.morr_rimed_ice,
                    _rhoa_scratch=rho,
                    _ice_to_snow_scratch=ice_to_snow)
    if refl_10cm_due:
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        # WRF wrapper T1D is the scheme-updated absolute temperature while
        # P1D remains the pre-call prepared pressure (F:730, :780, :913-914).
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)
