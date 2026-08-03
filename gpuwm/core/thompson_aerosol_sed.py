"""Aerosol-aware Thompson (``mp_physics=28``) fallout and phase cleanup.

Two launchers, both backed by ``gpuwm/core/kernels/thompson_aerosol_sed.cu``:

``launch_aa_cloud_sedimentation``
    WRF's cloud-water fallout with the droplet-NUMBER channel that classic
    Thompson does not have at all (module_mp_thompson.F:3644-3666 and
    :3823-3838).

``launch_aa_final_phase_cleanup``
    The two instantaneous phase transfers at :3943-3966, made
    number-conserving: melted cloud ice hands its number to the droplet
    population and homogeneously frozen cloud water hands its number back.

Neither writes ``state.nc``.  Both write the shared per-kilogram-per-second
accumulator ``ncten``, which WP-04's terminal state kernel applies once with
WRF's clamps (:3972-4021).  That split is the whole accumulator contract: WRF
clamps the droplet number exactly once per call, and four independent clamps
would be a silent physics change no unit test would flag.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
* It does not sediment rain, ice, snow or graupel.  The mp=28 adapter calls
  ``gpuwm.core.thompson``'s launchers for those, byte-for-byte unchanged;
  module_mp_thompson.F:3790-3936 contains no ``is_aerosol_aware`` branch and
  no nc/nwfa/nifa reference.
* It does not sediment aerosol.  ``nwfa`` and ``nifa`` have NO fallout term
  anywhere in module_mp_thompson.F.  Any implementation that adds one is
  wrong.
* It does not substep.  WRF's cloud fallout is a single pass with no
  ``onstep`` factor, unlike every other species (see the kernel header).
* It accumulates nothing at the surface.  Cloud mass and number leaving the
  lowest level are discarded by WRF, so a number budget over this launcher
  does not close.  It closes against the two END FLUXES, which is what
  ``tests/test_thompson_aerosol_sed_gpu.py`` asserts instead.

END-TO-END STATUS
-----------------
These two launchers are exercised inside a complete mp_physics=28 column call
-- eighteen launchers in WRF's driver order -- against the committed
``aero-nc-sed`` (109) and ``aero-reduces-to-classic`` (119) fixtures.  The
driver lives in the test module because
``gpuwm/core/microphysics_aerosol.py::_apply_thompson_aerosol`` (WP-09) does
not exist yet; when it does, the test should call it and the local driver
should be deleted.

MEASURED (RTX 5090, cupy 14.1.1): aero-nc-sed reproduces WRF BIT-EXACTLY in
qv, qc, qi, qs, qg, ni, nwfa, nifa, temperature, all three effective radii and
RAINNC, and to 1.2e-6 in qr/nr/nc.  aero-reduces-to-classic reproduces WRF to
9.5e-7 or better everywhere except one level of qr/nr; see the test module for
that level's localisation.  Both measurements require the shared saturation
fit in ``thompson_aerosol_common.cuh`` to be contraction-pinned; without that
one change all twenty cloud-free levels of aero-nc-sed condense spurious water
and lose up to 5.6 percent of the column CCN, because the FMA-contracted fit is
one ulp low and WRF opens its condensation block on ``ssatw > 1.E-15``
(:3400, with eps declared at :185).
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE
from gpuwm.core.thompson_aerosol_launch import (
    DEFAULT_THREADS,
    SED_MODULE,
    launch_grid,
    validate_fields,
)

#: Column-kernel launch width.  Matches ``gpuwm/core/thompson.py:16`` so the
#: aerosol and classic sedimentation passes tile identically.
_COLUMN_TPB = 32

#: Template instantiations in thompson_aerosol_sed.cu.  These mirror
#: ``THOMPSON_AA_KMAX_SHALLOW`` / ``THOMPSON_AA_KMAX_GENERIC``.
_SHALLOW_KMAX = 64
_KMAX = 256

#: Advertised vertical extent, matching ``thompson.VERTICAL_LEVEL_BOUNDS``.
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)

#: Diagnostic outputs ``launch_aa_cloud_sedimentation`` can fill, in the order
#: the kernel takes them.  Part of the published API; do not reorder.
DIAGNOSTIC_FIELDS = (
    "mass_velocity",
    "number_velocity",
    "cloud_mass",
    "cloud_number",
)


def _validate_surface_mask(name, value, surface_shape) -> None:
    if value.shape != surface_shape:
        raise ValueError(
            f"{name} must have shape {surface_shape}, got {value.shape}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def launch_aa_cloud_sedimentation(
        qc, cloud_number_entry, cloud_number_tendency,
        temperature, pressure, qv, vertical_velocity, dz,
        dt: float, *, reference_density,
        rain_active_columns=None, cloud_active_columns=None,
        diagnostics=None) -> None:
    """Apply WRF's number-weighted cloud-water fallout.

    Parameters
    ----------
    qc
        Cloud water mixing ratio, read-modify-written in place exactly as the
        frozen mp=8 launcher does.
    cloud_number_entry
        ``nc1d``: the per-kilogram droplet number as it entered the
        microphysics call.  READ ONLY.

        NOT raw ``state.nc``.  module_mp_thompson.F:1844-1846 rewrites the
        caller's own column on the way in::

            else                    ! qc1d(k) .le. R1
               qc1d(k) = 0.0
               nc1d(k) = 0.0

        so the entry droplet number every later block reads is the state array
        ZEROED wherever cloud water was absent at call entry.  Handing in the
        raw array instead gives a non-zero working droplet number in air that
        has no droplets, which stays bounded and never trips a health check.
    cloud_number_tendency
        ``ncten``: the shared per-kilogram-per-second accumulator.  Read for
        the working droplet number and then added to.
    reference_density
        WRF's HELD pre-adjustment density -- the value ``rc`` and ``nc`` are
        both formed on at :3216/:3486, before ``rho`` is refreshed at :3489.
        It is REQUIRED here, unlike in the classic launcher: forming the
        droplet number on the refreshed density instead is an invisible error
        that leaves every bound intact.
    rain_active_columns, cloud_active_columns
        WRF's post-source ``ANY(L_qr)`` and held ``ANY(L_qc)`` column guards,
        with the same meaning and the same ordering constraint as
        ``gpuwm.core.thompson.launch_cloud_sedimentation``.
    diagnostics
        Optional mapping of names in :data:`DIAGNOSTIC_FIELDS` to float32
        arrays shaped like ``qc``.  Supplying any of them selects a separate
        kernel entry point; the physics is identical and the unrequested
        outputs are passed as null pointers.  This exists so the oracle gate
        can pin WRF's intermediate columns rather than only the endpoints, and
        it is not used by the forecast adapter.

        * ``mass_velocity`` -- WRF's ``vtck``.
        * ``number_velocity`` -- WRF's ``vtnck``, which has no mp=8
          counterpart.
        * ``cloud_mass`` -- the working ``rc`` BEFORE the fallout is applied,
          i.e. the value ``sed_c`` is built from.
        * ``cloud_number`` -- the working ``nc`` AFTER the fallout is applied,
          i.e. WRF's :3835 with its floor of 10.
    """
    shape, _ = validate_fields({
        "qc": qc,
        "cloud_number_entry": cloud_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "reference_density": reference_density,
        "vertical_velocity": vertical_velocity,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(
            f"Thompson aerosol cloud fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"Thompson aerosol cloud sedimentation requires 2 <= nz <= "
            f"{_KMAX}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    surface_shape = (ny, nx)
    if rain_active_columns is not None:
        _validate_surface_mask(
            "rain_active_columns", rain_active_columns, surface_shape)
    if cloud_active_columns is not None:
        if rain_active_columns is None:
            raise ValueError(
                "cloud_active_columns requires rain_active_columns")
        _validate_surface_mask(
            "cloud_active_columns", cloud_active_columns, surface_shape)

    diagnostic_arguments = None
    if diagnostics:
        unknown = set(diagnostics) - set(DIAGNOSTIC_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown sedimentation diagnostics {sorted(unknown)}; "
                f"expected a subset of {list(DIAGNOSTIC_FIELDS)}")
        supplied = {name: diagnostics[name] for name in DIAGNOSTIC_FIELDS
                    if diagnostics.get(name) is not None}
        if supplied:
            validate_fields({"qc": qc, **supplied})
            diagnostic_arguments = tuple(
                diagnostics.get(name) for name in DIAGNOSTIC_FIELDS)

    suffix = "64" if nz <= _SHALLOW_KMAX else "256"
    if diagnostic_arguments is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_diagnostic"
    elif cloud_active_columns is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_with_masks"
    elif rain_active_columns is not None:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}_with_rain"
    else:
        kernel_name = f"thompson_aa_cloud_sediment_{suffix}"

    arguments = (qc, cloud_number_entry, cloud_number_tendency,
                 temperature, pressure, qv, reference_density)
    if diagnostic_arguments is not None:
        arguments += (rain_active_columns, cloud_active_columns)
    else:
        if rain_active_columns is not None:
            arguments += (rain_active_columns,)
        if cloud_active_columns is not None:
            arguments += (cloud_active_columns,)
    arguments += (vertical_velocity, dz)
    if diagnostic_arguments is not None:
        arguments += diagnostic_arguments
    arguments += (DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx))

    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel(SED_MODULE, kernel_name)(
        (blocks,), (_COLUMN_TPB,), arguments)


def launch_aa_final_phase_cleanup(
        qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
        cloud_number_tendency, pressure, qv, dt: float) -> None:
    """Melt cloud ice above 0 C and freeze cloud water below HGFR.

    module_mp_thompson.F:3943-3966.  Both transfers move NUMBER as well as
    mass, which classic Thompson does not do:

    * melting credits ``ncten`` with ``ni1d(k)`` -- the ENTRY ice number, not
      the live ``ni``.  ``ice_number_entry`` therefore has to be a per-call
      snapshot taken before any source network runs; handing in ``state.ni``
      would be a plausible-looking, silently wrong droplet source.  Like
      ``cloud_number_entry`` it is the entry array with :1870-1871's zeroing
      applied, i.e. zero wherever ``qi <= R1`` at call entry.
    * freezing debits ``ncten`` by ``nc1d(k) + ncten(k)*DT``, the true running
      per-kilogram droplet number, read after the melt branch and deliberately
      unclamped.

    ``qc``, ``qi``, ``ni`` and ``temperature`` are updated in place; ``nc`` is
    not touched.
    """
    _, size = validate_fields({
        "qc": qc,
        "qi": qi,
        "ni": ni,
        "temperature": temperature,
        "cloud_number_entry": cloud_number_entry,
        "ice_number_entry": ice_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "pressure": pressure,
        "qv": qv,
    })
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    grid, block = launch_grid(size, DEFAULT_THREADS)
    get_kernel(SED_MODULE, "thompson_aa_final_phase_cleanup")(
        grid, block,
        (qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
         cloud_number_tendency, pressure, qv, DTYPE(dt), np.int32(size)))


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "VERTICAL_LEVEL_BOUNDS",
    "launch_aa_cloud_sedimentation",
    "launch_aa_final_phase_cleanup",
]
