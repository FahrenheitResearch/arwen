"""Staged WRF v4.6.1 NSSL option-18 CUDA implementation.

Only independently admitted numerical slices live here until the full
mass/number/CCN/volume process driver and adaptive sedimentation pass the
official-WRF column gates.  Importing this module does not make
``mp_physics=18`` executable; global configuration remains fail-closed.
"""

from __future__ import annotations

import math

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE


_SHALLOW_KMAX = 64
_KMAX = 256
_COLUMN_TPB = 64


def _validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if not shape:
        raise ValueError("NSSL fields must be arrays")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def launch_effective_radius(
        air_density, qc, qndrop, qi, qni, qs, qns,
        re_cloud, re_ice, re_snow) -> None:
    """Diagnose NSSL cloud/ice/snow radii used by WRF radiation.

    Inputs are FP32 mixing ratios in the WRF Registry convention: mass in
    kg/kg, number in #/kg, and air density in kg/m3.  Outputs include the
    exact native option-18 driver bounds.  This diagnostic is admitted in
    isolation; it does not imply that the process network is executable.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "qc": qc,
        "qndrop": qndrop,
        "qi": qi,
        "qni": qni,
        "qs": qs,
        "qns": qns,
        "re_cloud": re_cloud,
        "re_ice": re_ice,
        "re_snow": re_snow,
    })
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_effective_radius")(
        (blocks,), (threads,),
        (air_density, qc, qndrop, qi, qni, qs, qns,
         re_cloud, re_ice, re_snow, np.int32(size)))


def launch_initial_state(
        air_density, qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh) -> None:
    """Apply WRF's mass-only NSSL first-step state initialization.

    This is the native two-moment/hail/predicted-CCN/density path through
    ``calcnfromq``.  It diagnoses absent number and volume moments, consumes
    CCN when droplet number is created, and returns negligible mass to vapor.
    Every field is mutated in place in WRF Registry mixing-ratio units.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    })
    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_initial_state")(
        (blocks,), (threads,),
        (air_density, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         np.int32(size)))


def launch_rain_self_collection(air_density, qr, qnr, dt_s: float) -> None:
    """Advance NSSL rain number through self-collection and breakup.

    ``qr`` (kg/kg) and ``qnr`` (#/kg) use WRF Registry units.  Rain mass is
    read-only; the trajectory-changing number moment is updated in place.
    This is the native option-18 two-moment process, including its mean-size
    bounds, 2-mm shutoff, exponential breakup efficiency, and timestep sink
    limiter.  Other warm-rain processes remain outside this admitted slice.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "qr": qr,
        "qnr": qnr,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_rain_self_collection")(
        (blocks,), (threads,),
        (air_density, qr, qnr, step32, np.int32(size)))


def launch_snow_aggregation(
        air_density, temperature_k, qs, qns, dt_s: float) -> None:
    """Advance the native NSSL two-moment snow aggregation sink.

    Snow mass (kg/kg) is read-only and snow number (#/kg) is updated in
    place.  This is WRF v4.6.1's default ``csacs`` process, including the
    temperature-dependent collection efficiency, native snow size diagnosis,
    per-process ten-percent depletion bound, and final mean-volume limiter.
    Deposition, melting, fragmentation, ice conversion, and sedimentation are
    separate cold-phase admission slices.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "temperature_k": temperature_k,
        "qs": qs,
        "qns": qns,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_snow_aggregation")(
        (blocks,), (threads,),
        (air_density, temperature_k, qs, qns, step32, np.int32(size)))


def launch_ice_deposition_conversion(
        full_theta, air_density, pressure_pa, exner,
        qv, qi, qni, qs, qns, dt_s: float) -> None:
    """Grow cloud ice by vapor deposition and convert large ice to snow.

    Potential temperature (K), vapor/ice/snow mass (kg/kg), and ice/snow
    number (#/kg) are mutated in place.  This is WRF v4.6.1's default
    ``icond=1, iscni=4`` cold-phase path: column-ice size/capacitance and
    ventilation, the two-pass deposition saturation bound, latent heating,
    and the 100-micron depositional conversion threshold are kept coupled.
    Sublimation and deposition onto existing snow/graupel/hail remain
    separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qi": qi,
        "qni": qni,
        "qs": qs,
        "qns": qns,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_ice_deposition_conversion")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner,
         qv, qi, qni, qs, qns, step32, np.int32(size)))


def launch_frozen_vapor_exchange(
        full_theta, air_density, pressure_pa, exner,
        qv, qi, qni, qs, qns, dt_s: float) -> None:
    """Exchange vapor with cloud ice and snow using native NSSL physics.

    Potential temperature (K), vapor/ice/snow mass (kg/kg), and ice/snow
    number (#/kg) are mutated in place.  The coupled slice includes signed
    deposition/sublimation for both frozen categories, native ventilation,
    the shared two-pass saturation limit, latent heating/cooling, number loss
    during sublimation, and default ``iscni=4`` depositional conversion.
    Graupel/hail exchange and their density moments remain separate.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qi": qi,
        "qni": qni,
        "qs": qs,
        "qns": qns,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_frozen_vapor_exchange")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner,
         qv, qi, qni, qs, qns, step32, np.int32(size)))


def launch_graupel_hail_vapor_exchange(
        full_theta, air_density, pressure_pa, exner,
        qv, qg, qng, qvolg, qh, qnh, qvolh, dt_s: float) -> None:
    """Exchange vapor with NSSL graupel and hail in native moment units.

    Potential temperature (K), vapor/graupel/hail mass (kg/kg), number
    (#/kg), and predicted volume (m3/kg air) are mutated in place.  This
    coupled slice includes density diagnosis, Milbrandt-Morrison
    ventilation, signed deposition/sublimation, the shared two-pass frozen
    saturation limit, latent heating/cooling, proportional number loss, and
    category-specific deposition density.  Collection, melting, conversion,
    breakup, and sedimentation remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_graupel_hail_vapor_exchange")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner,
         qv, qg, qng, qvolg, qh, qnh, qvolh,
         step32, np.int32(size)))


def launch_bigg_rain_freezing(
        full_theta, air_density, exner, temperature_k,
        qr, qnr, qg, qng, qvolg, dt_s: float) -> None:
    """Freeze the default Bigg option-2 rain tail into graupel.

    Potential temperature (K), rain/graupel mass (kg/kg), rain/graupel
    number (#/kg), and predicted graupel volume (m3/kg air) are mutated in
    place.  This is WRF v4.6.1's default ``ibiggopt=2, imurain=1,
    ifrzg=1`` process: the native 0.25-bin incomplete-gamma lookup and
    interpolation, strict -5 C and 8-mm gates, minimum-transfer gates,
    latent heating, frozen-drop density, and final two-moment size bounds
    remain coupled.  Optional snow routing and splinter production are
    deliberately outside this admitted slice.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "temperature_k": temperature_k,
        "qr": qr,
        "qnr": qnr,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_bigg_rain_freezing")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, temperature_k,
         qr, qnr, qg, qng, qvolg, step32, np.int32(size)))


def launch_warm_autoconversion(
        air_density, temperature_k, qc, qr, qndrop, qnr,
        dt_s: float) -> None:
    """Advance native NSSL Ziegler cloud-to-rain autoconversion.

    The four prognostic moments are mutated together in Registry units: cloud
    and rain mass are kg/kg, while droplet and rain number are #/kg.  This is
    the exact default option-18 ``dmrauto=0`` process, including cloud-number
    coalescence, the 7.51-micron initiation threshold, mass/number depletion
    guards, and the final two-moment mean-volume bounds.  Rain-cloud
    accretion, rain self-collection, condensation, and sedimentation are
    separate process slices.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "temperature_k": temperature_k,
        "qc": qc,
        "qr": qr,
        "qndrop": qndrop,
        "qnr": qnr,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_warm_autoconversion")(
        (blocks,), (threads,),
        (air_density, temperature_k, qc, qr, qndrop, qnr,
         step32, np.int32(size)))


def launch_rain_cloud_accretion(
        air_density, qc, qr, qndrop, qnr, dt_s: float) -> None:
    """Advance NSSL rain collection of cloud droplets.

    Cloud/rain mass (kg/kg) and number (#/kg) are mutated in Registry units.
    This is the default Ziegler two-moment accretion slice, including its
    initiation-radius gate, large/small collector branches, independent mass
    and number depletion guards, and final mean-volume bounds.
    """
    _, size = _validate_fields({
        "air_density": air_density,
        "qc": qc,
        "qr": qr,
        "qndrop": qndrop,
        "qnr": qnr,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_rain_cloud_accretion")(
        (blocks,), (threads,),
        (air_density, qc, qr, qndrop, qnr, step32, np.int32(size)))


def launch_clear_air_activation(
        full_theta, air_density, pressure_pa, exner, vertical_velocity,
        qv, qc, qndrop, qnn, dt_s: float) -> None:
    """Activate NSSL cloud droplets in clear, supersaturated updrafts.

    Potential temperature (K), vapor/cloud mass (kg/kg), droplet number
    (#/kg), and unactivated predicted CCN (#/kg) use WRF Registry units.
    This is the default option-18 clear-air branch of ``NUCOND``: its native
    two-iteration saturation adjustment, Twomey activation, CCN depletion,
    and cloud-droplet mean-mass bounds are coupled in one isolated slice.
    Cells containing existing cloud are left for the adjacent cloudy-water
    adjustment and cloud-interior renucleation launchers.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "vertical_velocity": vertical_velocity,
        "qv": qv,
        "qc": qc,
        "qndrop": qndrop,
        "qnn": qnn,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_clear_air_activation")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, vertical_velocity,
         qv, qc, qndrop, qnn, step32, np.int32(size)))


def launch_cloudy_water_adjustment(
        full_theta, air_density, pressure_pa, exner,
        qv, qc, qndrop, qnn, dt_s: float) -> None:
    """Adjust vapor and existing NSSL warm-cloud droplets on the GPU.

    Potential temperature (K), vapor/cloud mass (kg/kg), droplet number
    (#/kg), and unactivated predicted CCN (#/kg) use WRF Registry units.
    This bounded ``NUCOND`` slice owns full and partial cloud evaporation,
    adaptive RK2 weak-supersaturation condensation, CCN restoration, and
    native droplet mean-mass bounds.  Clear-air activation is handled by
    :func:`launch_clear_air_activation`; cloud-interior renucleation above
    0.5-percent supersaturation is delegated to
    :func:`launch_cloud_interior_renucleation`.  Rain/frozen transfer remains
    fail-closed.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "qndrop": qndrop,
        "qnn": qnn,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_cloudy_water_adjustment")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, air_density,
         qv, qc, qndrop, qnn, step32, np.int32(size),
         np.int32(0), np.int32(1), np.int32(0)))


def launch_cloud_interior_renucleation(
        full_theta, air_density, pressure_pa, exner, vertical_velocity,
        qv, qc, qndrop, qnn, dt_s: float) -> None:
    """Condense and renucleate droplets in supersaturated cloud interior.

    Inputs are contiguous FP32 ``(nz, ny, nx)`` arrays in WRF Registry
    units.  This is the default ``irenuc=2`` continuation of ``NUCOND`` for
    pre-existing cloud above its 0.5-percent supersaturation gate.  It couples
    adaptive RK2 condensation to Twomey/Cohard-Pinty activation, available-CCN
    depletion, the half-condensed-mass cap, vertical boundary gates, and final
    mean-mass bounds.  The separate maximum-supersaturation adjustment at a
    1.9 vapor/saturation ratio remains fail-closed.
    """
    shape, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "vertical_velocity": vertical_velocity,
        "qv": qv,
        "qc": qc,
        "qndrop": qndrop,
        "qnn": qnn,
    })
    if len(shape) != 3:
        raise ValueError(
            f"NSSL cloud-interior fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 3:
        raise ValueError(
            f"NSSL cloud-interior renucleation requires nz >= 3, got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_cloudy_water_adjustment")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, vertical_velocity,
         qv, qc, qndrop, qnn, step32, np.int32(size), np.int32(nz),
         np.int32(ny * nx), np.int32(1)))


def launch_primary_ice_nucleation(
        full_theta, air_density, pressure_pa, exner, vertical_velocity,
        dz, nuclei_minus, nuclei_center, nuclei_plus,
        qv, qi, qni, dt_s: float) -> None:
    """Apply default Meyers/Ferrier primary ice nucleation.

    All fields are contiguous FP32 arrays with a common shape.  ``qv`` and
    ``qi`` are kg/kg, ``qni`` is #/kg, and the three diagnosed ice-nuclei
    fields are #/m3 at the adjacent and current vertical levels.  The kernel
    owns the default ``icenucopt=1`` upwind-gradient source, ice-saturation
    and vapor limits, the 1e6-m-3 concentration cap, and coupled vapor/ice/
    number/latent-heat updates.  Neighboring deposition and collision
    processes remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "vertical_velocity": vertical_velocity,
        "dz": dz,
        "nuclei_minus": nuclei_minus,
        "nuclei_center": nuclei_center,
        "nuclei_plus": nuclei_plus,
        "qv": qv,
        "qi": qi,
        "qni": qni,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_primary_ice_nucleation")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, vertical_velocity,
         dz, nuclei_minus, nuclei_center, nuclei_plus, qv, qi, qni,
         step32, np.int32(size)))


def launch_ice_cloud_riming(
        full_theta, air_density, exner,
        qc, qndrop, qi, qni, dt_s: float) -> None:
    """Rime NSSL cloud droplets onto cloud ice.

    Potential temperature, cloud/ice mass (kg/kg), and cloud/ice number
    (#/kg) are mutated in place.  This is the default ``qiacw`` collision
    slice: strict droplet/crystal size and freezing-temperature gates,
    two-moment Seifert--Beheng collection geometry, differential fall speed,
    ten-percent source limiting, droplet-number removal, and latent heating.
    Ice-to-graupel conversion and neighboring vapor/freezing processes remain
    separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "qc": qc,
        "qndrop": qndrop,
        "qi": qi,
        "qni": qni,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_ice_cloud_riming")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, qc, qndrop, qi, qni,
         step32, np.int32(size)))


def launch_snow_cloud_riming(
        full_theta, air_density, exner,
        qc, qndrop, qs, qns, dt_s: float) -> None:
    """Rime NSSL cloud droplets onto snow aggregates.

    Potential temperature, cloud/snow mass (kg/kg), and cloud/snow number
    (#/kg) are mutated in place.  This is the default ``qsacw`` two-moment
    collection slice: Ziegler swept-volume collection, native cloud/snow
    mean-volume diagnosis, independent ten-percent mass/number depletion
    bounds, droplet-number removal, and latent heating.  Snow aggregation,
    snow-to-graupel conversion, melting, and neighboring collection processes
    remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "qc": qc,
        "qndrop": qndrop,
        "qs": qs,
        "qns": qns,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_snow_cloud_riming")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, qc, qndrop, qs, qns,
         step32, np.int32(size)))


def launch_graupel_cloud_riming(
        full_theta, air_density, exner,
        qc, qndrop, qg, qng, qvolg, dt_s: float) -> None:
    """Rime NSSL cloud droplets onto predicted-density graupel.

    Potential temperature, cloud/graupel mass (kg/kg), cloud/graupel number
    (#/kg), and graupel volume (m3/kg) are mutated in place.  This is the
    default ``qhacw`` two-moment collection slice: native droplet collection
    efficiency, Milbrandt--Morrison graupel fall speed, Seifert--Beheng
    collection geometry, independent fifty-percent mass/number depletion
    limits, latent heating, rime-density volume growth, and final graupel
    moment bounds.  Hail conversion and neighboring collection/freezing
    processes remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "qc": qc,
        "qndrop": qndrop,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_graupel_cloud_riming")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, qc, qndrop, qg, qng, qvolg,
         step32, np.int32(size)))


def launch_hail_cloud_riming(
        full_theta, air_density, exner, dz,
        qc, qndrop, qh, qnh, qvolh, dt_s: float) -> None:
    """Rime NSSL cloud droplets onto predicted-density hail.

    Potential temperature, cloud/hail mass (kg/kg), cloud/hail number (#/kg),
    and hail volume (m3/kg) are mutated in place.  Density, Exner, and cell
    depth (m) are read-only.  This is the default ``qhlacw`` two-moment
    collection slice: native droplet collection efficiency,
    Milbrandt--Morrison hail fall speed with the mandatory ``dz / dt`` cap,
    Seifert--Beheng collection geometry, independent fifty-percent
    mass/number depletion limits, latent heating, rime-density volume growth,
    and final hail moment bounds.  Neighboring hail growth, shedding, melting,
    conversion, and vapor-exchange processes remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "dz": dz,
        "qc": qc,
        "qndrop": qndrop,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_hail_cloud_riming")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, dz, qc, qndrop,
         qh, qnh, qvolh, step32, np.int32(size)))


def launch_rain_ice_collection_freezing(
        full_theta, air_density, pressure_pa, exner, temperature_k, qv,
        qr, qnr, qi, qni, qg, qng, qvolg, dt_s: float) -> None:
    """Collect cloud ice with cold rain and freeze captured rain.

    Potential temperature and rain/ice/graupel mass (kg/kg), number (#/kg),
    and predicted graupel volume (m3/kg) are mutated in place.  Density,
    pressure, Exner, temperature, and vapor are read-only inputs.  This is the
    isolated default ``iacr=2, iacrsize=5`` rain--ice interaction: reciprocal
    ``qraci``/``qiacr`` collection, WRF v4.6.1's official mass-only legacy
    fallback when the intended ``qiacr`` gate is false, the native
    150-micron rain-tail lookup,
    independent ten-percent mass/number caps, the rain-freezing heat-budget
    limit, latent heating, 900-kg/m3 frozen volume, and final moment bounds
    remain coupled.  Bigg freezing, snow routing, splinters, vapor exchange,
    and rain self-collection remain separate admission slices.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "temperature_k": temperature_k,
        "qv": qv,
        "qr": qr,
        "qnr": qnr,
        "qi": qi,
        "qni": qni,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_rain_ice_collection_freezing")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, temperature_k, qv,
         qr, qnr, qi, qni, qg, qng, qvolg, step32, np.int32(size)))


def launch_frozen_cross_collection(
        full_theta, air_density, exner, temperature_k, dz, qc,
        qr, qnr, qi, qni, qs, qns,
        qg, qng, qvolg, qh, qnh, qvolh, dt_s: float) -> None:
    """Apply native dry NSSL cross-collection among frozen categories.

    Potential temperature and rain/ice/snow/graupel/hail mass (kg/kg),
    donor number (#/kg), and predicted graupel/hail volume (m3/kg) are
    mutated in place.  Density, Exner, explicit temperature, cell depth,
    and cloud water are read-only.  The coupled slice includes snow collecting
    ice, graupel and hail collecting ice/snow/rain, independent native
    ten-percent donor limits, the hail ``dz / dt`` fall-speed cap, source
    routing, freezing heat, predicted-volume routing, and final moment bounds.
    Native two-moment snow--rain collection is intentionally a no-op.
    Wet growth, shedding, melting, and category conversion remain separate.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "temperature_k": temperature_k,
        "dz": dz,
        "qc": qc,
        "qr": qr,
        "qnr": qnr,
        "qi": qi,
        "qni": qni,
        "qs": qs,
        "qns": qns,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_frozen_cross_collection")(
        (blocks,), (threads,),
        (full_theta, air_density, exner, temperature_k, dz, qc,
         qr, qnr, qi, qni, qs, qns,
         qg, qng, qvolg, qh, qnh, qvolh,
         step32, np.int32(size)))


def launch_melting_liquid_shedding(
        full_theta, air_density, pressure_pa, exner, temperature_k, qv, dz,
        qc, qndrop, qr, qnr, qs, qns,
        qg, qng, qvolg, qh, qnh, qvolh, dt_s: float) -> None:
    """Apply coupled NSSL melting, wet growth, and liquid shedding.

    Potential temperature and cloud/rain/snow/graupel/hail mass (kg/kg),
    number (#/kg), and predicted graupel/hail volume (m3/kg) are mutated in
    place. Density, pressure, Exner, explicit temperature, vapor, and cell
    depth are read-only. The slice preserves WRF v4.6.1's cloud/rain
    collection support, snow and dense-particle melt limits, meltwater pore
    soaking, thermal wet-growth capacity, size-regime shedding, runtime
    ``imltshddmr=1`` rain-number routing, phase-change heat, and final
    two-moment bounds. Frozen-category conversion remains a separate slice.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "temperature_k": temperature_k,
        "qv": qv,
        "dz": dz,
        "qc": qc,
        "qndrop": qndrop,
        "qr": qr,
        "qnr": qnr,
        "qs": qs,
        "qns": qns,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_melting_liquid_shedding")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, temperature_k, qv, dz,
         qc, qndrop, qr, qnr, qs, qns,
         qg, qng, qvolg, qh, qnh, qvolh,
         step32, np.int32(size)))


def launch_secondary_ice_conversions(
        full_theta, air_density, pressure_pa, exner, temperature_k, qv, dz,
        qc, qndrop, qr, qnr, qi, qni, qs, qns,
        qg, qng, qvolg, qh, qnh, qvolh, dt_s: float) -> None:
    """Apply remaining default secondary-ice and category conversions.

    The coupled slice owns ``ibfc=1`` homogeneous droplet freezing,
    ``icfn=2`` contact freezing, active type-II Hallett--Mossop splintering,
    riming-driven ice/snow-to-graupel conversion, and the native two-moment
    post-init default ``ihlcnh=3`` graupel-to-hail conversion.  The default
    reverse hail-to-graupel switch remains off.  Already admitted cloud-riming
    rates are recomputed only as prerequisites; the mutations are the isolated
    full-minus-baseline process tendency including final moment-bound effects.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "temperature_k": temperature_k,
        "qv": qv,
        "dz": dz,
        "qc": qc,
        "qndrop": qndrop,
        "qr": qr,
        "qnr": qnr,
        "qi": qi,
        "qni": qni,
        "qs": qs,
        "qns": qns,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_secondary_ice_conversions")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, temperature_k, qv, dz,
         qc, qndrop, qr, qnr, qi, qni, qs, qns,
         qg, qng, qvolg, qh, qnh, qvolh,
         step32, np.int32(size)))


def launch_rain_evaporation(
        full_theta, air_density, pressure_pa, exner, qv, qr, qnr,
        dt_s: float) -> None:
    """Evaporate NSSL rain into subsaturated vapor with latent cooling.

    ``full_theta`` (K), vapor/rain mass (kg/kg), and rain number (#/kg) are
    mutated in place.  Density, pressure (Pa), and Exner are read-only.  This
    is the isolated default two-moment rain-evaporation slice, including the
    native saturation lookup, Wisner ventilation, 10-percent depletion cap,
    proportional number loss, and final mean-volume bounds.
    """
    _, size = _validate_fields({
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qr": qr,
        "qnr": qnr,
    })
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    threads = 256
    blocks = (size + threads - 1) // threads
    get_kernel("nssl2", "nssl2_rain_evaporation")(
        (blocks,), (threads,),
        (full_theta, air_density, pressure_pa, exner, qv, qr, qnr,
         step32, np.int32(size)))


def launch_rain_sedimentation(
        air_density, qr, qnr, dz, rainnc, rainncv, dt_s: float) -> None:
    """Apply native NSSL two-moment rain sedimentation.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays.  ``qr`` is
    kg/kg, ``qnr`` is #/kg, air density is kg/m3, and ``dz`` is metres.
    Surface ``rainnc`` and ``rainncv`` are contiguous FP32 ``(ny, nx)``
    arrays in millimetres.  The implementation preserves WRF's default
    adaptive substeps and hybrid reflectivity/mass-weighted rain-number
    correction; it is an independently admitted sedimentation slice rather
    than a complete NSSL process driver.
    """
    shape, _ = _validate_fields({
        "air_density": air_density,
        "qr": qr,
        "qnr": qnr,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(f"NSSL rain fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL rain sedimentation requires 2 <= nz <= {_KMAX}, got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    surface_shape = (ny, nx)
    for name, value in (("rainnc", rainnc), ("rainncv", rainncv)):
        if value.shape != surface_shape:
            raise ValueError(
                f"{name} must have shape {surface_shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("nssl2_rain_sediment_64" if nz <= _SHALLOW_KMAX
                   else "nssl2_rain_sediment_256")
    kernel = get_kernel("nssl2", kernel_name)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (air_density, qr, qnr, dz, rainnc, rainncv, step32,
            np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_snow_sedimentation(
        air_density, qs, qns, dz, snownc, snowncv, dt_s: float) -> None:
    """Apply native NSSL two-moment snow sedimentation.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays. ``qs`` is
    kg/kg, ``qns`` is #/kg, air density is kg/m3, and ``dz`` is metres.
    Surface ``snownc`` and ``snowncv`` are contiguous FP32 ``(ny, nx)``
    arrays in kilograms per square metre (millimetres liquid equivalent).
    The implementation preserves WRF's default Ferrier snow velocities,
    adaptive substeps, and mass-weighted number lower-bound correction.
    """
    shape, _ = _validate_fields({
        "air_density": air_density,
        "qs": qs,
        "qns": qns,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(f"NSSL snow fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL snow sedimentation requires 2 <= nz <= {_KMAX}, got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    surface_shape = (ny, nx)
    for name, value in (("snownc", snownc), ("snowncv", snowncv)):
        if value.shape != surface_shape:
            raise ValueError(
                f"{name} must have shape {surface_shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("nssl2_snow_sediment_64" if nz <= _SHALLOW_KMAX
                   else "nssl2_snow_sediment_256")
    kernel = get_kernel("nssl2", kernel_name)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (air_density, qs, qns, dz, snownc, snowncv, step32,
            np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_ice_sedimentation(
        air_density, qi, qni, dz, icenc, icencv, dt_s: float) -> None:
    """Apply native NSSL two-moment cloud-ice sedimentation.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays. ``qi`` is
    kg/kg, ``qni`` is #/kg, air density is kg/m3, and ``dz`` is metres.
    Surface accumulators are FP32 ``(ny, nx)`` arrays in kg/m2.  The kernel
    preserves WRF's adjusted-Ferrier ice velocities, adaptive substeps, and
    mass-weighted number lower-bound correction.
    """
    shape, _ = _validate_fields({
        "air_density": air_density,
        "qi": qi,
        "qni": qni,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(f"NSSL ice fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL ice sedimentation requires 2 <= nz <= {_KMAX}, got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    surface_shape = (ny, nx)
    for name, value in (("icenc", icenc), ("icencv", icencv)):
        if value.shape != surface_shape:
            raise ValueError(
                f"{name} must have shape {surface_shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("nssl2_ice_sediment_64" if nz <= _SHALLOW_KMAX
                   else "nssl2_ice_sediment_256")
    kernel = get_kernel("nssl2", kernel_name)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (air_density, qi, qni, dz, icenc, icencv, step32,
            np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_graupel_sedimentation(
        air_density, qg, qng, qvolg, dz,
        graupelnc, graupelncv, dt_s: float) -> None:
    """Apply native NSSL two-moment variable-density graupel fallout.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays. ``qg`` is
    kg/kg, ``qng`` is #/kg, ``qvolg`` is m3/kg air, density is kg/m3, and
    ``dz`` is metres. Surface accumulators are contiguous FP32 ``(ny, nx)``
    arrays in kg/m2. The kernel preserves WRF's default predicted-density
    diagnosis, Milbrandt--Morrison terminal velocities, adaptive substeps,
    volume fallout, and hybrid reflectivity/mass-weighted number correction.
    """
    shape, _ = _validate_fields({
        "air_density": air_density,
        "qg": qg,
        "qng": qng,
        "qvolg": qvolg,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(f"NSSL graupel fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL graupel sedimentation requires 2 <= nz <= {_KMAX}, "
            f"got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    surface_shape = (ny, nx)
    for name, value in (("graupelnc", graupelnc),
                        ("graupelncv", graupelncv)):
        if value.shape != surface_shape:
            raise ValueError(
                f"{name} must have shape {surface_shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("nssl2_graupel_sediment_64" if nz <= _SHALLOW_KMAX
                   else "nssl2_graupel_sediment_256")
    kernel = get_kernel("nssl2", kernel_name)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
            step32, np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_hail_sedimentation(
        air_density, qh, qnh, qvolh, dz,
        hailnc, hailncv, dt_s: float) -> None:
    """Apply native NSSL two-moment variable-density hail fallout.

    Volume fields are contiguous FP32 ``(nz, ny, nx)`` arrays. ``qh`` is
    kg/kg, ``qnh`` is #/kg, ``qvolh`` is m3/kg air, density is kg/m3, and
    ``dz`` is metres. Surface accumulators are contiguous FP32 ``(ny, nx)``
    arrays in kg/m2. The kernel preserves WRF's default hail shape parameter,
    predicted-density diagnosis, Milbrandt--Morrison terminal velocities,
    adaptive substeps, volume fallout, and hybrid number correction.
    """
    shape, _ = _validate_fields({
        "air_density": air_density,
        "qh": qh,
        "qnh": qnh,
        "qvolh": qvolh,
        "dz": dz,
    })
    if len(shape) != 3:
        raise ValueError(f"NSSL hail fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL hail sedimentation requires 2 <= nz <= {_KMAX}, "
            f"got {nz}")
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    surface_shape = (ny, nx)
    for name, value in (("hailnc", hailnc), ("hailncv", hailncv)):
        if value.shape != surface_shape:
            raise ValueError(
                f"{name} must have shape {surface_shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    kernel_name = ("nssl2_hail_sediment_64" if nz <= _SHALLOW_KMAX
                   else "nssl2_hail_sediment_256")
    kernel = get_kernel("nssl2", kernel_name)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
            step32, np.int32(nz), np.int32(ny), np.int32(nx)))


__all__ = [
    "launch_bigg_rain_freezing",
    "launch_clear_air_activation",
    "launch_cloud_interior_renucleation",
    "launch_cloudy_water_adjustment",
    "launch_effective_radius",
    "launch_frozen_cross_collection",
    "launch_frozen_vapor_exchange",
    "launch_graupel_hail_vapor_exchange",
    "launch_graupel_cloud_riming",
    "launch_graupel_sedimentation",
    "launch_hail_cloud_riming",
    "launch_hail_sedimentation",
    "launch_ice_deposition_conversion",
    "launch_ice_cloud_riming",
    "launch_ice_sedimentation",
    "launch_initial_state",
    "launch_melting_liquid_shedding",
    "launch_primary_ice_nucleation",
    "launch_rain_cloud_accretion",
    "launch_rain_evaporation",
    "launch_rain_ice_collection_freezing",
    "launch_rain_sedimentation",
    "launch_rain_self_collection",
    "launch_secondary_ice_conversions",
    "launch_snow_aggregation",
    "launch_snow_cloud_riming",
    "launch_snow_sedimentation",
    "launch_warm_autoconversion",
]
