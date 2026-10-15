"""WRF solar generations. The corrected generic generation is WRF 4.8."""
import numpy as np

SOLAR_GENERATIONS = {
    'wrf_461': {'phase_days': 1.0, 'sin_two_coefficient': 0.04089,
                'source': 'WRF v4.6.1 calc_coszen',
                'algorithm': 'wrf-v4.6.1-solar-eot-v1'},
    'wrf_480': {'phase_days': 0.5, 'sin_two_coefficient': 0.040849,
                'source': 'WRF v4.8.0 calc_coszen',
                'algorithm': 'wrf-v4.8.0-solar-eot-v1'},
}
DEFAULT_SOLAR_GENERATION = 'wrf_480'


def solar_generation(value=DEFAULT_SOLAR_GENERATION):
    if value not in SOLAR_GENERATIONS:
        raise ValueError(f'solar_generation={value!r} must be wrf_480 or wrf_461; '
                         'an unknown phase would change COSZEN and surface shortwave')
    return SOLAR_GENERATIONS[value]


def solar_period_seconds(cfg, rounded_seconds):
    """The WRF solar midpoint reads requested RADT, separately from STEPRA."""
    if getattr(cfg, 'solar_generation', DEFAULT_SOLAR_GENERATION) == 'wrf_461':
        return float(rounded_seconds)
    from gpuwm.config import effective_radt_minutes
    return float(effective_radt_minutes(cfg)) * 60.0


def solar_geometry_device(julian, xtime, gmt, latitude, longitude, *,
                          generation=DEFAULT_SOLAR_GENERATION, declination=None):
    """DECLIN/SOLCON, COSZEN and HRANG with WRF REAL(4) operation order."""
    import cupy as cp
    from gpuwm.core.kernels import load_module
    from gpuwm.core.rrtmg_legacy import _DEGRAD, _DPD
    solar_generation(generation)
    lat = cp.ascontiguousarray(cp.asarray(latitude, dtype=cp.float32))
    lon = cp.ascontiguousarray(cp.asarray(longitude, dtype=cp.float32))
    if lat.shape != lon.shape:
        raise ValueError('radiation latitude/longitude shapes must match')
    scalars = np.asarray([julian, xtime, gmt], np.float32)
    if not np.isfinite(scalars).all():
        raise ValueError('solar calendar scalars must be finite')
    module = load_module('rrtmg_legacy_solar')
    ds = cp.empty(2, cp.float32)
    coszen = cp.empty_like(lat)
    hrang = cp.empty_like(lat)
    module.get_function('rls_radconst')((1,), (1,),
        (scalars[0], _DEGRAD, _DPD, ds))
    if declination is not None:
        ds[0] = np.float32(declination)
    if lat.size:
        module.get_function('rls_calc_coszen_hrang')(((lat.size+255)//256,), (256,),
            (np.int64(lat.size), lat, lon, *scalars, _DEGRAD,
             np.int32(generation == 'wrf_480'), ds, coszen, hrang))
    return ds, coszen, hrang


def geometry_at(valid_time, latitude, longitude, *, hour_offset_seconds=0.,
                generation=DEFAULT_SOLAR_GENERATION, start_time=None,
                elapsed_seconds=0.):
    """Use the delivered call calendar; shift only XTIME for the midpoint."""
    if not np.isfinite(hour_offset_seconds):
        raise ValueError('hour_offset_seconds must be finite')
    if start_time is None:
        start_time = valid_time
        elapsed_seconds = 0.
    hour = (valid_time.hour + valid_time.minute/60. + valid_time.second/3600.
            + valid_time.microsecond/3.6e9)
    gmt = (start_time.hour + start_time.minute/60. + start_time.second/3600.
           + start_time.microsecond/3.6e9)
    # WRF adds separately rounded XTIME and RADT*0.5 in minute units.
    xtime = np.float32(np.float32(elapsed_seconds/60.) + np.float32(hour_offset_seconds/60.))
    return solar_geometry_device(valid_time.timetuple().tm_yday-1+hour/24.,
        xtime, gmt, latitude, longitude,
        generation=generation)
