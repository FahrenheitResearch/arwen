"""mkcase.py VARIANT OUT_TOML CYC RCYC HOURS [NX NY LAT LON]
Author one experiment.toml from the 5090-B c24s case (full HRRR grid under the shipped HRRR door at c909bccce; the
door recipe configs/recipes/hrrr_configuration_clock.toml is byte-unchanged from c909bccce to a80bc6b21).
VARIANT conus3: HRRR grid, 3 km, HRRR door as is (fixed 20 s, 6 substeps), new cycle and length.
VARIANT km1full: the same physics on a 1 km Lambert window inside HRRR's grid (fixed 6 s, 4 substeps, epssm 0.2,
  radt 5 min), statics built from the WPS 30 s geography (no [static] row at 1 km), high-res statics off,
  UP_HELI_MAX on (nwp_diagnostics 1), no restarts.
VARIANT km1lean: as km1full but WSM6/YSU/MM5 sfclay/Noah 4 levels/RRTMGP (the 5090-B k1 physics)."""
import re, sys
variant, out, cyc, rcyc, hours = sys.argv[1:6]
hours = int(hours)
t = open('/work/earth2-1km/kit/hrrr-door-c24s.toml').read()
def sub(t, k, v, count=1):
    pat = rf'(?m)^{re.escape(k)} = .*$'
    n = len(re.findall(pat, t))
    assert n == count, (k, n)
    return re.sub(pat, f'{k} = {v}', t)
def drop(t, k):
    return re.sub(rf'(?m)^{re.escape(k)} = .*\n', '', t)
def add_shared(t, line):
    return t.replace('[shared]\n', '[shared]\n' + line + '\n', 1)
start = f'{cyc[0:4]}-{cyc[4:6]}-{cyc[6:8]}T{cyc[8:10]}:00:00'
off = 3
t = sub(t, 'start_time', start)
t = sub(t, 'run_seconds', f'{hours * 3600}.0')
t = sub(t, 'cycle', f'"{rcyc[0:4]}-{rcyc[4:6]}-{rcyc[6:8]}T{rcyc[8:10]}"')
t = sub(t, 'hours', str(hours + off))
t = re.sub(r'(?m)^# .*\n', '', t)  # the c24s header comments describe another case
if variant == 'conus3':
    t = sub(t, 'name', f'"woof-hrrr-door-conus3km-{cyc}"')
else:
    nx, ny, lat, lon = sys.argv[6:10]
    t = sub(t, 'name', f'"woof-1km-{variant}-{cyc}"')
    t = sub(t, 'ref_lat', lat); t = sub(t, 'ref_lon', lon)
    t = sub(t, 'nx', nx); t = sub(t, 'ny', ny); t = sub(t, 'dx', '1000.0')
    t = sub(t, 'time_step', '6')
    for k in ('starting_time_step', 'max_time_step', 'min_time_step'):
        t = sub(t, k, '6')
    t = sub(t, 'time_step_sound', '4')
    t = sub(t, 'epssm', '0.2')
    t = sub(t, 'radt', '5.0')
    t = re.sub(r'(?ms)^\[static\]\n.*?(?=^\[|\Z)', '', t)
    t = t.rstrip('\n') + '\n\n[static.highres]\nenabled = false\ncache_root = "/work/earth2-1km/cache/highres-cache"\n'
    t = add_shared(t, 'nwp_diagnostics = 1')
    t = t.replace('"SMOIS"]', '"SMOIS", "UP_HELI_MAX"]')
    if variant == 'km1lean':
        for k in ('usemonalb', 'upper_wind_limiter_form', 'swint_opt', 'ruc_soilprop', 'ruc_snow', 'ruc_qvg_cold_start',
                  'ruc_irrigation', 'ruc_2m_diagnostic', 'rrtmg_cloud_optics_form', 'rdlai2d', 'mp_zero_out_thresh',
                  'mp_zero_out_all', 'mp_zero_out', 'mp28_aerosol_source', 'bl_mynn_version', 'alb_sol', 'aer_opt',
                  'bl_mynn_mixlength', 'scalar_pblmix', 'aer_init_opt', 'wif_input_opt', 'use_rap_aero_icbc', 'ra_physics'):
            t = drop(t, k)
        t = sub(t, 'gwd_opt', '0'); t = sub(t, 'mosaic_lu', '0'); t = sub(t, 'mosaic_soil', '0'); t = sub(t, 'sf_lake_physics', '0')
        t = sub(t, 'mp_physics', '6'); t = sub(t, 'bl_pbl_physics', '1'); t = sub(t, 'sf_sfclay_physics', '1')
        t = sub(t, 'sf_surface_physics', '2'); t = sub(t, 'num_soil_layers', '4')
        t = sub(t, 'wrf_rrtmg_compatibility', '"wrf-rrtmg-4-4-to-rte-rrtmgp-v2"'); t = sub(t, 'ra_rrtmg_variant', '"rte-rrtmgp"')
        t = add_shared(t, 'ra_lw_physics = 4\nra_sw_physics = 4\nicloud = 1\nswrad_scat = 1.0\no3input = 2\nuse_mp_re = 1\nsurface_radiation_policy = "required"\nisfflx = 1')
open(out, 'w').write(t)
print(variant, out, start, hours)
