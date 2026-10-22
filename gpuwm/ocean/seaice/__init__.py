"""GPU thermodynamic sea ice with a prescribed ocean heat boundary.

Original implementation of the zero-heat-capacity ice/snow column equations.
No Icepack code is included. See README.md for equations and scope.
"""
from pathlib import Path
import numpy as np
from gpuwm.core.device_cache import cuda_cache

STATE = ("ice_volume", "ice_snow_mass", "ice_skin", "ice_ocean_mask",
         "ice_rain", "ice_snowfall", "ice_growth", "ice_melt",
         "ice_energy", "ice_energy_error", "ice_ocean_exchange", "ice_ocean_heat", "ice_thickness_change",
         "ice_concentration_change")
OUTPUT = {"ice_thickness_change": ("ICETHICKCHANGE", "m"),
          "ice_concentration_change": ("ICECONCCHANGE", "1"),
          "ice_volume": ("ICEVOLUME", "m"),
          "ice_snow_mass": ("ICESNOWMASS", "kg m-2"),
          "ice_skin": ("ICETSK", "K"),
          "ice_growth": ("ICEGROWTH", "m"),
          "ice_melt": ("ICEMELT", "m"),
          "ice_energy": ("ICEENERGY", "J m-2"),
          "ice_energy_error": ("ICEENERGYERROR", "J m-2"),
          "ice_ocean_exchange": ("ICEOCEANEXCHANGE", "J m-2"),
          "ice_thickness": ("ICETHICK", "m"),
          "ice_snow_depth": ("ICESNOWDEPTH", "m")}

@cuda_cache(maxsize=8)
def _module():
    import cupy as cp
    from gpuwm.certify.kernel_manifest import record_module
    source = Path(__file__).with_name("thermodynamics.cu").read_text(encoding="utf-8")
    options = ("-std=c++17",)
    mod = cp.RawModule(code=source, options=options)
    record_module("seaice_thermodynamics", source=source, options=options, module=mod)
    return mod

def validate(cfg):
    from gpuwm.config import validate_seaice_parameters
    validate_seaice_parameters(cfg)

def _launch(name, fields, names, scalars=()):
    shape = fields["tsk"].shape
    arrays = []
    for key in names:
        a = fields[key]
        if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError(f"sea ice {key} requires contiguous float32 {shape}")
        arrays.append(a)
    n = fields["tsk"].size
    _module().get_function(name)(((n+127)//128,), (128,),
        (*arrays, *(np.float32(x) for x in scalars), np.int32(n)))

def initialize(fields, cfg):
    """Activate by default only on a domain containing ocean ice.

    All state is in the existing restart field dictionary. A checkpoint restore
    overwrites this cold seed before any integration. Thickness is a declared
    prior unless supplied in fields['ice_thickness']; it is never called an
    observed thickness. Lake columns retain their existing lake owner.
    """
    import cupy as cp
    validate(cfg)
    if not cfg.seaice_prognostic or not cfg.sf_surface_physics:
        return
    mask = ((fields["xice"] > 0) & (fields["lakemask"] < .5))
    if not bool(cp.any(mask)):
        return
    for key in STATE:
        fields[key] = cp.zeros_like(fields["tsk"])
    for key, prior in (("ice_thickness", cfg.seaice_initial_thickness),
                       ("ice_snow_depth", 0)):
        if key not in fields:
            fields[key] = cp.full_like(fields["tsk"], prior)
    invalid = (~cp.isfinite(fields["ice_thickness"]) | (fields["ice_thickness"] <= 0)) & mask
    if bool(cp.any(invalid)):
        raise ValueError("sea ice thickness must be finite and positive over initial ocean ice")
    _launch("ice_initialize", fields,
        ("xice", "xland", "lakemask", "snow", "tsk", "sst", "ice_thickness",
         "ice_snow_depth", "ice_volume", "ice_snow_mass", "ice_skin",
         "ice_ocean_mask"))

COLD_SEEDED = frozenset(STATE) | {"ice_thickness", "ice_snow_depth"}


def cold_seed_in_place(fields, cfg, absent):
    """Seed an existing ice column in place, as :func:`initialize` seeds it.

    For a warm start from a WRF restart, which carries no state of this
    model: ``absent`` names the carriers the restart could not fill.  The
    arrays are the driver's own (bound into its checkpoint manifest), so
    they are written in place, never replaced.
    """
    if "ice_volume" not in fields:
        return
    for key in STATE:
        if key in absent:
            fields[key][...] = 0
    if "ice_thickness" in absent:
        fields["ice_thickness"][...] = cfg.seaice_initial_thickness
    if "ice_snow_depth" in absent:
        fields["ice_snow_depth"][...] = 0
    _launch("ice_initialize", fields,
        ("xice", "xland", "lakemask", "snow", "tsk", "sst", "ice_thickness",
         "ice_snow_depth", "ice_volume", "ice_snow_mass", "ice_skin",
         "ice_ocean_mask"))


def capture_precipitation(driver, atmosphere):
    f = driver.fields
    if "ice_volume" not in f:
        return
    from gpuwm.core.physics import microphysics_scheme_sr_available
    ratio = driver.microphysics.sr if microphysics_scheme_sr_available(driver.mp_physics) else f["sr"]
    forcing = dict(f, ice_pending=driver._pending_rainbl,
                   ice_air=atmosphere["temperature"][0], ice_ratio=ratio)
    _launch("ice_precipitation", forcing,
        ("rainbl", "ice_pending", "ice_air", "ice_ratio", "ice_rain", "ice_snowfall"))

def step(driver, atmosphere, cfg):
    """After the LSM and before its energy ledger and PBL consume fluxes."""
    f = driver.fields
    if "ice_volume" not in f:
        return
    forcing = dict(f, ice_air=atmosphere["temperature"][0],
                   ice_qair=atmosphere["qv"][0])
    _launch("ice_step", forcing,
        ("xice", "ice_volume", "ice_snow_mass", "ice_skin", "ice_ocean_mask",
         "ice_rain", "ice_snowfall", "ice_growth", "ice_melt", "ice_energy",
         "ice_energy_error", "ice_ocean_exchange", "ice_ocean_heat",
         "ice_thickness_change", "ice_concentration_change", "ice_thickness", "ice_snow_depth",
         "tsk", "sst", "ice_air", "ice_qair", "psfc", "chs", "swdown", "glw",
         "hfx", "lh", "qfx", "qsfc", "albedo", "emiss", "snow", "snowh", "znt"),
        (driver.bldt_seconds, cfg.seaice_ocean_heat_flux,
         cfg.seaice_new_ice_thickness))

def history(fields):
    return {external: fields[key] for key, (external, _) in OUTPUT.items()
            if key in fields}

def blend_after_ocean(fields):
    if "ice_volume" in fields:
        _launch("ice_blend_ocean", fields, ("ice_ocean_mask", "xice", "ice_skin", "sst", "tsk"))


def reconcile_ocean(fields, cfg):
    from gpuwm.surface_ocean import ocean_option
    constants = ocean_option(cfg.sf_ocean_physics)["constants"]
    if "ice_volume" in fields:
        _launch("ice_ocean_freeze", fields,
            ("ice_ocean_mask", "xice", "ice_volume", "ice_skin",
             "ice_thickness", "ice_snow_mass", "ice_snow_depth", "ice_growth",
             "ice_energy", "ice_energy_error", "ice_ocean_exchange",
             "ice_thickness_change", "ice_concentration_change", "snow", "snowh", "albedo", "znt",
             "sst", "tml", "hml"),
            (cfg.seaice_new_ice_thickness, constants["rhowater"]*constants["cwater"]))
        blend_after_ocean(fields)
