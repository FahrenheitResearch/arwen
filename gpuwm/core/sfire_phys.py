"""Device SFIRE fuel parameters, Rothermel spread and fire heat release.

Equations and static fuel rows follow WRF v4.7.1 ``module_fr_fire_phys.F``.
The 13 Anderson and 40 Scott-Burgan models use their external fuel codes.
Arrays have C layout ``(ny, nx)``. No meteorological calculation runs on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import math

import numpy as np

from gpuwm.core import sfire_fuel_data
from gpuwm.fortran_namelist import parse_namelist_text

MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.sfire_phys:sfire_phys"
PARAM_FIELDS = ("fgip", "ischap", "betafl", "bbb", "fuel_time", "phiwc", "r_0", "iboros")
TABLE_FIELDS = ("fgi", "fgi_lh", "fueldepthm", "savr", "fuelmce", "fueldens", "st", "se", "weight", "ichap")
LOAD_FIELDS = ("fgi_1h", "fgi_10h", "fgi_100h", "fgi_1000h", "fgi_live")
FUEL_CODES = tuple(range(1, 14)) + (14,) + tuple(range(101,110)) + tuple(range(121,125)) + tuple(range(141,150)) + tuple(range(161,166)) + tuple(range(181,190)) + tuple(range(201,205))
_CATEGORY_FIELDS = tuple(name.lower() for name in vars(sfire_fuel_data) if name.isupper())


def _default_categories():
    return {name: list(getattr(sfire_fuel_data, name.upper())) for name in _CATEGORY_FIELDS}


@dataclass
class FuelTable:
    """WRF ``namelist.fire`` settings; partial assignments keep DATA defaults.

    Category 54 is enabled by default so the published 40-model table can use
    external code 204. WRF's source default 53 incorrectly excludes that row.
    ``nfuelcats`` in a supplied namelist is respected.
    """
    scalars: dict = field(default_factory=lambda: dict(
        cmbcnst=17.433e6, hfgl=17.e4, fuelmc_g=0.08, fuelmc_g_lh=1.20,
        fuelmc_c=1.0, nfuelcats=54, no_fuel_cat=14))
    categories: dict = field(default_factory=_default_categories)
    moisture: dict = field(default_factory=lambda: dict(
        moisture_classes=5, drying_lag=[1.,10.,100.,1000.,1.e9],
        wetting_lag=[1.4,14.,140.,1400.,1.e9], saturation_moisture=[2.5]*5,
        saturation_rain=[8.0]*5, rain_threshold=[0.05]*5,
        drying_model=[1]*5, wetting_model=[1]*5,
        moisture_class_name=['1-h','10-h','100-h','1000-h','Live'],
        fmc_gc_initialization=[2,2,2,2,3], fmc_1h=0.08, fmc_10h=0.08,
        fmc_100h=0.08, fmc_1000h=0.08, fmc_live=0.3))

    @classmethod
    def from_namelist(cls, path):
        return cls.from_namelist_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_namelist_text(cls, text):
        result = cls()
        sections = parse_namelist_text(text, allow_unset=True)
        targets = {"fuel_scalars": result.scalars, "fuel_categories": result.categories,
                   "fuel_moisture": result.moisture}
        for group, values in sections.items():
            if group not in targets:
                raise ValueError(f"namelist.fire group &{group} is unknown")
            target = targets[group]
            for name, supplied in values.items():
                if name not in target:
                    raise ValueError(f"namelist.fire &{group}/{name} is unknown")
                default = target[name]
                if isinstance(default, list):
                    if len(supplied) > len(default):
                        raise ValueError(f"{name} exceeds its {len(default)} fuel entries")
                    for i, value in enumerate(supplied):
                        if value is not None:
                            default[i] = value
                elif len(supplied) == 1 and supplied[0] is not None:
                    target[name] = supplied[0]
                elif supplied:
                    raise ValueError(f"{name} is a scalar")
        result.validate()
        return result

    def validate(self):
        for name,value in self.scalars.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite to define SFIRE fuel parameters")
        for name in ("nfuelcats","no_fuel_cat"):
            if int(self.scalars[name]) != self.scalars[name]:
                raise ValueError(f"{name} must be an integer fuel-table index")
        for name,values in self.categories.items():
            if name != "fuel_name" and any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must be finite to define SFIRE fuel coefficients")
        for name,values in self.moisture.items():
            if name == "moisture_class_name":
                continue
            numbers = values if isinstance(values,list) else [values]
            if any(not math.isfinite(value) for value in numbers):
                raise ValueError(f"{name} must be finite to define SFIRE moisture evolution")
        if self.scalars["cmbcnst"] <= 0:
            raise ValueError("cmbcnst must be positive to define fuel heat content")
        for name in ("fuelmc_g","fuelmc_g_lh","fuelmc_c"):
            if self.scalars[name] < 0:
                raise ValueError(f"{name} must be nonnegative to define fuel moisture")
        nf = int(self.scalars["nfuelcats"])
        no_fuel = int(self.scalars["no_fuel_cat"])
        if not 1 <= nf <= 60:
            raise ValueError("nfuelcats must be 1..60 to keep fuel-table lookup in bounds")
        if nf < 14 and 1 <= no_fuel <= nf:
            raise ValueError("no_fuel_cat must be outside a fuel table smaller than 14 categories")
        mc = int(self.moisture["moisture_classes"])
        if self.moisture["moisture_classes"] != mc or not 1 <= mc <= 5:
            raise ValueError("moisture_classes must be 1..5 to match SFIRE moisture state")
        for key in LOAD_FIELDS:
            if any(value < 0 for value in self.categories[key]):
                raise ValueError(f"{key} must be nonnegative so moisture weights are defined")
        for key in ("fgi", "fgi_lh"):
            if any(value < 0 for value in self.categories[key][:nf]):
                raise ValueError(f"{key} must be nonnegative to define combustible mass")
        for key in ("weight", "fueldepthm", "savr", "fuelmce", "fueldens", "se"):
            for i,value in enumerate(self.categories[key][:nf], start=1):
                if i != no_fuel and self.categories["fgi"][i-1] > 0 and value <= 0:
                    raise ValueError(f"{key} must be positive for combustible fuel row {i}")
        for key in ("drying_lag", "wetting_lag", "saturation_rain"):
            if any(value <= 0 for value in self.moisture[key][:mc]):
                raise ValueError(f"{key} must be positive to avoid undefined moisture time lags")
        for key in ("drying_model", "wetting_model"):
            if any(value != 1 for value in self.moisture[key][:mc]):
                raise ValueError(f"{key} supports model 1, the only model implemented in WRF v4.7.1")
        if any(value not in (0,1,2,3) for value in self.moisture["fmc_gc_initialization"][:mc]):
            raise ValueError("fmc_gc_initialization must be 0..3 to define initial moisture")

    def device_table(self):
        import cupy as cp
        self.validate()
        return cp.asarray(np.asarray([self.categories[key] for key in TABLE_FIELDS],
                                     dtype=np.float32).T.copy())

    def device_weights(self):
        import cupy as cp
        self.validate()
        loads = cp.asarray(np.asarray([self.categories[key] for key in LOAD_FIELDS],
                                      dtype=np.float32).T.copy())
        weights = cp.empty_like(loads)
        _kernel("sfire_moisture_weights")((1,), (128,), (loads, weights, np.int32(60)))
        return weights


def module_source(kernel_dir=None):
    from gpuwm.core.kernels import _preamble
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return _preamble(root) + "".join((root/name).read_text(encoding="utf-8") for name in
        ("glibc_flt32.cuh", "sfire_phys.cuh", "sfire_phys.cu"))


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_phys.cu")
        result = cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx,str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return result


def _kernel(name):
    import cupy as cp
    return _module(cp.cuda.Device().id).get_function(name)


def _array(value, shape=None):
    import cupy as cp
    result = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    if shape is not None and result.shape != tuple(shape):
        result = cp.ascontiguousarray(cp.broadcast_to(result, shape))
    return result


def set_fire_params(nfuel_cat, fmc_g=None, table=None, *, fire_fmc_read=1, nfuel_cat0=1):
    """Initialize the constant fire-grid coefficients on the device.

    A zero category uses ``nfuel_cat0``. Other nonburnable codes map to 14,
    including LANDFIRE's nonburnable 91..99 codes, as in WRF's mapping table.
    """
    import cupy as cp
    table = table or FuelTable()
    table.validate()
    cat = _array(nfuel_cat)
    shape = cat.shape
    moisture = _array(table.scalars["fuelmc_g"] if fmc_g is None else fmc_g, shape)
    output = {name: cp.empty_like(cat) for name in PARAM_FIELDS}
    status = cp.zeros((),dtype=cp.int32)
    n = cat.size
    _kernel("sfire_set_params")(((n+127)//128,), (128,),
        (cat,moisture,table.device_table(),*[output[name] for name in PARAM_FIELDS],
         np.int32(n),np.int32(nfuel_cat0),np.int32(table.scalars["no_fuel_cat"]),
         np.int32(table.scalars["nfuelcats"]),np.int32(fire_fmc_read),status,
         np.float32(table.scalars["fuelmc_g"]),
         np.float32(table.scalars["fuelmc_g_lh"]),
         np.float32(np.float32(table.scalars["cmbcnst"])*np.float32(4.30e-4))))
    if int(status.item()):
        raise ValueError("NFUEL_CAT selects a fuel row above namelist.fire nfuelcats")
    output["fmc_g"] = moisture
    return output


def fire_ros(propx, propy, vx, vy, dzdx, dzdy, params, *, fire_advection=0):
    """Return WRF's base, wind and slope contributions in m/s."""
    import cupy as cp
    shape = params["r_0"].shape
    inputs = [_array(value,shape) for value in (propx,propy,vx,vy,dzdx,dzdy)]
    outputs = [cp.empty(shape,dtype=cp.float32) for _ in range(3)]
    n = outputs[0].size
    _kernel("sfire_ros")(((n+127)//128,), (128,),
        (*inputs,*[params[key] for key in ("bbb","betafl","phiwc","r_0","ischap")],
         *outputs,np.int32(n),np.int32(fire_advection)))
    return tuple(outputs)


def heat_fluxes(dt, fgip, fuel_frac_burnt, fmc_g, table=None, *, xlv=2.5e6, return_dry_mass=False):
    """Sensible and latent fluxes in W/m2; dry burnt mass supports emissions."""
    import cupy as cp
    if dt <= 0:
        raise ValueError("fire heat-flux dt must be positive to define release per second")
    table = table or FuelTable()
    fgip = _array(fgip)
    burnt,moisture = (_array(value,fgip.shape) for value in (fuel_frac_burnt,fmc_g))
    outputs = [cp.empty_like(fgip) for _ in range(3)]
    n = fgip.size
    _kernel("sfire_heat_fluxes")(((n+127)//128,), (128,),
        (fgip,burnt,moisture,*outputs,np.int32(n),np.float32(dt),
         np.float32(table.scalars["cmbcnst"]),np.float32(xlv)))
    return tuple(outputs) if return_dry_mass else tuple(outputs[:2])


def weighted_moisture(fmc_fire_classes, nfuel_cat, table=None, *, nfuel_cat0=1):
    """Average already interpolated moisture classes using WRF fuel weights.

    ``fmc_fire_classes`` is ``(class, ny, nx)``. The atmosphere-to-fire
    interpolation is performed by the fire-grid interpolation routine.
    """
    import cupy as cp
    table = table or FuelTable()
    cat = _array(nfuel_cat)
    nc = int(table.moisture["moisture_classes"])
    classes = _array(fmc_fire_classes)
    if classes.ndim != 3 or classes.shape[0] < nc or classes.shape[1:] != cat.shape:
        raise ValueError("interpolated moisture must contain every active class on the fire grid")
    classes = cp.ascontiguousarray(classes[:nc])
    output = cp.empty_like(cat)
    n = cat.size
    _kernel("sfire_weighted_moisture")(((n+127)//128,), (128,),
        (classes,cat,table.device_weights(),output,np.int32(n),np.int32(nc),np.int32(nfuel_cat0)))
    return output
