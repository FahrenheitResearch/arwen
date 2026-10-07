"""Developer import of compiled ideal.exe state through the Rust decoder.

The ideal reference carries no real.exe soil requirement. Every field used
by the dynamics restoration is still supplied by the native initializer.
This helper is deliberately separate from the user-facing real-data door.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from gpuwm import netcdf_bridge
from gpuwm.ingest.wrfinput import RestoredDomain, _read_numeric


ATMOSPHERE_FIELDS = (
    "U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "T_INIT", "P", "PB", "AL", "ALB",
    "QVAPOR", "HGT", "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "ZNU", "ZNW", "C1H", "C2H",
    "C1F", "C2F", "C3H", "C4H", "C3F", "C4F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "CF1", "CF2", "CF3", "P_TOP",
)
FIRE_STATIC_FIELDS = ("NFUEL_CAT", "ZSF", "DZDXF", "DZDYF", "FMC_G")
MAPFACTOR_DIRECTIONS = {"MAPFAC_M": ("MAPFAC_MX", "MAPFAC_MY"),
                        "MAPFAC_U": ("MAPFAC_UX", "MAPFAC_UY"),
                        "MAPFAC_V": ("MAPFAC_VX", "MAPFAC_VY")}
OPTIONAL_ATMOSPHERE_FIELDS = ("QCLOUD", "QRAIN", "TKE", "UST", "ZNT", "TSK", "PSFC", "T2", "Q2", "U10", "V10", "XLAT", "XLONG",
                              *(name for pair in MAPFACTOR_DIRECTIONS.values() for name in pair))


def read_ideal_initial(path: str | Path, cfg) -> RestoredDomain:
    """Restore the compiled WRF initializer's exact atmosphere and fire inputs."""
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        missing = sorted(set(ATMOSPHERE_FIELDS) - dataset.variables.keys())
        if missing:
            raise ValueError(f"compiled ideal.exe input misses required native fields: {missing}")
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        expected = {"west_east": cfg.nx, "south_north": cfg.ny, "bottom_top": cfg.nz,
                    "west_east_stag": cfg.nx + 1, "south_north_stag": cfg.ny + 1, "bottom_top_stag": cfg.nz + 1}
        for name, length in expected.items():
            if dimensions.get(name) != length:
                raise ValueError(f"ideal.exe {name}={dimensions.get(name)} disagrees with configured {length}")
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        names = (*ATMOSPHERE_FIELDS, *FIRE_STATIC_FIELDS, *OPTIONAL_ATMOSPHERE_FIELDS)
        if bool(cfg.fire_smoke):
            if "fire_smoke" not in dataset.variables:
                raise ValueError("native tracer option 3 requires its compiled fire_smoke input volume")
            names += ("fire_smoke",)
        raw = {name: _read_numeric(dataset.variables[name]) for name in names if name in dataset.variables}
        if bool(cfg.fire_smoke) and raw["fire_smoke"].shape != (cfg.nz, cfg.ny, cfg.nx):
            raise ValueError("native fire_smoke input volume does not align with the atmospheric mass grid")
    # module_initialize_fire.F:232-238 sets only directional map factors;
    # its legacy MAPFAC_M/U/V fields remain zero. Preserve those words but
    # restore the factors the native solver actually consumes. Equal X/Y
    # directions prove that the shared isotropic state loses no geometry.
    restored_factors = {}
    for legacy, (x_name, y_name) in MAPFACTOR_DIRECTIONS.items():
        if np.any(raw[legacy] <= 0):
            if x_name not in raw or y_name not in raw:
                raise ValueError(f"zero legacy {legacy} requires the native {x_name}/{y_name} factors")
            if np.any(raw[x_name] <= 0) or not np.array_equal(raw[x_name], raw[y_name]):
                raise ValueError(f"the ideal importer cannot replace {legacy} with unequal or nonpositive directional factors")
            raw[f"LEGACY_{legacy}"] = raw[legacy]
            raw[legacy] = raw[x_name]
            restored_factors[legacy] = x_name
    attrs["IDEAL_EFFECTIVE_MAPFACTORS"] = restored_factors
    if not cfg.moist:
        raise ValueError("the official em_fire initial sounding is moist; dry dynamics would discard its QVAPOR")
    for name, value in raw.items():
        if not np.isfinite(value).all():
            raise ValueError(f"compiled ideal.exe {name} contains nonfinite values")
    return RestoredDomain(path=path, raw=MappingProxyType(raw), dimensions=MappingProxyType(dimensions),
                          global_attributes=MappingProxyType(attrs), mapped_variables=ATMOSPHERE_FIELDS,
                          auxiliary_variables=tuple(name for name in raw if name not in ATMOSPHERE_FIELDS))
