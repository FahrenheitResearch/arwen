"""Carry RUC category fractions through real-data physics initialization.

The fraction edits follow WRF v4.6.1 module_initialize_real.F:2640-2642
and module_soil_pre.F:335-337,350-356. WRF is distributed under its own
license; see licenses/LICENSE-WRF-public-domain.txt. No fraction is sorted
or renormalized here.
"""
from __future__ import annotations

import numpy as np


def ruc_mosaic_physics_inputs(cfg, static, *, landuse_attrs=None,
                              xice=None, fractional_seaice=False,
                              processed=False):
    """Return only the fraction keywords selected by this RUC configuration.

    Wrfinput carries real.exe's edited fractions. Prepared/static inputs do
    not, so merge inland-water fractions and match the sea-ice categories
    used by the engine's land-use initializer. As in that initializer, the
    cold-water TSK-only sea-ice trigger is not synthesized here.
    """
    if int(getattr(cfg, "sf_surface_physics", 0)) != 3:
        return {}
    selected = [("mosaic_lu", "LANDUSEF", "landusef"),
                ("mosaic_soil", "SOILCTOP", "soilctop")]
    result = {}
    for option, name, keyword in selected:
        if int(getattr(cfg, option, 0)) != 1:
            continue
        value = static.get(name)
        if value is None:
            raise ValueError(
                f"{option}=1 requires {name}; without category fractions "
                "RUC would silently run the dominant category")
        if processed:
            result[keyword] = value
            continue
        landmask = np.asarray(static["LANDMASK"], dtype=np.float32)
        ice = np.asarray(xice, dtype=np.float32)
        ice = np.broadcast_to(ice, landmask.shape)
        if name == "LANDUSEF":
            from gpuwm.core.noah_mosaic import real_exe_landusef
            result[keyword] = real_exe_landusef(
                value, landmask=landmask, xice=ice,
                iswater=int(landuse_attrs["ISWATER"]),
                islake=int(landuse_attrs["ISLAKE"]),
                isice=int(landuse_attrs["ISICE"]),
                fractional_seaice=bool(fractional_seaice))
        else:
            fractions = np.array(value, dtype=np.float32, copy=True)
            if fractions.ndim != 3 or fractions.shape[1:] != landmask.shape:
                raise ValueError(
                    "SOILCTOP must be (soil_cat, ny, nx); RUC would mix "
                    "fractions from different columns")
            seaice = ((landmask <= np.float32(.5))
                      & (ice >= np.float32(.02 if fractional_seaice else .5)))
            if np.any(seaice):
                if fractions.shape[0] < 16:
                    raise ValueError(
                        "SOILCTOP lacks category 16 for sea ice; real.exe "
                        "would address a missing soil category")
                fractions[:, seaice] = np.float32(0)
                fractions[15, seaice] = np.float32(1)
            result[keyword] = fractions
    return result


def wrfinput_ruc_mosaic_inputs(restored, cfg):
    """Read selected fractions through Rust without changing the off reader."""
    if (int(getattr(cfg, "sf_surface_physics", 0)) != 3
            or not (getattr(cfg, "mosaic_lu", 0)
                    or getattr(cfg, "mosaic_soil", 0))):
        return {}
    # The category axis by every name a WRF writer gives it: real.exe
    # writes land_cat_stag / soil_cat_stag (Registry.EM_COMMON:773-774 put
    # LANDUSEF and SOILCTOP on stagger Z, and WRF's I/O appends _stag; the
    # frozen v4.6.1 file wrf_direct_v461_contract.json records exactly
    # that), WPS's geo_em spells it land_cat / soil_cat.  Accepting only
    # the geo_em spelling refused every real.exe wrfinput under RUC mosaic.
    selected = {}
    if int(getattr(cfg, "mosaic_lu", 0)) == 1:
        selected["LANDUSEF"] = ("land_cat_stag", "land_cat")
    if int(getattr(cfg, "mosaic_soil", 0)) == 1:
        selected["SOILCTOP"] = ("soil_cat_stag", "soil_cat")
    fields = {name: restored.raw[name] for name in selected
              if name in restored.raw}
    missing = set(selected) - fields.keys()
    if missing:
        from gpuwm import netcdf_bridge
        from gpuwm.ingest.wrfinput import _read_numeric
        with netcdf_bridge.open_dataset(restored.path) as dataset:
            for name in selected:
                if name not in missing:
                    continue
                variable = dataset.variables.get(name)
                if variable is None:
                    continue
                dimensions = tuple(variable.dimensions)
                if dimensions and dimensions[0] == "Time":
                    dimensions = dimensions[1:]
                expected = tuple((category, "south_north", "west_east")
                                 for category in selected[name])
                if dimensions not in expected:
                    raise ValueError(
                        f"{name} axes {dimensions} != {expected[0]}; RUC "
                        "fractions would be assigned to different columns")
                fields[name] = _read_numeric(variable)
    return ruc_mosaic_physics_inputs(cfg, fields, processed=True)
