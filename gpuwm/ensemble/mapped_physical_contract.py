"""Physical interpretation from a validated mapped-source canonical contract."""
from __future__ import annotations

from pathlib import Path

from gpuwm.ensemble.physical_store import (
    FIELD_SCHEMA, canonical_grid_sha256, digest_file, validate_field_contract,
)

# The regular-source join and horizontal mapper explicitly implement this
# table. A source's validated canonical contract supplies every target unit.
_OUTPUTS = {
    "air_temperature": "TT", "air_pressure": "PRES", "specific_humidity": "SPFH",
    "eastward_wind": "UU", "northward_wind": "VV", "geopotential_height": "GHT",
    "surface_pressure": "PSFC", "terrain_height": "SOURCE_OROGRAPHY",
    "skin_temperature": "SKINTEMP", "air_temperature_2m": "T2", "specific_humidity_2m": "Q2",
    "eastward_wind_10m": "U10", "northward_wind_10m": "V10", "land_fraction": "LANDSEA",
    "snow_water_equivalent": "SNOW", "snow_depth": "SNOWH", "sea_ice_fraction": "XICE",
    "sea_surface_temperature": "SST", "lake_water_temperature": "LAKE_WATER_TEMP",
    "soil_temperature": "RW_SOIL_TEMPERATURE", "volumetric_soil_moisture": "RW_SOIL_MOISTURE",
    "cloud_water_mixing_ratio": "QC", "cloud_ice_mixing_ratio": "QI",
    "rain_water_mixing_ratio": "QR", "snow_mixing_ratio": "QS", "graupel_or_hail_mixing_ratio": "QG",
}
_HYDROMETEORS = {"cloud_water_mixing_ratio", "cloud_ice_mixing_ratio", "rain_water_mixing_ratio",
                  "snow_mixing_ratio", "graupel_or_hail_mixing_ratio"}
_WINDS = {"eastward_wind": ("northward_wind", "grid_x"),
          "northward_wind": ("eastward_wind", "grid_y"),
          "eastward_wind_10m": ("northward_wind_10m", "grid_x"),
          "northward_wind_10m": ("eastward_wind_10m", "grid_y")}


def mapped_physical_evidence_files(mapping_path, composition_path):
    from gpuwm import mapped_source, mapped_composition
    from gpuwm.ingest import horiz, water_temperature, soil_contract
    return {"source_mapping": Path(mapping_path), "source_composition": Path(composition_path),
            "canonical_regular_join": Path(mapped_source.__file__),
            "canonical_composition_join": Path(mapped_composition.__file__),
            "native_horizontal_mapper": Path(horiz.__file__),
            "water_temperature_assembly": Path(water_temperature.__file__),
            "soil_coordinate_contract": Path(soil_contract.__file__),
            "physical_field_contract_source": Path(__file__)}


def mapped_physical_field_contract(grid_identity, *, mapping_path, composition_path,
                                   source_identity, extra_evidence=None):
    """Revalidate canonical units, source axes and the ordinary native join.

    Canonical geopotential_height has already reached metres in the native
    decoder. This route passes GHT through; it never applies the Z/9.81
    conversion that belongs to a different regular-source input spelling.
    """
    from gpuwm import mapped_source, mapped_composition

    mapping_path, composition_path = Path(mapping_path), Path(composition_path)
    if source_identity.get("adapter") != "rw-wps-mapped-composition-v2":
        raise ValueError("mapped physical fields require the ordinary canonical source adapter")
    for key, path in (("mapping_sha256", mapping_path), ("composition_sha256", composition_path)):
        if digest_file(path) != source_identity.get(key):
            raise ValueError("mapped physical field definition differs from captured source authority")
    mapping = mapped_source.load_mapping(mapping_path)
    mapped_composition.load_composition(composition_path, mapping_path)
    evidence = {role: digest_file(path) for role, path in
                mapped_physical_evidence_files(mapping_path, composition_path).items()}
    for role, value in (extra_evidence or {}).items():
        if role in evidence and evidence[role] != value:
            raise ValueError("mapped physical field evidence has conflicting authorities")
        evidence[role] = value

    def row(units, dims, sources, operation, basis="scalar", source_units=None):
        return {"units": units, "dimensions": dims, "basis": basis, "source_fields": sources,
                "operation": operation, "source_units": units if source_units is None else source_units}

    arrays = {"levels_hpa": row("hPa", ["level"], ["air_pressure"],
              "Native regular-source join: representative column pressure in Pa divided by 100; "
              "the full physical column coordinate remains field__PRES.", source_units="Pa")}
    fields = mapping["fields"]
    policies = mapping["target"].get("initialization_policies", {})
    for canonical, output in _OUTPUTS.items():
        if canonical not in fields:
            if canonical not in _HYDROMETEORS:
                continue
            if policies.get(canonical) != "explicit_zero_with_adapter_validation":
                raise ValueError(f"mapped physical field {canonical} lacks a declared source or zero policy")
            arrays["field__"+output] = row("kg kg-1", ["level", "y", "x"], [canonical],
                "Canonical source explicitly omits this analyzed species; ordinary adapter supplies native zeros.")
            continue
        declared = fields[canonical]
        units = declared["units"]["target"]
        dims = [{"vertical": "level", "soil": "soil_level"}.get(axis, axis)
                for axis in declared["target_axes"]]
        basis, sources = "scalar", [canonical]
        operation = "Validated canonical target units retained through ordinary native horizontal mapping."
        if canonical in _WINDS:
            partner, basis = _WINDS[canonical]
            if partner not in fields or fields[partner]["units"]["target"] != units or units != "m s-1":
                raise ValueError("mapped vector pair lacks a common canonical speed authority")
            dims[-2:] = ["y", "x_stag"] if basis == "grid_x" else ["y_stag", "x"]
            sources = sorted((canonical, partner))
            operation = ("Canonical earth-relative vector pair is interpolated to target C-grid faces "
                         "by the native parabolic operator, then rotated into target grid basis.")
        elif canonical == "land_fraction":
            sources = ["target static LANDMASK"]
            operation = "Actual target static land/water classification replaces source land fraction."
        elif canonical == "geopotential_height":
            operation = "Canonical geopotential height in metres maps as GHT; no second conversion or division."
        elif canonical in {"soil_temperature", "volumetric_soil_moisture"}:
            operation = ("Canonical soil-layer order from verified composition; native masked land interpolation "
                         "and ordinary source-land repair/fill policy; units retained.")
        arrays["field__"+output] = row(units, dims, sources, operation, basis,
                                     source_units=declared["units"]["source"])
    for name, units, sources, operation in (
            ("water_temperature", "K", ["skin_temperature", "sea_surface_temperature", "lake_water_temperature"],
             "Native water-temperature policy selects source providers on the actual target static classes."),
            ("water_temperature_source", "1", ["native water-temperature provider table"],
             "Categorical provider identifier from the verified water-temperature assembly."),
            ("soil_no_source_land", "1", ["native soil donor search"],
             "Boolean target-land mask where the ordinary native search found no source-land donor.")):
        arrays["meta__"+name] = row(units, ["y", "x"], sources, operation)
    contract = {"schema": FIELD_SCHEMA, "grid_sha256": canonical_grid_sha256(grid_identity),
                "vertical": {"kind": ("pressure_levels" if mapping["coordinates"]["vertical"]["kind"] == "pressure"
                                       else "representative_pressure_levels"),
                             "units": "hPa", "values": "levels_hpa", "pressure_field": "field__PRES"},
                "arrays": arrays, "evidence": evidence}
    return validate_field_contract(contract, grid_identity)


def require_mapped_physical_field_contract(store, expected):
    """The native consumer refuses numerically unchanged but mislabelled fields."""
    actual = store.require_field_contract()
    if actual["vertical"] != expected["vertical"]:
        raise ValueError("mapped physical input vertical coordinate differs from its native source contract")
    present = set().union(*(frame["arrays"] for frame in store.document["frames"]))
    for name in present:
        if name not in expected["arrays"] or any(actual["arrays"][name][key] != expected["arrays"][name][key]
                for key in ("units", "dimensions", "basis")):
            raise ValueError(f"mapped physical input field {name} units or coordinates differ from native mapping")
    return actual
