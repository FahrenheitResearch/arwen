"""Author the packaged Earth-2 profile documents from the Earth-2 model table.

``gpuwm/authorities/earth2-models.v1.json`` is the source of truth: the
Earth2Studio vocabulary, its units, and each profile's level families.
This writes, for every profile row whose status is ``runnable``, the three
packaged authorities (mapping, composition, provenance) and the ERA5 ARCO
NetCDF companion-analysis mapping they borrow the land surface from.  The
documents are table data generated from table data; ``--check`` proves the
shipped bytes are what the table produces.

    python tools/author_earth2_profile.py [--check]
"""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "gpuwm" / "authorities"
TABLE = ROOT / "earth2-models.v1.json"
DONOR_NAME = "ecmwf-era5-arco-netcdf-analysis-donor"
GRAVITY_INVERSE = 0.1019716212977928  # 1 / 9.80665, the ERA5/GRIB tables' factor

#: ERA5 ARCO soil layers (cm) as `gpuwm fetch --source era5 --era5-provider
#: arco` names them, and their depths in metres.
_ERA5_SOIL = (("000007", 0.0, 0.07), ("007028", 0.07, 0.28),
              ("028100", 0.28, 1.0), ("100289", 1.0, 2.89))
_NOAH_LAYERS = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))

_POLICY_FIELDS = [
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio", "cloud_ice_mixing_ratio",
    "snow_mixing_ratio", "graupel_or_hail_mixing_ratio", "vertical_velocity",
    "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
]
_REQUIRED = [
    ("air_temperature", "vertical", "mass", "K"),
    ("specific_humidity", "vertical", "mass", "kg kg-1"),
    ("eastward_wind", "vertical", "mass", "m s-1"),
    ("northward_wind", "vertical", "mass", "m s-1"),
    ("geopotential_height", "vertical", "mass", "m"),
    ("air_pressure", "vertical", "mass", "Pa"),
    ("surface_pressure", None, "surface", "Pa"),
    ("terrain_height", None, "surface", "m"),
    ("skin_temperature", None, "surface", "K"),
    ("air_temperature_2m", None, "surface", "K"),
    ("specific_humidity_2m", None, "surface", "kg kg-1"),
    ("eastward_wind_10m", None, "surface", "m s-1"),
    ("northward_wind_10m", None, "surface", "m s-1"),
    ("land_fraction", None, "surface", "1"),
    ("soil_temperature", "soil", "soil", "K"),
    ("volumetric_soil_moisture", "soil", "soil", "m3 m-3"),
]
# Earth2Studio's lexicons carry one field under more than one key (HRRRLexicon: "msl" and the deprecated "mslp"), and a
# model package's own metadata picks one of them: nvidia/stormcast-v1-era5-hrrr's metadata.zarr names "mslp" where the
# ERA5-lineage models write "msl". Breakage it prevents: WOOF run_025e9dd2419ae75e (2026-10-26) ran StormCast and then
# stopped in prep at "mapped field(s) ['air_pressure_at_mean_sea_level'] have no matching variable in any of the 2
# supplied NetCDF file(s)" because the mapping accepted only "msl". Each selector accepts every key of its field.
_PRODUCER_ALIASES = {"msl": ("msl", "mslp")}

_TARGET_UNITS = {
    "geopotential_height": "m", "air_temperature": "K", "eastward_wind": "m s-1",
    "northward_wind": "m s-1", "specific_humidity": "kg kg-1",
    "air_temperature_2m": "K", "eastward_wind_10m": "m s-1",
    "northward_wind_10m": "m s-1", "air_pressure_at_mean_sea_level": "Pa",
    "air_pressure": "Pa",
    "relative_humidity": "%", "surface_pressure": "Pa",
}
#: The companion fields, their canonical shape, and their missing policy
#: (identical on both sides of the binding, which the composition checks).
_BORROWED = {
    "terrain_height": ("surface", "m", "reject"),
    "land_fraction": ("surface", "1", "reject"),
    "skin_temperature": ("surface", "K", "reject"),
    "specific_humidity_2m": ("surface", "kg kg-1", "reject"),
    "soil_temperature": ("soil", "K", "preserve_mask"),
    "volumetric_soil_moisture": ("soil", "m3 m-3", "preserve_mask"),
    "sea_surface_temperature": ("surface", "K", "preserve_mask"),
    "sea_ice_fraction": ("surface", "1", "preserve_mask"),
    "snow_water_equivalent": ("surface", "kg m-2", "reject"),
}


_BORROWED.update({
    "surface_pressure": ("surface", "Pa", "reject"),
    "snow_depth": ("surface", "m", "reject"),
})


def _required_fields():
    rows = []
    for name, extra, location, units in _REQUIRED:
        axes = [extra, "y", "x"] if extra else ["y", "x"]
        rows.append({"name": name, "axes": axes, "location": location, "target_units": units})
    return rows


def _target(name, interval):
    return {
        "name": name,
        "physics_suite": "source-independent",
        "max_dom": 4,
        "require_lateral_boundaries": True,
        "target_vertical_levels": 49,
        "soil_layer_count": 4,
        "boundary_interval_seconds": interval,
        "accept_boundary_interval_multiples": True,
        "required_fields": _required_fields(),
        "pressure_requirement": "air_pressure",
        "policy_controlled_fields": list(_POLICY_FIELDS),
    }


def _bound(name):
    location, units, missing = _BORROWED[name]
    axes = ["soil", "y", "x"] if location == "soil" else ["y", "x"]
    return {"provider": "composition_bound", "units": {"source": units, "target": units},
            "source_axes": axes, "target_axes": axes, "location": location,
            "staggering": "none", "missing": {"kind": missing}}


def profile_mapping(table, profile_id):
    profile = table["profiles"][profile_id]
    units = table["vocabulary_units"]
    levels = list(profile.get("level_values", profile.get("levels_hpa", [])))
    producer = {"lat": {"units": units["lat"]}, "lon": {"units": units["lon"]}}
    fields = {}
    for family, canonical in profile["level_families"].items():
        selectors = []
        for level in profile.get("level_selection", {}).get(family, levels):
            variable = f"{family}{level}" + profile.get("level_suffix", "")
            producer[variable] = {"units": units[family]}
            selectors.append({"format": "netcdf", "name": [variable], "level_value": level})
        unit = {"source": units[family], "target": _TARGET_UNITS[canonical]}
        if family == "z":
            unit["scale"] = GRAVITY_INVERSE
        fields[canonical] = {
            "selectors": selectors, "selector_stack_axis": "vertical", "units": unit,
            "source_axes": ["time", "lead", "vertical", "y", "x"],
            "target_axes": ["vertical", "y", "x"], "location": "mass",
            "staggering": "none", "missing": {"kind": "reject"},
        }
    if profile.get("level_selection"):
        for name, field in fields.items():
            field["missing"] = {"kind": "composition_fill"}
            if name in (rule["primary_field"] for output, rule in profile["vertical_merge"]["fields"].items() if output != rule["primary_field"]):
                field["dependency_only"] = True
        for name in profile.get("composition_fields", []):
            fields[name] = {"provider": "composition_bound", "units": {"source": _TARGET_UNITS[name], "target": _TARGET_UNITS[name]},
                "source_axes": ["vertical", "y", "x"], "target_axes": ["vertical", "y", "x"],
                "location": "mass", "staggering": "none", "missing": {"kind": "reject"}}
    if "air_pressure" not in fields:
        fields["air_pressure"] = {
            "selectors": [], "derivation": "pressure-from-coordinate",
            "units": {"source": "hPa", "target": "Pa", "scale": 100.0, "offset": 0.0},
            "source_axes": ["vertical", "y", "x"], "target_axes": ["vertical", "y", "x"],
            "location": "mass", "staggering": "none", "missing": {"kind": "reject"},
        }
    humidity_derivation = profile.get("humidity_derivation")
    if humidity_derivation:
        fields["specific_humidity"] = {
            "selectors": [], "derivation": "humidity-from-rh",
            "units": {"source": "kg kg-1", "target": "kg kg-1"},
            "source_axes": ["vertical", "y", "x"],
            "target_axes": ["vertical", "y", "x"], "location": "mass",
            "staggering": "none", "missing": {"kind": "reject"},
        }
    for variable, canonical in profile["surface"].items():
        names = list(_PRODUCER_ALIASES.get(variable, (variable,)))
        for name in names:
            producer[name] = {"units": units[variable]}
        fields[canonical] = {
            "selectors": [{"format": "netcdf", "name": names}],
            "units": {"source": units[variable], "target": _TARGET_UNITS[canonical]},
            "source_axes": ["time", "lead", "y", "x"], "target_axes": ["y", "x"],
            "location": "surface", "staggering": "none", "missing": {"kind": "reject"},
        }
    fields["surface_pressure"] = {
        "selectors": [], "derivation": "surface-pressure-from-sea-level",
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "staggering": "none", "missing": {"kind": "reject"},
    }
    donor_mapping = physical_donor(table, profile["donor"])
    donor_fields = donor_mapping["fields"]
    for name in profile["companion_analysis"]["fields"]:
        fields[name] = _bound(name)
        # Bind the effective donor contract, including inherited base metadata.
        fields[name]["missing"] = copy.deepcopy(donor_fields[name]["missing"])
    for variable in profile.get("diagnostic_channels", {}):
        producer[variable] = {"units": units[variable]}
        fields[variable] = {
            "selectors": [{"format": "netcdf", "name": [variable]}],
            "units": {"source": units[variable], "target": units[variable]},
            "source_axes": ["time", "lead", "y", "x"], "target_axes": ["y", "x"],
            "location": "surface", "staggering": "none",
            "missing": {"kind": "preserve_mask"}, "dependency_only": True,
        }
    target = _target("Earth2Studio forecast atmosphere with same-cycle physical analysis", profile["boundary_interval_seconds"])
    if {"soil_temperature", "volumetric_soil_moisture"}.issubset(profile["companion_analysis"]["fields"]):
        # Canonical source frames retain donor samples. The composition's
        # soil contract remaps these to the land model's layers downstream.
        target["soil_layer_count"] = donor_mapping["target"]["soil_layer_count"]
    target["initialization_policies"] = {
        "cloud_water_mixing_ratio": "explicit_zero_with_adapter_validation",
        "rain_water_mixing_ratio": "explicit_zero_with_adapter_validation",
        "cloud_ice_mixing_ratio": "explicit_zero_with_adapter_validation",
        "snow_mixing_ratio": "explicit_zero_with_adapter_validation",
        "graupel_or_hail_mixing_ratio": "explicit_zero_with_adapter_validation",
        "vertical_velocity": "explicit_zero_with_adapter_validation",
        "snow_water_equivalent": "borrowed_from_companion_analysis",
        "snow_depth": "not_published_by_the_ai_model",
        "sea_ice_fraction": "borrowed_from_companion_analysis",
    }
    document = {
        "schema": "rw-wps.mapping.v1",
        "name": profile_id.removesuffix("-v1"),
        "format": "netcdf",
        "producer_attributes": producer,
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": {"format": "netcdf", "name": ["lat"]},
                "longitude": {"format": "netcdf", "name": ["lon"]},
            },
            "vertical": {
                "kind": "pressure", "units": "hPa", "positive": "down", "levels": levels,
                "selector": {"format": "netcdf", "name": ["level"]},
            },
            "time": {
                "kind": "dimension",
                "units": "hours since 0001-01-01 00:00:00.0",
                "calendar": "proleptic_gregorian",
                "selector": {"name": ["time"]},
                "lead": {"selector": {"name": ["lead_time"]}, "units": "hours"},
            },
        },
        "fields": fields,
        "derivations": [
            {"name": "pressure-from-coordinate", "operation": "pressure_from_vertical_coordinate"},
            {"name": "surface-pressure-from-sea-level",
             "operation": "surface_pressure_from_sea_level",
             "sea_level_pressure": "air_pressure_at_mean_sea_level",
             "level_height": "geopotential_height", "pressure": "air_pressure",
             "surface_height": "terrain_height"},
        ],
        "target": target,
    }

    if profile.get("vertical_coordinate"):
        document["coordinates"]["vertical"].update(copy.deepcopy(profile["vertical_coordinate"]))
    if profile.get("grid"):
        document["grid"] = copy.deepcopy(profile["grid"])
    document["target"]["initialization_policies"].update(profile.get("initialization_policies", {}))
    document["derivations"] = [rule for rule in document["derivations"]
                               if any(field.get("derivation") == rule["name"] for field in fields.values())]
    if humidity_derivation:
        document["derivations"].append({
            "name": "humidity-from-rh", "operation": humidity_derivation,
            "relative_humidity": "relative_humidity",
            "temperature": "air_temperature", "pressure": "air_pressure",
        })
    return document


def donor_mapping():
    """ERA5 as `gpuwm fetch --source era5 --era5-provider arco` writes it."""

    def surface(variable, units, target_units=None, scale=None, missing="reject"):
        unit = {"source": units, "target": target_units or units}
        if scale is not None:
            unit["scale"] = scale
        return {"selectors": [{"format": "netcdf", "name": [variable]}], "units": unit,
                "source_axes": ["time", "y", "x"], "target_axes": ["y", "x"],
                "location": "surface", "staggering": "none", "missing": {"kind": missing}}

    def level(variable, units, target_units=None, scale=None):
        unit = {"source": units, "target": target_units or units}
        if scale is not None:
            unit["scale"] = scale
        return {"selectors": [{"format": "netcdf", "name": [variable]}], "units": unit,
                "source_axes": ["time", "vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"], "location": "mass",
                "staggering": "none", "missing": {"kind": "reject"}}

    def soil(prefix, units):
        return {"selectors": [{"format": "netcdf", "name": [prefix + code]} for code, _, _ in _ERA5_SOIL],
                "selector_stack_axis": "soil", "units": {"source": units, "target": units},
                "source_axes": ["time", "soil", "y", "x"], "target_axes": ["soil", "y", "x"],
                "location": "soil", "staggering": "none", "missing": {"kind": "preserve_mask"}}

    fields = {
        "geopotential_height": level("Z", "m2 s-2", "m", GRAVITY_INVERSE),
        "air_temperature": level("T", "K"),
        "specific_humidity": level("SPFH", "kg kg-1"),
        "eastward_wind": level("U", "m s-1"),
        "northward_wind": level("V", "m s-1"),
        "air_pressure": {
            "selectors": [], "derivation": "pressure-from-coordinate",
            "units": {"source": "hPa", "target": "Pa", "scale": 100.0, "offset": 0.0},
            "source_axes": ["vertical", "y", "x"], "target_axes": ["vertical", "y", "x"],
            "location": "mass", "staggering": "none", "missing": {"kind": "reject"},
        },
        "surface_pressure": surface("PSFC", "Pa"),
        "terrain_height": surface("SOILGEO", "m2 s-2", "m", GRAVITY_INVERSE),
        "skin_temperature": surface("SKINTEMP", "K"),
        "air_temperature_2m": surface("T2", "K"),
        "specific_humidity_2m": surface("Q2", "kg kg-1"),
        "eastward_wind_10m": surface("U10", "m s-1"),
        "northward_wind_10m": surface("V10", "m s-1"),
        "land_fraction": surface("LANDSEA", "1"),
        "soil_temperature": soil("ST", "K"),
        "volumetric_soil_moisture": soil("SM", "m3 m-3"),
        "sea_surface_temperature": surface("SST", "K", missing="preserve_mask"),
        "sea_ice_fraction": surface("SEAICE", "1", missing="preserve_mask"),
        "snow_water_equivalent": surface("SNOW_EC", "m", "kg m-2", 1000.0),
    }
    target = _target(
        "gpuwm/WRF-real ECMWF ERA5 (Google ARCO public Zarr as gpuwm fetch writes it), "
        "authored as the cross-source land-surface donor for AI-model atmospheres", 21600)
    target["initialization_policies"] = {
        "cloud_water_mixing_ratio": "explicit_zero_with_adapter_validation",
        "rain_water_mixing_ratio": "explicit_zero_with_adapter_validation",
        "cloud_ice_mixing_ratio": "explicit_zero_with_adapter_validation",
        "snow_mixing_ratio": "explicit_zero_with_adapter_validation",
        "graupel_or_hail_mixing_ratio": "explicit_zero_with_adapter_validation",
        "vertical_velocity": "explicit_zero_with_adapter_validation",
        "snow_water_equivalent": "published_snow_depth_water_equivalent",
        "snow_depth": "not_selected",
        "sea_ice_fraction": "published_sea_ice_cover",
    }
    return {
        "schema": "rw-wps.mapping.v1",
        "name": DONOR_NAME,
        "format": "netcdf",
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": {"format": "netcdf", "name": ["latitude"]},
                "longitude": {"format": "netcdf", "name": ["longitude"]},
            },
            "vertical": {
                "kind": "pressure", "units": "hPa", "positive": "down",
                "levels": [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225,
                           250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775,
                           800, 825, 850, 875, 900, 925, 950, 975, 1000],
                "selector": {"format": "netcdf", "name": ["level"]},
            },
            "time": {"kind": "dimension", "units": "seconds since 1970-01-01 00:00:00",
                     "selector": {"name": ["time"]}},
        },
        "fields": fields,
        "derivations": [
            {"name": "pressure-from-coordinate", "operation": "pressure_from_vertical_coordinate"},
        ],
        "target": target,
    }


def composition(donor_sha256, fields):
    def soil_selectors(code):
        return {"soil_temperature": {"format": "netcdf", "name": ["ST" + code]},
                "volumetric_soil_moisture": {"format": "netcdf", "name": ["SM" + code]}}

    return {
        "schema": "gpuwm-mapped-composition-v2",
        "name": "earth2studio-atmosphere-with-initialization-era5-land-surface",
        "mapping_binding": "input_manifest_sha256",
        "supplements": {},
        "field_sources": {
            "physical_analysis_surface": {
                "source_id": DONOR_NAME,
                "mapping_role": "physical_analysis_surface_mapping",
                "mapping_sha256": donor_sha256,
                "data_role": "physical_analysis_surface_data",
                "provenance_role": "physical_analysis_surface_provenance",
                "fields": list(fields),
                "grid_alignment": "bilinear_mask_aware_nearest",
                "time_alignment": "source_cycle_analysis_broadcast",
            },
        },
        "soil_layers": {
            "temperature_field": "soil_temperature",
            "moisture_field": "volumetric_soil_moisture",
            "depth_units": "m",
            "source_layers": [
                {"top": top, "bottom": bottom, "selectors": soil_selectors(code)}
                for code, top, bottom in _ERA5_SOIL
            ],
            "target_layers": [{"top": top, "bottom": bottom} for top, bottom in _NOAH_LAYERS],
            "remap": {"kind": "conservative_layer_means", "source_value_location": "layer_mean",
                      "target_value_location": "layer_mean", "coverage": "require_complete"},
            "missing": {"land": "reject",
                        "ocean": {"stage": "after_horizontal_interpolation",
                                  "temperature": "skin_temperature", "moisture": 1.0}},
        },
    }


def provenance(table, profile_id):
    profile = table["profiles"][profile_id]
    spec = table["donor_policy"]["donors"][profile["donor"]]
    result = {
        "schema": "gpuwm-cross-source-hybrid-provenance-v1",
        "source": "Earth2Studio atmosphere composed with a same-cycle physical analysis",
        "profile": profile_id,
        "contributing_sources": {
            "primary_atmosphere": {
                "product": "Earth2Studio NetCDF4Backend forecast output",
                "models_with_this_vocabulary": sorted(
                    name for name, row in table["models"].items()
                    if row.get("profile") == profile_id),
                "level_families": profile["level_families"],
                "levels_hpa": profile.get("levels_hpa"),
                "level_selection": profile.get("level_selection"),
                "surface_fields": profile["surface"],
                "units": "exact producer vocabulary declared by the mapping; conflicting file attributes refuse",
            },
            "physical_analysis_surface": {
                "product": spec["source"], "provider": spec["provider"],
                "mapping": spec["mapping"],
                "supplies": profile["companion_analysis"]["fields"],
                "clock": profile["companion_analysis"]["clock"],
                "grid_alignment": profile["companion_analysis"]["grid_alignment"],
                "surface_temperature_policy": spec.get("surface_temperature_policy", "published donor SST"),
            },
        },
        "why_hybrid": profile["companion_analysis"]["why"],
        "surface_pressure_policy": profile["derived"]["surface_pressure"],
        "frozen_surface_caveat": "The initialization analysis is carried unchanged while the AI atmosphere advances. The composition receipt names each carried valid time.",
        "input_identity": "File paths and SHA-256 values are sealed by gpuwm-mapped-source-inputs-v1",
    }

    if profile.get("level_values"):
        primary = result["contributing_sources"]["primary_atmosphere"]
        primary.pop("levels_hpa")
        primary["level_values"] = profile["level_values"]
        primary["vertical_coordinate"] = profile["vertical_coordinate"]
        primary["level_suffix"] = profile.get("level_suffix", "")
    if profile.get("diagnostic_channels"):
        result["contributing_sources"]["primary_atmosphere"]["diagnostic_channels"] = profile["diagnostic_channels"]
    return result


def _dump(document) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def physical_donor(table, donor_id):
    spec = table["donor_policy"]["donors"][donor_id]
    if "mapping_base" not in spec:
        return donor_mapping()
    donor = json.loads((ROOT / spec["mapping_base"]).read_text())
    donor["name"] = "earth2-" + donor_id + "-physical-analysis-donor"
    if "soil_layers" in spec:
        donor["target"]["soil_layer_count"] = len(spec["soil_layers"].get("source_layers", spec["soil_layers"].get("source_nodes", [])))
    if "vertical_levels_pa" in spec:
        donor["coordinates"]["vertical"]["levels"] = spec["vertical_levels_pa"]
    donor["fields"].update(copy.deepcopy(spec.get("field_overrides", {})))
    donor.setdefault("derivations", []).extend(spec.get("derivations_append", []))
    donor["target"]["initialization_policies"]["snow_water_equivalent"] = "borrowed_from_same_cycle_analysis"
    donor["target"]["initialization_policies"]["sea_ice_fraction"] = "borrowed_from_same_cycle_analysis"
    return donor


def variant_composition(table, profile, donor, donor_sha):
    fields = list(profile["companion_analysis"]["fields"])
    merge = profile.get("vertical_merge")
    if merge:
        fields += list(merge["fields"])
    result = composition(donor_sha, fields)
    binding = result["field_sources"]["physical_analysis_surface"]
    binding["source_id"] = donor["name"]
    binding["grid_alignment"] = profile["companion_analysis"]["grid_alignment"]
    if merge:
        binding["vertical_merge"] = merge
    spec = table["donor_policy"]["donors"][profile["donor"]]
    if "soil_layers" in spec:
        result["soil_layers"] = copy.deepcopy(spec["soil_layers"])
    result["name"] = "earth2-atmosphere-with-same-cycle-" + profile["donor"] + "-donor"
    return result


def documents():
    import hashlib
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    out = {}
    donors = {}
    for donor_id, spec in table["donor_policy"]["donors"].items():
        doc = physical_donor(table, donor_id)
        data = _dump(doc)
        out[spec["mapping"]] = data
        donors[donor_id] = (doc, hashlib.sha256(data).hexdigest())
    for profile_id, profile in table["profiles"].items():
        if profile.get("status") != "runnable":
            continue
        stem = "rw-wps-" + profile_id.removesuffix("-v1")
        doc, donor_sha = donors[profile["donor"]]
        out[stem + ".mapping.json"] = _dump(profile_mapping(table, profile_id))
        out[stem + ".composition.json"] = _dump(variant_composition(table, profile, doc, donor_sha))
        prov = provenance(table, profile_id)
        prov["donor"] = {"id": profile["donor"], "mapping": table["donor_policy"]["donors"][profile["donor"]]["mapping"],
                         "sha256": donor_sha, "clock": "source_cycle_analysis_broadcast"}
        if profile.get("vertical_merge"):
            prov["vertical_merge"] = profile["vertical_merge"]
        out[stem + ".provenance.json"] = _dump(prov)
    return out


def authority_pins(documents):
    """Regenerate the profile registry's digests from the same authored bytes."""
    import hashlib
    import re

    path = ROOT.parent / "source_authorities.py"
    text = path.read_text(encoding="utf-8")
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    for profile_id, profile in table["profiles"].items():
        if profile.get("status") != "runnable":
            continue
        stem = "rw-wps-" + profile_id.removesuffix("-v1")
        pattern = re.compile(r'("' + re.escape(profile_id) + r'": _profile\(.*?\n    \),)', re.S)
        def update(match):
            block = match.group(0)
            for role in ("mapping", "composition", "provenance"):
                digest = hashlib.sha256(documents[stem + "." + role + ".json"]).hexdigest()
                block = re.sub(r'(' + role + r'=")[0-9a-f]{64}("\s*,)',
                               lambda m: m[1] + digest + m[2], block)
            def donor_pin(match):
                name = match[2]
                data = documents.get(name)
                if data is None:
                    data = (ROOT / name).read_bytes()
                return match[1] + hashlib.sha256(data).hexdigest() + match[3]
            return re.sub(r'("file": "([^"]+)",\s*"sha256": ")[0-9a-f]{64}(")', donor_pin, block)
        text, count = pattern.subn(update, text)
        if count != 1:
            raise ValueError(f"profile registry must declare {profile_id} exactly once")
    return path, text.encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="refuse when a shipped document differs from what the table produces")
    args = parser.parse_args(argv)
    import hashlib
    stale = []
    authored = documents()
    outputs = [(ROOT / name, data) for name, data in authored.items()]
    outputs.append(authority_pins(authored))
    for path, data in outputs:
        name = path.name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                stale.append(name)
        else:
            path.write_bytes(data)
        print(f"{hashlib.sha256(data).hexdigest()}  {name}")
    if stale:
        print("stale: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
