"""Locate original soil inputs; numeric recovery belongs to the Rust reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from gpuwm.filesystem_paths import canonical_path, io_path

AUTHORITY_SCHEMA = "gpuwm-wrf-soil-authority-v1"


def _sha(path: Path) -> str:
    with io_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authority_from_vtable(path: Path) -> dict:
    """Read declared WPS quantities and layer bounds, without decoding data.

    GRIB1 level type 112 specifies centimetre layer bounds. Temperature
    selectors and truncated field suffixes do not establish the metgrid
    interpolation coordinate. Rust reads the actual SOIL_LEVELS axis and
    verifies its sorted depths against these source layer bounds.
    """
    water = {}
    for line in io_path(path).read_text(encoding="utf-8").splitlines():
        columns = [item.strip() for item in line.split("|")]
        if len(columns) < 7:
            continue
        name = columns[4]
        if not re.fullmatch(r"SOILM\d+", name):
            continue
        try:
            level_type = int(columns[1])
            if level_type != 112:
                raise ValueError("soil water layer bounds must use level type 112")
            if name in water:
                raise ValueError(f"duplicate soil water {name}")
            water[name] = (float(columns[2]) / 100.0,
                           float(columns[3]) / 100.0, columns[5])
        except ValueError as error:
            raise ValueError(f"{path}: invalid {name} source authority: {error}") from error
    if not water:
        raise ValueError(f"{path}: soil recovery needs SOILM fields with source layer bounds")
    ordered = sorted(water, key=lambda key: water[key][0])
    units = {water[key][2] for key in ordered}
    if len(units) != 1:
        raise ValueError(f"{path}: source soil layers declare inconsistent water units")
    unit = units.pop()
    compact = re.sub(r"[\s^*()]", "", unit.lower())
    quantities = {"kgm-2": "layer_water_mass", "kg/m2": "layer_water_mass",
                  "m": "equivalent_water_depth", "mm": "equivalent_water_depth",
                  "m3m-3": "volume_fraction", "m3/m3": "volume_fraction"}
    if compact not in quantities:
        raise ValueError(f"{path}: unsupported declared soil-water units {unit!r}")
    return {
        "schema": AUTHORITY_SCHEMA,
        "source_variable": "SOILM", "source_depth_variable": "SOIL_LEVELS",
        "source_quantity": quantities[compact], "source_units": unit,
        "source_layer_bounds_m": [[water[key][0], water[key][1]] for key in ordered],
        "source_depths_from_metgrid": True,
    }


def recover_supplied_soil(path, raw, attributes, *, source_directory=None):
    """Recover impossible mislabeled total water using original source inputs.

    Already physical volume fractions do not inspect additional inputs or
    change. Missing original evidence produces an actionable input error,
    before the caller initializes physics or launches a forecast.
    """
    if "SMOIS" not in raw or "LANDMASK" not in raw:
        return {}, None
    moisture = np.asarray(raw["SMOIS"])
    land = np.asarray(raw["LANDMASK"]) > .5
    if moisture.ndim != 3 or land.shape != moisture.shape[-2:]:
        return {}, None  # The main reader owns malformed domain geometry.
    if not np.any(moisture[:, land] > 1.0):
        return {}, None

    supplied_path = Path(path)
    directory = canonical_path(source_directory if source_directory is not None else supplied_path.parent)
    path = canonical_path(supplied_path)
    remedy = ("Keep the matching first met_em file and its producing Vtable beside the WRF inputs, "
              "or pass --soil-source DIR for their original directory. ArWen will recover the "
              "source layers automatically. Alternatively initialize directly from the source "
              "model through native preparation. The final four WRF layers cannot identify "
              "the source layer amounts on their own.")
    prefix = f"{supplied_path.name}: land soil moisture exceeds a volume fraction of 1; an upstream layer-water conversion may be missing. "
    domain = re.fullmatch(r"wrfinput_(d\d{2})", supplied_path.name)
    start = str(attributes.get("START_DATE", ""))
    if domain is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", start):
        raise ValueError(prefix + "The source domain and initialization time are not available. " + remedy)
    met_paths = [directory / f"met_em.{domain[1]}.{clock}.nc"
                 for clock in (start, start.replace(":", "_"))]
    present = [item for item in met_paths if io_path(item).is_file()]
    if not present:
        raise ValueError(prefix + "The matching source met_em file is missing. " + remedy)
    if len(present) > 1 and len({_sha(item) for item in present}) != 1:
        raise ValueError(prefix + "Two matching met_em filenames contain different data. " + remedy)
    met_path = present[0]
    # A normal WRF run links met_em from its producing WPS directory.
    # Follow that exact source link to find its table, without searching
    # unrelated runs or assuming that all nearby Vtables are equivalent.
    authority_roots = tuple(dict.fromkeys((directory, canonical_path(met_path).parent)))
    authority_paths = [root / name for root in authority_roots for name in (
        f"wrf-soil-authority.{domain[1]}.json", "wrf-soil-authority.json", "Vtable")]
    authority_path = next((item for item in authority_paths if io_path(item).is_file()), None)
    if authority_path is None:
        raise ValueError(prefix + "The producing Vtable or source-layer authority is missing. " + remedy)
    input_paths = {"wrfinput": path, "met_em": met_path, "authority": authority_path}
    hashes = {name: _sha(item) for name, item in input_paths.items()}
    if authority_path.name == "Vtable":
        authority = authority_from_vtable(authority_path)
    else:
        authority = json.loads(io_path(authority_path).read_text(encoding="utf-8"))
        if not isinstance(authority, dict) or authority.get("schema") != AUTHORITY_SCHEMA:
            raise ValueError(f"{authority_path}: expected {AUTHORITY_SCHEMA} source layer authority")
    from gpuwm.netcdf_bridge import recover_wrf_soil
    values, receipt = recover_wrf_soil(path, met_path, authority)
    if set(values) - {"SMOIS", "SH2O"} or "SMOIS" not in values:
        raise ValueError("soil recovery did not return the declared soil-water fields")
    for name, value in values.items():
        if name not in raw or value.shape != raw[name].shape:
            raise ValueError(f"soil recovery {name} shape does not match the WRF input")
    if any(_sha(item) != hashes[name] for name, item in input_paths.items()):
        raise ValueError("soil recovery inputs changed while being read; launch again with stable inputs")
    receipt = dict(receipt, input_files={
        name: {"path": str(item), "sha256": hashes[name]}
        for name, item in input_paths.items()}, authority=authority)
    return values, receipt
