"""Write CAMS-shaped GRIB2 files with ecCodes: a test and proof fixture, never an input source.

WHY THIS EXISTS.  The Atmosphere Data Store cannot be read from this project
until the account holder accepts the store's site policies, and no key-free
copy of the CAMS 3-D forecast exists (DESIGN 7.1).  The decode, remap and
boundary machinery still has to be proven on bytes encoded the way ECMWF
encodes CAMS, so this script asks ecCodes -- ECMWF's own GRIB library, the
encoder behind every CAMS file -- to write records of exactly the parameters
the ``cams-global`` and ``cams-oxidants`` source rows select: the WMO
chemical-constituent product template 4.40 for O3/NO2/CO/SO2, ECMWF-local
192/210 and 192/217 records for the aerosols and oxidants, specific humidity
on the 137 hybrid levels with the L137 coefficients in Section 4, and
single-level surface pressure.  ecCodes chooses the templates from the
paramId, so the fixture tests the decoder against ECMWF's encoding rather
than against the decoder's own idea of it.

THE VALUES ARE SYNTHETIC.  Smooth, plausible vertical profiles with a
horizontal gradient so every remap stage has structure to move.  They are not
a forecast and nothing may present them as one; files carry
``SYNTHETIC`` in their name and the manifest says so.

ecCodes is a test-fixture dependency only (``pip install eccodes`` in a
scratch environment); the engine decodes with its own Rust bridge.

Usage:
    python make_synthetic_cams.py OUT_DIR --area N,W,S,E --cycle 2025-07-30T00
        --leads 0,3,6 [--levels 1-137] [--packing simple|ccsds]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np

HERE = Path(__file__).resolve().parent
L137 = json.loads((HERE / "ifs_l137_half_levels.json").read_text())

# paramId -> (key used in the manifest, kind, molar mass g/mol or None)
PARAMS = {
    210203: ("ozone", "gas", 47.997),
    210121: ("nitrogen_dioxide", "gas", 46.005),
    210123: ("carbon_monoxide", "gas", 28.010),
    210122: ("sulphur_dioxide", "gas", 64.058),
    210001: ("sea_salt_0p03_0p5um", "aer", None),
    210002: ("sea_salt_0p5_5um", "aer", None),
    210003: ("sea_salt_5_20um", "aer", None),
    210004: ("dust_0p03_0p55um", "aer", None),
    210005: ("dust_0p55_0p9um", "aer", None),
    210006: ("dust_0p9_20um", "aer", None),
    210007: ("organic_matter_hydrophilic", "aer", None),
    210008: ("organic_matter_hydrophobic", "aer", None),
    210009: ("black_carbon_hydrophilic", "aer", None),
    210010: ("black_carbon_hydrophobic", "aer", None),
    210011: ("sulphate", "aer", None),
    217030: ("hydroxyl_radical", "gas", 17.007),
    217003: ("hydrogen_peroxide", "gas", 34.014),
    217032: ("nitrate_radical", "gas", 62.004),
    133: ("specific_humidity", "q", None),
}
MWDRY = 28.966

# Boundary-layer value (ppb for gases, ug/kg for aerosols), free-troposphere
# value, and an east-west gradient fraction.  Ozone also rises into the
# stratosphere.  Plausible magnitudes, not a forecast.
PROFILE = {
    "ozone": (40.0, 70.0, 0.25), "nitrogen_dioxide": (3.0, 0.05, 0.8),
    "carbon_monoxide": (130.0, 80.0, 0.2), "sulphur_dioxide": (1.0, 0.05, 0.6),
    "hydroxyl_radical": (1e-4, 3e-5, 0.1), "hydrogen_peroxide": (1.0, 0.5, 0.1),
    "nitrate_radical": (5e-3, 1e-3, 0.1),
    "sea_salt_0p03_0p5um": (2.0, 0.1, -0.5), "sea_salt_0p5_5um": (20.0, 0.5, -0.5),
    "sea_salt_5_20um": (15.0, 0.1, -0.5), "dust_0p03_0p55um": (0.5, 0.2, 0.6),
    "dust_0p55_0p9um": (1.5, 0.5, 0.6), "dust_0p9_20um": (12.0, 2.0, 0.6),
    "organic_matter_hydrophilic": (3.0, 0.5, 0.4),
    "organic_matter_hydrophobic": (1.0, 0.2, 0.4),
    "black_carbon_hydrophilic": (0.3, 0.05, 0.4),
    "black_carbon_hydrophobic": (0.2, 0.03, 0.4), "sulphate": (2.0, 0.5, 0.3),
}


def _half_pressure(ps):
    a = np.array([row[1] for row in L137], dtype=np.float64)
    b = np.array([row[2] for row in L137], dtype=np.float64)
    return a[:, None, None] + b[:, None, None] * ps[None]


def _grid(area, dx):
    north, west, south, east = area
    nj = int(round((north - south) / dx)) + 1
    ni = int(round((east - west) / dx)) + 1
    lat = north - dx * np.arange(nj)
    lon = west + dx * np.arange(ni)
    return lat, lon


def _fields(lat, lon, step_h):
    lon2, lat2 = np.meshgrid(lon, lat)
    ps = (101300.0 - 1500.0 * np.exp(-((lon2 - lon.mean()) / 3.0) ** 2)
          + 200.0 * np.sin(math.radians(15.0 * step_h)))
    ph = _half_pressure(ps)
    pf = 0.5 * (ph[:-1] + ph[1:])                     # (137, nj, ni)
    sigma = pf / ps[None]
    east = (lon2 - lon.min()) / max(lon.max() - lon.min(), 1e-9)
    q = 0.016 * sigma ** 3.5 * (0.8 + 0.4 * east[None]) + 3e-6
    out = {"specific_humidity": q}
    pbl = np.clip((sigma - 0.8) / 0.2, 0.0, 1.0)      # 1 near the ground
    for key, (bl, ft, grad) in PROFILE.items():
        value = ft + (bl - ft) * pbl
        value = value * (1.0 + grad * (east[None] - 0.5))
        if key == "ozone":                            # stratospheric ozone
            strat = 8000.0 * np.clip((5000.0 - pf) / 5000.0, 0.0, 1.0) ** 0.5
            value = value + strat
        molar = next(m for (k, _kind, m) in PARAMS.values() if k == key)
        if molar is not None:                          # ppb -> kg/kg moist
            value = value * 1e-9 * molar / MWDRY * (1.0 - q)
        else:                                          # ug/kg -> kg/kg moist
            value = value * 1e-9 * (1.0 - q)
        out[key] = value
    return ps, out


def _new(sample, lat, lon, cycle, step, packing):
    h = eccodes.codes_grib_new_from_samples(sample)
    eccodes.codes_set(h, "centre", "ecmf")
    eccodes.codes_set(h, "dataDate", int(cycle.strftime("%Y%m%d")))
    eccodes.codes_set(h, "dataTime", int(cycle.strftime("%H%M")))
    eccodes.codes_set(h, "typeOfProcessedData", "fc")
    eccodes.codes_set(h, "stepUnits", "h")
    eccodes.codes_set(h, "step", int(step))
    eccodes.codes_set(h, "gridType", "regular_ll")
    eccodes.codes_set(h, "Ni", int(lon.size))
    eccodes.codes_set(h, "Nj", int(lat.size))
    eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", float(lat[0]))
    eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", float(lon[0]) % 360.0)
    eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", float(lat[-1]))
    eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", float(lon[-1]) % 360.0)
    eccodes.codes_set(h, "iDirectionIncrementInDegrees", float(lon[1] - lon[0]))
    eccodes.codes_set(h, "jDirectionIncrementInDegrees", float(lat[0] - lat[1]))
    eccodes.codes_set(h, "iScansNegatively", 0)
    eccodes.codes_set(h, "jScansPositively", 0)
    return h


def _write(handle, values, packing, stream):
    eccodes.codes_set(handle, "packingType", "grid_simple")
    eccodes.codes_set(handle, "bitsPerValue", 16)
    eccodes.codes_set_values(handle, np.ascontiguousarray(values, dtype=np.float64).ravel())
    if packing == "ccsds":
        eccodes.codes_set(handle, "packingType", "grid_ccsds")
    eccodes.codes_write(handle, stream)
    eccodes.codes_release(handle)


def build(out_dir, area, cycle, leads, levels, packing="simple", dx=0.4, only=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid(area, dx)
    pv = [float(row[1]) for row in L137] + [float(row[2]) for row in L137]
    tag = cycle.strftime("%Y%m%dT%H")
    ml_path = out_dir / f"SYNTHETIC-cams-ml-{tag}.grib2"
    sl_path = out_dir / f"SYNTHETIC-cams-sl-{tag}.grib2"
    seen = {}
    with open(ml_path, "wb") as ml, open(sl_path, "wb") as sl:
        for step in leads:
            ps, fields = _fields(lat, lon, step)
            for pid, (key, _kind, _m) in PARAMS.items():
                if only is not None and key not in only:
                    continue
                for level in levels:
                    h = _new("GRIB2", lat, lon, cycle, step, packing)
                    eccodes.codes_set(h, "typeOfLevel", "hybrid")
                    eccodes.codes_set(h, "level", int(level))
                    eccodes.codes_set(h, "PVPresent", 1)
                    eccodes.codes_set_array(h, "pv", pv)
                    if PARAMS[pid][1] == "gas" and pid in (210203, 210121, 210123, 210122):
                        eccodes.codes_set(h, "productDefinitionTemplateNumber", 40)
                    eccodes.codes_set(h, "paramId", pid)
                    if level == levels[0] and step == leads[0]:
                        seen[key] = {
                            "paramId": pid,
                            "discipline": eccodes.codes_get(h, "discipline"),
                            "category": eccodes.codes_get(h, "parameterCategory"),
                            "parameter": eccodes.codes_get(h, "parameterNumber"),
                            "pdt": eccodes.codes_get(h, "productDefinitionTemplateNumber"),
                            "constituent_type": (eccodes.codes_get(h, "constituentType")
                                                 if eccodes.codes_is_defined(h, "constituentType")
                                                 else None),
                            "level_type": eccodes.codes_get(h, "typeOfFirstFixedSurface", ktype=int),
                        }
                    _write(h, fields[key][level - 1], packing, ml)
            h = _new("GRIB2", lat, lon, cycle, step, packing)
            eccodes.codes_set(h, "typeOfLevel", "surface")
            eccodes.codes_set(h, "paramId", 134)
            seen["surface_pressure"] = {
                "paramId": 134, "discipline": eccodes.codes_get(h, "discipline"),
                "category": eccodes.codes_get(h, "parameterCategory"),
                "parameter": eccodes.codes_get(h, "parameterNumber"),
                "pdt": eccodes.codes_get(h, "productDefinitionTemplateNumber"),
                "level_type": eccodes.codes_get(h, "typeOfFirstFixedSurface", ktype=int)}
            _write(h, ps, packing, sl)
    manifest = {
        "schema": "gpuwm.chem.synthetic-cams-fixture.v1",
        "SYNTHETIC": "values are smooth plausible profiles, not a forecast",
        "encoder": f"ecCodes {eccodes.codes_get_api_version()}",
        "area_nwse": list(area), "grid_deg": dx, "cycle": cycle.isoformat(),
        "leads_h": list(leads), "levels": [int(levels[0]), int(levels[-1]), len(levels)],
        "packing": packing, "encoded_as": seen,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ml_path, sl_path)},
    }
    with open(out_dir / f"SYNTHETIC-cams-{tag}.manifest.json", "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--area", required=True, help="N,W,S,E degrees")
    ap.add_argument("--cycle", required=True, help="YYYY-MM-DDTHH")
    ap.add_argument("--leads", default="0,3")
    ap.add_argument("--levels", default="1-137")
    ap.add_argument("--packing", default="simple", choices=("simple", "ccsds"))
    ap.add_argument("--params", default="",
                    help="comma list of manifest keys to write (default: all)")
    args = ap.parse_args(argv)
    area = tuple(float(v) for v in args.area.split(","))
    cycle = datetime.strptime(args.cycle, "%Y-%m-%dT%H")
    leads = [int(v) for v in args.leads.split(",")]
    lo, _, hi = args.levels.partition("-")
    levels = list(range(int(lo), int(hi or lo) + 1))
    only = set(filter(None, args.params.split(","))) or None
    manifest = build(args.out, area, cycle, leads, levels, args.packing, only=only)
    print(json.dumps({k: manifest[k] for k in ("files", "encoded_as")}, indent=1))


if __name__ == "__main__":
    main()
