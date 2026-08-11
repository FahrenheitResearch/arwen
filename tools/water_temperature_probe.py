"""Emit the water-temperature decision surface for one domain of a case.

What comes out, on the target grid, for one initialization:

============================  ==============================================
name                          meaning
============================  ==============================================
``SST``                       mapped SST, METGRID.TBL's own operator chain
``SKINTEMP``                  mapped skin temperature, METGRID masked=both
``SST_MINUS_SKINTEMP``        the two providers' disagreement, over water
``USED_SST``                  the 170..400 K admissibility mask the shipped
                              per-cell selector consumed -- true where the
                              old code took SST and false where it flipped
                              to SKINTEMP
``WATER_TEMP``                the assembled field under the named policy
``WATER_TEMP_SOURCE``         per-cell provider id (water_temperature.py)
``TSK``                       what the soil preprocessing publishes
``LANDMASK`` / ``LAKEMASK``   the target's own surface classes
============================  ==============================================

Two arms are written side by side, so the attribution is a subtraction
rather than an argument: ``--policy era5_class_coherent`` (the default) and
``--policy wrf_compat`` (the shipped per-cell selector).  Nothing here
reaches the network and nothing writes into the case.

Usage::

    python tools/water_temperature_probe.py CONFIG --domain 2 --out probe.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: What this tool calls itself in refusals and receipts.
PROBE_ROUTE = "the water-temperature probe"


def _domain_grid_and_static(exp, data, domain_id):
    """The grid and GEOG statics for one domain of a possibly nested case."""
    from gpuwm.static.projection import grids_from_projection_config
    from gpuwm.static.build import build_static
    from gpuwm.static.build import GeogSelection

    grids = grids_from_projection_config(exp)
    index = [dc.grid_id for dc in exp.domains].index(domain_id)
    grid = grids[index]
    selection = GeogSelection.from_case_data(data, domain_id=domain_id)
    static = build_static(grid, data.geog_root, selection=selection)
    return grid, static, selection


def probe(config: Path, domain_id: int, policies) -> dict:
    from gpuwm.case_data import load_experiment_case
    from gpuwm.ingest.horiz import interpolate_era5_to_lambert
    from gpuwm.ingest.soil import preprocess_noah_soil
    from gpuwm.ingest.water_temperature import (
        MIN_WATER_TEMPERATURE_K, MAX_WATER_TEMPERATURE_K, SOURCE_NAMES,
        WaterTemperatureStatics)
    from gpuwm.runtime import forcing_snapshots

    exp, data = load_experiment_case(config)
    grid, static, selection = _domain_grid_and_static(exp, data, domain_id)
    attrs = selection.landuse_global_attrs()
    lake = np.asarray(static["LU_INDEX"]) == int(attrs["ISLAKE"])
    land = np.asarray(static["LANDMASK"]) >= 0.5
    def statics_for(policy):
        """This probe's own water statics, named like a route."""
        return WaterTemperatureStatics.for_route(
            route=PROBE_ROUTE, policy=policy,
            landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
            landuse_attrs=attrs)

    from gpuwm.ingest.preflight import build_input_catalog

    catalog = build_input_catalog(data)
    snapshots = forcing_snapshots(data, catalog)
    source = snapshots[min(snapshots)]

    out = {
        "LANDMASK": np.asarray(static["LANDMASK"], dtype=np.float64),
        "LAKEMASK": lake.astype(np.float64),
        "LAT": np.asarray(grid.latlon_mass()[0], dtype=np.float64),
        "LON": np.asarray(grid.latlon_mass()[1], dtype=np.float64),
    }
    receipts = {}
    for policy in policies:
        met = interpolate_era5_to_lambert(
            source, grid, target_landmask=land,
            water_temperature_statics=statics_for(policy), backend="cpu",
            source_orography_catalog=catalog)
        fields = {name: np.asarray(value, dtype=np.float64)
                  for name, value in met.fields.items()
                  if np.ndim(value) == 2}
        soil = preprocess_noah_soil(
            met.fields, soil_type=static["SCT_DOM"],
            deep_soil_temperature=static["TMN"],
            landmask=static["LANDMASK"],
            water_temperature=met.water_temperature,
            water_temperature_policy=policy,
            route=PROBE_ROUTE)
        suffix = "" if policy == policies[0] else f"__{policy}"
        out[f"WATER_TEMP{suffix}"] = np.asarray(met.water_temperature)
        out[f"WATER_TEMP_SOURCE{suffix}"] = np.asarray(
            met.water_temperature_source)
        out[f"TSK{suffix}"] = np.asarray(soil.tsk, dtype=np.float64)
        if not suffix:
            sst = fields.get("SST")
            skin = fields["SKINTEMP"]
            out["SST"] = sst if sst is not None else np.full_like(skin, np.nan)
            out["SKINTEMP"] = skin
            out["SST_MINUS_SKINTEMP"] = np.where(
                land, np.nan, out["SST"] - skin)
            out["USED_SST"] = (
                np.isfinite(out["SST"])
                & (out["SST"] >= MIN_WATER_TEMPERATURE_K)
                & (out["SST"] <= MAX_WATER_TEMPERATURE_K)).astype(np.float64)
        receipts[policy] = met.water_temperature_receipt
    return out, receipts, {int(k): v for k, v in SOURCE_NAMES.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", action="append", default=None,
                        help="repeatable; the first is the reference arm")
    args = parser.parse_args(argv)
    policies = args.policy or ["era5_class_coherent", "wrf_compat"]
    fields, receipts, names = probe(args.config, args.domain, policies)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **fields)
    receipt_path = args.out.with_suffix(".json")
    receipt_path.write_text(json.dumps(
        {"domain": args.domain, "policies": policies,
         "provider_ids": names, "receipts": receipts},
        indent=2, default=str), encoding="utf-8")
    print(f"probe: {args.out}")
    print(f"receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
