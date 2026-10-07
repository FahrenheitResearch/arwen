"""Split an anonymous public inventory into accurate regional case manifests.

This builds acquisition manifests, not prepared simulation authorities. A
run manifest cannot be made from object listings: decoded geometry, radar
slots, surface reports and verification footprints must bind actual bytes.
No weather data are downloaded or prepared by this command.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

EXTRA_CASES = ("e4", "r3s8", "z5", "c3", "b3")
PRODUCTS = ("PrecipRate_00.00", "RadarQualityIndex_00.00",
            "SeamlessHSR_00.00", "RadarOnly_QPE_01H_00.00",
            "MultiSensor_QPE_01H_Pass2_00.00")


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("case clocks must carry a timezone")
    return result.astimezone(timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def object_time(key: str) -> datetime | None:
    match = re.search(r"(\d{8})[-_](\d{6})", key)
    if match is None:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc)


def select_objects(case: dict, inventory: dict) -> list[dict]:
    cycle, fork = utc(case["hrrr_cycle"]), utc(case["forecast_fork"])
    begin = utc(case["da_start"]) - timedelta(minutes=10)
    end = fork + timedelta(hours=6, minutes=5)
    hours = case["hrrr_forecast_hours"]
    if any(type(hour) is not int or hour < 0 or hour > 18 for hour in hours):
        raise ValueError("HRRR forecast hours must be ordinary 0..18 hour leads")
    if cycle + timedelta(hours=max(hours)) < end:
        raise ValueError("case forcing does not cover forecast verification")
    prefix = f"hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z."
    hrrr = {prefix + f"wrfprsf{hour:02d}.grib2" for hour in hours}
    hrrr.update(prefix + f"wrf{kind}f00.grib2" for kind in ("nat", "sfc"))
    selected: dict[tuple[str, str], dict] = {}
    for item in inventory["objects"]:
        bucket, key = item["bucket"], item["key"]
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("inventory object byte counts must be positive integers")
        include = bucket == "noaa-hrrr-bdp-pds" and key in hrrr
        taken = object_time(key)
        if bucket == "unidata-nexrad-level2" and taken is not None:
            include = (case["radar_site"] + "/" in key
                       and begin <= taken <= fork and not key.endswith("_MDM"))
        if bucket == "noaa-mrms-pds" and taken is not None:
            include = (begin <= taken <= end
                       and any(key.startswith(f"CONUS/{product}/")
                               for product in PRODUCTS))
        if include:
            identity = (bucket, key)
            if identity in selected and selected[identity] != item:
                raise ValueError("inventory gives conflicting object identities")
            selected[identity] = item
    actual_hrrr = {key for bucket, key in selected
                   if bucket == "noaa-hrrr-bdp-pds"}
    if actual_hrrr != hrrr:
        raise ValueError("required HRRR objects missing: "
                         + ", ".join(sorted(hrrr - actual_hrrr)))
    return [selected[key] for key in sorted(selected)]


def build_case(case: dict, grid: dict, inventory: dict,
               inventory_sha256: str) -> tuple[dict, dict]:
    objects = select_objects(case, inventory)
    counts = Counter(item["bucket"] if item["bucket"] != "noaa-mrms-pds"
                     else item["key"].split("/")[1] for item in objects)
    fork = utc(case["forecast_fork"])
    windows = [{"lead_hours": lead,
                "start": stamp(fork + timedelta(hours=lead - 1)),
                "end": stamp(fork + timedelta(hours=lead))}
               for lead in (1, 3, 6)]
    blockers = [
        {"id": "radar_slots", "needed": ["slots[].obs", "slots[].grid_wrfout",
                                          "slots[].leg_seconds", "slots[].analysis_time"],
         "reason": "Original accepted volume timestamps and frozen Vr/Z QC are absent. da_start is a proposed two-hour window, not a recovered first slot."},
        {"id": "prepared_authority", "needed": ["prepared_root", "authority_dir",
                                                  "physics_profile", "proof_sha256",
                                                  "source_manifest_sha256",
                                                  "prepared_content_sha256", "run_seconds"],
         "reason": "No prepared bundle exists. Event center and grid sizes do not pin Lambert geometry, eta coordinates, terrain/static fields, resolved physics or forcing bytes. Hashes must be computed after public preparation."},
        {"id": "surface_record", "needed": ["surface_obs", "station_table_sha256"],
         "reason": "Public IEM observations can be fetched by rw_asos, but the approved station list, elevation/age/completeness policy and frozen seam record are absent. No surface data are supplied by the AWS inventory."},
        {"id": "footprints", "needed": ["observed_tracked_footprints", "tracking_recipe", "footprint_qc_receipt"],
         "reason": "Event centers are not forecast-hour tracked observed footprints. No frozen footprint masks or approved public tracking recipe were supplied."},
        {"id": "truth_support", "needed": ["native_valid_masks", "native_qc", "mrms_product_contract", "common_support_receipt"],
         "reason": "MRMS object listings do not prove decoded native missing-value/QC masks, continuous support, product height equivalence or complete common verification support."},
        {"id": "historical_research_arm", "needed": ["lead_decision_on_research_only_features"],
         "reason": "Historical FX4S child analyses, slot recipe, additive noise and mean-preserving positivity are not reproduced by the current public candidate. No private source was read or copied."},
    ]
    manifest = {
        "schema": "gpuwm-da.public-regional-case-plan.v1",
        "case_id": case["id"],
        "status": "ACQUISITION_READY_RUN_MANIFEST_BLOCKED",
        "historical_case": dict(case),
        "historical_grid_sizes_only": dict(grid),
        "experiment": {"members": 32, "score_leads_hours": [1, 3, 6],
                       "free_forecast_seconds": 21600,
                       "required_shared_physics": {"moist_cq": True},
                       "matched_arms": ["public-da", "clean-start-no-da"],
                       "second_seed": "must be stated by operator and shared by matched arms"},
        "verification": {"hourly_rain_windows": windows,
                         "accumulation_endpoint_hours": [0, 1, 2, 3, 5, 6],
                         "reflectivity_interval_seconds": 120,
                         "fss_thresholds_mm_per_hour": [1, 5, 10],
                         "fss_neighborhood_widths_km": [10, 25, 50],
                         "spec": "SPEC-regional-rain.md"},
        "public_acquisition": {
            "inventory": "fetch-inventory.json",
            "source_inventory_sha256": inventory_sha256,
            "object_counts": dict(sorted(counts.items())),
            "bytes": sum(item["bytes"] for item in objects),
            "data_status": "listed objects only, no retrieval or decoded coverage proof",
            "fetch_argv": ["python3", "fetch_public.py", "fetch", "--inventory",
                           f"cases/{case['id']}/fetch-inventory.json", "--root",
                           "/workspace/shared-prepared/da-rerun-286/raw", "--receipt",
                           "receipts/public-fetch.jsonl", "--max-gib", "60"],
            "fetch_order": "Use one fetch process. Do not refetch shared objects already sealed in the shared-prepared receipt."},
        "public_preparation_doors": {
            "source": "hrrr-prs in tools.da_cycle_prepared",
            "hash_binder": "tools.da_background_ab.prepare_case",
            "native_source_door": "gpuwm.source_cli --source hrrr",
            "nested_geometry_author": "gpuwm.hrrr_route_inputs.write_hrrr_route_inputs",
            "native_required_flags": ["--source-root", "--namelist-input", "--valid-time",
                                      "--output-root", "--source-sha256s",
                                      "--source-sha256s-sha256", "--domain-spec",
                                      "--geog-root"],
            "prepared_da_contract": "regional-serial-fallback.py REQUIRED plus timestamped slots"},
        "missing": blockers,
        "qualification": "Historical tuning event. This case does not satisfy the spec's unseen-event gate.",
    }
    fetch_manifest = {"schema": "da-rerun.public-inventory/v1", "objects": objects,
                      "bytes": sum(item["bytes"] for item in objects),
                      "coverage": [row for row in inventory.get("coverage", [])
                                   if row.get("case") == case["id"]],
                      "note": "Acquisition listing only. No fabricated weather-data SHA-256 or run authority."}
    return manifest, fetch_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=EXTRA_CASES)
    args = parser.parse_args(argv)
    case_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    requested = set(args.case or EXTRA_CASES)
    cases = [case for case in case_doc["cases"] if case["id"] in requested]
    if {case["id"] for case in cases} != requested:
        raise ValueError("requested case metadata absent")
    summaries = []
    for case in cases:
        plan, fetch_manifest = build_case(case, case_doc["grid"], inventory, inventory_hash)
        dest = args.out / case["id"]
        dest.mkdir(parents=True, exist_ok=True)
        for name, doc in (("case.json", plan), ("fetch-inventory.json", fetch_manifest)):
            (dest / name).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        summaries.append({"case": case["id"], "objects": len(fetch_manifest["objects"]),
                          "GiB": fetch_manifest["bytes"] / 2**30,
                          "status": plan["status"], "missing": [item["id"] for item in plan["missing"]]})
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
