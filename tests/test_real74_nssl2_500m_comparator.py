"""Focused CPU-only tests for the real74 NSSL-2 streaming comparator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import netCDF4
import numpy as np
import pytest

from tools import compare_real74_nssl2_500m as comparator


CANDIDATE_POLICY = (
    comparator.REPOSITORY_ROOT / "configs"
    / "real74_nssl2_500m_comparison_candidate_v1.json"
)
RATIFIED_POLICY = (
    comparator.REPOSITORY_ROOT / "configs"
    / "real74_nssl2_500m_comparison_ratified_v1.json"
)
TINY_DOMAIN = comparator.DomainSpec(
    "d01", 1, 1, 1, 1, 1, 3, 2, 2, 12000.0, 60.0, 3600)


def _policy(*, interior: int = 0) -> dict[str, object]:
    payload = json.loads(CANDIDATE_POLICY.read_text(encoding="utf-8"))
    payload["surface_metric_convention"][
        "interior_exclusion_cells"] = interior
    return comparator.validate_policy(payload)


def _shape_for(layout: str) -> tuple[int, ...]:
    nz, ny, nx = TINY_DOMAIN.nz, TINY_DOMAIN.ny, TINY_DOMAIN.nx
    return {
        "mass3": (nz, ny, nx), "u3": (nz, ny, nx + 1),
        "v3": (nz, ny + 1, nx), "w3": (nz + 1, ny, nx),
        "surface": (ny, nx),
    }[layout]


_UNITS = {
    "speed": "m s-1", "temperature": "K", "pressure": "Pa",
    "geopotential": "m2 s-2", "length": "m",
    "mixing_ratio": "kg kg-1", "number": "# kg(-1)",
    "volume": "m(3) kg(-1)", "precipitation": "mm",
}


def _write_tiny_frame(path: Path, *, side: str, offset_field: str | None = None,
                      offset: float = 0.0, negative_field: str | None = None,
                      coordinate_offset: float = 0.0,
                      bad_units_field: str | None = None,
                      legacy_cpu_aliases: bool = False,
                      include_reflectivity: bool = True) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        for name, size in {
            "Time": 1, "DateStrLen": 19,
            "west_east": TINY_DOMAIN.nx,
            "south_north": TINY_DOMAIN.ny,
            "bottom_top": TINY_DOMAIN.nz,
            "west_east_stag": TINY_DOMAIN.nx + 1,
            "south_north_stag": TINY_DOMAIN.ny + 1,
            "bottom_top_stag": TINY_DOMAIN.nz + 1,
        }.items():
            dataset.createDimension(name, size)
        dataset.setncatts({
            "GRID_ID": 1, "PARENT_ID": 1 if side == "cpu" else 0,
            "I_PARENT_START": 1, "J_PARENT_START": 1,
            "PARENT_GRID_RATIO": 1, "DX": 12000.0, "DY": 12000.0,
            "DT": 60.0,
            "MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
            "STAND_LON": -83.9297, "MOAD_CEN_LAT": 39.6848,
            "POLE_LAT": 90.0, "POLE_LON": 0.0,
            "START_DATE": "1974-04-03_12:00:00",
            "SIMULATION_START_DATE": "1974-04-03_12:00:00",
        })
        if side == "cpu":
            dataset.MP_PHYSICS = 18
        else:
            dataset.GPUWM_WRITE_COMPLETE = 1
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list("1974-04-03_12:00:00"), dtype="S1")

        for field in comparator.FIELD_SPECS:
            name = field.cpu_name if side == "cpu" else field.gpu_name
            if legacy_cpu_aliases and side == "cpu":
                name = comparator.LEGACY_FOREIGN_CPU_ALIASES.get(
                    field.canonical, name)
            variable = dataset.createVariable(
                name, "f4", comparator._layout_dimensions(field.layout))
            variable.units = (
                "wrong" if field.canonical == bad_units_field
                else _UNITS[field.unit_family])
            variable.stagger = comparator._layout_stagger(field.layout)
            shape = _shape_for(field.layout)
            value = np.zeros(shape, dtype=np.float32)
            if field.canonical == "base_column_mass":
                value.fill(90000.0)
            elif field.canonical == "column_mass_perturbation":
                value.fill(1000.0)
            elif field.nonnegative:
                value.fill(1.0)
            if field.canonical == offset_field:
                value += np.float32(offset)
            if field.canonical == negative_field:
                value.flat[0] = np.float32(-1.0e-4)
            variable[0] = value

        lat = dataset.createVariable(
            "XLAT", "f4", ("Time", "south_north", "west_east"))
        lat.units, lat.stagger = "degree_north", ""
        lat[0] = np.asarray(
            [[39.0, 39.0, 39.0], [40.0, 40.0, 40.0]], dtype=np.float32
        ) + np.float32(coordinate_offset)
        lon = dataset.createVariable(
            "XLONG", "f4", ("Time", "south_north", "west_east"))
        lon.units, lon.stagger = "degree_east", ""
        lon[0] = np.asarray(
            [[-85.0, -84.0, -83.0], [-85.0, -84.0, -83.0]],
            dtype=np.float32)
        for suffix, dimensions, shape, stagger in (
                ("U", ("Time", "south_north", "west_east_stag"),
                 (2, 4), "X"),
                ("V", ("Time", "south_north_stag", "west_east"),
                 (3, 3), "Y")):
            lat_stag = dataset.createVariable(
                f"XLAT_{suffix}", "f4", dimensions)
            lat_stag.units, lat_stag.stagger = "degree_north", stagger
            lat_stag[0] = np.full(
                shape, 39.0 + coordinate_offset, dtype=np.float32)
            lon_stag = dataset.createVariable(
                f"XLONG_{suffix}", "f4", dimensions)
            lon_stag.units, lon_stag.stagger = "degree_east", stagger
            lon_stag[0] = np.full(shape, -84.0, dtype=np.float32)
        mapfac = dataset.createVariable(
            "MAPFAC_M", "f4", ("Time", "south_north", "west_east"))
        mapfac.units, mapfac.stagger = "", ""
        mapfac[0] = np.ones((2, 3), dtype=np.float32)
        for suffix, dimensions, shape, stagger in (
                ("U", ("Time", "south_north", "west_east_stag"),
                 (2, 4), "X"),
                ("V", ("Time", "south_north_stag", "west_east"),
                 (3, 3), "Y")):
            mapfac_stag = dataset.createVariable(
                f"MAPFAC_{suffix}", "f4", dimensions)
            mapfac_stag.units, mapfac_stag.stagger = "", stagger
            mapfac_stag[0] = np.ones(shape, dtype=np.float32)
        znu = dataset.createVariable("ZNU", "f4", ("Time", "bottom_top"))
        znu.units, znu.stagger = "", ""
        znu[0] = np.asarray([0.75, 0.25], dtype=np.float32)
        znw = dataset.createVariable(
            "ZNW", "f4", ("Time", "bottom_top_stag"))
        znw.units, znw.stagger = "", "Z"
        znw[0] = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
        if include_reflectivity:
            refl = dataset.createVariable(
                "REFL_10CM", "f4", comparator._layout_dimensions("mass3"))
            refl.units, refl.stagger = "dBZ", ""
            refl[0] = np.asarray([
                [[10.0, 20.0, 30.0], [40.0, 10.0, 20.0]],
                [[15.0, 25.0, 35.0], [45.0, 15.0, 25.0]],
            ], dtype=np.float32)


def test_candidate_policy_is_explicitly_unratified_and_rule_free():
    policy, identity = comparator.load_policy(CANDIDATE_POLICY)
    assert policy["status"] == "candidate-unratified"
    assert policy["authority"]["ratified_for_this_comparison"] is False
    assert policy["continuous_metric_rules"] == []
    assert identity["payload_sha256"] == comparator.stable_hash(policy)


def test_production_policy_is_prerun_ratified_and_scoped_to_reflectivity():
    policy, identity = comparator.load_policy(RATIFIED_POLICY)
    assert policy["status"] == "ratified"
    assert policy["authority"]["ratified_for_this_comparison"] is True
    assert policy["authority"]["approved_at_utc"] == "2026-07-22T22:18:03Z"
    assert policy["continuous_metric_rules"] == []
    events = policy["surface_events"]
    assert events["composite_reflectivity"]["minimum_fss"] == {
        "20": 0.9, "30": 0.8, "40": 0.7,
    }
    assert events["gridscale_precipitation"]["minimum_fss"] == {}
    assert events["hail_accumulation"]["minimum_fss"] == {}
    assert identity["payload_sha256"] == comparator.stable_hash(policy)


def test_native_nssl_aliases_are_explicit():
    aliases = {
        field.canonical: (field.cpu_name, field.gpu_name)
        for field in comparator.FIELD_SPECS
    }
    assert aliases["cloud_droplet_number"] == ("QNDROP", "QNDROP")
    assert aliases["graupel_volume"] == ("QVGRAUPEL", "QVGRAUPEL")
    assert aliases["hail_volume"] == ("QVHAIL", "QVHAIL")
    assert comparator.LEGACY_FOREIGN_CPU_ALIASES == {
        "cloud_droplet_number": "QNCLOUD",
        "graupel_volume": "QVOLG",
        "hail_volume": "QVOLH",
    }


def test_calendar_discovery_requires_exact_13_13_13_25(tmp_path):
    for domain, times in comparator.expected_calendar().items():
        for valid in times:
            name = valid.strftime(f"wrfout_{domain}_%Y-%m-%d_%H_%M_%S")
            (tmp_path / name).write_bytes(b"frame")
    assert len(comparator.discover_frames(tmp_path)) == 64

    next(iter(tmp_path.iterdir())).unlink()
    with pytest.raises(comparator.ComparisonError, match="not exact"):
        comparator.discover_frames(tmp_path)


def test_inspection_accepts_native_alias_units_dimensions_and_stagger(tmp_path):
    cpu_path, gpu_path = tmp_path / "cpu", tmp_path / "gpu"
    _write_tiny_frame(cpu_path, side="cpu")
    _write_tiny_frame(gpu_path, side="gpu")
    with netCDF4.Dataset(cpu_path) as cpu, netCDF4.Dataset(gpu_path) as gpu:
        cpu_schema = comparator.inspect_frame(
            cpu, side="cpu", domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME)
        gpu_schema = comparator.inspect_frame(
            gpu, side="gpu", domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME)
    assert cpu_schema["cloud_droplet_number"]["units"] == "# kg(-1)"
    assert gpu_schema["cloud_droplet_number"]["units"] == "# kg(-1)"


def test_inspection_rejects_wrong_units(tmp_path):
    path = tmp_path / "cpu"
    _write_tiny_frame(
        path, side="cpu", bad_units_field="cloud_droplet_number")
    with netCDF4.Dataset(path) as dataset, pytest.raises(
            comparator.ComparisonError, match="units"):
        comparator.inspect_frame(
            dataset, side="cpu", domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME)


def test_inspection_rejects_legacy_foreign_nssl_names(tmp_path):
    path = tmp_path / "cpu"
    _write_tiny_frame(path, side="cpu", legacy_cpu_aliases=True)
    with netCDF4.Dataset(path) as dataset, pytest.raises(
            comparator.ComparisonError, match="legacy/foreign QNCLOUD"):
        comparator.inspect_frame(
            dataset, side="cpu", domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME)


def test_streamed_pair_has_exact_metrics_percentiles_and_inventories(tmp_path):
    cpu_path, gpu_path = tmp_path / "cpu", tmp_path / "gpu"
    _write_tiny_frame(cpu_path, side="cpu")
    _write_tiny_frame(
        gpu_path, side="gpu", offset_field="cloud_droplet_number",
        offset=2.0)
    with tempfile.TemporaryDirectory() as temporary:
        _structure, fields, events, summaries = comparator.compare_frame_pair(
            cpu_path, gpu_path, domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME, policy=_policy(),
            chunk_values=4, temporary_root=Path(temporary))
    droplets = next(
        result for result in fields
        if result["field"] == "cloud_droplet_number")
    assert droplets["count"] == 12
    assert droplets["bias"] == pytest.approx(2.0)
    assert droplets["mae"] == pytest.approx(2.0)
    assert droplets["rmse"] == pytest.approx(2.0)
    assert droplets["max_abs"] == pytest.approx(2.0)
    assert droplets["p99.9_abs"] == pytest.approx(2.0)
    assert droplets["inventory"]["difference"] > 0.0
    assert events
    assert any(item.get("kind") == "particle_number" for item in summaries)
    changes = comparator.conservation_changes(summaries)
    assert changes
    assert {item["change_difference"] for item in changes} == {0.0}


def test_health_and_exact_geometry_fail_closed(tmp_path):
    cpu_path, gpu_path = tmp_path / "cpu", tmp_path / "gpu"
    _write_tiny_frame(cpu_path, side="cpu")
    _write_tiny_frame(gpu_path, side="gpu", negative_field="hail")
    with tempfile.TemporaryDirectory() as temporary, pytest.raises(
            comparator.ComparisonError, match="nonnegative"):
        comparator.compare_frame_pair(
            cpu_path, gpu_path, domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME, policy=_policy(),
            chunk_values=4, temporary_root=Path(temporary))

    _write_tiny_frame(gpu_path, side="gpu", coordinate_offset=0.01)
    with tempfile.TemporaryDirectory() as temporary, pytest.raises(
            comparator.ComparisonError, match="geometry tolerance"):
        comparator.compare_frame_pair(
            cpu_path, gpu_path, domain=TINY_DOMAIN,
            valid_time=comparator.START_TIME, policy=_policy(),
            chunk_values=4, temporary_root=Path(temporary))


def test_categorical_metrics_and_degenerate_hold_are_deterministic():
    cpu = np.asarray([[1.0, 1.0], [0.0, 0.0]])
    gpu = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    scores = comparator.categorical_scores(cpu, gpu, 0.5)
    assert (scores["hits"], scores["misses"], scores["false_alarms"],
            scores["correct_negatives"]) == (1, 1, 1, 1)
    assert scores["critical_success_index"] == pytest.approx(1.0 / 3.0)

    definition = {
        "units": "dBZ", "thresholds": [20.0],
        "neighborhood_radius_km": 5.0, "minimum_fss": {"20": 0.9},
    }
    rows = comparator.surface_event_metrics(
        np.zeros((2, 2)), np.zeros((2, 2)),
        event_name="composite_reflectivity", definition=definition,
        domain=TINY_DOMAIN, lead_seconds=0, interior_width=0,
        degenerate_floor=1.0e-4)
    assert rows[0]["fss"] == 1.0
    assert rows[0]["evaluation"] == "held_degenerate"


def test_candidate_rule_evaluation_never_claims_ratified_pass():
    policy = _policy()
    policy["continuous_metric_rules"] = [{
        "field": "cloud_droplet_number", "metric": "rmse",
        "operator": "<=", "value": 2.0, "units": "kg-1",
        "domains": ["d01"],
    }]
    policy = comparator.validate_policy(policy)
    result = {
        "field": "cloud_droplet_number", "domain": "d01",
        "lead_seconds": 0, "bias": 1.0, "mae": 1.0, "rmse": 1.0,
        "max_abs": 1.0, "correlation": 1.0,
        **{f"p{value:g}_abs": 1.0
           for value in policy["percentiles_absolute_error"]},
    }
    rows = comparator.apply_scientific_policy(result, policy)
    rmse = next(row for row in rows if row["metric"] == "rmse")
    assert rmse["evaluation"] == "candidate_pass"
    assert all(row["evaluation"] != "pass" for row in rows)


def _synthetic_inventory() -> list[dict[str, object]]:
    rows = []
    for domain, times in comparator.expected_calendar().items():
        for valid in times:
            name = valid.strftime(f"wrfout_{domain}_%Y-%m-%d_%H_%M_%S")
            rows.append({
                "logical_name": name, "bytes": 1,
                "sha256": comparator.stable_hash(name),
            })
    return rows


def _synthetic_registration(source: dict[str, object],
                            policy_identity: dict[str, object],
                            inventory: list[dict[str, object]],
                            *, created: str = "2026-07-22T12:00:00Z",
                            ) -> dict[str, object]:
    binding = {
        "source": source,
        "evaluator": comparator.file_identity(Path(comparator.__file__)),
        "policy": policy_identity,
        "calendar": comparator._calendar_summary(),
        "cpu_outputs": inventory,
        "field_aliases": {
            field.canonical: {
                "cpu": field.cpu_name, "gpu": field.gpu_name,
                "layout": field.layout, "unit_family": field.unit_family,
            }
            for field in comparator.FIELD_SPECS
        },
        "rejected_legacy_foreign_cpu_aliases": (
            comparator.LEGACY_FOREIGN_CPU_ALIASES),
    }
    return {
        "schema": comparator.REGISTRATION_SCHEMA,
        "created_at_utc": created,
        "binding_sha256": comparator.stable_hash(binding),
        "binding": binding,
    }


def test_registration_and_gpu_manifests_bind_prelaunch_source_and_hashes(
        tmp_path):
    _policy_payload, policy_identity = comparator.load_policy(CANDIDATE_POLICY)
    source = {"commit": "a" * 40, "tree": "b" * 40, "git_status": "clean"}
    inventory = _synthetic_inventory()
    registration = _synthetic_registration(source, policy_identity, inventory)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    comparator.validate_registration(
        registration, policy_identity=policy_identity,
        source_identity=source, cpu_inventory=inventory)

    topology = [{
        "grid_id": domain.grid_id,
        "parent_id": 0 if domain.grid_id == 1 else domain.parent_id,
        "i_parent_start": domain.i_parent_start,
        "j_parent_start": domain.j_parent_start,
        "parent_grid_ratio": domain.parent_grid_ratio,
        "parent_time_step_ratio": domain.parent_grid_ratio,
        "mass_shape": [domain.ny, domain.nx],
        "dx_m": domain.dx_m, "dt_s": domain.dt_s,
        "history_interval_s": float(domain.cadence_s), "mp_physics": 18,
    } for domain in comparator.DOMAINS]
    launch_binding = {
        "source": source, "topology": topology,
        "comparison_preregistration": {
            "registration": comparator.file_identity(registration_path),
            "registration_binding_sha256": registration["binding_sha256"],
            "registration_created_at_utc": registration["created_at_utc"],
            "policy": policy_identity,
        },
        "requested": {
            "start_time": comparator.START_TIME.isoformat(),
            "run_seconds": comparator.RUN_SECONDS,
            "restart_interval_s": 0,
            "expected_output_counts": {
                "d01": 13, "d02": 13, "d03": 13, "d04": 25,
            },
        },
    }
    launch = {
        "schema": comparator.GPU_LAUNCH_SCHEMA,
        "created_at_utc": "2026-07-22T12:01:00Z",
        "binding_sha256": comparator.stable_hash(launch_binding),
        "binding": launch_binding,
    }
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "launch-manifest.json").write_text(
        json.dumps(launch), encoding="utf-8")
    output_rows = {domain: [] for domain in ("d01", "d02", "d03", "d04")}
    for row in inventory:
        domain, _valid = comparator._parse_frame_name(str(row["logical_name"]))
        output_rows[domain].append({
            "path": str(tmp_path / str(row["logical_name"])),
            "bytes": row["bytes"], "sha256": row["sha256"],
        })
    completion = {
        "schema": comparator.GPU_COMPLETION_SCHEMA,
        "binding_sha256": launch["binding_sha256"],
        "output_counts": {"d01": 13, "d02": 13, "d03": 13, "d04": 25},
        "outputs": output_rows,
    }
    (tmp_path / "completion.json").write_text(
        json.dumps(completion), encoding="utf-8")
    provenance = comparator.validate_gpu_manifests(
        tmp_path, registration, source, inventory,
        registration_path, policy_identity)
    assert provenance["launch_binding_sha256"] == launch["binding_sha256"]

    late = dict(registration)
    late["created_at_utc"] = "2026-07-22T12:02:00Z"
    with pytest.raises(comparator.ComparisonError, match="after the GPU launch"):
        # Keep the direct launch binding coherent so this specifically proves
        # the independent pre-launch timestamp check.
        launch_binding["comparison_preregistration"][
            "registration_created_at_utc"] = late["created_at_utc"]
        launch_binding["comparison_preregistration"][
            "registration_binding_sha256"] = late["binding_sha256"]
        launch["binding_sha256"] = comparator.stable_hash(launch_binding)
        launch["binding"] = launch_binding
        (tmp_path / "metadata" / "launch-manifest.json").write_text(
            json.dumps(launch), encoding="utf-8")
        completion["binding_sha256"] = launch["binding_sha256"]
        (tmp_path / "completion.json").write_text(
            json.dumps(completion), encoding="utf-8")
        comparator.validate_gpu_manifests(
            tmp_path, late, source, inventory,
            registration_path, policy_identity)


def test_hashed_json_tsv_evidence_is_bound_and_nonoverwriting(tmp_path):
    report = {
        "report_fingerprint": "a" * 64,
        "registration_binding_sha256": "b" * 64,
        "policy_identity": {"payload_sha256": "c" * 64},
        "scientific_metric_rows": [{
            "domain": "d01", "lead_seconds": 0, "field": "hail",
            "metric": "rmse", "value": 0.0, "threshold_operator": None,
            "threshold": None, "evaluation": "report_only",
        }],
        "surface_event_results": [],
        "support_and_inventory_summaries": [],
        "conservation_change_summaries": [],
    }
    evidence = comparator.write_evidence(tmp_path, report)
    manifest = json.loads(
        (tmp_path / "evidence-manifest.json").read_text(encoding="utf-8"))
    assert evidence["evidence_binding_sha256"] == manifest["binding_sha256"]
    assert comparator.stable_hash(manifest["binding"]) == manifest[
        "binding_sha256"]
    assert (tmp_path / "comparison-metrics.tsv").read_text(
        encoding="utf-8").startswith("category\tdomain\t")
    with pytest.raises(comparator.ComparisonError, match="overwrite"):
        comparator.write_evidence(tmp_path, report)
