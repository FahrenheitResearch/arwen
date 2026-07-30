from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.case_data import PerDomainSourceOrography, SourceOrography
from gpuwm.verify.cases import real74_chain, real74_d02, real74_n5b


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "N5B-plan.json"
    path.write_text(json.dumps({
        "schema": 1,
        "metric": real74_n5b.METRIC,
        "variant": "real74_4dom-N5B-SHRINK",
        "d04": {"nx": 498, "ny": 498,
                "i_parent_start": 168, "j_parent_start": 168},
        "geometry": asdict(real74_chain.n5b_geometry()),
        "gate_convention": real74_chain._gate(
            "N5", real74_n5b.METRIC).convention,
    }), encoding="utf-8")
    return path


def test_n5b_geometry_matches_f22_amendment():
    assert real74_chain.n5b_geometry() == real74_chain.N5BGeometry(
        production_shape=(600, 600),
        shrink_shape=(498, 498),
        production_core=((100, 500), (100, 500)),
        shrink_core=((49, 449), (49, 449)),
    )


def test_shrink_is_exact_plan_replacement_with_common_core_coordinates(tmp_path):
    plan = _plan(tmp_path)
    production, _, geometry = real74_n5b.construct_variant(
        "production", config_path=real74_d02.PRODUCTION_CONFIG, plan_path=plan)
    shrink, _, shrink_geometry = real74_chain.construct_n5b_shrink_case(
        real74_d02.PRODUCTION_CONFIG)
    d04, original = shrink.domain(4), production.domain(4)
    assert (d04.run.nx, d04.run.ny) == (498, 498)
    assert (d04.i_parent_start, d04.j_parent_start) == (168, 168)
    assert d04.run.nx % d04.parent_grid_ratio == 0
    assert d04.run.ny % d04.parent_grid_ratio == 0
    assert d04.run.dx == original.run.dx
    assert d04.run.dt == original.run.dt
    assert d04.history_interval_s == original.history_interval_s
    assert original.run.restart_interval_s == 0
    assert geometry == shrink_geometry
    assert real74_n5b.core_coordinate_arrays(
        production, "production")["XLAT"].shape == (400, 400)

    assert real74_n5b.assert_common_core_coordinates(
        config_path=real74_d02.PRODUCTION_CONFIG,
        plan_path=plan,
    ) == "1e90893f2c259a3b10a9461ade81847c955f3de0fa3e03983c0f4de676312601"


def test_controller_crop_builds_consistent_shrink_and_rejects_uncropped(
        tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    production, case_data, _ = real74_n5b.construct_variant(
        "production", plan_path=plan)
    source_path = tmp_path / "synthetic-met-em-d04.nc"
    with netCDF4.Dataset(source_path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("level", 2)
        dataset.createDimension("south_north", 600)
        dataset.createDimension("west_east", 600)
        dataset.createDimension("south_north_stag", 601)
        dataset.createDimension("west_east_stag", 601)
        orography = dataset.createVariable(
            "SOILHGT", "f4", ("Time", "south_north", "west_east"))
        field = dataset.createVariable(
            "FIELD3D", "f4",
            ("Time", "level", "south_north", "west_east"))
        staggered = dataset.createVariable(
            "UU", "f4",
            ("Time", "level", "south_north", "west_east_stag"))
        values = np.arange(600 * 600, dtype=np.float32).reshape(600, 600)
        orography[0] = values
        field[0, 0], field[0, 1] = values, values + 1.0
        staggered[:] = 2.0
    declaration = case_data.source_orography
    assert isinstance(declaration, PerDomainSourceOrography)
    synthetic_declaration = PerDomainSourceOrography(tuple(
        (domain_id, SourceOrography(source_path, "SOILHGT")
         if domain_id == 4 else artifact)
        for domain_id, artifact in declaration.by_domain))
    synthetic_data = replace(
        case_data, source_orography=synthetic_declaration)

    def synthetic_case(*_args, **_kwargs):
        return production, synthetic_data

    monkeypatch.setattr(real74_d02, "construct_rung_case", synthetic_case)
    with pytest.raises(ValueError, match="generate-shrink-inputs"):
        real74_n5b.construct_variant("shrink", plan_path=plan)
    with pytest.raises(ValueError, match="generate-shrink-inputs"):
        real74_n5b.run_member(
            "unperturbed", tmp_path / "unused-frames",
            tmp_path / "unused-record.json", variant="shrink",
            plan_path=plan)

    manifest_path = real74_n5b.generate_shrink_d04_inputs(tmp_path / "crop")
    shrink, shrink_data, geometry = real74_n5b.construct_variant(
        "shrink", plan_path=plan, shrink_input_manifest=manifest_path)
    cropped = shrink_data.source_orography_for_domain(4)
    assert (shrink.domain(4).run.ny, shrink.domain(4).run.nx) == (498, 498)
    assert geometry.shrink_shape == (498, 498)
    with netCDF4.Dataset(cropped.path) as dataset:
        assert dataset.variables["SOILHGT"].shape == (1, 498, 498)
        assert dataset.variables["FIELD3D"].shape == (1, 2, 498, 498)
        assert dataset.variables["UU"].shape == (1, 2, 498, 499)
        assert np.array_equal(
            dataset.variables["SOILHGT"][0], values[51:549, 51:549])
        assert dataset.getncattr("i_parent_start") == 168
        assert dataset.getncattr("WEST-EAST_GRID_DIMENSION") == 499
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert provenance["f22_reference"] == real74_n5b.F22_REFERENCE
    assert provenance["window"]["notation"] == "[51:549)^2"
    assert provenance["source"]["sha256"] == real74_d02.sha256_file(source_path)
    assert provenance["result"]["sha256"] == real74_d02.sha256_file(cropped.path)


class _FakeState:
    def __init__(self):
        self.thp = np.ones((2, 3, 4), dtype=np.float32)


class _FakeNode:
    def __init__(self):
        self.state = _FakeState()


class _FakeModel:
    def __init__(self):
        self.domain = _FakeNode()

    def node(self, domain_id):
        assert domain_id == 1
        return self.domain


def test_one_ulp_perturbation_records_exact_float32_bits():
    model = _FakeModel()
    spec = real74_n5b.PerturbationSpec("fixture", 1, "thp", 1, 2, 3)
    document = real74_n5b.apply_one_ulp(model, spec)
    before = np.asarray(np.float32(document["before"])).view(np.uint32).item()
    after = np.asarray(np.float32(document["after"])).view(np.uint32).item()
    assert after == before + 1
    assert document["before_hex_bits"] == f"0x{before:08x}"
    assert document["after_hex_bits"] == f"0x{after:08x}"
    assert model.domain.state.thp[1, 2, 3] == np.float32(document["after"])
    assert "after gpuwm.core.model.build_experiment" in document[
        "application_surface"]


def test_reflectivity_fss_analytic_fixtures():
    event = np.zeros((9, 9), dtype=np.float64)
    event[4, 4] = 45.0
    assert real74_n5b.reflectivity_fss(event, event, dx_m=1000.0) == 1.0
    empty = np.zeros_like(event)
    assert real74_n5b.reflectivity_fss(empty, empty, dx_m=1000.0) == 1.0
    value = real74_n5b.reflectivity_fss(event, np.roll(event, 1, axis=1),
                                        dx_m=1000.0, radius_km=1.0)
    assert 0.0 < value < 1.0


def test_reflectivity_fss_two_frame_series_is_pooled_not_frame_mean():
    left = np.asarray([
        [[45.0, 0.0]],
        [[45.0, 45.0]],
    ])
    right = np.asarray([
        [[0.0, 0.0]],
        [[45.0, 45.0]],
    ])
    pooled = real74_n5b.reflectivity_fss(
        left, right, dx_m=1000.0, radius_km=0.001)
    assert pooled == pytest.approx(0.8)
    assert pooled != pytest.approx(0.5)  # arithmetic mean of frame FSS


def test_cold_pool_edge_and_gust_arrival_analytic_fixtures():
    left = np.zeros((3, 7, 7), dtype=np.float64)
    right = np.zeros_like(left)
    left[:, 3, 2] = 3.0
    right[:, 3, 3] = 3.0
    assert real74_n5b.cold_pool_edge_distance_km(
        left, right, dx_m=1000.0) == 1.0

    left.fill(0.0)
    right.fill(0.0)
    left[0:, 3, 3] = 3.0
    right[1:, 3, 3] = 3.0
    times = np.asarray([0.0, 60.0, 120.0])
    assert real74_n5b.gust_front_arrival_mae_min(
        left, right, times) == 1.0
    right.fill(0.0)
    assert np.isnan(real74_n5b.gust_front_arrival_mae_min(
        left, right, times))


def test_boundary_seeded_ci_object_count_analytic_fixture():
    times = np.asarray([0.0, 900.0, 1800.0])
    seeded = np.zeros((3, 30, 30), dtype=np.float64)
    seeded[:, 0:5, 10:15] = 45.0  # 25 km2, boundary-seeded, 30-min span
    matched = seeded.copy()
    assert real74_n5b.unmatched_boundary_ci_count(
        seeded, matched, times, dx_m=1000.0) == 0
    assert real74_n5b.unmatched_boundary_ci_count(
        seeded, np.zeros_like(seeded), times, dx_m=1000.0) == 1
    short = seeded[:2]
    assert real74_n5b.unmatched_boundary_ci_count(
        short, np.zeros_like(short), times[:2], dx_m=1000.0) == 0


def test_resolved_tke_ratio_and_discrepancy_mapping():
    y, x = np.mgrid[:4, :5]
    base = (x + 2 * y).astype(np.float64)
    left = np.broadcast_to(base, (2, 3, 4, 5)).copy()
    right = 2.0 * left
    mask = np.ones((4, 5), dtype=bool)
    ratio = real74_n5b.resolved_tke_ratio(
        left, left + 1, left - 1, right, right + 2, right - 2, mask=mask)
    assert ratio == pytest.approx(4.0)
    discrepancies = real74_n5b.metrics_to_discrepancies({
        "refl_fss": 0.9, "cold_pool_edge_km": 2.0,
        "gust_front_arrival_mae_min": 3.0,
        "unmatched_boundary_ci_count": 1.0, "tke_ratio": ratio,
    })
    assert discrepancies["one_minus_refl_fss"] == pytest.approx(0.1)
    assert discrepancies["tke_lower_excursion"] == 0.0
    assert discrepancies["tke_upper_excursion"] == 3.0


def test_evaluator_manifest_pins_commit_masks_and_cadence(tmp_path):
    exp, _, _ = real74_n5b.construct_variant(
        "production", plan_path=_plan(tmp_path))
    manifest = real74_n5b.evaluator_manifest(exp)
    assert manifest["evaluator_commit"] == real74_d02._git_commit()
    assert manifest["cadence"]["seconds"] == 900
    assert manifest["cadence"]["offset_seconds"] == [0, 900, 1800, 2700, 3600, 4500]
    assert set(manifest["masks"]) == {
        "verification_core", "cold_pool_environment", "gust_arrival_union",
        "ci_object_overlap", "inflow_fetch"}
    assert all(len(item["sha256"]) == 64 for item in manifest["masks"].values())
    assert manifest["parameters"]["ci_persistence_min"] == 20.0
    assert manifest["parameters"]["ci_minimum_area_km2"] == 25.0
    assert manifest["acceptance"] == {
        "reflectivity_fss_min": real74_n5b.F_GATE_FSS_MIN,
        "cold_pool_edge_distance_max_km":
            real74_n5b.F_GATE_EDGE_DISTANCE_MAX_KM,
        "gust_front_arrival_mae_max_min":
            real74_n5b.F_GATE_ARRIVAL_MAE_MAX_MIN,
        "unmatched_boundary_ci_count":
            real74_n5b.F_GATE_UNMATCHED_CI_COUNT,
        "tke_ratio_min": real74_n5b.F_GATE_TKE_RATIO_MIN,
        "tke_ratio_max": real74_n5b.F_GATE_TKE_RATIO_MAX,
    }


def test_evaluator_manifest_rejects_999_in_every_ratified_value(tmp_path):
    exp, _, _ = real74_n5b.construct_variant(
        "production", plan_path=_plan(tmp_path))
    original = real74_n5b.evaluator_manifest(exp)
    for section in ("parameters", "acceptance"):
        for name in original[section]:
            changed = json.loads(json.dumps(original))
            changed[section][name] = 999
            with pytest.raises(ValueError, match="ratified F value"):
                real74_n5b._validated_evaluator(
                    changed, dx_m=real74_n5b.F_GATE_D04_DX_M)
    changed = json.loads(json.dumps(original))
    changed["unratified_value"] = 999
    with pytest.raises(ValueError, match="closed-world"):
        real74_n5b._validated_evaluator(
            changed, dx_m=real74_n5b.F_GATE_D04_DX_M)
    changed = json.loads(json.dumps(original))
    changed["cadence"]["seconds"] = 999
    with pytest.raises(ValueError, match="cadence"):
        real74_n5b._validated_evaluator(
            changed, dx_m=real74_n5b.F_GATE_D04_DX_M)
    changed = json.loads(json.dumps(original))
    changed["masks"]["verification_core"]["shape"] = [999, 999]
    with pytest.raises(ValueError, match="mask hash"):
        real74_n5b._validated_evaluator(
            changed, dx_m=real74_n5b.F_GATE_D04_DX_M)


def _perturbation_document(member_id: str, seed: float) -> dict[str, object]:
    before = np.float32(seed)
    after = np.float32(np.nextafter(before, np.float32(np.inf)))
    spec = real74_n5b.DEFAULT_PERTURBATION_BY_ID[member_id]

    def bits(value):
        return f"0x{np.asarray(value).view(np.uint32).item():08x}"

    return {
        "operation": real74_n5b.PERTURBATION_OPERATION,
        "application_surface": real74_n5b.PERTURBATION_APPLICATION_SURFACE,
        "domain": f"d{spec.domain_id:02d}", "field": spec.field,
        "field_definition": real74_n5b.PERTURBATION_FIELD_DEFINITION,
        "index_order": "k,j,i", "k": spec.k, "j": spec.j, "i": spec.i,
        "before": float(before),
        "after": float(after), "before_hex_bits": bits(before),
        "after_hex_bits": bits(after),
    }


def _member_record(tmp_path: Path, member_id: str, *, perturbed: bool,
                   duration: float = 4500.0,
                   variant: str = "production") -> Path:
    outdir = tmp_path / member_id / "frames"
    outdir.mkdir(parents=True)
    frame = outdir / "fixture.bin"
    frame.write_bytes(member_id.encode("ascii"))
    inventory = [{
        "domain": "d04", "valid_time": "1974-04-03T12:00:00",
        "offset_seconds": 0, "relative_path": frame.name,
        "bytes": frame.stat().st_size, "sha256": real74_d02.sha256_file(frame),
    }]
    record = {
        "schema": 1, "metric": real74_n5b.METRIC, "id": member_id,
        "variant": variant, "outdir": str(outdir),
        "duration_seconds": duration, "cadence_seconds": 900,
        "restored_input_sha256": "a" * 64,
        "core_coordinate_sha256": "b" * 64,
        "one_ulp_perturbation": (
            _perturbation_document(member_id, float(len(member_id)))
            if perturbed else None),
        "frame_inventory": inventory,
        "frame_inventory_sha256": real74_d02.stable_hash(inventory),
        "evaluator_commit": real74_d02._git_commit(),
    }
    path = tmp_path / f"{member_id}.json"
    real74_d02.write_json(path, record)
    return path


def _zero_pair(_left, _right, *, evaluator):
    assert evaluator["core_coordinate_sha256"] == "b" * 64
    return {
        "refl_fss": 1.0, "cold_pool_edge_km": 0.0,
        "gust_front_arrival_mae_min": 0.0,
        "unmatched_boundary_ci_count": 0.0, "tke_ratio": 1.0,
    }


def _evaluator_path(tmp_path: Path) -> Path:
    exp, _, _ = real74_n5b.construct_variant("production")
    manifest = real74_n5b.evaluator_manifest(exp)
    manifest["core_coordinate_sha256"] = "b" * 64
    path = tmp_path / "N5B-evaluator.json"
    real74_d02.write_json(path, manifest)
    return path


def test_member_manifest_round_trip_and_freeze_rejections(tmp_path, monkeypatch):
    monkeypatch.setattr(
        real74_n5b, "_evaluate_same_geometry_snapshot_pair", _zero_pair)
    members = [_member_record(tmp_path, f"m{i:02d}", perturbed=True)
               for i in range(1, 4)]
    baseline = _member_record(tmp_path, "unperturbed", perturbed=False)
    ensemble_path = tmp_path / "N5B-same-geometry-ensemble.json"
    evaluator_path = _evaluator_path(tmp_path)
    ensemble = real74_n5b.emit_ensemble_manifest(
        members, baseline, ensemble_path,
        evaluator_manifest_path=evaluator_path)
    assert len(ensemble["members"]) == 3
    assert all(len(values) == 3
               for values in ensemble["same_geometry_discrepancies"].values())
    frozen_path = tmp_path / "N5B-frozen-envelope.json"
    frozen = real74_chain.freeze_n5b_envelope(ensemble_path, frozen_path)
    anchor_sha256 = real74_d02.sha256_file(
        real74_chain._n5b_freeze_anchor_path(frozen_path))
    assert frozen["geometry"]["production_shape"] == (600, 600)
    assert all(value == limit for value, limit in zip(
        frozen["allowed_discrepancies"].values(),
        frozen["stated_limits"].values(), strict=True))
    frozen_hash = real74_d02.sha256_file(frozen_path)
    with pytest.raises(FileExistsError, match="already exists"):
        real74_chain.freeze_n5b_envelope(ensemble_path, frozen_path)
    assert real74_d02.sha256_file(frozen_path) == frozen_hash
    member_bytes = members[0].read_bytes()
    members[0].write_bytes(member_bytes + b" ")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        real74_chain.verify_n5b_freeze(
            frozen_path, expected_anchor_sha256=anchor_sha256)
    members[0].write_bytes(member_bytes)
    assert real74_chain.verify_n5b_freeze(
        frozen_path, expected_anchor_sha256=anchor_sha256).sha256 == frozen_hash

    original = json.loads(ensemble_path.read_text(encoding="utf-8"))
    too_few = {**original, "members": original["members"][:2]}
    path = tmp_path / "too-few.json"
    path.write_text(json.dumps(too_few), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 3"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")

    missing_doc = json.loads(json.dumps(original))
    missing_doc["members"][0].pop("one_ulp_perturbation")
    path = tmp_path / "missing-doc.json"
    path.write_text(json.dumps(missing_doc), encoding="utf-8")
    with pytest.raises(ValueError, match="perturbation document"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")

    truthy_string = json.loads(json.dumps(original))
    truthy_string["members"][0]["one_ulp_perturbation"] = "field-0"
    path = tmp_path / "truthy-string.json"
    path.write_text(json.dumps(truthy_string), encoding="utf-8")
    with pytest.raises(ValueError, match="perturbation document"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")

    forged_member_bytes = members[0].read_bytes()
    forged_member = json.loads(forged_member_bytes)
    forged_document = forged_member["one_ulp_perturbation"]
    forged_document.update({
        "operation": "not-nextafter", "field": "made_up",
        "k": 999, "j": 999, "i": 999, "extra": "forged",
    })
    real74_d02.write_json(members[0], forged_member)
    forged = json.loads(json.dumps(original))
    forged["members"][0]["sha256"] = real74_d02.sha256_file(members[0])
    forged["members"][0]["one_ulp_perturbation"] = forged_document
    path = tmp_path / "forged-perturbation.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="perturbation document inventory"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")
    members[0].write_bytes(forged_member_bytes)

    forged_discrepancies = json.loads(json.dumps(original))
    for values in forged_discrepancies["same_geometry_discrepancies"].values():
        values[:] = [999.0] * len(values)
    path = tmp_path / "forged-discrepancies.json"
    path.write_text(json.dumps(forged_discrepancies), encoding="utf-8")
    with pytest.raises(ValueError, match="member-artifact re-derivation"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")

    bad_hash = json.loads(json.dumps(original))
    bad_hash["members"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad-hash.json"
    path.write_text(json.dumps(bad_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        real74_chain.freeze_n5b_envelope(path, tmp_path / "unused.json")

    frozen_bytes = frozen_path.read_bytes()
    forged_timestamp = json.loads(frozen_bytes)
    forged_timestamp["freeze_timestamp"] = "FORGED-POST-LOOK"
    frozen_path.write_text(json.dumps(forged_timestamp), encoding="utf-8")
    with pytest.raises(ValueError, match="external pre-look anchor"):
        real74_chain.verify_n5b_freeze(
            frozen_path, expected_anchor_sha256=anchor_sha256)
    frozen_path.write_bytes(frozen_bytes)


def test_manifest_emitter_rejects_wrong_duration(tmp_path):
    members = [_member_record(tmp_path, f"m{i:02d}", perturbed=True,
                              duration=(4499.0 if i == 2 else 4500.0))
               for i in range(1, 4)]
    baseline = _member_record(tmp_path, "unperturbed", perturbed=False)
    with pytest.raises(ValueError, match="4,500 seconds"):
        real74_n5b.emit_ensemble_manifest(
            members, baseline, tmp_path / "ensemble.json",
            evaluator_manifest_path=_evaluator_path(tmp_path))


def test_member_record_rejects_frame_hash_mismatch(tmp_path):
    member = _member_record(tmp_path, "m01", perturbed=True)
    record = json.loads(member.read_text(encoding="utf-8"))
    (Path(record["outdir"]) / "fixture.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="frame hash mismatch"):
        real74_n5b.validate_member_record(member, perturbed=True)


def test_observation_verifies_freeze_before_shifted_evaluator(tmp_path, monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("metric")
        raise AssertionError("shifted metric ran before freeze verification")

    monkeypatch.setattr(real74_n5b, "evaluate_record_pair", forbidden)
    with pytest.raises(FileNotFoundError):
        real74_n5b.emit_observation(
            tmp_path / "production.json", tmp_path / "shrink.json",
            tmp_path / "missing-freeze.json", tmp_path / "observation.json",
            freeze_anchor_sha256="0" * 64,
            evaluator_manifest_path=tmp_path / "evaluator.json")
    assert calls == []


def test_shifted_record_evaluator_has_no_unverified_default_bypass(tmp_path):
    with pytest.raises(TypeError, match="verified_freeze"):
        real74_n5b.evaluate_record_pair(
            tmp_path / "production.json", tmp_path / "shrink.json",
            evaluator_manifest_path=tmp_path / "evaluator.json")
