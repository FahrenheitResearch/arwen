from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.experiment import load_experiment
from gpuwm.ingest.prepared_cache import (
    PREPARED_CACHE_SCHEMA,
    _array_sha256,
    prepared_cache_identity,
)
from gpuwm.ingest.hrrr_physics import (
    _validate_prepared_near_surface,
    _validate_prepared_surface,
)
import tools.prepared_single_domain_forecast as runner


ROOT = Path(__file__).parents[1]


def test_prepared_runner_capability_query_is_side_effect_free_without_run_args(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--show-capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == runner.runner_capabilities()
    assert payload["schema"] == "gpuwm-runner-capabilities-v1"
    assert payload["supported_sources"] == ["20crv3", "era5", "gfs"]
    assert payload["physics_profile_ids"] == list(runner.PHYSICS_PROFILES)
    assert payload["report_schema"] == runner.REPORT_SCHEMA
    assert payload["window"]["limit_policy"] \
        == "prepared-cache-forcing-coverage"
    assert payload["window"]["maximum_run_seconds"] is None
    assert payload["window"]["run_seconds"]["whole_hour_required"] is True
    assert payload["window"]["source_forcing_cadence_hours"] == {
        "gfs": [1, 3],
        "era5": "uniform-positive-whole-hour",
        "20crv3": "manifest-bound-uniform-positive-whole-hour",
    }
    twentycr = payload["source_profiles"]["20crv3"]
    assert twentycr["member_identity"] \
        == "filename_memNNN_not_grib2_pdt"
    assert "NOT_ACCEPTANCE_GATED" in twentycr["readiness"]
    assert payload["output"]["io_modes"] == ["history"]
    assert payload["output"]["configurable_cadence"] is True
    cadence = payload["output"]["history_interval_seconds"]
    assert cadence["must_equal_hash_bound_experiment"] is True
    assert cadence["must_be_whole_model_steps"] is True
    assert cadence["must_evenly_divide_run"] is False
    assert cadence["last_scheduled_frame_may_precede_run_end"] is True
    assert payload["readiness"] \
        == "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
    assert payload["modes"]["forecast"]["available"] is True
    assert payload["standalone_rw_wps_wheel"]["runner_included"] is False
    assert payload["standalone_rw_wps_wheel"][
        "forecast_executor_included"] is False
    thompson = payload["physics_profiles"][runner.THOMPSON_PHYSICS_PROFILE]
    assert thompson["explicit_expert_consent_required"] is False
    assert thompson["table_root"]["environment"] \
        == runner.THOMPSON_TABLE_ROOT_ENV
    assert len(thompson["table_authority"]["assets"]) == 4
    nssl2 = payload["physics_profiles"][runner.NSSL2_PHYSICS_PROFILE]
    assert nssl2["readiness"] == "VALIDATION_CANDIDATE"
    assert nssl2["explicit_expert_consent_required"] is False
    assert nssl2["resolved_fixed_preset"] is True
    twentycr_wsm6 = payload["physics_profiles"][
        runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE]
    assert twentycr_wsm6["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert twentycr_wsm6["source_scope"] == ["20crv3"]
    assert payload["capability_query"]["requires_cupy"] is False
    materializer = payload["authority_materialization"]
    assert materializer["available"] is True
    assert materializer["mode_flag"] == "--materialize-authorities"
    assert materializer["receipt_schema"] \
        == runner.AUTHORITY_MATERIALIZATION_SCHEMA
    assert runner.MORRISON_PHYSICS_PROFILE in materializer[
        "source_physics_profile_ids"]["gfs"]
    assert materializer["source_physics_profile_ids"]["20crv3"][0] \
        == runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE
    assert list(tmp_path.iterdir()) == []


def test_prepared_capabilities_fail_closed_when_executor_modules_are_absent(
        monkeypatch):
    monkeypatch.setattr(
        runner,
        "_missing_forecast_executor_modules",
        lambda: ["gpuwm.core.model"],
    )

    payload = runner.runner_capabilities()

    assert payload["readiness"] == "FORECAST_EXECUTOR_OMITTED"
    assert payload["modes"]["forecast"]["available"] is False
    assert payload["modes"]["forecast"]["missing_executor_modules"] == [
        "gpuwm.core.model"]
    assert payload["modes"]["forecast"]["unavailable_reason"] \
        == "GPUWM forecast executor modules are absent"
    assert payload["supported_sources"] == []
    assert payload["physics_profile_ids"] == []
    assert payload["source_profiles"] == {}
    assert payload["physics_profiles"] == {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _artifact(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bind_test_thompson_runtime(tmp_path: Path, monkeypatch):
    root = tmp_path / "thompson-tables"
    root.mkdir()
    asset_type = type(runner.THOMPSON_CLASSIC_TABLE_ASSETS[0])
    assets = []
    for canonical in runner.THOMPSON_CLASSIC_TABLE_ASSETS:
        payload = f"test-{canonical.filename}".encode("ascii")
        path = root / canonical.filename
        path.write_bytes(payload)
        assets.append(asset_type(
            canonical.filename, len(payload), hashlib.sha256(payload).hexdigest()))
    assets = tuple(assets)

    def validate(path):
        assert Path(path).resolve() == root.resolve()
        for asset in assets:
            current = root / asset.filename
            if current.stat().st_size != asset.bytes \
                    or _sha256(current) != asset.sha256:
                raise ValueError(f"test Thompson table drift: {asset.filename}")
        return assets

    # The enable gate is retired; the table root stays overridable so a
    # test can point the byte validation at its own fixture set.
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(root))
    monkeypatch.setattr(runner, "THOMPSON_CLASSIC_TABLE_ASSETS", assets)
    monkeypatch.setattr(runner, "validate_thompson_table_assets", validate)
    return SimpleNamespace(root=root.resolve(), assets=assets)


@pytest.mark.parametrize(
    ("source", "profile"),
    (
        ("gfs", runner.PHYSICS_PROFILE),
        ("gfs", runner.THOMPSON_PHYSICS_PROFILE),
        ("era5", runner.MORRISON_PHYSICS_PROFILE),
        ("era5", runner.NSSL2_PHYSICS_PROFILE),
        ("20crv3", runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE),
        ("20crv3", runner.MORRISON_PHYSICS_PROFILE),
    ),
)
def test_materializer_publishes_exact_profile_and_preserves_descriptors(
        tmp_path, monkeypatch, source, profile):
    if profile == runner.THOMPSON_PHYSICS_PROFILE:
        _bind_test_thompson_runtime(tmp_path, monkeypatch)
    config_source = "gfs" if source in {"gfs", "20crv3"} else "era5"
    base_experiment = ROOT / "configs" / f"{config_source}_wrf_direct_proof.toml"
    base_wps = (
        ROOT / "configs" / f"{config_source}_wrf_direct_proof.namelist.wps")
    base = load_experiment(base_experiment)
    output = tmp_path / f"{source}-{profile[:4]}"

    receipt = runner.materialize_named_source_authorities(
        source=source,
        base_experiment_config=base_experiment,
        base_wps_namelist=base_wps,
        physics_profile=profile,
        output_directory=output,
    )

    generated = load_experiment(output / "experiment.toml")
    expected = runner._profile_runtime_switches(source, profile)
    assert receipt["schema"] == runner.AUTHORITY_MATERIALIZATION_SCHEMA
    assert receipt["status"] == "PASS"
    assert receipt["normalized_selected_physics"]["resolved"] == {
        **expected,
        "radiation_scheme_ids": list(
            runner.radiation_scheme_ids(generated.root.run)),
    }
    assert receipt["non_physics_descriptor"]["status"] == "EXACT_UNCHANGED"
    assert receipt["non_physics_descriptor"]["base_sha256"] \
        == receipt["non_physics_descriptor"]["generated_sha256"]
    assert generated.start_time == base.start_time
    assert generated.run_seconds == base.run_seconds
    assert generated.vertical == base.vertical
    assert generated.projection == base.projection
    assert [domain.grid_id for domain in generated.domains] \
        == [domain.grid_id for domain in base.domains]
    assert (output / "namelist.wps").read_bytes() == base_wps.read_bytes()
    persisted = json.loads(
        (output / "authority-receipt.json").read_text(encoding="utf-8"))
    assert persisted["generated"]["experiment_config"]["sha256"] \
        == _sha256(output / "experiment.toml")
    assert receipt["receipt"]["sha256"] \
        == _sha256(output / "authority-receipt.json")


def test_materializer_is_create_only_and_rejects_source_profile_mismatch(
        tmp_path):
    experiment = ROOT / "configs" / "gfs_wrf_direct_proof.toml"
    wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"
    output = tmp_path / "authorities"
    runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=experiment,
        base_wps_namelist=wps, physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        output_directory=output)
    with pytest.raises(FileExistsError):
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=output)
    with pytest.raises(ValueError, match="not available for source"):
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE,
            output_directory=tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_materializer_cli_publishes_receipted_authorities(tmp_path, capsys):
    output = tmp_path / "cli-authorities"
    assert runner.main([
        "--materialize-authorities",
        "--source", "era5",
        "--base-experiment-config",
        str(ROOT / "configs" / "era5_wrf_direct_proof.toml"),
        "--base-wps-namelist",
        str(ROOT / "configs" / "era5_wrf_direct_proof.namelist.wps"),
        "--physics-profile", runner.MORRISON_PHYSICS_PROFILE,
        "--output-directory", str(output),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert summary["physics_profile"] == runner.MORRISON_PHYSICS_PROFILE
    assert Path(summary["receipt"]["path"]) \
        == (output / "authority-receipt.json").resolve()


def test_materializer_preserves_hierarchy_and_nondefault_explicit_vertical(
        tmp_path):
    base = tmp_path / "hierarchy.toml"
    source = (ROOT / "configs" / "gfs_wrf_hierarchy_proof.toml").read_text(
        encoding="utf-8")
    assert source.count("hybrid_opt = 2") == 1
    base.write_text(
        source.replace("hybrid_opt = 2", "hybrid_opt = 1"),
        encoding="utf-8")
    wps = ROOT / "configs" / "gfs_wrf_hierarchy_proof.namelist.wps"
    output = tmp_path / "hierarchy-authorities"

    receipt = runner.materialize_named_source_authorities(
        source="gfs", base_experiment_config=base,
        base_wps_namelist=wps,
        physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        output_directory=output)
    generated = load_experiment(output / "experiment.toml")

    assert len(generated.domains) == 2
    assert generated.vertical.hybrid_opt == 1
    assert generated.vertical.eta_levels == load_experiment(base).vertical.eta_levels
    assert [domain.run.mp_physics for domain in generated.domains] == [10, 10]
    assert receipt["preserved"]["mass_levels"] == generated.root.run.nz
    assert receipt["preserved"]["hybrid_opt"] == 1


def test_materialized_descriptor_bytes_are_deterministic_across_case_roots(
        tmp_path):
    experiment = ROOT / "configs" / "gfs_wrf_direct_proof.toml"
    wps = ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps"
    receipts = [
        runner.materialize_named_source_authorities(
            source="gfs", base_experiment_config=experiment,
            base_wps_namelist=wps,
            physics_profile=runner.MORRISON_PHYSICS_PROFILE,
            output_directory=tmp_path / f"case-{index}")
        for index in range(2)
    ]

    assert (tmp_path / "case-0" / "experiment.toml").read_bytes() \
        == (tmp_path / "case-1" / "experiment.toml").read_bytes()
    assert receipts[0]["generated"]["experiment_config"]["sha256"] \
        == receipts[1]["generated"]["experiment_config"]["sha256"]
    assert receipts[0]["normalized_selected_physics"] \
        == receipts[1]["normalized_selected_physics"]


def _prepared_fixture(
        tmp_path: Path, source: str, *, adapter=None, hierarchy=False,
        physics_profile=runner.PHYSICS_PROFILE,
):
    if hierarchy and source not in {"gfs", "20crv3"}:
        raise ValueError("synthetic hierarchy fixture uses GFS-shaped configs")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    kind = "hierarchy" if hierarchy else "direct"
    config_source = "gfs" if source == "20crv3" else source
    config_name = f"{config_source}_wrf_{kind}_proof.toml"
    wps_name = f"{config_source}_wrf_{kind}_proof.namelist.wps"
    experiment_config = tmp_path / config_name
    wps_namelist = tmp_path / wps_name
    shutil.copy2(ROOT / "configs" / config_name, experiment_config)
    shutil.copy2(ROOT / "configs" / wps_name, wps_namelist)
    config_text = experiment_config.read_text(encoding="utf-8")
    if physics_profile == runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE:
        if source != "20crv3":
            raise ValueError("synthetic 20CRv3 profile requires source=20crv3")
        assert config_text.count("history_interval_s = 3600.0\n") == 1
        config_text = config_text.replace(
            "history_interval_s = 3600.0\n",
            "history_interval_s = 10800.0\n",
        )
    rendered, _exp, _receipt = runner._render_materialized_experiment(
        config_text, source=source, profile=physics_profile)
    experiment_config.write_text(rendered, encoding="utf-8")
    exp = load_experiment(experiment_config)
    forcing_hours = [0, 3] if source in {"gfs", "20crv3"} else [0, 6, 12]
    preprocessing = {
        "schema": "gpuwm-preprocess-implementation-v2",
        "backend": "cpu",
        "implementation": "test-source-neutral-preprocess",
        "workers": 2,
    }
    bridge_digest = hashlib.sha256(f"{source}-bridge".encode()).hexdigest()
    files = {}
    if source != "20crv3":
        files.update({
            "bridge": {
                "name": f"{source}_bridge.exe", "sha256": bridge_digest},
            "experiment_config": {
                "name": experiment_config.name,
                "sha256": _sha256(experiment_config),
            },
            "wps_namelist": {
                "name": wps_namelist.name,
                "sha256": _sha256(wps_namelist),
            },
        })
    if source == "gfs":
        files["series"] = {
            "name": "gfs-series.tsv",
            "sha256": hashlib.sha256(b"series").hexdigest(),
        }
        for hour in forcing_hours:
            files[f"grib-f{hour:03d}"] = {
                "name": f"gfs.f{hour:03d}.grib2",
                "sha256": hashlib.sha256(f"gfs-{hour}".encode()).hexdigest(),
            }
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "source": {
                "model": "GFS",
                "product": "pgrb2.0p25",
                "cycle": exp.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "files": files,
        }
    elif source == "era5":
        files.update({
            "grib": {
                "name": "era5.grib1",
                "sha256": hashlib.sha256(b"era5-grib").hexdigest(),
            },
            "vtable": {
                "name": "Vtable.ERA5",
                "sha256": hashlib.sha256(b"era5-vtable").hexdigest(),
            },
        })
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "files": files,
        }
    else:
        member = "072"
        rows = []
        for hour in forcing_hours:
            valid_time = exp.start_time + timedelta(hours=hour)
            for role in ("pl", "sfc"):
                filename = (
                    f"mem{member}_{valid_time:%Y%m%d%H}_{role}.grb2")
                rows.append({
                    "path": str((tmp_path / "raw" / filename).resolve()),
                    "filename": filename,
                    "member": member,
                    "valid_time": valid_time.isoformat(),
                    "role": role,
                    "bytes": 1000 + hour + (1 if role == "sfc" else 0),
                    "sha256": hashlib.sha256(filename.encode()).hexdigest(),
                })
        manifest = {
            "schema": runner._SOURCE_SCHEMA[source],
            "source": runner._TWENTYCRV3_SOURCE,
            "member_identity": runner._TWENTYCRV3_MEMBER_IDENTITY,
            "member": member,
            "valid_times": [
                (exp.start_time + timedelta(hours=hour)).isoformat()
                for hour in forcing_hours],
            "cadence_seconds": 10800,
            "file_count": len(rows),
            "files": rows,
        }
        manifest["content_sha256"] = hashlib.sha256(
            _canonical(manifest).encode()).hexdigest()
    if source == "20crv3":
        evidence = prepared / "source-evidence"
        evidence.mkdir()
        authorities = runner.twentycrv3_authority_sha256()
        authority_root = ROOT / "gpuwm" / "authorities"
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.mapping.json",
            evidence / "mapping.json")
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.composition.json",
            evidence / "composition.json")
        shutil.copy2(
            authority_root / "rw-wps-20crv3-member-grib2.provenance.json",
            evidence / "provenance-test.json")
        assert _sha256(evidence / "mapping.json") == authorities["mapping"]
        assert _sha256(evidence / "composition.json") \
            == authorities["composition"]
        assert _sha256(evidence / "provenance-test.json") \
            == authorities["provenance"]
        source_manifest = evidence / "input-manifest.json"
    else:
        source_manifest = prepared / "source-input-manifest.json"
    _write_json(source_manifest, manifest)
    manifest_digest = _sha256(source_manifest)
    mapped_authorities = None
    source_composition = None
    target_contract = None
    decoder_digests = None
    if source == "20crv3":
        mapped_authorities = dict(runner.twentycrv3_authority_sha256())
        decoder_digests = {
            role: hashlib.sha256(f"20crv3-{role}".encode()).hexdigest()
            for role in runner._TWENTYCRV3_DECODER_ROLES
        }
        decoder_paths = {
            role: str((tmp_path / f"{role}.exe").resolve())
            for role in runner._TWENTYCRV3_DECODER_ROLES
        }
        composition_document = json.loads(
            (prepared / "source-evidence" / "composition.json").read_text(
                encoding="utf-8"))
        source_composition = {
            "schema": "gpuwm-mapped-composition-receipt-v1",
            "status": "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED",
            "mapping": {
                "path": str((tmp_path / "mapping.json").resolve()),
                "sha256": mapped_authorities["mapping"],
            },
            "composition": {
                "path": str((tmp_path / "composition.json").resolve()),
                "sha256": mapped_authorities["composition"],
            },
            "input_manifest": {
                "path": str((tmp_path / "member072-manifest.json").resolve()),
                "sha256": manifest_digest,
            },
            "decoders": {
                role: {"path": decoder_paths[role],
                       "sha256": decoder_digests[role]}
                for role in sorted(runner._TWENTYCRV3_DECODER_ROLES)
            },
            "terrain_products": [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in manifest["files"] if row["role"] == "sfc"
            ],
            "terrain_provenance": {
                "provenance_path": str((tmp_path / "provenance.json").resolve()),
                "provenance_sha256": mapped_authorities["provenance"],
            },
            "alignment": {
                "strategy": "20crv3_in_band_surface_same_grid",
                "terrain_external_supplement": False,
                "member": manifest["member"],
                "member_identity": manifest["member_identity"],
                "surface_file_count": len(forcing_hours),
                "valid_time_count": len(forcing_hours),
                "canonical_receipt_content_sha256": hashlib.sha256(
                    b"canonical-20crv3-frames").hexdigest(),
                "coordinate_match": "same_decoded_grib2_grid_fingerprint",
                "terrain_invariant_across_all_times": True,
            },
            "soil_layers": composition_document["soil_layers"],
            "frame_count": len(forcing_hours),
            "valid_times": manifest["valid_times"],
            "frames": [{
                "header_sha256": hashlib.sha256(
                    f"header-{hour}".encode()).hexdigest(),
                "terrain_sha256": hashlib.sha256(
                    f"terrain-{hour}".encode()).hexdigest(),
                "field_count": 15,
            } for hour in forcing_hours],
        }
        source_composition["receipt_content_sha256"] = hashlib.sha256(
            _canonical(source_composition).encode()).hexdigest()
        target_contract = {
            "schema": "gpuwm-mapped-target-contract-v1",
            "max_dom": len(exp.domains),
            "boundary_interval_seconds": 10800,
        }

    domain_bundle = prepared
    if hierarchy:
        domain_bundle = prepared / "hierarchy-artifacts" / "domains" / "d01"
        domain_bundle.mkdir(parents=True)
    static_path = domain_bundle / "native-static.npz"
    if source == "20crv3":
        np.savez(static_path, STATIC=np.ones((1,), dtype=np.float64))
    else:
        static_path.write_bytes(b"hash-bound-test-static")
    geometry_path = domain_bundle / "geometry-receipt.json"
    geometry_payload = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": {"test": source},
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
        },
    }
    _write_json(geometry_path, geometry_payload)

    if source == "20crv3":
        source_identity = {
            "adapter": adapter or runner._SOURCE_ADAPTER[source],
            "mapping_sha256": mapped_authorities["mapping"],
            "composition_sha256": mapped_authorities["composition"],
            "input_manifest_sha256": manifest_digest,
            "composition_receipt_sha256": source_composition[
                "receipt_content_sha256"],
            "preprocessing": preprocessing,
        }
        if hierarchy:
            source_identity.update({
                "target_contract": target_contract,
                "nested_source_orography": {"schema": "test-orography-v1"},
                "hierarchy_implementation_sha256": {
                    "test": hashlib.sha256(b"hierarchy").hexdigest()},
            })
    else:
        source_identity = {
            "adapter": adapter or runner._SOURCE_ADAPTER[source],
            "input_manifest_schema": runner._SOURCE_SCHEMA[source],
            "input_manifest_sha256": manifest_digest,
            "decoder": {
                "name": files["bridge"]["name"],
                "sha256": bridge_digest,
                "implementation": runner._DECODER_IMPLEMENTATION[source],
            },
            "preprocessing": preprocessing,
        }
    if source == "gfs":
        source_identity.update({
            "implementation_sha256": {"test": "gfs"},
            "git_source_identity": {"commit": "test-gfs"},
        })
    if hierarchy:
        source_identity["grid_id"] = 1
    identity = prepared_cache_identity(
        bridge_manifest_sha256=manifest_digest,
        source_manifest_sha256=manifest_digest,
        static_cache_sha256=_sha256(static_path),
        namelist_sha256=_sha256(experiment_config),
        domain_config=exp.root,
        forcing_hours=forcing_hours,
        source_identity=source_identity,
    )
    cache = domain_bundle / "prepared-cache"
    cache.mkdir()
    array = np.asarray([1.0], dtype=np.float32)
    with (cache / "a00000.npy").open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    arrays = {
        "test": {
            "file": "a00000.npy",
            "shape": [1],
            "dtype": "float32",
            "nbytes": 4,
            "sha256": _array_sha256(array),
        },
    }
    intervals = [
        {
            "start_seconds": float(first * 3600),
            "end_seconds": float(second * 3600),
            "fields": list(runner._LBC_FIELDS),
        }
        for first, second in zip(forcing_hours, forcing_hours[1:])
    ]
    user_metadata = {
            "initial_valid_time": exp.start_time.isoformat(),
            "last_valid_time": (
                exp.start_time + timedelta(hours=forcing_hours[-1])).isoformat(),
            "forcing_hours": forcing_hours,
    }
    if not hierarchy:
        if source == "20crv3":
            user_metadata.update({
                "source_adapter": "mapped",
                "boundary_interval_seconds": (
                    forcing_hours[1] - forcing_hours[0]) * 3600,
                "composition_receipt_sha256": source_composition[
                    "receipt_content_sha256"],
            })
        else:
            user_metadata.update({
                "source_adapter": source,
                "boundary_interval_seconds": (
                    forcing_hours[1] - forcing_hours[0]) * 3600,
                "preprocessing": preprocessing,
            })
    elif source == "20crv3":
        user_metadata.update({
            "composition_receipt_sha256": source_composition[
                "receipt_content_sha256"],
            "mapped_target_contract": target_contract,
        })
    metadata = {
        "user": user_metadata,
        "state_names": [],
        "coord_arrays": [],
        "coord_scalars": {},
        "base_arrays": [],
        "base_scalars": {},
        "met_fields": sorted(runner._REQUIRED_MET_FIELDS),
        "surface_fields": sorted(runner._CANONICAL_SURFACE_FIELDS),
        "lbc": {
            "spec_bdy_width": exp.root.run.spec_bdy_width,
            "spec_zone": exp.root.run.spec_zone,
            "relax_zone": exp.root.run.relax_zone,
            "intervals": intervals,
        },
        "setup_fingerprint": "test",
    }
    basis = {
        "schema": PREPARED_CACHE_SCHEMA,
        "identity": identity,
        "metadata": metadata,
        "arrays": arrays,
        "payload_bytes": 4,
    }
    header = {
        **basis,
        "status": "READY",
        "created_utc": "2026-07-23T00:00:00+00:00",
        "content_sha256": hashlib.sha256(
            _canonical(basis).encode()).hexdigest(),
    }
    header_path = cache / "header.json"
    _write_json(header_path, header)

    boundary_seconds = (forcing_hours[1] - forcing_hours[0]) * 3600
    export_source = {
        "contract_sha256": _sha256(
            ROOT / "gpuwm" / "wrf_direct_v461_contract.json"),
        "geometry_receipt_sha256": _sha256(geometry_path),
        "prepared_content_sha256": header["content_sha256"],
        "prepared_header_sha256": _sha256(header_path),
        "resolved_physics_contract_sha256": (
            runner._resolved_wrf_direct_contract_sha256(
                exp.root.run.mp_physics)),
        "static_cache_sha256": _sha256(static_path),
    }
    cache_receipt = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "BUILT",
        "path": "prepared-cache",
        "content_sha256": header["content_sha256"],
        "array_count": 1,
        "payload_bytes": 4,
    }
    proof = {
        "schema": runner._PROOF_SCHEMA[source],
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "forcing_times": [
            (exp.start_time + timedelta(hours=hour)).isoformat()
            for hour in forcing_hours
        ],
        "forcing_hours": forcing_hours,
        "boundary_interval_seconds": boundary_seconds,
        "input_manifest_sha256": manifest_digest,
        "decoder_sha256": bridge_digest,
        "preprocessing": preprocessing,
        "preprocessing_receipt_sha256": hashlib.sha256(
            _canonical(preprocessing).encode()).hexdigest(),
        "source_inputs": {
            "manifest_schema": manifest["schema"],
            "manifest_sha256": manifest_digest,
            "files": manifest["files"],
        },
        "initialization_artifacts": {
            "source_manifest": _artifact(
                source_manifest, "source-input-manifest.json"),
            "static_cache": _artifact(static_path, "native-static.npz"),
            "geometry_receipt": _artifact(
                geometry_path, "geometry-receipt.json"),
            "prepared_cache": {
                "path": "prepared-cache",
                "content_sha256": header["content_sha256"],
                "payload_bytes": 4,
            },
            "wrf_files": {},
        },
        "prepared_cache": cache_receipt,
        "export": {
            "schema": "gpuwm-native-direct-wrf-export-v2",
            "status": "READY",
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "dimensions": {
                "nx": exp.root.run.nx,
                "ny": exp.root.run.ny,
                "nz": exp.root.run.nz,
            },
            "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
            "source": export_source,
        },
    }
    if source == "20crv3":
        proof = {
            "schema": runner._PROOF_SCHEMA[source],
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "forcing_times": manifest["valid_times"],
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "execution_inputs": {
                "decoders": {
                    role: {
                        "path": source_composition["decoders"][role]["path"],
                        "bytes": 1024 + index,
                        "sha256": decoder_digests[role],
                    }
                    for index, role in enumerate(
                        sorted(runner._TWENTYCRV3_DECODER_ROLES))
                },
                "experiment_config": _artifact(
                    experiment_config, str(experiment_config.resolve())),
                "wps_namelist": _artifact(
                    wps_namelist, str(wps_namelist.resolve())),
                "geog_root": str((tmp_path / "geog").resolve()),
                "geog_source_binding": (
                    "resolved_dataset_paths_plus_native_static_output_sha256"),
                "geog_resolution_tokens": ["default"],
                "geog_datasets": {},
            },
            "source_composition": source_composition,
            "preprocessing": preprocessing,
            "static": {
                "path": "native-static.npz",
                "bytes": static_path.stat().st_size,
                "sha256": _sha256(static_path),
                "fields": ["STATIC"],
            },
            "geometry": geometry_payload,
            "prepared_cache": cache_receipt,
            "export": {
                "schema": "gpuwm-native-direct-wrf-export-v2",
                "status": "READY",
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_seconds,
                "dimensions": {
                    "nx": exp.root.run.nx,
                    "ny": exp.root.run.ny,
                    "nz": exp.root.run.nz,
                },
                "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                "source": export_source,
            },
            "timing_seconds": {"total": 1.0},
        }
    if hierarchy:
        domain_receipt = {
            "schema": "gpuwm-native-domain-artifact-build-v1",
            "status": "READY",
            "grid_id": 1,
            "parent_id": 0,
            "boundary_mode": "external-specified",
            "valid_time": exp.start_time.isoformat(),
            "forcing_hours": forcing_hours,
            "artifacts": {
                "prepared_cache": {
                    "path": "prepared-cache",
                    "content_sha256": header["content_sha256"],
                    "payload_bytes": 4,
                    "array_count": 1,
                },
                "static_cache": {
                    "path": "native-static.npz",
                    "bytes": static_path.stat().st_size,
                    "sha256": _sha256(static_path),
                    "fields": ["STATIC"],
                },
                "geometry_receipt": {
                    "path": "geometry-receipt.json",
                    "sha256": _sha256(geometry_path),
                    "geometry": geometry_payload["geometry"],
                },
            },
            "verification": {
                "schema": PREPARED_CACHE_SCHEMA,
                "status": "PASS",
                "path": "prepared-cache",
                "content_sha256": header["content_sha256"],
                "array_count": 1,
                "payload_bytes": 4,
            },
        }
        _write_json(domain_bundle / "receipt.json", domain_receipt)
        hierarchy_root = prepared / "hierarchy-artifacts"
        artifact_manifest = {
            "schema": "gpuwm-native-domain-artifacts-v1",
            "domains": [{
                "grid_id": domain.grid_id,
                "prepared_cache": (
                    f"domains/d{domain.grid_id:02d}/prepared-cache"),
                "static_cache": (
                    f"domains/d{domain.grid_id:02d}/native-static.npz"),
                "geometry_receipt": (
                    f"domains/d{domain.grid_id:02d}/geometry-receipt.json"),
            } for domain in exp.domains],
        }
        artifact_manifest_path = hierarchy_root / "domain-artifacts.json"
        _write_json(artifact_manifest_path, artifact_manifest)
        hierarchy_receipt = {
            "schema": "gpuwm-native-hierarchy-artifact-build-v1",
            "status": "READY",
            "domain_count": len(exp.domains),
            "grid_ids": [domain.grid_id for domain in exp.domains],
            "manifest": {
                "path": "domain-artifacts.json",
                "sha256": _sha256(artifact_manifest_path),
            },
            "boundary_inventory": {
                "external": [1],
                "nested_parent_forced": [
                    domain.grid_id for domain in exp.domains[1:]],
            },
            "domains": [domain_receipt] + [
                {"grid_id": domain.grid_id} for domain in exp.domains[1:]],
        }
        _write_json(hierarchy_root / "receipt.json", hierarchy_receipt)
        d01_source = {
            key: value for key, value in export_source.items()
            if key != "contract_sha256"
        }
        d01_source.update({
            "mp_physics": exp.root.run.mp_physics,
            "microphysics": runner.stock_wrf_physics_inventory(
                exp.root.run.mp_physics).scheme,
        })
        wrf_manifest = {
            "schema": "gpuwm-native-direct-wrf-hierarchy-export-v1",
            "status": "READY",
            "valid_time": exp.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "hierarchy": runner._hierarchy_rows(exp),
            "source": {
                "contract_sha256": export_source["contract_sha256"],
                "domains": {
                    **{"d01": d01_source},
                    **{
                        f"d{domain.grid_id:02d}": {}
                        for domain in exp.domains[1:]
                    },
                },
                "input_provenance": {
                    "input_manifest_sha256": manifest_digest,
                    "decoder_sha256": bridge_digest,
                    "preprocessing": preprocessing,
                    "regular_source_adapter": source,
                    "native_artifact_manifest": (
                        "../hierarchy-artifacts/domain-artifacts.json"),
                    "native_artifact_manifest_sha256": _sha256(
                        artifact_manifest_path),
                },
            },
        }
        if source == "20crv3":
            wrf_manifest["source"]["input_provenance"] = {
                "mapping_sha256": mapped_authorities["mapping"],
                "composition_sha256": mapped_authorities["composition"],
                "input_manifest_sha256": manifest_digest,
                "decoder_sha256": decoder_digests,
                "preprocessing": preprocessing,
                "mapped_target_contract": target_contract,
                "regular_source_adapter": "rw-wps-mapped",
                "regular_source_forcing_times": manifest["valid_times"],
                "regular_source_static_catalog": {"schema": "test-catalog-v1"},
                "regular_source_orography": {"schema": "test-orography-v1"},
                "regular_source_coverage": {"schema": "test-coverage-v1"},
                "regular_source_topology": {"schema": "test-topology-v1"},
                "regular_source_hierarchy_implementation_sha256": {
                    "test": hashlib.sha256(b"hierarchy").hexdigest()},
                "native_artifact_manifest": (
                    "../hierarchy-artifacts/domain-artifacts.json"),
                "native_artifact_manifest_sha256": _sha256(
                    artifact_manifest_path),
            }
            shutil.copy2(static_path, prepared / "native-static.npz")
            shutil.copy2(geometry_path, prepared / "geometry-receipt.json")
        wrf_root = prepared / "wrf-native-input"
        wrf_root.mkdir()
        _write_json(wrf_root / "manifest.json", wrf_manifest)
        proof = {
            "schema": runner._HIERARCHY_PROOF_SCHEMA[source],
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "domain_count": len(exp.domains),
            "forcing_times": [
                (exp.start_time + timedelta(hours=hour)).isoformat()
                for hour in forcing_hours
            ],
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_seconds,
            "input_manifest_sha256": manifest_digest,
            "decoder_sha256": bridge_digest,
            "preprocessing": preprocessing,
            "artifact_receipt": hierarchy_receipt,
            "wrf_manifest": wrf_manifest,
        }
        if source == "20crv3":
            proof = {
                "schema": runner._HIERARCHY_PROOF_SCHEMA[source],
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "domain_count": len(exp.domains),
                "forcing_times": manifest["valid_times"],
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_seconds,
                "target_contract": target_contract,
                "execution_inputs": {
                    "decoders": {
                        role: {
                            "path": source_composition["decoders"][role]["path"],
                            "bytes": 1024 + index,
                            "sha256": decoder_digests[role],
                        }
                        for index, role in enumerate(
                            sorted(runner._TWENTYCRV3_DECODER_ROLES))
                    },
                    "experiment_config": _artifact(
                        experiment_config, str(experiment_config.resolve())),
                    "wps_namelist": _artifact(
                        wps_namelist, str(wps_namelist.resolve())),
                    "geog_root": str((tmp_path / "geog").resolve()),
                    "geog_source_binding": (
                        "per_domain_resolved_dataset_paths_plus_native_"
                        "static_output_sha256"),
                },
                "source_composition": source_composition,
                "preprocessing": preprocessing,
                "hierarchy_workers": 2,
                "root_static": {
                    "path": "native-static.npz",
                    "bytes": (prepared / "native-static.npz").stat().st_size,
                    "sha256": _sha256(prepared / "native-static.npz"),
                    "fields": ["STATIC"],
                },
                "root_geometry": geometry_payload,
                "static_catalog": {"schema": "test-catalog-v1"},
                "source_coverage": {"schema": "test-coverage-v1"},
                "artifact_receipt": hierarchy_receipt,
                "wrf_manifest": wrf_manifest,
                "timing_seconds": {"total": 1.0},
            }
    if source == "gfs":
        proof.update({
            "implementation_sha256": source_identity["implementation_sha256"],
            "git_source_identity": source_identity["git_source_identity"],
        })
    if source == "20crv3":
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode()).hexdigest()
    proof_path = prepared / "proof.json"
    _write_json(proof_path, proof)
    return SimpleNamespace(
        source=source, prepared=prepared, proof=proof_path,
        domain_bundle=domain_bundle,
        source_manifest=source_manifest, experiment=experiment_config,
        wps=wps_namelist, content_sha256=header["content_sha256"],
        run_seconds=exp.run_seconds)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_exact_portable_single_domain_authorities(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(tmp_path, source)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.source == source
    assert inputs.cache_reader.verify_all()["status"] == "PASS"
    assert inputs.physics_receipt["profile"] == runner.PHYSICS_PROFILE
    assert dict(inputs.landuse_identity) == dict(runner._LANDUSE_IDENTITY)
    assert inputs.forcing_hours == ((0, 3) if source == "gfs" else (0, 6, 12))


def _bind_synthetic_preflight_geometry(monkeypatch, *, hierarchy: bool):
    grid = SimpleNamespace(source="20crv3")
    if hierarchy:
        monkeypatch.setattr(
            runner, "validate_native_lambert_contracts",
            lambda exp, path, *, source_name: tuple(
                grid for _ in exp.domains))
    else:
        monkeypatch.setattr(
            runner, "validate_native_lambert_contract",
            lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {
            "STATIC": np.ones((ny, nx), dtype=np.float64)})


def _preflight_fixture(fixture, *, physics_profile=runner.PHYSICS_PROFILE):
    history_interval_seconds = load_experiment(
        fixture.experiment).root.history_interval_s
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=physics_profile,
        run_seconds=fixture.run_seconds,
        history_interval_seconds=history_interval_seconds)


def test_preflight_accepts_exact_member_20crv3_mapped_direct_d01(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture)

    assert inputs.layout == "mapped-direct-d01-v1"
    assert inputs.source_member == "072"
    assert inputs.forcing_hours == (0, 3)
    assert inputs.boundary_interval_seconds == 10800
    assert inputs.cache_identity["source_identity"]["adapter"] \
        == "rw-wps-20crv3-member-grib2-v1"
    assert {"mapped_mapping", "mapped_composition", "mapped_provenance"} \
        <= set(inputs.authority_paths)
    runner._verify_inputs_unchanged(inputs)


def test_20crv3_implemented_unverified_profile_retains_hash_bound_cadence(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "20crv3",
        physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(
        fixture, physics_profile=runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE)

    assert inputs.physics_receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert inputs.physics_receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    plan = inputs.physics_receipt["execution_plan"]
    assert plan["physics_overrides"] == []
    assert plan["source_experiment"]["domains"][0][
        "history_interval_seconds"] == 10800.0
    assert plan["executed_d01"]["domain"][
        "history_interval_seconds"] == 10800.0
    assert plan["execution_overrides"] == []
    assert inputs.physics_receipt["output_cadence"][
        "expected_frame_count"] == 2


@pytest.mark.parametrize(
    "profile",
    (
        runner.THOMPSON_PHYSICS_PROFILE,
        runner.MORRISON_PHYSICS_PROFILE,
        runner.NSSL2_PHYSICS_PROFILE,
    ),
)
def test_20crv3_materialized_profiles_reach_exact_prepared_preflight(
        tmp_path, monkeypatch, profile):
    if profile == runner.THOMPSON_PHYSICS_PROFILE:
        _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "20crv3", physics_profile=profile)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)

    inputs = _preflight_fixture(fixture, physics_profile=profile)

    assert inputs.source == "20crv3"
    assert inputs.physics_receipt["profile"] == profile
    assert inputs.physics_receipt["resolved"]["mp_physics"] in {8, 10, 18}
    assert inputs.physics_receipt["validated_domains"][0]["grid_id"] == 1


def test_20crv3_profile_is_source_scoped_and_does_not_weaken_gfs():
    exp = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")

    with pytest.raises(ValueError, match="experiment physics differs"):
        runner._validate_physics(
            exp, runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE,
            exp.run_seconds, 3600, source="gfs")


def test_20crv3_cache_identity_allows_only_the_legacy_same_scheme_default(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    header = json.loads(
        (fixture.domain_bundle / "prepared-cache" / "header.json").read_text(
            encoding="utf-8"))
    expected = header["identity"]
    observed = json.loads(_canonical(expected))
    assert observed["domain_config"]["run"].pop(
        "nest_microphysics_transition") == "same-scheme-only"

    selected, receipt = runner._resolve_cache_identity_compatibility(
        source="20crv3", observed=observed, expected=expected)

    assert selected == observed
    assert receipt["status"] == "COMPATIBLE_LEGACY_DEFAULT"
    assert receipt["compatibility_overrides"][0]["field"] \
        == "domain_config.run.nest_microphysics_transition"
    observed["domain_config"]["run"]["mp_physics"] = 10
    with pytest.raises(ValueError, match="cache identity differs"):
        runner._resolve_cache_identity_compatibility(
            source="20crv3", observed=observed, expected=expected)


def test_preflight_accepts_exact_member_20crv3_mapped_hierarchy_d01(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3", hierarchy=True)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=True)

    inputs = _preflight_fixture(fixture)

    assert inputs.layout == "mapped-hierarchy-d01-v1"
    assert inputs.domain_bundle_path == fixture.domain_bundle.resolve()
    assert inputs.source_domain_count > 1
    assert inputs.source_member == "072"
    assert inputs.export_source_receipt["mp_physics"] == 6
    runner._verify_inputs_unchanged(inputs)


def test_20crv3_preflight_rejects_stale_mapped_proof_content_hash(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["timing_seconds"]["total"] = 2.0
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="proof content hash is stale"):
        _preflight_fixture(fixture)


def test_20crv3_preflight_rejects_manifest_cadence_drift(tmp_path):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    manifest = json.loads(fixture.source_manifest.read_text(encoding="utf-8"))
    manifest["cadence_seconds"] = 3600
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        _canonical(content).encode()).hexdigest()
    _write_json(fixture.source_manifest, manifest)

    with pytest.raises(ValueError, match="manifest cadence"):
        _preflight_fixture(fixture)


def test_20crv3_preflight_rejects_member_swap_even_with_resealed_proof(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    manifest = json.loads(fixture.source_manifest.read_text(encoding="utf-8"))
    manifest["member"] = "071"
    for row in manifest["files"]:
        row["member"] = "071"
        row["filename"] = row["filename"].replace("mem072_", "mem071_", 1)
        row["path"] = str(Path(row["path"]).with_name(row["filename"]))
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = hashlib.sha256(
        _canonical(content).encode()).hexdigest()
    _write_json(fixture.source_manifest, manifest)
    manifest_sha256 = _sha256(fixture.source_manifest)

    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    composition = proof["source_composition"]
    composition["input_manifest"]["sha256"] = manifest_sha256
    composition["alignment"]["member"] = "071"
    composition["terrain_products"] = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in manifest["files"] if row["role"] == "sfc"
    ]
    composition.pop("receipt_content_sha256")
    composition["receipt_content_sha256"] = hashlib.sha256(
        _canonical(composition).encode()).hexdigest()
    proof.pop("proof_content_sha256")
    proof["proof_content_sha256"] = hashlib.sha256(
        _canonical(proof).encode()).hexdigest()
    _write_json(fixture.proof, proof)

    with pytest.raises(ValueError, match="source identity differs"):
        _preflight_fixture(fixture)


def test_20crv3_preflight_and_execution_bind_copied_provenance(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "20crv3")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    inputs = _preflight_fixture(fixture)
    inputs.authority_paths["mapped_provenance"].write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="inputs changed during execution"):
        runner._verify_inputs_unchanged(inputs)


def test_materialized_wsm6_physics_receipt_is_canonical():
    text = (ROOT / "configs" / "gfs_wrf_direct_proof.toml").read_text(
        encoding="utf-8")
    _rendered, exp, _materialization = (
        runner._render_materialized_experiment(
            text, source="gfs", profile=runner.PHYSICS_PROFILE))
    receipt = runner._validate_physics(
        exp, runner.PHYSICS_PROFILE, exp.run_seconds, 3600, source="gfs")

    assert receipt["resolved"] == {
        **runner.single_domain_runtime_switches(runner.PHYSICS_PROFILE),
        "radiation_scheme_ids": [0, 1],
    }


def test_preflight_accepts_exact_nssl2_validation_candidate_and_binds_receipt(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.NSSL2_PHYSICS_PROFILE)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.NSSL2_PHYSICS_PROFILE
    assert receipt["readiness"] == "VALIDATION_CANDIDATE"
    assert receipt["resolved"]["mp_physics"] == 18
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["epssm"] == 0.5
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["nssl2_contract"]["contract_id"] == (
        "wrf-v4.6.1-nssl-mp18-two-moment-hail-ccn-density-v1"
    )
    assert receipt["nssl2_contract"]["selector"] == 18
    assert receipt["nssl2_contract"]["resolved_default_mode"] == {
        "two_moment": True,
        "hail": True,
        "predicted_ccn": True,
        "density_moments": 2,
        "sixth_moments": 0,
    }
    assert receipt["radiation_substitution"] == {
        "contract": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
        "requested_wrf_scheme_ids": [4, 4],
        "resolved_gpuwm_scheme_ids": [4, 4],
        "resolved_gpuwm_solver": "RTE+RRTMGP",
    }


def test_nssl2_profile_rejects_wsm6_experiment_before_cache_restore(tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")

    with pytest.raises(ValueError, match="NSSL-2 validation-candidate"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.NSSL2_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_materialized_morrison_for_named_sources(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(
        tmp_path, source, physics_profile=runner.MORRISON_PHYSICS_PROFILE)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.MORRISON_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.MORRISON_PHYSICS_PROFILE
    assert receipt["readiness"] == "MODEL_VALIDATED_RUNTIME_PROFILE"
    assert receipt["resolved"]["mp_physics"] == 10
    assert receipt["resolved"]["morr_rimed_ice"] == 1
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["morrison_contract"]["rimed_ice_category"] == "hail"


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_accepts_guarded_thompson_for_each_prepared_source(
        tmp_path, monkeypatch, source):
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, source, physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    grid = SimpleNamespace(source=source)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, path, *, source_name: grid)
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source=source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    receipt = inputs.physics_receipt
    assert receipt["profile"] == runner.THOMPSON_PHYSICS_PROFILE
    assert receipt["readiness"] == "MODEL_VALIDATED_EXPERIMENTAL_RUNTIME"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["epssm"] == 0.5
    assert receipt["resolved"]["radiation_scheme_ids"] == [0, 1]
    contract = receipt["thompson_contract"]
    assert contract["selector"] == 8
    assert contract["transported_fields"] == [
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr"]
    assert contract["native_reflectivity"] == {
        "field": "REFL_10CM",
        "producer": "classic-Thompson same-call graupel-number shadow",
        "consumer": "gpuwm.core.refl.consume_refl_10cm",
        "fallback": None,
    }
    table = contract["table_authority"]
    assert table["root"] == str(runtime.root)
    assert table["payload_bytes"] == 379_839_912
    assert table["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes,
         "sha256": asset.sha256}
        for asset in runtime.assets
    ]
    table_identity = {
        key: table[key]
        for key in ("schema", "table_set", "wrf_version", "wrf_commit", "assets")
    }
    assert table["identity_sha256"] == hashlib.sha256(
        _canonical(table_identity).encode("ascii")).hexdigest()
    assert len(contract["implementation_sha256"]) == len(
        runner._THOMPSON_IMPLEMENTATION_FILES)
    assert len([
        name for name in inputs.file_sha256
        if name.startswith("thompson_table_")
    ]) == len(runtime.assets)


def test_thompson_profile_rejects_wsm6_cache_before_restore(
        tmp_path, monkeypatch):
    _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(tmp_path, "gfs")

    with pytest.raises(ValueError, match="guarded Thompson MP8"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_wsm6_profile_rejects_thompson_cache_before_restore(
        tmp_path, monkeypatch):
    _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)

    with pytest.raises(ValueError, match="supported prepared-cache profile"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_thompson_runs_without_the_retired_enable_gate(tmp_path, monkeypatch):
    """mp8 no longer demands GPUWM_EXPERIMENTAL_THOMPSON_MP8=1.

    The gate predated the packaging promotion: the classic tables ship
    as package data and `gpuwm doctor` byte-validates all four.  Keeping
    it meant the wizard's own default microphysics failed twice at
    runtime on a machine doctor had just called clean, with neither
    variable named in any document.  The table-root binding, which is
    what actually protected anything, is unchanged.
    """
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    fixture = _prepared_fixture(
        tmp_path, "gfs", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    exp = load_experiment(fixture.experiment)
    receipt = runner._validate_physics(
        exp, runner.THOMPSON_PHYSICS_PROFILE, exp.run_seconds, 3600)
    guard = receipt["thompson_contract"]["guard"]
    assert "experimental_runtime_environment" not in guard
    assert guard["table_root_source"] == "environment override"
    # And with no override at all, the packaged directory resolves.
    monkeypatch.delenv(runner.THOMPSON_TABLE_ROOT_ENV)
    from gpuwm.physics_compat import (packaged_thompson_table_root,
                                      thompson_table_root)
    assert Path(thompson_table_root()) == packaged_thompson_table_root()

    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(runtime.root))
    changed = tmp_path / "other-tables"
    changed.mkdir()
    monkeypatch.setenv(runner.THOMPSON_TABLE_ROOT_ENV, str(changed))
    with pytest.raises(RuntimeError,
                       match="changed between preflight and run"):
        runner._verify_thompson_runtime_environment(receipt)
    assert receipt["thompson_contract"]["table_authority"]["root"] == str(
        runtime.root)


def test_thompson_table_drift_is_rejected_by_preflight(tmp_path, monkeypatch):
    runtime = _bind_test_thompson_runtime(tmp_path, monkeypatch)
    fixture = _prepared_fixture(
        tmp_path, "era5", physics_profile=runner.THOMPSON_PHYSICS_PROFILE)
    (runtime.root / runtime.assets[0].filename).write_bytes(b"drift")

    with pytest.raises(ValueError, match="test Thompson table drift"):
        runner.preflight_prepared_forecast(
            source="era5", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.THOMPSON_PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_preflight_derives_hash_bound_hierarchy_d01_bundle(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs", hierarchy=True)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contracts",
        lambda exp, path, *, source_name: tuple(grid for _ in exp.domains))
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.layout == "hierarchy-d01-v1"
    assert inputs.domain_bundle_path == fixture.domain_bundle.resolve()
    assert inputs.source_domain_count > 1
    assert inputs.experiment.domains == (inputs.experiment.root,)


def test_hierarchy_d01_nssl2_authority_uses_dynamic_microphysics_label(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(
        tmp_path, "gfs", hierarchy=True,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE)
    grid = SimpleNamespace(source="gfs")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contracts",
        lambda exp, path, *, source_name: tuple(grid for _ in exp.domains))
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, actual_grid, cfg: {"status": "PASS"})
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, actual_grid, ny, nx: {"STATIC": np.ones((ny, nx))})

    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.NSSL2_PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)

    assert inputs.export_source_receipt["mp_physics"] == 18
    assert inputs.export_source_receipt["microphysics"] == "NSSL-2"


def test_preflight_rejects_explicit_hierarchy_bundle_outside_manifest(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs", hierarchy=True)
    wrong = tmp_path / "other-d01"
    wrong.mkdir()

    with pytest.raises(ValueError, match="manifest-derived hierarchy d01"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600,
            domain_bundle=wrong)


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_preflight_rejects_wrong_source_adapter_even_when_bundle_is_self_consistent(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(
        tmp_path, source, adapter="era5-grib1-direct-v1" if source == "gfs"
        else "gfs-pgrb2-0p25-direct-v1")
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "verify_native_static_receipt", lambda *args: {})
    monkeypatch.setattr(runner, "load_native_static_cache", lambda *args: {})

    with pytest.raises(ValueError, match="source identity differs"):
        runner.preflight_prepared_forecast(
            source=source, prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_preflight_rejects_pinned_proof_drift_before_cache_restore(
        tmp_path):
    fixture = _prepared_fixture(tmp_path, "gfs")
    with pytest.raises(ValueError, match="proof SHA differs"):
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256="0" * 64,
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_preflight_rejects_resolved_wrf_contract_drift(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "era5")
    proof = json.loads(fixture.proof.read_text(encoding="utf-8"))
    proof["export"]["source"]["resolved_physics_contract_sha256"] = "0" * 64
    _write_json(fixture.proof, proof)
    monkeypatch.setattr(
        runner, "validate_native_lambert_contract", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "verify_native_static_receipt", lambda *args: {})
    monkeypatch.setattr(runner, "load_native_static_cache", lambda *args: {})

    with pytest.raises(ValueError, match="export source hashes differ"):
        runner.preflight_prepared_forecast(
            source="era5", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600)


def test_output_claim_is_create_only_and_preserves_existing_tree(tmp_path):
    output = tmp_path / "run"
    assert runner.claim_output_directory(output) == output.resolve()
    marker = output / "owned-by-first-run.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.claim_output_directory(output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_output_claim_refuses_to_modify_the_prepared_input_tree(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()

    with pytest.raises(ValueError, match="protected input tree"):
        runner.claim_output_directory(
            prepared / "forecast", protected_roots=(prepared,))

    assert not (prepared / "forecast").exists()


def test_durable_wrfout_inventory_reads_atomic_writer_subdirectory(tmp_path):
    output = tmp_path / "model-output"
    wrfout = output / "wrfout"
    wrfout.mkdir(parents=True)
    first = wrfout / "wrfout_d01_2026-07-21_12_00_00"
    second = wrfout / "wrfout_d01_2026-07-21_13_00_00"
    ignored = output / "wrfout_d01_wrong_location"
    first.write_bytes(b"first-complete-frame")
    second.write_bytes(b"second-complete-frame")
    ignored.write_bytes(b"must-not-be-inventoried")

    inventory = runner._durable_wrfout_inventory(output)

    assert inventory == [
        {
            "path": str(first.resolve()),
            "bytes": first.stat().st_size,
            "sha256": _sha256(first),
        },
        {
            "path": str(second.resolve()),
            "bytes": second.stat().st_size,
            "sha256": _sha256(second),
        },
    ]


def test_restored_source_adapter_hint_is_optional_but_never_conflicting():
    runner._validate_restored_source_adapter({}, "gfs")
    runner._validate_restored_source_adapter({"source_adapter": "gfs"}, "gfs")
    runner._validate_restored_source_adapter(
        {"source_adapter": "mapped"}, "20crv3")

    with pytest.raises(ValueError, match="source adapter differs"):
        runner._validate_restored_source_adapter(
            {"source_adapter": "era5"}, "gfs")
    with pytest.raises(ValueError, match="source adapter differs"):
        runner._validate_restored_source_adapter(
            {"source_adapter": "20crv3"}, "20crv3")


def test_disabled_optional_physics_records_zero_updates():
    assert runner._physics_update_count(None) == 0
    assert runner._physics_update_count(SimpleNamespace(update_count=7)) == 7

    with pytest.raises(ValueError, match="cannot be negative"):
        runner._physics_update_count(SimpleNamespace(update_count=-1))


def test_post_restore_content_pin_is_checked_before_forecast():
    expected = "a" * 64
    runner._validate_restored_cache_receipt({
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "RESTORED",
        "content_sha256": expected,
    }, expected)

    with pytest.raises(ValueError, match="caller-pinned content"):
        runner._validate_restored_cache_receipt({
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "RESTORED",
            "content_sha256": "b" * 64,
        }, expected)


def test_source_neutral_surface_contract_accepts_exact_noah_inventory():
    shape = (3, 4)
    fields = {
        name: np.ones((4, *shape) if name in {"TSLB", "SMOIS", "SH2O"}
                      else shape, dtype=np.float32)
        for name in runner._CANONICAL_SURFACE_FIELDS
    }
    fields["LANDMASK"][:] = 0.0
    fields["XLAND"][:] = 2.0

    # The prepared-surface contract is Noah's; the soil rank it enforces is
    # now resolved from the SCHEME (config.soil_layer_count), so the cfg
    # stand-in has to say which scheme it is rather than only how big.
    actual = _validate_prepared_surface(
        SimpleNamespace(fields=fields),
        SimpleNamespace(ny=3, nx=4, sf_surface_physics=2, num_soil_layers=4))

    assert actual is fields


def test_source_neutral_near_surface_contract_accepts_exact_staggering():
    cfg = SimpleNamespace(ny=3, nx=4)
    result = SimpleNamespace(
        surface_pressure=np.full((3, 4), 95_000.0),
        surface_qv=np.full((3, 4), 0.01),
    )
    met = SimpleNamespace(fields={
        "T2": np.full((3, 4), 290.0),
        "SKINTEMP": np.full((3, 4), 291.0),
        "U10": np.zeros((3, 5)),
        "V10": np.zeros((4, 4)),
        "LANDSEA": np.ones((3, 4)),
    })

    actual = _validate_prepared_near_surface(result, met, cfg)

    assert actual["surface_pressure"].shape == (3, 4)
    assert actual["U10"].shape == (3, 5)
    assert actual["V10"].shape == (4, 4)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("surface_pressure", np.asarray(95_000.0), "surface_pressure has shape"),
        ("surface_qv", np.full((2, 3), np.nan), "surface_qv must be finite"),
        ("T2", np.full((2, 3), 500.0), "T2 is outside the physical range"),
        ("U10", np.zeros((2, 3)), "U10 has shape"),
        ("LANDSEA", np.full((2, 3), 0.5), "LANDSEA must be exactly binary"),
    ],
)
def test_source_neutral_near_surface_contract_rejects_bad_arrays(
        field, value, error):
    cfg = SimpleNamespace(ny=2, nx=3)
    result = SimpleNamespace(
        surface_pressure=np.full((2, 3), 95_000.0),
        surface_qv=np.full((2, 3), 0.01),
    )
    met_fields = {
        "T2": np.full((2, 3), 290.0),
        "SKINTEMP": np.full((2, 3), 291.0),
        "U10": np.zeros((2, 4)),
        "V10": np.zeros((3, 3)),
        "LANDSEA": np.ones((2, 3)),
    }
    if field in {"surface_pressure", "surface_qv"}:
        setattr(result, field, value)
    else:
        met_fields[field] = value

    with pytest.raises(ValueError, match=error):
        _validate_prepared_near_surface(
            result, SimpleNamespace(fields=met_fields), cfg)


def test_source_neutral_surface_contract_rejects_land_identity_drift():
    shape = (2, 3)
    fields = {
        name: np.ones((4, *shape) if name in {"TSLB", "SMOIS", "SH2O"}
                      else shape, dtype=np.float32)
        for name in runner._CANONICAL_SURFACE_FIELDS
    }
    fields["LANDMASK"][:] = 0.0
    fields["XLAND"][:] = 1.0

    with pytest.raises(ValueError, match="XLAND differs"):
        _validate_prepared_surface(
            SimpleNamespace(fields=fields),
            SimpleNamespace(ny=2, nx=3, sf_surface_physics=2,
                            num_soil_layers=4))


def test_output_due_thompson_consumes_native_refl_10cm_stash():
    state = SimpleNamespace(
        qv=object(), physics=SimpleNamespace(mp_physics=8))
    native = object()
    consumed = []

    result = runner._consume_due_native_refl_10cm(
        state, 1, lambda actual: consumed.append(actual) or native)

    assert result is native
    assert consumed == [state]
    assert runner._consume_due_native_refl_10cm(
        state, 0, lambda actual: pytest.fail("initial frame consumed a stash")) \
        is None


def _retime_single_domain(exp, *, cadence_seconds, run_seconds=None):
    run_seconds = (
        float(exp.run_seconds) if run_seconds is None else float(run_seconds))
    domain = replace(
        exp.root,
        history_interval_s=float(cadence_seconds),
        run=replace(
            exp.root.run,
            output_interval_s=float(cadence_seconds),
            run_seconds=run_seconds,
        ),
    )
    return replace(exp, run_seconds=run_seconds, domains=(domain,))


@pytest.mark.parametrize(
    ("cadence_seconds", "expected_offsets", "expected_suffixes",
     "last_equals_run_end"),
    (
        (900.0, [0.0, 900.0, 1800.0, 2700.0, 3600.0, 4500.0,
                 5400.0, 6300.0, 7200.0, 8100.0, 9000.0, 9900.0,
                 10800.0], ["00_00_00", "03_00_00"], True),
        (10800.0, [0.0, 10800.0], ["00_00_00", "03_00_00"], True),
        (7200.0, [0.0, 7200.0], ["00_00_00", "02_00_00"], False),
    ),
)
def test_hash_bound_history_cadence_uses_floor_schedule_for_any_due_output(
        cadence_seconds, expected_offsets, expected_suffixes,
        last_equals_run_end):
    base = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    exp = _retime_single_domain(base, cadence_seconds=cadence_seconds)

    receipt = runner._validate_hash_bound_history_cadence(
        exp, cadence_seconds)
    schedule = runner._history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=cadence_seconds)

    assert [record[0] for record in schedule] == expected_offsets
    assert schedule[0][2].endswith(expected_suffixes[0])
    assert schedule[-1][2].endswith(expected_suffixes[1])
    assert receipt["expected_frame_count"] == len(schedule)
    assert receipt["last_scheduled_offset_seconds"] == expected_offsets[-1]
    assert receipt["last_scheduled_valid_time"] == schedule[-1][1].isoformat()
    assert receipt["run_end_offset_seconds"] == exp.run_seconds
    assert receipt["last_scheduled_equals_run_end"] is last_equals_run_end
    assert receipt["run_end_frame_scheduled"] is last_equals_run_end


def test_hash_bound_history_cadence_rejects_cli_mismatch_and_nonstep_cadence():
    base = load_experiment(ROOT / "configs" / "gfs_wrf_direct_proof.toml")
    subhourly = _retime_single_domain(base, cadence_seconds=900.0)
    with pytest.raises(ValueError, match="exactly match the hash-bound"):
        runner._validate_hash_bound_history_cadence(subhourly, 1800.0)

    nonstep = _retime_single_domain(base, cadence_seconds=71.0)
    with pytest.raises(ValueError, match="whole number of exact model time steps"):
        runner._validate_hash_bound_history_cadence(nonstep, 71.0)


def test_cli_requires_explicit_source_physics_run_and_history_output_contract():
    args = runner._parse_args([
        "--source", "gfs",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.PHYSICS_PROFILE,
        "--run-seconds", "10800",
        "--history-interval-seconds", "900",
        "--io-mode", "history",
        "--outdir", "run",
    ])
    assert args.source == "gfs"
    assert args.physics_profile == runner.PHYSICS_PROFILE
    assert args.io_mode == "history"
    assert args.history_interval_seconds == 900.0


def test_cli_accepts_explicit_nssl2_validation_candidate_profile():
    args = runner._parse_args([
        "--source", "era5",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.NSSL2_PHYSICS_PROFILE,
        "--run-seconds", "43200",
        "--history-interval-seconds", "3600",
        "--io-mode", "history",
        "--outdir", "run",
    ])

    assert args.physics_profile == runner.NSSL2_PHYSICS_PROFILE


def test_cli_accepts_explicit_guarded_thompson_profile():
    args = runner._parse_args([
        "--source", "gfs",
        "--prepared-root", "prepared",
        "--proof-sha256", "1" * 64,
        "--source-manifest-sha256", "2" * 64,
        "--prepared-content-sha256", "3" * 64,
        "--experiment-config", "experiment.toml",
        "--wps-namelist", "namelist.wps",
        "--physics-profile", runner.THOMPSON_PHYSICS_PROFILE,
        "--run-seconds", "10800",
        "--history-interval-seconds", "3600",
        "--io-mode", "history",
        "--outdir", "run",
    ])

    assert args.physics_profile == runner.THOMPSON_PHYSICS_PROFILE


# ---------------------------------------------------------------------------
# First-time-user papercuts from the 2026-07-30 Linux pilots.
# ---------------------------------------------------------------------------

def _wizard_config(tmp_path):
    """A real wizard-emitted single-domain TOML (not a hand-built stub)."""
    from gpuwm.cli import main as cli_main

    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=25.76,-80.19", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    assert rc == 0
    return out


def test_clock_flags_default_to_the_hash_bound_experiment(tmp_path, capsys):
    """PP-15: flags that can hold only one value must not be required.

    --run-seconds and --history-interval-seconds are validated to equal
    the hash-bound experiment exactly, so requiring the user to retype
    them bought nothing -- and cost a wasted cycle to anyone who tried
    to shorten a retry and was told the values must match.
    """
    config = _wizard_config(tmp_path)
    capsys.readouterr()

    run_seconds, cadence = runner._clock_defaults(config)
    exp = load_experiment(config)
    assert run_seconds == float(exp.run_seconds)
    assert cadence == float(exp.root.history_interval_s)

    args = SimpleNamespace(
        experiment_config=config, run_seconds=None,
        history_interval_seconds=None)
    runner._resolve_clock_arguments(args)
    assert args.run_seconds == run_seconds
    assert args.history_interval_seconds == cadence
    noted = capsys.readouterr().err
    assert "--run-seconds defaulted to" in noted
    assert "--history-interval-seconds defaulted to" in noted

    # Explicit values are left exactly alone (and stay subject to the
    # unchanged exact-match guard downstream).
    explicit = SimpleNamespace(
        experiment_config=config, run_seconds=1.0,
        history_interval_seconds=2.0)
    runner._resolve_clock_arguments(explicit)
    assert (explicit.run_seconds, explicit.history_interval_seconds) == (
        1.0, 2.0)
    assert capsys.readouterr().err == ""


def test_clock_flags_are_optional_in_the_parser(tmp_path):
    config = _wizard_config(tmp_path)
    args = runner._parse_args([
        "--source", "gfs", "--prepared-root", str(tmp_path / "prep"),
        "--proof-sha256", "0" * 64,
        "--source-manifest-sha256", "1" * 64,
        "--prepared-content-sha256", "2" * 64,
        "--experiment-config", str(config),
        "--wps-namelist", str(config.with_suffix(".namelist.wps")),
        "--physics-profile", next(iter(runner.PHYSICS_PROFILES)),
        "--io-mode", "history", "--outdir", str(tmp_path / "run"),
    ])
    assert args.run_seconds is None
    assert args.history_interval_seconds is None


def test_clock_defaults_refuse_a_config_they_cannot_read(tmp_path):
    bad = tmp_path / "not-an-experiment.toml"
    bad.write_text("[nothing]\nhere = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pass them explicitly"):
        runner._clock_defaults(bad)


def test_the_stability_abort_names_the_dominant_term_and_the_remedy():
    """PP-9: the abort quoted one `cfl` number and no remedy.

    `cfl` is dt * max(u/dx, w/dz_min) over two terms three orders of
    magnitude apart: with the certified 49-level eta profile dz_min is
    about 14.3 m, so a tropical abort read `cfl 27.70` while its
    horizontal Courant number was 0.15.  Nothing said "shorten dt".
    """
    run = SimpleNamespace(dt=60.0, dx=12_000.0, ztop=20_000.0)
    state = SimpleNamespace(dz_min=14.3)

    tropical = runner._stability_diagnosis(
        {"u_max": 29.3, "w_max": 6.62}, state, run)
    assert "VERTICAL Courant 27.7" in tropical
    assert "14.3 m thinnest layer" in tropical
    assert "REMEDY: retry with a lower dt" in tropical
    assert "time_step to about 15 s" in tropical
    assert "+22% wall time" in tropical

    # A genuine horizontal violation is named as such, not mislabelled.
    horizontal = runner._stability_diagnosis(
        {"u_max": 4000.0, "w_max": 0.1}, state, run)
    assert "HORIZONTAL Courant 20.0" in horizontal
    assert "dx 12.000 km" in horizontal
    assert "REMEDY: retry with a lower dt" in horizontal


def test_the_materializer_accepts_the_wizard_s_own_advisory_fetch_table(
        tmp_path):
    """Node-3 #2: `unknown table(s)/top-level key(s) ['fetch']`.

    The wizard writes that advisory table into every config it emits,
    and `gpuwm check` and `rw-wps` both accept it -- only the
    materializer, which handed the raw dict to build_experiment, did
    not.  Materialization has to run BEFORE the front door, so this
    rejected the wizard's own output at the first step of the route.
    """
    config = _wizard_config(tmp_path)
    base = config.read_text(encoding="utf-8")
    assert "[fetch]" in base

    rendered, exp, receipt = runner._render_materialized_experiment(
        base, source="gfs",
        profile="morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    # Preserved verbatim, not silently dropped: it is provenance.
    assert "[fetch]" in rendered
    assert receipt["profile_validation"]["profile"] == (
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    assert (receipt["base_non_physics_descriptor_sha256"]
            == receipt["generated_non_physics_descriptor_sha256"])

    # A malformed [fetch] table is still refused rather than ignored.
    broken = base.replace('source = "gfs"', 'source = "not-a-source"')
    with pytest.raises(ValueError):
        runner._render_materialized_experiment(
            broken, source="gfs",
            profile="morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
