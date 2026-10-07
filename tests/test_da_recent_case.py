from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import da_recent_case as case


def test_actual_public_authority_matches_model_clock_and_pinned_profile(tmp_path):
    doc = case.author(tmp_path / "case", start="2026-10-03T21:00:00Z", latitude=31,
                      longitude=-89, nx=33, ny=33, smoke=True, smoke_run_hours=2)
    assert doc["source"] == "rap-native"
    assert doc["members"] == 4 and doc["free_forecast_seconds"] == 120
    assert doc["analysis_times_utc"] == ["2026-10-03T22:00:00Z"]
    assert doc["history_interval_seconds"] == 120
    assert doc["initial_source_coverage"]["status"] == "PASS"
    authority = case.read_json(tmp_path / "case/authority/authority-receipt.json")
    physics = authority["normalized_selected_physics"]["resolved"]
    assert physics["moist_cq"] is True
    assert physics["mp_physics"] == 28 and physics["num_soil_layers"] == 9
    assert physics["bl_mynn_version"] == "gsd_41"
    argv = doc["prepare_argv"]
    supplements = [argv[index + 1] for index, value in enumerate(argv) if value == "--supplement"]
    assert len(supplements) == 3
    assert [row.rsplit("f", 1)[-1] for row in supplements] == ["00.grib2", "01.grib2", "02.grib2"]
    assert "--sealed-prepared-cache" not in argv and "--physics-profile" not in argv
    assert not any(row["role"] == "independent_reference_analysis" for row in doc["model_objects"])
    assert doc["independent_reference_status"] == "not requested for integration smoke"
    assert doc["bbox"][0] < -89 < doc["bbox"][2]
    assert doc["bbox"][1] < 31 < doc["bbox"][3]
    obs = case.read_json(tmp_path / "case/observation-case.json")
    assert len(obs["analysis_times_utc"]) == 6
    assert obs["smoke_inventory_selection"] == "--first-slot-only required"
    # The roster is discovered from the domain, not a regional default.
    assert {"KDGX", "KLIX", "KMOB"} <= set(obs["radar_sites"])
    assert obs["radar_sites"] == sorted(obs["radar_sites"])
    assert obs["radar_roster_basis"] == doc["radar_roster_basis"]
    assert "250 km reach touches the domain" in obs["radar_roster_basis"]
    # Real PIB metadata lies in the accepted last-row half-cell, beyond the
    # center-extrema bbox that previously missed it during station discovery.
    pib_lat, pib_lon = 31.4671, -89.3371
    assert doc["model_mass_bbox"][3] < pib_lat < doc["bbox"][3]
    assert doc["bbox"][0] < pib_lon < doc["bbox"][2]
    assert obs["bbox_basis"] == "native full mass-cell corner footprint"
    assert obs["truth_bbox"][0] < doc["bbox"][0] < doc["bbox"][2] < obs["truth_bbox"][2]
    assert obs["truth_bbox"][1] < doc["bbox"][1] < doc["bbox"][3] < obs["truth_bbox"][3]
    assert obs["truth_padding_m"] == 75000
    padding = obs["truth_padding_provenance"]
    assert padding["minimum_ground_rim_bound_m"] >= 75000
    assert padding["virtual_grid_definition"]["translated_from"] == padding["model_grid_definition"]
    assert padding["virtual_grid_definition"]["offset_cells"] == [-padding["rim_cells_per_side"]] * 2
    assert len(padding["native_library_sha256"]) == 64
    assert obs["surface_networks_basis"] == "domain-bbox"
    assert {"MS_ASOS", "LA_ASOS", "AL_ASOS"} <= set(obs["surface_networks"])


def test_actual_author_cli_preserves_generic_surface_network_authority(tmp_path):
    out = tmp_path / "generic"
    assert case.main(["--out", str(out), "--nx", "33", "--ny", "33", "--smoke",
                      "--surface-networks", "KS_ASOS", "OK_ASOS"]) == 0
    obs = case.read_json(out / "observation-case.json")
    assert obs["surface_networks"] == ["KS_ASOS", "OK_ASOS"]
    assert obs["surface_networks_basis"] == "caller"
    assert case.normalize_surface_networks("KS_ASOS, OK_ASOS") == obs["surface_networks"]


@pytest.mark.parametrize("networks", [[], [1], ["KS_ASOS", "KS_ASOS"], [""], ["bad-network"]])
def test_invalid_surface_network_authority_is_refused(networks):
    with pytest.raises(ValueError, match="surface networks"):
        case.normalize_surface_networks(networks)


def test_author_declares_actual_large_domain_coverage_refusal(tmp_path):
    doc = case.author(tmp_path / "large", start="2026-10-03T21:00:00Z", latitude=31,
                      longitude=-89, nx=601, ny=601)
    assert doc["initial_source_coverage"]["status"] == "REFUSED"
    # The window is every target's bilinear cell since f807525cd (WPS
    # metgrid's sixteen-point stencil falls back to four-point at HRRR's
    # own edge), one cell narrower each side than the parabolic halo the
    # refusal named before (j=-39..565); the domain still leaves HRRR.
    assert "j=-38..564" in doc["initial_source_coverage"]["error"]
    # About 27 GiB per member: one per card inside the 28 GiB budget, never two.
    assert doc["native_memory_plan"]["packing_candidate_per_32gib_card"] == 1


def fixture(tmp_path, monkeypatch):
    """Synthetic receipt fixture. No fixture hashes describe weather data."""
    authority, prepared = tmp_path / "authority", tmp_path / "prepared"
    authority.mkdir(); prepared.mkdir()
    case.dump(authority / "authority-receipt.json", {"synthetic": True})
    case.write(authority / "experiment.toml", "synthetic config")
    case.write(authority / "namelist.wps", "synthetic WPS")
    case.dump(prepared / "proof.json", {"synthetic": True})
    case.dump(prepared / "source-evidence/input-manifest.json", {"synthetic": True})
    case.dump(prepared / "prepared-cache/header.json", {"content_sha256": "c" * 64})
    grid_path = prepared / "wrf-native-input/wrfinput_d01"
    case.write(grid_path, "synthetic grid fixture, not meteorological fields")
    obs = tmp_path / "radar.nc"; case.write(obs, "synthetic radar fixture")
    surface = tmp_path / "surface.json"; case.dump(surface, {"synthetic": True})
    plan = {"schema": "gpuwm-da.recent-case-plan.v1", "source": case.SOURCE,
            "initial_source": "hrrr-native", "physics_profile": case.PROFILE,
            "initial_source_coverage": {"status": "PASS"}, "nx": 33, "ny": 33, "nz": 49,
            "authority_dir": str(authority), "authority_receipt_sha256": case.sha(authority / "authority-receipt.json"),
            "model_start_utc": "2026-10-03T21:00:00Z", "analysis_times_utc": ["2026-10-03T22:00:00Z"],
            "forecast_fork_utc": "2026-10-03T22:00:00Z", "run_seconds": 7200,
            "free_forecast_seconds": 120, "members": 4, "smoke_only": True,
            "model_objects": [{"bucket": "noaa-rap-pds", "key": "synthetic-model"}]}
    slot = {"obs": str(obs), "obs_sha256": case.sha(obs), "grid_wrfout": str(grid_path),
            "grid_wrfout_sha256": case.sha(grid_path), "leg_seconds": 3600,
            "analysis_time": "2026-10-03T22:00:00Z"}
    slots = {"schema": "da-rerun.recent-slots.v1", "status": "prepared-smoke-first-slot",
             "model_start_utc": plan["model_start_utc"], "forecast_fork_utc": plan["forecast_fork_utc"],
             "slots": [slot], "surface_obs": str(surface), "surface_sha256": case.sha(surface)}
    fetched = {"schema": "da-rerun.recent-fetch.v1", "status": "complete", "fetch_processes": 1,
               "objects": [{"bucket": "noaa-rap-pds", "key": "synthetic-model", "bytes": 5,
                            "sha256": "a" * 64, "path": str(tmp_path / "retired-raw")}]}
    slots["fetch_receipt_sha256"] = case.document_sha(fetched)
    inputs = SimpleNamespace(experiment=SimpleNamespace(root=SimpleNamespace(run=SimpleNamespace(nx=33, ny=33, nz=49)),
                               start_time=datetime(2026, 10, 3, 21)), proof={"initial_source": {"source": "hrrr-native"}})
    calls = []
    def preflight(**kwargs):
        calls.append(kwargs)
        return inputs
    grid = SimpleNamespace(nx=33, ny=33, nz=49, identity_sha256=lambda: "d" * 64)
    monkeypatch.setattr(case, "_prepared_preflight", preflight)
    monkeypatch.setattr(case, "_read_grid", lambda path: grid)
    def radar(path, grid):
        calls.append(("radar", path))
        return {"valid_time": "2026-10-03T22:00:00Z"}
    def surface_read(path):
        calls.append(("surface", path))
        return {"valid_times": ["2026-10-03T22:00:00"], "reports": []}
    monkeypatch.setattr(case, "_verify_radar", radar)
    monkeypatch.setattr(case, "_verify_surface", surface_read)
    paths = {"plan": tmp_path / "case-plan.json", "slots": tmp_path / "slots.json", "fetch": tmp_path / "fetch.json"}
    for key, doc in (("plan", plan), ("slots", slots), ("fetch", fetched)):
        case.dump(paths[key], doc)
    kwargs = {"prepared_root": prepared, "slots_path": paths["slots"], "fetch_receipt_path": paths["fetch"],
              "engine_sha": "e" * 40, "run_out": tmp_path / "run", "manifest_path": tmp_path / "run.json"}
    return plan, slots, fetched, paths, kwargs, calls, inputs


def test_finalize_binds_real_file_bytes_and_calls_public_readers(tmp_path, monkeypatch):
    plan, slots, fetched, paths, kwargs, calls, inputs = fixture(tmp_path, monkeypatch)
    doc = case.finalize(paths["plan"], **kwargs)
    assert doc["members"] == 4 and doc["free_forecast_seconds"] == 120
    assert doc["source"] == "rap-native" and doc["history_interval_seconds"] == 120
    assert doc["surface_dewpoint_error_k"] == 2
    assert doc["proof_sha256"] == case.sha(kwargs["prepared_root"] / "proof.json")
    assert calls[0]["physics_profile"] == case.PROFILE
    assert calls[0]["prepared_content_sha256"] == "c" * 64
    assert [call[0] for call in calls[1:]] == ["radar", "surface"]
    assert case.read_json(kwargs["manifest_path"]) == doc
    with pytest.raises(ValueError, match="create-only"):
        case.finalize(paths["plan"], **kwargs)


def test_earlier_declared_smoke_plan_records_four_member_runtime_contract(tmp_path, monkeypatch):
    plan, slots, fetched, paths, kwargs, calls, inputs = fixture(tmp_path, monkeypatch)
    plan["members"] = 32
    case.dump(paths["plan"], plan)
    doc = case.finalize(paths["plan"], **kwargs)
    assert doc["members"] == 4 and doc["authored_member_count"] == 32


@pytest.mark.parametrize("problem", ["clock", "different_grid", "changed_radar", "partial_fetch", "missing_model", "wrong_donor", "wrong_scope", "geometry_refused", "forcing_short"])
def test_finalize_refuses_mismatched_preparation_or_observation_authorities(tmp_path, monkeypatch, problem):
    plan, slots, fetched, paths, kwargs, calls, inputs = fixture(tmp_path, monkeypatch)
    if problem == "clock": slots["slots"][0]["analysis_time"] = "2026-10-03T23:00:00Z"
    if problem == "different_grid":
        other = tmp_path / "different-grid.nc"; case.write(other, "other")
        slots["slots"][0].update(grid_wrfout=str(other), grid_wrfout_sha256=case.sha(other))
    if problem == "changed_radar": case.write(slots["slots"][0]["obs"], "changed bytes")
    if problem == "partial_fetch": fetched["status"] = "in-progress"
    if problem == "missing_model": fetched["objects"] = []
    if problem == "wrong_donor": inputs.proof["initial_source"]["source"] = "rap-native"
    if problem == "wrong_scope": slots["status"] = "prepared"
    if problem == "geometry_refused": plan["initial_source_coverage"]["status"] = "REFUSED"
    if problem == "forcing_short": plan["run_seconds"] = 3600
    for key, doc in (("plan", plan), ("slots", slots), ("fetch", fetched)):
        case.dump(paths[key], doc)
    with pytest.raises(ValueError):
        case.finalize(paths["plan"], **kwargs)
    assert not kwargs["manifest_path"].exists()
