"""A recent case at another spacing and forcing, its fine companion grid and its run plan."""
import io
import json
import threading
import time

import pytest

from tools import da_recent_case as case
from tools import da_recent_observations as obs
from tools import da_recent_run as runner
from tools.da_cycle_prepared import build_parser


def test_coarse_case_scales_clock_and_localization_and_names_gfs_forcing(tmp_path):
    doc = case.author(tmp_path / "case", start="2026-10-01T18:00:00Z", latitude=38.5, longitude=-97.5,
                      nx=68, ny=68, da_hours=3, dx_m=9000, forcing="gfs",
                      surface_networks=["KS_ASOS", "OK_ASOS"])
    assert doc["source"] == "gdas" and doc["forcing"] == "gfs"
    assert doc["dx_m"] == 9000 and doc["time_step_seconds"] == 30
    assert doc["analysis_times_utc"] == ["2026-10-01T19:00:00Z", "2026-10-01T20:00:00Z", "2026-10-01T21:00:00Z"]
    assert doc["run_seconds"] == 9 * 3600 and doc["free_forecast_seconds"] == 21600
    assert doc["localization"]["horizontal_loc_m"] == 36000 and doc["localization"]["sfc_horizontal_loc_m"] == 36000
    assert doc["localization"]["vertical_loc_m"] == 6000 and doc["localization"]["sfc_vertical_loc_m"] == 3000
    assert doc["perturbation_length_scale_km"] == 40
    forcing = [row for row in doc["model_objects"] if row["role"] == "lateral_forcing"]
    assert forcing[0]["key"] == "gfs.20261001/18/atmos/gfs.t18z.pgrb2.0p25.f000"
    assert [row["lead_hours"] for row in forcing] == list(range(10))
    assert {row["bucket"] for row in forcing} == {"noaa-gfs-bdp-pds"}
    argv = doc["prepare_argv"]
    assert argv[argv.index("--source") + 1] == "gdas"
    assert sum(value.startswith("gdas_pgrb2_in_band_surface=") for value in argv) == 10
    assert doc["initial_source_coverage"]["status"] == "PASS"
    authority = (tmp_path / "case/authority/experiment.toml").read_text()
    assert "time_step = 30" in authority and "dx = 9000.0" in authority
    assert doc["aerosol"] == {"mp28_aerosol_source": "climatology", "use_rap_aero_icbc": False}
    assert 'mp28_aerosol_source = "climatology"' in authority and "use_rap_aero_icbc = false" in authority
    assert doc["native_memory_plan"]["packing_candidate_per_32gib_card"] in (1, 2, 4)


@pytest.mark.parametrize("change, message", [
    ({"start": "2026-10-01T19:00:00Z"}, "cycle hours"),
    ({"da_hours": 6}, "through lead 9"),
    ({"dx_m": 9000, "time_step_s": 45}, "whole time step"),
    ({"nx": 60}, "span at least"),
])
def test_coarse_case_refuses_unforced_or_unclocked_shapes(tmp_path, change, message):
    kwargs = dict(start="2026-10-01T18:00:00Z", latitude=38.5, longitude=-97.5, nx=68, ny=68,
                  da_hours=3, dx_m=9000, forcing="gfs")
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        case.author(tmp_path / "case", **kwargs)


def test_companion_grid_refines_parent_interior_on_its_own_lattice(tmp_path):
    case.author(tmp_path / "coarse", start="2026-10-01T18:00:00Z", latitude=38.5, longitude=-97.5,
                nx=81, ny=79, da_hours=3, dx_m=9000, forcing="gfs")
    fine = case.companion(tmp_path / "coarse/case-plan.json", tmp_path / "fine", shared_root="/box/fine",
                          geog_root="/box/geog")
    assert fine["schema"] == "gpuwm-da.recent-companion-grid.v1"
    assert (fine["nx"], fine["ny"], fine["dx_m"], fine["time_step_seconds"]) == (3 * 69, 3 * 67, 3000.0, 20)
    assert fine["members"] == 1
    assert fine["placement"] == {"parent_grid_ratio": 3, "i_parent_start": 7, "j_parent_start": 7,
                                 "parent_cells_x": 69, "parent_cells_y": 67,
                                 "alignment": "same projection and center; fine corners fall on coarse cell corners"}
    assert not (tmp_path / "fine/observation-case.json").exists()
    assert not any(row["role"] == "independent_reference_analysis" for row in fine["model_objects"])
    method = fine["start_from_analysis"]
    assert method["issued_utc"] == "2026-10-01T21:00:00Z" and method["issued_seconds_after_model_start"] == 10800
    assert method["argv_template"][3] == "downscale"
    assert "/box/fine/prepared/wrf-native-input/wrfinput_d01" in method["argv_template"]
    argv = method["argv_template"]
    assert argv[argv.index("--child-config") + 1] == fine["child_run_config"]
    assert argv[argv.index("--child-config-sha256") + 1] == case.sha(fine["child_run_config"])
    from gpuwm.offline_child import resolve_child_run_config
    child = resolve_child_run_config(fine["child_run_config"])
    assert (child.nx, child.ny, child.dx, child.dt) == (3 * 69, 3 * 67, 3000.0, 20.0)
    assert child.specified and not child.nested and child.run_seconds == 21600
    # The fine footprint is the coarse one less six coarse cells a side.
    coarse = case.read_json(tmp_path / "coarse/case-plan.json")
    assert coarse["bbox"][0] < fine["bbox"][0] < fine["bbox"][2] < coarse["bbox"][2]
    assert coarse["bbox"][1] < fine["bbox"][1] < fine["bbox"][3] < coarse["bbox"][3]


def test_run_plan_reads_the_case_localization_and_scales(tmp_path):
    for name in ("surface", "grid", "obs"):
        (tmp_path / name).write_bytes(b"fixture")
    document = {"engine_sha": runner.source_revision(), "source": "gdas",
        "prepared_root": str(tmp_path / "prepared"), "authority_dir": str(tmp_path / "authority"),
        "physics_profile": "fixture-profile", "proof_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64, "prepared_content_sha256": "c" * 64,
        "run_seconds": 32400, "history_interval_seconds": 120,
        "model_start_utc": "2026-10-01T18:00:00Z", "forecast_fork_utc": "2026-10-01T21:00:00Z",
        "slots": [{"obs": str(tmp_path / "obs"), "grid_wrfout": str(tmp_path / "grid"),
                   "leg_seconds": 3600, "analysis_time": stamp}
                  for stamp in ("2026-10-01T19:00:00Z", "2026-10-01T20:00:00Z", "2026-10-01T21:00:00Z")],
        "surface_obs": str(tmp_path / "surface"), "out": str(tmp_path / "runs"),
        "members": 32, "smoke_only": False, "free_forecast_seconds": 21600,
        "perturbation_length_scale_km": 40, "localization": case.localization_for(9000)}
    rows = runner.plan(document, seed=7)
    da, control = [build_parser().parse_args(row["argv"][3:]) for row in rows]
    assert da.horizontal_loc_m == control.horizontal_loc_m == 36000
    assert da.vertical_loc_m == 6000 and da.sfc_horizontal_loc_m == 36000 and da.sfc_vertical_loc_m == 3000
    assert da.length_scale_km == control.length_scale_km == 40
    assert da.leg_durations_seconds == control.leg_durations_seconds == [3600, 3600, 3600, 21600]
    one = build_parser().parse_args([*rows[0]["argv"][3:], "--forecast-members-per-card", "1"])
    assert one.forecast_members_per_card == "1"


def test_concurrent_fetch_streams_keep_one_process_and_inventory_order(tmp_path, monkeypatch):
    class Reply(io.BytesIO):
        headers = {"ETag": "\"e\""}
    active, peak, guard = [0], [0], threading.Lock()

    def get(url, timeout):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05 if url.endswith("0") else 0.01)
        with guard:
            active[0] -= 1
        return Reply(b"abc")
    monkeypatch.setattr(obs.urllib.request, "urlopen", get)
    rows = [{"bucket": "noaa-mrms-pds", "key": f"a/data{index}", "bytes": 3, "etag": "\"e\"",
             "last_modified": "2026-10-01T00:00:00Z"} for index in range(8)]
    inv = {"objects": rows}
    receipt_path = tmp_path / "fetch.json"
    obs.fetch(inv, tmp_path / "raw", receipt_path, 1, streams=4)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["fetch_processes"] == 1 and receipt["fetch_streams"] == 4 and receipt["status"] == "complete"
    assert [row["key"] for row in receipt["objects"]] == [row["key"] for row in rows]
    assert peak[0] > 1
    obs.validate_fetched(inv, receipt)


@pytest.mark.parametrize("dx, step", [(3000, 20), (4500, 15), (6000, 20), (9000, 30), (12000, 30), (13500, 30)])
def test_time_step_divides_history_and_radiation_and_keeps_the_deck_clock(dx, step):
    # 3 km keeps HRRR's 20 s; coarser grids take the largest step at or under
    # 10/3 s per km that divides the 120 s history and the 900 s radiation clock.
    assert case.time_step_for(dx) == step
    assert 120 % step == 0 and 900 % step == 0
