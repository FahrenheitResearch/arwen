from copy import deepcopy
import datetime as dt
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import da_recent_observations as obs


def case():
    return {"schema": "da-rerun.recent-observations.v1", "model_start_utc": "2026-10-03T21:00:00Z",
            "analysis_times_utc": ["2026-10-03T22:00:00Z", "2026-10-03T23:00:00Z", "2026-10-04T00:00:00Z"],
            "observation_cutoff_utc": "2026-10-04T00:00:00Z", "bbox": [-93, 27, -85, 35],
            "radar_sites": ["KAAA", "KBBB"]}


def obj(bucket, key, size=3):
    return {"bucket": bucket, "key": key, "bytes": size, "etag": '"e"', "last_modified": "2026-10-05T00:00:00Z"}


def volume(**change):
    fields = dict(complete=True, sweeps_incomplete=0, sweeps_in_volume=4,
                  end_time="2026-10-03T21:59:59")
    fields.update(change)
    return SimpleNamespace(**fields)


def account():
    return {"params": {"engine": "region-global"},
            "totals": {"engine": "region-global", "sweeps_dealiased": 1,
                       "gates_offered": 3, "accounting_balances": True},
            "sweeps": [{"native": {"library": "/native/region.so"}}]}


def contribution():
    return SimpleNamespace(dealias=account(), z_count=np.array([3]), vr_count=np.array([3]),
                           cc_qc={"params": {}}, counts=SimpleNamespace(to_payload=lambda: {"gates_considered": 6}))


def surface(stamp="2026-10-03T22:00:00", station="AAA", observed="2026-10-03T21:59:00"):
    return {"schema": "gpuwm-obs.asos-surface.v2", "status": "READY", "station_table_sha256": "frozen",
            "provenance": {"source": "iem-metar", "uri": "/data/hour.csv", "sha256": "csvhash", "is_stub": False},
            "valid_times": [stamp], "match_seconds": 900, "min_report_rate": .8,
            "stations": [{"station_id": station, "lat": 31, "lon": -89}],
            "reports": [{"station_id": station, "valid_time": stamp, "observation_time": observed,
                         "values": {"temperature_2m": 290.0, "dewpoint_2m": 287.0}}],
            "screen": {"range_drops": 0, "dewpoint_above_temperature_drops": 0,
                       "stations_dropped_by_screen": [], "stations_dropped_by_completeness": []}}


def test_positive_hourly_schedule_and_cutoff():
    origin, analyses = obs.validate_case(case())
    assert (analyses[-1]-origin).total_seconds() == 10800
    bad = case(); bad["analysis_times_utc"][0] = bad["model_start_utc"]
    with pytest.raises(ValueError, match="positive hourly"):
        obs.validate_case(bad)
    bad = case(); bad["observation_cutoff_utc"] = "2026-10-03T23:00:00Z"
    with pytest.raises(ValueError, match="cutoff"):
        obs.validate_case(bad)


@pytest.mark.parametrize("stamp", ["2026-10-03T21:00:00", "2026-10-03T21:00:00-05:00"])
def test_case_rejects_implicit_or_non_utc_clock(stamp):
    bad = case(); bad["model_start_utc"] = stamp
    with pytest.raises(ValueError, match="explicit UTC"):
        obs.validate_case(bad)


def test_inventory_selects_past_candidates_across_midnight_and_brackets():
    def listing(bucket, prefix):
        if "KAAA" in prefix:
            return [obj(bucket, prefix+"KAAA20261003_215500_V06"),
                    obj(bucket, prefix+"KAAA20261003_220100_V06"),
                    obj(bucket, prefix+"KAAA20261003_235900_V06")]
        if "KBBB" in prefix:
            return []
        return [obj(bucket, prefix+"MRMS_Test_20261004-000000.grib2.gz"),
                obj(bucket, prefix+"MRMS_Test_20261004-010000.grib2.gz"),
                obj(bucket, prefix+"MRMS_Test_20261004-040000.grib2.gz")]
    inv = obs.inventory(case(), listing=listing)
    assert inv["slots"][0]["radar_candidates"]["KAAA"] == ["2026/10/03/KAAA/KAAA20261003_215500_V06", "2026/10/04/KAAA/KAAA20261003_215500_V06"]
    assert all("220100" not in key for key in inv["slots"][0]["radar_candidates"]["KAAA"])
    assert inv["slots"][-1]["radar_candidates"]["KAAA"]
    assert inv["raw_payloads_fetched"] is False
    assert len(inv["objects"]) == len({(row["bucket"], row["key"]) for row in inv["objects"]})


@pytest.mark.parametrize("change,message", [({"complete": False}, "complete"),
    ({"sweeps_incomplete": 1}, "complete"), ({"sweeps_in_volume": 0}, "complete"),
    ({"end_time": None}, "radial end"), ({"end_time": "2026-10-03T22:00:01Z"}, "after"),
    ({"end_time": "2026-10-03T21:40:00Z"}, "stale")])
def test_causal_volume_rejects_incomplete_future_stale_or_unknown(change, message):
    with pytest.raises(ValueError, match=message):
        obs.causal_volume(volume(**change), obs.utc("2026-10-03T22:00:00Z"), 900)


def test_causal_native_naive_utc_clock_is_accepted():
    assert obs.causal_volume(volume(), obs.utc("2026-10-03T22:00:00Z"), 900) == obs.utc("2026-10-03T21:59:59Z")


def test_object_publication_upper_bound_must_be_causal():
    row = obj("unidata-nexrad-level2", "object")
    analysis = obs.utc("2026-10-03T22:00:00Z")
    with pytest.raises(ValueError, match="after analysis"):
        obs.causal_object(row, analysis)
    row["last_modified"] = "2026-10-03T21:58:00Z"
    obs.causal_object(row, analysis)
    row.pop("last_modified")
    with pytest.raises(ValueError, match="upper-bound"):
        obs.causal_object(row, analysis)


def test_archive_replay_judges_a_restamped_object_by_its_scan_time():
    # 2024 archive objects re-stamped 2025-07-15 by bucket migration; every
    # pre-migration replay case refused.  The real key and stamp:
    key = "2024/05/21/KDMX/KDMX20240521_175821_V06"
    row = {**obj("unidata-nexrad-level2", key), "last_modified": "2025-07-15T22:59:04.000Z"}
    assert obs.archive_restamp(row) is not None
    assert obs.causal_object(row, obs.utc("2024-05-21T18:00:00Z")) == obs.CAUSAL_BY_SCAN_TIME
    # The scan time is still a cutoff: a scan started after the window end
    # is refused, re-stamped or not.
    with pytest.raises(ValueError, match="scan time is after"):
        obs.causal_object(row, obs.utc("2024-05-21T17:55:00Z"))


def test_archive_replay_keeps_the_refusal_for_genuinely_late_objects():
    analysis = obs.utc("2025-08-01T18:00:00Z")
    # Scanned after the migration and uploaded late: LastModified is real.
    late = {**obj("unidata-nexrad-level2", "2025/08/01/KDMX/KDMX20250801_175821_V06"),
            "last_modified": "2025-08-01T18:30:00Z"}
    assert obs.archive_restamp(late) is None
    with pytest.raises(ValueError, match="LastModified is after"):
        obs.causal_object(late, analysis)
    # A bucket without a re-stamp row keeps the LastModified rule for a
    # pre-migration scan too.
    other = {**obj("noaa-nexrad-level2", "2024/05/21/KDMX/KDMX20240521_175821_V06"),
             "last_modified": "2025-07-15T22:59:04Z"}
    assert obs.archive_restamp(other) is None
    with pytest.raises(ValueError, match="LastModified is after"):
        obs.causal_object(other, obs.utc("2024-05-21T18:00:00Z"))
    # LastModified before the re-upload date is the ordinary bound.
    early = {**obj("unidata-nexrad-level2", "2024/05/21/KDMX/KDMX20240521_175821_V06"),
             "last_modified": "2024-05-21T18:30:00Z"}
    assert obs.archive_restamp(early) is None
    with pytest.raises(ValueError, match="LastModified is after"):
        obs.causal_object(early, obs.utc("2024-05-21T18:00:00Z"))
    assert obs.causal_object({**early, "last_modified": "2024-05-21T17:59:00Z"},
                             obs.utc("2024-05-21T18:00:00Z")) == obs.CAUSAL_BY_LAST_MODIFIED


def test_archive_restamps_are_table_rows():
    for row in obs.ARCHIVE_RESTAMPS:
        assert row["bucket"] in obs.BUCKETS
        obs.utc(row["restamped_from"])
        obs.utc(row["scanned_before"])


def test_radar_tten_windows_offers_a_replayed_2024_case():
    from tools import radar_tten_windows as rtw
    end = obs.utc("2024-05-21T18:00:00Z")
    stamp = "2025-07-15T22:59:04.000Z"
    listing = [{**obj("unidata-nexrad-level2", f"2024/05/21/KDMX/KDMX20240521_{t}_V06"),
                "last_modified": stamp} for t in ("174255", "175038", "175821", "180604")]
    kept, refused = rtw.candidates_for(listing, end, per_site=3)
    # Inside the age window and not after the end, newest first.
    assert [row["key"] for row in kept] == [
        "2024/05/21/KDMX/KDMX20240521_175821_V06", "2024/05/21/KDMX/KDMX20240521_175038_V06"]
    assert {row["causal_basis"] for row in kept} == {obs.CAUSAL_BY_SCAN_TIME}
    assert refused == []


def test_smoke_inventory_uses_first_slot_and_only_explicit_models():
    calls = []
    def listing(bucket, prefix):
        calls.append((bucket, prefix))
        if bucket == "noaa-hrrr-bdp-pds":
            return [obj(bucket, "hrrr.20261003/conus/hrrr.t21z.wrfnatf00.grib2")]
        return []
    requests = [{"bucket": "noaa-hrrr-bdp-pds", "key": "hrrr.20261003/conus/hrrr.t21z.wrfnatf00.grib2"}]
    inv = obs.inventory(case(), listing=listing, first_slot_only=True, model_requests=requests)
    assert inv["scope"] == "smoke-first-slot"
    assert len(inv["slots"]) == 1 and inv["slots"][0]["analysis_time"] == "2026-10-03T22:00:00Z"
    assert not any(bucket == "noaa-mrms-pds" for bucket, _prefix in calls)
    assert any(prefix == "hrrr.20261003/conus/hrrr.t21z." for _bucket, prefix in calls)
    assert inv["model_requests"] == requests


def test_native_case_model_request_wrapper_is_accepted_without_rewriting_rows():
    rows = [{"bucket": "noaa-rap-pds", "key": "rap.20261003/rap.t21z.awp130bgrbf00.grib2"}]
    assert obs.model_request_rows({"schema": "da-rerun.public-object-requests.v1", "objects": rows}) is rows
    with pytest.raises(ValueError, match="wrapper schema"):
        obs.model_request_rows({"schema": "wrong", "objects": rows})


def test_native_case_surface_network_list_is_normalized_without_mutation():
    value = ["LA_ASOS", "MS_ASOS", "AL_ASOS"]
    assert obs.surface_networks_argument(value) == "LA_ASOS,MS_ASOS,AL_ASOS"
    assert value == ["LA_ASOS", "MS_ASOS", "AL_ASOS"]
    assert obs.surface_networks_argument("LA_ASOS, MS_ASOS") == "LA_ASOS,MS_ASOS"
    with pytest.raises(ValueError, match="invalid network"):
        obs.surface_networks_argument(["LA_ASOS", "LA_ASOS"])


def test_truth_extent_is_separately_frozen_and_never_model_clipped():
    doc = case()
    with pytest.raises(ValueError, match="independently widened"):
        obs.truth_bbox(doc)
    doc.update(truth_bbox=[-94, 26, -84, 36], truth_padding_m=75000)
    assert obs.truth_bbox(doc) == [-94, 26, -84, 36]
    doc["truth_bbox"] = doc["bbox"][:]
    with pytest.raises(ValueError, match="complete model"):
        obs.truth_bbox(doc)
    doc.update(truth_bbox=[-94, 26, -84, 36], truth_padding_m=50000)
    with pytest.raises(ValueError, match="75000"):
        obs.truth_bbox(doc)


def test_nonempty_dealias_params_alone_cannot_claim_solver_execution():
    c = contribution(); c.dealias = {"params": {"engine": "region-global"}, "totals": {}}
    with pytest.raises(ValueError, match="native solver"):
        obs.validate_contribution(c)
    c = contribution(); c.dealias["sweeps"] = [{"native": {"skipped": "no finite gate"}}]
    with pytest.raises(ValueError, match="native solver"):
        obs.validate_contribution(c)


def test_radar_support_and_cc_qc_are_required():
    obs.validate_contribution(contribution())
    for field in ("z_count", "vr_count"):
        c = contribution(); setattr(c, field, np.array([0]))
        with pytest.raises(ValueError, match="usable Z or Vr"):
            obs.validate_contribution(c)
    c = contribution(); c.cc_qc = {}
    with pytest.raises(ValueError, match="dual-pol"):
        obs.validate_contribution(c)


def test_surface_future_stale_wrong_slot_and_stub_refused():
    analysis = obs.utc("2026-10-03T22:00:00Z")
    obs.validate_surface(surface(), analysis)
    for bad in [surface(observed="2026-10-03T22:00:01"), surface(observed="2026-10-03T21:40:00"),
                surface(stamp="2026-10-03T23:00:00")]:
        with pytest.raises(ValueError): obs.validate_surface(bad, analysis)
    bad = surface(); bad["provenance"]["is_stub"] = True
    with pytest.raises(ValueError): obs.validate_surface(bad, analysis)


def test_surface_union_preserves_native_fields_station_union_and_no_reuse():
    a = surface(); b = surface("2026-10-03T23:00:00", "BBB", "2026-10-03T22:59:00")
    combined = obs.combine_surfaces([a, b])
    assert combined["reports"] == a["reports"] + b["reports"]
    assert [s["station_id"] for s in combined["stations"]] == ["AAA", "BBB"]
    assert combined["min_report_rate"] == 0
    assert combined["provenance"]["source"] == "iem-metar"
    assert a["min_report_rate"] == .8
    with pytest.raises(ValueError, match="reused"):
        obs.combine_surfaces([a, a])
    b["station_table_sha256"] = "changed"
    with pytest.raises(ValueError, match="station identity"):
        obs.combine_surfaces([a, b])


@pytest.mark.parametrize("key", ["../escape", "a/../../escape", "/absolute", "a\\bad", "a/./same", "a//same", "a:stream"])
def test_raw_destination_traversal_and_noncanonical_identity_refused(tmp_path, key):
    with pytest.raises(ValueError):
        obs.raw_path(tmp_path, obj("noaa-mrms-pds", key))


def test_fetch_receipt_binds_inventory_and_actual_bytes(tmp_path):
    raw = tmp_path/"raw.bin"; raw.write_bytes(b"abc")
    row = obj("noaa-mrms-pds", "a/data")
    inv = {"objects": [row]}
    receipt = {"schema": "da-rerun.recent-fetch.v1", "status": "complete",
               "inventory_sha256": obs.document_sha(inv), "objects": [{**row, "path": str(raw), "sha256": obs.sha(raw)}]}
    obs.validate_fetched(inv, receipt)
    tamper = deepcopy(receipt); tamper["objects"][0]["key"] = "a/other"
    with pytest.raises(ValueError, match="roster"):
        obs.validate_fetched(inv, tamper)
    tamper = deepcopy(receipt); tamper["inventory_sha256"] = "tampered"
    with pytest.raises(ValueError, match="hash-bound"):
        obs.validate_fetched(inv, tamper)
    raw.write_bytes(b"abd")
    with pytest.raises(ValueError, match="source identity"):
        obs.validate_fetched(inv, receipt)


def test_one_fetch_controller_checks_etag_size_and_retains_input_inventory(tmp_path, monkeypatch):
    class Reply(io.BytesIO):
        headers = {"ETag": '"e"'}
    calls = []
    def get(url, timeout):
        calls.append(url); return Reply(b"abc")
    monkeypatch.setattr(obs.urllib.request, "urlopen", get)
    inv = {"objects": [obj("noaa-mrms-pds", "a/data")]}
    original = deepcopy(inv)
    receipt_path = tmp_path/"fetch.json"
    obs.fetch(inv, tmp_path/"raw", receipt_path, 1)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["fetch_processes"] == 1 and len(calls) == 1
    assert inv == original
    obs.validate_fetched(inv, receipt)
    assert not (tmp_path/"raw/.recent-observation-fetch.lock").exists()
    with pytest.raises(ValueError, match="existing raw"):
        obs.fetch(inv, tmp_path/"raw", receipt_path, 1)


def cached_ancestor(tmp_path):
    row = obj("noaa-mrms-pds", "a/data")
    inv = {"objects": [row], "case_authority": "new explicit bbox"}
    root = tmp_path/"raw"
    path = obs.raw_path(root, row); path.parent.mkdir(parents=True); path.write_bytes(b"abc")
    ancestor = {"schema": "da-rerun.recent-fetch.v1", "status": "complete", "inventory_sha256": "0"*64,
                "objects": [{**row, "path": str(path), "sha256": obs.sha(path), "fetched_at_utc": "2026-10-05T01:07:17Z"}]}
    original = tmp_path/"ancestor.json"; obs.save(original, ancestor)
    return inv, root, path, ancestor, obs.sha(original)


def test_cached_reuse_hashes_actual_bytes_makes_zero_http_calls_and_preserves_fetch_time(tmp_path, monkeypatch):
    inv, root, path, ancestor, fingerprint = cached_ancestor(tmp_path)
    monkeypatch.setattr(obs.urllib.request, "urlopen", lambda *args, **kw: pytest.fail("cached reuse must make zero HTTP calls"))
    before = deepcopy(ancestor)
    out = tmp_path/"new-receipt.json"
    obs.fetch(inv, root, out, 1, reuse_receipt=ancestor, reuse_receipt_sha256=fingerprint)
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "complete" and receipt["reused_objects"] == 1 and receipt["downloaded_objects"] == 0
    assert receipt["ancestor_receipt_sha256"] == fingerprint
    assert receipt["objects"][0]["fetched_at_utc"] == "2026-10-05T01:07:17Z"
    assert receipt["objects"][0]["sha256"] == obs.sha(path)
    assert receipt["inventory_sha256"] == obs.document_sha(inv)
    assert ancestor == before
    obs.validate_fetched(inv, receipt)


@pytest.mark.parametrize("field,value", [("bytes", 4), ("etag", '"other"'), ("last_modified", "2026-10-05T00:01:00Z")])
def test_reuse_refuses_metadata_mismatch_without_adapting_it(tmp_path, field, value):
    inv, root, path, ancestor, fingerprint = cached_ancestor(tmp_path)
    inv["objects"][0][field] = value
    with pytest.raises(ValueError, match="metadata or actual cached"):
        obs.fetch(inv, root, tmp_path/"new.json", 1, reuse_receipt=ancestor, reuse_receipt_sha256=fingerprint)
    assert path.read_bytes() == b"abc"


def test_reuse_refuses_tampered_raw_or_partial_and_wrong_path_roster(tmp_path):
    inv, root, path, ancestor, fingerprint = cached_ancestor(tmp_path)
    partial = deepcopy(ancestor); partial["status"] = "in-progress"
    with pytest.raises(ValueError, match="COMPLETE ancestor"):
        obs.fetch(inv, root, tmp_path/"new.json", 1, reuse_receipt=partial, reuse_receipt_sha256=fingerprint)
    wrong = deepcopy(ancestor); wrong["objects"][0]["key"] = "a/untracked"
    with pytest.raises(ValueError, match="ancestor source path"):
        obs.fetch(inv, root, tmp_path/"new.json", 1, reuse_receipt=wrong, reuse_receipt_sha256=fingerprint)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="metadata or actual cached"):
        obs.fetch(inv, root, tmp_path/"new.json", 1, reuse_receipt=ancestor, reuse_receipt_sha256=fingerprint)
    assert path.read_bytes() == b"abd"


def test_new_fetch_receipt_cannot_overwrite_ancestor(tmp_path):
    inv, root, path, ancestor, fingerprint = cached_ancestor(tmp_path)
    inventory_path = tmp_path/"inventory.json"; obs.save(inventory_path, inv)
    ancestor_path = tmp_path/"ancestor.json"
    with pytest.raises(ValueError, match="preserve its COMPLETE ancestor"):
        obs.main(["fetch", "--inventory", str(inventory_path), "--root", str(root),
                  "--receipt", str(ancestor_path), "--reuse-receipt", str(ancestor_path)])
    assert obs.sha(ancestor_path) == fingerprint


@pytest.mark.parametrize("workers", [1, 4])
def test_full_prepare_smoke_dispatch_uses_public_doors_and_hashes(tmp_path, monkeypatch, workers):
    """Exercise orchestration, not scientific numerics or network availability."""
    import gpuwm.obs.cc_qc
    import gpuwm.obs.dealias
    import gpuwm.obs.frontdoor
    import gpuwm.obs.nexrad
    import gpuwm.obs.radar_grid
    import gpuwm.obs.superob
    import gpuwm.obs.sweeps
    import gpuwm.obs.target_grid
    from tools.da_recent_case import observation_config
    authored_plan = {**case(), "smoke_only": True, "requested_da_hours": 3}
    doc = observation_config(authored_plan, radar_sites=["KDGX", "KHDC"])
    analysis = doc["analysis_times_utc"][0]
    original_networks = doc["surface_networks"][:]
    rows = [obj("unidata-nexrad-level2", f"2026/10/03/{site}/{site}20261003_215500_V06") for site in doc["radar_sites"]]
    for row in rows: row["last_modified"] = "2026-10-03T21:59:00Z"
    inv = {"case": doc, "scope": "smoke-first-slot", "objects": rows,
           "slots": [{"analysis_time": analysis,
                      "radar_candidates": {site: [row["key"]] for site, row in zip(doc["radar_sites"], rows)}}],
           "publication_policy": "archive replay"}
    fetched = {"schema": "da-rerun.recent-fetch.v1", "status": "complete",
               "inventory_sha256": obs.document_sha(inv), "objects": []}
    for index, row in enumerate(rows):
        path = tmp_path/f"raw-{index}"; path.write_bytes(b"abc")
        fetched["objects"].append({**row, "path": str(path), "sha256": obs.sha(path)})
    binary = tmp_path/"native.exe"; binary.write_bytes(b"native fixture")
    grid = tmp_path/"grid.nc"; grid.write_bytes(b"prepared georeference fixture")
    monkeypatch.setattr(gpuwm.obs.nexrad, "find_nexrad_bin", lambda: binary)
    monkeypatch.setattr(gpuwm.obs.dealias, "engine_unavailable_reason", lambda engine: None)
    monkeypatch.setattr(gpuwm.obs.frontdoor.FrontDoor, "find", lambda self: binary)
    monkeypatch.setattr(gpuwm.obs.target_grid.TargetGrid, "from_wrfout", lambda path: SimpleNamespace())
    decoded = []
    def decode(executable, **kw):
        decoded.append(kw); kw["out"].write_bytes(b"native pack fixture")
    monkeypatch.setattr(gpuwm.obs.nexrad, "run_decode", decode)
    monkeypatch.setattr(gpuwm.obs.nexrad, "run_verify", lambda *args, **kw: {"status": "PASS"})
    monkeypatch.setattr(gpuwm.obs.sweeps, "read_sweep_pack", lambda path: volume())
    monkeypatch.setattr(gpuwm.obs.superob, "superob_volume", lambda *args, **kw: contribution())
    monkeypatch.setattr(gpuwm.obs.superob, "merge_contributions", lambda *args, **kw: SimpleNamespace())
    def write(path, *args, **kw):
        assert kw["valid_time"] == analysis
        path.write_bytes(b"native radar output fixture")
        return {"status": "PASS", "sha256": obs.sha(path)}
    monkeypatch.setattr(gpuwm.obs.radar_grid, "write_radar_grid", write)
    calls = []
    def native(command, log):
        calls.append(command)
        if "--out" in command:
            path = command[command.index("--out")+1]
            if command[1] == "decode": obs.save(path, surface())
            else: open(path, "w").write("native fixture")
        return {"status": "PASS"}, .001
    monkeypatch.setattr(obs, "run_json", native)
    result = obs.prepare(inv, fetched, grid, tmp_path/"prepared-observations", workers=workers)
    assert result["status"] == "prepared-smoke-first-slot"
    # Concurrent sites still merge in roster order.
    assert [row["site"] for row in result["slots"][0]["radar_sources"]] == doc["radar_sites"]
    assert result["forecast_fork_utc"] == analysis
    assert result["requested_full_case_fork_utc"] == doc["analysis_times_utc"][-1]
    assert len(result["slots"]) == 1 and result["scientific_gate"] == "pending"
    assert result["inventory_sha256"] == obs.document_sha(inv)
    assert result["slots"][0]["grid_wrfout_sha256"] == obs.sha(grid)
    assert all(call["censor_flags"] and call["moments"] == ("REF", "VEL", "RHO") for call in decoded)
    fetch = next(command for command in calls if command[1] == "fetch")
    assert fetch[fetch.index("--end")+1] == "2026-10-03T22:00:00"
    station_command = next(command for command in calls if command[1] == "stations")
    assert station_command[station_command.index("--networks")+1] == ",".join(original_networks)
    assert all(isinstance(value, str) for value in station_command)
    assert doc["surface_networks"] == original_networks
    assert sum(command[1] == "verify" for command in calls) == 2
