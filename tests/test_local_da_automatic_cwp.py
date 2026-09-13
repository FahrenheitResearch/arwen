"""Deterministic listing controls and real pack/join/superob regression calls.

Native process and final netCDF publication are test doubles. The fixture
container, production readers, CWP cross-check, QC and mosaic are real code.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.obs import goes_window as gw
from gpuwm.obs.goes_cwp import CwpErrorModel, GoesCwpError, grid_cwp, read_cwp_pack
from gpuwm.obs.goes_cwp_policy import utc, stamp
from goes_pack_fixtures import (PROJECTION, CWP_SCHEMA_V2, CLOUDTOP_SCHEMA_V2,
    _encode, source, sibling_block, write_cwp_pack, write_cloudtop_pack)

WHEN = utc("2026-09-12T18:10:00Z")
SOURCE = gw.Source("G19", "C", 6)
POLICY = gw.AcquisitionPolicy(sources=(SOURCE,))


class Grid:
    """Analytic target geometry; no geographic projection is substituted silently."""
    nz, ny, nx = 3, 1, 4
    lat = np.full((1, 4), 35.0)
    lon = np.array([[-98., -97., -96., -95.]])
    terrain_m = np.zeros((1, 4))
    z_w = np.broadcast_to(np.array([0., 3000., 6000., 12000.])[:, None, None], (4, 1, 4))

    def identity_sha256(self):
        return "analytic-target-4-columns"

    def mass_index(self, lat, lon):
        return (np.asarray(lon) + 98 + 180) % 360 - 180, np.asarray(lat) - 35

    def inside(self, i, j):
        return (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)

    def level_index(self, i, j, height):
        level = np.searchsorted([0, 3000, 6000, 12000], height, side="right") - 1
        return np.where((level >= 0) & (level < 3), level, -1)


def listing(source_=SOURCE, *, start=WHEN - timedelta(minutes=5), end=WHEN - timedelta(minutes=2),
            products=("COD", "CPS", "ACTP", "ACHA"), published=None):
    published = end + timedelta(seconds=40) if published is None else published
    start, end = utc(start), utc(end)
    def token(moment):
        return moment.strftime("%Y%j%H%M%S") + str(moment.microsecond // 100000)
    rows = []
    for product in products:
        name = f"OR_ABI-L2-{product}{source_.sector}-M{source_.mode}_{source_.satellite}_s{token(start)}_e{token(end)}_c{token(published)}.nc"
        key = f"ABI-L2-{product}{source_.sector}/{start:%Y}/{start:%j}/{start:%H}/{name}"
        rows.append(dict(product=product, key=key, filename=name, url="https://example.invalid/"+key,
                         scan_start=stamp(start), scan_end=stamp(end), size_bytes=len(product.encode()), last_modified=stamp(published)))
    return dict(schema=gw.LIST_SCHEMA, status="READY", satellite=source_.satellite, sector=source_.sector,
        mode=source_.mode, products=list(gw.REQUIRED_PRODUCTS+gw.OPTIONAL_PRODUCTS),
        scans=[dict(scan_start=stamp(start), complete=len(rows)==4, missing_products=sorted(set(gw.REQUIRED_PRODUCTS+gw.OPTIONAL_PRODUCTS)-set(products)), granules=rows)])


def select(record, **kwargs):
    return gw.select_scans(record, SOURCE, when=WHEN, publication_cutoff=WHEN, policy=POLICY, **kwargs)


class Native:
    identity = {"test_double": True}

    def __init__(self, records=None, *, phase=(1, 0, 4, np.nan), race=False, cache_hit=False,
                 bad_pair=False, fail_source=None, corrupt_pack=False):
        self.records = records or {SOURCE.id: listing()}
        self.phase = phase
        self.race, self.cache_hit, self.bad_pair = race, cache_hit, bad_pair
        self.fail_source, self.corrupt_pack = fail_source, corrupt_pack
        self.calls, self.written, self.current = [], {}, None

    def call(self, command, arguments, *, schema):
        args = dict(zip(arguments[::2], arguments[1::2])) if command != "fetch" else dict(zip(arguments[:-1:2], arguments[1:-1:2]))
        self.calls.append((command, list(arguments), schema))
        if command in ("list", "fetch"):
            src = gw.Source(args["--satellite"], args["--sector"], int(args["--mode"]))
            if src.id == self.fail_source:
                raise RuntimeError("source unreachable")
            record = self.records.get(src.id, {**listing(src), "scans": []})
            if command == "list":
                return copy.deepcopy(record)
            scan = next(s for s in record["scans"] if utc(s["scan_start"]) == utc(args["--start"]))
            self.current = (src, scan)
            files = []
            for row in scan["granules"]:
                if row["product"] not in args["--products"].split(","):
                    continue
                path = Path(args["--out"]) / row["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(row["product"].encode())
                self.written[path] = row
                files.append(dict(product=row["product"], key=row["key"] + (".revision" if self.race else ""),
                    path=str(path), bytes=path.stat().st_size, sha256=gw.sha256(path), cache_hit=self.cache_hit))
            return dict(schema=schema, satellite=src.satellite, sector=src.sector, mode=src.mode,
                        scans=[dict(scan_start=scan["scan_start"], complete=True, files=files)])
        src, scan = self.current
        path = Path(args["--out"])
        phase = np.array([self.phase], np.float32)
        common = dict(satellite=src.satellite, sector=src.sector, scan_start=scan["scan_start"],
            scan_end=scan["granules"][0]["scan_end"], lat=Grid.lat, lon=Grid.lon,
            x_scan_rad=[0, .001, .002, .003], y_scan_rad=[.08])
        if command == "cwp":
            sources = []
            for product in gw.REQUIRED_PRODUCTS:
                file = Path(args["--"+product.lower()])
                sources.append({**source(product, dqf_plane=product.lower()+"_dqf"),
                    "filename": file.name, "sha256": gw.sha256(file)})
            write_cwp_pack(path, cod=np.full((1,4),10.), cps=np.full((1,4),15.), phase=phase,
                sources=sources, schema=CWP_SCHEMA_V2,
                dqf_planes={p:np.zeros((1,4)) for p in gw.REQUIRED_PRODUCTS}, **common)
            if self.corrupt_pack:
                path.write_bytes(path.read_bytes()[:-1]+b"x")
        elif command == "cloud-top":
            sibling = sibling_block(args["--pairs-with"])
            if self.bad_pair:
                sibling["content_sha256"] = "0"*64
            write_cloudtop_pack(path, cloud_top_height_m=np.full((1,4),8000.),
                schema=CLOUDTOP_SCHEMA_V2, sibling=sibling, **common)
            raw = path.read_bytes(); meta_length = int.from_bytes(raw[12:16], "little")
            meta = json.loads(raw[64:64+meta_length]); rawfile = Path(args["--acha"])
            meta["sources"][0].update(filename=rawfile.name, sha256=gw.sha256(rawfile))
            path.write_bytes(_encode(meta, raw[64+meta_length:]))
        else:
            raise AssertionError(command)
        return {"schema": schema, "status": "READY", "pack": {"path": str(path)}}


def capture_writer(captured):
    def write(path, product, grid, *, valid_time):
        captured["product"] = product
        # The sink is intentionally not a fake .nc decoder. Only existence
        # and content hashing are tested at this substituted publication edge.
        Path(path).write_text(json.dumps({"test_sink": True, "valid_time": valid_time,
            "mask": product.cwp_mask.tolist(), "provenance": product.provenance}, allow_nan=False))
        return {"path": str(path), "sha256": gw.sha256(path)}
    return write


def acquire(tmp_path, native=None, *, policy=POLICY, when=WHEN, used=()):
    captured = {}
    result = gw.acquire_window(grid=Grid(), when=when, directory=tmp_path/"out", cache=tmp_path/"cache",
        native=native or Native(), policy=policy, used_scan_ids=used,
        now=lambda: when+timedelta(seconds=10), write_grid=capture_writer(captured))
    return result, captured


def test_default_policy_is_visible_and_roundtrips():
    policy = gw.AcquisitionPolicy()
    restored = gw.AcquisitionPolicy.from_payload(json.loads(json.dumps(policy.to_payload())))
    assert restored == policy
    assert policy.error_model["floor_liquid_g_m2"] == 50
    assert policy.max_age_seconds == 1800
    assert {s.satellite for s in policy.sources} == {"G16","G17","G18","G19"}
    assert not any(s.sector.startswith("M") for s in policy.sources)


@pytest.mark.parametrize("field,value", [("max_age_seconds",0),("max_age_seconds",True),("scan_lookback_seconds",0)])
def test_invalid_policy_refused(field, value):
    with pytest.raises(ValueError):
        replace(POLICY, **{field:value}).validate()


@pytest.mark.parametrize("products", [("COD","CPS"),("COD","ACTP"),("CPS","ACTP"),()])
def test_incomplete_trio_is_not_an_observation(products):
    scans, decisions = select(listing(products=products))
    assert not scans and decisions[0]["status"] == "incomplete"


def test_missing_cloudtop_does_not_disable_complete_cwp():
    scans, _ = select(listing(products=gw.REQUIRED_PRODUCTS))
    assert len(scans) == 1 and set(scans[0].granules)==set(gw.REQUIRED_PRODUCTS)


def test_late_cloudtop_is_not_joined_to_an_earlier_cutoff():
    record = listing()
    record["scans"][0]["granules"][-1]["last_modified"] = stamp(WHEN+timedelta(seconds=1))
    scans, _ = select(record)
    assert "ACHA" not in scans[0].granules


def test_unknown_publication_is_usable_with_actual_receipt_and_explicit_uncertainty(tmp_path):
    from gpuwm.local_da_fetch import assigned_document
    record = listing()
    record['scans'][0]['granules'][0]['last_modified'] = ''
    result, captured = acquire(tmp_path, Native({SOURCE.id: record}))
    assert result.path is not None
    scan = result.receipt['scans'][0]
    assert scan['publication_utc'] is None
    assert 'actual local receipt' in scan['publication_basis']
    assert result.receipt['sources'][0]['decisions'][0]['warning']
    document = {'schema':'gpuwm-obs.goes-grid.v1', 'valid_time':stamp(WHEN),
                'provenance':{'acquisition':result.receipt}}
    assert assigned_document(document, WHEN, [WHEN], 900.)


@pytest.mark.parametrize("change,status", [
    (lambda r:r.update(scan_end=stamp(WHEN+timedelta(seconds=1))), "future_scan"),
    (lambda r:r.update(last_modified=stamp(WHEN+timedelta(seconds=1))), "published_after_cutoff"),
])
def test_scan_time_and_publication_gates(change,status):
    record=listing()
    change(record["scans"][0]["granules"][0])
    scans, decisions=select(record)
    assert not scans and decisions[0]["status"]==status


def test_fractional_start_and_late_unconsumed_scan_are_kept():
    record=listing(start=WHEN-timedelta(minutes=8,microseconds=900000), end=WHEN-timedelta(minutes=6))
    scans,_=select(record)
    assert len(scans)==1 and scans[0].start.microsecond==100000
    assert not select(record,used_scan_ids=[scans[0].id])[0]


@pytest.mark.parametrize("age,kept", [(1800,True),(1801,False)])
def test_age_boundary_is_explicit(age,kept):
    record=listing(start=WHEN-timedelta(seconds=age+200),end=WHEN-timedelta(seconds=age),products=gw.REQUIRED_PRODUCTS)
    assert bool(select(record)[0]) == kept


def test_duplicate_products_and_scan_id_are_refused():
    for duplicate in ("product","scan"):
        record=listing()
        if duplicate=="product":
            record["scans"][0]["granules"].append(copy.deepcopy(record["scans"][0]["granules"][0]))
        else:
            record["scans"].append(copy.deepcopy(record["scans"][0]))
        with pytest.raises(gw.AcquisitionError):
            select(record)


def test_future_label_cannot_change_native_satellite_identity():
    record=listing();record["satellite"]="G18"
    with pytest.raises(gw.AcquisitionError):select(record)






def test_real_packs_join_grid_errors_and_publication(tmp_path):
    native=Native(); result,captured=acquire(tmp_path,native)
    assert result.path and result.path.is_file()
    product=captured["product"]
    np.testing.assert_array_equal(product.cwp_mask,[[1,1,1,0]])
    np.testing.assert_allclose(product.cwp_obs[0,:3],[100,0,91.7],rtol=1e-6)
    np.testing.assert_array_equal(product.obs_level[0,:3],[2,1,2])
    np.testing.assert_allclose(product.cwp_err[0,:3],[50,50,100])
    assert [c[0] for c in native.calls]==["list","fetch","cwp","fetch","cloud-top"]
    fetch=native.calls[1][1]
    assert fetch[fetch.index("--start")+1]==fetch[fetch.index("--end")+1]
    assert result.receipt["observed_columns"]==3
    assert len(result.receipt["consumed_scan_ids"])==1
    assert all(gw.sha256(a["path"])==a["sha256"] for a in result.assets)


def test_all_missing_pixels_yield_no_grid_not_clear_sky(tmp_path):
    result,_=acquire(tmp_path,Native(phase=(np.nan,)*4))
    assert result.path is None
    assert result.receipt["observed_columns"]==0
    assert result.receipt["consumed_scan_ids"]==[]
    assert result.manifest_path.is_file()




def test_corrupted_pack_is_recorded_and_not_published(tmp_path):
    result,_=acquire(tmp_path,Native(corrupt_pack=True))
    assert result.path is None
    assert "GoesPackError" in result.receipt["sources"][0]["attempts"][0]["reason"]


def test_wrong_cloudtop_sibling_is_not_joined(tmp_path):
    result,captured=acquire(tmp_path,Native(bad_pair=True))
    np.testing.assert_array_equal(captured["product"].obs_level[0,:3],[1,1,1])
    record=result.receipt["scans"][0]
    assert "Refusing to join" in record["cloud_top_failure"]
    assert not record["gridding"]["join"]["performed"]


def test_revision_race_does_not_silently_substitute_bytes(tmp_path):
    result,_=acquire(tmp_path,Native(race=True))
    assert result.path is None
    assert "selected revision" in result.receipt["sources"][0]["attempts"][0]["reason"]


def test_cached_arrival_is_not_invented(tmp_path):
    result,_=acquire(tmp_path,Native(cache_hit=True))
    assert all(r["first_receipt_utc"] is None for r in result.receipt["scans"][0]["source_files"])


def test_used_scan_is_not_fetched_again(tmp_path):
    native=Native();used=select(listing())[0][0].id
    result,_=acquire(tmp_path,native,used=[used])
    assert [c[0] for c in native.calls]==["list"]
    assert result.path is None


def test_failed_generations_do_not_collide_on_retry(tmp_path):
    failed,_=acquire(tmp_path,Native(race=True))
    success,_=acquire(tmp_path)
    assert failed.manifest_path.parent != success.manifest_path.parent
    assert failed.manifest_path.is_file() and success.path.is_file()


def test_overlapping_satellites_offer_one_observation_per_column(tmp_path):
    other=gw.Source("G18","C",6)
    native=Native({SOURCE.id:listing(),other.id:listing(other)})
    result,captured=acquire(tmp_path,native,policy=replace(POLICY,sources=(SOURCE,other)))
    assert captured["product"].cwp_mask.sum()==3
    assert sum(r["selected_columns"] for r in result.receipt["scans"])==3
    assert len(result.receipt["consumed_scan_ids"])==1


def test_one_unreachable_satellite_does_not_disable_the_other(tmp_path):
    other=gw.Source("G18","C",6)
    native=Native({SOURCE.id:listing()},fail_source=other.id)
    result,_=acquire(tmp_path,native,policy=replace(POLICY,sources=(SOURCE,other)))
    assert result.path is not None
    assert any(s["status"]=="unavailable" for s in result.receipt["sources"])








def test_optional_cloudtop_transport_failure_keeps_complete_optical_trio(tmp_path):
    class MissingTop(Native):
        def call(self, command, arguments, *, schema):
            if command == "fetch" and arguments[arguments.index("--products")+1] == "ACHA":
                raise OSError("optional ACHA download failed")
            return super().call(command, arguments, schema=schema)
    result, captured = acquire(tmp_path, MissingTop())
    assert result.path
    record = result.receipt["scans"][0]
    assert "optional ACHA" in record["cloud_top_failure"]
    assert record["gridding"]["join"]["performed"] is False
    assert {r["product"] for r in record["source_files"]} == set(gw.REQUIRED_PRODUCTS)
    np.testing.assert_array_equal(captured["product"].obs_level[0,:3], [1,1,1])


def test_optical_age_is_not_rejuvenated_by_cloudtop(tmp_path):
    record = listing()
    top = record["scans"][0]["granules"][-1]
    top["scan_end"] = stamp(WHEN)
    top["last_modified"] = stamp(WHEN)
    result, _ = acquire(tmp_path, Native({SOURCE.id: record}))
    assert result.receipt["max_observation_age_seconds"] == 120
    assert result.receipt["scans"][0]["scan_end"] == stamp(WHEN)


def test_all_transport_failures_are_not_reported_as_empty_listings(tmp_path):
    result, _ = acquire(tmp_path, Native(fail_source=SOURCE.id))
    assert result.path is None
    assert result.receipt["status"] == "unavailable"
    assert result.receipt["sources"][0]["status"] == "unavailable"








def test_explicit_cwp_valid_time_cannot_hide_future_scan_end():
    from gpuwm.local_da_fetch import assigned_document
    doc = {"schema":"gpuwm-obs.goes-grid.v1", "valid_time":stamp(WHEN),
        "provenance":{"pack":{"scan_start":stamp(WHEN-timedelta(minutes=2)),
                              "scan_end":stamp(WHEN+timedelta(seconds=1))}}}
    assert not assigned_document(doc, WHEN, [WHEN], 300)
    doc["provenance"]["pack"]["scan_end"] = stamp(WHEN)
    assert assigned_document(doc, WHEN, [WHEN], 300)


def test_automatic_assignment_checks_actual_receipt_cutoff(tmp_path):
    from gpuwm.local_da_fetch import assigned_document
    result, captured = acquire(tmp_path)
    doc = {"schema":"gpuwm-obs.goes-grid.v1", "valid_time":stamp(WHEN),
        "provenance":captured["product"].provenance}
    assert assigned_document(doc, WHEN, [WHEN], 300)
    assert not assigned_document(doc, WHEN+timedelta(minutes=5), [WHEN,WHEN+timedelta(minutes=5)], 300)
    doc = copy.deepcopy(doc)
    doc["provenance"]["acquisition"]["scans"][0]["source_files"][0]["available_on_disk_utc"] = stamp(WHEN+timedelta(hours=1))
    assert not assigned_document(doc, WHEN, [WHEN], 300)


def _dispatch_setup(tmp_path, monkeypatch, native):
    """Patch I/O boundaries only; the actual dispatcher and freeze logic run."""
    from gpuwm import local_da_fetch as fetch
    from gpuwm.obs import radar_grid, goes_grid
    from gpuwm.da import obs_point, obs_surface
    clock = [WHEN+timedelta(seconds=10)]
    acquired = gw.acquire_window
    monkeypatch.setattr(gw, "NativeGoes", lambda *a, **k: native)
    captured = {}
    def acquire_with_sink(**kwargs):
        return acquired(**kwargs, now=lambda:clock[0], write_grid=capture_writer(captured))
    monkeypatch.setattr(gw, "acquire_window", acquire_with_sink)
    def reader(path, *, expected_grid):
        document = json.loads(Path(path).read_text())
        return {"schema":"gpuwm-obs.goes-grid.v1", "valid_time":document["valid_time"],
                "provenance":document["provenance"]}
    monkeypatch.setattr(goes_grid, "read_goes_grid", reader)
    monkeypatch.setattr(radar_grid, "read_radar_grid", lambda path, **k: {"radar_test_marker":str(path)})
    monkeypatch.setattr(obs_point, "read_tables", lambda paths: ([{"table":str(p)} for p in paths], []))
    monkeypatch.setattr(obs_surface, "read_record", lambda path: {"surface":str(path)})
    backend = SimpleNamespace(root=tmp_path, grid=Grid(), analysis_times=[WHEN,WHEN+timedelta(minutes=5)],
        plan={"review_sha256":"review-for-dispatch-test", "request":{"obs_tables":[]},
              "selected":{"cadence_seconds":300}, "observations":[{
                "id":"cloud-water-path", "route":"cloud-water-path", "status":"candidate",
                "files":[], "binary":"native-test-double", "acquisition_policy":POLICY.to_payload()}]})
    return fetch, backend, clock


def test_dispatcher_freezes_real_acquisition_and_resume_calls_no_native(tmp_path, monkeypatch):
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    first = fetch.observation_window(backend, 0, WHEN, {})
    assert first["cwp"] is not None
    calls = list(native.calls)
    native.records = {SOURCE.id:listing(end=WHEN)}
    clock[0] += timedelta(hours=1)
    recovered = fetch.observation_window(backend, 0, WHEN, {})
    assert recovered["cwp"] == first["cwp"]
    assert native.calls == calls
    frozen = json.loads((tmp_path/"observations/cycle_000/window.json").read_text())
    assert {a["kind"] for a in frozen["assets"]} == {"cwp", "cwp-acquisition", "cwp-source"}


def test_consumed_scan_does_not_enter_a_second_cycle(tmp_path, monkeypatch):
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    assert fetch.observation_window(backend, 0, WHEN, {})["cwp"] is not None
    nfetch = sum(c[0] == "fetch" for c in native.calls)
    clock[0] += timedelta(minutes=5)
    second = fetch.observation_window(backend, 1, WHEN+timedelta(minutes=5), {})
    assert second["cwp"] is None
    assert sum(c[0] == "fetch" for c in native.calls) == nfetch
    assert second["receipts"][0]["status"] == "empty"


def test_late_unconsumed_scan_is_accepted_by_later_cycle(tmp_path, monkeypatch):
    record = listing(published=WHEN+timedelta(seconds=60))
    native = Native({SOURCE.id:record})
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    assert fetch.observation_window(backend, 0, WHEN, {})["cwp"] is None
    clock[0] += timedelta(minutes=5)
    second = fetch.observation_window(backend, 1, WHEN+timedelta(minutes=5), {})
    assert second["cwp"] is not None
    assert second["cwp"]["provenance"]["acquisition"]["max_observation_age_seconds"] == 420
    assert fetch.observation_window(backend, 0, WHEN, {})["cwp"] is None


@pytest.mark.parametrize("kind", ["cwp", "cwp-source", "cwp-acquisition"])
def test_modified_frozen_assets_are_fatal_not_refetched(tmp_path, monkeypatch, kind):
    from gpuwm.local_da import PlanError
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    fetch.observation_window(backend, 0, WHEN, {})
    receipt = json.loads((tmp_path/"observations/cycle_000/window.json").read_text())
    path = Path(next(a["path"] for a in receipt["assets"] if a["kind"]==kind))
    path.write_bytes(path.read_bytes()+b"modified")
    calls = list(native.calls)
    with pytest.raises(PlanError):
        fetch.observation_window(backend, 0, WHEN, {})
    assert native.calls == calls


def test_changed_prior_consumption_record_is_fatal_in_new_window(tmp_path, monkeypatch):
    from gpuwm.local_da import PlanError
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    fetch.observation_window(backend, 0, WHEN, {})
    receipt = json.loads((tmp_path/"observations/cycle_000/window.json").read_text())
    path = Path(next(a["path"] for a in receipt["assets"] if a["kind"]=="cwp-acquisition"))
    path.write_bytes(path.read_bytes()+b"modified")
    clock[0] += timedelta(minutes=5)
    with pytest.raises(PlanError):
        fetch.observation_window(backend, 1, WHEN+timedelta(minutes=5), {})
    assert not (tmp_path/"observations/cycle_001/window.json").exists()


def test_satellite_result_coexists_with_other_observation_assets(tmp_path, monkeypatch):
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    table = tmp_path/"table.csv"; table.write_text("table-fixture")
    surface = tmp_path/"surface.json"; surface.write_text("surface-fixture")
    radar = tmp_path/"radar.nc"; radar.write_text("radar-fixture")
    backend.plan["request"]["obs_tables"] = [str(table)]
    backend.plan["observations"].extend([
        {"id":"radar", "route":"radar-grid", "status":"candidate", "files":[]},
        {"id":"surface", "route":"surface-v1", "status":"candidate"}])
    monkeypatch.setattr(fetch,"_radar_fetch",lambda *a:radar)
    monkeypatch.setattr(fetch,"_surface_fetch",lambda *a:[surface])
    out = fetch.observation_window(backend,0,WHEN,{})
    assert out["cwp"] is not None and out["radar"] is not None
    assert len(out["rows"]) == len(out["surface"]) == 1


def test_operator_probe_uses_actual_checkpoint_cwp_provider():
    cfg = SimpleNamespace(mp_physics=10)
    setup = dict(c1h=np.ones(2), c2h=np.zeros(2), dnw=np.full(2,-.5), mub2d=np.full((1,1),90000.))
    state = dict(mup=np.zeros((1,1)), qc=np.full((2,1,1),.001),
                 qi=np.full((2,1,1),.0001), qs=np.zeros((2,1,1)))
    assert gw.cwp_operator_refusal(state, setup, cfg) is None
    del state["qs"]
    reason = gw.cwp_operator_refusal(state, setup, cfg)
    assert "qs" in reason and "unavailable" in reason
    assert "qs" not in state


def test_unavailable_operator_skips_auto_fetch_but_keeps_other_routes(tmp_path, monkeypatch):
    native = Native()
    fetch, backend, clock = _dispatch_setup(tmp_path, monkeypatch, native)
    backend.cwp_unavailable_reason = "configured column has no required ice species"
    result = fetch.observation_window(backend,0,WHEN,{})
    assert result["cwp"] is None and not native.calls
    assert result["receipts"][0]["reason"] == backend.cwp_unavailable_reason


def test_nonobject_native_listing_is_a_named_failure(tmp_path):
    class BadListing(Native):
        def call(self, *args, **kwargs):
            return []
    result, _ = acquire(tmp_path, BadListing())
    assert result.receipt["status"] == "unavailable"
    assert "not an object" in result.receipt["sources"][0]["reason"]


def test_bound_frontdoor_does_not_resolve_another_binary(tmp_path, monkeypatch):
    from gpuwm.obs.frontdoor import FrontDoor
    import subprocess
    binary = tmp_path / "reviewed-goes"
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout='{"schema":"test-contract"}', stderr='')
    monkeypatch.setattr(subprocess,"run",run)
    monkeypatch.setattr(FrontDoor,"require",lambda self:pytest.fail("unexpected binary re-resolution"))
    door = FrontDoor("rw_goes","TEST_ENV","test", "abi")
    assert door.run("list",["--sector","C"],schema="test-contract",binary=binary)["schema"] == "test-contract"
    assert seen == [[str(binary),"list","--sector","C"]]


def test_frontdoor_refuses_nonobject_json(monkeypatch):
    from gpuwm.obs.frontdoor import FrontDoor
    import subprocess
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout='[]',stderr=''))
    with pytest.raises(RuntimeError,match="JSON object"):
        FrontDoor("rw_goes","TEST_ENV","test","abi").run("list",[],schema="test",binary=Path("unused"))


@pytest.mark.parametrize("explicit", [False, True])
def test_preparation_checks_cwp_operator_before_forecast(tmp_path, monkeypatch, explicit):
    import sys
    import types
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm.local_da import PlanError
    radar_owner = types.ModuleType("gpuwm.da.radar_assimilation")
    def no_reflectivity(*a, **k):
        raise NotImplementedError("reflectivity not part of this test")
    radar_owner.scheme_reflectivity_provider = no_reflectivity
    state_owner = types.ModuleType("gpuwm.ensemble.state_sha")
    state_owner.serialized_state_attrs = lambda:("mup", "qc", "qi", "qs")
    monkeypatch.setitem(sys.modules,"gpuwm.da.radar_assimilation",radar_owner)
    monkeypatch.setitem(sys.modules,"gpuwm.ensemble.state_sha",state_owner)
    # The real preflight method probes a column that lacks the required qs.
    state = SimpleNamespace(mup=np.zeros((1,1)), qc=np.ones((2,1,1))*.001,
        qi=np.zeros((2,1,1)), qs=None, p=np.ones((2,1,1))*90000,
        al=np.ones((2,1,1)), alt=np.ones((2,1,1)))
    backend = SimpleNamespace(_reference_grid=lambda:None, _host=PreparedBackend._host,
        state=state, grid=Grid(), route_receipts=[], exp=SimpleNamespace(root=SimpleNamespace(run=SimpleNamespace(mp_physics=1))),
        setup=dict(thb=np.ones((2,1,1))*300, c1h=np.ones(2),c2h=np.zeros(2),dnw=np.full(2,-.5),mub2d=np.ones((1,1))*90000),
        plan={"observations":[{"route":"cloud-water-path", "status":"ready" if explicit else "candidate"}],
              "request":{"satellite_grids":["explicit.nc"] if explicit else [], "radar_grids":[], "obs_tables":[]}})
    if explicit:
        with pytest.raises(PlanError,match="required condensate"):
            PreparedBackend._validate_observation_inputs(backend)
    else:
        PreparedBackend._validate_observation_inputs(backend)
        assert "qs" in backend.cwp_unavailable_reason
        assert backend.route_receipts[-1]["status"] == "unavailable"


def test_rust_mosaic_retains_older_valid_columns_without_averaging(tmp_path):
    result, captured = acquire(tmp_path)
    product = captured['product']
    original = result.receipt['scans'][0]
    mosaic = gw.Mosaic(Grid())
    mosaic.offer(product, original)
    newer = replace(product, cwp_mask=np.array([[1,0,0,0]], np.int8), cwp_obs=np.full((1,4),200.))
    mosaic.offer(newer, {**original, 'scan_end': stamp(WHEN), 'optical_scan_end': stamp(WHEN), 'scan_id':'newer'})
    receipt = {'policy': POLICY.to_payload()}
    final = mosaic.finish(receipt)
    np.testing.assert_allclose(final.cwp_obs[0,:3], [200,0,91.7], rtol=1e-6)
    np.testing.assert_array_equal(final.cwp_mask, [[1,1,1,0]])
    np.testing.assert_array_equal(final.cwp_err[0,:3], product.cwp_err[0,:3])
    assert sum(row['selected_columns'] for row in receipt['scans']) == 3


def test_native_qc_is_not_replaced_with_an_extra_time_of_day_clamp(tmp_path):
    when = utc('2026-09-12T06:10:00Z')
    record = listing(start=when-timedelta(minutes=5),end=when-timedelta(minutes=2))
    result, captured = acquire(tmp_path, Native({SOURCE.id:record}), when=when)
    np.testing.assert_array_equal(captured['product'].cwp_mask, [[1,1,1,0]])
    assert result.receipt['observed_columns'] == 3


def test_explicit_fractional_scan_end_is_preserved_and_rejected_if_future():
    from gpuwm.local_da_fetch import assigned_document
    document = {'schema':'gpuwm-obs.goes-grid.v1', 'valid_time':stamp(WHEN),
        'provenance':{'pack':{'scan_start':stamp(WHEN-timedelta(minutes=2)),
                              'scan_end':stamp(WHEN+timedelta(microseconds=100000))}}}
    assert not assigned_document(document, WHEN, [WHEN], 900.)


def test_review_enables_native_auto_cwp_without_supplied_files(tmp_path, monkeypatch):
    import gpuwm.local_da_observations as review
    from test_local_da_plan import request
    from gpuwm.local_da import derive_rung, configuration
    from gpuwm.obs import nexrad
    binary = tmp_path / 'native-observation-door'
    binary.write_bytes(b'fixed executable fixture')
    monkeypatch.setattr(review, '_door_record', lambda *args: (str(binary), 'fixture ABI available'))
    monkeypatch.setattr(nexrad, 'find_nexrad_bin', lambda: None)
    monkeypatch.setattr(review.importlib.util, 'find_spec', lambda name: None)
    req = request()
    rung = derive_rung(req, req.scale)
    _, _, exp, _ = configuration(req, rung)
    rows = review.inspect_routes(req, rung, exp)
    cwp = next(row for row in rows if row['route'] == 'cloud-water-path')
    assert cwp['status'] == 'candidate' and cwp['files'] == []
    assert cwp['binary_sha256'] == gw.sha256(binary)
    assert cwp['acquisition_policy']['sources']
