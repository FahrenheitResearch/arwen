"""Candidate metadata and native payload binding, without CDS or a forecast."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import cli, companion_domains, companion_forcing, era5_acquisition, era5_member, fetch, go_cli, runplan
from test_case_data import make_case_toml


def candidate(tmp_path, monkeypatch):
    path = make_case_toml(tmp_path, files=False)
    path.write_text(path.read_text().replace("run_seconds = 3600.0", "run_seconds = 46800.0") +
        '\n[projection]\nmap_proj="lambert"\nref_lat=36.0\nref_lon=-98.0\ntruelat1=30.0\ntruelat2=60.0\nstand_lon=-98.0\n' +
        '\n[fetch]\nsource="era5"\ncycle="1999-05-03T12"\nhours=18\ncadence=6\narea="35.5,-98.5,36.5,-97.5"\n')
    # Geometry itself has separate native integration proof. This test owns
    # the preserved clock, paths, WPS cadence and actual fetch argument seam.
    monkeypatch.setattr(companion_domains, "native_domain_outlines", lambda exp: [])
    return dict(schema=companion_forcing.REQUEST_SCHEMA, config_path=str(path),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        output_path=str(tmp_path / "candidate.toml"), product_type="ensemble_members",
        cadence_hours=3, member=7, provider="cds")


def test_candidate_binds_real_command_wps_and_cache_without_changing_run(tmp_path, monkeypatch):
    request = candidate(tmp_path, monkeypatch)
    source = Path(request["config_path"]); before = source.read_bytes()
    result = companion_forcing.edit_configuration(request)
    raw = tomllib.loads(Path(result["config_path"]).read_text())
    assert source.read_bytes() == before
    assert raw["experiment"]["run_seconds"] == 46800
    assert raw["fetch"]["hours"] == 15
    assert raw["fetch"]["cadence"] == 3
    assert raw["fetch"]["member"] == 7
    assert raw["case_data"]["forcing_interval_s"] == 10800
    from gpuwm.namelist_import import parse_namelist_text
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["share"]["interval_seconds"] == [10800]
    assert wps["share"]["end_date"] == ["1999-05-04_01:00:00"]
    parsed = cli.build_parser().parse_args(result["fetch_argv"])
    assert (parsed.member, parsed.era5_product, parsed.cadence, parsed.retrieve) == ("7", "ensemble_members", 3, True)
    assert result["acquisition_started"] is False and result["forecast_started"] is False
    request.update(output_path=str(tmp_path / "member8.toml"), member=8)
    other = companion_forcing.edit_configuration(request)
    assert result["selection"]["forcing_path"] != other["selection"]["forcing_path"]
    request.update(output_path=str(tmp_path / "rean.toml"), product_type="reanalysis", member=None, cadence_hours=1)
    hourly = companion_forcing.edit_configuration(request)
    assert hourly["selection"]["boundary_window_hours"] == 13
    assert "member" not in hourly["configuration"]["fetch"]
    assert hourly["configuration"]["case_data"]["forcing_interval_s"] == 3600


@pytest.mark.parametrize("change", [dict(expected_sha256="0"*64), dict(member=10), dict(member=None),
    dict(cadence_hours=1), dict(cadence_hours=3.0), dict(provider="arco")])
def test_candidate_refusal_publishes_nothing(tmp_path, monkeypatch, change):
    request = candidate(tmp_path, monkeypatch); request.update(change)
    with pytest.raises(ValueError):
        companion_forcing.edit_configuration(request)
    assert not (tmp_path / "candidate.toml").exists()
    assert not (tmp_path / "candidate.namelist.wps").exists()


def test_existing_eda_forcing_still_reaches_verified_reuse(tmp_path):
    forcing = tmp_path / fetch.ERA5_COMBINED_NAME; forcing.write_bytes(b"unverified")
    hints = dict(source="era5", cycle="2013-05-31T18", hours=3, cadence=3, area="35.5,-98.5,36.5,-97.5",
        out=str(tmp_path), era5_provider="cds", era5_product="ensemble_members", member=0, retrieve=True)
    args = runplan.declared_forcing_fetch({"fetch": hints}, SimpleNamespace(forcing=[forcing]))
    assert args[args.index("--member")+1] == "0" and "--retrieve" in args
    command = go_cli.fetch_command({**hints, "data": tmp_path})
    assert command[command.index("--member")+1] == "0" and "--retrieve" in command


def test_cli_forwards_eda_to_actual_retrieval_contract(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(era5_acquisition, "retrieve_era5", lambda **kw: captured.append(kw))
    args = cli.build_parser().parse_args(["fetch", "--source", "era5", "--cycle", "2013-05-31T18", "--hours", "3",
        "--area=35.5,-98.5,36.5,-97.5", "--out", str(tmp_path), "--era5-product", "ensemble_members",
        "--era5-provider", "cds", "--member", "0", "--cadence", "3", "--retrieve"])
    assert fetch.fetch_main(args) == 0
    assert captured[0]["member"] == 0 and captured[0]["product_type"] == "ensemble_members" and captured[0]["cadence"] == 3


def _native_bridge():
    try:
        return era5_member.require_bridge()
    except (ValueError, OSError):
        pytest.skip("matching native EDA member bridge is not installed")


def test_bulk_acquisition_selects_native_identity_and_reuses_only_matching_receipt(tmp_path, monkeypatch):
    """Synthetic full header census; values are deliberately NOT a science fixture."""
    bridge = _native_bridge()
    monkeypatch.setattr(era5_member, "require_bridge", lambda: bridge)
    real = (Path(__file__).resolve().parents[1] / "tools/grib1_bridge/tests/fixtures/era5-eda-ten-t2m.grib").read_bytes()
    calls = []
    class Client:
        def retrieve(self, dataset, request, destination=None):
            if destination is None:
                from types import SimpleNamespace
                return SimpleNamespace(download=lambda path: self.retrieve(dataset, request, path))
            calls.append((dataset, request))
            assert request["product_type"] == ["ensemble_members"]
            assert "number" not in request and "member" not in request
            pressure = "pressure-levels" in dataset
            params = fetch.ERA5_REQUIRED_PRESSURE if pressure else {**fetch.ERA5_REQUIRED_SURFACE, **fetch.ERA5_OPTIONAL_SURFACE, 129: "z"}
            qualified = [(128, parameter) for parameter in params]
            lake_parameters = {"lake_mix_layer_temperature": 8,
                               "lake_ice_temperature": 13, "lake_ice_depth": 14}
            if not pressure:
                qualified.extend((228, parameter) for name, parameter in lake_parameters.items()
                                 if name in request["variable"])
            levels = fetch.ERA5_PRESSURE_LEVELS_HPA if pressure else (0,)
            messages = bytearray()
            for clock in request["time"]:
                for table, parameter in qualified:
                    for level in levels:
                        for m in range(10):
                            data = bytearray(real[m*130:(m+1)*130])
                            data[8+3] = table
                            data[8+8] = parameter
                            data[8+9] = 100 if pressure else 1
                            data[8+10:8+12] = level.to_bytes(2, "big")
                            data[8+15] = int(clock[:2])
                            messages.extend(data)
            Path(destination).write_bytes(messages)
    monkeypatch.setattr(era5_acquisition, "_client", lambda progress: Client())
    options = dict(cycle=datetime(2013, 5, 31, 18), hours=3, cadence=3,
        area="35.5,-98.5,36.5,-97.5", out=tmp_path, product_type="ensemble_members", member=7, progress=lambda _: None)
    target = era5_acquisition.retrieve_era5(**options)
    assert len(calls) == 2  # All levels/variables in two jobs, not 222 jobs.
    report = era5_member.check_member(target, 7, bridge=bridge)
    assert report["messages"] == 416
    receipt = json.loads((tmp_path / "era5-acquisition.json").read_text())
    assert receipt["member_selection"]["input_messages"] == 4160
    assert receipt["request"]["member"] == 7
    before = target.read_bytes()
    assert era5_acquisition.retrieve_era5(**options) == target and len(calls) == 2
    with pytest.raises(FileExistsError):
        era5_acquisition.retrieve_era5(**{**options, "member": 8})
    assert target.read_bytes() == before and len(calls) == 2
    # A matching request/digest pair cannot hide a payload for another member.
    tampered = bytearray(before)
    for start in range(0, len(tampered), 130): tampered[start+8+49] = 8
    target.write_bytes(tampered)
    receipt["artifact"]["sha256"] = hashlib.sha256(tampered).hexdigest()
    (tmp_path / "era5-acquisition.json").write_text(json.dumps(receipt))
    with pytest.raises(FileExistsError):
        era5_acquisition.retrieve_era5(**options)
    assert len(calls) == 2


@pytest.mark.parametrize("change", [
    {"table_qualified_census": False},
    {"lake_surface_parameters": None},
    {"lake_surface_parameters": {"center": 98, "table": 128,
                                 "parameters": [8, 13, 14], "surface_only": True}},
])
def test_member_bridge_rejects_incomplete_lake_capabilities(monkeypatch, change):
    from gpuwm import bridges
    report = dict(schema=era5_member.SCHEMA, byte_preserving=True,
        members=list(range(10)), complete_input_census=True,
        local_definitions=[1, 17, 36], table_qualified_census=True,
        lake_surface_parameters={"center": 98, "table": 228,
                                 "parameters": [8, 13, 14], "surface_only": True})
    monkeypatch.setattr(bridges, "find_bridge", lambda name: Path("matching-bridge"))
    monkeypatch.setattr(era5_member, "_invoke", lambda *args, **kwargs: report)
    assert era5_member.require_bridge() == Path("matching-bridge")
    report.update(change)
    with pytest.raises(ValueError, match="table-qualified ERA5 lake-field"):
        era5_member.require_bridge()


def test_cds_progress_exposes_actual_request_counts_and_bytes_without_raw_client_details(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import threading
    from gpuwm import progress as progress_mod
    events=[]
    partial_seen = threading.Event()
    class Client:
        def __init__(self, information):self.information=information
        def retrieve(self,dataset,request):
            self.information("fetch era5: CDS request queued")
            self.information("fetch era5: CDS request running")
            def download(destination):
                partial_seen.clear()
                partial = Path(destination + ".download")
                partial.write_bytes(b"metadata")
                assert partial_seen.wait(3), "No byte update arrived before the CDS file completed"
                partial.write_bytes(b"metadata transfer fixture")
                partial.rename(destination)
            return SimpleNamespace(content_length=len(b"metadata transfer fixture"), download=download)
    monkeypatch.setattr(era5_acquisition,"_client",lambda information:Client(information))
    monkeypatch.setattr(era5_acquisition,"_validate",lambda *args,**kwargs:SimpleNamespace(checks=("protocol-only validation fixture",)))
    def observe(event, **fields):
        events.append({"event":event,**fields})
        if event == "fetch_progress" and fields.get("bytes") == len(b"metadata"):
            partial_seen.set()
    with progress_mod.event_sink(observe):
        target=era5_acquisition.retrieve_era5(cycle=datetime(1997,5,27),hours=3,cadence=1,area="30,-99,31,-98",out=tmp_path,progress=lambda message:None)
    completed=[e for e in events if e["event"]=="fetch_completed"]
    assert len(completed)==2 and all(e["bytes"]==len(b"metadata transfer fixture") for e in completed)
    snapshots=[e["acquisition"] for e in events if "acquisition" in e]
    assert any(e["phase"]=="cds_queued" for e in snapshots)
    assert any(e["phase"]=="downloading" for e in snapshots)
    assert any(e.get("bytes")==len(b"metadata") and e.get("expected_bytes")==len(b"metadata transfer fixture") for e in events)
    assert snapshots[-1]["phase"]=="ready"
    assert snapshots[-1]["requests_completed"]==snapshots[-1]["requests_total"]==2
    assert snapshots[-1]["forcing_times_completed"]==snapshots[-1]["forcing_times_total"]==4
    assert snapshots[-1]["bytes_available"]==target.stat().st_size


def test_preparation_metadata_uses_the_existing_native_stage_counts():
    from gpuwm import progress as progress_mod
    events=[]
    with progress_mod.event_sink(lambda event,**fields:events.append({"event":event,**fields})):
        with progress_mod.prep_stage("native_static_fields",label="Preparing geography",count=3,index=2,backend="native"):
            pass
    details=[e["preparation"] for e in events if e.get("code")=="preparation_progress"]
    assert [e["event"] for e in details]==["started","finished"]
    assert all(e["index"]==2 and e["count"]==3 and e["backend"]=="native" for e in details)
    assert details[-1]["elapsed_seconds"]>=0


def test_arco_reanalysis_is_offered_and_builds_a_candidate(tmp_path, monkeypatch):
    """The 2.7.3 sweep: the editor stopped refusing a shipped provider.

    ARCO was absent from the provider list because full forcing through
    it had not been qualified, while `gpuwm fetch --era5-provider arco`
    ships it and everything downstream of the provider id here is
    provider-generic.  EDA stays CDS-only for a product reason -- the
    ARCO archive carries no ensemble members -- and says so.
    """
    providers = {p["id"]: p for p in companion_forcing.capabilities()["providers"]}
    assert set(providers) == {"cds", "arco"}
    assert providers["arco"]["requires_credentials"] is False
    assert [p["id"] for p in providers["arco"]["products"]] == ["reanalysis"]

    request = candidate(tmp_path, monkeypatch)
    request.update(provider="arco", product_type="reanalysis", member=None,
                   cadence_hours=6)
    result = companion_forcing.edit_configuration(request)
    assert result["selection"]["provider"] == "arco"
    raw = tomllib.loads(Path(result["config_path"]).read_text())
    assert raw["fetch"]["era5_provider"] == "arco"

    # An unknown provider still refuses, naming the two that exist.
    request["provider"] = "not-a-provider"
    with pytest.raises(ValueError) as caught:
        companion_forcing.edit_configuration(request)
    assert "'cds'" in str(caught.value) and "'arco'" in str(caught.value)

    # EDA on ARCO refuses for its own product reason, not for evidence.
    with pytest.raises(ValueError, match="EDA requires CDS"):
        era5_member.validate_selection(
            product_type="ensemble_members", member=3, cadence=3,
            provider="arco")
