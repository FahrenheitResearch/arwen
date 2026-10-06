"""Annual public archive grammar and native preparation reachability."""
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
import pytest
from gpuwm import cf_archive_fetch as cf, fetch, fetch_routes
from gpuwm.source_drivability import drivability_for


@pytest.mark.parametrize("cycle", ["1836-01-01T00", "1925-03-18T12", "1957-12-18T12", "2015-12-31T18"])
def test_supported_analysis_windows(cycle):
    times = cf.validate_window("20crv3-cf", datetime.fromisoformat(cycle), 3, 3)
    assert len(times) == 2


@pytest.mark.parametrize("cycle,hours", [("1835-12-31T21",3),("2015-12-31T21",3),("1925-03-18T13",3),("1925-03-18T12",4)])
def test_coverage_and_native_clock_refuse_before_download(cycle,hours):
    with pytest.raises(ValueError):
        cf.validate_window("20crv3-cf", datetime.fromisoformat(cycle), hours, 3)


def test_annual_files_switch_distribution_and_keep_year_boundary():
    times = cf.validate_window("20crv3-cf",datetime(1980,12,31,21),3,3)
    rows = cf.plans("20crv3-cf",times,fetch.parse_area("25,-110,50,-75"))
    assert len(rows) == 28
    assert "/prsSI/air.1980.nc?" in rows[0]["url"]
    assert "/prsMO/air.1981.nc?" in rows[13]["url"]
    assert rows[-2]["url"].endswith("timeInvariantSI/hgt.sfc.nc")
    assert rows[-1]["url"].endswith("timeInvariantSI/land.nc")
    query = parse_qs(urlsplit(rows[0]["url"]).query)
    assert query["time_start"] == ["1980-12-31T21:00:00Z"]
    assert query["time_end"] == ["1980-12-31T21:00:00Z"]
    assert {r["stem"] for r in rows if r["primary"]} >= {"tsoil","soilw","skt"}


def test_native_cf_front_door_and_go_share_bound_handoff():
    assert fetch.source_argument("20cr-netcdf") == "20crv3-cf"
    assert fetch.native_cf_fetch_contract("20cr-netcdf") == "native-cf-subset-v1"
    assert fetch_routes.publishes_prep_handoff("20crv3-cf")
    assert fetch.fetch_accepts_area("20crv3-cf")
    assert not drivability_for("20crv3-cf").get("requires_source_root",False)
    assert drivability_for("20crv3-cf")["chain"] == "prepared:staged"
    fetch.validate_fetch_hints({"source":"20crv3-cf","cycle":"1925-03-18T12", "hours":3,"cadence":3,"area":"25,-110,50,-75"},source="test")
    assert fetch.resolve_latest_cycle("20crv3-cf",6,cadence=6) == datetime(2015,12,31,15)
    fetch.validate_fetch_hints({"source":"20crv3-cf","cycle":"1925-03-18T12", "hours":3,"cadence":3,"source_root":"staged"},source="test")


def test_staged_root_is_explicit_while_native_default_downloads():
    from gpuwm.source_drivability import local_input_requested
    assert not local_input_requested({"source":"20crv3-cf"})
    assert local_input_requested({"source":"20crv3-cf","source_root":"staged"})
    assert fetch_routes.source_root_layout("20crv3-cf")["inputs"]["format"] == "netcdf"


def test_water_boundary_derivation_consumes_a_single_canonical_frame():
    import numpy as np
    from gpuwm.source_authorities import packaged_authorities
    from gpuwm.mapped_source import load_mapping, CanonicalField, _evaluate_derivation
    mapping = load_mapping(packaged_authorities("20crv3-netcdf-v1")["mapping"])
    skin = mapping["fields"]["skin_temperature"]
    sst = mapping["fields"]["sea_surface_temperature"]
    assert sst["source_axes"] == skin["target_axes"] == ["y","x"]
    assert sst["units"]["source"] == skin["units"]["target"] == "K"
    values = np.asarray([[280.0,281.0],[282.0,283.0]])
    source = CanonicalField(name="skin_temperature",values=values,axes=("y","x"),units="K",location="surface",
                            staggering="none",missing_count=0,source_references=("source.skt",))
    operation = next(row for row in mapping["derivations"] if row["name"] == sst["derivation"])
    received,axes,references = _evaluate_derivation(operation,{"skin_temperature":source},None,sst,"sea_surface_temperature")
    np.testing.assert_array_equal(received,values)
    assert axes == ("y","x") and references == ("source.skt",)


def test_managed_cache_recovers_changed_requests_and_damaged_receipts(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from gpuwm import netcdf_bridge
    downloads = []
    bindings = []
    def download(url,path):
        downloads.append(url)
        path.write_bytes(url.encode())
    def run(argv,**kwargs):
        if argv[1] == "--abi":
            return SimpleNamespace(stdout="bind_published_invariants_v1")
        bindings.append(argv)
        Path(argv[6]).write_bytes(b"bound-invariant")
        return SimpleNamespace(stdout=json.dumps({"schema":"gpuwm-published-invariant-binding-v1"}))
    monkeypatch.setattr(netcdf_bridge,"resolve_netcdf_bin",lambda:Path("reader"))
    monkeypatch.setattr(netcdf_bridge,"_run",run)
    monkeypatch.setattr(cf,"_download",download)
    monkeypatch.setattr(cf,"_validate",lambda *args:None)
    kwargs = dict(source="20crv3-cf",cycle=datetime(1925,3,18,12),hours=3,cadence=3,
                  area=fetch.parse_area("25,-110,50,-75"),out=tmp_path / "inputs",workers=1)
    first = cf.acquire(**kwargs)
    assert len(downloads) == 15 and len(bindings) == 1
    from gpuwm.go_cli import managed_download_dir
    request = {"source":"20crv3-cf","cycle":"1925-03-18T12","hours":3,"cadence":3,"area":"25,-110,50,-75"}
    managed = managed_download_dir(tmp_path / "case",request)
    cf.acquire(**dict(kwargs,out=managed))
    assert managed_download_dir(tmp_path / "case",request) == managed
    # Keep the recovery counts below scoped to the explicitly supplied output.
    downloads.clear()
    cf.acquire(**kwargs)
    assert len(downloads) == 0 and len(bindings) == 2
    (kwargs["out"] / "air.1925.nc.fetch.json").write_text("broken receipt")
    (kwargs["out"] / "invariant.nc").write_bytes(b"damaged invariant")
    cf.acquire(**kwargs)
    assert len(downloads) == 1 and len(bindings) == 3
    original = (kwargs["out"] / "hgt.1925.nc").read_bytes()
    kwargs["area"] = fetch.parse_area("20,-110,50,-75")
    last = cf.acquire(**kwargs)
    assert len(downloads) == 16 and len(bindings) == 4
    assert last["request"] != first["request"]
    assert any(path.read_bytes() == original for path in kwargs["out"].glob("hgt.1925.nc.previous-request*"))
    handoff = json.loads((kwargs["out"] / "prep-arguments.json").read_text())
    assert "--input-list" in handoff["argv"] and "--supplement" in handoff["argv"]
