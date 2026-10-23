"""Init-state table and admission (CPU only, no network)."""
import pytest
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import time

from gpuwm.hydro import initial_state as s


@pytest.fixture(autouse=True)
def _own_fabric_cache(tmp_path, monkeypatch):
    """The forcing-day cache sits beside the fabric cache; keep it in the test's folder."""
    monkeypatch.setenv("GPUWM_HYDRO_FABRIC_CACHE", str(tmp_path / "fabric-cache"))


_CONCURRENT_PREPARATION = r'''
import importlib.util, json, sys, time
from pathlib import Path
module, table, root, name, date = sys.argv[1:]
spec = importlib.util.spec_from_file_location('gpuwm.hydro.initial_state', module)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
s.TABLE = Path(table)
root = Path(root)
def event(message):
    with (root / (name + '.events')).open('a') as stream:
        stream.write(message + '\n')
def extract(argv, **kwargs):
    event('extract')
    deadline = time.monotonic() + 20
    while not (root / 'release').exists():
        assert time.monotonic() < deadline, 'extraction release timeout'
        time.sleep(.01)
    Path(argv[3]).write_bytes(b'metadata-test-only; not NetCDF')
    return type('Result', (), dict(returncode=0, stderr=''))()
s._rw_zarr = lambda: Path('rw_zarr')
s.subprocess.run = extract
s.run_stage = lambda *args: {}
try:
    result = s.prepare(fabric=root / 'fabric.gpkg', date=date,
        area=[38, -123, 39, -122], out=root / name,
        fetch_dir=root / 'source', progress=event)
    (root / (name + '.result')).write_text(json.dumps(result))
except Exception as error:
    event('error: ' + repr(error))
    raise
'''


@pytest.mark.parametrize('second', ['same', 'changed', 'interrupted'])
def test_prepare_concurrent_source_ownership(tmp_path, second):
    """Separate processes must share a completed fetch and recover a dead owner."""
    def start(name, date):
        return subprocess.Popen([sys.executable, '-B', '-c', _CONCURRENT_PREPARATION,
            str(s.__file__), str(s.TABLE), str(tmp_path), name, date],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def events(name):
        path = tmp_path / (name + '.events')
        return path.read_text() if path.exists() else ''

    def wait_for(predicate):
        deadline = time.monotonic() + 15
        while not predicate():
            assert time.monotonic() < deadline, (events('first'), events('second'))
            time.sleep(.01)

    first = start('first', '2023-01-01T00:00:00Z')
    other = None
    try:
        wait_for(lambda: 'extract' in events('first'))
        prior_request = (tmp_path / 'source/state.request.json').read_bytes()
        date = '2023-01-02T00:00:00Z' if second == 'changed' else '2023-01-01T00:00:00Z'
        other = start('second', date)
        wait_for(lambda: any(word in events('second') for word in ('waiting for shared source', 'extract', 'error:')))
        if second == 'interrupted':
            first.kill()
            first.communicate(timeout=5)
        (tmp_path / 'release').touch()
        if second != 'interrupted':
            _, error = first.communicate(timeout=15)
            assert first.returncode == 0, error
        _, error = other.communicate(timeout=15)
        assert other.returncode == 0, error
        assert (tmp_path / 'source/state.request.json').read_bytes() == prior_request
        count = events('first').count('\nextract\n') + events('second').count('\nextract\n')
        assert count == (1 if second == 'same' else 2)
        assert (tmp_path / 'second.result').is_file()
        if second == 'same':
            assert json.loads((tmp_path / 'first.result').read_text())['fetch'] == json.loads((tmp_path / 'second.result').read_text())['fetch']
        if second == 'interrupted':
            assert not (tmp_path / 'source/fetch-receipt.json').exists()
    finally:
        for process in (first, other):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_table_rows_and_admission():
    t = s.load_table()
    assert t["schema"] == "arwen.hydro-world.init-sources.v1"
    row = s.admit(t, "era5-arco")
    assert row["licence"]["spdx"] == "CC-BY-4.0" and row["access"]["account_required"] is False
    # Account-gated and reader-less rows are refused with the row's own reason.
    for name in ("era5-land-cds", "gldas-noah025-3h-2.1", "nwm-retrospective-v3"):
        with pytest.raises(PermissionError, match=t["rows"][name]["status"]):
            s.admit(t, name)
    with pytest.raises(KeyError):
        s.admit(t, "no-such-row")


def test_fetch_variables_follow_the_row():
    row = s.admit(s.load_table(), "era5-arco")
    state = dict(s._state_variables(row))
    assert state["volumetric_soil_water_layer_4"] == "m3 m-3" and state["soil_type"] == "code"
    forcing = dict(s._forcing_variables(row))
    assert set(forcing) == {"total_precipitation", "snowfall", "snowmelt", "potential_evaporation"}
    times = s.hours(s.parse_time("2022-12-31T23:00:00Z"), 2)
    assert times == ["2022-12-31T23:00:00Z", "2023-01-01T00:00:00Z"]


@pytest.fixture
def extraction(monkeypatch):
    transport = SimpleNamespace(calls=[], stages=[],
                                payload=b"metadata-test-only; not NetCDF")

    def extract(argv, **kwargs):
        transport.calls.append(json.loads(Path(argv[2]).read_text()))
        Path(argv[3]).write_bytes(transport.payload)
        return SimpleNamespace(returncode=0, stderr="")

    def stage(command, request, workdir):
        transport.stages.append((command, request))
        return {}

    monkeypatch.setattr(s, "_rw_zarr", lambda: Path("rw_zarr"))
    monkeypatch.setattr(s.subprocess, "run", extract)
    monkeypatch.setattr(s, "run_stage", stage)
    return transport


@pytest.mark.parametrize('prior', ['empty'])
def test_prepare_repeated_output_preserves_prior_files(tmp_path, monkeypatch, extraction, prior):
    calls = extraction.calls
    out = tmp_path / "initial-water"
    preserved = tmp_path / "original"
    preserved.write_text("original data")
    out.mkdir()
    args = dict(fabric=tmp_path / "fabric.gpkg", date="2023-01-01T00:00:00Z",
                area=[38, -123, 39, -122], out=out, progress=lambda _: None)
    first = s.prepare(**args)
    first_path = Path(first["output_directory"])
    snapshot = {p: p.read_bytes() for p in first_path.rglob("*") if p.is_file()}
    second = s.prepare(**args)
    assert second["output_directory"] != first["output_directory"]
    assert len(calls) == 1  # same extraction reused across output runs
    args["date"] = "2023-01-02T00:00:00Z"
    changed = s.prepare(**args)
    repeated = s.prepare(**args)
    assert changed["output_directory"] != repeated["output_directory"]
    assert len(calls) == 2  # changed request downloaded once, then reused
    assert first["requested_output_directory"] == str(out.absolute())
    assert all(p.read_bytes() == value for p, value in snapshot.items())
    assert preserved.read_text() == "original data"


@pytest.mark.parametrize('change', ['area'])
def test_fetch_reuse_binds_request_and_preserves_other_domains(tmp_path, monkeypatch, extraction, change):
    table = copy.deepcopy(s.load_table())
    calls, stages = extraction.calls, extraction.stages
    monkeypatch.setattr(s, "load_table", lambda: table)
    args = dict(fabric=tmp_path / "fabric.gpkg", date="2023-01-01T00:00:00Z",
                area=[38, -123, 39, -122], fetch_dir=tmp_path / "shared-source",
                spinup_days=1, progress=lambda _: None)
    first = s.prepare(**args, out=tmp_path / "first")
    original = {str(p): p.read_bytes() for p in args["fetch_dir"].rglob("*") if p.is_file()}
    assert len(calls) == 2
    s.prepare(**args, out=tmp_path / "same")
    assert len(calls) == 2
    args["area"] = [40, -121, 41, -120]
    second = s.prepare(**args, out=tmp_path / "second")
    assert len(calls) == 4
    assert second["fetch"]["area_south_west_north_east"] == args["area"]
    assert stages[-4][1]["grid_file"] != str(args["fetch_dir"] / "state.nc")
    for path, contents in original.items():
        assert Path(path).read_bytes() == contents
    s.prepare(**args, out=tmp_path / "second-repeated")
    assert len(calls) == 4
    assert first["fetch"]["area_south_west_north_east"] == [38, -123, 39, -122]


@pytest.mark.parametrize('damage', ['invalid_size', 'duplicate_slice', 'legacy'])
def test_fetch_payload_integrity_recovery(tmp_path, monkeypatch, extraction, damage):
    """Do not pass changed sealed inputs to native stages, or rewrite old inputs."""
    calls, mapped = extraction.calls, []
    payload = b"opaque transport fixture; not NetCDF"
    extraction.payload = payload

    def stage(command, request, workdir):
        if command == "weights":
            mapped.append(Path(request["grid_file"]))
        return {}

    monkeypatch.setattr(s, "run_stage", stage)
    cache = tmp_path / "source"
    args = dict(fabric=tmp_path / "fabric.gpkg", date="2023-01-01T00:00:00Z",
                area=[38, -123, 39, -122], fetch_dir=cache, progress=lambda _: None)
    s.prepare(**args, out=tmp_path / "first")
    receipt_path = cache / "fetch-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    part = receipt["slices"][0]
    if damage == "invalid_size":
        part["size_bytes"] = str(len(payload))
    elif damage == "duplicate_slice":
        receipt["slices"].append(dict(part))
    else:
        part.pop("size_bytes", None)
        part.pop("sha256", None)
    receipt_path.write_text(json.dumps(receipt))
    preserved = {p: p.read_bytes() for p in cache.iterdir() if p.is_file()}
    s.prepare(**args, out=tmp_path / "second")
    assert len(calls) == (1 if damage == "legacy" else 2)
    assert (mapped[-1] == cache / "state.nc") == (damage == "legacy")
    assert mapped[-1].read_bytes() == payload
    assert all(p.read_bytes() == contents for p, contents in preserved.items())
    s.prepare(**args, out=tmp_path / "third")
    assert len(calls) == (1 if damage == "legacy" else 2)


def test_fetch_receipt_records_actual_payload_bytes(tmp_path, extraction):
    payload = b"opaque transport fixture; not NetCDF"
    extraction.payload = payload
    receipt = s.fetch(s.admit(s.load_table(), "era5-arco"), area=[38, -123, 39, -122],
                      state_times=["2023-01-01T00:00:00Z"], forcing_times=[],
                      out=tmp_path / "source", progress=lambda _: None)
    part = receipt["slices"][0]
    assert part["size_bytes"] == len(payload)
    assert part["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("live", ["covered", "unavailable"])
def test_live_start_spins_up_to_the_record_end_and_carries(tmp_path, monkeypatch, extraction, live):
    """A forecast past the reanalysis's last hour (every live run) spins up to
    that hour and carries the state to the start, with the live analysis soil
    where it covers the domain, else the named era5_carry_over fallback.
    Breakage it prevents: the ERA5 fetch past its coverage failed every live
    run's hydro preparation, so the forecast ran with hydro off."""
    monkeypatch.setattr(s, "source_last_time", lambda row: s.parse_time("2026-10-15T23:00:00Z"))

    def live_soil(**kwargs):
        if live == "unavailable":
            raise LookupError("this domain lies outside it")
        return tmp_path / "live-state", {"row": s.LIVE_ROW, "valid_time": "2026-10-21T08:00:00Z"}

    monkeypatch.setattr(s, "live_soil", live_soil)
    result = s.prepare(fabric=tmp_path / "fabric.gpkg", date="2026-10-21T08:00:00Z", spinup_days=2,
                       area=[38, -123, 39, -122], out=tmp_path / "w", progress=lambda _: None)
    fetched = sorted(t for call in extraction.calls for t in call["times"])
    assert fetched[0] == "2026-10-13T23:00:00Z" and fetched[-1] == "2026-10-15T23:00:00Z"
    spin = dict(extraction.stages)["spinup"]
    assert spin["start"] == "2026-10-13T23:00:00Z" and spin["hours"] == 48
    assert spin["carry_to"] == "2026-10-21T08:00:00Z"
    water = result["initial_water"]
    assert water["carried_hours"] == 129
    if live == "covered":
        assert water["method"] == "live_analysis_soil_over_carried_spinup"
        assert spin["soil_override"] == str(tmp_path / "live-state")
    else:
        assert water["method"] == "era5_carry_over" and water["fallback"] is True
        assert "soil_override" not in spin and "outside it" in water["live_analysis_unavailable"]


def test_a_later_start_reuses_the_whole_forcing_days(tmp_path, monkeypatch, extraction):
    """Whole UTC days of spin-up forcing are fetched once per area and reused by a
    later start; only its new days and edge hours are fetched. Breakage it prevents:
    every start re-fetched all 8,666 ERA5 chunks (about 300 s on the HRRR recipe)."""
    args = dict(fabric=tmp_path / "fabric.gpkg", spinup_days=3, area=[38, -123, 39, -122],
                progress=lambda _: None)
    s.prepare(date="2023-01-10T00:00:00Z", out=tmp_path / "a", **args)
    first = [call["times"] for call in extraction.calls]
    extraction.calls.clear()
    s.prepare(date="2023-01-11T00:00:00Z", out=tmp_path / "b", **args)
    second = [call["times"] for call in extraction.calls]
    whole = lambda calls: sorted(t[0][:10] for t in calls if len(t) == 24 and t[0][11:13] == "00")  # noqa: E731
    assert whole(first) == ["2023-01-08", "2023-01-09"]
    # The later start fetches only the whole day it adds (01-09 comes from the cache)
    # plus its own partial edge hours.
    assert whole(second) == ["2023-01-10"]


def _holed_stages(extraction, monkeypatch, tmp_path, hole_days, fillable):
    """run_stage that reports NaN records on HOLE_DAYS (forcing-gaps) and patches them
    (forcing-fill); downloads succeed only when FILLABLE."""
    def stage(command, request, workdir):
        extraction.stages.append((command, request))
        if command == "forcing-gaps":
            return {"gaps": [{"file": f, "time": f"{day}T{h:02d}:00:00Z", "variables": ["potential_evaporation"]}
                             for f in request["files"] for day in hole_days
                             if f.endswith(".nc") and day in Path(f).name for h in range(24)]}
        if command == "forcing-fill":
            return {"files": [{"file": fill["file"], "output": fill["output"], "still_missing": [],
                               "patched": [{"time": rec["time"], "variable": src["variable"]}
                                           for rec in fill["records"] for src in rec["sources"]]}
                              for fill in request["fills"]]}
        return {}
    downloads = []

    def download(url, path):
        downloads.append(url)
        if not fillable:
            raise OSError("HTTP Error 404: Not Found")
        Path(path).write_bytes(b"day file fixture")
        return {}
    monkeypatch.setattr(s, "run_stage", stage)
    monkeypatch.setattr(s, "_download", download)
    return downloads


def test_store_holes_are_filled_from_the_next_source_and_the_receipt_names_each_segment(
        tmp_path, monkeypatch, extraction):
    """A day the store serves as NaN inside its declared record is filled from the
    row's gap_fill source (ARCO raw ERA5T day files), so the spin-up runs every hour it
    was asked for, to the forecast start, and the receipt names the source of each
    segment. Breakage it prevents (2.9.4 WOOF readiness g2/g4, 2026-10-22): the ARCO
    store served 2026-10-10 and 2026-10-14 as NaN; starts inside the record gave up
    ('potential_evaporation missing over a catchment', hydro off) and carried starts
    ran 1,992 of 2,160 h."""
    downloads = _holed_stages(extraction, monkeypatch, tmp_path, ["2026-10-10"], fillable=True)
    result = s.prepare(fabric=tmp_path / "fabric.gpkg", date="2026-10-12T00:00:00Z", spinup_days=4,
                       area=[38, -123, 39, -122], out=tmp_path / "w", progress=lambda _: None)
    assert len(downloads) == 1 and "/2026/10/10/potential_evaporation/surface.nc" in downloads[0]
    spin = dict(extraction.stages)["spinup"]
    assert spin["hours"] == 96 and spin["start"] == "2026-10-08T00:00:00Z" and "carry_to" not in spin
    assert any(f.endswith("-filled.nc") and "2026-10-10" in f for f in spin["forcing_files"])
    water = result["initial_water"]
    assert water["spinup_hours"] == {"requested": 96, "ran": 96}
    assert [(seg["first"], seg["last"], seg["hours"]) for seg in water["segments"]] == [
        ("2026-10-08T01:00:00Z", "2026-10-09T23:00:00Z", 47),
        ("2026-10-10T00:00:00Z", "2026-10-10T23:00:00Z", 24),
        ("2026-10-11T00:00:00Z", "2026-10-12T00:00:00Z", 25)]
    assert "era5t-arco-raw-days" in water["segments"][1]["source"]
    assert result["gap_fill"]["records_patched"] == 24


def test_a_hole_nothing_fills_moves_the_whole_spinup_earlier_and_says_so(tmp_path, monkeypatch, extraction):
    """A hole no fill source closes never makes the spin-up short or gives up: it ends
    before the hole, starts earlier by as much (its full length) and is carried to the
    start, with the reason in the receipt."""
    _holed_stages(extraction, monkeypatch, tmp_path, ["2026-10-10"], fillable=False)
    monkeypatch.setattr(s, "live_soil", lambda **kw: (_ for _ in ()).throw(LookupError("outside it")))
    result = s.prepare(fabric=tmp_path / "fabric.gpkg", date="2026-10-12T00:00:00Z", spinup_days=4,
                       area=[38, -123, 39, -122], out=tmp_path / "w", progress=lambda _: None)
    spin = [request for command, request in extraction.stages if command == "spinup"][-1]
    assert spin["hours"] == 96 and spin["start"] == "2026-10-05T23:00:00Z"
    assert spin["carry_to"] == "2026-10-12T00:00:00Z"
    water = result["initial_water"]
    assert water["spinup_end"] == "2026-10-09T23:00:00Z" and water["spinup_hours"]["ran"] == 96
    assert water["held_back"][0]["unfilled_from"] == "2026-10-10T00:00:00Z"
    assert water["segments"][-1]["source"].startswith("carried") and water["segments"][-1]["hours"] == 49


def _bbox_stage(extraction, monkeypatch, box):
    def stage(command, request, workdir):
        extraction.stages.append((command, request))
        return {"area_south_west_north_east": list(box)} if command == "bbox" else {}
    monkeypatch.setattr(s, "run_stage", stage)


def test_the_footprint_prefetch_is_exactly_what_the_preparation_reads(tmp_path, monkeypatch, extraction):
    """2.9.4 R11 (2026-10-22): a first run on a new region fetched its 89 ERA5 forcing days
    only after the fabric's own sources were downloaded and the fabric built. The fetch
    now starts from the footprint's area beside them; the preparation that follows reads
    those very files and extracts nothing more."""
    footprint = {"south": 35.55, "west": -87.81, "north": 36.84, "east": -85.79}
    hint = s.footprint_area(footprint)
    assert hint == [35.55 - 0.75, -87.81 - 0.75, 36.84 + 0.75, -85.79 + 0.75]
    # The fabric's own box: the footprint plus the bbox stage's 0.5 degree pad.
    _bbox_stage(extraction, monkeypatch, [35.05, -88.31, 37.34, -85.29])
    out = tmp_path / "initial"
    fetched = s.prefetch(date="2023-01-04T00:00:00Z", out=out, area=hint, spinup_days=3,
                         progress=lambda _: None)
    assert fetched["forcing_days"]["fetched_days"] == 2
    prefetched = len(extraction.calls)
    assert prefetched == 2 + 2  # state + edge hours, and the two whole days
    result = s.prepare(fabric=tmp_path / "fabric.gpkg", date="2023-01-04T00:00:00Z", out=out,
                       spinup_days=3, area_hint=hint, progress=lambda _: None)
    assert len(extraction.calls) == prefetched
    assert {tuple(call["area"]) for call in extraction.calls} == {tuple(hint)}
    assert result["source_area"] == {"area_south_west_north_east": hint, "from": "footprint"}
    assert result["forcing_days"]["reused_days"] == 2 and result["forcing_days"]["fetched_days"] == 0


def test_a_fabric_reaching_past_the_footprint_area_reads_its_own_box(tmp_path, monkeypatch, extraction):
    hint = s.footprint_area({"south": 35.0, "west": -88.0, "north": 36.0, "east": -86.0})
    box = [34.0, -88.5, 36.5, -85.5]  # a catchment drawn a degree south of the footprint
    _bbox_stage(extraction, monkeypatch, box)
    result = s.prepare(fabric=tmp_path / "fabric.gpkg", date="2023-01-04T00:00:00Z", out=tmp_path / "w",
                       spinup_days=2, area_hint=hint, progress=lambda _: None)
    assert result["source_area"] == {"area_south_west_north_east": box, "from": "fabric box"}
    assert {tuple(call["area"]) for call in extraction.calls} == {tuple(box)}


def test_state_and_forcing_days_are_fetched_at_the_same_time(tmp_path, monkeypatch, extraction):
    """The whole forcing days no longer wait for the state and edge extraction (11-14 s of
    every first run's preparation): both are in flight together."""
    import threading
    both = threading.Barrier(2, timeout=10)
    inner = s.subprocess.run

    def extract(argv, **kwargs):
        if "forcing-days" in argv[3] or Path(argv[3]).name == "state.nc":
            both.wait()  # the state extraction and the day extraction must meet here
        return inner(argv, **kwargs)
    monkeypatch.setattr(s.subprocess, "run", extract)
    monkeypatch.setattr(s, "source_last_time", lambda row: None)
    s.prepare(fabric=tmp_path / "fabric.gpkg", date="2023-01-03T00:00:00Z", out=tmp_path / "w",
              spinup_days=2, area=[38, -123, 39, -122], progress=lambda _: None)
    assert not both.broken
