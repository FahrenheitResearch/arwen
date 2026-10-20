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


@pytest.mark.parametrize("prior", ["empty", "result", "file", "symlink", "current"])
def test_prepare_repeated_output_preserves_prior_files(tmp_path, monkeypatch, prior):
    calls = []
    monkeypatch.setattr(s, "_rw_zarr", lambda: Path("rw_zarr"))

    def extract(argv, **kwargs):
        calls.append(json.loads(Path(argv[2]).read_text()))
        Path(argv[3]).write_bytes(b"metadata-test-only; not NetCDF")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(s.subprocess, "run", extract)
    monkeypatch.setattr(s, "run_stage", lambda *args: {})
    out = tmp_path / "initial-water"
    preserved = tmp_path / "original"
    preserved.write_text("original data")
    if prior == "current":
        monkeypatch.chdir(tmp_path)
        out = Path(".")
    elif prior == "file":
        out.write_text("original data")
    elif prior == "symlink":
        out.symlink_to(preserved)
    else:
        out.mkdir()
        if prior == "result":
            (out / "prior.json").write_text("original data")
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
    if prior == "file":
        assert out.read_text() == "original data"
    elif prior == "symlink":
        assert out.is_symlink()
    elif prior == "result":
        assert (out / "prior.json").read_text() == "original data"


@pytest.mark.parametrize("change", ["area", "date", "store", "variable", "missing_payload"])
def test_fetch_reuse_binds_request_and_preserves_other_domains(tmp_path, monkeypatch, change):
    table = copy.deepcopy(s.load_table())
    calls, stages = [], []
    monkeypatch.setattr(s, "load_table", lambda: table)
    monkeypatch.setattr(s, "_rw_zarr", lambda: Path("rw_zarr"))

    def extract(argv, **kwargs):
        request = json.loads(Path(argv[2]).read_text())
        calls.append(request)
        Path(argv[3]).write_bytes(b"metadata-test-only; not NetCDF")
        return SimpleNamespace(returncode=0, stderr="")

    def stage(command, request, workdir):
        stages.append((command, request))
        return {}

    monkeypatch.setattr(s.subprocess, "run", extract)
    monkeypatch.setattr(s, "run_stage", stage)
    args = dict(fabric=tmp_path / "fabric.gpkg", date="2023-01-01T00:00:00Z",
                area=[38, -123, 39, -122], fetch_dir=tmp_path / "shared-source",
                spinup_days=1, progress=lambda _: None)
    first = s.prepare(**args, out=tmp_path / "first")
    original = {str(p): p.read_bytes() for p in args["fetch_dir"].rglob("*") if p.is_file()}
    assert len(calls) == 2
    s.prepare(**args, out=tmp_path / "same")
    assert len(calls) == 2
    if change == "area":
        args["area"] = [40, -121, 41, -120]
    elif change == "date":
        args["date"] = "2023-01-02T00:00:00Z"
    elif change == "store":
        table["rows"]["era5-arco"]["access"]["store"] += "-other"
    elif change == "variable":
        table["rows"]["era5-arco"]["state"]["soil_water"]["layers"][0]["variable"] += "_new"
    else:
        (args["fetch_dir"] / "state.nc").unlink()
        original.pop(str(args["fetch_dir"] / "state.nc"))
    second = s.prepare(**args, out=tmp_path / "second")
    assert len(calls) == 4
    assert second["fetch"]["area_south_west_north_east"] == args["area"]
    assert stages[-4][1]["grid_file"] != str(args["fetch_dir"] / "state.nc")
    for path, contents in original.items():
        assert Path(path).read_bytes() == contents
    s.prepare(**args, out=tmp_path / "second-repeated")
    assert len(calls) == 4
    assert first["fetch"]["area_south_west_north_east"] == [38, -123, 39, -122]


@pytest.mark.parametrize("damage", ["same_size", "truncated", "missing_size", "missing_digest",
                                   "wrong_size", "wrong_digest", "invalid_size", "duplicate_slice", "legacy"])
def test_fetch_payload_integrity_recovery(tmp_path, monkeypatch, damage):
    """Do not pass changed sealed inputs to native stages, or rewrite old inputs."""
    calls, mapped = [], []
    payload = b"opaque transport fixture; not NetCDF"
    monkeypatch.setattr(s, "_rw_zarr", lambda: Path("rw_zarr"))

    def extract(argv, **kwargs):
        calls.append(argv)
        Path(argv[3]).write_bytes(payload)
        return SimpleNamespace(returncode=0, stderr="")

    def stage(command, request, workdir):
        if command == "weights":
            mapped.append(Path(request["grid_file"]))
        return {}

    monkeypatch.setattr(s.subprocess, "run", extract)
    monkeypatch.setattr(s, "run_stage", stage)
    cache = tmp_path / "source"
    args = dict(fabric=tmp_path / "fabric.gpkg", date="2023-01-01T00:00:00Z",
                area=[38, -123, 39, -122], fetch_dir=cache, progress=lambda _: None)
    s.prepare(**args, out=tmp_path / "first")
    receipt_path = cache / "fetch-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    part = receipt["slices"][0]
    if damage == "same_size":
        (cache / "state.nc").write_bytes(b"X" + payload[1:])
    elif damage == "truncated":
        (cache / "state.nc").write_bytes(payload[:-1])
    elif damage == "missing_size":
        part.pop("size_bytes", None)
        part["sha256"] = hashlib.sha256(payload).hexdigest()
    elif damage == "missing_digest":
        part.pop("sha256", None)
        part["size_bytes"] = len(payload)
    elif damage == "wrong_size":
        part["size_bytes"] = len(payload) + 1
    elif damage == "wrong_digest":
        part["sha256"] = "0" * 64
    elif damage == "invalid_size":
        part["size_bytes"] = str(len(payload))
    elif damage == "duplicate_slice":
        receipt["slices"].append(dict(part))
    else:
        part.pop("size_bytes", None)
        part.pop("sha256", None)
    if damage not in ("same_size", "truncated"):
        receipt_path.write_text(json.dumps(receipt))
    preserved = {p: p.read_bytes() for p in cache.iterdir() if p.is_file()}
    s.prepare(**args, out=tmp_path / "second")
    assert len(calls) == (1 if damage == "legacy" else 2)
    assert (mapped[-1] == cache / "state.nc") == (damage == "legacy")
    assert mapped[-1].read_bytes() == payload
    assert all(p.read_bytes() == contents for p, contents in preserved.items())
    s.prepare(**args, out=tmp_path / "third")
    assert len(calls) == (1 if damage == "legacy" else 2)


def test_fetch_receipt_records_actual_payload_bytes(tmp_path, monkeypatch):
    payload = b"opaque transport fixture; not NetCDF"
    monkeypatch.setattr(s, "_rw_zarr", lambda: Path("rw_zarr"))

    def extract(argv, **kwargs):
        Path(argv[3]).write_bytes(payload)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(s.subprocess, "run", extract)
    receipt = s.fetch(s.admit(s.load_table(), "era5-arco"), area=[38, -123, 39, -122],
                      state_times=["2023-01-01T00:00:00Z"], forcing_times=[],
                      out=tmp_path / "source", progress=lambda _: None)
    part = receipt["slices"][0]
    assert part["size_bytes"] == len(payload)
    assert part["sha256"] == hashlib.sha256(payload).hexdigest()
