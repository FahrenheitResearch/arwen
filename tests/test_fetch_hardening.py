"""The audited lying states of the fetch/cache state machines.

Each gate names the state it closes, reproduces the fault that reached
it (an injected transport failure, an injected wait timeout, a real
second process holding the output), and then asserts the property the
directory must have afterwards: either it resumes truthfully, or it
refuses honestly, or the suspect bytes were set aside -- never a
readable receipt describing bytes that are not there.

Nothing here touches the network.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time

import pytest

import gpuwm.fetch as fetch
from gpuwm import fetch_guard, geog_assets, table_assets
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport

_AREA = "30,-100,40,-90"
_CYCLE = datetime(2026, 7, 28, 6)
_HRRR_CYCLE = datetime(2026, 7, 28, 5)


@pytest.fixture(autouse=True)
def isolated_lock_root(tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()
    yield
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()


def _grib2(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _gfs_download(url: str, destination: Path, **kwargs) -> None:
    destination.write_bytes(_grib2(fetch.GFS_SUBSET_RECORD_COUNT))


def _hrrr_product(request, *, workers, retries, expected_count=-1):
    request.destination.write_bytes(_grib2(
        hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
        else hrrr_transport.ATMOSPHERE_RECORD_COUNT))
    request.index_path.write_text("1:0:fixture\n", encoding="ascii")
    return {"kind": request.kind}


def _gfs(out: Path, hours: tuple[int, ...], **kwargs):
    return fetch.fetch_gfs(
        cycle=_CYCLE, hours=hours, area=fetch.parse_area(_AREA), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kw: fetch.GFS_SUBSET_RECORD_COUNT,
        **kwargs)


def _manifest(out: Path) -> dict:
    return json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())


# ---------------------------------------------------------------------------
# LS-1 -- an interrupted --force-refetch left a manifest claiming bytes
#         it had already replaced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fail_on_hour", [0, 3])
def test_interrupted_force_leaves_no_receipt_claiming_replaced_bytes(
        tmp_path, monkeypatch, fail_on_hour):
    out = tmp_path / "gfs"
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    _gfs(out, (0, 3, 6))
    original = _manifest(out)
    claimed = {item["name"]: item["sha256"] for item in original["files"]}
    # Three payloads, the series, and the three front-door receipts the
    # route writes beside them: the manifest claims every canonical file
    # in the directory, so every one of them has to survive a force.
    assert set(claimed) == {
        "gfs.t06z.pgrb2.0p25.f000.subset.grib2",
        "gfs.t06z.pgrb2.0p25.f003.subset.grib2",
        "gfs.t06z.pgrb2.0p25.f006.subset.grib2",
        "gfs-series.tsv", "SHA256SUMS", "inputs.txt", "prep-command.txt"}

    def fail_partway(url, destination, **kwargs):
        if f"f{fail_on_hour:03d}" in url:
            raise RuntimeError("injected transport failure mid-force")
        _gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", fail_partway)
    with pytest.raises(RuntimeError, match="injected transport failure"):
        _gfs(out, (0, 3, 6), force=True)

    # THE LIE, if it were still there: fetch-manifest.json readable, its
    # digests describing payloads that force has moved or replaced.
    surviving = out / fetch.FETCH_MANIFEST_NAME
    if surviving.is_file():
        for item in json.loads(surviving.read_text())["files"]:
            path = out / item["name"]
            assert path.is_file(), item["name"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == \
                item["sha256"], item["name"]

    # Every original payload is still on disk under a quarantine name --
    # nothing was deleted.
    for name, digest in claimed.items():
        aside = list(out.glob(f"{name}.rejected-*"))
        assert len(aside) == 1, name
        assert hashlib.sha256(aside[0].read_bytes()).hexdigest() == digest

    # And an ordinary re-run is refused honestly rather than resumed.
    if surviving.is_file():
        fetch.check_prior_request(out, source="gfs", cycle=_CYCLE,
                                  area=fetch.parse_area(_AREA))
    else:
        with pytest.raises(ValueError, match="carries no readable"):
            fetch.check_prior_request(out, source="gfs", cycle=_CYCLE,
                                      area=fetch.parse_area(_AREA))

    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    repaired = _manifest(Path(str(_gfs(out, (0, 3, 6), force=True).parent)))
    assert repaired["forecast_hours"] == [0, 3, 6]


def test_force_sweeps_receipts_before_any_payload(tmp_path, monkeypatch):
    """The ORDER is the property: no payload moves while a receipt lives."""

    out = tmp_path / "gfs"
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    _gfs(out, (0, 3))

    order: list[str] = []
    real_quarantine = fetch_guard.quarantine

    def record(path, *, tag="rejected"):
        order.append(path.name)
        return real_quarantine(path, tag=tag)

    monkeypatch.setattr(fetch_guard, "quarantine", record)
    _gfs(out, (0, 3), force=True)

    # The manifest first -- while it is canonical the directory still
    # presents itself as a completed fetch -- then the rest of the front
    # door, every file of which names payloads, then the series, and
    # only then a payload.
    receipts = len(fetch.FETCH_RECEIPT_NAMES)
    assert order[0] == fetch.FETCH_MANIFEST_NAME
    assert set(order[:receipts]) == set(fetch.FETCH_RECEIPT_NAMES)
    assert order[receipts] == "gfs-series.tsv"
    assert set(order[receipts + 1:]) == {
        "gfs.t06z.pgrb2.0p25.f000.subset.grib2",
        "gfs.t06z.pgrb2.0p25.f003.subset.grib2"}


# ---------------------------------------------------------------------------
# LS-2 -- a shorter forced refetch left old forecast hours canonical and
#         unlisted, contradicting the documented force contract
# ---------------------------------------------------------------------------

def test_a_shorter_force_leaves_no_unlisted_canonical_payload(tmp_path,
                                                              monkeypatch):
    out = tmp_path / "gfs"
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    _gfs(out, (0, 3, 6))
    out_of_window = out / "gfs.t06z.pgrb2.0p25.f006.subset.grib2"
    stale_sidecar = out / "gfs.t06z.pgrb2.0p25.f006.subset.grib2.idx"
    stale_sidecar.write_text("stale index\n", encoding="ascii")
    kept = out_of_window.read_bytes()

    _gfs(out, (0, 3), force=True)

    manifest = _manifest(out)
    assert manifest["forecast_hours"] == [0, 3]
    listed = {item["name"] for item in manifest["files"]}
    # The out-of-window payload is gone from the canonical name AND from
    # the receipt, and its bytes survive under quarantine.
    assert out_of_window.name not in listed
    assert not out_of_window.exists()
    assert not stale_sidecar.exists()
    aside = list(out.glob(f"{out_of_window.name}.rejected-*"))
    assert len(aside) == 1 and aside[0].read_bytes() == kept
    assert list(out.glob(f"{stale_sidecar.name}.rejected-*"))
    # Every canonical file left in the directory is one the receipt claims.
    canonical = {path.name for path in out.iterdir()
                 if path.is_file() and ".rejected-" not in path.name}
    assert canonical == listed | {fetch.FETCH_MANIFEST_NAME}


def test_force_leaves_earlier_quarantine_evidence_alone(tmp_path,
                                                        monkeypatch):
    out = tmp_path / "gfs"
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    _gfs(out, (0, 3))
    evidence = out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2.rejected-1"
    evidence.write_bytes(b"older evidence")

    _gfs(out, (0, 3), force=True)
    assert evidence.read_bytes() == b"older evidence"


# ---------------------------------------------------------------------------
# LS-3 -- two writers into one output directory
# ---------------------------------------------------------------------------

_HOLDER = """
    import sys, time
    from pathlib import Path
    from gpuwm import fetch_guard

    target, flag = sys.argv[1], Path(sys.argv[2])
    release = flag.with_suffix(".release")
    with fetch_guard.hold("fetch-out", target, timeout_s=30):
        flag.write_text("held", encoding="utf-8")
        deadline = time.monotonic() + 60
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
"""


def _spawn_holder(out: Path, flag: Path):
    child = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_HOLDER), str(out), str(flag)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ),
        cwd=str(Path(__file__).resolve().parents[1]))
    deadline = time.monotonic() + 60
    while not flag.exists() and time.monotonic() < deadline:
        assert child.poll() is None, child.communicate()
        time.sleep(0.05)
    assert flag.exists(), "the holder never took the lock"
    return child


def _fetch(source: str, out: Path):
    if source == "gfs":
        return fetch.fetch_gfs(
            cycle=_CYCLE, hours=(0, 3), area=fetch.parse_area(_AREA),
            out=out, progress=lambda line: None,
            derived_bar=lambda cycle, **kw: fetch.GFS_SUBSET_RECORD_COUNT)
    return fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=(0,), area=None,
                            out=out, progress=lambda line: None)


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_a_second_writer_refuses_instead_of_interleaving(tmp_path,
                                                         monkeypatch, source):
    out = tmp_path / "out"
    flag = tmp_path / "held.flag"
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, "0")
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    monkeypatch.setattr(hrrr_transport, "_download_product", _hrrr_product)
    child = _spawn_holder(out, flag)
    try:
        with pytest.raises(fetch_guard.FetchLockBusy) as caught:
            _fetch(source, out)
        assert "refuses rather than interleave" in str(caught.value)
        # It refused BEFORE writing anything into the directory.
        assert not out.exists() or list(out.iterdir()) == []
    finally:
        flag.with_suffix(".release").write_text("go", encoding="utf-8")
        child.wait(timeout=60)


def test_a_second_writer_may_queue_instead_of_refusing(tmp_path,
                                                       monkeypatch):
    """The other half of the contract: the loser waits, then proceeds."""

    out = tmp_path / "out"
    flag = tmp_path / "held.flag"
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, "30")
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    child = _spawn_holder(out, flag)
    flag.with_suffix(".release").write_text("go", encoding="utf-8")
    try:
        manifest = _fetch("gfs", out)
    finally:
        child.wait(timeout=60)
    assert json.loads(manifest.read_text())["forecast_hours"] == [0, 3]


def test_the_front_door_guard_and_the_transfer_share_one_lock(monkeypatch,
                                                              tmp_path):
    """The guard authorises the transfer; a gap between them is the race."""

    import inspect
    source = inspect.getsource(fetch.fetch_main)
    guard_index = source.index("check_prior_request(args.out, source=source")
    hold_index = source.rindex('fetch_guard.hold("fetch-out", args.out)',
                               0, guard_index)
    assert source.index("manifest = fetch_gfs(") > hold_index


# ---------------------------------------------------------------------------
# LS-4 -- the HRRR wait-timeout receipt claimed a half-fetched hour
# ---------------------------------------------------------------------------

class _LiveCycle:
    """A publication schedule plus a clock that only the sleeper moves."""

    def __init__(self, published: set[tuple[str, int]]):
        self.published = published
        self.now = 0.0

    def probe(self, url: str) -> bool:
        for product, hour in self.published:
            if f"{product}f{hour:02d}.grib2" in url:
                return True
        return False

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def clock(self) -> float:
        return self.now


def test_a_timed_out_hour_is_not_claimed_by_files_or_sha256sums(
        tmp_path, monkeypatch):
    """The lie: forecast_hours [0] beside an hour-1 file in SHA256SUMS."""

    # Hour 1's atmosphere publishes; its soil never does.
    live = _LiveCycle({("wrfnat", 0), ("wrfprs", 0), ("wrfnat", 1)})
    monkeypatch.setattr(hrrr_transport, "_download_product", _hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=_HRRR_CYCLE, hours=(0, 1), area=None, out=out,
                  transport="auto", wait=True, probe=live.probe,
                  sleeper=live.sleep, clock=live.clock)
    with pytest.raises(RuntimeError, match="timed out"):
        fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                         progress=lambda line: None)

    partial = out / "hrrr.t05z.wrfnatf01.grib2"
    assert partial.is_file(), "the fetched hour-1 atmosphere was lost"

    manifest = _manifest(out)
    assert manifest["forecast_hours"] == [0]
    names = [item["name"] for item in manifest["files"]]
    assert names == ["hrrr.t05z.wrfnatf00.grib2", "hrrr.t05z.soilf00.grib2",
                     "SHA256SUMS"]
    sums = (out / "SHA256SUMS").read_text()
    assert partial.name not in sums
    assert len(sums.splitlines()) == 2
    # Every claimed name belongs to a complete hour, and every claim is true.
    for item in manifest["files"]:
        assert hashlib.sha256((out / item["name"]).read_bytes()).hexdigest() \
            == item["sha256"]

    # The unclaimed hour-1 atmosphere is still resumable: the next run
    # re-verifies it under the ordinary bars instead of re-downloading.
    live.published.add(("wrfprs", 1))
    downloaded: list[str] = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _hrrr_product(request, workers=workers, retries=retries,
                             expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    final = fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                             progress=lambda line: None)
    assert downloaded == ["hrrr.t05z.soilf01.grib2"]
    assert json.loads(final.read_text())["forecast_hours"] == [0, 1]


@pytest.mark.parametrize("hours", [(0, 1), (0, 1, 2)])
def test_hrrr_checkpoints_after_every_completed_hour(tmp_path, monkeypatch,
                                                     hours):
    """F-4: an ordinary kill after good hours left no receipt at all."""

    monkeypatch.setattr(hrrr_transport, "_download_product", _hrrr_product)
    published: list[list[int]] = []
    real_write = fetch.write_fetch_manifest

    def record(out, payload):
        published.append(list(payload["forecast_hours"]))
        return real_write(out, payload)

    monkeypatch.setattr(fetch, "write_fetch_manifest", record)
    fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=hours, area=None,
                     out=tmp_path / "hrrr", progress=lambda line: None)

    prefixes = [list(hours[:index + 1]) for index in range(len(hours))]
    assert published == prefixes + [list(hours)]


def test_a_kill_after_a_complete_hour_leaves_a_usable_receipt(tmp_path,
                                                              monkeypatch):
    out = tmp_path / "hrrr"

    def die_on_hour_one(request, *, workers, retries, expected_count=-1):
        if "f01" in request.destination.name:
            raise RuntimeError("injected kill after hour 0")
        return _hrrr_product(request, workers=workers, retries=retries,
                             expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", die_on_hour_one)
    with pytest.raises(RuntimeError, match="injected kill"):
        fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=(0, 1), area=None,
                         out=out, progress=lambda line: None)

    manifest = _manifest(out)
    assert manifest["forecast_hours"] == [0]
    fetch.check_prior_request(out, source="hrrr", cycle=_HRRR_CYCLE,
                              area=None)

    monkeypatch.setattr(hrrr_transport, "_download_product", _hrrr_product)
    resumed = fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=(0, 1), area=None,
                               out=out, progress=lambda line: None)
    assert json.loads(resumed.read_text())["forecast_hours"] == [0, 1]


# ---------------------------------------------------------------------------
# LS-5 -- a stale canonical `.part` no run could resume, overwrite or
#         quarantine
# ---------------------------------------------------------------------------

def test_a_stale_canonical_part_never_blocks_the_next_run(tmp_path,
                                                          monkeypatch):
    out = tmp_path / "hrrr"
    out.mkdir()
    dest = out / "hrrr.t05z.wrfnatf00.grib2"
    stale = dest.with_suffix(dest.suffix + ".part")
    stale.write_bytes(b"a killed run's half-written subset")

    payload = _grib2(hrrr_transport.ATMOSPHERE_RECORD_COUNT)
    ranges = [(0, len(payload) - 1)]

    monkeypatch.setattr(hrrr_transport, "_head", lambda url: {
        "content-length": str(len(payload)), "accept-ranges": "bytes"})
    monkeypatch.setattr(hrrr_transport, "_request_bytes",
                        lambda url: (b"1:0:d=x:PRES:1 hybrid level:\n", {}))
    monkeypatch.setattr(hrrr_transport, "_parse_index",
                        lambda body, size: (hrrr_transport.IndexRow(
                            1, 0, "PRES", "1 hybrid level", "1:0:x"),))
    monkeypatch.setattr(hrrr_transport, "_atmosphere_selection",
                        lambda rows, expected_count=None: (0,))
    monkeypatch.setattr(hrrr_transport, "_coalesce",
                        lambda rows, selected, size: (
                            hrrr_transport.ByteRange(*ranges[0], "x", "x"),))

    def fake_range(url, byte_range, path, retries):
        path.write_bytes(payload)

    monkeypatch.setattr(hrrr_transport, "_download_range", fake_range)

    hrrr_transport._download_subset(
        url="https://example.invalid/x", index_url="https://example.invalid/x.idx",
        index_path=out / "x.idx", destination=dest, kind="atmosphere",
        workers=2, retries=1, expected_count=None)

    assert dest.read_bytes() == payload
    # The stale part was never opened, never overwritten, never deleted.
    assert stale.read_bytes() == b"a killed run's half-written subset"
    # And no NEW canonical part was created for the next run to trip on.
    assert sorted(path.name for path in out.iterdir()) == [
        dest.name, stale.name, "x.idx"]


def test_a_stale_part_is_swept_by_force(tmp_path, monkeypatch):
    out = tmp_path / "hrrr"
    monkeypatch.setattr(hrrr_transport, "_download_product", _hrrr_product)
    fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=(0,), area=None, out=out,
                     progress=lambda line: None)
    stale = out / "hrrr.t05z.wrfnatf00.grib2.part"
    stale.write_bytes(b"legacy partial")

    fetch.fetch_hrrr(cycle=_HRRR_CYCLE, hours=(0,), area=None, out=out,
                     force=True, progress=lambda line: None)
    assert not stale.exists()
    aside = list(out.glob(f"{stale.name}.rejected-*"))
    assert len(aside) == 1 and aside[0].read_bytes() == b"legacy partial"


# ---------------------------------------------------------------------------
# LS-6 -- the Rust soil handoff orphan no run could select
# ---------------------------------------------------------------------------

def test_an_orphaned_pressure_file_is_set_aside_before_the_rust_run(
        tmp_path, monkeypatch):
    """A kill between the backbone's rename and Python's own left a
    canonical `wrfprs` orphan: the next backbone run refused because its
    destination existed, and force targeted only the `soil` name."""

    out = tmp_path / "hrrr"
    out.mkdir()
    orphan = out / "hrrr.t05z.wrfprsf00.grib2"
    orphan.write_bytes(b"an orphan from a killed handoff")
    dest = out / "hrrr.t05z.soilf00.grib2"
    payload = _grib2(hrrr_transport.SOIL_RECORD_COUNT)

    def fake_backbone(*, binary, cycle, hour, kind, host, mode, out,
                      cache_dir, progress):
        landing = out / "hrrr.t05z.wrfprsf00.grib2"
        if landing.exists():
            raise AssertionError(
                "the backbone would refuse: its destination already exists")
        landing.write_bytes(payload)
        return {"name": landing.name, "bytes": len(payload),
                "wall_seconds": 0.1, "source": "s3", "mode": "idx-subset",
                "mode_reason": "test", "grib_url": "https://example.invalid/x",
                "selected_record_count": hrrr_transport.SOIL_RECORD_COUNT,
                "idx_name": None, "probe": {}}

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake_backbone)
    bar, url = fetch._download_one_hrrr_product(
        engine="rust", engine_bin=Path("rw_fetch"), mode="auto",
        cycle=_HRRR_CYCLE, hour=0, kind="soil", host="s3",
        url="https://example.invalid/x", dest=dest, dest_name=dest.name,
        source_name=orphan.name, out=out, label="f00 soil", cache_dir=None,
        workers=1, retries=1, bar_kind="hrrr-soil",
        certified=hrrr_transport.SOIL_RECORD_COUNT,
        accept_inventory_change=False, progress=lambda line: None)

    assert dest.read_bytes() == payload
    aside = list(out.glob(f"{orphan.name}.rejected-*"))
    assert len(aside) == 1
    assert aside[0].read_bytes() == b"an orphan from a killed handoff"
    assert bar.expected == hrrr_transport.SOIL_RECORD_COUNT


# ---------------------------------------------------------------------------
# LS-13 -- the shared table staging temp
# ---------------------------------------------------------------------------

def _table_asset(root: Path, payload: bytes):
    from gpuwm.core.thompson_contract import TableAsset
    return TableAsset(filename="externalized_table.dat", bytes=len(payload),
                      sha256=hashlib.sha256(payload).hexdigest())


def test_table_install_is_bound_to_the_bytes_it_verified(tmp_path):
    """A second writer owning the fixed temp used to install bytes that
    failed the pin, under the first writer's successful check."""

    root = tmp_path / "tables"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    payload = b"pinned table bytes" * 16
    asset = _table_asset(root, payload)
    (source / asset.filename).write_bytes(payload)

    # The legacy fixed staging name, owned by "another writer".
    legacy = root / f".{asset.filename}.fetch-partial"
    legacy.write_bytes(b"evil")

    installed = table_assets.fetch_asset_from_dir(root, asset, source)
    assert installed.read_bytes() == payload
    assert legacy.read_bytes() == b"evil"   # untouched, and never installed


def test_table_staging_names_are_unique_per_writer(tmp_path):
    root = tmp_path / "tables"
    asset = _table_asset(root, b"x")
    names = {table_assets._staging_path(root, asset).name for _ in range(50)}
    assert len(names) == 50
    assert all(str(os.getpid()) in name for name in names)


# ---------------------------------------------------------------------------
# Geog: LS-9, LS-10, LS-11, LS-12
# ---------------------------------------------------------------------------

_INDEX = """\
type = continuous
projection = regular_ll
dx = 0.5
dy = 0.5
known_x = 1.0
known_y = 1.0
known_lat = -89.75
known_lon = -179.75
wordsize = 2
tile_x = 4
tile_y = 4
tile_z = 1
"""


def _geog_archive(datasets: tuple[str, ...]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        for name in datasets:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
            tile = b"\x00\x01" * 16
            data = tarfile.TarInfo(f"{name}/00001-00004.00001-00004")
            data.size = len(tile)
            tar.addfile(data, io.BytesIO(tile))
            payload = _INDEX.encode()
            index = tarfile.TarInfo(f"{name}/index")
            index.size = len(payload)
            tar.addfile(index, io.BytesIO(payload))
    return buffer.getvalue()


def _geog_pin(dataset: str, payload: bytes):
    return geog_assets.GeogArchive(
        dataset, f"{dataset}.tar.bz2", len(payload),
        hashlib.sha256(payload).hexdigest(), 4096)


class _GeogResponse:
    def __init__(self, payload: bytes, status: int = 200, headers=None):
        self._data = io.BytesIO(payload)
        self.status = status
        self.headers = dict(headers or {})

    def read(self, n: int = -1) -> bytes:
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _geog_transport(files: dict[str, bytes], *, fail: set[str] = frozenset()):
    calls: list = []

    def urlopen_fn(request):
        calls.append(request)
        name = request.full_url.rsplit("/", 1)[1]
        if name in fail:
            raise OSError("injected download failure")
        payload = files[name]
        header = request.headers.get("Range")
        if header:
            offset = int(header.split("=", 1)[1].rstrip("-"))
            return _GeogResponse(payload[offset:], status=206, headers={
                "ETag": '"v1"',
                "Content-Range":
                    f"bytes {offset}-{len(payload) - 1}/{len(payload)}"})
        return _GeogResponse(payload, headers={"ETag": '"v1"'})

    urlopen_fn.calls = calls
    return urlopen_fn


@pytest.mark.parametrize("poison", ["truncate", "delete", "add"])
def test_a_poisoned_tile_tree_is_no_longer_classified_valid(tmp_path,
                                                            monkeypatch,
                                                            poison):
    """LS-9: a corrupt tile beside an intact index was accepted forever."""

    payload = _geog_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_geog_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    transport = _geog_transport({"alpha_ds.tar.bz2": payload})
    assert geog_assets.fetch_geog(
        root=root, datasets=("alpha_ds",), source="hf",
        progress=lambda *_: None, urlopen_fn=transport) == 1
    ok, detail = geog_assets.validate_dataset_dir(root, "alpha_ds")
    assert ok, detail

    tile = root / "alpha_ds" / "00001-00004.00001-00004"
    if poison == "truncate":
        tile.write_bytes(tile.read_bytes()[:4])
    elif poison == "delete":
        tile.unlink()
    else:
        (root / "alpha_ds" / "99999-99999.99999-99999").write_bytes(b"x" * 8)

    ok, detail = geog_assets.validate_dataset_dir(root, "alpha_ds")
    assert not ok, detail
    assert "no longer matches" in detail
    # And fetch does real work again rather than returning zero.
    assert geog_assets.fetch_geog(
        root=root, datasets=("alpha_ds",), source="hf",
        progress=lambda *_: None, urlopen_fn=transport) == 1
    assert geog_assets.validate_dataset_dir(root, "alpha_ds")[0]


def test_an_install_without_a_receipt_still_validates_index_only(tmp_path,
                                                                 monkeypatch):
    """Pre-1.1.3 installs carry no corpus receipt and must not be broken."""

    payload = _geog_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_geog_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    geog_assets.fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                           progress=lambda *_: None,
                           urlopen_fn=_geog_transport(
                               {"alpha_ds.tar.bz2": payload}))
    (root / geog_assets.GEOG_FETCH_MANIFEST_NAME).unlink()
    ok, detail = geog_assets.validate_dataset_dir(root, "alpha_ds")
    assert ok and "no local" in detail


def test_provenance_is_published_before_the_archive_is_removed(tmp_path,
                                                               monkeypatch):
    """LS-10: a kill after install lost the only record of the source."""

    first = _geog_archive(("alpha_ds",))
    second = _geog_archive(("beta_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (
        _geog_pin("alpha_ds", first), _geog_pin("beta_ds", second)))
    root = tmp_path / "WPS_GEOG"
    transport = _geog_transport(
        {"alpha_ds.tar.bz2": first, "beta_ds.tar.bz2": second},
        fail={"beta_ds.tar.bz2"})

    with pytest.raises(geog_assets.GeogFetchError):
        geog_assets.fetch_geog(
            root=root, datasets=("alpha_ds", "beta_ds"), source="hf",
            progress=lambda *_: None, urlopen_fn=transport)

    manifest = json.loads(
        (root / geog_assets.GEOG_FETCH_MANIFEST_NAME).read_text())
    entry = manifest["archives"]["alpha_ds.tar.bz2"]
    assert entry["archive_sha256"] == hashlib.sha256(first).hexdigest()
    assert entry["datasets"]["alpha_ds"]["files"] == 2
    assert geog_assets.installed_receipt(root, "alpha_ds") is not None


def test_a_concurrent_fetchs_entries_are_merged_not_dropped(tmp_path,
                                                            monkeypatch):
    """LS-11: last writer wins used to delete the other run's datasets.

    The injection matters: the competing entry has to appear *after*
    this run has already read the manifest, which is exactly when a
    start-of-run read plus an end-of-run replace loses it.
    """

    payload = _geog_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_geog_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    other = {"url": "https://example.invalid/other_ds.tar.bz2",
             "source": "hf", "archive_bytes": 1, "archive_sha256": "0" * 64,
             "pinned": True,
             "datasets": {"other_ds": {"files": 2, "bytes": 40}},
             "fetched_at": "2026-07-30T00:00:00Z"}

    real_extract = geog_assets.extract_datasets

    def extract_then_let_the_other_writer_finish(*args, **kwargs):
        counts = real_extract(*args, **kwargs)
        manifest = geog_assets._load_manifest(root)
        manifest["archives"]["other_ds.tar.bz2"] = other
        geog_assets._write_manifest(root, manifest)
        return counts

    monkeypatch.setattr(geog_assets, "extract_datasets",
                        extract_then_let_the_other_writer_finish)
    geog_assets.fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                           progress=lambda *_: None,
                           urlopen_fn=_geog_transport(
                               {"alpha_ds.tar.bz2": payload}))

    archives = json.loads(
        (root / geog_assets.GEOG_FETCH_MANIFEST_NAME).read_text())["archives"]
    assert set(archives) == {"other_ds.tar.bz2", "alpha_ds.tar.bz2"}


# ---------------------------------------------------------------------------
# LS-14 -- a quarantine name collision overwrote older evidence
# ---------------------------------------------------------------------------

def test_quarantining_twice_in_one_tick_keeps_both_artifacts(tmp_path,
                                                             monkeypatch):
    """``.rejected-<time_ns>`` collides; ``os.replace`` then destroys."""

    monkeypatch.setattr(time, "time_ns", lambda: 1_700_000_000_000_000_000)
    out = tmp_path / "out"
    out.mkdir()
    kept = []
    for generation in (b"first rejected payload", b"second rejected payload"):
        victim = out / "hrrr.t05z.wrfnatf00.grib2"
        victim.write_bytes(generation)
        fetch._quarantine_rejected(victim, lambda line: None, "hrrr f00")
        kept.append(generation)

    aside = sorted(out.glob("hrrr.t05z.wrfnatf00.grib2.rejected-*"))
    assert len(aside) == 2
    assert {path.read_bytes() for path in aside} == set(kept)


def test_a_partial_bound_to_another_object_is_not_appended_to(tmp_path,
                                                              monkeypatch):
    """LS-12: resume used the byte count alone, so two generations of an
    archive could be stitched into one file."""

    payload = _geog_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_geog_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    archive_dir = root / geog_assets.ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True)
    partial = archive_dir / "alpha_ds.tar.bz2"
    partial.write_bytes(b"A" * 100)
    geog_assets._resume_sidecar(partial).write_text(json.dumps({
        "url": "https://example.invalid/some_other_object.tar.bz2",
        "etag": '"other"', "last_modified": None}), encoding="utf-8")

    transport = _geog_transport({"alpha_ds.tar.bz2": payload})
    lines: list[str] = []
    assert geog_assets.fetch_geog(
        root=root, datasets=("alpha_ds",), source="hf",
        progress=lines.append, urlopen_fn=transport) == 1

    assert any("not a recorded prefix" in line for line in lines)
    (request,) = transport.calls
    assert request.headers.get("Range") is None   # started from zero
    aside = list(archive_dir.glob("alpha_ds.tar.bz2.rejected-*"))
    assert len(aside) == 1 and aside[0].read_bytes() == b"A" * 100


def test_a_resume_that_lands_at_the_wrong_offset_is_refused(tmp_path,
                                                            monkeypatch):
    payload = _geog_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_geog_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    archive_dir = root / geog_assets.ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True)
    partial = archive_dir / "alpha_ds.tar.bz2"
    partial.write_bytes(payload[:100])
    url = geog_assets.archive_url("alpha_ds.tar.bz2", "hf")
    geog_assets._resume_sidecar(partial).write_text(json.dumps({
        "url": url, "etag": '"v1"', "last_modified": None}), encoding="utf-8")

    def lying_range(request):
        # A 206 that starts somewhere other than where we asked.
        return _GeogResponse(payload[500:], status=206, headers={
            "Content-Range": f"bytes 500-{len(payload) - 1}/{len(payload)}"})

    with pytest.raises(geog_assets.GeogFetchError,
                       match="does not start there"):
        geog_assets.fetch_geog(root=root, datasets=("alpha_ds",),
                               source="hf", progress=lambda *_: None,
                               urlopen_fn=lying_range)
    assert partial.read_bytes() == payload[:100]   # untouched


def test_an_unbound_partial_restarts_only_where_there_is_no_pin(tmp_path,
                                                               monkeypatch):
    """Two values of the dimension that decides it: pinned vs drift."""

    payload = _geog_archive(("alpha_ds",))
    url = geog_assets.archive_url("alpha_ds.tar.bz2", "ncar")

    for allow_drift, expect_range in ((False, "bytes=100-"), (True, None)):
        root = tmp_path / f"WPS_GEOG-{allow_drift}"
        archive_dir = root / geog_assets.ARCHIVE_SUBDIR
        archive_dir.mkdir(parents=True)
        partial = archive_dir / "alpha_ds.tar.bz2"
        partial.write_bytes(payload[:100])
        transport = _geog_transport({"alpha_ds.tar.bz2": payload})
        geog_assets.download_archive(
            url, partial, expected_bytes=len(payload),
            progress=lambda *_: None, label="alpha_ds",
            strict_size=not allow_drift, urlopen_fn=transport)
        (request,) = transport.calls
        assert request.headers.get("Range") == expect_range, allow_drift


def test_the_resume_record_is_removed_once_the_archive_is_whole(tmp_path):
    payload = _geog_archive(("alpha_ds",))
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    dest = archive_dir / "alpha_ds.tar.bz2"
    geog_assets.download_archive(
        geog_assets.archive_url("alpha_ds.tar.bz2", "hf"), dest,
        expected_bytes=len(payload), progress=lambda *_: None,
        label="alpha_ds",
        urlopen_fn=_geog_transport({"alpha_ds.tar.bz2": payload}))
    assert dest.read_bytes() == payload
    assert not geog_assets._resume_sidecar(dest).exists()


# ---------------------------------------------------------------------------
# Publication leaves no fixed-name staging litter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hours", [(0, 3), (0, 3, 6)])
def test_no_fixed_tmp_name_is_ever_left_in_an_output(tmp_path, monkeypatch,
                                                     hours):
    out = tmp_path / "gfs"
    monkeypatch.setattr(gfs_transport, "_download", _gfs_download)
    _gfs(out, hours)
    assert not list(out.glob("*.tmp"))
    assert not list(out.glob("*.tmp.*"))
