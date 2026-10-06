"""A preparation's decode is budgeted against HOST RAM, never the card.

Named breakage: the 2026-10-04T12 and 2026-10-05T00 runs of a 240 h, 81-lead
GFS 0.25-degree template were SIGKILLed in their prepare stage on a 64 GiB
host beside a 96 GB card, after ``gpuwm go`` printed that they fit "the
93.93 GiB budget".  Their leads were whole-globe s3 objects (about 540 MB
each); one bridge run parsed all 81 (measured 46.4 GiB RSS) and the
preparation then held all 81 decoded leads as float64 (measured 85.8 GiB).

These tests run on the CPU with a fake host-memory reader; no card is used.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import gfs_direct, go_cli
from gpuwm.core import preflight, streaming
from gpuwm.ingest import host_decode_window as window_module
from gpuwm.ingest.host_decode_window import (
    DECODE_HOST_SHARE, GFS_FULL_FILE_LEAD_BYTES, batch_price, decode_window,
    gfs_decoded_lead_bytes)

GIB = 1024 ** 3
#: The measured whole-globe GFS lead: a 540 MB object decoding to 134
#: float32 fields (5 x 23 levels to 50 hPa + 19 surface) on 1440 x 721.
LEAD = GFS_FULL_FILE_LEAD_BYTES
DECODED = gfs_decoded_lead_bytes(5 * 23 + 19)


# -- the window ----------------------------------------------------------

def test_a_64_gib_host_decodes_a_240_h_whole_globe_series_in_windows():
    window = decode_window(leads=81, threads=8, lead_bytes=LEAD,
                           decoded_lead_bytes=DECODED, available=60 * GIB)
    assert 1 < window.leads < 81
    assert window.threads == 8
    assert window.batch_bytes <= DECODE_HOST_SHARE * 60 * GIB
    # One more lead would not fit the share.
    assert batch_price(window.leads + 1, 8, LEAD, DECODED) \
        > DECODE_HOST_SHARE * 60 * GIB
    # The unbounded batch is far over the share, even before the 81 float64
    # leads the preparation used to hold beside it (measured 85.8 GiB).
    assert batch_price(81, 8, LEAD, DECODED) > 1.5 * DECODE_HOST_SHARE * 60 * GIB


def test_a_small_host_narrows_threads_before_leads_and_never_below_one():
    small = decode_window(leads=81, threads=16, lead_bytes=LEAD,
                          decoded_lead_bytes=DECODED, available=12 * GIB)
    assert 1 <= small.threads < 16
    assert small.leads >= small.threads
    assert small.batch_bytes <= DECODE_HOST_SHARE * 12 * GIB
    tiny = decode_window(leads=81, threads=16, lead_bytes=LEAD,
                         decoded_lead_bytes=DECODED, available=1 * GIB)
    assert (tiny.leads, tiny.threads) == (1, 1)


def test_unreadable_host_ram_leaves_the_batch_as_asked():
    window = decode_window(leads=81, threads=8, lead_bytes=LEAD,
                           decoded_lead_bytes=DECODED, available=None)
    assert (window.leads, window.threads, window.budget_bytes) == (81, 8, None)
    assert "unreadable" in window.sentence(81)


def test_a_wide_host_or_a_crop_series_decodes_in_one_window():
    wide = decode_window(leads=81, threads=8, lead_bytes=LEAD,
                         decoded_lead_bytes=DECODED, available=400 * GIB)
    assert wide.leads == 81
    crop = decode_window(leads=81, threads=8, lead_bytes=4 * 1000 ** 2,
                         decoded_lead_bytes=16 * 1000 ** 2, available=30 * GIB)
    assert crop.leads == 81


# -- the planner: a host smaller than its card ---------------------------

_MEXICO_LIKE = """\
[experiment]
name = "host-smaller-than-card"
start_time = 2026-10-05T00:00:00
run_seconds = 864000.0
restart_interval_s = 0.0

[fetch]
source = "gfs"
cycle = "2026-10-05T00"
hours = 240
cadence = 3

[shared]
nz = 49
ztop = 20000.0
p_top = 5000.0
moist = true
moist_cq = true
mp_physics = 10
ra_lw_physics = 4
ra_sw_physics = 4
sf_sfclay_physics = 91
sf_surface_physics = 2
bl_pbl_physics = 1

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 158
ny = 158
time_step = 18
dx = 9000.0
history_interval_s = 3600.0
"""


@pytest.fixture()
def host_smaller_than_card(tmp_path, monkeypatch):
    """A 96 GB card (93.93 GiB free) on a 64 GiB host with 62 GiB available."""

    config = tmp_path / "host-smaller-than-card.toml"
    config.write_text(textwrap.dedent(_MEXICO_LIKE), encoding="utf-8")

    def probe(*_args, **_kwargs):
        return {"free_bytes": int(93.93 * GIB), "total_bytes": 96 * GIB,
                "name": "test card", "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", probe)
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(window_module, "available_host_bytes",
                        lambda: 62 * GIB)
    monkeypatch.setattr(window_module, "threads_available", lambda: 8)
    return config


def test_go_prices_the_preparation_on_a_host_smaller_than_the_card(
        host_smaller_than_card):
    plan = {"config": str(host_smaller_than_card), "source": "gfs",
            "cadence": 3}
    gate = go_cli.memory_gate(plan)
    verdict = gate["verdict"]
    # The card's number is named as the card's ...
    assert "GiB card budget" in verdict
    assert re.search(r"the [0-9.]+ GiB budget", verdict) is None
    # ... and the preparation is weighed against the host's own RAM.
    assert gate["host"] == {"available_bytes": 62 * GIB,
                            "total_bytes": 64 * GIB,
                            "budget_bytes": 62 * GIB}
    assert ("budgeted against this host's RAM, not the card: 62.00 GiB "
            "available of 64.00 GiB") in verdict
    window = gate["decode_window"]
    assert window is not None
    assert window.available_bytes == 62 * GIB
    assert 1 < window.leads < 81
    assert window.batch_bytes <= DECODE_HOST_SHARE * 62 * GIB
    assert f"decode window {window.leads} of 81 lead(s)" in verdict
    assert gate["decode_warning"] is None


def test_go_warns_a_host_that_cannot_hold_one_whole_globe_lead(
        host_smaller_than_card, monkeypatch):
    monkeypatch.setattr(window_module, "available_host_bytes",
                        lambda: 2 * GIB)
    gate = go_cli.memory_gate({"config": str(host_smaller_than_card),
                               "source": "gfs", "cadence": 3})
    assert gate["host"]["budget_bytes"] == 2 * GIB
    assert gate["decode_window"].leads == 1
    assert "more than the 2.00 GiB this host has available" \
        in gate["decode_warning"]


def test_the_assistant_still_reads_the_card_budget():
    from gpuwm.gui.assistant import plan

    match = plan._BUDGET.search("it fits the 93.93 GiB card budget with 1 GiB")
    assert match is not None and match.group(1) == "93.93"


# -- the decoded leads: read on access, a window held --------------------

def _decoded_tree(root: Path, hours, levels=2, ny=3, nx=4):
    for hour in hours:
        folder = root / f"f{hour:03d}"
        folder.mkdir(parents=True)
        for index, name in enumerate(gfs_direct._THREE_D):
            values = np.arange(levels * ny * nx, dtype="<f4") + 1000 * hour + index
            values.tofile(folder / f"{name}.f32le")
        for index, name in enumerate(gfs_direct._TWO_D):
            values = np.arange(ny * nx, dtype="<f4") + 1000 * hour - index
            values.tofile(folder / f"{name}.f32le")
    return gfs_direct._BridgeSnapshots(
        root=root, cycle=datetime(2026, 10, 5), hours=hours,
        levels_hpa=np.asarray([1000.0, 500.0]),
        latitude=np.asarray([10.0, 10.25, 10.5]),
        longitude=np.asarray([0.0, 0.25, 0.5, 0.75]), ny=ny, nx=nx)


def test_decoded_leads_are_read_on_access_and_only_a_window_is_held(tmp_path):
    hours = (0, 3, 6, 9, 12)
    series = _decoded_tree(tmp_path, hours)
    assert len(series) == 5
    assert series.valid_times == tuple(
        datetime(2026, 10, 5) + timedelta(hours=hour) for hour in hours)
    assert series._cache == {}  # nothing read for the times
    for index, hour in enumerate(hours):
        snapshot = series[index]
        expected = np.fromfile(tmp_path / f"f{hour:03d}" / "T.f32le",
                               dtype="<f4").reshape(2, 3, 4).astype(np.float64)
        assert np.array_equal(snapshot.fields["T"], expected)
        assert snapshot.valid_time == datetime(2026, 10, 5) + timedelta(hours=hour)
        assert len(series._cache) <= gfs_direct._SNAPSHOT_CACHE_LEADS
    # A re-read lead holds the same values as its first read.
    first = series[0]
    assert np.array_equal(first.fields["PSFC"], _decoded_tree(
        tmp_path / "again", (0,))[0].fields["PSFC"])
    # A tail slice and a transform stay lazy.
    tail = series[2:]
    assert isinstance(tail, gfs_direct._BridgeSnapshots)
    assert tail.hours == (6, 9, 12) and tail._cache == {}
    seen = []

    def mark(snapshot):
        seen.append(snapshot.valid_time)
        return snapshot

    marked = gfs_direct._map_snapshots(series, mark)
    assert seen == []
    marked[1]
    assert seen == [datetime(2026, 10, 5, 3)]
    # The grid alone reads no field.
    meta = series.metadata(4)
    assert dict(meta.fields) == {}
    assert np.array_equal(meta.longitude, series.longitude)
    assert gfs_direct._sequence_valid_times(series) == series.valid_times


# -- the whole-cycle decode in windows -----------------------------------

def test_a_whole_series_wider_than_the_window_decodes_in_merged_batches(
        tmp_path, monkeypatch):
    leads = tmp_path / "leads"
    leads.mkdir()
    records = []
    for hour in (0, 3, 6, 9):
        path = leads / f"gfs.t00z.pgrb2.0p25.f{hour:03d}"
        path.write_bytes(b"GRIB")
        records.append((hour, path.resolve()))
    series = tmp_path / "gfs-series.tsv"
    series.write_text("".join(f"{hour}\t{path.name}\t{81 if hour == 0 else 96}\n"
                              for hour, path in records).replace(
        "gfs.t00z", str(leads) + "/gfs.t00z"), encoding="utf-8")
    monkeypatch.setattr(gfs_direct, "_file_bytes", lambda _path: GIB)
    monkeypatch.setattr(window_module, "available_host_bytes",
                        lambda: 4 * GIB)
    calls = []

    def bridge(command, env=None, **_kwargs):
        calls.append(([str(part) for part in command], dict(env or {})))
        if command[1] == "--merge-batches":
            output = Path(command[2])
            output.mkdir()
            (output / "gate.tsv").write_text("merged\n")
            return SimpleNamespace(returncode=0, stdout=f"PASS\t{output}",
                                   stderr="")
        table, output = Path(command[2]), Path(command[3])
        output.mkdir()
        (output / "gate.tsv").write_text(
            "status\tPASS\npressure_levels_pa\t100000,50000\n")
        for line in table.read_text(encoding="utf-8").splitlines():
            (output / f"f{int(line.split(chr(9))[0]):03d}").mkdir()
        return SimpleNamespace(returncode=0, stdout=f"PASS\t{output}",
                               stderr="")

    monkeypatch.setattr(gfs_direct.subprocess, "run", bridge)
    monkeypatch.setattr(gfs_direct, "_parse_gate", lambda path: dict(
        line.split("\t") for line in Path(path).read_text().splitlines()))
    decoded = tmp_path / "scratch" / "decoded"
    decoded.parent.mkdir()
    one_run = ["bridge", "--series", str(series), str(decoded), "2026-10-05 00:00:00"]
    completed = gfs_direct._decode_series_bounded(
        one_run, series=series, records=tuple(records), decoded=decoded,
        scratch=decoded.parent, bridge=Path("bridge"),
        cycle_time=datetime(2026, 10, 5), levels_pa_csv=None,
        environment=None)
    assert completed.returncode == 0
    assert completed.stdout == f"PASS\t{decoded}"
    batches = [argv for argv, _env in calls if "--lead-batch" in argv]
    assert len(batches) == 4  # one lead each on this host
    assert all(env["GPUWM_GFS_BRIDGE_THREADS"] == "1" for _argv, env in calls
               if "--lead-batch" in _argv)
    # The first batch derives the ladder; every later one is given it.
    assert "--pressure-levels-pa" not in batches[0]
    assert all(argv[argv.index("--pressure-levels-pa") + 1] == "100000,50000"
               for argv in batches[1:])
    merges = [argv for argv, _env in calls if "--merge-batches" in argv]
    assert len(merges) == 1 and len(merges[0]) == 4 + 4
    # Every lead's arrays sit beside the merged receipts, as one run writes.
    assert sorted(path.name for path in decoded.iterdir()) == [
        "f000", "f003", "f006", "f009", "gate.tsv"]


def test_a_whole_series_that_fits_is_the_one_bridge_run(tmp_path, monkeypatch):
    path = tmp_path / "gfs.f000"
    path.write_bytes(b"GRIB")
    monkeypatch.setattr(window_module, "available_host_bytes",
                        lambda: 512 * GIB)
    calls = []
    monkeypatch.setattr(gfs_direct.subprocess, "run", lambda command, **kw: (
        calls.append(list(command)) or SimpleNamespace(returncode=0,
                                                        stdout="PASS", stderr="")))
    one_run = ["bridge", "--series", "s.tsv", "out", "cycle"]
    gfs_direct._decode_series_bounded(
        one_run, series=tmp_path / "s.tsv",
        records=((0, path), (3, path)), decoded=tmp_path / "out",
        scratch=tmp_path, bridge=Path("bridge"),
        cycle_time=datetime(2026, 10, 5), levels_pa_csv=None, environment=None)
    assert calls == [one_run]


# -- the as-posted decode in windows -------------------------------------

def test_leads_that_post_at_once_decode_in_host_sized_batches(
        tmp_path, monkeypatch):
    """The 2026-10-05T00 shape: every later lead ready in the same second.

    On a host that fits one lead per batch, f001..f003 posted together
    decode as three batches of one, each on the window's thread count,
    instead of one bridge run over all of them.
    """

    from test_posted_preparation import _gfs_as_posted_route

    monkeypatch.setattr(gfs_direct, "_file_bytes", lambda _path: GIB)
    monkeypatch.setattr(window_module, "available_host_bytes",
                        lambda: 4 * GIB)

    def post_the_rest(replay):
        replay.publish(1)
        replay.publish(2)
        replay.publish(3)

    _proof, _root, _manifest, _replay, seen = _gfs_as_posted_route(
        tmp_path, monkeypatch, actions=[post_the_rest])
    assert [batch for batch, _ in seen["loads"]] == [(0,), (1,), (2,), (3,)]
    decodes = [call for call in seen["bridge"] if "--lead-batch" in call]
    assert len(decodes) == 4
