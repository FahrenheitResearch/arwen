"""CPU tests for the nowcast frames producer (tools/nowcast).

Everything here runs without the model package: the ledger is fed
synthetic object names in the exact forms the S3 archives use, frames are
a tiny synthetic ensemble, and the CLI is driven only as far as its
refusals and ``--plan-only``.  The GPU rollout itself is proven on the box.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.nowcast import frames as F  # noqa: E402

SOURCES = REPOSITORY_ROOT / "tools" / "nowcast" / "sources.toml"
RUNNER = REPOSITORY_ROOT / "tools" / "nowcast" / "run_nowcast.py"
SOURCE_ID = "stormscope-3km-10min"
ISSUE = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def row():
    return F.source_row(F.load_sources(SOURCES), SOURCE_ID)


def _goes(dt: datetime) -> str:
    return dt.strftime("%Y%j%H%M%S") + str(dt.microsecond // 100000)


def abi_key(end: datetime, sat: str = "19") -> str:
    start = end - timedelta(seconds=158)
    return (f"noaa-goes{sat}/ABI-L2-MCMIPC/{end:%Y/%j/%H}/OR_ABI-L2-MCMIPC-M6_G{sat}"
            f"_s{_goes(start)}_e{_goes(end)}_c{_goes(end + timedelta(seconds=30))}.nc")


def mrms_key(stamp: datetime, product: str = "MergedReflectivityQCComposite_00.50") -> str:
    return (f"s3://noaa-mrms-pds/CONUS/{product}/{stamp:%Y%m%d}/"
            f"MRMS_{product}_{stamp:%Y%m%d-%H%M%S}.grib2.gz")


def glm_key(start: datetime, sat: str = "19") -> str:
    end = start + timedelta(seconds=20)
    return (f"s3://noaa-goes{sat}/GLM-L2-LCFA/{start:%Y/%j/%H}/OR_GLM-L2-LCFA_G{sat}"
            f"_s{_goes(start)}_e{_goes(end)}_c{_goes(end + timedelta(seconds=2))}.nc")


def fill_ledger(row, issue=ISSUE, skip=(), extra=None, sat="19"):
    """A ledger holding what a causal run reads for every slot of every stream.

    ``skip`` holds (stream, slot index) pairs to leave out; ``extra`` maps
    (stream, slot index) to object URIs to add for that slot.
    """
    ledger = F.InputLedger(row["streams"])
    slots = F.history_slots(issue, row["step_minutes"], row["history_frames"])
    extra = extra or {}
    for k, nominal in enumerate(slots):
        with ledger.slot(nominal):
            if ("goes-abi-mcmip", k) not in skip:
                ledger.record(abi_key(nominal - timedelta(seconds=65), sat))
            if ("mrms-composite", k) not in skip:
                ledger.record(mrms_key(nominal - timedelta(seconds=84)))
            if ("mrms-base", k) not in skip:
                ledger.record(mrms_key(nominal - timedelta(seconds=82), "MergedBaseReflectivityQC_00.50"))
            if ("glm-lcfa", k) not in skip:
                # the source reads files starting from bin start - 20 s up to and
                # including the bin end, which here is the slot itself
                t = nominal - timedelta(minutes=5, seconds=20)
                while t <= nominal:
                    ledger.record(glm_key(t, sat))
                    t += timedelta(seconds=20)
            for (stream, kk), uris in extra.items():
                if kk == k:
                    for uri in uris:
                        ledger.record(uri)
    return ledger, slots


def check(row, ledger, slots, issue=ISSUE, sat="goes19"):
    return F.check_history(ledger, issue, slots, sat, float(row["history_tolerance_seconds"]))


# ------------------------------------------------------------- causality


def test_a_causal_history_passes_and_its_table_names_every_slot(row):
    ledger, slots = fill_ledger(row)
    table = check(row, ledger, slots)
    assert len(table) == len(row["streams"]) * len(slots)
    assert F.is_causal(ledger.records, ISSUE)
    doc = ledger.to_json(ISSUE, table)
    assert doc["causal"] is True
    # the GLM file that starts exactly at the issue time is the newest input
    assert doc["latest_input_time"] == "2026-10-01T18:00:00Z"
    assert all(o["stamp"] for o in doc["objects"])


def test_an_input_stamped_after_the_issue_time_is_refused(row):
    # the MRMS file nearest 18:00 is stamped 18:00:36, after an 18:00 issue
    late = mrms_key(ISSUE + timedelta(seconds=36))
    ledger, slots = fill_ledger(row, skip={("mrms-composite", 5)},
                                extra={("mrms-composite", 5): [late]})
    with pytest.raises(F.NowcastRefusal, match="after the issue time") as exc:
        check(row, ledger, slots)
    assert "hindcast scored as a forecast" in str(exc.value)
    assert not F.is_causal(ledger.records, ISSUE)


def test_an_input_stamped_exactly_at_the_issue_time_passes(row):
    exact = mrms_key(ISSUE)
    ledger, slots = fill_ledger(row, skip={("mrms-composite", 5)},
                                extra={("mrms-composite", 5): [exact]})
    check(row, ledger, slots)
    assert F.is_causal(ledger.records, ISSUE)


def test_an_object_no_stream_describes_is_refused(row):
    ledger, slots = fill_ledger(row, extra={("goes-abi-mcmip", 0): ["s3://some-bucket/unknown/file.bin"]})
    with pytest.raises(F.NowcastRefusal, match="matches no stream"):
        check(row, ledger, slots)


# ------------------------------------------------------------- history slots


def test_two_slots_resolving_to_the_same_object_are_refused(row):
    same = abi_key(ISSUE - timedelta(minutes=10, seconds=65))  # slot 4's object
    ledger, slots = fill_ledger(row, skip={("goes-abi-mcmip", 5)},
                                extra={("goes-abi-mcmip", 5): [same]})
    with pytest.raises(F.NowcastRefusal, match="same object") as exc:
        check(row, ledger, slots)
    assert "stood still" in str(exc.value)


def test_a_duplicate_stamp_under_another_name_is_refused(row):
    # same product time, different file name (a re-issued object)
    twin = mrms_key(ISSUE - timedelta(minutes=10, seconds=84)).replace(".grib2.gz", ".grib2.gz")
    twin = twin.replace("noaa-mrms-pds", "noaa-mrms-pds-mirror")
    ledger, slots = fill_ledger(row, skip={("mrms-composite", 5)},
                                extra={("mrms-composite", 5): [twin]})
    with pytest.raises(F.NowcastRefusal, match="same stamp|same object"):
        check(row, ledger, slots)


def test_a_slot_more_than_five_minutes_off_nominal_is_refused(row):
    early = mrms_key(ISSUE - timedelta(minutes=6))
    ledger, slots = fill_ledger(row, skip={("mrms-composite", 5)},
                                extra={("mrms-composite", 5): [early]})
    with pytest.raises(F.NowcastRefusal, match="off nominal") as exc:
        check(row, ledger, slots)
    assert "jumped" in str(exc.value)


def test_a_slot_exactly_five_minutes_off_nominal_passes(row):
    edge = mrms_key(ISSUE - timedelta(minutes=5))
    ledger, slots = fill_ledger(row, skip={("mrms-composite", 5)},
                                extra={("mrms-composite", 5): [edge]})
    check(row, ledger, slots)


def test_a_missing_history_frame_is_refused(row):
    for stream in ("goes-abi-mcmip", "glm-lcfa"):
        ledger, slots = fill_ledger(row, skip={(stream, 2)})
        with pytest.raises(F.NowcastRefusal, match="read nothing for the history slot"):
            check(row, ledger, slots)


def test_two_objects_for_one_single_slot_are_refused(row):
    # a source that fell back to a second candidate: the ledger cannot say which fed the frame
    second = mrms_key(ISSUE - timedelta(seconds=204))
    ledger, slots = fill_ledger(row, extra={("mrms-composite", 5): [second]})
    with pytest.raises(F.NowcastRefusal, match="2 objects"):
        check(row, ledger, slots)


def test_a_read_outside_any_slot_is_refused(row):
    ledger, slots = fill_ledger(row)
    ledger.record(abi_key(ISSUE - timedelta(minutes=30, seconds=65)))
    with pytest.raises(F.NowcastRefusal, match="outside any history slot"):
        check(row, ledger, slots)


def test_request_times_land_before_each_slot(row):
    streams = {s["name"]: s for s in row["streams"]}
    assert F.request_time(ISSUE, streams["goes-abi-mcmip"]) == ISSUE - timedelta(minutes=5)
    assert F.request_time(ISSUE, streams["mrms-composite"]) == ISSUE - timedelta(seconds=60)
    assert F.request_time(ISSUE, streams["glm-lcfa"]) == ISSUE - timedelta(minutes=5)


# ------------------------------------------------------------- satellite


@pytest.mark.parametrize("when, sat", [("2024-05-21T18:00Z", "goes16"),
                                       ("2025-03-14T18:00Z", "goes16"),
                                       ("2026-08-02T18:00Z", "goes19"),
                                       ("2026-10-01T18:00Z", "goes19")])
def test_goes_east_by_date(row, when, sat):
    assert F.goes_east_satellite(F.parse_utc(when), row["goes_east"]) == sat


@pytest.mark.parametrize("when", ["2025-04-04T00:00Z", "2025-04-05T18:00Z", "2025-04-06T23:59Z",
                                  "2017-06-01T18:00Z"])
def test_the_handover_gap_and_dates_before_the_record_are_refused(row, when):
    with pytest.raises(F.NowcastRefusal, match="no GOES-East period") as exc:
        F.goes_east_satellite(F.parse_utc(when), row["goes_east"])
    assert "wrong scan geometry" in str(exc.value)


def test_a_requested_satellite_that_was_not_goes_east_is_refused(row):
    with pytest.raises(F.NowcastRefusal, match="was not GOES-East"):
        F.check_satellite("goes16", F.parse_utc("2026-08-02T18:00Z"), row["goes_east"])
    assert F.check_satellite(None, F.parse_utc("2026-08-02T18:00Z"), row["goes_east"]) == "goes19"


def test_inputs_from_the_wrong_satellite_are_refused(row):
    ledger, slots = fill_ledger(row, sat="16")
    with pytest.raises(F.NowcastRefusal, match="comes from goes16") as exc:
        check(row, ledger, slots)
    assert "wrong scan geometry" in str(exc.value)


# ------------------------------------------------------------- revision pin


def test_the_revision_pin(row):
    pin = row["revision"]
    assert pin == "62f0fd2fa52c3cff67c931daac18cdc0d9f58d2a"
    assert F.check_revision(None, pin) == pin
    assert F.check_revision(pin, pin) == pin
    assert F.check_revision(f"hf://nvidia/stormscope-goes-mrms@{pin}", pin) == pin
    with pytest.raises(F.NowcastRefusal, match="not the pinned revision") as exc:
        F.check_revision("0123456789abcdef0123456789abcdef01234567", pin)
    assert "weights that did not make the frames" in str(exc.value)
    with pytest.raises(F.NowcastRefusal):
        F.check_revision("hf://nvidia/stormscope-goes-mrms@main", pin)


# ------------------------------------------------------------- read taps


def test_tap_reads_records_sync_and_async_reads_into_their_slot(row, tmp_path):
    cached = tmp_path / "cached.grib2"
    cached.write_bytes(b"synthetic object")

    class Source:
        def fetch(self, uri):
            return str(cached)

        async def fetch_async(self, uri):
            return str(cached)

    src = Source()
    ledger = F.InputLedger(row["streams"])
    F.tap_reads(src, "fetch", ledger)
    F.tap_reads(src, "fetch_async", ledger)
    slot = ISSUE - timedelta(minutes=10)
    key_a = mrms_key(slot - timedelta(seconds=84))
    key_b = abi_key(slot - timedelta(seconds=65))
    with ledger.slot(slot):
        src.fetch(key_a)
        asyncio.run(src.fetch_async(key_b))
        src.fetch(key_a)  # a cache hit is not a second object
    assert [r.stream for r in ledger.records] == ["mrms-composite", "goes-abi-mcmip"]
    assert all(r.slot == slot for r in ledger.records)
    want = hashlib.sha256(b"synthetic object").hexdigest()
    assert all(r.sha256 == want and r.nbytes == 16 for r in ledger.records)
    assert ledger.records[1].satellite == "goes19"
    assert ledger.records[1].stamp == slot - timedelta(seconds=65)


# ------------------------------------------------------------- lattice


HRRR = {"truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5, "ref_lat": 38.5,
        "earth_radius_m": 6371229.0}


def test_the_projection_reproduces_the_published_hrrr_grid_corners():
    # NCEP's HRRR CONUS grid: 1799 x 1059 at 3 km, first point 21.138123 N
    # 237.280472 E, last point 47.842195 N 299.082807 E (sphere 6371229 m).
    x0, y0 = F.lcc_forward(21.138123, 237.280472, HRRR)
    xs = float(x0) + 3000.0 * np.arange(1799)
    ys = float(y0) + 3000.0 * np.arange(1059)
    block = F.lattice_from_projection_coords(xs, ys, HRRR)
    lat, lon = block["corners_latlon"]["jn_in"]
    assert F.great_circle_m(lat, lon, 47.842195, 299.082807 - 360.0, 6371229.0) < 100.0
    assert block["dx_m"] == pytest.approx(3000.0) and block["dy_m"] == pytest.approx(3000.0)
    assert block["row_order"] == "south_to_north"
    lat_back, lon_back = F.lcc_inverse(x0, y0, HRRR)
    assert float(lat_back) == pytest.approx(21.138123, abs=1e-9)
    assert float(lon_back) == pytest.approx(237.280472 - 360.0, abs=1e-9)


def test_the_lattice_from_uniform_projection_coordinates_round_trips():
    xs = -1_000_000.0 + 3000.0 * np.arange(40)
    ys = -500_000.0 + 3000.0 * np.arange(30)
    block = F.lattice_from_projection_coords(xs, ys, HRRR)
    assert (block["nx"], block["ny"]) == (40, 30)
    assert block["x0_m"] == pytest.approx(-1_000_000.0) and block["y0_m"] == pytest.approx(-500_000.0)
    assert block["ref_lon"] == block["stand_lon"] == -97.5
    F.check_lattice_corners(block)
    moved = json.loads(json.dumps(block))
    moved["corners_latlon"]["j0_i0"][0] += 0.01  # about 1.1 km
    with pytest.raises(F.NowcastRefusal, match="from where its projection numbers put it"):
        F.check_lattice_corners(moved)


def test_a_lattice_whose_spacing_is_not_uniform_is_refused():
    xs = -1_000_000.0 + 3000.0 * np.arange(40)
    xs[25:] += 150.0  # one step of 3150 m
    ys = -500_000.0 + 3000.0 * np.arange(30)
    with pytest.raises(F.NowcastRefusal, match="not uniformly spaced") as exc:
        F.lattice_from_projection_coords(xs, ys, HRRR)
    assert "wrong columns" in str(exc.value)


def test_the_lattice_from_a_model_latlon_grid_matches_an_independent_projection():
    # the engine's own WPS Lambert transcription (6370000 m sphere) makes the
    # lat/lon; this module's projection must recover a uniform 3 km lattice
    from gpuwm.static.lambert import LambertGrid

    grid = LambertGrid(ref_lat=38.5, ref_lon=-97.5, truelat1=38.5, truelat2=38.5,
                       stand_lon=-97.5, dx=3000.0, dy=3000.0, e_we=61, e_sn=41)
    jj, ii = np.meshgrid(np.arange(40, dtype=float) + 1.0, np.arange(60, dtype=float) + 1.0,
                         indexing="ij")
    lat, lon = grid.ij_to_latlon(ii, jj)
    proj = dict(HRRR, earth_radius_m=6370000.0)
    block = F.lattice_from_latlon(lat, lon, proj)
    assert block["dx_m"] == pytest.approx(3000.0, abs=1e-3)
    assert block["dy_m"] == pytest.approx(3000.0, abs=1e-3)
    assert (block["nx"], block["ny"]) == (60, 40)
    # the same lat/lon read on HRRR's sphere is not the same lattice: it is
    # still a lattice, but its spacing and origin move, which is why the
    # adapter never assumes two "3 km Lambert" grids are one
    other = F.lattice_from_latlon(lat, lon, HRRR)
    assert abs(other["dx_m"] - 3000.0) > 0.5
    bent = lon.copy()
    bent[:, 30:] += 0.002  # a 170 m kink halfway across
    with pytest.raises(F.NowcastRefusal):
        F.lattice_from_latlon(lat, bent, proj)


# ------------------------------------------------------------- frames and receipt


def _synthetic_members(rng, members, ny, nx, lead):
    field = rng.normal(20.0, 15.0, size=(members, ny, nx)).astype(np.float32) + lead / 10.0
    field[:, 0, 0] = np.nan  # outside coverage
    field[:, -1, -1] = -10.0  # model clear air
    return field


def _write_tiny_run(root: Path, members=3, ny=4, nx=5, steps=3, batches=((0, 2), (2, 1))):
    rng = np.random.default_rng(7)
    writer = F.FrameWriter(root, ISSUE, 10, members, ny, nx,
                           ["refc", "refc_base", "glm_density"], "refc")
    truth = {}
    for b0, nb in batches:
        for k in range(1, steps + 1):
            for v in ("refc", "refc_base", "glm_density"):
                slab = _synthetic_members(rng, nb, ny, nx, 10 * k)
                writer.write(10 * k, v, b0, slab)
                truth.setdefault((k, v), np.zeros((members, ny, nx), np.float32))[b0:b0 + nb] = slab
    return writer, truth


def test_the_receipt_round_trips_with_the_sha256_of_each_frame(row, tmp_path):
    root = tmp_path / "frames"
    writer, truth = _write_tiny_run(root)
    frames = writer.finish()
    assert [f["lead_minutes"] for f in frames] == [10, 20, 30]
    assert frames[0]["file"] == "20261001T1810Z/refc.f32"
    ledger, slots = fill_ledger(row)
    table = check(row, ledger, slots)
    inputs_sha = F.write_json_atomic(root / "inputs.json", ledger.to_json(ISSUE, table))
    xs = -1_000_000.0 + 3000.0 * np.arange(5)
    ys = -500_000.0 + 3000.0 * np.arange(4)
    receipt = F.build_receipt(
        status=F.STATUS_READY,
        source={"kind": "nowcast", "id": SOURCE_ID, "package": row["package"],
                "revision": row["revision"], "code": {}},
        issue=ISSUE, ledger_json=ledger.to_json(ISSUE, table), members=3, variable="refc",
        units="dBZ", lattice=F.lattice_from_projection_coords(xs, ys, HRRR), frames=frames,
        sampler=row["sampler"], seed=1, timing={"seconds_per_step_steady": 1.0},
        peak_device_bytes=123, inputs_sha256=inputs_sha)
    F.write_receipt(root, receipt)

    back = F.read_receipt(root)
    assert back["status"] == "READY" and back["causal"] is True
    assert back["schema"] == "gpuwm-obs.nowcast-frames.v1"
    assert back["latest_input_time"] == "2026-10-01T18:00:00Z"
    for k, frame in enumerate(back["frames"], start=1):
        data = (root / frame["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == frame["sha256"]
        assert len(data) == 3 * 4 * 5 * 4
        got = F.load_frame(root, frame)
        np.testing.assert_array_equal(got, truth[(k, "refc")])
        assert np.isnan(got[:, 0, 0]).all() and (got[:, -1, -1] == -10.0).all()
        for v in ("refc_base", "glm_density"):
            extra = frame["extra"][v]
            raw = np.fromfile(root / extra["file"], dtype="<f4").reshape(3, 4, 5)
            np.testing.assert_array_equal(raw, truth[(k, v)])
            assert hashlib.sha256((root / extra["file"]).read_bytes()).hexdigest() == extra["sha256"]
    assert not list(root.rglob("*.partial"))

    # one flipped byte in one frame is refused
    path = root / back["frames"][1]["file"]
    raw = bytearray(path.read_bytes())
    raw[7] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(F.NowcastRefusal, match="does not match the receipt"):
        F.read_receipt(root)


def test_unfinished_and_refused_receipts_are_not_consumable(row, tmp_path):
    for status in (F.STATUS_RUNNING, F.STATUS_REFUSED, F.STATUS_FAILED):
        root = tmp_path / status
        F.write_receipt(root, F.build_receipt(
            status=status, source={"id": SOURCE_ID}, issue=ISSUE, ledger_json=None,
            members=1, variable="refc", units="dBZ", lattice=None, frames=[]))
        with pytest.raises(F.NowcastRefusal, match="not READY"):
            F.read_receipt(root)


def test_frame_writer_refusals(tmp_path):
    writer = F.FrameWriter(tmp_path, ISSUE, 10, 2, 3, 3, ["refc"], "refc")
    with pytest.raises(F.NowcastRefusal, match="positive multiple"):
        writer.write(15, "refc", 0, np.zeros((2, 3, 3)))
    with pytest.raises(F.NowcastRefusal, match="expected"):
        writer.write(10, "refc", 0, np.zeros((2, 3, 4)))
    writer.write(10, "refc", 0, np.zeros((1, 3, 3)))
    with pytest.raises(F.NowcastRefusal, match="written twice"):
        writer.write(10, "refc", 0, np.zeros((1, 3, 3)))
    with pytest.raises(F.NowcastRefusal, match="1 of 2 members"):
        writer.finish()


def test_a_gap_in_frame_leads_is_refused(tmp_path):
    writer = F.FrameWriter(tmp_path, ISSUE, 10, 1, 2, 2, ["refc"], "refc")
    writer.write(10, "refc", 0, np.zeros((1, 2, 2)))
    writer.write(30, "refc", 0, np.zeros((1, 2, 2)))
    with pytest.raises(F.NowcastRefusal, match="have a gap"):
        writer.finish()


# ------------------------------------------------------------- the CLI, short of the model


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(RUNNER), *args], capture_output=True, text=True,
                          cwd=cwd or REPOSITORY_ROOT, timeout=120)


def test_plan_only_prints_the_slots_and_requests(tmp_path):
    res = _cli("--source", SOURCE_ID, "--issue", "2026-10-01T18:00Z", "--minutes", "120",
               "--members", "8", "--out", str(tmp_path / "x"), "--plan-only")
    assert res.returncode == 0, res.stderr
    plan = json.loads(res.stdout)
    assert plan["satellite"] == "goes19" and plan["steps"] == 12
    assert plan["slots"][0] == "2026-10-01T17:10:00Z" and plan["slots"][-1] == "2026-10-01T18:00:00Z"
    assert plan["requests"]["radar"][-1] == "2026-10-01T17:59:00Z"
    assert plan["requests"]["satellite"][-1] == "2026-10-01T17:55:00Z"
    assert plan["valid_times"][-1] == "2026-10-01T20:00:00Z"
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("extra, match", [
    (["--satellite", "goes16"], "was not GOES-East"),
    (["--revision", "main"], "not the pinned revision"),
    (["--minutes", "125"], "not a positive multiple"),
])
def test_cli_refusals_write_a_refused_receipt_before_any_model_import(tmp_path, extra, match):
    out = tmp_path / "frames"
    res = _cli("--source", SOURCE_ID, "--issue", "2026-10-01T18:00Z", "--out", str(out), *extra)
    assert res.returncode == 3, res.stderr
    assert match in res.stderr
    receipt = json.loads((out / "nowcast.json").read_text())
    assert receipt["status"] == "REFUSED" and match in receipt["refusal"]
    assert not (out / "inputs.json").exists()


def test_cli_refuses_a_handover_gap_issue_time(tmp_path):
    res = _cli("--source", SOURCE_ID, "--issue", "2025-04-05T18:00Z",
               "--out", str(tmp_path / "f"), "--plan-only")
    assert res.returncode == 3 and "no GOES-East period" in res.stderr


def test_cli_will_not_overwrite_a_ready_receipt(tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    (out / "nowcast.json").write_text(json.dumps({"schema": F.SCHEMA, "status": "READY"}))
    res = _cli("--source", SOURCE_ID, "--issue", "2026-10-01T18:00Z", "--out", str(out))
    assert res.returncode == 3 and "already READY" in res.stderr


def test_a_naive_issue_time_is_rejected():
    with pytest.raises(ValueError, match="no zone"):
        F.parse_utc("2026-10-01T18:00")
    assert F.parse_utc("2026-10-01T18:00Z") == ISSUE
    assert F.frame_dirname(ISSUE + timedelta(minutes=10)) == "20261001T1810Z"


# ------------------------------------------------------------- review fixes (box-waste and coverage)


def _drop_glm(ledger, slot, starts):
    """Remove the GLM files of one slot whose start is in ``starts``."""
    ledger.records = [r for r in ledger.records
                      if not (r.stream == "glm-lcfa" and r.slot == slot and r.start in starts)]


def test_a_lightning_bin_with_one_missing_file_passes(row):
    ledger, slots = fill_ledger(row)
    _drop_glm(ledger, slots[3], {slots[3] - timedelta(minutes=2)})
    table = check(row, ledger, slots)
    assert any(t["stream"] == "glm-lcfa" for t in table)


def test_a_lightning_outage_inside_a_bin_is_refused(row):
    # two back-to-back 20 s files gone: 40 s of the bin were never read, and
    # the gridded source would count them as no lightning
    ledger, slots = fill_ledger(row)
    gone = {slots[3] - timedelta(minutes=2), slots[3] - timedelta(minutes=2, seconds=-20)}
    _drop_glm(ledger, slots[3], gone)
    with pytest.raises(F.NowcastRefusal, match="does not cover the bin") as exc:
        check(row, ledger, slots)
    assert "outage of the source as no lightning" in str(exc.value)


def test_a_lightning_bin_missing_its_tail_is_refused(row):
    ledger, slots = fill_ledger(row)
    tail = {slots[5] - timedelta(seconds=s) for s in (0, 20, 40, 60)}
    _drop_glm(ledger, slots[5], tail)
    with pytest.raises(F.NowcastRefusal, match="does not cover the bin"):
        check(row, ledger, slots)


def test_a_listed_file_the_read_skipped_counts_as_missing(row, tmp_path):
    ledger, slots = fill_ledger(row, skip={("glm-lcfa", 2)})
    t = slots[2] - timedelta(minutes=5, seconds=20)
    with ledger.slot(slots[2]):
        while t <= slots[2]:
            # every file of the bin was listed, but the source found none on S3
            ledger.record(glm_key(t), local_path=tmp_path / "never-written.nc")
            t += timedelta(seconds=20)
    assert all(r.missing for r in ledger.records if r.slot == slots[2] and r.stream == "glm-lcfa")
    with pytest.raises(F.NowcastRefusal, match="no object names its span"):
        check(row, ledger, slots)


def test_bin_coverage_gap_measures_the_largest_unread_stretch():
    t0 = ISSUE - timedelta(minutes=5)

    def rec(s, e):
        return F.InputRecord(uri="", bucket="", key="", stream="glm-lcfa", stamp=s,
                             satellite=None, slot=ISSUE, start=s, end=e)

    full = [rec(t0 + timedelta(seconds=20 * k), t0 + timedelta(seconds=20 * k + 20)) for k in range(15)]
    assert F.bin_coverage_gap(full, t0, ISSUE) == 0.0
    assert F.bin_coverage_gap(full[:7] + full[9:], t0, ISSUE) == 40.0
    assert F.bin_coverage_gap(full[1:], t0, ISSUE) == 20.0
    assert F.bin_coverage_gap([], t0, ISSUE) is None


def test_the_radar_candidates_are_cut_at_the_slot():
    slot = ISSUE
    near = [(ISSUE + timedelta(seconds=40), "after"), (ISSUE - timedelta(seconds=80), "before"),
            (ISSUE, "exact"), (ISSUE - timedelta(seconds=200), "earlier")]
    kept = F.causal_candidates(near, slot)
    assert [u for _, u in kept] == ["before", "exact", "earlier"]
    # one missing file falls back to the one before it, never to one after the slot
    assert F.causal_candidates(near[:1] + near[3:], slot) == [near[3]]
    assert F.causal_candidates(near, None) == near


def test_unobserved_radar_cells_are_not_written_as_clear_air():
    valid = np.ones((3, 4), dtype=bool)
    valid[0, 0] = False
    raw = np.full((6, 3, 4), -99.0, dtype=np.float32)  # observed no echo stays
    raw[:, 1, 1] = 35.0
    raw[2, 2, 3] = -999.0  # a radar outage in one history slot
    raw[5, 0, 2] = np.nan  # beyond the source grid
    raw[4, 0, 0] = -999.0  # outside the model mask already
    keep, counts = F.unobserved_cells(raw, valid, -900.0)
    assert counts == [0, 0, 1, 0, 0, 1]
    assert not keep[2, 3] and not keep[0, 2] and not keep[0, 0]
    assert keep[1, 1] and keep[2, 2]
    assert int(keep.sum()) == 12 - 3


@pytest.mark.parametrize("issue, match", [
    ("2026-10-01T18:03Z", "5-minute grid"),
    ("2020-10-14T00:30Z", "before the input archive"),
    ("2025-04-07T00:30Z", "falls in no GOES-East period|goes16 period"),
])
def test_plans_the_sources_would_reject_mid_fetch_are_refused(tmp_path, issue, match):
    res = _cli("--source", SOURCE_ID, "--issue", issue, "--out", str(tmp_path / "f"), "--plan-only")
    assert res.returncode == 3, res.stderr
    assert re.search(match, res.stderr), res.stderr
    assert "rented box" in res.stderr or "scan geometr" in res.stderr


def test_an_issue_on_the_five_minute_grid_plans(tmp_path):
    res = _cli("--source", SOURCE_ID, "--issue", "2026-10-01T18:05Z", "--out", str(tmp_path / "f"),
               "--plan-only")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["requests"]["satellite"][-1] == "2026-10-01T18:00:00Z"


def test_a_history_field_with_no_finite_value_is_refused():
    slots = F.history_slots(ISSUE, 10, 6)
    ok = [[True, True]] * 6
    F.check_history_fields(ok, ["refc", "refc_base"], slots, "radar")
    bad = [list(r) for r in ok]
    bad[4][1] = False
    with pytest.raises(F.NowcastRefusal, match="refc_base has no finite value") as exc:
        F.check_history_fields(bad, ["refc", "refc_base"], slots, "radar")
    assert "on the card" in str(exc.value)


def test_the_card_lock_is_the_engines_and_refuses_a_second_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    from gpuwm.supervisor import default_lock_path

    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert F.card_lock_path(uuid) == default_lock_path(uuid)
    with F.card_lock(uuid, "first") as path:
        assert path.exists()
        with pytest.raises(F.NowcastRefusal, match="held by another run") as exc:
            with F.card_lock(uuid, "second"):
                pass
        assert "run out of memory" in str(exc.value)
    with F.card_lock(uuid, "third"):
        pass  # released after the first holder
    # torch prints the UUID bare; the lock key is nvidia-smi's form either way
    assert F.nvidia_smi_uuid(uuid[4:]) == uuid == F.nvidia_smi_uuid(uuid)


def test_a_stale_partial_frame_cannot_leak_into_a_new_run(tmp_path):
    stale = tmp_path / "20261001T1810Z" / "refc.f32.partial"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"\x7f" * (4 * 2 * 2 * 4))  # a failed run with 4 members
    writer = F.FrameWriter(tmp_path, ISSUE, 10, 2, 2, 2, ["refc"], "refc")
    writer.write(10, "refc", 0, np.ones((1, 2, 2)))
    writer.write(10, "refc", 1, np.full((1, 2, 2), 2.0))
    frames = writer.finish()
    data = F.load_frame(tmp_path, frames[0])
    assert (tmp_path / frames[0]["file"]).stat().st_size == 2 * 2 * 2 * 4
    np.testing.assert_array_equal(data[0], 1.0)
    np.testing.assert_array_equal(data[1], 2.0)
