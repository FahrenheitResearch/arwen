"""D-09 (2.8.8 acceptance): the local-DA surface route decodes the window it fetched.

The acceptance row D1 fetched one METAR at 22:01 for a 22:15 analysis and
then decoded only the slot 22:15:00..22:15:00 at the hourly stride.  The
decoder (``rw_asos decode``) matches a report to the valid time nearest it
and only within ``--match-seconds`` (600 s by default for METARs), so the
22:01 report matched nothing and 0 surface reports reached the filter.

These tests drive ``_surface_fetch`` itself with the archive calls
replaced by the committed real IEM rows (``tests/fixtures/
asos_surface_real``).  The decode is either the real writer, when
``GPUWM_RW_ASOS`` names an ``rw_asos`` binary, or the decoder's documented
matching rule applied to the arguments ``_surface_fetch`` passed.  The
analysis is placed at 13:30 with a 15 minute cadence, so the only fetched
reports are the seven taken at 13:15 -- fifteen minutes before the
analysis, the same shape as D1.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from gpuwm import local_da_fetch
from gpuwm.da import obs_surface

FIXTURES = Path(__file__).parent / "fixtures" / "asos_surface_real"
WHEN = datetime(2024, 5, 21, 13, 30, tzinfo=timezone.utc)
CADENCE = 900.0
# The runtime's own schedule for a first cycle: this analysis and the next.
SCHEDULE = [WHEN, WHEN + timedelta(seconds=CADENCE)]
ROUTE = {"boxes": [[-95.0, 40.0, -92.0, 43.0]], "networks": ["IA_ASOS"]}


TAKEN = datetime(2024, 5, 21, 13, 15, tzinfo=timezone.utc)


def _taken(record) -> set[datetime]:
    """Report instants, read the way the adapter reads them."""
    return {obs_surface._parse_seam_time(r["observation_time"], where="test")
            for r in record["reports"]}


def _parse(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _option(arguments, name, default=None):
    return arguments[arguments.index(name) + 1] if name in arguments else default


def _fetched_rows(arguments):
    """The committed archive rows inside the fetch's own [start, end]."""
    start, end = _parse(_option(arguments, "--start")), _parse(_option(arguments, "--end"))
    with open(FIXTURES / "observations_subset.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = []
    for row in rows:
        taken = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if start <= taken <= end:
            keep.append(row)
    return rows and list(rows[0]), keep


def _documented_decode(arguments, csv_path: Path, out: Path):
    """``rw_asos decode``'s matching rule (asos.rs ``match_reports``).

    Valid times run from --start to --end at the stride; each report goes
    to the ONE valid time nearest it and only within --match-seconds
    (METAR default 600 s, default stride 60 min).  Only the matching is
    mirrored; the record carries what ``_select_reports`` reads.
    """
    start, end = _parse(_option(arguments, "--start")), _parse(_option(arguments, "--end"))
    if "--step-minutes" in arguments:
        step = timedelta(minutes=int(_option(arguments, "--step-minutes")))
    else:
        step = timedelta(hours=int(_option(arguments, "--step-hours", "1")))
    match = int(_option(arguments, "--match-seconds", "600"))
    targets, when = [], start
    while when <= end:
        targets.append(when)
        when += step
    chosen: dict[tuple[str, int], dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            taken = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            distance, index = min((abs((taken - t).total_seconds()), i) for i, t in enumerate(targets))
            if distance > match:
                continue
            key = (row["station"], index)
            if key not in chosen or distance < chosen[key]["distance"]:
                chosen[key] = dict(distance=distance, report=dict(
                    station_id=row["station"],
                    valid_time=targets[index].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    observation_time=taken.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    values={}, flags=[]))
    reports = [entry["report"] for entry in chosen.values()]
    if not reports:
        raise RuntimeError(
            f"rw_asos: no station cleared the screens for {_option(arguments, '--start')} .. "
            f"{_option(arguments, '--end')}")
    out.write_text(json.dumps(dict(schema=obs_surface.OBS_SCHEMA, status="READY",
                                   provenance={}, reports=reports, station_table_sha256="0" * 64,
                                   stations=sorted({r["station_id"] for r in reports}))))


@pytest.fixture
def fake_archive(monkeypatch):
    from gpuwm.obs import frontdoor

    real_run = frontdoor.ASOS.run
    calls = []
    exe = os.environ.get("GPUWM_RW_ASOS")
    real_decode = bool(exe and Path(exe).is_file())

    def run(subcommand, arguments, *, schema=None):
        calls.append((subcommand, list(arguments)))
        out = Path(_option(arguments, "--out"))
        if subcommand == "stations":
            shutil.copyfile(FIXTURES / "stations.json", out)
            return {"stations": str(out)}
        if subcommand == "fetch":
            header, rows = _fetched_rows(arguments)
            with open(out, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            return {"rows": len(rows)}
        assert subcommand == "decode"
        if real_decode:
            return real_run(subcommand, arguments, schema=schema)
        _documented_decode(arguments, Path(_option(arguments, "--obs")), out)
        return {"decoded": str(out)}

    # FrontDoor is frozen; _surface_fetch imports ASOS at call time.
    monkeypatch.setattr(frontdoor, "ASOS", SimpleNamespace(run=run))
    return calls


def test_reports_fetched_inside_the_cadence_are_decoded_and_routed(tmp_path, fake_archive):
    records = local_da_fetch._surface_fetch(ROUTE, tmp_path, WHEN, CADENCE)
    fetch = next(args for name, args in fake_archive if name == "fetch")
    decode = next(args for name, args in fake_archive if name == "decode")
    # The decode window is the fetch window: nothing fetched is dropped
    # for being away from the analysis instant.
    assert _option(decode, "--start") == _option(fetch, "--start")
    assert _option(decode, "--end") == _option(fetch, "--end") == "2024-05-21T13:30:00Z"
    record = obs_surface.read_record(records[0])
    assert _taken(record) == {TAKEN}
    assert len({r["station_id"] for r in record["reports"]}) == 7
    # And the adapter routes every one of them to this analysis.
    counts = {key: 0 for key in ("reports_refused_flagged", "reports_without_observation_time",
                                 "reports_outside_window", "reports_routed_to_other_analysis",
                                 "reports_superseded_same_station")}
    chosen = obs_surface._select_reports(record, WHEN, SCHEDULE, CADENCE, counts)
    assert len(chosen) == 7
    assert counts["reports_outside_window"] == counts["reports_routed_to_other_analysis"] == 0


def test_the_d1_shape_one_report_fourteen_minutes_early(tmp_path, monkeypatch, fake_archive):
    """D1 itself: analysis 22:15, cadence 900 s, the station's report at 22:01."""
    when = datetime(2024, 5, 21, 13, 29, tzinfo=timezone.utc)
    # 13:15 is fourteen minutes before 13:29, as 22:01 was before 22:15.
    records = local_da_fetch._surface_fetch(ROUTE, tmp_path, when, CADENCE)
    record = obs_surface.read_record(records[0])
    assert _taken(record) == {TAKEN}


@pytest.mark.skipif(not (os.environ.get("GPUWM_RW_ASOS") and Path(os.environ["GPUWM_RW_ASOS"]).is_file()),
                    reason="GPUWM_RW_ASOS does not name an rw_asos binary")
def test_the_real_writer_accepts_the_decode_arguments(tmp_path):
    """The real binary takes the minute stride over the fetch window."""
    exe = os.environ["GPUWM_RW_ASOS"]
    out = tmp_path / "surface.json"
    subprocess.run([exe, "decode", "--stations", str(FIXTURES / "stations.json"),
                    "--obs", str(FIXTURES / "observations_subset.csv"),
                    "--start", "2024-05-21T13:15:00Z", "--end", "2024-05-21T13:30:00Z",
                    "--step-minutes", "1", "--match-seconds", "30",
                    "--min-report-rate", "0", "--out", str(out)],
                   check=True, capture_output=True, timeout=120)
    record = obs_surface.read_record(out)
    assert TAKEN in _taken(record)
