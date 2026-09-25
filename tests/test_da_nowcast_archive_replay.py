"""Replaying an archived window through the nowcast front door.

Pure tests: no network, no GPU, no subprocess execution.  The archive
is a fake that answers the window it is ASKED for, exactly as the real
listing does, so what these tests pin is which window the front door
asks about: the one the caller named, or whatever time it happens to be
when the command is typed.

The last class here drives the WHOLE door with every stage replaced by
a fake that leaves behind the artifacts the next stage reads, so what
the run writes into ``nowcast-receipt.json`` is pinned by reading the
file the door actually wrote.  A block that is asserted only by calling
the function that builds it can be correct and still never reach the
receipt -- which is exactly how the frames block shipped documented and
unobserved.

A replay is the whole point of the fixture: the volumes below exist at
two epochs, the requested window and the day the command is typed on.
A survey that reads the wall clock finds the second set, sites the
domain on weather that has nothing to do with the window, and measures
a motion vector from it.  Nothing downstream can recover from that,
because every later stage takes the domain as given.

Sites here are synthetic ids; real station names never enter the tree's
generic code or its fixtures (standing owner rule).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.da_nowcast import iso, parse_iso

#: The wall clock the command is typed at, six weeks after the window.
WALL = datetime(2026, 9, 18, 17, 30, tzinfo=timezone.utc)
#: The archived window a replay asks for.  Six 900 s cycles put the
#: model init on 04:00Z, which is the front door's whole-hour rule.
WINDOW_END = datetime(2026, 8, 5, 5, 30, tzinfo=timezone.utc)
SITE = "QQQQ"


def iso_ms(stamp: datetime) -> str:
    """The pack's own spelling of a radial collection instant."""

    return (stamp.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{stamp.microsecond // 1000:03d}Z")


class _FrozenDatetime(datetime):
    """``datetime`` with one hand held: ``now`` is the typing time."""

    @classmethod
    def now(cls, tz=None):
        return WALL if tz is not None else WALL.replace(tzinfo=None)


class FakeArchive:
    """Volumes at a fixed cadence around each epoch, listed by window.

    The real listing answers the window it is given and nothing else,
    which is the only property these tests need it to have.
    """

    def __init__(self, *epochs, cadence_s: int = 300, span_s: int = 7200):
        self.stamps: list[datetime] = []
        for epoch in epochs:
            steps = span_s // cadence_s + 1
            self.stamps.extend(epoch - timedelta(seconds=cadence_s * k)
                               for k in range(steps))
        self.stamps.sort()
        self.listed: list[tuple[datetime, datetime]] = []
        self.calls = 0

    #: How long a volume takes to scan after the instant its key names,
    #: and how long after its start the archive publishes it.  A volume
    #: is complete 240 s after its first radial and on the shelf 60 s
    #: after that, which is the shape the real archive has (a KTLX volume
    #: of 2026-09-19 started 12:02:36, ended 12:09:16 and was published
    #: at 12:09:16).
    VOLUME_SECONDS = 240
    PUBLISH_SECONDS = 300

    def name(self, stamp: datetime) -> str:
        return f"{SITE}{stamp:%Y%m%d_%H%M%S}_V06"

    def within(self, start: str, end: str) -> list[datetime]:
        first, last = parse_iso(start), parse_iso(end)
        return [s for s in self.stamps if first <= s <= last]

    def end_of(self, stamp: datetime) -> datetime:
        return stamp + timedelta(seconds=self.VOLUME_SECONDS)

    def published_at(self, stamp: datetime) -> datetime:
        return stamp + timedelta(seconds=self.PUBLISH_SECONDS)

    # -- the rw_nexrad seam ------------------------------------------------
    def run_list(self, binary, *, site, start, end, bucket=None,
                 limit=None) -> dict:
        self.listed.append((parse_iso(start), parse_iso(end)))
        # The archive lists a volume by its START and only once it is on
        # the shelf: a replay window six weeks back sees every volume it
        # names, a live listing at the wall clock does not yet see the
        # one being scanned.
        return {"volumes": [
            {"filename": self.name(stamp),
             "key": f"{stamp:%Y/%m/%d}/{site}/{self.name(stamp)}",
             "valid_time": iso(stamp),
             "last_modified": iso_ms(self.published_at(stamp))}
            for stamp in self.within(start, end)
            if self.published_at(stamp) <= WALL]}

    def run_fetch(self, binary, *, site, start, end, out, bucket=None,
                  **kwargs) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        picked = self.within(start, end)
        for stamp in picked:
            (out / self.name(stamp)).write_bytes(b"level-ii volume")
        return {"downloaded": len(picked)}

    def run_decode(self, binary, *, volume, out, **kwargs) -> dict:
        out.write_bytes(b"sweep pack")
        return {"status": "PASS"}

    def run_verify(self, binary, *, pack) -> dict:
        return {"status": "PASS"}

    def stamp_of(self, path) -> datetime:
        """The key time the fake wrote into this pack's file name."""

        text = Path(path).name[len(SITE):len(SITE) + 15]
        return datetime.strptime(text, "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc)

    def read_sweep_pack(self, path):
        # What the real reader lifts from a pack the tree's own rw_nexrad
        # wrote: the volume's own start and end instants beside the site.
        stamp = self.stamp_of(path)
        return SimpleNamespace(
            site=SimpleNamespace(lat_deg=40.0, lon_deg=-90.0, alt_m=200.0),
            sweeps=(), start_time=iso_ms(stamp),
            end_time=iso_ms(self.end_of(stamp)), complete=True)

    def echo_stats(self, volume, **kwargs) -> dict:
        # Two volumes, two centroids, so the survey has a displacement
        # to call a motion from.  The numbers are the fixture's.
        self.calls += 1
        return {"gates": 4000, "max_dbz": 58.0,
                "centroid_east_km": 10.0 * self.calls,
                "centroid_north_km": 4.0 * self.calls}


def install(monkeypatch, archive: FakeArchive) -> None:
    """Point the survey's subprocess seams at the fake archive."""

    import tools.da_nowcast as front
    from gpuwm.obs import nexrad, sweeps

    monkeypatch.setattr(nexrad, "find_nexrad_bin",
                        lambda: Path("rw_nexrad"))
    monkeypatch.setattr(nexrad, "run_list", archive.run_list)
    monkeypatch.setattr(nexrad, "run_fetch", archive.run_fetch)
    monkeypatch.setattr(nexrad, "run_decode", archive.run_decode)
    monkeypatch.setattr(nexrad, "run_verify", archive.run_verify)
    monkeypatch.setattr(sweeps, "read_sweep_pack", archive.read_sweep_pack)
    monkeypatch.setattr(front, "echo_stats", archive.echo_stats)


def replay(tmp_path: Path, monkeypatch, window_end: str,
           extra: tuple[str, ...] = ()) -> tuple[FakeArchive, dict]:
    """One ``da_nowcast run --stop-after survey`` against the fake."""

    import tools.da_nowcast as front

    archive = FakeArchive(WINDOW_END, WALL)
    install(monkeypatch, archive)
    monkeypatch.setattr(front, "datetime", _FrozenDatetime)
    out = tmp_path / "case"
    code = front.main(["run", "--site", SITE,
                       "--window-end", window_end,
                       "--out", str(out), "--stop-after", "survey",
                       *extra])
    assert code == 0
    survey = json.loads((out / "receipts" / "00-survey.json")
                        .read_text(encoding="utf-8"))
    plan = json.loads((out / "receipts" / "01-plan.json")
                      .read_text(encoding="utf-8"))
    return archive, {"survey": survey, "plan": plan}


class TestTheSurveyAsksAboutTheWindowItWasGiven:
    def test_a_past_window_lists_that_windows_volumes(
            self, tmp_path, monkeypatch):
        archive, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        assert archive.listed, "the survey never asked the archive"
        start, end = archive.listed[0]
        assert end == WINDOW_END
        assert start < WINDOW_END
        newest = parse_iso(
            receipts["survey"]["survey_volumes"][-1]["valid_time"])
        assert newest <= WINDOW_END
        assert newest.date() == WINDOW_END.date()

    def test_it_does_not_list_the_day_the_command_was_typed(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        for volume in receipts["survey"]["survey_volumes"]:
            assert parse_iso(volume["valid_time"]).date() \
                == WINDOW_END.date()

    def test_the_lag_is_measured_against_the_window_end(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        # 300 s cadence: the newest volume in the window is minutes
        # behind its end, never the six weeks the wall clock would read.
        assert receipts["survey"]["archive_lag_seconds"] <= 900.0

    def test_the_receipt_records_which_clock_the_survey_used(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        clock = receipts["survey"]["clock"]
        assert clock["mode"] == "window"
        assert clock["at"] == iso(WINDOW_END)

    def test_the_domain_is_sited_on_the_windows_echo(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        # Proof the box came from echo the survey actually measured in
        # the window, rather than from the antenna fallback.
        assert receipts["plan"]["domain_center"]["basis"].startswith(
            "echo centroid")


class TestLiveIsUnchanged:
    def test_latest_still_asks_the_wall_clock(self, tmp_path, monkeypatch):
        archive, receipts = replay(tmp_path, monkeypatch, "latest")
        _, end = archive.listed[0]
        assert end == WALL
        newest = parse_iso(
            receipts["survey"]["survey_volumes"][-1]["valid_time"])
        assert newest.date() == WALL.date()

    def test_the_receipt_says_the_wall_clock_was_used(self, tmp_path,
                                                      monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, "latest")
        clock = receipts["survey"]["clock"]
        assert clock["mode"] == "wall"
        assert clock["at"] == iso(WALL)


class TestAdmissionIsOnTheVolumesEnd:
    """A volume exists when its last radial has been collected.

    The archive names a volume by its start, so a listing up to the
    window end can name a volume that was still being scanned when the
    window ended.  Admitting that one dated its top cut to a moment
    before the antenna reached it, and a replay could assimilate data
    the atmosphere had not produced yet.
    """

    def test_the_volume_still_scanning_at_the_window_end_is_refused(
            self, tmp_path, monkeypatch):
        archive, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        survey = receipts["survey"]
        # The newest KEY in the window starts at the window end itself and
        # ends 240 s later; the one before it ended 60 s before the window
        # end and is the newest complete volume.
        assert survey["newest_volume"] == archive.name(
            WINDOW_END - timedelta(seconds=300))
        assert survey["newest_volume_end_time"] == iso_ms(
            WINDOW_END - timedelta(seconds=60))
        refused = survey["admission"]["refused_incomplete_at_clock"]
        assert [r["volume"] for r in refused] == [archive.name(WINDOW_END)]
        assert refused[0]["end_time"] == iso_ms(
            WINDOW_END + timedelta(seconds=240))
        assert iso(WINDOW_END) in refused[0]["reason"]
        assert survey["admission"]["clock"] == iso(WINDOW_END)

    def test_the_lag_is_the_age_of_the_newest_complete_data(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        survey = receipts["survey"]
        # 60 s from the admitted volume's last radial to the window end,
        # not the 300 s its key name is behind, and not the 0 s the
        # refused volume's key name would have read.
        assert survey["archive_lag_seconds"] == 60.0
        assert "last radial" in survey["archive_lag_definition"]

    def test_the_receipt_keeps_the_three_clocks_apart(
            self, tmp_path, monkeypatch):
        archive, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        newest = receipts["survey"]["survey_volumes"][-1]
        stamp = WINDOW_END - timedelta(seconds=300)
        assert newest["valid_time"] == iso(stamp)
        assert newest["start_time"] == iso_ms(stamp)
        assert newest["end_time"] == iso_ms(archive.end_of(stamp))
        assert newest["availability_time"] == iso_ms(
            archive.published_at(stamp))
        assert newest["complete"] is True
        assert receipts["survey"]["newest_volume_availability_time"] \
            == newest["availability_time"]

    def test_the_motion_baseline_is_between_the_scans(
            self, tmp_path, monkeypatch):
        _, receipts = replay(tmp_path, monkeypatch, iso(WINDOW_END))
        older, newer = receipts["survey"]["survey_volumes"]
        motion = receipts["survey"]["motion"]
        assert motion["baseline_seconds"] == (
            parse_iso(newer["start_time"])
            - parse_iso(older["start_time"])).total_seconds()

    def test_a_live_window_ends_no_later_than_the_newest_complete_volume(
            self, tmp_path, monkeypatch):
        archive, receipts = replay(tmp_path, monkeypatch, "latest")
        survey = receipts["survey"]
        # At the wall clock the volume being scanned is not on the shelf,
        # so the newest listed one is complete and is admitted as it is.
        newest_stamp = WALL - timedelta(seconds=300)
        assert survey["newest_volume"] == archive.name(newest_stamp)
        assert survey["admission"]["refused_incomplete_at_clock"] == []
        assert parse_iso(receipts["plan"]["plan"]["window_end"]) <= parse_iso(
            survey["newest_volume_end_time"])

    def test_a_pack_without_an_end_time_is_refused_with_the_remedy(
            self, tmp_path, monkeypatch, capsys):
        import tools.da_nowcast as front
        from gpuwm.obs import sweeps

        archive = FakeArchive(WINDOW_END, WALL)
        install(monkeypatch, archive)
        monkeypatch.setattr(front, "datetime", _FrozenDatetime)
        # A pack written by an rw_nexrad from before the instants existed
        # carries no end_time; the survey must not admit it on its start.
        monkeypatch.setattr(
            sweeps, "read_sweep_pack",
            lambda path: SimpleNamespace(
                site=SimpleNamespace(lat_deg=40.0, lon_deg=-90.0,
                                     alt_m=200.0),
                sweeps=()))
        with pytest.raises(SystemExit) as refusal:
            front.main(["run", "--site", SITE,
                        "--window-end", iso(WINDOW_END),
                        "--out", str(tmp_path / "case"),
                        "--stop-after", "survey"])
        message = str(refusal.value.code)
        assert "carries no end_time" in message
        assert "rebuild it from this tree" in message


class TestTheClockItself:
    def test_a_window_end_is_the_surveys_clock(self):
        from tools.da_nowcast import survey_clock

        clock = survey_clock(iso(WINDOW_END), now=WALL)
        assert clock.at == WINDOW_END
        assert clock.mode == "window"
        assert clock.window_end == WINDOW_END

    def test_latest_has_no_window_yet_so_it_is_the_wall(self):
        from tools.da_nowcast import survey_clock

        clock = survey_clock("latest", now=WALL)
        assert clock.at == WALL
        assert clock.mode == "wall"
        assert clock.window_end is None

    def test_a_malformed_window_end_is_refused_before_any_byte_moves(self):
        from tools.da_nowcast import survey_clock

        with pytest.raises(SystemExit):
            survey_clock("the day before yesterday", now=WALL)


# ---------------------------------------------------------------------------
# the whole door, on fakes: what lands in the receipt
# ---------------------------------------------------------------------------
#: The case shape the whole-run test drives.  Small on purpose: nothing
#: here integrates anything, and every number below is only ever read
#: back out of the receipt.
CYCLES, FREE_LEGS, MEMBERS = 2, 2, 2

WPS_NAMELIST = """&share
/
"""

CASE_TOML = """[[domain]]
name = "root"
nx = 136
ny = 134
dx = 3000.0
dy = 3000.0
nz = 49
history_interval_s = 3600.0

[fetch]
cycle = "{cycle}"
hours = 6
area = "35.0,-98.0,36.0,-97.0"
cadence = 3
"""


class StageDouble:
    """Every ``run_stage`` call, and the artifacts each one leaves.

    The door reads four things back off disk between stages: the case
    TOML and its WPS namelist from the wizard, the input manifest from
    the fetch, and ``proof.json`` from the preparation.  This writes
    those four and records the rest, so the run reaches its own receipt
    without a subprocess, a card or a byte of real data.
    """

    def __init__(self, case_dir: Path, case_name: str, cycle: str):
        self.case_dir = case_dir
        self.case_name = case_name
        self.cycle = cycle
        self.stages: list[str] = []

    def __call__(self, name, argv, *, cwd, receipts_dir, index):
        self.stages.append(name)
        if name == "domain":
            root = self.case_dir / "case"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{self.case_name}.toml").write_text(
                CASE_TOML.format(cycle=self.cycle), encoding="utf-8")
            (root / f"{self.case_name}.namelist.wps").write_text(
                WPS_NAMELIST, encoding="utf-8")
        elif name == "manifest":
            data = self.case_dir / "data"
            data.mkdir(parents=True, exist_ok=True)
            (data / "gfs-input-manifest.json").write_text(
                json.dumps({"files": []}), encoding="utf-8")
        elif name == "prepare":
            prepared = self.case_dir / "prepared"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps(
                {"prepared_cache": {"content_sha256": "0" * 64}}),
                encoding="utf-8")
        return {"stage": name, "index": index}


def whole_run(tmp_path: Path, monkeypatch,
              extra: tuple[str, ...] = ()) -> tuple[dict, StageDouble]:
    """One complete ``da_nowcast run`` on fakes; the receipt it wrote."""

    import tools.da_nowcast as front

    archive = FakeArchive(WINDOW_END, WALL)
    install(monkeypatch, archive)
    monkeypatch.setattr(front, "datetime", _FrozenDatetime)
    out = tmp_path / "case"
    case_name = f"nowcast_{SITE.lower()}_{WINDOW_END:%Y%m%d%H%M}"
    # The background cycle the wizard would be given; the fake TOML
    # quotes it back so the fetch stage's hints are the run's own.
    double = StageDouble(out, case_name, f"{WINDOW_END:%Y-%m-%d}T00")
    monkeypatch.setattr(front, "run_stage", double)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"not run: every stage is a double")
    geog = tmp_path / "WPS_GEOG"
    geog.mkdir()
    code = front.main([
        "run", "--site", SITE, "--window-end", iso(WINDOW_END),
        "--out", str(out), "--source", "gfs",
        "--cycles", str(CYCLES), "--cycle-seconds", "900",
        "--free-legs", str(FREE_LEGS), "--members", str(MEMBERS),
        "--bridge", str(bridge), "--geog-root", str(geog),
        "--no-verify", *extra])
    assert code == 0
    receipt = json.loads((out / "nowcast-receipt.json")
                         .read_text(encoding="utf-8"))
    return receipt, double


class TestTheReceiptTheRunWrites:
    def test_the_written_receipt_names_the_forecast_frames(
            self, tmp_path, monkeypatch):
        """Read off the file, not off the function that builds it."""
        receipt, _ = whole_run(tmp_path, monkeypatch)
        frames = receipt["outputs"]["forecast_frames"]
        assert frames["dir"].endswith("composites")
        assert Path(frames["dir"]).parent.name == "cycle"
        assert frames["free_legs"] == ["leg02", "leg03"]
        assert frames["leg_number_offset"] == 0
        assert "wrfout_legNN" in frames["wrfout"]
        assert "_dNN" in frames["nested"]
        # the written receipt carries the frame's limit as well as its
        # path, because the receipt is the machine-readable contract and
        # a tool reading it does not open the page beside it
        assert "one-level" in frames["vertical"]
        assert "three-dimensional leg state is not kept" in frames["vertical"]
        assert frames["measured_field"].startswith("composite reflectivity")

    def test_the_whole_run_reached_its_own_receipt(self, tmp_path,
                                                   monkeypatch):
        """The stages ran in order, so the receipt is a whole run's."""
        _, double = whole_run(tmp_path, monkeypatch)
        assert double.stages[:6] == ["domain", "authority", "fetch",
                                     "manifest", "prepare", "forecast"]
        assert double.stages[-2:] == ["cycle", "render"]

    def test_the_receipt_says_which_clock_the_survey_used(
            self, tmp_path, monkeypatch):
        receipt, _ = whole_run(tmp_path, monkeypatch)
        assert receipt["survey"]["clock"]["mode"] == "window"
        assert receipt["survey"]["clock"]["at"] == iso(WINDOW_END)

    def test_a_run_that_asked_for_no_nest_says_so(self, tmp_path,
                                                  monkeypatch):
        receipt, _ = whole_run(tmp_path, monkeypatch)
        assert receipt["sizing"]["nested_free_forecast"] is False
        assert receipt["sizing"]["nest"] == {"requested": False}

    def test_a_run_that_asked_for_a_nest_says_what_it_asked_for(
            self, tmp_path, monkeypatch):
        receipt, _ = whole_run(
            tmp_path, monkeypatch,
            extra=("--nest-half-width-km", "45", "--nest-members", "1"))
        assert receipt["sizing"]["nested_free_forecast"] is True
        nest = receipt["sizing"]["nest"]
        assert nest["half_width_km"] == 45.0
        assert nest["members"] == 1
