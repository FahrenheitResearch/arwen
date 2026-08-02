"""Initializing a run from a GFS FORECAST LEAD, not only from f000.

The defect these cover: ``rw-wps --source gfs`` refused any experiment
whose ``start_time`` was not the cycle itself, so a user who wanted the
f174..f240 window had to integrate 240 hours to reach it.  There is no
product reason for that -- a GFS f174 instantaneous record has the same
shape as an f000 one, and WPS/real initializes from forecast leads
routinely -- so ``start_time = cycle + K`` is admitted for any K the
fetched series carries, with one warning line and honest receipts.

Two hour vocabularies appear throughout and are never interchanged:
SOURCE leads (f000, f018, f174) name NOAA products, and MODEL forcing
offsets (0, 3, 6) count from ``start_time``.  They coincide exactly when
the lead is zero, which is why a lead-zero run is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from gpuwm import fetch
from gpuwm.gfs_direct import (
    initial_condition_provenance,
    resolve_initial_forecast_lead,
)
from gpuwm.prepared_single_domain_forecast import (
    _gfs_manifest_source_receipt,
    proof_initial_forecast_lead,
)


CYCLE = datetime(2026, 7, 29, 0, 0, 0)
LADDER = (0, 3, 6, 9, 12, 15, 18, 21, 24)


# ---------------------------------------------------------------------------
# The lead itself
# ---------------------------------------------------------------------------

def test_a_start_time_at_any_fetched_lead_resolves_to_that_lead():
    for lead in LADDER:
        assert resolve_initial_forecast_lead(
            start_time=CYCLE + timedelta(hours=lead),
            cycle_time=CYCLE, source_hours=LADDER) == lead


def test_the_lead_zero_case_is_still_exactly_the_cycle():
    assert resolve_initial_forecast_lead(
        start_time=CYCLE, cycle_time=CYCLE, source_hours=LADDER) == 0
    # And it is still the analysis, named as one.
    assert initial_condition_provenance(
        cycle_time=CYCLE, lead_hours=0) == {
        "schema": "gpuwm-gfs-initial-condition-provenance-v1",
        "cycle": "2026-07-29T00:00:00Z",
        "initial_forecast_lead_hours": 0,
        "model_start_time": "2026-07-29T00:00:00Z",
        "initial_condition_kind": "analysis",
        "forecast_generating_process_id": 81,
        "statement": "initialized from GFS cycle 2026-07-29T00:00:00Z "
                     "analysis (f000)",
    }


def test_a_lead_the_series_does_not_carry_is_refused_by_name():
    """Negative control: the refusal names the lead AND the fetched set."""

    with pytest.raises(ValueError) as caught:
        resolve_initial_forecast_lead(
            start_time=CYCLE + timedelta(hours=174),
            cycle_time=CYCLE, source_hours=LADDER)
    message = str(caught.value)
    assert "f174" in message
    assert "does not carry" in message
    # The set, so the user can see what they DO have.
    assert "f000, f003" in message and "f024" in message
    assert "--forecast-start-hour 174" in message


def test_a_start_before_the_cycle_and_a_part_hour_start_are_refused():
    with pytest.raises(ValueError, match="BEFORE GFS cycle"):
        resolve_initial_forecast_lead(
            start_time=CYCLE - timedelta(hours=3),
            cycle_time=CYCLE, source_hours=LADDER)
    with pytest.raises(ValueError, match="whole number of hours"):
        resolve_initial_forecast_lead(
            start_time=CYCLE + timedelta(hours=3, minutes=30),
            cycle_time=CYCLE, source_hours=LADDER)


def test_a_forecast_lead_is_never_relabelled_as_an_analysis():
    receipt = initial_condition_provenance(cycle_time=CYCLE, lead_hours=174)
    assert receipt["initial_condition_kind"] == "forecast"
    assert receipt["forecast_generating_process_id"] == 96
    assert receipt["cycle"] == "2026-07-29T00:00:00Z"
    assert receipt["model_start_time"] == "2026-08-05T06:00:00Z"
    assert receipt["statement"] == (
        "initialized from GFS cycle 2026-07-29T00:00:00Z at lead f174: "
        "the initial condition is itself a 174 h forecast")
    # The word "analysis" must not appear anywhere in a lead receipt.
    assert "analysis" not in json.dumps(receipt)


# ---------------------------------------------------------------------------
# Fetch planning
# ---------------------------------------------------------------------------

def test_a_fetch_window_may_begin_at_a_lead_and_f000_is_unchanged():
    # Unchanged: --hours is the window LENGTH and the default start is 0.
    assert fetch.gfs_forecast_hours(6, 3) == (0, 3, 6)
    assert fetch.gfs_forecast_hours(6, 3, 0) == (0, 3, 6)
    assert fetch.gfs_forecast_hours(2, 1, None) == (0, 1, 2)
    # New: the same window, 174 hours into the forecast.
    assert fetch.gfs_forecast_hours(66, 3, 174)[:2] == (174, 177)
    assert fetch.gfs_forecast_hours(66, 3, 174)[-1] == 240
    assert fetch.gfs_forecast_hours(6, 3, 24) == (24, 27, 30)


def test_an_off_cadence_or_over_horizon_lead_is_refused():
    with pytest.raises(ValueError, match="not on the 3 h cadence"):
        fetch.gfs_forecast_hours(6, 3, 25)
    with pytest.raises(ValueError, match="horizon"):
        fetch.gfs_forecast_hours(6, 3, 381)
    with pytest.raises(ValueError, match="nonnegative forecast lead"):
        fetch.gfs_forecast_hours(6, 3, -3)


def test_the_fetch_hint_table_carries_the_lead_and_scopes_it():
    fetch.validate_fetch_hints(
        {"source": "gfs", "hours": 6, "forecast_start_hour": 174},
        source="case.toml")
    fetch.validate_fetch_hints(
        {"source": "hrrr", "cycle": "2026-07-29T18", "hours": 6,
         "forecast_start_hour": 3},
        source="case.toml")
    with pytest.raises(ValueError, match="source = gfs\\|gdas\\|hrrr only"):
        fetch.validate_fetch_hints(
            {"source": "era5", "hours": 6, "forecast_start_hour": 3},
            source="case.toml")
    with pytest.raises(ValueError, match="nonnegative forecast lead"):
        fetch.validate_fetch_hints(
            {"source": "gfs", "hours": 6, "forecast_start_hour": -1},
            source="case.toml")
    # A rotten HRRR hint: a 13Z cycle stops at f18, so f12 + 9 h is a
    # window NOAA never published.  Caught at config load, not at the
    # download, and in the words the fetch itself would use.
    with pytest.raises(ValueError, match="horizon f18"):
        fetch.validate_fetch_hints(
            {"source": "hrrr", "cycle": "2026-07-29T13", "hours": 9,
             "forecast_start_hour": 12},
            source="case.toml")


# ---------------------------------------------------------------------------
# The same lead, on HRRR
# ---------------------------------------------------------------------------

def test_an_hrrr_fetch_window_may_begin_at_a_lead_and_f00_is_unchanged():
    """``--hours`` stays the LENGTH; the default start is still f00.

    The engine below this has been lead-aware since the source-window
    contract landed -- ``hrrr_source_window`` keys off ``start_hour``,
    the bridge's series validation keys off ``observed[0]``, and the
    hierarchy's forcing horizon is model-relative.  What was missing was
    a way to say it on the front doors: this function took no ``start``
    at all and hardcoded ``range(0, hours + 1)``.
    """

    extended = datetime(2026, 7, 29, 18)   # 00/06/12/18Z reach f48
    standard = datetime(2026, 7, 29, 13)   # every other cycle stops at f18

    assert fetch.hrrr_forecast_hours(3, extended) == (0, 1, 2, 3)
    assert fetch.hrrr_forecast_hours(3, extended, 0) == (0, 1, 2, 3)
    assert fetch.hrrr_forecast_hours(3, extended, None) == (0, 1, 2, 3)
    assert fetch.hrrr_forecast_hours(3, extended, 6) == (6, 7, 8, 9)
    assert fetch.hrrr_forecast_hours(2, standard, 16) == (16, 17, 18)


def test_an_hrrr_lead_past_the_cycle_horizon_is_refused_by_name():
    standard = datetime(2026, 7, 29, 13)
    with pytest.raises(ValueError, match="horizon f18"):
        fetch.hrrr_forecast_hours(3, standard, 16)
    with pytest.raises(ValueError, match="horizon f48"):
        fetch.hrrr_forecast_hours(3, datetime(2026, 7, 29, 18), 46)
    with pytest.raises(ValueError, match="nonnegative forecast lead"):
        fetch.hrrr_forecast_hours(3, standard, -1)


def test_the_hrrr_lead_flag_is_no_longer_refused_for_the_isobaric_ladder(
        tmp_path, capsys):
    """The refusal a lead used to meet named a flag nobody had typed.

    ``--forecast-start-hour`` rode the level-ladder bundle, so an HRRR
    request for f06 was declined with a sentence about isobaric ladders.
    ERA5 still refuses it -- for the reason that actually applies.
    """

    from gpuwm.cli import main as cli_main

    assert cli_main(
        ["fetch", "--source", "era5", "--cycle", "2026-07-29T18",
         "--hours", "6", "--area", "30,-100,40,-90",
         "--forecast-start-hour", "6", "--out", str(tmp_path)]) != 0
    message = capsys.readouterr().err
    assert "reanalysis" in message
    assert "isobaric" not in message


# ---------------------------------------------------------------------------
# Manifest authoring over a tail of an existing fetch
# ---------------------------------------------------------------------------

def _fetched_directory(tmp_path: Path, hours=LADDER) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    files = []
    for hour in hours:
        name = f"gfs.t00z.pgrb2.0p25.f{hour:03d}.subset.grib2"
        (out / name).write_bytes(f"GRIB-{hour}".encode())
        files.append({
            "name": name, "role": "gfs-subset", "forecast_hour": hour,
            "bytes": (out / name).stat().st_size,
            "sha256": fetch.sha256_file(out / name), "url": None,
        })
    (out / "gfs-series.tsv").write_text("".join(
        f"{hour}\t{item['name']}\t{81 if hour == 0 else 96}\n"
        for hour, item in zip(hours, files)), encoding="utf-8")
    (out / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA,
        "source": "gfs",
        "cycle": "2026-07-29T00:00:00Z",
        "forecast_hours": list(hours),
        "area": None,
        "files": files,
        "payload_bytes": sum(item["bytes"] for item in files),
    }), encoding="utf-8")
    for name in ("bridge", "namelist.wps", "case.toml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    return out


def test_a_manifest_may_be_authored_over_the_tail_of_an_existing_fetch(
        tmp_path):
    """The tester's own situation: f000..f024 already on disk, run at f018.

    Nothing is re-downloaded and nothing existing is edited: a second,
    hash-bound series naming only the tail is written beside the first,
    and the manifest binds that one.
    """

    out = _fetched_directory(tmp_path)
    path, digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=tmp_path / "bridge",
        wps_namelist=tmp_path / "namelist.wps",
        experiment_config=tmp_path / "case.toml",
        forecast_start_hour=18, progress=lambda *_: None)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["schema"] == fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
    grib_roles = sorted(
        role for role in payload["files"] if role.startswith("grib-f"))
    assert grib_roles == ["grib-f018", "grib-f021", "grib-f024"]
    # The cycle is the CYCLE, never the start time.
    assert payload["source"]["cycle"] == "2026-07-29T00:00:00Z"

    tail = out / "gfs-series-f018.tsv"
    assert payload["files"]["series"]["name"] == tail.name
    assert tail.read_text(encoding="utf-8").splitlines()[0].startswith(
        "18\t")
    assert all(line.endswith("\t96")
               for line in tail.read_text(encoding="utf-8").splitlines())
    # The original series is untouched and still readable.
    assert (out / "gfs-series.tsv").read_text(
        encoding="utf-8").startswith("0\t")


def test_authoring_over_a_tail_refuses_a_lead_or_a_window_it_lacks(tmp_path):
    out = _fetched_directory(tmp_path)
    common = dict(
        out=out, bridge=tmp_path / "bridge",
        wps_namelist=tmp_path / "namelist.wps",
        experiment_config=tmp_path / "case.toml",
        progress=lambda *_: None)
    with pytest.raises(ValueError) as caught:
        fetch.author_gfs_front_door_manifest(
            forecast_start_hour=174, **common)
    assert "f000" in str(caught.value) and "f024" in str(caught.value)
    # f024 is fetched, but nothing follows it: no boundary time at all.
    with pytest.raises(ValueError, match="at least one lateral boundary"):
        fetch.author_gfs_front_door_manifest(
            forecast_start_hour=24, **common)


# ---------------------------------------------------------------------------
# The forecast runner's own authority checks
# ---------------------------------------------------------------------------

def _experiment(start_time: datetime, p_top: float = 10000.0):
    return SimpleNamespace(
        start_time=start_time,
        vertical=SimpleNamespace(p_top=p_top))


def _manifest(cycle: str = "2026-07-29T00:00:00Z"):
    return {"source": {"model": "GFS", "product": "pgrb2.0p25",
                       "cycle": cycle}}


def _proof(lead: int):
    return {"initial_condition": initial_condition_provenance(
        cycle_time=CYCLE, lead_hours=lead)}


def test_the_runner_derives_the_cycle_by_subtracting_the_declared_lead():
    receipt = _gfs_manifest_source_receipt(
        _manifest(), _experiment(CYCLE + timedelta(hours=18)), _proof(18))
    assert receipt["identity"]["cycle"] == "2026-07-29T00:00:00Z"

    # Lead zero is the pre-existing behaviour, unchanged.
    receipt = _gfs_manifest_source_receipt(
        _manifest(), _experiment(CYCLE), _proof(0))
    assert receipt["identity"]["cycle"] == "2026-07-29T00:00:00Z"

    # A proof from before this release carries no provenance block, and
    # there is exactly one lead it can have meant.
    receipt = _gfs_manifest_source_receipt(
        _manifest(), _experiment(CYCLE), {})
    assert receipt["identity"]["cycle"] == "2026-07-29T00:00:00Z"


def test_the_runner_refuses_a_manifest_from_a_different_cycle():
    with pytest.raises(ValueError, match="forecast lead f018"):
        _gfs_manifest_source_receipt(
            _manifest("2026-07-28T00:00:00Z"),
            _experiment(CYCLE + timedelta(hours=18)), _proof(18))


def test_the_runner_refuses_a_proof_that_calls_a_lead_an_analysis():
    """Negative control for the relabelling this feature must never do."""

    lying = {"initial_condition": {
        "schema": "gpuwm-gfs-initial-condition-provenance-v1",
        "cycle": "2026-07-29T00:00:00Z",
        "initial_forecast_lead_hours": 174,
        "model_start_time": "2026-08-05T06:00:00Z",
        "initial_condition_kind": "analysis",
        "forecast_generating_process_id": 81,
        "statement": "initialized from GFS cycle 2026-07-29T00:00:00Z",
    }}
    with pytest.raises(ValueError, match="is not an\\s+analysis"):
        proof_initial_forecast_lead(lying)

    with pytest.raises(ValueError, match="unreadable initial forecast lead"):
        proof_initial_forecast_lead(
            {"initial_condition": {"initial_forecast_lead_hours": -3}})
    with pytest.raises(ValueError, match="provenance is malformed"):
        proof_initial_forecast_lead({"initial_condition": 174})


def test_the_runner_refuses_provenance_that_contradicts_the_experiment():
    """cycle + lead must BE start_time, in the document, not by luck."""

    proof = _proof(18)
    with pytest.raises(ValueError, match="disagrees with"):
        _gfs_manifest_source_receipt(
            _manifest(), _experiment(CYCLE + timedelta(hours=18)),
            {"initial_condition": {
                **proof["initial_condition"],
                "model_start_time": "2026-07-29T21:00:00Z"}})


# ---------------------------------------------------------------------------
# The front door itself, entered the way rw-wps enters it
# ---------------------------------------------------------------------------

def _front_door_case(tmp_path, *, start_hour: int, series_hours,
                     run_hours: int = 3):
    """A wizard config at cycle+K, plus the series/manifest pair beside it.

    Everything up to the decode is real: the config comes out of the
    wizard, the manifest binds each file by its own sha256, and the
    front door verifies it.  The bridge is a stand-in, so a case that
    passes the lead gate proceeds to the decode and fails THERE -- which
    is precisely the evidence that admission happened.
    """

    from gpuwm.cli import main as cli_main
    from gpuwm.physics_compat import WSM6_PROFILE_ID

    config = tmp_path / "wizard" / "case.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main([
        "domain", "--point=39.7,-96.6", "--card", "24gb", "--ladder", "12",
        "--source", "gfs", "--cycle", "2026-07-29T18",
        "--physics-profile", WSM6_PROFILE_ID,
        "--hours", str(run_hours),
        "--forecast-start-hour", str(start_hour),
        "--out", str(config)]) == 0

    data = tmp_path / "data"
    data.mkdir()
    names = {}
    for hour in series_hours:
        name = f"f{hour:03d}.grib2"
        (data / name).write_bytes(f"GRIB-{hour}".encode())
        names[hour] = name
    series = data / "series.tsv"
    series.write_text("".join(
        f"{hour}\t{names[hour]}\t{81 if hour == 0 else 96}\n"
        for hour in series_hours), encoding="utf-8")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stand-in")
    # The door checks os.access(..., X_OK) before it reaches any of the
    # gates under test, and on POSIX a freshly written file does not
    # have it.  (On Windows X_OK is true for every existing file, which
    # is how a first draft of this passed there and refused here.)
    bridge.chmod(bridge.stat().st_mode | stat.S_IXUSR)
    # The wizard's OWN namelist.wps, emitted beside the config, so the
    # geometry contract this door checks is the real one.
    namelist = config.parent / f"{config.stem}.namelist.wps"
    assert namelist.is_file()
    static_input = tmp_path / "static.npz"
    static_input.write_bytes(b"static")
    static_receipt = tmp_path / "static.json"
    static_receipt.write_text("{}", encoding="utf-8")

    roles = {
        "series": series, "bridge": bridge, "wps_namelist": namelist,
        "experiment_config": config, "static_input": static_input,
        "static_receipt": static_receipt,
    }
    roles.update({f"grib-f{hour:03d}": data / names[hour]
                  for hour in series_hours})
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text(json.dumps({
        "schema": "gpuwm-gfs-direct-input-manifest-v1",
        "source": {"model": "GFS", "product": "pgrb2.0p25",
                   "cycle": "2026-07-29T18:00:00Z"},
        "files": {
            role: {"name": path.name,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in roles.items()},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [
        "--series", str(series),
        "--cycle", "2026-07-29_18:00:00",
        "--bridge", str(bridge),
        "--wps-namelist", str(namelist),
        "--experiment-config", str(config),
        "--static-input", str(static_input),
        "--static-receipt", str(static_receipt),
        "--input-manifest", str(manifest),
        "--input-manifest-sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "--output-root", str(tmp_path / "prepared"),
    ]


def test_a_lead_the_fetch_lacks_is_refused_at_the_door_as_a_sentence(
        tmp_path, capsys):
    """Hard refusal #1, and it arrives as one sentence, not a traceback."""

    from gpuwm.gfs_direct import main as gfs_main

    argv = _front_door_case(tmp_path, start_hour=6, series_hours=(0, 3))
    capsys.readouterr()
    assert gfs_main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("rw-wps --source gfs: ")
    assert "f006" in lines[0] and "f000, f003" in lines[0]


def test_a_lead_the_fetch_carries_is_admitted_with_one_warning(
        tmp_path, capsys):
    """Warn, not block.  The run proceeds to the decode and fails there.

    On the base commit this same case died at "GFS cycle must equal
    experiment start_time" without ever reaching the bridge.
    """

    from gpuwm.gfs_direct import main as gfs_main

    argv = _front_door_case(tmp_path, start_hour=3, series_hours=(0, 3, 6))
    capsys.readouterr()
    assert gfs_main(argv) == 2
    err = capsys.readouterr().err
    assert "warning: initialized from GFS cycle 2026-07-29T18:00:00Z at " \
           "lead f003: the initial condition is itself a 3 h forecast" in err
    # It got PAST the lead gate.  The refusal it did hit is a stand-in
    # INPUT further down the door (the static receipt this fixture does
    # not build), which means the manifest verification, the physics
    # gate, and the geometry and vertical contracts all ran at a nonzero
    # lead.  The one thing it is not is the start_time equality this
    # release removed.
    assert "native static receipt" in err
    assert "must equal experiment start_time" not in err
    # And the unused f000 prefix is named rather than silently decoded.
    assert "before f003" in err and "f000" in err


# ---------------------------------------------------------------------------
# The lead through the front door: cadence, and the two ends of the GFS axis
#
# An independent tester drove published 1.4.0 from PyPI and found the lead
# feature half-reachable: the wizard wrote a cadence its own printed fetch
# refuses, `gpuwm go` dropped the cadence key entirely so a config asking
# for hourly boundaries ran with 3-hourly ones at exit 0, and the two ends
# of the GFS forecast axis -- the f120 cadence break and NOMADS retention --
# were reported as a publication delay and as a urllib traceback.
# ---------------------------------------------------------------------------

def test_the_wizard_emits_a_cadence_that_contains_the_lead(tmp_path, capsys):
    """C-08: `--forecast-start-hour 4` used to write `cadence = 3`.

    Step 1 of the wizard's own printed recipe then exited 2 with "f004 is
    not on the 3 h cadence", and for two leads in every three the
    one-command `gpuwm go` path could not be made to work at all.
    """
    import tomllib
    from gpuwm.cli import main as cli_main

    out = tmp_path / "leg4.toml"
    assert cli_main([
        "domain", "--point=35.5,-97.5", "--vram-gib", "12", "--root-dx", "12",
        "--hours", "2", "--source", "gfs", "--cycle", "2026-08-01T00",
        "--forecast-start-hour", "4", "--out", str(out)]) == 0
    printed = capsys.readouterr().out

    table = tomllib.loads(out.read_text(encoding="utf-8"))["fetch"]
    assert table["cadence"] == 1
    assert table["forecast_start_hour"] == 4
    # The planner that would refuse it accepts it.
    assert fetch.gfs_forecast_hours(
        table["hours"], table["cadence"], table["forecast_start_hour"])[0] == 4
    # And the printed command carries the cadence, or it downloads a
    # different window from the one the config was written for.
    line = next(l for l in printed.splitlines() if "gpuwm fetch" in l)
    assert "--cadence 1" in line
    assert "--forecast-start-hour 4" in line


def test_a_lead_on_the_default_grid_still_emits_the_default_cadence(
        tmp_path, capsys):
    """Unchanged where it was already right: f006 stays a 3 h window."""
    import tomllib
    from gpuwm.cli import main as cli_main

    out = tmp_path / "leg6.toml"
    assert cli_main([
        "domain", "--point=35.5,-97.5", "--vram-gib", "12", "--root-dx", "12",
        "--hours", "3", "--source", "gfs", "--cycle", "2026-08-01T00",
        "--forecast-start-hour", "6", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert tomllib.loads(out.read_text(encoding="utf-8"))["fetch"][
        "cadence"] == 3
    line = next(l for l in printed.splitlines() if "gpuwm fetch" in l)
    assert "--cadence" not in line


def test_go_carries_the_configs_cadence_into_its_fetch(tmp_path):
    """C-07: the key `go` executed five of six of."""
    from gpuwm import go_cli

    plan = {"source": "gfs", "cycle": "2026-08-01T00", "hours": 3,
            "area": "16,-118,54,-76", "data": tmp_path / "data",
            "cadence": 1, "forecast_start_hour": 4}
    command = go_cli.fetch_command(plan)
    assert command[command.index("--cadence") + 1] == "1"
    assert command[command.index("--forecast-start-hour") + 1] == "4"
    # Absent cadence stays absent: HRRR refuses the flag outright.
    assert "--cadence" not in go_cli.fetch_command({**plan, "cadence": None})


def test_go_refuses_a_run_whose_boundaries_are_not_what_the_config_asked(
        tmp_path):
    """C-07's other half: exit 0 and "validity PASS" on a 3x coarser clock."""
    from gpuwm import go_cli

    plan = {"cadence": 1, "config": tmp_path / "cad1.toml",
            "run": tmp_path / "run"}
    coarse = {"input": {"boundary_interval_seconds": 10800}}
    message = go_cli._boundary_interval_refusal(plan, coarse)
    assert message is not None
    assert "3 h lateral boundaries" in message
    assert "cadence = 1" in message
    # Agreement is silent, and so is a receipt that does not record it.
    assert go_cli._boundary_interval_refusal(
        plan, {"input": {"boundary_interval_seconds": 3600}}) is None
    assert go_cli._boundary_interval_refusal(plan, {}) is None
    assert go_cli._boundary_interval_refusal(
        {**plan, "cadence": None}, coarse) is None


def test_the_gfs_hourly_cadence_break_is_named_as_a_break(tmp_path):
    """C-02: f121/f122/f124 are permanent 404s, not a publication delay."""
    # The last legal hourly window.
    assert fetch.gfs_forecast_hours(4, 1, 116)[-1] == 120
    with pytest.raises(ValueError) as caught:
        fetch.gfs_forecast_hours(4, 1, 118)
    message = str(caught.value)
    assert "published every hour only through f120" in message
    assert "--cadence 3" in message
    assert "not published" not in message
    assert "yet" not in message
    # 3-hourly is unaffected all the way to the horizon.
    assert fetch.gfs_forecast_hours(3, 3, 381)[-1] == 384


def test_an_off_grid_lead_past_f120_is_told_where_the_3h_grid_is():
    with pytest.raises(ValueError, match="f117 or f120"):
        fetch.gfs_forecast_hours(6, 1, 118)


def test_a_cycle_nomads_no_longer_serves_refuses_in_one_sentence():
    """C-09: the probe reads S3 (years), the download reads NOMADS (days)."""
    from urllib.error import HTTPError

    stale = fetch.nomads_reach_refusal(
        "gfs", datetime(2020, 1, 1), 0,
        HTTPError("http://x", 403, "Forbidden", {}, None))
    assert "no longer serves it" in stale
    assert "HTTP 403" in stale
    assert "rolling window" in stale
    # A failure that is not about age keeps the transport's own answer
    # rather than inventing a retention story.
    fresh = fetch.nomads_reach_refusal(
        "gfs", datetime.now(), 0,
        HTTPError("http://x", 500, "Server Error", {}, None))
    assert "no longer serves it" not in fresh
    assert "HTTP 500" in fresh


def test_a_rotten_cadence_and_lead_pairing_is_caught_at_config_load():
    """The 1.4.0 emission, met at load instead of at the download."""
    with pytest.raises(ValueError, match="not on the 3 h cadence"):
        fetch.validate_fetch_hints(
            {"source": "gfs", "hours": 3, "cadence": 3,
             "forecast_start_hour": 4}, source="case.toml")
    # The pairing the wizard emits now loads clean.
    fetch.validate_fetch_hints(
        {"source": "gfs", "hours": 2, "cadence": 1,
         "forecast_start_hour": 4}, source="case.toml")
    # And the f120 break is caught here too.
    with pytest.raises(ValueError, match="every hour only through f120"):
        fetch.validate_fetch_hints(
            {"source": "gfs", "hours": 4, "cadence": 1,
             "forecast_start_hour": 118}, source="case.toml")
