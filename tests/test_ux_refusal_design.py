"""Refusal-design findings from the 2026-08-18 persona walks: R1, N14, N11.

Each test drives the same front door the walk drove and pins the
sentence that door owes.  The Refusals law is the spec: a refusal names
the concrete breakage it prevents AND prints a remedy the user can
type; a nonzero exit with neither is a defect row.

* R1  -- ``gpuwm check`` on a wizard-emitted config with no declared
         budget and no measurable card exited 2 after three
         "not measured" lines, with no sentence saying why or what to
         type next (replay walk C, step 1).
* N14 -- the wizard refused arithmetic it derived itself: ``--root-dx
         9`` chooses dt = 45 s and the profile's cudt = 300 s, then the
         loader refuses 300/45 = 20/3.  The user supplied neither
         number, so the wizard adjusts its own derivation and says so.
* N11 -- ``gpuwm fetch --source gfs`` left a pile (three grib2 + tsv +
         manifest) instead of the four-file front door DATA.md promises,
         and the handoff needed a second ``--author-front-door-manifest``
         call.  The fetch now writes the same four files the table
         routes write, and the prep door authors the digest binding
         itself when ``--source-manifest`` is omitted.

R2 (the gfs-direct eta-ladder refusal with no remedy) lives in
``tests/test_gfs_direct.py`` beside the vertical-contract suite, and
N18 (the run-folder line on a refused real ``gpuwm go``) lives in
``tests/test_go_chain.py`` beside the gate-ordering suite.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import pytest

from gpuwm import fetch
from gpuwm.cli import main as cli_main


def _wizard_emit(tmp_path, *extra, name="area.toml"):
    """One real ``gpuwm domain`` emission, the walk's own door."""

    out = tmp_path / name
    rc = cli_main([
        "domain", "--point=35.30,-97.50", "--card", "24gb",
        "--source", "gfs", "--cycle", "2026-08-18T18",
        "--out", str(out), *extra])
    return rc, out


# ---------------------------------------------------------------------------
# R1: the check door's fail-closed exit speaks
# ---------------------------------------------------------------------------

def test_check_with_nothing_to_verify_against_names_the_missing_budget(
        tmp_path, monkeypatch, capsys):
    """Exit 2 stays, and it now arrives with a sentence and a remedy.

    The wizard prints ``gpuwm check CONFIG`` as its step 2.  On a box
    with no measurable card and no declared budget every gate is
    "not measured" and the command exits 2 -- which is right (fail
    closed) and was silent (defect).  The refusal must name the missing
    budget declaration and print the exact lines that supply one: the
    check flags themselves, and the wizard flags that print the filled-in
    form.
    """

    rc, config = _wizard_emit(tmp_path)
    assert rc == 0
    capsys.readouterr()

    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy fails
    rc = cli_main(["check", str(config)])
    captured = capsys.readouterr()
    assert rc == 2
    spoken = captured.err
    assert "REFUSED" in spoken
    assert "no VRAM budget was declared" in spoken
    assert "--budget-gib" in spoken and "--vram-gib" in spoken
    assert str(config) in spoken, "the remedy is a line the user can type"
    assert "gpuwm domain --card" in spoken
    assert "Traceback" not in spoken


def test_check_with_a_declared_budget_still_passes_quietly(
        tmp_path, monkeypatch, capsys):
    """The declared form the wizard's step-2 comment offers still works."""

    rc, config = _wizard_emit(tmp_path)
    assert rc == 0
    capsys.readouterr()

    monkeypatch.setitem(sys.modules, "cupy", None)
    rc = cli_main(["check", str(config),
                   "--budget-gib", "19.45", "--vram-gib", "24"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "REFUSED" not in captured.err


# ---------------------------------------------------------------------------
# N14: the wizard never refuses its own derivation
# ---------------------------------------------------------------------------

def test_root_dx_9_emits_a_whole_step_cudt_and_says_so(tmp_path, capsys):
    """dt = 45 s and cudt = 300 s are both wizard-derived; the wizard
    reconciles them (315 s = 7 steps) instead of exiting 2."""

    rc, config = _wizard_emit(tmp_path, "--root-dx", "9")
    printed = capsys.readouterr().out
    assert rc == 0, "the wizard refused arithmetic it derived itself"
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    root = raw["domain"][0]
    assert root["time_step"] == 45
    assert root["cudt_minutes"] == 5.25
    assert "cudt" in printed, "the adjustment is spoken, not silent"
    assert "5.25" in printed


def test_root_dx_12_keeps_the_profile_cadence_verbatim(tmp_path, capsys):
    """dt = 60 s divides 300 s: nothing is adjusted, nothing is said."""

    rc, config = _wizard_emit(tmp_path, "--root-dx", "12")
    printed = capsys.readouterr().out
    assert rc == 0
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert raw["domain"][0]["cudt_minutes"] == 5.0
    assert "adjusted" not in printed


def test_cadence_snap_survives_a_clock_whose_minutes_need_searching():
    """A tropical 7 km clock (17.5 s) has no representable cadence at
    round(300/17.5) = 17 steps; the snap must land on one whose minutes
    round-trip exactly through the float the TOML carries."""

    from gpuwm.domain_wizard import snap_cadences_to_clock

    physics = {"radt": 12.0, "cu_physics": 1, "cudt_minutes": 5.0}
    snapped, notes = snap_cadences_to_clock(Fraction(35, 2), physics)
    minutes = snapped["cudt_minutes"]
    seconds = Fraction(minutes) * 60
    steps = seconds / Fraction(35, 2)
    assert steps.denominator == 1, "the loader's own arithmetic"
    assert notes, "an adjustment this real is spoken"
    # And a clock that already divides everything makes no note.
    same, silent = snap_cadences_to_clock(Fraction(60), physics)
    assert same == physics
    assert silent == ()


# ---------------------------------------------------------------------------
# N11: the GFS fetch leaves the four-file front door
# ---------------------------------------------------------------------------

def _grib2_stream(count):
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * count


def _fetched_gfs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch, "gfs_live_index", lambda *args, **kwargs: None)
    from tools import download_gfs_native_subset as gfs_transport

    monkeypatch.setattr(
        gfs_transport, "_download",
        lambda url, destination, **kw: destination.write_bytes(
            _grib2_stream(fetch.GFS_SUBSET_RECORD_COUNT)))
    out = tmp_path / "gfs"
    fetch.fetch_gfs(
        cycle=datetime(2026, 8, 18, 18), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    return out


def test_gfs_fetch_writes_the_four_file_front_door(tmp_path, monkeypatch):
    """SHA256SUMS + inputs.txt + prep-command.txt + fetch-manifest.json:
    the same contract every table route leaves (DATA.md)."""

    out = _fetched_gfs_dir(tmp_path, monkeypatch)

    assert (out / fetch.FETCH_MANIFEST_NAME).is_file()
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    named = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
             for line in sums}
    for hour in (0, 3, 6):
        name = f"gfs.t18z.pgrb2.0p25.f{hour:03d}.subset.grib2"
        assert named[name] == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()
    assert "gfs-series.tsv" in named

    inputs = (out / "inputs.txt").read_text(encoding="utf-8").splitlines()
    assert [Path(line).name for line in inputs] == [
        f"gfs.t18z.pgrb2.0p25.f{hour:03d}.subset.grib2"
        for hour in (0, 3, 6)]

    command = (out / "prep-command.txt").read_text(encoding="utf-8")
    assert "gpuwm prep" in command
    assert "--source gfs" in command
    assert "--gfs-series" in command
    assert "--cycle 2026-08-18_18:00:00" in command
    assert "yours to supply" in command
    assert "--wps-namelist" in command and "--experiment-config" in command


def test_bare_prep_follows_the_fetch_with_no_author_call(
        tmp_path, monkeypatch, capsys):
    """The prep door authors and digest-binds the input manifest itself
    when --source-manifest is omitted, so the walk's step 22 vanishes."""

    from gpuwm.source_cli import main as prep_main

    out = _fetched_gfs_dir(tmp_path, monkeypatch)
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"hashed, never launched: --dry-run")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text("[run]\n", encoding="utf-8")

    rc = prep_main([
        "--source", "gfs",
        "--gfs-series", str(out / "gfs-series.tsv"),
        "--cycle", "2026-08-18_18:00:00",
        "--wps-namelist", str(wps),
        "--experiment-config", str(config),
        "--geog-root", str(tmp_path / "WPS_GEOG"),
        "--output-root", str(tmp_path / "prepared"),
        "--bridge", str(bridge),
        "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    manifest = out / fetch.GFS_INPUT_MANIFEST_NAME
    assert manifest.is_file(), "the door authored the binding itself"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    printed = captured.out.replace("\\", "/")
    assert str(manifest).replace("\\", "/") in printed, (
        "the composed command binds the authored manifest")
    assert digest in captured.out
    assert "authored" in captured.err


def test_prep_without_a_fetch_directory_still_names_both_doors(
        tmp_path, capsys):
    """No fetched directory to author from: the refusal names the fetch
    and the explicit --source-manifest pair, never a bare demand."""

    from gpuwm.source_cli import main as prep_main

    series = tmp_path / "empty" / "gfs-series.tsv"
    series.parent.mkdir()
    series.write_text("0\tmissing.grib2\t81\n", encoding="utf-8")
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"hashed, never launched")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text("[run]\n", encoding="utf-8")

    rc = prep_main([
        "--source", "gfs",
        "--gfs-series", str(series),
        "--cycle", "2026-08-18_18:00:00",
        "--wps-namelist", str(wps),
        "--experiment-config", str(config),
        "--geog-root", str(tmp_path / "WPS_GEOG"),
        "--output-root", str(tmp_path / "prepared"),
        "--bridge", str(bridge),
        "--dry-run"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "run the fetch first" in captured.err
    assert "--source-manifest" in captured.err
    assert "Traceback" not in captured.err
