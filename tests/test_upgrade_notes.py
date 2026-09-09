"""The release notes and the doctor's upgrade note say the same things.

THE BREAKAGE THIS PREVENTS: the 2.7.0 candidate carried seventeen
default-on result changes (map-factor mixing, total-water pressure,
km_opt=2 surface drag, KF shallow 2400, NSSL/Morrison/GF/P3/MYNN
transcription corrections, ERA5 lake water, HRRR soil downscaling, ...)
and two classes of file that stop loading (every 2.6.5 checkpoint, a
namelist without input_from_file), and neither the CHANGELOG's release
section nor ``gpuwm doctor``'s upgrade note mentioned one of them.  The
two surfaces were written independently, so each could be complete
while the other said nothing.

:mod:`gpuwm.upgrade_notes` is the one table; this file pins CHANGELOG.md
to it entry by entry and drives the doctor through both of its doors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import doctor, upgrade_notes, whats_changed

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def _release_section(text: str, release: str) -> str:
    """The CHANGELOG text between ``## <release>`` and the next ``## ``."""

    start = text.index(f"## {release} (")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _require_changelog() -> str:
    if not CHANGELOG.is_file():
        pytest.skip("the CHANGELOG pin needs the source tree, not an install")
    return CHANGELOG.read_text(encoding="utf-8")


@pytest.mark.parametrize("release", sorted(upgrade_notes.RESULTS_THAT_MOVE))
def test_the_changelog_carries_every_recorded_entry_verbatim(release):
    """Both sections, every entry, as a `- ` bullet, inside the release's
    own section and not merely somewhere in the file."""

    section = _release_section(_require_changelog(), release)
    assert f"### {upgrade_notes.results_heading(release)}" in section
    assert f"### {upgrade_notes.COMPATIBILITY_HEADING}" in section
    missing = [entry for entry in (upgrade_notes.results_that_move(release)
                                   + upgrade_notes.restart_and_import(release))
               if f"- {entry}\n" not in section]
    assert not missing, (
        "CHANGELOG.md's %s section lacks %d entr%s recorded in "
        "gpuwm/upgrade_notes.py; paste upgrade_notes.changelog_sections(%r) "
        "in:\n  %s" % (release, len(missing),
                       "y" if len(missing) == 1 else "ies", release,
                       "\n  ".join(entry[:80] for entry in missing)))


def test_the_changelog_sections_are_exactly_the_generated_text():
    """Not just present: the block is byte for byte what the table says,
    so an entry edited in the CHANGELOG alone fails here too."""

    section = _release_section(_require_changelog(), "2.7.0")
    assert upgrade_notes.changelog_sections("2.7.0") in section


def test_every_entry_names_a_configuration_and_carries_no_dash_punctuation():
    """House style for release text: terse, config-named, no em-dashes and
    no double-hyphen used as punctuation (CLI flags are not punctuation)."""

    selectors = ("km_opt", "mp_physics", "cu_physics", "sf_sfclay_physics",
                 "[tiles]", "gpuwm go", "diff_6th_opt", "feedback",
                 "num_soil_layers", "wrfout", "ww_pp", "ERA5", "HRRR",
                 "hydrostatic", "checkpoint", "input_from_file",
                 "use_theta_m", "prepared cache")
    for entry in (upgrade_notes.results_that_move("2.7.0")
                  + upgrade_notes.restart_and_import("2.7.0")):
        assert "—" not in entry, entry
        assert " -- " not in entry, entry
        assert any(word in entry for word in selectors), entry


def test_the_upgrade_note_from_2_6_5_carries_the_results_list(tmp_path):
    """The one-time note after an upgrade lists every recorded entry."""

    state = tmp_path / "doctor-state.json"
    state.write_text(json.dumps({"version": "2.6.5"}), encoding="utf-8")
    note = doctor.upgrade_note("2.7.0", state)
    assert note is not None
    assert upgrade_notes.results_heading("2.7.0") in note
    for entry in (upgrade_notes.results_that_move("2.7.0")
                  + upgrade_notes.restart_and_import("2.7.0")):
        assert entry in note, entry[:80]
    # Said once: the second run records nothing new and prints nothing.
    assert doctor.upgrade_note("2.7.0", state) is None


def test_since_reports_the_same_lines_without_touching_state(capsys,
                                                             monkeypatch,
                                                             tmp_path):
    """`gpuwm doctor --since 2.6.5` is a query: it prints the note and
    leaves the one-time state file exactly as it found it."""

    state = tmp_path / "doctor-state.json"
    state.write_text(json.dumps({"version": "2.6.5"}), encoding="utf-8")
    monkeypatch.setenv(doctor.DOCTOR_STATE_ENV, str(state))
    monkeypatch.setattr(doctor, "collect_checks",
                        lambda *a, **k: pytest.fail("--since ran the estate"))
    monkeypatch.setattr("gpuwm.version_cli.install_shape",
                        lambda: {"version": "2.7.0"})

    code = doctor.doctor_main(SimpleNamespace(since="2.6.5", json=False,
                                              source=None))
    out = capsys.readouterr().out
    assert code == 0
    assert "what changed since 2.6.5" in out
    for line in whats_changed.since("2.6.5", "2.7.0"):
        assert line in out, line[:80]
    assert json.loads(state.read_text(encoding="utf-8"))["version"] == "2.6.5"


def test_since_the_installed_version_says_nothing_changed(capsys,
                                                          monkeypatch):
    monkeypatch.setattr("gpuwm.version_cli.install_shape",
                        lambda: {"version": "2.7.0"})
    assert doctor.doctor_main(SimpleNamespace(since="2.7.0")) == 0
    assert "nothing recorded as changed since 2.7.0" in capsys.readouterr().out


def test_the_doctor_parser_accepts_since():
    parser = argparse.ArgumentParser()
    doctor.register_cli(parser.add_subparsers())
    args = parser.parse_args(["doctor", "--since", "2.6.5"])
    assert args.since == "2.6.5"
    assert args.func is doctor.doctor_main
    assert parser.parse_args(["doctor"]).since is None
