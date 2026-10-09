"""WRF &time_control cycling reaches RunConfig.cycling through the importer.

Emitted only when .true., so a namelist that omits it or spells the default
imports to the bytes it always did.  Both MYNN generations consume it
(tests/test_mynn_carry_config.py admits both), and the operational fork's
&time_control translation does not eat it on the way.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from gpuwm.namelist_import import (
    _Section, _translate_operational_fork_time_controls, import_namelists)
from gpuwm.physics_source_defaults import with_physics_selector_comment

FIXTURES = Path(__file__).parent / "fixtures" / "source_requests"


def _import(tmp_path, line: str | None) -> str:
    source = (FIXTURES / "hrrr_wrf.nl.c18c").read_text()
    if line is not None:
        source = source.replace("&time_control\n",
                                f"&time_control\n {line}\n", 1)
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(with_physics_selector_comment(source, {}))
    text, _ = import_namelists(FIXTURES / "hrrr_namelist.wps.c18", path)
    return text


def test_cycling_true_selects_the_cycled_start(tmp_path):
    text = _import(tmp_path, "cycling = .true.,")
    assert "cycling = true\n" in text
    shared = tomllib.loads(text)["shared"]
    assert shared["cycling"] is True
    assert shared["bl_mynn_version"] == "gsd_41"


def test_cycling_false_or_absent_imports_the_same_bytes(tmp_path):
    absent = _import(tmp_path, None)
    false = _import(tmp_path, "cycling = .false.,")
    assert "cycling =" not in absent
    assert false == absent


def test_cycling_true_reaches_the_model_under_both_mynn_generations(tmp_path):
    """The breakage this guards: a cycled fork namelist imported as a fresh
    start.  The fork's &time_control translation once substituted
    cycling = .true. with a fresh start; on a route that read it there, the
    run would integrate a cold MYNN start while the CHANGELOG says cycling
    reaches the model.  Under either generation the key must arrive."""
    for version, selectors in (("wrf_461", {"bl_mynn_version": "wrf_461"}),
                               ("gsd_41", {})):
        source = (FIXTURES / "hrrr_wrf.nl.c18c").read_text().replace(
            "&time_control\n", "&time_control\n cycling = .true.,\n", 1)
        # The fork route is recognised from the file's own name.
        (tmp_path / version).mkdir()
        path = tmp_path / version / "hrrr_wrf.nl"
        path.write_text(with_physics_selector_comment(source, selectors))
        text, report = import_namelists(
            FIXTURES / "hrrr_namelist.wps.c18", path)
        shared = tomllib.loads(text)["shared"]
        assert shared["cycling"] is True, version
        # wrf_461 is RunConfig's default, so [shared] omits it.
        assert shared.get("bl_mynn_version", "wrf_461") == version
        assert "cycling" not in {s.key for s in report.substitutions}
        assert ("time_control", "cycling") not in {
            (d.section, d.key) for d in report.dropped}


def test_the_fork_time_control_translation_leaves_cycling_alone():
    """Whatever order the importer calls it in, the fork translation must
    not consume cycling: that is the one key _translate_namelists maps
    onto [shared] cycling, and a consumer here would turn it into a drop."""
    section = _Section("time_control", {"cycling": [True]}, "namelist.input")
    dropped, substitutions = [], []
    _translate_operational_fork_time_controls(
        section, 3600.0, lambda *args: dropped.append(args), substitutions)
    assert section.entries == {"cycling": [True]}
    assert all(args[1] != "cycling" for args in dropped)
    assert all(item.key != "cycling" for item in substitutions)
