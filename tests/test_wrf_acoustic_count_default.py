"""D8 (2.8.9): a fixed-step WRF folder runs WRF's own acoustic count.

NCAR's CONUS 12 km benchmark leaves ``time_step_sound`` unset, so wrf.exe
derives 4 (solve_em.F); WOOF's terrain clock raised the restart leg to 6,
10 percent per step and further from WRF's own output.  The import now
pins the clock to the count the namelist (or WRF's rule for it) names,
and the acoustic receipt states the count that ran.
"""
from __future__ import annotations

from dataclasses import dataclass

from gpuwm.acoustic_adaptation import (SlopeReading, acoustic_receipt,
                                       derive_acoustics)
from gpuwm.experiment import load_experiment
from gpuwm.namelist_import import import_namelists

from test_namelist_import import (INPUT_TEXT, _fixed_through_adaptive_input,
                                  _pair, _single_domain_pair)


def _load(tmp_path, toml_text):
    out = tmp_path / "d8.toml"
    out.write_text(toml_text)
    return load_experiment(out)


def test_a_fixed_step_wrf_namelist_runs_wrfs_count_pinned(tmp_path):
    """time_step_sound unset at 60 s on 12 km: WRF's rule gives 4, and the
    terrain clock may not raise it."""
    toml_text, report = import_namelists(*_pair(tmp_path))
    assert 'terrain_clock = "pinned"' in toml_text
    applied = {a.key: a for a in report.defaults_applied}
    assert applied["terrain_clock"].value == "pinned"
    assert applied["time_step_sound"].value == 4
    assert "solve_em.F" in applied["time_step_sound"].reason
    exp = _load(tmp_path, toml_text)
    assert all(dc.run.terrain_clock == "pinned" for dc in exp.domains)
    assert all(dc.run.time_step_sound == 4 for dc in exp.domains)


def test_an_explicit_time_step_sound_is_the_count_that_runs(tmp_path):
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,",
                             " hybrid_opt = 2,\n time_step_sound = 8,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    exp = _load(tmp_path, toml_text)
    assert exp.root.run.time_step_sound == 8
    assert exp.root.run.terrain_clock == "pinned"


def test_wrfs_rule_for_zero_follows_the_step(tmp_path):
    """A 120 s step on 12 km: 300 * 120 / 12000 = 3 -> 2 * (2 + 1) = 6.
    The old import wrote 4 here, a count wrf.exe never runs."""
    inp = (INPUT_TEXT.replace(" time_step = 60,", " time_step = 120,")
           .replace(" radt = 12, 3,", " radt = 12, 4,")
           .replace(" cudt = 5, 0,", " cudt = 0, 0,"))
    toml_text, report = import_namelists(*_single_domain_pair(tmp_path, inp))
    applied = {a.key: a for a in report.defaults_applied}
    assert applied["time_step_sound"].value == 6
    assert "time_step_sound = 6" in toml_text
    assert _load(tmp_path, toml_text).root.run.time_step_sound == 6


def test_a_stated_terrain_clock_is_the_inputs_choice(tmp_path):
    """A selector comment naming WOOF's clock keeps it: that statement is
    the input's choice, not WRF's."""
    inp = ('! gpuwm-physics-selectors-v1: {"terrain_clock":"local_face"}\n'
           + INPUT_TEXT)
    wps, namelist = _pair(tmp_path, inp=inp)
    toml_text, report = import_namelists(wps, namelist)
    assert 'terrain_clock = "local_face"' in toml_text
    assert 'terrain_clock = "pinned"' not in toml_text
    assert _load(tmp_path, toml_text).root.run.terrain_clock == "local_face"


def test_an_adaptive_namelist_keeps_the_terrain_clock(tmp_path):
    """WRF derives the count from a step the adaptive clock chooses, so
    the input names no fixed count and the terrain clock stays."""
    inp = _fixed_through_adaptive_input(60, least=30)
    toml_text, _ = import_namelists(*_single_domain_pair(tmp_path, inp))
    assert "terrain_clock" not in toml_text
    assert _load(tmp_path, toml_text).root.run.terrain_clock == "local_face"


@dataclass(frozen=True)
class _Run:
    dx: float = 12000.0
    dy: float = 12000.0
    epssm: float = 0.1
    time_step_sound: int = 4


@dataclass(frozen=True)
class _Clock:
    grid_id: int
    time_step_sound: int


def test_the_acoustic_receipt_states_the_count_that_ran():
    """NCAR 12 km restart receipt: this section said 4 / AS_CONFIGURED
    while the terrain clock section said 6 / ADAPTED, and 6 ran."""
    rule = derive_acoustics(1, _Run(), SlopeReading("d01", 0.09, ("y", 65, 145)))
    assert rule.time_step_sound == 4 and rule.status == "AS_CONFIGURED"
    row = acoustic_receipt([rule], [_Clock(1, 6)])["domains"][0]
    assert row["time_step_sound"] == 6
    assert row["slope_rule_time_step_sound"] == 4
    assert row["set_by"] == "terrain_clock"
    assert row["status"] == "ADAPTED"
    # The clock that holds the count changes nothing.
    same = acoustic_receipt([rule], [_Clock(1, 4)])["domains"][0]
    assert same == acoustic_receipt([rule])["domains"][0]


def test_a_terrain_clock_checkpoint_under_the_pinned_default_says_why():
    """A checkpoint from an earlier import of the same WRF folder ran the
    terrain clock; resuming it under the 2.8.9 default is still refused
    (its count can differ), and the refusal names the default and the
    line that resumes it."""
    from dataclasses import replace

    import pytest

    from gpuwm.config import (TERRAIN_CLOCK_PINNED_RESTART_NOTICE,
                              TERRAIN_CLOCK_RESTART_BREAK_NOTICE)
    from gpuwm.io import restart
    from test_restart import _cfg

    earlier = _cfg()
    assert earlier.terrain_clock == "local_face"
    stored = restart.configuration_echo(earlier)
    pinned = replace(earlier, terrain_clock="pinned")
    with pytest.raises(restart.RestartMismatchError,
                       match="terrain_clock") as caught:
        restart._require_config_match(stored, pinned, "earlier.npz")
    assert TERRAIN_CLOCK_PINNED_RESTART_NOTICE in str(caught.value)
    assert TERRAIN_CLOCK_RESTART_BREAK_NOTICE not in str(caught.value)
    # A pinned checkpoint resumes under pinned, with no note.
    restart._require_config_match(
        restart.configuration_echo(pinned), pinned, "pinned.npz")
