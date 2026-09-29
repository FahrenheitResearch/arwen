"""History intervals that are not a finite positive number of seconds are refused by name.

`gpuwm domain --history-interval=inf` (or --nest-history-interval with a
nest) reached ``Fraction(inf)`` in the clock derivation and left as an
OverflowError traceback at exit 1; NaN left as "cannot convert NaN to
integer ratio", zero and negative as "run and output/checkpoint
intervals must be positive", none of them naming the option.  A zero
--cadence divided by zero in the fetch window.
"""

from __future__ import annotations

import pytest

from gpuwm.cli import main
from gpuwm.domain_wizard import derived_time_step_s

BASE = ["domain", "--point", "40,-100", "--card", "12gb", "--source", "gfs",
        "--cycle", "2026-09-27T00", "--hours", "1", "--point-extent-km", "100"]


@pytest.mark.parametrize("option,extra", [
    ("--history-interval", []),
    ("--nest-history-interval", ["--ladder", "12-3"]),
])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_interval_is_named_and_writes_nothing(tmp_path, capsys, option, extra, value):
    out = tmp_path / "case.toml"
    with pytest.raises(SystemExit) as stopped:
        main([*BASE, "--out", str(out), *extra, f"{option}={value}"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert f"argument {option}" in err and "above zero" in err
    assert "Traceback" not in err
    assert not out.exists()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_zero_or_negative_boundary_cadence_is_named(tmp_path, capsys, value):
    out = tmp_path / "case.toml"
    with pytest.raises(SystemExit) as stopped:
        main([*BASE, "--out", str(out), f"--cadence={value}"])
    assert stopped.value.code == 2
    assert "argument --cadence" in capsys.readouterr().err
    assert not out.exists()


def test_a_valid_interval_still_authors(tmp_path, capsys):
    out = tmp_path / "case.toml"
    assert main([*BASE, "--out", str(out), "--history-interval", "1800"]) == 0
    assert out.exists()
    capsys.readouterr()


def test_the_clock_helper_names_a_nonfinite_interval_and_ignores_an_unused_nest_one():
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError, match="history interval must be a finite"):
            derived_time_step_s(40.0, 12000.0, run_seconds=3600.0, ratios=(),
                                history_interval_s=bad)
        with pytest.raises(ValueError, match="nest history interval must be a finite"):
            derived_time_step_s(40.0, 12000.0, run_seconds=3600.0, ratios=(3,),
                                nest_history_interval_s=bad)
    # A single-domain ladder never reads the nest interval.
    assert derived_time_step_s(40.0, 12000.0, run_seconds=3600.0, ratios=(),
                               nest_history_interval_s=float("inf")) > 0
