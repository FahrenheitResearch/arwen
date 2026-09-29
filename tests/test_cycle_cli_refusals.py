"""`gpuwm cycle` refusals are sentences at exit 2, not tracebacks.

A CycleRefusal is a RuntimeError by design (each one carries what it
observed), and the front door's refusal boundary re-raised every
RuntimeError outside the fetch family: `--cycles 0`, a child step that
does not divide the parent's, a placement with no parent geography, all
left as tracebacks at exit 1.  A NaN step left as "Invalid literal for
Fraction: 'nan'", which names no option.
"""

from __future__ import annotations

import pytest

from gpuwm.cli import main


def _cycle(tmp_path, *extra):
    return ["cycle", "--root", str(tmp_path / "run"), "--epoch-anchor",
            "2026-09-27T00:00:00Z", "--parent-kind", "replay", "--dry-run", *extra]


def test_a_cycle_refusal_is_a_sentence_at_exit_2(tmp_path, capsys):
    # 900 s is not a whole number of the default 120 s parent steps.
    assert main(_cycle(tmp_path, "--cycle-seconds", "900", "--cycles", "2")) == 2
    err = capsys.readouterr().err
    assert "gpuwm cycle:" in err and "observed:" in err
    assert "Traceback" not in err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("option,value", [
    ("--cycles", "0"), ("--cycle-seconds", "nan"), ("--parent-dt-seconds", "-1"),
    ("--child-dt-seconds", "inf"), ("--child-slots", "-1"),
    ("--max-forecast-only-cycles", "-1"), ("--accept-snap-offset-seconds", "nan"),
    ("--port-steps", "0"), ("--port-timeout", "-1"), ("--parent-dx-m", "0"),
    ("--placement-threshold", "nan"), ("--retire-below-strength", "inf"),
    ("--min-separation-km", "-1"), ("--child-nx", "0"), ("--child-dx-m", "-inf")])
def test_numeric_cycle_options_are_refused_by_name(tmp_path, capsys, option, value):
    argv = _cycle(tmp_path, "--cycle-seconds", "960", "--cycles", "2")
    with pytest.raises(SystemExit) as stopped:
        main([*argv, f"{option}={value}"])
    assert stopped.value.code == 2
    assert f"argument {option}" in capsys.readouterr().err
