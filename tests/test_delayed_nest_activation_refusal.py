"""Delayed nest activation refuses at load, instead of accepting and crashing.

Defect #205: a [[domain]] child whose start_time is later than
[experiment].start_time passed every plan-time and init-time gate, then
deterministically killed the run at the activation epoch.  The mechanism:
the tree history writer consumes a microphysics-time REFL_10CM stash for
every frame after the experiment's own t = 0
(``gpuwm.runtime._submit_tree_history_frame`` ->
``gpuwm.core.refl.consume_refl_10cm``), and the stash is produced only
inside a microphysics step of the same domain.  A delayed child's first
history frame is due AT its activation epoch, before any of its steps has
run, so the consume raises "REFL_10CM output is due but no
microphysics-time field is stashed" and the run dies there -- measured at
32,350 s of burned integration on 2026-08-19, reproduced identically
across three supervisor restarts.

Drew's refusals law: a config the product cannot run refuses upfront,
naming the concrete breakage.  The refusal tests here fail against the
tree as it stood at ``bcc58ff03``; the all-domains-at-init companion
passes before and after.
"""

from __future__ import annotations

import pytest

import gpuwm.cli as cli
from gpuwm.experiment import load_experiment
from gpuwm.explain import split

from test_experiment import _write

#: 1800 s after the experiment start: aligned to d01's 60 s step and
#: inside the run window, so no structural refusal can catch it -- before
#: the guard, this config loaded and the run burned hours.
_DELAYED_D02 = "start_time = 1974-04-03T12:30:00"


def test_a_delayed_child_start_refuses_at_load_naming_the_breakage(
        tmp_path):
    """The earliest shared gate refuses, and the message carries all four
    halves: the feature name, the concrete breakage, the epoch it kills,
    and the workaround."""

    path = _write(tmp_path, d02=_DELAYED_D02)
    with pytest.raises(ValueError) as caught:
        load_experiment(path)
    message = str(caught.value)
    assert "delayed nest activation" in message
    assert "microphysics-time REFL_10CM" in message
    assert "activation epoch" in message
    assert "Start every [[domain]] at [experiment].start_time" in message


def test_the_refusal_is_a_sentence_with_a_remedy_not_a_traceback(tmp_path):
    """Message-layer contract, same as the [case_data] refusals: the
    action half is short and carries the remedy; the stash mechanism is
    behind ``--explain``."""

    path = _write(tmp_path, d02=_DELAYED_D02)
    with pytest.raises(ValueError) as caught:
        load_experiment(path)
    action, why = split(str(caught.value))
    assert "Start every [[domain]] at [experiment].start_time" in action
    assert action.count("\n") <= 3, action
    assert why, "the stash mechanism belongs behind --explain"
    assert "no microphysics-time field is stashed" in why


def test_the_refusal_reaches_the_shipped_cli_at_rc_2(tmp_path, capsys):
    """The exit code, not just the exception: the shipped entry point
    refuses upfront at rc 2 instead of accepting and dying hours in."""

    path = _write(tmp_path, d02=_DELAYED_D02)
    assert cli.main(["run", str(path)]) == 2
    assert "delayed nest activation" in capsys.readouterr().err


def test_all_domains_at_init_pass_the_same_gate_untouched(tmp_path):
    """No gate widening: the default (no [[domain]] start_time) and an
    explicit start equal to the experiment's both still load."""

    exp = load_experiment(_write(tmp_path))
    assert exp.domain_start_offset_exact(2) == 0
    explicit = _write(tmp_path,
                      d02="start_time = 1974-04-03T12:00:00")
    exp2 = load_experiment(explicit)
    assert exp2.domain_start_time(2) == exp2.start_time


def test_a_misaligned_delayed_start_keeps_its_structural_refusal(tmp_path):
    """Deliberate ordering pin: a delayed start that is ALSO misaligned
    to the parent step keeps the precise structural message it always
    had -- that config stays invalid even once delayed activation gains
    its activation-epoch stash."""

    path = _write(tmp_path, d02="start_time = 1974-04-03T12:00:50")
    with pytest.raises(ValueError, match="parent step boundary"):
        load_experiment(path)
