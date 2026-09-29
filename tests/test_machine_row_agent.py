"""A machine row changed to another host sends the agent again, and a failed helper call keeps its words.

The defects: a changed row kept the note that the agent had been sent, so
the next call went to an agent that was not on the new host (a new
address, or a cloud machine's new instance under the same workspace path);
and a helper that failed on the machine showed the page plain words while
its own words (a traceback) went nowhere, so whoever looked after the
machine had nothing to read.
"""

from __future__ import annotations

from gpuwm.gui.machines import Registry


def test_a_machine_row_changed_to_another_host_sends_the_agent_again(tmp_path):
    registry = Registry(tmp_path / "machines.toml")
    registry.put({"name": "box", "host": "old-host"})
    before = registry.get("box")
    before._agent_sent.add(before.agent_path())
    assert registry.get("box")._agent_sent  # the same row keeps what it sent
    registry.update("box", host="new-host")
    assert registry.get("box")._agent_sent == set()


def test_a_helper_that_fails_on_the_machine_says_plain_words_and_logs_its_own(tmp_path, monkeypatch, capsys):
    """The page read "gpuwm's helper stopped with an error" and the helper's own words went nowhere."""

    import subprocess

    import pytest

    from gpuwm.gui.machines import MachineError

    registry = Registry(tmp_path / "machines.toml")
    registry.put({"name": "box", "host": "me@box"})
    machine = registry.get("box")
    monkeypatch.setattr(machine, "ensure_agent", lambda: None)
    failed = subprocess.CompletedProcess([], 1, b"", b"Traceback (most recent call last):\nKeyError: 'files'\n")
    monkeypatch.setattr(machine, "run", lambda *a, **k: failed)
    with pytest.raises(MachineError) as caught:
        machine.call("launch", "--run", "r1")
    assert "helper stopped with an error" in str(caught.value)
    logged = capsys.readouterr().err
    assert "launch on box ended with exit 1" in logged and "KeyError: 'files'" in logged
