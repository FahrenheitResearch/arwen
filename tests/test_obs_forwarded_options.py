"""An observation instrument door hands every token after its name to the binary.

``REMAINDER`` alone did not: argparse still read a leading ``--help`` as
the wrapper's own (so ``gpuwm obs mrms --help`` printed the wrapper, not
the binary's grammar) and refused ``--abi``, ``--version`` and every
other leading native switch as unrecognized, for all six instruments.
"""

from __future__ import annotations

import subprocess

import pytest

from gpuwm.cli import build_parser, main
from gpuwm.obs import cli as obs_cli

INSTRUMENTS = sorted(obs_cli._INSTRUMENTS)


@pytest.mark.parametrize("instrument", INSTRUMENTS)
@pytest.mark.parametrize("tokens", [
    ["--help"], ["-h"], ["--version"], ["--abi"],
    ["list", "--start", "2026-09-27T00:00:00Z"],
    ["--unknown-option", "value"], ["--explain"], []])
def test_instrument_argv_is_forwarded_verbatim(instrument, tokens):
    args = build_parser().parse_args(["obs", instrument, *tokens])
    assert args.argv == tokens
    assert args.func is obs_cli._instrument_main


def test_the_parent_and_radar_doors_keep_their_own_grammar(capsys):
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["obs", "--help"])
    assert stopped.value.code == 0
    assert "radar" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["obs", "radar", "sites", "--no-such-flag"])
    assert stopped.value.code == 2
    args = build_parser().parse_args(["obs", "--explain", "mrms", "--abi"])
    assert args.explain is True and args.argv == ["--abi"]


def test_leading_native_switches_reach_the_binary(monkeypatch, tmp_path):
    """The real front door, end to end, with the binary's launch observed."""
    seen = []

    class Door:
        def require(self):
            return tmp_path / "rw_mrms"

    def run(command, check):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    from gpuwm.obs import frontdoor
    monkeypatch.setitem(frontdoor.FRONT_DOORS, "mrms", Door())
    monkeypatch.setattr(subprocess, "run", run)
    assert main(["obs", "mrms", "--help"]) == 0
    assert main(["obs", "mrms", "--abi"]) == 0
    assert [command[1:] for command in seen] == [["--help"], ["--abi"]]
