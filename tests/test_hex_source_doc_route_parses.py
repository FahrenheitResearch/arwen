"""The documented hex-as-ArWen-source route parses against the real CLI.

Breakage (2.9.2 hex-as-source door test, DOOR-hex-as-arwen-source.md): the
hex-global section of docs/native-mapped-source-authoring.md gave
``gpuwm prep --source hex-global --input FRAME ... --supplement
hex_in_band_surface=FRAME ...`` and nothing else.  Run as written it exited
64 ("choose exactly one of an existing --source-manifest plus digest or
--author-input-manifest") and omitted the config, namelist, geography root
and output root every route needs, while ``gpuwm domain --source hex-global``
had no way to emit that config at all (it refused the row's undeclared
cadence even with ``--cadence``).

The section now carries the whole route as one fenced block.  This test
feeds every command in that block to the parser that would receive it --
a full ``parse_args``, the way tests/test_verification_recipe_flags.py
holds VERIFICATION.md's recipe -- so a required argument the block drops,
or a flag the door renames, goes red here and not on a user's box.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

import doc_command_parity as parity

DOC = parity.REPO_ROOT / "docs" / "native-mapped-source-authoring.md"


def _hex_route_commands() -> list[list[str]]:
    """Every command of the fenced block that plans ``--source hex-global``."""
    blocks: list[list[str]] = []
    inside = False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(?:```|~~~)", line):
            if inside:
                inside = False
            else:
                inside = True
                blocks.append([])
            continue
        if inside:
            blocks[-1].append(line)
    route = [block for block in blocks
             if any("--source hex-global" in line for line in block)]
    assert len(route) == 1, "exactly one fenced block carries the hex-global route"
    joined = "\n".join(route[0]).replace("\\\n", " ")
    return [shlex.split(line) for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


COMMANDS = _hex_route_commands()


def test_the_block_is_the_whole_route():
    doors = [parity.resolve_door(" ".join(tokens), parity.doors())[0]
             for tokens in COMMANDS]
    assert doors == ["gpuwm domain", "gpuwm fetch-geog", "gpuwm prep",
                     "gpuwm sim"], doors


def test_the_domain_line_names_the_cadence_the_row_does_not_declare():
    domain = next(t for t in COMMANDS if t[:2] == ["gpuwm", "domain"])
    assert "--cadence" in domain
    assert "--source" in domain and domain[domain.index("--source") + 1] == "hex-global"


def test_the_prep_line_carries_what_exit_64_asked_for():
    prep = next(t for t in COMMANDS if t[:2] == ["gpuwm", "prep"])
    for flag in ("--author-input-manifest", "--experiment-config",
                 "--wps-namelist", "--geog-root", "--output-root"):
        assert flag in prep, flag
    inputs = [prep[i + 1] for i, t in enumerate(prep) if t == "--input"]
    supplements = [prep[i + 1] for i, t in enumerate(prep) if t == "--supplement"]
    assert len(inputs) == 3
    assert supplements == [f"hex_in_band_surface={path}" for path in inputs]


@pytest.mark.parametrize("index", range(len(COMMANDS)),
                         ids=lambda i: " ".join(COMMANDS[i][:2]))
def test_each_documented_command_parses_against_its_door(index):
    tokens = COMMANDS[index]
    known = parity.doors()
    door, _ = parity.resolve_door(" ".join(tokens), known)
    assert door in known, door
    parser = known[door]
    # Everything after the door's own words is what the parser receives.
    rest = tokens[len(door.split()):]
    try:
        parser.parse_args(rest)
    except SystemExit as exc:  # argparse's refusal, now with the command named
        pytest.fail(f"`{' '.join(tokens)}` does not parse against {door} "
                    f"(argparse exit {exc.code})")
