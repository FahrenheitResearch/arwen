"""The European radar front door: is it reachable, and does it match its docs.

Two kinds of test here and they answer different questions.

**Reachability.** ``gpuwm obs radar`` has to exist on the real product parser,
be dispatched by the real dispatch table, and be listed in the real bundle
inventory. Every one of those is a place a subcommand has silently failed to
ship before: a parser entry with no dispatch line raises ``AttributeError`` at
the first invocation, and a front door whose binary is in no bundle can only be
reached by installing a Rust toolchain, which is not shipping.

**The flag contract, in both directions.** A flag the parser defines and the
docs do not is undiscoverable; a flag the docs name and the parser does not is
a lie that fails in a user's terminal. Both directions are asserted against
``docs/european-radar-front-door.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pytest

from gpuwm.obs import cli as obs_cli

DOC = (Path(__file__).resolve().parent.parent / "docs"
       / "european-radar-front-door.md")

#: argparse's own, never documented per-command.
_BUILTIN = {"-h", "--help"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuwm")
    obs_cli.register_cli(parser.add_subparsers(dest="command", required=True))
    return parser


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:                      # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):   # noqa: SLF001
            return action.choices
    return {}


def _walk(parser: argparse.ArgumentParser, prefix: str) -> dict[str, set[str]]:
    """``{"gpuwm obs radar pack": {"--file", ...}}`` for every leaf command."""

    children = _subparsers(parser)
    if children:
        found: dict[str, set[str]] = {}
        for name, child in children.items():
            found.update(_walk(child, f"{prefix} {name}"))
        return found
    options: set[str] = set()
    for action in parser._actions:                      # noqa: SLF001
        options.update(
            option for option in action.option_strings if option not in _BUILTIN)
    return {prefix: options}


def parser_flags() -> dict[str, set[str]]:
    """Every leaf under ``gpuwm obs radar``, which is what DOC governs.

    Scoped to the ``radar`` subtree rather than to all of ``gpuwm obs``.
    ``gpuwm obs`` also carries one passthrough door per instrument
    (``mrms``, ``stage4``, ``asos``, ``goes``, ``opera``, ``odim``),
    which take no gpuwm flags of their own -- they hand ARGS to a binary
    that owns its own grammar and prints its own ``--help``. DOC is the
    *European radar* reference and correctly says nothing about them, so
    walking from the root would make this comparison fail on commands it
    was never written to describe. The regex in :func:`documented_flags`
    has always said ``gpuwm obs radar``; this makes the other side agree.
    """

    obs = _subparsers(_parser())["obs"]
    return _walk(_subparsers(obs)["radar"], "gpuwm obs radar")


def documented_flags() -> dict[str, set[str]]:
    """Flags the command reference names, per ``### gpuwm obs radar X`` heading."""

    text = DOC.read_text(encoding="utf-8")
    _, _, reference = text.partition("## Command reference")
    assert reference, f"{DOC} has no '## Command reference' section"
    found: dict[str, set[str]] = {}
    current: str | None = None
    for line in reference.splitlines():
        heading = re.match(r"^###\s+(gpuwm obs radar \w+)\s*$", line)
        if heading:
            current = heading.group(1)
            found[current] = set()
            continue
        if current is None:
            continue
        found[current].update(re.findall(r"`(--[a-z0-9-]+)`", line))
    return found


# --------------------------------------------------------------- reachable

def test_obs_radar_is_on_the_real_product_parser():
    """Not this module's parser -- the one ``gpuwm`` actually builds."""

    from gpuwm.cli import build_parser

    obs = _subparsers(build_parser()).get("obs")
    assert obs is not None, "`gpuwm obs` is not a subcommand of the product CLI"
    assert "radar" in _subparsers(obs)


def test_obs_is_in_the_dispatch_table_not_only_the_parser():
    """A parser entry with no dispatch line raises AttributeError on first use.

    ``_dispatch`` routes a fixed list of commands to ``args.func`` and
    everything else falls through to the config-driven run path, which reads
    ``args.config`` -- an attribute an ``obs`` namespace does not have. The
    failure is an AttributeError traceback, not a refusal, and it only ever
    appears when a user runs the command.
    """

    from gpuwm import cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    dispatch = source.partition("def _dispatch(")[2]
    assert '"obs"' in dispatch.partition("return args.func(args)")[0]


def test_the_decoder_ships_in_the_bridge_bundle():
    """Reachable means obtainable without a Rust toolchain."""

    from gpuwm import bridge_assets

    names = {artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert "rw_odim" in names, (
        "rw_odim is in no bundle, so `gpuwm obs radar` can only be reached "
        "from a source checkout with cargo installed")


def test_the_doctor_leg_reports_a_remedy_when_the_binary_is_absent(
        monkeypatch, capsys, tmp_path):
    """Missing is an answer, not a traceback, and the remedy is the useful half."""

    from gpuwm.obs import frontdoor

    monkeypatch.setenv("GPUWM_RW_ODIM", str(tmp_path / "definitely-absent"))
    with pytest.raises(FileNotFoundError):
        frontdoor.ODIM.find()

    monkeypatch.delenv("GPUWM_RW_ODIM", raising=False)
    monkeypatch.setattr(frontdoor.FrontDoor, "find", lambda self: None)
    args = argparse.Namespace()
    assert obs_cli._radar_doctor(args) == 0            # noqa: SLF001
    import json
    record = json.loads(capsys.readouterr().out)
    assert record["found"] is None
    assert record["abi_matches"] is False
    assert record["remedy"]
    assert record["searched"]


# ---------------------------------------------------------- both directions

def test_every_parser_flag_is_documented():
    parser = parser_flags()
    documented = documented_flags()
    undocumented = {
        command: sorted(flags - documented.get(command, set()))
        for command, flags in parser.items()
        if flags - documented.get(command, set())
    }
    assert not undocumented, (
        f"flags the parser defines and {DOC.name} does not name: "
        f"{undocumented}")


def test_every_documented_flag_exists_in_the_parser():
    parser = parser_flags()
    documented = documented_flags()
    invented = {
        command: sorted(flags - parser.get(command, set()))
        for command, flags in documented.items()
        if flags - parser.get(command, set())
    }
    assert not invented, (
        f"flags {DOC.name} names that the parser does not define: {invented}")


def test_the_two_directions_cover_the_same_commands():
    """A whole command missing from the docs passes both directions above.

    Each of them iterates one side's commands, so a command absent from the
    reference contributes no flags to compare and neither test notices.
    """

    parser = set(parser_flags())
    documented = set(documented_flags())
    assert parser == documented, (
        f"only in the parser: {sorted(parser - documented)}; "
        f"only in {DOC.name}: {sorted(documented - parser)}")


def test_the_instrument_fires_on_a_known_bad_doc(tmp_path, monkeypatch):
    """Validate the checker in the direction that matters: it must fail.

    A doc-vs-parser comparison that silently passes on a broken doc is the
    most expensive kind of green. Point it at a reference that invents a flag
    and drops a real one, and it must report both.
    """

    broken = tmp_path / "broken.md"
    broken.write_text(
        "## Command reference\n"
        "### gpuwm obs radar doctor\n"
        "### gpuwm obs radar volumes\n"
        "| `--dir` | fine |\n"
        "### gpuwm obs radar pack\n"
        "| `--file` | present |\n"
        "| `--invented` | not a real flag |\n"
        "### gpuwm obs radar nyquist\n"
        "| `--file` | fine |\n"
        "### gpuwm obs radar sites\n"
        "| `--bbox` | fine |\n"
        "### gpuwm obs radar grid\n"
        "| `--pack` | fine |\n",
        encoding="utf-8")
    monkeypatch.setattr(
        __import__(__name__, fromlist=["DOC"]), "DOC", broken)

    with pytest.raises(AssertionError, match="--invented"):
        test_every_documented_flag_exists_in_the_parser()
    with pytest.raises(AssertionError, match="--out"):
        test_every_parser_flag_is_documented()


def test_the_instrument_is_silent_on_the_real_doc():
    """The other direction of the same validation: no false alarm."""

    test_every_documented_flag_exists_in_the_parser()
    test_every_parser_flag_is_documented()
    test_the_two_directions_cover_the_same_commands()


# ------------------------------------------------------- refusals argparse
#                                                          cannot express

def test_pack_refuses_both_file_and_dir():
    args = argparse.Namespace(
        file=Path("v.h5"), dir=Path("."), stamp=None, out=Path("o.pack"),
        quantities=None, max_elevation_deg=None, max_range_km=None)
    with pytest.raises(SystemExit, match="exactly one"):
        obs_cli._radar_pack(args)                       # noqa: SLF001


def test_pack_refuses_neither_file_nor_dir():
    args = argparse.Namespace(
        file=None, dir=None, stamp=None, out=Path("o.pack"),
        quantities=None, max_elevation_deg=None, max_range_km=None)
    with pytest.raises(SystemExit, match="exactly one"):
        obs_cli._radar_pack(args)                       # noqa: SLF001


def test_pack_refuses_a_stamp_that_cannot_mean_anything():
    """``--stamp`` selects among a directory's volumes; a file is one volume.

    Accepting and ignoring it would let a caller believe they had selected
    something.
    """

    args = argparse.Namespace(
        file=Path("v.h5"), dir=None, stamp="20260814T120000Z",
        out=Path("o.pack"), quantities=None, max_elevation_deg=None,
        max_range_km=None)
    with pytest.raises(SystemExit, match="--stamp"):
        obs_cli._radar_pack(args)                       # noqa: SLF001


@pytest.mark.parametrize("value", ["1,2,3", "1,2,3,4,5", "a,b,c,d",
                                   "0,55,10,50"])
def test_the_bbox_type_refuses_what_it_cannot_mean(value):
    with pytest.raises(argparse.ArgumentTypeError):
        obs_cli._bbox(value)                            # noqa: SLF001


def test_the_bbox_type_accepts_a_real_box():
    assert obs_cli._bbox("3,50,8,55") == (3.0, 50.0, 8.0, 55.0)  # noqa: SLF001
