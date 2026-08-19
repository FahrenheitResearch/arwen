"""Docs and parser must name the same flags for the arbitrary-input door.

The complaint this whole surface answers is that a working feature was
invisible.  A doc naming a flag that does not exist, or a flag no doc names,
recreates that invisibility one flag at a time -- so the two are held to each
other here rather than by review.

Both directions are asserted against the REAL parser, never a transcribed
list, for the same reason tests/test_explain_layering.py introspects it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from gpuwm.adapt import _SUPPORTED_FORMATS, register_cli


REPO = Path(__file__).resolve().parent.parent
#: Every doc that documents `gpuwm adapt` flags.  A flag must be named by at
#: least one of them; a flag named by any of them must exist.
DOCS = (
    REPO / "docs" / "cli-reference.md",
    REPO / "docs" / "intermediate-format.md",
    REPO / "docs" / "arbitrary-verified-adapters.md",
    REPO / "docs" / "adapt-validation-contract.md",
)
#: Swept onto every subcommand by gpuwm/cli.py, documented with the flag
#: itself rather than per subcommand.
_UNIVERSAL = {"--explain", "--help"}


def _adapt_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gpuwm")
    register_cli(root.add_subparsers(dest="command"))
    return root._subparsers._group_actions[0].choices["adapt"]


def _parser_flags() -> set[str]:
    flags: set[str] = set()
    for action in _adapt_parser()._actions:
        flags.update(option for option in action.option_strings
                     if option.startswith("--"))
    return flags - _UNIVERSAL


def _documented_flags() -> dict[str, list[Path]]:
    """Flags each doc names, restricted to plausible adapt flags."""

    found: dict[str, list[Path]] = {}
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for match in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", text):
            found.setdefault(match, []).append(doc)
    return found


def test_every_documented_adapt_flag_exists_in_the_parser():
    parser_flags = _parser_flags()
    documented = _documented_flags()
    # Only hold docs to flags they present as adapt's own: a doc may
    # legitimately mention rw-wps flags too.  The failure this guards is a
    # doc naming an adapt flag that adapt does not have, so scope to the
    # `gpuwm adapt` invocation blocks.
    named: set[str] = set()
    for doc in DOCS:
        for block in re.findall(
            r"gpuwm adapt[^\n]*(?:\n[ ]+[^\n]*)*", doc.read_text(encoding="utf-8")
        ):
            named.update(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", block))
    missing = sorted(named - parser_flags - _UNIVERSAL)
    assert not missing, (
        f"docs show `gpuwm adapt` flags the parser does not define: {missing}; "
        f"parser defines {sorted(parser_flags)}"
    )


def test_every_adapt_parser_flag_is_documented():
    parser_flags = _parser_flags()
    documented = set(_documented_flags())
    undocumented = sorted(parser_flags - documented)
    assert not undocumented, (
        f"`gpuwm adapt` defines flags no doc names: {undocumented}; "
        f"document them in {[str(doc.relative_to(REPO)) for doc in DOCS]}"
    )


def test_the_format_specification_exists_and_is_named_by_the_docs():
    spec = REPO / "docs" / "intermediate-format.md"
    assert spec.is_file(), "the intermediate-format specification must exist"
    reference = (REPO / "docs" / "cli-reference.md").read_text(encoding="utf-8")
    assert "intermediate-format.md" in reference, (
        "the CLI reference must point at the format specification, or the "
        "specification is invisible again"
    )


@pytest.mark.parametrize("source_format", _SUPPORTED_FORMATS)
def test_the_specification_names_every_authorable_format(source_format):
    spec = (REPO / "docs" / "intermediate-format.md").read_text(encoding="utf-8")
    assert source_format in spec.lower(), (
        f"gpuwm adapt authors {source_format!r} but the specification never "
        "names it"
    )


def test_vtable_is_optional_so_netcdf_can_reach_the_front_door():
    """The regression that made NetCDF unreachable: --vtable required."""

    for action in _adapt_parser()._actions:
        if "--vtable" in action.option_strings:
            assert not action.required, (
                "--vtable must not be globally required: a NetCDF descriptor "
                "has no WPS Vtable, so requiring it closes the front door on "
                "every NetCDF source"
            )
            return
    pytest.fail("--vtable is not defined at all")
