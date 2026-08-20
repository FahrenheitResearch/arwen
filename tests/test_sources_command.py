"""``gpuwm sources``: the human door onto the source registry.

Two defects are guarded here, both of them the same law -- a refusal
must name a remedy the reader can actually take.

**The refusal named a command that does not exist.**  A registered but
not-runnable source asked for through ``gpuwm fetch --source X`` refused
with ``see: `gpuwm sources` for what each registered source can do
today.``  There was no ``gpuwm sources``: the real parser answered
``invalid choice: 'sources'`` at exit 2, so the one sentence a stuck
reader was handed sent them to a dead end.

**The registry had no human door at all.**  The same facts were
reachable only as ``gpuwm run-plan --sources``, a JSON query flag on the
machine-facing execution front door.  Handing a person who just hit a
refusal a JSON query is not a front door (ship-only-what-users-can-reach),
so the command in the sentence is the one that now exists.

Everything the command prints is COPIED from the registry document
``gpuwm.runplan.source_inventory`` builds.  There is no second table and
no branch on a source id, which is what keeps adding a model to table
work: the graft cell below appends one row to the registry and expects
it in the printed listing with no edit here or in the renderer.
"""

from __future__ import annotations

import json
import re

import pytest


def _parser():
    from gpuwm.cli import build_parser

    return build_parser()


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    """Run one gpuwm invocation through the REAL front door.

    ``cli.main``, not ``args.func`` -- the refusal print boundary and
    the exit-2 convention live in ``main``, and a cell that called the
    handler directly would be testing a path no reader takes.
    """

    from gpuwm.cli import main

    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------
# The sentence that started it
# --------------------------------------------------------------------------

def _cited_commands(text: str) -> set[str]:
    """Every ``gpuwm <subcommand>`` a message tells the reader to run."""

    return {match.group(1) for match in re.finditer(r"`gpuwm ([a-z][a-z0-9-]*)",
                                                    text)}


def test_the_no_fetch_route_refusal_names_a_command_that_exists():
    """The #222 defect, at the exact call site that shipped it."""

    from gpuwm import fetch_routes
    from gpuwm.source_adapters import source_adapters

    subcommands = set(_parser()._subparsers._group_actions[0].choices)
    stranded = next(
        adapter.source_id for adapter in source_adapters()
        if not adapter.runnable
        and adapter.source_id not in set(fetch_routes.route_ids())
        and adapter.source_id not in fetch_routes.LEGACY_ROUTE_SOURCES
        and adapter.source_id not in fetch_routes._REFUSALS)  # noqa: SLF001
    with pytest.raises(ValueError) as excinfo:
        fetch_routes.route_for(stranded)
    text = str(excinfo.value)
    cited = _cited_commands(text)
    assert cited, f"the refusal offers no command at all: {text}"
    assert cited <= subcommands, (
        f"refusal cites {sorted(cited - subcommands)}, which `gpuwm` does "
        f"not have: {text}")


def test_no_refusal_in_the_package_cites_a_command_that_does_not_exist():
    """The sweep, so the next one is caught by a suite and not a user."""

    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "gpuwm"
    subcommands = set(_parser()._subparsers._group_actions[0].choices)
    offenders: list[str] = []
    for source in sorted(package.rglob("*.py")):
        text = source.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if '"' not in line and "'" not in line:
                continue
            for name in _cited_commands(line) - subcommands:
                offenders.append(f"{source.relative_to(package.parent)}:"
                                 f"{number} cites `gpuwm {name}`")
    assert not offenders, "\n".join(offenders)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

def test_the_listing_carries_every_registered_row_and_nothing_else(capsys):
    from gpuwm.source_adapters import source_adapters

    code, out, _err = _run(capsys, "sources")
    assert code == 0
    registered = [adapter.source_id for adapter in source_adapters()]
    listed = [line.split()[0] for line in out.splitlines()
              if line.startswith("  ") and line.strip()
              and not line.strip().startswith("id ")]
    assert listed == registered


def test_the_listing_says_how_many_rows_and_how_many_are_runnable(capsys):
    from gpuwm.source_adapters import source_capability_manifest

    manifest = source_capability_manifest()
    _code, out, _err = _run(capsys, "sources")
    assert str(manifest["source_count"]) in out
    assert str(manifest["runnable_source_count"]) in out


def test_one_source_id_prints_that_row_in_full(capsys):
    """The optional argument: one row, spelled out."""

    from gpuwm.source_adapters import source_adapters

    adapter = next(a for a in source_adapters() if a.runnable)
    code, out, _err = _run(capsys, "sources", adapter.source_id)
    assert code == 0
    assert adapter.source_id in out
    assert adapter.file_family in out
    assert adapter.status.value in out
    # The run-plan verdict travels with the row: a picker that offered a
    # launch the resolver then refuses is the defect this column exists
    # to prevent.
    assert "intent" in out.lower()


def test_an_alias_reaches_its_row(capsys):
    from gpuwm.source_adapters import source_adapters

    adapter = next((a for a in source_adapters() if a.aliases), None)
    if adapter is None:
        pytest.skip("no registered source declares an alias")
    code, out, _err = _run(capsys, "sources", adapter.aliases[0])
    assert code == 0
    assert adapter.source_id in out


def test_an_unknown_id_refuses_by_name_with_a_reachable_remedy(capsys):
    from gpuwm.source_adapters import source_capability_manifest

    code, out, err = _run(capsys, "sources", "no-such-source")
    assert code == 2
    text = out + err
    assert "no-such-source" in text
    assert str(source_capability_manifest()["source_count"]) in text
    assert _cited_commands(text) <= set(
        _parser()._subparsers._group_actions[0].choices)


def test_the_json_mode_is_byte_for_byte_the_run_plan_document(capsys):
    """One reply, two spellings -- so the doors cannot drift apart."""

    from gpuwm.runplan import SOURCES_SCHEMA, source_inventory

    code, out, _err = _run(capsys, "sources", "--json")
    assert code == 0
    document = json.loads(out)
    assert document["schema"] == SOURCES_SCHEMA
    assert document == source_inventory()


def test_a_grafted_registry_row_appears_with_no_edit_here(capsys, monkeypatch):
    """THE arbitrary acceptance test for this door.

    Adding a model is appending a row.  A per-model line in the renderer
    -- a display-name dict, a curated list, a status translation table --
    fails this cell.
    """

    import dataclasses

    from gpuwm import source_adapters as registry

    grafted = dataclasses.replace(
        registry.source_adapters()[0], source_id="probe-arbitrary-model",
        aliases=())
    monkeypatch.setattr(
        registry, "_ADAPTERS", (*registry.source_adapters(), grafted))
    monkeypatch.setattr(
        registry, "_ALIASES",
        {**registry._ALIASES,  # noqa: SLF001 - the graft IS the test
         "probe-arbitrary-model": grafted})

    _code, listing, _err = _run(capsys, "sources")
    assert "probe-arbitrary-model" in listing
    code, row, _err = _run(capsys, "sources", "probe-arbitrary-model")
    assert code == 0
    assert grafted.file_family in row
