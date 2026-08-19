"""Every source spelling a user can READ must be a spelling they can TYPE.

Origin: the 6 h model battery (2026-08-17).  The Canadian global model ran
as ``--source gem-gdps`` only after ``--source gem`` was refused by name --
``gem`` was the row's ``upstream_model_id``, the model's own name, but not
one of its aliases.  The front door refused correctly and listed every id,
so nothing was silently wrong; the model was simply unreachable under the
name it is known by, which is the discoverability half of "engine-proven is
not shipped".

The breakage these tests prevent: a source whose name appears in the
documentation or in its own registry row cannot be selected at the front
door.  Both directions are gated, so the fix is table work either way --
add the alias, or stop publishing the name.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuwm.source_adapters import get_source_adapter, source_adapters

_REPO = Path(__file__).resolve().parents[1]
_CLI_REFERENCE = _REPO / "docs" / "cli-reference.md"

#: Commands whose ``--source`` is the model registry's.  Other commands own
#: unrelated ``--source`` flags (the static-asset mirror, the radar feed),
#: and those spellings are theirs, not the registry's.
_PREP_DOORS = ("rw-wps", "gpuwm prep", "gpuwm-wrf-init")


def _alias_table() -> dict[str, tuple[str, ...]]:
    """The published id -> other spellings table, parsed from the docs."""

    text = _CLI_REFERENCE.read_text(encoding="utf-8")
    section = text.split("## Source spellings", 1)
    assert len(section) == 2, "the cli reference lost its source spellings table"
    body = section[1].split("\n## ", 1)[0]
    table: dict[str, tuple[str, ...]] = {}
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        named = re.fullmatch(r"`([^`]+)`", cells[0])
        if named is None:                    # the header and the rule row
            continue
        source_id = named.group(1)
        others = tuple(
            value.strip().strip("`") for value in cells[1].split(",")
            if value.strip() and value.strip() != "--"
        )
        table[source_id] = others
    return table


def _documented_prose_spellings() -> dict[str, tuple[str, ...]]:
    """``--source X`` spellings written beside a preparation front door."""

    found: dict[str, list[str]] = {}
    paths = sorted((_REPO / "docs").rglob("*.md")) + [_REPO / "README.md"]
    for path in paths:
        if not path.exists():
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1):
            if not any(door in line for door in _PREP_DOORS):
                continue
            for spelling in re.findall(r"--source[= ]+`?([A-Za-z0-9][\w.-]*)",
                                       line):
                if spelling.isupper():            # MODEL, NAME: metavariables
                    continue
                found.setdefault(spelling, []).append(
                    f"{path.relative_to(_REPO)}:{number}")
    return {name: tuple(places) for name, places in found.items()}


def test_the_documentation_scan_is_not_vacuous():
    """Validate the instrument: an empty scan would pass every assertion."""

    prose = _documented_prose_spellings()
    assert len(prose) >= 5, prose
    assert "hrrr" in prose and "mapped" in prose
    table = _alias_table()
    assert len(table) == len(source_adapters())


@pytest.mark.parametrize("spelling", sorted(_documented_prose_spellings()))
def test_every_documented_source_spelling_resolves(spelling):
    where = ", ".join(_documented_prose_spellings()[spelling])
    try:
        get_source_adapter(spelling)
    except ValueError as error:                     # pragma: no cover - the gate
        pytest.fail(f"--source {spelling} is documented at {where}: {error}")


def test_the_published_spelling_table_is_exactly_the_registry():
    """Docs and registry drift in either direction is the same defect."""

    table = _alias_table()
    registry = {
        adapter.source_id: tuple(adapter.aliases)
        for adapter in source_adapters()
    }
    assert set(table) == set(registry)
    for source_id, others in registry.items():
        assert table[source_id] == others, source_id


@pytest.mark.parametrize(
    "source_id,spelling",
    [(adapter.source_id, spelling)
     for adapter in source_adapters()
     for spelling in (adapter.source_id, *adapter.aliases)],
)
def test_every_registry_spelling_resolves_to_its_own_row(source_id, spelling):
    assert get_source_adapter(spelling).source_id == source_id


@pytest.mark.parametrize(
    "source_id,model_name",
    [(adapter.source_id, adapter.upstream_model_id)
     for adapter in source_adapters() if adapter.upstream_model_id],
)
def test_a_rows_own_model_name_is_a_spelling_the_front_door_accepts(
        source_id, model_name):
    """The name the model is KNOWN by, not only the row's id.

    A user reaches for the model's name first.  Where a row declares one,
    it has to select a row for that same model -- adding the alias is one
    table cell, and leaving it out costs a user a refusal on the name the
    upstream publisher uses.
    """

    resolved = get_source_adapter(model_name)
    assert resolved.upstream_model_id == model_name, (
        f"--source {model_name} (the model {source_id} decodes) resolves to "
        f"{resolved.source_id}, a row for a different model")
