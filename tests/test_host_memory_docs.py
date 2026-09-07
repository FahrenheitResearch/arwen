"""The host-memory page, held against the decoders it describes.

A user's three-domain ERA5 forecast printed one word -- ``Terminated``
-- and nothing else.  Nothing in the product had priced the cost that
killed it: ``gpuwm check``'s itemization is device memory by its own
label (``IngestMemoryEstimate``, ``gpuwm/core/preflight.py``), and
``docs/public/HARDWARE.md`` sized the card carefully and said nothing at
all about RAM.  So the page now carries a host-memory section, and the
numbers in it are the ones a reader is going to multiply by their own
grid.

Which makes them exactly the kind of number that rots.  ``205 full 2-D
arrays`` is not a fact about ERA5; it is a fact about the retrieval THIS
code asks CDS for -- five pressure-level variables on the 37 standard
levels plus twenty single-level ones -- and it moves the day someone
adds a variable to :func:`gpuwm.fetch.era5_request_template`.  The same
goes for the GFS ladder in :mod:`gpuwm.gfs_direct`.  A page that quotes
a stale inventory is worse than one that quotes none, because a reader
sizes a machine from it.

So every figure the section states is recomputed here from the code that
decides it, and the committed table rows are rebuilt byte for byte.  The
binding is the enumeration kind (``tests/doc_command_parity.py`` names
the two): the code declares a set, the document presents a figure
derived from it, and the two are held against each other.  The command
bindings on the same section -- that its ``gpuwm fetch`` line names a
real door with real flags -- are already covered by
``tests/test_docs_extras_agree_with_code.py``.
"""

from __future__ import annotations

from datetime import datetime
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE = REPO_ROOT / "docs" / "public" / "HARDWARE.md"

#: The section heading the figures below live under.
_SECTION = "## Host memory (RAM), which nothing above prices"

#: Bytes one decoded value occupies on the host.  The decoders enforce
#: float64 (``gpuwm/ingest/grib.py`` refuses any other dtype into an
#: ``Era5Snapshot``), so this is not a choice the page is making.
_FLOAT64 = 8

#: How many copies of the decoded forcing the config-driven ``gpuwm
#: run`` route keeps: the raw partials cache and the frozen-snapshot
#: cache in ``gpuwm/ingest/grib.py`` hold disjoint bytes, and neither is
#: cleared.
_COPIES_RETAINED = 2

#: The two grids the worked table prices, and the valid-time count it
#: prices them over.
_GLOBAL_GRID = (721, 1440)
_REGIONAL_GRID = (161, 201)
_WORKED_VALID_TIMES = 8


def _section_text() -> str:
    text = HARDWARE.read_text(encoding="utf-8")
    assert _SECTION in text, (
        f"{HARDWARE.name} has no host-memory section; the page sizes the "
        f"card and says nothing about the RAM the forcing decode needs")
    body = text.split(_SECTION, 1)[1]
    return body.split("\n## ", 1)[0]


def _era5_fields_per_valid_time() -> tuple[int, int, int, int]:
    """``(pressure vars, levels, single vars, 2-D arrays)`` for ERA5.

    Read off the request the tool itself writes, so the page cannot
    quote an inventory the retrieval no longer asks for.
    """

    from gpuwm.fetch import (ERA5_PRESSURE_LEVELS_HPA, Area,
                             era5_request_template)

    template = era5_request_template(
        cycle=datetime(2014, 11, 4), hours=24,
        area=Area(25.0, -110.0, 50.0, -70.0))
    by_dataset = {request["dataset"]: request["request"]
                  for request in template["requests"]}
    pressure = by_dataset["reanalysis-era5-pressure-levels"]["variable"]
    single = by_dataset["reanalysis-era5-single-levels"]["variable"]
    levels = len(ERA5_PRESSURE_LEVELS_HPA)
    return (len(pressure), levels, len(single),
            len(pressure) * levels + len(single))


def _gfs_fields_per_valid_time() -> tuple[int, int, int, int]:
    """The same three counts for the certified GFS ladder."""

    from gpuwm import gfs_direct

    three = len(gfs_direct._THREE_D)
    levels = len(gfs_direct._CERTIFIED_PRESSURE_LEVELS_HPA)
    two = len(gfs_direct._TWO_D)
    return three, levels, two, three * levels + two


def _worked_row(label: str, grid: tuple[int, int], fields: int) -> str:
    """One row of the worked table, rebuilt from the inventory."""

    ny, nx = grid
    per_time = fields * _FLOAT64 * ny * nx
    total = per_time * _WORKED_VALID_TIMES * _COPIES_RETAINED
    if per_time >= 1024 ** 3:
        per_time_text = f"{per_time / 1024 ** 3:.2f} GiB"
    else:
        per_time_text = f"{per_time / 1024 ** 2:.1f} MiB"
    return (f"| {label} | {ny} x {nx} | {per_time_text} | "
            f"**{total / 1024 ** 3:.2f} GiB** |")


# ---------------------------------------------------------------------------
# The figures the section states
# ---------------------------------------------------------------------------

def test_the_page_states_the_era5_inventory_the_retrieval_asks_for():
    pressure, levels, single, fields = _era5_fields_per_valid_time()
    section = _section_text()
    sentence = (f"{pressure} pressure-level\nvariables on the {levels} "
                f"standard levels plus {single} single-level variables")
    assert " ".join(sentence.split()) in " ".join(section.split()), (
        f"{HARDWARE.name}'s host-memory section does not state the ERA5 "
        f"retrieval inventory as {pressure} x {levels} + {single}")
    assert f"{fields} full 2-D arrays" in " ".join(section.split()), (
        f"the section does not say that one ERA5 valid time is {fields} "
        f"2-D arrays, which is the number a reader multiplies")


def test_the_page_states_the_certified_gfs_ladder():
    three, levels, two, fields = _gfs_fields_per_valid_time()
    section = " ".join(_section_text().split())
    assert f"{three} x {levels} + {two} = {fields}" in section, (
        f"the section's GFS arithmetic is not {three} x {levels} + {two} "
        f"= {fields}, which is what gpuwm/gfs_direct.py decodes")


def test_the_worked_table_is_the_arithmetic_of_that_inventory():
    """The rows, rebuilt from the code, byte for byte."""

    _, _, _, fields = _era5_fields_per_valid_time()
    section = _section_text()
    for label, grid in (("global, 0.25 deg", _GLOBAL_GRID),
                        ("40 x 50 deg box, 0.25 deg", _REGIONAL_GRID)):
        row = _worked_row(label, grid, fields)
        assert row in section, (
            f"{HARDWARE.name}'s worked table does not carry the row the "
            f"decoder's own inventory produces:\n  want: {row}")


def test_the_row_builder_is_sensitive_to_the_inventory():
    """The control: a different field count is a different row.

    Without this, a builder that returned a constant would make the
    table test pass forever and prove nothing about the page.
    """

    _, _, _, fields = _era5_fields_per_valid_time()
    row = _worked_row("global, 0.25 deg", _GLOBAL_GRID, fields)
    assert _worked_row("global, 0.25 deg", _GLOBAL_GRID, fields + 1) != row
    assert _worked_row("global, 0.25 deg", _REGIONAL_GRID, fields) != row
    assert row not in _worked_row("global, 0.25 deg", _GLOBAL_GRID,
                                  fields + 1)


# ---------------------------------------------------------------------------
# The diagnostic the section sells
# ---------------------------------------------------------------------------

def test_run_seconds_really_is_absent_from_the_catalog_the_page_names():
    """The section tells a reader a short run does NOT shorten the decode.

    That advice is only useful because it is exact: the catalog is a
    function of the forcing FILES and of nothing else, so a 3600 s
    re-run isolates a decode failure from a forecast failure.  Give
    :func:`build_input_catalog` a forecast-length argument and the
    paragraph becomes wrong, silently, on a page a reader is debugging
    from.
    """

    from gpuwm.ingest.preflight import build_input_catalog

    assert list(
        inspect.signature(build_input_catalog).parameters) == ["case_data"], (
        "build_input_catalog now takes more than the case data; "
        "docs/public/HARDWARE.md tells readers a shorter run_seconds "
        "cannot shrink the decode, which may no longer be true")
    section = " ".join(_section_text().split())
    assert "run_seconds` is not an input to the catalog" in section


# ---------------------------------------------------------------------------
# The page is reachable from the top of itself
# ---------------------------------------------------------------------------

def test_the_opening_paragraph_points_at_the_host_memory_section():
    """The title says VRAM; a reader sizing RAM has to be sent there."""

    head = HARDWARE.read_text(encoding="utf-8").split(_SECTION, 1)[0]
    intro = head.split("## The short version", 1)[0]
    anchor = "#host-memory-ram-which-nothing-above-prices"
    assert anchor in intro, (
        "nothing above the fold on a page titled 'Hardware and VRAM "
        "sizing' tells a reader that host RAM is sized further down")


@pytest.mark.parametrize("phrase", [
    "float64",
    "8-16x",
    "Killed",
    "Terminated",
    "worker-01.stderr.log",
])
def test_the_section_names_what_the_reader_has_in_front_of_them(phrase):
    """The symptom and the rule of thumb, both spelled out.

    A reader arrives here holding one word from their shell.  The
    section is only findable if it contains that word.
    """

    assert phrase in _section_text(), phrase
