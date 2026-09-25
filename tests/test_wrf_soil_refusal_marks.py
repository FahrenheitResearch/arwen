"""The Python side's word-match on the reader's refusals is held to them.

``gpuwm/ingest/wrf_soil_recovery.py`` decides WHICH REMEDY a user is
given by matching words inside the refusal the Rust reader returned:
``_LAYER_REFUSAL_MARKS`` means the declaration is wrong, and
``_CONVERSION_REFUSAL_MARK`` means both files are right and the
arithmetic between them is not.  Everything else means the wrong pair of
files was found.

That match is a hand transcription of another crate's sentences and
nothing pinned the pair.  Either side could be reworded alone: reword
the Rust and a user with a wrong declaration is told to go and find
files they already have, which is what a field report recorded;
reword the Python marks and the same thing happens silently.

The pairing is read out of the Rust SOURCE rather than out of a built
binary, deliberately.  The strings are compile-time literals, so the
source is the same evidence the binary would give, and a test that
needed ``cargo`` would skip on every machine that has no toolchain --
which is every machine this most needs to run on.  What the source
cannot show is whether the crate builds, and that is not this file's
question.

WHAT THIS FILE ASSERTS

* every refusal in the reader is accounted for, so a new one cannot
  appear unclassified;
* exactly the three refusals about the declared layer set carry a layer
  mark;
* exactly the one refusal about the conversion carries the conversion
  mark, and no refusal carries both, because the remedies are chosen in
  order and two marks on one sentence would make that order the thing
  deciding a user's answer.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuwm.ingest import wrf_soil_recovery

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
READER = (REPOSITORY_ROOT / "tools" / "rustwx" / "crates" / "rw-netcdf"
          / "src" / "soil_recovery.rs")

#: ``Err("...")`` and ``Err(format!("..."))``, which is every way this
#: module returns a refusal.
_REFUSAL = re.compile(r'Err\(\s*(?:format!\(\s*)?"((?:\\.|[^"\\])*)"', re.S)

#: The three refusals about the DECLARED LAYER SET, each by a fragment
#: that is NOT one of the marks, so this file does not check a word
#: against itself.  A rewording of any of the three fails here and is
#: then a deliberate change to both sides rather than a silent one.
LAYER_REFUSAL_FRAGMENTS = (
    "every source depth must fall inside one declared layer",
    "soil layer(s) and the authority declares only",
    "does not match declared layer",
)

#: The conversion refusal, likewise by a fragment that is not its mark.
CONVERSION_REFUSAL_FRAGMENT = "source-layer conversion produces an invalid"

#: The reader had 35 refusals when this file was written.  A floor, not
#: an equality: the point is that the extraction below is reading the
#: file rather than quietly matching nothing.
MINIMUM_REFUSALS = 30


def _refusals() -> tuple[str, ...]:
    text = READER.read_text(encoding="utf-8")
    return tuple(match.group(1) for match in _REFUSAL.finditer(text))


def _marked(message: str) -> bool:
    return any(mark in message
               for mark in wrf_soil_recovery._LAYER_REFUSAL_MARKS)


def test_the_reader_is_actually_being_read():
    """Validate the instrument before believing what it counts."""
    assert READER.is_file(), f"{READER} is gone; this test checks nothing"
    found = _refusals()
    assert len(found) >= MINIMUM_REFUSALS, (
        f"only {len(found)} refusals extracted from {READER.name}; the "
        "pattern has stopped matching and every count below is vacuous")


@pytest.mark.parametrize("fragment", LAYER_REFUSAL_FRAGMENTS)
def test_each_layer_refusal_carries_a_layer_mark(fragment):
    """The Python side must recognise this sentence as a layer refusal."""
    matching = [message for message in _refusals() if fragment in message]
    assert len(matching) == 1, (
        f"{fragment!r} appears in {len(matching)} refusals; the reader has "
        "been reworded and gpuwm/ingest/wrf_soil_recovery.py has to be "
        "brought with it")
    assert _marked(matching[0]), (
        f"the reader refuses with {matching[0]!r}, which carries none of "
        f"{wrf_soil_recovery._LAYER_REFUSAL_MARKS}, so a user whose "
        "declaration is wrong would be sent to look for files they "
        "already have")


def test_no_other_refusal_carries_a_layer_mark():
    """The marks have to be discriminating, not merely present."""
    marked = [message for message in _refusals() if _marked(message)]
    assert len(marked) == len(LAYER_REFUSAL_FRAGMENTS), (
        f"{len(marked)} refusals carry a layer mark: "
        + "; ".join(repr(message[:80]) for message in marked))
    for fragment in LAYER_REFUSAL_FRAGMENTS:
        assert any(fragment in message for message in marked)


def test_the_conversion_refusal_carries_its_own_mark_and_not_a_layer_one():
    """It is a third class, and the remedy order must not be what decides."""
    matching = [message for message in _refusals()
                if CONVERSION_REFUSAL_FRAGMENT in message]
    assert len(matching) == 1
    message = matching[0]
    assert wrf_soil_recovery._CONVERSION_REFUSAL_MARK in message
    assert not _marked(message), (
        "the conversion refusal carries a LAYER mark as well as its own, "
        "so which remedy a user gets is decided by the order of two ifs "
        "rather than by what went wrong")


def test_the_conversion_refusal_names_the_three_numbers_that_diagnose_it():
    """Value, layer index and thickness, because those separate its causes."""
    message = next(m for m in _refusals()
                   if CONVERSION_REFUSAL_FRAGMENT in m)
    for placeholder in ("{volume}", "{thickness}", "{value}", "{top}",
                        "{bottom}"):
        assert placeholder in message, (
            f"{placeholder} is not printed, so the refusal states that a "
            "conversion failed without stating what it converted")
    assert "authority layer index" in message


def test_the_conversion_remedy_names_what_the_reader_printed():
    """A remedy that repeats the reader's numbers, not a file hunt."""
    remedy = wrf_soil_recovery._conversion_remedy(
        "wrf-soil-authority.d01.json", "met_em.d01.2026-09-18_00.nc",
        {"source_variable": "SOILM", "source_quantity": "layer_water_mass",
         "source_units": "kg m-2"})
    assert "SOILM" in remedy
    assert "layer_water_mass" in remedy
    assert "kg m-2" in remedy
    assert "source_layer_bounds_m" in remedy
    assert "thickness" in remedy
    # It must NOT send the user back to look for the files it just read.
    assert "Keep the matching first met_em file" not in remedy
