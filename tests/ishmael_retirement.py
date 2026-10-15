"""Historical import bytes from before the namelist door rule (2026-10-12).

Two importer decisions changed on the golden namelist lane, and the byte
controls pinned before it were generated with the old ones:

* ``mp_physics = 55`` (Jensen-ISHMAEL) imported as Morrison (10), and the
  emitted TOML said so in one header line.  It is now refused by name
  (gpuwm.config._MP_UNPORTED_SCHEMES).  The pinned inputs asked for 55;
  these helpers ask for Morrison by its own number instead and take exactly
  the retired header line out of the historical bytes.
* an omitted ``&dynamics top_lid`` resolved to the profile or RunConfig
  rigid lid (``top_lid = true``); it is now WRF's Registry default, the
  open top (``top_lid = false``).  The pinned inputs omit top_lid, so the
  one ``top_lid`` line of the historical bytes is read as the open top.

Exactly those two lines, once each, and nothing more -- every other byte is
still the parent's.
"""
from __future__ import annotations

ISHMAEL_SPELLING = "mp_physics = 55, 55"
MORRISON_SPELLING = "mp_physics = 10, 10"
RETIRED_LINE = ("#   mp_physics 55 (ISHMAEL) -> mp_physics 10 "
                "(Morrison 2-moment)\n")
SUBSTITUTION_HEADER = "# Physics substitutions (ratified, never silent):\n"
RIGID_LID_LINE = "\ntop_lid = true\n"
OPEN_TOP_LINE = "\ntop_lid = false\n"


def requested_by_own_number(input_text: str) -> str:
    """The pinned input with ISHMAEL's spelling replaced by Morrison's."""
    assert input_text.count(ISHMAEL_SPELLING) == 1, input_text
    assert "top_lid" not in input_text, "the pinned inputs omit top_lid"
    return input_text.replace(ISHMAEL_SPELLING, MORRISON_SPELLING)


def without_retired_line(old: bytes) -> bytes:
    """The historical bytes as today's door writes them for the same
    request: the one ISHMAEL substitution line removed and the omitted
    top_lid read as WRF's open top."""
    line = RETIRED_LINE.encode("utf-8")
    assert old.count(line) == 1, "the pinned bytes carry the retired line once"
    rigid, open_top = RIGID_LID_LINE.encode(), OPEN_TOP_LINE.encode()
    assert old.count(rigid) == 1, "the pinned bytes resolved the lid rigid"
    return old.replace(line, b"").replace(rigid, open_top)


def with_retired_line(emitted: str) -> str:
    """``emitted`` as the parent wrote it, for controls pinned by hash: the
    retired line put back first under the substitution header, and the
    omitted top_lid written as the parent's rigid lid."""
    assert emitted.count(SUBSTITUTION_HEADER) == 1, emitted
    assert emitted.count(OPEN_TOP_LINE) == 1, emitted
    return emitted.replace(
        SUBSTITUTION_HEADER, SUBSTITUTION_HEADER + RETIRED_LINE).replace(
        OPEN_TOP_LINE, RIGID_LID_LINE)
