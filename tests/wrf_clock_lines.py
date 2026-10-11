"""The emission a fixed-step WRF namelist gained in 2.8.9 (D8).

A fixed-step WRF namelist imports with ``terrain_clock = "pinned"``, so
the importer runs the acoustic substep count wrf.exe runs instead of the
terrain clock's (tests/test_wrf_acoustic_count_default.py).  The byte
controls pinned before that ruling compare everything else: these helpers
take out exactly the D8 lines, once, and nothing more.
"""
from __future__ import annotations

import re

#: The block :func:`gpuwm.namelist_import._translate_namelists` emits for a
#: fixed-step WRF namelist.
WRF_CLOCK_BLOCK = (
    "# The namelist's fixed step and acoustic count are the ones wrf.exe runs;\n"
    "# the engine runs them as written and records the terrain clock's verdict\n"
    "# as advice in the receipt.\n"
    'terrain_clock = "pinned"\n')

#: The pre-D8 report line for an omitted time_step_sound that WRF's rule
#: resolves to 4 (the only count the pinned controls hold).
_OLD_SOUND_LINE = (
    "  time_step_sound = 4: WRF Registry default 0 means automatic "
    "selection; 4 is WRF's choice at conventional dt/dx and gpuwm requires "
    "an explicit even value")
_NEW_SOUND_LINE = re.compile(
    r"^  time_step_sound = 4: WRF Registry default 0 means WRF derives the "
    r"count \(solve_em\.F: .*; at .* that is 4$", re.M)
_CLOCK_LINE = re.compile(
    r"^  terrain_clock = pinned: a fixed-step WRF namelist names the acoustic "
    r"substep count wrf\.exe runs \(\d+\);.*\n", re.M)


def without_wrf_clock_lines(text: str) -> str:
    """``text`` without the D8 block; a fixed-through-adaptive import
    (which carried its own pinned block before D8) has none to take."""
    count = text.count(WRF_CLOCK_BLOCK)
    assert count in (0, 1), count
    return text.replace(WRF_CLOCK_BLOCK, "")


def without_wrf_clock_report(text: str) -> str:
    """A SubstitutionReport.format() as the pre-D8 importer wrote it."""
    text, clocks = _CLOCK_LINE.subn("", text)
    assert clocks in (0, 1), clocks
    text, sounds = _NEW_SOUND_LINE.subn(_OLD_SOUND_LINE, text)
    assert sounds in (0, 1), sounds
    return text
