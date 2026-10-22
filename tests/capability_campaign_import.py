"""Import lines the 2.9.4 capabilities campaign added to every imported TOML.

The namelist importer now writes, for every namelist, the switch lane/cap-
eclipse made default-on (``ra_sw_eclipse``, WRF's &physics key, ArWen's
default 1 with a reason naming the eclipses inside the run window) and
lane/cap-hail's HAILCAST selectors (``hailcast_opt`` at its automatic -1,
``haildt`` at WRF's 60 s), the way it already writes the lightning package.
Byte controls pinned before the campaign take exactly these three lines,
once each, out of today's text; every other byte is still the pin's.
"""
from __future__ import annotations

CAMPAIGN_IMPORT_LINES = (
    "ra_sw_eclipse = 1\n",
    "hailcast_opt = -1\n",
    "haildt = 60.0\n",
)


def without_campaign_import_lines(text: str) -> str:
    for line in CAMPAIGN_IMPORT_LINES:
        assert text.count(line) == 1, line
        text = text.replace(line, "", 1)
    return text
