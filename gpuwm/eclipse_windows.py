"""When WRF's eclipse table has an eclipse, as plain time windows.

WRF's ``ra_sw_eclipse = 1`` (Registry.EM_COMMON:2451) dims incoming
shortwave by the Moon's obscuration, computed from the Besselian elements in
``run/eclipse_besselian_elements.dat``.  phys/module_ra_eclipse.F reads a row
only when its year and day of year match the radiation call and the call's
hour (UTC, at ``xtime + radt``) lies inside the row's ``[tmin, tmax]``
(columns 4 and 5); at every other call the obscuration is zero and the
shortwave is exactly the one ``ra_sw_eclipse = 0`` computes.

This module answers one question from that table, for the namelist door:
does any eclipse row's window overlap a forecast window?  When none does,
the switch is inert for that run and imports as exactly what WRF integrates.
When one does, ArWen has no eclipse obscuration on this line, so the door
refuses rather than run an undimmed sun under the shadow.

The table is WRF 4.6.1's own file, packaged byte for byte and pinned by
SHA-256 in ``eclipse_manifest.json`` (the same pin the eclipse port reads,
so the two never disagree about which eclipses exist).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json

from gpuwm.data_assets import data_path

TABLE_PATH = data_path("wrf_radiation/eclipse_besselian_elements.dat")
MANIFEST_PATH = TABLE_PATH.with_name("eclipse_manifest.json")


@dataclass(frozen=True)
class EclipseWindow:
    """One table row's validity window, in UTC."""

    start: datetime
    end: datetime

    def label(self) -> str:
        return (f"{self.start:%Y-%m-%d %H:%M}-{self.end:%H:%M} UTC"
                if self.start.date() == self.end.date() or
                self.end == datetime.combine(self.start.date(),
                                             datetime.min.time())
                + timedelta(days=1)
                else f"{self.start:%Y-%m-%d %H:%M} to "
                     f"{self.end:%Y-%m-%d %H:%M} UTC")


@lru_cache(maxsize=1)
def eclipse_windows() -> tuple[EclipseWindow, ...]:
    """Every row of the pinned table as a UTC window, in table order.

    Refuses a table whose bytes do not match the manifest: a changed table
    moves or invents eclipses, and an import decision read from it would be
    about some other sky.
    """
    pin = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = TABLE_PATH.read_bytes()
    if (hashlib.sha256(data).hexdigest() != pin["sha256"]
            or len(data) != int(pin["bytes"])):
        raise ValueError(
            f"{TABLE_PATH} does not match {MANIFEST_PATH.name}: a changed "
            "Besselian table moves or invents eclipses")
    windows = []
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 32:
            raise ValueError(f"{TABLE_PATH}:{number} has {len(fields)} "
                             "columns; WRF's rows have 32")
        year, julday = int(fields[0]), int(fields[1])
        tmin, tmax = float(fields[3]), float(fields[4])
        day = datetime(year, 1, 1) + timedelta(days=julday - 1)
        windows.append(EclipseWindow(start=day + timedelta(hours=tmin),
                                     end=day + timedelta(hours=tmax)))
    return tuple(windows)


def eclipses_during(start: datetime, end: datetime
                    ) -> tuple[EclipseWindow, ...]:
    """The table windows that overlap ``[start, end]`` (inclusive)."""
    return tuple(window for window in eclipse_windows()
                 if window.start <= end and window.end >= start)
