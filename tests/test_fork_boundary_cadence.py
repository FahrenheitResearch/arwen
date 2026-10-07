"""Shipped fork-source configurations take their lateral boundaries hourly.

plains-t2-bias (WOOF-FIX-PROGRAM 2026-10-06): on the Plains 2025-03-14
HRRR-lattice crop, with same-hour RAP 18Z boundaries, a ``[fetch] cadence
= 3`` row alone left the westernmost ASOS stations 0.94 K warm and 114
W/m2 sunnier than HRRR at f02 (the boundary state linearly interpolated
over 3 h thins the inflow cloud).  Hourly boundaries cut that to 0.52 K
and 42 W/m2 and put 2 m temperature within 0.08 K of HRRR at every hour
f01-f06 on the same 416-433 stations.  RAP and HRRR post every hour, the
route table's and the container registry's default spacing for every
fork source is already hourly, and the domain wizard writes that default
for a rap-native source when no cadence is named.  The 3 h row was a
written override: the WOOF-HRRR door's hand-authored configuration and
the two shipped HRRR recipes carried the same ``cadence = 3`` line.  This
guard holds every shipped configuration under ``configs/`` whose source
is a fork source to hourly boundaries or none, so the override cannot
return through a shipped file; a door that authors its own configuration
by hand is outside its reach.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tomllib

import pytest

from gpuwm import fetch, fetch_routes

ROOT = Path(__file__).parents[1]
SHIPPED = sorted((ROOT / "configs").rglob("*.toml"))
FORK_SOURCES = {"rap", "rap-native", "hrrr", "hrrr-native", "hrrr-prs"}


def _fork_configs():
    found = []
    for path in SHIPPED:
        fetch_table = tomllib.loads(path.read_text(encoding="utf-8")).get("fetch") or {}
        if fetch_table.get("source") in FORK_SOURCES:
            found.append(path)
    return found


def _relative_id(path: Path) -> str:
    return path.relative_to(ROOT / "configs").as_posix()


@pytest.mark.parametrize("path", _fork_configs(), ids=_relative_id)
def test_shipped_fork_config_boundaries_are_hourly(path):
    fetch_table = tomllib.loads(path.read_text(encoding="utf-8"))["fetch"]
    assert fetch_table.get("cadence", 1) == 1, (
        f"{_relative_id(path)}: a fork-source configuration writes "
        f"{fetch_table.get('cadence')} h boundaries; the source posts hourly "
        "and a coarser row warmed the inflow rim (plains-t2-bias, 2026-10-06)")


def test_the_fork_recipes_are_covered():
    names = {p.stem for p in _fork_configs()}
    assert {"hrrr_v4_gsd41", "hrrr_configuration_cut",
            "conus_hrrr_configuration", "hrrr_native_3km_demo"} <= names


@pytest.mark.parametrize("source", sorted(FORK_SOURCES))
def test_every_fork_source_defaults_to_hourly_boundaries(source):
    """A configuration that writes no cadence resolves hourly for every
    fork source, through the registry row and, for the table routes,
    through the route's own default."""
    assert fetch.container_default_cadence(source) == 1
    if source in fetch_routes.route_ids():
        assert fetch_routes.route_for(source).default_cadence == 1


def test_rap_native_route_default_is_hourly_over_a_twelve_hour_window():
    route = fetch_routes.route_for("rap-native")
    assert route.default_cadence == 1
    leads = fetch_routes.resolve_leads(route, datetime(2025, 3, 14, 18), 12)
    assert leads == tuple(range(13))
