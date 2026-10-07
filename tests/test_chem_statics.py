"""The chem preparation-time arrays reach the state on every preparation route.

The breakage this prevents: the dust statics and the sulfur latitude and
longitude were written only by gpuwm.runtime.prepare_real_case, so a chem
forecast prepared on the mapped (hrrr-prs), ERA5 or GFS route reached the dust
process with no erodible cell and was refused at init.  CPU-only: the routes
may build their state on the host, so the writer must not assume the card.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.chem_table import chem_names, load_sets
from gpuwm.core import chem_dust, chem_sulfur
from gpuwm.core.chem_state import process_attr
from gpuwm.core.chem_statics import attach_chem_statics, put_plane

ROUTES = ("gpuwm/runtime.py", "gpuwm/mapped_direct.py",
          "gpuwm/era5_direct.py", "gpuwm/gfs_direct.py")


def _state(table, ny=3, nx=4):
    state = SimpleNamespace(chem=SimpleNamespace(table=table))
    for module in (chem_dust, chem_sulfur):
        for alloc in module.ALLOCATES:
            shape = {"2d": (ny, nx), "rows_2d": (5, ny, nx)}.get(alloc.shape)
            if shape is not None:
                setattr(state, process_attr(alloc),
                        np.zeros(shape, dtype=alloc.dtype))
    return state


class _Grid:
    def latlon_mass(self):
        lat = np.linspace(30.0, 32.0, 12, dtype=np.float64).reshape(3, 4)
        return lat, -lat - 80.0


def test_host_state_takes_lat_lon_without_geography_and_dust_waits():
    table = load_sets(chem_names("gocart_simple"), ())
    state = _state(table)
    attach_chem_statics(state, _Grid(), None)
    by_name = {a.name: a for a in chem_sulfur.ALLOCATES}
    lat = getattr(state, process_attr(by_name["sulfur_xlat"]))
    np.testing.assert_array_equal(
        lat, _Grid().latlon_mass()[0].astype(np.float32))
    ready = {a.name: a for a in chem_dust.ALLOCATES}["dust_statics_ready"]
    # No geography: the dust statics stay unwritten, and the dust process's
    # init refuses by name rather than emitting from no erodible cell.
    assert not getattr(state, process_attr(ready)).any()


def test_chem_off_state_is_untouched():
    attach_chem_statics(SimpleNamespace(), object(), None)


def test_put_plane_writes_host_arrays_in_float32():
    state = SimpleNamespace(a=np.zeros((2, 2), np.float32))
    put_plane(state, "a", [[1.0, 2.0], [3.0, 1.0 / 3.0]])
    assert state.a.dtype == np.float32
    assert state.a[1, 1] == np.float32(1.0 / 3.0)


@pytest.mark.parametrize("route", ROUTES)
def test_every_real_preparation_route_attaches_the_chem_statics(route):
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / route).read_text(
        encoding="utf-8")
    assert "attach_chem_statics(initial_result.state, grid," in source, route
