"""The batched native geometry must retain every old f64 word and refusal."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static import rust_bridge as bridge
from gpuwm.static.lambert import LambertGrid
from gpuwm.static.projection import MercatorGrid, PolarStereoGrid


@pytest.mark.parametrize("projection", [LambertGrid, MercatorGrid, PolarStereoGrid])
@pytest.mark.parametrize("hemisphere", [1, -1])
def test_batched_geometry_matches_native_words_all_staggers(projection, hemisphere):
    grid = projection(ref_lat=hemisphere * 35.0, ref_lon=179.0,
                      truelat1=hemisphere * 30.0, truelat2=hemisphere * 60.0,
                      stand_lon=179.0, dx=12000.0, dy=12000.0,
                      e_we=9, e_sn=8)
    grids = (grid, grid.nest(2, 2, 3, 6, 5), grid.translated(2, -1))
    kinds = (6, 0, 5, 2, 1, 4, 3)
    for current in grids:
        handle = current._rust_handle(bridge)
        for stagger in range(4):
            ny, nx = current._stagger_shape(stagger)
            batch = bridge.grid_arrays(handle, stagger, kinds, ny, nx)
            for kind, value in zip(kinds, batch):
                old = bridge.grid_array(handle, stagger, kind, ny, nx)
                assert value.shape == old.shape
                assert value.dtype == old.dtype
                assert value.tobytes() == old.tobytes()
            assert not np.shares_memory(batch[0], batch[1])
        fields = current.native_static_geometry()
        expected = {"MAPFAC_M": current.mapfac_m(),
                    "MAPFAC_U": current.mapfac_u(), "MAPFAC_V": current.mapfac_v()}
        expected["F"], expected["E"] = current.coriolis_m()
        expected["SINALPHA"], expected["COSALPHA"] = current.rotation_m()
        assert list(fields) == list(expected)
        assert all(fields[n].tobytes() == expected[n].tobytes() for n in fields)


def test_batched_geometry_rejects_invalid_abi_arguments_without_writing():
    grid = LambertGrid(ref_lat=35, ref_lon=-97, truelat1=30, truelat2=60,
                       stand_lon=-97, dx=12000, dy=12000, e_we=5, e_sn=4)
    handle = grid._rust_handle(bridge)
    library = bridge.load()
    function = library.gpuwm_static_grid_arrays
    out = np.full(84, 123.0)
    pointer = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    for args in ((handle, 0, 0, pointer, 84), (handle, 0, 128, pointer, 84),
                 (handle, 9, 127, pointer, 84), (handle, 0, 127, None, 84),
                 (handle, 0, 127, pointer, 83), (0, 0, 127, pointer, 84)):
        assert function(*args) != 0
        assert np.all(out == 123.0)


def test_old_library_and_repeated_fields_keep_independent_arrays(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "load", lambda: SimpleNamespace())
    monkeypatch.setattr(bridge, "_REPORTED_FALLBACKS", set())
    calls = []
    def old(handle, stagger, kind, ny, nx):
        calls.append(kind)
        return np.full((ny, nx), float(kind))
    monkeypatch.setattr(bridge, "grid_array", old)
    first = bridge.grid_arrays(1, 0, (4, 2, 4), 2, 3)
    second = bridge.grid_arrays(1, 0, (3,), 2, 3)
    assert calls == [4, 2, 4, 3]
    assert not np.shares_memory(first[0], first[2])
    first[0][0, 0] = 99
    assert first[2][0, 0] == 4
    assert second[0][0, 0] == 3
    assert capsys.readouterr().err.count("using the existing native array calls") == 1


def test_geometry_respects_overridden_public_field():
    class CustomGrid(LambertGrid):
        def mapfac_m(self):
            return np.full((3, 4), 42.0)
    grid = CustomGrid(ref_lat=35, ref_lon=-97, truelat1=30, truelat2=60,
                      stand_lon=-97, dx=12000, dy=12000, e_we=5, e_sn=4)
    assert np.all(grid.native_static_geometry()["MAPFAC_M"] == 42.0)
