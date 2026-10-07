"""Native refined continuation cells must not become atmosphere aliases."""
from types import SimpleNamespace

import numpy as np
import pytest

from tilestream.fire_inventory import fire_field_layout, fire_transfer_segments, fire_wind_segments
from tilestream.hoststore import manifest_from_arrays
from tilestream.spec import plan_tiles
from tilestream.sfire import configure_tile


@pytest.mark.parametrize("periodic", [False, True])
def test_refined_owned_cells_and_physical_halos_have_exactly_one_writer(periodic):
    specs = plan_tiles(96, 72, 32, 24, 16, periodic=periodic)
    counts = np.zeros((72 * 3 + 2, 96 * 4 + 2), np.int32)
    for spec in specs:
        (tile_slice, full_slice), = fire_transfer_segments(spec, (3, 4, 1, 1), "scatter")
        counts[full_slice] += 1
        assert (tile_slice[0].stop-tile_slice[0].start, tile_slice[1].stop-tile_slice[1].start) == counts[full_slice].shape
        (source, dest), = fire_transfer_segments(spec, (3, 4, 1, 1), "gather")
        assert source[0].start >= 0 and source[1].start >= 0
        assert source[0].stop <= counts.shape[0] and source[1].stop <= counts.shape[1]
        assert dest[0].start >= 0 and dest[1].start >= 0
    assert np.all(counts == 1)


def test_refinement_is_bound_by_native_field_name_and_dimensions():
    shape = (218, 386)
    assert fire_field_layout("state/qv", shape, 72, 96) is None
    arrays = {"state/qv": np.zeros((16, 72, 96), np.float32),
              "fire/grid.lfn": np.zeros(shape, np.float32)}
    specs = {row.name: row for row in manifest_from_arrays(arrays, 16, 72, 96)}
    assert specs["fire/grid.lfn"].shape(16, 72, 96) == shape
    assert specs["fire/grid.lfn"].shape(16, 56, 64) == (170, 258)
    with pytest.raises(ValueError, match="unbound refinement"):
        fire_field_layout("fire/grid.lfn", (217, 386), 72, 96)


def test_periodic_atmosphere_tile_keeps_physical_fire_boundaries():
    spec = plan_tiles(96, 72, 32, 24, 16, periodic=True)[0]
    fire = SimpleNamespace(sr_x=4, sr_y=3, grid=SimpleNamespace())
    state = SimpleNamespace(physics=SimpleNamespace(fire=fire))
    configure_tile(state, spec)
    assert fire._tile_atmos_domain == (17, 64, 17, 56)
    assert fire.grid.domain == (65, 256, 49, 168)
    assert fire.grid.global_guard_domain == (65, 448, 49, 264)
    assert fire.grid.owned_domain == (65, 192, 49, 120)


def test_periodic_air_does_not_overwrite_native_fire_terminal_wind_with_alias():
    for name, shape in (("fire/uah", (72, 97)), ("fire/vah", (73, 96))):
        counts = np.zeros(shape, np.int32)
        for spec in plan_tiles(96, 72, 32, 24, 16, periodic=True):
            (local, owned), = fire_wind_segments(spec, name, "scatter")
            counts[owned] += 1
            if owned[1].stop == 97:
                assert spec.i1 == 96
                assert local[1].stop == spec.nx - spec.ci0 + 1
        assert np.all(counts == 1)


def test_declared_moisture_class_axes_survive_nz_collision_and_global_history_time():
    names = ("fire/grid.moisture.fmc_gc", "fire/grid.moisture.fmc_equi",
             "fire/grid.moisture.fmc_lag")
    values = {"state/qv":np.zeros((16,12,17),np.float32),
        "fire/lfn_time":np.array([-125.5],np.float32),
        **{name:np.zeros((16,12,17),np.float32) for name in names}}
    manifest = {row.name:row for row in manifest_from_arrays(values,16,12,17)}
    for name in names:
        assert manifest[name].shape(59,640,360) == (16,640,360)
    assert manifest["state/qv"].shape(59,640,360) == (59,640,360)
    assert manifest["fire/lfn_time"].shape(59,640,360) == (1,)
