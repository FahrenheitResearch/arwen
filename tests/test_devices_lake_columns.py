"""[devices] admission prices each rank's sparse lake arrays at its lake columns.

Breakage (open-defects ledger A13, 2026-10-05): a rank's CLM lake work
arrays are allocated at the rank's own lake column count
(``gpuwm.core.lake.LakeModel.refresh_columns``), but admission priced them
at every column of the rank, about 0.75 GiB of price per 2x2 rank of the
HRRR grid.  A bound that is too small would be the opposite failure (an
admitted rank exhausting its pool), so the bound must cover every column
the run's ``lakemask`` can mark.
"""

from dataclasses import replace
from datetime import datetime

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import preflight as pf
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.devices_memory import estimate_devices, rank_lake_column_bounds
from gpuwm.core.mynn_pbl_scratch import mynn_pricing_memory
from gpuwm.experiment import experiment_from_run_config
from gpuwm.prepared_single_domain_forecast import _devices_lake_mask


def _config(**kwargs):
    values = dict(nx=320, ny=240, nz=50, dx=3000.0, dy=3000.0,
                  ztop=20000.0, dt=15.0, run_seconds=120.0,
                  moist=True, mp_physics=28, bl_pbl_physics=5,
                  sf_sfclay_physics=5, sf_surface_physics=3,
                  num_soil_layers=9, ra_physics=0, cu_physics=0,
                  sf_lake_physics=1)
    return RunConfig(**(values | kwargs))


def test_rank_windows_count_their_own_lake_columns():
    from tilestream.multigpu import plan_split
    specs = plan_split(320, 240, 28, gx=2, gy=2, periodic_x=False,
                       periodic_y=False)
    mask = np.zeros((240, 320), bool)
    mask[10:20, 10:30] = True          # 200 columns, top-left corner only
    counts = rank_lake_column_bounds(specs, mask)
    for spec, count in zip(specs, counts):
        window = mask[spec.cj0:spec.cj0 + spec.cny, spec.ci0:spec.ci0 + spec.cnx]
        assert count == int(window.sum())
    assert max(counts) == 200 and min(counts) == 0


def test_a_lake_free_rank_prices_no_sparse_lake_work():
    cfg = _config()
    exp = replace(experiment_from_run_config(cfg, datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=4, grid=(2, 2)))
    mask = np.zeros((cfg.ny, cfg.nx), bool)
    mask[10:20, 10:30] = True
    with mynn_pricing_memory(total_bytes=96 * pf.GIB, free_bytes=96 * pf.GIB):
        bound = estimate_devices(exp, vram_gib=96)
        priced = estimate_devices(exp, vram_gib=96, lake_mask=mask)
    for before, after in zip(bound["rank_shapes"], priced["rank_shapes"]):
        assert after["resident_bytes"] < before["resident_bytes"]
        assert "lake_columns_bound" in after
    # The carried horizontal lake fields stay full size on every rank.
    full = pf.physics_array_shapes(cfg, lake_columns=0)
    assert full["fields/lake_columns"][1:] == (cfg.ny, cfg.nx)
    assert full["lake/columns"] == (full["lake/columns"][0], 0)


def test_the_bound_follows_both_lake_routes_and_drops_the_sea():
    """lakeini's flag-0 route (water at or above lake_min_elev) and the land
    cover's lake category; sea-level water of another category is no lake."""
    cfg = _config(nx=6, ny=4)
    landmask = np.ones((4, 6), np.float32)
    landmask[0, :] = 0                # sea, at 0 m
    landmask[2, 1] = 0                # an upland water cell
    landmask[3, 3] = 0                # a lake-category cell below 5 m
    hgt = np.full((4, 6), 200.0, np.float32)
    hgt[0, :] = 0.0
    hgt[3, 3] = -20.0
    lu = np.full((4, 6), 10.0, np.float32)
    lu[0, :] = 17
    lu[2, 1] = 17
    lu[3, 3] = 21
    bound = _devices_lake_mask(
        cfg, geography={"LANDMASK": landmask, "HGT_M": hgt, "LU_INDEX": lu},
        landuse={"ISLAKE": 21, "ISWATER": 17})
    assert bound.sum() == 2 and bound[2, 1] and bound[3, 3]
    # Without the land-use identity every water column counts.
    assert _devices_lake_mask(
        cfg, geography={"LANDMASK": landmask, "HGT_M": hgt, "LU_INDEX": lu}).sum() == 8


def test_the_prepared_bound_covers_water_and_the_lake_record():
    cfg = _config(nx=6, ny=4)
    landmask = np.ones((4, 6), np.float32)
    landmask[0, :2] = 0               # water
    lakes = np.zeros((4, 6), np.float32)
    lakes[3, 5] = 1                   # a recorded lake on a land cell
    bound = _devices_lake_mask(cfg, geography={"LANDMASK": landmask,
                                               "LAKEMASK": lakes})
    assert bound.sum() == 3 and bound[0, 0] and bound[3, 5]
    # After the store loader the run's own mask is exact.
    exact = np.zeros((4, 6), np.float32)
    exact[2, 2] = 1
    assert _devices_lake_mask(cfg, geography={"LANDMASK": landmask},
                              inventory={"driver/fields/lakemask": exact}).sum() == 1
    # Nothing readable, or no lake model: the every-column price stands.
    assert _devices_lake_mask(cfg, geography={}) is None
    assert _devices_lake_mask(replace(cfg, sf_lake_physics=0),
                              geography={"LANDMASK": landmask}) is None
