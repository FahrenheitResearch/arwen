"""The single-layer UCM wired end to end through the physics driver.

The column, Noah and Noah-MP oracles prove the kernel against WRF; this
proves the wiring: ``sf_urban_physics = 1`` builds through
``initialize_physics``, the driver runs its surface step with the real
``gpuwm.core.urban_ucm`` on the city columns, and nothing reaches a column
that is not a city.  Against the same driver at ``sf_urban_physics = 0``,
after three surface steps:

* every field on the non-urban columns is byte-identical (the UCM and its
  hand-over touch urban columns only, and every scheme between them is
  column-local);
* the city columns' skin temperature, fluxes and 2 m temperature moved;
* the urban state advanced from its cold start, and the surface-driver
  overrides published the UCM's own U10/V10/PSIM/PSIH/GZ1OZ0/AKMS and
  AKHS = CHS on the city columns, as module_surface_driver.F:3001-3021 does.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

_STEPS = 3


def _run(option: int, **overrides):
    import cupy as cp

    from test_urban_default_off_identity import _small_driver

    state, cfg, driver = _small_driver(sf_urban_physics=option, **overrides)
    for _ in range(_STEPS):
        driver.compute(state, cfg)
    fields = {name: cp.asnumpy(value) for name, value in driver.fields.items()
              if hasattr(value, "dtype")}
    return driver, fields


@pytest.mark.gpu
@requires_gpu
def test_the_ucm_runs_in_the_driver_and_touches_only_city_columns():
    _, off = _run(0)
    driver, on = _run(1)
    urban = driver.urban
    city = np.zeros(off["tsk"].shape, dtype=bool)
    city[:, :2] = True                       # _small_driver's ISURBAN cells
    assert np.array_equal(np.asarray(urban.urban_mask.get()), city)

    for name, before in off.items():
        after = on[name]
        if after.shape != before.shape or before.ndim < 2:
            continue
        rural = before[..., ~city], after[..., ~city]
        assert rural[0].tobytes() == rural[1].tobytes(), name
    for name in ("tsk", "hfx", "lh", "qfx", "t2", "albedo", "grdflx"):
        assert not np.array_equal(off[name][city], on[name][city]), name
    for name in list(on):
        if on[name].dtype.kind == "f":
            assert np.isfinite(on[name]).all(), name

    # the urban state left its cold start (TSK = 295 K everywhere)
    for name in ("tr_urb2d", "tb_urb2d", "tg_urb2d", "tc_urb2d"):
        assert np.all(on[name][city] != np.float32(295.0)), name
    assert np.all(on["ust_urb2d"][city] > 0)
    # module_surface_driver.F:3001-3021 on the city columns
    for grid, urb in (("u10", "u10_urb2d"), ("v10", "v10_urb2d"),
                      ("psim", "psim_urb2d"), ("psih", "psih_urb2d"),
                      ("gz1oz0", "gz1oz0_urb2d"), ("akms", "akms_urb2d")):
        assert on[grid][city].tobytes() == on[urb][city].tobytes(), grid
    assert on["akhs"][city].tobytes() == on["chs"][city].tobytes()
    assert np.all(on["chs"][city] >= np.float32(0.01))
