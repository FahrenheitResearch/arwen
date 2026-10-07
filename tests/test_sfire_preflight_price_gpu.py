"""The preflight prices the SFIRE coupler at the bytes an actual one allocates."""
import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("spotting", [False, True])
@pytest.mark.parametrize("nfmc", [5, 7])
def test_fire_array_shapes_equal_an_actual_coupler(spotting, nfmc):
    import cupy as cp
    from test_sfire_restart_coupled_gpu import _coupled_state
    from gpuwm.core.preflight import fire_array_shapes
    from gpuwm.core.sfire_coupler import FireCoupler

    state, cfg, driver = _coupled_state(cp, spotting=spotting)
    cfg = dataclasses.replace(cfg, nfmc=nfmc)
    driver.fire = FireCoupler(state, cfg, driver.fields)
    actual = sum(int(value.nbytes) for value in driver.fire.arrays().values())
    tendencies = driver.fire_tendencies
    actual += sum(int(getattr(tendencies, name).nbytes)
                  for name in ("ru", "rv", "rtheta", "rqv", "rqc")
                  if getattr(tendencies, name, None) is not None)
    priced = sum(4 * int(np.prod(shape)) for shape in fire_array_shapes(cfg).values())
    assert priced == actual


from test_preflight import d01_cfg  # noqa: E402,F401  (the module fixture)


def test_fire_off_prices_nothing_and_fire_on_prices_the_coupler(d01_cfg):
    from gpuwm.core.preflight import physics_array_shapes
    assert not any(name.startswith(("fire/", "fire_tendencies/"))
                   for name in physics_array_shapes(d01_cfg))
    on = physics_array_shapes(dataclasses.replace(
        d01_cfg, ifire=2, sr_x=4, sr_y=4, moist=True))
    assert on["fire/fine_padded"] == (43, d01_cfg.ny * 4 + 2, d01_cfg.nx * 4 + 2)
