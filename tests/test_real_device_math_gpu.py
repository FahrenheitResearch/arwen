"""Portable-library-dependent REAL gates, run after the sibling patch lands."""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest import real as host
from gpuwm.ingest import real_device as dev

pytestmark = requires_gpu
# The twins run on the card: cupy is this module's own import (GPU shard 1u).
cupy = pytest.importorskip("cupy")


def identical(a, b):
    b = b.get()
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", [
    "_potential_temperature_from_temperature", "_temperature_from_potential_temperature",
    "_moist_specific_volume", "_saturation_mixing_ratio", "_mixing_ratio_to_relative_humidity"])
def test_thermodynamics(name):
    rng = np.random.default_rng(28164)
    t = rng.uniform(180, 350, (17, 7, 11))
    p = rng.uniform(1000, 120000, t.shape)
    q = rng.uniform(0, 0.05, t.shape)
    args = (t, p)
    if name == "_moist_specific_volume":
        args = (t, q, p)
    elif name == "_saturation_mixing_ratio":
        rh = rng.uniform(-30, 130, t.shape)
        rh[0, 0, :5] = [np.nan, np.inf, -np.inf, 0.0, -0.0]
        args = (t, p, rh)
    elif name == "_mixing_ratio_to_relative_humidity":
        args = (t, p, q)
    identical(getattr(host, name)(*args), getattr(dev, name)(*args))


def test_surface_thermodynamics():
    rng = np.random.default_rng(28165)
    t = rng.uniform(210, 310, (17, 23))
    p = rng.uniform(75000, 110000, t.shape)
    z = rng.uniform(-20, 3500, t.shape)
    q = rng.uniform(-0.01, 0.035, t.shape)
    identical(host.surface_pressure_from_surface(p, z, z + 37.0, t, q),
              dev.surface_pressure_from_surface(p, z, z + 37.0, t, q))
    identical(host._surface_relative_humidity(t - 3.0, t),
              dev._surface_relative_humidity(t - 3.0, t))


def test_opt2_geopotential():
    rng = np.random.default_rng(28166)
    coord = make_vertical_coord(19, hybrid_opt=2)
    terrain = rng.uniform(-20, 3500, (7, 11))
    base = host._make_real_base(coord, terrain, 5000.0, 290.0, hypsometric_opt=2)
    mu = rng.uniform(65000, 98000, terrain.shape)
    alpha = rng.uniform(0.5, 20, (19, *terrain.shape))
    alpha[0, 0, :4] = [0.0, -0.0, 1e-45, 1e-40]
    identical(host._fp32_geopotential_split(base, coord, mu, alpha, 2),
              dev._fp32_geopotential_split(base, coord, mu, alpha, 2))
