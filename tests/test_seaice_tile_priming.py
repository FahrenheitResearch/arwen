"""X12: a tile buffer carries the sea ice carriers its domain activated.

The model activates only on a domain containing ocean ice
(gpuwm.ocean.seaice.initialize), and a tile buffer is built on neutral
geography with none, so the 5 km CONUS WRF-input door (seaice_prognostic on,
ocean ice in the domain) was refused at attach: "missing on tile
['fields/ice_concentration_change', ...]".
"""
from __future__ import annotations

import sys
import types

import numpy as np

from gpuwm.ocean import seaice


def _numpy_as_cupy(monkeypatch):
    stub = types.ModuleType("cupy")
    stub.zeros_like = np.zeros_like
    monkeypatch.setitem(sys.modules, "cupy", stub)


def test_a_buffer_takes_the_domains_ice_carriers(monkeypatch):
    _numpy_as_cupy(monkeypatch)
    domain = {"tsk": np.ones((4, 5), np.float32),
              **{key: np.ones((4, 5), np.float32) for key in seaice.COLD_SEEDED}}
    tile = {"tsk": np.full((2, 3), 7, np.float32)}
    primed = seaice.prime_like(tile, domain)
    assert set(primed) == {f"fields/{key}" for key in seaice.COLD_SEEDED}
    for key in seaice.COLD_SEEDED:
        assert tile[key].shape == (2, 3) and tile[key].dtype == np.float32


def test_a_domain_without_ice_primes_nothing(monkeypatch):
    _numpy_as_cupy(monkeypatch)
    tile = {"tsk": np.zeros((2, 3), np.float32)}
    assert seaice.prime_like(tile, {"tsk": np.zeros((4, 5), np.float32)}) == ()
    assert set(tile) == {"tsk"}
