"""Checkpoint radiation identities bind domain geography for every spectrum."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.radiation_composition import ComposedRadiation
from tilestream import driver, restart_stream


class Spectrum:
    def __init__(self, latitude, longitude):
        self.start_time = datetime(2026, 5, 29, 18)
        self.latitude_deg = latitude
        self.longitude_deg = longitude
        self.restart_identity = {"algorithm": "declared-test-spectrum"}


def radiation(latitude, longitude, shared):
    lw = Spectrum(latitude, longitude)
    sw = lw if shared else Spectrum(latitude.copy(), longitude.copy())
    return ComposedRadiation(lw.start_time, latitude, longitude,
                             longwave_adapter=lw, shortwave_adapter=sw)


@pytest.mark.parametrize("shared_leaf", [False, True])
@pytest.mark.parametrize("interrupt_header", [False, True])
def test_composed_header_uses_domain_geography_and_restores_each_leaf(shared_leaf, interrupt_header):
    lat = np.arange(30, dtype=np.float32).reshape(5, 6) + 40.
    lon = lat * np.float32(0.2)
    reference = radiation(lat, lon, shared_leaf)
    tiled = radiation(lat[1:4, 2:5].copy(), lon[1:4, 2:5].copy(), shared_leaf)
    state = SimpleNamespace(physics=SimpleNamespace(radiation_callable=tiled))
    setup = restart_stream.DomainSetup(arrays={}, scalars={}, scheme_geography={
        key: value for key, value in driver.geography_inventory(
            SimpleNamespace(physics=SimpleNamespace(radiation_callable=reference))).items()
        if not key.startswith("setup/")})
    owners = tuple(driver._scheme_geography_owners(state.physics))
    originals = [(owner, owner.latitude_deg, owner.longitude_deg) for _, owner in owners]
    class StopHeader(Exception):
        pass
    try:
        with restart_stream.domain_header_view(setup, state) as view:
            assert view.physics.radiation_callable.restart_identity() == reference.restart_identity()
            if interrupt_header:
                raise StopHeader
    except StopHeader:
        assert interrupt_header
    for owner, latitude, longitude in originals:
        assert owner.latitude_deg is latitude
        assert owner.longitude_deg is longitude
    assert tiled.restart_identity() != reference.restart_identity()
    # A changed actual input stays visible; this does not erase coordinate identity.
    setup.scheme_geography["radiation/component_0/latitude_deg"] = lat + 1.
    with restart_stream.domain_header_view(setup, state) as view:
        assert view.physics.radiation_callable.restart_identity() != reference.restart_identity()
