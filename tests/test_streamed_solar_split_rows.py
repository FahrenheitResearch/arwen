"""A streamed frame carries SWDDIR/SWDDIF/SWDDNI once radiation hands them over (X12).

A resident run's first frame lacks the three solar split rows and every frame
after the first radiation call has them (``io.history_layout``,
``_swdd_from_scheme``).  The streamed frame plan was taken once, at the first
frame, against a state whose driver never steps, so a streamed 5 km CONUS HRRR
run published 238 of the resident run's 241 variables at f01.
"""
from __future__ import annotations

from types import SimpleNamespace

from gpuwm.core import streaming


class _Driver:
    _swdd_from_scheme = False


def test_the_planned_state_is_measured_with_the_live_flag_and_restored():
    driver = _Driver()
    state = SimpleNamespace(physics=driver)
    with streaming._solar_split_as(state, True):
        assert driver._swdd_from_scheme is True
    assert "_swdd_from_scheme" not in vars(driver)
    assert driver._swdd_from_scheme is False
    driver._swdd_from_scheme = True
    with streaming._solar_split_as(state, False):
        assert driver._swdd_from_scheme is False
    assert driver._swdd_from_scheme is True
    with streaming._solar_split_as(state, None):
        assert driver._swdd_from_scheme is True


def _domain(flag):
    domain = streaming.StreamedDomain.__new__(streaming.StreamedDomain)
    physics = None if flag is None else SimpleNamespace(_swdd_from_scheme=flag)
    domain._run = SimpleNamespace(tiles=[SimpleNamespace(physics=physics)])
    return domain


def test_the_live_flag_is_read_off_a_tile_buffer():
    assert _domain(True)._live_solar_split() is True
    assert _domain(False)._live_solar_split() is False
    assert _domain(None)._live_solar_split() is None
    empty = streaming.StreamedDomain.__new__(streaming.StreamedDomain)
    empty._run = SimpleNamespace(tiles=[])
    assert empty._live_solar_split() is None
