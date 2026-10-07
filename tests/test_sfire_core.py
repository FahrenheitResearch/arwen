"""Geometry and option contracts that do not open a CUDA device.

Native Fortran output-word comparisons live in test_sfire_wrf471_parity.py.
"""

import pytest

from gpuwm.core.sfire_core import CoreOptions, IgnitionLine, _bounds, _tiles


def test_tiles_cover_domain_without_stage_overlap():
    domain = _bounds((42,48),domain=(4,43,4,37))
    assert _tiles(domain,[(4,23,4,37),(24,43,4,37)]) == (
        (4,23,4,37),(24,43,4,37))
    with pytest.raises(ValueError,match="Overlapping"):
        _tiles(domain,[(4,24,4,37),(24,43,4,37)])
    with pytest.raises(ValueError,match="cover every"):
        _tiles(domain,[(4,23,4,37)])


def test_domain_retains_boundary_derivative_halo():
    assert _bounds((6,9)) == (1,7,1,4)
    with pytest.raises(ValueError,match="surrounding halo"):
        _bounds((6,9),domain=(0,7,1,4))
    with pytest.raises(ValueError,match="two nodes"):
        _bounds((3,9))


def test_options_name_wrfs_unimplemented_equations():
    assert CoreOptions().upwinding == 9
    assert CoreOptions().upwinding_reinit == 4
    with pytest.raises(ValueError,match="Lax-Friedrichs"):
        CoreOptions(upwind_split=2)
    with pytest.raises(ValueError,match="fuel method 2"):
        CoreOptions(fuel_left_method=2)


def test_ignition_requires_defined_propagation_time():
    args = dict(start_x=0,start_y=0,end_x=10,end_y=0,start_time=1,end_time=2,radius=5)
    with pytest.raises(ValueError,match="positive"):
        IgnitionLine(**args,ros=0)
    assert IgnitionLine(**args,ros=1).end_x == 10
