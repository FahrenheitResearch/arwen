"""The new Registry arrays use their soil axis and source output metadata."""
import pytest
from gpuwm.io.wrfout import WrfoutWriter


@pytest.mark.parametrize('name',('TVGL_URB','SMG_URB'))
def test_ground_layers_are_soil_not_atmospheric_levels(name):
    writer=object.__new__(WrfoutWriter)
    writer.nz=20;writer.ny=3;writer.nx=4;writer.soil_layers=4
    assert writer._dims_for(name,(4,3,4))==('Time','soil_layers_stag','south_north','west_east')
    with pytest.raises(ValueError,match='must have shape'):
        writer._dims_for(name,(20,3,4))
