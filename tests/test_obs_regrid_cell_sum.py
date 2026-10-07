"""ABI-1 emission additions, gated against independent serial f64 arithmetic."""
import numpy as np
import pytest

from gpuwm import obs_regrid_bridge as bridge
from gpuwm.verify.cell_sum_ref import apply as reference
from gpuwm.verify.cell_sum_ref import ulp_table


@pytest.fixture
def sum_bridge():
    try:
        bridge._sum_library()
    except (FileNotFoundError, OSError, bridge.ObsRegridBridgeError) as error:
        pytest.skip(f"cell-sum library is not loadable: {error}")
    return bridge


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_sum_serial_float64_parity(sum_bridge, n):
    index = np.resize(np.array([0, 1, -1, 2], dtype=np.int64), 6*n*n)
    values = np.array([[1.0, 3.0, 0.1], [2**40, 2**-20, 7.0]])
    valid = np.array([[True, False, True], [True, True, True]])
    out, mask, totals = bridge.apply_sum_plan(source_index=index, values=values,
        valid=valid, destination_shape=(2,2), split_n=n)
    expected, expected_mask, expected_totals = reference(index, values, valid, (2,2), n)
    # All nonnegative finite outputs admit direct unsigned word ULP distance.
    for actual, golden in ((out, expected), (np.array(list(totals.values())), expected_totals)):
        assert ulp_table(actual,golden) == dict(max_ulp=0,nonzero=0,count=actual.size)
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("n", [4, 6])
def test_three_km_cell_split_onto_fine_grid(sum_bridge, n):
    lat, lon = np.meshgrid(np.arange(2)*0.027, 100+np.arange(2)*0.027, indexing="ij")
    dlat, dlon = np.meshgrid(-0.0135+(np.arange(2*n)+0.5)*0.027/n,
                            100-0.0135+(np.arange(2*n)+0.5)*0.027/n, indexing="ij")
    index, reachable, _ = bridge.build_plan(method="cell_sum_split", source_latitude=lat,
        source_longitude=lon, destination_latitude=dlat, destination_longitude=dlon,
        max_distance_m=20.0, split_n=n)
    out, valid, totals = bridge.apply_sum_plan(source_index=index, values=np.full((2,2),36.0),
        valid=np.ones((2,2),bool), destination_shape=dlat.shape, split_n=n)
    np.testing.assert_array_equal(out, np.full(dlat.shape,36.0/n**2))
    assert ulp_table(out,np.full(dlat.shape,36.0/n**2)) == dict(max_ulp=0,nonzero=0,count=out.size)
    assert valid.all() and reachable.all()
    assert totals == dict(total_source_mass=144.0,total_remapped_mass=144.0,
                          unreachable_mass=0.0,masked_mass=0.0)


def test_old_build_has_named_sum_refusal(monkeypatch):
    monkeypatch.setattr(bridge, "load", lambda: object())
    with pytest.raises(bridge.ObsRegridBridgeError, match="predates the cell-sum remap"):
        bridge._sum_library()


def test_irregular_split_refusal(sum_bridge):
    lat = np.array([[30.0,30.0],[30.01,30.012]])
    lon = np.array([[100.0,100.01],[100.0,100.01]])
    with pytest.raises(bridge.ObsRegridBridgeError,match="irregular source grid"):
        bridge.build_plan(method="cell_sum_split",source_latitude=lat,source_longitude=lon,
            destination_latitude=lat,destination_longitude=lon,max_distance_m=100,split_n=2)
