"""CUDA uses the same original index plans for pre-published support."""
import numpy as np
import pytest
from conftest import requires_gpu
from test_atmospheric_window import source, target

from gpuwm.ingest.atmospheric_window import AtmosphericWindow, WindowedAtmosphericField
from gpuwm.ingest.horiz import _RegularGpuPlan, interpolate_era5_to_lambert

pytestmark = requires_gpu


@pytest.mark.parametrize("method", ["parabolic", "bilinear"])
@pytest.mark.parametrize("edge", [False, True])
def test_cuda_pre_cropped_operand_keeps_original_fp32_coordinates(method, edge):
    import cupy as cp
    values = np.random.default_rng(1242).normal(size=(3, 63, 81))
    y, x = np.array([[22.123456789]]), np.array([[34.123456789]])
    if edge:
        y, x = np.array([[0., 1.00000001]]), np.array([[79.9999999, 80.]])
    plan = _RegularGpuPlan(np.arange(63, dtype=np.float64), np.arange(81, dtype=np.float64), y, x)
    support = plan._source_support
    window = AtmosphericWindow((63, 81), (support.rows.start, support.rows.stop),
                                (support.columns.start, support.columns.stop))
    operand = WindowedAtmosphericField(window.crop(values).copy(), window)
    actual = plan.apply(operand, method=method, source_support=True)
    assert cp.asnumpy(actual).tobytes() == cp.asnumpy(plan.apply(values, method=method)).tobytes()


def test_cuda_actual_mapped_join_all_fields_on_two_domains(source):
    import cupy as cp
    _, bundle = source
    grids = (target(), target(37., -102.))
    full = bundle.regular_snapshots()[0]
    small = bundle.regular_snapshots().for_grids(grids)[0]
    for grid in grids:
        expected = interpolate_era5_to_lambert(full, grid, backend="cuda")
        actual = interpolate_era5_to_lambert(small, grid, backend="cuda")
        assert expected.fields.keys() == actual.fields.keys()
        for name in expected.fields:
            assert cp.asnumpy(actual.fields[name]).tobytes() == cp.asnumpy(expected.fields[name]).tobytes(), name
