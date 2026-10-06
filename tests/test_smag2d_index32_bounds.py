"""Large or invalid grids retain wide scalar-flux addressing."""

import pytest

from gpuwm.core.smag2d import scalar_index32_fits


@pytest.mark.parametrize("shape", [
    (50, 600, 600), (50, 400, 400), (50, 1057, 1797),
    (1, 1, 1073741822),
])
def test_operational_and_last_safe_capacity_use_uint32(shape):
    assert scalar_index32_fits(*shape)


@pytest.mark.parametrize("shape", [
    (1, 1, 1073741823),  # Staggered box is 2**32 elements.
    (50, 10000, 10000), (0, 600, 600), (-1, 600, 600),
    (50, 0, 600), (50, 600, 0), (1, 1, 0x7fffffff),
    (1, 1, 0x80000000), (2**63, 2**63, 2**63),
])
def test_overflow_and_invalid_axes_retain_size_t(shape):
    assert not scalar_index32_fits(*shape)
