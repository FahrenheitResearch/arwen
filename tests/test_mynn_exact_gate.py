"""The exact driver gate must inspect every recorded word, including zeros."""
import numpy as np
import pytest

import _mynn_families as F
import _mynn_families_gsd41 as G


def _record(module):
    blocks = next(iter(module.load(1 if module is F else False).values()))
    expected = module.step_outputs(blocks)
    return {name: value.copy() for name, value in expected.items()}, expected


@pytest.mark.parametrize("name", ("rqsblten", "dozone"))
def test_stock_gate_includes_recorded_inactive_tendencies(name):
    actual, expected = _record(F)
    actual[name].view(np.uint32).flat[0] ^= np.uint32(1)
    table = F.max_ulp_table([(1, actual, expected)])
    assert table.get(name, [0])[0] > 0


@pytest.mark.parametrize("module", (F, G), ids=("stock", "fork"))
def test_bit_gate_rejects_a_zero_sign_bit_change(module):
    actual, expected = _record(module)
    name = next(name for name in module.PROFILE_OUTPUTS
                if np.any(expected[name] == 0))
    where = np.flatnonzero(expected[name].reshape(-1) == 0)[0]
    actual[name].view(np.uint32).flat[where] ^= np.uint32(0x80000000)
    table = module.bit_mismatch_table([(1, actual, expected)])
    assert table[name] == [1]


@pytest.mark.parametrize("module", (F, G), ids=("stock", "fork"))
def test_bit_gate_counts_all_recorded_outputs(module):
    actual, expected = _record(module)
    table = module.bit_mismatch_table([(1, actual, expected)])
    assert set(table) == set(expected)
    assert not any(max(steps) for steps in table.values())
