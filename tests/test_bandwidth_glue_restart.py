"""Pointer-bound launch tables must not enter checkpoint state."""

from types import SimpleNamespace

import numpy as np


def test_grouped_add_launch_is_rebuilt_infrastructure():
    from gpuwm.io.restart import state_manifest

    field = np.array([1.0, -0.0], dtype=np.float32)
    state = SimpleNamespace(thp=field)
    before = state_manifest(state)
    state._glue_add_launch = (((1234, 2), (5678, 2)), lambda: None)
    after = state_manifest(state)
    assert set(after) == set(before) == {'state/thp'}
    assert after['state/thp'] is field
