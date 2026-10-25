"""A complete recipe run must seal its registered GSD reflectivity planes."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import state_digest
from gpuwm.core.gsd_diagnostics import PERSISTENT, WORK
from gpuwm.io import restart


@pytest.mark.parametrize("changed", ["refl_10cm_1km", "refl_10cm_4km"])
def test_complete_gsd_inventory_seals_and_hashes_held_reflectivity(monkeypatch, changed):
    from tilestream import physics_inventory

    state = SimpleNamespace(elapsed_seconds=3600., physics=None, _scratch={
        name: np.zeros((2, 3), np.float32) for name in (*PERSISTENT, *WORK)
    })
    state._scratch["refl_10cm"] = np.zeros((4, 2, 3), np.float32)
    base = {"state/thp": np.zeros((4, 2, 3), np.float32)}
    monkeypatch.setattr(restart, "state_manifest", lambda state: dict(base))
    clock = SimpleNamespace(dtbc_fp32=20.)
    before = state_digest.canonical_state_digest(state, clock)
    for name in PERSISTENT:
        assert f"scratch/{name}" in before["field_order"]
    state._scratch[changed][0, 0] = 17.25
    after = state_digest.canonical_state_digest(state, clock)
    assert before["sha256"] != after["sha256"]

    # The ordinary resident and store paths retain the same diagnostic bytes.
    store = {**base, **restart._scratch_manifest(state),
             **state_digest._canonical_extra_manifest(state)}
    monkeypatch.setattr(physics_inventory, "carrier_inventory", lambda value: value)
    streamed = state_digest.canonical_store_digest(
        store, {"elapsed_seconds": 3600., "driver": None}, clock)
    assert streamed == after


def test_unknown_reflectivity_member_is_still_rejected():
    state = SimpleNamespace(_scratch={"refl_unregistered": np.zeros((2, 3), np.float32)})
    with pytest.raises(RuntimeError, match="not a concrete registered member"):
        state_digest._canonical_extra_manifest(state)
