from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.streamed_state import CanonicalStateRefused, CanonicalStoreState


def test_canonical_state_never_exposes_last_slab_as_domain():
    slab = np.zeros((4, 2, 8), np.float32)
    full = np.ones((4, 24, 8), np.float32)
    driver = SimpleNamespace(fields={"glw": slab[0]}, stepra=3)
    template = SimpleNamespace(thp=slab, pb=slab+2, physics=driver,
                               c1h=np.ones(4), rogue=slab+4)
    geo = {"setup/pb": full+2}
    store = {"thp": full, "fields/glw": full[0], "scratch/test": full[0]}
    state = CanonicalStoreState(template, SimpleNamespace(nx=8, ny=24, nz=4),
        store=store, geography=geo, scalars={"elapsed_seconds": 720.},
        inventory={"thp": slab, "fields/glw": driver.fields["glw"]},
        geography_inventory={"setup/pb": template.pb})
    assert state.thp is full
    assert state.pb is geo["setup/pb"]
    assert state.physics.fields["glw"] is store["fields/glw"]
    state.physics.stepra = 7
    assert state.physics.stepra == 7
    assert driver.stepra == 3
    assert state.c1h is template.c1h
    assert state.ny == 24 and state.elapsed_seconds == 720.
    assert state.existing_scratch("test") is full[0] or np.shares_memory(state.existing_scratch("test"), full)
    with pytest.raises(CanonicalStateRefused, match="no canonical store"):
        _ = state.rogue
    with pytest.raises(CanonicalStateRefused, match="explicit bounded allocator"):
        state.scratch((4,), "coupler")


def test_canonical_state_missing_key_refuses_before_publication():
    with pytest.raises(CanonicalStateRefused, match="missing canonical array thp"):
        CanonicalStoreState(SimpleNamespace(), SimpleNamespace(nx=8, ny=24, nz=4),
            store={}, geography={}, scalars={}, inventory={"thp": np.zeros((4, 2, 8))},
            geography_inventory={})


def test_coupler_scratch_is_explicit_and_cached():
    calls = []
    def allocate(shape, slot, dtype):
        calls.append((shape, slot, dtype))
        return np.zeros(shape, dtype=dtype)
    state = CanonicalStoreState(SimpleNamespace(), SimpleNamespace(nx=8, ny=24, nz=4),
        store={}, geography={}, scalars={}, inventory={}, geography_inventory={},
        scratch_allocator=allocate)
    table = state.scratch((4, 32), "rolling", dtype=np.float32)
    assert state.scratch((4, 32), "rolling", dtype=np.float32) is table
    assert len(calls) == 1
    with pytest.raises(CanonicalStateRefused, match="shape/dtype mismatch"):
        state.scratch((4, 33), "rolling", dtype=np.float32)
