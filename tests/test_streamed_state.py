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


def _canonical_projection_owner():
    from gpuwm.core.streaming import StreamedDomain

    slab = np.zeros((4, 2, 8), np.float32)
    full = np.full((4, 24, 8), 7.0, np.float32)
    plane = np.full((24, 8), 3.0, np.float32)
    template = SimpleNamespace(thp=slab)
    store = {"state/thp": full, "scratch/uh_follow_window": plane}
    state = CanonicalStoreState(
        template, SimpleNamespace(nx=8, ny=24, nz=4), store=store,
        geography={}, scalars={}, inventory={"state/thp": slab},
        geography_inventory={})
    owner = object.__new__(StreamedDomain)
    owner._state = state
    owner._run = SimpleNamespace(store=store)
    owner.host_store = True
    return owner, state, template, store


def test_canonical_parent_projection_borrows_live_arrays_without_device_copy(
        monkeypatch):
    import cupy as cp

    owner, state, template, store = _canonical_projection_owner()

    def no_device_copy(*_args, **_kwargs):
        pytest.fail("a canonical parent projected its full host domain to the device")

    monkeypatch.setattr(cp, "asarray", no_device_copy)
    monkeypatch.setattr(cp, "asnumpy", no_device_copy)
    store["state/thp"] += 11.0
    assert owner.sync_to_state() == 0
    assert state.thp is store["state/thp"]
    assert np.all(state.thp == 18.0)
    assert np.all(template.thp == 0.0)
    state.existing_scratch("uh_follow_window").fill(0.0)
    assert owner.sync_from_state(("scratch/uh_follow_window",)) == 0
    assert np.all(store["scratch/uh_follow_window"] == 0.0)
    assert owner.sync_to_state(("state/thp",), window=(2, 5, 1, 4)) == 0


@pytest.mark.parametrize("direction", ["to", "from"])
def test_canonical_projection_refuses_a_replaced_store_instead_of_reading_stale(
        direction):
    from gpuwm.core.streaming import StreamingRefused

    owner, _, _, store = _canonical_projection_owner()
    owner._run.store = {key: value.copy() for key, value in store.items()}
    with pytest.raises(StreamingRefused, match="does not alias.*stale domain"):
        if direction == "to":
            owner.sync_to_state(("state/thp",))
        else:
            owner.sync_from_state(("state/thp",))


def test_canonical_proxy_metadata_is_precise_infrastructure_not_a_cache_exemption():
    from gpuwm.io import restart

    owner, state, _, _ = _canonical_projection_owner()
    for name in ("_template_metadata", "_canonical_arrays", "_view_cache",
                 "_canonical_store", "_canonical_geography", "_canonical_scalars",
                 "_scratch_allocator", "cfg", "nx", "ny", "nz"):
        assert restart.classify_state_attr(name) == "infra"
    state._canonical_new_carrier = np.ones((24, 8), np.float32)
    with pytest.raises(restart.RestartManifestError, match="_canonical_new_carrier"):
        owner.sync_to_state()
    with pytest.raises(restart.RestartManifestError, match="_canonical_new_carrier"):
        restart.state_manifest(state)
