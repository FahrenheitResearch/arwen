"""A reconstruction replaces allocation ownership without replacing the stepper."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import streaming


def test_moved_store_digest_reads_prognostic_keys_and_detects_poison():
    from gpuwm.core.streamed_relocation import StreamedChildReconstruction
    from gpuwm.ensemble.state_sha import live_state_sha256
    arrays = {"thp": np.zeros((2, 3, 4), np.float32),
              "mup": np.ones((3, 4), np.float32)}
    stream = SimpleNamespace(store={f"state/{key}": value.copy() for key, value in arrays.items()})
    state = SimpleNamespace(_streamed_domain=stream)
    before = StreamedChildReconstruction.state_digest(state)
    assert before == live_state_sha256(SimpleNamespace(**arrays))
    stream.store["state/thp"][0, 0, 0] = 999.
    assert StreamedChildReconstruction.state_digest(state) != before


def owner(*, closed=False, nx=8):
    run = SimpleNamespace(cfg=SimpleNamespace(nx=nx, ny=8, nz=4), closed=closed,
                          store={"scratch/test": np.ones((8, nx))}, drain=lambda: None)
    state = SimpleNamespace()
    return streaming.StreamedDomain(run, streaming.StreamingDecision(True, "test"), state=state)


def test_rebind_keeps_stepper_and_new_canonical_store():
    old, new = owner(closed=True), owner()
    old_state = old.state
    new_state = new.state
    new_run = new.tiled_run
    old.steps = 37
    old._frame = object()
    old.rebind_after_reconstruction(new, state=new_state)
    assert old.tiled_run is new_run
    assert old.store is new_run.store
    assert old.steps == 37
    assert old.state is new_state
    assert new_state._streamed_domain is old
    assert not hasattr(old_state, "_streamed_domain")
    assert old._frame is None
    assert new.tiled_run is None
    with pytest.raises(streaming.StreamingRefused, match="not the one"):
        old(old_state, new_run.cfg)


@pytest.mark.parametrize("old_closed,new_closed,nx,match", [
    (False, False, 8, "close the outgoing"),
    (True, True, 8, "open and unconsumed"),
    (True, False, 9, "never domain extent"),
])
def test_rebind_refuses_invalid_transfer_without_mutation(old_closed, new_closed, nx, match):
    old, new = owner(closed=old_closed), owner(closed=new_closed, nx=nx)
    before, after = old.tiled_run, new.tiled_run
    with pytest.raises(streaming.StreamingRefused, match=match):
        old.rebind_after_reconstruction(new, state=new.state)
    assert old.tiled_run is before
    assert new.tiled_run is after


def test_replacement_drain_failure_leaves_transfer_uncommitted():
    old, new = owner(closed=True), owner()
    before, after = old.tiled_run, new.tiled_run
    def fail():
        raise RuntimeError("drain failed")
    after.drain = fail
    with pytest.raises(RuntimeError, match="drain failed"):
        old.rebind_after_reconstruction(new, state=new.state)
    assert old.tiled_run is before
    assert new.tiled_run is after


def test_store_child_builder_binds_the_live_node(monkeypatch):
    from gpuwm.core import nest_stream
    node = SimpleNamespace(parent=object())
    bundle = SimpleNamespace(boundaries=None, template=object(), store={}, scalars={}, geography={})
    hook, factory = object(), object()
    monkeypatch.setattr(nest_stream, "make_nest_tile_hook", lambda actual: hook if actual is node else None)
    monkeypatch.setattr(streaming, "prepared_tile_state_factory", lambda *a, **kw: factory)
    monkeypatch.setattr(streaming, "attach", lambda *a, **kw: kw)
    monkeypatch.setattr(streaming, "streamed_store_inventory", lambda: object())
    result = streaming.store_domain_builder(bundle, node=node)(None, SimpleNamespace(), None)
    assert result["tile_hook"] is hook
    assert result["tile_state_factory"] is factory
    assert result["store"] is bundle.store
    bundle.boundaries = object()
    with pytest.raises(streaming.StreamingRefused, match="also carry tabulated"):
        streaming.store_domain_builder(bundle, node=node)(None, SimpleNamespace(), None)
