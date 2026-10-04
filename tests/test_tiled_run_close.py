"""CPU lifecycle proof: drain before releasing tile/closure allocation owners."""
import gc
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from tilestream.driver import TiledRun, TiledRunError


class Resource:
    pass


def run_stub(monkeypatch, *, failing=False):
    run = object.__new__(TiledRun)
    log = []
    failure = [failing]
    current = [0]
    class Device:
        def __init__(self, device):
            self.id = device
        def __enter__(self):
            self.previous = current[0]
            current[0] = self.id
            return self
        def __exit__(self, *exc):
            current[0] = self.previous
    class Stream:
        def __init__(self, name):
            self.name = name
        def synchronize(self):
            assert current[0] == 1, "stream synchronization used the caller's card"
            log.append(self.name)
            if failure[0]:
                raise RuntimeError("drain failed")
    def synchronize_device():
        assert current[0] == 1, "the stream owner card was not synchronized"
        log.append("device")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        Device=Device, runtime=SimpleNamespace(
            deviceSynchronize=synchronize_device, getDevice=lambda: current[0]))))
    run._device_id = 1
    run._pending = True
    run._closed = False
    run._streams = [Stream("compute")]
    run._copy_in = [Stream("copy-in")]
    run._copy_out = [Stream("copy-out")]
    run._home = {"carrier": np.arange(12, dtype=np.float32)}
    run.tiles = [Resource(), Resource()]
    run.graph_steppers = [Resource()]
    run.health = Resource()
    run.observer = Resource()
    # Model the constructor's three closures, which retain additional owners
    # independently of the public tiles/graph lists.
    def closure():
        resource = Resource()
        return lambda *a, **kw: resource, weakref.ref(resource)
    refs = [weakref.ref(v) for v in run.tiles + run.graph_steppers]
    refs += [weakref.ref(run.health), weakref.ref(run.observer)]
    for key in ("_sweep", "_set_live_config", "_reseed_clock"):
        fn, ref = closure()
        setattr(run, key, fn)
        refs.append(ref)
    run.scalars = {"elapsed_seconds": 12, "call_counts": {"radiation": 2}}
    run.cfg = SimpleNamespace(grid_id=2)
    return run, log, failure, refs


def test_close_drains_all_streams_and_releases_closure_owners(monkeypatch):
    run, log, _, refs = run_stub(monkeypatch)
    store = run._home
    original = store["carrier"].copy()
    run.close()
    assert log == ["compute", "copy-in", "copy-out", "device"]
    assert sys.modules["cupy"].cuda.runtime.getDevice() == 0
    assert run.closed and not run._pending
    gc.collect()
    assert all(ref() is None for ref in refs)
    # Closing the owner neither clears nor mutates external canonical storage.
    np.testing.assert_array_equal(store["carrier"], original)
    assert run.scalars == {"elapsed_seconds": 12, "call_counts": {"radiation": 2}}
    run.close()
    run.drain()
    assert len(log) == 4


def test_failed_drain_keeps_every_owner_live_and_can_retry(monkeypatch):
    run, log, failure, refs = run_stub(monkeypatch, failing=True)
    store = run._home
    with pytest.raises(RuntimeError, match="drain failed"):
        run.close()
    assert sys.modules["cupy"].cuda.runtime.getDevice() == 0
    assert not run.closed and run._pending
    assert run._home is store
    assert all(ref() is not None for ref in refs)
    failure[0] = False
    run.close()
    gc.collect()
    assert run.closed and all(ref() is None for ref in refs)


def test_close_drains_when_failed_sweep_never_marked_pending(monkeypatch):
    run, log, _, refs = run_stub(monkeypatch)
    run._pending = False
    run.close()
    assert log == ["compute", "copy-in", "copy-out", "device"]
    assert run.closed and all(ref() is None for ref in refs)


def test_compute_wait_uses_owner_card_and_restores_caller(monkeypatch):
    """A store reader on card 0 must wait the tile's card-1 stream."""
    run, log, _, _ = run_stub(monkeypatch)
    run.sync_compute()
    assert log == ["compute"]
    assert sys.modules["cupy"].cuda.runtime.getDevice() == 0
    assert run._pending


def test_sweep_uses_owner_card_and_restores_caller(monkeypatch):
    """Re-entering saved tile streams from a different card must not fail."""
    from gpuwm.core import dycore

    run, log, _, _ = run_stub(monkeypatch)
    def sweep(*args):
        assert sys.modules["cupy"].cuda.runtime.getDevice() == 1
        log.append("sweep")
    run._sweep = sweep
    monkeypatch.setattr(dycore, "finish_wrf_cfl_domain_step", lambda *a, **kw: None)
    run.sweep()
    assert log == ["sweep"]
    assert sys.modules["cupy"].cuda.runtime.getDevice() == 0


@pytest.mark.parametrize("operation", ["store", "sweep", "reseed", "sync"])
def test_closed_owner_refuses_stale_use(monkeypatch, operation):
    run, _, _, _ = run_stub(monkeypatch)
    run.close()
    with pytest.raises(TiledRunError, match="closed"):
        if operation == "store":
            _ = run.store
        elif operation == "sweep":
            run.sweep()
        elif operation == "reseed":
            run.reseed_clock({})
        else:
            run.sync_compute()


@pytest.mark.parametrize("fail", [False, True])
def test_whole_run_entry_closes_its_owner_on_success_and_failure(monkeypatch, fail):
    from tilestream import driver
    log = []
    class Owner:
        def __init__(self, *a, **kw):
            log.append("construct")
        def sweep(self, *a, **kw):
            log.append("sweep")
            if fail:
                raise RuntimeError("step failed")
        def close(self):
            log.append("close")
    monkeypatch.setattr(driver, "TiledRun", Owner)
    if fail:
        with pytest.raises(RuntimeError, match="step failed"):
            driver.run_tiled({}, object(), 12, 12)
    else:
        driver.run_tiled({}, object(), 12, 12)
    assert log == ["construct", "sweep", "close"]


@pytest.mark.parametrize("sweep_fails", [False, True])
def test_whole_run_preserves_primary_error_when_close_fails(monkeypatch, sweep_fails):
    from tilestream import driver
    primary = RuntimeError("original step failed")
    cleanup = RuntimeError("copy-out drain failed")
    class Owner:
        def __init__(self, *a, **kw):
            pass
        def sweep(self, *a, **kw):
            if sweep_fails:
                raise primary
        def close(self):
            raise cleanup
    monkeypatch.setattr(driver, "TiledRun", Owner)
    with pytest.raises(RuntimeError) as caught:
        driver.run_tiled({}, object(), 12, 12)
    if sweep_fails:
        assert caught.value is primary
        assert primary.__notes__ == [
            "tiled-run cleanup also failed: RuntimeError: copy-out drain failed"]
    else:
        assert caught.value is cleanup
