"""Original-step entry timing, owner restoration and failure lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from gpuwm.ensemble.scheduled_production_physics import ScheduledProductionPhysics


class _Owner:
    def __init__(self, members=2):
        self.members, self._active, self._error = members, False, None
        self.drivers = tuple(SimpleNamespace(state=SimpleNamespace(physics=None),
            radiation_due_override=None, metadata=f"member {member}") for member in range(members))
        for driver in self.drivers:
            driver.state.physics = driver
        self.trace = []

    def begin(self, configs):
        self.trace.append(("begin", configs, tuple(driver.radiation_due_override for driver in self.drivers)))
        self._active = True

    def member_compute(self, member, state, cfg):
        self.trace.append(("compute", member, cfg))
        return member

    def abort(self, error):
        self._error = error

    def finish(self):
        self._active = False
        self.trace.append(("finish",))
        if self._error is not None:
            raise self._error

    def close(self):
        assert not self._active
        self.trace.append(("close",))


def _group():
    group = object.__new__(ScheduledProductionPhysics)
    group.owner = _Owner()
    group._initialize_coordination()
    return group


def test_metadata_is_captured_only_at_actual_physics_entry_after_original_refreshes():
    group = _group()
    group.arm()
    arrived = threading.Barrier(2)
    def callback(member):
        state = group.owner.drivers[member].state
        with group.member_scope(member):
            # This stands for the original callback's own before_step and
            # diagnostics. Arm must never run either of those itself.
            state.physics.radiation_due_override = True
            assert state.physics.metadata == f"member {member}"
            arrived.wait(timeout=2)
            return state.physics.compute(state, f"current cfg {member}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        assert [future.result(timeout=2) for future in futures] == [0, 1]
    group.finish()
    assert group.owner.trace[0] == ("begin", ("current cfg 0", "current cfg 1"), (True, True))
    assert all(driver.state.physics is driver for driver in group.owner.drivers)
    assert group.receipt["entry_groups"] == 1 and not group._armed


def test_missing_physics_entry_wakes_peer_and_restores_original_owners():
    group = _group()
    group.arm()
    def callback(member):
        with group.member_scope(member):
            if member == 0:
                state = group.owner.drivers[member].state
                return state.physics.compute(state, "cfg")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        with pytest.raises(ValueError, match="without reaching its physics entry"):
            futures[0].result(timeout=2)
        assert futures[1].result(timeout=2) is None
    with pytest.raises(ValueError, match="without reaching its physics entry"):
        group.finish()
    assert all(driver.state.physics is driver for driver in group.owner.drivers)
    group.close()


def test_post_entry_failure_finishes_active_owner_before_release():
    group = _group()
    group.arm()
    entered = threading.Barrier(2)
    error = RuntimeError("original step failed after physics")
    def callback(member):
        with group.member_scope(member):
            state = group.owner.drivers[member].state
            state.physics.compute(state, "cfg")
            entered.wait(timeout=2)
            if member == 0:
                raise error
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        with pytest.raises(RuntimeError, match="after physics"):
            futures[0].result(timeout=2)
        futures[1].result(timeout=2)
    with pytest.raises(RuntimeError, match="after physics"):
        group.finish()
    assert not group.owner._active and not group._armed
    assert all(driver.state.physics is driver for driver in group.owner.drivers)
    group.close()


def test_premature_finish_cannot_disarm_callbacks_that_are_still_running():
    group = _group()
    group.arm()
    with pytest.raises(RuntimeError, match="precedes"):
        group.finish()
    assert group._armed


def test_step_wrapper_restores_driver_before_original_post_step_consumer():
    group = _group()
    group.arm()
    def numerical_step(state, cfg, **unused):
        return state.physics.compute(state, cfg)
    def callback(member):
        state = group.owner.drivers[member].state
        result = group.wrap_step(member, numerical_step)(state, "current")
        assert state.physics is group.owner.drivers[member]
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        assert [future.result(timeout=2) for future in futures] == [0, 1]
    group.finish()


def test_pre_entry_callback_failure_aborts_waiting_peer_and_can_close():
    group = _group()
    group.arm()
    error = RuntimeError("original pre-step validation failed")
    def callback(member):
        try:
            if member == 1:
                raise error
            with group.member_scope(member):
                state = group.owner.drivers[member].state
                return state.physics.compute(state, "cfg")
        except BaseException as caught:
            group.abort(caught)
            raise
        finally:
            group.callback_finished(member)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        for future in futures:
            with pytest.raises(RuntimeError, match="pre-step validation"):
                future.result(timeout=2)
    with pytest.raises(RuntimeError, match="pre-step validation"):
        group.finish()
    assert all(driver.state.physics is driver for driver in group.owner.drivers)
    assert not group.owner._active
    group.close()


def test_callback_completion_without_scope_or_entry_wakes_peer():
    group = _group()
    group.arm()
    def callback(member):
        try:
            if member == 0:
                with group.member_scope(member):
                    state = group.owner.drivers[member].state
                    return state.physics.compute(state, "cfg")
        finally:
            group.callback_finished(member)
            group.callback_finished(member)  # Idempotent with scope exit.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, member) for member in range(2)]
        with pytest.raises(ValueError, match="without reaching its physics entry"):
            futures[0].result(timeout=2)
        futures[1].result(timeout=2)
    with pytest.raises(ValueError, match="without reaching its physics entry"):
        group.finish()
    assert not group.owner._active
    group.close()
