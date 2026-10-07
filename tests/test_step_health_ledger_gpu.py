"""Scheme health reads are deferred to the end of each domain step.

The forecast executor routes the scheme status words (microphysics, YSU, KF,
RRTMGP, ...) through gpuwm.core.health_ledger while a domain steps and drains
them once when the step returns.  A healthy step is unchanged; a flagged step
still raises the scheme's own refusal from the same step, before history or a
checkpoint can see it.  --health-debug keeps the immediate reads.
"""
import pytest
from conftest import requires_gpu

from test_model_stability_gpu import _model

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("adaptive", [False, True])
def test_flagged_scheme_refuses_in_its_own_step_before_output(adaptive):
    import cupy as cp
    from gpuwm.core import dycore, health_ledger
    from gpuwm.core.model import execute_experiment

    exp, model = _model(adaptive)
    count = 0
    deferred = []
    history, checkpoints = [], []

    def describe(flags):
        raise FloatingPointError(f"test scheme flagged {flags:#x}")

    def flagged_step(state, cfg, **kwargs):
        nonlocal count
        dycore.step(state, cfg, **kwargs)
        count += 1
        deferred.append(health_ledger.active() is not None)
        status = cp.asarray([3 if count == 3 else 0], dtype=cp.uint32)
        # Deferred: returns 0 now, refuses at the step's drain.
        assert health_ledger.read_status(
            status, site="test-scheme", describe=describe) == 0

    with pytest.raises(FloatingPointError, match="test scheme flagged 0x3"):
        execute_experiment(
            model, steppers={1: flagged_step}, experiment=exp,
            history_handler=lambda tree, node, ticks: history.append(
                node.clock.elapsed_seconds),
            restart_handler=lambda tree, ticks: checkpoints.append(
                tree.root.clock.elapsed_seconds),
            pool_trim_per_period=False)
    assert count == 3
    assert all(deferred) and len(deferred) == 3
    # Nothing at or after the flagged step's time reached history/checkpoint.
    assert all(seconds < 3.0 for seconds in history + checkpoints)
    assert health_ledger.active() is None


def test_health_debug_keeps_immediate_reads():
    from gpuwm.core import dycore, health_ledger
    from gpuwm.core.model import execute_experiment

    exp, model = _model(False)
    seen = []

    def step(state, cfg, **kwargs):
        dycore.step(state, cfg, **kwargs)
        seen.append(health_ledger.active())

    execute_experiment(model, steppers={1: step}, experiment=exp,
                       health_debug=True, pool_trim_per_period=False)
    assert seen and all(ledger is None for ledger in seen)
