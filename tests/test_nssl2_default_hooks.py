"""Contract gates for the concrete NSSL default condensation hook."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import nssl2_default_hooks as default_hooks
from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
from gpuwm.core.nssl2_runtime import NSSL2RuntimeFields


_SHAPE = (3, 1, 2)


class _State:
    def __init__(self):
        self.p = np.full(_SHAPE, 85000.0, dtype=np.float32)
        self.w = np.zeros((_SHAPE[0] + 1, *_SHAPE[1:]), dtype=np.float32)
        self._scratch = {}

    def scratch(self, shape, name):
        value = self._scratch.get(name)
        if value is None:
            value = np.empty(shape, dtype=np.float32)
            self._scratch[name] = value
        return value


def _workspace(shape=_SHAPE):
    return NSSL2DriverWorkspace(
        state=np.zeros((16, *shape), dtype=np.float32),
        category_surface_export=np.zeros((5, *shape[1:]), dtype=np.float32),
        shape=shape,
    )


def _fields(state):
    return NSSL2RuntimeFields(
        theta=np.full(_SHAPE, 300.0, dtype=np.float32),
        rho=np.full(_SHAPE, 1.1, dtype=np.float32),
        pressure=state.p,
        pii=np.full(_SHAPE, 0.95, dtype=np.float32),
        dz=np.full(_SHAPE, 125.0, dtype=np.float32),
        w=state.w,
    )


def test_default_hook_binds_direct_workspace_nucond_once(monkeypatch):
    state = _State()
    workspace = _workspace()
    fields = _fields(state)
    captured = []

    def fused(actual_workspace, actual_fields):
        return None

    def launch(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(default_hooks, "launch_nucond", launch)
    hooks = default_hooks.make_nssl2_default_runtime_hooks(
        state, 2.5, fused, validate_values=False)

    assert hooks.fused_gs is fused
    assert hooks.nucond is None
    assert hooks.qv_excess is None
    assert hooks.nucond_qvexcess is not None
    hooks.nucond_qvexcess(workspace, fields)

    assert len(captured) == 1
    args, kwargs = captured[0]
    expected_prefix = (
        fields.theta, fields.rho, state.p, fields.pii, state.w)
    for actual, expected in zip(args[:5], expected_prefix, strict=True):
        assert actual is expected
    for offset, name in enumerate(
            ("qv", "qc", "qr", "qi", "qs", "qndrop", "qnr", "qni",
             "qns", "qnn"),
            start=5):
        expected = workspace.field(name)
        assert np.shares_memory(args[offset], expected)
        assert args[offset].ctypes.data == expected.ctypes.data
    assert args[15] == 2.5
    assert kwargs == {
        "supersaturation_scratch":
            state._scratch[default_hooks.NSSL2_NUCOND_SCRATCH],
        "concentration_space": True,
        "validate_values": False,
    }


def test_missing_or_invalid_fused_hook_fails_before_scratch_allocation():
    state = _State()
    with pytest.raises(
            default_hooks.NSSL2ProductionConfigurationError,
            match="fused_gs"):
        default_hooks.make_nssl2_default_runtime_hooks(state, 1.0, None)
    with pytest.raises(TypeError, match="fused_gs"):
        default_hooks.make_nssl2_default_runtime_hooks(state, 1.0, 7)
    assert state._scratch == {}


@pytest.mark.parametrize("step", [0.0, -1.0, np.nan, np.inf])
def test_invalid_step_fails_before_scratch_allocation(step):
    state = _State()
    with pytest.raises(ValueError, match="positive finite"):
        default_hooks.make_nssl2_default_runtime_hooks(
            state, step, lambda workspace, fields: None)
    assert state._scratch == {}


def test_cross_state_runtime_fields_fail_before_launch(monkeypatch):
    state = _State()
    other = _State()
    launches = []
    monkeypatch.setattr(
        default_hooks, "launch_nucond",
        lambda *args, **kwargs: launches.append((args, kwargs)))
    hooks = default_hooks.make_nssl2_default_runtime_hooks(
        state, 1.0, lambda workspace, fields: None)
    with pytest.raises(
            default_hooks.NSSL2ProductionConfigurationError,
            match="different DomainState"):
        hooks.nucond_qvexcess(_workspace(), _fields(other))
    assert launches == []


@pytest.mark.parametrize("failure", ["shape", "dtype", "contiguous"])
def test_scratch_contract_is_fail_closed(failure):
    state = _State()
    if failure == "shape":
        value = np.empty((2, 1, 2), dtype=np.float32)
        match = "shape"
    elif failure == "dtype":
        value = np.empty(_SHAPE, dtype=np.float64)
        match = "float32"
    else:
        value = np.empty((3, 1, 4), dtype=np.float32)[:, :, ::2]
        match = "C-contiguous"
    state._scratch[default_hooks.NSSL2_NUCOND_SCRATCH] = value
    with pytest.raises((TypeError, ValueError), match=match):
        default_hooks.make_nssl2_default_runtime_hooks(
            state, 1.0, lambda workspace, fields: None)


def test_source_has_one_combined_nucond_and_no_registry_round_trip():
    import inspect

    source = inspect.getsource(
        default_hooks.make_nssl2_default_runtime_hooks)
    assert source.count("launch_nucond(") == 1
    assert "concentration_space=True" in source
    assert "launch_qvexcess" not in source
    assert "scatter" not in source.lower()
    assert "gather" not in source.lower()


def test_invalid_state_structure_is_rejected():
    def fused(workspace, fields):
        return None

    with pytest.raises(
            default_hooks.NSSL2ProductionConfigurationError,
            match="state.p"):
        default_hooks.make_nssl2_default_runtime_hooks(
            SimpleNamespace(p=np.ones((2,), dtype=np.float32)), 1.0, fused)

    with pytest.raises(
            default_hooks.NSSL2ProductionConfigurationError,
            match="DomainState.scratch"):
        default_hooks.make_nssl2_default_runtime_hooks(
            SimpleNamespace(
                p=np.ones(_SHAPE, dtype=np.float32),
                w=np.zeros((_SHAPE[0] + 1, *_SHAPE[1:]), dtype=np.float32)),
            1.0,
            fused,
        )
