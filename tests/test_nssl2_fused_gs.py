"""Contract gates for the independent fused NSSL GS entry point."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import nssl2_fused_gs as fused


def _workspace(shape=(4, 2, 3)):
    state = np.zeros((16, *shape), dtype=np.float32)
    export = np.zeros((5, *shape[1:]), dtype=np.float32)
    return fused.NSSL2DriverWorkspace(state, export, shape)


def _environment(shape=(4, 2, 3)):
    cells = [np.ones(shape, dtype=np.float32) for _ in range(7)]
    w = np.ones((shape[0] + 1, *shape[1:]), dtype=np.float32)
    # theta, rho, pressure, exner, interface w, t0 scratch, t7 scratch, dz
    return (*cells[:4], w, *cells[4:])


def test_launcher_passes_one_workspace_and_exact_environment(monkeypatch):
    workspace = _workspace()
    environment = _environment()
    calls = []

    class Kernel:
        def __call__(self, grid, block, args):
            calls.append((grid, block, args))

    def get_kernel(module, symbol):
        assert module == "nssl2_fused_gs"
        assert symbol in ("nssl2_prepare_fused_gs", "nssl2_fused_gs")
        return Kernel()

    monkeypatch.setattr(fused, "get_kernel", get_kernel)
    fused.launch_fused_gs(workspace, *environment, 12.5)

    assert len(calls) == 2
    grid, block, args = calls[0]
    assert grid == (1,)
    assert block == (128,)
    theta, rho, pressure, exner, w, temperature, target, dz = environment
    expected_prepass = (
        temperature, target, workspace.state, theta, rho, pressure, exner
    )
    assert all(
        actual is expected
        for actual, expected in zip(args[:7], expected_prepass)
    )
    assert args[7] == np.int32(24)

    grid, block, args = calls[1]
    assert grid == (1,)
    assert block == (128,)
    expected_fused = (
        workspace.state, theta, rho, pressure, exner,
        temperature, w, target, dz,
    )
    assert all(
        actual is expected
        for actual, expected in zip(args[:9], expected_fused)
    )
    assert args[9] == np.float32(12.5)
    # nz, ncol, n, then the hail switch: the default lane runs with
    # WRF's hail category present (nssl_hail_on resolving to 1).
    assert args[10:] == (
        np.int32(4), np.int32(6), np.int32(24), np.int32(1))


def test_callback_adapter_preserves_the_narrow_hook(monkeypatch):
    workspace = _workspace()
    environment = _environment()
    calls = []
    monkeypatch.setattr(
        fused,
        "launch_fused_gs",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    theta, rho, pressure, exner, w, temperature, target, dz = environment
    fields = SimpleNamespace(
        theta=theta, rho=rho, pressure=pressure, pii=exner, w=w, dz=dz
    )
    callback = fused.NSSL2FusedGS(temperature, target, 30.0)
    callback(workspace, fields)
    assert calls == [((
        workspace, theta, rho, pressure, exner, w,
        temperature, target, dz, 30.0,
    ), {"hail_on": True})]

    # The variant's hail switch is carried by the adapter, not smuggled in
    # from module state: an adapter built hail-off must forward hail-off.
    calls.clear()
    fused.NSSL2FusedGS(temperature, target, 30.0, hail_on=False)(
        workspace, fields)
    assert calls[0][1] == {"hail_on": False}


@pytest.mark.parametrize("dt_s", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_launcher_rejects_invalid_step_before_compilation(dt_s):
    with pytest.raises(ValueError, match="positive finite"):
        fused.launch_fused_gs(_workspace(), *_environment(), dt_s)


def test_launcher_rejects_non_scalar_step_before_compilation():
    with pytest.raises(TypeError, match="positive finite"):
        fused.launch_fused_gs(_workspace(), *_environment(), object())


def test_launcher_validates_workspace_and_environment_before_compilation():
    environment = list(_environment())
    with pytest.raises(TypeError, match="NSSL2DriverWorkspace"):
        fused.launch_fused_gs(object(), *environment, 1.0)

    workspace = _workspace()
    bad_state = fused.NSSL2DriverWorkspace(
        np.zeros((15, *workspace.shape), dtype=np.float32),
        workspace.category_surface_export,
        workspace.shape,
    )
    with pytest.raises(ValueError, match="workspace state"):
        fused.launch_fused_gs(bad_state, *environment, 1.0)

    bad_dtype = list(environment)
    bad_dtype[2] = bad_dtype[2].astype(np.float64)
    with pytest.raises(TypeError, match="pressure_pa must be float32"):
        fused.launch_fused_gs(workspace, *bad_dtype, 1.0)

    bad_shape = list(environment)
    bad_shape[5] = np.zeros((24,), dtype=np.float32)
    with pytest.raises(ValueError, match="temperature_k must have shape"):
        fused.launch_fused_gs(workspace, *bad_shape, 1.0)

    bad_velocity = list(environment)
    bad_velocity[4] = np.zeros((5, 2, 6), dtype=np.float32)[:, :, ::2]
    assert not bad_velocity[4].flags.c_contiguous
    with pytest.raises(ValueError, match="vertical_velocity must be C-contiguous"):
        fused.launch_fused_gs(workspace, *bad_velocity, 1.0)

    cell_velocity = list(environment)
    cell_velocity[4] = np.zeros(workspace.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="interface field"):
        fused.launch_fused_gs(workspace, *cell_velocity, 1.0)

    noncontiguous = list(environment)
    noncontiguous[7] = np.zeros((4, 2, 6), dtype=np.float32)[:, :, ::2]
    assert not noncontiguous[7].flags.c_contiguous
    with pytest.raises(ValueError, match="dz must be C-contiguous"):
        fused.launch_fused_gs(workspace, *noncontiguous, 1.0)


def test_fused_launcher_does_not_import_or_call_isolated_process_surface():
    source = inspect.getsource(fused)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "gpuwm.core.nssl2" not in imported_modules


def test_cuda_reproduces_wrf_two_stage_vertical_velocity_centering():
    source = (
        Path(__file__).parents[1]
        / "gpuwm" / "core" / "kernels" / "nssl2_fused_gs.cu"
    ).read_text(encoding="utf-8")
    assert "const int velocity_kp = min(k + 1, nz - 1);" in source
    assert "const float w_mass = __fmul_rn" in source
    assert "const float w_mass_kp = __fmul_rn" in source
    assert "__fadd_rn(w_mass_kp, w_mass)" in source
