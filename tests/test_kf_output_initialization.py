"""KF must initialize its outputs even on columns that return early."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("poison", [0x7FC12345, 0xDEADBEEF])
def test_launcher_overwrites_output_residue(monkeypatch, phase, poison):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kf
    from gpuwm.core.kernels import get_kernel
    from test_kf_workspace import _bits, _differing, _fresh, _kf_batch, _launch

    host = _kf_batch(8 * kf._TPB + 8)
    nz, ny, nx = host["temperature"].shape
    dev = {name: cp.asarray(value) for name, value in host.items()}
    expected = _fresh(nz, ny, nx)
    expected["cloud_base"].fill(-1)
    expected["cloud_top"].fill(-1)
    ws = cp.empty(kf.kf_workspace_floats(nz, ny * nx), dtype=cp.float32)
    _launch(get_kernel("kf", "kf_column"), dev, expected,
            nz, ny, nx, ws, 0, ny * nx, phase)
    expected_bits = _bits(expected)
    triggered = cp.asnumpy(expected["triggered"])
    assert (triggered != 0).any() and (triggered == 0).any()

    original_empty = cp.empty

    def poisoned_empty(*args, **kwargs):
        array = original_empty(*args, **kwargs)
        array.view(cp.uint32).fill(np.uint32(poison))
        return array

    # A partial final tile also checks reuse after a different sounding.
    monkeypatch.setattr(kf, "kf_tile_columns", lambda fn, ncol: 3 * kf._TPB)
    monkeypatch.setattr(cp, "empty", poisoned_empty)
    actual = kf.launch_kf(**dev, dx=12000.0, dt=60.0, cudt=300.0,
                          phase_mode=kf.KFPhaseMode(phase))
    assert set(actual) == set(expected)
    assert not _differing(expected_bits, _bits(actual))


@pytest.mark.gpu
@pytest.mark.parametrize("phase", range(4))
def test_regrouping_matches_single_warp_blocks(monkeypatch, phase):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kf
    from test_kf_workspace import _bits, _differing, _kf_batch

    dev = {name: cp.asarray(value) for name, value in _kf_batch(513).items()}
    args = dict(dx=12000.0, dt=60.0, cudt=300.0,
                phase_mode=kf.KFPhaseMode(phase))
    grouped = _bits(kf.launch_kf(**dev, **args))
    monkeypatch.setattr(kf, "_TPB", kf._WS_LANES)
    monkeypatch.setattr(kf, "KF_TILE_BLOCKS_PER_SM", 8)
    single_warp = _bits(kf.launch_kf(**dev, **args))
    assert not _differing(grouped, single_warp)
