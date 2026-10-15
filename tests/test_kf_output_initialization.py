"""KF must initialize its outputs even on columns that return early."""
from __future__ import annotations

import numpy as np
import pytest

#: The arrays the column kernel itself writes (launch_kf derives the
#: integer views and the ``rainc`` alias from them afterwards).
_KERNEL_OUTPUTS = (
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten", "rqscuten",
    "updraft_mass_flux", "downdraft_mass_flux", "cldfra_dp", "cldfra_sh",
    "qc_kf", "qi_kf", "raincv", "pratec", "nca_seconds", "cutop", "cubot",
    "shall", "cape_before", "cape_after", "timec")


@pytest.mark.gpu
@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("poison", [0x7FC12345, 0xDEADBEEF])
def test_launcher_overwrites_output_residue(monkeypatch, phase, poison):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kf
    from test_kf_workspace import (_INPUTS, _bits, _differing, _fresh,
                                   _kf_batch, _launch)

    host = _kf_batch(8 * kf._TPB + 8)
    nz, ny, nx = host["temperature"].shape
    dev = {name: cp.asarray(value) for name, value in host.items()}
    expected = _fresh(nz, ny, nx)
    ws = cp.empty(kf.kf_workspace_floats(nz, ny * nx), dtype=cp.float32)
    _launch(kf._kf_function("kf_column"), dev, expected,
            nz, ny, nx, ws, 0, ny * nx, phase)
    expected_bits = _bits(expected)
    shall = expected_bits["shall"].view(np.float32)
    assert (shall < 1.5).any() and (shall > 1.5).any()

    original_empty = cp.empty

    def poisoned_empty(*args, **kwargs):
        array = original_empty(*args, **kwargs)
        array.view(cp.uint32).fill(np.uint32(poison))
        return array

    # A partial final tile also checks reuse after a different sounding.
    monkeypatch.setattr(kf, "kf_tile_columns", lambda fn, ncol: 3 * kf._TPB)
    monkeypatch.setattr(cp, "empty", poisoned_empty)
    actual = kf.launch_kf(*(dev[name] for name in _INPUTS), dx=12000.0,
                          dt=60.0, cudt_minutes=5.0,
                          phase_mode=kf.KFPhaseMode(phase),
                          cloud_diagnostics=True)
    assert set(_KERNEL_OUTPUTS) <= set(actual)
    assert set(expected) == set(_KERNEL_OUTPUTS)
    got = _bits({name: actual[name] for name in _KERNEL_OUTPUTS})
    assert not _differing(expected_bits, got)


@pytest.mark.gpu
@pytest.mark.parametrize("phase", range(4))
def test_regrouping_matches_single_warp_blocks(monkeypatch, phase):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kf
    from test_kf_workspace import _INPUTS, _bits, _differing, _kf_batch

    host = _kf_batch(513)
    dev = [cp.asarray(host[name]) for name in _INPUTS]
    args = dict(dx=12000.0, dt=60.0, cudt_minutes=5.0,
                phase_mode=kf.KFPhaseMode(phase), cloud_diagnostics=True)
    grouped = _bits(kf.launch_kf(*dev, **args))
    monkeypatch.setattr(kf, "_TPB", kf._WS_LANES)
    monkeypatch.setattr(kf, "KF_TILE_BLOCKS_PER_SM", 8)
    single_warp = _bits(kf.launch_kf(*dev, **args))
    assert not _differing(grouped, single_warp)
