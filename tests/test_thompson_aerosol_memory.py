"""Dead diagnostic storage must not change aerosol forecasts or entry rewrites."""

import numpy as np
import pytest

from conftest import requires_gpu
from test_thompson_aerosol_adapter import _FIXTURES, _build_case, _tables_or_skip


DEAD_SLOTS = (
    "nwfa_entry_m3", "nifa_entry_m3", "rc_entry", "nc_entry_m3",
    "nu_c_entry", "l_qc_entry",
)


@requires_gpu
@pytest.mark.parametrize("scenario", _FIXTURES)
def test_three_calls_match_the_full_entry_output_path_bitwise(monkeypatch, scenario):
    # A repeated real call catches scratch reuse as well as first-step bits.
    import cupy as cp
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
    from gpuwm.core import thompson_aerosol_state as entry

    _tables_or_skip()
    compact, cfg, dt, *_ = _build_case(cp, scenario)
    full, _, _, *_ = _build_case(cp, scenario)
    snapshot, cloud = entry.launch_aerosol_entry_snapshot, entry.launch_aerosol_entry_cloud_number

    def legacy_snapshot(*args):
        args = list(args)
        args[-2:] = [full.scratch(full.p.shape, "old_" + name)
                     for name in DEAD_SLOTS[:2]]
        return snapshot(*args)

    def legacy_cloud(*args):
        args = list(args)
        args[-4:] = [full.scratch(full.p.shape, "old_" + name,
                                 dtype=np.int32 if name in DEAD_SLOTS[-2:] else np.float32)
                     for name in DEAD_SLOTS[2:]]
        return cloud(*args)

    for step in range(3):
        _apply_thompson_aerosol(compact, cfg, dt, refl_10cm_due=(step == 0))
        with monkeypatch.context() as patch:
            patch.setattr(entry, "launch_aerosol_entry_snapshot", legacy_snapshot)
            patch.setattr(entry, "launch_aerosol_entry_cloud_number", legacy_cloud)
            _apply_thompson_aerosol(full, cfg, dt, refl_10cm_due=(step == 0))
        for name, value in vars(compact).items():
            if isinstance(value, cp.ndarray):
                np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),
                                              cp.asnumpy(getattr(full, name)).view(np.uint32),
                                              err_msg=f"{scenario}, step {step}, {name}")
        for name, value in compact._scratch.items():
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),
                                          cp.asnumpy(full._scratch[name]).view(np.uint32),
                                          err_msg=f"{scenario}, step {step}, {name}")
    removed = sum(full._scratch["old_" + name].nbytes for name in DEAD_SLOTS)
    assert removed == 24 * compact.p.size
    assert not any("mp_thompson_aero_" + name in compact._scratch for name in DEAD_SLOTS)


@requires_gpu
@pytest.mark.parametrize("mask", range(16))
def test_optional_cloud_outputs_keep_selected_diagnostics_and_state(mask):
    import cupy as cp
    from gpuwm.core.thompson_aerosol_state import launch_aerosol_entry_cloud_number

    qc = cp.asarray([-1e-4, 0, 1e-12, 2e-12, 2e-4, np.nan], dtype=cp.float32)
    nc = cp.asarray([2, 0, 1e8, 1e9, 1e7, 1e8], dtype=cp.float32)
    rho = cp.asarray([0.3, 0.5, 0.8, 1.1, 1.2, 0.7], dtype=cp.float32)
    full_qc, full_nc = qc.copy(), nc.copy()
    outputs = [cp.empty(qc.shape, dtype=cp.float32 if i < 2 else cp.int32) for i in range(4)]
    selected = [cp.empty_like(array) if mask & (1 << i) else None
                for i, array in enumerate(outputs)]
    launch_aerosol_entry_cloud_number(full_qc, full_nc, rho, *outputs)
    launch_aerosol_entry_cloud_number(qc, nc, rho, *selected)
    for left, right in ((qc, full_qc), (nc, full_nc),
                        *((array, outputs[i]) for i, array in enumerate(selected) if array is not None)):
        np.testing.assert_array_equal(cp.asnumpy(left).view(np.uint32),
                                      cp.asnumpy(right).view(np.uint32))
