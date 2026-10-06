"""Combined merged driver flags on actual device arrays and reference calls."""
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_ruc_lsm_fused import (
    _call, _case, _copy, _equal, _fractional_ice_and_monthly_lai,
    _operational_params)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("snow", ["wrf_461", "wrf_45"])
def test_combined_driver_flags_match_reference_after_each_of_two_calls(snow):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference

    # Sample ice, water, cold snow, warm land and lake columns from the
    # existing heterogeneous fixture. Its first seventeen columns alone
    # would all lie in the ice band and omit the land/lake driver arms.
    bench, driver, atmosphere, cold = _case(97 * 61, 9, "mixed")
    indices = np.asarray(
        [0, 3, 6, 7, 80, 83, 90, 485, 510, 539, 971, 1484,
         2910, 2911, 3040, 4000, 5916], dtype=np.int32)
    device_indices = cp.asarray(indices)

    def selected_columns(array):
        if isinstance(array, cp.ndarray) and array.shape[-2:] == (1, 97 * 61):
            return cp.ascontiguousarray(array[..., device_indices])
        return array.copy() if hasattr(array, "copy") else array

    driver.fields = {name: selected_columns(value) for name, value in driver.fields.items()}
    atmosphere = {name: selected_columns(value) for name, value in atmosphere.items()}
    cold = cold[..., indices]
    width = len(indices)
    params = _operational_params(driver.ruc_params, 9, rdlai2d=True, fractional_seaice=1)
    assert params.rdlai2d and params.xice_threshold == 0.02
    assert _fractional_ice_and_monthly_lai(driver.fields, width) > 0
    prescribed_lai = cp.asnumpy(driver.fields["lai"]).copy()
    # An invalid ground-vapour state activates the first-call air fallback.
    # Warmer lowest-level air activates the logarithmic temperature branch.
    driver.fields["qvg"][...] = cp.float32(0.0)
    driver.fields["qcg"][...] = cp.float32(0.03)
    atmosphere["temperature"][0][...] = driver.fields["tsk"] + cp.float32(4.0)
    arms = {name: _copy(driver.fields) for name in
            ("fused", "reference", "public_cold_start", "flux_diagnostic")}
    selectors = dict(ruc_soilprop="wrf_45", ruc_irrigation="wrf_461", ruc_snow=snow,
                     ruc_qvg_cold_start="air", ruc_2m_diagnostic="log_profile")
    functions = {
        "fused": partial(ruc_lsm_step, **selectors),
        "reference": partial(_ruc_lsm_step_reference, **selectors),
        "public_cold_start": partial(ruc_lsm_step, **{**selectors, "ruc_qvg_cold_start": "wrf"}),
        "flux_diagnostic": partial(ruc_lsm_step, **{**selectors, "ruc_2m_diagnostic": "flux"}),
    }

    def different(left, right, names):
        return any(bool(cp.any(left[name].view(cp.uint32) != right[name].view(cp.uint32)))
                   for name in names)

    for k in (1, 2):
        running = list(arms) if k == 1 else ["fused", "reference"]
        census = {}
        for name in running:
            fields = arms[name]
            bench.forcing(SimpleNamespace(fields=fields), k, 59, (1, width), cold)
            census[name] = _call(functions[name], fields, atmosphere, params, k)
        assert census["fused"] == census["reference"]
        assert census["fused"]["land"] > 0 and census["fused"]["sea_ice"] > 0
        # Existing whole-call helper requires exact bytes for every carried
        # device array, including T2/TH2/Q2 and the snow/soil output profiles.
        _equal(arms["fused"], arms["reference"])
        np.testing.assert_array_equal(cp.asnumpy(arms["fused"]["lai"]).view(np.uint32),
                                      prescribed_lai.view(np.uint32))
        if k == 1:
            assert different(arms["fused"], arms["public_cold_start"],
                             ("qvg", "qcg", "tsk", "hfx")), "air cold-start flag was not live"
            assert bool(cp.any(atmosphere["temperature"][0] > arms["fused"]["tsk"]))
            assert different(arms["fused"], arms["flux_diagnostic"],
                             ("t2", "th2", "q2")), "log-profile flag was not live"
