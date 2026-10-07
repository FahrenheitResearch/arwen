"""Chemical arrays join the selected fork dynamics and zero-out operators.

Registry index 1 is dummy and PARAM_FIRST_SCALAR is 2. Active qv has
P_QV=2. The WRF operator compares that numeric index separately in its
chem and tracer calls, so each first real member takes the qv-index rule.
The tests assert operator behavior, not HRRR-Smoke forecast parity.
"""
from dataclasses import replace

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("mode,every_array,boundary", [
    (0, 1, "specified"), (1, 0, "specified"),
    (1, 1, "specified"), (2, 1, "specified"),
    (2, 1, "nested"), (2, 1, "periodic"),
])
def test_zero_out_uses_separate_chemical_and_tracer_registry_indices(mode, every_array, boundary):
    import cupy as cp
    from gpuwm.core.microphysics import microphysics_zero_out
    from test_chem_transport import _bubble_case

    cfg, state = _bubble_case(
        chem_sets="cams_aq,smoke,tracer_test", chem_sources="cams-global,rave-3km",
        mp_zero_out=mode, mp_zero_out_all=every_array,
        mp_zero_out_thresh=1.0e-12, spec_zone=2,
        specified=boundary == "specified", nested=boundary == "nested")
    shape = state.p.shape
    donor = np.full(shape, np.float32(5.0e-13))
    donor[:, 0, :] = np.float32(-2.0e-13)
    donor[:, -1, :] = np.float32(-2.0e-13)
    donor[:, :, 0] = np.float32(-2.0e-13)
    donor[:, :, -1] = np.float32(-2.0e-13)
    donor[:, 3, 3] = np.float32(-3.0e-13)
    donor[:, 4, 4] = np.float32(2.0e-12)
    for row in state.chem.transported:
        getattr(state, row.state_attr)[...] = cp.asarray(donor)
    state.chemdiag_ledger_emitted.fill(17.0)
    before_weather = {name: getattr(state, name).get()
                      for name in ("u", "v", "w", "thp", "php", "mup")}
    microphysics_zero_out(state, cfg)
    sz = 2 if boundary != "periodic" else 0
    # The table order makes o3 first in chem and smoke first in tracer.
    first_members = {"o3", "smoke"}
    for row in state.chem.transported:
        expected = donor.copy()
        if mode and every_array:
            expected[:, 0, :] = 0
            expected[:, -1, :] = 0
            expected[:, :, 0] = 0
            expected[:, :, -1] = 0
            interior = expected[:, sz:shape[1]-sz, sz:shape[2]-sz]
            if row.name in first_members:
                if mode == 2:
                    interior[interior < 0] = 0
            else:
                interior[interior < np.float32(1.0e-12)] = 0
        np.testing.assert_array_equal(getattr(state, row.state_attr).get(), expected,
                                      err_msg=row.name)
    for name, expected in before_weather.items():
        np.testing.assert_array_equal(getattr(state, name).get().view(np.uint8),
                                      expected.view(np.uint8), err_msg=name)
    np.testing.assert_array_equal(state.chemdiag_ledger_emitted.get(),
                                  np.full(7, 17.0, np.float64))


def test_zero_out_classifies_from_declared_array_and_not_family():
    import cupy as cp
    from gpuwm.core.microphysics import microphysics_zero_out
    from test_chem_transport import _bubble_case

    cfg, state = _bubble_case(chem_sets="tracer_test", mp_zero_out=1,
                              mp_zero_out_all=1, mp_zero_out_thresh=1.0e-12)
    state.chem.transported = tuple(replace(row, wrf_array="chem" if index == 0 else "tracer")
                                   for index, row in enumerate(state.chem.transported))
    for row in state.chem.transported:
        getattr(state, row.state_attr).fill(cp.float32(5.0e-13))
    microphysics_zero_out(state, cfg)
    # Both rows become the first real member of their declared array.
    # Their unchanged tracer families cannot merge the two groups.
    for row in state.chem.transported:
        np.testing.assert_array_equal(getattr(state, row.state_attr).get(),
                                      np.full(state.p.shape, 5.0e-13, np.float32))


@pytest.mark.parametrize("mixing", [
    {"diff_6th_form": "noaa_wrf39", "diff_6th_opt": 2,
     "diff_6th_factor": 0.12, "diff_6th_factor2": 0.04},
    {"diff_6th_form": "noaa_wrf39", "diff_6th_opt": 2,
     "diff_6th_factor": 0.12, "diff_6th_factor2": 0.04, "km_opt": 4},
])
def test_combined_fork_filter_uses_each_declared_array_factor_on_actual_gpu_fields(mixing):
    import cupy as cp
    from gpuwm.core.dycore import _save_time_t, prepare_fixed_tendencies
    from test_chem_transport import _bubble_case

    cfg, state = _bubble_case(chem_sets="cams_aq,tracer_test", **mixing)
    rng = np.random.default_rng(7)
    field = rng.uniform(0.002, 0.018, state.p.shape).astype(np.float32)
    state.qv[...] = cp.asarray(field)
    state.chem_passive_1[...] = state.qv
    state.chem_o3[...] = state.qv
    _save_time_t(state)
    prepare_fixed_tendencies(state, cfg)
    shape = state.p.shape
    moist = state.scratch(shape, "smag_rqv").get()
    chemical = state.scratch(shape, "smag_rchem_o3").get()
    tracer = state.scratch(shape, "smag_rchem_passive_1").get()
    assert np.any(moist != 0)
    np.testing.assert_array_equal(chemical.view(np.uint32), moist.view(np.uint32))
    assert not np.array_equal(tracer.view(np.uint32), moist.view(np.uint32))
    # Same input and actual operator, now giving moisture the independently
    # pinned tracer factor1. This is an exact control, not a ratio tolerance.
    control = replace(cfg, diff_6th_factor2=0.12)
    prepare_fixed_tendencies(state, control)
    np.testing.assert_array_equal(state.scratch(shape, "smag_rqv").get().view(np.uint32),
                                  tracer.view(np.uint32))


def test_fork_chem_mix6_off_skips_both_declared_chemical_arrays():
    import cupy as cp
    from gpuwm.core.dycore import _save_time_t, prepare_fixed_tendencies
    from test_chem_transport import _bubble_case

    cfg, state = _bubble_case(chem_sets="cams_aq,tracer_test",
                              diff_6th_form="noaa_wrf39", diff_6th_opt=2,
                              diff_6th_factor2=0.04, chem_mix6_off=True)
    rng = np.random.default_rng(8)
    state.qv[...] = cp.asarray(rng.uniform(0.002, 0.018, state.p.shape).astype(np.float32))
    for row in state.chem.transported:
        getattr(state, row.state_attr)[...] = state.qv
    _save_time_t(state)
    prepare_fixed_tendencies(state, cfg)
    assert np.any(state.scratch(state.p.shape, "smag_rqv").get() != 0)
    for row in state.chem.transported:
        assert not np.any(state.scratch(state.p.shape, "smag_r" + row.state_attr).get())
