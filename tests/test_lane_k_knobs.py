"""Non-vacuous value-pair gates for Lane K's newly exposed knobs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _shortwave_column():
    k = 4
    return dict(
        p3d=np.linspace(90000.0, 30000.0, k, dtype=np.float32),
        p8w=np.linspace(100000.0, 20000.0, k + 1, dtype=np.float32),
        t3d=np.array([280.0, 270.0, 260.0, 250.0], np.float32),
        t8w=np.full(k + 1, 260.0, np.float32),
        dz8w=np.full(k, 500.0, np.float32),
        qv3d=np.full(k, 1.0e-3, np.float32),
        qc3d=np.full(k, 1.0e-4, np.float32),
        qr3d=np.zeros(k, np.float32),
        qi3d=np.full(k, 1.0e-5, np.float32),
        qs3d=np.full(k, 1.0e-5, np.float32),
        qg3d=np.zeros(k, np.float32),
        cldfra3d=np.full(k, 0.5, np.float32),
        o33d=np.full(k, 1.0e-4, np.float32),
        re_cloud=np.full(k, 15.0e-6, np.float32),
        re_ice=np.full(k, 30.0e-6, np.float32),
        re_snow=np.full(k, 50.0e-6, np.float32),
        tsk=280.0, albedo=0.2, xland=1.0, xice=0.0, snow=0.0,
        xlat=30.0, xcoszen=0.5, solcon=1361.0,
        icloud=1, warm_rain=False, cldovrlp=2, idcor=0,
        yr=2000, julian=100.0, mp_physics=8,
    )


def test_o3input_values_select_distinguishable_legacy_wrapper_profiles():
    from gpuwm.core import rrtmg_legacy_prep as prep

    kwargs = _shortwave_column()
    calculated = prep.swrad_prep(
        **kwargs, o3input=0, has_reqc=0, has_reqi=0, has_reqs=0)
    supplied = prep.swrad_prep(
        **kwargs, o3input=2, has_reqc=0, has_reqi=0, has_reqs=0)

    assert not np.array_equal(calculated["o3vmr"], supplied["o3vmr"])
    np.testing.assert_array_equal(
        supplied["o3vmr"][:4], kwargs["o33d"])


def test_use_mp_re_values_select_calculated_or_scheme_radii():
    from gpuwm.core import rrtmg_legacy_prep as prep
    from gpuwm.core.rrtmg_legacy import legacy_scheme_declares_radii

    kwargs = _shortwave_column()
    outputs = {}
    for value in (0, 1):
        has_req = int(legacy_scheme_declares_radii(8, value))
        outputs[value] = prep.swrad_prep(
            **kwargs, o3input=2,
            has_reqc=has_req, has_reqi=has_req, has_reqs=has_req)

    assert legacy_scheme_declares_radii(8, 0) is False
    assert legacy_scheme_declares_radii(8, 1) is True
    assert not np.array_equal(
        outputs[0]["relqmcl"], outputs[1]["relqmcl"])
    assert not np.array_equal(
        outputs[0]["reicmcl"], outputs[1]["reicmcl"])


def test_seaice_albedo_default_values_change_only_ruc_ice_columns():
    from gpuwm.core.ruc_runtime import (
        RucRuntimeParameters, _ruc_seaice_albedo_override)

    albbck = np.array([0.2, 0.3, 0.4], np.float32)
    xice = np.array([0.0, 0.5, 1.0], np.float32)
    low, low_mask = _ruc_seaice_albedo_override(
        albbck, xice, 0.55, arrays=np)
    high, high_mask = _ruc_seaice_albedo_override(
        albbck, xice, 0.75, arrays=np)

    np.testing.assert_array_equal(low_mask, high_mask)
    np.testing.assert_array_equal(low[~low_mask], high[~high_mask])
    assert np.all(low[low_mask] == np.float32(0.55))
    assert np.all(high[high_mask] == np.float32(0.75))
    assert not np.array_equal(low, high)
    assert RucRuntimeParameters(
        seaice_albedo_default=0.55
    ).restart_identity()["seaice_albedo_default"] == 0.55


def test_rdmaxalb_values_choose_geogrid_or_vegparm_snow_albedo():
    from gpuwm.core.noah import (
        VEG_COLS, NoahParams, noah_initial_snow_albedo)

    veg = np.zeros((2, len(VEG_COLS)), np.float64)
    veg[:, VEG_COLS.index("maxalb")] = [80.0, 70.0]
    params = NoahParams(
        veg=veg, soil=np.zeros((1, 10)), gen=np.zeros(16),
        lucats=2, slcats=1, bare=1, natural=1,
        lutype="test", sltype="test")
    supplied = np.array([[10.0, 20.0]], np.float64)
    categories = np.array([[1, 2]], np.int32)

    read = noah_initial_snow_albedo(
        supplied, categories, params, rdmaxalb=True)
    table = noah_initial_snow_albedo(
        supplied, categories, params, rdmaxalb=False)

    np.testing.assert_array_equal(read, np.array([[0.1, 0.2]], np.float32))
    np.testing.assert_array_equal(table, np.array([[0.8, 0.7]], np.float32))
    assert not np.array_equal(read, table)


def test_legacy_adapter_o3input_identity_must_match_run_config():
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation

    adapter = object.__new__(RRTMGLegacyRadiation)
    adapter.o3input = 0
    cfg = SimpleNamespace(
        icloud=1, cldovrlp=2, idcor=0, ghg_input=0, aer_opt=0,
        swint_opt=0, o3input=0)
    adapter._check_pins(cfg)
    cfg.o3input = 2
    try:
        adapter._check_pins(cfg)
    except ValueError as exc:
        assert "constructed for o3input=0" in str(exc)
    else:
        raise AssertionError("mismatched ozone routing identity was accepted")
