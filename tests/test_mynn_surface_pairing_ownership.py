"""WRF v4.6.1 ownership gates for MYNN with RUC and Noah-MP."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp


def _host(array):
    return np.ascontiguousarray(cp.asnumpy(array))


def _capture_mynn_pbl_inputs(driver, atmosphere, cfg, monkeypatch):
    """Run the real PBL while recording the fields at its call boundary."""
    import gpuwm.core.physics as physics_module

    captured = {}
    real = physics_module.mynn_pbl_step

    def capture(pbl_atmosphere, fields, **kwargs):
        for name in (
                "tsk", "ust", "hfx", "qfx", "qsfc", "ch",
                "t2", "q2", "th2"):
            captured[name] = _host(fields[name]).copy()
        return real(pbl_atmosphere, fields, **kwargs)

    monkeypatch.setattr(physics_module, "mynn_pbl_step", capture)
    driver._run_mynn_pbl(atmosphere, cfg)
    return captured


@requires_gpu
def test_noahmp_post_lsm_2m_categories_match_the_wrf_source_transcription():
    """Cover water, full ice, urban, partial ice and vegetated land."""
    from gpuwm.core.noahmp_runtime import (
        NoahmpRuntimeParameters,
        _noahmp_post_lsm_diagnostics,
    )
    from tools.mynn_surface_pairing_wrf461_oracle.transcribe_pairing import (
        noahmp_post_lsm_diagnostics,
    )

    params = NoahmpRuntimeParameters()
    identity = params.land_use
    shape = (2, 3)
    ivgtyp = np.array([
        [identity.iswater, identity.isice, identity.isurban],
        [identity.isice, 10, 10],
    ], np.int32)
    xice = np.array([[0.0, 1.0, 0.0], [0.25, 0.0, 0.0]], np.float32)
    fields = {
        "ivgtyp": cp.asarray(ivgtyp),
        "xice": cp.asarray(xice),
        "psfc": cp.asarray(np.array(
            [[100000.0, 99500.0, 99000.0],
             [98500.0, 98000.0, 97500.0]], np.float32)),
        "tsk": cp.asarray(np.array(
            [[296.0, 268.0, 303.0], [271.0, 302.0, 299.0]], np.float32)),
        "hfx": cp.asarray(np.array(
            [[45.0, -20.0, 80.0], [10.0, 120.0, -15.0]], np.float32)),
        "qfx": cp.asarray(np.array(
            [[1e-5, -2e-6, 3e-5], [4e-6, 2e-5, -1e-6]], np.float32)),
        "qsfc": cp.asarray(np.array(
            [[0.010, 0.002, 0.012], [0.003, 0.011, 0.008]], np.float32)),
        # Last column exercises each <1e-5 guard without changing category
        # coverage in the other five columns.
        "chs2": cp.asarray(np.array(
            [[0.010, 0.008, 0.012], [0.007, 0.015, 0.0]], np.float32)),
        "cqs2": cp.asarray(np.array(
            [[0.009, 0.006, 0.011], [0.005, 0.013, 0.0]], np.float32)),
        "fvegxy": cp.asarray(np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.65, 0.25]], np.float32)),
        "t2mvxy": cp.asarray(np.array(
            [[290.0, 265.0, 301.0], [269.0, 300.5, 298.5]], np.float32)),
        "t2mbxy": cp.asarray(np.array(
            [[291.0, 266.0, 302.0], [270.0, 301.5, 299.5]], np.float32)),
        "q2mvxy": cp.asarray(np.array(
            [[0.008, 0.001, 0.010], [0.002, 0.0095, 0.007]], np.float32)),
        "q2mbxy": cp.asarray(np.array(
            [[0.009, 0.0015, 0.011], [0.0025, 0.0105, 0.008]], np.float32)),
        "t2": cp.full(shape, np.float32(-1.0e36)),
        "th2": cp.full(shape, np.float32(-1.0e36)),
        "q2": cp.full(shape, np.float32(-1.0e36)),
    }
    expected = noahmp_post_lsm_diagnostics(
        ivgtyp=ivgtyp, xice=xice, iswater=identity.iswater,
        isice=identity.isice, isurban=identity.isurban, lcz=identity.lcz,
        **{name: _host(fields[name]) for name in (
            "psfc", "tsk", "hfx", "qfx", "qsfc", "chs2", "cqs2")},
        fveg=_host(fields["fvegxy"]), t2mv=_host(fields["t2mvxy"]),
        t2mb=_host(fields["t2mbxy"]), q2mv=_host(fields["q2mvxy"]),
        q2mb=_host(fields["q2mbxy"]),
    )
    _noahmp_post_lsm_diagnostics(fields, params=params)
    np.testing.assert_array_equal(_host(fields["t2"]), expected["t2"])
    np.testing.assert_array_equal(_host(fields["q2"]), expected["q2"])
    np.testing.assert_allclose(
        _host(fields["th2"]), expected["th2"], rtol=2.0e-7, atol=0.0)


@requires_gpu
@pytest.mark.parametrize("mp_physics", [6, 8])
def test_mynn_ruc_fractional_seaice_stages_immediate_and_wait_fields(
        mp_physics, monkeypatch):
    """Port MYNN_SEAICE_WRAPPER's two calls and its split ownership sets."""
    import gpuwm.core.physics as physics_module
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS
    from tools.mynn_surface_pairing_wrf461_oracle.transcribe_pairing import (
        MYNN_FRACTIONAL_IMMEDIATE,
        MYNN_FRACTIONAL_WAIT_FOR_LSM,
        get_local_ice_tsk,
        mynn_fractional_seaice_staging,
    )
    from test_ruc_runtime import _build

    state, cfg, driver = _build(
        nx=4, ny=2, nz=12, water_columns=0, ice_rows=1,
        mp_physics=mp_physics, sf_sfclay_physics=5, bl_pbl_physics=5)
    driver.fields["xice"][0, :] = cp.float32(0.65)
    atmosphere = _prepare_atmosphere(state)
    driver.fields["psfc"][...] = atmosphere["p_interface"][0]
    tsk_before = _host(driver.fields["tsk"]).copy()
    sst_before = _host(driver.fields["tsk_sea"]).copy()

    calls = []
    real = physics_module.launch_mynn_surface_layer

    def capture(inputs, mol, ustm, result, **kwargs):
        regime_before = _host(result.regime).copy()
        value = real(inputs, mol, ustm, result, **kwargs)
        calls.append({
            "outputs": {
                name: _host(getattr(result, name)).copy()
                for name in MYNN_SURFACE_OUTPUTS
            },
            "tsk": _host(inputs["tsk"]).copy(),
            "xland": _host(inputs["xland"]).copy(),
            "mavail": _host(inputs["mavail"]).copy(),
            "regime_before": regime_before,
        })
        return value

    monkeypatch.setattr(
        physics_module, "launch_mynn_surface_layer", capture)
    driver._run_sfclay(atmosphere, cfg)
    assert len(calls) == 2
    active = _host(driver.fields["xice"]) >= np.float32(0.5)
    temperatures = get_local_ice_tsk(
        xice=_host(driver.fields["xice"]), sst=sst_before, tsk=tsk_before,
        itimestep=1)
    np.testing.assert_array_equal(calls[0]["tsk"], temperatures["tsk_ice"])
    np.testing.assert_array_equal(calls[1]["tsk"], temperatures["tsk_sea"])
    np.testing.assert_array_equal(
        _host(driver.fields["tsk_sea"]), temperatures["tsk_sea"])
    np.testing.assert_array_equal(
        calls[1]["regime_before"], calls[0]["outputs"]["regime"])
    expected = mynn_fractional_seaice_staging(
        ice=calls[0]["outputs"], sea=calls[1]["outputs"],
        xice=_host(driver.fields["xice"]), active=active)
    for name in MYNN_FRACTIONAL_IMMEDIATE:
        np.testing.assert_array_equal(
            _host(driver.fields[name]), expected[name], err_msg=name)
    for name in MYNN_FRACTIONAL_WAIT_FOR_LSM:
        np.testing.assert_array_equal(
            _host(driver.fields[name]), expected[name], err_msg=name)
        np.testing.assert_array_equal(
            _host(driver.fields[f"{name}_sea"]),
            expected[f"{name}_sea"], err_msg=f"{name}_sea")
    np.testing.assert_array_equal(
        _host(driver.fields["regime"])[active],
        calls[1]["outputs"]["regime"][active])
    assert np.all(calls[1]["xland"][active] == np.float32(2.0))
    assert np.all(calls[1]["mavail"][active] == np.float32(1.0))
    assert np.all(calls[1]["tsk"][active] >= np.float32(271.4))


@requires_gpu
def test_mynn_ruc_fractional_wrapper_runs_both_calls_without_ice(monkeypatch):
    """FRACTIONAL_SEAICE selects the wrapper, not an any-ice shortcut."""
    import gpuwm.core.physics as physics_module
    from gpuwm.core.physics import _prepare_atmosphere
    from test_ruc_runtime import _build

    state, cfg, driver = _build(
        nx=3, ny=2, nz=12, water_columns=0, ice_rows=0,
        mp_physics=6, sf_sfclay_physics=5, bl_pbl_physics=5)
    atmosphere = _prepare_atmosphere(state)
    calls = 0
    real = physics_module.launch_mynn_surface_layer

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(physics_module, "launch_mynn_surface_layer", count)
    driver._run_sfclay(atmosphere, cfg)
    assert calls == 2


@requires_gpu
@pytest.mark.parametrize("mp_physics", [6, 8])
def test_mynn_ruc_writer_order_matches_wrf_for_two_microphysics_schemes(
        mp_physics, monkeypatch):
    """Byte-check every shared flux/coefficient after both real writers."""
    import gpuwm.core.ruc_runtime as runtime
    from gpuwm.core.physics import _prepare_atmosphere
    from tools.mynn_surface_pairing_wrf461_oracle.transcribe_pairing import (
        ruc_post_lsm_diagnostics,
        ruc_seam_ownership,
    )
    from test_ruc_runtime import _build

    state, cfg, driver = _build(
        nx=4, ny=2, nz=12, water_columns=0, mp_physics=mp_physics,
        sf_sfclay_physics=5, bl_pbl_physics=5)
    atmosphere = _prepare_atmosphere(state)
    driver.fields["psfc"][...] = atmosphere["p_interface"][0]
    driver._run_sfclay(atmosphere, cfg)
    after_mynn = {
        name: _host(driver.fields[name]).copy()
        for name in ("ust", "flhc", "flqc", "chs", "chs2", "cqs2", "cpm",
                     "tsk", "hfx", "qfx", "lh", "qsfc", "znt")
    }

    captured = {}
    real = runtime.ruc_land_surface_step

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(runtime, "ruc_land_surface_step", capture)
    driver._run_ruc(atmosphere, cfg, 1)
    result = captured["result"]
    lsm = {
        "tsk": _host(result.soilt),
        "hfx": _host(result.hfx),
        "qfx": _host(result.qfx),
        "lh": _host(result.lh),
        "qsfc": _host(result.qsfc),
        "znt": _host(result.znt),
    }
    expected = ruc_seam_ownership(
        mynn=after_mynn, lsm=lsm, rho=_host(atmosphere["rho"][0]),
        mavail=_host(driver.fields["mavail"]))
    for name in ("ust", "flhc", "flqc", "chs", "chs2", "cqs2", "cpm",
                 "tsk", "hfx", "qfx", "lh", "qsfc", "znt"):
        np.testing.assert_array_equal(
            _host(driver.fields[name]), expected[name], err_msg=name)

    diag = ruc_post_lsm_diagnostics(
        psfc=_host(driver.fields["psfc"]), tsk=expected["tsk"],
        hfx=expected["hfx"], qfx=expected["qfx"], qsfc=expected["qsfc"],
        chs2=expected["chs2"], cqs2=expected["cqs2"], cqs=expected["cqs"],
        t1=_host(atmosphere["temperature"][0]),
        qv1=_host(atmosphere["qv"][0]), rho1=_host(atmosphere["rho"][0]),
        p1=_host(atmosphere["pressure"][0]))
    for name in ("t2", "th2", "q2"):
        np.testing.assert_array_equal(
            _host(driver.fields[name]), diag[name], err_msg=name)
    pbl = _capture_mynn_pbl_inputs(
        driver, atmosphere, cfg, monkeypatch)
    for name in ("tsk", "ust", "hfx", "qfx", "qsfc"):
        np.testing.assert_array_equal(pbl[name], expected[name], err_msg=name)
    for name in ("t2", "q2", "th2"):
        np.testing.assert_array_equal(pbl[name], diag[name], err_msg=name)


@requires_gpu
@pytest.mark.parametrize("mp_physics", [6, 8])
def test_mynn_noahmp_writer_order_matches_wrf_for_two_microphysics_schemes(
        mp_physics, monkeypatch):
    """Noah-MP overwrites fluxes, never MYNN's exchange coefficients/UST."""
    import gpuwm.core.noahmp_runtime as runtime
    from gpuwm.core.physics import _prepare_atmosphere
    from tools.mynn_surface_pairing_wrf461_oracle.transcribe_pairing import (
        noahmp_flux_writeback,
        noahmp_post_lsm_diagnostics,
        noahmp_seam_ownership,
    )
    from test_noahmp_runtime import _build

    state, cfg, driver = _build(
        nx=4, ny=2, nz=12, water_columns=0, mp_physics=mp_physics,
        sf_sfclay_physics=5, bl_pbl_physics=5)
    atmosphere = _prepare_atmosphere(state)
    driver.fields["psfc"][...] = atmosphere["p_interface"][0]
    driver._run_sfclay(atmosphere, cfg)
    after_mynn = {
        name: _host(driver.fields[name]).copy()
        for name in ("ust", "chs", "chs2", "cqs2", "flhc", "flqc")
    }

    captured = {}
    real = runtime._write_back_slab

    def capture(fields, j, i, result, *, nsoil, dt):
        captured["j"] = _host(j)
        captured["i"] = _host(i)
        captured["result"] = {
            name: _host(result[name]).copy()
            for name in ("trad", "fsh", "ecan", "edir", "etran",
                         "fcev", "fgev", "fctr", "qsfc", "z0wrf")
        }
        return real(fields, j, i, result, nsoil=nsoil, dt=dt)

    monkeypatch.setattr(runtime, "_write_back_slab", capture)
    driver._run_noahmp(atmosphere, cfg, 1)

    shape = (cfg.ny, cfg.nx)
    raw = {}
    for name, values in captured["result"].items():
        slab = np.empty(shape, np.float32)
        slab[captured["j"], captured["i"]] = values
        raw[name] = slab
    lsm = noahmp_flux_writeback(**raw)
    expected = noahmp_seam_ownership(mynn=after_mynn, lsm=lsm)
    for name in ("ust", "chs", "chs2", "cqs2", "flhc", "flqc",
                 "tsk", "hfx", "qfx", "lh", "qsfc", "znt"):
        np.testing.assert_array_equal(
            _host(driver.fields[name]), expected[name], err_msg=name)

    identity = driver.noahmp_params.land_use
    diag = noahmp_post_lsm_diagnostics(
        ivgtyp=_host(driver.fields["ivgtyp"]),
        xice=_host(driver.fields["xice"]), iswater=identity.iswater,
        isice=identity.isice, isurban=identity.isurban, lcz=identity.lcz,
        psfc=_host(driver.fields["psfc"]), tsk=expected["tsk"],
        hfx=expected["hfx"], qfx=expected["qfx"], qsfc=expected["qsfc"],
        chs2=expected["chs2"], cqs2=expected["cqs2"],
        fveg=_host(driver.fields["fvegxy"]),
        t2mv=_host(driver.fields["t2mvxy"]),
        t2mb=_host(driver.fields["t2mbxy"]),
        q2mv=_host(driver.fields["q2mvxy"]),
        q2mb=_host(driver.fields["q2mbxy"]))
    np.testing.assert_array_equal(_host(driver.fields["t2"]), diag["t2"])
    np.testing.assert_array_equal(_host(driver.fields["q2"]), diag["q2"])
    np.testing.assert_allclose(
        _host(driver.fields["th2"]), diag["th2"], rtol=2.0e-7, atol=0.0)
    pbl = _capture_mynn_pbl_inputs(
        driver, atmosphere, cfg, monkeypatch)
    for name in ("tsk", "ust", "hfx", "qfx", "qsfc"):
        np.testing.assert_array_equal(pbl[name], expected[name], err_msg=name)
    np.testing.assert_array_equal(pbl["t2"], diag["t2"])
    np.testing.assert_array_equal(pbl["q2"], diag["q2"])
    np.testing.assert_allclose(
        pbl["th2"], diag["th2"], rtol=2.0e-7, atol=0.0)
