"""Scalar authority, unchanged defaults and actual solver-entry profiles (CPU)."""
from datetime import datetime
import inspect

import numpy as np
import pytest

from gpuwm.core.trace_gases import (
    CLASSIC_GASES, LEGACY_LW_GASES, LEGACY_SW_GASES,
    validate_trace_gas_overrides)
from gpuwm.core.rrtm import rrtm_default_trace_gases
from gpuwm.core.rrtmg_sw import option4_trace_gases, expf
from gpuwm.core import rrtmg_legacy_prep as prep

F = np.float32
DECLARED = dict(co2=600e-6, n2o=450e-9, ch4=2200e-9, o2=.21,
                cfc11=400e-12, cfc12=900e-12, cfc22=350e-12, ccl4=250e-12)


@pytest.mark.parametrize("value", [0, -1e-6, float("nan"), float("inf"),
                                     True, np.bool_(False), 1.01, 1e-100])
def test_unrepresentable_explicit_fraction_is_not_a_solver_operand(value):
    with pytest.raises(ValueError):
        validate_trace_gas_overrides({"co2": value})


def test_ordinary_oxygen_and_positive_high_co2_are_owned_not_trace_capped():
    original = {"o2": .209488, "co2": .02}
    got = validate_trace_gas_overrides(original)
    assert got == original and got is not original
    original["o2"] = .1
    assert got["o2"] == .209488
    assert validate_trace_gas_overrides({"o2": 1.0}) == {"o2": 1.0}
    with pytest.raises(ValueError, match="unknown trace gas"):
        validate_trace_gas_overrides({"invented": 1e-6})
    with pytest.raises(TypeError, match="mapping"):
        validate_trace_gas_overrides([("co2", 1e-4)])


@pytest.mark.parametrize("year", [1900, 1974, 2000, 2021, 2050])
def test_default_year_formulas_remain_distinct_and_byte_exact(year):
    # Independent old wrapper expressions, including the two rounding orders.
    classic = dict(co2=float((280 + 90*np.exp(.02*(year-2000)))*1e-6),
                   n2o=319e-9, ch4=1774e-9)
    legacy_co2 = F(F(1e-6) * F(F(280) + F(F(90) * expf(F(F(.02)*(year-2000))))))
    legacy = np.array([legacy_co2, F(1774e-9), F(319e-9), F(.209488)], F)
    assert rrtm_default_trace_gases(year) == classic
    assert rrtm_default_trace_gases(year, {}) == classic
    assert np.asarray(option4_trace_gases(year), F).tobytes() == legacy.tobytes()
    assert np.asarray(option4_trace_gases(year, {}), F).tobytes() == legacy.tobytes()
    assert prep.lw_trace_gases(year)["co2"] == legacy_co2
    assert prep.lw_trace_gases(year, {}) == prep.lw_trace_gases(year)


def test_declared_values_replace_only_selected_operands():
    for function, supported in ((rrtm_default_trace_gases, CLASSIC_GASES),
                                (prep.lw_trace_gases, LEGACY_LW_GASES)):
        old = function(2021)
        changed = function(2021, {"co2": DECLARED["co2"]})
        assert {k: v for k,v in changed.items() if k != "co2"} == {
            k: v for k,v in old.items() if k != "co2"}
        function(2021, {k: DECLARED[k] for k in supported})
    with pytest.raises(ValueError, match="no absorption operand.*o2"):
        rrtm_default_trace_gases(2021, {"o2": .21})
    with pytest.raises(ValueError, match="no absorption operand.*cfc11"):
        option4_trace_gases(2021, {"cfc11": DECLARED["cfc11"]})


def _legacy_profile(longwave):
    from test_rrtmg_legacy_prep import _sw_fixtures, SW_DAY, _sw_prep_kwargs
    kw = _sw_prep_kwargs(_sw_fixtures(), SW_DAY[0])
    if longwave:
        kw = {k:v for k,v in kw.items() if k in inspect.signature(prep.lwrad_prep).parameters}
        kw.update(emiss=F(.95), nlayers=prep.compute_lw_nlayers(
            len(kw["p3d"])+1, kw["p8w"][-1]))
    return kw


@pytest.mark.parametrize("longwave", [False, True])
def test_all_legacy_operands_reach_model_and_above_model_layers(longwave):
    scalar = prep.lwrad_prep if longwave else prep.swrad_prep
    batch = prep.lwrad_prep_batch if longwave else prep.swrad_prep_batch
    supported = LEGACY_LW_GASES if longwave else LEGACY_SW_GASES
    kw = _legacy_profile(longwave)
    overrides = {k: DECLARED[k] for k in supported}
    baseline = scalar(**kw)
    got = scalar(**kw, trace_gas_overrides=overrides)
    bkw = {k: np.stack([v, v]) if isinstance(v, np.ndarray) and v.ndim else v
           for k,v in kw.items()}
    batched = batch(**bkw, trace_gas_overrides=overrides)
    for gas, value in overrides.items():
        name = gas+"vmr"
        assert got[name].size > len(kw["p3d"]), "above-model layer is exercised"
        np.testing.assert_array_equal(got[name], np.full_like(got[name], F(value)))
        np.testing.assert_array_equal(batched[name], np.stack([got[name], got[name]]))
        assert not np.array_equal(got[name], baseline[name]), gas
    # Supplying an empty declaration must not affect clouds, ozone or profiles.
    empty = scalar(**kw, trace_gas_overrides={})
    for key, value in baseline.items():
        if isinstance(value, np.ndarray):
            assert empty[key].dtype == value.dtype and empty[key].tobytes() == value.tobytes(), key


def test_classic_molecules_reach_the_cavallo_buffer_with_existing_rounding():
    from test_rrtm_longwave import _column_block
    from gpuwm.core.rrtm_lw import mm5atm_columns, _AMDN, _AMDC
    from gpuwm.core.rrtm_tables import load_rrtm_lw_tables
    kw = _column_block(ncol=2)
    for gas in CLASSIC_GASES:
        kw[gas+"vmr"] = DECLARED[gas]
    got = mm5atm_columns(np, load_rrtm_lw_tables(), **kw)
    assert got["wkl"].shape[1] > kw["t"].shape[1]
    # MM5ATM turns mixing fractions into molecules/cm2. Compare the full
    # column (including the buffer) using its WRF-prescribed rounding order.
    for gas, index, mass in (("co2", 1, None), ("n2o", 3, _AMDN), ("ch4", 5, _AMDC)):
        fraction = F(DECLARED[gas]) if mass is None else F(DECLARED[gas]/mass)*F(mass)
        np.testing.assert_array_equal(got["wkl"][:,:,index], fraction*got["coldry"])


def test_selected_spectra_distinguish_applied_inactive_and_missing_operand():
    from test_radiation_composition import _cfg
    from gpuwm.core.radiation_composition import trace_gas_override_status
    assert trace_gas_override_status(_cfg(1,1), {"co2": 6e-4}) == "applied"
    assert trace_gas_override_status(_cfg(4,1,"rrtmg_legacy"), DECLARED) == "applied"
    assert trace_gas_override_status(_cfg(0,1), {"co2": 6e-4}) == "inactive"
    for cfg, gas in ((_cfg(1,1), "o2"), (_cfg(0,4,"rrtmg_legacy"), "cfc11")):
        with pytest.raises(ValueError, match="no absorption operand"):
            trace_gas_override_status(cfg, {gas: DECLARED[gas]})
    with pytest.raises(ValueError, match="unknown trace gas"):
        trace_gas_override_status(_cfg(0,0), {"invented": 1e-6})


def test_legacy_sw_n2o_interface_is_not_claimed_as_absorption():
    from test_radiation_composition import _cfg
    from gpuwm.core.radiation_composition import (
        trace_gas_override_status, trace_gas_override_consumption)
    only_sw = _cfg(0,4,"rrtmg_legacy")
    assert trace_gas_override_status(only_sw, {"n2o": 8e-7}) == "inactive"
    assert trace_gas_override_status(only_sw, {"n2o": 8e-7,"co2": 8e-4}) == "partially_applied"
    assert trace_gas_override_consumption(only_sw, {"n2o": 8e-7,"co2": 8e-4}) == {
        "n2o": (), "co2": ("sw",)}
    assert trace_gas_override_consumption(_cfg(1,4,"rrtmg_legacy"), {"n2o": 8e-7}) == {
        "n2o": ("lw",)}


def test_modern_preserves_noaa_defaults_and_accepts_ordinary_oxygen():
    from gpuwm.core.rrtmgp import trace_gases, coefficient_gas_names, load_gas_tables
    assert trace_gases(datetime(1974,1,1)) == {"co2": 336.85*1e-6}
    assert trace_gases(datetime(1974,1,1), {"co2": 330e-6,"o2": .209488}) == {
        "co2": 330e-6, "o2": .209488}
    for kind in ("lw", "sw"):
        assert coefficient_gas_names(kind) == load_gas_tables(kind).gas_names
