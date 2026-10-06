"""Source units and missing-data guards for prescribed smoke optics."""
from pathlib import Path
import json

import numpy as np
import pytest

from gpuwm.core import rrtmg_aerosol_optics as ao

F = np.float32
REPO = Path(__file__).parents[1]
ORACLE = REPO / "tests/data/smoke_aer3_oracle/smoke_aer3_oracle.npz"


def _array(value):
    return np.array([[value]], dtype=F)


def test_source_mass_units_give_the_layer_optical_depth():
    # 100 ug/kg * 1 kg/m3 * 100 m = 0.01 g/m2; source extinction
    # 4.5 m2/g yields layer AOD 0.045. Two equal layers each have that AOD.
    result = ao.smoke_aod_from_dry_mixing_ratio(
        np.full((1, 2), 100, F), np.ones((1, 2), F), np.full((1, 2), 100, F))
    np.testing.assert_array_equal(result, np.full((1, 2), F(0.045), F))


@pytest.mark.parametrize("input_name,value", [
    ("smoke", -1), ("smoke", np.nan), ("smoke", np.inf),
    ("smoke", 11000), ("rho", 0), ("rho", -1), ("dz", 0),
])
def test_corrupt_external_profiles_are_refused(input_name, value):
    arrays = dict(smoke=_array(100), rho=_array(1), dz=_array(100))
    arrays[input_name] = _array(value)
    with pytest.raises(ValueError, match="physically invalid"):
        ao.smoke_aod_from_dry_mixing_ratio(arrays["smoke"], arrays["rho"], arrays["dz"])


def test_native_posted_concentration_uses_donor_thermodynamics():
    # UPP emits kg/m3. For this source coefficient a 100 ug/kg tracer
    # appears as about 1.14e-7 kg/m3, not 100 ug/m3 or layer AOD.
    p, t = _array(95000), _array(290)
    q = _array(100)
    emitted = ((F(1) / F(287.04)) * (p / t) * q) * F(1.e-9)
    recovered = ao.smoke_dry_mixing_ratio_from_posted_density(emitted, p, t)
    np.testing.assert_allclose(recovered, q, rtol=2.e-7, atol=0)
    changed_donor = ao.smoke_dry_mixing_ratio_from_posted_density(emitted, p, t * F(2))
    np.testing.assert_allclose(changed_donor, q * F(2), rtol=2.e-7, atol=0)
    with pytest.raises(ValueError, match="donor_pressure"):
        ao.smoke_dry_mixing_ratio_from_posted_density(emitted, _array(0), t)


@pytest.mark.parametrize("smoke,enabled,error", [
    (None, True, "requires smoke_aod"),
    (_array(.1), False, "disabled"),
    (_array(-.1), True, "physically invalid"),
    (_array(np.nan), True, "physically invalid"),
    (np.zeros((2, 1), F), True, "shape"),
    (np.zeros((1, 1), np.float64), True, "float32"),
    (None, 1, "explicit boolean"),
])
def test_selected_smoke_refuses_missing_or_invalid_data_before_radiation(smoke, enabled, error):
    args = [_array(x) for x in (95000, 290, .005, 100, 1.e9, 1.e6)]
    with pytest.raises(ValueError, match=error):
        ao.aer3_sw_optics(*args, smoke_aod=smoke, smoke_feedback=enabled)


def test_smoke_twin_matches_the_compiled_source_oracle():
    assert ORACLE.is_file(), "compile the source smoke oracle before running this gate"
    with np.load(ORACLE) as fixture:
        receipt = json.loads(str(fixture["receipt"]))
        assert receipt["compiled"] is True
        assert receipt["commit"] == "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
        assert receipt["sources"]["module_add_emiss_burn.F"] == "024c838f64e5c5de098a63374b2163760f680e1e7d9bf81d882fc051dea3b31f"
        inputs = [fixture[f"in/{key}"] for key in ("p", "t", "qv", "dz8w", "nwfa", "nifa")]
        smoke_aod = ao.smoke_aod_from_dry_mixing_ratio(
            fixture["in/smoke_ugkg"], fixture["in/rho_dry"], fixture["in/dz8w"])
        assert np.array_equal(smoke_aod.view(np.uint32), fixture["out/smoke_aod"].view(np.uint32))
        recovered = ao.smoke_dry_mixing_ratio_from_posted_density(
            fixture["out/pm_posted_kgm3"], fixture["in/p"], fixture["in/t"])
        assert np.array_equal(recovered.view(np.uint32), fixture["out/smoke_recovered_ugkg"].view(np.uint32))
        actual = ao.aer3_sw_optics(*inputs, smoke_aod=smoke_aod, smoke_feedback=True)
        for value, key in zip(actual, ("tauaer", "ssaaer", "asyaer", "taod5503d")):
            assert np.array_equal(value.view(np.uint32), fixture[f"out/{key}"].view(np.uint32)), key
        assert (smoke_aod > F(3)).any(), "oracle must reach the per-layer cap"
        assert (smoke_aod == F(0)).any(), "oracle must include clean layers"
