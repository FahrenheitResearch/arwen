"""The offline door supplies target initialization context and keeps water updates."""
import numpy as np
import pytest

from gpuwm.offline_child import map_microphysics_to_nssl18, MomentDiagnosisRequired


def _morrison():
    fields = {name: np.zeros((2, 3, 4), np.float32)
              for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng")}
    fields["qv"].fill(0.01)
    fields["qc"].fill(0.0001)
    fields["qr"].fill(1e-12)
    return fields


def test_native_initialization_receives_real_density_and_keeps_water_changes(monkeypatch):
    from gpuwm.core import nssl2_offline_init
    source = _morrison()
    rho = np.full(source["qv"].shape, 0.75, np.float32)
    called = []

    def native(fields, density):
        np.testing.assert_array_equal(density, rho)
        called.append(True)
        result = {name: value.copy() for name, value in fields.items()}
        result["qv"] += result["qr"]
        result["qr"].fill(0)
        result["qndrop"] = np.full(rho.shape, 2e7, np.float32)
        result["qnn"] -= result["qndrop"]
        return result, {"method": "native-calcnfromq-test-seam", "mass_return_to_vapor_preserved": True}

    monkeypatch.setattr(nssl2_offline_init, "initialize_missing_moments", native)
    result, receipt = map_microphysics_to_nssl18(
        source, source_mp_physics=10, morr_rimed_ice=0,
        air_density=lambda: rho)
    assert called == [True]
    assert receipt["diagnosed_target_moments"] == ("qndrop",)
    assert receipt["target_initialization"]["mass_return_to_vapor_preserved"]
    assert np.all(result["qr"] == 0)
    np.testing.assert_array_equal(result["qnr"], source["nr"])
    assert np.all(source["qr"] > 0), "parent input is unchanged"
    np.testing.assert_allclose(sum(result[name] for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")),
                               sum(source[name] for name in ("qv", "qc", "qr", "qi", "qs", "qg")), rtol=0, atol=2e-9)


def test_missing_context_still_refuses_and_clear_air_needs_no_initializer():
    source = _morrison()
    with pytest.raises(MomentDiagnosisRequired, match="actual child air density"):
        map_microphysics_to_nssl18(source, source_mp_physics=10, morr_rimed_ice=0)
    source["qc"].fill(0)
    def no_density():
        pytest.fail("clear air must not allocate a target initializer")
    result, receipt = map_microphysics_to_nssl18(
        source, source_mp_physics=10, morr_rimed_ice=0, air_density=no_density)
    assert receipt["target_initialization"] is None
    assert np.all(result["qndrop"] == 0)
