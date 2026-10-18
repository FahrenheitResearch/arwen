"""Working random consumers warn; absent consumers refuse concrete breakage."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.request import EnsembleRequest


@pytest.mark.parametrize("controls", [
    {"sppt": True}, {"skebs": True}, {"spp": {"conv": 1}},
    {"spp": {"pbl": 1}}, {"spp": {"lsm": 1}},
])
def test_implemented_random_controls_are_admitted_with_warning(controls):
    request = EnsembleRequest(2, stochastic=controls)
    from gpuwm.ensemble.calibration_admission import random_warnings
    assert random_warnings(stochastic=request.stochastic) == [
        "uncalibrated spread, WRF default amplitudes"]


def test_uniform_initial_wind_request_has_an_actual_initializer(tmp_path):
    from gpuwm.ensemble.production import PreparedEnsembleSession
    request = EnsembleRequest(2, perturbation={"kind": "uniform-wind", "amplitude": 0.5})
    session = PreparedEnsembleSession(request, output_directory=tmp_path)
    session._require_member_inputs(SimpleNamespace(domains=()))
    assert callable(session._initialization_callback(0))


@pytest.mark.parametrize("selector", [
    "rand_perturb", "multi_perturb", "perturb_bdy", "perturb_chem_bdy",
    "pert_cld3", "pert_deng", "pert_farms", "pert_mynn", "pert_noah", "pert_thom",
])
def test_native_missing_consumers_name_the_lost_output(selector):
    from gpuwm.ensemble_admission import refuse_configured_random
    with pytest.raises(ValueError, match=selector) as caught:
        refuse_configured_random({"stoch": {selector: [1]}})
    assert "no runtime consumer" in str(caught.value)
    assert ("RAND_PERT history and restart field would be missing" if selector == "rand_perturb"
            else "unperturbed") in str(caught.value)
    assert "Next:" in str(caught.value)


def test_disabled_controls_have_no_warning_or_owner():
    from gpuwm.ensemble.calibration_admission import random_warnings
    controls = {"sppt": False, "skebs": {"psi": False, "theta": False},
                "spp": {"conv": 0, "pbl": 0, "lsm": 0}}
    assert random_warnings(stochastic=controls) == []
    assert EnsembleRequest(2, stochastic=controls).stochastic == controls


def test_unknown_initial_descriptor_refuses_the_missing_initializer():
    with pytest.raises(ValueError, match="no initial-state consumer"):
        EnsembleRequest(2, perturbation={"kind": "absent"})


def test_initial_wind_guard_is_restart_infrastructure():
    from gpuwm.io.restart import classify_state_attr
    assert classify_state_attr("_ensemble_initial_wind") == "infra"


def test_stochastic_authority_receipt_warns_only_for_active_controls():
    from gpuwm.config import RunConfig
    from gpuwm.ensemble.stochastic_authority import configure_member_inputs
    from gpuwm.ensemble.stochastic_model import StochasticModelProvider
    @dataclass(frozen=True)
    class Domain:
        grid_id: int
        run: object
    @dataclass(frozen=True)
    class Experiment:
        domains: tuple
    @dataclass(frozen=True)
    class Inputs:
        experiment: object
    inputs = Inputs(Experiment((Domain(1, RunConfig(nx=16, ny=12, nz=40,
        dx=3000., dy=3000., ztop=16000., dt=12., run_seconds=36.)),)))
    bound, receipt = configure_member_inputs(
        StochasticModelProvider.from_mapping({"sppt": True}), inputs)
    assert bound is inputs
    assert receipt["warnings"] == ["uncalibrated spread, WRF default amplitudes"]
    assert configure_member_inputs(StochasticModelProvider.from_mapping({"sppt": False}), inputs) == (inputs, None)
