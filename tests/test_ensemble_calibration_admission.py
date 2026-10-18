"""Public random doors admit implemented consumers and refuse lost operations."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest

from gpuwm.ensemble.calibration_admission import refuse_public_arguments
from gpuwm.ensemble.door import request_for_payload
from gpuwm.ensemble.request import EnsembleRequest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("controls", [
    {"sppt": True}, {"sppt": {"stddev": 0.25}}, {"skebs": True},
    {"skebs": {}}, {"spp": True}, {"spp": {"pbl": 1}},
    {"spp": {"lsm": 1}}, {"spp_configs": {"pbl": {"stddev": 0.5}}},
])
@pytest.mark.parametrize("members", [1, 20])
def test_implemented_random_request_keeps_its_controls(controls, members):
    assert EnsembleRequest(members, stochastic=controls).stochastic == controls


def test_mutated_request_revalidates_the_actual_provider_schema():
    controls = {"sppt": False}
    request = EnsembleRequest(2, stochastic=controls)
    controls["sppt"] = True
    assert EnsembleRequest.from_mapping(request) is request
    controls["absent"] = True
    with pytest.raises(ValueError, match="stochastic controls must declare only"):
        EnsembleRequest.from_mapping(request)


@pytest.mark.parametrize("command", ["go", "ensemble", "run", "resume", "branch"])
@pytest.mark.parametrize("family", ["sppt", "skebs", "spp"])
def test_public_config_check_admits_implemented_families(command, family, tmp_path):
    config = tmp_path / "case.toml"
    config.write_text(f'[ensemble]\nmembers=2\n[ensemble.stochastic]\n{family}=true\n')
    refuse_public_arguments(SimpleNamespace(command=command, config=config))
    assert request_for_payload(config.read_bytes()).stochastic == {family: True}


@pytest.mark.parametrize("mode", ["options", "inline", "path"])
def test_run_plan_keeps_active_supported_controls(mode, tmp_path):
    config = '[ensemble]\nmembers=2\n[ensemble.stochastic]\nsppt=true\n'
    plan = {"config": {"inline": ""}}
    if mode == "options":
        plan["run_options"] = {"ensemble": {"members": 2, "stochastic": {"sppt": True}}}
    elif mode == "inline":
        plan["config"] = {"inline": config}
    else:
        (tmp_path / "case.toml").write_text(config)
        plan["config"] = {"path": "case.toml"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    refuse_public_arguments(SimpleNamespace(command="run-plan", plan=path))


@pytest.mark.parametrize("selector", ["rand_perturb", "multi_perturb", "perturb_bdy", "pert_mynn", "pert_thom"])
def test_native_absent_consumer_refuses_before_headers(selector, tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    native.joinpath("namelist.input").write_text(f"&stoch\n {selector}=1,\n/\n")
    with pytest.raises(ValueError, match=selector) as caught:
        refuse_public_arguments(SimpleNamespace(command=None, wrfinput=native))
    assert "no runtime consumer" in str(caught.value)
    assert ("RAND_PERT history and restart field would be missing" if selector == "rand_perturb"
            else "unperturbed") in str(caught.value)


def test_disabled_stochastic_and_none_remain_ordinary():
    controls = {"sppt": False, "skebs": {"psi": False, "theta": False},
                "spp": {"conv": 0, "pbl": 0, "lsm": 0}}
    assert EnsembleRequest(2, stochastic=controls).stochastic == controls
    assert EnsembleRequest(2, perturbation="none").perturbation is None


@pytest.mark.parametrize("reference", ["none", "gpuwm.da.perturb", "experimental-stub"])
def test_shipped_string_references_keep_their_own_owner(reference):
    from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
    refuse_uncalibrated_random(perturbation=reference)


def test_every_shipped_config_has_no_calibration_refusal():
    configs = sorted((REPO / "configs").rglob("*.toml"))
    assert any(path.name == "may1999_tiny_2member.toml" for path in configs)
    for config in configs:
        refuse_public_arguments(SimpleNamespace(command="go", config=config))


_GUARD = r'''
import json, runpy, socket, subprocess, sys
events = []
def forbid(kind):
    def blocked(*args, **kwargs):
        events.append(kind)
        raise SystemExit("GUARD: " + kind + " before the calibration refusal")
    return blocked
socket.socket.connect = forbid("network connect")
socket.create_connection = forbid("network connect")
spawn, original = forbid("process spawn"), subprocess.Popen.__init__
def popen(self, args, *rest, **kwargs):
    # CPython's ctypes.util.find_library reads the Linux linker cache with
    # `ldconfig -p` while GPU libraries are imported.  That is a library
    # lookup, not a download, a device call or work for the request.
    argv = list(args) if isinstance(args, (list, tuple)) else [args]
    if argv and str(argv[0]).rsplit("/", 1)[-1] == "ldconfig":
        return original(self, args, *rest, **kwargs)
    return spawn(self, args, *rest, **kwargs)
subprocess.Popen.__init__ = popen
module, argv = sys.argv[1], sys.argv[2:]
sys.argv = [module] + argv
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
except SystemExit as exit:
    code = exit.code
else:
    code = 0
print("GUARD_EVENTS=" + json.dumps(events))
sys.exit(code if isinstance(code, int) else 3)
'''

@pytest.mark.parametrize("command", ["ensemble", "go", "run"])
@pytest.mark.parametrize("family", ["sppt", "skebs", "spp"])
def test_real_entry_admits_supported_controls_before_startup(command, family, tmp_path):
    config = tmp_path / "case.toml"
    config.write_text(f'[ensemble]\nmembers=2\n[ensemble.stochastic]\n{family}=true\n')
    result = _run_guarded("gpuwm", [command, str(config)])
    assert 'GUARD_EVENTS=["process spawn"]' in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("command", ["ensemble", "go", "run", "resume", "branch"])
def test_real_entry_refuses_unimplemented_selector_before_startup(command, tmp_path):
    config = tmp_path / "case.toml"
    config.write_text('[shared]\npert_mynn=true\n')
    argv = [command, str(config)]
    if command == "branch":
        argv += ["--outdir", str(tmp_path / "branch")]
    result = _run_guarded("gpuwm", argv)
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 2
    assert "pert_mynn has no runtime consumer" in result.stderr
    assert "unperturbed" in result.stderr


def _run_guarded(module, argv):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-c", _GUARD, module, *argv], cwd=REPO,
        env=environment, capture_output=True, text=True, timeout=180)
