"""Public random requests fail before source acquisition or GPU admission."""
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.calibration_admission import UNCALIBRATED_SPREAD_REASON
from gpuwm.ensemble.door import request_for_payload
from gpuwm.ensemble.request import EnsembleRequest

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("controls", [
    {"perturbation": {}}, {"perturbation": {"kind": "uniform-wind", "amplitude": 0.5}},
    {"stochastic": {"sppt": True}}, {"stochastic": {"sppt": {"stddev": 0.0}}},
    {"stochastic": {"skebs": True}}, {"stochastic": {"skebs": {}}},
    {"stochastic": {"skebs": {"psi": False}}}, {"stochastic": {"spp": True}},
    {"stochastic": {"spp": {"pbl": 1}}}, {"stochastic": {"spp": {"lsm": 1}}},
    {"stochastic": {"spp_configs": {"pbl": {"stddev": 0.5}}}},
])
@pytest.mark.parametrize("members", [1, 20])
def test_uncalibrated_random_request_refuses_before_provider_construction(controls, members, monkeypatch):
    from gpuwm.ensemble import stochastic_model
    monkeypatch.setattr(stochastic_model.StochasticModelProvider, "from_mapping",
        lambda *a, **k: pytest.fail("uncalibrated request reached its numerical provider"))
    with pytest.raises(ValueError, match="calibrated against observations") as caught:
        EnsembleRequest(members, **controls)
    assert str(caught.value) == UNCALIBRATED_SPREAD_REASON


def test_explicitly_disabled_stochastic_controls_remain_ordinary():
    controls = {"sppt": False, "skebs": {"psi": False, "theta": False},
                "spp": {"conv": 0, "pbl": 0, "lsm": 0}}
    assert EnsembleRequest(20, stochastic=controls).stochastic == controls
    payload = b'[ensemble]\nmembers=2\n[ensemble.stochastic]\nsppt=false\nskebs=false\nspp=false\n'
    assert request_for_payload(payload).members == 2


def test_mutated_request_is_revalidated_at_the_run_door():
    controls = {"sppt": False}
    request = EnsembleRequest(20, stochastic=controls)
    controls["sppt"] = True
    with pytest.raises(ValueError, match="calibrated against observations"):
        EnsembleRequest.from_mapping(request)


@pytest.mark.parametrize("selector", ["spp", "spp_conv", "spp_pbl", "spp_lsm", "sppt", "skebs", "stoch_force_opt"])
@pytest.mark.parametrize("ensemble", [True, False])
def test_native_domain_switch_refuses_during_captured_config_admission(selector, ensemble):
    prefix = '[ensemble]\nmembers=2\n' if ensemble else ''
    payload = (prefix + '[[domains]]\n[domains.run]\n' + selector + '=1\n').encode()
    with pytest.raises(ValueError, match="spread and probabilities would be meaningless"):
        request_for_payload(payload)


@pytest.mark.parametrize("command", ["go", "ensemble"])
def test_go_door_refuses_before_fetch_chain_or_gpu_session(command, tmp_path, monkeypatch):
    from gpuwm import go_cli
    from gpuwm.ensemble import production
    config = tmp_path/'random.toml'
    config.write_text('[ensemble]\nmembers=20\n[ensemble.perturbation]\namplitude=0.5\n')
    monkeypatch.setattr(go_cli, "_go_launch", lambda *a, **k: pytest.fail("download chain was entered"))
    monkeypatch.setattr(production, "PreparedEnsembleSession",
                        lambda *a, **k: pytest.fail("GPU ensemble session was constructed"))
    with pytest.raises(ValueError, match="calibrated against observations"):
        go_cli.go_main(SimpleNamespace(config=config, command=command))
    assert not config.with_suffix('').exists()


def test_native_source_request_preserves_member_descriptors_without_random_amplitudes():
    sources = ({"source": "hrrr", "cycle": "2024-01-01T00:00:00Z"},
               {"source": "rap", "cycle": "2024-01-01T00:00:00Z"})
    request = EnsembleRequest(2, sources=sources)
    assert request.sources == sources and request.perturbation is None and request.stochastic is None


@pytest.mark.parametrize("kind", ["wrfinput", "met_em"])
def test_input_directory_cli_refuses_before_native_loading(kind, tmp_path, monkeypatch, capsys):
    import json
    from gpuwm import wrfinput_forecast, metem_forecast
    module = wrfinput_forecast if kind == "wrfinput" else metem_forecast
    target = "run_wrf_forecast" if kind == "wrfinput" else "run_metem_forecast"
    monkeypatch.setattr(module, target, lambda *a, **k: pytest.fail("native input or GPU admission was entered"))
    flag = "--wrfinput" if kind == "wrfinput" else "--met-em"
    request = {"members": 20, "stochastic": {"sppt": True}}
    assert module.main([flag, str(tmp_path/'absent-inputs'), '--outdir', str(tmp_path/'out'),
                       '--ensemble-request', json.dumps(request)]) == 2
    assert UNCALIBRATED_SPREAD_REASON in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize("kind", ["wrfinput", "met_em"])
@pytest.mark.parametrize("selector", ["sppt", "skebs", "spp", "spp_lsm", "rand_perturb"])
def test_native_namelist_refuses_before_header_reads(kind, selector, tmp_path, monkeypatch):
    from gpuwm import wrfinput_forecast, metem_forecast, wrfinput_door, metem_door
    (tmp_path/'namelist.input').write_text('&stoch\n '+selector+' = 1,\n/\n')
    resolver = wrfinput_door if kind == 'wrfinput' else metem_door
    name = 'resolve_wrfinput_run' if kind == 'wrfinput' else 'resolve_metem_run'
    monkeypatch.setattr(resolver, name, lambda *a, **k: pytest.fail('native headers were read'))
    run = wrfinput_forecast.run_wrf_forecast if kind == 'wrfinput' else metem_forecast.run_metem_forecast
    with pytest.raises(ValueError, match='calibrated against observations'):
        run(tmp_path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('reference', ['gpuwm.da.perturb', 'experimental-stub', 'none'])
@pytest.mark.parametrize('command', ['run', 'cycle'])
def test_legacy_command_keeps_its_284_references(command, reference, tmp_path, monkeypatch):
    """2.8.4 shipped these references; the calibration gate must not touch them."""
    from tools import ensemble_forecast
    shipped = REPO/'configs'/'ensemble'/'may1999_tiny_2member.toml'
    text = shipped.read_text(encoding='utf-8').replace(
        'perturbation = "gpuwm.da.perturb"', 'perturbation = "%s"' % reference)
    if reference == 'none':
        text = text[:text.index('[ensemble.perturbation_options]')]
    config = tmp_path/'ensemble.toml'
    config.write_text(text.replace('base_config = "', 'base_config = "%s/' % shipped.parent.as_posix()))
    reached = []
    def run_ensemble(cfg, root, **kwargs):
        reached.append(cfg.perturbation)
        return SimpleNamespace(status='COMPLETE', ens_root=root, manifest_path=root/'m.json',
                               ran=(), skipped=())
    def run_cycles(cfg, root, **kwargs):
        reached.append(cfg.perturbation)
        return SimpleNamespace(status='COMPLETE', ens_root=root, manifest_path=root/'m.json',
                               cycles_run=())
    monkeypatch.setattr(ensemble_forecast, 'run_ensemble', run_ensemble)
    monkeypatch.setattr(ensemble_forecast, 'run_cycles', run_cycles)
    args = [command, '--ensemble-config', str(config), '--ens-root', str(tmp_path/'out')]
    if command == 'cycle':
        args += ['--cycles', '1', '--cycle-seconds', '600']
    assert ensemble_forecast.main(args) == 0
    assert reached == [reference]


@pytest.mark.parametrize('command', ['go', 'ensemble', 'run', 'resume'])
def test_common_cli_refuses_before_provenance_or_capability_probes(command, tmp_path, monkeypatch, capsys):
    from gpuwm import cli, provenance_gate, capabilities
    config = tmp_path/'config.toml'
    config.write_text('[ensemble]\nmembers=20\n[ensemble.stochastic]\nsppt=true\n')
    def forbidden(*a, **k):
        pytest.fail('uncalibrated request entered a startup resource probe')
    monkeypatch.setattr(provenance_gate, 'announce', forbidden)
    monkeypatch.setattr(capabilities, 'require_for_command', forbidden)
    monkeypatch.setattr(cli, '_dispatch', forbidden)
    assert cli.main([command, str(config)]) == 2
    assert UNCALIBRATED_SPREAD_REASON in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['options', 'inline', 'path'])
def test_run_plan_random_controls_refuse_before_startup_probes(mode, tmp_path, monkeypatch, capsys):
    import json
    from gpuwm import cli, provenance_gate
    config = '[ensemble]\nmembers=20\n[ensemble.stochastic]\nsppt=true\n'
    plan = {'schema':'gpuwm.run-plan.v1', 'name':'refusal', 'route':'experiment', 'config':{'inline':''}}
    if mode == 'options':
        plan['run_options'] = {'ensemble': {'members':20, 'stochastic':{'sppt':True}}}
    elif mode == 'inline':
        plan['config'] = {'inline':config}
    else:
        (tmp_path/'config.toml').write_text(config)
        plan['config'] = {'path':'config.toml'}
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a, **k: pytest.fail('startup probe was entered'))
    assert cli.main(['run-plan', str(path)]) == 2
    assert UNCALIBRATED_SPREAD_REASON in capsys.readouterr().err


@pytest.mark.parametrize('tree', [False, True])
@pytest.mark.parametrize('spelling', ['separate', 'equals', 'abbreviated'])
def test_prepared_cli_refuses_before_startup_or_prepared_loading(tree, spelling, tmp_path, monkeypatch, capsys):
    from gpuwm import prepared_single_domain_forecast, prepared_domain_tree_forecast, provenance_gate
    module = prepared_domain_tree_forecast if tree else prepared_single_domain_forecast
    config = tmp_path/'config.toml'
    config.write_text('[ensemble]\nmembers=20\n[ensemble.stochastic]\nsppt=true\n')
    monkeypatch.setattr(provenance_gate, 'announce_for_main', lambda *a, **k: pytest.fail('startup probe was entered'))
    argv = (['--experiment-config', str(config)] if spelling == 'separate' else
            ['--experiment-config='+str(config)] if spelling == 'equals' else ['--experiment', str(config)])
    assert module.main(argv) == 2
    assert UNCALIBRATED_SPREAD_REASON in capsys.readouterr().err


def test_sim_refuses_before_bundle_loading(tmp_path, monkeypatch):
    from gpuwm import stage_cli
    config = tmp_path/'config.toml'
    config.write_text('[ensemble]\nmembers=20\n[ensemble.stochastic]\nsppt=true\n')
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda *a, **k: pytest.fail('prepared bundle was opened'))
    assert stage_cli.sim_main(SimpleNamespace(experiment_config=config)) == 2


# Real member trajectories are not random amplitudes: they stay admitted at
# every door that refuses random spread.
LAGGED = ({"source": "hrrr", "cycle": "2026-10-01T18:00:00Z"},
          {"source": "hrrr", "cycle": "2026-10-01T17:00:00Z"},
          {"source": "hrrr", "cycle": "2026-10-01T16:00:00Z"})
MULTI_MODEL = ({"source": "hrrr", "cycle": "2026-10-01T18:00:00Z"},
               {"source": "rap", "cycle": "2026-10-01T18:00:00Z"},
               {"source": "gfs", "cycle": "2026-10-01T18:00:00Z"})


def _sources_toml(sources):
    rows = "".join('[[ensemble.sources]]\nsource="%s"\ncycle="%s"\n' % (item["source"], item["cycle"])
                   for item in sources)
    return '[ensemble]\nmembers=%d\n' % len(sources) + rows


@pytest.mark.parametrize("sources", [LAGGED, MULTI_MODEL], ids=["time-lagged", "multi-model"])
def test_lagged_and_multi_model_members_are_admitted_at_every_door_check(sources, tmp_path):
    from gpuwm.ensemble.calibration_admission import refuse_public_arguments
    request = EnsembleRequest(len(sources), sources=sources)
    assert request.sources == sources and request.perturbation is None and request.stochastic is None
    config = tmp_path/'members.toml'
    config.write_text(_sources_toml(sources))
    admitted = request_for_payload(config.read_bytes())
    assert admitted.members == len(sources)
    assert [dict(item) for item in admitted.sources] == [dict(item) for item in sources]
    for command in ("go", "ensemble", "run", "resume"):
        refuse_public_arguments(SimpleNamespace(command=command, config=config))
    refuse_public_arguments(SimpleNamespace(command="sim", experiment_config=config))
    import json
    refuse_public_arguments(SimpleNamespace(command=None, wrfinput=None, met_em=None,
        ensemble_request=json.dumps({"members": len(sources), "sources": list(sources)})))


@pytest.mark.parametrize("kind", ["time-lagged", "multi-model"])
def test_explicit_lagged_and_multi_model_recipes_resolve_without_the_refusal(kind):
    from datetime import datetime, timedelta, timezone
    from gpuwm.ensemble.automatic_sources import EnsembleSourceContext, resolve_ensemble_sources
    from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe
    start = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)
    end = start + timedelta(hours=12)
    context = EnsembleSourceContext(start, end, object(), SourceTrajectory("hrrr", start), "requested-trajectory")
    trajectories = () if kind == "time-lagged" else (
        SourceTrajectory("hrrr", start), SourceTrajectory("rap", start))
    recipe = build_recipe(source="hrrr", cycle=start, start=start, end=end, count=2,
                          base_seed=5, kind=kind, trajectories=trajectories)
    selection = resolve_ensemble_sources(2, context, recipe=recipe)
    assert selection.mode == "source-recipe" and selection.recipe.kind == kind
    assert selection.stochastic is None and selection.amplitude is None
    assert len({member.trajectory.identity for member in selection.recipe.members}) == 2
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources({"members": 2, "stochastic": {"sppt": True}}, context, recipe=recipe)


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


STOCHASTIC_CONFIGS = {
    "sppt": '[ensemble]\nmembers=20\n[ensemble.stochastic]\nsppt=true\n',
    "skebs": '[ensemble]\nmembers=20\n[ensemble.stochastic]\nskebs=true\n',
    "spp": '[ensemble]\nmembers=20\n[ensemble.stochastic]\nspp=true\n',
}


def _plan(tmp_path, config):
    import json
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'schema': 'gpuwm.run-plan.v1', 'name': 'calibration-door',
                                'route': 'experiment', 'config': {'path': str(config)}}))
    return plan


def _real_door(tmp_path, family="sppt"):
    import json
    config = tmp_path/'random.toml'
    config.write_text(STOCHASTIC_CONFIGS[family])
    native = tmp_path/'native'
    native.mkdir()
    request = json.dumps({"members": 20, "perturbation": {"kind": "uniform-wind", "amplitude": 0.5}})
    return {
        "go": ("gpuwm", ["go", str(config)]),
        "ensemble": ("gpuwm", ["ensemble", str(config)]),
        "run": ("gpuwm", ["run", str(config)]),
        "run-plan": ("gpuwm", ["run-plan", str(_plan(tmp_path, config))]),
        "wrfinput": ("gpuwm.wrfinput_forecast", ["--wrfinput", str(native), "--outdir",
                                                 str(tmp_path/'out'), "--ensemble-request", request]),
        "met_em": ("gpuwm.metem_forecast", ["--met-em", str(native), "--outdir",
                                            str(tmp_path/'out'), "--ensemble-request", request]),
    }


def _run_guarded(module, argv):
    import os
    import subprocess
    import sys
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-c", _GUARD, module, *argv], cwd=REPO, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


@pytest.mark.parametrize("family", ["sppt", "skebs", "spp"])
@pytest.mark.parametrize("door", ["go", "ensemble", "run", "run-plan", "wrfinput", "met_em"])
def test_real_entry_point_refuses_before_any_process_or_network(door, family, tmp_path):
    """Run the actual module entry, not mocks: no spawn, no socket, exit 2."""
    if door in ("wrfinput", "met_em") and family != "sppt":
        pytest.skip("the native doors take one request descriptor, covered once")
    module, argv = _real_door(tmp_path, family)[door]
    result = _run_guarded(module, argv)
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 2, result.stdout + result.stderr
    assert UNCALIBRATED_SPREAD_REASON in result.stderr
    assert not (tmp_path/'out').exists()


def _none_ensemble(tmp_path):
    config = tmp_path/'none.toml'
    config.write_text('[ensemble]\nmembers=2\nperturbation="none"\n')
    return config


@pytest.mark.parametrize("which", ["may1999_tiny_2member", "perturbation-none"])
@pytest.mark.parametrize("door", ["ensemble", "go", "run-plan"])
def test_real_entry_point_passes_shipped_and_none_ensembles_through_the_check(door, which, tmp_path):
    """The 2.8.4 shipped overlay and perturbation = "none" are not refused.

    The guard trips on the startup banner's first process spawn, which the
    command reaches only after the calibration check has admitted it.
    """
    config = (REPO/'configs'/'ensemble'/'may1999_tiny_2member.toml'
              if which == "may1999_tiny_2member" else _none_ensemble(tmp_path))
    argv = ["run-plan", str(_plan(tmp_path, config))] if door == "run-plan" else [door, str(config)]
    result = _run_guarded("gpuwm", argv)
    assert UNCALIBRATED_SPREAD_REASON not in result.stdout + result.stderr
    assert 'GUARD_EVENTS=["process spawn"]' in result.stdout, result.stdout + result.stderr


def test_none_perturbation_is_the_absent_key():
    assert EnsembleRequest(2, perturbation="none").perturbation is None
    assert request_for_payload(b'[ensemble]\nmembers=2\nperturbation="none"\n').perturbation is None


@pytest.mark.parametrize("reference", ["none", "gpuwm.da.perturb", "experimental-stub"])
def test_284_string_references_pass_the_calibration_check(reference):
    from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
    refuse_uncalibrated_random(perturbation=reference)


def test_every_shipped_config_passes_the_calibration_check():
    from gpuwm.ensemble.calibration_admission import refuse_public_arguments
    configs = sorted((REPO/'configs').rglob('*.toml'))
    assert any(path.name == 'may1999_tiny_2member.toml' for path in configs)
    for path in configs:
        for args in (SimpleNamespace(command="go", config=path),
                     SimpleNamespace(command="sim", experiment_config=path)):
            try:
                refuse_public_arguments(args)
            except Exception as error:
                assert UNCALIBRATED_SPREAD_REASON not in str(error), path
