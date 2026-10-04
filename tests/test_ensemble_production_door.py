from types import SimpleNamespace

import pytest

from gpuwm.ensemble.door import request_for_config, request_for_payload, production_run_scope
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.runtime_context import current_session


def test_config_ensemble_thresholds_and_cli_override_remain_one_request(tmp_path):
    config = tmp_path / "forecast.toml"
    config.write_text('[ensemble]\nmembers = 20\nbase_seed = 73\n'
                      '[ensemble.thresholds]\nwind10 = [10.0, 25.0]\n')
    request = request_for_config(config, members=4, keep_member_files=True)
    assert request.members == 4 and request.base_seed == 73
    assert request.keep_member_files
    assert request.thresholds == {"wind10": [10.0, 25.0]}
    config.write_text('[ensemble]\nmembers = 2\nunknown = 1\n')
    with pytest.raises(ValueError, match="unknown ensemble"):
        request_for_config(config)


def test_plain_config_never_constructs_an_ensemble_session(tmp_path):
    config = tmp_path / "forecast.toml"
    config.write_text('[experiment]\nname = "domain"\n')
    assert request_for_config(config) is None
    with production_run_scope(None, output_directory=tmp_path,
            session_factory=lambda *args, **kw: pytest.fail("ordinary path created a session")):
        assert current_session() is None


def test_nested_front_doors_share_the_exact_request_and_reset_after_failure(tmp_path):
    created = []
    def factory(request, **kwargs):
        session = SimpleNamespace(request=request, output_directory=kwargs["output_directory"])
        created.append(session)
        return session
    request = EnsembleRequest(10)
    with pytest.raises(RuntimeError, match="forecast"):
        with production_run_scope(request, output_directory=tmp_path, session_factory=factory) as session:
            with production_run_scope(request.receipt(), output_directory=tmp_path / "chain") as nested:
                assert nested is session
            with pytest.raises(ValueError, match="different ensemble request"):
                with production_run_scope(4, output_directory=tmp_path):
                    pass
            raise RuntimeError("forecast")
    assert len(created) == 1 and current_session() is None


def test_cli_and_run_plan_use_the_same_validated_ensemble_request(tmp_path):
    from gpuwm.cli import build_parser
    from gpuwm.runplan import _run_option
    parser = build_parser()
    for command in ("go", "ensemble", "run"):
        args = parser.parse_args([command, "forecast.toml", "--members", "20", "--keep-member-files"])
        assert args.members == 20 and args.keep_member_files
    request = _run_option("ensemble", {"members": 20, "thresholds": {"wind10": [25]}}, tmp_path)
    assert request == EnsembleRequest(20, thresholds={"wind10": [25]}).receipt()


def test_captured_request_does_not_reopen_a_changed_source(tmp_path):
    config = tmp_path / "forecast.toml"
    payload = b"[ensemble]\nmembers = 10\nbase_seed = 17\n"
    config.write_bytes(payload)
    config.write_text("[ensemble]\nmembers = 99\n")
    assert request_for_payload(payload, members=4) == EnsembleRequest(4, base_seed=17)


def test_supervisor_carries_ensemble_overrides_to_every_worker(tmp_path):
    from gpuwm.supervisor import _worker_command, _parser
    command = _worker_command(tmp_path / "source.toml", tmp_path / "captured.toml",
        tmp_path / "out", restart=tmp_path / "checkpoint.npz", health_debug=False,
        members=20, keep_member_files=True)
    args = _parser().parse_args(command[3:])
    assert args.members == 20 and args.keep_member_files
    assert args.restart == tmp_path / "checkpoint.npz"


def test_actual_worker_uses_captured_ensemble_scope_and_success_manifest(tmp_path, monkeypatch):
    import hashlib
    from gpuwm import case_data, runtime, supervisor
    from gpuwm.ensemble import production
    payload = b"[ensemble]\nmembers = 10\nbase_seed = 17\n"
    source, captured, out = tmp_path / "source.toml", tmp_path / "captured.toml", tmp_path / "out"
    source.write_text("[ensemble]\nmembers = 99\n")
    captured.write_bytes(payload)
    out.mkdir()
    for key, value in {"GPUWM_RUN_ID": "ensemble-worker",
        "GPUWM_CONFIG_DIGEST": hashlib.sha256(payload).hexdigest(),
        "GPUWM_STARTED_AT_UTC": "2026-10-02T00:00:00Z",
        "GPUWM_GPU_UUID": "GPU-test", "GPUWM_GPU_DRIVER": "test",
        "GPUWM_GPU_NAME": "test", "GPUWM_INPUT_HASHES_JSON": "{}"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(supervisor.INPUT_AUTHORITIES_ENV, raising=False)
    exp = SimpleNamespace(restart_interval_s=None, domains=(), run_seconds=24.)
    def load(data, **kwargs):
        assert data == payload
        return exp, object()
    monkeypatch.setattr(case_data, "load_experiment_case_bytes", load)
    monkeypatch.setattr(supervisor, "_validate_worker_resolved_input_inventory", lambda *args: None)
    monkeypatch.setattr(production, "PreparedEnsembleSession",
        lambda request, **kwargs: SimpleNamespace(request=request))
    manifest = out / "ensemble-run.json"
    manifest.write_text('{"status":"PASS"}')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    def run(*args, **kwargs):
        assert current_session().request == EnsembleRequest(4, base_seed=17, keep_member_files=True)
        return runtime.ExperimentRunSummary((), 24., True,
            ensemble_manifest=manifest, ensemble_manifest_sha256=digest)
    monkeypatch.setattr(runtime, "run_experiment", run)
    capsules = []
    monkeypatch.setattr(supervisor, "emit_run_capsule", lambda *args, **kwargs: capsules.append(kwargs))
    assert supervisor._worker_main(SimpleNamespace(config=source, config_payload=captured,
        outdir=out, restart=None, health_debug=False, members=4, keep_member_files=True)) == 0
    assert current_session() is None
    assert capsules[0]["output"]["ensemble_manifest"] == {"path": str(manifest), "sha256": digest}
    assert supervisor.read_heartbeat(out / supervisor.HEARTBEAT_NAME).status == "complete"


@pytest.mark.parametrize("kind", ["wrfinput", "met_em"])
def test_input_directory_doors_carry_the_full_request_to_the_worker(kind, tmp_path, monkeypatch):
    import json
    from gpuwm import wrfinput_forecast, metem_forecast
    module = wrfinput_forecast if kind == "wrfinput" else metem_forecast
    seen = []
    name = "run_wrf_forecast" if kind == "wrfinput" else "run_metem_forecast"
    monkeypatch.setattr(module, name, lambda *args, **kwargs: seen.append(kwargs) or 0)
    # One member: an input directory is one trajectory's files, so these
    # doors refuse N > 1 (tests/test_ensemble_member_inputs.py).
    request = EnsembleRequest(1, base_seed=17, keep_member_files=True,
                              stochastic={"sppt": False}, thresholds={"wind10": [25.]})
    flag = "--wrfinput" if kind == "wrfinput" else "--met-em"
    assert module.main([flag, str(tmp_path), "--outdir", str(tmp_path / "out"),
        "--ensemble-request", json.dumps(request.receipt()), "--_worker"]) == 0
    assert seen[0]["ensemble_request"] == request.receipt()


@pytest.mark.parametrize("flag", ["--wrfinput", "--met-em"])
def test_common_run_directory_input_door_carries_member_option(flag, tmp_path, monkeypatch):
    from gpuwm import capabilities, cli, wrfinput_forecast, metem_forecast
    module, name = ((wrfinput_forecast, "run_wrf_forecast") if flag == "--wrfinput"
                    else (metem_forecast, "run_metem_forecast"))
    seen = []
    monkeypatch.setattr(module, name, lambda *args, **kwargs: seen.append(kwargs) or 0)
    # The forecast is replaced above, so nothing here needs a card runtime.
    # On an install without CuPy (the public CPU leg) the door's own runtime
    # check refused with exit 2 before the member option was carried.
    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    assert cli.main(["run", flag, str(tmp_path), "--outdir", str(tmp_path / "out"),
                     "--members", "1", "--keep-member-files"]) == 0
    assert seen[0]["ensemble_request"] == EnsembleRequest(1, keep_member_files=True).receipt()
