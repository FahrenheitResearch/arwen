"""Public prepared-tree continuation forwards to the existing strict runner."""
import shlex
from types import SimpleNamespace

import pytest

from gpuwm import stage_cli
from gpuwm.cli import main as cli_main
from test_stage_seams import _authority, _single_domain_bundle, _tree_bundle


@pytest.mark.parametrize("sealed", [False, True])
def test_restart_changes_only_the_existing_runner_operands(tmp_path, sealed):
    root = _tree_bundle(tmp_path / "prepared")
    config, _ = _authority(tmp_path / "authority")
    bundle = stage_cli.resolve_bundle(root)
    kwargs = dict(experiment_config=config, wps_namelist=None, outdir=tmp_path / "out")
    original = stage_cli.sim_command(bundle, **kwargs)
    checkpoint = tmp_path / "earlier run" / "gpuwmrst_d02.npz"
    command = stage_cli.sim_command(bundle, **kwargs, restart=checkpoint,
                                    sealed_forcing_extension=sealed)
    index = command.index("--restart")
    assert command[index + 1] == str(checkpoint)
    del command[index:index + 2]
    if sealed:
        command.remove("--sealed-forcing-extension")
    assert command == original


def test_public_printed_restart_command_uses_actual_parser_and_spends_nothing(tmp_path, capsys):
    from gpuwm import prepared_domain_tree_forecast as runner
    root = _tree_bundle(tmp_path / "prepared")
    config, _ = _authority(tmp_path / "authority")
    checkpoint = tmp_path / "earlier run" / "gpuwmrst_d02.npz"
    output = tmp_path / "continued"
    assert cli_main(["sim", str(root), "--experiment-config", str(config),
        "--outdir", str(output), "--restart", str(checkpoint),
        "--sealed-forcing-extension", "--print-command"]) == 0
    command = shlex.split(capsys.readouterr().out.strip())
    parsed = runner.build_parser().parse_args(command[3:])
    assert parsed.restart == checkpoint and parsed.sealed_forcing_extension
    assert parsed.experiment_config == config
    assert parsed.experiment_config_sha256 == stage_cli._sha256(config)
    assert parsed.preparation_receipt_sha256 == stage_cli._sha256(root / "proof.json")
    assert not output.exists()


def test_public_sim_dispatches_checkpoint_to_same_runner_and_preserves_return_code(tmp_path, monkeypatch):
    from gpuwm import prepared_domain_tree_forecast as runner
    root = _tree_bundle(tmp_path / "prepared")
    config, _ = _authority(tmp_path / "authority")
    checkpoint = tmp_path / "checkpoint.npz"
    calls = []
    def execute(argv):
        parsed = runner.build_parser().parse_args(argv)
        assert parsed.restart == checkpoint
        assert parsed.experiment_config == config
        assert not parsed.sealed_forcing_extension
        calls.append(parsed)
        return 17
    monkeypatch.setattr(runner, "main", execute)
    assert cli_main(["sim", str(root), "--experiment-config", str(config),
        "--outdir", str(tmp_path / "out"), "--restart", str(checkpoint)]) == 17
    assert len(calls) == 1


def test_public_sim_keeps_the_runners_checkpoint_identity_refusal(tmp_path, monkeypatch, capsys):
    from gpuwm import capabilities, prepared_domain_tree_forecast as runner, provenance_gate
    root = _tree_bundle(tmp_path / "prepared")
    config, _ = _authority(tmp_path / "authority")
    checkpoint = tmp_path / "wrong-identity.npz"
    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *_: None)
    monkeypatch.setattr(capabilities, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "preflight_prepared_tree", lambda **kwargs:
                        SimpleNamespace(experiment=SimpleNamespace(domains=())))
    def restore(inputs, **kwargs):
        assert kwargs["restart"] == checkpoint
        raise runner.RestartMismatchError("checkpoint experiment identity differs")
    monkeypatch.setattr(runner, "run_prepared_tree", restore)
    assert cli_main(["sim", str(root), "--experiment-config", str(config),
        "--outdir", str(tmp_path / "out"), "--restart", str(checkpoint)]) == 2
    assert "--restart refused: checkpoint experiment identity differs" in capsys.readouterr().err


def test_single_bundle_still_requires_hierarchy_receipts_for_extension(tmp_path, capsys):
    flags = ["--sealed-forcing-extension"]
    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    output = tmp_path / "out"
    assert cli_main(["sim", str(root), "--experiment-config", str(config),
        "--wps-namelist", str(wps), "--outdir", str(output), *flags]) == 2
    assert "single prepared bundle binds its complete configuration and stop time" in capsys.readouterr().err
    assert not output.exists()
