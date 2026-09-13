"""Audit C-062/C-074: real front-door composition, no forecast executed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import stage_cli
from gpuwm.cli import build_parser
from test_stage_seams import _authority, _single_domain_bundle, _tree_bundle


def operands(tmp_path, layout="single"):
    root = (_single_domain_bundle if layout == "single" else _tree_bundle)(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    return stage_cli.resolve_bundle(root), dict(experiment_config=config,
        wps_namelist=wps, outdir=tmp_path / "out")


@pytest.mark.parametrize("stream_init", ["auto", "resident", "store"])
def test_single_stage_relays_validated_streaming_operands(tmp_path, stream_init):
    bundle, kw = operands(tmp_path)
    command = stage_cli.sim_command(bundle, **kw, tiles='{"mode":"auto"}', stream_init=stream_init)
    assert json.loads(command[command.index("--tiles") + 1])["mode"] == "auto"
    assert command[command.index("--stream-init") + 1] == stream_init
    from gpuwm.prepared_single_domain_forecast import build_parser as runner_parser
    # The emitted command is accepted by the actual owning runner.
    args = runner_parser().parse_args(command[3:])
    assert args.stream_init == stream_init


@pytest.mark.parametrize("tiles", ["[]", '{"mode":"invalid"}', '{"unknown":1}', '{bad'])
def test_invalid_streaming_never_claims_output(tmp_path, capsys, tiles):
    bundle, kw = operands(tmp_path)
    args = SimpleNamespace(prepared_root=bundle["document"].parent,
        runner="auto", print_command=False, run_stamp="off", tiles=tiles,
        **kw)
    assert stage_cli.sim_main(args) == 2
    assert not kw["outdir"].exists()
    assert "tiles" in capsys.readouterr().err


@pytest.mark.parametrize("option", [{"tiles":'{"mode":"auto"}'}, {"stream_init":"store"}])
def test_tree_does_not_silently_swallow_single_runner_overrides(tmp_path, option):
    bundle, kw = operands(tmp_path, "tree")
    with pytest.raises(stage_cli.StageRefusal, match="tree"):
        stage_cli.sim_command(bundle, **kw, **option)


def test_sim_parser_exposes_runner_streaming_flags(tmp_path):
    _, kw = operands(tmp_path)
    args = build_parser().parse_args(["sim", str(tmp_path / "prepared"),
        "--experiment-config", str(kw["experiment_config"]), "--outdir", str(kw["outdir"]),
        "--tiles", '{"mode":"auto"}', "--stream-init", "store"])
    assert args.stream_init == "store"
    assert json.loads(args.tiles)["mode"] == "auto"


def test_tree_named_profile_reaches_owning_runner_parser(tmp_path):
    bundle, kw = operands(tmp_path, "tree")
    command = stage_cli.sim_command(bundle, **kw, physics_profile="test-assertion")
    from gpuwm.prepared_domain_tree_forecast import build_parser as tree_parser
    args = tree_parser().parse_args(command[3:])
    assert args.physics_profile == "test-assertion"


def test_missing_single_wps_refuses_before_output_claim(tmp_path):
    bundle, kw = operands(tmp_path)
    kw["wps_namelist"] = None
    args = SimpleNamespace(prepared_root=bundle["document"].parent,
        runner="auto", print_command=False, run_stamp="off", **kw)
    assert stage_cli.sim_main(args) == 2
    assert not kw["outdir"].exists()


@pytest.mark.parametrize("changed_domain", [None, 1, 2])
def test_tree_named_profile_checks_every_domain_without_rewriting(changed_domain):
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm import prepared_single_domain_forecast as single
    switches = single._profile_runtime_switches("gfs", single.PHYSICS_PROFILE)
    domains = [SimpleNamespace(grid_id=i, run=SimpleNamespace(**switches)) for i in (1, 2)]
    if changed_domain is not None:
        domains[changed_domain - 1].run.mp_physics = 18
    exp = SimpleNamespace(domains=tuple(domains), root=domains[0])
    before = [vars(d.run).copy() for d in domains]
    if changed_domain is None:
        receipt = tree.validate_physics_profile(exp, source="gfs", profile=single.PHYSICS_PROFILE)
        assert [d["grid_id"] for d in receipt["validated_domains"]] == [1, 2]
    else:
        with pytest.raises(ValueError, match=f"d{changed_domain:02d}"):
            tree.validate_physics_profile(exp, source="gfs", profile=single.PHYSICS_PROFILE)
    assert before == [vars(d.run) for d in domains]


def test_unnamed_tree_profile_has_no_assertion():
    from gpuwm.prepared_domain_tree_forecast import validate_physics_profile
    assert validate_physics_profile(object(), source="gfs", profile=None) is None
