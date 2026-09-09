"""Tiled GFS admission and actual preparation must select the same backend."""
from argparse import Namespace
from pathlib import Path

import pytest

from gpuwm import go_cli, source_cli


def config(tmp_path, mode="auto", store="host"):
    path = tmp_path / "forecast.toml"
    path.write_text(f'[tiles]\nmode = "{mode}"\nstore = "{store}"\n'
                    '[[domain]]\ngrid_id = 1\nparent_id = 0\n', encoding="utf-8")
    return path


def command(plan, tmp_path):
    return go_cli.prepare_command(plan, tmp_path / "bridge", manifest=tmp_path / "manifest.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-09-08_18:00:00", geog_root=tmp_path / "geog")


def plan(path, tmp_path):
    return {"source": "gfs", "config": path, "data": tmp_path / "data",
            "authority": tmp_path / "authority", "prepared": tmp_path / "prepared"}


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_go_and_direct_prep_forward_the_cpu_choice_from_the_same_configuration(tmp_path, mode, capsys):
    path = config(tmp_path, mode)
    before = path.read_bytes()
    args = Namespace(source="gfs", experiment_config=path, preprocess_backend=None)
    source_cli._apply_configuration_preprocess_default(args)
    assert args.preprocess_backend == "cpu"
    assert "CPU preprocessing" in capsys.readouterr().err
    argv = command(plan(path, tmp_path), tmp_path)
    assert argv[argv.index("--preprocess-backend") + 1] == "cpu"
    assert path.read_bytes() == before


@pytest.mark.parametrize("backend", ["cpu", "cuda", "auto"])
def test_explicit_prep_backend_is_preserved_without_reading_the_config(tmp_path, backend):
    args = Namespace(source="gfs", experiment_config=tmp_path / "missing.toml", preprocess_backend=backend)
    source_cli._apply_configuration_preprocess_default(args)
    assert args.preprocess_backend == backend


@pytest.mark.parametrize("mode,store", [("off", "host"), ("auto", "device")])
def test_non_host_tiled_workflows_keep_the_existing_backend_defaults(tmp_path, mode, store):
    path = config(tmp_path, mode, store)
    args = Namespace(source="gfs", experiment_config=path, preprocess_backend=None)
    source_cli._apply_configuration_preprocess_default(args)
    assert args.preprocess_backend is None
    assert "--preprocess-backend" not in command(plan(path, tmp_path), tmp_path)


def test_other_sources_do_not_read_a_configuration_or_change_backend(tmp_path):
    args = Namespace(source="hrrr", experiment_config=tmp_path / "missing.toml", preprocess_backend=None)
    source_cli._apply_configuration_preprocess_default(args)
    assert args.preprocess_backend is None


def test_invalid_gfs_config_does_not_silently_choose_a_backend(tmp_path):
    path = tmp_path / "invalid.toml"; path.write_text("[tiles\n", encoding="utf-8")
    args = Namespace(source="gfs", experiment_config=path, preprocess_backend=None)
    with pytest.raises(ValueError):
        source_cli._apply_configuration_preprocess_default(args)
