"""Actual source-door parser and preparation authority materialization."""
import os
from pathlib import Path
import tomllib

import pytest

from gpuwm import tuning_prepare as tp


def test_cpu_flags_reach_the_real_source_cli_parser(tmp_path):
    from gpuwm import source_cli
    handoff = {"argv": ["--source", "hrrr-prs", "--input-list", str(tmp_path / "inputs.txt"),
                         "--author-input-manifest", str(tmp_path / "upstream.json")]}
    arguments = tp.preparation_arguments(handoff, wps=tmp_path / "namelist.wps", config=tmp_path / "experiment.toml",
        geog_root=tmp_path / "geog", bundle_root=tmp_path / "prepared", manifest_path=tmp_path / "own-inputs.json", workers=3)
    parser = source_cli._parser()
    parsed = parser.parse_args(arguments)
    assert parsed.preprocess_backend == "cpu"
    assert parsed.preprocess_workers == 3
    assert parsed.hierarchy_workers == 1
    assert parsed.experiment_config == tmp_path / "experiment.toml"
    assert parsed.author_input_manifest == tmp_path / "own-inputs.json"
    assert handoff["argv"][-1] == str(tmp_path / "upstream.json")
    # Parsing validates the real flag interface, not native GRIB execution.
    assert not (tmp_path / "prepared").exists()


def test_a_second_preparation_authority_refuses(tmp_path):
    handoff = {"argv": ["--source", "hrrr-prs", "--preprocess-workers", "48"]}
    with pytest.raises(ValueError, match="two authorities"):
        tp.preparation_arguments(handoff, wps=tmp_path / "namelist.wps", config=tmp_path / "experiment.toml",
            geog_root=tmp_path / "geog", bundle_root=tmp_path / "prepared", manifest_path=tmp_path / "manifest.json", workers=2)


def test_config_materialization_changes_only_shared_cache_path(tmp_path):
    source = tmp_path / "group.toml"
    source.write_text('[experiment]\nname="group"\n[shared]\nsf_surface_physics=3\n'
                      '[[domain]]\ngrid_id=1\nnx=288\n[[domain]]\ngrid_id=2\nnx=216\n'
                      '[fetch]\nsource="hrrr-prs"\ncycle="2026-10-03T00"\n'
                      '[static.highres]\nenabled=true\ncache_root="prior-relative-cache"\n', encoding="utf-8")
    original = source.read_bytes()
    target, receipt = tp.materialize_config(source, tmp_path / "authority", cache_root=tmp_path / "shared-cache")
    before = tomllib.loads(original.decode())
    after = tomllib.loads(target.read_text())
    assert after["static"]["highres"].pop("cache_root") == str((tmp_path / "shared-cache").resolve())
    before["static"]["highres"].pop("cache_root")
    assert after == before
    assert source.read_bytes() == original
    assert receipt["source_config_sha256"] == tp._sha(source)
    assert receipt["experiment_config_sha256"] == tp._sha(target)


def test_cpu_environment_retires_its_own_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_PHYSICS_PARAMS", "inherited")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with tp.cpu_environment(tmp_path / "tmp", 2):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["GPUWM_NO_LOCAL_GPU"] == "1"
        assert os.environ["RAYON_NUM_THREADS"] == "2"
        assert "GPUWM_PHYSICS_PARAMS" not in os.environ
    assert os.environ["GPUWM_PHYSICS_PARAMS"] == "inherited"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_cpu_limit_reuses_the_native_nested_cgroup_budget(monkeypatch):
    from gpuwm.ingest import preparation_workers
    monkeypatch.setattr(preparation_workers, "cpu_budget", lambda: {"available_cpus": 2})
    assert tp.cpu_limit() == 2
    assert tp.cpu_limit(48) == 2
    assert tp.cpu_limit(1) == 1
    with pytest.raises(ValueError, match="positive integer"):
        tp.cpu_limit(0)
