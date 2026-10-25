"""The native HRRR root's 1 km child fetches its own static recipe."""

import json
from pathlib import Path

import pytest

from gpuwm.cli import main
from gpuwm.geog_assets import archive_for, geography_for_run
from gpuwm.go_cli import geography_refusal


RECIPE_DATASETS = {"lai_modis_30s", "bnu_soiltype_top", "bnu_soiltype_bot"}


@pytest.fixture
def native_root_and_child(tmp_path):
    recipe = (Path(__file__).resolve().parents[1] / "gpuwm/data/recipes/"
              "hrrr_v4_gsd41.toml")
    config = tmp_path / "nested.toml"
    config.write_text(recipe.read_text(encoding="utf-8") + """

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 850
j_parent_start = 500
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 150
ny = 150
specified = false
nested = true
history_interval_s = 900.0
# Child terrain drag is a separate unsupported route on this base.
gwd_opt = 0
""", encoding="utf-8")
    return config


def test_fetch_config_includes_native_sources_off_grid_child_datasets(
        native_root_and_child, capsys):
    assert main(["fetch-geog", "--config", str(native_root_and_child),
                 "--list"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert RECIPE_DATASETS <= set(plan["datasets"])
    assert "lai_modis_10m" not in plan["datasets"]
    assert "soilgrids" not in plan["datasets"]
    assert plan["static_sources"] == ["hrrr-conus-v4"]


def test_go_names_missing_off_grid_child_datasets(
        native_root_and_child, tmp_path, capsys, monkeypatch):
    from gpuwm import runplan

    def no_acquisition(*args, **kwargs):
        pytest.fail("go started acquisition before refusing missing geography")

    monkeypatch.setattr(runplan, "_staged_chain", no_acquisition)
    monkeypatch.setattr(runplan, "_hrrr_chain", no_acquisition)
    geog = tmp_path / "WPS_GEOG"
    # Stage everything else so only the reported run's missing recipe remains.
    for name in geography_for_run(native_root_and_child)["datasets"]:
        if name in RECIPE_DATASETS:
            continue
        for child in archive_for(name).index_subdirs or ("",):
            index = geog / name / child / "index"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text("", encoding="utf-8")
    refusal = geography_refusal(geog, config=native_root_and_child)
    assert refusal is not None
    for name in RECIPE_DATASETS:
        assert f"{name}/index" in refusal
    assert "before the fetch stage downloads" in refusal
    assert f"gpuwm fetch-geog --config {native_root_and_child}" in refusal
    outdir = tmp_path / "run"
    for mode in ("--dry-run", "--prepare-only"):
        assert main(["go", str(native_root_and_child), mode,
                     "--geog-root", str(geog), "--outdir", str(outdir)]) == 2
        err = capsys.readouterr().err
        for name in RECIPE_DATASETS:
            assert f"{name}/index" in err
        assert not outdir.exists()
    for name in RECIPE_DATASETS:
        index = geog / name / "index"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("", encoding="utf-8")
    assert geography_refusal(geog, config=native_root_and_child) is None
