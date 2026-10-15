"""A native-grid recipe must acquire its declared static field source."""
from pathlib import Path

from gpuwm.geog_assets import geography_for_run


def test_source_recipe_acquires_complete_statics_without_unpinned_drag():
    recipe = Path(__file__).resolve().parents[1] / "configs/recipes/hrrr_v4_gsd41.toml"
    plan = geography_for_run(recipe)
    assert plan["static_sources"] == ["hrrr-conus-v4"]
    assert plan["datasets"] == []
    assert plan["static_bytes"] == 1112724420


def test_source_recipe_dry_run_does_not_require_unused_geography(tmp_path):
    from gpuwm.cli import main

    recipe = Path(__file__).resolve().parents[1] / "configs/recipes/hrrr_v4_gsd41.toml"
    assert main(["go", str(recipe), "--dry-run", "--products", "none",
                 "--geog-root", str(tmp_path / "absent-geography"),
                 "--outdir", str(tmp_path / "run")]) == 0
