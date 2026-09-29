"""Picture admission must price the selected products, not an obsolete full set."""
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from gpuwm import disk_budget, runplan


@pytest.fixture(autouse=True)
def measured_catalog_fallback(monkeypatch):
    # Make the saved-size tests independent of the machine's renderer install.
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"products": None})


def projection(tmp_path, products, nx=36, ny=36):
    plan = SimpleNamespace(route="prepared", config_intent=None, run_dir=tmp_path,
                           run_options={"render_products": products})
    exp = SimpleNamespace(run_seconds=3600, restart_interval_s=0, domains=[
        SimpleNamespace(grid_id=1, history_interval_s=3600,
                        run=SimpleNamespace(nx=nx, ny=ny, nz=49))])
    return runplan._disk_projection(plan, exp, raw={}, data=None, fetch_arguments=None)


def test_one_product_is_not_charged_for_the_full_catalog(tmp_path):
    one = projection(tmp_path, "t2")
    all_products = projection(tmp_path, "all")
    assert 0 < one["picture_bytes"] < all_products["picture_bytes"] / 10
    free = one["total_bytes"] - one["picture_bytes"] + 10_000_000
    assert disk_budget.disk_refusal(one, free) is None
    assert disk_budget.disk_refusal(all_products, free) is not None


def test_picture_estimate_uses_horizontal_grid_size(tmp_path):
    small = projection(tmp_path, "all", 36, 36)
    large = projection(tmp_path, "all", 156, 124)
    assert small["picture_bytes"] < large["picture_bytes"]


def test_default_picture_estimate_matches_measured_small_grid(tmp_path):
    # Complete two-frame default inventories are retained in the fixture.
    predicted = projection(tmp_path, None)["picture_bytes"]
    assert 87_000_000 <= predicted <= 130_000_000


def test_projection_covers_complete_measured_runs_without_old_inflation(tmp_path):
    records = json.loads((Path(__file__).parent / "fixtures" /
                          "picture-disk-measurements.json").read_text())
    assert len(records) == 26
    for row in records:
        predicted = projection(tmp_path, "all", row["nx"], row["ny"])["picture_bytes"]
        assert row["bytes"] <= predicted <= 1.6 * row["bytes"], row


@pytest.mark.parametrize("spec", ["t2,t2", "t2,2m_temperature"])
def test_aliases_and_duplicates_are_not_charged_twice(tmp_path, spec):
    assert projection(tmp_path, spec)["picture_bytes"] == projection(tmp_path, "t2")["picture_bytes"]


def test_group_overlap_is_not_charged_twice(tmp_path):
    assert projection(tmp_path, "all,t2,direct")["picture_bytes"] == projection(tmp_path, "all")["picture_bytes"]
    assert projection(tmp_path, "direct,derived")["picture_bytes"] < projection(tmp_path, "all")["picture_bytes"]
    assert projection(tmp_path, "all,variables")["picture_bytes"] > projection(tmp_path, "all")["picture_bytes"]


def test_windows_are_priced_only_after_they_close():
    price = disk_budget.projected_picture_bytes
    assert price(36, 36, 3600, 3600, "qpf_6h") == 0
    assert price(36, 36, 6 * 3600, 3600, "qpf_6h") > 0
    assert price(36, 36, 0, 3600, "qpf_total") == 0
    assert price(36, 36, 3600, 900, "qpf_1h") == price(36, 36, 3600, 3600, "qpf_1h")
    assert price(36, 36, 3600, 900, "uh_2to5km_run_max") == price(
        36, 36, 3600, 3600, "uh_2to5km_run_max") > 0


def test_explicit_unknown_product_and_section_are_still_priced():
    price = disk_budget.projected_picture_bytes
    unknown = price(36, 36, 3600, 3600, "future_product")
    assert unknown > 0
    assert price(36, 36, 3600, 3600, "xsec:theta/levels=280,290,300") == unknown


def test_image_size_saturates_outside_measured_grid_brackets():
    price = disk_budget.projected_picture_bytes
    assert price(1000, 1000, 3600, 3600) == price(156, 124, 3600, 3600)
    assert price(1, 1, 3600, 3600) == price(36, 36, 3600, 3600)


def test_vertical_levels_do_not_price_pictures():
    domain = SimpleNamespace(grid_id=1, history_interval_s=3600,
                             run=SimpleNamespace(nx=36, ny=36, nz=20))
    exp = SimpleNamespace(run_seconds=3600, restart_interval_s=0, domains=[domain])
    before = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None,
                                              chain=None, render=True)
    domain.run.nz = 80
    after = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None,
                                             chain=None, render=True)
    assert before["picture_bytes"] == after["picture_bytes"]
    assert after["history_bytes"] == 4 * before["history_bytes"]


def test_new_catalog_fact_is_only_a_table_row(monkeypatch):
    import copy
    table = copy.deepcopy(disk_budget._picture_table())
    before = disk_budget.projected_picture_bytes(36, 36, 3600, 3600)
    table["products"]["future_product"] = ["direct", 0, 1000, 2000]
    monkeypatch.setattr(disk_budget, "_picture_table", lambda: table)
    assert disk_budget.projected_picture_bytes(36, 36, 3600, 3600) == before + int(2000 * table["headroom"])


def test_recipe_cache_key_includes_picture_measurements(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setenv("RECIPE_WORK", str(tmp_path / "work"))
    script = Path(__file__).parents[1] / "tools/wiki_seed/design_recipes.py"
    spec = importlib.util.spec_from_file_location("picture_recipe_key", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module._measured_key()
    table = tmp_path / "picture.json"
    table.write_bytes(disk_budget.picture_table_path().read_bytes() + b"\n")
    monkeypatch.setattr(disk_budget, "picture_table_path", lambda: table)
    assert module._measured_key() != before


def test_child_and_forecast_share_product_prices(tmp_path):
    from gpuwm.offline_child_run import child_cadence, child_disk_projection
    cfg = SimpleNamespace(nx=36, ny=36, nz=49, grid_id=2, dt=60,
                          run_seconds=3600, output_interval_s=900, restart_interval_s=3600)
    spec = "t2,xsec:wa=1,2,5@5,qpf_1h"
    child = child_disk_projection(cfg, child_cadence(cfg), keep_checkpoints=1,
                                 render_products=spec, outdir=tmp_path)
    assert child["pictures_per_frame"] == 3
    assert child["picture_bytes"] == disk_budget.projected_picture_bytes(36, 36, 3600, 900, spec)
    cfg.nx, cfg.ny = 156, 124
    larger = child_disk_projection(cfg, child_cadence(cfg), keep_checkpoints=1,
                                  render_products=spec, outdir=tmp_path)
    assert larger["picture_bytes"] > child["picture_bytes"]


def test_child_final_frame_off_cadence_is_charged(tmp_path):
    from gpuwm.offline_child_run import child_cadence, child_disk_projection
    cfg = SimpleNamespace(nx=36, ny=36, nz=49, grid_id=2, dt=60,
                          run_seconds=4 * 3600, output_interval_s=3 * 3600,
                          restart_interval_s=3600)
    child = child_disk_projection(cfg, child_cadence(cfg), keep_checkpoints=1,
                                 render_products="t2,qpf_total", outdir=tmp_path)
    assert child["domains"][0]["history_frames"] == 3
    regular = disk_budget.projected_picture_bytes(36, 36, 4 * 3600, 3 * 3600, "t2,qpf_total")
    assert child["picture_bytes"] > regular


def test_renderer_catalog_drives_both_selection_and_prices(monkeypatch):
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "group_keywords": ["all", "direct"], "local_run": {"products": [
            {"name": "2m_temperature", "kind": "direct", "minimum_hour": 0},
            {"name": "future_product", "kind": "direct", "minimum_hour": 0}]}})
    assert disk_budget.pictures_per_frame("all,t2") == 2
    price = disk_budget.projected_picture_bytes
    assert price(36, 36, 3600, 3600, "all,t2") == price(
        36, 36, 3600, 3600, "t2,future_product")
