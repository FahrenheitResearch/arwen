"""Offline admission includes the newly drawable products with real byte receipts."""
import json
from pathlib import Path

import pytest

from gpuwm import disk_budget, runplan

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = (
    "10m_wind_gusts", "categorical_rain", "categorical_freezing_rain",
    "categorical_ice_pellets", "categorical_snow", "precipitation_type",
    "snowfall_window", "snow_10to1_window", "snow_kuchera_window",
)


@pytest.mark.parametrize("product", PRODUCTS)
def test_offline_all_prices_every_new_drawable_product_from_receipts(monkeypatch, product):
    record = json.loads((ROOT / "tools/wiki_seed/runs/bytes-per-product/sizes.json").read_text())
    table = disk_budget._picture_table()
    assert product in table["products"], f"offline all silently omits {product}"
    receipts = [row for row in record["pictures"] if row[1] == product]
    assert {tuple(row[2]) for row in receipts} >= {(36, 36), (156, 124)}
    assert all(row[0] in record["runs"] and row[4] > 0 for row in receipts)
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"products": None})
    offline = disk_budget._picture_products("all", run_hours=6, measured=True)
    assert product in offline
    before = disk_budget.projected_picture_bytes(36, 36, 6 * 3600, 3600, "all")
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"local_run": {"products": [
        {"name": name, "kind": row[0], "minimum_hour": row[1]}
        for name, row in table["products"].items() if row[0] != "generic"]}})
    assert disk_budget.projected_picture_bytes(36, 36, 6 * 3600, 3600, "all") == before
