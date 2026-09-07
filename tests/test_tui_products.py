"""Terminal plot presets use real selectors without changing model output."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

from gpuwm import tui_products


def test_presets_use_supported_catalog_selectors_and_stored_snow_fields():
    document = tui_products.presets()
    assert document["schema"] == "gpuwm-tui-plot-presets-v1"
    assert document["default"] == "general"
    assert [row["id"] for row in document["presets"][:6]] == [
        "general", "tornado", "hurricane", "snow", "rain", "wind"]
    assert [row["id"] for row in document["presets"][6:]] == [
        "fire", "temperature", "aviation", "terrain", "coastal"]
    assert len(document["presets"][0]["products"]) == 25
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "tools/rustwx/crates/rustwx-products/tests/fixtures/"
                           "product_catalog_inventory_v1.json").read_text())
    known = {name for lane in inventory["lanes"].values() for name in lane}
    for preset in document["presets"]:
        assert len(preset["products"]) == len(set(preset["products"]))
        assert preset["label"] and preset["description"]
        assert set(preset["products"]) <= known | {"var:SNOWH", "var:SNOW"}
    # New task presets use canonical engine selectors only. The stored-field
    # escape hatch remains limited to the existing snow request above.
    for preset in document["presets"][6:]:
        assert set(preset["products"]) <= known
    assert len({row["id"] for row in document["presets"]}) == len(document["presets"])
    from gpuwm.io.history_selection import HISTORY_VOCABULARY
    assert {"SNOWH", "SNOW"} <= HISTORY_VOCABULARY


def test_catalog_queries_the_existing_engine_only_in_inspection_mode(monkeypatch):
    from gpuwm import bridges, runplan
    entered = []

    @contextmanager
    def inspection():
        entered.append(True)
        try:
            yield
        finally:
            entered.pop()

    def catalog():
        assert entered == [True]
        return {"engine": "rust", "products": [{"name": "real_engine_selector"}]}

    monkeypatch.setattr(bridges, "inspection_only", inspection)
    monkeypatch.setattr(runplan, "render_catalog", catalog)
    result = tui_products.catalog_document()
    assert result["products"] == [{"name": "real_engine_selector"}]
    assert result["presets"] == tui_products.presets()
    assert entered == []


def test_catalog_refusal_is_preserved_without_a_substitute_product_list(monkeypatch):
    from gpuwm import runplan
    refusal = {"engine": None, "products": None, "error": "Stage the native renderer"}
    monkeypatch.setattr(runplan, "render_catalog", lambda: refusal)
    result = tui_products.catalog_document()
    assert {key: result[key] for key in refusal} == refusal


def test_reading_presets_imports_no_forecast_or_gpu_runtime():
    code = """
import sys
from gpuwm.tui_products import presets
assert len(presets()['presets'][0]['products']) == 25
assert 'cupy' not in sys.modules
assert 'gpuwm.runplan' not in sys.modules
assert 'gpuwm.prepared_single_domain_forecast' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
