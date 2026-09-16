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
    # 24, not 25: simulated_ir_satellite left this preset and the
    # hurricane one when the lane record gained its reason.  A preset
    # that promises a product the lane can never draw is drift.
    assert len(document["presets"][0]["products"]) == 24
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
assert len(presets()['presets'][0]['products']) == 24
assert 'cupy' not in sys.modules
assert 'gpuwm.runplan' not in sys.modules
assert 'gpuwm.prepared_single_domain_forecast' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_preset_ships_a_product_this_lane_can_never_fill():
    """A preset that promises what the lane cannot draw is drift."""
    for preset in tui_products.presets()["presets"]:
        assert "simulated_ir_satellite" not in preset["products"], preset["id"]
    record = tui_products.lane_capabilities()
    reason = record["unavailable"]["simulated_ir_satellite"]
    assert "radiative-transfer operator" in reason
    # The alternative on this lane is named, and named as a DIFFERENT
    # quantity: asserting equivalence would repeat the wrong-quantity
    # defect this tree has already paid for.
    assert "var:wrf_olr" in reason
    assert "W m-2" in reason and "not a brightness temperature" in reason.lower()
    # No model is named: lane rule 7 bans those tokens in generic code,
    # and the sentence has to stand on the lane, not on a source.
    assert "GRIB" in reason


def test_the_catalog_states_every_lane_unserved_preset_product_once():
    """At plan review, with the recorded reason verbatim."""
    record = tui_products.lane_capabilities()
    block = tui_products.preset_availability(record)
    assert set(block) == {row["id"] for row in tui_products.presets()["presets"]}
    for slug in ("cloud_cover", "10m_wind_gusts", "precipitation_type"):
        assert block["general"][slug] == record["unavailable"][slug]
    assert "2m_temperature" not in block["general"]
    assert "composite_reflectivity" not in block["general"]


def test_the_catalog_document_carries_the_statement_and_its_basis(monkeypatch):
    from gpuwm import bridges, runplan

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust", "products": [{"name": "2m_temperature"}]})
    document = tui_products.catalog_document()
    assert "10m_wind_gusts" in document["preset_availability"]["general"]
    assert document["preset_availability_basis"]
    assert document["lane_record_sha256"] == tui_products.lane_capabilities()["sha256"]
    # The curated list still round-trips byte-identical: narrowing it
    # would hide the finding rather than report it.
    assert document["presets"] == tui_products.presets()


def test_the_renderers_own_requirement_rows_name_the_missing_selector(monkeypatch):
    from gpuwm import runplan, rustwx

    requirements = rustwx.parse_catalog_requirements("\n".join([
        "NEEDS\t2m_temperature\ttemperature_2m_agl",
        "NEEDS\t10m_wind_gusts\twind_gust_10m_agl",
        "PLANNED\ttemperature_2m_agl",
    ]))
    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust", "products": [{"name": "2m_temperature"}]})
    monkeypatch.setattr(rustwx, "catalog_requirements", lambda *a, **k: requirements)
    document = tui_products.catalog_document()
    general = document["preset_availability"]["general"]
    assert "wind_gust_10m_agl" in general["10m_wind_gusts"]
    assert rustwx.REQUIREMENTS_BASIS in general["10m_wind_gusts"]
    assert "2m_temperature" not in general
    assert rustwx.REQUIREMENTS_BASIS in document["preset_availability_basis"]


def test_a_renderer_that_cannot_be_asked_states_its_basis_and_does_not_refuse(monkeypatch):
    """Unmeasured is not impossible: the block stays, saying what it knows."""
    from gpuwm import runplan, rustwx

    def raising(*args, **kwargs):
        raise RuntimeError("the renderer could not be launched")

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust", "products": [{"name": "2m_temperature"}]})
    monkeypatch.setattr(rustwx, "catalog_requirements", raising)
    document = tui_products.catalog_document()
    assert "preset_availability" in document
    assert document["preset_availability"]["general"] == {
        slug: tui_products.lane_capabilities()["unavailable"][slug]
        for slug in ("10m_wind_gusts", "precipitation_type", "cloud_cover")}
    basis = document["preset_availability_basis"]
    assert "could not be asked" in basis and "packaged lane record only" in basis
    assert document["presets"] == tui_products.presets()


def test_the_catalog_lists_the_observation_grid_lane_as_its_own(monkeypatch):
    """Five engine products nothing in this tree listed anywhere."""
    from gpuwm import rustwx_lanes, runplan

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust", "products": [{"name": "2m_temperature"}]})
    lanes = tui_products.catalog_document()["lanes"]
    assert set(lanes["observation-grid"]["products"]) == set(
        rustwx_lanes.OBSGRID_PRODUCTS)
    # Named as a lane of its own, with what it takes: these five would
    # refuse on every wrfout frame, so they are not in that catalog.
    assert "wrfout" in lanes["observation-grid"]["input"]
    assert lanes["observation-grid"]["engine"] == rustwx_lanes.OBSGRID_NAME
