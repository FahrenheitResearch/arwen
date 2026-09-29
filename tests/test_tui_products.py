"""Terminal plot presets use real selectors without changing model output."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm import tui_products


def test_presets_use_supported_catalog_selectors_and_stored_snow_fields():
    document = tui_products.presets()
    assert document["schema"] == "gpuwm-tui-plot-presets-v1"
    assert document["default"] == "general"
    assert [row["id"] for row in document["presets"][:6]] == [
        "general", "tornado", "hurricane", "snow", "rain", "wind"]
    assert [row["id"] for row in document["presets"][6:]] == [
        "fire", "temperature", "aviation", "terrain", "coastal"]
    # 22: simulated_ir_satellite left this preset when the lane record
    # gained its reason, and 10m_wind_gusts and precipitation_type left it
    # when a run of the default preset was measured drawing 20 of 24:
    # no wrfout carries a gust or a precipitation category.  cloud_cover
    # (a total cloud fraction no wrfout carries) became cloud_cover_levels,
    # the layer panel the same history does draw.  A preset that promises
    # a product the lane can never draw is drift.
    assert len(document["presets"][0]["products"]) == 22
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "tools/rustwx/crates/rustwx-products/tests/fixtures/"
                           "product_catalog_inventory_v1.json").read_text())
    known = {name for lane in inventory["lanes"].values() for name in lane}
    for preset in document["presets"]:
        assert len(preset["products"]) == len(set(preset["products"]))
        assert preset["label"] and preset["description"]
        # Stored variables in STORE spelling: `var:SNOWH` named no stored
        # variable (the import stores it as wrf_snowh) and was dropped on
        # every run.
        assert set(preset["products"]) <= known | {"var:wrf_snowh",
                                                   "var:wrf_snow"}
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
assert len(presets()['presets'][0]['products']) == 22
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


def test_the_catalog_states_every_lane_unserved_preset_product_once(
        monkeypatch):
    """At plan review, with the recorded reason verbatim.

    No shipped preset names an unserved product any more, so the block is
    empty for every one of them; the statement itself is proven on a
    preset that does name one.
    """
    record = tui_products.lane_capabilities()
    block = tui_products.preset_availability(record)
    assert set(block) == {row["id"] for row in tui_products.presets()["presets"]}
    assert all(rows == {} for rows in block.values()), block
    monkeypatch.setattr(tui_products, "presets", lambda: {"presets": [
        {"id": "probe", "products": ["2m_temperature", "cloud_cover",
                                     "10m_wind_gusts"]}]})
    probe = tui_products.preset_availability(record)["probe"]
    for slug in ("cloud_cover", "10m_wind_gusts"):
        assert probe[slug] == record["unavailable"][slug]
    assert "2m_temperature" not in probe


def test_the_catalog_document_carries_the_statement_and_its_basis(monkeypatch):
    from gpuwm import bridges, runplan

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust", "products": [{"name": "2m_temperature"}]})
    document = tui_products.catalog_document()
    assert document["preset_availability"]["general"] == {}
    assert document["preset_availability_basis"]
    assert document["lane_record_sha256"] == tui_products.lane_capabilities()["sha256"]
    # The curated list still round-trips byte-identical: narrowing it
    # would hide the finding rather than report it.
    assert document["presets"] == tui_products.presets()


def test_the_picker_does_not_price_a_preset_against_a_fileless_plan(monkeypatch):
    """The plan is not the run.

    The renderer build's fileless requirement pair used to be folded
    in beside the packaged record here.  Measured on the shipped
    wheel against a real child, it called sixteen of the shipped snow
    preset's twenty-one products undrawable, 2m_temperature and
    500mb_height_winds among them, on the run that then drew 143
    pictures of exactly those products.  The same reading retired the
    same pair at the downscale door; this was its last caller, so the
    picker asks the renderer nothing but its catalog and the block
    says which measurement it rests on.
    """
    from gpuwm import runplan

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust",
                                 "products": [{"name": "2m_temperature"}]})
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("the picker launched the renderer"))
    document = tui_products.catalog_document()
    general = document["preset_availability"]["general"]
    # The general preset names nothing this lane cannot draw, and the
    # renderer was not launched to learn that.
    assert general == {}
    basis = document["preset_availability_basis"]
    assert basis == tui_products.PRESET_AVAILABILITY_BASIS
    assert "store catalog at render time" in basis


def test_the_block_never_narrows_the_curated_preset(monkeypatch):
    """A preset is a curated request, not a promise about one install."""
    from gpuwm import runplan

    monkeypatch.setattr(runplan, "render_catalog",
                        lambda: {"engine": "rust",
                                 "products": [{"name": "2m_temperature"}]})
    document = tui_products.catalog_document()
    assert document["presets"] == tui_products.presets()
    assert set(document["preset_availability"]) == {
        row["id"] for row in tui_products.presets()["presets"]}


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
