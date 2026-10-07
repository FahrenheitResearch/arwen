"""Optional renderer rows remain metadata and follow native recipe selectors."""
import json
from pathlib import Path
import re

from gpuwm.rustwx import CHEM_PRODUCT_ROWS, catalog_verdict

ROOT = Path(__file__).resolve().parents[1]


def test_chem_product_inventory_matches_native_sources_and_recipes():
    expected = (
        ("smoke_near_surface", "SMOKE_SFC", None, "ug/m^3"),
        ("smoke_column", "SMOKE_COLUMN", None, "mg/m^2"),
        ("pm25_near_surface", "PM2_5_DRY", 1, "ug/m^3"),
        ("aod_550", "AOD5502D", None, "1"),
        ("dust_near_surface", "DUST_SFC", None, "ug/m^3"),
        ("ozone_near_surface", "o3", 1, "ppb"),
    )
    assert CHEM_PRODUCT_ROWS == expected
    native = (ROOT / "tools/rustwx/crates/rw-wrfbatch/src/wrf_process.rs").read_text()
    rows = re.findall(r'ChemProductRow \{ source: "([^"]+)", also: &\[([^\]]*)\], '
                      r'store_name: "([^"]+)"', native)
    assert [(source, slug) for source, _, slug in rows] == [
        (source, slug) for slug, source, _, _ in expected]
    # The coupled fire's bulk smoke draws on the two smoke maps, in their
    # units, and on nothing else.
    also = {slug: re.findall(r'"([^"]+)"', extra) for _, extra, slug in rows}
    assert also == {"smoke_near_surface": ["SFIRE_SMOKE_SFC"],
                    "smoke_column": ["SFIRE_SMOKE_COLUMN"],
                    "pm25_near_surface": [], "aod_550": [],
                    "dust_near_surface": [], "ozone_near_surface": []}
    inventory = json.loads((ROOT / "tools/rustwx/crates/rustwx-products/tests/fixtures/"
                            "product_catalog_inventory_v1.json").read_text())
    assert {row[0] for row in expected} <= set(inventory["lanes"]["direct"])


def test_inventory_does_not_make_absent_sources_drawable():
    slugs = [row[0] for row in CHEM_PRODUCT_ROWS]
    absent = [(slug, "direct", "excluded", "missing source", "") for slug in slugs]
    assert catalog_verdict(absent, slugs) == ("", [(slug, "missing source") for slug in slugs])
    rows = [(slug, "direct", "renderable", "present source", "") for slug in slugs]
    assert catalog_verdict(rows, slugs)[1] == []
