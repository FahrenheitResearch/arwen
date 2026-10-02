"""WPS's bare 30s token uses its own table entries and records identity."""
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.namelist_import import import_namelists
from test_namelist_import import WPS_TEXT, _pair


def test_thirty_second_namelist_import_records_every_static_field(tmp_path):
    wps = WPS_TEXT.replace("'default', 'default'", "'30s', '30s'")
    text, report = import_namelists(*_pair(tmp_path, wps=wps))
    reference, _ = import_namelists(*_pair(tmp_path))
    assert text == reference
    notice = next(entry for entry in report.notices if "geog_data_res '30s'" in entry)
    assert "WPS 4.6 GEOGRID.TBL.ARW" in notice
    assert "d01:" in notice and "d02:" in notice
    for field, dataset in (
            ("terrain", "topo_gmted2010_30s"),
            ("landuse", "modis_landuse_20class_30s_with_lakes"),
            ("soil_top", "soiltype_top_30s"),
            ("soil_bottom", "soiltype_bot_30s"),
            ("greenfrac", "greenfrac_fpar_modis"),
            ("lai", "lai_modis_10m"),
            ("albedo", "albedo_modis"),
            ("snow_albedo", "maxsnowalb_modis"),
            ("soil_temperature", "soiltemp_1deg")):
        assert f"{field}={dataset}" in notice
    assert "no dataset substitution is applied" in report.format()


def test_thirty_second_notice_preserves_field_specific_token_order(tmp_path):
    wps = WPS_TEXT.replace("'default', 'default'", "'30s+5m', 'default'")
    _, report = import_namelists(*_pair(tmp_path, wps=wps))
    notice = next(entry for entry in report.notices if "geog_data_res '30s'" in entry)
    assert "d01:" in notice and "d02:" not in notice
    assert "terrain=topo_gmted2010_5m" in notice
    assert "soil_top=soiltype_top_30s" in notice


@pytest.mark.skipif(not os.environ.get("GPUWM_TEST_GEOG_ROOT"),
                    reason="staged WPS geography was not supplied")
def test_thirty_second_selector_builds_real_datasets_through_rust():
    from gpuwm.static import rust_bridge
    from gpuwm.static.build import GeogSelection, build_static
    from gpuwm.static.lambert import LambertGrid

    root = Path(os.environ["GPUWM_TEST_GEOG_ROOT"])
    assert rust_bridge.route("build_static") is rust_bridge
    selection = GeogSelection.from_tokens(root, "30s")
    grid = LambertGrid(ref_lat=-29.0, ref_lon=-51.0,
                       truelat1=-20.0, truelat2=-40.0, stand_lon=-51.0,
                       dx=12000.0, dy=12000.0, e_we=17, e_sn=17)
    coverage = {}
    fields = build_static(grid, root, selection=selection,
                          source_coverage_report=coverage)
    default = build_static(grid, root, selection=GeogSelection.fallback(root))
    assert set(coverage) == {
        "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
        "lai", "albedo", "snow_albedo", "soil_temperature"}
    assert fields.keys() == default.keys()
    for name, values in fields.items():
        assert np.isfinite(values).all(), name
        assert values.tobytes() == default[name].tobytes(), name
    assert fields["HGT_M"].shape == (16, 16)
    assert fields["LANDUSEF"].shape == (21, 16, 16)
    assert np.any(fields["HGT_M"] > 0.0)
