"""Alternative WPS soil categories use the common static preparation path."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import geog_assets
from gpuwm.static.build import GeogSelection, build_static


def test_bnu_soil_selection_changes_only_top_and_bottom_soil(tmp_path):
    selection = GeogSelection.from_tokens(tmp_path, "bnu_soil_30s+default")
    baseline = GeogSelection.fallback(tmp_path)
    assert selection.soil_top == "bnu_soiltype_top"
    assert selection.soil_bottom == "bnu_soiltype_bot"
    for name in ("terrain", "landuse", "greenfrac", "lai", "albedo",
                 "snow_albedo", "soil_temperature"):
        assert selection.path(name) == baseline.path(name)
    # Each field uses the first token declaring it, as WPS does.
    first_default = GeogSelection.from_tokens(tmp_path, "default+bnu_soil_30s")
    assert first_default.soil_top == baseline.soil_top
    combined = GeogSelection.from_tokens(tmp_path, "bnu_soil_30s+5m")
    assert combined.terrain == "topo_gmted2010_5m"
    assert combined.soil_top == "bnu_soiltype_top"


def test_bnu_soil_selection_is_independent_per_domain(tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n max_dom=2,\n/\n&geogrid\n"
                   " geog_data_res='bnu_soil_30s+default',\n/\n")
    data = SimpleNamespace(geog_root=tmp_path, wps_namelist=wps)
    assert GeogSelection.from_case_data(data, 1).soil_top == "bnu_soiltype_top"
    assert GeogSelection.from_case_data(data, 2).soil_top == "soiltype_top_30s"


def test_bnu_soil_archives_are_optional_pinned_upstream_rows():
    expected = {
        "bnu_soiltype_top": (8198836, 933120270,
            "7a7eb86d585c3dc6b32297f1ea4622d929aacadc70f16e14e0193a06f233038a"),
        "bnu_soiltype_bot": (8078617, 933120273,
            "49306298749e3ed2a6172dfe7af0023a5be0e9f5f5230a2cfd6421ebce3e81b1"),
    }
    for name, (size, unpacked, digest) in expected.items():
        row = geog_assets.archive_for(name)
        assert row.archive_bytes == size
        assert row.archive_sha256 == digest
        assert row.extracted_bytes == unpacked
        assert row.required_by == ()
        assert row.optional_for == (geog_assets.GEOG_CONSUMER_WRF,)
        assert row.available_sources == ("ncar",)
        assert not row.in_mandatory_bundle
        assert name not in geog_assets.parse_datasets("wrf")
        assert name not in geog_assets.MANDATORY_BUNDLE_DATASETS


def test_highres_can_retain_selected_wps_soil_without_changing_default_echo(tmp_path):
    from gpuwm.static.highres_production import parse_static_table
    raw = {"highres": {"enabled": True, "cache_root": str(tmp_path),
                        "fields": "all", "landcover_source": "annual-nlcd"}}
    default = parse_static_table(raw, source="test", base_dir=tmp_path)
    assert default.soil_source == "soilgrids"
    assert "soil_source" not in default.echo()
    raw["highres"]["soil_source"] = "wps-geog"
    configured = parse_static_table(raw, source="test", base_dir=tmp_path)
    assert configured.soil_source == "wps-geog"
    assert configured.echo()["soil_source"] == "wps-geog"
    from gpuwm.static.highres_production import (
        parse_sealed_static_highres, prepared_highres_settings_match,
        static_highres_identity)
    restored = parse_sealed_static_highres(static_highres_identity(configured),
        source="prepared", base_dir=tmp_path)
    assert restored.soil_source == "wps-geog"
    assert prepared_highres_settings_match(configured.echo(), restored)
    assert not prepared_highres_settings_match(configured.echo(), default)
    raw["highres"]["soil_source"] = "unknown"
    with pytest.raises(ValueError, match="soil_source"):
        parse_static_table(raw, source="test", base_dir=tmp_path)


def test_selected_wps_soil_does_not_fetch_soilgrids(tmp_path, monkeypatch):
    from datetime import date
    from gpuwm.static import highres_production as production
    from gpuwm.static.highres_fetch import CoverageError, FootprintBBox

    monkeypatch.setattr(production, "_fetch_terrain",
        lambda *args, **kwargs: (None, {"terrain_bytes_fetched": 0}))
    raster = SimpleNamespace(receipt=lambda: {"source": "landcover"})
    monkeypatch.setattr(production, "fetch_landcover",
        lambda *args, **kwargs: ([], raster))
    def outside(*args, **kwargs):
        raise CoverageError("outside land-cover raster")
    monkeypatch.setattr(production, "derive_landcover_window", outside)
    def unexpected(*args, **kwargs):
        raise AssertionError("selected WPS soil must not download SoilGrids")
    monkeypatch.setattr(production, "fetch_soilgrids", unexpected)
    _, _, soil, manifest = production._fetch_and_bind(
        FootprintBBox(37.5, 38., -123., -122.), tmp_path, date(2026, 10, 3),
        coverage=SimpleNamespace(), soil_source="wps-geog")
    assert soil == {} and manifest["soilgrids_windows"] == {}
    assert manifest["bytes_fetched"] == 0


@pytest.mark.skipif(not os.environ.get("GPUWM_TEST_BNU_GEOG_ROOT"),
                    reason="rebuilt Rust static-fields bridge not supplied")
def test_selected_wps_soil_helper_retains_fraction_bits_and_reconciles_water():
    from gpuwm.static.highres import selected_geog_soil
    fraction = np.zeros((16, 1, 3))
    fraction[1, 0, 0] = 0.20000000298023224
    fraction[2, 0, 0] = 0.7999999970197678
    fraction[13, 0, 1:] = 1.
    fields, audit = selected_geog_soil({"SOILCTOP": fraction, "SOILCBOT": fraction},
                                      np.array([[1., 1., 0.]]))
    assert fields["SOILCTOP"][:, :, 0].tobytes() == fraction[:, :, 0].tobytes()
    assert fields["SOILCTOP"][:, :, 1].tobytes() == fraction[:, :, 0].tobytes()
    assert fields["SCT_DOM"].tolist() == [[3., 3., 14.]]
    assert audit["top_0_30cm"]["water_soil_land_cells_from_nearest_land"] == 1


@pytest.mark.skipif(not os.environ.get("GPUWM_TEST_BNU_GEOG_ROOT"),
                    reason="staged BNU and baseline WPS geography not supplied")
@pytest.mark.parametrize("spacing", [2250.0, 750.0])
def test_real_bnu_tiles_build_through_rust_and_preserve_other_fields(spacing):
    from gpuwm.static import rust_bridge
    from gpuwm.static.lambert import LambertGrid

    root = Path(os.environ["GPUWM_TEST_BNU_GEOG_ROOT"])
    assert rust_bridge.route("build_static") is rust_bridge
    grid = LambertGrid(ref_lat=37.75, ref_lon=-122.2,
                       truelat1=30.0, truelat2=60.0, stand_lon=-122.2,
                       dx=spacing, dy=spacing, e_we=33, e_sn=33)
    baseline = build_static(grid, root, selection=GeogSelection.fallback(root))
    coverage = {}
    candidate = build_static(
        grid, root, selection=GeogSelection.from_tokens(root, "bnu_soil_30s"),
        source_coverage_report=coverage)
    assert candidate.keys() == baseline.keys()
    soil = {"SOILCTOP", "SOILCBOT", "SCT_DOM", "SCB_DOM"}
    for name in candidate.keys() - soil:
        assert candidate[name].tobytes() == baseline[name].tobytes(), name
    for fractions, dominant in (("SOILCTOP", "SCT_DOM"),
                               ("SOILCBOT", "SCB_DOM")):
        assert candidate[fractions].shape == (16, 32, 32)
        assert np.isfinite(candidate[fractions]).all()
        assert np.all((candidate[dominant] >= 1) & (candidate[dominant] <= 16))
        np.testing.assert_allclose(candidate[fractions].sum(axis=0), 1., atol=1.e-7)
        assert np.any(candidate[dominant] != baseline[dominant])
    assert set(coverage) == {"terrain", "landuse", "soil_top", "soil_bottom",
                             "greenfrac", "lai", "albedo", "snow_albedo",
                             "soil_temperature"}
