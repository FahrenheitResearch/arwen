"""Built-area estimates exercise the compiled Rust static-fields seam."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static import rust_bridge
from gpuwm.static.highres_production import HighresStaticConfig, static_highres_identity
from gpuwm.static.urban_fraction import (
    URBAN_FRACTION_ALGORITHM, NLCD_FRACTION_ALGORITHM,
    NLCD_IMPERVIOUS_CLASS_MIDPOINTS, estimate_urban_fraction,
)


def _fields():
    fractions = np.zeros((61, 1, 5), np.float64)
    # pure low intensity, mixed high intensity, mixed rural, water, two urban types
    fractions[50, 0, 0] = 1.0
    fractions[52, 0, 1] = 0.7
    fractions[9, 0, 1] = 0.3
    fractions[52, 0, 2] = 0.3
    fractions[9, 0, 2] = 0.7
    fractions[16, 0, 3] = 1.0
    fractions[50, 0, 4] = 0.3
    fractions[52, 0, 4] = 0.7
    return {"LANDUSEF": fractions,
            "LU_INDEX": np.array([[51, 53, 10, 17, 53]], np.float64),
            "LANDMASK": np.array([[1, 1, 1, 0, 1]], np.float64)}


@pytest.fixture
def compiled_bridge():
    if rust_bridge.unavailable_reason() is not None:
        pytest.skip("the static-fields bridge is not built on this host")
    assert hasattr(rust_bridge.load(), "gpuwm_static_highres_urban_fraction")


def test_mixed_nlcd_cover_uses_each_type_and_dominant_mask(compiled_bridge):
    from gpuwm.core.urban_tables import load_urban_params
    low, _, high = map(float, load_urban_params(1, 0).FRC_URB_TBL)
    fields = _fields()
    before = {name: values.tobytes() for name, values in fields.items()}
    fraction, audit = estimate_urban_fraction(fields, option=1, use_wudapt_lcz=0)
    np.testing.assert_array_equal(fraction, [[low, 0.7 * high, 0, 0,
                                             0.3 * low + 0.7 * high]])
    assert audit["dominant_urban_cells"] == 3
    assert audit["urban_cover_outside_dominant_cells"] == 1
    assert "not observed imperviousness" in audit["interpretation"]
    assert audit["algorithm"] == URBAN_FRACTION_ALGORITHM
    assert before == {name: values.tobytes() for name, values in fields.items()}


def test_lcz_uses_its_own_urbparm_rows(compiled_bridge):
    from gpuwm.core.urban_tables import load_urban_params
    params = load_urban_params(1, 1)
    fraction, _ = estimate_urban_fraction(_fields(), option=1, use_wudapt_lcz=1)
    assert fraction[0, 0] == float(params.FRC_URB_TBL[0])
    assert fraction[0, 1] == 0.7 * float(params.FRC_URB_TBL[2])


def test_nlcd_source_intensities_replace_table_built_area_only(compiled_bridge):
    fields = _fields()
    source = np.zeros((5, 1, 5))
    source[1, 0, 0] = 1.0  # open space is distinct from low density morphology
    source[3, 0, 1] = 0.7  # medium developed, 50-79% impervious
    source[0, 0, 1] = 0.3
    source[4, 0, 2] = 1.0  # rural dominant mask remains authoritative
    source[4, 0, 3] = 1.0  # water mask remains authoritative
    source[2, 0, 4] = 0.3
    source[4, 0, 4] = 0.7
    weights = np.ones((1, 5))
    before = {name: value.tobytes() for name, value in fields.items()}
    fraction, audit = estimate_urban_fraction(fields, option=1,
        use_wudapt_lcz=0, source_fractions=source, source_weight=weights)
    np.testing.assert_array_equal(fraction,
        [[.10, .7 * .645, 0, 0, .3 * .345 + .7 * .90]])
    assert audit["algorithm"] == NLCD_FRACTION_ALGORITHM
    assert audit["source_covered_urban_cells"] == 3
    assert list(NLCD_IMPERVIOUS_CLASS_MIDPOINTS) == [21, 22, 23, 24]
    assert before == {name: value.tobytes() for name, value in fields.items()}


def test_nlcd_fraction_identity_cannot_reuse_table_weighted_preparation():
    config = HighresStaticConfig(True, Path.cwd() / "cache", sf_urban_physics=1,
                                landcover_source="annual-nlcd")
    assert config.echo()["urban_fraction"] == NLCD_FRACTION_ALGORITHM
    assert replace(config, landcover_source="cglc-modis-lcz").echo()["urban_fraction"] == URBAN_FRACTION_ALGORITHM
    from gpuwm.static.highres_production import parse_sealed_static_highres
    restored = parse_sealed_static_highres(static_highres_identity(config), source="test", base_dir=Path("."))
    assert restored.echo() == config.echo()
    corrupted = {**static_highres_identity(config), "urban_fraction": URBAN_FRACTION_ALGORITHM}
    with pytest.raises(ValueError, match="sealed built area"):
        parse_sealed_static_highres(corrupted, source="test", base_dir=Path("."))


def test_real_legacy_bridge_cannot_silently_ignore_source_classes(monkeypatch):
    import ctypes
    import os
    old_path = os.environ.get("GPUWM_TEST_OLD_URBAN_BRIDGE")
    if not old_path:
        pytest.skip("the prior compiled urban bridge is not supplied")
    old = ctypes.CDLL(old_path)
    assert hasattr(old, "gpuwm_static_highres_urban_fraction")
    assert not hasattr(old, "gpuwm_static_highres_urban_fraction_source_v1")
    monkeypatch.setattr(rust_bridge, "load", lambda: old)
    with pytest.raises(rust_bridge.StaticBridgeError, match="source-class urban fractions"):
        estimate_urban_fraction(_fields(), option=1, use_wudapt_lcz=0,
            source_fractions=np.ones((5, 1, 5)), source_weight=np.ones((1, 5)))


def test_wrong_algorithm_receipt_is_rejected_after_actual_bridge_execution(
        compiled_bridge, monkeypatch):
    monkeypatch.setattr(rust_bridge, "highres_audit_json",
                        lambda handle: {"algorithm": "other-algorithm"})
    with pytest.raises(rust_bridge.StaticBridgeError, match="returned algorithm"):
        estimate_urban_fraction(_fields(), option=1, use_wudapt_lcz=0)


def test_corrupt_cover_cannot_silently_default_to_a_whole_cell(compiled_bridge):
    fields = _fields()
    fields["LANDUSEF"][:, 0, 0] = 0.0
    with pytest.raises(rust_bridge.StaticBridgeError, match="silently fall back"):
        estimate_urban_fraction(fields, option=1, use_wudapt_lcz=0)


def test_off_identity_keeps_every_old_operand_and_no_fraction_key():
    off = HighresStaticConfig(True, Path("cache"))
    assert static_highres_identity(off) == {
        "enabled": True, "cache_root": "cache", "on_refuse": "error",
        "terrain_source": "auto", "fields": "auto", "landcover_source": "auto"}
    assert "urban_fraction" not in off.echo()
    assert "urban_fraction" not in replace(off, sf_urban_physics=1,
                                          fields="terrain").echo()


def test_custom_nonurban_inventory_keeps_its_existing_no_canopy_path(monkeypatch):
    from test_static_highres_landcover_sources import (
        _stub_categories, _Stub, _noah_baseline, _grid, N,
    )
    from gpuwm.static.highres import (
        build_highres_overrides, baseline_ocean_mask,
        CGLC_MODIS_LCZ_TO_MODIS21,
    )
    _stub_categories(monkeypatch)
    baseline = _noah_baseline()
    fields, audit = build_highres_overrides(
        _grid(40.0, -100.0, dx=1000.0, n=N + 1),
        terrain=None, landcover=_Stub("synthetic"),
        soil_sources={("sand", "0-5cm"): _Stub("soil")},
        baseline_ocean=baseline_ocean_mask(baseline), baseline=baseline,
        landcover_mapping=CGLC_MODIS_LCZ_TO_MODIS21, category_count=27)
    assert fields["LANDUSEF"].shape == (27, N, N)
    assert "FRC_URB2D" not in fields
    assert "urban_fraction" not in audit


@pytest.mark.parametrize("option", [2, 3])
def test_bep_bem_retained_legends_keep_bytes_and_prior_fraction_behavior(
        monkeypatch, option):
    import json
    from test_static_highres_landcover_sources import (
        _stub_categories, _Stub, _noah_baseline, _grid, N,
    )
    from gpuwm.static.highres import (
        build_highres_overrides, baseline_ocean_mask, landcover_legend,
    )
    _stub_categories(monkeypatch)
    baseline = _noah_baseline()
    mapping, count = landcover_legend("annual-nlcd", sf_urban_physics=option)
    slucm, _ = landcover_legend("annual-nlcd", sf_urban_physics=1)
    assert json.dumps(mapping).encode() == json.dumps(slucm).encode()
    assert type(mapping) is dict
    fields, audit = build_highres_overrides(
        _grid(40.0, -100.0, dx=1000.0, n=N + 1),
        terrain=None, landcover=_Stub("annual-nlcd-2024"),
        soil_sources={("sand", "0-5cm"): _Stub("soil")},
        baseline_ocean=baseline_ocean_mask(baseline), baseline=baseline,
        landcover_mapping=mapping, category_count=count)
    assert fields["LANDUSEF"].shape == (61, N, N)
    assert "FRC_URB2D" not in fields
    assert "urban_fraction" not in audit


def test_an_explicit_fraction_request_cannot_lose_its_table_binding(monkeypatch):
    from test_static_highres_landcover_sources import (
        _stub_categories, _Stub, _noah_baseline, _grid, N,
    )
    from gpuwm.static.highres import (
        build_highres_overrides, baseline_ocean_mask, landcover_legend,
    )
    _stub_categories(monkeypatch)
    baseline = _noah_baseline()
    mapping, count = landcover_legend("annual-nlcd", sf_urban_physics=1)
    mapping[21] = 52  # Requested estimate, with its class-table binding corrupted.
    with pytest.raises(ValueError, match="requested SLUCM area fraction"):
        build_highres_overrides(
            _grid(40.0, -100.0, dx=1000.0, n=N + 1),
            terrain=None, landcover=_Stub("annual-nlcd-2024"),
            soil_sources={("sand", "0-5cm"): _Stub("soil")},
            baseline_ocean=baseline_ocean_mask(baseline), baseline=baseline,
            landcover_mapping=mapping, category_count=count)


def test_real_highres_overlay_publishes_fraction_with_its_estimate_receipt(
        compiled_bridge, tmp_path, monkeypatch):
    from datetime import date
    from test_static_highres_landcover_sources import (
        _stub_categories, _stub_fetch, _noah_baseline, _grid, N, MODIS21_ATTRS,
    )
    from gpuwm.static.highres_production import apply_highres_statics
    from gpuwm.static import highres
    from gpuwm.core.urban_tables import load_urban_params
    _stub_categories(monkeypatch)
    _stub_fetch(monkeypatch, {}, pixels={"61": 144})
    def categories(source, grid, mapping, *, category_count):
        fractions = np.zeros((category_count, grid.e_sn - 1, grid.e_we - 1))
        fractions[mapping[61] - 1] = 1.0
        return fractions
    monkeypatch.setattr(highres, "resample_mapped_categories", categories)
    config = HighresStaticConfig(True, tmp_path, sf_urban_physics=1, use_wudapt_lcz=1)
    fields, receipt = apply_highres_statics(_noah_baseline(), _grid(40.0, -100.0, dx=1000.0, n=N + 1),
        config=config, domain_id=1, case_date=date(2026, 7, 1), landuse_attrs=MODIS21_ATTRS)
    np.testing.assert_array_equal(fields["FRC_URB2D"], np.full((N, N),
        float(load_urban_params(1, 1).FRC_URB_TBL[10])))
    assert receipt["override_audit"]["urban_fraction"]["algorithm"] == URBAN_FRACTION_ALGORITHM


def test_native_static_cache_roundtrip_keeps_built_area_bytes(tmp_path):
    from test_native_wrf_contract import _grid, _complete_native_static
    from gpuwm.native_wrf_contract import (
        native_static_export_fields, write_native_static_cache, load_native_static_cache,
    )
    grid = _grid()
    fields = _complete_native_static(grid)
    fields["FRC_URB2D"] = np.array([[0, .25, .5, .9], [0, .4, .8, 1], [0, 0, 0, 0]], np.float64)
    first = tmp_path / "native-static.npz"
    first_receipt = write_native_static_cache(first, native_static_export_fields(fields, grid))
    restored = load_native_static_cache(first, grid, 3, 4)
    assert restored["FRC_URB2D"].tobytes() == fields["FRC_URB2D"].tobytes()
    second = tmp_path / "republished.npz"
    second_receipt = write_native_static_cache(second, native_static_export_fields(restored, grid))
    assert first_receipt["sha256"] == second_receipt["sha256"]


@pytest.mark.parametrize("bad", [np.zeros((2, 4)), np.full((3, 4), -0.1),
                                 np.full((3, 4), 1.1), np.full((3, 4), np.nan)])
def test_native_static_rejects_invalid_built_area(bad):
    from test_native_wrf_contract import _grid, _complete_native_static
    from gpuwm.native_wrf_contract import validate_native_static_fields
    grid = _grid()
    fields = _complete_native_static(grid)
    fields["FRC_URB2D"] = bad
    with pytest.raises(ValueError, match="FRC_URB2D"):
        validate_native_static_fields(fields, grid, 3, 4)
