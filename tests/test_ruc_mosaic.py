"""WRF column oracles pin mixture order and the live irrigation state update."""

from dataclasses import fields
import hashlib
import numpy as np
import pytest

from gpuwm.core.ruc import ruc_surface_parameters, ruc_land_surface_step
from ruc_mosaic_fixture import surface_cases, driver_calls


def test_mosaic_oracle_bytes_match_the_recorded_fortran_build():
    from gpuwm.core.ruc_contract import (RUC_MOSAIC_DRIVER_ORACLE_ASSET,
                                         RUC_MOSAIC_SURFACE_ORACLE_ASSET)
    from ruc_mosaic_fixture import ORACLE
    for asset in (RUC_MOSAIC_SURFACE_ORACLE_ASSET, RUC_MOSAIC_DRIVER_ORACLE_ASSET):
        payload = (ORACLE / asset.relative_path).read_bytes()
        assert len(payload) == asset.bytes
        assert hashlib.sha256(payload).hexdigest() == asset.sha256


def test_surface_mosaics_match_unmodified_wrf_bits():
    # Prevent normalized roughness, water-soil inclusion, or averaging forest
    # classes from silently replacing WRF's specific mixture semantics.
    for row, inputs, keywords in surface_cases():
        result = ruc_surface_parameters(*inputs, **keywords)
        for descriptor in fields(result):
            name = descriptor.name
            dtype = np.int32 if name == "iforest" else np.float32
            expected = np.array([float(row[name])], dtype=dtype)
            np.testing.assert_array_equal(getattr(result, name).view(np.uint32),
                                          expected.view(np.uint32), err_msg=f"{row['case']}:{name}")


def test_full_mosaic_columns_match_unmodified_wrf_bits():
    # Compare every returned state/flux carrier over two steps, for both tables.
    for label, values, keywords, expected in driver_calls():
        result = ruc_land_surface_step(values, **keywords)
        for name, reference in expected.items():
            np.testing.assert_array_equal(getattr(result, name).view(np.uint32),
                                          reference.view(np.uint32), err_msg=f"{label}:{name}")


def test_irrigation_changes_only_active_root_columns_and_is_not_ignored():
    label, values, keywords, expected = next(driver_calls())
    result = ruc_land_surface_step(values, **keywords)
    # The first column irrigates; exactly 0.75 greenness and the no-cover
    # column do not. The deep reservoir remains below the irrigated layer.
    assert result.soilmois[0, 0] > values["soilmois"][0, 0]
    assert result.soilmois[0, 1] < result.soilmois[0, 0]
    assert result.soilmois[-1, 0] < result.soilmois[0, 0]


@pytest.mark.parametrize("field", ["landusef", "soilctop"])
def test_missing_fraction_source_refuses_instead_of_inventing_cover(field):
    row, inputs, keywords = next(surface_cases())
    keywords[field] = None
    with pytest.raises(ValueError, match=f"requires {field}"):
        ruc_surface_parameters(*inputs, **keywords)


def test_zero_land_area_refuses_before_mosaic_division():
    row, inputs, keywords = next(surface_cases())
    keywords["landusef"].fill(0)
    with pytest.raises(ValueError, match="zero area"):
        ruc_surface_parameters(*inputs, **keywords)


def test_category_first_fractions_preserve_two_horizontal_axes():
    row, inputs, keywords = list(surface_cases())[6]
    tiled = [np.broadcast_to(value, (2, 3)).copy() for value in inputs]
    tiled[0][1, 2] = 6
    tiled_keywords = dict(keywords)
    for name in ("landusef", "soilctop"):
        tiled_keywords[name] = np.broadcast_to(keywords[name][:, None],
                                              (keywords[name].shape[0], 2, 3)).copy()
    actual = ruc_surface_parameters(*tiled, **tiled_keywords)
    flat_keywords = dict(tiled_keywords)
    for name in ("landusef", "soilctop"):
        flat_keywords[name] = tiled_keywords[name].reshape(-1, 6)
    flat = ruc_surface_parameters(*(value.reshape(6) for value in tiled), **flat_keywords)
    for descriptor in fields(actual):
        np.testing.assert_array_equal(getattr(actual, descriptor.name).ravel(),
                                      getattr(flat, descriptor.name))
