"""Native refined input geometry and competing-authority regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.wrfinput import _validate_wrfinput_geometry
from gpuwm.ingest.wrfinput_sfire import fire_input_extents, fire_static_fields
from gpuwm.sfire_coordinates import fire_coordinate_mode


def test_native_refinement_extension_is_pinned_by_atmosphere_and_namelist():
    cfg = SimpleNamespace(ifire=2, sr_x=3, sr_y=2, nfmc=5)
    dims = {"west_east": 6, "south_north": 4}
    extents = {**dims, **fire_input_extents(cfg, dims)}
    var = SimpleNamespace(dimensions=("Time", "south_north_subgrid", "west_east_subgrid"))
    _validate_wrfinput_geometry("NFUEL_CAT", var, extents, np.ones((10, 21), np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_wrfinput_geometry("NFUEL_CAT", var, extents, np.ones((8, 18), np.float32))
    var.dimensions = ("Time", "south_north", "west_east")
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_wrfinput_geometry("NFUEL_CAT", var, extents, np.ones((10, 21), np.float32))


def test_foreign_static_preserves_native_word_arrays_and_metric_authority():
    fx = np.array([[6.25, 18.75]], np.float32)
    restored = SimpleNamespace(raw={"FXLAT": fx, "FXLONG": fx, "LFN_TIME": np.array([0], np.float32)},
                               global_attributes={"MAP_PROJ": 0, "CEN_LAT": 0.0})
    cfg = SimpleNamespace(ifire=2, fire_static="")
    fields = fire_static_fields(restored, cfg)
    assert fields["FXLAT"] is fx
    assert fields["LFN_TIME"] is restored.raw["LFN_TIME"]
    assert fire_coordinate_mode(fields) == "metric"
    cfg.fire_static = "other-bound-bundle.npz"
    with pytest.raises(ValueError, match="one bound static authority"):
        fire_static_fields(restored, cfg)


def test_inactive_fire_does_not_import_or_allocate_fine_grid():
    assert fire_input_extents(SimpleNamespace(ifire=0), {}) == {}
    assert fire_static_fields(SimpleNamespace(), SimpleNamespace(ifire=0)) is None


def test_actual_native_fmep_dimension_truncation_keeps_shape_and_name_guards():
    from gpuwm.ingest.wrfinput_sfire import FMEP_DIMENSION_ALIAS
    cfg = SimpleNamespace(ifire=2, sr_x=3, sr_y=2, nfmc=5)
    dims = {"west_east": 6, "south_north": 4}
    extents = {**dims, **fire_input_extents(cfg, dims)}
    var = SimpleNamespace(dimensions=("Time", FMEP_DIMENSION_ALIAS, "south_north", "west_east"))
    _validate_wrfinput_geometry("FMEP", var, extents, np.zeros((2, 4, 6), np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_wrfinput_geometry("FMEP", var, extents, np.zeros((1, 4, 6), np.float32))
    var.dimensions = ("Time", "unbound_two_layer_axis", "south_north", "west_east")
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_wrfinput_geometry("FMEP", var, extents, np.zeros((2, 4, 6), np.float32))


def test_actual_native_fmc_class_stagger_name_still_has_exact_nfmc_extent():
    from gpuwm.ingest.wrfinput_sfire import FMC_GC_DIMENSION_ALIAS
    dims = {"west_east": 6, "south_north": 4}
    cfg = SimpleNamespace(ifire=2, sr_x=3, sr_y=2, nfmc=5)
    extents = {**dims, **fire_input_extents(cfg, dims)}
    var = SimpleNamespace(dimensions=("Time", FMC_GC_DIMENSION_ALIAS, "south_north", "west_east"))
    _validate_wrfinput_geometry("FMC_GC", var, extents, np.zeros((5, 4, 6), np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_wrfinput_geometry("FMC_GC", var, extents, np.zeros((6, 4, 6), np.float32))
