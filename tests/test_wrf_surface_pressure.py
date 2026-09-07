"""Native pressure reconstruction against the independent WRF Fortran body."""
import ctypes
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.cpu_backend import CpuPreprocessBackend

ORACLE = json.loads((Path(__file__).parent / "data/wrf-sfcprs3-fortran.json").read_text())


def native():
    try:
        return CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"native CPU bridge is not available: {error}")


def inputs():
    cases = ORACLE["cases"]
    p = np.array([case["pressure"] for case in cases]).T[:, None, :].copy()
    z = np.array([case["height"] for case in cases]).T[:, None, :].copy()
    terrain = np.array([[case["terrain"] for case in cases]])
    slp = np.array([[case["slp"] for case in cases]])
    return p, z, terrain, slp


@pytest.mark.parametrize("workers", [1, 3, 17])
@pytest.mark.parametrize("reverse", [False, True])
def test_sfcprs3_matches_independent_fortran_in_every_column(workers, reverse):
    p, z, terrain, slp = inputs()
    if reverse:
        p, z = p[::-1], z[::-1]
    actual = native().surface_pressure_from_sea_level(p, z, terrain, slp, workers=workers)
    expected = np.array([[case["expected_fp32_bits"] for case in ORACLE["cases"]]], dtype=np.uint32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected)


@pytest.mark.parametrize("field, index, value, message", [
    (0, (2, 0, 4), 92500., "pressure"),
    (1, (3, 0, 4), float("nan"), "non-finite"),
    (2, (0, 4), float("inf"), "non-finite"),
    (3, (0, 4), 0., "non-finite"),
])
def test_invalid_column_has_position_and_does_not_reach_forecast(field, index, value, message):
    arrays = list(inputs())
    arrays[field][index] = value
    with pytest.raises(ValueError, match="row 0, column 4.*" + message):
        native().surface_pressure_from_sea_level(*arrays, workers=3)


def test_unbracketed_slp_is_a_named_column_error():
    p, z, terrain, slp = inputs()
    terrain[0, 6], slp[0, 6] = 50., 10000.
    with pytest.raises(ValueError, match="row 0, column 6.*interpolation"):
        native().surface_pressure_from_sea_level(p, z, terrain, slp)


def test_native_failure_leaves_whole_output_untouched():
    backend = native()
    p, z, terrain, slp = inputs()
    backend.surface_pressure_from_sea_level(p, z, terrain, slp)  # bind ABI
    call = backend._library.gpuwm_wrf_sfcprs3_from_f64
    order = np.arange(p.shape[0], dtype=np.uint64)
    output = np.full(terrain.shape, -1234, dtype=np.float32)
    slp[0, 5] = slp[0, 2] = float("nan")
    first = ctypes.c_size_t()
    args = (z.ctypes.data, p.ctypes.data, terrain.ctypes.data, slp.ctypes.data,
            order.ctypes.data, output.ctypes.data, p.shape[0], terrain.size, 3,
            ctypes.addressof(first))
    assert call(*args) == 3 and first.value == 2
    assert np.all(output == -1234)
    order[1] = order[0]
    assert call(*args) == 2 and first.value == ctypes.c_size_t(-1).value
    assert np.all(output == -1234)
    assert call(None, *args[1:]) == 1


def test_old_library_only_refuses_when_the_new_operation_is_requested():
    backend = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    backend._library = SimpleNamespace()
    with pytest.raises(ValueError, match="lacks WRF sea-level pressure reconstruction"):
        backend.surface_pressure_from_sea_level(*inputs())


def test_false_branch_initializes_real_state(monkeypatch):
    from test_initial_perturbation import _application_fixture
    from gpuwm.ingest import real
    native()
    cfg, coord, grid, snapshot, terrain, orography = _application_fixture()
    snapshot = replace(snapshot, fields={**snapshot.fields, "PMSL": np.full(terrain.shape, 101100., np.float32)})
    expected = native().surface_pressure_from_sea_level(
        np.broadcast_to(snapshot.levels_hpa[:, None, None] * 100., snapshot.fields["GHT"].shape),
        snapshot.fields["GHT"], terrain, snapshot.fields["PMSL"])
    monkeypatch.setattr(real, "surface_pressure_from_surface",
                        lambda *a, **kw: pytest.fail("false policy used the true operation"))
    result = real.initialize_real(snapshot, cfg, coord, terrain, grid=grid,
        source_orography=orography, sfcp_to_sfcp=False, p_top=10000.,
        preprocess_backend="cpu", state_backend="preprocess", column_workers=3)
    np.testing.assert_array_equal(result.surface_pressure, expected)
    for name in ("thp", "php", "mup", "qv", "u", "v"):
        assert np.isfinite(getattr(result.state, name)).all(), name


def test_true_policy_does_not_require_or_call_sea_level_operation(monkeypatch):
    from test_initial_perturbation import _initialize
    monkeypatch.setattr(CpuPreprocessBackend, "surface_pressure_from_sea_level",
                        lambda *a, **kw: pytest.fail("true policy called sfcprs3"))
    result, _ = _initialize()
    assert np.isfinite(result.surface_pressure).all()


def test_false_policy_reports_missing_pressure_field_before_preprocessing():
    from test_initial_perturbation import _application_fixture
    from gpuwm.ingest.real import initialize_real
    cfg, coord, _, snapshot, terrain, orography = _application_fixture()
    with pytest.raises(KeyError, match="PMSL"):
        initialize_real(snapshot, cfg, coord, terrain,
                        source_orography=orography, sfcp_to_sfcp=False)


@pytest.mark.parametrize("units,value", [("Pa", 101100.), ("hPa", 1011.)])
def test_met_em_slp_uses_rust_unit_conversion_and_shared_pressure_operation(tmp_path, units, value):
    from test_metem_ingest import case
    from gpuwm.ingest.metem import read_met_em
    from gpuwm.metem_door import metgrid_initialization_controls
    def add_slp(ds):
        ds.FLAG_PSFC = ds.FLAG_SOILHGT = ds.FLAG_SLP = 1
        var = ds.createVariable("PMSL", "f4", ("Time", "south_north", "west_east"))
        var.units = units
        var[:] = value
    met = read_met_em(case(tmp_path, mutate=add_slp))
    controls = metgrid_initialization_controls(met, SimpleNamespace(controls={}))
    assert controls["sfcp_to_sfcp"] is False
    np.testing.assert_array_equal(met.snapshot.fields["PMSL"], np.full((5,5), 101100., np.float32))
    p = np.broadcast_to(met.snapshot.levels_hpa[:,None,None]*100., met.snapshot.fields["GHT"].shape)
    result = native().surface_pressure_from_sea_level(p, met.snapshot.fields["GHT"],
        met.statics["HGT_M"], met.snapshot.fields["PMSL"])
    np.testing.assert_array_equal(result, met.snapshot.fields["PMSL"])


def test_flagged_met_em_missing_slp_names_the_missing_input(tmp_path):
    from test_metem_ingest import case
    from gpuwm.ingest.metem import read_met_em, MetgridRefusal
    def flag(ds):
        ds.FLAG_SLP = 1
    with pytest.raises(MetgridRefusal, match="PMSL"):
        read_met_em(case(tmp_path, mutate=flag))


@pytest.mark.parametrize("unaligned", [None, 0, 1, 2, 3])
def test_ffi_aligns_offset_buffers_without_copying_aligned_inputs(unaligned):
    arrays = list(inputs())
    if unaligned is not None:
        original = arrays[unaligned]
        offset = np.ndarray(original.shape, dtype=original.dtype,
                            buffer=bytearray(original.nbytes+1), offset=1)
        offset[:] = original
        assert offset.flags.c_contiguous and not offset.flags.aligned
        arrays[unaligned] = offset
    def intercept(height, pressure, terrain, slp, *_):
        pointers = (pressure, height, terrain, slp)
        assert all(pointer % 8 == 0 for pointer in pointers)
        for index, array in enumerate(arrays):
            if index != unaligned:
                assert pointers[index] == array.ctypes.data, "aligned input was copied"
        return 2  # no native dereference: a failing wrapper must be safe too
    backend = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    backend._library = SimpleNamespace(gpuwm_wrf_sfcprs3_from_f64=intercept)
    with pytest.raises(ValueError, match="sfcprs3"):
        backend.surface_pressure_from_sea_level(*arrays)
