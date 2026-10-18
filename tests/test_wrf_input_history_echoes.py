"""Native history echoes preserve source words, type, and named axes."""
from types import SimpleNamespace

import numpy as np


def test_native_echoes_preserve_scalar_and_soil_records_without_mutating_input():
    from gpuwm.io.history_layout import wrf_file_grid_history_fields

    raw = {
        "CFN": np.asarray(np.float32(.1337), dtype=np.float64),
        "BATHYMETRY_FLAG": np.asarray(1, dtype=np.float64),
        "ZS": np.asarray([.05, .25, .7, 1.5], dtype=np.float64),
    }
    snapshot = {name: value.copy() for name, value in raw.items()}
    fields = wrf_file_grid_history_fields(SimpleNamespace(restored=SimpleNamespace(raw=raw)))
    assert fields["CFN"].shape == fields["BATHYMETRY_FLAG"].shape == ()
    assert fields["CFN"].dtype == fields["ZS"].dtype == np.float32
    assert fields["BATHYMETRY_FLAG"].dtype == np.int32
    np.testing.assert_array_equal(fields["CFN"].view("u4"), np.array(.1337, np.float32).view("u4"))
    np.testing.assert_array_equal(fields["ZS"], np.asarray(snapshot["ZS"], np.float32))
    for name, value in raw.items():
        np.testing.assert_array_equal(value, snapshot[name])


def test_native_scalar_soil_and_face_records_survive_the_real_writer(tmp_path):
    from gpuwm.io.wrfout import WrfoutWriter
    from gpuwm.verify_exact.replay import read_frame

    fields = {
        "CFN": np.array(.1337, np.float32),
        "BATHYMETRY_FLAG": np.array(1, np.int32),
        "ZS": np.array([.05, .25, .7, 1.5], np.float32),
        "C1F": np.linspace(0, 1, 6, dtype=np.float32),
        "MAPFAC_UX": np.full((3, 5), np.float32(1.0143)),
        "ISEEDARR_SPPT": np.zeros(8, np.int32),
        "SAVE_TOPO_FROM_REAL": np.zeros((), np.int32),
    }
    path = tmp_path / "wrfout_echoes"
    with WrfoutWriter(path, nx=4, ny=3, nz=5, dx=3000., dy=3000., soil_layers=4) as writer:
        writer.write_frame("2024-01-01_00:00:00", fields)
    actual = read_frame(path)
    for name, expected in fields.items():
        assert actual[name].shape == expected.shape
        assert actual[name].dtype == expected.dtype
        assert actual[name].tobytes() == expected.tobytes()


def test_initialized_history_slots_do_not_mutate_native_inputs_or_invent_active_seeds():
    from gpuwm.io.history_layout import wrf_file_grid_history_fields
    raw = {"SAVE_TOPO_FROM_REAL": np.asarray(1, np.int32)}
    cfg = SimpleNamespace(nx=4, ny=3, dx=3000., dy=3000., use_adaptive_time_step=False,
                          spp_conv=1, spp_pbl=0, spp_lsm=0)
    fields = wrf_file_grid_history_fields(SimpleNamespace(restored=SimpleNamespace(raw=raw)), cfg)
    assert raw["SAVE_TOPO_FROM_REAL"] == 1
    assert fields["SAVE_TOPO_FROM_REAL"].item() == 0
    assert "ISEEDARRAY_SPP_CONV" not in fields
    assert fields["ISEEDARRAY_SPP_PBL"].dtype == np.int32
    assert fields["ISEEDARRAY_SPP_PBL"].tolist() == [0] * 8
    assert fields["NEST_POS"].shape == fields["SST_INPUT"].shape == (3, 4)


def test_stock_nominal_geometry_does_not_refuse_unused_or_absent_map_factors():
    from gpuwm.static.rust_bridge import wrf_cell_geometry
    for args, kwargs in (((np.zeros((2, 3)), np.full((2, 3), np.nan)), {}),
                         ((), {"shape": (2, 3)})):
        area, distance = wrf_cell_geometry(3000., 2000., *args, **kwargs)
        np.testing.assert_array_equal(area, np.full((2, 3), 6000000., np.float32))
        np.testing.assert_array_equal(distance, np.full((2, 3), 3000., np.float32))


def test_stock_scalar_history_axes_match_recorded_metadata():
    import os
    from pathlib import Path
    import pytest
    from gpuwm.io.wrf_input_history_schema import WRF_INPUT_HISTORY_ROWS, WRF_INITIALIZED_HISTORY_ROWS
    from gpuwm.verify_exact.recording import load_run_record
    recordings = Path(os.environ.get("GPUWM_EXACT_RECORDINGS", "/work/cache/combo-rec-full"))
    if not recordings.is_dir():
        pytest.skip("needs actual stock metadata")
    schemas = WRF_INPUT_HISTORY_ROWS | WRF_INITIALIZED_HISTORY_ROWS
    for combo in ("A071", "A088"):
        reference = load_run_record(recordings / "scalar/iowa-convective" / combo)
        scalars = {name for name in schemas if reference.fields[name].shape == ()}
        assert len(scalars) == 20
        for name in scalars:
            assert schemas[name]["history_dimensions"] == [], name


def test_recorded_cell_geometry_matches_stock_full_planes():
    import os
    from pathlib import Path
    import pytest
    from gpuwm import netcdf_bridge
    from gpuwm.static.rust_bridge import wrf_cell_geometry
    from gpuwm.verify_exact.recording import load_run_record
    recordings = Path(os.environ.get("GPUWM_EXACT_RECORDINGS", "/work/cache/combo-rec-full"))
    if not recordings.is_dir():
        pytest.skip("needs actual native inputs and stock recordings")
    for combo in ("A071", "A088"):
        reference = load_run_record(recordings / "scalar/iowa-convective" / combo)
        stock = reference.raw(reference.raw_steps[0])
        with netcdf_bridge.open_dataset(recordings / "inputs/iowa-convective" / combo / "wrfinput_d01") as dataset:
            area, distance = wrf_cell_geometry(dataset.getncattr("DX"), dataset.getncattr("DY"),
                np.asarray(dataset.variables["MAPFAC_MX"][:])[0], np.asarray(dataset.variables["MAPFAC_MY"][:])[0])
        for name, actual in (("AREA2D", area), ("DX2D", distance)):
            assert actual.shape == stock[name].shape
            np.testing.assert_array_equal(actual.view("u4"), stock[name].view("u4"), err_msg=name)


def test_native_noah_table_replacement_matches_recorded_snow_albedo():
    import os
    from pathlib import Path
    import pytest
    from gpuwm import netcdf_bridge
    from gpuwm.core.noah import load_tables, pack_params, VEG_COLS
    from gpuwm.static.rust_bridge import wrf_noah_snow_albedo
    from gpuwm.verify_exact.recording import load_run_record
    recordings = Path(os.environ.get("GPUWM_EXACT_RECORDINGS", "/work/cache/combo-rec-full"))
    if not recordings.is_dir():
        pytest.skip("needs actual native inputs and stock recordings")
    reference = load_run_record(recordings / "scalar/iowa-convective/A071")
    stock = reference.raw(reference.raw_steps[0])["SNOALB"]
    with netcdf_bridge.open_dataset(recordings / "inputs/iowa-convective/A071/wrfinput_d01") as dataset:
        params = pack_params(load_tables(str(dataset.getncattr("MMINLU"))))
        categories = np.asarray(dataset.variables["IVGTYP"][0], dtype="i4")
        raw = np.asarray(dataset.variables["SNOALB"][0], dtype="f4")
    # This is a genuine overwritten input, not an echo or a tolerance check.
    assert np.count_nonzero(raw.view("u4") != stock.view("u4")) > 0
    actual = wrf_noah_snow_albedo(params.veg[:, VEG_COLS.index("maxalb")], categories)
    np.testing.assert_array_equal(actual.view("u4"), stock.view("u4"))
