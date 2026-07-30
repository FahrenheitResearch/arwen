from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.verify.cases import real74_chain, real74_n5s
from gpuwm.verify.cases.real74_n5s import read_wrfbdy, read_wrfinput
from gpuwm.verify.n5s_common import (
    RESTORED_INPUT_FILENAMES, restored_input_sha256, write_json,
)
from gpuwm.verify.n5s_metrics import (
    build_ensemble_evidence, build_gpu_evidence, first_object_time,
    fss_distance, has_qualifying_object, make_registration, rmse,
    score_run_pair, validate_registration, verify_run_artifact,
)
from tools.n5s.derive_wps_namelist import derive
from tools.n5s.perturb_ulp import perturb


@pytest.mark.parametrize(
    ("ra_physics", "ra_lw_physics", "ra_sw_physics"),
    ((0, -1, -1), (0, 1, 1)),
)
def test_restored_model_does_not_inject_rrtmgp_for_other_radiation_schemes(
        ra_physics, ra_lw_physics, ra_sw_physics):
    cfg = SimpleNamespace(
        ra_physics=ra_physics,
        ra_lw_physics=ra_lw_physics,
        ra_sw_physics=ra_sw_physics,
    )
    assert real74_n5s._restored_model_radiation(
        cfg, datetime(2026, 7, 22), None, None, None) is None


def _variable(dataset, name, dimensions, values, *, compressed=False,
              dtype="f4"):
    options = {"zlib": True, "complevel": 1} if compressed else {}
    variable = dataset.createVariable(name, dtype, dimensions, **options)
    variable[...] = np.asarray(values, dtype=dtype)
    return variable


def _small_wrfinput(path: Path, *, nz: int = 2, ny: int = 4, nx: int = 5,
                    moisture_names=("QVAPOR",)) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", nz),
                ("bottom_top_stag", nz + 1), ("south_north", ny),
                ("south_north_stag", ny + 1), ("west_east", nx),
                ("west_east_stag", nx + 1)):
            dataset.createDimension(name, size)
        mass = ("Time", "bottom_top", "south_north", "west_east")
        _variable(dataset, "U", ("Time", "bottom_top", "south_north",
                                  "west_east_stag"), 1.0)
        _variable(dataset, "V", ("Time", "bottom_top", "south_north_stag",
                                  "west_east"), 2.0)
        _variable(dataset, "W", ("Time", "bottom_top_stag", "south_north",
                                  "west_east"), 3.0)
        t = np.arange(nz * ny * nx, dtype=np.float32).reshape(1, nz, ny, nx)
        _variable(dataset, "T", mass, t)
        _variable(dataset, "T_INIT", mass, np.float32(300.0))
        _variable(dataset, "PH", ("Time", "bottom_top_stag", "south_north",
                                   "west_east"), 4.0)
        _variable(dataset, "PHB", ("Time", "bottom_top_stag", "south_north",
                                    "west_east"), 5.0)
        _variable(dataset, "MU", ("Time", "south_north", "west_east"), 6.0)
        _variable(dataset, "MUB", ("Time", "south_north", "west_east"), 7.0)
        for index, name in enumerate(moisture_names):
            _variable(dataset, name, mass, 0.001 * (index + 1))


def _real_shaped_d01_wrfinput(path: Path) -> None:
    """Minimal mapped fixture with the registered d01 WRF geometry."""
    nz, ny, nx = 49, 200, 250
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", nz),
                ("bottom_top_stag", nz + 1), ("south_north", ny),
                ("south_north_stag", ny + 1), ("west_east", nx),
                ("west_east_stag", nx + 1)):
            dataset.createDimension(name, size)
        mass = ("Time", "bottom_top", "south_north", "west_east")
        for name, dims, value in (
                ("U", ("Time", "bottom_top", "south_north",
                       "west_east_stag"), 1.0),
                ("V", ("Time", "bottom_top", "south_north_stag",
                       "west_east"), 2.0),
                ("W", ("Time", "bottom_top_stag", "south_north",
                       "west_east"), 3.0),
                ("T", mass, 1.0), ("T_INIT", mass, 300.0),
                ("PH", ("Time", "bottom_top_stag", "south_north",
                        "west_east"), 4.0),
                ("PHB", ("Time", "bottom_top_stag", "south_north",
                         "west_east"), 5.0),
                ("MU", ("Time", "south_north", "west_east"), 6.0),
                ("MUB", ("Time", "south_north", "west_east"), 7.0),
                ("QVAPOR", mass, 0.001),
                ("C1H", ("Time", "bottom_top"), 1.0),
                ("C2H", ("Time", "bottom_top"), 0.0),
                ("C1F", ("Time", "bottom_top_stag"), 1.0),
                ("C2F", ("Time", "bottom_top_stag"), 0.0),
                ("MAPFAC_M", ("Time", "south_north", "west_east"), 1.0),
                ("MAPFAC_U", ("Time", "south_north", "west_east_stag"),
                 2.0),
                ("MAPFAC_V", ("Time", "south_north_stag", "west_east"),
                 4.0)):
            _variable(dataset, name, dims, value, compressed=True)


def _real_shaped_scalar_wrfinput(path: Path) -> None:
    """WRF Registry scalars are one-element variables on ``Time``."""
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        for name, value in (
                ("CF1", 1.0), ("CF2", 2.0), ("CF3", 3.0),
                ("P_TOP", 10000.0), ("XTIME", 0.0)):
            _variable(dataset, name, ("Time",), [value])
        _variable(dataset, "ITIMESTEP", ("Time",), [0], dtype="i4")


def _small_wrfbdy(path: Path, *, omit: str | None = None,
                  records: int = 2, cadence_seconds: int = 21600,
                  include_nextbdytime: bool = False) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", records)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 200)
        dataset.createDimension("south_north_stag", 201)
        dataset.createDimension("west_east", 250)
        dataset.createDimension("west_east_stag", 251)
        dataset.createDimension("bottom_top", 49)
        dataset.createDimension("bottom_top_stag", 50)
        dataset.createDimension("bdy_width", 5)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        origin = datetime(1974, 4, 3, 12)
        time_values = tuple(
            origin + timedelta(seconds=index * cadence_seconds)
            for index in range(records))
        for index, value in enumerate(time_values):
            text = value.strftime("%Y-%m-%d_%H:%M:%S")
            times[index] = np.frombuffer(text.encode("ascii"), dtype="S1")
        if include_nextbdytime:
            next_name = (
                "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
            next_times = dataset.createVariable(
                next_name, "S1", ("Time", "DateStrLen"))
            for index, value in enumerate(time_values):
                text = (value + timedelta(seconds=cadence_seconds)).strftime(
                    "%Y-%m-%d_%H:%M:%S")
                next_times[index] = np.frombuffer(
                    text.encode("ascii"), dtype="S1")
        fields = {
            "U": ("bottom_top", "south_north", "west_east_stag", 6.5),
            "V": ("bottom_top", "south_north_stag", "west_east", 6.5),
            "T": ("bottom_top", "south_north", "west_east", 13.0),
            "PH": ("bottom_top_stag", "south_north", "west_east", 52.0),
            "MU": (None, "south_north", "west_east", 6.0),
            "QVAPOR": ("bottom_top", "south_north", "west_east", 0.013),
        }
        for field, (zdim, ydim, xdim, coupled_value) in fields.items():
            for suffix, side_dim in (
                    ("XS", ydim), ("XE", ydim),
                    ("YS", xdim), ("YE", xdim)):
                dims = (["Time", "bdy_width"]
                        + ([] if zdim is None else [zdim]) + [side_dim])
                for infix, value in (("B", 1.0), ("BT", 0.01)):
                    name = f"{field}_{infix}{suffix}"
                    if name != omit:
                        _variable(
                            dataset, name, tuple(dims),
                            coupled_value if infix == "B" else 0.0,
                            compressed=True)


def _explicit_geometry(nz: int, ny: int, nx: int) -> dict[str, int]:
    return {
        "bottom_top": nz, "bottom_top_stag": nz + 1,
        "south_north": ny, "south_north_stag": ny + 1,
        "west_east": nx, "west_east_stag": nx + 1,
        "soil_layers_stag": 4,
    }


@pytest.mark.parametrize(
    ("domain", "nz", "ny", "nx"),
    (("d01", 3, 7, 9), ("d02", 4, 11, 13)),
)
def test_wrfinput_accepts_explicit_arbitrary_domain_geometry(
        tmp_path, domain, nz, ny, nx):
    path = tmp_path / f"wrfinput_{domain}"
    _small_wrfinput(path, nz=nz, ny=ny, nx=nx)
    restored = read_wrfinput(
        path, require_complete=False,
        expected_dimensions=_explicit_geometry(nz, ny, nx))
    assert restored.raw["U"].shape == (nz, ny, nx + 1)
    assert restored.raw["V"].shape == (nz, ny + 1, nx)
    assert restored.raw["PH"].shape == (nz + 1, ny, nx)


def test_wrfinput_explicit_geometry_remains_caller_pinned(tmp_path):
    path = tmp_path / "wrfinput_d01"
    _small_wrfinput(path, nz=3, ny=7, nx=9)
    wrong = _explicit_geometry(3, 7, 10)
    with pytest.raises(ValueError, match=r"U shape mismatch: expected pinned"):
        read_wrfinput(
            path, require_complete=False, expected_dimensions=wrong)


def test_wrfinput_wsm6_inventory_requires_active_mass_species(tmp_path):
    path = tmp_path / "wrfinput_d01"
    active = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW")
    _small_wrfinput(path, moisture_names=active)
    cfg = SimpleNamespace(moist=True, mp_physics=6)
    with pytest.raises(ValueError, match="QGRAUP"):
        read_wrfinput(
            path, cfg=cfg,
            expected_dimensions=_explicit_geometry(2, 4, 5))


def test_wrfinput_wsm6_inventory_rejects_morrison_number_moment(tmp_path):
    path = tmp_path / "wrfinput_d01"
    active = (
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN",
    )
    _small_wrfinput(path, moisture_names=active)
    cfg = SimpleNamespace(moist=True, mp_physics=6)
    with pytest.raises(ValueError, match=r"inactive WRF moisture.*QNRAIN"):
        read_wrfinput(
            path, require_complete=False, cfg=cfg,
            expected_dimensions=_explicit_geometry(2, 4, 5))


def test_wrfinput_nssl_inventory_accepts_exact_registry_names(tmp_path):
    from gpuwm.verify.cases.real74_n5s import NSSL_MOISTURE_MAP

    path = tmp_path / "wrfinput_d01"
    names = tuple(NSSL_MOISTURE_MAP)
    _small_wrfinput(path, moisture_names=names)
    cfg = SimpleNamespace(moist=True, mp_physics=18)
    restored = read_wrfinput(
        path, require_complete=False, cfg=cfg,
        expected_dimensions=_explicit_geometry(2, 4, 5))
    assert set(names) <= set(restored.raw)
    assert all(restored.raw[name].shape == (2, 4, 5) for name in names)


def test_wrfinput_nssl_inventory_requires_every_native_moment(tmp_path):
    from gpuwm.verify.cases.real74_n5s import NSSL_MOISTURE_MAP

    path = tmp_path / "wrfinput_d01"
    names = tuple(name for name in NSSL_MOISTURE_MAP if name != "QVHAIL")
    _small_wrfinput(path, moisture_names=names)
    cfg = SimpleNamespace(moist=True, mp_physics=18)
    with pytest.raises(ValueError, match="QVHAIL"):
        read_wrfinput(
            path, cfg=cfg,
            expected_dimensions=_explicit_geometry(2, 4, 5))


def test_wrfinput_thompson_moisture_selector_is_explicitly_supported(tmp_path):
    path = tmp_path / "wrfinput_d01"
    active = (
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN", "QNICE",
    )
    _small_wrfinput(path, moisture_names=active)
    cfg = SimpleNamespace(moist=True, mp_physics=8)
    restored = read_wrfinput(
        path, require_complete=False, cfg=cfg,
        expected_dimensions=_explicit_geometry(2, 4, 5))
    assert set(active) <= set(restored.raw)


def test_wrfinput_unknown_moisture_selector_remains_fail_closed(tmp_path):
    path = tmp_path / "wrfinput_d01"
    _small_wrfinput(path)
    cfg = SimpleNamespace(moist=True, mp_physics=55)
    with pytest.raises(
            ValueError, match="unsupported active wrfinput mp_physics=55"):
        read_wrfinput(
            path, require_complete=False, cfg=cfg,
            expected_dimensions=_explicit_geometry(2, 4, 5))


def test_nssl_wrfinput_mapping_restores_current_and_rk_initial_fields():
    from gpuwm.verify.cases.real74_n5s import (
        MOISTURE_MAP, NSSL_MOISTURE_MAP, REFLECTIVITY_MICROPHYSICS,
        _restore_active_moisture,
    )

    expected = {
        "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
        "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
        "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
        "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
        "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
    }
    assert NSSL_MOISTURE_MAP == expected
    assert MOISTURE_MAP["QNRAIN"] == "nr"
    assert NSSL_MOISTURE_MAP["QNRAIN"] == "qnr"
    assert REFLECTIVITY_MICROPHYSICS == frozenset((1, 6, 8, 10, 18))

    shape = (2, 3, 4)
    raw = {
        wrf_name: np.full(shape, index + 1.0, np.float32)
        for index, wrf_name in enumerate(expected)
    }
    fields = {}
    for state_name in expected.values():
        fields[state_name] = np.zeros(shape, np.float32)
        fields[f"{state_name}0"] = np.full(shape, -1.0, np.float32)
    fields["nr"] = np.full(shape, -99.0, np.float32)
    state = SimpleNamespace(**fields)
    cfg = SimpleNamespace(moist=True, mp_physics=18)

    _restore_active_moisture(state, raw, cfg, np)

    for wrf_name, state_name in expected.items():
        np.testing.assert_array_equal(getattr(state, state_name), raw[wrf_name])
        np.testing.assert_array_equal(
            getattr(state, f"{state_name}0"), raw[wrf_name])
    np.testing.assert_array_equal(state.nr, -99.0)


def test_importer_cpu_round_trip_and_wrfbdy_tables(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    wrfbdy = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(wrfbdy)
    restored = read_wrfinput(wrfinput, require_complete=False)
    inverse = restored.wrf_frame()
    assert np.array_equal(inverse["U"], restored.raw["U"])
    assert np.array_equal(inverse["V"], restored.raw["V"])
    assert np.array_equal(inverse["W"], restored.raw["W"])
    assert np.array_equal(inverse["T"], restored.raw["T"])
    assert np.array_equal(inverse["PH"], restored.raw["PH"])
    assert np.array_equal(inverse["MU"], restored.raw["MU"])
    assert np.array_equal(inverse["QVAPOR"], restored.raw["QVAPOR"])
    with pytest.raises(ValueError, match="missing mapped WRF variable"):
        read_wrfinput(wrfinput)

    boundaries = read_wrfbdy(
        wrfbdy, run_seconds=4500, restored=restored)
    assert len(boundaries.intervals) == 2
    assert boundaries.intervals[0].start_seconds == 0
    assert boundaries.intervals[0].end_seconds == 21600
    assert boundaries.intervals[1].end_seconds == 43200
    assert boundaries.interval_at(4500) is boundaries.intervals[0]
    assert set(boundaries.intervals[0].fields) == {
        "u", "v", "theta", "phi", "mu", "qv"}
    np.testing.assert_array_equal(
        boundaries.intervals[0].fields["theta"].west.value,
        np.full((49, 200, 5), 13.0, dtype=np.float32))


def test_wrfinput_accepts_real_shaped_scalar_records(tmp_path):
    path = tmp_path / "wrfinput_d01"
    _real_shaped_scalar_wrfinput(path)
    restored = read_wrfinput(path, require_complete=False)
    for name in ("CF1", "CF2", "CF3", "P_TOP", "XTIME", "ITIMESTEP"):
        assert restored.raw[name].shape == ()


def test_wrfinput_rejects_unmapped_variable(tmp_path):
    path = tmp_path / "wrfinput_d01"
    _small_wrfinput(path)
    with netCDF4.Dataset(path, "a") as dataset:
        _variable(
            dataset, "UNMAPPED_SCIENCE_FIELD",
            ("Time", "bottom_top", "south_north", "west_east"), 1.0)
    with pytest.raises(ValueError, match="UNMAPPED_SCIENCE_FIELD"):
        read_wrfinput(path, require_complete=False)


def test_wrfinput_rejects_mapped_shape_mismatch(tmp_path):
    path = tmp_path / "wrfinput_d01"
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", 2), ("south_north", 4),
                ("west_east", 5), ("west_east_stag", 6)):
            dataset.createDimension(name, size)
        _variable(dataset, "U", ("Time", "bottom_top", "south_north"), 1.0)
    with pytest.raises(
            ValueError,
            match=r"U shape mismatch: expected pinned .* got \(2, 4\)"):
        read_wrfinput(path, require_complete=False)


def test_wrfinput_rejects_self_consistent_wrong_domain_geometry(tmp_path):
    path = tmp_path / "wrfinput_d01"
    _small_wrfinput(path)
    with pytest.raises(
            ValueError,
            match=(r"U shape mismatch: expected pinned \(49, 200, 251\).*"
                   r"got \(2, 4, 6\)")):
        read_wrfinput(path, require_complete=False)


def test_wrfbdy_rejects_missing_qv_side(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    path = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(path, omit="QVAPOR_BTYE")
    with pytest.raises(ValueError, match="QVAPOR_BTYE"):
        read_wrfbdy(
            path, run_seconds=4500,
            restored=read_wrfinput(wrfinput, require_complete=False))


def test_wrfbdy_rejects_hourly_forcing_tables(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    path = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables["Times"][1] = np.frombuffer(
            b"1974-04-03_13:00:00", dtype="S1")
    with pytest.raises(ValueError, match="six-hour forcing cadence"):
        read_wrfbdy(
            path, run_seconds=4500,
            restored=read_wrfinput(wrfinput, require_complete=False))


def test_wrfbdy_accepts_explicit_hourly_multi_record_tables(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    path = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(
        path, records=3, cadence_seconds=3600,
        include_nextbdytime=True)
    boundaries = read_wrfbdy(
        path, run_seconds=5400, forcing_interval_seconds=3600,
        restored=read_wrfinput(wrfinput, require_complete=False))
    assert [(value.start_seconds, value.end_seconds)
            for value in boundaries.intervals] == [
                (0, 3600), (3600, 7200), (7200, 10800)]


def test_wrfbdy_accepts_explicit_hourly_single_record_tables(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    path = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(
        path, records=1, cadence_seconds=3600,
        include_nextbdytime=True)
    boundaries = read_wrfbdy(
        path, run_seconds=900, forcing_interval_seconds=3600,
        restored=read_wrfinput(wrfinput, require_complete=False))
    assert len(boundaries.intervals) == 1
    assert boundaries.intervals[0].end_seconds == 3600


def test_wrfbdy_rejects_explicit_cadence_and_nextbdytime_mismatch(tmp_path):
    wrfinput = tmp_path / "wrfinput_d01"
    path = tmp_path / "wrfbdy_d01"
    _real_shaped_d01_wrfinput(wrfinput)
    _small_wrfbdy(
        path, records=1, cadence_seconds=3600,
        include_nextbdytime=True)
    next_name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables[next_name][0] = np.frombuffer(
            b"1974-04-03_14:00:00", dtype="S1")
    with pytest.raises(ValueError, match="nextbdytime"):
        read_wrfbdy(
            path, run_seconds=900, forcing_interval_seconds=3600,
            restored=read_wrfinput(wrfinput, require_complete=False))


def test_production_wrfbdy_decodes_to_wrfinput_perimeter():
    """CPU handoff check: real.exe LBCs equal the 12Z initial perimeter."""
    root = Path(__file__).resolve().parents[1] / "out" / "n5s-inputs"
    wrfinput = root / "wrfinput_d01"
    wrfbdy = root / "wrfbdy_d01"
    if not wrfinput.is_file() or not wrfbdy.is_file():
        pytest.skip("registered N5S handoff inputs are not present")

    restored = read_wrfinput(wrfinput)
    raw = restored.raw
    boundaries = read_wrfbdy(
        wrfbdy, run_seconds=4500, restored=restored)
    interval = boundaries.intervals[0]
    mu = np.asarray(raw["MUB"] + raw["MU"], dtype=np.float32)
    mux = np.asarray(
        np.float32(0.5) * (mu + np.roll(mu, 1, axis=1)),
        dtype=np.float32)
    muy = np.asarray(
        np.float32(0.5) * (mu + np.roll(mu, 1, axis=0)),
        dtype=np.float32)
    mux[:, 0] = mu[:, 0]
    muy[0, :] = mu[0, :]
    mux = np.concatenate((mux, mux[:, :1]), axis=1)
    muy = np.concatenate((muy, muy[:1, :]), axis=0)
    mux[:, -1] = mu[:, -1]
    muy[-1, :] = mu[-1, :]
    c1h = raw["C1H"][:, None, None]
    c2h = raw["C2H"][:, None, None]
    c1f = raw["C1F"][:, None, None]
    c2f = raw["C2F"][:, None, None]
    chm = np.asarray(c1h * mu[None] + c2h, dtype=np.float32)
    chf = np.asarray(c1f * mu[None] + c2f, dtype=np.float32)
    expected = {
        "u": np.asarray(
            np.asarray((c1h * mux[None] + c2h) * raw["U"],
                       dtype=np.float32) / raw["MAPFAC_U"][None],
            dtype=np.float32),
        "v": np.asarray(
            np.asarray((c1h * muy[None] + c2h) * raw["V"],
                       dtype=np.float32) / raw["MAPFAC_V"][None],
            dtype=np.float32),
        "theta": np.asarray(chm * raw["T"], dtype=np.float32),
        "phi": np.asarray(chf * raw["PH"], dtype=np.float32),
        "mu": raw["MU"][None],
        "qv": np.asarray(chm * raw["QVAPOR"], dtype=np.float32),
    }
    width = boundaries.spec_bdy_width
    for name, field in expected.items():
        strips = {
            "west": field[..., :width],
            "east": field[..., -width:][..., ::-1],
            "south": field[..., :width, :],
            "north": field[..., -width:, :][..., ::-1, :],
        }
        decoded = interval.fields[name]
        for side_name, expected_strip in strips.items():
            actual = getattr(decoded, side_name).value
            assert actual.shape == expected_strip.shape
            # Four FP32 mass/coupling operations have gamma_4 ~= 2.4e-7.
            # Normalizing WRF's expression tree to gpuwm's reduces the
            # observed handoff residual below two FP32 epsilons.
            np.testing.assert_allclose(
                actual, expected_strip, rtol=2.0e-7, atol=2.0e-6,
                err_msg=f"{name} {side_name} boundary mismatch")


def test_wps_derivation_uses_bundle_cadence_and_explicit_table_path(tmp_path):
    source = tmp_path / "namelist.wps"
    source.write_text(
        """&share
 start_date = 'old', 'old', 'old', 'old',
 end_date = 'old', 'old', 'old', 'old',
 interval_seconds = 1,
/
&metgrid
 fg_name = 'FILE',
/
""", encoding="utf-8")
    actual = derive(source)
    assert "end_date = '1974-04-03_18:00:00', '1974-04-03_12:00:00'" in actual
    assert "interval_seconds = 21600," in actual
    assert "nocolons = .true.," in actual
    assert "opt_metgrid_tbl_path = './'," in actual


def test_binary_runners_source_checked_default_environment():
    script_root = Path(__file__).parents[1] / "tools" / "n5s"
    for name in (
            "prep_met_em.sh", "run_real.sh", "run_ensemble.sh",
            "tmux_worker.sh"):
        text = (script_root / name).read_text(encoding="utf-8")
        assert 'ENV_FILE="$SCRIPT_DIR/wsl-env.sh"' in text
        assert "--env-file" in text
        assert 'source "$ENV_FILE"' in text
    profile = (script_root / "wsl-env.sh").read_text(encoding="utf-8")
    assert "/home/drew/WRF_BUILD/LIBRARIES" in profile
    assert "JASPERLIB" in profile
    assert "LD_LIBRARY_PATH" in profile
    assert "missing required directory" in profile
    worker = (script_root / "tmux_worker.sh").read_text(encoding="utf-8")
    assert 'mkdir -p -- "$(dirname -- "$output")"' in worker


def test_perturb_ulp_changes_exactly_one_documented_value(tmp_path):
    source = tmp_path / "wrfinput_d01"
    destination = tmp_path / "member" / "wrfinput_d01"
    _small_wrfinput(source)
    record = perturb(source, destination, field="T", index=(1, 2, 3))
    with netCDF4.Dataset(source) as left, netCDF4.Dataset(destination) as right:
        a = np.asarray(left.variables["T"][:])
        b = np.asarray(right.variables["T"][:])
    changed = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
    assert changed.tolist() == [[0, 1, 2, 3]]
    assert record["before_hex_bits"] != record["after_hex_bits"]
    assert record["netcdf_index"] == [0, 1, 2, 3]
    assert json.loads((destination.parent / "perturbation.json").read_text(
        encoding="utf-8"))["operation"] == "numpy.nextafter"


def test_metrics_analytic_rmse_fss_objects_and_quiet_timing():
    left = np.zeros((5, 5), dtype=np.float64)
    right = np.full((5, 5), 3.0, dtype=np.float64)
    assert rmse(left, right) == 3.0
    assert fss_distance(left, left, threshold=40.0, half_width_cells=1) == 0.0
    echo_a = left.copy()
    echo_b = left.copy()
    echo_a[0, 0] = 50.0
    echo_b[-1, -1] = 50.0
    assert fss_distance(
        echo_a, echo_b, threshold=40.0, half_width_cells=0) == 1.0
    object_field = np.zeros((5, 5), dtype=np.float64)
    object_field[1:3, 1:3] = 45.0
    assert has_qualifying_object(
        object_field, threshold=40.0, min_cells=4, connectivity=8)
    assert not has_qualifying_object(
        object_field, threshold=40.0, min_cells=5, connectivity=8)
    quiet = {300: np.zeros((1, 5, 5)), 600: np.zeros((1, 5, 5))}
    assert first_object_time(
        quiet, threshold=40.0, min_cells=1, duration_seconds=600,
        cadence_seconds=300) == 900.0
    active = dict(quiet)
    active[600] = np.full((1, 5, 5), 45.0)
    assert first_object_time(
        active, threshold=40.0, min_cells=1, duration_seconds=600,
        cadence_seconds=300) == 600.0


def test_registration_refuses_absent_hash_and_side_mismatch():
    registration = make_registration(
        run_minutes=30, history_minutes=5, commit="1" * 40)
    broken = deepcopy(registration)
    del broken["expected_samples"]
    with pytest.raises(ValueError, match="missing pins"):
        validate_registration(broken)
    broken = deepcopy(registration)
    broken["parameters"]["reflectivity_threshold_dbz"] = 35.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_registration(broken)


def _write_frame(path: Path, *, offset: float, seconds: int) -> None:
    nz, ny, nx = 1, 12, 12
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", nz),
                ("bottom_top_stag", nz + 1), ("south_north", ny),
                ("south_north_stag", ny + 1), ("west_east", nx),
                ("west_east_stag", nx + 1)):
            dataset.createDimension(name, size)
        fields = {
            "U": (("Time", "bottom_top", "south_north", "west_east_stag"),
                  np.zeros((1, nz, ny, nx + 1))),
            "V": (("Time", "bottom_top", "south_north_stag", "west_east"),
                  np.zeros((1, nz, ny + 1, nx))),
            "W": (("Time", "bottom_top_stag", "south_north", "west_east"),
                  np.zeros((1, nz + 1, ny, nx))),
            "T": (("Time", "bottom_top", "south_north", "west_east"),
                  np.full((1, nz, ny, nx), offset + seconds / 1800.0)),
            "PH": (("Time", "bottom_top_stag", "south_north", "west_east"),
                   np.zeros((1, nz + 1, ny, nx))),
            "MU": (("Time", "south_north", "west_east"),
                   np.zeros((1, ny, nx))),
            "QVAPOR": (("Time", "bottom_top", "south_north", "west_east"),
                       np.zeros((1, nz, ny, nx))),
            "REFL_10CM": (("Time", "bottom_top", "south_north", "west_east"),
                          np.zeros((1, nz, ny, nx))),
        }
        for name, (dimensions, values) in fields.items():
            _variable(dataset, name, dimensions, values)


def _make_run(root: Path, name: str, registration, *, offset: float,
              perturbation: bool) -> Path:
    run = root / name
    run.mkdir(parents=True)
    write_json(run / "n5s-preregistration.json", registration)
    (run / "exit.status").write_text("0\n", encoding="utf-8")
    if perturbation:
        write_json(run / "perturbation.json", {
            "field": "T", "k": 0, "j": int(offset) + 1,
            "i": int(offset) + 2, "before_hex_bits": "0x00000000",
            "after_hex_bits": "0x00000001"})
    start = datetime(1974, 4, 3, 12)
    for domain in ("d01", "d02", "d03", "d04"):
        for seconds in (0, 1800):
            valid = start + timedelta(seconds=seconds)
            path = run / valid.strftime(
                f"wrfout_{domain}_%Y-%m-%d_%H_%M_%S")
            _write_frame(path, offset=offset, seconds=seconds)
    return run


@pytest.fixture(scope="module")
def n5s_evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("n5s-evidence")
    restored = root / "restored"
    restored.mkdir()
    for index, name in enumerate(RESTORED_INPUT_FILENAMES):
        (restored / name).write_bytes(f"fixture-{index}".encode("ascii"))
    registration = make_registration(
        run_minutes=30, history_minutes=30, commit="2" * 40)
    write_json(root / "n5s-preregistration.json", registration)
    unperturbed = _make_run(
        root, "unperturbed", registration, offset=0.0, perturbation=False)
    members = [
        _make_run(root, f"member-{index:02d}", registration,
                  offset=float(index + 1), perturbation=True)
        for index in range(3)]
    gpu = _make_run(root, "gpu", registration, offset=0.0, perturbation=False)
    write_json(root / "exit-status-ledger.json", {
        "schema": 1,
        "runs": [
            {"id": run.name, "exit_status": 0}
            for run in [unperturbed, *members]
        ],
        "all_succeeded": True,
    })
    digest = restored_input_sha256(restored)
    (gpu / "restored_input_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    ensemble = build_ensemble_evidence(
        ensemble_root=root, unperturbed_directory=unperturbed,
        member_directories=members, registration=registration,
        restored_inputs=restored, output=root / "n5s-ensemble.json")
    candidate_path = root / "N5S-gpu-evidence.json"
    candidate = build_gpu_evidence(
        gpu_directory=gpu, unperturbed_directory=unperturbed,
        registration=registration, restored_inputs=restored,
        output=candidate_path)
    return root, candidate_path, ensemble, candidate


def test_emitters_round_trip_through_schema_authority(n5s_evidence):
    root, candidate_path, ensemble, candidate = n5s_evidence
    report = real74_chain.evaluate_n5s_shadow(candidate_path, root)
    assert report["passed"] is True
    assert report["expected_cpu_pairs"] == 3
    assert report["restored_input_sha256"] == ensemble["restored_input_sha256"]
    assert set(ensemble["cpu_pair_distances"]) == set(
        candidate["gpu_vs_unperturbed_distances"])
    assert all(row["gpu_distance"] <= row["cpu_e95"]
               for row in report["comparisons"])
    assert ensemble["registration"] == candidate["registration"]
    for record in [ensemble["unperturbed"], *ensemble["members"]]:
        artifact = json.loads((root / record["relative_path"]).read_text(
            encoding="utf-8"))
        assert artifact["registration"] == ensemble["registration"]
        assert all(
            set(frame) == {
                "domain", "seconds", "filename", "byte_length", "sha256"}
            for frame in artifact["frames"])
    gpu_artifact = json.loads(
        (root / candidate["gpu_candidate"]["relative_path"]).read_text(
            encoding="utf-8"))
    assert gpu_artifact["registration"] == candidate["registration"]


def test_final_evaluator_rejects_registration_free_candidate_summary(
        n5s_evidence):
    root, _candidate_path, _ensemble, candidate = n5s_evidence
    registration_free = deepcopy(candidate)
    del registration_free["registration"]
    path = root / "registration-free-candidate.json"
    write_json(path, registration_free)

    with pytest.raises(ValueError, match="registration"):
        real74_chain.evaluate_n5s_shadow(path, root)


def test_final_evaluator_rejects_edited_candidate_distance(n5s_evidence):
    root, _candidate_path, _ensemble, candidate = n5s_evidence
    edited = deepcopy(candidate)
    metric = next(iter(edited["gpu_vs_unperturbed_distances"]))
    edited["gpu_vs_unperturbed_distances"][metric] = 1.0e30
    path = root / "edited-distance-candidate.json"
    write_json(path, edited)

    with pytest.raises(ValueError, match="artifact re-scoring"):
        real74_chain.evaluate_n5s_shadow(path, root)


def test_final_evaluator_rehashes_candidate_frames(n5s_evidence):
    root, candidate_path, _ensemble, candidate = n5s_evidence
    artifact = json.loads(
        (root / candidate["gpu_candidate"]["relative_path"]).read_text(
            encoding="utf-8"))
    frame = root / "gpu" / artifact["frames"][0]["filename"]
    original = frame.read_bytes()
    frame.write_bytes(original + b"tampered")
    try:
        with pytest.raises(ValueError, match="frame inventory"):
            real74_chain.evaluate_n5s_shadow(candidate_path, root)
    finally:
        frame.write_bytes(original)


def test_final_report_loader_reconstructs_instead_of_trusting_boolean(
        n5s_evidence):
    root, candidate_path, _ensemble, _candidate = n5s_evidence
    report = real74_chain.evaluate_n5s_shadow(candidate_path, root)
    report["passed"] = False
    path = root / "N5S-controller-report.json"
    write_json(path, report)

    loaded = real74_chain._load_gate_report(
        path, "N5S_matched_physics_wrf_shadow")

    assert loaded["passed"] is True
    assert loaded["score_binding_sha256"] == report["score_binding_sha256"]


def test_frame_deletion_after_manifest_fails_artifact_verification(n5s_evidence):
    root, _candidate_path, ensemble, _candidate = n5s_evidence
    record = ensemble["members"][0]
    artifact_path = root / record["relative_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    run = artifact_path.parent
    frame = run / artifact["frames"][0]["filename"]
    original = frame.read_bytes()
    frame.unlink()
    try:
        with pytest.raises(ValueError, match="frame times|frame inventory"):
            verify_run_artifact(run, ensemble["registration"])
    finally:
        frame.write_bytes(original)


def test_contradictory_registrations_raise_before_scoring(n5s_evidence):
    root, _candidate_path, ensemble, _candidate = n5s_evidence
    conflicting = make_registration(
        run_minutes=30, history_minutes=30, commit="2" * 40,
        parameters={"reflectivity_threshold_dbz": 35.0})
    with pytest.raises(ValueError, match="registrations differ"):
        score_run_pair(
            root / "member-00", root / "member-01",
            ensemble["registration"], conflicting)


def test_manifest_builder_rejects_failed_status_ledger_before_frames(
        n5s_evidence):
    root, _candidate_path, ensemble, _candidate = n5s_evidence
    ledger_path = root / "exit-status-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    failed = deepcopy(ledger)
    failed["all_succeeded"] = False
    write_json(ledger_path, failed)
    try:
        with pytest.raises(ValueError, match="ledger reports a failed run"):
            build_ensemble_evidence(
                ensemble_root=root,
                unperturbed_directory=root / "unperturbed",
                member_directories=sorted(root.glob("member-*")),
                registration=ensemble["registration"],
                restored_inputs=root / "restored",
                output=root / "not-written.json")
    finally:
        write_json(ledger_path, ledger)


def test_manifest_builder_resolves_bare_run_names_under_ensemble_root(
        tmp_path):
    """Production layout is ``runs/<member>/wrfout_*`` with bare CLI IDs."""
    root = tmp_path / "runs"
    root.mkdir()
    restored = tmp_path / "restored"
    restored.mkdir()
    for index, name in enumerate(RESTORED_INPUT_FILENAMES):
        (restored / name).write_bytes(f"fixture-{index}".encode("ascii"))
    registration = make_registration(
        run_minutes=30, history_minutes=30, commit="3" * 40)
    write_json(root / "n5s-preregistration.json", registration)
    unperturbed = _make_run(
        root, "unperturbed", registration, offset=0.0, perturbation=False)
    members = [
        _make_run(root, f"member-{index:02d}", registration,
                  offset=float(index + 1), perturbation=True)
        for index in range(3)
    ]
    write_json(root / "exit-status-ledger.json", {
        "schema": 1,
        "runs": [
            {"id": run.name, "exit_status": 0}
            for run in [unperturbed, *members]
        ],
        "all_succeeded": True,
    })

    evidence = build_ensemble_evidence(
        ensemble_root=root,
        unperturbed_directory=unperturbed.name,
        member_directories=[member.name for member in members],
        registration=registration, restored_inputs=restored,
        output=root / "n5s-ensemble.json")

    assert evidence["unperturbed"]["relative_path"].startswith(
        "unperturbed/")
    assert [row["id"] for row in evidence["members"]] == [
        member.name for member in members]


@pytest.mark.parametrize("rejection", [
    "too_few_members", "missing_perturbation", "hash_mismatch",
    "short_duration", "coverage_gap", "cadence_mismatch",
])
def test_schema_authority_rejects_each_required_path(
        n5s_evidence, tmp_path, rejection):
    root, _candidate_path, ensemble_base, candidate_base = n5s_evidence
    ensemble = deepcopy(ensemble_base)
    candidate = deepcopy(candidate_base)
    match rejection:
        case "too_few_members":
            ensemble["members"] = ensemble["members"][:2]
        case "missing_perturbation":
            del ensemble["members"][0]["one_ulp_perturbation"]
        case "hash_mismatch":
            ensemble["members"][0]["sha256"] = "0" * 64
        case "short_duration":
            candidate["domain_durations_seconds"]["d04"] = 1799
        case "coverage_gap":
            metric = next(name for name in ensemble["cpu_pair_distances"]
                          if name.startswith("storm_object_timing_difference:d01:"))
            del ensemble["cpu_pair_distances"][metric]
            del candidate["gpu_vs_unperturbed_distances"][metric]
        case "cadence_mismatch":
            candidate["cadence"] = "history_interval_seconds=60"
    write_json(root / "n5s-ensemble.json", ensemble)
    candidate_path = tmp_path / "candidate.json"
    write_json(candidate_path, candidate)
    with pytest.raises(ValueError):
        real74_chain.evaluate_n5s_shadow(candidate_path, root)


def test_ignored_wrfinput_disjoint_from_consumed():
    """A name in both IGNORED and MAPPED would be silently skipped by
    read_wrfinput's loop, leaving restoration incomplete — forbid overlap."""
    from gpuwm.verify.cases.real74_n5s import (ALLOWED_WRFINPUT,
                                               IGNORED_WRFINPUT,
                                               MAPPED_WRFINPUT)
    assert not (IGNORED_WRFINPUT & MAPPED_WRFINPUT)
    assert not (IGNORED_WRFINPUT & ALLOWED_WRFINPUT)
    # Spot-pins: the production real.exe standards stay ignored, and the
    # registered restoration set stays consumed.
    assert "THM" in IGNORED_WRFINPUT
    assert "SST" in IGNORED_WRFINPUT
    assert "THM" not in ALLOWED_WRFINPUT
