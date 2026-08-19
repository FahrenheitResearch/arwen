"""One reader for both models, and the units it refuses to guess.

The half of this module that reads stored variables is exercised against real
history files written here.  The half that calls the mandated science core is
exercised through a stand-in handle: what needs pinning is not that the core
computes ``t2`` -- that is its job and its tests -- but that this module
converts what the core returns into seam units from a *declared* table and
refuses the result when the declaration is wrong.  A silent 273.15 is the
kind of error that survives an entire campaign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm.verify.obs import model_source
from gpuwm.verify.obs.contracts import SEAM_BOUNDS


def _write_frame(path, *, valid_time, reflectivity, rain, landmask=None):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("bottom_top", reflectivity.shape[0])
        dataset.createDimension("south_north", reflectivity.shape[1])
        dataset.createDimension("west_east", reflectivity.shape[2])
        refl = dataset.createVariable(
            "REFL_10CM", "f4",
            ("Time", "bottom_top", "south_north", "west_east"))
        refl[0] = reflectivity
        rainnc = dataset.createVariable(
            "RAINNC", "f4", ("Time", "south_north", "west_east"))
        rainnc[0] = rain
        if landmask is not None:
            land = dataset.createVariable(
                "LANDMASK", "f4", ("Time", "south_north", "west_east"))
            land[0] = landmask
        dataset.setncattr("SIMULATION_START_DATE", valid_time)


def _run_directory(tmp_path, hours=(12, 13), landmask=None):
    generator = np.random.default_rng(3)
    for index, hour in enumerate(hours):
        # Colons are legal in WRF's own naming and illegal on this platform's
        # filesystem; the discovery regex accepts both spellings for exactly
        # that reason, and the underscore spelling is what gets written here.
        name = f"wrfout_d01_2026-08-03_{hour:02d}_00_00"
        _write_frame(
            tmp_path / name, valid_time=f"2026-08-03_{hour:02d}:00:00",
            reflectivity=generator.uniform(-20.0, 60.0, size=(4, 6, 7)),
            rain=np.full((6, 7), 2.5 * index, dtype=np.float64),
            landmask=landmask)
    return tmp_path


class _StandInCore:
    """A handle-shaped stand-in for the science core, for the units pins only."""

    def __init__(self, values):
        self._values = dict(values)

    class WrfFile:  # noqa: D401 - shape only
        def __init__(self, path):
            self.path = path
            self.dx = 3000.0

    def getvar(self, _handle, name, meta=False, **_kwargs):
        if name not in self._values:
            raise KeyError(name)
        return self._values[name]

    def latlon_coords(self, _handle):
        latitude = np.full((6, 7), 37.0)
        longitude = np.full((6, 7), -97.0)
        return latitude, longitude

    def ll_to_xy(self, _handle, latitude, longitude):
        return np.array([2.5]), np.array([3.5])


# --------------------------------------------------------------------------
# the pin on the mandated core
# --------------------------------------------------------------------------


def test_the_science_core_pin_matches_what_is_installed():
    core = model_source.require_science_core()
    assert hasattr(core, "getvar") and hasattr(core, "WrfFile")


def test_the_pin_here_is_the_same_pin_the_tree_already_carries():
    from tools.flagship.products import PINNED_WRF_RUST_VERSION

    assert model_source.PINNED_WRF_RUST_VERSION == PINNED_WRF_RUST_VERSION


def test_the_window_accepts_a_newer_core_and_still_refuses_an_older_one():
    """The pin is a WINDOW, not an equality.

    An exact ``==0.2.35`` did two harmful things at once: pip DOWNGRADED a
    user who already ran a newer wrf-rust, and these runtime checks refused
    the newer core if they kept it.  Both directions are asserted here so a
    future edit cannot quietly restore the equality.
    """
    from gpuwm import science_core

    assert science_core.version_supported("0.2.35")   # the floor holds
    assert science_core.version_supported("0.2.38")   # the reported case
    assert science_core.version_supported("0.2.39")   # the certified core
    assert science_core.version_supported("0.2.50")   # a newer core stays in
    assert science_core.version_supported("0.2.99")
    assert not science_core.version_supported("0.2.34")  # below the floor
    assert not science_core.version_supported("0.3.0")   # at the ceiling
    assert not science_core.version_supported("1.0.0")
    # An unnameable reader is refused rather than silently accepted.
    assert not science_core.version_supported(None)
    assert not science_core.version_supported("0.0.1-not-the-pin")


def test_the_install_floor_outranks_the_runtime_floor_and_stays_separate():
    """Two floors, two breakages, and neither may absorb the other.

    ``SCIENCE_CORE_INSTALL_FLOOR`` (0.2.39) is about what pip may
    RESOLVE: below it a 3.14 box gets a wheel-less core, falls back to an
    sdist whose pyo3 caps at 3.13, and -- because pip fails a whole
    resolution when one requirement fails -- installs no gpuwm at all.
    ``SCIENCE_CORE_FLOOR`` (0.2.35) is about a core that is ALREADY
    installed, and no breakage has ever been named for 0.2.35..0.2.38: a
    0.2.38 sitting on a 3.13 box renders every product it ever did.

    Collapsing them into one number is the tempting edit, and it is a
    refusal with nothing behind it -- gpuwm would start rejecting a
    working core to enforce an install-time wheel-matrix fact.  This
    asserts the split survives, in both directions.
    """
    from gpuwm import science_core

    runtime = science_core.version_tuple(science_core.SCIENCE_CORE_FLOOR)
    install = science_core.version_tuple(
        science_core.SCIENCE_CORE_INSTALL_FLOOR)
    assert runtime is not None and install is not None
    # The install floor is the STRICTER of the two, and strictly so.
    assert runtime < install
    # The gap between them is the range that installs no longer resolve
    # but a running box is still judged on its merits.
    assert science_core.version_supported("0.2.38")
    assert science_core.SCIENCE_CORE_REQUIREMENT == (
        f"wrf-rust>={science_core.SCIENCE_CORE_INSTALL_FLOOR},"
        f"<{science_core.SCIENCE_CORE_CEILING}")


def test_the_certified_release_sits_inside_the_window_it_is_recorded_beside():
    """The release the suites run on cannot fall out of the declared window.

    ``SCIENCE_CORE_CERTIFIED`` records which wrf-rust the recorded test
    counts belong to; the floor/ceiling record what a user is allowed to
    install.  Nothing forces those three strings to agree, and the failure
    they would produce is silent: a tree that certifies against a release
    its own requirement string refuses to install, so `pip install
    gpuwm[render]` hands the user a core no suite was ever run on.  This is
    the arithmetic that catches it -- it is a consistency check between
    three constants in one module, not a gate on anybody's environment.
    """
    from gpuwm import science_core

    assert science_core.version_supported(science_core.SCIENCE_CORE_CERTIFIED)
    floor = science_core.version_tuple(science_core.SCIENCE_CORE_FLOOR)
    certified = science_core.version_tuple(
        science_core.SCIENCE_CORE_CERTIFIED)
    assert floor is not None and certified is not None
    # The floor is the OLDEST certified release, so it cannot outrank the
    # one actually exercised.
    assert floor <= certified


def test_the_requirement_string_is_the_one_pyproject_installs():
    """One window, written once: the extra and the runtime check agree.

    ``SCIENCE_CORE_REQUIREMENT`` is the INSTALL window -- floor 0.2.39,
    the oldest release with a wheel for every interpreter gpuwm
    supports -- because that string is what pip is handed.  The runtime
    floor is older and deliberately so; both axes, and the environment
    marker this line used to carry, are asserted in
    tests/test_render_extra_python_ceiling.py.  Splitting on ``;`` is
    kept so a future marker cannot make this test fail for a reason it
    does not name.
    """
    import tomllib

    from gpuwm import science_core

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    render = data["project"]["optional-dependencies"]["render"]
    declared = [item.split(";")[0].strip()
                for item in render if item.startswith("wrf-rust")]
    assert declared == [science_core.SCIENCE_CORE_REQUIREMENT]


# --------------------------------------------------------------------------
# frame discovery
# --------------------------------------------------------------------------


def test_frames_are_discovered_and_keyed_by_seam_timestamp(tmp_path):
    frames = model_source.discover_frames(_run_directory(tmp_path), "d01")
    assert sorted(frames) == ["2026-08-03T12:00:00", "2026-08-03T13:00:00"]


def test_a_run_directory_with_no_frames_is_a_refusal(tmp_path):
    with pytest.raises(ValueError, match="holds no d01 history frames"):
        model_source.discover_frames(tmp_path, "d01")


def test_an_unparseable_frame_name_is_a_refusal_not_a_skip(tmp_path):
    (tmp_path / "wrfout_d01_garbage").write_bytes(b"")
    with pytest.raises(ValueError, match="invalid history-frame name"):
        model_source.discover_frames(tmp_path, "d01")


# --------------------------------------------------------------------------
# stored fields, read for real
# --------------------------------------------------------------------------


def test_composite_reflectivity_is_the_column_max_of_the_stored_field(tmp_path):
    directory = _run_directory(tmp_path)
    source = model_source.WrfHistorySource(directory, domain="d01")
    composite = source.composite_reflectivity("2026-08-03T12:00:00")

    path = source.frame_path("2026-08-03T12:00:00")
    with netCDF4.Dataset(path) as dataset:
        expected = np.max(np.asarray(dataset.variables["REFL_10CM"][0]),
                          axis=0)
    assert composite.shape == (6, 7)
    assert np.allclose(composite, expected)


def test_precipitation_is_a_run_total_and_differences_are_positive(tmp_path):
    directory = _run_directory(tmp_path)
    source = model_source.WrfHistorySource(directory, domain="d01")
    early = source.precipitation_accumulation("2026-08-03T12:00:00")
    late = source.precipitation_accumulation("2026-08-03T13:00:00")
    assert np.all(early == 0.0)
    assert np.allclose(late - early, 2.5)


def test_valid_times_are_ascending_and_the_reader_records_its_choices(tmp_path):
    source = model_source.WrfHistorySource(_run_directory(tmp_path),
                                           domain="d01")
    assert source.valid_times() == ("2026-08-03T12:00:00",
                                    "2026-08-03T13:00:00")
    record = source.record()
    assert record["reflectivity_variable"] == "REFL_10CM"
    assert record["reflectivity_reduction"] == "column maximum over k"
    assert record["science_core"] == "wrf-rust"
    assert record["unit_conversions"]["mslp"]["factor"] == 100.0
    assert record["cross_check_operator_pins"] == {"use_varint": False,
                                                   "use_liqskin": False}


def test_asking_for_a_frame_that_does_not_exist_is_a_refusal(tmp_path):
    source = model_source.WrfHistorySource(_run_directory(tmp_path),
                                           domain="d01")
    with pytest.raises(ValueError, match="no d01 frame at"):
        source.composite_reflectivity("2026-08-03T20:00:00")


# --------------------------------------------------------------------------
# units are declared, converted once, and checked
# --------------------------------------------------------------------------


def _with_stand_in_core(tmp_path, values):
    source = model_source.WrfHistorySource(_run_directory(tmp_path),
                                           domain="d01")
    source._wrf = _StandInCore(values)
    source._handles.clear()
    return source


def test_a_declared_conversion_is_applied_once(tmp_path):
    source = _with_stand_in_core(tmp_path, {
        "t2": np.full((6, 7), 291.0),
        "dp2m": np.full((6, 7), 12.0),      # Celsius, per the declared table
        "slp": np.full((6, 7), 1013.0),     # hPa, per the declared table
        "wspd10": np.full((6, 7), 6.0),
    })
    assert np.allclose(
        source.surface_field("2026-08-03T12:00:00", "temperature_2m"), 291.0)
    assert np.allclose(
        source.surface_field("2026-08-03T12:00:00", "dewpoint_2m"),
        285.15)
    assert np.allclose(
        source.surface_field("2026-08-03T12:00:00", "mslp"), 101300.0)
    assert np.allclose(
        source.surface_field("2026-08-03T12:00:00", "wind_speed_10m"), 6.0)


def test_a_wrong_conversion_fails_loudly_on_the_first_frame(tmp_path):
    # The core returns dewpoint in Kelvin while the table declares Celsius:
    # the result lands 273 K too high and the seam bound catches it.
    source = _with_stand_in_core(tmp_path, {
        "dp2m": np.full((6, 7), 285.0),
    })
    with pytest.raises(ValueError, match="outside the seam bound"):
        source.surface_field("2026-08-03T12:00:00", "dewpoint_2m")


def test_the_seam_bound_named_in_the_refusal_is_the_registered_one(tmp_path):
    source = _with_stand_in_core(tmp_path, {"t2": np.full((6, 7), 12.0)})
    low, high = SEAM_BOUNDS["temperature_2m"]
    with pytest.raises(ValueError) as excinfo:
        source.surface_field("2026-08-03T12:00:00", "temperature_2m")
    assert f"[{low:g}, {high:g}]" in str(excinfo.value)


def test_a_non_finite_diagnostic_is_refused(tmp_path):
    source = _with_stand_in_core(
        tmp_path, {"t2": np.full((6, 7), np.nan)})
    with pytest.raises(ValueError, match="non-finite"):
        source.surface_field("2026-08-03T12:00:00", "temperature_2m")


def test_a_variable_with_no_declared_conversion_is_refused(tmp_path):
    source = _with_stand_in_core(tmp_path, {})
    with pytest.raises(ValueError, match="no declared conversion"):
        source.surface_field("2026-08-03T12:00:00", "visibility")


def test_the_cross_check_operator_comes_from_the_core_not_from_here(tmp_path):
    source = _with_stand_in_core(
        tmp_path, {"maxdbz": np.full((6, 7), 44.0)})
    assert np.allclose(source.core_maxdbz("2026-08-03T12:00:00"), 44.0)


def test_the_station_locator_goes_through_the_cores_projection(tmp_path):
    source = _with_stand_in_core(tmp_path, {})
    locate = source.station_locator()
    position = locate("KXYZ", 37.1, -97.2)
    assert (position.station_id, position.x, position.y) == ("KXYZ", 2.5, 3.5)


def test_the_grid_carries_terrain_and_this_domains_spacing(tmp_path):
    source = _with_stand_in_core(
        tmp_path, {"terrain": np.full((6, 7), 320.0)})
    grid = source.grid()
    assert grid.shape == (6, 7)
    assert grid.dx_m == 3000.0
    assert np.allclose(grid.terrain_m, 320.0)


def test_the_land_mask_is_read_off_the_run_and_is_a_flag(tmp_path):
    # The registered station admission keeps land points only, so the mask
    # has to come off the arm being scored rather than off a static file
    # somebody remembered to point at.
    mask = np.zeros((6, 7), dtype=np.float64)
    mask[2:5, 1:4] = 1.0
    source = model_source.WrfHistorySource(
        _run_directory(tmp_path, landmask=mask), domain="d01")
    land = source.land_mask()
    assert land.dtype == np.bool_
    assert land.shape == (6, 7)
    assert np.array_equal(land, mask >= 0.5)


def test_a_land_mask_that_is_not_a_flag_is_refused(tmp_path):
    # A fractional land fraction read as a 0/1 flag would silently admit
    # every coastal station, which is the opposite of what the rule says.
    fractional = np.full((6, 7), 0.35, dtype=np.float64)
    source = model_source.WrfHistorySource(
        _run_directory(tmp_path, landmask=fractional), domain="d01")
    with pytest.raises(ValueError, match="0/1 flag"):
        source.land_mask()


def test_a_run_with_no_land_mask_says_so_rather_than_inventing_one(tmp_path):
    source = model_source.WrfHistorySource(_run_directory(tmp_path),
                                           domain="d01")
    with pytest.raises(Exception):
        source.land_mask()
