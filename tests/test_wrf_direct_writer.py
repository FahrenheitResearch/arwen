"""The wrfinput/wrfbdy export writes on the Rust NetCDF writer BY DEFAULT.

2.5.0's law is that NetCDF read and write are Drew's Rust.  The
WPS-to-WRF handoff pair is the largest NetCDF write in the tree -- every
gridded field of every domain, on the bare default of ``gpuwm prep``,
``gpuwm go``, ``gpuwm-wrf-init``, ``rw-wps`` and ``gpuwm
import-namelist`` -- and it was the last one still going out through
netCDF4.  These tests pin the flip itself, in the shape the wrfout tape's
own flip is pinned (``tests/test_wrfout_engine.py``):

* the DEFAULT engine is ``rust`` -- no flag, no env -- and it writes the
  classic CDF-2 container through ``gpuwm.io.nc_writer_bridge``, which is
  the container stock WRF's own ``real.exe`` writes at
  ``io_form_input = 2``;
* the export's closing verification -- the finiteness read-back over
  every float variable -- runs on Rust too, through the writer crate's
  own sweep and the ``rw_netcdf`` reader, never on netCDF4;
* the netCDF4 engine remains reachable ONLY as an explicit escape
  (``GPUWM_WRFINPUT_WRITER=python``), documented as a workaround;
* a box where the Rust library is missing gets a refusal naming the
  concrete breakage and both remedies -- never a silent engine downgrade.

The two engines' equivalence on a real contract is
``test_the_two_engines_write_the_same_file``, which is the dual-write
discipline ``tests/test_wrfout_dual_write.py`` established.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm import wrf_direct
from gpuwm.io import nc_writer_bridge

_RUST_UNAVAILABLE = nc_writer_bridge.unavailable_reason()

# Small in the horizontal, the contract's own extent in the vertical and
# the soil: the frozen v4.6.1 prototypes for the base-profile scalars are
# 49/50/4 long, so those axes are what they are and the horizontal is
# free.
NX, NY, NZ = 4, 3, 49
SOIL = 4

_GEOMETRY = {
    "center_lat": 39.5, "center_lon": -84.25,
    "truelat1": 30.0, "truelat2": 60.0,
    "ref_lat": 39.5, "stand_lon": -84.25,
    "map_proj": "lambert",
}


def _require_rust_writer():
    if _RUST_UNAVAILABLE:
        pytest.skip(f"the Rust NetCDF writer library is not built: "
                    f"{_RUST_UNAVAILABLE}")


def _contract():
    return wrf_direct._load_contract()["wrfinput"]


def _updates():
    return wrf_direct._global_updates(
        valid_time=datetime(2026, 8, 18, 0, 0, 0),
        nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0, dt=15.0,
        geometry=_GEOMETRY)


def _fields(*, poison=None):
    """A handful of real field names at this test's extents."""
    mass = np.arange(NZ * NY * NX, dtype=np.float64).reshape(NZ, NY, NX)
    surface = np.arange(NY * NX, dtype=np.float64).reshape(NY, NX) * 0.5
    fields = {
        "T": mass * 1e-3 + 290.0,
        "QVAPOR": mass * 1e-6,
        "MU": surface,
        "HGT": surface * 12.0,
        "TSK": surface + 280.0,
    }
    if poison is not None:
        fields["T"] = fields["T"].copy()
        fields["T"][2, 1, 1] = poison
    return fields


def _write(path, *, engine=None, poison=None):
    contract = _contract()
    dimensions = wrf_direct._dimensions(
        contract, nx=NX, ny=NY, nz=NZ, num_soil_layers=SOIL)
    wrf_direct._write_wrfinput(
        path, contract, dimensions, _updates(), _fields(poison=poison),
        "2026-08-18_00:00:00", engine=engine)
    return contract


def _validate(path, contract, *, engine=None, expect_attrs=True):
    return wrf_direct._validate_file(
        path, contract, nx=NX, ny=NY, nz=NZ, num_soil_layers=SOIL,
        expected_global_attributes=(
            wrf_direct._domain_global_attributes(_updates())
            if expect_attrs else None),
        engine=engine)


# ---------------------------------------------------------------------------
# Engine resolution: the default is the flip.
# ---------------------------------------------------------------------------

def test_the_default_engine_is_rust(monkeypatch):
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    assert wrf_direct.resolve_wrfinput_engine() == "rust"
    assert wrf_direct.resolve_wrfinput_engine(None) == "rust"


def test_the_env_escape_selects_python(monkeypatch):
    monkeypatch.setenv(wrf_direct.WRFINPUT_WRITER_ENV, "python")
    assert wrf_direct.resolve_wrfinput_engine() == "python"


def test_the_explicit_argument_wins_over_the_env(monkeypatch):
    monkeypatch.setenv(wrf_direct.WRFINPUT_WRITER_ENV, "python")
    assert wrf_direct.resolve_wrfinput_engine("rust") == "rust"


def test_an_unknown_engine_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(wrf_direct.WRFINPUT_WRITER_ENV, "fortran")
    with pytest.raises(ValueError, match="fortran"):
        wrf_direct.resolve_wrfinput_engine()


# ---------------------------------------------------------------------------
# The bare default writes the classic container through the Rust seam.
# ---------------------------------------------------------------------------

def test_the_bare_default_writes_the_classic_container(tmp_path, monkeypatch):
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfinput_d01"
    contract = _write(path)
    with netCDF4.Dataset(path) as ds:
        # The container IS the flip: HDF5 NETCDF4 came from the C
        # library, 64-bit-offset classic comes from the Rust writer --
        # and classic is what stock WRF's own real.exe writes.
        assert ds.file_format == "NETCDF3_64BIT_OFFSET"
        assert list(ds.variables) == [
            item["name"] for item in contract["variables"]]
        assert ds.dimensions["Time"].isunlimited()
        assert len(ds.dimensions["bottom_top"]) == NZ
        assert len(ds.dimensions["west_east"]) == NX
        stamp = b"".join(np.asarray(ds.variables["Times"][:]).ravel())
        assert stamp == b"2026-08-18_00:00:00"
        theta = ds.variables["T"]
        theta.set_auto_maskandscale(False)
        assert np.asarray(theta[0]).tobytes() == \
            _fields()["T"].astype("f4").tobytes()
        # A frozen prototype the export did not supply survives the flip.
        assert float(np.asarray(ds.variables["P_TOP"][:]).ravel()[0]) > 0.0
        # And no attribute this pair never carried has appeared.
        assert "GPUWM_WRITE_COMPLETE" not in ds.ncattrs()


def test_the_bare_default_never_opens_netcdf4(tmp_path, monkeypatch):
    """The positive claim, not just the absence of a flag.

    The concrete breakage this guards: an export that quietly reopened
    the file with the C library for its own verification would keep a
    Python NetCDF read on the preparation path while every visible sign
    said Rust.
    """
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()

    class _Forbidden:
        def __getattr__(self, name):
            raise AssertionError(
                f"the default wrfinput path reached netCDF4.{name}")

    monkeypatch.setattr(wrf_direct, "netCDF4", _Forbidden())
    path = tmp_path / "wrfinput_d01"
    contract = _write(path)
    receipt = _validate(path, contract)
    assert receipt["bytes"] == path.stat().st_size
    assert len(receipt["sha256"]) == 64


def test_a_missing_rust_library_refuses_with_both_remedies(
        tmp_path, monkeypatch):
    """No silent downgrade: the refusal names the breakage and the escape.

    The concrete breakage a fallback would cause: the handoff pair's
    container would silently depend on which box built what, and the
    'default' writer would quietly stop being the one that is tested.
    """
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    monkeypatch.setattr(
        nc_writer_bridge, "unavailable_reason",
        lambda: "FileNotFoundError: the Rust NetCDF writer library was "
                "not found (simulated)")
    path = tmp_path / "wrfinput_d01"
    with pytest.raises(RuntimeError) as refusal:
        _write(path)
    message = str(refusal.value)
    assert "Rust NetCDF writer" in message
    assert wrf_direct.WRFINPUT_WRITER_ENV in message   # the documented escape
    assert "workaround" in message                     # named as one
    assert not path.exists()


# ---------------------------------------------------------------------------
# The read-back verification survives the flip, in Rust.
# ---------------------------------------------------------------------------

def test_a_non_finite_field_is_named_by_the_default_read_back(
        tmp_path, monkeypatch):
    """The gate this export has always ended with: a NaN never reaches
    stock WRF, where it surfaces hours later as a blown-up integration
    with no trace of where it came from."""
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfinput_d01"
    contract = _write(path, poison=np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        _validate(path, contract)


def test_the_read_back_refuses_a_truncated_file_rather_than_passing_it(
        tmp_path, monkeypatch):
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfinput_d01"
    contract = _write(path)
    payload = path.read_bytes()
    path.write_bytes(payload[:-64])
    with pytest.raises(Exception):
        _validate(path, contract)


def test_global_attribute_drift_is_still_caught(tmp_path, monkeypatch):
    """The Rust reader hands attributes back as plain JSON numbers, so
    the contract's declared width is what the comparison restores them
    to.  Without that, every float32 global would read as drift."""
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfinput_d01"
    contract = _write(path)
    _validate(path, contract)                       # clean
    drifted = dict(wrf_direct._domain_global_attributes(_updates()))
    drifted["CEN_LAT"] = 12.5
    with pytest.raises(ValueError, match="drift"):
        wrf_direct._validate_file(
            path, contract, nx=NX, ny=NY, nz=NZ, num_soil_layers=SOIL,
            expected_global_attributes=drifted)


# ---------------------------------------------------------------------------
# The workaround, and the two engines' equivalence.
# ---------------------------------------------------------------------------

def test_the_python_escape_still_writes_the_frozen_hdf5_container(
        tmp_path, monkeypatch):
    monkeypatch.setenv(wrf_direct.WRFINPUT_WRITER_ENV, "python")
    path = tmp_path / "wrfinput_d01"
    contract = _write(path)
    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == "NETCDF4"
    _validate(path, contract)


BNX, BNY, BNZ, BDY = 4, 3, 5, 5

_WRF_BY_LOGICAL = {"u": "U", "v": "V", "phi": "PH", "theta": "T",
                   "mu": "MU", "qv": "QVAPOR"}
_SUFFIX_BY_SIDE = {"west": "XS", "east": "XE", "south": "YS", "north": "YE"}


class _BoundaryCache:
    """The slice of ``PreparedCache`` ``_wrfbdy_fields`` asks for.

    The shapes are DERIVED from the frozen contract rather than guessed:
    each logical/side pair's WRF-side extents come from the contract's
    own dimension list, and this inverts ``_lbc_to_wrf``'s transpose to
    hand back what the prepared cache would have stored.  Deterministic
    per key, so the two engines are handed identical arrays and any
    difference downstream is the writer's.
    """

    def __init__(self):
        contract = wrf_direct._load_contract()["wrfbdy"]
        dimensions = wrf_direct._dimensions(
            contract, nx=BNX, ny=BNY, nz=BNZ, num_soil_layers=SOIL)
        self._target = {
            item["name"]: tuple(int(dimensions[dim])
                                for dim in item["dimensions"][1:])
            for item in contract["variables"]
        }

    def array(self, key):
        _lbc, index, logical, side, kind = key.split("/")
        name = f"{_WRF_BY_LOGICAL[logical]}_B{_SUFFIX_BY_SIDE[side]}"
        target = self._target[name]
        if logical == "mu":
            shape = ((1, target[1], target[0]) if side in {"west", "east"}
                     else (1, target[0], target[1]))
        elif side in {"west", "east"}:
            shape = (target[1], target[2], target[0])
        else:
            shape = (target[1], target[0], target[2])
        seed = (int(index) * 7 + len(name)) % 97
        count = int(np.prod(shape))
        return (np.arange(count, dtype=np.float64).reshape(shape)
                + seed) * 1e-2


def _write_bdy(path, *, engine=None):
    contract = wrf_direct._load_contract()["wrfbdy"]
    dimensions = wrf_direct._dimensions(
        contract, nx=BNX, ny=BNY, nz=BNZ, num_soil_layers=SOIL)
    times = [datetime(2026, 8, 18, hour) for hour in (0, 1, 2)]
    wrf_direct._write_wrfbdy(
        path, contract, dimensions,
        wrf_direct._global_updates(
            valid_time=times[0], nx=BNX, ny=BNY, nz=BNZ,
            dx=3000.0, dy=3000.0, dt=15.0, geometry=_GEOMETRY),
        _BoundaryCache(), times, 3600, engine)
    return contract


def test_the_boundary_file_is_written_and_verified_on_the_default_engine(
        tmp_path, monkeypatch):
    """wrfbdy is the multi-RECORD half of the pair: three boundary times,
    115 variables, three of them character.  The classic writer lays
    records out by stride, so a wrong stride here would hand WRF shifted
    boundary tendencies rather than an error."""
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfbdy_d01"
    contract = _write_bdy(path)
    wrf_direct._validate_file(
        path, contract, nx=BNX, ny=BNY, nz=BNZ, num_soil_layers=SOIL)
    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == "NETCDF3_64BIT_OFFSET"
        assert len(ds.dimensions["Time"]) == 3
        stamps = ["".join(row.astype(str))
                  for row in np.asarray(ds.variables["Times"][:])]
        assert stamps == ["2026-08-18_00:00:00", "2026-08-18_01:00:00",
                          "2026-08-18_02:00:00"]
        # The NEXT-time metadata record is one interval ahead of the last
        # boundary time, on the last record -- the value a wrong record
        # stride would smear.
        nxt = ds.variables[
            "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"]
        assert "".join(np.asarray(nxt[2]).astype(str)) == \
            "2026-08-18_03:00:00"


def test_the_two_engines_write_the_same_boundary_file(tmp_path, monkeypatch):
    _require_rust_writer()
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    rust = tmp_path / "wrfbdy_rust"
    python = tmp_path / "wrfbdy_python"
    _write_bdy(rust, engine="rust")
    _write_bdy(python, engine="python")
    with netCDF4.Dataset(rust) as left, netCDF4.Dataset(python) as right:
        assert list(left.variables) == list(right.variables)
        for name, variable in left.variables.items():
            other = right.variables[name]
            variable.set_auto_maskandscale(False)
            other.set_auto_maskandscale(False)
            assert np.asarray(variable[:]).tobytes() == \
                np.asarray(other[:]).tobytes(), name


def test_the_two_engines_write_the_same_file(tmp_path, monkeypatch):
    """Same inventory, same dimensions, same attributes, same bits.

    Only the container differs, which is the one thing the flip is
    allowed to change.
    """
    _require_rust_writer()
    monkeypatch.delenv(wrf_direct.WRFINPUT_WRITER_ENV, raising=False)
    rust = tmp_path / "wrfinput_rust"
    python = tmp_path / "wrfinput_python"
    contract = _write(rust, engine="rust")
    _write(python, engine="python")
    _validate(rust, contract, engine="rust")
    _validate(python, contract, engine="python")

    with netCDF4.Dataset(rust) as left, netCDF4.Dataset(python) as right:
        assert list(left.variables) == list(right.variables)
        assert {name: len(dim) for name, dim in left.dimensions.items()} == \
            {name: len(dim) for name, dim in right.dimensions.items()}
        assert sorted(left.ncattrs()) == sorted(right.ncattrs())
        for name in left.ncattrs():
            got, want = left.getncattr(name), right.getncattr(name)
            assert np.asarray(got).dtype == np.asarray(want).dtype, name
            assert np.array_equal(np.asarray(got), np.asarray(want)), name
        for name, variable in left.variables.items():
            other = right.variables[name]
            assert variable.dimensions == other.dimensions, name
            assert variable.dtype == other.dtype, name
            assert variable.ncattrs() == other.ncattrs(), name
            for attribute in variable.ncattrs():
                assert variable.getncattr(attribute) == \
                    other.getncattr(attribute), (name, attribute)
            variable.set_auto_maskandscale(False)
            other.set_auto_maskandscale(False)
            assert np.asarray(variable[:]).tobytes() == \
                np.asarray(other[:]).tobytes(), name
