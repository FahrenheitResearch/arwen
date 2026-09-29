"""NetCDF is decoded in Rust, and there is no Python decoder behind it.

Both directions are asserted, because only asserting the happy one lets a
silent Python fallback survive every test:

* with ``rw_netcdf`` present the decode runs through it, and the numbers
  it produces match the C library's on a file the C library wrote;
* with ``rw_netcdf`` absent the decode REFUSES BY NAME, naming the
  binary and the command that installs it -- it does not quietly reach
  for ``netCDF4``.

The comparison is cross-lane on purpose: the fixture is written by
``netCDF4`` (the other side's real writer), never by a hand-rolled byte
layout of our own, and the result is reported as matched COUNTS rather
than a boolean, so a reader that matches nothing cannot pass by exiting
zero.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gpuwm import netcdf_bridge
from conftest import requires_netcdf_bridge

netCDF4 = pytest.importorskip("netCDF4")


def _bridge() -> Path:
    """The Rust decoder, or skip -- never a Python substitute."""

    try:
        return netcdf_bridge.resolve_netcdf_bin()
    except netcdf_bridge.NetcdfBridgeMissing:
        pytest.skip("rw_netcdf is not built; build tools/rustwx to run this")


def _write_source(path: Path) -> dict[str, np.ndarray]:
    """A CF file written by the C library -- the other lane's writer."""

    rng = np.random.default_rng(20260814)
    nt, nz, ny, nx = 3, 4, 5, 6
    fields = {
        "temperature": rng.normal(280.0, 8.0, (nt, nz, ny, nx)),
        "humidity": rng.uniform(0.0, 0.02, (nt, nz, ny, nx)),
    }
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", nt)
        dataset.createDimension("level", nz)
        dataset.createDimension("y", ny)
        dataset.createDimension("x", nx)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2026-03-01 00:00:00"
        time.calendar = "standard"
        time[:] = [0.0, 6.0, 12.0]
        level = dataset.createVariable("level", "f8", ("level",))
        level.units = "Pa"
        level[:] = [100000.0, 85000.0, 70000.0, 50000.0]
        latitude = dataset.createVariable("latitude", "f8", ("y",))
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        latitude[:] = np.linspace(30.0, 34.0, ny)
        longitude = dataset.createVariable("longitude", "f8", ("x",))
        longitude.standard_name = "longitude"
        longitude.units = "degrees_east"
        longitude[:] = np.linspace(-100.0, -95.0, nx)
        for name, values in fields.items():
            variable = dataset.createVariable(
                name, "f8", ("time", "level", "y", "x"))
            variable.units = "K" if name == "temperature" else "kg kg-1"
            variable[:] = values
    return fields


def test_mapped_source_does_not_import_a_python_netcdf_decoder():
    """The decoder is gone from the source, not merely unused at runtime.

    Parsed rather than grepped: a comment mentioning netCDF4 -- and this
    change deliberately leaves one explaining the absence -- must not be
    able to fail or pass this.
    """

    module = Path(netcdf_bridge.__file__).with_name("mapped_source.py")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"netCDF4", "xarray", "h5py", "h5netcdf", "scipy"}
    assert not (imported & banned), (
        f"gpuwm/mapped_source.py imports a Python NetCDF decoder: "
        f"{sorted(imported & banned)}")


def test_rust_decode_matches_the_c_library_on_its_own_output(tmp_path):
    """Cross-lane, by COUNTS: same variables, same shapes, same numbers."""

    bridge = _bridge()
    source = tmp_path / "source.nc"
    _write_source(source)

    dataset = netcdf_bridge.open_dataset(source, executable=bridge)
    with netCDF4.Dataset(source) as reference:
        reference_names = set(reference.variables)
        # A decoder that lists nothing must not be able to pass.
        assert reference_names, "fixture wrote no variables"
        missing = reference_names - set(dataset.variables)
        assert not missing, (
            f"rw_netcdf omitted {len(missing)} of {len(reference_names)} "
            f"variables the C library reports: {sorted(missing)}")

        compared = 0
        elements = 0
        for name in sorted(reference_names):
            expected = np.asarray(reference.variables[name][:], dtype=np.float64)
            got = np.asarray(dataset.variables[name][:], dtype=np.float64)
            assert got.shape == expected.shape, f"{name}: shape differs"
            np.testing.assert_array_equal(got, expected, err_msg=name)
            compared += 1
            elements += expected.size

    assert compared == len(reference_names)
    # Counts, not a boolean: a real number of real values was checked.
    assert compared >= 6, f"only {compared} variables compared"
    assert elements >= 3 * 4 * 5 * 6 * 2, f"only {elements} elements compared"


def test_cf_reference_times_are_decoded_by_the_bridge(tmp_path):
    """The calendar arithmetic happens in Rust, and it is right."""

    bridge = _bridge()
    source = tmp_path / "times.nc"
    _write_source(source)
    dataset = netcdf_bridge.open_dataset(source, executable=bridge)
    times = dataset.variables["time"].times()
    assert times == (
        datetime(2026, 3, 1, 0, 0, 0),
        datetime(2026, 3, 1, 6, 0, 0),
        datetime(2026, 3, 1, 12, 0, 0),
    )


def test_a_non_time_variable_refuses_to_be_read_as_times(tmp_path):
    bridge = _bridge()
    source = tmp_path / "times.nc"
    _write_source(source)
    dataset = netcdf_bridge.open_dataset(source, executable=bridge)
    with pytest.raises(netcdf_bridge.NetcdfDecodeError, match="CF reference-time"):
        dataset.variables["level"].times()


def test_missing_bridge_refuses_by_name_instead_of_falling_back(
        tmp_path, monkeypatch):
    """The other direction: no Rust, no decode, and a NAMED refusal.

    This is the assertion that makes the first one mean something.  If a
    Python decoder were still reachable, this call would quietly succeed.
    """

    source = tmp_path / "source.nc"
    _write_source(source)
    # No candidate on the ladder exists, and the override names nothing.
    monkeypatch.delenv(netcdf_bridge.NETCDF_ENV, raising=False)
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates",
                        lambda: (tmp_path / "absent" / "rw_netcdf",))

    with pytest.raises(netcdf_bridge.NetcdfBridgeMissing) as raised:
        netcdf_bridge.open_dataset(source)

    message = str(raised.value)
    assert "rw_netcdf" in message, "the refusal must name the binary"
    assert "no Python fallback" in message
    assert netcdf_bridge.NETCDF_ENV in message, (
        "the refusal must name the override that fixes it")


def test_environment_override_naming_a_missing_file_is_a_hard_error(
        tmp_path, monkeypatch):
    """An explicit override that is wrong must not fall through."""

    monkeypatch.setenv(netcdf_bridge.NETCDF_ENV,
                       str(tmp_path / "nowhere" / "rw_netcdf"))
    with pytest.raises(FileNotFoundError, match=netcdf_bridge.NETCDF_ENV):
        netcdf_bridge.find_netcdf_bin()


def test_bridge_answers_its_declared_abi_contract():
    """The installed binary speaks THIS release's schema, not any schema."""

    bridge = _bridge()
    completed = subprocess.run([str(bridge), "--abi"],
                               capture_output=True, text=True)
    assert completed.returncode == 0
    assert netcdf_bridge.INVENTORY_SCHEMA in completed.stdout
    assert netcdf_bridge.DUMP_SCHEMA in completed.stdout


@requires_netcdf_bridge
@pytest.mark.parametrize("container", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF4"])
def test_fixed_character_records_preserve_exact_shape_and_bytes(tmp_path, container):
    bridge = _bridge()
    path = tmp_path / "times.nc"
    values = np.frombuffer(b"2021-12-30_17:00:00" + b"2021-12-30_18:00:00" + b"a b\x00c" + b"\x00" * 14, dtype="S1").reshape(3, 19)
    with netCDF4.Dataset(path, "w", format=container) as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        var = ds.createVariable("Times", "S1", ("Time", "DateStrLen"), zlib=container == "NETCDF4")
        var[:] = values
    with netcdf_bridge.open_dataset(path, executable=bridge) as ds:
        got = ds.variables["Times"][:]
        assert got.shape == (3, 19)
        assert got.dtype == np.dtype("S1")
        assert got.tobytes() == values.tobytes()
        assert ds.variables["Times"][1].tobytes() == values[1].tobytes()


@requires_netcdf_bridge
def test_variable_length_text_is_not_truncated_to_characters(tmp_path):
    bridge = _bridge()
    path = tmp_path / "text.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("record", 2)
        ds.createVariable("labels", str, ("record",))[:] = np.array(["long text", "more text"], dtype=object)
    with netcdf_bridge.open_dataset(path, executable=bridge) as ds:
        with pytest.raises(netcdf_bridge.NetcdfDecodeError, match="fixed-width ASCII"):
            ds.variables["labels"][:]


def test_an_undeclared_default_fill_reads_as_missing_as_the_c_library_masks_it(tmp_path):
    """A masked element with no ``_FillValue`` is the default fill on disk.

    netCDF4 masks it; the bridge used to hand it back as the finite number
    9.97e36.  Cross-lane: the C library's masked cells are the bridge's NaN
    cells, and a declared ``_FillValue`` still replaces the default.
    """

    bridge = _bridge()
    source = tmp_path / "fill.nc"
    with netCDF4.Dataset(source, "w") as out:
        out.createDimension("x", 4)
        plain = out.createVariable("plain", "f4", ("x",))
        plain[:] = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
        double = out.createVariable("double", "f8", ("x",))
        double[:] = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[1, 0, 0, 0])
        declared = out.createVariable("declared", "f4", ("x",), fill_value=-999.0)
        declared[:] = np.array([9.969209968386869e36, -999.0, 3.0, 4.0], dtype=np.float32)
    dataset = netcdf_bridge.open_dataset(source, executable=bridge)
    with netCDF4.Dataset(source) as reference:
        for name in ("plain", "double", "declared"):
            expected = np.ma.getmaskarray(reference.variables[name][:])
            got = np.isnan(np.asarray(dataset.variables[name][:], dtype=np.float64))
            np.testing.assert_array_equal(got, expected, err_msg=name)
            assert expected.sum() == 1, name


def test_a_reader_older_than_the_default_fill_rule_is_masked_the_same():
    """An installed ``rw_netcdf`` that predates the rule says nothing about it."""

    variable = type("V", (), {"attributes": {}, "stored_dtype": "F32"})()
    stored = np.array([1.0, float(np.float32(9.969209968386869e36))])
    masked = netcdf_bridge._mask_default_fill(variable, stored, {"applied": True})
    assert masked[0] == 1.0 and np.isnan(masked[1])
    # a reader that applied the rule itself is not second-guessed
    kept = netcdf_bridge._mask_default_fill(
        variable, stored, {"applied": True, "default_fill_rule": True})
    assert kept[1] > 9e36
    declared = type("V", (), {"attributes": {"_FillValue": -1.0}, "stored_dtype": "F32"})()
    assert netcdf_bridge._mask_default_fill(declared, stored, {})[1] > 9e36


def test_an_older_reader_s_unpacked_default_fill_is_found_where_the_c_library_masks(tmp_path):
    """A packed field's default fill arrives unpacked from a reader without the rule.

    The fallback compared the stored -32767 against values the reader had
    already multiplied by ``scale_factor`` and shifted by ``add_offset``,
    so a packed short never matched.  Cross-lane: the reader's own unpacked,
    unmasked answer (all a reader without the rule gives for a variable
    with no fill attributes), masked here, masks exactly the cells netCDF4
    masks, and the same holds through an explicit unit transform.
    """

    import netCDF4

    bridge = _bridge()
    source = tmp_path / "packed.nc"
    with netCDF4.Dataset(source, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("x", 5)
        packed = out.createVariable("packed", "i2", ("x",))
        packed.scale_factor = np.float32(0.01)
        packed.add_offset = np.float32(273.15)
        packed.set_auto_maskandscale(False)
        packed[:] = np.array([100, -32767, 250, -200, 0], dtype=np.int16)
    with netCDF4.Dataset(source) as reference:
        expected = np.ma.getmaskarray(reference.variables["packed"][:])
    assert expected.tolist() == [False, True, False, False, False]

    dataset = netcdf_bridge.open_dataset(source, executable=bridge)
    variable = dataset.variables["packed"]
    record = {"applied": True,
              "scale_factor": float(np.float32(0.01)),
              "add_offset": float(np.float32(273.15))}
    for transform in (None, (1.8, -459.67)):
        # What a reader without the rule hands on: unpacked, nothing masked
        # but declared markers, of which this variable has none.
        handed, _ = dataset._decode("packed", raw=True, scale=True,
                                    unit_transform=transform)
        assert np.isfinite(handed).all()
        masked = netcdf_bridge._mask_default_fill(
            variable, handed, record, unit_transform=transform)
        np.testing.assert_array_equal(np.isnan(masked), expected)
        # the current reader agrees on its own
        current, _ = dataset._decode("packed", unit_transform=transform)
        np.testing.assert_array_equal(np.isnan(current), expected)
