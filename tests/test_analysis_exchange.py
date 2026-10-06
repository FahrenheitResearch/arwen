"""A WRF input file an analysis edits in place comes back as it arrived.

The exchange contract, at the byte level.  A CPU analysis that edits a WRF
file in place hands this model a file and expects the same file back
wherever nothing changed.  The fixture here is shaped like such a file:
classic 64-bit-offset container written by the netCDF C library, one Time
record, every variable the reader requires, the two variables the analysis
carries that this model has no state for (``REFL_10CM``,
``RAD_TTEN_DFI``), integer and character variables, and a handful of the
names the reader skips.

Bytes are compared, not values with a tolerance: a low bit lost on the way
through the reader, a variable written back in a different type, a moved
data section or a dropped variable are the defects this exists to catch,
and each of them survives a tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm import netcdf_bridge
from gpuwm.config import RunConfig, soil_layer_count
from gpuwm.ingest import wrfinput as wi
from gpuwm.io import analysis_exchange as exchange
from gpuwm.io import nc_writer_bridge as bridge

NZ, NY, NX = 3, 4, 6

#: Names the fixture carries beyond the reader's required inventory, with
#: the dimensions each takes after ``Time``.
_EXTRA_DIMENSIONS = {
    # The analysis program's own carried variables.
    "REFL_10CM": wi._MASS_3D_DIMS,
    "RAD_TTEN_DFI": wi._MASS_3D_DIMS,
    # One of each other pass-through kind.
    "RAD_TTEN_DFI_1": wi._MASS_3D_DIMS,
    "UP_HELI_MAX": wi._MASS_2D_DIMS,
    "QVG": wi._MASS_2D_DIMS,
    "OA1": wi._MASS_2D_DIMS,
    "BF": ("bottom_top",),
    # WRF input variables this model does not consume.
    "THM": wi._MASS_3D_DIMS,
    "P_HYD": wi._MASS_3D_DIMS,
    "CLDFRA": wi._MASS_3D_DIMS,
    "ZS": ("soil_layers_stag",),
    "DZS": ("soil_layers_stag",),
    "RDX": (),
    "RDY": (),
}
_INTEGER_VARIABLES = ("ISLTYP", "IVGTYP", "ITIMESTEP")
_PASSTHROUGH_IN_FIXTURE = ("REFL_10CM", "RAD_TTEN_DFI", "RAD_TTEN_DFI_1",
                           "UP_HELI_MAX", "QVG", "OA1", "BF")


def _cfg() -> RunConfig:
    return RunConfig(nx=NX, ny=NY, nz=NZ, dx=3000., dy=3000., ztop=10000.,
                     dt=1., run_seconds=0., moist=True, mp_physics=8,
                     sf_surface_physics=3, num_soil_layers=9,
                     sf_sfclay_physics=5, bl_pbl_physics=5)


def _dimensions(cfg) -> dict[str, int]:
    return dict(west_east=cfg.nx, west_east_stag=cfg.nx + 1,
                south_north=cfg.ny, south_north_stag=cfg.ny + 1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz + 1,
                soil_layers_stag=soil_layer_count(cfg))


def _names(cfg) -> list[str]:
    required_moisture, _ = wi.active_moisture_inventory(cfg)
    names = [name for name in wi.REQUIRED_WRFINPUT
             if name not in wi.ALL_MOISTURE_WRFINPUT]
    names += sorted(required_moisture)
    names += ["SEAICE", "ALBBCK", "LAI", "P_TOP", "SST", "SOILT1", "QKE",
              "XLAT", "XLONG", "XTIME", "ITIMESTEP"]
    names += list(_EXTRA_DIMENSIONS)
    return list(dict.fromkeys(names))


def _values(name: str, shape: tuple[int, ...], rng) -> np.ndarray:
    """Finite values with busy low bits, inside what the reader admits."""

    if name in _INTEGER_VARIABLES:
        return rng.integers(1, 12, size=shape).astype(np.int32)
    if name == "LANDMASK":
        return (rng.random(shape) > 0.4).astype(np.float32)
    if name in ("SMOIS", "SH2O"):
        return (0.1 + 0.3 * rng.random(shape)).astype(np.float32)
    return (rng.random(shape) * 7.0 + 0.125).astype(np.float32)


def _write_analysis_file(path: Path, cfg) -> dict[str, np.ndarray]:
    """A complete analysis-shaped file; returns what was stored, by name."""

    rng = np.random.default_rng(20261003)
    sizes = _dimensions(cfg)
    stored: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        for name, size in sizes.items():
            dataset.createDimension(name, size)
        dataset.setncatts({
            "TITLE": " OUTPUT FROM REAL_EM V3.9 PREPROCESSOR",
            "START_DATE": "2026-10-03_12:00:00",
            "GRID_ID": np.int32(1),
            "MP_PHYSICS": np.int32(cfg.mp_physics),
            "SF_SURFACE_PHYSICS": np.int32(cfg.sf_surface_physics),
            "BL_PBL_PHYSICS": np.int32(cfg.bl_pbl_physics),
            "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
            "HYBRID_OPT": np.int32(2),
            "ETAC": np.float32(0.2),
        })
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(b"2026-10-03_12:00:00", dtype="S1")
        for name in _names(cfg):
            dims = _EXTRA_DIMENSIONS.get(name)
            if dims is None:
                dims = wi.WRFINPUT_DIMENSIONS[name]
            shape = tuple(sizes[dim] for dim in dims)
            values = _values(name, shape, rng)
            variable = dataset.createVariable(
                name, values.dtype.str[1:], ("Time", *dims))
            if name in ("SMOIS", "SH2O"):
                variable.units = "m3 m-3"
            variable.description = f"fixture {name}"
            variable[0] = values
            stored[name] = values
    return stored


def _read(path: Path, cfg):
    try:
        return wi.read_wrfinput(path, cfg=cfg,
                                expected_dimensions=_dimensions(cfg))
    except netcdf_bridge.NetcdfBridgeMissing as error:
        pytest.skip(f"the Rust NetCDF reader is not built: {error}")


def _write_back(restored, target, **options):
    try:
        return exchange.write_back(restored, target, **options)
    except FileNotFoundError as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")


def _partials(directory: Path) -> list[str]:
    return [item.name for item in directory.iterdir() if "partial" in item.name]


def test_the_reader_takes_a_complete_analysis_file_and_names_every_variable(
        tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    restored = _read(path, cfg)

    # What the file carries and this model has no state for is accepted by
    # name and restored into nothing.
    carried = [name for name in stored
               if name in wi.ANALYSIS_PASSTHROUGH_WRFINPUT]
    assert sorted(carried) == sorted(_PASSTHROUGH_IN_FIXTURE)
    dispositions = exchange.variable_dispositions(
        restored, ["Times", *stored])
    assert set(dispositions) == {"Times", *stored}
    for name in carried:
        assert name not in restored.raw
        assert dispositions[name] == (
            exchange.PASSED_THROUGH, wi.ANALYSIS_PASSTHROUGH_WRFINPUT[name])
    for name in ("T", "QVAPOR", "U", "V", "MU", "TSK", "SMOIS", "Q2",
                 "SOILT1", "TH2", "QNRAIN", "ISLTYP", "SST", "SEAICE"):
        assert dispositions[name][0] == exchange.REWRITTEN, name
    for name in ("Times", "THM", "RDX", "CLDFRA"):
        assert dispositions[name][0] == exchange.PASSED_THROUGH, name


def test_a_name_outside_every_inventory_is_still_refused(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    _write_analysis_file(path, cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createVariable(
            "ANALYSIS_NOBODY_NAMED", "f4",
            ("Time", "south_north", "west_east"))[0] = 1.0
    with pytest.raises(ValueError, match="ANALYSIS_NOBODY_NAMED"):
        _read(path, cfg)


def test_the_pass_through_names_have_no_consumer_and_a_reason():
    for name, reason in wi.ANALYSIS_PASSTHROUGH_WRFINPUT.items():
        assert name not in wi.ALLOWED_WRFINPUT
        assert name not in wi.IGNORED_WRFINPUT
        assert len(reason) > 40, name


def test_a_name_in_two_pass_through_rows_is_refused():
    with pytest.raises(ValueError, match="QVG appears twice"):
        wi._analysis_passthrough((("one reason", ("QVG",)),
                                  ("another reason", ("QVG",))))


def test_a_zero_step_write_back_is_byte_identical(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    restored = _read(path, cfg)
    target = tmp_path / "wrf_inout.back"

    receipt = _write_back(restored, target)

    assert target.read_bytes() == path.read_bytes()
    assert receipt["byte_identical"] is True
    assert receipt["template_sha256"] == receipt["target_sha256"]
    assert receipt["schema"] == exchange.RECEIPT_SCHEMA
    assert set(receipt["variables"]) == {"Times", *stored}
    # Every restored variable really was rewritten, through float64 and
    # back, not skipped: identity by rewriting is the claim.
    rewritten = [name for name, entry in receipt["variables"].items()
                 if entry["disposition"] == exchange.REWRITTEN]
    assert sorted(rewritten) == sorted(restored.raw)
    assert receipt["counts"][exchange.REWRITTEN] == len(restored.raw)
    assert (receipt["counts"][exchange.PASSED_THROUGH]
            == len(stored) + 1 - len(restored.raw))
    assert receipt["variables"]["ISLTYP"]["stored_type"] == "<i4"
    assert receipt["variables"]["RAD_TTEN_DFI"]["disposition"] == (
        exchange.PASSED_THROUGH)
    assert _partials(tmp_path) == []


def test_an_update_changes_its_own_variable_and_nothing_else(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    restored = _read(path, cfg)
    target = tmp_path / "wrf_inout.next"
    # A forecast's values, float32 as the model holds them.
    advanced = (stored["T"] + np.float32(1.5)).astype(np.float32)

    receipt = _write_back(restored, target, updates={"T": advanced})

    assert receipt["byte_identical"] is False
    assert receipt["updated"] == ["T"]
    assert receipt["variables"]["T"]["source"] == "update"
    assert path.stat().st_size == target.stat().st_size
    before = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    after = np.frombuffer(target.read_bytes(), dtype=np.uint8)
    changed = np.flatnonzero(before != after)
    # Every changed byte lies inside one run no longer than T's slab.
    assert 0 < changed.size <= advanced.size * 4
    assert changed[-1] - changed[0] < advanced.size * 4
    with netCDF4.Dataset(target) as dataset:
        dataset.set_auto_mask(False)
        assert dataset["T"][0].tobytes() == advanced.tobytes()
        for name, values in stored.items():
            if name != "T":
                assert dataset[name][0].tobytes() == values.tobytes(), name


def test_a_value_that_would_be_rounded_is_refused_and_leaves_no_file(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    restored = _read(path, cfg)
    target = tmp_path / "wrf_inout.next"
    inexact = stored["T"].astype(np.float64) + 0.1

    with pytest.raises(bridge.NcWriteError, match="'T'.*not exactly"):
        _write_back(restored, target, updates={"T": inexact})

    assert not target.exists()
    assert _partials(tmp_path) == []


def test_updates_reach_only_variables_the_reader_restores(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    restored = _read(path, cfg)
    target = tmp_path / "wrf_inout.next"

    with pytest.raises(ValueError, match="RAD_TTEN_DFI"):
        _write_back(restored, target,
                    updates={"RAD_TTEN_DFI": stored["RAD_TTEN_DFI"]})
    with pytest.raises(ValueError, match="QNEW"):
        _write_back(restored, target, updates={"QNEW": stored["T"]})
    with pytest.raises(ValueError, match="shape"):
        _write_back(restored, target, updates={"T": stored["T"][:-1]})
    assert not target.exists()
    assert _partials(tmp_path) == []


def test_an_existing_target_is_never_written_over(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    _write_analysis_file(path, cfg)
    restored = _read(path, cfg)

    with pytest.raises(bridge.NcWriteError, match="already exists"):
        _write_back(restored, path)


def test_a_container_the_patch_cannot_keep_is_refused_by_name(tmp_path):
    """An HDF5 file has no fixed offsets to rewrite in place."""

    cfg = _cfg()
    classic = tmp_path / "classic"
    stored = _write_analysis_file(classic, cfg)
    hdf5 = tmp_path / "wrf_inout"
    sizes = _dimensions(cfg)
    with netCDF4.Dataset(classic) as source, \
            netCDF4.Dataset(hdf5, "w", format="NETCDF4") as dataset:
        dataset.setncatts({name: source.getncattr(name)
                           for name in source.ncattrs()})
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        for name, size in sizes.items():
            dataset.createDimension(name, size)
        for name, variable in source.variables.items():
            copy = dataset.createVariable(
                name, variable.dtype, variable.dimensions)
            copy.setncatts({key: variable.getncattr(key)
                            for key in variable.ncattrs()})
            copy[:] = variable[:]
    # The reader takes either container; only the way back is refused.
    restored = _read(hdf5, cfg)
    assert restored.raw["T"].astype(np.float32).tobytes() == (
        stored["T"].tobytes())

    with pytest.raises(bridge.NcWriteError, match="classic 'CDF' container"):
        _write_back(restored, tmp_path / "wrf_inout.back")
    assert _partials(tmp_path) == []


def test_the_inventory_sorts_every_variable_from_the_header_alone(tmp_path):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    stored = _write_analysis_file(path, cfg)
    try:
        document = exchange.inventory(path)
    except netcdf_bridge.NetcdfBridgeMissing as error:
        pytest.skip(f"the Rust NetCDF reader is not built: {error}")

    classes = document["classes"]
    assert document["variable_count"] == len(stored) + 1
    assert sum(document["counts"].values()) == document["variable_count"]
    assert classes[exchange.UNNAMED] == []
    assert sorted(classes[exchange.ANALYSIS_CARRIED]) == sorted(
        _PASSTHROUGH_IN_FIXTURE)
    assert {"THM", "P_HYD", "CLDFRA", "RDX", "RDY", "ZS", "DZS"} == set(
        classes[exchange.NOT_CONSUMED])
    assert {"Times", "XLAT", "XLONG", "XTIME", "ITIMESTEP"} == set(
        classes[exchange.AUXILIARY])
    assert "T" in classes[exchange.RESTORED]

    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createVariable(
            "ANALYSIS_NOBODY_NAMED", "f4",
            ("Time", "south_north", "west_east"))[0] = 1.0
    assert exchange.inventory(path)["classes"][exchange.UNNAMED] == [
        "ANALYSIS_NOBODY_NAMED"]
    assert exchange.main(["inventory", str(path)]) == 1


def test_the_round_trip_command_reads_its_configuration_from_the_file(
        tmp_path, capsys):
    cfg = _cfg()
    path = tmp_path / "wrf_inout"
    _write_analysis_file(path, cfg)
    target = tmp_path / "wrf_inout.back"
    receipt_path = tmp_path / "receipt.json"
    _read(path, cfg)  # skips here when the reader is not built

    try:
        status = exchange.main(["round-trip", str(path), str(target),
                                "--receipt", str(receipt_path)])
    except FileNotFoundError as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")

    assert status == 0
    assert target.read_bytes() == path.read_bytes()
    import json
    summary = json.loads(capsys.readouterr().out)
    assert summary["byte_identical"] is True
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["variables"]["REFL_10CM"]["disposition"] == (
        exchange.PASSED_THROUGH)
    # The fixture is complete for a start, so nothing is missing.
    assert summary["forecast_start_missing"] == []
    # A second write to the same target is refused, and says so at exit 2.
    assert exchange.main(["round-trip", str(path), str(target)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_a_file_a_forecast_cannot_start_from_still_round_trips(tmp_path):
    """Exchange and start are separate questions, answered separately."""

    cfg = _cfg()
    complete = tmp_path / "complete"
    _write_analysis_file(complete, cfg)
    path = tmp_path / "wrf_inout"
    with netCDF4.Dataset(complete) as source, \
            netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.setncatts({name: source.getncattr(name)
                           for name in source.ncattrs()})
        for name, dim in source.dimensions.items():
            dataset.createDimension(
                name, None if dim.isunlimited() else len(dim))
        for name, variable in source.variables.items():
            if name in ("AL", "ALB"):
                continue
            copy = dataset.createVariable(
                name, variable.dtype, variable.dimensions)
            copy.setncatts({key: variable.getncattr(key)
                            for key in variable.ncattrs()})
            copy[:] = variable[:]
    _read(complete, cfg)  # skips here when the reader is not built
    with pytest.raises(ValueError, match="missing mapped WRF variable"):
        _read(path, cfg)

    try:
        receipt = exchange.round_trip(path, tmp_path / "wrf_inout.back")
    except FileNotFoundError as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")

    assert receipt["byte_identical"] is True
    assert receipt["forecast_start_missing"] == ["AL", "ALB"]
