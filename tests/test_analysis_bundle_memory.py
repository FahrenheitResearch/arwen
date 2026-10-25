"""wrf-analysis-exchange/1: the analysis bundle's words held in memory.

Every exchange is checked against the file path it mirrors on the same inputs: make_bundle
(increment=..., ph="balance", coordinate=...) for the analysis and increment words, the bundle
it writes for :meth:`AnalysisExchange.write`, and the door's ``iau-split.npz`` for
:func:`plan_from_exchange`.  Words are compared as unsigned integers, so -0.0 and +0.0 differ.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from analysis_bundle_fixtures import bundle_identity, write_wrf_state, write_wrfbdy
from gpuwm.da import analysis_door as door
from gpuwm.io import analysis_bundle as ab
from test_analysis_door import no_diagnostics  # noqa: F401 - pytest fixture

NAMELIST = """&time_control
 run_hours = 1,
 start_year = 2022, start_month = 06, start_day = 13, start_hour = 23,
 end_year = 2022, end_month = 06, end_day = 14, end_hour = 00,
 interval_seconds = 3600,
/
&domains
 time_step = 60, max_dom = 1,
 e_we = 8, e_sn = 7, e_vert = 5,
/
&dynamics
 hypsometric_opt = 2,
/
"""
COORDINATE = ("C1F", "C2F", "C1H", "C2H", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN", "DNW")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame(path: Path):
    """A history frame as a frame sink holds it: stored words, dimensions, global attributes."""

    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        state = {name: np.asarray(v[0]) for name, v in dataset.variables.items() if name != "Times"}
        dims = {name: tuple(v.dimensions[1:]) for name, v in dataset.variables.items() if name != "Times"}
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        valid = b"".join(dataset["Times"][0]).decode()
    return state, dims, attrs, valid


def _increment(background: Path, *, theta=1.5, mass=40.0, rain=-2.0e-3):
    state, _dims, _attrs, _valid = _frame(background)
    inc = {name: np.zeros(state[name].shape, np.float64) for name in ("U", "V", "W", "T", "PH", "MU", "QVAPOR")}
    inc["T"][1, 2, 3] = theta
    inc["U"][:, 3, 4] = 2.0
    inc["MU"][1, 1] = mass
    inc["QVAPOR"][0, 2, 2] = 1.0e-4
    qr = np.zeros(state["QRAIN"].shape, np.float64)
    qr[1, 2, 3] = rain                  # takes the rain below zero: the analysis clips it
    qr[2, 3, 3] = 2.0e-4
    inc["QRAIN"] = qr
    return inc


def _write_increment_form(path: Path, inc, background: Path):
    """The producer's increment form, float64 (what a producer stages for make_bundle)."""

    with netCDF4.Dataset(background) as dataset:
        sizes = {name: len(dim) for name, dim in dataset.dimensions.items()}
    ab._write_increment_file(path, inc, sizes=sizes, valid_time="2022-06-13_23:00:00", precision="f8")
    return path


def _coordinate(path: Path):
    mine = ab.coordinate_of(path)
    state, _dims, _attrs, _valid = _frame(path)
    assert all(np.array_equal(mine[name], state[name].astype(np.float64)) for name in COORDINATE)
    return mine


@pytest.fixture
def made(tmp_path):
    background = write_wrf_state(tmp_path / "wrfout.nc")
    coordinate = write_wrf_state(tmp_path / "wrfinput_d01")
    inc = _increment(background)
    staged = _write_increment_form(tmp_path / "inc.nc", inc, background)
    bundle, record = ab.make_bundle(tmp_path / "bundle", background=background, increment=staged,
                                    iau_window_s=300.0, form="wrfout", coordinate=coordinate,
                                    **bundle_identity())
    state, dims, attrs, valid = _frame(background)
    exchange = ab.make_exchange(state=state, attributes=attrs, valid_time=valid, coordinate=_coordinate(coordinate),
                                increment=inc, iau_window_s=300.0, form="wrfout", dimensions=dims,
                                coordinate_grid=ab.grid_fingerprint(coordinate), **bundle_identity())
    return dict(background=background, coordinate=coordinate, bundle=bundle, record=record, exchange=exchange,
                inc=inc, tmp=tmp_path)


def _words(a):
    a = np.ascontiguousarray(a)
    return a.view(f"u{a.dtype.itemsize}")


def _file_record(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        return np.asarray(dataset[name][0])


def test_the_exchange_holds_make_bundles_analysis_words(made):
    exchange, bundle = made["exchange"], made["bundle"]
    state = exchange.state()
    with netCDF4.Dataset(bundle.analysis) as dataset:
        names = [n for n in dataset.variables if n != "Times"]
    for name in names:
        theirs = _file_record(bundle.analysis, name)
        mine = np.asarray(state[name])
        assert mine.dtype == theirs.dtype, name
        assert np.array_equal(_words(mine), _words(theirs)), name
    # the moved fields are the increment's and the engine's PH; the clip is recorded
    assert {"T", "U", "MU", "QVAPOR", "QRAIN", "PH"} <= set(exchange.analysis)
    assert exchange.record["analysis_from_increment"]["clipped_negative"]["QRAIN"]["values"] == 1
    assert exchange.record["balance"]["mass_moved"] is True


def test_the_exchange_holds_make_bundles_increment_words(made):
    exchange, bundle = made["exchange"], made["bundle"]
    files = ab.read_increment(bundle.increment)
    assert sorted(files) == sorted(exchange.increment)
    for name, theirs in files.items():
        mine = exchange.increment[name]
        assert mine.dtype == np.float64
        assert np.array_equal(_words(mine), _words(theirs)), name
    assert exchange.record["increment"]["exact_background"] == made["record"]["increment"]["exact_background"]


def test_the_identity_blocks_are_the_bundles(made):
    exchange, bundle = made["exchange"], made["bundle"]
    m = bundle.manifest
    for key in ("valid_time", "grid", "static", "physics", "iau", "versions"):
        assert exchange.metadata[key] == m[key], key
    assert exchange.metadata["analysis"] == dict(m["analysis"], form="exchange")
    assert exchange.metadata["schema"] == ab.EXCHANGE_SCHEMA
    assert set(exchange.metadata) | {"files"} == set(m)


def test_written_with_its_template_it_is_make_bundles_bundle_byte_for_byte(made):
    exchange, bundle = made["exchange"], made["bundle"]
    out = exchange.write(made["tmp"] / "exported", template=made["background"])
    for name in ("analysis.nc", "increment.nc", "manifest.json"):
        assert _sha(out.root / name) == _sha(bundle.root / name), name


def test_written_from_memory_it_verifies_and_holds_the_same_arrays(made):
    exchange, bundle = made["exchange"], made["bundle"]
    out = exchange.write(made["tmp"] / "thin")
    assert ab.load_bundle(out.root).manifest["files"].keys() == bundle.manifest["files"].keys()
    with netCDF4.Dataset(bundle.analysis) as dataset:
        names = [n for n in dataset.variables if n != "Times"]
    for name in names:
        assert np.array_equal(_words(_file_record(out.analysis, name)), _words(_file_record(bundle.analysis, name)))
    assert _sha(out.root / "increment.nc") == _sha(bundle.root / "increment.nc")


def test_without_dimensions_and_without_a_template_writing_is_refused(made, tmp_path):
    state, _dims, attrs, valid = _frame(made["background"])
    exchange = ab.make_exchange(state=state, attributes=attrs, valid_time=valid,
                                coordinate=_coordinate(made["coordinate"]), increment=made["inc"],
                                iau_window_s=300.0, **bundle_identity())
    with pytest.raises(ab.BundleRefusal, match="dimensions"):
        exchange.write(tmp_path / "x")


def test_the_record_lists_the_increment_arrays_and_its_metadata_validates(made):
    exchange = made["exchange"]
    record = exchange.exchange_record(lambda a: "0123456789abcdef")
    assert set(record["arrays"]) == set(exchange.increment)
    assert all(entry["dtype"] == "f8" for entry in record["arrays"].values())
    with pytest.raises(ab.BundleRefusal, match="16-digit"):
        exchange.exchange_record(lambda a: "nope")
    ab.validate_exchange_metadata(json.loads(json.dumps(exchange.metadata)))


def test_another_coordinate_grid_is_refused(made, tmp_path):
    state, dims, attrs, valid = _frame(made["background"])
    other = write_wrf_state(tmp_path / "other", xlat_offset=1.0)
    with pytest.raises(ab.BundleRefusal, match="coordinate"):
        ab.make_exchange(state=state, attributes=attrs, valid_time=valid, coordinate=_coordinate(made["coordinate"]),
                         increment=made["inc"], iau_window_s=300.0, dimensions=dims,
                         coordinate_grid=ab.grid_fingerprint(other), **bundle_identity())


def test_an_increment_the_forecast_never_integrates_is_refused(made):
    state, dims, attrs, valid = _frame(made["background"])
    inc = dict(made["inc"], P=np.zeros(state["P"].shape))
    with pytest.raises(ab.BundleRefusal, match="not a prognostic"):
        ab.make_exchange(state=state, attributes=attrs, valid_time=valid, coordinate=_coordinate(made["coordinate"]),
                         increment=inc, iau_window_s=300.0, **bundle_identity())


def _run(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "namelist.input").write_text(NAMELIST, encoding="ascii")
    start = write_wrf_state(root / "wrfinput_d01")
    write_wrfbdy(root / "wrfbdy_d01", start, width=2)
    return root


@pytest.mark.parametrize("flags", [dict(iau="on", iau_mode="3d", iau_fields="all", iau_window_seconds=300.0),
                                   dict(iau="on", iau_mode="3d", iau_fields="dynamics", iau_window_seconds=300.0),
                                   dict(iau="off")])
def test_the_plan_from_an_exchange_is_the_doors_split(made, tmp_path, flags):
    run = _run(tmp_path / "run")
    request = {"analysis": made["bundle"].root, "iau": None, "iau_mode": None, "iau_fields": None,
               "iau_boundary": None, "iau_window_seconds": None, "analysis_work": None, "plan": None}
    request.update(flags)
    checkpoint = tmp_path / "gpuwmrst_d01_2022-06-13_23_00_00.npz"
    checkpoint.write_bytes(b"restart")
    resolved = door.resolve(request, directory=run, outdir=tmp_path / "out", restart=checkpoint)
    plan = json.loads(Path(resolved["plan"]).read_text())
    at_once, spread = door.read_split(Path(plan["forcing"]["path"]), plan["forcing"]["sha256"])
    mine = door.plan_from_exchange(made["exchange"], dict(flags), hypsometric_opt=door.hypsometric_opt_of(
        run / "namelist.input"), coordinate=_coordinate(run / "wrfinput_d01"),
        run_grid=ab.grid_fingerprint(run / "wrfinput_d01"),
        run_static_sha256=ab.static_fingerprint(run / "wrfinput_d01"))
    assert mine["start"] == plan["start"] == "cycling-restart"
    assert mine["iau"] == plan["iau"]
    assert sorted(mine["forcing"]["at_once"]) == sorted(at_once)
    assert sorted(mine["forcing"]["spread"]) == sorted(spread)
    for part, theirs in (("at_once", at_once), ("spread", spread)):
        for name, value in theirs.items():
            assert np.array_equal(_words(mine["forcing"][part][name]), _words(value)), (part, name)
    rebuilt = tmp_path / "background-file.nc"
    ab.write_background(made["bundle"].analysis, made["bundle"].increment, rebuilt)
    for name, value in mine["background"].items():
        assert np.array_equal(_words(value), _words(_file_record(rebuilt, name))), name


@pytest.mark.parametrize("continuation", [True, False])
def test_the_generated_plan_applies_without_file_plan_keys(made, no_diagnostics, continuation):
    from datetime import datetime
    from test_analysis_door import _host_state, _tree

    plan = door.plan_from_exchange(made["exchange"], {"iau": "on"}, hypsometric_opt=2,
                                  coordinate=_coordinate(made["coordinate"]), run_seconds=3600.)
    assert "work" not in plan and "run_directory" not in plan
    state = _host_state(made["background"])
    model, nodes = _tree(state)
    out = made["tmp"] / "applied"
    out.mkdir()
    worker = door.AnalysisInitialization(None, plan, outdir=out)
    how = {"continuation": "in-memory"} if continuation else {"restart": made["background"]}
    worker.apply_door_plan(model=model, nodes=nodes, start_time=datetime(2022,6,13,23),
                           run_seconds=3600., **how)
    receipt = worker.run_report_blocks(None)["analysis_bundle"]
    assert receipt["plan"] is None
    assert receipt["applied_now"]["applied"] is True
    assert receipt["mass"]["mode"] == "spread"
    start = json.loads((out / "forecast-start.json").read_text())
    assert start["analysis_bundle"] == made["exchange"].manifest_sha256()
    assert start["run_directory"] is None


def test_a_plan_refuses_a_file_and_a_cold_start_boundary(made):
    coordinate = _coordinate(made["coordinate"])
    with pytest.raises(door.AnalysisDoorRefusal, match="names a file"):
        door.plan_from_exchange(made["exchange"], {"analysis_work": "/x"}, hypsometric_opt=2, coordinate=coordinate)
    with pytest.raises(door.AnalysisDoorRefusal, match="G8b"):
        door.plan_from_exchange(made["exchange"], {"iau_boundary": "analysis"}, hypsometric_opt=2,
                                coordinate=coordinate)
    with pytest.raises(door.AnalysisDoorRefusal, match="G10"):
        door.plan_from_exchange(made["exchange"], {}, hypsometric_opt=2, coordinate=coordinate, run_seconds=60.0)
