# tests/test_wrfout.py
import os
import gc
from pathlib import Path
import queue
import threading
import time
import weakref
from contextlib import nullcontext, suppress
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import netCDF4
import pytest
from conftest import requires_gpu
from gpuwm.io.wrfout import WrfoutWriter


_HANDOFF_POLL_SECONDS = 0.05


class _DoneEvent:
    @staticmethod
    def synchronize():
        return None


def _manual_async_writer(abort_event, *, ticket_queue=None):
    """CPU-only AsyncDomainWrfoutWriter shell (no CuPy stream allocation)."""
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer.nx = writer.ny = writer.nz = 1
    writer.dx = writer.dy = 1.0
    writer.soil_layers = 4
    writer.title = "test"
    writer.global_attrs = {}
    writer._queue = (AsyncDomainWrfoutWriter._new_ticket_queue()
                     if ticket_queue is None else ticket_queue)
    writer.stream = nullcontext()
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._closed = False
    writer._abort_event = abort_event
    writer.paths = []
    writer._thread = threading.Thread(target=writer._worker, daemon=True)
    writer._thread.start()
    return writer


def _cpu_ticket(path, *, fields=None, device_refs=(), pinned_refs=()):
    from gpuwm.io.wrfout import _AsyncFrame

    return _AsyncFrame(
        path=Path(path), time_str="1974-04-03_12:00:00",
        fields=({"T": np.zeros((1, 1, 1), np.float32)}
                if fields is None else fields),
        event=_DoneEvent(), device_refs=device_refs,
        pinned_refs=pinned_refs)


def _queue_cpu_ticket(writer, path, *, fields=None, device_refs=(),
                      pinned_refs=()):
    ticket = _cpu_ticket(
        path, fields=fields, device_refs=device_refs,
        pinned_refs=pinned_refs)
    writer._admit(ticket)
    return ticket


def _await_worker(event, writer, *, reaching):
    """Block until the worker fires ``event``, or until it can no longer.

    The properties these handoff tests assert are orderings and object
    lifetimes.  None of them has a wall-clock component, so none of them
    should be decided by one.  A fixed deadline here decides "was the box
    busy", and it decides it inside a suite that takes ~37 minutes and runs
    on loaded machines -- so it reports a scheduling delay as a failure of
    the writer, in the one neighbourhood (corruption detection) where a
    test nobody believes is worse than no test.

    The wait is therefore bounded by the WORKER's liveness rather than by
    the clock: a slow machine waits longer and still passes, while a worker
    that stopped fails at once and with its own recorded cause instead of a
    timeout.  Only a worker that is alive and permanently wedged can hang,
    which is a real defect and the one outcome worth hanging for.
    """
    while not event.wait(timeout=_HANDOFF_POLL_SECONDS):
        if not writer._thread.is_alive():
            if event.is_set():
                break
            # Prefer the worker's own exception over a bare assertion.
            writer._raise_failure()
            raise AssertionError(
                f"the wrfout worker stopped without {reaching}")
    return True


def _dead_full_async_writer(failure):
    """Minimal writer whose sole worker has stopped with one queued ticket."""
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = queue.Queue(maxsize=1)
    writer._queue.put(object())
    writer._condition = threading.Condition()
    writer._pending = 1
    writer._failure = failure
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    writer._thread = threading.Thread(target=lambda: None)
    writer._thread.start()
    writer._thread.join(timeout=1.0)
    assert not writer._thread.is_alive()
    return writer

def test_wrfout_roundtrip(tmp_path):
    # Colons are illegal in Windows filenames, so the on-disk name uses dashes.
    p = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    nz, ny, nx = 4, 3, 5
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0) as w:
        w.write_frame("1974-04-03_18:00:00", {
            "T": np.zeros((nz, ny, nx), np.float32),
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
        })
        w.write_frame("1974-04-03_18:05:00", {
            "T": np.full((nz, ny, nx), 2.0, np.float32),
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
        })
    ds = netCDF4.Dataset(p)
    assert ds.dimensions["west_east"].size == nx
    assert ds.dimensions["west_east_stag"].size == nx + 1
    assert ds.variables["T"].dimensions == ("Time", "bottom_top", "south_north", "west_east")
    assert ds.variables["U"].dimensions[-1] == "west_east_stag"
    times = ["".join(c.decode() for c in row) for row in ds.variables["Times"][:]]
    assert times == ["1974-04-03_18:00:00", "1974-04-03_18:05:00"]
    assert float(ds.variables["T"][1].max()) == 2.0
    assert ds.getncattr("DX") == 100.0
    ds.close()


def test_wrfout_moist_terrain_fields_and_attrs(tmp_path):
    """Phase 2 Task 12: QVAPOR/QCLOUD/QRAIN, HGT, and the terrain-consistent
    (per-column 3-D) PHB base roundtrip, and every variable carries the WRF
    attribute set (stagger / MemoryOrder / description / units /
    FieldType)."""
    p = tmp_path / "wrfout_moist.nc"
    nz, ny, nx = 4, 3, 5
    rng = np.random.default_rng(12)
    # Terrain-consistent base geopotential: per-column, monotone in k.
    phb = np.cumsum(rng.uniform(1.0, 2.0, (nz + 1, ny, nx)),
                    axis=0).astype(np.float32)
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0) as w:
        w.write_frame("0001-01-01_00:00:00", {
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "V": np.ones((nz, ny + 1, nx), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "PHB": phb,
            "HGT": np.full((ny, nx), 250.0, np.float32),
            "QVAPOR": np.full((nz, ny, nx), 0.014, np.float32),
            "QCLOUD": np.zeros((nz, ny, nx), np.float32),
            "QRAIN": np.zeros((nz, ny, nx), np.float32),
        })
    ds = netCDF4.Dataset(p)
    try:
        v = ds.variables
        # Dimensions/staggering per WRF conventions.
        assert v["QVAPOR"].dimensions == ("Time", "bottom_top",
                                          "south_north", "west_east")
        assert v["QCLOUD"].dimensions == v["QVAPOR"].dimensions
        assert v["QRAIN"].dimensions == v["QVAPOR"].dimensions
        assert v["HGT"].dimensions == ("Time", "south_north", "west_east")
        assert v["PHB"].dimensions == ("Time", "bottom_top_stag",
                                       "south_north", "west_east")
        # stagger attribute: "X"/"Y"/"Z" on staggered fields, "" otherwise.
        assert v["U"].stagger == "X"
        assert v["V"].stagger == "Y"
        assert v["W"].stagger == "Z"
        assert v["PHB"].stagger == "Z"
        assert v["QVAPOR"].stagger == ""
        assert v["HGT"].stagger == ""
        # MemoryOrder / FieldType / units / description (WRF Registry).
        assert v["QVAPOR"].MemoryOrder == "XYZ"
        assert v["HGT"].MemoryOrder == "XY "
        assert int(v["QVAPOR"].FieldType) == 104
        assert v["QVAPOR"].units == "kg kg-1"
        assert v["HGT"].units == "m"
        assert v["PHB"].units == "m2 s-2"
        for name in ("QVAPOR", "QCLOUD", "QRAIN", "HGT", "PHB"):
            assert v[name].description != ""
        # Terrain-consistent PHB base: per-column values roundtrip exactly.
        np.testing.assert_array_equal(np.asarray(v["PHB"][0]), phb)
        assert float(np.asarray(v["QVAPOR"][0]).max()) == np.float32(0.014)
    finally:
        ds.close()


def test_wrfout_roundtrips_opted_in_physics_surface_fields(tmp_path):
    p = tmp_path / "wrfout_physics.nc"
    nz, ny, nx = 2, 3, 4
    fields = {
        "RAINC": np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) / 10,
        "SWDOWN": np.full((ny, nx), 625.5, dtype=np.float32),
        "GLW": np.full((ny, nx), 312.25, dtype=np.float32),
        "OLR": np.full((ny, nx), 241.75, dtype=np.float32),
    }
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0) as w:
        w.write_frame("1974-04-03_12:00:00", fields)

    with netCDF4.Dataset(p) as ds:
        expected_metadata = {
            "RAINC": ("mm", "ACCUMULATED TOTAL CUMULUS PRECIPITATION"),
            "SWDOWN": ("W m-2", "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE"),
            "GLW": ("W m-2", "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE"),
            # Registry.EM_COMMON:1839, the row a wrf-python/wrf-rust OLR
            # recipe reads.  Same 2-D mass grid as its two neighbours.
            "OLR": ("W m-2", "TOA OUTGOING LONG WAVE"),
        }
        for name, expected in fields.items():
            assert name in ds.variables
            variable = ds.variables[name]
            assert variable.dtype == np.dtype(np.float32)
            assert (variable.units, variable.description) == expected_metadata[name]
            np.testing.assert_array_equal(np.asarray(variable[0]), expected)
        olr = ds.variables["OLR"]
        assert olr.dimensions == ("Time", "south_north", "west_east")
        assert olr.stagger == ""
        assert olr.MemoryOrder == "XY "
        assert int(olr.FieldType) == 104


def test_live_state_history_fields_cover_mp10_frozen_precip_and_noah():
    """Every live WRF-history array is handed to both frame builders.

    Use NumPy sentinels so this inventory regression remains CPU-only; the
    host and device frame paths consume the same pure mapping helper.
    """
    from gpuwm.io.wrfout import _live_state_history_fields

    nz, ny, nx = 3, 2, 4
    atmospheric = {
        name: np.full((nz, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(
            ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"), 1)
    }
    surface = {
        name: np.full((ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("snow", "snowh", "snowc"), 21)
    }
    soil = {
        name: np.full((4, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("tslb", "smois", "sh2o"), 31)
    }
    microphysics = SimpleNamespace(
        rainnc=np.full((ny, nx), 41, dtype=np.float32),
        snownc=np.full((ny, nx), 42, dtype=np.float32),
        graupelnc=np.full((ny, nx), 43, dtype=np.float32))
    state = SimpleNamespace(
        **atmospheric,
        physics=SimpleNamespace(
            fields={**surface, **soil}, microphysics=microphysics,
            noah_params=object()))

    history = _live_state_history_fields(state)
    expected = {
        "QICE": state.qi, "QSNOW": state.qs, "QGRAUP": state.qg,
        "QNCLOUD": state.nc, "QNRAIN": state.nr, "QNICE": state.ni,
        "QNSNOW": state.ns, "QNGRAUPEL": state.ng,
        "RAINNC": microphysics.rainnc, "SNOWNC": microphysics.snownc,
        "GRAUPELNC": microphysics.graupelnc,
        "SNOW": surface["snow"], "SNOWH": surface["snowh"],
        "SNOWC": surface["snowc"], "TSLB": soil["tslb"],
        "SMOIS": soil["smois"], "SH2O": soil["sh2o"],
    }
    assert set(history) == set(expected)
    for name, sentinel in expected.items():
        assert history[name] is sentinel, name


def test_wrfout_mp10_precip_snow_and_noah_schema(tmp_path):
    """The added history inventory uses WRF names, dimensions, and values."""
    p = tmp_path / "wrfout_history_schema"
    nz, ny, nx = 3, 2, 4
    fields = {
        name: np.full((nz, ny, nx), value, dtype=np.float32)
        for value, name in enumerate((
            "QICE", "QSNOW", "QGRAUP", "QNCLOUD", "QNRAIN", "QNICE",
            "QNSNOW", "QNGRAUPEL"), 1)
    }
    fields.update({
        name: np.full((ny, nx), value, dtype=np.float32)
        for value, name in enumerate((
            "RAINNC", "SNOWNC", "GRAUPELNC", "SNOW", "SNOWH", "SNOWC"),
            21)
    })
    fields.update({
        name: np.full((4, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("TSLB", "SMOIS", "SH2O"), 31)
    })
    with WrfoutWriter(
            p, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
            field_schema=fields) as writer:
        assert set(fields) <= set(writer.ds.variables)
        writer.write_frame("1974-04-03_12:00:00", fields)

    with netCDF4.Dataset(p) as ds:
        atmosphere_dims = (
            "Time", "bottom_top", "south_north", "west_east")
        surface_dims = ("Time", "south_north", "west_east")
        soil_dims = (
            "Time", "soil_layers_stag", "south_north", "west_east")
        assert ds.dimensions["soil_layers_stag"].size == 4
        for name in ("QICE", "QSNOW", "QGRAUP", "QNCLOUD", "QNRAIN",
                     "QNICE", "QNSNOW", "QNGRAUPEL"):
            assert ds[name].dimensions == atmosphere_dims
        for name in ("RAINNC", "SNOWNC", "GRAUPELNC", "SNOW", "SNOWH",
                     "SNOWC"):
            assert ds[name].dimensions == surface_dims
        for name in ("TSLB", "SMOIS", "SH2O"):
            assert ds[name].dimensions == soil_dims
            assert ds[name].stagger == "Z"
        for name, expected in fields.items():
            np.testing.assert_array_equal(np.asarray(ds[name][0]), expected)
            assert ds[name].description != ""


@pytest.mark.parametrize(
    "grid_id,parent_id,i_start,j_start,ratio,dt,expected_step",
    [
        (1, 0, 1, 1, 1, 60.0, 75),
        (2, 1, 63, 51, 4, 15.0, 300),
        (3, 2, 167, 117, 3, 5.0, 900),
        (4, 3, 151, 151, 3, 1.6666666, 2700),
    ])
def test_wrfout_real74_time_topology_vertical_and_coordinate_metadata(
        tmp_path, grid_id, parent_id, i_start, j_start, ratio, dt,
        expected_step):
    """The generic output identity matches WRF's four-domain metadata."""
    from datetime import timedelta
    from gpuwm.io.wrfout import wrf_global_attrs

    start = datetime(1974, 4, 3, 12)
    valid = start + timedelta(minutes=75)
    grid = SimpleNamespace(
        truelat1=30.0, truelat2=60.0, stand_lon=-98.0,
        ref_lat=35.0, ref_lon=-97.0, cen_lat=35.0, cen_lon=-97.0,
        moad_cen_lat=35.0)
    attrs = wrf_global_attrs(
        grid, start, grid_id=grid_id, parent_id=parent_id,
        i_parent_start=i_start, j_parent_start=j_start,
        parent_grid_ratio=ratio, dt=dt, hybrid_opt=2, etac=0.2)
    nz, ny, nx = 3, 2, 4
    fields = {
        "T": np.zeros((nz, ny, nx), dtype=np.float32),
        "U": np.zeros((nz, ny, nx + 1), dtype=np.float32),
        "V": np.zeros((nz, ny + 1, nx), dtype=np.float32),
        "W": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "XLAT": np.zeros((ny, nx), dtype=np.float32),
        "XLONG": np.zeros((ny, nx), dtype=np.float32),
        "XLAT_U": np.zeros((ny, nx + 1), dtype=np.float32),
        "XLONG_U": np.zeros((ny, nx + 1), dtype=np.float32),
        "XLAT_V": np.zeros((ny + 1, nx), dtype=np.float32),
        "XLONG_V": np.zeros((ny + 1, nx), dtype=np.float32),
        "P_TOP": np.asarray(10000.0, dtype=np.float32),
        "ZNU": np.asarray([0.8, 0.5, 0.2], dtype=np.float32),
        "ZNW": np.asarray([1.0, 0.65, 0.35, 0.0], dtype=np.float32),
    }
    p = tmp_path / f"wrfout_d{grid_id:02d}"
    with WrfoutWriter(
            p, nx=nx, ny=ny, nz=nz, dx=12000.0 / ratio,
            dy=12000.0 / ratio, global_attrs=attrs) as writer:
        writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), fields)

    with netCDF4.Dataset(p) as ds:
        assert tuple(int(getattr(ds, name)) for name in (
            "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
            "PARENT_GRID_RATIO", "HYBRID_OPT")) == (
                grid_id, parent_id, i_start, j_start, ratio, 2)
        assert float(ds.DT) == np.float32(dt)
        assert float(ds.ETAC) == np.float32(0.2)
        assert ds["XTIME"].dtype == np.dtype(np.float32)
        assert ds["ITIMESTEP"].dtype == np.dtype(np.int32)
        assert float(ds["XTIME"][0]) == 75.0
        assert int(ds["ITIMESTEP"][0]) == expected_step
        assert ds["XTIME"].units == "minutes since 1974-04-03 12:00:00"
        assert ds["P_TOP"].dimensions == ("Time",)
        assert ds["ZNU"].dimensions == ("Time", "bottom_top")
        assert ds["ZNW"].dimensions == ("Time", "bottom_top_stag")
        assert ds["ZNW"].stagger == "Z"
        assert ds["T"].coordinates == "XLONG XLAT XTIME"
        assert ds["U"].coordinates == "XLONG_U XLAT_U XTIME"
        assert ds["V"].coordinates == "XLONG_V XLAT_V XTIME"
        assert ds["W"].coordinates == "XLONG XLAT XTIME"
        assert ds["XLAT"].coordinates == "XLONG XLAT"
        assert ds["XLONG_U"].coordinates == "XLONG_U XLAT_U"
        assert ds["XLAT_V"].coordinates == "XLONG_V XLAT_V"


@pytest.mark.gpu
@requires_gpu
def test_state_frame_from_domain_state():
    """state_frame builds the standard wrfout frame from a DomainState:
    winds / T (theta - 300) / PH / MU plus the terrain-consistent base
    fields PHB (a flat base state's 1-D column broadcast to 3-D) / MUB /
    HGT, P_TOP/ZNU/ZNW vertical identity, and QVAPOR/QCLOUD/QRAIN exactly
    when the state carries moisture."""
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.io.wrfout import state_frame

    vc = make_vertical_coord(4)

    def sounding(z):
        return np.full_like(np.asarray(z, dtype=np.float64), 300.0)

    cfg = RunConfig(nx=6, ny=5, nz=4, dx=100.0, dy=100.0, ztop=4000.0,
                    dt=0.5, run_seconds=1.0, moist=True)
    b = make_base_state(vc, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    s.qv[...] = 0.004
    f = state_frame(s)
    assert set(f) == {"T", "U", "V", "W", "PH", "MU", "PHB", "MUB", "HGT",
                      "QVAPOR", "QCLOUD", "QRAIN", "P_TOP", "ZNU", "ZNW"}
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32
               for a in f.values())
    assert f["PHB"].shape == (cfg.nz + 1, cfg.ny, cfg.nx)
    np.testing.assert_array_equal(f["PHB"][:, 2, 3],
                                  np.asarray(b.phb, np.float32))
    assert f["HGT"].shape == (cfg.ny, cfg.nx) and not f["HGT"].any()
    np.testing.assert_array_equal(f["MUB"],
                                  np.full((cfg.ny, cfg.nx),
                                          np.float32(b.mub)))
    # Diagnostic pressure requires a diagnosed state (every production
    # writer path runs the EOS before writing; a fresh init has p == 0).
    from gpuwm.core.diagnostics import update_diagnostics
    update_diagnostics(s)
    diagnostic = state_frame(s, include_diagnostic_pressure=True)
    assert set(diagnostic) == set(f) | {"P", "PB", "PSFC"}
    assert diagnostic["P"].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert diagnostic["PB"].shape == (cfg.nz, cfg.ny, cfg.nx)
    # No physics driver attached: PSFC is WRF phy_prep's linear-in-z
    # surface extrapolation of the FULL (moist) pressure
    # (module_big_step_utilities_em.F:5566-5578).  Recompute it in
    # float64 from the frame's own output fields.
    z_if = (diagnostic["PH"].astype(np.float64)
            + diagnostic["PHB"]) / 9.81
    z_mid = 0.5 * (z_if[:-1] + z_if[1:])
    w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
    p_full = diagnostic["P"].astype(np.float64) + diagnostic["PB"]
    np.testing.assert_allclose(
        diagnostic["PSFC"], w1 * p_full[0] + (1.0 - w1) * p_full[1],
        rtol=5.0e-6)
    # Downward extrapolation from an at-rest hydrostatic column: PSFC
    # exceeds the lowest half-level pressure and sits near p_surf (the
    # painted qv shifts the diagnosed moist pressure ~1% off the dry
    # base, so this is a sanity bound, not a pin).
    assert np.all(diagnostic["PSFC"] > p_full[0])
    np.testing.assert_allclose(diagnostic["PSFC"], cfg.p_surf, rtol=0.03)
    np.testing.assert_array_equal(f["QVAPOR"],
                                  np.full((cfg.nz, cfg.ny, cfg.nx),
                                          np.float32(0.004)))
    # isothermal-theta base at rest: T = theta - 300 = 0
    assert not f["T"].any()

    # dry state: no moisture variables in the frame
    dry = RunConfig(nx=6, ny=5, nz=4, dx=100.0, dy=100.0, ztop=4000.0,
                    dt=0.5, run_seconds=1.0)
    fd = state_frame(init_at_rest(dry, vc, b))
    assert not ({"QVAPOR", "QCLOUD", "QRAIN"} & set(fd))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("frame_name", ["state_frame", "_device_state_frame"])
def test_psfc_source_follows_the_surface_switch_not_the_driver(frame_name):
    """PSFC comes from the driver only when the surface refreshes it.

    A physics driver allocates ``psfc`` at 100000 Pa and rewrites it from
    ``p_interface[0]`` only inside the surface/PBL cadence block, which a
    microphysics-only composition never enters.  Both writer frames used
    to select the driver's array by asking whether a driver existed, so
    an mp-only ideal case at ``p_surf = 85000`` published the untouched
    seed: 100000.00 Pa against a true 84998.16 Pa, a 150 hPa fabrication
    on the default forecast history path.

    Both directions are pinned at both sites: surface off takes WRF's
    ``phy_prep`` extrapolation, surface on takes the driver's field
    (proved with a sentinel no computation could produce).
    """
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.physics import initialize_physics
    from gpuwm.io import wrfout as wrfout_module

    frame = getattr(wrfout_module, frame_name)

    def _psfc(cfg):
        vc = make_vertical_coord(cfg.nz)
        b = make_base_state(
            vc, lambda z: np.full_like(np.asarray(z, dtype=np.float64), 300.0),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        s = init_at_rest(cfg, vc, b)
        update_diagnostics(s)
        driver = initialize_physics(s, cfg, glw=0.0)
        out = frame(s, include_diagnostic_pressure=True)
        return driver, np.asarray(cp.asnumpy(out["PSFC"]), dtype=np.float64)

    common = dict(nx=6, ny=5, nz=90, dx=1000.0, dy=1000.0, ztop=20000.0,
                  dt=1.0, run_seconds=1.0, moist=True, p_surf=85000.0)

    # Direction 1: microphysics only.  The seed is never refreshed, so the
    # frame must NOT publish it.
    mp_only, psfc = _psfc(RunConfig(**common, mp_physics=6))
    assert mp_only.surface_enabled is False
    assert float(cp.asnumpy(mp_only.fields["psfc"]).ravel()[0]) == 100000.0
    np.testing.assert_allclose(psfc, 84998.16, atol=0.01)
    assert abs(psfc - 100000.0).min() > 14000.0    # the fabrication is gone

    # Direction 2: a surface-bearing composition.  The driver's array is
    # the authority, sentinel-proved.
    surf_cfg = RunConfig(**common, mp_physics=6, sf_sfclay_physics=1,
                         sf_surface_physics=2, bl_pbl_physics=1)
    vc = make_vertical_coord(surf_cfg.nz)
    b = make_base_state(
        vc, lambda z: np.full_like(np.asarray(z, dtype=np.float64), 300.0),
        p_surf=surf_cfg.p_surf, ztop=surf_cfg.ztop)
    s = init_at_rest(surf_cfg, vc, b)
    update_diagnostics(s)
    driver = initialize_physics(s, surf_cfg, glw=0.0)
    assert driver.surface_enabled is True
    driver.fields["psfc"][...] = np.float32(12345.0)
    out = frame(s, include_diagnostic_pressure=True)
    np.testing.assert_array_equal(
        np.asarray(cp.asnumpy(out["PSFC"])),
        np.full((surf_cfg.ny, surf_cfg.nx), np.float32(12345.0)))


def test_wrf_time_str():
    """WRF Times formatting: seconds from 0001-01-01_00:00:00, calendar
    rollover included (shared helper single-sources what straka/igw
    duplicated).

    The month and year roll too, which they did not before v1.1.3: the
    helper incremented only the day field, so this test stopping at day two
    was the whole reason ``0001-01-32`` and ``0001-01-366`` could ship.  The
    cases below cross a month end, a leap-less February, and a year end.
    """
    from gpuwm.io.wrfout import wrf_time_str
    assert wrf_time_str(0.0) == "0001-01-01_00:00:00"
    assert wrf_time_str(305.0) == "0001-01-01_00:05:05"
    assert wrf_time_str(7200.0) == "0001-01-01_02:00:00"
    assert wrf_time_str(86400.0 + 3661.0) == "0001-01-02_01:01:01"
    assert wrf_time_str(30 * 86400.0) == "0001-01-31_00:00:00"
    assert wrf_time_str(31 * 86400.0) == "0001-02-01_00:00:00"
    assert wrf_time_str(58 * 86400.0) == "0001-02-28_00:00:00"
    assert wrf_time_str(59 * 86400.0) == "0001-03-01_00:00:00"
    assert wrf_time_str(365 * 86400.0) == "0002-01-01_00:00:00"
    with pytest.raises(ValueError, match="non-negative"):
        wrf_time_str(-1.0)
    with pytest.raises(ValueError, match="overflows"):
        wrf_time_str(1.0e15)


def test_async_writer_netCDF_sessions_are_process_serial(monkeypatch,
                                                         tmp_path):
    import gpuwm.io.wrfout as wrfout

    active = 0
    peak = 0
    guard = threading.Lock()

    class SlowWriter:
        def __init__(self, path, **_kwargs):
            self.path = path

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", SlowWriter)
    abort = threading.Event()
    first = _manual_async_writer(abort)
    second = _manual_async_writer(abort)
    _queue_cpu_ticket(first, tmp_path / "d01")
    _queue_cpu_ticket(second, tmp_path / "d02")
    first.close()
    second.close()
    assert peak == 1
    assert not first._thread.is_alive()
    assert not second._thread.is_alive()


def test_async_worker_releases_device_refs_after_d2h_before_write(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout

    write_started = threading.Event()
    finish_write = threading.Event()

    class BlockingWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            write_started.set()
            # No deadline, and no assertion, on purpose.  This wait spans the
            # main thread's gc.collect(), whose cost is a function of the heap
            # the rest of the suite has already built -- not of this writer.
            # A bound here used to fail INSIDE the worker, where _worker's
            # `except BaseException` files it as self._failure and sets the
            # experiment-wide abort event, so a slow collection was reported
            # as `RuntimeError: per-domain wrfout writer failed`: a false
            # corruption alarm raised by the corruption detector's own test.
            # The release below is set unconditionally in this test's finally,
            # so an unbounded wait cannot outlive the test.
            finish_write.wait()

        def __exit__(self, *_args):
            return False

    class DeviceArray:
        pass

    monkeypatch.setattr(wrfout, "WrfoutWriter", BlockingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        device = DeviceArray()
        pinned_owner = np.zeros((1, 1, 1), dtype=np.float32)
        field = pinned_owner.view()
        assert field.base is pinned_owner
        device_ref = weakref.ref(device)
        pinned_ref = weakref.ref(pinned_owner)
        # `field` is a view of `pinned_owner`, so numpy's own base chain keeps
        # the backing alive independently of ticket.pinned_refs -- and in the
        # production path too, where host_fields come from
        # np.frombuffer(memory, ...).  Watching only `pinned_owner` therefore
        # cannot tell "pinned_refs retained it" from "the view's .base did":
        # dropping ticket.pinned_refs early leaves this test green.  The
        # unviewed buffer below is reachable ONLY through ticket.pinned_refs,
        # so it holds the explicit mechanism to its stated contract.
        unviewed = np.zeros((1, 1, 1), dtype=np.float32)
        unviewed_ref = weakref.ref(unviewed)
        ticket = _queue_cpu_ticket(
            writer, tmp_path / "d01", fields={"T": field},
            device_refs=(device,), pinned_refs=(pinned_owner, unviewed))
        ticket_ref = weakref.ref(ticket)
        del device, pinned_owner, field, ticket, unviewed

        _await_worker(write_started, writer, reaching="entering write_frame")
        gc.collect()
        assert device_ref() is None
        assert pinned_ref() is not None
        assert unviewed_ref() is not None
        assert ticket_ref() is not None
        finish_write.set()
        writer.drain()
        gc.collect()
        assert pinned_ref() is None
        assert unviewed_ref() is None
        assert ticket_ref() is None
    finally:
        finish_write.set()
        writer.close()


def test_async_write_failure_does_not_retain_field_backing(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout

    class FailingWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, fields):
            arr = fields["T"]
            assert arr.shape == (1, 1, 1)
            raise OSError("injected field write failure")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    writer = _manual_async_writer(threading.Event())
    pinned_owner = np.zeros((1, 1, 1), dtype=np.float32)
    field = pinned_owner.view()
    assert field.base is pinned_owner
    pinned_ref = weakref.ref(pinned_owner)
    ticket = _queue_cpu_ticket(
        writer, tmp_path / "d01", fields={"T": field},
        pinned_refs=(pinned_owner,))
    ticket_ref = weakref.ref(ticket)
    del pinned_owner, field, ticket

    try:
        with pytest.raises(
                RuntimeError, match="per-domain wrfout writer failed") as raised:
            writer.drain()
        cause = raised.value.__cause__
        assert isinstance(cause, OSError)
        assert str(cause) == "injected field write failure"
        assert cause is writer._failure
        gc.collect()
        assert ticket_ref() is None
        assert pinned_ref() is None
        assert cause.__traceback__ is None
        assert "in write_frame" in writer._failure_traceback
        assert "injected field write failure" in writer._failure_traceback
    finally:
        with suppress(RuntimeError):
            writer.close()


def test_async_worker_holds_no_completed_ticket_while_idle(monkeypatch,
                                                           tmp_path):
    import gpuwm.io.wrfout as wrfout

    class NoOpWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            pass

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", NoOpWriter)
    writer = _manual_async_writer(threading.Event())
    ticket = _queue_cpu_ticket(writer, tmp_path / "d01")
    ticket_ref = weakref.ref(ticket)
    del ticket

    try:
        writer.drain()
        gc.collect()
        assert ticket_ref() is None
    finally:
        writer.close()


def test_async_admission_raises_recorded_failure_when_worker_dies_with_full_queue(
        tmp_path):
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    attempted = threading.Event()

    class SignallingQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            if timeout is not None:
                attempted.set()
            return super().put(item, block=block, timeout=timeout)

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = SignallingQueue(maxsize=1)
    # Use the base implementation so filling the queue does not signal the
    # simulated worker-death transition before admission starts.
    queue.Queue.put(writer._queue, object())
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    failure = OSError("worker failed while admission was blocked")

    def fail_worker():
        if attempted.wait(timeout=1.0):
            writer._failure = failure

    writer._thread = threading.Thread(target=fail_worker)
    writer._thread.start()
    with pytest.raises(RuntimeError, match="per-domain wrfout writer failed") as raised:
        writer._admit(_cpu_ticket(tmp_path / "blocked"))
    writer._thread.join(timeout=1.0)

    assert raised.value.__cause__ is failure
    assert writer.pending == 0
    assert writer._queue.qsize() == 1


def test_async_interrupted_admission_rolls_back_pending_and_drain_returns(
        tmp_path):
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    class InterruptingQueue:
        @staticmethod
        def put(_item, *, timeout):
            assert timeout > 0.0
            raise KeyboardInterrupt("injected interrupted put")

    class LiveWorker:
        @staticmethod
        def is_alive():
            return True

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = InterruptingQueue()
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    writer._thread = LiveWorker()

    with pytest.raises(KeyboardInterrupt, match="injected interrupted put"):
        writer._admit(_cpu_ticket(tmp_path / "interrupted"))
    assert writer.pending == 0
    writer.drain()


def test_async_interrupted_admission_after_insert_stays_balanced_and_drains(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    consumed = threading.Event()

    class NoOpWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            consumed.set()

        def __exit__(self, *_args):
            return False

    ticket_queue_type = type(AsyncDomainWrfoutWriter._new_ticket_queue())

    class InsertThenInterruptQueue(ticket_queue_type):
        def __init__(self):
            super().__init__(maxsize=1)
            self._interrupt_next_ticket = True

        def put(self, item, block=True, timeout=None):
            super().put(item, block=block, timeout=timeout)
            if item is not None and self._interrupt_next_ticket:
                self._interrupt_next_ticket = False
                raise KeyboardInterrupt("injected interrupt after insertion")

    monkeypatch.setattr(wrfout, "WrfoutWriter", NoOpWriter)
    writer = _manual_async_writer(
        threading.Event(), ticket_queue=InsertThenInterruptQueue())
    drained = threading.Event()
    drainer = None
    pending_after_consume = None
    try:
        ticket = _cpu_ticket(tmp_path / "inserted")
        with pytest.raises(
                KeyboardInterrupt, match="injected interrupt after insertion"):
            writer._admit(ticket)
        assert ticket.admitted
        _await_worker(consumed, writer, reaching="consuming the ticket")
        pending_after_consume = writer.pending

        def drain():
            writer.drain()
            drained.set()

        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()
        drain_terminated = drained.wait(timeout=1.0)
    finally:
        # Keep the deliberately anti-hang regression self-cleaning while it
        # is red against an implementation that drives _pending negative.
        with writer._condition:
            if writer._pending < 0:
                writer._pending = 0
                writer._condition.notify_all()
        if drainer is not None:
            drainer.join(timeout=1.0)
        writer.close()

    assert pending_after_consume == 0
    assert drain_terminated


@pytest.mark.parametrize("operation", ["drain", "close"])
def test_async_shutdown_terminates_for_dead_worker_with_full_queue(operation):
    failure = OSError("worker died with queued work")
    writer = _dead_full_async_writer(failure)
    finished = threading.Event()
    raised = []

    def shut_down():
        try:
            getattr(writer, operation)()
        except BaseException as exc:
            raised.append(exc)
        finally:
            finished.set()

    caller = threading.Thread(target=shut_down, daemon=True)
    caller.start()
    assert finished.wait(timeout=1.0), f"{operation} blocked on a dead worker"
    caller.join(timeout=1.0)
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert raised[0].__cause__ is failure


def test_async_ticket_queues_bound_synchronized_four_domain_bursts():
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    class Ticket:
        admitted = False

    domain_count = 4
    ticket_queues = [
        AsyncDomainWrfoutWriter._new_ticket_queue()
        for _ in range(domain_count)
    ]
    first_burst = [Ticket() for _ in range(domain_count)]
    second_burst = [Ticket() for _ in range(domain_count)]
    third_burst = [Ticket() for _ in range(domain_count)]

    for ticket_queue, ticket in zip(ticket_queues, first_burst):
        ticket_queue.put(ticket)
    worker_held = [ticket_queue.get() for ticket_queue in ticket_queues]
    assert worker_held == first_burst
    for ticket_queue, ticket in zip(ticket_queues, second_burst):
        ticket_queue.put(ticket)

    # One complete history burst may wait across the domain queues while each
    # worker owns its current handoff: eight admitted tickets, globally.
    assert len(worker_held) + sum(q.qsize() for q in ticket_queues) == (
        2 * domain_count)

    admitted = [threading.Event() for _ in range(domain_count)]

    def submit_third(domain_index):
        ticket_queues[domain_index].put(third_burst[domain_index])
        admitted[domain_index].set()

    producers = [
        threading.Thread(target=submit_third, args=(domain_index,))
        for domain_index in range(domain_count)
    ]
    for producer in producers:
        producer.start()
    assert all(not event.wait(timeout=0.05) for event in admitted)

    for ticket_queue in ticket_queues:
        ticket_queue.task_done()
    worker_held = [ticket_queue.get() for ticket_queue in ticket_queues]
    assert worker_held == second_burst
    assert all(event.wait(timeout=1.0) for event in admitted)
    for producer in producers:
        producer.join(timeout=1.0)
        assert not producer.is_alive()
    assert len(worker_held) + sum(q.qsize() for q in ticket_queues) == (
        2 * domain_count)
    assert [ticket_queue.get() for ticket_queue in ticket_queues] == third_burst


def test_per_domain_writer_builds_static_metadata_once(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout
    import gpuwm.runtime as runtime

    metadata = {"XLAT": np.arange(4, dtype=np.float64).reshape(2, 2)}
    metadata_calls = []
    global_start_calls = []

    def build_metadata(grid, static_fields):
        metadata_calls.append((grid, static_fields))
        return metadata

    class CapturingWriter:
        def __init__(self, **_kwargs):
            self.pending = 0
            self.paths = []
            self.submissions = []

        def submit(self, *args, **kwargs):
            self.submissions.append((args, kwargs))

        def drain(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runtime, "_metadata_frame", build_metadata)
    def global_attrs(_grid, start_time, *_args, **_kwargs):
        global_start_calls.append(start_time)
        return {}
    monkeypatch.setattr(runtime, "_global_wrf_attrs", global_attrs)
    monkeypatch.setattr(wrfout, "AsyncDomainWrfoutWriter", CapturingWriter)

    cfg = SimpleNamespace(
        grid_id=1,
        start_time=datetime(1974, 4, 3, 12, 5),
        # sf_surface_physics is not optional in a RunConfig stand-in any more:
        # the soil axis is resolved from the SCHEME (config.soil_layer_count),
        # so a fake that declares a layer count without declaring whose it is
        # can no longer be asked for its geometry.
        run=SimpleNamespace(nx=2, ny=2, nz=1, dx=1.0, dy=1.0,
                            sf_surface_physics=0, num_soil_layers=4))
    grid = object()
    static_fields = {"HGT_M": np.zeros((2, 2), dtype=np.float32)}
    case = SimpleNamespace(
        static_fields=static_fields, geog_selection=None,
        initial_result=SimpleNamespace(
            coord=SimpleNamespace(hybrid_opt=2, etac=0.2)))
    node = SimpleNamespace(
        cfg=cfg, grid=grid, state=object(),
        clock=SimpleNamespace(tick_den=1))
    model = SimpleNamespace(
        _prepared_by_grid_id={1: case},
        walk_parent_first=lambda: (node,))

    writers = wrfout.PerDomainWrfoutWriters(
        model, tmp_path, start_time=datetime(1974, 4, 3, 12),
        title="test")
    writers.submit(node, 0)
    writers.submit(node, 900)

    assert metadata_calls == [(grid, static_fields)]
    assert global_start_calls == [datetime(1974, 4, 3, 12, 5)]
    submissions = writers._writers[1].submissions
    assert submissions[0][1]["extra_fields"] is metadata
    assert submissions[1][1]["extra_fields"] is metadata


def test_failing_writer_close_joins_every_domain_and_aborts_publication(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout

    published = []

    class FailingWriter:
        def __init__(self, path, **_kwargs):
            self.path = Path(path)

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            if self.path.name == "d01":
                raise OSError("injected netCDF failure")
            published.append(self.path.name)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    abort = threading.Event()
    first = _manual_async_writer(abort)
    second = _manual_async_writer(abort)
    _queue_cpu_ticket(first, tmp_path / "d01")
    _await_worker(abort, first, reaching="signalling the publication abort")
    with pytest.raises(RuntimeError, match="wrfout writing was aborted"):
        _queue_cpu_ticket(second, tmp_path / "d02")

    writers = object.__new__(wrfout.PerDomainWrfoutWriters)
    writers._writers = {1: first, 2: second}
    writers._abort_event = abort
    writers.last_durable_wrfout = None
    with pytest.raises(RuntimeError, match="per-domain wrfout writer failed"):
        writers.close()
    assert not first._thread.is_alive()
    assert not second._thread.is_alive()
    assert published == []
    time.sleep(0.02)
    assert published == []


_REFERENCE_WRFOUT = os.path.join(
    os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                   "gpuwm-fixture-unset/wrf74-bundle"),
    "wrfout_reference", "wrfout_d01_1974-04-03_13_00_00")


@pytest.mark.skipif(not __import__("pathlib").Path(_REFERENCE_WRFOUT).is_file(),
                    reason="WRF_1974_MP55 reference bundle not present")
def test_var_meta_matches_reference_wrfout_attributes():
    """Every _VAR_META description/units string matches what the group's
    WRF v4.6.1 actually wrote (the Registry strings, via the reference
    wrfout).  Guards against drift for all variables the reference
    carries; gpuwm-only extensions (e.g. PSIM/PSIH) are exempt."""
    from gpuwm.io.wrfout import _VAR_META

    with netCDF4.Dataset(_REFERENCE_WRFOUT) as ds:
        checked = 0
        for name, (desc, units) in _VAR_META.items():
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            assert var.description == desc, (
                f"{name}: {var.description!r} != {desc!r}")
            assert var.units == units, f"{name}: {var.units!r} != {units!r}"
            checked += 1
    assert checked >= 40


def test_p3s_rime_pair_is_published_and_its_carriers_are_not():
    """mp=50's history inventory, and the two fields WRF keeps out of it.

    WRF gives QIR/QIB the history ``h`` in their IO string
    (``Registry.EM_COMMON:555-558``, ``i0rhusdf``), and without them an
    mp=50 wrfout is a one-moment ice field: rime fraction QIR/QICE and rime
    density QIR/QIB are the two indices P3's whole ice inventory is built
    on, and neither is recoverable from QICE alone.  th_old/qv_old are
    restart-only in WRF (``:1598-1599``, ``rusd`` -- no ``h``), so
    publishing them would be a gpuwm invention, not a transcription.

    Presence-guarded, so no other scheme's inventory moves.
    """
    from gpuwm.io.wrfout import _live_state_history_fields

    class _P3:
        qi = np.zeros((4, 3, 2), np.float32)
        ni = np.zeros((4, 3, 2), np.float32)
        nr = np.zeros((4, 3, 2), np.float32)
        qir = np.full((4, 3, 2), 1.0e-5, np.float32)
        qib = np.full((4, 3, 2), 2.0e-8, np.float32)
        th_old = np.full((4, 3, 2), 300.0, np.float32)
        qv_old = np.full((4, 3, 2), 1.0e-3, np.float32)

    fields = _live_state_history_fields(_P3())
    assert fields["QIR"] is _P3.qir
    assert fields["QIB"] is _P3.qib
    assert fields["QICE"] is _P3.qi
    # P3 has ONE ice category: these are absent, not zero.
    assert not ({"QSNOW", "QGRAUP", "QHAIL"} & set(fields))
    # Restart-only, following WRF's own IO strings.
    assert not ({"TH_OLD", "QV_OLD"} & set(fields))

    class _Morrison:
        qi = np.zeros((4, 3, 2), np.float32)
        nc = np.zeros((4, 3, 2), np.float32)

    assert not ({"QIR", "QIB"} & set(_live_state_history_fields(_Morrison())))
