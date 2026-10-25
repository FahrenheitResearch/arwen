"""Column-blocked CPU initialization gives the whole-domain bytes.

THE BREAKAGE the blocks fix: ``initialize_real`` held every float64 work
array of the whole domain at once, about 174 GiB on a 4160 x 2450 x 50 root,
so each 1 km CONUS preparation needed 115-245 GB of host RAM and two of
them were SIGKILLed together by a container's memory limit (ledger A4).
The CPU preparation now builds row blocks and keeps only finished fields
whole.  These tests hold every block count to the whole-domain result:
every state array, every result array and every receipt, on the three
lanes (FLAG_SH RH, use_sh_qv with a dry slot the floor catches, and the
pressure-level RH lane through sfcprs3), with an aerosol donor whose own
level ladder differs from the analysis'.
"""

from datetime import datetime

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.real import (
    INITIALIZATION_BLOCK_COLUMNS, _initialization_block_rows, initialize_real)

NY, NX, NZ = 9, 7, 10
VALID = datetime(2026, 7, 20, 6)


def _field(levels, scale, offset, *, seed, stagger=None):
    rng = np.random.default_rng(seed)
    ny, nx = NY, NX
    if stagger == "u":
        nx += 1
    elif stagger == "v":
        ny += 1
    shape = (levels.size, ny, nx) if levels is not None else (ny, nx)
    return np.asarray(offset + scale * rng.random(shape), dtype=np.float32)


def _column_varying(levels, *, specific=True, seed=0, dry_slot=False,
                    hydrometeors=True, numbers=False, terrain=None):
    rng = np.random.default_rng(seed)
    nlev = levels.size
    rows = np.arange(NY)[:, None]
    cols = np.arange(NX)[None, :]
    psfc = (99000.0 - 6.0 * (terrain if terrain is not None else 0.0)
            + 300.0 * np.sin(rows + 0.7 * cols)).astype(np.float32)
    pressure = np.broadcast_to(levels[:, None, None] * 100.0, (nlev, NY, NX)).copy()
    temperature = (215.0 + 75.0 * (pressure / 100000.0) ** 0.22
                   + rng.normal(0.0, 0.8, pressure.shape))
    height = -7900.0 * np.log(pressure / 100000.0) + rng.normal(0.0, 3.0, pressure.shape)
    fields = {
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": _field(levels, 12.0, -4.0, seed=seed + 1, stagger="u"),
        "VV": _field(levels, 10.0, -5.0, seed=seed + 2, stagger="v"),
        "PSFC": psfc,
        "T2": (288.0 + rng.normal(0.0, 1.5, (NY, NX))).astype(np.float32),
        "U10": _field(None, 6.0, -3.0, seed=seed + 3, stagger="u"),
        "V10": _field(None, 6.0, -3.0, seed=seed + 4, stagger="v"),
    }
    if specific:
        spfh = 0.012 * (pressure / 100000.0) ** 3 * (0.6 + 0.4 * rng.random(pressure.shape))
        if dry_slot:
            # A sharp dry slot between moister levels, in every other
            # column: WRF's second-order vertical operator undershoots it.
            spfh[nlev // 2, ::2, :] = 1.0e-9
            spfh[nlev // 2 - 1, ::2, :] *= 3.0
            spfh[nlev // 2 + 1, ::2, :] *= 3.0
        fields.update({
            "PRES": pressure.astype(np.float32),
            "SPFH": spfh.astype(np.float32),
            "Q2": (0.009 + 0.002 * rng.random((NY, NX))).astype(np.float32),
        })
    else:
        fields.update({
            "RH": (30.0 + 60.0 * rng.random(pressure.shape)).astype(np.float32),
            "D2": (fields["T2"] - 4.0 - 3.0 * rng.random((NY, NX))).astype(np.float32),
            "PMSL": (101300.0 + 200.0 * rng.random((NY, NX))).astype(np.float32),
        })
    if hydrometeors:
        for index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
            value = index * 1.0e-5 * rng.random(pressure.shape)
            value[value < 2.0e-6 * index] = 0.0
            fields[name] = value.astype(np.float32)
    if numbers:
        fields["QNWFA"] = (1.0e8 * rng.random(pressure.shape)).astype(np.float32)
        fields["QNIFA"] = (1.0e5 * rng.random(pressure.shape)).astype(np.float32)
        fields["QNWFA_SFC"] = (1.0e8 * rng.random((NY, NX))).astype(np.float32)
        fields["QNIFA_SFC"] = (1.0e5 * rng.random((NY, NX))).astype(np.float32)
    if terrain is not None:
        fields["SOURCE_OROGRAPHY"] = (terrain + rng.normal(0.0, 15.0, terrain.shape)).astype(np.float32)
    return fields


def _terrain():
    rows = np.arange(NY)[:, None]
    cols = np.arange(NX)[None, :]
    return (300.0 + 120.0 * np.sin(0.9 * rows) * np.cos(0.6 * cols)).astype(np.float64)


def _initialize(case, block_rows, *, boundary_only=False, velocity=None, tracers=False):
    terrain = _terrain()
    eta = np.linspace(1.0, 0.0, NZ + 1)
    levels = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0])
    donor = None
    options = {}
    if case == "flag_sh_mp28_donor":
        fields = _column_varying(levels, terrain=terrain)
        donor_levels = np.array([100.0, 250.0, 400.0, 550.0, 700.0, 800.0, 900.0, 975.0])
        donor_fields = _column_varying(donor_levels, terrain=terrain, seed=7,
                                       hydrometeors=False, numbers=True)
        donor = HorizontalSnapshot(valid_time=VALID, levels_hpa=donor_levels,
                                   fields=donor_fields)
        mp, use_sh_qv, sfcp = 28, False, True
    elif case == "use_sh_qv_dry_slot":
        fields = _column_varying(levels, terrain=terrain, dry_slot=True, seed=3)
        mp, use_sh_qv, sfcp = 8, True, True
    else:
        fields = _column_varying(levels, specific=False, hydrometeors=False, seed=5)
        fields["SOURCE_OROGRAPHY"] = terrain.astype(np.float32)
        mp, use_sh_qv, sfcp = 6, False, False
    if velocity:
        fields[velocity] = _field(levels, 16.0, -8.0, seed=91)
    boundary_tracers = ({"smoke": _field(levels, 50.0, 0.0, seed=92)}
                        if tracers else {})
    snapshot = HorizontalSnapshot(valid_time=VALID, levels_hpa=levels, fields=fields,
                                  boundary_tracers=boundary_tracers)
    cfg = RunConfig(
        nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0, ztop=18000.0, dt=15.0,
        run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        mp_physics=mp)
    if donor is not None:
        options["aerosol_snapshot"] = donor
    timing = {}
    result = initialize_real(
        snapshot, cfg, make_vertical_coord(NZ, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, p_top=10000.0, use_sh_qv=use_sh_qv, sfcp_to_sfcp=sfcp,
        preprocess_backend="cpu", preprocess_workers=3, state_backend="preprocess",
        landmask=np.ones((NY, NX), dtype=np.float32), boundary_only=boundary_only,
        column_block_rows=block_rows, timing_report=timing, **options)
    return result, timing


def _arrays(value):
    return {name: array for name, array in vars(value).items()
            if isinstance(array, np.ndarray)}


def _assert_same(whole, blocked):
    for name in ("surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
                 "total_pressure", "total_geopotential", "total_specific_volume",
                 "integrated_moisture_pressure"):
        left, right = getattr(whole, name), getattr(blocked, name)
        if left is None:
            assert right is None, name
            continue
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name
    for name, left in _arrays(whole.base).items():
        assert left.tobytes() == getattr(blocked.base, name).tobytes(), f"base.{name}"
    state_whole, state_blocked = _arrays(whole.state), _arrays(blocked.state)
    assert set(state_whole) == set(state_blocked)
    for name, left in state_whole.items():
        assert left.tobytes() == state_blocked[name].tobytes(), f"state.{name}"
    assert (whole.state._external_scalar_boundary_fields
            == blocked.state._external_scalar_boundary_fields)
    for name in ("hydrometeor_initialization", "aerosol_initialization",
                 "surface_moisture_floor", "prognostic_moisture_floor",
                 "initial_perturbation", "vertical_velocity_initialization"):
        assert repr(getattr(whole, name)) == repr(getattr(blocked, name)), name
    assert set(whole.boundary_tracers) == set(blocked.boundary_tracers)
    for name, value in whole.boundary_tracers.items():
        assert value.tobytes() == blocked.boundary_tracers[name].tobytes(), name


@pytest.fixture(scope="module")
def native_bridge():
    try:
        from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
        resolve_preprocess_backend("cpu", workers=1)._native.prepare_vertical_geometry
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")


@pytest.mark.parametrize("case", ("flag_sh_mp28_donor", "use_sh_qv_dry_slot",
                                  "pressure_level_sfcprs3"))
@pytest.mark.parametrize("boundary_only", (False, True))
def test_every_block_count_gives_the_whole_domain_bytes(native_bridge, case, boundary_only):
    whole, whole_timing = _initialize(case, NY, boundary_only=boundary_only)
    for rows in (1, 2, 4, 5):
        blocked, timing = _initialize(case, rows, boundary_only=boundary_only)
        _assert_same(whole, blocked)
        # One total per stage, the same stages either way.
        assert set(timing) == set(whole_timing) | {"domain_state_allocation"}


def test_the_dry_slot_case_really_exercises_the_merged_floor(native_bridge):
    whole, _ = _initialize("use_sh_qv_dry_slot", NY)
    receipt = whole.prognostic_moisture_floor
    assert receipt and receipt["floored_cells"] > 0, (
        "the fixture no longer makes the vertical operator undershoot; the "
        "block merge of the floor receipt is then untested")


def test_the_donor_case_carries_analyzed_aerosol_and_receipts(native_bridge):
    whole, _ = _initialize("flag_sh_mp28_donor", NY)
    receipt = whole.hydrometeor_initialization
    assert receipt["vertical_disposition"], "the replay receipt was not built"
    assert receipt["number_moments"]["separate_aerosol_donor"] is True
    assert np.count_nonzero(whole.state.nwfa) > 0


@pytest.mark.parametrize("velocity", ("WW", "OMEGA"))
@pytest.mark.parametrize("boundary_only", (False, True))
def test_blocks_retain_vertical_motion_and_smoke(native_bridge, velocity, boundary_only):
    """Row blocking must retain the later native-source analysis capabilities."""
    options = dict(boundary_only=boundary_only, velocity=velocity, tracers=True)
    whole, _ = _initialize("flag_sh_mp28_donor", NY, **options)
    assert np.count_nonzero(whole.state.w) > 0
    assert np.count_nonzero(whole.boundary_tracers["smoke"]) > 0
    assert whole.vertical_velocity_initialization["source_field"] == velocity
    for rows in (1, 4):
        blocked, _ = _initialize("flag_sh_mp28_donor", rows, **options)
        _assert_same(whole, blocked)


def test_blocks_trim_once_per_block_and_restore_the_pool_trim(native_bridge):
    from gpuwm.ingest import real

    trims = []
    original = real._return_freed_host_memory
    real._return_freed_host_memory = lambda: trims.append(1)
    try:
        _initialize("pressure_level_sfcprs3", 2)
        blocked = len(trims)
        trims.clear()
        _initialize("pressure_level_sfcprs3", NY)
        whole = len(trims)
    finally:
        real._return_freed_host_memory = original
    assert real._POOL_TRIM.get() is True
    # Five blocks of two rows (the last one), each trimmed once, plus the
    # handful of whole-domain trims; never one per helper pool per block.
    assert blocked < whole + 2 * 5


def test_block_rows_are_sized_from_the_domain_and_refused_off_the_host():
    cfg = RunConfig(nx=4160, ny=2450, nz=50, dx=1000.0, dy=1000.0, ztop=20000.0,
                    dt=6.0, run_seconds=60.0, moist=True)
    rows = _initialization_block_rows(cfg, None, eligible=True)
    assert rows * cfg.nx >= INITIALIZATION_BLOCK_COLUMNS > (rows - 1) * cfg.nx
    assert _initialization_block_rows(cfg, None, eligible=False) == cfg.ny
    # Each of 48 column workers takes at least 16 rows of a block: one or
    # two rows each left the per-level NumPy steps queued on the
    # interpreter lock (a blocked boundary-only 1 km time took 352 s where
    # the whole build took 145 s).
    assert _initialization_block_rows(cfg, None, eligible=True, workers=48) == 768
    # Where the fastest blocks' work arrays would take more than an eighth
    # of the host memory left, they narrow toward the least-memory size.
    from gpuwm.ingest.real import initialization_column_bytes
    per_row = initialization_column_bytes(cfg, 50) * cfg.nx
    assert _initialization_block_rows(cfg, None, eligible=True, workers=48,
                                      source_levels=50, available=10**15) == 768
    assert _initialization_block_rows(cfg, None, eligible=True, workers=48,
                                      source_levels=50, available=8 * 200 * per_row) == 200
    assert _initialization_block_rows(cfg, None, eligible=True, workers=48,
                                      source_levels=50, available=0) == rows
    small = RunConfig(nx=100, ny=100, nz=50, dx=1000.0, dy=1000.0, ztop=20000.0,
                      dt=6.0, run_seconds=60.0, moist=True)
    assert _initialization_block_rows(small, None, eligible=True) == small.ny
    assert _initialization_block_rows(cfg, 10**6, eligible=False) == cfg.ny
    with pytest.raises(ValueError, match="holds its state on the card"):
        _initialization_block_rows(cfg, 64, eligible=False)
    with pytest.raises(ValueError, match="positive"):
        _initialization_block_rows(cfg, 0, eligible=True)
