"""WRF real's second-order vertical operator at a sharp dry slot.

A real-data initialization on the use_sh_qv lane refused a whole domain
with ``interpolated specific-humidity qv is invalid``: the forcing column's
vapour was finite and non-negative on every source level, and WRF real's
second-order Lagrange vertical operator took the target levels that land
inside a mid-tropospheric dry slot (43 kPa; source 1.0e-3, 3.4e-6, 2.3e-5,
3.6e-4 kg/kg on four consecutive levels) below zero.  WRF's own real.exe
writes that negative to wrfinput, because the use_sh_qv lane is the one
lane that never passes through ``rh_to_mxrat`` and so never meets the
``qv_min`` floor the RH lane gets for free.

The column that did it is kept under ``tests/data`` and fed through the
engine's own real here, on the float64 reference vertical operator, so the
undershoot is WRF's arithmetic and not a fabricated array.  Without
:func:`gpuwm.ingest.real._floor_sh_vertical_undershoot` the first test in
this file fails on the refusal it is named for.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest import real as real_module
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.real import initialize_real

FIXTURE = Path(__file__).with_name("data") / "real_init_dry_slot_column.json"
NY, NX = 2, 3
#: The target levels of the 55-level ladder below that fall inside the
#: dry slot, and the floor of what the operator makes of them there.
SLOT_LEVELS = [41, 42]
SLOT_MIN = -1.15e-05
#: The ladder the column is initialized onto: WRF's hybrid coordinate,
#: tanh-clustered toward the surface, 5 kPa model top.
NZ, STRETCH, P_TOP = 55, 1.6, 5000.0


def _column() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _columns(column: dict, name: str) -> np.ndarray:
    value = np.asarray(column[name], dtype=np.float64)
    return np.broadcast_to(value[:, None, None], (value.size, NY, NX)).copy()


def _snapshot(column: dict, *, smooth_slot: bool = False) -> HorizontalSnapshot:
    """The fixture's column as a specific-humidity-authority snapshot."""
    from gpuwm.core import constants as c

    pressure = _columns(column, "PB") + _columns(column, "P")
    theta = _columns(column, "T") + 300.0
    temperature = theta * (pressure / float(c.P0)) ** (float(c.RD) / float(c.CP))
    qv = np.maximum(_columns(column, "QVAPOR"), 0.0)
    if smooth_slot:
        # The same column without its slot: levels 25..28 replaced by a
        # monotone bridge between their neighbours.  Everything else --
        # the rest of the column, the ladder, the operator -- is held
        # fixed, so the floor going dormant is attributable to the slot.
        qv[25:29] = np.linspace(qv[24], qv[29], 6)[1:-1]
    spfh = qv / (1.0 + qv)
    # WPS's 9.81 m s-2 convention, the one GHT is written under.
    z_w = (_columns(column, "PHB") + _columns(column, "PH")) / 9.81
    ght = 0.5 * (z_w[:-1] + z_w[1:])
    nsource = pressure.shape[0]
    q2 = max(float(column["Q2"]), 0.0)

    def f32(value):
        return np.ascontiguousarray(value, dtype=np.float32)

    fields = {
        "TT": f32(temperature), "PRES": f32(pressure), "SPFH": f32(spfh),
        "GHT": f32(ght),
        "UU": f32(np.full((nsource, NY, NX + 1), 5.0)),
        "VV": f32(np.full((nsource, NY + 1, NX), -3.0)),
        "PSFC": f32(np.full((NY, NX), column["PSFC"])),
        "T2": f32(np.full((NY, NX), column["T2"])),
        "Q2": f32(np.full((NY, NX), q2 / (1.0 + q2))),
        "U10": f32(np.full((NY, NX + 1), 2.0)),
        "V10": f32(np.full((NY + 1, NX), -1.0)),
    }
    levels_hpa = np.asarray(np.mean(pressure, axis=(1, 2)) / 100.0)
    return HorizontalSnapshot(
        valid_time=datetime(2026, 9, 13, 15, 45), levels_hpa=levels_hpa,
        fields=fields, specific_humidity_authority=True, analyzed_species=())


class _ReferenceVerticalPlan:
    """WRF real's vertical interpolation through the NumPy reference."""

    def __init__(self, source, surface, target):
        self.source = np.asarray(source, dtype=np.float32)
        self.surface = np.asarray(surface, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def apply(self, field, surface_value, **options):
        from gpuwm.verify.npref import np_wrf_real_vert_interp

        options.pop("values_are_finite", None)
        return np.asarray(np_wrf_real_vert_interp(
            field, surface_value, self.source, self.surface, self.target,
            **options), dtype=np.float32)


class _ReferencePreprocessBackend:
    """The preprocessing ABI over the NumPy reference operator.

    Deliberately not the packaged Rust CPU bridge: the undershoot under
    test is a property of WRF's own second-order Lagrange operator, and
    this test must not skip on a machine where the bridge is unbuilt.
    Spelled out here rather than imported from ``test_real_init``, which
    carries the same double, because that module imports ``conftest`` and
    so cannot be imported from a source distribution.
    """

    name = "cpu-reference-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def regular_plan(*args, **kwargs):
        raise AssertionError("horizontal preprocessing is unused here")

    masked_nearest = regular_plan
    rotate_earth_to_grid = regular_plan
    era5_rh_to_water = regular_plan

    @staticmethod
    def prepare_wrf_vertical(source, surface, target):
        return _ReferenceVerticalPlan(source, surface, target)

    @staticmethod
    def receipt():
        return {"backend": "cpu-reference-test"}


def _real_on_the_column(column: dict, *, smooth_slot: bool = False):
    cfg = RunConfig(
        nx=NX, ny=NY, nz=NZ, dx=float(column["attrs"]["DX"]),
        dy=float(column["attrs"]["DY"]), ztop=20000.0, dt=1.25,
        run_seconds=900.0, hybrid_opt=2, etac=0.2, hypsometric_opt=2,
        moist=True, terrain_opt=1, mp_physics=8)
    coord = make_vertical_coord(NZ, stretch=STRETCH, hybrid_opt=2, etac=0.2)
    terrain = np.full((NY, NX), float(column["HGT"]), dtype=np.float64)
    return initialize_real(
        _snapshot(column, smooth_slot=smooth_slot), cfg, coord, terrain,
        source_orography=terrain, p_top=P_TOP, sfcp_to_sfcp=True,
        column_workers=1,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="preprocess")


def test_the_dry_slot_undershoot_is_floored_at_wrf_qv_min_and_receipted():
    result = _real_on_the_column(_column())
    qv = np.asarray(result.state.qv, dtype=np.float64)
    assert np.isfinite(qv).all()
    assert float(qv.min()) == pytest.approx(real_module._WRF_QV_MIN_VALUE,
                                            rel=1e-6)
    receipt = result.prognostic_moisture_floor
    assert receipt["policy"] == (
        "use-sh-qv-vertical-undershoot-floored-to-wrf-qv-min-value")
    assert receipt["levels"] == SLOT_LEVELS
    assert receipt["floored_cells"] == len(SLOT_LEVELS) * NY * NX
    assert receipt["columns"] == NY * NX
    assert SLOT_MIN < receipt["min_pre_floor"] < 0.0
    assert receipt["min_pre_floor_at"]["level"] == SLOT_LEVELS[0]
    assert 40_000.0 < receipt["min_pre_floor_at"]["pressure_pa"] < 46_000.0
    assert receipt["qv_min_value"] == real_module._WRF_QV_MIN_VALUE
    assert receipt["qv_min_p_safe"] == real_module._WRF_QV_MIN_P_SAFE
    assert receipt["wrf_reference"]["wrf_version"] == "v4.7.1"
    # The floored levels carry WRF's floor and nothing else moved: the
    # levels either side of the slot are the operator's own values.
    for level in SLOT_LEVELS:
        np.testing.assert_array_equal(
            qv[level], np.float32(real_module._WRF_QV_MIN_VALUE))
    assert float(qv[SLOT_LEVELS[0] - 1].min()) > 1.0e-4
    assert float(qv[SLOT_LEVELS[-1] + 1].min()) > 1.0e-4


def test_the_floor_is_dormant_on_the_same_column_without_its_slot():
    result = _real_on_the_column(_column(), smooth_slot=True)
    assert result.prognostic_moisture_floor == {}
    qv = np.asarray(result.state.qv, dtype=np.float64)
    assert float(qv.min()) > real_module._WRF_QV_MIN_VALUE


def test_the_garbage_guard_keeps_refusing_by_name():
    qv = np.full((4, 2, 3), 2.0e-3)
    pressure = np.full((4, 2, 3), 80_000.0)
    # Nothing negative: the array comes back as it is, with no receipt.
    same, receipt = real_module._floor_sh_vertical_undershoot(qv, pressure)
    np.testing.assert_array_equal(same, qv)
    assert receipt == {}
    # A value inside [0, qv_min) is NOT the floor's business: only what the
    # operator took below zero moves, so no artifact that passed before the
    # floor existed changes.
    tiny = qv.copy()
    tiny[2, 1, 2] = 5.0e-7
    same, receipt = real_module._floor_sh_vertical_undershoot(tiny, pressure)
    assert same[2, 1, 2] == 5.0e-7 and receipt == {}
    # A non-finite value is refused naming where it is.
    bad = qv.copy()
    bad[3, 1, 2] = np.nan
    with pytest.raises(ValueError, match=(
            r"interpolated specific-humidity qv is invalid \| observed: "
            r"non_finite_cells=1, first at level 3 row 1 column 2 value nan")):
        real_module._refuse_non_finite_prognostic_qv(bad)
    # A negative under WRF's conditional rule (p < qv_min_p_safe) is
    # floored and receipted.
    deep = qv.copy()
    deep[0, 0, 1] = -2.0e-6
    floored, receipt = real_module._floor_sh_vertical_undershoot(deep, pressure)
    assert floored[0, 0, 1] == real_module._WRF_QV_MIN_VALUE
    assert receipt["floored_cells"] == 1 and receipt["levels"] == [0]
    assert receipt["floored_cells_at_or_above_qv_min_p_safe"] == 0
    assert receipt["min_pre_floor"] == -2.0e-6
    assert receipt["min_pre_floor_at"] == {
        "level": 0, "row": 0, "column": 1, "pressure_pa": 80_000.0}


def test_a_negative_where_the_conditional_rule_stops_is_floored_not_refused():
    """At or above qv_min_p_safe the floor still applies, counted apart.

    WRF's conditional qv_min rule stops at qv_min_p_safe, but the RH lane's
    rh_to_mxrat1 floors at 1e-6 there unconditionally, and the refusal this
    replaced named no breakage a floored value would cause: a negative there
    stopped the whole domain's initialization for a cell the floor makes a
    valid state.  The receipt keeps the count, the extreme and its pressure.
    """
    qv = np.full((4, 2, 3), 2.0e-3)
    pressure = np.full((4, 2, 3), 80_000.0)
    qv[0, 0, 1] = -2.0e-6
    qv[2, 1, 0] = -5.0e-7
    pressure[0, 0, 1] = 110_000.0
    floored, receipt = real_module._floor_sh_vertical_undershoot(qv, pressure)
    assert floored[0, 0, 1] == real_module._WRF_QV_MIN_VALUE
    assert floored[2, 1, 0] == real_module._WRF_QV_MIN_VALUE
    assert float(floored.min()) == real_module._WRF_QV_MIN_VALUE
    assert receipt["floored_cells"] == 2
    assert receipt["floored_cells_at_or_above_qv_min_p_safe"] == 1
    assert receipt["levels"] == [0, 2] and receipt["columns"] == 2
    assert receipt["min_pre_floor"] == -2.0e-6
    assert receipt["min_pre_floor_at"] == {
        "level": 0, "row": 0, "column": 1, "pressure_pa": 110_000.0}
