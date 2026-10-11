"""The fast SBM bin state, its unit layer, constants and behaviour presets.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A conversion that is right on paper and wrong in WRF's terms: a cloud/rain
  split one bin off, a QI/QS split keyed on the wrong bin, a clip keyed on
  the wrong bulk field (WRF clips every snow bin on QS alone, so fresh small
  ice vanishes whenever large snow is absent), or a number diagnostic in the
  wrong unit.
* A ``chem_new`` layout that disagrees with WRF's scalar indices, which
  would put aerosol where drops belong the day the engine carries the bins.

The expectations are hand-computed from the physics: one bin holding q kg/kg
of drops of mass x g is q kg/kg of cloud or rain and q / x * 1000 drops per
kg.  A spectrum shifted by one bin must fail those expectations, so the
check is shown to be able to fail.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gpuwm.core.fast_sbm import constants as C
from gpuwm.core.fast_sbm import state as S
from gpuwm.core.fast_sbm.constants import FsbmBehavior, FsbmRefusal
from gpuwm.core.fast_sbm.tables import load_tables


@pytest.fixture(scope="module")
def tables():
    return load_tables("float64")


def _bins(nz: int, ff1=None, ff5=None, ff6=None, ff8=None) -> S.FsbmColumnBins:
    zero33 = np.zeros((nz, 33))
    return S.FsbmColumnBins(
        ff1=zero33.copy() if ff1 is None else np.asarray(ff1, dtype=float),
        ff5=zero33.copy() if ff5 is None else np.asarray(ff5, dtype=float),
        ff6=zero33.copy() if ff6 is None else np.asarray(ff6, dtype=float),
        ff8=np.zeros((nz, 43)) if ff8 is None else np.asarray(ff8, dtype=float))


def _random_bins(rng, nz: int) -> S.FsbmColumnBins:
    def spectrum(width):
        return 10.0 ** rng.uniform(-12.0, -3.0, size=(nz, width))
    return _bins(nz, spectrum(33), spectrum(33), spectrum(33),
                 10.0 ** rng.uniform(0.0, 9.0, size=(nz, 43)))


# ---------------------------------------------------------------------------
# Constants and presets
# ---------------------------------------------------------------------------

def test_the_constants_are_wrfs_literals():
    assert (C.NKR, C.NKR_AEROSOL, C.ICEMAX, C.JMAX, C.JBREAK) == (33, 43, 3, 33, 18)
    assert (C.KRMIN_BREAKUP, C.KRDROP, C.KRICE, C.KR_ICEMPL, C.KRFREEZ) == (31, 15, 18, 9, 21)
    assert C.COL == 0.23105 and C.COL != math.log(2.0) / 3.0
    assert (C.NCOND, C.NCOLL) == (3, 1)
    assert (C.AA1_MY, C.BB1_MY, C.AA2_MY, C.BB2_MY) == (2.53e12, 5.42e3, 3.41e13, 6.13e3)
    assert (C.AL1, C.AL2, C.ALCR, C.ALCR_G) == (2500.0, 2834.0, 0.5, 100.0)
    assert (C.TCRIT, C.TTCOAL) == (270.15, 233.15)
    assert (C.IONS, C.MWAERO, C.RO_SOLUTE) == (3, 115.0, 1.79)
    assert (C.HAIL_OPT_DEFAULT, C.ICEPROCS_DEFAULT) == (1, 1)


def test_the_chem_new_layout_is_wrfs():
    layout = (C.P_FF1I01, C.P_FF1I33, C.P_FF5I01, C.P_FF5I33, C.P_FF6I01,
              C.P_FF6I33, C.P_FF8I01, C.P_FF8I43)
    assert layout == (2, 34, 35, 67, 68, 100, 101, 143)
    assert C.N_BIN_SCALARS == 142


def test_the_behaviour_presets_are_the_design_contract():
    wrf = FsbmBehavior.wrf_v480()
    assert (wrf.rainnc_velocity, wrf.rainnc_source, wrf.nucleation_water_sink,
            wrf.dtcond_carry, wrf.breakup_retry_cap, wrf.cloud_base_rule,
            wrf.fall_speed_lookup) == (
        "tile_last", "level1_after_fall", False, True, 64, "wrf_v480",
        "wrf_v480")
    fixed = FsbmBehavior.corrected()
    assert (fixed.rainnc_velocity, fixed.rainnc_source,
            fixed.nucleation_water_sink, fixed.dtcond_carry,
            fixed.breakup_retry_cap, fixed.cloud_base_rule,
            fixed.fall_speed_lookup) == (
        "own_column", "bottom_outflow", True, False, 64, "first_level",
        "own_bin")
    with pytest.raises(AttributeError):
        fixed.dtcond_carry = True  # frozen


def test_every_behaviour_names_its_cloud_base_rule():
    """D9's field has no default: a behaviour built field by field without
    it would otherwise pick the ``Cloud_Base_Super`` points silently, and an
    A/B arm made from a preset keeps that preset's rule."""

    five = dict(rainnc_velocity="own_column", rainnc_source="bottom_outflow",
                nucleation_water_sink=True, dtcond_carry=False,
                breakup_retry_cap=64)
    with pytest.raises(TypeError, match="cloud_base_rule"):
        FsbmBehavior(**five)
    arm = dataclasses.replace(FsbmBehavior.wrf_v480(),
                              rainnc_velocity="own_column")
    assert arm.cloud_base_rule == "wrf_v480"
    arm = dataclasses.replace(FsbmBehavior.corrected(),
                              nucleation_water_sink=False)
    assert arm.cloud_base_rule == "first_level"
    assert C.CLOUD_BASE_RULES == ("wrf_v480", "first_level")


def test_every_behaviour_names_its_fall_speed_lookup():
    """D10's field has no default either: a behaviour built without it
    would otherwise take one collision table or the other silently, and an
    A/B arm made from a preset keeps that preset's lookup."""

    six = dict(rainnc_velocity="own_column", rainnc_source="bottom_outflow",
               nucleation_water_sink=True, dtcond_carry=False,
               breakup_retry_cap=64, cloud_base_rule="first_level")
    with pytest.raises(TypeError, match="fall_speed_lookup"):
        FsbmBehavior(**six)
    arm = dataclasses.replace(FsbmBehavior.wrf_v480(),
                              rainnc_velocity="own_column")
    assert arm.fall_speed_lookup == "wrf_v480"
    arm = dataclasses.replace(FsbmBehavior.corrected(),
                              cloud_base_rule="wrf_v480")
    assert arm.fall_speed_lookup == "own_bin"
    assert C.FALL_SPEED_LOOKUPS == ("wrf_v480", "own_bin")


@pytest.mark.parametrize("change", [
    {"rainnc_velocity": "last_column"},
    {"rainnc_source": "outflow"},
    {"nucleation_water_sink": 1},
    {"breakup_retry_cap": 0},
    {"breakup_retry_cap": True},
    {"cloud_base_rule": "intended"},
    {"cloud_base_rule": None},
    {"fall_speed_lookup": "beard"},
    {"fall_speed_lookup": None},
])
def test_a_malformed_behaviour_is_refused(change):
    fields = dict(rainnc_velocity="own_column", rainnc_source="bottom_outflow",
                  nucleation_water_sink=True, dtcond_carry=False,
                  breakup_retry_cap=64, cloud_base_rule="first_level",
                  fall_speed_lookup="own_bin")
    fields.update(change)
    with pytest.raises(FsbmRefusal, match="FsbmBehavior"):
        FsbmBehavior(**fields)


def test_the_refusal_carries_where_and_condition():
    refusal = FsbmRefusal("ONECOND1 :7070", "temperature jump above 3 K")
    assert refusal.where == "ONECOND1 :7070"
    assert refusal.condition == "temperature jump above 3 K"
    assert str(refusal) == "ONECOND1 :7070: temperature jump above 3 K"
    assert isinstance(refusal, RuntimeError)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hail_opt", [1, 0])
def test_bins_to_distributions_to_bins_round_trips(tables, hail_opt):
    rng = np.random.default_rng(20260923)
    nz = 40
    bins = _random_bins(rng, nz)
    rho = S.rho_cgs_from_si(rng.uniform(0.3, 1.25, size=nz))
    dist = S.bins_to_distributions(bins, rho, tables, hail_opt)
    if hail_opt == 1:
        assert not np.any(dist.ff4r) and np.all(dist.ff5r > 0)
    else:
        assert not np.any(dist.ff5r) and np.all(dist.ff4r > 0)
    bulk = S.bulk_diagnostics(dist, rho, tables, hail_opt)
    # every level is far above the clip, so the clip must change nothing
    assert np.all(bulk.qc + bulk.qr > 1e-13) and np.all(bulk.qs > 1e-13)
    back = S.distributions_to_bins(dist, rho, tables, bulk, hail_opt)
    for name in ("ff1", "ff5", "ff6", "ff8"):
        original = getattr(bins, name)
        relative = np.abs(getattr(back, name) - original) / original
        assert relative.max() <= 1e-15, (name, relative.max())


def test_the_entry_conversion_keeps_wrfs_operation_order(tables):
    q = np.array([[3.7e-5] * 33])
    rho = np.array([1.1e-3])
    dist = S.bins_to_distributions(_bins(1, ff1=q), rho, tables)
    xl = tables.xl
    for k in range(33):
        expected = (((q[0, k] * rho[0]) / C.COL) / xl[k]) / xl[k] / 3.0
        assert dist.ff1r[0, k] == expected


def test_negative_bins_stay_unless_the_point_loop_zeroing_is_asked(tables):
    ff1 = np.full((2, 33), 1e-6)
    ff1[0, 4] = -1e-9
    ff8 = np.full((2, 43), 1e3)
    ff8[1, 7] = -5.0
    rho = np.array([1e-3, 1e-3])
    kept = S.bins_to_distributions(_bins(2, ff1=ff1, ff8=ff8), rho, tables)
    assert kept.ff1r[0, 4] < 0 and kept.fccn[1, 7] < 0
    zeroed = S.bins_to_distributions(_bins(2, ff1=ff1, ff8=ff8), rho, tables,
                                     zero_negative=True)
    assert zeroed.ff1r[0, 4] == 0 and zeroed.fccn[1, 7] == 0
    assert np.all(zeroed.ff1r >= 0) and np.all(zeroed.fccn >= 0)


# ---------------------------------------------------------------------------
# Bulk diagnostics against hand-computed spectra
# ---------------------------------------------------------------------------

def _single_bin(tables, species: str, bin_one_based: int, q: float,
                hail_opt: int = 1):
    ff = np.zeros((1, 33))
    ff[0, bin_one_based - 1] = q
    rho = np.array([1.05e-3])
    bins = _bins(1, **{species: ff})
    dist = S.bins_to_distributions(bins, rho, tables, hail_opt)
    return S.bulk_diagnostics(dist, rho, tables, hail_opt)


def _expect(bulk, **nonzero):
    for field in ("qc", "qr", "qi", "qs", "qg", "qnc", "qnr", "qni", "qns",
                  "qng", "qna"):
        want = nonzero.get(field, 0.0)
        got = getattr(bulk, field)[0]
        if want == 0.0:
            assert got == 0.0, (field, got)
        else:
            assert got == pytest.approx(want, rel=1e-14), field


def test_cloud_and_rain_split_at_bin_15(tables):
    q = 2.5e-4
    radii = S.drop_radii(tables)
    assert round(radii[14] * 1e4, 1) == 50.8  # bin 15, 50.8 um
    for k in (1, 14):
        _expect(_single_bin(tables, "ff1", k, q),
                qc=q, qnc=q / tables.xl[k - 1] * 1000.0)
    for k in (15, 33):
        _expect(_single_bin(tables, "ff1", k, q),
                qr=q, qnr=q / tables.xl[k - 1] * 1000.0)


def test_cloud_ice_and_snow_split_at_bin_18(tables):
    q = 4.0e-6
    for k in (1, 18):
        _expect(_single_bin(tables, "ff5", k, q),
                qi=q, qni=q / tables.xs[k - 1] * 1000.0)
    for k in (19, 33):
        _expect(_single_bin(tables, "ff5", k, q),
                qs=q, qns=q / tables.xs[k - 1] * 1000.0)


def test_dense_ice_is_hail_or_graupel_by_hail_opt(tables):
    q = 1.0e-3
    k = 25
    _expect(_single_bin(tables, "ff6", k, q, hail_opt=1),
            qg=q, qng=q / tables.xh[k - 1] * 1000.0)
    _expect(_single_bin(tables, "ff6", k, q, hail_opt=0),
            qg=q, qng=q / tables.xg[k - 1] * 1000.0)


def test_aerosol_number_is_col_times_the_spectrum(tables):
    ff8 = np.linspace(1e6, 5e8, 43)[None, :]
    rho = np.array([9.0e-4])
    dist = S.bins_to_distributions(_bins(1, ff8=ff8), rho, tables)
    bulk = S.bulk_diagnostics(dist, rho, tables)
    assert bulk.qna[0] == pytest.approx(C.COL * math.fsum(ff8[0]), rel=1e-14)


def test_a_full_spectrum_sums_in_bin_order(tables):
    """Marshall-Palmer-like drops across both classes: QC and QR are the
    sums of the bins either side of the split, QNC and QNR likewise."""
    radii = S.drop_radii(tables)
    q = 1e-3 * np.exp(-radii / 0.05) * radii ** 3
    q = (q / q.sum() * 2.0e-3)[None, :]
    rho = np.array([1.1e-3])
    dist = S.bins_to_distributions(_bins(1, ff1=q), rho, tables)
    bulk = S.bulk_diagnostics(dist, rho, tables)
    cloud, rain = slice(0, 14), slice(14, 33)
    assert bulk.qc[0] == pytest.approx(math.fsum(q[0, cloud]), rel=1e-14)
    assert bulk.qr[0] == pytest.approx(math.fsum(q[0, rain]), rel=1e-14)
    assert bulk.qnc[0] == pytest.approx(
        math.fsum(q[0, cloud] / tables.xl[cloud]) * 1000.0, rel=1e-14)
    assert bulk.qnr[0] == pytest.approx(
        math.fsum(q[0, rain] / tables.xl[rain]) * 1000.0, rel=1e-14)
    # the accumulation is sequential, as the Fortran loop's is
    mass = (1.0 / rho[0]) * C.COL * dist.ff1r[0] * tables.xl * tables.xl * 3.0
    running = 0.0
    for value in mass[:14]:
        running = running + value
    assert bulk.qc[0] == running


def test_a_spectrum_shifted_by_one_bin_fails_the_check(tables):
    """Negative control: the same mass one bin up is rain, not cloud, and
    half as many drops; the hand-computed expectation must reject it."""
    q = 2.5e-4
    shifted = _single_bin(tables, "ff1", 15, q)
    with pytest.raises(AssertionError):
        _expect(shifted, qc=q, qnc=q / tables.xl[13] * 1000.0)
    heavier = _single_bin(tables, "ff1", 21, q)
    with pytest.raises(AssertionError):
        _expect(heavier, qr=q, qnr=q / tables.xl[19] * 1000.0)


def test_iceprocs_off_leaves_the_ice_diagnostics_zero(tables):
    rng = np.random.default_rng(7)
    bins = _random_bins(rng, 3)
    rho = np.full(3, 1e-3)
    dist = S.bins_to_distributions(bins, rho, tables)
    bulk = S.bulk_diagnostics(dist, rho, tables, iceprocs=0)
    for field in ("qi", "qs", "qg", "qni", "qns", "qng"):
        assert not np.any(getattr(bulk, field)), field
    assert np.all(bulk.qc > 0) and np.all(bulk.qna > 0)


# ---------------------------------------------------------------------------
# The exit clip, keyed as WRF keys it
# ---------------------------------------------------------------------------

def _bulk(nz, **fields):
    base = {f: np.zeros(nz) for f in ("qc", "qr", "qi", "qs", "qg", "qnc",
                                      "qnr", "qni", "qns", "qng", "qna")}
    base.update({k: np.asarray(v, dtype=float) for k, v in fields.items()})
    return S.FsbmBulk(**base)


def test_the_clip_zeroes_exactly_below_1e_13_per_species(tables):
    nz = 4
    below = np.nextafter(1e-13, 0.0)
    rho = np.full(nz, 1e-3)
    dist = S.FsbmDistributions(
        ff1r=np.ones((nz, 33)), ff3r=np.ones((nz, 33)),
        ff4r=np.zeros((nz, 33)), ff5r=np.ones((nz, 33)),
        fccn=np.full((nz, 43), 1e-30))
    bulk = _bulk(
        nz,
        # level 0: all at the threshold, nothing clipped
        # level 1: drops keyed on qc + qr: 0.6e-13 + 0.3e-13 is below
        # level 2: large cloud ice but qs below: every snow bin goes
        # level 3: qg just below: every dense-ice bin goes
        qc=[1e-13, 0.6e-13, 1.0, 1.0],
        qr=[0.0, 0.3e-13, 0.0, 0.0],
        qi=[0.0, 0.0, 1.0e-3, 0.0],
        qs=[1e-13, 1.0, below, 1.0],
        qg=[1e-13, 1.0, 1.0, below])
    out = S.distributions_to_bins(dist, rho, tables, bulk)
    assert np.all(out.ff1[0] > 0) and np.all(out.ff5[0] > 0) and np.all(out.ff6[0] > 0)
    assert not np.any(out.ff1[1]) and np.all(out.ff5[1] > 0) and np.all(out.ff6[1] > 0)
    assert np.all(out.ff1[2] > 0) and not np.any(out.ff5[2]) and np.all(out.ff6[2] > 0)
    assert np.all(out.ff1[3] > 0) and np.all(out.ff5[3] > 0) and not np.any(out.ff6[3])
    # aerosol is never clipped, however small
    assert np.all(out.ff8 > 0)
    # clip=None converts everything
    unclipped = S.distributions_to_bins(dist, rho, tables, bulk, clip=None)
    assert np.all(unclipped.ff1 > 0) and np.all(unclipped.ff5 > 0)


def test_the_clip_on_graupel_keys_on_qg_too(tables):
    rho = np.array([1e-3])
    dist = S.FsbmDistributions(
        ff1r=np.zeros((1, 33)), ff3r=np.zeros((1, 33)),
        ff4r=np.ones((1, 33)), ff5r=np.zeros((1, 33)),
        fccn=np.zeros((1, 43)))
    kept = S.distributions_to_bins(dist, rho, tables, _bulk(1, qg=[1e-13]),
                                   hail_opt=0)
    assert np.all(kept.ff6 > 0)
    gone = S.distributions_to_bins(dist, rho, tables, _bulk(1, qg=[5e-14]),
                                   hail_opt=0)
    assert not np.any(gone.ff6)


# ---------------------------------------------------------------------------
# chem_new layout
# ---------------------------------------------------------------------------

def test_the_chem_new_layout_round_trips(tables):
    rng = np.random.default_rng(11)
    bins = _random_bins(rng, 5)
    chem = S.to_chem_new(bins)
    assert chem.shape == (5, 142)
    # column c holds WRF scalar index c + 2
    assert np.array_equal(chem[:, C.P_FF1I01 - 2], bins.ff1[:, 0])
    assert np.array_equal(chem[:, C.P_FF1I33 - 2], bins.ff1[:, 32])
    assert np.array_equal(chem[:, C.P_FF5I01 - 2], bins.ff5[:, 0])
    assert np.array_equal(chem[:, C.P_FF6I33 - 2], bins.ff6[:, 32])
    assert np.array_equal(chem[:, C.P_FF8I01 - 2], bins.ff8[:, 0])
    assert np.array_equal(chem[:, C.P_FF8I43 - 2], bins.ff8[:, 42])
    back = S.from_chem_new(chem)
    for name in ("ff1", "ff5", "ff6", "ff8"):
        assert np.array_equal(getattr(back, name), getattr(bins, name))
    with pytest.raises(FsbmRefusal):
        S.from_chem_new(np.zeros((5, 143)))


# ---------------------------------------------------------------------------
# Drop radii and refusals
# ---------------------------------------------------------------------------

def test_drop_radii_span_2_um_to_3_25_mm(tables):
    radii = S.drop_radii(tables)
    assert radii.shape == (33,)
    assert radii[0] == pytest.approx(2.0e-4, rel=1e-4)
    assert radii[32] == pytest.approx(0.3251, rel=1e-4)
    assert radii[14] == pytest.approx(tables.rlec[14], rel=1e-4)
    expected = (3.0 * tables.xl / 4.0 / 3.141593 / 1.0) ** (1.0 / 3.0)
    assert np.array_equal(radii, expected)


def test_malformed_inputs_are_refused(tables):
    bins = _bins(3)
    rho = np.full(3, 1e-3)
    with pytest.raises(FsbmRefusal, match="hail_opt"):
        S.bins_to_distributions(bins, rho, tables, hail_opt=2)
    with pytest.raises(FsbmRefusal, match="rho_cgs"):
        S.bins_to_distributions(bins, np.full(2, 1e-3), tables)
    with pytest.raises(FsbmRefusal, match="positive"):
        S.bins_to_distributions(bins, np.array([1e-3, 0.0, 1e-3]), tables)
    with pytest.raises(FsbmRefusal, match="ff8"):
        S.bins_to_distributions(_bins(3, ff8=np.zeros((3, 33))), rho, tables)
    dist = S.bins_to_distributions(bins, rho, tables)
    with pytest.raises(FsbmRefusal, match="iceprocs"):
        S.bulk_diagnostics(dist, rho, tables, iceprocs=2)
    with pytest.raises(FsbmRefusal, match="bulk.qc"):
        S.distributions_to_bins(dist, rho, tables, _bulk(2))


def test_rho_cgs_from_si_is_wrfs_product():
    rho = np.array([1.2, 0.41])
    assert np.array_equal(S.rho_cgs_from_si(rho), rho * 0.001)
