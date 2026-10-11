"""The fast SBM derived tables and collision kernels against WRF's own Fortran.

``gpuwm/core/fast_sbm/derived.py`` transcribes what ``FAST_HUCMINIT`` derives
from the lookup tables (the Bott categories and Courant numbers, the mass
grids, ``DROPRADII``, the aerosol spectra, the coalescence efficiency matrix,
the spontaneous breakup weights) and ``Kernals_KS``, the collision kernels at
a pressure.  Every warm-rain process item of the fast-sbm program reads these
arrays, so they are gated here against the oracle dumps (fast-sbm design
5.3, 5.5) and against the physics they claim to encode.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A collision step that sends coalesced mass to the wrong bin: an ``ima``
  one category off, or a Courant number from a mass grid WRF never used.
  ``ima`` is gated exactly and ``chucm`` to relative 1e-12, and the analytic
  check proves each summed mass sits in the bracket ``ima`` names.
* A coalescence efficiency from the wrong branch, the wrong fall speeds or
  the wrong unit: the matrix is gated against the dump to 1e-12, the small
  drops are checked to coalesce with unit efficiency, and Low-List values
  are recomputed from Low and List (1982a) in SI units for pairs across the
  large-drop range.
* A kernel interpolated between the wrong slices or in the wrong direction,
  missing the time step, ``ln2/3`` or ``ECOALMASSM``, or a transposed
  partner left untransposed:
  every one of the 36 families at every one of the ten dumped pressures is
  gated to 1e-12, and one entry bent by 1e-6 must fail that gate.
* An aerosol spectrum or spontaneous breakup that creates or loses matter:
  the spectra must integrate to the mode totals inside WRF's radius window
  and the breakup fragments of one drop must carry that drop's mass.
* A loader view WRF never ran with: every float64 and float32 table view is
  compared byte for byte with the arrays WRF's READ filled in the promoted
  (O64) and native (O32) builds, which the tables item could not see.

The native build (O32) is not gated: its differences from the float64 host
are reported in float32 ULP through ``gpuwm.core.fp32_ulp`` (run with ``-s``
to read the table).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gpuwm.core.fast_sbm import derived as D
from gpuwm.core.fast_sbm import tables as T
from gpuwm.core.fast_sbm.constants import COL, FsbmBehavior, FsbmRefusal, NKR
from gpuwm.core.fp32_ulp import MISMATCH, fp32_ulp_distance
from tools.fast_sbm_wrf480_oracle import fixtures as fx

#: The oracle gate for the float64 host against the O64 build (design 5.5).
RTOL_O64 = 1e-12
#: The step the oracle ran FAST_HUCMINIT with (``tables_*`` manifest).
DT_ORACLE = 12.0
#: The oracle is WRF as written, so its tables are built with WRF's
#: behaviour (finding D10 decides ``ECOALMASSM``).
WRF = FsbmBehavior.wrf_v480()

#: The host FsbmDerived field holding each array of the dump's ``derived``
#: record that is gated to relative 1e-12.
RELATIVE_FIELDS = {
    "chucm": "chucm",
    "xl_mg": "xl_mg_6078",
    "xl_mg_bott": "xl_mg_1656",
    "ecoalmassm": "ecoalmassm",
    "brkweight": "brkweight",
    "prob": "prob",
    "gain_var_new": "gain_var_new",
    "nnd": "nnd",
    "fccnr_con": "fccnr_con",
    "fccnr_mar": "fccnr_mar",
    "xccn": "xccn",
    "rccn": "rccn",
    "dropradii": "dropradii",
}
#: Dump fields gated exactly, with the host field holding each.
EXACT_FIELDS = {
    "ima": "ima",
    "ikr_spon_break": "ikr_spon_break",
    "scale_ccn_factor": "scale_ccn_factor",
    "dt": "dt",
}

#: The FsbmTables contract for the stacked kernels, written out here so a
#: loader that relabels a slice is caught.
YWLL_SLICE_ORDER_HPA = (1000, 750, 500)
ICE_SLICE_ORDER_HPA = (300, 500, 750)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t64():
    return T.load_tables("float64")


@pytest.fixture(scope="module")
def t32():
    return T.load_tables("float32")


@pytest.fixture(scope="module")
def derived64(t64):
    return D.build_derived(t64, DT_ORACLE, behavior=WRF)


def _record(build: str, record: str) -> dict[str, np.ndarray]:
    dump = fx.load_fixture(f"tables_{build}")
    prefix = record + "."
    return {key[len(prefix):]: value[0] for key, value in dump.items()
            if key.startswith(prefix)}


@pytest.fixture(scope="module")
def dump64():
    return _record("o64", "derived")


@pytest.fixture(scope="module")
def dump32():
    return _record("o32", "derived")


@pytest.fixture(scope="module")
def kernels64():
    return fx.load_fixture("kernels_o64")


@pytest.fixture(scope="module")
def kernels32():
    return fx.load_fixture("kernels_o32")


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

def _relative_failures(label: str, host, want, rtol: float) -> list[str]:
    """Every element where ``host`` misses ``want`` by more than ``rtol``.

    A reference zero must be matched by an exact zero, since no relative
    tolerance exists there.  Shapes must agree exactly.
    """
    host = np.asarray(host, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if host.shape != want.shape:
        return [f"{label}: shape {host.shape}, oracle {want.shape}"]
    if not np.all(np.isfinite(host)):
        return [f"{label}: non-finite host values"]
    diff = np.abs(host - want)
    bad = np.where(want == 0.0, diff != 0.0, diff > rtol * np.abs(want))
    out = []
    for index in zip(*np.nonzero(bad)):
        one_based = tuple(int(i) + 1 for i in index)
        out.append(f"{label}{one_based}: host {host[index]!r}, oracle "
                   f"{want[index]!r}, relative "
                   f"{diff[index] / max(abs(want[index]), 1e-300):.3e}")
    return out


def _kernel_failures(kernels: D.FsbmKernels, dump: dict, n: int,
                     rtol: float) -> list[str]:
    failures = []
    for family in D.KERNEL_FAMILIES:
        failures += _relative_failures(
            f"{family}@{float(dump['kernels.p_hpa'][n]):g}hPa",
            getattr(kernels, family), dump[f"kernels.{family}"][n], rtol)
    return failures


# ---------------------------------------------------------------------------
# Table values: the loader views against WRF's READ (cross-lane rung)
# ---------------------------------------------------------------------------

def _loader_array(tables, field: str) -> tuple[str, np.ndarray]:
    prefix, _, suffix = field.rpartition("_")
    if prefix and suffix.endswith("mb"):
        order = (YWLL_SLICE_ORDER_HPA if prefix == "ywll"
                 else ICE_SLICE_ORDER_HPA)
        return prefix, getattr(tables, prefix)[order.index(int(suffix[:-2]))]
    return field, getattr(tables, field)


@pytest.mark.parametrize("build, precision",
                         [("o64", "float64"), ("o32", "float32")])
def test_every_loader_view_is_byte_identical_to_wrfs_read(build, precision):
    """load_tables(precision) against the arrays WRF's READ filled.

    Breakage prevented: a derived table gated against the oracle while the
    table it came from differs from the one the Fortran held.  The tables
    item built its loader without the oracle, so this is where its float64
    and float32 views meet WRF's own READ, byte for byte, dtype for dtype.
    """
    record = _record(build, "read")
    tables = T.load_tables(precision)
    covered = set()
    for field, expected in record.items():
        if field in ("dt", "bin_log"):   # the step, and log10(bin_mass)
            continue
        name, mine = _loader_array(tables, field)
        covered.add(name)
        assert mine.dtype == expected.dtype, (field, mine.dtype, expected.dtype)
        assert mine.shape == expected.shape, field
        assert mine.tobytes() == expected.tobytes(), field
    loader_fields = {f.name for f in dataclasses.fields(T.FsbmTables)}
    assert covered == loader_fields - {"precision", "root"}


# ---------------------------------------------------------------------------
# Derived tables against O64
# ---------------------------------------------------------------------------

def test_ima_is_wrfs_exactly(derived64, dump64):
    """The Bott category of every drop pair equals WRF's, integer for integer.

    Breakage prevented: coalesced mass landing one bin off.  ``ima`` keeps
    WRF's one-based values, including the zeros WRF leaves for every pair
    with bin 33.
    """
    assert derived64.ima.dtype == np.int32
    assert np.array_equal(derived64.ima, dump64["ima"])
    assert np.all(derived64.ima[:, NKR - 1] == 0)
    assert np.all(derived64.ima[NKR - 1, :] == 0)


def test_derived_tables_match_o64(derived64, dump64):
    """chucm, both mass grids, ECOALMASSM, BRKWEIGHT, Prob, Gain_Var_New,
    NND, the aerosol spectra and grids and DROPRADII within relative 1e-12
    of the promoted build; ikr_spon_break, Scale_CCN_Factor and dt exactly.

    Breakage prevented: a derived table that only resembles WRF's (a
    rounded pi, a reordered sum, a lost correction row), which every
    collision, breakup and nucleation gate downstream would inherit.
    """
    failures = []
    for mine, theirs in RELATIVE_FIELDS.items():
        failures += _relative_failures(mine, getattr(derived64, mine),
                                       dump64[theirs], RTOL_O64)
    assert not failures, "\n".join(failures[:20])
    assert derived64.ikr_spon_break == int(dump64["ikr_spon_break"]) == 33
    assert derived64.scale_ccn_factor == float(dump64["scale_ccn_factor"])
    assert derived64.dt == float(dump64["dt"]) == DT_ORACLE
    gated = set(RELATIVE_FIELDS.values()) | set(EXACT_FIELDS)
    assert gated == set(dump64), "a dumped derived quantity is not gated"


def test_the_derived_gate_can_fail(derived64, dump64):
    """Negative control: a Courant number bent by 1e-10 relative, and one
    ECOALMASSM entry bent the same way, each fail the 1e-12 gate."""
    for name in ("chucm", "ecoalmassm"):
        bent = np.array(getattr(derived64, name), copy=True)
        index = tuple(int(i) for i in np.argwhere(bent > 0.01)[0])
        bent[index] *= 1.0 + 1e-10
        failures = _relative_failures(name, bent, dump64[RELATIVE_FIELDS[name]],
                                      RTOL_O64)
        assert len(failures) == 1, failures


# ---------------------------------------------------------------------------
# Kernels against O64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(10))
def test_kernels_at_each_dumped_pressure_match_o64(t64, derived64, kernels64, n):
    """Every kernel family after Kernals_KS(12 s, 33, p) within relative
    1e-12 of the promoted build at 1000, 950, 850, 750, 700, 600, 500, 400,
    300 and 250 hPa.

    Breakage prevented: a kernel interpolated between the wrong slices or
    in the wrong direction, outside its slice range, without dt, ln2/3 or
    ECOALMASSM, or a transposed partner left as is.
    """
    p = float(kernels64["kernels.p_z"][n])
    dt = float(kernels64["kernels.dtime_coal"][n])
    assert p == float(kernels64["kernels.p_hpa"][n]) * 1000.0
    kernels = D.kernels_at_pressure(t64, derived64, p, dt)
    assert (kernels.p_cgs, kernels.dt_coll) == (p, dt)
    failures = _kernel_failures(kernels, kernels64, n, RTOL_O64)
    assert not failures, "\n".join(failures[:20])
    assert np.array_equal(D.drop_drop_kernel(t64, derived64, p, dt),
                          kernels.cwll)


def test_every_dumped_kernel_family_is_a_host_family(kernels64):
    """Breakage prevented: a family the oracle dumped that the host never
    forms, so the gate above silently skips it."""
    dumped = {key.split(".", 1)[1] for key in kernels64}
    dumped -= {"p_hpa", "p_z", "dtime_coal"}
    assert dumped == set(D.KERNEL_FAMILIES)
    host = {f.name for f in dataclasses.fields(D.FsbmKernels)}
    assert host - {"p_cgs", "dt_coll"} == set(D.KERNEL_FAMILIES)


def test_the_kernel_gate_can_fail(t64, derived64, kernels64):
    """Negative control: one kernel entry bent by 1e-6 relative fails the
    kernel gate, and only that entry does."""
    n = 4   # 700 hPa, inside both interpolation ranges
    p = float(kernels64["kernels.p_z"][n])
    dt = float(kernels64["kernels.dtime_coal"][n])
    kernels = D.kernels_at_pressure(t64, derived64, p, dt)
    bent_cwll = np.array(kernels.cwll, copy=True)
    index = tuple(int(i) for i in np.argwhere(bent_cwll > 0.0)[0])
    bent_cwll[index] *= 1.0 + 1e-6
    bent = dataclasses.replace(kernels, cwll=bent_cwll)
    failures = _kernel_failures(bent, kernels64, n, RTOL_O64)
    assert len(failures) == 1 and failures[0].startswith("cwll@700hPa"), failures


def test_the_init_call_is_the_edge_slices(t64, derived64):
    """Kernals_KS(dt, nkr, 7.6E6) at FAST_HUCMINIT (:6215) sits above every
    slice range: the drop-drop kernel is YWLL at 1000 hPa and every ice
    family the 750 hPa slice.  Those values are superseded at every
    collision call (:8318), which is why FsbmDerived carries no kernels.

    Breakage prevented: a host that stores the init kernels and hands them
    to a collision step at another pressure.
    """
    k = D.kernels_at_pressure(t64, derived64, 7.6e6, DT_ORACLE)
    dlnr = math.log(2.0) / 3.0
    assert np.array_equal(
        k.cwll, derived64.ecoalmassm * (t64.ywll[0] * DT_ORACLE * dlnr))
    assert np.array_equal(k.cwlg, t64.ywlg[2] * (DT_ORACLE * dlnr))
    assert np.array_equal(k.cwgl, k.cwlg.T)
    assert np.array_equal(k.cwii_2_3, t64.ywii[2][:, :, 1, 2] * (DT_ORACLE * dlnr))
    for family in ("cwig_1", "cwig_2", "cwig_3", "cwih_1", "cwih_2", "cwih_3"):
        assert not np.any(getattr(k, family))


# ---------------------------------------------------------------------------
# The native build, reported in float32 ULP
# ---------------------------------------------------------------------------

def test_o32_differences_reported_in_ulp(t64, t32, dump32, kernels32):
    """Report, not a gate: the native build's derived tables and kernels
    against the float64 host, in float32 ULP through gpuwm.core.fp32_ulp.

    Two host inputs are measured: the float64 tables (the whole distance
    from WRF as it runs to the reference) and the float32 tables (the part
    left once the host reads what the native build reads).  The only
    assertions are coverage and finiteness: breakage prevented is a report
    that silently drops a field or hides a NaN behind a number.
    """
    lines = ["O32 against the float64 host, max float32 ULP (count of "
             "elements that differ):"]
    measured = set()
    for label, tables in (("float64 tables", t64), ("float32 tables", t32)):
        derived = D.build_derived(tables, DT_ORACLE, behavior=WRF)
        lines.append(f"  host on {label}:")
        ima_differ = int(np.count_nonzero(derived.ima != dump32["ima"]))
        lines.append(f"    ima: {ima_differ} pairs differ")
        for mine, theirs in RELATIVE_FIELDS.items():
            ulp = fp32_ulp_distance(
                np.asarray(getattr(derived, mine), dtype=np.float32),
                np.asarray(dump32[theirs], dtype=np.float32))
            assert int(ulp.max()) < MISMATCH, mine
            measured.add(mine)
            lines.append(f"    {mine}: {int(ulp.max())} "
                         f"({int(np.count_nonzero(ulp))})")
        worst: dict[str, int] = {}
        for n in range(10):
            kernels = D.kernels_at_pressure(
                tables, derived, float(kernels32["kernels.p_z"][n]),
                float(kernels32["kernels.dtime_coal"][n]))
            for family in D.KERNEL_FAMILIES:
                ulp = fp32_ulp_distance(
                    np.asarray(getattr(kernels, family), dtype=np.float32),
                    np.asarray(kernels32[f"kernels.{family}"][n],
                               dtype=np.float32))
                assert int(ulp.max()) < MISMATCH, family
                worst[family] = max(worst.get(family, 0), int(ulp.max()))
        measured |= set(worst)
        lines.append("    kernels, worst over the ten pressures: " + ", ".join(
            f"{family} {value}" for family, value in worst.items() if value))
    print("\n".join(lines))
    assert measured == set(RELATIVE_FIELDS) | set(D.KERNEL_FAMILIES)


# ---------------------------------------------------------------------------
# Analytic and published checks
# ---------------------------------------------------------------------------

def test_courant_numbers_bracket_the_summed_mass(derived64):
    """chucm lies in [0, 1); for every pair the summed mass x_i + x_j lies
    between x[ima] and x[ima + 1] (one-based), with the Courant number its
    log-position in that bracket; a pair folded into the next category
    (chucm above 1 - 1e-8, :1669) sits on x[ima] to the tables' five
    digits; a pair above the largest bin keeps ima = 0 and chucm = 0.

    Breakage prevented: a Bott flux aimed at a bin that cannot hold the
    coalesced mass, from a search or fold that disagrees with the grid.
    """
    x = np.asarray(derived64.xl_mg_bott)
    chucm, ima = derived64.chucm, derived64.ima
    assert np.all(chucm >= 0.0) and np.all(chucm < 1.0)
    assert np.array_equal(chucm, chucm.T) and np.array_equal(ima, ima.T)
    folded = clamped = 0
    for i in range(NKR):
        for j in range(NKR):
            x0 = x[i] + x[j]
            k = int(ima[i, j])
            if k == 0:
                assert x0 > x[NKR - 1] and chucm[i, j] == 0.0, (i + 1, j + 1)
                continue
            assert 1 <= k <= NKR - 1
            low, high = x[k - 1], x[k]
            if chucm[i, j] > 0.0:
                assert low < x0 <= high, (i + 1, j + 1)
                expected = math.log(x0 / low) / math.log(2.0)
                assert abs(chucm[i, j] - expected) <= 1e-12, (i + 1, j + 1)
            elif abs(x0 / low - 1.0) < 1e-4:
                folded += 1          # on the lower bracket
            else:
                # min(nkr-1, kk-1) (:1673) keeps the last category when the
                # fold would leave the grid: the sum sits on x[ima + 1].
                assert k == NKR - 1 and abs(x0 / high - 1.0) < 1e-4, (i + 1, j + 1)
                clamped += 1
    assert folded > 0 and clamped > 0


def _diameter_cm(mass_g: float) -> float:
    return (6.0 * mass_g / (math.pi * 1.0)) ** (1.0 / 3.0)


def test_small_drops_coalesce_with_unit_efficiency(derived64, dump64, t64):
    """E = 1 wherever the smaller drop's diameter is below 100 um (:8827),
    in the host and in WRF's dump; everywhere E lies in [1e-30, 1]; the
    correction row (:8770-8772) is 1e-30 at (33, 25..31) and leaves the
    transposed entries alone.

    Breakage prevented: cloud droplets that bounce, or an efficiency above
    one that creates collected mass.
    """
    diam = [_diameter_cm(float(m)) for m in t64.xl]
    for ecoal in (derived64.ecoalmassm, dump64["ecoalmassm"]):
        assert np.all(ecoal >= 1e-30) and np.all(ecoal <= 1.0)
        small = 0
        for i in range(NKR):
            for j in range(NKR):
                if min(diam[i], diam[j]) < 100e-4:
                    assert ecoal[i, j] == 1.0, (i + 1, j + 1)
                    small += 1
        assert small == NKR * NKR - (NKR - 14) ** 2   # bins 1-14 are small
        assert np.all(ecoal[NKR - 1, 24:31] == 1e-30)
    # The transposed entries keep ecoalmass's own values, which for rows
    # 29-31 are well above the floor, so the matrix is not symmetric there.
    for j in range(25, 32):
        own = D.coalescence_efficiency(t64.xl[j - 1], t64.xl[NKR - 1], t64.xl,
                                       derived64.dropradii, t64.vr1,
                                       fall_speed_lookup="wrf_v480")
        assert derived64.ecoalmassm[j - 1, NKR - 1] == own
        assert dump64["ecoalmassm"][j - 1, NKR - 1] == pytest.approx(
            own, rel=1e-12)
    assert np.all(derived64.ecoalmassm[28:31, NKR - 1] > 0.2)


def _low_list_si(bin_l: int, bin_s: int, tables, speed_bin) -> float:
    """Low and List (1982a), from the paper, in SI units.

    E = a (1 + D_S/D_L)**-2 exp(-b sigma E_T**2 / S_C) for E_T < 5 uJ, else
    0, with a = 0.778, b = 2.61e6 m2 J-2, sigma = 7.28e-2 N/m, S_C = pi sigma
    (D_L**3 + D_S**3)**(2/3), S_T = pi sigma (D_L**2 + D_S**2), CKE = pi/12
    rho D_L**3 D_S**3 / (D_L**3 + D_S**3) (V_L - V_S)**2, E_T = CKE + S_T -
    S_C.  Diameters from the bin masses with the true pi and water at 1000
    kg/m3; speeds from the VR1 table (cm/s) of the bin ``speed_bin`` names.
    """
    m_l = float(tables.xl[bin_l - 1]) * 1e-3
    m_s = float(tables.xl[bin_s - 1]) * 1e-3
    d_l = (6.0 * m_l / (math.pi * 1000.0)) ** (1.0 / 3.0)
    d_s = (6.0 * m_s / (math.pi * 1000.0)) ** (1.0 / 3.0)
    v_l = float(tables.vr1[speed_bin(bin_l) - 1]) * 0.01
    v_s = float(tables.vr1[speed_bin(bin_s) - 1]) * 0.01
    sigma = 7.28e-2
    cube = d_l ** 3 + d_s ** 3
    cke = math.pi / 12.0 * 1000.0 * d_l ** 3 * d_s ** 3 / cube * (v_l - v_s) ** 2
    s_t = math.pi * sigma * (d_l ** 2 + d_s ** 2)
    s_c = math.pi * sigma * cube ** (2.0 / 3.0)
    e_t = cke + s_t - s_c
    if e_t >= 5e-6:
        return 0.0
    return 0.778 * (1.0 + d_s / d_l) ** -2 * math.exp(
        -2.61e6 * sigma * e_t ** 2 / s_c)


#: One-based (larger, smaller) bin pairs whose mean radius is at least
#: 500 um, where ecoaldiam takes Low-List alone (:8847-8849): equal drops,
#: neighbours, wide pairs, and (32, 28) whose collision energy is above
#: 5 uJ, so E_LL is 0 and ecoaldiam's floor gives 1e-30.
LOW_LIST_PAIRS = ((25, 25), (26, 24), (28, 20), (29, 26), (31, 22),
                  (30, 29), (33, 30), (32, 28))
#: WRF's pi (3.1415927) differs from the true one by 1.5e-8 relative, which
#: moves the diameters by 5e-9 and, through the exponent, E by up to 5.5e-9
#: (pair 29, 26); every formula slip is orders of magnitude larger.
RTOL_PAPER = 1e-7


def _wrf_speed_bin(k: int) -> int:
    """vTBeard's pick for a bin-centre drop: the next bin up (finding D10)."""
    return min(k + 1, NKR)


def test_low_list_values_recomputed_from_the_paper(t64, derived64, dump64):
    """Low-List efficiencies recomputed from Low and List (1982a), in SI,
    equal ECOALMASSM in WRF's dump and in the host for large-drop pairs.

    Breakage prevented: a coalescence efficiency whose formula or units
    drifted from the published parameterization while still matching some
    other transcription of it.
    """
    for bin_l, bin_s in LOW_LIST_PAIRS:
        d_l = _diameter_cm(float(t64.xl[bin_l - 1]))
        d_s = _diameter_cm(float(t64.xl[bin_s - 1]))
        assert 0.25 * (d_l + d_s) >= 500e-4 and d_s >= 100e-4
        expected = max(_low_list_si(bin_l, bin_s, t64, _wrf_speed_bin), 1e-30)
        for ecoal in (dump64["ecoalmassm"], derived64.ecoalmassm):
            value = float(ecoal[bin_s - 1, bin_l - 1])
            assert abs(value - expected) <= RTOL_PAPER * expected, (
                bin_l, bin_s, value, expected)
    assert float(derived64.ecoalmassm[27, 31]) == 1e-30     # (28, 32): E_T > 5 uJ


def test_finding_d10_wrf_uses_the_next_bins_fall_speed(t64, derived64, dump64):
    """Finding D10: every drop radius ecoalmass forms (pi 3.1415927) lies
    3.18e-8 relative above its DROPRADII (pi 3.141593), so vTBeard returns
    the next bin's speed.  The paper's formula with each bin's own speed
    misses WRF's dump and matches fall_speed_lookup="own_bin".

    Breakage prevented: an unrecorded change of fall speed inside every
    Beard-Ochs and Low-List efficiency, in either direction.
    """
    for k in range(NKR):
        radius = _diameter_cm(float(t64.xl[k])) / 2.0
        wrf_radius = ((6.0 / 3.1415927) * float(t64.xl[k])) ** (1.0 / 3.0) / 2.0
        assert wrf_radius / derived64.dropradii[k] - 1.0 == pytest.approx(
            3.1831e-8, rel=1e-3)
        assert radius > derived64.dropradii[k]
    own = D.build_derived(t64, DT_ORACLE, fall_speed_lookup="own_bin")
    assert own.fall_speed_lookup == "own_bin"
    for bin_l, bin_s in LOW_LIST_PAIRS:
        if bin_l == bin_s:
            continue
        expected = max(_low_list_si(bin_l, bin_s, t64, lambda k: k), 1e-30)
        wrf = float(dump64["ecoalmassm"][bin_s - 1, bin_l - 1])
        value = float(own.ecoalmassm[bin_s - 1, bin_l - 1])
        assert abs(value - expected) <= RTOL_PAPER * expected, (bin_l, bin_s)
        if not (wrf == expected == 1e-30):
            assert abs(wrf - expected) > 1e-3 * expected, (bin_l, bin_s)
    # (33, 30): 0.3228 with WRF's speeds, the floor with the drops' own.
    assert float(dump64["ecoalmassm"][29, 32]) == pytest.approx(0.32276, rel=1e-4)
    assert float(own.ecoalmassm[29, 32]) == 1e-30
    changed = np.count_nonzero(own.ecoalmassm != derived64.ecoalmassm)
    assert changed == 295


def test_finding_d10_the_corrected_default_builds_each_bins_own_fall_speed(
        t64, derived64, dump64):
    """D10 is fixed by default.  The corrected preset (the reference
    default) names the ``own_bin`` lookup, the bare ``build_derived`` call
    takes it from that preset, and ``behavior=`` selects WRF's lookup for
    the oracle.  Inside bins 1-32, the only pairs ``coll_xxx_lwf`` reads
    (both indices capped at nkr - 1, :680-689), the two tables differ in
    272 of 1024 entries; the largest is pair (30, 32), 0.2702 with WRF's
    speeds and the 1e-30 floor with each bin's own.

    Breakage prevented: a corrected run that carries WRF's next-bin fall
    speeds in every coalescence efficiency under a corrected label, which
    the milestone 1 evidence did while the lookup was an argument of
    ``build_derived`` alone, defaulting to WRF's.
    """
    corrected = FsbmBehavior.corrected()
    assert corrected.fall_speed_lookup == "own_bin"
    assert WRF.fall_speed_lookup == "wrf_v480"
    bare = D.build_derived(t64, DT_ORACLE)
    chosen = D.build_derived(t64, DT_ORACLE, behavior=corrected)
    own = D.build_derived(t64, DT_ORACLE, fall_speed_lookup="own_bin")
    assert bare.fall_speed_lookup == chosen.fall_speed_lookup == "own_bin"
    assert np.array_equal(bare.ecoalmassm, own.ecoalmassm)
    assert np.array_equal(chosen.ecoalmassm, own.ecoalmassm)
    assert derived64.fall_speed_lookup == "wrf_v480"
    np.testing.assert_allclose(derived64.ecoalmassm, dump64["ecoalmassm"],
                               rtol=RTOL_O64, atol=0.0)
    inner = (slice(0, NKR - 1), slice(0, NKR - 1))
    assert np.count_nonzero(bare.ecoalmassm[inner]
                            != derived64.ecoalmassm[inner]) == 272
    assert float(derived64.ecoalmassm[29, 31]) == pytest.approx(0.27025,
                                                                rel=1e-4)
    assert float(bare.ecoalmassm[29, 31]) == 1e-30
    # Everything but ECOALMASSM is the same table.
    for field in dataclasses.fields(D.FsbmDerived):
        if field.name in ("ecoalmassm", "fall_speed_lookup"):
            continue
        a, b = getattr(bare, field.name), getattr(derived64, field.name)
        assert np.array_equal(a, b), field.name
    # A lookup that disagrees with the behaviour it is built for is refused.
    with pytest.raises(FsbmRefusal, match="D10"):
        D.build_derived(t64, DT_ORACLE, behavior=WRF,
                        fall_speed_lookup="own_bin")
    with pytest.raises(FsbmRefusal, match="D10"):
        D.build_derived(t64, DT_ORACLE, behavior=corrected,
                        fall_speed_lookup="wrf_v480")
    with pytest.raises(FsbmRefusal, match="FsbmBehavior"):
        D.build_derived(t64, DT_ORACLE, behavior="corrected")
    # The routine-level functions name the lookup; neither picks one.
    with pytest.raises(TypeError, match="fall_speed_lookup"):
        D.coalescence_efficiency_matrix(t64.xl, derived64.dropradii, t64.vr1)
    with pytest.raises(TypeError, match="fall_speed_lookup"):
        D.coalescence_efficiency(t64.xl[0], t64.xl[1], t64.xl,
                                 derived64.dropradii, t64.vr1)


#: The three lognormal modes (N cm-3, mean radius cm, sigma) WRF writes,
#: maritime and continental (:3777-3803), restated for an independent check.
MODES = {
    "fccnr_mar": ((340.0, 0.005e-4, 1.6), (60.0, 0.035e-4, 2.0),
                  (3.1, 0.31e-4, 2.7)),
    "fccnr_con": ((1000.0, 0.008e-4, 1.6), (800.0, 0.034e-4, 2.1),
                  (0.72, 0.46e-4, 2.2)),
}


@pytest.mark.parametrize("field", sorted(MODES))
def test_lognormal_spectra_integrate_to_the_mode_totals(derived64, field):
    """COL times the sum of FCCNR over the bins inside 0.48 nm < r < 0.4 um
    equals the modes' number inside the cells those bins span, computed
    from erf.  The bins are a midpoint rule in ln r with step ln2/3, so the
    comparison adds the midpoint rule's own h**2/24 end correction and
    holds to 1e-6; without it, to 5e-5.  Bins outside the window are zero.

    Breakage prevented: an aerosol spectrum that activates more or fewer
    droplets than the stated modes hold (a lost ln2/3, a mode dropped, a
    radius grid off by a bin).
    """
    spectrum = np.asarray(getattr(derived64, field))
    radii = np.asarray(derived64.rccn)
    inside = (radii > 0.48e-7) & (radii < 0.4e-4)
    assert np.all(spectrum[~inside] == 0.0) and np.all(spectrum[inside] > 0.0)
    first, last = np.nonzero(inside)[0][[0, -1]]
    assert (first + 1, last + 1) == (7, 35)
    h = math.log(2.0) / 3.0
    assert np.allclose(np.diff(np.log(radii)), h, rtol=1e-12, atol=0.0)
    lo = math.log(radii[first]) - h / 2.0
    hi = math.log(radii[last]) + h / 2.0
    exact = slope_lo = slope_hi = 0.0
    for number, mean, sigma in MODES[field]:
        s = math.log(sigma)
        mu = math.log(mean)
        exact += number / 2.0 * (math.erf((hi - mu) / (math.sqrt(2.0) * s))
                                 - math.erf((lo - mu) / (math.sqrt(2.0) * s)))
        for edge, sign in ((lo, -1.0), (hi, 1.0)):
            density = number / (math.sqrt(2.0 * math.pi) * s) * math.exp(
                -(edge - mu) ** 2 / (2.0 * s * s))
            slope = -density * (edge - mu) / (s * s)
            if sign > 0:
                slope_hi += slope
            else:
                slope_lo += slope
    midpoint = math.fsum(COL * float(v) for v in spectrum[inside])
    assert abs(midpoint / exact - 1.0) < 5e-5
    corrected = midpoint + h * h / 24.0 * (slope_hi - slope_lo)
    assert abs(corrected / exact - 1.0) < 1e-6, (midpoint, corrected, exact)


def test_spontaneous_breakup_redistribution_conserves_mass(derived64, dump64,
                                                           t64):
    """The fragments of one breaking drop of bin ikr_spon_break carry its
    mass: sum_i Gain_Var_New(ikr, i) x_i * x_i = x_ikr (Spont_Rain_BreakUp
    adds Gain_Var_New(j, i) x_i fragments of mass x_i per breaking drop,
    :275), in the host and in WRF's dump.  ikr_spon_break is the first bin
    with a radius of at least 0.3 cm.

    Breakage prevented: spontaneous breakup that creates or destroys rain.
    """
    ikr = derived64.ikr_spon_break
    radii = np.asarray(derived64.dropradii)
    assert radii[ikr - 1] >= 0.3 and np.all(radii[:ikr - 1] < 0.3)
    xl = [float(v) for v in t64.xl]
    for weights in (derived64.gain_var_new, dump64["gain_var_new"]):
        fragments = math.fsum(float(weights[ikr - 1, i]) * xl[i] * xl[i]
                              for i in range(ikr - 1))
        assert abs(fragments / xl[ikr - 1] - 1.0) < 1e-13
        assert np.all(weights[ikr - 1, :ikr - 1] > 0.0)
        assert not np.any(weights[:ikr - 1, :]) and weights[ikr - 1, ikr - 1] == 0
    assert derived64.prob[ikr - 1] == pytest.approx(
        2.94e-7 * math.exp(34.0 * radii[ikr - 1]), rel=1e-14)


# ---------------------------------------------------------------------------
# Contract details
# ---------------------------------------------------------------------------

def test_results_are_read_only_and_float64(t32, derived64):
    """Breakage prevented: one caller's edit of a shared derived table or
    kernel changing what every other caller computes with."""
    for field in dataclasses.fields(D.FsbmDerived):
        value = getattr(derived64, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, field.name
            expected = np.int32 if field.name == "ima" else np.float64
            assert value.dtype == expected, field.name
    kernels = D.kernels_at_pressure(T.load_tables("float64"), derived64,
                                    8.0e5, DT_ORACLE)
    for family in D.KERNEL_FAMILIES:
        array = getattr(kernels, family)
        assert not array.flags.writeable and array.dtype == np.float64
        assert array.shape == (NKR, NKR)
    from_f32 = D.build_derived(t32, DT_ORACLE)
    assert from_f32.table_precision == "float32"
    assert from_f32.chucm.dtype == np.float64


def test_the_ice_mass_grids_are_the_tables_in_mg(t64, derived64):
    """xs_mg, xg_mg, xh_mg and xi_mg are x*1.e3 of the same loop (:6077-6085).

    Breakage prevented: an ice collision in the ice milestone reading a mass
    grid in g where WRF hands it mg."""
    for name in ("xs", "xg", "xh", "xi", "xl"):
        assert np.array_equal(getattr(derived64, f"{name}_mg"),
                              np.asarray(getattr(t64, name)) * 1.0e3)


@pytest.mark.parametrize("bad", [0.0, -12.0, float("nan"), float("inf")])
def test_refusals_name_the_bad_input(t64, derived64, bad):
    """A step or pressure that would make every kernel zero, negative or NaN
    is refused, and so is an aerosol regime or fall-speed lookup WRF does
    not define."""
    with pytest.raises(FsbmRefusal, match="dt"):
        D.build_derived(t64, bad)
    with pytest.raises(FsbmRefusal, match="p_cgs"):
        D.kernels_at_pressure(t64, derived64, bad, DT_ORACLE)
    with pytest.raises(FsbmRefusal, match="dt_coll"):
        D.kernels_at_pressure(t64, derived64, 8.0e5, bad)
    with pytest.raises(FsbmRefusal, match="IType"):
        D.lognormal_modes_aerosol(t64.xl, 3)
    with pytest.raises(FsbmRefusal, match="fall_speed_lookup"):
        D.build_derived(t64, DT_ORACLE, fall_speed_lookup="beard")
