"""Fast SBM sedimentation and surface precipitation against WRF's own Fortran.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A fall scheme that is right on paper and wrong in WRF's terms: a substep
  count off by one (``NSUB = INT(2 DT/TFALL)+1`` with ``TFALL`` taken from the
  level heights, not the layer depths), a divisor that is the layer depth
  instead of the distance between level centres, a top level treated like
  the others, or a bin falling that WRF leaves still (the 1e-20 activity
  test).  Each moves rain to the ground at the wrong time or in the wrong
  amount, and the warm column cannot be gated against WRF afterwards.
* Fall speeds corrected with the wrong pressure: WRF multiplies the table by
  ``sqrt(1e6/p)`` in CGS.  The negative control scales the table by 1.001
  and the probe gate must fail.
* Surface precipitation that is not WRF's: WRF's RAINNC is the level-1
  spectrum after the fall times the fall speed times dt, with the fall speed
  of the tile's LAST column (D1), not what left the column (D2).  The
  reference must reproduce that byte for byte, and its corrected forms must
  be measurably different where WRF's is wrong.

What the oracle is.  ``tests/data/fast_sbm_oracle`` holds WRF v4.8.0's
``module_mp_fast_sbm.F`` run twice: O64 (every REAL promoted to double, the
tight gate) and O32 (WRF as it runs, the envelope).  ``sweep_falflux_*`` is
``FALFLUXHUCM_Z`` on a uniform 100 m grid and the KC-1 stretched grid, two
spectra and two steps, up to 44 substeps.  ``kc1_probe_*`` records the drop
fall inside ``FAST_SBM`` at 6 steps of the warm column KC-1 for all four
columns, and the diagnostics after the surface loop.  ``kc1_replay_*`` holds
the full state before and after 16 calls.

Tolerances (design 5.5): process records relative 1e-10 with a floor of
1e-12 of the record's scale (its largest magnitude), NSUB exact.  The O32
envelope is ``abs(host - O32) <= abs(O64 - O32) + 1e-9 scale``.

Measured on these fixtures (2026-09-23), reported, and held by the tests
below:

* The host reproduces every O64 falflux sweep and probe record, every probe
  fall speed, and the probe RAINNC and RAINNCV bit for bit.
* D1: every column's RAINNC uses the fall-speed factor of column c4
  (850 hPa, last in the tile); for c1 to c3 that inflates RAINNC by 8.2 to
  8.7 percent over the column's own factor.
* D2: at the probe steps WRF's RAINNCV exceeds the bottom-face outflow by
  3.7 to 72 percent where more than 1e-6 mm fell (and 43 times at the leading
  edge of the rain, 2e-18 mm); with each column's own speed the level-1 form
  still misses the outflow by -4 to +59 percent.
* D8: weighted by the scheme's own divisors, column plus outflow closes to
  1e-12 on every record.  With layer depths (the physical column) it does
  not: on the KC-1 grid (layers 2.5 percent deeper per level) 1/81 of the
  outflow, 1.23 percent, stays behind as new water at every call where rain
  reaches the ground, and water falling out of the top level shrinks the
  column (0.07 and 0.14 percent of it in one call on the oracle's stretched
  sweep).  Moves between the levels in between keep the physical column,
  because the layer-to-divisor ratio is the same at each of them.
* D4 meets D2: at least one replay call counts level-1 drops as rain that
  the exit clip then deletes from the column (qc + qr < 1e-13 at exit).
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.fast_sbm import sedimentation as S
from gpuwm.core.fast_sbm import state as ST
from gpuwm.core.fast_sbm.constants import COL, FsbmBehavior, FsbmRefusal, NKR
from gpuwm.core.fast_sbm.tables import load_tables
from tools.fast_sbm_wrf480_oracle import fixtures as F

REL = 1e-10
FLOOR = 1e-12
ENVELOPE_FLOOR = 1e-9
EPS32 = 2.0 ** -23


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables():
    return load_tables("float64")


@pytest.fixture(scope="module")
def sweep64():
    return F.load_fixture("sweep_falflux_o64")


@pytest.fixture(scope="module")
def sweep32():
    return F.load_fixture("sweep_falflux_o32")


@pytest.fixture(scope="module")
def kc1_dt():
    return float(F.load_fixture("kc1_case")["case.dt"][0])


@pytest.fixture(scope="module")
def kc1_initial():
    summary = F.load_fixture("kc1_summary_o64")
    return {key.split(".", 1)[1]: value[0] for key, value in summary.items()
            if key.startswith("initial.")}


def _probe(build: str) -> dict[str, dict[tuple[int, int], F.Record]]:
    """Probe records by type, keyed ``(step, column)``, column one-based."""
    out: dict[str, dict[tuple[int, int], F.Record]] = {}
    types = ("fall_before", "fall_after", "fall_ice_before", "fall_ice_after",
             "diag")
    for rec in F.iter_probe(f"kc1_probe_{build}", types=types):
        key = (int(rec["step"]), int(rec["i"]))
        if rec.type.startswith("fall_ice"):
            out.setdefault(rec.type, {})[key + (int(rec["species"]),)] = rec
        else:
            out.setdefault(rec.type, {})[key] = rec
    return out


@pytest.fixture(scope="module")
def probe64():
    return _probe("o64")


@pytest.fixture(scope="module")
def probe32():
    return _probe("o32")


@pytest.fixture(scope="module")
def replay64():
    return {step: (pre, post) for step, pre, post in F.iter_replay("kc1_replay_o64")}


def _sweep_records(dump) -> list[dict]:
    n = dump["case.dt"].shape[0]
    return [{name: dump[f"case.{name}"][c] for name in
             ("grid", "profile", "dt", "zcgs", "rhocgs", "pcgs", "vrx",
              "ffx_in", "ffx_out")} for c in range(n)]


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def _scale(ref) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(ref))) if ref.size else 0.0


def _gate_ratio(host, ref, scale: float | None = None) -> np.ndarray:
    """``|host - ref| / (1e-10 |ref| + 1e-12 scale)``; the gate is ``<= 1``."""
    host = np.asarray(host, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scale = _scale(ref) if scale is None else scale
    bound = REL * np.abs(ref) + FLOOR * scale
    diff = np.abs(host - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bound > 0.0, diff / bound, np.where(diff > 0.0, np.inf, 0.0))
    return ratio


def _gate_failures(host, ref, scale: float | None = None) -> int:
    return int(np.count_nonzero(_gate_ratio(host, ref, scale) > 1.0))


def _probe_fall_gate(vr1, probe, dt) -> list[str]:
    """The sedimentation probe gate: host speeds from ``vr1`` and host fall.

    Speeds come from the table the caller passes, not from the record, so an
    error in the speed table reaches both the ``vrx`` check and the fall.
    """
    failures = []
    for key, before in sorted(probe["fall_before"].items()):
        after = probe["fall_after"][key]
        v = S.corrected_velocities(vr1, before["pcgs"])
        n_v = _gate_failures(v, before["vrx"])
        result = S.falflux(before["ffx"], v, before["rhocgs"], before["zcgs"], dt)
        n_f = _gate_failures(result.ffx, after["ffx"])
        if n_v or n_f:
            failures.append(f"step {key[0]} column {key[1]}: {n_v} speeds, "
                            f"{n_f} fall values outside the gate")
    return failures


def _nsub_float32(v, z, dt) -> int:
    """``TFALL`` and ``NSUB`` evaluated in float32, as the O32 build does."""
    tfall = np.float32(1.0e10)
    eps = np.float32(1.0e-20)
    for k in range(len(z)):
        tfall = min(tfall, np.float32(z[k]) / (np.float32(v[k]) + eps))
    return int((np.float32(2.0) * np.float32(dt)) / tfall) + 1


# ---------------------------------------------------------------------------
# Oracle gates: geometry and fall speeds
# ---------------------------------------------------------------------------

def test_level_heights_are_the_zcgs_fast_sbm_builds(kc1_initial, probe64):
    zcgs = S.level_heights_cgs(kc1_initial["dz8w"][0])
    for key, before in probe64["fall_before"].items():
        assert _gate_failures(zcgs, before["zcgs"]) == 0, key
    assert np.array_equal(S.p_cgs_from_si(kc1_initial["p"][3]),
                          probe64["fall_before"][(10, 4)]["pcgs"])


def test_corrected_velocities_are_the_speeds_wrf_falls_with(tables, sweep64, probe64):
    for rec in _sweep_records(sweep64):
        v = S.corrected_velocities(tables.vr1, rec["pcgs"])
        assert _gate_failures(v, rec["vrx"]) == 0
    for key, before in probe64["fall_before"].items():
        v = S.corrected_velocities(tables.vr1, before["pcgs"])
        assert _gate_failures(v, before["vrx"]) == 0, key


# ---------------------------------------------------------------------------
# Oracle gates: FALFLUXHUCM_Z
# ---------------------------------------------------------------------------

def test_falflux_reproduces_every_o64_sweep_record(sweep64):
    records = _sweep_records(sweep64)
    assert len(records) == 8
    grids = {(int(r["grid"]), int(r["profile"]), float(r["dt"])) for r in records}
    assert len(grids) == 8
    max_nsub = 0
    for rec in records:
        result = S.falflux(rec["ffx_in"], rec["vrx"], rec["rhocgs"], rec["zcgs"],
                           rec["dt"])
        assert _gate_failures(result.ffx, rec["ffx_out"]) == 0, (
            rec["grid"], rec["profile"], rec["dt"])
        # Bins WRF leaves still are returned untouched.
        still = ~result.active
        assert np.array_equal(result.ffx[:, still], rec["ffx_in"][:, still])
        assert np.all(result.nsub[still] == 0)
        max_nsub = max(max_nsub, int(result.nsub.max()))
    # The family reaches the multi-substep path well beyond one substep.
    assert max_nsub >= 40


def test_falflux_reproduces_every_o64_sedimentation_probe_record(probe64, tables, kc1_dt):
    assert len(probe64["fall_before"]) == 24
    assert _probe_fall_gate(tables.vr1, probe64, kc1_dt) == []
    # With the record's own speeds too, so the fall is gated on its own.
    for key, before in probe64["fall_before"].items():
        result = S.falflux(before["ffx"], before["vrx"], before["rhocgs"],
                           before["zcgs"], kc1_dt)
        assert _gate_failures(result.ffx, probe64["fall_after"][key]["ffx"]) == 0


def test_the_velocity_table_scaled_by_1_001_fails_the_probe_gate(probe64, tables, kc1_dt):
    failures = _probe_fall_gate(np.asarray(tables.vr1) * 1.001, probe64, kc1_dt)
    # Every record fails on its speeds; the records with drops fail on the fall.
    assert len(failures) == len(probe64["fall_before"])
    moved = [f for f in failures if not f.endswith(" 0 fall values outside the gate")]
    assert len(moved) >= 20


def _nsub_cases(sweep64, probe64, dt_kc1):
    for rec in _sweep_records(sweep64):
        yield (f"sweep grid {rec['grid']} profile {rec['profile']} dt {rec['dt']}",
               rec["ffx_in"], rec["vrx"], rec["rhocgs"], rec["zcgs"],
               float(rec["dt"]), rec["ffx_out"])
    for key, before in sorted(probe64["fall_before"].items()):
        yield (f"probe step {key[0]} column {key[1]}", before["ffx"], before["vrx"],
               before["rhocgs"], before["zcgs"], dt_kc1,
               probe64["fall_after"][key]["ffx"])


def test_nsub_is_exact(sweep64, probe64, kc1_dt):
    """WRF's NSUB, and no neighbour of it, reproduces every falling bin.

    The records do not carry NSUB, so it is identified from what WRF
    returned: for every bin that fell, the host's NSUB passes the gate and
    NSUB - 1 and NSUB + 1 are strictly further from the record.
    """
    checked = 0
    for label, ffx, v, rho, z, dt, ref in _nsub_cases(sweep64, probe64, kc1_dt):
        result = S.falflux(ffx, v, rho, z, dt)
        scale = _scale(ref)
        for kr in np.nonzero(result.active)[0]:
            n0 = int(result.nsub[kr])
            _, n_rule = S.falflux_substep_count(v[:, kr], z, dt)
            assert n_rule == n0

            def worst(n):
                out, _ = S.falflux_bin(ffx[:, kr], v[:, kr], rho, z, dt, n)
                return float(np.max(_gate_ratio(out, ref[:, kr], scale)))

            here = worst(n0)
            assert here <= 1.0, (label, kr + 1, n0)
            neighbours = [worst(n) for n in (n0 - 1, n0 + 1) if n >= 1]
            assert all(here < other for other in neighbours), (label, kr + 1, n0)
            checked += 1
    assert checked > 700


def test_nsub_on_native_inputs_is_the_float32_count(sweep32, probe32, kc1_dt):
    """On O32's own inputs the host picks the bins and NSUB O32 picks."""
    cases = [(r["ffx_in"], r["vrx"], r["zcgs"], float(r["dt"]))
             for r in _sweep_records(sweep32)]
    cases += [(b["ffx"], b["vrx"], b["zcgs"], kc1_dt)
              for b in probe32["fall_before"].values()]
    for ffx, v, z, dt in cases:
        result = S.falflux(ffx.astype(np.float64), v.astype(np.float64),
                           np.ones(z.shape[0]), z.astype(np.float64), dt)
        active32 = np.any(ffx >= np.float32(1.0e-20), axis=0)
        assert np.array_equal(result.active, active32)
        for kr in np.nonzero(active32)[0]:
            assert result.nsub[kr] == _nsub_float32(v[:, kr], z, dt)


def test_o32_sweep_envelope(sweep64, sweep32):
    """``abs(host - O32) <= abs(O64 - O32) + 1e-9 scale``, elementwise.

    The host runs the case as the O64 build received it (the case defined in
    float64), so ``O64`` is the promoted build on the same case and the
    envelope is the one the design states.  The host run on O32's own
    float32-rounded inputs is gated separately below against float32
    rounding: elementwise, about 2 percent of its values carry O32 rounding
    that O64, on inputs that differ in the last float32 place, happens not
    to show, so the envelope is not the right instrument there.
    """
    for rec64, rec32 in zip(_sweep_records(sweep64), _sweep_records(sweep32)):
        host = S.falflux(rec64["ffx_in"], rec64["vrx"], rec64["rhocgs"],
                         rec64["zcgs"], rec64["dt"]).ffx
        o32 = rec32["ffx_out"].astype(np.float64)
        scale = _scale(o32)
        bound = np.abs(rec64["ffx_out"] - o32) + ENVELOPE_FLOOR * scale
        assert np.all(np.abs(host - o32) <= bound), (rec64["grid"], rec64["profile"])


def test_host_on_native_inputs_is_within_float32_rounding(sweep32, probe64, probe32, kc1_dt):
    """Host on O32's own inputs against O32: one float32 epsilon of scale.

    Measured: at most 0.46 float32 epsilons of the record scale over the
    sweep and the probe.  For the probe the envelope is also held in the
    record's max norm against O64, whose KC-1 trajectory is its own.
    """
    worst = 0.0
    for rec in _sweep_records(sweep32):
        host = S.falflux(rec["ffx_in"].astype(np.float64), rec["vrx"].astype(np.float64),
                         rec["rhocgs"].astype(np.float64), rec["zcgs"].astype(np.float64),
                         float(rec["dt"])).ffx
        o32 = rec["ffx_out"].astype(np.float64)
        worst = max(worst, float(np.max(np.abs(host - o32))) / _scale(o32))
    for key, before in probe32["fall_before"].items():
        host = S.falflux(before["ffx"].astype(np.float64), before["vrx"].astype(np.float64),
                         before["rhocgs"].astype(np.float64),
                         before["zcgs"].astype(np.float64), kc1_dt).ffx
        o32 = probe32["fall_after"][key]["ffx"].astype(np.float64)
        o64 = probe64["fall_after"][key]["ffx"]
        scale = _scale(o32)
        if scale == 0.0:
            assert not np.any(host)
            continue
        diff = float(np.max(np.abs(host - o32)))
        worst = max(worst, diff / scale)
        assert diff <= float(np.max(np.abs(o64 - o32))) + ENVELOPE_FLOOR * scale, key
    assert worst <= EPS32


# ---------------------------------------------------------------------------
# Oracle gates: surface precipitation
# ---------------------------------------------------------------------------

def _level1_after_fall(probe, key, species=None):
    if species is None:
        after = probe["fall_after"][key]
    else:
        after = probe["fall_ice_after"][key + (species,)]
    rho = probe["fall_after"][key]["rhocgs"]
    return after["ffx"][0] * rho[0]


def _tile_level1_speeds(tables, probe, step, vr):
    rows = [S.corrected_velocities(vr, probe["fall_before"][(step, i)]["pcgs"])[0]
            for i in range(1, 5)]
    return np.array(rows)


def test_surface_precip_level1_is_wrfs_rainncv_exactly(tables, probe64, replay64, kc1_dt):
    """RAINNCV and RAINNC reproduced exactly as WRF computes them.

    For each column at each probe step: the level-1 drops after the fall,
    WRF's level-1 speed ``VR1_Z(KRR,1)`` as the diag probe recorded it after
    the loop, and RAINNC before the call from the replay record.  The whole
    surface block also runs the snow and hail accumulations on their
    (zero) level-1 bins, which must change nothing.
    """
    wrf = FsbmBehavior.wrf_v480()
    rainy = 0
    for key, diag in sorted(probe64["diag"].items()):
        step, i = key
        pre, post = replay64[step]
        drops = _level1_after_fall(probe64, key)
        v_rec = diag["vr1_z_level1_after_loop"]
        rainncv = S.surface_precip_level1(drops, v_rec, tables.xl, tables.ro1bl,
                                          COL, kc1_dt)
        assert rainncv == float(diag["rainncv"]), key

        v_drops = S.surface_fall_speeds(
            _tile_level1_speeds(tables, probe64, step, tables.vr1), wrf)[i - 1]
        v_snow = S.surface_fall_speeds(
            _tile_level1_speeds(tables, probe64, step, tables.vr3), wrf)[i - 1]
        v_hail = S.surface_fall_speeds(
            _tile_level1_speeds(tables, probe64, step, tables.vr5), wrf)[i - 1]
        assert np.array_equal(v_drops, v_rec)
        block = S.surface_precip_wrf_block(
            drops, v_drops, tables, kc1_dt,
            snow_level1=_level1_after_fall(probe64, key, 2), v_snow=v_snow,
            dense_level1=_level1_after_fall(probe64, key, 3), v_dense=v_hail,
            hail_opt=1, rainnc=float(pre["rainnc"][i - 1]),
            snownc=float(pre["snownc"][i - 1]),
            graupelnc=float(pre["graupelnc"][i - 1]))
        assert block.rainncv == float(diag["rainncv"]), key
        assert block.rainnc == float(diag["rainnc"]), key
        for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                     "graupelncv"):
            assert getattr(block, name) == float(post[name][i - 1]), (key, name)
        rainy += rainncv > 0.0
    assert rainy >= 15


def test_d1_every_column_rains_with_the_last_columns_speed(tables, probe64, kc1_dt):
    """D1: WRF's level-1 speed is the tile's last column's, for every column.

    The RAINNC ratio of the WRF form to the own-column form equals the
    pressure-factor ratio ``sqrt(p_own / p_last)`` to 1e-12.
    """
    wrf = FsbmBehavior.wrf_v480()
    corrected = FsbmBehavior.corrected()
    ratios = []
    for key, diag in sorted(probe64["diag"].items()):
        step, i = key
        own_rows = _tile_level1_speeds(tables, probe64, step, tables.vr1)
        last = own_rows[3]
        assert np.array_equal(diag["vr1_z_level1_after_loop"], last)
        assert np.array_equal(S.surface_fall_speeds(own_rows, wrf)[i - 1], last)
        assert np.array_equal(S.surface_fall_speeds(own_rows, corrected)[i - 1],
                              own_rows[i - 1])
        drops = _level1_after_fall(probe64, key)
        wrf_form = S.surface_precip_level1(drops, last, tables.xl, tables.ro1bl,
                                           COL, kc1_dt)
        own_form = S.surface_precip_level1(drops, own_rows[i - 1], tables.xl,
                                           tables.ro1bl, COL, kc1_dt)
        if i == 4 or wrf_form == 0.0:
            assert wrf_form == own_form
            continue
        pcgs = probe64["fall_before"]
        factor_ratio = (S.pressure_factor(pcgs[(step, 4)]["pcgs"][:1])[0]
                        / S.pressure_factor(pcgs[(step, i)]["pcgs"][:1])[0])
        assert abs(wrf_form / own_form / factor_ratio - 1.0) <= 1e-12, key
        ratios.append(factor_ratio)
    # c1-c3 sit near 1000 hPa and c4 at 850 hPa: an 8 to 9 percent excess.
    assert ratios and 1.08 < min(ratios) and max(ratios) < 1.09


def test_replay_surface_increments(tables, replay64, kc1_initial, kc1_dt):
    """RAINNCV and RAINNC of all 16 replay calls from the post-call state.

    The level-1 drops are recovered from the post-call bins with the entry
    conversion, so agreement is to the gate, not bit for bit.  Where the
    exit clip (D4: drop bins zeroed where qc + qr < 1e-13 at exit) removed
    the level-1 drops after WRF counted them as rain, the recorded RAINNCV
    must be no more than the clipped water could deliver in one step,
    ``10 dt v_max rho (qc + qr)``.
    """
    wrf = FsbmBehavior.wrf_v480()
    p_cgs = S.p_cgs_from_si(kc1_initial["p"])
    own = np.array([S.corrected_velocities(tables.vr1, p_cgs[i])[0] for i in range(4)])
    v_surface = S.surface_fall_speeds(own, wrf)
    rho = ST.rho_cgs_from_si(kc1_initial["rho"])
    clipped_points = 0
    clipped_rain = 0
    compared = 0
    for step, (pre, post) in sorted(replay64.items()):
        scale = _scale(post["rainncv"])
        for i in range(4):
            q_level1 = float(post["qc"][i, 0] + post["qr"][i, 0])
            ref_v = float(post["rainncv"][i])
            if q_level1 < 1e-13:
                clipped_points += 1
                assert not np.any(post["chem"][i, 0, 0:NKR])
                bound = 10.0 * kc1_dt * float(np.max(v_surface[i])) * rho[i, 0] * q_level1
                assert 0.0 <= ref_v <= bound * (1.0 + 1e-12), (step, i + 1)
                clipped_rain += ref_v > 0.0
                continue
            bins = ST.from_chem_new(post["chem"][i])
            dist = ST.bins_to_distributions(bins, rho[i], tables)
            host_v = S.surface_precip_level1(dist.ff1r[0], v_surface[i], tables.xl,
                                             tables.ro1bl, COL, kc1_dt)
            host = S.surface_precip_level1(dist.ff1r[0], v_surface[i], tables.xl,
                                           tables.ro1bl, COL, kc1_dt,
                                           start=float(pre["rainnc"][i]))
            assert _gate_failures(host_v, ref_v, scale) == 0, (step, i + 1)
            assert _gate_failures(host, float(post["rainnc"][i])) == 0, (step, i + 1)
            compared += 1
    assert compared + clipped_points == 64 and compared >= 20
    # The clip deletes water WRF has just counted as rain at least once.
    assert clipped_rain >= 1


def test_rainncv_on_native_inputs_is_within_float32_rounding(tables, probe32, kc1_dt):
    """Host on O32's level-1 drops and speeds against O32's RAINNCV.

    O32 forms each term as an 8-factor float32 product and sums 33 positive
    terms in float32, so its relative error is at most (8 + 33) float32
    half-epsilons.  Measured: 2.1e-7.
    """
    bound = 41 * 2.0 ** -24
    for key, diag in probe32["diag"].items():
        drops = (probe32["fall_after"][key]["ffx"][0]
                 * probe32["fall_after"][key]["rhocgs"][0]).astype(np.float64)
        host = S.surface_precip_level1(drops, diag["vr1_z_level1_after_loop"].astype(np.float64),
                                       tables.xl, tables.ro1bl, COL, kc1_dt)
        ref = float(diag["rainncv"])
        assert abs(host - ref) <= bound * abs(ref), key


# ---------------------------------------------------------------------------
# The column stage and the behaviour switch
# ---------------------------------------------------------------------------

def _probe_distributions(before) -> ST.FsbmDistributions:
    nz = before["ffx"].shape[0]
    zero = np.zeros((nz, NKR))
    return ST.FsbmDistributions(ff1r=before["ffx"] * before["rhocgs"][:, None],
                                ff3r=zero.copy(), ff4r=zero.copy(),
                                ff5r=zero.copy(), fccn=np.zeros((nz, 43)))


def test_sediment_column_is_the_fast_sbm_stage(tables, probe64, kc1_dt):
    wrf = FsbmBehavior.wrf_v480()
    corrected = FsbmBehavior.corrected()
    for key, before in sorted(probe64["fall_before"].items()):
        step, i = key
        fall = S.sediment_column(_probe_distributions(before), tables, before["pcgs"],
                                 before["rhocgs"], before["zcgs"], kc1_dt,
                                 hail_opt=1, iceprocs=1)
        after = probe64["fall_after"][key]
        assert _gate_failures(fall.dist.ff1r, after["ffx"] * after["rhocgs"][:, None]) == 0
        for species, result in ((2, fall.snow), (3, fall.dense)):
            assert not np.any(result.active)
            assert np.array_equal(result.ffx, probe64["fall_ice_after"][key + (species,)]["ffx"])
        v_last = _tile_level1_speeds(tables, probe64, step, tables.vr1)
        v_wrf = S.surface_fall_speeds(v_last, wrf)[i - 1]
        sp = S.surface_precip_column(wrf, fall, v_wrf, tables, kc1_dt)
        ref = float(probe64["diag"][key]["rainncv"])
        assert _gate_failures(sp.rainncv, ref, max(ref, 1e-300)) == 0, key
        sp_c = S.surface_precip_column(corrected, fall, v_last[i - 1], tables, kc1_dt)
        assert sp_c.rainncv == S.surface_precip_outflow(fall.drops.outflow, tables.xl,
                                                        tables.ro1bl, COL)
        # iceprocs = 0 skips the ice calls and gives the same drops.
        warm = S.sediment_column(_probe_distributions(before), tables, before["pcgs"],
                                 before["rhocgs"], before["zcgs"], kc1_dt, iceprocs=0)
        assert warm.snow is None and warm.dense is None
        assert np.array_equal(warm.dist.ff1r, fall.dist.ff1r)


# ---------------------------------------------------------------------------
# D2: WRF's RAINNC against the outflow, measured on KC-1
# ---------------------------------------------------------------------------

def test_d2_wrf_rainnc_is_not_the_outflow(tables, probe64, kc1_initial, kc1_dt):
    """D2 on the 24 KC-1 probe records, and the outflow form's closure.

    The outflow form equals what the scheme's own budget lost (column
    content with the scheme's divisors) to 1e-12 of the column's content;
    WRF's level-1 form does not.  Measured: WRF/outflow 1.037 to 1.72 where
    more than 1e-6 mm fell; own-column level-1/outflow 0.958 to 1.59.
    """
    wrf_ratios, own_ratios = [], []
    for key, before in sorted(probe64["fall_before"].items()):
        step, i = key
        v = S.corrected_velocities(tables.vr1, before["pcgs"])
        result = S.falflux(before["ffx"], v, before["rhocgs"], before["zcgs"], kc1_dt)
        depths = S.scheme_depths(before["zcgs"])
        rho = before["rhocgs"][:, None]
        lost = (S.column_content(before["ffx"] * rho, depths)
                - S.column_content(result.ffx * rho, depths))
        whole = S.surface_precip_outflow(S.column_content(before["ffx"] * rho, depths),
                                         tables.xl, tables.ro1bl, COL)
        outflow_mm = S.surface_precip_outflow(result.outflow, tables.xl, tables.ro1bl, COL)
        lost_mm = S.surface_precip_outflow(lost, tables.xl, tables.ro1bl, COL)
        assert abs(outflow_mm - lost_mm) <= 1e-12 * whole, key

        wrf_mm = float(probe64["diag"][key]["rainncv"])
        own_mm = S.surface_precip_level1(result.ffx[0] * before["rhocgs"][0], v[0],
                                         tables.xl, tables.ro1bl, COL, kc1_dt)
        if outflow_mm > 1e-6:
            wrf_ratios.append(wrf_mm / outflow_mm)
            own_ratios.append(own_mm / outflow_mm)
    assert len(wrf_ratios) >= 12
    assert min(wrf_ratios) > 1.03 and max(wrf_ratios) < 1.8
    assert min(own_ratios) > 0.955 and max(own_ratios) < 1.6
    # WRF's form is never within 3 percent of what left the column.
    assert min(abs(r - 1.0) for r in wrf_ratios) > 0.03


# ---------------------------------------------------------------------------
# Analytic checks
# ---------------------------------------------------------------------------

def _single_bin_speeds(nz: int, bin_index: int, speed: float) -> np.ndarray:
    v = np.full((nz, NKR), 1.0)
    v[:, bin_index] = speed
    return v


def test_a_single_bin_pulse_falls_at_its_fall_speed_and_conserves():
    """Uniform grid: the centroid moves at v exactly and nothing is lost.

    100 m levels, density falling with height, one bin at 7 m/s.  While the
    pulse is aloft its number-weighted centroid moves down ``v dt`` per call
    to 1e-12; as it rains out, column content plus accumulated outflow stays
    the initial content to 1e-12, including a load on the top level.
    """
    nz, kr = 60, 19
    zcgs = S.level_heights_cgs(np.full(nz, 100.0))
    rho = 1.2e-3 * np.exp(-zcgs / 8.0e5)
    v = _single_bin_speeds(nz, kr, 700.0)
    dt = 12.0
    depths = S.scheme_depths(zcgs)
    assert np.array_equal(depths, np.full(nz, 1.0e4))

    ffx = np.zeros((nz, NKR))
    ffx[38:43, kr] = np.array([1.0, 3.0, 5.0, 3.0, 1.0]) / rho[38:43]

    def centroid(values):
        f = values[:, kr] * rho
        return float(np.dot(zcgs, f) / f.sum())

    z0 = centroid(ffx)
    state = ffx
    for call in range(1, 9):
        result = S.falflux(state, v, rho, zcgs, dt)
        assert result.nsub[kr] == int(2.0 * dt / (zcgs[0] / 700.0)) + 1
        assert result.outflow[kr] == 0.0
        state = result.ffx
        expected = z0 - 700.0 * dt * call
        assert abs(centroid(state) - expected) <= 1e-12 * 700.0 * dt * call

    ffx2 = ffx.copy()
    ffx2[nz - 1, kr] = 2.0 / rho[nz - 1]
    initial = S.column_content(ffx2 * rho[:, None], depths)[kr]
    state, rained = ffx2, 0.0
    for _ in range(90):
        result = S.falflux(state, v, rho, zcgs, 60.0)
        rained += result.outflow[kr]
        state = result.ffx
        content = S.column_content(state * rho[:, None], depths)[kr]
        assert abs(content + rained - initial) <= 1e-12 * initial
    assert rained > 0.99 * initial


def _kc1_stretched(nz: int = 40):
    dz = 50.0 * 1.025 ** np.arange(nz)
    return dz, S.level_heights_cgs(dz)


def test_d8_the_stretched_grid_column_does_not_close(tables, sweep64, probe64,
                                                      kc1_initial, kc1_dt):
    """D8: the divisor is the distance between level centres, not the layer.

    With the scheme's own divisors (``scheme_depths``) content plus outflow
    closes to 1e-12, so the outflow-based surface precipitation is exactly
    what the scheme removed.  The physical column (layer depths) does not
    close.  On a grid whose layers deepen by a constant ratio r (1.025 in
    KC-1) every level below the top has layer depth over divisor
    ``2/(1+r)``, so moves between those levels keep the physical column too;
    the non-closure sits at the two ends:

    * at the ground the column loses only ``2/(1+r)`` of what leaves it, so
      ``1 - 2/(1+r)`` of the outflow (1/81, 1.23 percent, for r = 1.025)
      stays behind as new water;
    * the top level's divisor is the centre distance below it, so water
      falling out of the top level shrinks the physical column (the oracle's
      stretched sweep with the top level loaded loses 0.07 and 0.14 percent
      of the column in one call).
    """
    ratio = 1.025
    ground = 1.0 - 2.0 / (1.0 + ratio)

    # Analytic: a pulse below the top level rains out of the KC-1 grid.
    nz, kr = 40, 25
    dz, zcgs = _kc1_stretched(nz)
    rho = 1.1e-3 * np.exp(-zcgs / 8.0e5)
    v = _single_bin_speeds(nz, kr, 600.0)
    ffx = np.zeros((nz, NKR))
    ffx[20:24, kr] = np.array([1.0, 2.0, 2.0, 1.0]) / rho[20:24]
    layer = dz * 100.0
    own = S.scheme_depths(zcgs)
    start_layer = S.column_content(ffx * rho[:, None], layer)[kr]
    start_own = S.column_content(ffx * rho[:, None], own)[kr]
    state, rained = ffx, 0.0
    for _ in range(40):
        result = S.falflux(state, v, rho, zcgs, 60.0)
        rained += result.outflow[kr]
        state = result.ffx
        end_own = S.column_content(state * rho[:, None], own)[kr]
        assert abs(end_own + rained - start_own) <= 1e-12 * start_own
    end_layer = S.column_content(state * rho[:, None], layer)[kr]
    assert rained > 0.9 * start_own
    assert abs((end_layer + rained - start_layer) / rained - ground) <= 1e-9

    # The oracle's stretched sweep.
    dz_sweep = 50.0 * ratio ** np.arange(40)
    top_losses = []
    for rec in _sweep_records(sweep64):
        if int(rec["grid"]) != 2:
            continue
        assert _gate_failures(S.level_heights_cgs(dz_sweep), rec["zcgs"]) == 0
        result = S.falflux(rec["ffx_in"], rec["vrx"], rec["rhocgs"], rec["zcgs"], rec["dt"])
        r = rec["rhocgs"][:, None]
        own_before = S.column_content(rec["ffx_in"] * r, S.scheme_depths(rec["zcgs"]))
        own_after = S.column_content(rec["ffx_out"] * r, S.scheme_depths(rec["zcgs"]))
        assert np.all(np.abs(own_after + result.outflow - own_before)
                      <= 1e-12 * own_before + 1e-300)
        lay_before = S.column_content(rec["ffx_in"] * r, dz_sweep * 100.0)
        lay_after = S.column_content(rec["ffx_out"] * r, dz_sweep * 100.0)
        gap = S.surface_precip_outflow(lay_after + result.outflow - lay_before,
                                       tables.xl, tables.ro1bl, COL)
        content = S.surface_precip_outflow(lay_before, tables.xl, tables.ro1bl, COL)
        outflow = S.surface_precip_outflow(result.outflow, tables.xl, tables.ro1bl, COL)
        if np.any(rec["ffx_in"][-1]):
            top_losses.append(-gap / content)
        else:
            # Interior moves only: the physical column closes but for the
            # ground's share of whatever left.
            assert abs(gap - ground * outflow) <= 1e-12 * content
    assert len(top_losses) == 2 and all(5e-4 < loss < 2e-3 for loss in top_losses)

    # KC-1 probe records: per call, relative to the water that left.
    layer_kc1 = kc1_initial["dz8w"][0] * 100.0
    assert np.allclose(layer_kc1[1:] / layer_kc1[:-1], ratio, rtol=1e-15, atol=0.0)
    ground_share = []
    for key, before in sorted(probe64["fall_before"].items()):
        after = probe64["fall_after"][key]
        result = S.falflux(before["ffx"], before["vrx"], before["rhocgs"],
                           before["zcgs"], kc1_dt)
        r = before["rhocgs"][:, None]
        depths = S.scheme_depths(before["zcgs"])
        own_before = S.column_content(before["ffx"] * r, depths)
        own_after = S.column_content(after["ffx"] * r, depths)
        assert np.all(np.abs(own_after + result.outflow - own_before)
                      <= 1e-12 * own_before + 1e-300), key
        lay_before = S.column_content(before["ffx"] * r, layer_kc1)
        lay_after = S.column_content(after["ffx"] * r, layer_kc1)
        out_mm = S.surface_precip_outflow(result.outflow, tables.xl, tables.ro1bl, COL)
        gap_mm = S.surface_precip_outflow(lay_after + result.outflow - lay_before,
                                          tables.xl, tables.ro1bl, COL)
        content_mm = S.surface_precip_outflow(lay_before, tables.xl, tables.ro1bl, COL)
        if not np.any(before["ffx"][-1]):
            assert abs(gap_mm - ground * out_mm) <= 1e-12 * content_mm + 1e-300, key
        elif out_mm == 0.0:
            assert gap_mm <= 0.0, key
        if out_mm > 1e-3:
            ground_share.append(gap_mm / out_mm)
    # Where rain reached the ground the column keeps 1 to 1.24 percent of it.
    assert len(ground_share) >= 8
    assert 0.009 < min(ground_share) and max(ground_share) <= ground * (1 + 1e-6)


# ---------------------------------------------------------------------------
# Refusals and contract
# ---------------------------------------------------------------------------

def test_zero_fall_speeds_are_wrfs_bare_stop():
    cases = [r for r in F.iter_refusals(build="o64") if r.routine == "FALFLUXHUCM_Z"]
    assert len(cases) == 1 and cases[0].exit_status == 0 and cases[0].message == ""
    rec = cases[0].records[0]
    with pytest.raises(FsbmRefusal) as info:
        S.falflux(rec["ffx"], rec["vrx"], rec["rhocgs"], rec["zcgs"], rec["dt"])
    assert ":5360" in info.value.where
    # A bin below the activity threshold never reaches the check.
    quiet = np.array(rec["ffx"]) * 1e-30
    result = S.falflux(quiet, rec["vrx"], rec["rhocgs"], rec["zcgs"], rec["dt"])
    assert not np.any(result.active)


def test_inputs_that_cannot_be_honoured_refuse():
    nz = 5
    zcgs = S.level_heights_cgs(np.full(nz, 100.0))
    rho = np.full(nz, 1e-3)
    ffx = np.zeros((nz, NKR))
    ffx[2, 10] = 1.0
    v = np.full((nz, NKR), 100.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx, -v, rho, zcgs, 12.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx[:1], v[:1], rho[:1], zcgs[:1], 12.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx, v, rho, zcgs[::-1], 12.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx, v[:, :10], rho, zcgs, 12.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx, v, rho, zcgs, 0.0)
    with pytest.raises(FsbmRefusal):
        S.falflux(ffx, v, -rho, zcgs, 12.0)
    with pytest.raises(FsbmRefusal):
        S.falflux_bin(ffx[:, 10], v[:, 10], rho, zcgs, 12.0, 0)


def test_ice_is_milestone_3_unless_zero(tables):
    nz = 6
    zcgs = S.level_heights_cgs(np.full(nz, 100.0))
    rho = np.full(nz, 1e-3)
    p = np.full(nz, 9.0e5)
    zero = np.zeros((nz, NKR))
    snow = zero.copy()
    snow[3, 20] = 1.0
    dist = ST.FsbmDistributions(ff1r=zero.copy(), ff3r=snow, ff4r=zero.copy(),
                                ff5r=zero.copy(), fccn=np.zeros((nz, 43)))
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.sediment_column(dist, tables, p, rho, zcgs, 12.0)
    hail = ST.FsbmDistributions(ff1r=zero.copy(), ff3r=zero.copy(), ff4r=zero.copy(),
                                ff5r=snow, fccn=np.zeros((nz, 43)))
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.sediment_column(hail, tables, p, rho, zcgs, 12.0)
    # Graupel bins are not the dense class when hail_opt = 1, but they are
    # when hail_opt = 0.
    graupel = ST.FsbmDistributions(ff1r=zero.copy(), ff3r=zero.copy(), ff4r=snow,
                                   ff5r=zero.copy(), fccn=np.zeros((nz, 43)))
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.sediment_column(graupel, tables, p, rho, zcgs, 12.0, hail_opt=0)
    # iceprocs = 0 never calls the ice falls.
    assert S.sediment_column(dist, tables, p, rho, zcgs, 12.0, iceprocs=0).snow is None

    one = np.zeros(NKR)
    one[5] = 1.0
    v = np.full(NKR, 100.0)
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.surface_precip_wrf_block(one, v, tables, 12.0, snow_level1=one, v_snow=v)
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.surface_precip_wrf_block(one, v, tables, 12.0, dense_level1=one, v_dense=v)
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.surface_precip_outflow_block(one, tables, snow_outflow=one)
    with pytest.raises(FsbmRefusal, match="milestone 3"):
        S.surface_precip_outflow_block(one, tables, dense_outflow=one)
    with pytest.raises(FsbmRefusal):
        S.surface_precip_wrf_block(one, v, tables, 12.0, snow_level1=np.zeros(NKR))
    # Zero ice adds exactly nothing through the transcribed accumulations.
    plain = S.surface_precip_wrf_block(one, v, tables, 12.0, rainnc=3.0)
    zeros = S.surface_precip_wrf_block(one, v, tables, 12.0, rainnc=3.0,
                                       snow_level1=np.zeros(NKR), v_snow=v,
                                       dense_level1=np.zeros(NKR), v_dense=v,
                                       snownc=1.0, graupelnc=2.0)
    assert (zeros.rainnc, zeros.rainncv) == (plain.rainnc, plain.rainncv)
    assert (zeros.snownc, zeros.snowncv, zeros.graupelnc, zeros.graupelncv) == (
        1.0, 0.0, 2.0, 0.0)


def test_the_functions_do_not_modify_their_inputs(tables, probe64, kc1_dt):
    before = probe64["fall_before"][(80, 1)]
    args = [np.array(before[name]) for name in ("ffx", "vrx", "rhocgs", "zcgs")]
    copies = [a.copy() for a in args]
    S.falflux(*args, kc1_dt)
    S.fall_species(args[0] * args[2][:, None], args[1], args[2], args[3], kc1_dt)
    for a, c in zip(args, copies):
        assert np.array_equal(a, c)
    dist = _probe_distributions(before)
    kept = dist.ff1r.copy()
    S.sediment_column(dist, tables, before["pcgs"], before["rhocgs"], before["zcgs"], kc1_dt)
    assert np.array_equal(dist.ff1r, kept)
    ff = dist.ff1r[0].copy()
    S.surface_precip_level1(dist.ff1r[0], before["vrx"][0], tables.xl, tables.ro1bl,
                            COL, kc1_dt)
    assert np.array_equal(dist.ff1r[0], ff)


def test_the_module_holds_no_device_code():
    import gpuwm.core.fast_sbm.sedimentation as module
    text = open(module.__file__, encoding="utf-8").read()
    assert "cupy" not in text and "np.sum(" not in text
