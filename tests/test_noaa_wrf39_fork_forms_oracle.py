"""The HRRR door's operational-branch forms against the branch's own Fortran.

The HRRR door (gpuwm/data/physics_sources/request-defaults.v1.toml) selects
forms that exist only in the operational RAP/HRRR branch (NOAA-EMC/HRRR
v4.1.21, WRFV3.9), not in stock WRF v4.6.1.  tools/fork_forms_oracle
compiles the branch's own statements for each (fetched from its public
commit, sha256-pinned in fetch_fork.py), runs WOOF's production device path
on the same words, and compares every output word.  The fixtures in
tests/data/oracles/fork_forms are the branch's words (gfortran 13.3.0 -O0,
no contraction, glibc 2.39; pack_fixtures.py); these tests rerun only the
WOOF side.

* mp_zero_out = 2, mp_zero_out_thresh = 1e-12, mp_zero_out_all = 1: the
  branch's microphysics_zero_outb/_outa on the moist and scalar arrays,
  six configurations, 120 columns x 20 levels each.
* alb_sol = 1: radconst, calc_coszen at the interval midpoint and the
  ALBSOL block, 128 columns, four start times, three calls each.  This is
  the test that fails on the pre-fix path (NumPy's float32 sin/cos/arcsin
  on the host gave COSZEN up to 8 ULP off and ALBSOL up to 2 ULP).
* upper_wind_limiter_form = noaa_wrf39: advance_w's saved-wind limiter,
  12 cases of 160 columns x 50 levels.
* diff_6th_form = noaa_wrf39 + diff_6th_factor2: the per-row factor and
  step, and sixth_order_diffusion on 16 rows in three regimes (24 x 20
  columns, 12 levels; the lane ran six).
* rrtmg_cloud_optics_form = noaa_wrf39: WRF v4.6.1's RRTMG with the
  branch's two wrapper cloud rules, the legacy RRTMG column oracle's 128
  columns and nine regimes.
* ruc_soilprop / ruc_irrigation / ruc_snow = wrf_45, ruc_qvg_cold_start =
  air, ruc_2m_diagnostic = log_profile: the branch's LSMRUC and
  SFCDIAGS_RUCLSM inside its surface-driver arm, 102 columns, 8 steps.
  SNOWFALLAC is the one field graded by a rule instead of by bits: the
  branch accumulates metres under a Registry label of mm (its own
  inconsistency, documented, not copied); WOOF accumulates mm.

Strict build (GPUWM_WRF_EXACT=1): every word bitwise.  Default arithmetic
flushes subnormals to zero on the device; there the only words allowed to
differ are the ones that class predicts (named per test), and every other
word must still match bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_GPU

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "oracles" / "fork_forms"
TOOLS = REPO / "tools" / "fork_forms_oracle"
F = np.float32
FLT_MIN = np.float32(1.17549435e-38)


def gpu_gate(fn):
    return pytest.mark.gpu(
        pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")(fn))


def _tool(name, where=TOOLS):
    """Load a tool module by path under a name of its own: the RUC and RRTMG
    oracle directories both hold a run_woof.py."""
    import importlib.util
    key = f"_fork_forms_oracle_{where.name}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    if str(where) not in sys.path:
        sys.path.insert(0, str(where))
    spec = importlib.util.spec_from_file_location(key, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _strict():
    from gpuwm import wrf_exact
    return bool(wrf_exact.ENABLED)


def _load(name):
    return dict(np.load(FIXTURE / name))


def _tiny(a):
    """Zero or subnormal: the words a flush-to-zero device can move."""
    a = np.asarray(a, np.float32)
    return np.abs(a) < FLT_MIN


def _differ(got, want):
    return (np.ascontiguousarray(got, np.float32).view(np.uint32)
            != np.ascontiguousarray(want, np.float32).view(np.uint32))


# ----------------------------------------------------------------- CPU

def test_every_fixture_carries_its_recorded_hash():
    prov = json.loads((FIXTURE / "provenance.json").read_text())
    names = sorted(p.name for p in FIXTURE.glob("*.npz"))
    assert names == sorted(prov["fixture_sha256"])
    for name in names:
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert digest == prov["fixture_sha256"][name], name
    # The branch sources are the pinned ones.
    pins = _tool("fetch_fork").PIN
    for part in ("zero_out", "albsol"):
        for path, digest in prov[part]["sources"].items():
            assert pins[path] == digest, (part, path)


def test_the_bit_gate_is_not_vacuous():
    a = np.linspace(0.1, 1.0, 64, dtype=np.float32)
    b = a.copy()
    b.view(np.uint32)[5] += 1
    assert int(_differ(a, b).sum()) == 1
    neg = np.array([-0.0], np.float32)
    assert bool(_differ(neg, np.zeros(1, np.float32))[0])   # signed zero counts


def test_diff6_rows_take_the_branch_factor_and_step():
    """rk_tendency rows: diff_6th_factor and grid%dt; moist and scalar rows:
    diff_6th_factor2 and rk_scalar_tend's dt_rk*(rk_order-rk_step+1) with
    dt_rk = grid%dt/3. (the branch's float32 words, from its own
    expression compiled by gfortran); TKE: diff_6th_factor and that step."""
    d6 = _tool("diff6_regimes")
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    data = _load("diff6.npz")
    full = dict(zip(data["steps/dt"].tolist(), data["steps/full"]))
    assert any(full[k] != np.float32(k) for k in full), \
        "a step whose rebuilt full step is not dt itself must be in the set"
    dry = ("smag_ru", "smag_rv", "smag_rw", "smag_rth")
    for dt, want_full in full.items():
        cfg = d6._fork_cfg(float(np.float32(dt)))
        for slot, _array, _st in d6.ROWS:
            got_dt = np.float32(_diff6_dt(cfg, slot))
            want_dt = np.float32(dt) if slot in dry else want_full
            assert got_dt.view(np.uint32) == want_dt.view(np.uint32), (dt, slot)
            want_factor = np.float32(0.12 if slot in dry + ("smag_rtke",) else 0.04)
            assert np.float32(_diff6_factor(cfg, slot)) == want_factor, (dt, slot)


def test_alb_sol_runs_the_device_fork_cosine():
    """The forecast's alb_sol step is solar_albedo_radiation_call, whose
    cosine is the swint unit's fork calc_coszen; the NumPy radconst /
    calc_coszen twins no longer feed it."""
    import ast
    source = (REPO / "gpuwm" / "core" / "physics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_run_radiation")
    text = ast.get_source_segment(source, run)
    assert "solar_albedo_radiation_call(" in text
    # The strict (GPUWM_WRF_EXACT=1) arm is the fork's device cosine; a
    # default process keeps 2.9.1's solar generations in the else arm.
    strict = [n for n in ast.walk(run) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == "WRF_EXACT"
              and "solar_albedo_radiation_call(" in ast.unparse(n)]
    assert len(strict) == 1
    body = ast.unparse(ast.Module(body=strict[0].body, type_ignores=[]))
    assert "solar_albedo_radiation_call(" in body
    assert "calc_coszen" not in body and "radconst" not in body
    albedo = (REPO / "gpuwm" / "core" / "solar_albedo.py").read_text(encoding="utf-8")
    assert '"swint_coszen_loc"' in albedo


def test_chklowq_follows_the_ruc_lineage_on_the_host():
    """The branch (LSMRUC:1083-1087) and WRF v4.0-4.5 zero CHKLOWQ only under
    the MYJ PBL; v4.6.1 drops that test.  On humid evaporating columns the
    v4.6.1 lineage writes 0 and the wrf_45 lineage 1."""
    sys.path.insert(0, str(REPO / "tests"))
    from ruc_mosaic_fixture import driver_calls
    from gpuwm.core.ruc import ruc_land_surface_step
    _label, values, keywords, _expected = next(driver_calls())
    t = values["t3d"].astype(np.float64)
    p = values["p8w"].astype(np.float64)

    def qsat(tk):
        es = 611.2 * np.exp(17.67 * (tk - 273.15) / (tk - 29.65))
        return 0.622 * es / (p - es)

    values["qv3d"][...] = (0.97 * qsat(t)).astype(np.float32)
    values["soilt"][...] = (t + 3.0).astype(np.float32)
    values["tso"][...] = (t + 3.0).astype(np.float32)
    values["qvg"][...] = qsat(t + 3.0).astype(np.float32)
    values["soilmois"][...] = 0.45
    values["sh2o"][...] = 0.45
    v461 = ruc_land_surface_step({k: v.copy() for k, v in values.items()},
                                 **keywords)
    v45 = ruc_land_surface_step({k: v.copy() for k, v in values.items()},
                                **dict(keywords, snow="wrf_45"))
    assert np.all(np.asarray(v461.chklowq) == 0.0)
    assert np.all(np.asarray(v45.chklowq) == 1.0)


# ----------------------------------------------------------------- GPU

ZERO_OUT_CONFIGS = [c[0] for c in _tool("zero_out_oracle").CONFIGS]


@gpu_gate
@pytest.mark.parametrize("config", ZERO_OUT_CONFIGS)
def test_zero_out_matches_the_branch(config):
    z = _tool("zero_out_oracle")
    name, mp, mode, thresh, specified = next(c for c in z.CONFIGS if c[0] == config)
    fields = z.make_fields(mp, thresh)
    woof = z.run_woof(fields, mp, mode, thresh, specified)
    data = _load("zero_out.npz")
    misses = {}
    for species, got in woof.items():
        want = (fields[species].view(np.uint32)
                ^ data[f"{config}/{species}"]).view(np.float32)
        bad = _differ(got, want)
        if not _strict():
            # Default arithmetic: a negative subnormal input compares as -0
            # on the device, so the floor keeps it where the branch zeroes it.
            bad &= ~_tiny(fields[species])
        if bad.any():
            misses[species] = int(bad.sum())
    assert not misses, f"{config}: words differing from the branch {misses}"


ALBSOL_CASES = range(len(_tool("albsol_oracle").STARTS))


@gpu_gate
@pytest.mark.parametrize("case", ALBSOL_CASES)
def test_alb_sol_matches_the_branch(case):
    import cupy as cp
    from gpuwm.core.solar_albedo import solar_albedo_radiation_call
    a = _tool("albsol_oracle")
    inputs = a.inputs_for()
    data = _load("albsol.npz")
    scalars = data["scalars"][case]
    lat = cp.asarray(inputs["lat_case"][case].reshape(1, a.NCOL))
    lon = cp.asarray(inputs["lon_case"][case].reshape(1, a.NCOL))
    fields = {n: cp.asarray(np.asarray(inputs[n], F).reshape(1, a.NCOL))
              for n in ("albedo", "albbck", "xland", "snow", "xice")}
    fields["ivgtyp"] = cp.asarray(np.asarray(inputs["ivgtyp"], np.int32).reshape(1, a.NCOL))
    fields["albsol"] = cp.full((1, a.NCOL), -1.0, cp.float32)
    fields["albbcksol"] = cp.full((1, a.NCOL), -1.0, cp.float32)
    flushable = (_tiny(inputs["snow"]) & (np.asarray(inputs["snow"], F) != 0)) | (
        _tiny(inputs["xice"]) & (np.asarray(inputs["xice"], F) != 0))
    misses = {}
    for k in range(a.NCALLS):
        julian, xtime, gmt = (F(v) for v in scalars[k])
        cosine = solar_albedo_radiation_call(fields, lat, lon, julian, xtime, gmt,
                                             F(a.RADT), initialize=(k == 0))
        got = {"albsol": fields["albsol"], "albbcksol": fields["albbcksol"],
               "coszen": cosine}
        for f, name in enumerate(("albsol", "albbcksol", "coszen")):
            bad = _differ(cp.asnumpy(got[name]).reshape(-1), data["out"][case, k, f])
            if not _strict():
                # Default arithmetic: a subnormal SNOW or XICE reads as zero
                # in the mask test, so that column is normalised.
                bad &= ~flushable
            if bad.any():
                misses[(k, name)] = int(bad.sum())
    assert not misses, f"case {case}: words differing from the branch {misses}"


@gpu_gate
def test_the_pre_fix_numpy_cosine_misses_the_branch():
    """Negative control: the alb_sol oracle catches the defect it fixed."""
    import cupy as cp
    from gpuwm.core.rrtmg_legacy import calc_coszen, radconst
    a = _tool("albsol_oracle")
    inputs = a.inputs_for()
    data = _load("albsol.npz")
    differing = 0
    for case in ALBSOL_CASES:
        julian, xtime, gmt = (F(v) for v in data["scalars"][case][1])
        declin, _ = radconst(julian)
        cosine = calc_coszen(julian, F(xtime + F(F(a.RADT) * F(0.5))), gmt,
                             inputs["lat_case"][case], inputs["lon_case"][case], declin)
        differing += int(_differ(np.asarray(cosine, F), data["out"][case, 1, 2]).sum())
    assert differing > 0


@gpu_gate
def test_upper_wind_limiter_matches_the_branch():
    u = _tool("upper_wind_regimes")
    data = _load("upper_wind.npz")
    seed = 0
    misses = {}
    changed = 0
    for specified in (True, False):
        for terrain3d in (False, True):
            for substeps in (1, 2, 4):
                seed += 1
                case = u.make_case(seed, specified, substeps, terrain3d)
                ug, vg = u.run_woof(case)
                for name, got, inp in (("u", ug, case["u"]), ("v", vg, case["v"])):
                    xor = data[f"case{seed}/{name}"]
                    changed += int(np.count_nonzero(xor))
                    want = (inp.view(np.uint32) ^ xor).view(np.float32)
                    bad = _differ(got, want)
                    if bad.any():
                        misses[(seed, name)] = int(bad.sum())
    assert changed > 1000, "the cases must exercise the limiter"
    assert not misses, f"words differing from the branch {misses}"


DIFF6_REGIMES = list(_tool("pack_fixtures").FIXTURE_DIFF6_REGIMES)


@gpu_gate
@pytest.mark.parametrize("regime", DIFF6_REGIMES)
def test_diff6_fork_form_matches_the_branch(regime):
    import cupy as cp
    d = _tool("diff6_regimes")
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor, launch_diff6_to_edge
    data = _load("diff6.npz")
    cfg = d._fork_cfg(20.0)
    seed = d.REGIMES.index(regime) * len(d.ROWS)
    misses = {}
    for slot, _array, st in d.ROWS:
        seed += 1
        c = d._case(regime, slot, st, seed)
        tend = cp.zeros(c["f"].shape, cp.float32)
        launch_diff6_to_edge(cp.asarray(c["f"]), tend, cp.asarray(c["mut"]),
                             cp.asarray(c["c1"]), cp.asarray(c["c2"]),
                             float(F(_diff6_factor(cfg, slot))), float(F(_diff6_dt(cfg, slot))),
                             2, stagger=st, phb=cp.asarray(c["phb"]),
                             msfu=cp.asarray(c["msfu"]), msfv=cp.asarray(c["msfv"]),
                             msft=cp.asarray(c["msft"]), slopeopt=1, thresh=0.05,
                             dx=d.DX, dy=d.DX)
        got = tend.get()
        want = data[f"{regime}/{slot}"]
        bad = _differ(got, want)
        if not _strict():
            # Default arithmetic flushes subnormal results to zero.  The
            # filter's limiter tests the sign of flux*(f(i)-f(i-1)); where
            # the field holds words below 1e-12 that product falls below
            # FLT_MIN, flushes to 0 and flips the branch (measured: tendency
            # words up to 1.1e-12 on qc, qr and qg, all within three points
            # of such words).  Only those points, and words that are zero
            # or subnormal on either side, may move.
            small = (c["f"] != 0) & (np.abs(c["f"]) < np.float32(1e-12))
            bad &= ~(_tiny(got) | _tiny(want) | _near(small))
        if bad.any():
            misses[slot] = int(bad.sum())
    assert not misses, f"{regime}: words differing from the branch {misses}"


def _near(mask):
    """Points within the sixth-order stencil (3 points) of a masked point."""
    out = mask.copy()
    for axis in (1, 2):
        for shift in (1, 2, 3):
            out |= np.roll(mask, shift, axis) | np.roll(mask, -shift, axis)
    return out


RRTMG_FIXTURE = REPO / "tests" / "data" / "oracles" / "rrtmg_legacy" / "column"
RRTMG_REGIMES = sorted(json.loads((RRTMG_FIXTURE / "regimes.json").read_text())["regimes"])


@gpu_gate
@pytest.mark.parametrize("regime", RRTMG_REGIMES)
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="graded on the legacy RRTMG column oracle's strict (GPUWM_WRF_EXACT=1) route; a default process keeps 2.9.1's solar generations")
def test_rrtmg_noaa_wrf39_cloud_form_matches_the_branch_rules(regime, monkeypatch):
    import cupy  # noqa: F401
    run_woof = _tool("run_woof", REPO / "tools" / "rrtmg_legacy_column_oracle")
    monkeypatch.setenv("RRTMG_CLOUD_OPTICS_FORM", "noaa_wrf39")
    meta = json.loads((RRTMG_FIXTURE / "regimes.json").read_text())
    woof = run_woof.run_regime(RRTMG_FIXTURE, meta, regime)
    want = _load("rrtmg_fork_cloud.npz")
    graded = {"rthratenlw": "rthratenlw", "rthratensw_wrapper": "rthratensw",
              "rthraten": "rthraten", "glw": "glw", "olr": "olr", "gsw": "gsw",
              "swdown": "swdown", "coszen": "coszen", "cldfra": "cldfra",
              "swddir": "swddir", "swddif": "swddif", "o3rad": "o3rad"}
    misses = {}
    for wk, ok in graded.items():
        key = f"{regime}/{wk}"
        if wk == "o3rad" and meta["regimes"][regime]["o3input"] != 2:
            continue    # o3input = 0: the wrapper builds O3DATA itself
        assert key in want and ok in woof, (regime, wk)
        bad = _differ(woof[ok], want[key])
        if bad.any():
            misses[wk] = int(bad.sum())
    assert not misses, f"{regime}: values differing from the branch rules {misses}"


RUC_FIXTURES = sorted(p.name for p in FIXTURE.glob("ruc_fork_*.npz"))


@gpu_gate
@pytest.mark.parametrize("name", RUC_FIXTURES)
def test_ruc_fork_forms_match_the_branch(tmp_path, name):
    import hashlib as _h
    data = np.load(FIXTURE / name)
    receipt = json.loads(bytes(data["receipt"]).decode())
    assert receipt["lineage"] == "fork"
    tool = REPO / "tools" / "ruc_lsm_gpu_oracle"
    columns = _tool("columns", tool)
    run_woof = _tool("run_woof", tool)
    layout = _tool("layout", tool)
    o = receipt["options"]
    case = columns.build_case(
        nzs=o["nzs"], nsteps=o["nsteps"], dt=o["dt"],
        fractional_seaice=o["fractional_seaice"], mosaic=bool(o["mosaic"]),
        lakemodel=o["lakemodel"], rdlai2d=bool(o["rdlai2d"]))
    case["lineage"] = "fork"
    run_woof.run(case, tmp_path)
    digest = _h.sha256((tmp_path / "inputs.bin").read_bytes()).hexdigest()
    assert digest == receipt["inputs_sha256"]
    woof = np.load(tmp_path / "woof.npz")
    differing = {}
    for field in layout.INIT_2D + layout.INIT_PROFILES:
        bad = _differ(woof[f"init__{field}"], data[f"init__{field}"])
        if bad.any():
            differing[("init", field)] = int(bad.sum())
    snowfall = 0
    for k in range(1, o["nsteps"] + 1):
        for field in layout.OUT_2D + columns.PROFILES:
            want = data[f"step{k}__" + (layout.SFCEVP_ONCE_ACC
                                        if field == "sfcevp" else field)]
            got = woof[f"step{k}__{field}"]
            if field == "snowfallac":
                # The branch accumulates metres (module_sf_ruclsm.F:2168)
                # under its Registry's "[mm]"; WOOF accumulates mm.  The
                # run totals agree to float32 rounding of the unit factor.
                snowfall += int(np.count_nonzero(want))
                rel = np.abs(got.astype(np.float64) * 1.0e-3 - want) / np.maximum(
                    np.abs(want.astype(np.float64)), 1e-30)
                bad = (rel > 2.0e-7) & (want != 0)
                bad |= (want == 0) & (got != 0)
            else:
                bad = _differ(got, want)
            if bad.any():
                differing[(k, field)] = int(bad.sum())
    assert snowfall > 0, "the case must accumulate snowfall"
    assert not differing, f"words differing from the branch: {differing}"
