"""mp_physics=28 on real model columns: the port's own kernels, compiled for
the host, against WRF v4.6.1's own Fortran, process rate by process rate.

The breakage each gate prevents, named:

1. ``test_rate_readback_anchors_match_the_kernels`` -- the harness reads
   every process rate out of the production kernels by inserting readback
   calls at one anchor line per kernel, next to local variables it names
   (tools/thompson_real_column_parity/port_rates.py): the three aerosol
   units for mp=28 and the classic kernels of thompson.cu for mp=8.  A
   kernel edit that moves an anchor or renames a rate variable breaks the
   harness, and with it the only process-by-process check of either port
   against WRF on real columns.  Pure text; runs everywhere.

2. ``test_the_fixture_covers_every_repaired_regime`` -- the committed
   fixture was cut per regime from WRF's own checkpoint streams
   (tools/thompson_real_column_parity/make_fixture.py).  A re-cut that
   loses a regime would leave the repair that lives there unguarded while
   gate 3 stays green.  Pure data; runs everywhere.

3. ``test_real_columns_against_wrf461`` -- the fixture
   ``tests/data/thompson_real_columns_wrf461.npz`` holds 42 saved real-data
   model columns and the answers unmodified WRF v4.6.1 gave for them
   (gfortran 13.3, glibc 2.39).  The port's production adapter runs on the
   same float32 inputs through the host backend, and every process rate,
   the final state, reflectivity and surface precipitation are compared.

   What it pins: the cells where a quantity differs from WRF by more than
   1e-2 relative and neither rounding rule explains the gap (the port's own
   response to a one-unit nudge of every input, or four float32 units of
   the largest value the cell held; plus the two named rules for the rates
   rounding itself decides, see fixture_check.py).  Every process rate and
   every final-state quantity must have none.  (Until the no-micro column
   exit and the terminal vapour floor were repaired the final vapour was
   pinned at four such cells: G, WRF floors vapour at 1.E-10 at every level
   of a column with microphysics (:3974) and the port left a level at
   exactly zero at zero.)  Undoing any of the twelve repairs
   of 2026-09-23 fails this gate: measured for the entry rewrite (the echo
   then differs by 9.4 dB, and ice nucleation in four cells) and the cloud
   fallout gate (cloud water and droplet number in two cells), and the
   port as it stood before the repairs fails it on 20 rates and 10
   final-state quantities, its echo 9.4 dB from WRF.  Reflectivity is
   held to 0.01 dB.

   Where it runs: a POSIX box with a C++ compiler and the staged Thompson
   table set, or the Windows desktop through WSL with the same.  It skips,
   naming the missing piece, where either is absent -- the hosted CI runners
   carry no ``freezeH2O.dat`` (254 MB, not packaged).

4. ``test_real_columns_against_wrf461_mp8`` -- the same 42 columns with the
   answers WRF's same module gives them as classic Thompson
   (``tests/data/thompson_real_columns_wrf461_mp8.npz``, written by
   ``make_fixture.py --mp8``), graded through ``gpuwm.core.microphysics.
   _apply_thompson``.  mp=8 runs the shared ``thompson.cu`` kernels, and
   before this gate nothing compared its process rates with WRF's.  It pins,
   per rate and per final-state quantity, the cells beyond 1e-2 that no
   rounding rule explains, EXACTLY: a count that grows is a new difference
   from WRF, and a count that shrinks is a repair that must retire its pin
   here.  Every count it recorded when it was committed (f382c8825) was
   retired by the repair that closed it, so the pins are empty and the
   echo is held to 0.05 dB (see ``PINNED_RATE_CELLS_MP8``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "thompson_real_column_parity"
_FIXTURE = _ROOT / "tests" / "data" / "thompson_real_columns_wrf461.npz"
_FIXTURE_MP8 = (_ROOT / "tests" / "data"
                / "thompson_real_columns_wrf461_mp8.npz")

#: Final-state cells beyond 1e-2 relative that no rounding rule explains, per
#: quantity, on the committed fixture.  Quantities absent here must have none.
#: Empty: the one entry it held, ``"qv": 4`` for the unrepaired vapour floor
#: (G, module_mp_thompson.F:2020 and :3974), was retired with its repair.
PINNED_FINAL_CELLS: dict[str, int] = {}
REFL_MAX_DB = 1.0e-2

#: Every regime a repair acts in, by the label make_fixture.py gives it.
REQUIRED_REGIMES = (
    "A entry ice without number",
    "B cold snow riming",
    "C rain emptied at the source stage",
    "C/F graupel emptied at the source stage",
    "D/H mass above R1 below it as a concentration",
    "E ice collection",
    "F melting graupel",
    "G vapour at zero",
    "K orphan ice number where WRF nucleates",
    "K orphan rain number",
    "L cloud formed where the gate is shut",
    "O graupel at or below R1 at entry",
    "rain evaporation at saturation",
    "rain break-up diameter",
)


def test_rate_readback_anchors_match_the_kernels():
    sys.path.insert(0, str(_TOOL))
    try:
        import port_rates
        from instrument_wrf_rates import RATES
    finally:
        sys.path.remove(str(_TOOL))
    from gpuwm.core.kernels import module_source

    for mp, table in port_rates.ANCHORS_BY_MP.items():
        for module, anchors in table.items():
            text = module_source(module)
            instrumented = port_rates.instrument(module, text, mp)
            for _anchor, rate_map in anchors:
                for wrf_name, expr in rate_map.items():
                    assert (wrf_name in RATES
                            or wrf_name in port_rates.DIAGNOSTICS)
                    assert wrf_name not in port_rates.NOT_CARRIED[mp]
                    assert f"/* {wrf_name} */" in instrumented, (
                        mp, module, wrf_name)
                    # every rate expression is a local the kernel really
                    # declares
                    assert (f" {expr} = " in text or f" {expr};" in text
                            or f"double {expr}" in text), (mp, module, expr)
            # nothing but the guarded readback was inserted: every added
            # line is a HOST_RATE call or its #ifdef/#endif guard
            added = (len(instrumented.splitlines())
                     - len(text.splitlines()))
            expected = sum(len(rate_map) + 2 for _a, rate_map in anchors)
            assert added == expected, (mp, module, added, expected)
            calls = [line for line in instrumented.splitlines()
                     if "HOST_RATE(" in line]
            assert len(calls) == sum(len(m) for _a, m in anchors), (
                mp, module)
    # mp=8 reads back every WRF rate classic Thompson carries: each is either
    # in one of the classic kernels' maps or named as not carried.
    carried = set()
    for anchors in port_rates.ANCHORS_MP8.values():
        for _anchor, rate_map in anchors:
            carried |= set(rate_map)
    missing = [name for name in RATES
               if name not in carried
               and name not in port_rates.NOT_CARRIED_MP8]
    assert not missing, missing


def test_the_fixture_covers_every_repaired_regime():
    assert _FIXTURE.exists(), _FIXTURE
    z = np.load(_FIXTURE)
    labels = set(str(label) for label in z["labels"])
    missing = [r for r in REQUIRED_REGIMES if r not in labels]
    assert not missing, missing
    receipt = str(z["wrf_build_receipt"])
    # The WRF source the answers came from, by content.
    assert ("fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"
            in receipt), receipt


def _wsl_python():
    wsl = shutil.which("wsl.exe")
    if wsl is None:
        return None
    probe = subprocess.run(
        [wsl, "--exec", "bash", "-lc",
         "command -v g++ >/dev/null && python3 -c 'import numpy'"],
        capture_output=True, check=False)
    if probe.returncode != 0:
        return None

    def unix(path):
        done = subprocess.run([wsl, "--exec", "wslpath", "-a", str(path)],
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()

    return wsl, unix


def _run_tool(tmp_path, script_name, *args):
    """Run one of the harness scripts on the host backend and return its
    JSON: in process's own Python on a POSIX box with a C++ compiler, and
    through WSL on Windows."""
    script = _TOOL / script_name
    build = tmp_path / "host-kernels"
    if os.name == "nt":
        found = _wsl_python()
        if found is None:
            pytest.skip("no native C++ compiler on Windows and no WSL with "
                        "g++ and numpy to build the host kernels in")
        wsl, unix = found
        cmd = [wsl, "--exec", "env",
               f"GPUWM_HOST_KERNEL_BUILD_DIR={unix(build)}",
               "python3", unix(script),
               *[unix(a) if isinstance(a, Path) else a for a in args]]
        env = None
    else:
        if shutil.which(os.environ.get("CXX", "g++")) is None:
            pytest.skip("no C++ compiler to build the host kernels")
        cmd = [sys.executable, str(script), *[str(a) for a in args]]
        env = dict(os.environ, GPUWM_HOST_KERNEL_BUILD_DIR=str(build))
    done = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          check=False)
    if done.returncode != 0:
        tail = (done.stdout + done.stderr)[-3000:]
        lowered = tail.lower()
        if "freezeh2o" in lowered or ("table" in lowered and (
                "missing" in lowered or "not found" in lowered)):
            pytest.skip("the classic Thompson table set is not staged "
                        "(freezeH2O.dat is not packaged):\n" + tail)
        raise AssertionError(f"{script_name} failed:\n" + tail)
    return json.loads(done.stdout)


def _run_check(tmp_path, fixture=_FIXTURE):
    return _run_tool(tmp_path, "fixture_check.py", fixture)


def test_real_columns_against_wrf461(tmp_path):
    assert _FIXTURE.exists(), _FIXTURE
    result = _run_check(tmp_path)

    lines = []
    for name, entry in result["rates"].items():
        if entry["n_far_unexplained"]:
            lines.append(f"rates.{name}: {entry['n_far_unexplained']} cells "
                         "beyond 1e-2 that rounding does not explain")
    for name, entry in result["final"].items():
        if "n_far_unexplained" not in entry:
            continue
        pinned = PINNED_FINAL_CELLS.get(name, 0)
        if entry["n_far_unexplained"] != pinned:
            lines.append(f"final.{name}: {entry['n_far_unexplained']} cells "
                         f"beyond 1e-2 unexplained, pinned {pinned}")
    for name in ("T_exit", "rainnc", "snownc", "graupelnc"):
        if result["final"][name]["rel_max"] > 1.0e-5:
            lines.append(f"final.{name}: relative "
                         f"{result['final'][name]['rel_max']:.3e}")
    if result["refl"]["max_abs_db"] > REFL_MAX_DB:
        lines.append(f"refl: {result['refl']['max_abs_db']:.3e} dB from WRF")
    assert not lines, "\n".join(lines)


# ---------------------------------------------------------------------------
# mp_physics=8 on the same columns.
# ---------------------------------------------------------------------------

#: Process-rate cells beyond 1e-2 that no rounding rule explains, per rate, on
#: the committed mp=8 fixture: none.  Each entry this table held was a WRF
#: v4.6.1 rule the classic kernels of thompson.cu did not follow, the
#: counterpart of a repair the mp=28 kernels carry (their comments name the
#: WRF lines), read cell by cell off this fixture and retired with its
#: repair:
#:
#: * A, :1855-1858, ice arriving with mass and no number given 5 micron
#:   crystals before its size clamps and read so by the nucleation (:2627):
#:   ``pri_ide`` 9, ``pni_ide`` 8, ``prs_ide`` 1, ``pni_iau`` 9, ``prs_iau``
#:   9, the rain-collects-ice set (``pri_rci``, ``pni_rci``, ``prr_rci``,
#:   ``pnr_rci``, ``prg_rci``, 1 each, whose gate reads the ice size), the
#:   rain freezing and rain-snow collection the rain limiter scales with it
#:   (``pri_rfz``, ``prg_rfz``, ``prr_rcs``, 1 each), one ``pni_sci``,
#:   ``prs_sci`` 1 and one cell each of rain evaporation;
#: * E, :2649 and :2713, collected ice turned into a number with WRF's D0i
#:   minimum crystal mass: ``pni_sci`` 2;
#: * O, :2703, the graupel sublimation number gated on the entry mixing
#:   ratio: ``png_gde`` 7;
#: * C, :3067-3091, rain whose post-source concentration is at or below R1
#:   removed with its entry mass: ``prv_rev`` 3 and ``pnr_rev`` 3, which
#:   read that rain.
PINNED_RATE_CELLS_MP8: dict[str, int] = {}
#: Final-state cells beyond 1e-2 that no rounding rule explains, per
#: quantity, on the committed mp=8 fixture: none.
#:
#: Retired: K, the entry rewrite (:1844-1942), repaired in the classic
#: adapter, took ``qs`` from 7 to 6, ``ni`` from 11 to 7 and ``nr`` from 2
#: to 1; L, the cloud fallout gate read from the L_qc the adjustment leaves
#: (:3485, :3645), took ``qc`` from 2 to none; A took ``qi`` from 6, ``qs``
#: from 6, ``nr`` from 1 to none, ``ni`` from 7 to 1, ``qr`` from 3 to 2,
#: ``qg`` from 9 to 7 and ``ng`` from 52 to 51; C and F, :3067-3160, rain
#: and graupel emptied at the source stage removed and the private graupel
#: number re-balanced there, took ``qr`` from 2, ``qg`` from 7 and ``ng``
#: from 51 to none; H, :4024-4040, the terminal ice bound in its
#: per-kilogram form for ice whose mixing ratio passed R1 but whose
#: concentration did not, took ``ni`` from 1 to none.
PINNED_FINAL_CELLS_MP8: dict[str, int] = {}
#: The largest reflectivity gap from WRF on the mp=8 fixture, in dB, and the
#: cells beyond 0.1 dB (9.29 dB and 5 cells until A, C, F and O were
#: repaired).
PINNED_REFL_MP8 = {"max_abs_db": 0.05, "n_gt_0p1_db": 0}
#: The exit temperature's largest relative gap from WRF on the mp=8 fixture
#: (2.7e-5 until A was repaired).
PINNED_T_EXIT_REL_MP8 = 1.0e-5


def test_real_columns_against_wrf461_mp8(tmp_path):
    assert _FIXTURE_MP8.exists(), _FIXTURE_MP8
    z = np.load(_FIXTURE_MP8)
    z28 = np.load(_FIXTURE)
    # the same columns as the mp=28 fixture, and WRF's same source
    assert int(z["mp_physics"]) == 8
    for key in ("col_p", "col_th", "col_qv", "col_qc", "col_qi"):
        assert np.array_equal(z[key], z28[key]), key
    assert ("fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"
            in str(z["wrf_build_receipt"]))
    result = _run_check(tmp_path, _FIXTURE_MP8)
    assert result["mp_physics"] == 8

    lines = []
    for name, entry in result["rates"].items():
        pinned = PINNED_RATE_CELLS_MP8.get(name, 0)
        if entry["n_far_unexplained"] != pinned:
            lines.append(f"rates.{name}: {entry['n_far_unexplained']} cells "
                         f"beyond 1e-2 unexplained, pinned {pinned}")
    for name, entry in result["final"].items():
        if "n_far_unexplained" not in entry:
            continue
        pinned = PINNED_FINAL_CELLS_MP8.get(name, 0)
        if entry["n_far_unexplained"] != pinned:
            lines.append(f"final.{name}: {entry['n_far_unexplained']} cells "
                         f"beyond 1e-2 unexplained, pinned {pinned}")
    if result["final"]["T_exit"]["rel_max"] > PINNED_T_EXIT_REL_MP8:
        lines.append(f"final.T_exit: relative "
                     f"{result['final']['T_exit']['rel_max']:.3e}")
    for name in ("rainnc", "snownc", "graupelnc"):
        if result["final"][name]["rel_max"] > 1.0e-5:
            lines.append(f"final.{name}: relative "
                         f"{result['final'][name]['rel_max']:.3e}")
    if result["refl"]["max_abs_db"] > PINNED_REFL_MP8["max_abs_db"]:
        lines.append(f"refl: {result['refl']['max_abs_db']:.3e} dB from WRF")
    if result["refl"]["n_gt_0p1_db"] != PINNED_REFL_MP8["n_gt_0p1_db"]:
        lines.append(f"refl: {result['refl']['n_gt_0p1_db']} cells beyond "
                     f"0.1 dB, pinned {PINNED_REFL_MP8['n_gt_0p1_db']}")
    assert not lines, "\n".join(lines)


# ---------------------------------------------------------------------------
# Named WRF rules on built columns (tools/thompson_real_column_parity/
# rule_checks.py), mp=28 and mp=8 through their production adapters.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rules(tmp_path_factory):
    return _run_tool(tmp_path_factory.mktemp("rules"), "rule_checks.py")


@pytest.mark.parametrize("mp", (28, 8))
def test_a_column_wrf_leaves_at_its_no_micro_exit_is_returned_untouched(
        rules, mp):
    """module_mp_thompson.F:2020 and :3974.

    Breakage prevented: a column with no condensate above R1 and no ice
    supersaturation leaves mp_thompson before the source loop, so WRF
    neither floors its vapour at 1.E-10 nor clamps its aerosol; every other
    column leaves with every level's vapour at 1.E-10 or above.  The port
    applied neither rule: it clamped the aerosol of every column (up to 1.5
    percent of nwfa at one saved analysis cell) and left zero vapour at zero
    in columns with microphysics (1 to 71 levels per saved forecast frame,
    all 2,794 analysis levels of that kind) against WRF v4.6.1's own
    Fortran.  Column 0 is the no-micro column, column 1 the same with cloud.
    """
    got = rules[f"G_no_micro_column_mp{mp}"]
    assert got["qv_top"] == [0.0, pytest.approx(1.0e-10, rel=1e-7)]
    assert got["qc_orphan"] == [0.0, 0.0]
    assert got["qi_orphan"] == [0.0, 0.0]
    assert got["ni_orphan"] == [0.0, 0.0]
    if mp == 28:
        # untouched above the 9999.E6 ceiling, and clamped to it
        assert got["nwfa_max"][0] == pytest.approx(2.0e10, rel=1e-7)
        assert got["nwfa_max"][1] == pytest.approx(9999.0e6, rel=1e-7)


@pytest.mark.parametrize("mp", (28, 8))
def test_ice_above_0c_keeps_its_number_inside_the_size_bounds(rules, mp):
    """module_mp_thompson.F:3033-3055.

    Breakage prevented: WRF's cloud ice mass/number balance runs at every
    level of a column with microphysics, so ice at a level above 0 C (which
    advection and analyses both leave) reaches the fallout with a number
    inside the 5 to 300 micron bounds.  The port applied the balance only in
    its cold source network, which returns above 0 C, so ice there fell with
    no number (the fallout then gave it 5 micron crystals) or with whatever
    out-of-bounds number it arrived with: three ice-number cells of a saved
    analysis state differed from WRF v4.6.1's own Fortran beyond 1e-2, and
    the cloud the melt makes from that ice moved with them.  Level 5 holds
    ice with no number, level 4 ice with a number far above the 5 micron
    bound; both sit near 282 to 285 K.
    """
    got = rules[f"Awarm_ice_number_mp{mp}"]
    assert all(t > 273.15 for t in got["temperature"])
    assert got["ni_after_sources"] == pytest.approx(got["ni_balance"],
                                                    rel=1.0e-6)


@pytest.mark.parametrize("mp", (28, 8))
def test_cloud_and_ice_at_or_below_r1_melt_before_they_are_removed(
        rules, mp):
    """module_mp_thompson.F:3943-3966, then :4007-4009 and :4023-4027.

    Breakage prevented: WRF melts any positive ice at a warm level into
    cloud water, and only the terminal apply after that removes cloud and
    ice at or below R1, so a level whose cloud and melted ice are each at or
    below R1 but together above it keeps that cloud.  The port's cloud and
    ice fallout removed both before the melt (mp=28's ice fallout is the
    shared thompson.cu kernel): 0 to 2 cloud cells of 1.6e-12 to 2.0e-12
    kg/kg per saved forecast frame, and the droplet number on them, came
    back as zero against WRF v4.6.1's own Fortran.  Level 2 holds 0.8e-12
    kg/kg each of cloud and ice, level 3 0.4e-12 each, near 280 K.
    """
    got = rules[f"J_melt_before_r1_mp{mp}"]
    both = float(np.float32(0.8e-12) + np.float32(0.8e-12))
    assert got["qc"][2] == pytest.approx(both, rel=1.0e-6)
    assert got["qc"][3] == 0.0
    assert got["qi"] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("mp", (28, 8))
def test_melting_snow_blends_with_the_rain_pass_s_own_fall_speed(rules, mp):
    """module_mp_thompson.F:3612-3634 and :3722-3724.

    Breakage prevented: melting snow falls at vts*SR + (1-SR)*vtrk(k) with
    SR = rs/(rs+rr), whatever the level's own rain: a level without rain has
    rr = R1 and the rain speed of the level above, and a column without rain
    has vtrk = 0.  The port blended only where the level itself held rain, at
    a rain speed and rr formed from the snow's density, so melting snow
    without rain fell unblended and the snow a melting layer leaves near R1
    differed from WRF v4.6.1's own Fortran (1 to 15 cells per saved forecast
    frame beyond 1e-2, up to 2.3e-9 kg/kg on the analysis states through the
    snow substep count).  Before the repair the two melting arms removed
    exactly what the non-melting one did; after it they remove what the rule
    says: SR = 0.84 of it with no rain, and 34 times it under rain aloft.
    """
    got = rules[f"I_melting_snow_blend_mp{mp}"]
    assert 0.5 < got["solid_fraction"] < 0.95
    assert got["vtrk"] > 10.0 * got["vts"]
    assert got["removed"]["dry"] == pytest.approx(got["rule"]["dry"],
                                                  rel=1.0e-3)
    assert got["removed"]["rain_above"] == pytest.approx(
        got["rule"]["rain_above"], rel=1.0e-3)


@pytest.mark.parametrize("mp", (28, 8))
def test_a_species_is_present_where_its_mixing_ratio_passes_r1(rules, mp):
    """module_mp_thompson.F:1827-1949 with :2783-2825.

    Breakage prevented: WRF's L_qc, L_qs, L_qg and L_qr are the MIXING
    RATIO tests of the entry block, and every process gated on them runs
    wherever they hold, with rs(k), rg(k) and rc(k) whatever q*rho is.  The
    classic source networks gated on the concentration instead, so in thin
    air, where a mixing ratio just above R1 is a concentration at or below
    it, snow and graupel did not melt or sublimate, rain did not collect
    cloud and the conservation limits skipped them: 454 snow-melt, 547
    graupel-melt and 209 rain-collects-cloud cells per five saved forecast
    frames differed from WRF v4.6.1's own Fortran beyond 1e-2, and 691 snow
    cells of the final state.  Snow of 1.1e-12 kg/kg at a level near 276 K
    where rho is about 0.8 melts in WRF and in both ports now; the classic
    port kept it.
    """
    got = rules[f"P_presence_mp{mp}"]
    assert got["concentration"][0] <= 1.0e-12 < 1.1e-12
    assert all(t > 273.15 for t in got["temperature"])
    assert got["qs"] == 0.0
    assert got["qg"] == 0.0


def test_the_classic_rain_evaporation_hands_the_fallout_wrfs_l_qr(rules):
    """module_mp_thompson.F:3236, :3252-3253 and :3501-3572.

    Breakage prevented: WRF's rain fallout forms rr(k) only where L_qr holds
    (:3236, the post-source mixing ratio), from the TAU+1 density there, and
    from the post-condensation density, floored at R1, where the rain
    evaporation rewrote it (:3568).  The classic rain fallout guessed L_qr
    from the post-evaporation mixing ratio and could not tell the floored
    rewrite from the held pair, so the rain number of saved real-data frames
    differed from WRF v4.6.1's own Fortran at thousands of levels beyond
    2e-6 and the echo by up to 13.8 dB.  The classic rain evaporation now
    marks the three cases in the fallout's density: zero, the held density,
    the negative of its own.
    """
    got = rules["R_classic_rain_presence"]
    assert got["density"][0] == 0.0
    assert got["density"][1] == pytest.approx(-got["own"][1], rel=1.0e-7)
    assert got["qr"][1] < 1.0e-4
    assert got["density"][2] == got["held"][2]
    assert got["qr"][2] == pytest.approx(1.0e-4, rel=1.0e-7)


def test_graupel_at_or_below_r1_leaves_the_call_as_zero(rules):
    """module_mp_thompson.F:4058-4063.

    Breakage prevented: WRF's terminal apply writes graupel at or below R1
    as zero, mass and number, in every column.  The port zeroed it only in
    the columns whose graupel fallout ran, so a residue the sources left in
    any other column stayed in the state, the history and every between-step
    reader (188,969 cells of one saved 19,600-column history frame carried
    0 < qg <= 1e-12).  mp=8 and mp=28 share the kernel.
    """
    got = rules["T_terminal_graupel_zero"]
    above = float(np.nextafter(np.float32(1.0e-12), np.float32(1.0)))
    assert got["qg"][:5] == [0.0, 0.0, above, pytest.approx(2.0e-12), 0.0]
    assert got["qg"][5] == pytest.approx(1.0e-4)
    assert got["ng_zero"] == [True, True, False, False, True, False]
