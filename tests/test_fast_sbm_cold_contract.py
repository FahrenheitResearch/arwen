"""The cold half's contract: behaviour fields, tile inputs, results, routine map.

Milestone 2 of the fast-sbm program builds WRF's ice phase in four items that
run in parallel (ice nucleation and phase changes, deposition, cold
collisions, fall and exit) and a fifth that joins them into the full call.
This item lands everything they share first, so none of them edits the same
dataclass.  What each gate here prevents:

* A routine of the pinned WRF file that no item owns and no reason excuses:
  the cold half would ship with a process silently missing.  The file is
  parsed (every SUBROUTINE and FUNCTION, 45 of them), and a map with a
  routine removed, an owner blanked, an unknown owner, a wrong line range,
  a part outside its routine or a milestone 1 part without a host fails.
  Every code line of every routine must lie in a part, so a part dropped
  from a routine that keeps others fails too; and the owners of each
  routine's parts, in order, are a literal table here, so a part that
  loses its owner, turns into a reason or is dropped from a routine split
  by argument (JERRATE_KS, JERTIMESC_KS, JERSUPSAT_KS, JERDFUN_KS) fails
  until the table is edited on purpose.
* A cold-half behaviour field with a usable default, or a preset that does
  not set it: an A/B arm built field by field would take WRF's defect or
  its fix silently.  Leaving any of the eight out raises ``TypeError``
  naming it; both presets set each; malformed values are refused.
* A WRF behaviour the corrected preset copies with no catalogue row: the
  finding set is closed (D1 to D18, S1 to S3), and D15's, D17's and D18's
  claims about the ice pre-pass (its loop order, the scalar w_stag, the
  wind average in both copies, the halo loops that make a tile's edge
  depend on the order of the tiles) are read off the file.
* A catalogue citation that drifted off the file: every finding and dead
  path names text that must appear in its first cited range.
* A transcribed constant that is not WRF's literal at the cited line.
* A dense-ice class chosen by a bare integer, or two options sharing an
  identity; an option whose WRF array names (mass grid, spectrum, fall
  speed, capacity, bulk density, the two kernels it meets drops with) are
  not the ones the pinned file uses for it in its hail_opt branches, its
  ONECOND2 and ONECOND3 calls and its cold block, so a hail run would take
  graupel's arrays with every other test green.
* Ice mass grids that differ from the drop grid, which would make WRF's
  bin-by-bin merges create or destroy mass: the identity holds in both
  table precisions and in both oracle builds' READs, and a grid moved by
  one ULP fails it (the negative control).
* ``FREEZ`` or ``J_W_MELT`` assigning a mass grid, which would make the
  host's "the grids are tables" rule wrong; and the cold routines reading
  a module table milestone 1 does not load.
* Cold tile inputs a call could not use (missing, misshapen, non-finite, a
  tile outside its domain, a halo whose interior disagrees with the tile),
  refused by name, and a single-column helper whose halo record lies; a
  flat layout that is not the documented formula (a producer packing WRF
  arrays by that formula would disagree with a consumer using the
  accessor).
* A tally whose collected mass has no destination, or one its pair's
  collision routine cannot send it to (read off the cold block), and a
  result that does not name its dense-ice option.
* Cold inputs that change the warm call: the same KC-1 replay call with and
  without them is byte-identical under both presets.
* The D8 field under the corrected label while the sedimentation still
  divides by the level spacing: the warm call refuses ``"layer_depth"``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import math
import re
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fast_sbm import catalogue as K
from gpuwm.core.fast_sbm import column as COL_MOD
from gpuwm.core.fast_sbm import constants as C
from gpuwm.core.fast_sbm import state as S
from gpuwm.core.fast_sbm.column import (
    COLD_INCREMENT_PROCESSES,
    COLD_TILE_FIELDS,
    HALO_FIELDS,
    HALO_NEIGHBOURS,
    HALO_NONE,
    HALO_TILE_IS_DOMAIN,
    INCREMENT_PROCESSES,
    TALLY_CLASSES,
    TALLY_DESTINATIONS,
    TALLY_PAIRS,
    WARM_INCREMENT_PROCESSES,
    FsbmIncrement,
    FsbmPoint,
    FsbmResult,
    FsbmTally,
    FsbmTile,
    FsbmWarmResult,
    check_cold_inputs,
    default_processes,
    fast_sbm_warm_call,
    whole_domain_tile,
)
from gpuwm.core.fast_sbm.constants import FsbmBehavior, FsbmRefusal
from gpuwm.core.fast_sbm.derived import FsbmDerived, FsbmKernels, build_derived
from gpuwm.core.fast_sbm.tables import FsbmTables, load_tables
from tools.fast_sbm_wrf480_oracle import fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED = REPO_ROOT / K.PINNED_SOURCE_PATH
HARNESS_PINS = REPO_ROOT / "tools" / "fast_sbm_wrf480_oracle" / "sources.sha256"
WRF = FsbmBehavior.wrf_v480()
CORRECTED = FsbmBehavior.corrected()
DT = 12.0

COLD_FIELD_NAMES = ("snow_clip_key", "melt_step", "fusion_heat",
                    "homogeneous_freezing", "dsupice_w", "dsupice_wind",
                    "supsat_small_ratio", "fall_depth")
PRESET_VALUES = {
    "snow_clip_key": ("qs", "qi_plus_qs"),
    "melt_step": ("linear", "exponential"),
    "fusion_heat": ("wrf_v480", "al2_minus_al1"),
    "homogeneous_freezing": ("water_supersaturated", "any_saturation"),
    "dsupice_w": ("wrf_v480", "own_point"),
    "dsupice_wind": ("wrf_v480", "own_point"),
    "supsat_small_ratio": ("wrf_v480", "rederived"),
    "fall_depth": ("level_spacing", "level_spacing"),
}


@functools.lru_cache(maxsize=None)
def source() -> str:
    data = PINNED.read_bytes()
    assert hashlib.sha256(data).hexdigest() == K.PINNED_SOURCE_SHA256, (
        f"{PINNED} is not the pinned WRF v4.8.0 module_mp_fast_sbm.F")
    return data.decode("utf-8")


@functools.lru_cache(maxsize=None)
def source_lines() -> tuple[str, ...]:
    return tuple(source().split("\n"))


@functools.lru_cache(maxsize=None)
def parsed() -> tuple[K.ParsedRoutine, ...]:
    return tuple(K.parse_routines(source()))


@functools.lru_cache(maxsize=None)
def tables(precision: str = "float64") -> FsbmTables:
    return load_tables(precision)


# ---------------------------------------------------------------------------
# The pinned file
# ---------------------------------------------------------------------------

def test_the_pinned_source_in_the_tree_is_the_harness_pin():
    """The copy the map is checked against is the file the oracle builds:
    its LF hash is the harness pin and the design's, and it is LF."""
    data = PINNED.read_bytes()
    assert b"\r" not in data
    digest = hashlib.sha256(data).hexdigest()
    assert digest == K.PINNED_SOURCE_SHA256
    assert f"source {digest} - phys/module_mp_fast_sbm.F" in \
        HARNESS_PINS.read_text()
    assert data.count(b"\n") + (not data.endswith(b"\n")) == \
        K.PINNED_SOURCE_LINES


# ---------------------------------------------------------------------------
# The routine map
# ---------------------------------------------------------------------------

def test_every_routine_in_the_pinned_file_has_an_owner_or_a_reason():
    found = parsed()
    assert len(found) == 45
    assert K.routine_map_problems(found) == []
    assert K.routine_map_problems(found, text=source()) == []


# The owners of each routine's parts, in the map's order ("not ported" for a
# part with a reason).  A routine split by argument rather than by line
# (JERRATE_KS, JERTIMESC_KS, JERSUPSAT_KS, JERDFUN_KS) cannot lose its cold
# part to the line coverage check; it loses it here.
EXPECTED_PART_OWNERS = {
    "SBM_fast": ("not ported",),
    "Spontanous_Init": ("fsbm-derived-tables",),
    "Spont_Rain_BreakUp": ("fsbm-collision-breakup",),
    "BreakUp_Snow": ("not ported",),
    "coll_xyy_lwf": ("fsbm-cold-collision",),
    "coll_xxx_lwf": ("fsbm-collision-breakup",),
    "coll_xyx_lwf": ("fsbm-cold-collision",),
    "coll_xyz_lwf": ("fsbm-cold-collision",),
    "coll_xxy_lwf": ("not ported",),
    "INTERPOL_SE": ("fsbm-cold-collision",),
    "modkrn_KS": ("fsbm-collision-breakup", "fsbm-cold-collision", "not ported"),
    "coll_breakup_KS": ("fsbm-collision-breakup",),
    "courant_bott_KS": ("fsbm-derived-tables",),
    "POLYSVP": ("fsbm-condensation",),
    "JERRATE_KS": ("fsbm-condensation", "fsbm-deposition"),
    "JERTIMESC_KS": ("fsbm-condensation", "fsbm-deposition"),
    "JERSUPSAT_KS": ("fsbm-condensation", "fsbm-deposition"),
    "JERDFUN_KS": ("fsbm-condensation", "fsbm-deposition"),
    "JERNEWF_KS": ("fsbm-condensation",),
    "JERDFUN_NEW_KS": ("fsbm-condensation",),
    "Relaxation_Time": ("not ported",),
    "JERNUCL01_KS": ("fsbm-nucleation", "fsbm-ice-nucleation-phase",
                     "fsbm-ice-nucleation-phase"),
    "WATER_NUCLEATION": ("fsbm-nucleation",),
    "ICE_NUCL": ("fsbm-ice-nucleation-phase",),
    "Cloud_Base_Super": ("fsbm-nucleation",),
    "supmax_COEFF": ("fsbm-nucleation",),
    "LogNormal_modes_Aerosol": ("fsbm-derived-tables",),
    "FAST_SBM": ("fsbm-warm-column", "fsbm-cold-column", "fsbm-fall-exit",
                 "not ported", "not ported"),
    "FALFLUXHUCM_Z": ("fsbm-sedimentation", "fsbm-fall-exit"),
    "FAST_HUCMINIT": ("fsbm-bins-tables", "fsbm-derived-tables", "not ported"),
    "Kernals_KS": ("fsbm-derived-tables",),
    "ckern_z": ("fsbm-derived-tables",),
    "FREEZ": ("fsbm-ice-nucleation-phase",),
    "J_W_MELT": ("fsbm-ice-nucleation-phase",),
    "ONECOND1": ("fsbm-condensation",),
    "ONECOND2": ("fsbm-deposition",),
    "ONECOND3": ("fsbm-deposition",),
    "COAL_BOTT_NEW": ("fsbm-collision-breakup", "fsbm-cold-collision"),
    "BREAKINIT_KS": ("fsbm-derived-tables", "fsbm-bins-tables"),
    "ecoalmass": ("fsbm-derived-tables",),
    "ecoaldiam": ("fsbm-derived-tables",),
    "ecoalLowList": ("fsbm-derived-tables",),
    "ecoalOchs": ("fsbm-derived-tables",),
    "COLLENERGY": ("fsbm-derived-tables",),
    "vTBeard": ("fsbm-derived-tables",),
}


def test_each_routines_parts_have_the_owners_on_record():
    assert {r.name: tuple(p.owner or "not ported" for p in r.parts)
            for r in K.ROUTINES} == EXPECTED_PART_OWNERS
    assert len(EXPECTED_PART_OWNERS) == len(parsed())


def _drop_lines(name, owner, lines):
    # The map with the ranges ``lines`` removed from ``name``'s part owned by
    # ``owner`` (the whole part when it has no range left).
    out = []
    for r in K.ROUTINES:
        if r.name == name:
            parts = []
            for p in r.parts:
                if p.owner == owner:
                    kept = tuple(span for span in p.lines if span not in lines)
                    if not kept:
                        continue
                    p = dataclasses.replace(p, lines=kept)
                parts.append(p)
            r = dataclasses.replace(r, parts=tuple(parts))
        out.append(r)
    return tuple(out)


def _uncovered(problems, name) -> set[int]:
    # The code lines a problem list reports as in no part, for ``name``.
    out: set[int] = set()
    for problem in problems:
        prefix = f"{name}: code lines "
        if problem.startswith(prefix) and problem.endswith(" lie in no part"):
            for span in problem[len(prefix):-len(" lie in no part")].split(", "):
                first, _, last = span.partition("-")
                out.update(range(int(first), int(last or first) + 1))
    return out


@pytest.mark.parametrize("name,owner,lines", [
    ("COAL_BOTT_NEW", "fsbm-cold-collision", ((8437, 8570), (8572, 8602), (8603, 8640))),
    ("FAST_SBM", "fsbm-cold-column", ((4724, 4763),)),
    ("FAST_SBM", "fsbm-cold-column", ((4667, 4674),)),
    ("FAST_SBM", "fsbm-fall-exit", ((5072, 5121),)),
    ("JERNUCL01_KS", "fsbm-ice-nucleation-phase", ((3100, 3111),)),
    ("modkrn_KS", "fsbm-cold-collision", ((1395, 1420), (1481, 1503))),
])
def test_a_part_dropped_from_a_routine_that_keeps_others_fails(name, owner, lines):
    """Negative control of the line coverage: each of these routines keeps
    another owned part, so only the coverage check can see the loss, and it
    names exactly the dropped code lines no other part covers."""
    mutated = _drop_lines(name, owner, lines)
    assert K.routine_map_problems(parsed(), mutated) == []
    problems = K.routine_map_problems(parsed(), mutated, text=source())
    routine = next(r for r in mutated if r.name == name)
    covered = {n for p in routine.parts for a, b in p.lines for n in range(a, b + 1)}
    dropped = {n for a, b in lines for n in K.code_lines(source(), a, b)} - covered
    assert dropped and _uncovered(problems, name) == dropped, problems


@pytest.mark.parametrize("name,index", [("JERRATE_KS", 1), ("JERDFUN_KS", 1),
                                        ("FREEZ", 0), ("FAST_SBM", 1)])
def test_a_part_that_loses_its_owner_to_a_reason_fails(name, index):
    """Negative control of the literal table: the map with one owner turned
    into a reason still passes the map checker, and the table catches it."""
    routines = []
    for r in K.ROUTINES:
        if r.name == name:
            parts = list(r.parts)
            parts[index] = dataclasses.replace(parts[index], owner=None, host=None,
                                               reason="not ported")
            r = dataclasses.replace(r, parts=tuple(parts))
        routines.append(r)
    assert K.routine_map_problems(parsed(), routines, text=source()) == []
    owners = {r.name: tuple(p.owner or "not ported" for p in r.parts)
              for r in routines}
    assert owners != EXPECTED_PART_OWNERS


def test_the_old_ports_routine_list_is_the_parsed_one_less_three():
    """The 09-13/14 port transcribed 42 routines; the file has 45.  The
    three it skipped are the preprocessor stub and the two init routines
    milestone 1 transcribed (FAST_HUCMINIT, BREAKINIT_KS)."""
    old = {"spontanous_init", "spont_rain_breakup", "breakup_snow",
           "coll_xyy_lwf", "coll_xxx_lwf", "coll_xyx_lwf", "coll_xyz_lwf",
           "coll_xxy_lwf", "interpol_se", "modkrn_ks", "coll_breakup_ks",
           "courant_bott_ks", "polysvp", "jerrate_ks", "jertimesc_ks",
           "jersupsat_ks", "jerdfun_ks", "jernewf_ks", "jerdfun_new_ks",
           "relaxation_time", "jernucl01_ks", "water_nucleation", "ice_nucl",
           "cloud_base_super", "supmax_coeff", "lognormal_modes_aerosol",
           "fast_sbm", "falfluxhucm_z", "kernals_ks", "ckern_z", "freez",
           "j_w_melt", "onecond1", "onecond2", "onecond3", "coal_bott_new",
           "ecoalmass", "ecoaldiam", "ecoallowlist", "ecoalochs",
           "collenergy", "vtbeard"}
    names = {r.name.lower() for r in parsed()}
    assert names - old == {"sbm_fast", "fast_hucminit", "breakinit_ks"}
    assert old <= names


def _replace_routine(name, **changes):
    return tuple(dataclasses.replace(r, **changes) if r.name == name else r
                 for r in K.ROUTINES)


def _replace_first_part(name, **changes):
    out = []
    for r in K.ROUTINES:
        if r.name == name:
            parts = (dataclasses.replace(r.parts[0], **changes),) + r.parts[1:]
            r = dataclasses.replace(r, parts=parts)
        out.append(r)
    return tuple(out)


@pytest.mark.parametrize("mutation,expect", [
    (lambda: tuple(r for r in K.ROUTINES if r.name != "FREEZ"),
     "FREEZ (:6560-6636) has neither an owner nor a reason"),
    (lambda: _replace_first_part("ICE_NUCL", owner=None),
     "needs exactly one of an owner and a reason"),
    (lambda: _replace_first_part("BreakUp_Snow", reason=None),
     "needs exactly one of an owner and a reason"),
    (lambda: _replace_first_part("J_W_MELT", reason="dead"),
     "needs exactly one of an owner and a reason"),
    (lambda: _replace_first_part("ONECOND2", owner="fsbm-ice"),
     "is no item"),
    (lambda: _replace_first_part("ONECOND1", host=None),
     "names no host"),
    (lambda: _replace_routine("ONECOND3", lines=(7716, 8260)),
     "lines (7716, 8260)"),
    (lambda: _replace_first_part("FREEZ", lines=((6560, 6640),)),
     "lie outside"),
    (lambda: _replace_routine("FREEZ", parts=()),
     "no part has an owner or a reason"),
    (lambda: _replace_routine("ICE_NUCL", module="module_mp_fast_sbm"),
     "module"),
    (lambda: K.ROUTINES + (dataclasses.replace(K.ROUTINES[0],
                                               name="ICE_NUCL_2"),),
     "in the map but not in the file"),
    (lambda: _replace_first_part("coll_xxy_lwf", reason="  "),
     "the reason is empty"),
])
def test_a_map_that_leaves_a_routine_unowned_fails(mutation, expect):
    problems = K.routine_map_problems(parsed(), mutation())
    assert any(expect in p for p in problems), problems


def test_every_milestone_1_part_names_a_host_that_exists():
    for routine in K.ROUTINES:
        for part in routine.parts:
            if part.host is None:
                assert part.owner not in K.M1_ITEMS, routine.name
                continue
            module, name = part.host.rsplit(".", 1)
            found = importlib.import_module(f"gpuwm.core.fast_sbm.{module}")
            assert callable(getattr(found, name)), part.host


def test_every_process_item_of_milestone_2_owns_a_part():
    owners = {p.owner for r in K.ROUTINES for p in r.parts}
    for item in ("fsbm-ice-nucleation-phase", "fsbm-deposition",
                 "fsbm-cold-collision", "fsbm-fall-exit", "fsbm-cold-column"):
        assert item in owners, item
    assert owners - {None} <= set(K.ITEMS)


def test_the_parser_reads_every_header_form_and_nothing_else():
    text = "\n".join([
        "module outer",                                 # 1
        "contains",                                     # 2
        "\tSUBROUTINE A_one (x, &",                     # 3
        "   & y)",                                      # 4
        "  ! subroutine commented_out(x)",              # 5
        "  CALL B_two(x)",                              # 6
        "  END SUBROUTINE A_one",                       # 7
        " double precision FUNCTION B_two(x)",          # 8
        " end function B_two",                          # 9
        "  real function c3(y)",                        # 10
        "  End Function c3",                            # 11
        "end module outer",                             # 12
        "   subroutine loose",                          # 13
        "   end subroutine loose",                      # 14
    ])
    assert K.parse_routines(text) == [
        K.ParsedRoutine("A_one", "subroutine", "outer", 3, 7),
        K.ParsedRoutine("B_two", "function", "outer", 8, 9),
        K.ParsedRoutine("c3", "function", "outer", 10, 11),
        K.ParsedRoutine("loose", "subroutine", None, 13, 14),
    ]
    with pytest.raises(ValueError, match="has no END"):
        K.parse_routines("subroutine x\n")
    with pytest.raises(ValueError, match="closes no routine"):
        K.parse_routines("end subroutine x\n")
    with pytest.raises(ValueError, match="opens inside"):
        K.parse_routines("subroutine x\nsubroutine y\n")


def test_the_cold_routines_read_only_the_tables_recorded():
    """The old port's table needs, checked against the file: FREEZ,
    J_W_MELT and ICE_NUCL read no module variable, ONECOND1 to 3 read only
    FR_LIM and FRH_LIM, and COAL_BOTT_NEW's reads are all loaded and gated
    by milestone 1."""
    recorded = [r for r in K.ROUTINES if r.module_reads is not None]
    assert {r.name for r in recorded} >= {"FREEZ", "J_W_MELT", "ICE_NUCL",
                                          "ONECOND2", "ONECOND3",
                                          "COAL_BOTT_NEW"}
    for routine in recorded:
        assert K.module_variable_reads(source(), routine.name) == \
            routine.module_reads, routine.name
    coal = next(r for r in K.ROUTINES if r.name == "COAL_BOTT_NEW")
    assert set(coal.module_reads) <= _carried()
    # The detector can find a read: FAST_SBM reads the fall-speed tables.
    assert "vr1" in K.module_variable_reads(source(), "FAST_SBM")


def _carried() -> set[str]:
    return ({f.name for f in dataclasses.fields(FsbmTables)}
            | {f.name for f in dataclasses.fields(FsbmDerived)}
            | {f.name for f in dataclasses.fields(FsbmKernels)})


def test_fast_sbm_itself_reads_no_table_milestone_1_does_not_carry():
    """The cold parts of FAST_SBM (the pre-pass, the dispatch, the ice fall)
    are covered too: every module variable FAST_SBM mentions is a carried
    table or one of the named working and polarimetric arrays, each cited
    at a line of FAST_SBM that mentions it.  A new table read anywhere in
    FAST_SBM fails, as the negative control shows."""
    uncarried = set(K.module_variable_reads(source(), "FAST_SBM")) - _carried()
    assert uncarried == set(K.FAST_SBM_UNCARRIED_READS)
    fast_sbm = next(r for r in parsed() if r.name == "FAST_SBM")
    lines = source_lines()
    for name, (line, what) in K.FAST_SBM_UNCARRIED_READS.items():
        assert fast_sbm.first <= line <= fast_sbm.last, name
        assert name in lines[line - 1].lower(), (name, line)
        assert what.strip(), name
    # Negative control: the pre-pass reading a table nothing carries
    # (RADXX, the polarimetric radii) shows up in the difference.
    edited = list(lines)
    assert edited[4401 - 1].strip() == "SUPICE(K)=EW1N/ES2N-1."
    edited[4401 - 1] = "              SUPICE(K)=EW1N/ES2N-1.+RADXX(1,1)"
    reads = set(K.module_variable_reads("\n".join(edited), "FAST_SBM"))
    assert reads - _carried() == set(K.FAST_SBM_UNCARRIED_READS) | {"radxx"}


def test_nothing_assigns_fr_lim():
    """FR_LIM and FRH_LIM are never set (dead path fr_lim); the detector
    finds a real module assignment in FAST_HUCMINIT (xl_mg, :6078)."""
    for routine in parsed():
        targets = K.assignment_targets(source(), routine.name)
        assert not targets & {"fr_lim", "frh_lim"}, routine.name
    assert "xl_mg" in K.assignment_targets(source(), "FAST_HUCMINIT")


def test_freez_and_j_w_melt_take_the_mass_grids_as_tables():
    """J_W_MELT declares the grids INTENT(INOUT) and neither routine
    assigns them, so the host passes the loaded tables (the old port's
    fix 2).  The detector sees the spectra they do assign."""
    for name in K.GRIDS_ARE_TABLES:
        targets = K.assignment_targets(source(), name)
        assert not targets & set(K.GRID_DUMMIES), (name, targets)
        assert "ff1" in targets
    melt = next(r for r in parsed() if r.name == "J_W_MELT")
    inout = set()
    for _, statement in K.statements(source(), melt.first, melt.last):
        found = K.declaration(statement)
        if found and "intent(inout)" in statement.lower().replace(" ", ""):
            inout.update(found[0])
    assert set(K.GRID_DUMMIES) <= inout, inout


# ---------------------------------------------------------------------------
# Findings and dead paths
# ---------------------------------------------------------------------------

def test_every_catalogue_citation_is_on_the_pinned_file():
    lines = source_lines()
    ids = [f.id for f in K.FINDINGS]
    keys = [d.key for d in K.DEAD_PATHS]
    assert len(set(ids)) == len(ids) and len(set(keys)) == len(keys)
    assert set(ids) == {f"D{n}" for n in range(1, 19)} | {"S1", "S2", "S3"}
    for record in K.FINDINGS + K.DEAD_PATHS:
        for first, last in record.lines:
            assert 1 <= first <= last <= K.PINNED_SOURCE_LINES, record
        first, last = record.lines[0]
        assert any(record.anchor in lines[n - 1]
                   for n in range(first, last + 1)), record
    for finding in K.FINDINGS:
        assert finding.owner in K.ITEMS, finding.id


def test_a_drifted_citation_fails_the_anchor_check():
    lines = source_lines()
    d11 = next(f for f in K.FINDINGS if f.id == "D11")
    first, _ = d11.lines[0]
    assert d11.anchor in lines[first - 1]
    assert d11.anchor not in lines[first]


def test_every_behaviour_field_names_a_finding_and_back():
    fields = {f.name for f in dataclasses.fields(FsbmBehavior)}
    selected = {f.field for f in K.FINDINGS if f.field is not None}
    assert selected == fields
    for name, ids in C.COLD_FIELDS.items():
        assert {f.id for f in K.FINDINGS if f.field == name} == set(ids), name


def _compact(line: str) -> str:
    return line.replace(" ", "").replace("\t", "").upper()


def test_d15_w_stag_is_one_scalar_carried_in_j_i_k_order():
    """The claim item 7 implements ``dsupice_w="wrf_v480"`` from: w_stag is
    a scalar local (so a cold point reads whatever the last warm-enough
    point left), and the pre-pass visits points in (j, i, k) order."""
    lines = source_lines()
    declared = _compact(lines[4158 - 1])
    assert ",W_STAG," in declared and "W_STAG(" not in declared
    assert _compact(lines[4374 - 1]) == "DOJ=JTS,JTE"
    assert _compact(lines[4375 - 1]) == "DOI=ITS,ITE"
    assert _compact(lines[4404 - 1]) == "DOK=KTS,KTE"
    assigned = [n for n in range(4372, 4513)
                if _compact(lines[n - 1]).startswith("W_STAG=")]
    assert assigned == [4407, 4409]
    d15 = next(f for f in K.FINDINGS if f.id == "D15")
    assert "(j, i, k)" in d15.consequence and "most recent" in \
        d15.consequence


def test_d17_the_pre_pass_averages_mass_point_winds_in_both_copies():
    """Both copies of the wind (:4411-4417 for the dead DTFREEZ_XYZ,
    :4497-4503 for DSUPICE_XYZ) take the four-point average away from the
    last column and row and the point's own wind on it; the behaviour field
    selects WRF's form or the point's own, and the presets differ."""
    lines = source_lines()
    d17 = next(f for f in K.FINDINGS if f.id == "D17")
    assert d17.field == "dsupice_wind" and d17.owner == "fsbm-cold-column"
    for first, last in d17.lines:
        text = [_compact(lines[n - 1]) for n in range(first, last + 1)]
        assert text[0] == "IF(I.LT.IDE-1.AND.J.LT.JDE-1)THEN"
        for n, (w, a) in enumerate((("UX", "U"), ("VX", "V")), start=1):
            assert text[n] == (f"{w}=25.*({a}(I,K,J)+{a}(I+1,K,J)"
                               f"+{a}(I,K,J+1)+{a}(I+1,K,J+1))"), (first, n)
        assert text[4:6] == ["UX=U(I,K,J)*100.", "VX=V(I,K,J)*100."]
    assert _compact(lines[4504 - 1]).startswith(
        "DSUPICE_XYZ(I,K,J)=(UX*DERIVS_X")
    assert (WRF.dsupice_wind, CORRECTED.dsupice_wind) == ("wrf_v480",
                                                          "own_point")


def test_d18_a_tiles_halo_depends_on_the_order_of_the_tiles():
    """The loops D18 rests on, read off the file: the halo bounds are one
    column beyond the tile and inside the domain; the itimestep 1 reset and
    T_OLD run over them, so they read (and at step 1 write) the neighbours'
    th_old and qv_old; the exit writes th_old and qv_old on the tile's own
    columns only, which is what a later tile's halo then sees.  The port
    defines the halo as the neighbours' pre-call state (the MPI halo
    exchange's answer), in both presets, so D18 has no behaviour field."""
    lines = source_lines()
    assert [_compact(lines[n - 1]) for n in range(4262, 4266)] == [
        "I_START=MAX(1,ITS-1)", "J_START=MAX(1,JTS-1)",
        "I_END=MIN(IDE-1,ITE+1)", "J_END=MIN(JDE-1,JTE+1)"]
    assert _compact(lines[4351 - 1]) == "IF(ITIMESTEP==1)THEN"
    for first in (4352, 4362):
        assert [_compact(lines[n - 1]) for n in range(first, first + 3)] == [
            "DOJ=J_START,J_END", "DOK=KTS,KTE", "DOI=I_START,I_END"], first
    assert _compact(lines[4355 - 1]) == "TH_OLD(I,K,J)=TH_PHY(I,K,J)"
    assert _compact(lines[4356 - 1]) == "QV_OLD(I,K,J)=QV(I,K,J)"
    assert _compact(lines[4367 - 1]) == "T_OLD(I,K,J)=TH_OLD(I,K,J)*PI_PHY(I,K,J)"
    assert [_compact(lines[n - 1]) for n in range(5267, 5272)] == [
        "DOJ=JTS,JTE", "DOK=KTS,KTE", "DOI=ITS,ITE",
        "TH_OLD(I,K,J)=TH_PHY(I,K,J)", "QV_OLD(I,K,J)=QV(I,K,J)"]
    d18 = next(f for f in K.FINDINGS if f.id == "D18")
    assert d18.field is None and d18.owner == "fsbm-cold-column"
    assert d18.lines[0] == (4351, 4360) and (5267, 5274) in d18.lines
    assert "before the call" in d18.port
    assert "D18" in FsbmTile.__doc__ and "before this call" in FsbmTile.__doc__


# ---------------------------------------------------------------------------
# The behaviour fields
# ---------------------------------------------------------------------------

def test_the_cold_fields_are_the_contract():
    assert tuple(C.COLD_FIELDS) == COLD_FIELD_NAMES
    by_name = {f.name: f for f in dataclasses.fields(FsbmBehavior)}
    for name in COLD_FIELD_NAMES:
        assert by_name[name].kw_only, name
        allowed = PRESET_VALUES[name]
        assert by_name[name].default not in allowed, name


@pytest.mark.parametrize("name", COLD_FIELD_NAMES)
def test_no_cold_field_has_a_usable_default(name):
    fields = dataclasses.asdict(CORRECTED)
    del fields[name]
    with pytest.raises(TypeError, match=repr(name)):
        FsbmBehavior(**fields)


def test_both_presets_set_every_cold_field():
    for name, (wrf, corrected) in PRESET_VALUES.items():
        assert getattr(WRF, name) == wrf, name
        assert getattr(CORRECTED, name) == corrected, name
    # An arm made from a preset keeps every other field of that preset.
    arm = dataclasses.replace(WRF, melt_step="exponential")
    assert (arm.fusion_heat, arm.snow_clip_key) == ("wrf_v480", "qs")
    assert FsbmBehavior(**dataclasses.asdict(CORRECTED)) == CORRECTED


@pytest.mark.parametrize("name", COLD_FIELD_NAMES)
@pytest.mark.parametrize("bad", ["", "WRF_V480", None, 1])
def test_a_malformed_cold_value_is_refused(name, bad):
    with pytest.raises(FsbmRefusal, match=name):
        dataclasses.replace(CORRECTED, **{name: bad})


def test_milestone_1_fields_are_still_checked_first():
    seven = dict(rainnc_velocity="own_column", rainnc_source="bottom_outflow",
                 nucleation_water_sink=True, dtcond_carry=False,
                 breakup_retry_cap=64, cloud_base_rule="first_level",
                 fall_speed_lookup="own_bin")
    with pytest.raises(FsbmRefusal, match="rainnc_velocity"):
        FsbmBehavior(**{**seven, "rainnc_velocity": "last"})
    with pytest.raises(TypeError, match="8 required keyword arguments"):
        FsbmBehavior(**seven)


def test_fusion_heat_is_wrfs_literal_per_site_or_al2_minus_al1():
    for site, value in C.FUSION_HEAT_WRF.items():
        assert C.fusion_heat(WRF, site) == value
        assert C.fusion_heat(CORRECTED, site) == C.AL2 - C.AL1 == 334.0
        line = source_lines()[C.FUSION_HEAT_LINES[site] - 1]
        literal = {333.0: "333.", 320.0: "320.", 334.0: "334.0D0"}[value]
        assert f"{literal}*" in line.replace(" ", ""), (site, line)
    with pytest.raises(FsbmRefusal, match="site"):
        C.fusion_heat(WRF, "sublimation")
    with pytest.raises(FsbmRefusal, match="FsbmBehavior"):
        C.fusion_heat("wrf_v480", "freez")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: (name, value, line, the literal as WRF writes it there)
COLD_CONSTANTS = [
    ("KRFREEZ", 21, 4092, "TCRIT/21"),
    ("BFREEZMAX", 0.66, 4093, "0.6600E00"),
    ("TTCOAL", 233.15, 4097, "2.3315E02"),
    ("AFREEZMY", 3.333e-5, 4097, "0.3333E-04"),
    ("BFREEZMY", 0.66, 4097, "2.3315E02,0.3333E-04,0.6600E00"),
    ("TCRIT", 270.15, 4099, "2.7015E02"),
    ("ALCR", 0.5, 3907, "ALCR = 0.5"),
    ("ALCR_G", 100.0, 3908, "ALCR_G = 100.0"),
    ("KR_ICEMPL", 9, 3910, "kr_icempl=9"),
    ("KRICE", 18, 3900, "KRICE=18"),
    ("ICEMPL", 1, 3894, "icempl=1"),
    ("IBREAKUP", 1, 3880, "IBREAKUP = 1"),
    ("SNOW_BREAKUP_ON", 1, 3881, "Snow_BreakUp_On = 1"),
    ("SPONT_RAIN_BREAKUP_ON", 1, 3882, "Spont_Rain_BreakUp_On = 1"),
    ("JIWEN_FAN_MELT", 1, 3884, "JIWEN_FAN_MELT = 1"),
    ("HAIL_OPT_DEFAULT", 1, 3886, "hail_opt = 1"),
    ("ICEPROCS_DEFAULT", 1, 3891, "ICEPROCS = 1"),
    ("KR_SNOW_MAX", 35, 110, "KR_SNOW_MAX = 35"),
    ("KR_SNOW_MIN", 34, 111, "KR_SNOW_MIN = 34"),
    ("USE_CLOUD_BASE_NUC", 1, 3024, "Use_cloud_base_nuc = 1"),
    ("T_NUCL_DROP_MIN", -80.0, 3025, "T_NUCL_DROP_MIN = -80.0D0"),
    ("T_NUCL_ICE_MIN", -37.0, 3026, "T_NUCL_ICE_MIN = -37.0D0"),
    ("ICE_NUCL_METHOD", 0, 3030, "ice_nucl_method = 0"),
    ("ISIGN_TQ_ICENUCL", 1, 3031, "ISIGN_TQ_ICENUCL = 1"),
    ("DELSUPICE_MAX", 59.0, 3033, "DELSUPICE_MAX = 59.0D0"),
    ("T_HOMOGENEOUS_FREEZING", -38.0, 3101, "TPC <= -38.0D0"),
]


@pytest.mark.parametrize("name,value,line,literal", COLD_CONSTANTS)
def test_the_cold_constants_are_wrfs_literals_at_their_lines(name, value, line,
                                                              literal):
    assert getattr(C, name) == value
    assert type(getattr(C, name)) is type(value)
    assert literal in source_lines()[line - 1], (name, line)


def test_a_wrong_literal_is_not_found_at_the_cited_line():
    assert "0.3334E-04" not in source_lines()[4097 - 1]
    assert "0.3333E-04" not in source_lines()[4096 - 1]


# ---------------------------------------------------------------------------
# The dense-ice class
# ---------------------------------------------------------------------------

def test_the_dense_ice_class_is_a_named_option():
    hail, graupel = C.dense_ice("hail"), C.dense_ice("graupel")
    assert C.dense_ice() is hail and C.DENSE_ICE_DEFAULT == "hail"
    assert hail.hail_opt == C.HAIL_OPT_DEFAULT == 1 and graupel.hail_opt == 0
    assert hail.identity != graupel.identity
    assert "hail" in hail.identity and "graupel" in graupel.identity
    table_fields = {f.name for f in dataclasses.fields(FsbmTables)}
    kernel_fields = {f.name for f in dataclasses.fields(FsbmKernels)}
    for option in (hail, graupel):
        for name in (option.mass_grid, option.fall_speed, option.capacity,
                     option.bulk_density):
            assert name in table_fields, (option.name, name)
        assert set(option.kernels) <= kernel_fields
        assert option.distribution in {
            f.name for f in dataclasses.fields(S.FsbmDistributions)}
    assert {k for o in (hail, graupel) for k in o.kernels} == {
        spec[3] for pair, spec in TALLY_PAIRS.items()
        if "graupel" in pair or "hail" in pair}


def _hail_opt_branches(first, last, pattern):
    # The names ``pattern`` matches in the ``hail_opt == 1`` branch and in
    # the ``else`` branch of the IF block inside lines first..last.
    lines = source_lines()
    branch, found = None, {True: set(), False: set()}
    for n in range(first, last + 1):
        code = _compact(K.strip_comment(lines[n - 1]))
        if code.startswith("IF(HAIL_OPT==1)"):
            branch = True
            continue
        if code == "ELSE" and branch:
            branch = False
            continue
        if code == "ENDIF" and branch is not None:
            break
        if branch is not None:
            found[branch] |= {m.lower() for m in re.findall(pattern, code)}
    return found[True], found[False]


def _dense_ice_problems(option) -> list[str]:
    """Where ``option``'s WRF array names differ from the pinned file."""
    hail = option.hail_opt == 1
    pick = (lambda both: both[0]) if hail else (lambda both: both[1])
    problems = []
    for what, first, last, pattern, got in (
            ("mass_grid", 4236, 4252, r"/(X[A-Z])\(KRR\)", option.mass_grid),
            ("distribution", 4558, 4578, r"^(FF\dR)\(KRR\)=CHEM_NEW", option.distribution),
            ("fall_speed", 4936, 4948, r"=(VR\d)_Z3D", option.fall_speed)):
        names = pick(_hail_opt_branches(first, last, pattern))
        if names != {got}:
            problems.append(f"{what}: the file has {sorted(names)}, the option {got!r}")
    # ONECOND2 and ONECOND3 take each dense slot as (spectrum, spectrum in,
    # mass grid, capacity, bulk density); the option's slot is the group of
    # its spectrum.
    lines = source_lines()
    groups = set()
    for n in range(4730, 4760):
        code = _compact(K.strip_comment(lines[n - 1]))
        parts = [a.lower() for a in code.strip(",&").split(",")]
        if len(parts) == 5 and parts[0] == option.distribution:
            groups.add(tuple(parts))
    want = (option.distribution, option.distribution[:-1] + "in",
            option.mass_grid, option.capacity, option.bulk_density)
    if groups != {want}:
        problems.append(f"onecond slot: the file has {sorted(groups)}, the option {want}")
    # The cold block's drop collisions with this class: (g1, g?) is drops
    # collected by it, (g?, g1) it collecting drops.
    g = "g5" if hail else "g4"
    coal = next(r for r in parsed() if r.name == "COAL_BOTT_NEW")
    call = re.compile(r"call\s+coll_x[xy][xyz]_lwf\s*\(([^)&]*)", re.I)
    by_order = {"drop_first": set(), "drop_second": set()}
    for n in range(coal.first, coal.last + 1):
        found = call.search(lines[n - 1])
        if not found:
            continue
        args = [a.strip().lower() for a in found.group(1).split(",")]
        kernel = next(a for a in args if a.startswith("cw"))
        if args[:2] == ["g1", g]:
            by_order["drop_first"].add(kernel)
        elif args[:2] == [g, "g1"]:
            by_order["drop_second"].add(kernel)
    file_kernels = (by_order["drop_first"], by_order["drop_second"])
    if file_kernels != ({option.kernels[0]}, {option.kernels[1]}):
        problems.append(f"kernels: the file has {file_kernels}, the option {option.kernels}")
    pairs = (TALLY_PAIRS[f"drop_{option.name}"][3], TALLY_PAIRS[f"{option.name}_drop"][3])
    if pairs != option.kernels:
        problems.append(f"kernels: the tally pairs say {pairs}, the option {option.kernels}")
    return problems


@pytest.mark.parametrize("name", ["hail", "graupel"])
def test_the_dense_ice_names_are_the_files_for_its_option(name):
    """Read off the pinned file: the hail_opt branches of the entry
    conversion (:4239-4252), the spectrum slot (:4561-4577) and the fall
    speeds (:4941-4945); the dense slot of the ONECOND2 and ONECOND3 calls
    (:4735-4736, :4754-4755); and the kernels of the drop collisions with the
    class in COAL_BOTT_NEW's cold block (:8476-8537), in the order the tally
    pairs use."""
    assert _dense_ice_problems(C.dense_ice(name)) == []


@pytest.mark.parametrize("field,value", [
    ("mass_grid", "xg"), ("distribution", "ff4r"), ("fall_speed", "vr4"),
    ("capacity", "rgec"), ("bulk_density", "ro4bl"),
    ("kernels", ("cwlg", "cwgl")), ("kernels", ("cwhl", "cwlh"))])
def test_a_dense_ice_option_with_the_other_classs_names_fails(field, value):
    """Negative control: hail given one of graupel's names, or its kernels
    swapped, is caught by the check above."""
    swapped = dataclasses.replace(C.dense_ice("hail"), **{field: value})
    assert _dense_ice_problems(swapped) != []


@pytest.mark.parametrize("name", [1, 0, True, "Hail", "", None, "graupel "])
def test_a_dense_ice_class_not_named_is_refused(name):
    with pytest.raises(FsbmRefusal, match="dense_ice"):
        C.dense_ice(name)


@pytest.mark.parametrize("option", ["hail", "graupel"])
def test_the_option_names_where_the_state_layer_puts_the_dense_bins(option):
    chosen = C.dense_ice(option)
    nz = 2
    ff6 = np.full((nz, 33), 1.0e-6)
    bins = S.FsbmColumnBins(ff1=np.zeros((nz, 33)), ff5=np.zeros((nz, 33)),
                            ff6=ff6, ff8=np.zeros((nz, 43)))
    dist = S.bins_to_distributions(bins, np.full(nz, 1.0e-3), tables(),
                                   chosen.hail_opt)
    assert np.all(getattr(dist, chosen.distribution) > 0.0)
    other = "ff4r" if chosen.distribution == "ff5r" else "ff5r"
    assert not np.any(getattr(dist, other))


# ---------------------------------------------------------------------------
# The mass-grid identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_ice_mass_grids_are_the_drop_grid(precision):
    assert K.mass_grid_mismatches(tables(precision)) == ()


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_wrfs_own_reads_hold_the_identity(build):
    dump = fx.load_fixture(f"tables_{build}")
    xl = dump["read.xl"][0]
    for name in ("xs", "xg", "xh"):
        assert np.array_equal(dump[f"read.{name}"][0], xl), name
    xi = dump["read.xi"][0]
    for habit in range(3):
        assert np.array_equal(xi[:, habit], xl), habit


def test_the_derived_mg_grids_hold_the_identity():
    derived = build_derived(tables(), DT, behavior=WRF)
    for name in ("xs_mg", "xg_mg", "xh_mg"):
        assert np.array_equal(getattr(derived, name), derived.xl_mg), name
    for habit in range(3):
        assert np.array_equal(derived.xi_mg[:, habit], derived.xl_mg)


@pytest.mark.parametrize("grid,expect", [
    ("xs", ("xs",)), ("xg", ("xg",)), ("xh", ("xh",)), ("xi", ("xi[:, 1]",))])
def test_a_grid_moved_by_one_ulp_fails_the_identity(grid, expect):
    """Negative control: one bin one ULP off, and the check names it."""
    base = tables()
    moved = np.array(getattr(base, grid), copy=True)
    if grid == "xi":
        moved[20, 1] = np.nextafter(moved[20, 1], np.inf)
    else:
        moved[20] = np.nextafter(moved[20], np.inf)
    assert K.mass_grid_mismatches(
        dataclasses.replace(base, **{grid: moved})) == expect


def test_demonstration_a_merge_conserves_mass_only_on_equal_grids():
    """A demonstration, not a gate: why the identity above matters.  WRF
    adds crystal number density to snow bin by bin (:4811).  On equal grids
    the snow mass after is the two masses before; on a grid one percent off
    in one bin it is not.  The arithmetic calls no port code, so it can
    fail only when the tables change, which
    ``test_the_ice_mass_grids_are_the_drop_grid`` already gates.

    The last assertion is a trip-wire: the cold-column item owns the merges
    (``MASS_GRID_MERGES``), and when it names the host of its part of
    FAST_SBM this test fails until it is retargeted to call that host, so
    it gates the real merges."""
    rng = np.random.default_rng(20260924)
    f_snow = 10.0 ** rng.uniform(-8.0, -2.0, 33)
    f_crystal = 10.0 ** rng.uniform(-8.0, -2.0, 33)

    def mass(f, x):
        return float(np.sum(3.0 * C.COL * x * x * f))

    xs = np.asarray(tables().xs)
    xi = np.asarray(tables().xi)[:, 1]
    before = mass(f_snow, xs) + mass(f_crystal, xi)
    after = mass(f_snow + f_crystal, xs)
    assert abs(after - before) <= 1e-14 * before
    xi_off = xi.copy()
    xi_off[25] *= 1.01
    before_off = mass(f_snow, xs) + mass(f_crystal, xi_off)
    assert abs(after - before_off) > 1e-6 * before_off
    fast_sbm = next(r for r in K.ROUTINES if r.name == "FAST_SBM")
    merges = [p for p in fast_sbm.parts if p.owner == "fsbm-cold-column"]
    assert len(merges) == 1 and all(
        any(a <= n <= b for a, b in merges[0].lines)
        for n in K.MASS_GRID_MERGES)
    assert merges[0].host is None, (
        f"the merge host {merges[0].host} exists: retarget this test to "
        f"call it at MASS_GRID_MERGES, so it gates the real merges")


# ---------------------------------------------------------------------------
# The tile
# ---------------------------------------------------------------------------

def _synthetic_tile(ni: int = 3, nj: int = 2, nz: int = 5) -> FsbmTile:
    rng = np.random.default_rng(ni * 100 + nj * 10 + nz)
    ncol = ni * nj
    th = 280.0 + rng.uniform(0.0, 5.0, (ncol, nz))
    return FsbmTile(
        th=th, qv=rng.uniform(1e-3, 5e-3, (ncol, nz)), th_old=th - 0.1,
        qv_old=rng.uniform(1e-3, 5e-3, (ncol, nz)),
        p=np.linspace(9.0e4, 5.0e4, nz)[None, :] + np.zeros((ncol, 1)),
        pii=np.linspace(0.97, 0.82, nz)[None, :] + np.zeros((ncol, 1)),
        rho=np.full((ncol, nz), 1.0), dz=np.full((ncol, nz), 200.0),
        w=np.zeros((ncol, nz + 1)), ff1=np.zeros((ncol, nz, 33)),
        ff5=np.zeros((ncol, nz, 33)), ff6=np.zeros((ncol, nz, 33)),
        ff8=np.zeros((ncol, nz, 43)), xland=np.ones(ncol),
        rainnc=np.zeros(ncol), ni=ni)


def _cold_tile(ni: int = 3, nj: int = 2, nz: int = 5) -> FsbmTile:
    tile = _synthetic_tile(ni, nj, nz)
    rng = np.random.default_rng(7)
    return whole_domain_tile(tile, dx=1000.0, dy=1500.0,
                             u=rng.uniform(-5, 5, (ni * nj, nz)),
                             v=rng.uniform(-5, 5, (ni * nj, nz)))


def test_the_tile_gains_the_cold_inputs_unsupplied_by_default():
    names = {f.name for f in dataclasses.fields(FsbmTile)}
    assert set(COLD_TILE_FIELDS) <= names
    assert set(COLD_TILE_FIELDS) == {
        "u", "v", "dx", "dy", "ide", "jde", "its", "jts", "halo_fill",
        "snownc", "graupelnc", *(f"halo_{n}" for n in HALO_FIELDS)}
    assert HALO_FIELDS == ("th", "qv", "th_old", "qv_old", "p", "pii")
    tile = _synthetic_tile()
    for name in COLD_TILE_FIELDS:
        value = getattr(tile, name)
        assert np.size(value) == 0 or value == 0, name
    assert tile.halo_fill == HALO_NONE and tile.halo_fill_name == "none"
    with pytest.raises(FsbmRefusal, match="tile.u is not supplied"):
        check_cold_inputs(tile)


def test_every_tile_field_is_numeric_so_tiles_compare_field_by_field():
    """Milestone 1's refusal tests snapshot a tile field by field with
    ``np.array_equal(..., equal_nan=True)``, which a None or a string
    field breaks; every field, supplied or not, stays numeric."""
    for tile in (_synthetic_tile(), _cold_tile()):
        for name in tile.__dataclass_fields__:
            if name == "ni":
                continue
            value = getattr(tile, name)
            assert np.array_equal(value, np.array(value, copy=True),
                                  equal_nan=True), name


@pytest.mark.parametrize("ni,nj", [(1, 1), (3, 2), (4, 1)])
def test_whole_domain_tile_fills_and_records_its_halo(ni, nj):
    tile = _synthetic_tile(ni, nj)
    rng = np.random.default_rng(3)
    u = rng.uniform(-5, 5, (ni * nj, tile.nz))
    v = rng.uniform(-5, 5, (ni * nj, tile.nz))
    out = whole_domain_tile(tile, dx=1000.0, dy=1500.0, u=u, v=v)
    check_cold_inputs(out)
    assert (out.ide, out.jde, out.its, out.jts) == (ni + 1, nj + 1, 1, 1)
    assert out.halo_fill == HALO_TILE_IS_DOMAIN
    assert out.halo_fill_name == "tile_is_domain"
    assert (out.dx, out.dy) == (1000.0, 1500.0)
    assert np.array_equal(out.snownc, np.zeros(ni * nj))
    for name in HALO_FIELDS:
        halo = getattr(out, f"halo_{name}")
        own = getattr(tile, name)
        for j in range(nj + 2):
            for i in range(ni + 2):
                c = (min(max(j, 1), nj) - 1) * ni + min(max(i, 1), ni) - 1
                assert np.array_equal(halo[out.halo_index(i, j)], own[c])
    for j in range(1, nj + 2):
        for i in range(1, ni + 2):
            c = (min(j, nj) - 1) * ni + min(i, ni) - 1
            assert np.array_equal(out.u[out.wind_index(i, j)], u[c])
            assert np.array_equal(out.v[out.wind_index(i, j)], v[c])
    # The input tile is untouched and still has no cold inputs.
    assert tile.halo_fill == HALO_NONE and np.size(tile.u) == 0


def test_the_flat_layouts_are_the_documented_formulas():
    """The layouts four items and the oracle build on, pinned by literal
    index and by an independent construction: ``halo_*`` rows at ``e = j *
    (ni + 2) + i`` and ``u``, ``v`` rows at ``e = (j - 1) * (ni + 1) + (i -
    1)``.  Every other test fills and reads through the accessors, so a
    transposed layout would stay self-consistent there; here a producer
    packing arrays by the documented formula (numpy's edge padding of the
    (j, i) grid) must agree with the accessors byte for byte."""
    ni, nj = 3, 2
    tile = _synthetic_tile(ni, nj)
    nz = tile.nz
    assert (tile.row_length, tile.nrows) == (ni, nj)
    for (i, j), row in {(0, 0): 0, (4, 0): 4, (0, 1): 5, (1, 1): 6,
                        (4, 3): 19}.items():
        assert tile.halo_index(i, j) == row, (i, j)
    for (i, j), row in {(1, 1): 0, (4, 1): 3, (1, 2): 4, (4, 3): 11}.items():
        assert tile.wind_index(i, j) == row, (i, j)
    rng = np.random.default_rng(5)
    u = rng.uniform(-5, 5, (ni * nj, nz))
    v = rng.uniform(-5, 5, (ni * nj, nz))
    out = whole_domain_tile(tile, dx=1000.0, dy=1500.0, u=u, v=v)
    for name in HALO_FIELDS:
        own = np.asarray(getattr(tile, name), dtype=np.float64)
        expect = np.pad(own.reshape(nj, ni, nz), ((1, 1), (1, 1), (0, 0)),
                        mode="edge").reshape(-1, nz)
        assert getattr(out, f"halo_{name}").tobytes() == expect.tobytes(), name
    for name, wind in (("u", u), ("v", v)):
        expect = np.pad(wind.reshape(nj, ni, nz), ((0, 1), (0, 1), (0, 0)),
                        mode="edge").reshape(-1, nz)
        assert getattr(out, name).tobytes() == expect.tobytes(), name


def test_whole_domain_tile_keeps_the_tiles_own_accumulations():
    """``snownc`` and ``graupelnc`` default to the tile's own when it
    carries them, as copies; a value passed in wins and is copied too."""
    tile = _synthetic_tile()
    tile.snownc = np.arange(1.0, 7.0)
    tile.graupelnc = np.arange(10.0, 16.0)
    rng = np.random.default_rng(9)
    winds = dict(u=rng.uniform(-5, 5, tile.th.shape),
                 v=rng.uniform(-5, 5, tile.th.shape))
    out = whole_domain_tile(tile, dx=1000.0, dy=1000.0, **winds)
    for name in ("snownc", "graupelnc"):
        kept, own = getattr(out, name), getattr(tile, name)
        assert np.array_equal(kept, own), name
        assert not np.shares_memory(kept, own), name
    given = np.full(6, 9.0)
    out = whole_domain_tile(tile, dx=1000.0, dy=1000.0, snownc=given, **winds)
    given[0] = -1.0
    assert np.array_equal(out.snownc, np.full(6, 9.0))
    assert np.array_equal(out.graupelnc, tile.graupelnc)


def test_a_tile_inside_a_larger_domain_carries_its_neighbours():
    """The KC-2 shape: a 3 x 3 tile in the corner of a 4 x 4 domain, its
    halo filled from the domain where the domain has columns and by copy
    outside it, where the pre-pass never reads."""
    ni = nj = 3
    domain = _synthetic_tile(4, 4)
    tile_cols = [(j - 1) * 4 + (i - 1) for j in range(1, 4) for i in range(1, 4)]
    names = [f.name for f in dataclasses.fields(FsbmTile)
             if f.name not in COLD_TILE_FIELDS and f.name != "ni"]
    tile = FsbmTile(**{n: np.array(getattr(domain, n))[tile_cols]
                       for n in names}, ni=3)
    out = tile.copy()
    out.u = np.zeros(((nj + 1) * (ni + 1), tile.nz))
    out.v = np.zeros(((nj + 1) * (ni + 1), tile.nz))
    out.dx = out.dy = 500.0
    out.ide, out.jde, out.its, out.jts = 5, 5, 1, 1
    for name in HALO_FIELDS:
        full = np.asarray(getattr(domain, name))
        halo = np.empty(((nj + 2) * (ni + 2), tile.nz))
        for j in range(nj + 2):
            for i in range(ni + 2):
                di, dj = min(max(i, 1), 4), min(max(j, 1), 4)
                halo[out.halo_index(i, j)] = full[(dj - 1) * 4 + di - 1]
        setattr(out, f"halo_{name}", halo)
    out.halo_fill = HALO_NEIGHBOURS
    out.snownc = np.zeros(9)
    out.graupelnc = np.zeros(9)
    check_cold_inputs(out)
    # The east and north halo hold the domain's fourth column and row.
    assert np.array_equal(out.halo_th[out.halo_index(4, 2)],
                          np.asarray(domain.th)[(2 - 1) * 4 + 3])
    out.halo_fill = HALO_TILE_IS_DOMAIN
    with pytest.raises(FsbmRefusal, match="whole domain"):
        check_cold_inputs(out)


def _mutate(tile, name, value):
    setattr(tile, name, value)


def _mutate_item(tile, name, index, value):
    array = np.array(getattr(tile, name), copy=True)
    array[index] = value
    setattr(tile, name, array)


@pytest.mark.parametrize("change,match", [
    (lambda t: _mutate(t, "u", np.zeros((0, 0))), "tile.u is not supplied"),
    (lambda t: _mutate(t, "v", np.zeros((11, 5))), "tile.v has shape"),
    (lambda t: _mutate_item(t, "u", (3, 2), np.nan), r"tile.u\[3, 2\]"),
    (lambda t: _mutate(t, "dx", 0.0), "tile.dx"),
    (lambda t: _mutate(t, "dy", -1.0), "tile.dy"),
    (lambda t: _mutate(t, "dx", math.inf), "tile.dx"),
    (lambda t: _mutate(t, "dy", True), "tile.dy must be a number"),
    (lambda t: _mutate(t, "ide", 3), "do not lie inside"),
    (lambda t: _mutate(t, "jts", 0), "first column"),
    (lambda t: _mutate(t, "its", 2), "do not lie inside"),
    (lambda t: _mutate(t, "jde", 3.0), "tile.jde must be an int"),
    (lambda t: _mutate(t, "halo_qv_old", np.zeros((0, 0))),
     "tile.halo_qv_old is not supplied"),
    (lambda t: _mutate_item(t, "halo_th_old", (6, 1), 300.0),
     "halo's interior must repeat"),
    (lambda t: _mutate_item(t, "halo_p", (0, 0), -1.0),
     "tile.halo_p must be positive"),
    (lambda t: _mutate_item(t, "halo_pii", (0, 4), np.inf),
     r"tile.halo_pii\[0, 4\]"),
    (lambda t: _mutate_item(t, "halo_pii", (0, 0), 0.0),
     "tile.halo_pii must be positive"),
    (lambda t: _mutate(t, "halo_fill", HALO_NONE), "tile.halo_fill"),
    (lambda t: _mutate(t, "halo_fill", 7), "tile.halo_fill"),
    (lambda t: (_mutate(t, "ide", 9), _mutate(t, "jde", 9)), "whole domain"),
    (lambda t: _mutate(t, "snownc", np.zeros(0)),
     "tile.snownc is not supplied"),
    (lambda t: _mutate(t, "snownc", np.zeros(7)), "tile.snownc has shape"),
    (lambda t: _mutate_item(t, "graupelnc", 2, np.nan),
     r"tile.graupelnc\[2\]"),
])
def test_a_cold_input_the_call_could_not_use_is_refused_by_name(change, match):
    tile = _cold_tile()
    check_cold_inputs(tile)
    change(tile)
    with pytest.raises(FsbmRefusal, match=match):
        check_cold_inputs(tile)


def test_whole_domain_tile_refuses_winds_of_the_wrong_shape():
    tile = _synthetic_tile()
    with pytest.raises(FsbmRefusal, match="u has shape"):
        whole_domain_tile(tile, dx=1.0, dy=1.0, u=np.zeros((5, 5)),
                          v=np.zeros((6, 5)))


def test_a_tile_copy_is_deep_and_keeps_every_scalar():
    tile = _cold_tile()
    copy = tile.copy()
    for name in tile.__dataclass_fields__:
        a, b = getattr(tile, name), getattr(copy, name)
        if isinstance(a, np.ndarray):
            assert b is not a and np.array_equal(a, b), name
        else:
            assert type(a) is type(b) and a == b, name


# ---------------------------------------------------------------------------
# Results, increments, tallies
# ---------------------------------------------------------------------------

_BULK_NAMES = ("qc", "qr", "qi", "qs", "qg", "qnc", "qnr", "qni", "qns",
               "qng", "qna")


def _result(**extra) -> FsbmWarmResult:
    bulk = {k: np.full((1, 2), float(n)) for n, k in enumerate(_BULK_NAMES)}
    return FsbmWarmResult(tile=_synthetic_tile(1, 1, 2), rainncv=np.zeros(1),
                          outflow_mm=np.zeros(1), bulk=bulk,
                          kz_cloud_base=np.zeros(1), nucleated=np.zeros(1),
                          nucleation_vapour=np.zeros(1),
                          clipped=np.zeros(1), **extra)


def test_the_result_gains_the_cold_outputs():
    names = {f.name for f in dataclasses.fields(FsbmWarmResult)}
    assert {"snowncv", "graupelncv", "tallies", "dense_ice"} <= names
    assert FsbmResult is FsbmWarmResult
    result = _result()
    for name in ("qi", "qs", "qg", "qni", "qns", "qng"):
        assert getattr(result, name) is result.bulk[name], name
    assert result.tallies is None and result.snowncv is None
    assert result.dense_ice is None


@pytest.mark.parametrize("option", ["hail", "graupel"])
def test_a_result_records_its_dense_ice_option_by_identity(option):
    chosen = C.dense_ice(option)
    result = _result(dense_ice=chosen)
    assert result.dense_ice is chosen
    assert result.dense_ice.identity == f"fast-sbm/dense-ice={option}"


@pytest.mark.parametrize("bad", [1, 0, "hail", True])
def test_a_result_refuses_a_dense_ice_that_is_not_the_named_option(bad):
    with pytest.raises(FsbmRefusal, match="dense_ice must be a DenseIce"):
        _result(dense_ice=bad)


def test_the_increments_name_every_process_and_carry_the_ice_spectra():
    assert set(WARM_INCREMENT_PROCESSES).isdisjoint(COLD_INCREMENT_PROCESSES)
    assert INCREMENT_PROCESSES == (WARM_INCREMENT_PROCESSES
                                   + COLD_INCREMENT_PROCESSES)
    point = FsbmPoint(1, 1, 1, 1, 1)
    inc = FsbmIncrement("freez", point, d_ff1=-np.ones(33), d_fccn=None,
                        d_t=0.1, d_qv=0.0, d_ff2=np.zeros((33, 3)),
                        d_ff5=np.ones(33))
    assert inc.d_ff3 is None and inc.d_ff4 is None
    old = FsbmIncrement("onecond1", point, None, None, 0.0, 0.0)
    assert (old.d_ff2, old.d_ff3, old.d_ff4, old.d_ff5) == (None,) * 4
    with pytest.raises(FsbmRefusal, match="freezing"):
        FsbmIncrement("freezing", point, None, None, 0.0, 0.0)


def test_the_tallies_name_their_pair_and_class():
    point = FsbmPoint(1, 1, 1, 1)
    zero = np.zeros((33, 33))
    assert set(TALLY_CLASSES) == {spec[0] for spec in TALLY_PAIRS.values()}
    kernel_fields = {f.name for f in dataclasses.fields(FsbmKernels)}
    for pair, (klass, first, second, kernel) in TALLY_PAIRS.items():
        assert kernel in kernel_fields, pair
        to = TALLY_DESTINATIONS[pair][0]
        if first == second:
            tally = FsbmTally(pair, point, zero, None, to)
            with pytest.raises(FsbmRefusal, match="self-collection"):
                FsbmTally(pair, point, zero, zero, to)
        else:
            tally = FsbmTally(pair, point, zero, zero, to)
            with pytest.raises(FsbmRefusal, match="needs taken_y"):
                FsbmTally(pair, point, zero, None, to)
        assert tally.tally_class == klass
        liquid = "drop" in (first, second)
        assert klass == ("liquid_liquid" if first == second == "drop" else
                         "liquid_ice" if liquid else "ice_ice")
    with pytest.raises(FsbmRefusal, match="pair"):
        FsbmTally("crystal_snow", point, zero, zero, "snow")
    with pytest.raises(FsbmRefusal, match="taken_x has shape"):
        FsbmTally("snow_snow", point, np.zeros((33, 32)), None, "snow")
    with pytest.raises(FsbmRefusal, match="taken_y has shape"):
        FsbmTally("drop_snow", point, zero, np.zeros(33), "hail")
    with pytest.raises(TypeError, match="destination"):
        FsbmTally("drop_snow", point, zero, zero)


_SPECIES = {"g1": "drop", "g3": "snow", "g4": "graupel", "g5": "hail"}


def test_the_tally_destinations_are_where_the_cold_block_puts_the_mass():
    """Read off COAL_BOTT_NEW itself: each collision call's routine form
    says where the collected mass goes (xxx and xyx keep it in the first
    spectrum, xyy puts it in the second, xyz in the third), and its kernel
    names the pair.  Every pair's destinations are exactly the ones some
    call can produce, so a tally can close against the mass moved."""
    lines = source_lines()
    coal = next(r for r in parsed() if r.name == "COAL_BOTT_NEW")
    call = re.compile(r"call\s+coll_(x[xy][xyz])_lwf\s*\(([^)&]*)", re.I)
    by_kernel: dict[str, set[str]] = {}
    for n in range(coal.first, coal.last + 1):
        found = call.search(lines[n - 1])
        if not found:
            continue
        form = found.group(1).lower()
        args = [a.strip().lower() for a in found.group(2).split(",")]
        spectra = [a for a in args if a in _SPECIES]
        kernel = next(a for a in args if a.startswith("cw"))
        target = {"xxx": 0, "xyx": 0, "xyy": 1, "xyz": 2}[form]
        by_kernel.setdefault(kernel, set()).add(_SPECIES[spectra[target]])
    assert set(by_kernel) == {spec[3] for spec in TALLY_PAIRS.values()}
    for pair, (_, first, second, kernel) in TALLY_PAIRS.items():
        assert by_kernel[kernel] == set(TALLY_DESTINATIONS[pair]), pair
    point = FsbmPoint(1, 1, 1, 1)
    zero = np.zeros((33, 33))
    for pair, (_, first, second, _) in TALLY_PAIRS.items():
        taken_y = None if first == second else zero
        for wrong in {"drop", "snow", "graupel", "hail", "crystal", None} - \
                set(TALLY_DESTINATIONS[pair]):
            with pytest.raises(FsbmRefusal, match="destination"):
                FsbmTally(pair, point, zero, taken_y, wrong)


def _tally(pair, destination):
    zero = np.zeros((33, 33))
    first, second = TALLY_PAIRS[pair][1:3]
    return FsbmTally(pair, FsbmPoint(1, 1, 1, 1), zero,
                     None if first == second else zero, destination)


@pytest.mark.parametrize("option,tallies,match", [
    ("hail", [("drop_snow", "hail"), ("snow_drop", "snow"),
              ("snow_drop", "hail"), ("drop_hail", "hail"),
              ("snow_snow", "snow"), ("drop_drop", "drop")], None),
    ("graupel", [("drop_snow", "graupel"), ("drop_graupel", "graupel"),
                 ("graupel_drop", "hail")], None),
    ("hail", [("drop_snow", "graupel")], "run's dense ice is hail"),
    ("graupel", [("snow_drop", "hail")], "run's dense ice is graupel"),
    ("hail", [("drop_graupel", "graupel")], "the hail build leaves empty"),
    ("graupel", [("hail_drop", "hail")], "the graupel build leaves empty"),
    (None, [("drop_drop", "drop")], "must name its dense_ice"),
    (None, [], "must name its dense_ice"),
])
def test_a_results_tallies_agree_with_its_dense_ice(option, tallies, match):
    chosen = None if option is None else C.dense_ice(option)
    made = [_tally(pair, to) for pair, to in tallies]
    if match is None:
        assert _result(dense_ice=chosen, tallies=made).tallies is made
    else:
        with pytest.raises(FsbmRefusal, match=match):
            _result(dense_ice=chosen, tallies=made)


# ---------------------------------------------------------------------------
# The warm call with the contract in place (KC-1)
# ---------------------------------------------------------------------------

def _f8(a) -> np.ndarray:
    return np.array(a, dtype=np.float64)


@functools.lru_cache(maxsize=None)
def _kc1_statics():
    return next(fx.iter_records("kc1_summary_o64", types=["initial"]))


@functools.lru_cache(maxsize=None)
def _kc1_pre(step: int):
    for n, pre, _ in fx.iter_replay("kc1_replay_o64"):
        if n == step:
            return pre
    raise KeyError(step)


def _kc1_tile(step: int) -> FsbmTile:
    """The tile FAST_SBM received at a KC-1 replay step (O64 build)."""
    s, pre = _kc1_statics(), _kc1_pre(step)
    return FsbmTile.from_chem(
        _f8(pre["chem"]), th=_f8(pre["th"]), qv=_f8(pre["qv"]),
        th_old=_f8(pre["th_old"]), qv_old=_f8(pre["qv_old"]), p=_f8(s["p"]),
        pii=_f8(s["pii"]), rho=_f8(s["rho"]), dz=_f8(s["dz8w"]),
        w=_f8(pre["w"]), xland=_f8(s["xland"]), rainnc=_f8(pre["rainnc"]))


@functools.lru_cache(maxsize=None)
def _derived(fall_speed_lookup: str):
    return build_derived(tables(), DT, fall_speed_lookup=fall_speed_lookup)


def _run(tile, step, behavior):
    return fast_sbm_warm_call(tile, step, DT, tables(),
                              _derived(behavior.fall_speed_lookup),
                              behavior=behavior, processes=default_processes())


def _with_cold_inputs(tile: FsbmTile) -> FsbmTile:
    rng = np.random.default_rng(11)
    return whole_domain_tile(tile, dx=1000.0, dy=1000.0,
                             u=rng.uniform(-9, 9, tile.th.shape),
                             v=rng.uniform(-9, 9, tile.th.shape),
                             snownc=rng.uniform(0, 3, tile.ncol),
                             graupelnc=rng.uniform(0, 3, tile.ncol))


@pytest.mark.parametrize("behavior", [WRF, CORRECTED],
                         ids=["wrf_v480", "corrected"])
def test_the_cold_inputs_leave_the_warm_call_byte_identical(behavior):
    """One KC-1 replay call with and without the cold inputs, under each
    preset: every state field and every result array is the same bytes,
    the cold inputs come back as they went in, and the warm call's surface
    snow and dense ice are exact zeros."""
    step = 40
    plain_tile = _kc1_tile(step)
    cold_tile = _with_cold_inputs(plain_tile)
    plain = _run(plain_tile, step, behavior)
    cold = _run(cold_tile, step, behavior)
    for name in plain.tile.__dataclass_fields__:
        if name in COLD_TILE_FIELDS:
            continue
        assert np.asarray(getattr(plain.tile, name)).tobytes() == \
            np.asarray(getattr(cold.tile, name)).tobytes(), name
    for name in ("rainncv", "outflow_mm", "kz_cloud_base", "nucleated",
                 "nucleation_vapour", "clipped"):
        assert np.asarray(getattr(plain, name)).tobytes() == \
            np.asarray(getattr(cold, name)).tobytes(), name
    for name in plain.bulk:
        assert plain.bulk[name].tobytes() == cold.bulk[name].tobytes(), name
    for name in COLD_TILE_FIELDS:
        assert np.array_equal(getattr(cold.tile, name),
                              getattr(cold_tile, name)), name
    for result in (plain, cold):
        assert result.snowncv.tobytes() == np.zeros(4).tobytes()
        assert result.graupelncv.tobytes() == np.zeros(4).tobytes()
        assert result.tallies is None and result.dense_ice is None
        assert not np.any(result.qi) and not np.any(result.qg)


def test_the_warm_call_records_only_warm_increments():
    step = 40
    result = fast_sbm_warm_call(_kc1_tile(step), step, DT, tables(),
                                _derived(WRF.fall_speed_lookup), behavior=WRF,
                                processes=default_processes(),
                                record_increments=True)
    kinds = {inc.process for inc in result.increments}
    assert kinds and kinds <= set(WARM_INCREMENT_PROCESSES)
    for inc in result.increments:
        assert (inc.d_ff2, inc.d_ff3, inc.d_ff4, inc.d_ff5) == (None,) * 4


class _NoCalls:
    def __getattr__(self, name):
        def fail(**kwargs):
            raise AssertionError(f"process {name} was called")
        return fail


def test_the_warm_call_refuses_a_non_finite_cold_input_untouched():
    tile = _with_cold_inputs(_kc1_tile(40))
    tile.halo_qv_old = np.array(tile.halo_qv_old, copy=True)
    tile.halo_qv_old[0, 3] = np.nan
    before = tile.copy()
    with pytest.raises(FsbmRefusal, match=r"tile.halo_qv_old\[0, 3\]"):
        fast_sbm_warm_call(tile, 40, DT, tables(),
                           _derived(WRF.fall_speed_lookup), behavior=WRF,
                           processes=_NoCalls())
    for name in tile.__dataclass_fields__:
        if name != "ni":
            assert np.array_equal(getattr(tile, name), getattr(before, name),
                                  equal_nan=True), name


def test_d8_layer_depth_is_refused_until_the_fall_exit_item_builds_it():
    """Both presets divide by the level spacing today, and a behaviour that
    asks for the layer depth is refused by name before any process runs,
    so no result carries WRF's divisor under the corrected label.  The
    refusal cites D8 so the fall-exit item's guard sweep finds it."""
    assert WRF.fall_depth == CORRECTED.fall_depth == "level_spacing"
    arm = dataclasses.replace(CORRECTED, fall_depth="layer_depth")
    tile = _kc1_tile(40)
    with pytest.raises(FsbmRefusal, match=r"fall_depth 'layer_depth' \(D8\)"):
        fast_sbm_warm_call(tile, 40, DT, tables(),
                           _derived(arm.fall_speed_lookup), behavior=arm,
                           processes=_NoCalls())
    guard = Path(COL_MOD.__file__).read_text(encoding="utf-8")
    assert "Guard for D8, retired by the fall-exit item" in guard
