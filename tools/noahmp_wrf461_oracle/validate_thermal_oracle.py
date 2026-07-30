#!/usr/bin/env python3
"""Validate the Noah-MP *thermal* leaf oracle fixture and its zero-probe sweep.

Same gate as ``validate_leaf_oracle.py``, applied to a second, independently
owned leaf group: TSNOSOI, HRT, HSTEP, PHASECHANGE, FRH2O.  It checks four
independent things:

1. **Structure.** Every declared leaf is present with the declared case list
   and the declared input/output slot counts.
2. **Bit fidelity.** The ``bits`` column round-trips to the ``value`` column
   exactly, so downstream ``max_ulp 0`` comparison can use ``bits`` alone.
3. **Undefined-slot convention.** Every ``live=0`` output slot carries the
   harness's 0.0 pre-fill.
4. **Discrimination.** For every input slot of every leaf, zeroing that slot
   must move at least one live output in at least one case where the slot was
   not already zero -- unless the slot is in that leaf's declared inert set,
   in which case it must move nothing anywhere.

``check_branch_coverage`` then asserts, per leaf, that the branches this option
identity leaves live were actually taken, and -- where the dead branch would
have produced a different, recognisable value -- that the dead branch was NOT
taken.  ``BOTFLX != 0`` is the executable statement that ``OPT_TBOT == 1`` was
off; ``BI == -CI`` at the top layer is the executable statement that
``OPT_STC == 2`` was off.

Usage:
    validate_thermal_oracle.py THERMAL.csv DISCRIMINATION.csv
"""

from __future__ import annotations

import csv
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path


def _f32(bits: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]


def _r32(value: float) -> float:
    """Round a Python float to FP32.

    The sum of two FP32 values is exact in FP64, so rounding that sum once
    here reproduces the single FP32 addition gfortran emitted -- there is no
    double rounding to worry about for the two-term identities below.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


NSNOW = 3
NSOIL = 4
NLAY = NSNOW + NSOIL

# leaf -> (cases, n_int, n_in, n_out, inert{(name, index): reason})
LEAVES: dict[str, dict] = {
    "hrt": {
        "cases": (
            "soil_only_isnow0", "one_snow_layer", "two_snow_layers",
            "three_snow_layers", "three_snow_layers_alt", "warm_soil_isnow0",
        ),
        "n_int": 1, "n_in": 5 * NLAY + 4, "n_out": 4 * NLAY + 1,
        "inert": {
            ("dt", 0): "declared INTENT(IN) at 5397; the body never "
                       "references it -- HRT forms a tendency, HSTEP steps it",
        },
    },
    "hstep": {
        "cases": (
            "soil_only_isnow0", "one_snow_layer", "two_snow_layers",
            "three_snow_layers", "three_snow_layers_alt", "short_step_isnow0",
        ),
        "n_int": 1, "n_in": 5 * NLAY + 1, "n_out": 5 * NLAY,
        "inert": {
            ("ci", 4): "ROSR12 overwrites C(NSOIL) with 0.0 at 5565 before "
                       "any read of C, and the CI(NSOIL) output is "
                       "P(NSOIL) = DELTA(NSOIL) at 5582",
        },
    },
    "tsnosoi": {
        "cases": (
            "soil_only_isnow0", "one_snow_layer", "two_snow_layers",
            "three_snow_layers", "three_snow_layers_alt", "warm_soil_isnow0",
        ),
        "n_int": 5, "n_in": 5 * NLAY + 7, "n_out": NLAY + 1,
        "inert": {
            ("sag", 0): "declared INTENT(IN) at 5284; never referenced "
                        "anywhere in the routine",
            ("tg", 0): "declared INTENT(IN) at 5286; referenced only in "
                       "SSOIL2 at 5359, after the RETURN at 5346",
            ("ice", 0): "declared INTENT(IN) at 5275; never referenced "
                        "anywhere in the routine",
            ("ist", 0): "declared INTENT(IN) at 5279; referenced only in the "
                        "diagnostic WRITE at 5367, after the RETURN at 5346",
            ("iloc", 0): "declared INTENT(IN) at 5273; referenced only in the "
                         "diagnostic WRITE at 5367, after the RETURN at 5346",
            ("jloc", 0): "declared INTENT(IN) at 5274; referenced only in the "
                         "diagnostic WRITE at 5367, after the RETURN at 5346",
            **{("dzsnso", k): "referenced only in the ERR_EST accumulation "
                              "at 5352, after the RETURN at 5346"
               for k in range(-NSNOW + 1, NSOIL + 1)},
        },
    },
    "phasechange": {
        "cases": (
            "soil_melt_and_freeze", "soil_column_refreeze",
            "no_layer_partial_melt", "no_layer_total_melt",
            "no_layer_heat_deficit", "three_snow_melt", "three_snow_freeze",
            "lake_two_snow_ist2", "one_snow_mixed", "fact_sign_probe",
        ),
        "n_int": 4,
        "n_in": 4 * NLAY + 2 * NSNOW + 2 * NSOIL + 3 + 3 * NSOIL,
        "n_out": NLAY + 2 * NSNOW + 2 * NSOIL + 4 + NLAY,
        "inert": {
            ("iloc", 0): "declared INTENT(IN) at 5608, never referenced",
            ("jloc", 0): "declared INTENT(IN) at 5609, never referenced",
            **{("hcpct", k): "declared INTENT(IN) at 5617, never referenced"
               for k in range(-NSNOW + 1, NSOIL + 1)},
            **{("dzsnso", k): "DZSNSO is read only for soil layers "
                              "(5666-5667, 5687, 5804-5805)"
               for k in range(-NSNOW + 1, 1)},
        },
    },
    # DEAD under the pinned identity: FRH2O's only call site is 5692, inside
    # IF (OPT_FRZ == 2), and the pinned identity is opt_frz = 1.  Pinned
    # anyway because it is a self-contained leaf; nothing calls the port.
    "frh2o": {
        "cases": (
            "warm_shortcut", "cold_loam", "clay_bexp_above_blim",
            "dry_sh2o_below_two_pct", "trace_moisture_column",
            "swlk_high_clamp", "swlk_low_clamp", "sand_wet_cold",
        ),
        "n_int": 0, "n_in": 6, "n_out": 1, "inert": {},
    },
}


class ThermalOracleError(RuntimeError):
    pass


def load_leaves(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ThermalOracleError(f"{path} is empty")
    table: dict[tuple[str, str, str], dict] = defaultdict(dict)
    for row in rows:
        key = (row["leaf"], row["case"], row["role"])
        table[key][int(row["slot"])] = row
    return table


def check_structure(table) -> None:
    seen_leaves = {key[0] for key in table}
    if seen_leaves != set(LEAVES):
        raise ThermalOracleError(
            f"leaf set mismatch: extra={sorted(seen_leaves - set(LEAVES))}, "
            f"missing={sorted(set(LEAVES) - seen_leaves)}")
    for leaf, spec in LEAVES.items():
        cases = {key[1] for key in table if key[0] == leaf}
        if cases != set(spec["cases"]):
            raise ThermalOracleError(
                f"{leaf}: case mismatch "
                f"extra={sorted(cases - set(spec['cases']))} "
                f"missing={sorted(set(spec['cases']) - cases)}")
        for case in spec["cases"]:
            for role, count in (("int", spec["n_int"]), ("in", spec["n_in"]),
                                ("out", spec["n_out"])):
                got = len(table.get((leaf, case, role), {}))
                if got != count:
                    raise ThermalOracleError(
                        f"{leaf}/{case}: role {role} has {got} slots, "
                        f"expected {count}")


def check_bits(table) -> int:
    checked = 0
    for (leaf, case, role), slots in table.items():
        for slot, row in slots.items():
            from_bits = _f32(row["bits"])
            printed = float(row["value"])
            if struct.pack("<f", printed) != struct.pack("<f", from_bits):
                raise ThermalOracleError(
                    f"{leaf}/{case}/{role}[{slot}]: bits {row['bits']} "
                    f"({from_bits!r}) does not round-trip the printed value "
                    f"{row['value']}")
            if role == "out" and row["live"] == "1":
                if not math.isfinite(from_bits):
                    raise ThermalOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is "
                        f"non-finite: {from_bits}")
            if role == "out" and row["live"] == "0":
                if row["bits"] != "00000000":
                    raise ThermalOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is flagged "
                        f"live=0 but carries {row['bits']}, not the 0.0 "
                        f"pre-fill")
            checked += 1
    return checked


def check_discrimination(disc_path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(disc_path.open(newline="")))
    if not rows:
        raise ThermalOracleError(f"{disc_path} is empty")
    probes: dict[tuple[str, int], list] = defaultdict(list)
    for row in rows:
        probes[(row["leaf"], int(row["slot"]))].append(
            (row["name"], int(row["index"]), int(row["already_zero"]),
             int(row["noutputs_changed"])))

    counts: dict[str, int] = defaultdict(int)
    for (leaf, slot), entries in sorted(probes.items()):
        spec = LEAVES[leaf]
        name, index = entries[0][0], entries[0][1]
        declared_inert = (name, index) in spec["inert"]
        informative = [e for e in entries if e[2] == 0]
        moved = [e for e in entries if e[3] > 0]
        if not informative:
            raise ThermalOracleError(
                f"{leaf}: input slot {slot} ({name}[{index}]) is zero in "
                f"every case, so the fixture cannot tell whether the port "
                f"reads it. Give at least one case a non-zero value.")
        if declared_inert and moved:
            raise ThermalOracleError(
                f"{leaf}: {name}[{index}] is declared inert "
                f"({spec['inert'][(name, index)]}) but zeroing it moved "
                f"{moved[0][3]} outputs")
        if not declared_inert and not moved:
            raise ThermalOracleError(
                f"{leaf}: zeroing {name}[{index}] moves no live output in "
                f"any of {len(entries)} cases, and it is not declared inert. "
                f"Either the fixture cannot detect a dropped variable or the "
                f"argument really is dead -- decide and record which.")
        counts[leaf] += 1
    for leaf, spec in LEAVES.items():
        if counts[leaf] != spec["n_in"]:
            raise ThermalOracleError(
                f"{leaf}: discrimination sweep covered {counts[leaf]} of "
                f"{spec['n_in']} input slots")
    return counts


def _lookup(table, leaf, case, role, name, index):
    for row in table[(leaf, case, role)].values():
        if row["name"] == name and int(row["index"]) == index:
            return row
    raise ThermalOracleError(f"{leaf}/{case}: no {role} {name}[{index}]")


def check_branch_coverage(table) -> None:
    """Explicit branch assertions, live and dead, for each thermal leaf."""

    def out(leaf, case, name, index):
        return _f32(_lookup(table, leaf, case, "out", name, index)["bits"])

    def inp(leaf, case, name, index):
        return _f32(_lookup(table, leaf, case, "in", name, index)["bits"])

    def isnow(leaf, case):
        row = _lookup(table, leaf, case, "int", "isnow", 0)
        return int(float(row["value"]))

    # ---------------------------------------------------------------- HRT --
    topologies = {isnow("hrt", case) for case in LEAVES["hrt"]["cases"]}
    if topologies != {0, -1, -2, -3}:
        raise ThermalOracleError(
            f"hrt: ISNOW coverage is {sorted(topologies)}, not every "
            f"topology the pinned three-snow-layer stack admits")
    deep = [c for c in LEAVES["hrt"]["cases"] if isnow("hrt", c) == -3]
    if len(deep) < 2:
        raise ThermalOracleError(
            "hrt: fewer than two ISNOW=-3 cases, so the layer -2 slots have a "
            "single reader and freezing them to that reader's own value is a "
            "no-op no mutant can detect")

    for case in LEAVES["hrt"]["cases"]:
        top = isnow("hrt", case) + 1
        # 5453: AI(ISNOW+1) = 0.0, the head branch at 5425/5452.
        if out("hrt", case, "ai", top) != 0.0:
            raise ThermalOracleError(
                f"hrt/{case}: AI(ISNOW+1) must be exactly 0.0 (5453)")
        # 5467: CI(NSOIL) = 0.0, the foot branch at 5437/5465.
        if out("hrt", case, "ci", NSOIL) != 0.0:
            raise ThermalOracleError(
                f"hrt/{case}: CI(NSOIL) must be exactly 0.0 (5467)")
        # 5456: OPT_STC == 1 gives BI = -CI at the top layer.  The dead
        # OPT_STC == 2 form at 5459 adds DF/(0.5*Z*Z*HCPCT), which is strictly
        # positive here, so this equality is what rules that branch out.
        if out("hrt", case, "bi", top) != -out("hrt", case, "ci", top):
            raise ThermalOracleError(
                f"hrt/{case}: BI(ISNOW+1) != -CI(ISNOW+1); the OPT_STC == 2 "
                f"diagonal at 5459 appears to have been taken")
        # 5464/5468: the interior and foot diagonals.
        for k in range(top + 1, NSOIL + 1):
            want = -_r32(out("hrt", case, "ai", k) + out("hrt", case, "ci", k))
            if out("hrt", case, "bi", k) != want:
                raise ThermalOracleError(
                    f"hrt/{case}: BI({k}) != -(AI({k})+CI({k}))")
        # 5443-5446: OPT_TBOT == 2.  Under the dead OPT_TBOT == 1 branch
        # (5441) BOTFLX would be exactly 0.0.
        if out("hrt", case, "botflx", 0) == 0.0:
            raise ThermalOracleError(
                f"hrt/{case}: BOTFLX is exactly 0.0, which is the dead "
                f"OPT_TBOT == 1 value at 5441")
        # The interior branch at 5431-5436 must exist in every case.
        if top + 1 > NSOIL - 1:
            raise ThermalOracleError(
                f"hrt/{case}: no interior layer, so 5431-5436 is unreached")
        # PHI must be non-zero somewhere, or its term is undetectable.
        if all(inp("hrt", case, "phi", k) == 0.0
               for k in range(top, NSOIL + 1)):
            raise ThermalOracleError(
                f"hrt/{case}: PHI is zero on every read layer, so the "
                f"5430/5436/5447 term cannot be discriminated")

    # -------------------------------------------------------------- HSTEP --
    topologies = {isnow("hstep", case) for case in LEAVES["hstep"]["cases"]}
    if topologies != {0, -1, -2, -3}:
        raise ThermalOracleError(
            f"hstep: ISNOW coverage is {sorted(topologies)}")
    for case in LEAVES["hstep"]["cases"]:
        top = isnow("hstep", case) + 1
        # Slots above ISNOW are INTENT(INOUT) and untouched: echo, bit for bit.
        for name in ("ai", "bi", "ci", "rhsts", "stc"):
            for k in range(-NSNOW + 1, top):
                got = _lookup(table, "hstep", case, "out", name, k)["bits"]
                want = _lookup(table, "hstep", case, "in", name, k)["bits"]
                if got != want:
                    raise ThermalOracleError(
                        f"hstep/{case}: {name}({k}) is above ISNOW and must "
                        f"echo its input, got {got} want {want}")
        # 5582: ROSR12 sets P(NSOIL) = DELTA(NSOIL), and HSTEP's P is CI while
        # its DELTA is RHSTS, so the two outputs must agree bit for bit.
        got = _lookup(table, "hstep", case, "out", "ci", NSOIL)["bits"]
        want = _lookup(table, "hstep", case, "out", "rhsts", NSOIL)["bits"]
        if got != want:
            raise ThermalOracleError(
                f"hstep/{case}: CI(NSOIL) must equal RHSTS(NSOIL) (5582), "
                f"got {got} want {want}")
        # 5526-5528: the temperature update must actually move.
        moved = [k for k in range(top, NSOIL + 1)
                 if out("hstep", case, "stc", k)
                 != inp("hstep", case, "stc", k)]
        if not moved:
            raise ThermalOracleError(
                f"hstep/{case}: no layer temperature changed, so the ROSR12 "
                f"solve and the 5527 update are both unexercised")
        # 5508-5510: the coefficient scaling must move BI off its input.
        for k in range(top, NSOIL + 1):
            if out("hstep", case, "bi", k) == inp("hstep", case, "bi", k):
                raise ThermalOracleError(
                    f"hstep/{case}: BI({k}) did not change, so 1.0 + BI*DT "
                    f"at 5509 is unexercised")

    # ------------------------------------------------------------ TSNOSOI --
    if "tsnosoi" not in LEAVES:
        return
    topologies = {isnow("tsnosoi", case)
                  for case in LEAVES["tsnosoi"]["cases"]}
    if topologies != {0, -1, -2, -3}:
        raise ThermalOracleError(
            f"tsnosoi: ISNOW coverage is {sorted(topologies)}")
    deep = [c for c in LEAVES["tsnosoi"]["cases"] if isnow("tsnosoi", c) == -3]
    if len(deep) < 2:
        raise ThermalOracleError(
            "tsnosoi: fewer than two ISNOW=-3 cases, so the layer -2 slots "
            "have a single reader")
    for case in LEAVES["tsnosoi"]["cases"]:
        top = isnow("tsnosoi", case) + 1
        # STC above ISNOW is INTENT(INOUT) and untouched: echo, bit for bit.
        for k in range(-NSNOW + 1, top):
            got = _lookup(table, "tsnosoi", case, "out", "stc", k)["bits"]
            want = _lookup(table, "tsnosoi", case, "in", "stc", k)["bits"]
            if got != want:
                raise ThermalOracleError(
                    f"tsnosoi/{case}: STC({k}) is above ISNOW and must echo "
                    f"its input, got {got} want {want}")
        # The HRT + HSTEP chain must actually move the column.
        moved = [k for k in range(top, NSOIL + 1)
                 if out("tsnosoi", case, "stc", k)
                 != inp("tsnosoi", case, "stc", k)]
        if not moved:
            raise ThermalOracleError(
                f"tsnosoi/{case}: no layer temperature changed, so neither "
                f"the 5324 HRT call nor the 5330 HSTEP call is exercised")
        # EFLXB is HRT's BOTFLX under OPT_TBOT == 2 (5443-5446).  The dead
        # OPT_TBOT == 1 branch at 5441 would leave it exactly 0.0.
        if out("tsnosoi", case, "eflxb", 0) == 0.0:
            raise ThermalOracleError(
                f"tsnosoi/{case}: EFLXB is exactly 0.0, which is the dead "
                f"OPT_TBOT == 1 value at 5441")
        # ZBOT must vary across cases, or no freeze mutant on it can be
        # killed and a port could hard-code ZBOT_TABLE.
    zbots = {inp("tsnosoi", case, "zbot", 0)
             for case in LEAVES["tsnosoi"]["cases"]}
    if len(zbots) < 2:
        raise ThermalOracleError(
            "tsnosoi: ZBOT takes one value across the fixture, so a port "
            "that hard-codes ZBOT_TABLE would be indistinguishable")
    snowhs = {inp("tsnosoi", case, "snowh", 0)
              for case in LEAVES["tsnosoi"]["cases"]}
    if len(snowhs) < 2 or 0.0 in snowhs:
        raise ThermalOracleError(
            "tsnosoi: SNOWH must be non-zero and varying, otherwise "
            "ZBOTSNO = ZBOT - SNOWH at 5314 cannot discriminate it")

    # -------------------------------------------------------- PHASECHANGE --
    if "phasechange" not in LEAVES:
        return
    cases = LEAVES["phasechange"]["cases"]
    topologies = {isnow("phasechange", case) for case in cases}
    if topologies != {0, -1, -2, -3}:
        raise ThermalOracleError(
            f"phasechange: ISNOW coverage is {sorted(topologies)}")
    deep = [c for c in cases if isnow("phasechange", c) == -3]
    if len(deep) < 2:
        raise ThermalOracleError(
            "phasechange: fewer than two ISNOW=-3 cases")

    def ist(case):
        return int(float(
            _lookup(table, "phasechange", case, "int", "ist", 0)["value"]))

    if {ist(c) for c in cases} != {1, 2}:
        raise ThermalOracleError(
            "phasechange: IST must take both 1 (soil, supercool loop live at "
            "5671-5697) and 2 (lake, loop skipped)")

    # Echo of the INTENT(INOUT) region above ISNOW.
    for case in cases:
        top = isnow("phasechange", case) + 1
        for k in range(-NSNOW + 1, top):
            for name in ("stc", "snice", "snliq"):
                if name != "stc" and k > 0:
                    continue
                got = _lookup(table, "phasechange", case,
                              "out", name, k)["bits"]
                want = _lookup(table, "phasechange", case,
                               "in", name, k)["bits"]
                if got != want:
                    raise ThermalOracleError(
                        f"phasechange/{case}: {name}({k}) is above ISNOW and "
                        f"must echo its input, got {got} want {want}")

    def imelt(case, k):
        return int(out("phasechange", case, "imelt", k))

    # Every IMELT state the pinned identity can produce must appear.
    states = {imelt(c, k) for c in cases
              for k in range(isnow("phasechange", c) + 1, NSOIL + 1)}
    if states != {0, 1, 2}:
        raise ThermalOracleError(
            f"phasechange: IMELT states seen are {sorted(states)}; 1 (melt, "
            f"5701) and 2 (refreeze, 5704) and 0 must all occur")

    # 5735-5752, the layerless-snow block.  PONDING = TEMP1 - SNEQV is zero
    # unless it ran, and SNEQV must both survive and vanish across the fixture.
    ran = [c for c in cases if out("phasechange", c, "ponding", 0) != 0.0]
    if not ran:
        raise ThermalOracleError(
            "phasechange: the ISNOW==0/SNEQV>0/XM(1)>0 block at 5735 never "
            "ran, so neither PONDING nor QMELT's 5749 term is exercised")
    survived = [c for c in ran if out("phasechange", c, "sneqv", 0) > 0.0]
    exhausted = [c for c in ran if out("phasechange", c, "sneqv", 0) == 0.0]
    if not survived or not exhausted:
        raise ThermalOracleError(
            f"phasechange: the 5738 MAX(0, TEMP1-XM(1)) needs a case that "
            f"leaves snow and one that removes it all "
            f"(survived={survived}, exhausted={exhausted})")

    # 5723/5727: the two IMELT resets.  They need FACT < 0 to fire at all --
    # see run_thermal.F90 -- so at least one case must carry a negative FACT
    # and come back with IMELT == 0 on a layer whose STC was driven to TFRZ.
    probes = []
    for case in cases:
        top = isnow("phasechange", case) + 1
        for k in range(top, NSOIL + 1):
            if inp("phasechange", case, "fact", k) < 0.0 \
                    and imelt(case, k) == 0 \
                    and out("phasechange", case, "stc", k) == _r32(273.16) \
                    and inp("phasechange", case, "stc", k) != _r32(273.16):
                probes.append((case, k))
    if len(probes) < 2:
        raise ThermalOracleError(
            "phasechange: the IMELT resets at 5723 and 5727 are unbound; "
            "they need a negative FACT on a layer on each side of TFRZ")

    # 5786-5789, the BARLAGE branch: a snow layer whose ice is fully melted
    # comes back with SNICE == 0 and STC == TFRZ.
    barlage = [(c, k) for c in cases
               for k in range(isnow("phasechange", c) + 1, 1)
               if out("phasechange", c, "snice", k) == 0.0
               and inp("phasechange", c, "snice", k) > 0.0]
    if not barlage:
        raise ThermalOracleError(
            "phasechange: no snow layer melts its ice away, so the BARLAGE "
            "block at 5786-5789 is unreached")

    # 5766 vs 5771: snow freezing and soil freezing must both occur.
    snow_freeze = [(c, k) for c in cases
                   for k in range(isnow("phasechange", c) + 1, 1)
                   if imelt(c, k) == 2]
    soil_freeze = [(c, k) for c in cases for k in range(1, NSOIL + 1)
                   if imelt(c, k) == 2]
    if not snow_freeze or not soil_freeze:
        raise ThermalOracleError(
            f"phasechange: refreezing must occur in snow (5766) and in soil "
            f"(5771); snow={snow_freeze[:2]} soil={soil_freeze[:2]}")

    # The soil parameters are read only inside IF(IST==1) and IF(STC<TFRZ)
    # at 5683-5689, so they must vary across the cases that read them or no
    # freeze mutant on them can be killed.
    for name in ("smcmax", "psisat", "bexp"):
        for k in range(1, NSOIL + 1):
            values = {inp("phasechange", c, name, k) for c in cases}
            if len(values) < 2:
                raise ThermalOracleError(
                    f"phasechange: {name}({k}) takes one value across the "
                    f"fixture, so a port that hard-codes it is invisible")

    # -------------------------------------------------------------- FRH2O --
    if "frh2o" not in LEAVES:
        return
    cases = LEAVES["frh2o"]["cases"]
    tfrz_eps = _r32(_r32(273.16) - _r32(1.0e-3))

    # 5872: the warm shortcut returns SMC untouched, and no other case may
    # take it -- otherwise the Newton loop below is not what is being pinned.
    warm = [c for c in cases if inp("frh2o", c, "tkelv", 0) > tfrz_eps]
    if len(warm) != 1:
        raise ThermalOracleError(
            f"frh2o: exactly one case must satisfy TKELV > TFRZ-1.0E-3 "
            f"(5872); got {warm}")
    if out("frh2o", warm[0], "free", 0) != inp("frh2o", warm[0], "smc", 0):
        raise ThermalOracleError(
            f"frh2o/{warm[0]}: the 5873 shortcut must return SMC exactly")

    # 5866: BEXP on both sides of BLIM = 5.5.
    above = [c for c in cases if inp("frh2o", c, "bexp", 1) > 5.5]
    below = [c for c in cases if inp("frh2o", c, "bexp", 1) <= 5.5]
    if not above or not below:
        raise ThermalOracleError(
            f"frh2o: BEXP must straddle BLIM = 5.5 (5866); "
            f"above={above} below={below}")

    # 5883: SWL = SMC - SH2O > SMC - 0.02 needs SH2O < 0.02.
    if not [c for c in cases
            if inp("frh2o", c, "tkelv", 0) <= tfrz_eps
            and inp("frh2o", c, "sh2o", 0) < 0.02]:
        raise ThermalOracleError(
            "frh2o: no cold case has SH2O < 0.02, so the initial SWL clamp "
            "at 5883 is unreached")

    # 5887: SWL < 0 after that clamp needs SMC < 0.02.
    if not [c for c in cases
            if inp("frh2o", c, "tkelv", 0) <= tfrz_eps
            and inp("frh2o", c, "smc", 0) < 0.02]:
        raise ThermalOracleError(
            "frh2o: no cold case has SMC < 0.02, so the SWL < 0 clamp at "
            "5887 is unreached")

    # 5899 / 5900: the two in-loop clamps land FREE on SMC - (SMC-0.02) and
    # on SMC - 0 respectively, which is how they are visible from outside.
    if not [c for c in cases
            if inp("frh2o", c, "tkelv", 0) <= tfrz_eps
            and out("frh2o", c, "free", 0) == _r32(0.02)]:
        raise ThermalOracleError(
            "frh2o: no case leaves SWL at the SMC-0.02 clamp (5899)")
    if not [c for c in cases
            if inp("frh2o", c, "tkelv", 0) <= tfrz_eps
            and out("frh2o", c, "free", 0) == inp("frh2o", c, "smc", 0)]:
        raise ThermalOracleError(
            "frh2o: no cold case leaves SWL at the 0.0 clamp (5900)")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    leaves_path, disc_path = Path(argv[1]), Path(argv[2])
    try:
        table = load_leaves(leaves_path)
        check_structure(table)
        nvalues = check_bits(table)
        counts = check_discrimination(disc_path)
        check_branch_coverage(table)
    except ThermalOracleError as error:
        print(f"thermal oracle validation FAILED: {error}", file=sys.stderr)
        return 1
    total_cases = sum(len(spec["cases"]) for spec in LEAVES.values())
    print(f"thermal oracle ok: {len(LEAVES)} leaves, {total_cases} cases, "
          f"{nvalues} pinned FP32 values, "
          f"{sum(counts.values())} discriminating input probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
