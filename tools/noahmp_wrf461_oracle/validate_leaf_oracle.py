#!/usr/bin/env python3
"""Validate the Noah-MP per-leaf oracle fixture and its discrimination sweep.

This is the gate that stops a leaf fixture from being a test that cannot fail.
It checks four independent things:

1. **Structure.** Every declared leaf is present with the declared case list
   and the declared input/output slot counts.  A renamed or dropped case is a
   hard failure, not a silently smaller fixture.
2. **Bit fidelity.** The ``bits`` column round-trips to the ``value`` column
   exactly, so downstream ``max_ulp 0`` comparison can use ``bits`` alone.
3. **Undefined-slot convention.** Every ``live=0`` output slot carries the
   harness's 0.0 pre-fill, so the CPU/CUDA ports can adopt the same
   convention deterministically.
4. **Discrimination.** For every input slot of every leaf, zeroing that slot
   must move at least one live output in at least one case where the slot was
   not already zero -- unless the slot is in that leaf's declared inert set,
   in which case it must move nothing anywhere.  The inert sets below are the
   executable statement of which WRF arguments this option identity does not
   consume; each carries the source line that makes it inert.

Usage:
    validate_leaf_oracle.py LEAVES.csv DISCRIMINATION.csv
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


# leaf -> (cases, n_int, n_in, n_out, inert{(name, index): reason})
LEAVES: dict[str, dict] = {
    "atm": {
        "cases": (
            "warm_dry_noon", "warm_rain_afternoon", "mixed_phase_lower",
            "mixed_phase_upper", "cold_all_snow", "near_freezing_snow",
            "nocturnal_dark", "hot_bright_convective",
        ),
        "n_int": 0, "n_in": 11, "n_out": 17,
        "inert": {
            ("prcpsnow", 0): "read only inside IF(OPT_SNF==4) at 1217-1228",
            ("prcpgrpl", 0): "read only inside IF(OPT_SNF==4) at 1217-1228",
            ("prcphail", 0): "read only inside IF(OPT_SNF==4) at 1217-1228",
        },
    },
    "esat": {
        "cases": (
            "clamp_low_minus50", "deep_cold_minus28", "frost_minus7",
            "freezing_zero", "cool_plus6", "warm_plus27",
            "clamp_high_plus50",
        ),
        "n_int": 0, "n_in": 1, "n_out": 4, "inert": {},
    },
    "rosr12": {
        "cases": (
            "soil_only_isnow0", "one_snow_layer", "two_snow_layers",
            "three_snow_layers", "three_snow_layers_alt",
        ),
        "n_int": 1, "n_in": 28, "n_out": 21,
        "inert": {
            ("a", -2): "A is read only for K > NTOP (line 5575); "
                       "min NTOP over the fixture is -2",
            ("c", 4): "C(NSOIL) is overwritten with 0.0 at line 5565 "
                      "before any read of C",
        },
    },
    "csnow": {
        "cases": (
            "snow_free_isnow0", "single_thin_layer", "two_layer_ripe",
            "three_layer_cold", "ice_saturated_clamp", "three_layer_dense",
        ),
        "n_int": 1, "n_in": 9, "n_out": 15, "inert": {},
        # ISNOW only sets the ISNOW+1..0 loop bound at 2547, 2553 and 2561.
        # Every output slot at or above ISNOW is INTENT(OUT)-undefined, and the
        # per-layer formulas are independent, so a CSNOW that computes all
        # NSNOW layers unconditionally is indistinguishable from the real one
        # on every *defined* output.  Narrowing mutants (freeze to 0, -1, -2)
        # are killed because they leave the harness pre-fill in live slots;
        # widening mutants to the fixture's deepest ISNOW are not.  ISNOW's
        # detectability therefore belongs to CSNOW's callers -- THERMOPROP
        # reads it again at 2454 and 2497 -- and thermoprop's own mutation
        # study is clean on it.
        "partial": {
            ("isnow", 0): (
                "loop bound only; outputs above ISNOW are INTENT(OUT)-"
                "undefined (2547, 2553, 2561)",
                ("freeze:three_layer_cold", "freeze:three_layer_dense"),
            ),
        },
    },
    "tdfcnd": {
        "cases": (
            "wet_unfrozen_loam", "dry_below_kersten", "partly_frozen",
            "bone_dry_zero_smc", "saturated_loam", "quartz_rich_sand",
            "clay_wilting",
        ),
        "n_int": 0, "n_in": 4, "n_out": 1, "inert": {},
    },
    "thermoprop": {
        "cases": (
            "bare_soil_wet", "bare_soil_frozen", "one_snow_layer",
            "two_snow_layers", "three_snow_layers",
            "three_snow_layers_dense", "urban_override",
            "lake_thawed_column", "lake_frozen_column", "lake_mixed_column",
        ),
        "n_int": 4, "n_in": 44, "n_out": 30,
        "inert": {
            ("tg", 0): "declared INTENT(IN) at 2422, never referenced",
            ("ur", 0): "declared INTENT(IN) at 2424, never referenced",
            ("lat", 0): "declared INTENT(IN) at 2425, never referenced",
            ("z0m", 0): "declared INTENT(IN) at 2426, never referenced",
            ("zlvl", 0): "declared INTENT(IN) at 2427, never referenced",
            ("vegtyp", 0): "declared INTENT(IN) at 2428, never referenced",
            ("stc", -2): "STC is read only as STC(IZ), IZ=1..NSOIL, inside "
                         "the IST==2 branch at 2483-2493",
            ("stc", -1): "STC is read only as STC(IZ), IZ=1..NSOIL, inside "
                         "the IST==2 branch at 2483-2493",
            ("stc", 0): "STC is read only as STC(IZ), IZ=1..NSOIL, inside "
                        "the IST==2 branch at 2483-2493",
        },
    },
    "wdfcnd1": {
        "cases": (
            "loam_unfrozen", "loam_partly_impermeable", "fully_impermeable",
            "dry_factr_clamp", "saturated", "sand_unfrozen",
        ),
        "n_int": 0, "n_in": 6, "n_out": 2, "inert": {},
    },
    "wdfcnd2": {
        "cases": (
            "ice_free", "trace_ice_vkwgt_high", "heavy_ice_vkwgt_low",
            "dry_factr_clamp", "saturated_ice_free", "sand_moderate_ice",
        ),
        "n_int": 0, "n_in": 6, "n_out": 2, "inert": {},
    },
}


class LeafOracleError(RuntimeError):
    pass


def load_leaves(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise LeafOracleError(f"{path} is empty")
    table: dict[tuple[str, str, str], dict] = defaultdict(dict)
    for row in rows:
        key = (row["leaf"], row["case"], row["role"])
        table[key][int(row["slot"])] = row
    return table


def check_structure(table) -> None:
    seen_leaves = {key[0] for key in table}
    if seen_leaves != set(LEAVES):
        raise LeafOracleError(
            f"leaf set mismatch: extra={sorted(seen_leaves - set(LEAVES))}, "
            f"missing={sorted(set(LEAVES) - seen_leaves)}")
    for leaf, spec in LEAVES.items():
        cases = {key[1] for key in table if key[0] == leaf}
        if cases != set(spec["cases"]):
            raise LeafOracleError(
                f"{leaf}: case mismatch extra={sorted(cases - set(spec['cases']))} "
                f"missing={sorted(set(spec['cases']) - cases)}")
        for case in spec["cases"]:
            for role, count in (("int", spec["n_int"]), ("in", spec["n_in"]),
                                ("out", spec["n_out"])):
                got = len(table.get((leaf, case, role), {}))
                if got != count:
                    raise LeafOracleError(
                        f"{leaf}/{case}: role {role} has {got} slots, "
                        f"expected {count}")


def check_bits(table) -> int:
    checked = 0
    for (leaf, case, role), slots in table.items():
        for slot, row in slots.items():
            from_bits = _f32(row["bits"])
            printed = float(row["value"])
            if struct.pack("<f", printed) != struct.pack("<f", from_bits):
                raise LeafOracleError(
                    f"{leaf}/{case}/{role}[{slot}]: bits {row['bits']} "
                    f"({from_bits!r}) does not round-trip the printed value "
                    f"{row['value']}")
            if role == "out" and row["live"] == "1":
                if not math.isfinite(from_bits):
                    raise LeafOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is "
                        f"non-finite: {from_bits}")
            if role == "out" and row["live"] == "0":
                if row["bits"] != "00000000":
                    raise LeafOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is flagged "
                        f"live=0 but carries {row['bits']}, not the 0.0 "
                        f"pre-fill")
            checked += 1
    return checked


def check_discrimination(disc_path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(disc_path.open(newline="")))
    if not rows:
        raise LeafOracleError(f"{disc_path} is empty")
    # (leaf, slot) -> [(name, index, already_zero, nchanged)]
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
            raise LeafOracleError(
                f"{leaf}: input slot {slot} ({name}[{index}]) is zero in "
                f"every case, so the fixture cannot tell whether the port "
                f"reads it. Give at least one case a non-zero value.")
        if declared_inert and moved:
            raise LeafOracleError(
                f"{leaf}: {name}[{index}] is declared inert "
                f"({spec['inert'][(name, index)]}) but zeroing it moved "
                f"{moved[0][3]} outputs")
        if not declared_inert and not moved:
            raise LeafOracleError(
                f"{leaf}: zeroing {name}[{index}] moves no live output in "
                f"any of {len(entries)} cases, and it is not declared inert. "
                f"Either the fixture cannot detect a dropped variable or the "
                f"argument really is dead -- decide and record which.")
        counts[leaf] += 1
    for leaf, spec in LEAVES.items():
        if counts[leaf] != spec["n_in"]:
            raise LeafOracleError(
                f"{leaf}: discrimination sweep covered {counts[leaf]} of "
                f"{spec['n_in']} input slots")
    return counts


def check_branch_coverage(table) -> None:
    """A handful of explicit branch assertions the fixture must satisfy."""
    def out(leaf, case, name, index):
        for row in table[(leaf, case, "out")].values():
            if row["name"] == name and int(row["index"]) == index:
                return _f32(row["bits"])
        raise LeafOracleError(f"{leaf}/{case}: no output {name}[{index}]")

    def inp(leaf, case, name, index):
        for row in table[(leaf, case, "in")].values():
            if row["name"] == name and int(row["index"]) == index:
                return _f32(row["bits"])
        raise LeafOracleError(f"{leaf}/{case}: no input {name}[{index}]")

    # ATM: every Jordan-91 FPICE branch is reached.
    fpice = {c: out("atm", c, "fpice", 0) for c in LEAVES["atm"]["cases"]}
    if fpice["warm_dry_noon"] != 0.0:
        raise LeafOracleError("atm: expected FPICE=0 above TFRZ+2.5")
    if fpice["near_freezing_snow"] != 1.0:
        raise LeafOracleError("atm: expected FPICE=1 below TFRZ+0.5")
    if not 0.0 < fpice["mixed_phase_lower"] < 1.0:
        raise LeafOracleError("atm: the 1190 ramp branch was not reached")
    if abs(fpice["mixed_phase_upper"] - 0.6) > 1e-6:
        raise LeafOracleError("atm: the FPICE=0.6 shelf was not reached")
    if out("atm", "nocturnal_dark", "swdown", 0) != 0.0:
        raise LeafOracleError("atm: the COSZ<=0 branch was not reached")

    if out("atm", "nocturnal_dark", "solad", 1) != 0.0 \
            or inp("atm", "nocturnal_dark", "soldn", 0) <= 0.0:
        raise LeafOracleError(
            "atm: the nocturnal case must carry SOLDN>0 with COSZ<0, "
            "otherwise the COSZ<=0 branch is indistinguishable from its "
            "complement and no COSZ mutant can be killed")

    # TDFCND: every AKE branch at 2654-2672 must be selected by some case, and
    # the selection is asserted from the *inputs* so it cannot be faked.
    frozen, kersten, dry, saturated = [], [], [], []
    for case in LEAVES["tdfcnd"]["cases"]:
        smc = inp("tdfcnd", case, "smc", 0)
        sh2o = inp("tdfcnd", case, "sh2o", 0)
        smcmax = inp("tdfcnd", case, "smcmax", 1)
        satratio = smc / smcmax
        if sh2o + 0.0005 < smc:
            frozen.append(case)
        elif satratio > 0.1:
            kersten.append(case)
            if abs(satratio - 1.0) < 1e-6:
                saturated.append(case)
        else:
            dry.append(case)
    if not frozen or not kersten or not dry or not saturated:
        raise LeafOracleError(
            "tdfcnd: AKE branch coverage incomplete "
            f"(frozen={frozen}, kersten={kersten}, dry={dry}, "
            f"saturated={saturated})")
    for case in dry:
        smcmax = inp("tdfcnd", case, "smcmax", 1)
        gammd = (1.0 - smcmax) * 2700.0
        thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
        if abs(out("tdfcnd", case, "df", 0) - thkdry) > 2e-6 * abs(thkdry):
            raise LeafOracleError(
                f"tdfcnd/{case}: AKE=0 must collapse DF to THKDRY")

    # CSNOW: the MIN(1.0, ...) ice-volume clamp and its EPORE consequence.
    if out("csnow", "ice_saturated_clamp", "snicev", -1) != 1.0:
        raise LeafOracleError("csnow: SNICEV clamp at 2548 was not reached")
    if out("csnow", "ice_saturated_clamp", "epore", -1) != 0.0:
        raise LeafOracleError("csnow: EPORE=0 consequence was not reached")

    # THERMOPROP: the urban override at 2470 (DF(1) is then re-blended by the
    # ISNOW==0 interface rule at 2504, so only layers 2..4 stay at 3.24).
    for iz in (2, 3, 4):
        if abs(out("thermoprop", "urban_override", "df", iz) - 3.24) > 1e-6:
            raise LeafOracleError(
                f"thermoprop: urban DF({iz}) override at 2470 not reached")
    if abs(out("thermoprop", "urban_override", "df", 1) - 3.24) <= 1e-6:
        raise LeafOracleError(
            "thermoprop: DF(1) should be re-blended at 2504 after the urban "
            "override, so it must not remain exactly 3.24")
    # Both IST==2 branches, as whole columns, plus a mixed column.  The pair of
    # whole columns is what makes STC(1:4) mutation-detectable.
    # DF(1) is always re-blended by the interface rule at 2503-2507, so only
    # layers 2..4 keep the raw TKWAT/TKICE the lake branch assigned.
    for iz in (2, 3, 4):
        if abs(out("thermoprop", "lake_thawed_column", "df", iz) - 0.6) > 1e-6:
            raise LeafOracleError(
                f"thermoprop: lake_thawed_column DF({iz}) must be TKWAT")
        if abs(out("thermoprop", "lake_frozen_column", "df", iz) - 2.2) > 1e-6:
            raise LeafOracleError(
                f"thermoprop: lake_frozen_column DF({iz}) must be TKICE")
    if abs(out("thermoprop", "lake_mixed_column", "df", 2) - 2.2) > 1e-6 \
            or abs(out("thermoprop", "lake_mixed_column", "df", 4) - 2.2) > 1e-6:
        raise LeafOracleError(
            "thermoprop: lake_mixed_column must freeze layers 2 and 4")
    # The ISNOW==0 snow/soil interface blend at 2504 needs a non-zero SNOWH.
    if inp("thermoprop", "bare_soil_frozen", "snowh", 0) <= 0.0:
        raise LeafOracleError(
            "thermoprop: no ISNOW==0 case has SNOWH>0, so the 2504 blend is "
            "exercised only in its degenerate SNOWH=0 form")

    # WDFCND2: the SICE>0 VKWGT branch must reduce WDF below the ice-free
    # formula, because FACTR1 <= FACTR2 and the exponent is positive.  When
    # the soil is dry enough that FACTR2 hits the 0.01 clamp, FACTR1 == FACTR2
    # and the blend is a no-op up to rounding, so that case is excluded.
    ice_branch: list[str] = []
    for case in LEAVES["wdfcnd2"]["cases"]:
        sice = inp("wdfcnd2", case, "sice", 0)
        smc = inp("wdfcnd2", case, "smc", 0)
        smcmax = inp("wdfcnd2", case, "smcmax", 1)
        bexp = inp("wdfcnd2", case, "bexp", 1)
        dwsat = inp("wdfcnd2", case, "dwsat", 1)
        factr1 = min(0.05 / smcmax, max(0.01, smc / smcmax))
        factr2 = max(0.01, smc / smcmax)
        ice_free = dwsat * factr2 ** (bexp + 2.0)
        wdf = out("wdfcnd2", case, "wdf", 0)
        if sice > 0.0 and factr1 < factr2:
            if wdf >= ice_free * (1.0 - 1e-5):
                raise LeafOracleError(
                    f"wdfcnd2/{case}: the SICE>0 branch at 9222 did not "
                    f"change WDF ({wdf} vs ice-free {ice_free})")
            ice_branch.append(case)
        elif sice == 0.0 and abs(wdf - ice_free) > 2e-6 * ice_free:
            raise LeafOracleError(
                f"wdfcnd2/{case}: ice-free WDF disagrees with 9220")
    if not ice_branch:
        raise LeafOracleError(
            "wdfcnd2: no case reaches the SICE>0 blend with FACTR1 < FACTR2, "
            "so the VKWGT weighting at 9224 is never actually exercised")

    # WDFCND1: FCR=1 must zero both outputs.
    if out("wdfcnd1", "fully_impermeable", "wdf", 0) != 0.0 or \
            out("wdfcnd1", "fully_impermeable", "wcnd", 0) != 0.0:
        raise LeafOracleError("wdfcnd1: FCR=1 must zero WDF and WCND")


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
    except LeafOracleError as error:
        print(f"leaf oracle validation FAILED: {error}", file=sys.stderr)
        return 1
    total_cases = sum(len(spec["cases"]) for spec in LEAVES.values())
    print(f"leaf oracle ok: {len(LEAVES)} leaves, {total_cases} cases, "
          f"{nvalues} pinned FP32 values, "
          f"{sum(counts.values())} discriminating input probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
