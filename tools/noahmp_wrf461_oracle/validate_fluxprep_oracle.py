#!/usr/bin/env python3
"""Validate the SFCDIF1 / RAGRB / STOMATA oracle fixture and its sweep.

Sibling of ``validate_leaf_oracle.py`` for the flux-preparation leaves, with
the same four gates:

1. **Structure.** Every declared leaf is present with the declared case list
   and the declared input/output slot counts.
2. **Bit fidelity.** The ``bits`` column round-trips to the ``value`` column
   exactly, and every live output is finite.
3. **Undefined-slot convention.** Every ``live=0`` output slot carries the
   harness's 0.0 pre-fill.  (These three leaves have no undefined outputs:
   every output slot is written on every path, so ``live`` is 1 throughout and
   this gate is vacuous but kept so the schema stays shared.)
4. **Discrimination.** For every FP32 input slot, substituting the probe value
   must move at least one live output in at least one case where the slot did
   not already carry that value -- unless the slot is declared inert, in which
   case it must move nothing anywhere.  Each inert entry carries the WRF line
   number that makes it inert.  The probe is 0.0 everywhere except SFCDIF1's
   ZLVL, where 0.0 would satisfy `IF(ZLVL <= ZPD)` at :4650 and call
   wrf_error_fatal; that one substitution is declared in ``PROBES`` with its
   reason and the CSV carries the probe value used.

Plus a fifth that is specific to this file: **branch coverage**, asserted from
the *inputs*.  Where the branch cannot be read off one input directly, the
selector is recomputed here in FP64 from the inputs alone.  Those replicas
decide booleans only -- never values -- and the fixture's cases are placed with
margin on each side, so an FP64/FP32 disagreement cannot flip one.

Usage:
    validate_fluxprep_oracle.py FLUXPREP.csv DISCRIMINATION.csv
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


# WRF module parameters, module_sf_noahmplsm.F:204-220.
GRAV = 9.80616
VKC = 0.40
TFRZ = 273.16
CPAIR = 1004.64

RAGRB_CASES = (
    "neutral_forest", "unstable_forest", "stable_grass", "stable_clamp_mozg",
    "mpe_floors_tmp1", "mpe_floors_kh", "rb_high_clamp", "rb_low_clamp",
)
SFCDIF1_CASES = (
    "neutral_start", "unstable_moderate", "stable_moderate",
    "sign_flip_counts", "sign_flip_resets", "mozsgn_already_two",
    "mpe_floors_tmp1", "moz_clamped_to_one", "clamps_bind",
    "degenerate_guards",
)
STOMATA_CASES = (
    "forest_wj_limited", "dry_canopy_ci_floor", "crop_we_limited",
    "c4_pathway",
    "dark_early_return", "fveg_clamp_folnmx_zero", "highland_crop_b_nonneg",
    "co2_starved_fine", "co2_starved_coarse", "igs_zero_dormant",
)

# leaf -> (cases, n_int, n_in, n_out, inert{(name, index): reason})
LEAVES: dict[str, dict] = {
    "ragrb": {
        "cases": RAGRB_CASES,
        "n_int": 4, "n_in": 17, "n_out": 6,
        "inert": {
            ("tv", 0): "INTENT(IN) at 4507, never referenced in the body",
            ("mozg", 0): "INTENT(INOUT) at 4519 but assigned 0.0 at 4536 "
                         "before any read",
            ("vegtyp", 0): "INTENT(IN) at 4500, never referenced in the body",
            ("iloc", 0): "INTENT(IN) at 4498, never referenced in the body",
            ("jloc", 0): "INTENT(IN) at 4499, never referenced in the body",
        },
    },
    "sfcdif1": {
        "cases": SFCDIF1_CASES,
        "n_int": 4, "n_in": 16, "n_out": 10,
        "inert": {
            ("iloc", 0): "INTENT(IN) at 4595, never referenced in the body",
            ("jloc", 0): "INTENT(IN) at 4596, never referenced in the body",
        },
    },
    "stomata": {
        "cases": STOMATA_CASES,
        "n_int": 3, "n_in": 25, "n_out": 2,
        "inert": {
            ("vegtyp", 0): "INTENT(IN) at 5013, never referenced in the body",
            ("iloc", 0): "INTENT(IN) at 5011, never referenced in the body",
            ("jloc", 0): "INTENT(IN) at 5012, never referenced in the body",
        },
    },
}


# Input slots probed with something other than 0.0, and why.
PROBES: dict[str, dict[tuple[str, int], tuple[float, str]]] = {
    "sfcdif1": {
        ("zlvl", 0): (
            3.0,
            "0.0 would satisfy IF(ZLVL <= ZPD) at 4650 and call "
            "wrf_error_fatal; 3.0 m is above every ZPD in the case table and "
            "equal to no case's own ZLVL",
        ),
    },
}


class FluxprepOracleError(RuntimeError):
    pass


def load_leaves(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise FluxprepOracleError(f"{path} is empty")
    table: dict[tuple[str, str, str], dict] = defaultdict(dict)
    for row in rows:
        key = (row["leaf"], row["case"], row["role"])
        table[key][int(row["slot"])] = row
    return table


def check_structure(table) -> None:
    seen_leaves = {key[0] for key in table}
    if seen_leaves != set(LEAVES):
        raise FluxprepOracleError(
            f"leaf set mismatch: extra={sorted(seen_leaves - set(LEAVES))}, "
            f"missing={sorted(set(LEAVES) - seen_leaves)}")
    for leaf, spec in LEAVES.items():
        cases = {key[1] for key in table if key[0] == leaf}
        if cases != set(spec["cases"]):
            raise FluxprepOracleError(
                f"{leaf}: case mismatch "
                f"extra={sorted(cases - set(spec['cases']))} "
                f"missing={sorted(set(spec['cases']) - cases)}")
        for case in spec["cases"]:
            for role, count in (("int", spec["n_int"]), ("in", spec["n_in"]),
                                ("out", spec["n_out"])):
                got = len(table.get((leaf, case, role), {}))
                if got != count:
                    raise FluxprepOracleError(
                        f"{leaf}/{case}: role {role} has {got} slots, "
                        f"expected {count}")


def check_bits(table) -> int:
    checked = 0
    for (leaf, case, role), slots in table.items():
        for slot, row in slots.items():
            from_bits = _f32(row["bits"])
            printed = float(row["value"])
            if struct.pack("<f", printed) != struct.pack("<f", from_bits):
                raise FluxprepOracleError(
                    f"{leaf}/{case}/{role}[{slot}]: bits {row['bits']} "
                    f"({from_bits!r}) does not round-trip the printed value "
                    f"{row['value']}")
            if role == "out" and row["live"] == "1":
                if not math.isfinite(from_bits):
                    raise FluxprepOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is "
                        f"non-finite: {from_bits}")
            if role == "out" and row["live"] == "0":
                if row["bits"] != "00000000":
                    raise FluxprepOracleError(
                        f"{leaf}/{case}/out[{slot}] {row['name']} is flagged "
                        f"live=0 but carries {row['bits']}, not the 0.0 "
                        f"pre-fill")
            checked += 1
    return checked


def check_discrimination(disc_path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(disc_path.open(newline="")))
    if not rows:
        raise FluxprepOracleError(f"{disc_path} is empty")
    probes: dict[tuple[str, int], list] = defaultdict(list)
    for row in rows:
        probes[(row["leaf"], int(row["slot"]))].append(
            (row["name"], int(row["index"]), int(row["already_at_probe"]),
             int(row["noutputs_changed"]), _f32(row["probe_bits"])))

    counts: dict[str, int] = defaultdict(int)
    for (leaf, slot), entries in sorted(probes.items()):
        spec = LEAVES[leaf]
        name, index = entries[0][0], entries[0][1]
        declared = PROBES.get(leaf, {}).get((name, index))
        used = {e[4] for e in entries}
        if len(used) != 1:
            raise FluxprepOracleError(
                f"{leaf}: {name}[{index}] was probed with more than one "
                f"value {sorted(used)}")
        value = used.pop()
        if value != 0.0 and declared is None:
            raise FluxprepOracleError(
                f"{leaf}: {name}[{index}] was probed with {value} rather than "
                f"0.0 and that substitution is not declared in PROBES")
        if declared is not None and declared[0] != value:
            raise FluxprepOracleError(
                f"{leaf}: {name}[{index}] declares probe {declared[0]} but "
                f"the sweep used {value}")
        declared_inert = (name, index) in spec["inert"]
        informative = [e for e in entries if e[2] == 0]
        moved = [e for e in entries if e[3] > 0]
        if not informative:
            raise FluxprepOracleError(
                f"{leaf}: input slot {slot} ({name}[{index}]) already equals "
                f"the probe value in every case, so the fixture cannot tell "
                f"whether the port reads it. Give at least one case a "
                f"different value.")
        if declared_inert and moved:
            raise FluxprepOracleError(
                f"{leaf}: {name}[{index}] is declared inert "
                f"({spec['inert'][(name, index)]}) but probing it moved "
                f"{moved[0][3]} outputs")
        if not declared_inert and not moved:
            raise FluxprepOracleError(
                f"{leaf}: probing {name}[{index}] moves no live output in "
                f"any of {len(entries)} cases, and it is not declared inert. "
                f"Either the fixture cannot detect a dropped variable or the "
                f"argument really is dead -- decide and record which.")
        counts[leaf] += 1
    for leaf, spec in LEAVES.items():
        if counts[leaf] != spec["n_in"]:
            raise FluxprepOracleError(
                f"{leaf}: discrimination sweep covered {counts[leaf]} of "
                f"{spec['n_in']} input slots")
    return counts


# ---------------------------------------------------------------------------
# FP64 branch selectors.  Booleans only.
# ---------------------------------------------------------------------------

def _ragrb_flags(iter_, v):
    """Which branches RAGRB (4483-4579) takes, from the inputs alone."""
    flags = {"iter_gt_1": iter_ > 1, "tmp1_floor": False,
             "mozg_clamped": False, "mozg_negative": False,
             "kh_floor": False, "rb_low": False, "rb_high": False}
    mozg = 0.0
    fhg = v["fhg"]
    if iter_ > 1:
        tmp1 = VKC * (GRAV / v["tah"]) * v["hg"] / (v["rhoair"] * CPAIR)
        if abs(tmp1) <= v["mpe"]:
            tmp1 = v["mpe"]
            flags["tmp1_floor"] = True
        molg = -(v["fv"] ** 3) / tmp1
        raw = (v["zpd"] - v["z0mg"]) / molg
        mozg = min(raw, 1.0)
        flags["mozg_clamped"] = raw > 1.0
    flags["mozg_negative"] = mozg < 0.0
    if mozg < 0.0:
        fhgnew = (1.0 - 15.0 * mozg) ** (-0.25)
    else:
        fhgnew = 1.0 + 4.7 * mozg
    fhg = fhgnew if iter_ == 1 else 0.5 * (fhg + fhgnew)
    cwpc = (v["cwp"] * v["vai"] * v["hcan"] * fhg) ** 0.5
    flags["kh_floor"] = VKC * v["fv"] * (v["hcan"] - v["zpd"]) < v["mpe"]
    tmprb = cwpc * 50.0 / (1.0 - math.exp(-cwpc / 2.0))
    rb = tmprb * math.sqrt(v["dleaf"] / v["uc"])
    flags["rb_low"] = rb < 5.0
    flags["rb_high"] = rb > 50.0
    return flags


def _sfcdif1_flags(iter_, mozsgn, v):
    """Which branches SFCDIF1 (4583-4743) takes, from the inputs alone."""
    flags = {"iter_gt_1": iter_ > 1, "tmp1_floor": False,
             "moz_clamped": False, "sign_flip": False, "reset": False,
             "moz_negative": False, "clamps_bind": False,
             "guards_fire": False}
    mozold = v["moz"]
    tmpcm = math.log((v["zlvl"] - v["zpd"]) / v["z0m"])
    tmpch = math.log((v["zlvl"] - v["zpd"]) / v["z0h"])
    tmpcm2 = math.log((2.0 + v["z0m"]) / v["z0m"])
    tmpch2 = math.log((2.0 + v["z0h"]) / v["z0h"])
    if iter_ == 1:
        moz = moz2 = 0.0
    else:
        tvir = (1.0 + 0.61 * v["qair"]) * v["sfctmp"]
        tmp1 = VKC * (GRAV / tvir) * v["h"] / (v["rhoair"] * CPAIR)
        if abs(tmp1) <= v["mpe"]:
            tmp1 = v["mpe"]
            flags["tmp1_floor"] = True
        mol = -(v["fv"] ** 3) / tmp1
        raw = (v["zlvl"] - v["zpd"]) / mol
        raw2 = (2.0 + v["z0h"]) / mol
        moz, moz2 = min(raw, 1.0), min(raw2, 1.0)
        flags["moz_clamped"] = raw > 1.0 and raw2 > 1.0
    fm, fh, fm2, fh2 = v["fm"], v["fh"], v["fm2"], v["fh2"]
    if mozold * moz < 0.0:
        mozsgn += 1
        flags["sign_flip"] = True
    if mozsgn >= 2:
        moz = moz2 = 0.0
        fm = fh = fm2 = fh2 = 0.0
        flags["reset"] = True
    flags["moz_negative"] = moz < 0.0
    if moz < 0.0:
        t1 = (1.0 - 16.0 * moz) ** 0.25
        t2 = math.log((1.0 + t1 * t1) / 2.0)
        t3 = math.log((1.0 + t1) / 2.0)
        fmnew = 2.0 * t3 + t2 - 2.0 * math.atan(t1) + 1.5707963
        fhnew = 2.0 * t2
        t12 = (1.0 - 16.0 * moz2) ** 0.25
        t22 = math.log((1.0 + t12 * t12) / 2.0)
        t32 = math.log((1.0 + t12) / 2.0)
        fm2new = 2.0 * t32 + t22 - 2.0 * math.atan(t12) + 1.5707963
        fh2new = 2.0 * t22
    else:
        fmnew = -5.0 * moz
        fhnew = fmnew
        fm2new = -5.0 * moz2
        fh2new = fm2new
    if iter_ == 1:
        fm, fh, fm2, fh2 = fmnew, fhnew, fm2new, fh2new
    else:
        fm = 0.5 * (fm + fmnew)
        fh = 0.5 * (fh + fhnew)
        fm2 = 0.5 * (fm2 + fm2new)
        fh2 = 0.5 * (fh2 + fh2new)
    binds = (fh > 0.9 * tmpch, fm > 0.9 * tmpcm,
             fh2 > 0.9 * tmpch2, fm2 > 0.9 * tmpcm2)
    flags["clamps_bind"] = all(binds)
    fh, fm = min(fh, 0.9 * tmpch), min(fm, 0.9 * tmpcm)
    fh2, fm2 = min(fh2, 0.9 * tmpch2), min(fm2, 0.9 * tmpcm2)
    diffs = (tmpcm - fm, tmpch - fh, tmpcm2 - fm2, tmpch2 - fh2)
    flags["guards_fire"] = all(abs(d) <= v["mpe"] for d in diffs)
    return flags


def _stomata_flags(v):
    """Which branches STOMATA (5005-5137) takes, from the inputs alone."""
    flags = {"early_return": False, "fveg_clamp": v["fveg"] < 1.0e-6,
             "folnmx_floor": v["folnmx"] < v["mpe"], "fnf_clamped": False,
             "cea_from_ei": False, "cea_from_ea": False,
             "cea_from_quarter_ei": False,
             "wj_min": False, "wc_min": False, "we_min": False,
             "cs_floor": False, "b_nonneg": False, "b_negative": False,
             "ci_cp_clamp": False, "ci_zero_clamp": False}
    apar_scale = v["apar"] / max(v["fveg"], 1.0e-6)
    cf = v["sfcprs"] / (8.314 * v["sfctmp"]) * 1.0e6
    if apar_scale <= 0.0:
        flags["early_return"] = True
        return flags
    fnf_raw = v["foln"] / max(v["mpe"], v["folnmx"])
    flags["fnf_clamped"] = fnf_raw >= 1.0
    fnf = min(fnf_raw, 1.0)
    tc = v["tv"] - TFRZ
    ppf = 4.6 * apar_scale
    j = ppf * v["qe25"]
    kc = v["kc25"] * v["akc"] ** ((tc - 25.0) / 10.0)
    ko = v["ko25"] * v["ako"] ** ((tc - 25.0) / 10.0)
    awc = kc * (1.0 + v["o2"] / ko)
    cp = 0.5 * kc / ko * v["o2"] * 0.21
    f2 = 1.0 + math.exp((-2.2e05 + 710.0 * (tc + TFRZ)) / (8.314 * (tc + TFRZ)))
    vcmx = v["vcmx25"] / f2 * fnf * v["btran"] * v["avcmx"] ** ((tc - 25.0) / 10.0)
    c3 = v["c3psn"]
    ci = 0.7 * v["co2"] * c3 + 0.4 * v["co2"] * (1.0 - c3)
    rlb = v["rb"] / cf
    quarter = 0.25 * v["ei"] * c3 + 0.40 * v["ei"] * (1.0 - c3)
    inner = min(v["ea"], v["ei"])
    cea = max(quarter, inner)
    flags["cea_from_quarter_ei"] = quarter > inner
    flags["cea_from_ei"] = (not flags["cea_from_quarter_ei"]) and v["ea"] >= v["ei"]
    flags["cea_from_ea"] = (not flags["cea_from_quarter_ei"]) and v["ea"] < v["ei"]
    for _ in range(3):
        clipped = max(ci - cp, 0.0)
        if clipped == 0.0:
            flags["ci_cp_clamp"] = True
        wj = clipped * j / (ci + 2.0 * cp) * c3 + j * (1.0 - c3)
        wc = clipped * vcmx / (ci + awc) * c3 + vcmx * (1.0 - c3)
        we = 0.5 * vcmx * c3 + 4000.0 * vcmx * ci / v["sfcprs"] * (1.0 - c3)
        smallest = min(wj, wc, we)
        if smallest == wj:
            flags["wj_min"] = True
        elif smallest == wc:
            flags["wc_min"] = True
        else:
            flags["we_min"] = True
        psn = smallest * v["igs"]
        cs_raw = v["co2"] - 1.37 * rlb * v["sfcprs"] * psn
        if cs_raw < v["mpe"]:
            flags["cs_floor"] = True
        cs = max(cs_raw, v["mpe"])
        a = v["mp"] * psn * v["sfcprs"] * cea / (cs * v["ei"]) + v["bp"]
        b = (v["mp"] * psn * v["sfcprs"] / cs + v["bp"]) * rlb - 1.0
        c = -rlb
        if b >= 0.0:
            flags["b_nonneg"] = True
            q = -0.5 * (b + math.sqrt(b * b - 4.0 * a * c))
        else:
            flags["b_negative"] = True
            q = -0.5 * (b - math.sqrt(b * b - 4.0 * a * c))
        rs = max(q / a, c / q)
        ci_raw = cs - psn * v["sfcprs"] * 1.65 * rs
        if ci_raw < 0.0:
            flags["ci_zero_clamp"] = True
        ci = max(ci_raw, 0.0)
    return flags


def _inputs(table, leaf, case):
    return {row["name"]: _f32(row["bits"])
            for row in table[(leaf, case, "in")].values()}


def _ints(table, leaf, case):
    return {row["name"]: int(float(row["value"]))
            for row in table[(leaf, case, "int")].values()}


def _out(table, leaf, case, name):
    for row in table[(leaf, case, "out")].values():
        if row["name"] == name:
            return _f32(row["bits"])
    raise FluxprepOracleError(f"{leaf}/{case}: no output {name}")


def _require(cover: dict[str, list], leaf: str, *names: str) -> None:
    missing = [name for name in names if not cover.get(name)]
    if missing:
        raise FluxprepOracleError(
            f"{leaf}: no case exercises {missing}; branch coverage incomplete")


def check_branch_coverage(table) -> None:
    # ---- RAGRB -----------------------------------------------------------
    cover: dict[str, list] = defaultdict(list)
    for case in RAGRB_CASES:
        v = _inputs(table, "ragrb", case)
        ints = _ints(table, "ragrb", case)
        if v["zpd"] >= v["hcan"]:
            raise FluxprepOracleError(
                f"ragrb/{case}: HCAN must exceed ZPD (KH would be negative)")
        flags = _ragrb_flags(ints["iter"], v)
        for name, hit in flags.items():
            if hit:
                cover[name].append(case)
            else:
                cover["not_" + name].append(case)
        if _out(table, "ragrb", case, "ramg") != 0.0:
            raise FluxprepOracleError(f"ragrb/{case}: RAMG is 0.0 at 4574")
        if _out(table, "ragrb", case, "rahg") \
                != _out(table, "ragrb", case, "rawg"):
            raise FluxprepOracleError(f"ragrb/{case}: RAWG = RAHG at 4576")
        if _out(table, "ragrb", case, "mozg") != 0.0 and ints["iter"] == 1:
            raise FluxprepOracleError(
                f"ragrb/{case}: ITER==1 must leave MOZG at 0.0 (4536)")
        rb = _out(table, "ragrb", case, "rb")
        if not 5.0 <= rb <= 50.0:
            raise FluxprepOracleError(
                f"ragrb/{case}: RB={rb} escapes the 4577 clamp")
    _require(cover, "ragrb", "iter_gt_1", "not_iter_gt_1", "tmp1_floor",
             "mozg_clamped", "mozg_negative", "not_mozg_negative", "kh_floor",
             "rb_low", "rb_high")
    interior = [c for c in RAGRB_CASES
                if c not in cover["rb_low"] and c not in cover["rb_high"]]
    if len(interior) < 3:
        raise FluxprepOracleError(
            "ragrb: fewer than three cases leave RB inside the clamp, so UC "
            "and DLEAF are barely observable")

    # ---- SFCDIF1 ---------------------------------------------------------
    cover = defaultdict(list)
    for case in SFCDIF1_CASES:
        v = _inputs(table, "sfcdif1", case)
        ints = _ints(table, "sfcdif1", case)
        if v["zlvl"] <= v["zpd"]:
            raise FluxprepOracleError(
                f"sfcdif1/{case}: ZLVL <= ZPD would take the wrf_error_fatal "
                f"path at 4650, which no fixture row may pin")
        flags = _sfcdif1_flags(ints["iter"], ints["mozsgn"], v)
        for name, hit in flags.items():
            if hit:
                cover[name].append(case)
            else:
                cover["not_" + name].append(case)
        mozsgn_out = _out(table, "sfcdif1", case, "mozsgn")
        expected = ints["mozsgn"] + (1 if flags["sign_flip"] else 0)
        if mozsgn_out != float(expected):
            raise FluxprepOracleError(
                f"sfcdif1/{case}: MOZSGN {mozsgn_out} != {expected} "
                f"predicted from the inputs at 4677")
        if flags["reset"] and _out(table, "sfcdif1", case, "moz") != 0.0:
            raise FluxprepOracleError(
                f"sfcdif1/{case}: the 4678 reset must zero MOZ")
    _require(cover, "sfcdif1", "iter_gt_1", "not_iter_gt_1", "tmp1_floor",
             "moz_clamped", "sign_flip", "not_sign_flip", "reset",
             "not_reset", "moz_negative", "not_moz_negative", "clamps_bind",
             "guards_fire")
    if len(cover["reset"]) < 2:
        raise FluxprepOracleError(
            "sfcdif1: the 4678 reset must be reached both by an incoming "
            "MOZSGN >= 2 and by a sign flip that pushes it there")

    # ---- STOMATA ---------------------------------------------------------
    cover = defaultdict(list)
    for case in STOMATA_CASES:
        v = _inputs(table, "stomata", case)
        flags = _stomata_flags(v)
        for name, hit in flags.items():
            if hit:
                cover[name].append(case)
            else:
                cover["not_" + name].append(case)
        if v["c3psn"] == 0.0:
            cover["c4"].append(case)
        if flags["early_return"]:
            if _out(table, "stomata", case, "psn") != 0.0:
                raise FluxprepOracleError(
                    f"stomata/{case}: the 5085 early return must leave PSN=0")
            cf = v["sfcprs"] / (8.314 * v["sfctmp"]) * 1.0e6
            want = cf / v["bp"]
            got = _out(table, "stomata", case, "rs")
            if abs(got - want) > 1e-5 * abs(want):
                raise FluxprepOracleError(
                    f"stomata/{case}: the early return must leave "
                    f"RS = CF/BP ({want}), got {got}")
    _require(cover, "stomata", "early_return", "not_early_return",
             "fveg_clamp", "folnmx_floor", "fnf_clamped", "not_fnf_clamped",
             "cea_from_ei", "cea_from_ea", "cea_from_quarter_ei",
             "wj_min", "wc_min", "we_min", "cs_floor", "b_nonneg",
             "b_negative", "ci_cp_clamp", "ci_zero_clamp", "c4")
    floors = {}
    for case in cover["cs_floor"]:
        floors[case] = _inputs(table, "stomata", case)["mpe"]
    if len(set(floors.values())) < 2:
        raise FluxprepOracleError(
            "stomata: every case that reaches the 5118 MPE floor carries the "
            "same MPE, so no mutant that freezes MPE to that constant can be "
            f"killed (saw {floors})")


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
    except FluxprepOracleError as error:
        print(f"fluxprep oracle validation FAILED: {error}", file=sys.stderr)
        return 1
    total_cases = sum(len(spec["cases"]) for spec in LEAVES.values())
    print(f"fluxprep oracle ok: {len(LEAVES)} leaves, {total_cases} cases, "
          f"{nvalues} pinned FP32 values, "
          f"{sum(counts.values())} discriminating input probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
