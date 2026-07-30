#!/usr/bin/env python3
"""Validate the Noah-MP ENERGY fixture emitted by run_energy.F90.

Three separate jobs, in order of how much they are worth:

1. **Branch coverage, asserted from the inputs.**  ENERGY is a composition, so
   what its fixture has to pin is the branching.  Every live branch under the
   pinned option identity is asserted here from the *entry state*, never from
   the outputs, so a coincidence in the answer cannot satisfy it.

2. **The dead-option audit.**  The pinned identity kills whole routines --
   SFCDIF2 (opt_sfc=1), CANRES/CALHUM (opt_crs=1), SNOWALB_BATS (opt_alb=2),
   the OPT_STC==2 snow-surface reset (opt_stc=1), the CLM and SSiB BTRAN legs
   (opt_btr=1), the Sellers RSURF legs (opt_rsf=1).  This file states that in
   executable form so "we did not port it" is checkable rather than asserted.

3. **The whole-column cross-check.**  Cases 1-4 of the ENERGY fixture stand on
   the same four columns as ``noahmp-sflx.csv``, reached through WRF's own
   ATM/PHENOLOGY/PRECIP_HEAT.  Every quantity ENERGY writes that also appears
   as a NOAHMP_SFLX output must therefore agree **bit for bit**, except the
   soil-moisture and snow-mass state that WATER updates after ENERGY returns.
   Having two fixtures is only worth something if they are made to disagree
   when one of them is wrong, which is what this does.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import struct
import sys
from pathlib import Path

NSNOW = 3
NSOIL = 4
TFRZ = 273.16

CASES = (
    "veg_warm_day_dry",
    "veg_warm_night_rain",
    "snowpack_frozen_soil",
    "bare_thin_snow_melt",
    "veg_calm_desert_dry",
    "veg_deep_snow_saturated",
    "veg_subfreezing_canopy",
    "urban_snowfree",
    "veg_single_snow_layer",
)

#: The four ENERGY cases that reproduce a ``noahmp-sflx.csv`` column.
MIRROR_CASES = CASES[:4]

#: WRF Registry defaults; run_energy.F90 sets exactly these.
PINNED_OPTIONS = {
    "DVEG": 4, "OPT_CRS": 1, "OPT_BTR": 1, "OPT_RUN": 3, "OPT_SFC": 1,
    "OPT_FRZ": 1, "OPT_INF": 1, "OPT_RAD": 3, "OPT_ALB": 2, "OPT_SNF": 1,
    "OPT_TBOT": 2, "OPT_STC": 1, "OPT_RSF": 1, "OPT_SOIL": 1, "OPT_PEDO": 1,
    "OPT_CROP": 0, "OPT_IRR": 0, "OPT_IRRM": 0, "OPT_INFDV": 0, "OPT_TDRN": 0,
    "SOIL_UPDATE_STEPS": 1, "CALCULATE_SOIL": 1,
}

#: Names emitted as integers rather than as FP32 bit patterns.
INTEGER_ROLES = {"opt", "cfg"}
INTEGER_NAMES = {
    "ICE", "IST", "ISNOW", "ILOC", "JLOC", "NROOT", "ISBARREN", "URBAN_FLAG",
    "FROZEN_CANOPY", "FROZEN_GROUND",
}

#: ENERGY writes these, and NOAHMP_SFLX then hands them to WATER, which writes
#: them again before the column returns.  They are the only outputs the
#: whole-column cross-check may exempt, and the exemption is by name, so a new
#: disagreement anywhere else fails.
DOWNSTREAM_REWRITTEN = (
    {f"sh2o_{k}" for k in range(1, NSOIL + 1)}
    | {f"smc_{k}" for k in range(1, NSOIL + 1)}
    | {f"snice_{k}" for k in range(-NSNOW + 1, 1)}
    | {f"snliq_{k}" for k in range(-NSNOW + 1, 1)}
    | {"sneqv", "sneqvo", "snowh"}
)


def _f32(value: float) -> int:
    """IEEE-754 binary32 bit pattern of ``value``."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class Fixture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, str]] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["case", "role", "name", "hex", "value"]:
                raise SystemExit(f"{path}: unexpected header {reader.fieldnames}")
            for row in reader:
                self.rows.append({k: v.strip() for k, v in row.items()})
        self.by: dict[tuple[str, str], dict[str, str]] = collections.defaultdict(dict)
        for row in self.rows:
            key = (row["case"], row["role"])
            if row["name"] in self.by[key]:
                raise SystemExit(
                    f"{path}: duplicate {row['case']}/{row['role']}/{row['name']}")
            self.by[key][row["name"]] = row["hex"]

    # -- typed accessors ---------------------------------------------------
    def bits(self, case: str, role: str, name: str) -> int:
        return int(self.by[(case, role)][name], 16)

    def real(self, case: str, role: str, name: str) -> float:
        return _from_bits(self.bits(case, role, name))

    def integer(self, case: str, role: str, name: str) -> int:
        raw = self.bits(case, role, name)
        return raw - (1 << 32) if raw >= (1 << 31) else raw

    def has(self, case: str, role: str, name: str) -> bool:
        return name in self.by[(case, role)]


# ---------------------------------------------------------------------------
# 0. structure
# ---------------------------------------------------------------------------
def check_structure(fx: Fixture) -> None:
    seen = []
    for row in fx.rows:
        if row["case"] not in seen:
            seen.append(row["case"])
    if seen != list(CASES):
        raise SystemExit(f"case order {seen} != {list(CASES)}")

    roles = {row["role"] for row in fx.rows}
    if roles != {"opt", "cfg", "par", "seed", "in", "out", "undef"}:
        raise SystemExit(f"unexpected roles {sorted(roles)}")

    # Every FP32 row's decimal rendering must round-trip to its own bit
    # pattern; the hex column is authoritative and this proves the two agree.
    for row in fx.rows:
        if row["role"] in INTEGER_ROLES or row["name"] in INTEGER_NAMES:
            continue
        base = row["name"].split("_")[0]
        if base in INTEGER_NAMES or base == "IMELT":
            continue
        bits = int(row["hex"], 16)
        if _f32(float(row["value"])) != bits:
            raise SystemExit(
                f"{row['case']}/{row['role']}/{row['name']}: decimal "
                f"{row['value']} does not round-trip to {row['hex']}")

    # Each case must carry the same slot set, so no case is silently thinner.
    # 'out' and 'undef' partition one set: which side a slot falls on depends
    # on ISNOW and COSZ, so only the union is required to be constant.
    reference = {role: set(fx.by[(CASES[0], role)]) for role in
                 ("cfg", "par", "seed", "in")}
    reference["out"] = set(fx.by[(CASES[0], "out")]) | \
        set(fx.by[(CASES[0], "undef")])
    for case in CASES[1:]:
        for role, names in reference.items():
            got = set(fx.by[(case, role)])
            if role == "out":
                got |= set(fx.by[(case, "undef")])
            if got != names:
                missing = sorted(names - got)
                extra = sorted(got - names)
                raise SystemExit(
                    f"{case}/{role}: slot set differs (missing {missing}, "
                    f"extra {extra})")
        overlap = set(fx.by[(case, "out")]) & set(fx.by[(case, "undef")])
        if overlap:
            raise SystemExit(f"{case}: {sorted(overlap)} is both out and undef")

    check_undefined_slots(fx)


def check_undefined_slots(fx: Fixture) -> None:
    """The undefined set must be exactly what the WRF source leaves unwritten.

    Nothing here is judgement: each entry is a range the pinned source assigns
    over, so a port that writes a value into a buried slot, or that fails to
    write one it should, is caught by name.
    """
    for case in CASES:
        isnow = fx.integer(case, "in", "ISNOW")
        cosz = fx.real(case, "in", "COSZ")
        want = set()
        # CSNOW :2547-2565 and THERMOPROP :2413-2427 / PHASECHANGE :5063 all
        # loop from ISNOW+1, leaving the buried slots INTENT(OUT)-undefined.
        for k in range(-NSNOW + 1, NSOIL + 1):
            if k <= isnow:
                want |= {f"IMELT_{k:+d}", f"HCPCT_{k:+d}"}
        for k in range(-NSNOW + 1, 1):
            if k <= isnow:
                want |= {f"SNICEV_{k:+d}", f"SNLIQV_{k:+d}", f"EPORE_{k:+d}"}
        # BTRANI :2164-2172 covers 1..NROOT only.
        nroot = fx.integer(case, "par", "NROOT")
        want |= {f"BTRANI_{k:+d}" for k in range(nroot + 1, NSOIL + 1)}
        # ALBEDO :2922 exits past TWOSTREAM at night without writing
        # FREVD/FREVI/FREGD/FREGI or BGAP/WGAP; SURRAD :3111-3112 then carries
        # the first four into FSRV/FSRG.
        if cosz <= 0.0:
            want |= {"FSRV", "FSRG", "BGAP", "WGAP"}
        got = set(fx.by[(case, "undef")])
        if got != want:
            raise SystemExit(
                f"{case}: undefined set {sorted(got)} != {sorted(want)}")
        for name in got:
            if fx.bits(case, "undef", name) != 0:
                raise SystemExit(
                    f"{case}/undef/{name} is not the 0.0 marker")


# ---------------------------------------------------------------------------
# 1. the pinned option identity, and what it kills
# ---------------------------------------------------------------------------
def check_options(fx: Fixture) -> None:
    for name, want in PINNED_OPTIONS.items():
        got = fx.integer(CASES[0], "opt", name)
        if got != want:
            raise SystemExit(f"option {name} = {got}, expected {want}")

    # What the identity makes dead, stated where a reader will look for it.
    dead = {
        "SFCDIF2": ("OPT_SFC", 1),
        "CANRES/CALHUM": ("OPT_CRS", 1),
        "SNOWALB_BATS": ("OPT_ALB", 2),
        "ENERGY OPT_STC==2 snow-surface reset (:2384-2396)": ("OPT_STC", 1),
        "ENERGY BTRAN CLM/SSiB legs (:2153-2163)": ("OPT_BTR", 1),
        "ENERGY Sellers RSURF legs (:2188-2202)": ("OPT_RSF", 1),
        "gecros crop chain": ("OPT_CROP", 0),
        "irrigation": ("OPT_IRR", 0),
    }
    for routine, (option, value) in dead.items():
        if fx.integer(CASES[0], "opt", option) != value:
            raise SystemExit(
                f"{option} moved off {value}; {routine} is no longer dead and "
                "the port's NotImplementedError guards are wrong")

    # Slice: no case may use the ICE == 1 emissivity leg or an IST == 2 lake
    # column.  Both are outside the option identity this project admits, and
    # noahmp-sflx.csv does not admit them either.
    for case in CASES:
        if fx.integer(case, "in", "ICE") != 0:
            raise SystemExit(f"{case}: ICE != 0 is outside the admitted slice")
        if fx.integer(case, "in", "IST") != 1:
            raise SystemExit(f"{case}: IST != 1 is outside the admitted slice")


# ---------------------------------------------------------------------------
# 2. branch coverage, asserted from the entry state
# ---------------------------------------------------------------------------
def check_branch_coverage(fx: Fixture) -> None:
    taken: dict[str, list[str]] = collections.defaultdict(list)

    for case in CASES:
        elai = fx.real(case, "in", "ELAI")
        esai = fx.real(case, "in", "ESAI")
        vai = elai + esai
        fveg = fx.real(case, "in", "FVEG")
        snowh = fx.real(case, "in", "SNOWH")
        uu = fx.real(case, "in", "UU")
        vv = fx.real(case, "in", "VV")
        tv = fx.real(case, "in", "TV")
        tg = fx.real(case, "in", "TG")
        isnow = fx.integer(case, "in", "ISNOW")
        hvt = fx.real(case, "par", "HVT")
        urban = fx.integer(case, "par", "URBAN_FLAG")
        nroot = fx.integer(case, "par", "NROOT")

        # :2062-2063  VAI > 0 -> VEG
        taken["veg" if vai > 0.0 else "bare"].append(case)
        # :2278 / :2325  the tile average has a vegetated and a bare-only leg
        taken["tile_veg" if (vai > 0.0 and fveg > 0.0)
              else "tile_bare"].append(case)
        # :2068-2073  the FSNO branch
        taken["fsno_snow" if snowh > 0.0 else "fsno_none"].append(case)
        # :2057  UR = MAX(SQRT(UU**2+VV**2), 1.0)
        taken["ur_clamped" if math.hypot(uu, vv) < 1.0
              else "ur_free"].append(case)
        # :2090-2096  ZPD from the canopy or lifted onto the snow surface
        if vai > 0.0:
            taken["zpd_snow" if snowh > 0.65 * hvt
                  else "zpd_canopy"].append(case)
        # :2101-2106  the urban override of Z0MG/ZPDG/Z0M/ZPD
        taken["urban" if urban else "nonurban"].append(case)
        # :2211-2223  the two psychrometric branches, independently
        taken["frozen_canopy" if tv <= TFRZ else "thawed_canopy"].append(case)
        taken["frozen_ground" if tg <= TFRZ else "thawed_ground"].append(case)
        # SNOW_INIT topology: every ISNOW the initializer can produce
        taken[f"isnow{isnow}"].append(case)

        # :2159-2166  the GX = MIN(1., MAX(0., GX)) clamp, both legs, from the
        # root-zone soil water alone
        for k in range(1, nroot + 1):
            sh2o = fx.real(case, "in", f"SH2O_+{k}")
            wlt = fx.real(case, "par", f"SMCWLT_+{k}")
            ref = fx.real(case, "par", f"SMCREF_+{k}")
            if sh2o < wlt:
                taken["btran_gx_low_clamp"].append(case)
            if sh2o > ref:
                taken["btran_gx_high_clamp"].append(case)

        # :2185  MIN(1., SH2O(1)/SMCMAX(1)) inside L_RSURF
        sh2o1 = fx.real(case, "in", "SH2O_+1")
        smcmax1 = fx.real(case, "par", "SMCMAX_+1")
        taken["rsurf_saturated" if sh2o1 >= smcmax1
              else "rsurf_unsaturated"].append(case)
        # :2199  the dry-soil, snow-free RSURF saturation at 1.E6
        if sh2o1 < 0.01 and snowh == 0.0:
            taken["rsurf_dry_cap"].append(case)
        # :2204-2206  the urban, snow-free RSURF cap
        if urban and snowh == 0.0:
            taken["rsurf_urban_cap"].append(case)

    required = (
        "veg", "bare", "tile_veg", "tile_bare",
        "fsno_snow", "fsno_none", "ur_clamped", "ur_free",
        "zpd_snow", "zpd_canopy", "urban", "nonurban",
        "frozen_canopy", "thawed_canopy", "frozen_ground", "thawed_ground",
        "isnow0", "isnow-1", "isnow-2", "isnow-3",
        "btran_gx_low_clamp", "btran_gx_high_clamp",
        "rsurf_saturated", "rsurf_unsaturated",
        "rsurf_dry_cap", "rsurf_urban_cap",
    )
    missing = [name for name in required if not taken[name]]
    if missing:
        raise SystemExit(f"branch coverage gap: {missing}")

    # The two psychrometric flags must be shown to move independently, or the
    # fixture cannot tell LATHEAV from LATHEAG.
    pairs = set()
    for case in CASES:
        pairs.add((fx.real(case, "in", "TV") <= TFRZ,
                   fx.real(case, "in", "TG") <= TFRZ))
    if len(pairs) != 4:
        raise SystemExit(
            f"frozen_canopy/frozen_ground only reach {sorted(pairs)}; all four "
            "combinations are needed to separate LATHEAV from LATHEAG")

    # PHASECHANGE has to actually melt somewhere, or QMELT/PONDING/IMELT are
    # pinned at zero and the port could drop them.
    if not any(fx.real(case, "out", "QMELT") > 0.0 for case in CASES):
        raise SystemExit("no case melts: QMELT is zero everywhere")
    if not any(fx.real(case, "out", "PONDING") > 0.0 for case in CASES):
        raise SystemExit("no case ponds: PONDING is zero everywhere")
    if not any(fx.has(case, "out", f"IMELT_{k:+d}")
               and fx.integer(case, "out", f"IMELT_{k:+d}") != 0
               for case in CASES for k in range(-NSNOW + 1, NSOIL + 1)):
        raise SystemExit("IMELT is zero in every slot of every case")

    # ZPDG >= ZLVL (:2109) is unreachable for a positive ZREF: ZLVL is
    # MAX(ZPD,HVT) + ZREF and ZPDG is either SNOWH (<= ZPD) or 0.65*HVT
    # (< HVT), so the test needs ZREF <= 0.  Assert the premise instead of
    # pretending to cover the branch.
    for case in CASES:
        if fx.real(case, "in", "ZREF") <= 0.0:
            raise SystemExit(
                f"{case}: ZREF <= 0 would make the ZPDG >= ZLVL branch at "
                ":2109 reachable, and no case covers it")

    return None


# ---------------------------------------------------------------------------
# 3. the whole-column cross-check
# ---------------------------------------------------------------------------
def _load_sflx(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    table: dict[tuple[str, str], dict[str, str]] = collections.defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            field = row["field"].strip().lower()
            index = row["index"].strip()
            key = field if index == "0" else f"{field}_{index}"
            table[(row["case"].strip(), row["stage"].strip())][key] = \
                row["value"].strip()
    return table


def _normalise(name: str) -> str:
    """``HCPCT_+1`` -> ``hcpct_1``; run_sflx.F90 does not force the sign."""
    return name.lower().replace("_+", "_")


def cross_check_sflx(fx: Fixture, sflx_path: Path) -> int:
    sflx = _load_sflx(sflx_path)
    total_matched = 0
    for case in MIRROR_CASES:
        column = sflx[(case, "output")]
        if not column:
            raise SystemExit(f"{sflx_path}: no output stage for case {case}")
        matched: list[str] = []
        mismatched: list[tuple[str, str, str]] = []
        for name, hexval in fx.by[(case, "out")].items():
            key = _normalise(name)
            if key not in column:
                continue
            try:
                want = _f32(float(column[key]))
            except ValueError:
                continue
            if want == int(hexval, 16):
                matched.append(key)
            else:
                mismatched.append((key, hexval, f"{want:08X}"))

        unexpected = [m for m in mismatched if m[0] not in DOWNSTREAM_REWRITTEN]
        if unexpected:
            lines = "\n".join(
                f"    {k}: energy {a} vs sflx {b}" for k, a, b in unexpected)
            raise SystemExit(
                f"{case}: ENERGY disagrees with the whole-column fixture on "
                f"{len(unexpected)} field(s) that nothing downstream of ENERGY "
                f"writes:\n{lines}")
        if len(matched) < 80:
            raise SystemExit(
                f"{case}: only {len(matched)} fields could be cross-checked "
                "against the whole-column fixture; the comparison has gone "
                "vacuous")
        total_matched += len(matched)
        print(f"  {case}: {len(matched)} fields bit-identical to "
              f"noahmp-sflx.csv, {len(mismatched)} rewritten downstream")

    # Negative control: the comparator must be able to fail.  Flip one bit of
    # one cross-checked field and require the comparison to reject it.
    case = MIRROR_CASES[0]
    probe = "FSA"
    poisoned = int(fx.by[(case, "out")][probe], 16) ^ 1
    column = sflx[(case, "output")]
    if _f32(float(column[_normalise(probe)])) == poisoned:
        raise SystemExit("negative control did not perturb anything")
    return total_matched


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--sflx", type=Path, default=None,
                    help="noahmp-sflx.csv, for the whole-column cross-check")
    args = ap.parse_args(argv)

    fx = Fixture(args.fixture)
    check_structure(fx)
    check_options(fx)
    check_branch_coverage(fx)
    print(f"OK: {len(fx.rows)} rows, {len(CASES)} cases, every live branch of "
          "ENERGY covered from the entry state")

    if args.sflx is not None:
        print("whole-column cross-check:")
        total = cross_check_sflx(fx, args.sflx)
        print(f"OK: {total} ENERGY outputs bit-identical to noahmp-sflx.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
