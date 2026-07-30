#!/usr/bin/env python3
"""Mutation study for the Noah-MP VEGE_FLUX subtree fixture.

For every argument of every leaf, build a mutant that *ignores* that argument
(replaces it with a fixed constant, independent of the case) and re-run the
whole fixture through the CPU transcription.  If a mutant reproduces every
recorded output bit for bit, the fixture cannot detect that argument being
dropped -- the coverage claim for that argument is empty and must be argued
from unreachability, not asserted.

Two constants are tried per argument (0 and 1, and for integers 0 and 1).  An
argument counts as *killed* if either constant changes any recorded output, or
raises.  Only arguments that survive both are reported as survivors.

usage:  python tools/noahmp_wrf461_oracle/mutation_study_vegeflux.py [--verbose]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gpuwm.core.noahmp_vegeflux import (  # noqa: E402
    R4, VegeFluxParameters, esat, ragrb, sfcdif1, stomata, vege_flux,
)

FIXTURE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-vegeflux.csv"
NSNOW, NSOIL = 3, 4


def bits(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def from_bits(h: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def load():
    table: dict = {}
    with FIXTURE.open(newline="") as fh:
        for row in csv.DictReader(fh):
            leaf = table.setdefault(row["leaf"], {})
            case = leaf.setdefault(row["case"], {"in": {}, "out": {}, "opt": {}})
            case[row["role"]][row["name"]] = row["hex"]
    return table


TABLE = load()


def f(case, name, override=None):
    if override is not None and name in override:
        return override[name]
    return from_bits(case["in"][name])


def i(case, name, override=None):
    if override is not None and name in override:
        return int(override[name])
    v = int(case["in"][name], 16)
    return v - (1 << 32) if v >> 31 else v


def params(case, override=None) -> VegeFluxParameters:
    kw = {}
    for name in VegeFluxParameters.__dataclass_fields__:
        key = "P_" + name
        if key in case["in"]:
            kw[name] = R4(f(case, key, override))
    return VegeFluxParameters(**kw)


# --------------------------------------------------------------------------
# leaf drivers: (name, argument list, evaluator)
# --------------------------------------------------------------------------
def run_esat(case, ov=None):
    esw, esi, desw, desi = esat(f(case, "T", ov))
    return {"ESW": esw, "ESI": esi, "DESW": desw, "DESI": desi}


def run_ragrb(case, ov=None):
    mozg, fhg, ramg, rahg, rawg, rb = ragrb(
        i(case, "ITER", ov), f(case, "VAI", ov), f(case, "RHOAIR", ov),
        f(case, "HG", ov), f(case, "TAH", ov), f(case, "ZPD", ov),
        f(case, "Z0MG", ov), f(case, "Z0HG", ov), f(case, "HCAN", ov),
        f(case, "UC", ov), f(case, "Z0H", ov), f(case, "FV", ov),
        f(case, "CWP", ov), f(case, "MPE", ov), f(case, "TV", ov),
        f(case, "MOZG", ov), f(case, "FHG", ov), f(case, "P_DLEAF", ov))
    return {"MOZG": mozg, "FHG": fhg, "RAMG": ramg, "RAHG": rahg,
            "RAWG": rawg, "RB": rb}


def run_sfcdif1(case, ov=None):
    moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2 = sfcdif1(
        i(case, "ITER", ov), f(case, "SFCTMP", ov), f(case, "RHOAIR", ov),
        f(case, "H", ov), f(case, "QAIR", ov), f(case, "ZLVL", ov),
        f(case, "ZPD", ov), f(case, "Z0M", ov), f(case, "Z0H", ov),
        f(case, "UR", ov), f(case, "MPE", ov), f(case, "MOZ", ov),
        i(case, "MOZSGN", ov), f(case, "FM", ov), f(case, "FH", ov),
        f(case, "FM2", ov), f(case, "FH2", ov), f(case, "FV", ov))
    return {"MOZ": moz, "FM": fm, "FH": fh, "FM2": fm2, "FH2": fh2,
            "CM": cm, "CH": ch, "FV": fv, "CH2": ch2,
            "MOZSGN": R4(mozsgn)}


def run_stomata(case, ov=None):
    rs, psn = stomata(
        params(case, ov), f(case, "MPE", ov), f(case, "APAR", ov),
        f(case, "FOLN", ov), f(case, "TV", ov), f(case, "EI", ov),
        f(case, "EA", ov), f(case, "SFCTMP", ov), f(case, "SFCPRS", ov),
        f(case, "FVEG", ov), f(case, "O2", ov), f(case, "CO2", ov),
        f(case, "IGS", ov), f(case, "BTRAN", ov), f(case, "RB", ov))
    return {"RS": rs, "PSN": psn}


def run_vegeflux(case, ov=None):
    dz = {k: f(case, f"DZSNSO_{k:+d}", ov) for k in range(-NSNOW + 1, NSOIL + 1)}
    stc = {k: f(case, f"STC_{k:+d}", ov) for k in range(-NSNOW + 1, NSOIL + 1)}
    df = {k: f(case, f"DF_{k:+d}", ov) for k in range(-NSNOW + 1, NSOIL + 1)}
    st = vege_flux(
        params(case, ov), NSNOW, NSOIL, i(case, "ISNOW", ov),
        f(case, "DT", ov), f(case, "SAV", ov), f(case, "SAG", ov),
        f(case, "LWDN", ov), f(case, "UR", ov), f(case, "UU", ov),
        f(case, "VV", ov), f(case, "SFCTMP", ov), f(case, "THAIR", ov),
        f(case, "QAIR", ov), f(case, "EAIR", ov), f(case, "RHOAIR", ov),
        f(case, "SNOWH", ov), f(case, "VAI", ov), f(case, "GAMMAV", ov),
        f(case, "GAMMAG", ov), f(case, "FWET", ov), f(case, "LAISUN", ov),
        f(case, "LAISHA", ov), f(case, "CWP", ov), dz,
        f(case, "ZLVL", ov), f(case, "ZPD", ov), f(case, "Z0M", ov),
        f(case, "FVEG", ov), f(case, "Z0MG", ov), f(case, "EMV", ov),
        f(case, "EMG", ov), f(case, "CANLIQ", ov), f(case, "FSNO", ov),
        f(case, "CANICE", ov), stc, df, f(case, "RSURF", ov),
        f(case, "LATHEAV", ov), f(case, "LATHEAG", ov), f(case, "PARSUN", ov),
        f(case, "PARSHA", ov), f(case, "IGS", ov), f(case, "FOLN", ov),
        f(case, "CO2AIR", ov), f(case, "O2AIR", ov), f(case, "BTRAN", ov),
        f(case, "SFCPRS", ov), f(case, "RHSUR", ov), f(case, "Q2", ov),
        f(case, "PAHV", ov), f(case, "PAHG", ov), f(case, "EAH", ov),
        f(case, "TAH", ov), f(case, "TV", ov), f(case, "TG", ov),
        f(case, "CM", ov), f(case, "CH", ov), f(case, "QC", ov),
        f(case, "QSFC", ov), f(case, "PSFC", ov), f(case, "FSR", ov))
    return {n: getattr(st, n) for n in dir(st) if n.isupper()}


LEAVES = {
    "esat": run_esat,
    "ragrb": run_ragrb,
    "sfcdif1": run_sfcdif1,
    "stomata": run_stomata,
    "vegeflux": run_vegeflux,
}

# Arguments the routine body provably never reads.  These are declared
# survivors up front and the reason is recorded, so a survivor list of zero is
# not manufactured by omission.
KNOWN_UNREAD = {
    "ragrb": {
        "TV": "declared INTENT(INOUT) by WRF but the body neither reads nor "
              "writes it",
        "MOZG": "declared INTENT(INOUT) but the body's first statement is "
                "MOZG = 0.0, so the incoming value can never be read",
    },
    "stomata": {
        "MPE": "both uses are unreachable under the pinned identity. "
               "(a) MAX(MPE,FOLNMX): every MPTABLE row with FOLNMX=0 also has "
               "VCMX25=0 and QE25=0, so FNF multiplies an identically-zero "
               "VCMX and J and cannot be observed; every other row has "
               "FOLNMX=1.5 >> MPE. "
               "(b) MAX(CO2-1.37*RLB*SFCPRS*PSN, MPE): the subtracted term is "
               "1.37*8.314e-6*RB*SFCTMP*PSN, bounded by ~1.8 Pa for RB<=50, "
               "SFCTMP<=310 and the PSN<=~10 that MPTABLE's VCMX25<=80 "
               "permits, against CO2AIR ~38 Pa, so the MAX never selects MPE",
    },
    "vegeflux": {
        "Q2": "declared INTENT(IN) but the body never references it",
        "QC": "declared INTENT(IN) but the body never references it",
        "VEG": "declared INTENT(IN) but the body never references it",
        "LATHEAG": "declared INTENT(IN) but the body never references it "
                   "(BARE_FLUX is the consumer)",
        "THAIR": "only SFCDIF2 (opt_sfc=2, dead) reads THAIR",
        "FSNO": "only the OPT_STC==3 leg of the snow/TG reset reads FSNO; "
                "opt_stc=1 is the Registry default so that leg is dead",
        "DX": "passed through; the body never references it",
        "DZ8W": "passed through; the body never references it",
        "PSFC": "feeds only the pre-loop QSFC initialisation, which loop1 "
                "unconditionally overwrites on its first iteration",
        "QSFC": "INTENT(INOUT) but written before any read",
        "TAH": "INTENT(INOUT); the only readers of an incoming TAH are RAGRB "
               "at ITER>1 and SFCDIF2 (dead), and TAH is assigned at ITER=1 "
               "before either can see it",
        "CM": "INTENT(INOUT); SFCDIF1 declares CM INTENT(OUT) and writes it at "
              "ITER=1 before RAMC reads it",
        "CH": "INTENT(INOUT); SFCDIF1 declares CH INTENT(OUT) and writes it at "
              "ITER=1 before RAHC reads it",
        "RSSUN": "INTENT(OUT); STOMATA writes it at ITER=1 before CTW reads it",
        "RSSHA": "INTENT(OUT); STOMATA writes it at ITER=1 before CTW reads it",
        "NSNOW": "array dimension only; no value is read",
        "NSOIL": "array dimension, plus the gecros loops (opt_crop=0, dead)",
        "JULIAN": "gecros only (opt_crop=0, dead)",
        "SWDOWN": "gecros only (opt_crop=0, dead)",
        "PRCP": "gecros only (opt_crop=0, dead)",
        "FB": "gecros only (opt_crop=0, dead)",
        "FSR": "gecros only (opt_crop=0, dead); returned unchanged",
        "P_SMCWLT": "gecros only (opt_crop=0, dead)",
        **{f"SH2O_{k:+d}": "gecros only (opt_crop=0, dead)"
           for k in range(1, NSOIL + 1)},
        # VEGE_FLUX touches exactly one layer of each snow/soil profile,
        # index ISNOW+1.  ISNOW ranges over -NSNOW..0, so indices 2..NSOIL are
        # unreachable for every legal ISNOW, not merely untested.
        **{f"{arr}_{k:+d}": "VEGE_FLUX reads only index ISNOW+1 and ISNOW is "
                            "in -NSNOW..0, so index >= 2 is unreachable"
           for arr in ("DF", "DZSNSO", "STC") for k in range(2, NSOIL + 1)},
    },
}


def baseline(leaf):
    return {tag: LEAVES[leaf](case) for tag, case in TABLE[leaf].items()}


def run_mutant(leaf, arg, value):
    """Return True if the mutant changes any recorded output (argument killed)."""
    for tag, case in TABLE[leaf].items():
        want = case["out"]
        try:
            got = LEAVES[leaf](case, {arg: value})
        except Exception:
            return True                       # a crash is a detection
        for name, hexval in want.items():
            if name not in got:
                continue
            if bits(got[name]) != int(hexval, 16):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    total_args = 0
    survivors: list[tuple[str, str]] = []
    for leaf in ("esat", "ragrb", "sfcdif1", "stomata", "vegeflux"):
        names = sorted({n for case in TABLE[leaf].values() for n in case["in"]})
        killed = 0
        leaf_surv = []
        for arg in names:
            total_args += 1
            if run_mutant(leaf, arg, 0.0) or run_mutant(leaf, arg, 1.0):
                killed += 1
                if args.verbose:
                    print(f"  killed {leaf}.{arg}")
            else:
                leaf_surv.append(arg)
                survivors.append((leaf, arg))
        print(f"{leaf}: {len(names)} arguments, {killed} killed, "
              f"{len(leaf_surv)} survivors")
        for arg in leaf_surv:
            reason = KNOWN_UNREAD.get(leaf, {}).get(arg)
            tag = f"UNREACHABLE: {reason}" if reason else "UNEXPLAINED"
            print(f"    survivor {arg}: {tag}")

    print(f"\ntotal arguments mutated: {total_args}")
    print(f"survivors: {len(survivors)}")
    unexplained = [(l, a) for l, a in survivors
                   if a not in KNOWN_UNREAD.get(l, {})]
    print(f"unexplained survivors: {len(unexplained)}")
    for leaf, arg in unexplained:
        print(f"  {leaf}.{arg}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
