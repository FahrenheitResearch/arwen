#!/usr/bin/env python3
"""Mutation study for the Noah-MP radiation fixtures.

One mutant per input column of every leaf.  A mutant *ignores* that column:
its value is replaced, in every row, by the value row 0 carries -- exactly
what a transcription that read the argument once and cached it would compute.
The mutant is then replayed against the fixture.

  KILLED   the mutant disagrees with the oracle on at least one row, so the
           fixture would catch that argument being dropped.
  SURVIVED the mutant reproduces the fixture bit for bit.  A survivor is a
           statement about the *fixture*, not about the code, and it has to
           be discharged one of two ways:
             (a) the argument is genuinely dead in the pinned source -- prove
                 it by pointing at the body, or
             (b) the argument is constant down the whole fixture, in which
                 case the fixture is too weak and must be widened.
           "Probably fine" is not a discharge.

``--check`` exits non-zero unless every survivor appears in EXPECTED_SURVIVORS
below, so a future fixture edit that quietly weakens coverage fails loudly.
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gpuwm.core import noahmp_radiation as rad  # noqa: E402

ORACLE = REPO / "gpuwm" / "data" / "noahmp" / "oracle"


def _f(h: str) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])


def _rows(leaf: str):
    with (ORACLE / f"noahmp-radiation-{leaf}.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _v(r, base, n=2):
    return tuple(_f(r[f"{base}{i}"]) for i in range(1, n + 1))


def _pair(*bs):
    return [f"{b}{i}" for b in bs for i in (1, 2)]


# --------------------------------------------------------------------------
# per-leaf: which columns are inputs, which are outputs, how to evaluate a row
# --------------------------------------------------------------------------
def ev_snow_age(r):
    return list(rad.snow_age(
        _f(r["tau0"]), _f(r["grain_growth"]), _f(r["extra_growth"]),
        _f(r["dirt_soot"]), _f(r["swemx"]), _f(r["dt"]), _f(r["tg"]),
        _f(r["sneqvo"]), _f(r["sneqv"]), _f(r["tauss_in"])))


def ev_snowalb_class(r):
    a, sd, si = rad.snowalb_class(_f(r["swemx"]), int(r["nband"]),
                                  _f(r["qsnow"]), _f(r["dt"]), _f(r["albold"]))
    return [a, sd[0], sd[1], si[0], si[1]]


def ev_groundalb(r):
    a, b = rad.groundalb(_v(r, "albsat"), _v(r, "albdry"), _v(r, "alblak"),
                         int(r["nsoil"]), int(r["nband"]), int(r["ice"]),
                         int(r["ist"]), _f(r["fsno"]), (_f(r["smc1"]),),
                         _v(r, "albsnd"), _v(r, "albsni"), _f(r["cosz"]),
                         _f(r["tg"]))
    return [*a, *b]


def ev_surrad(r):
    return list(rad.surrad(
        _f(r["mpe"]), _f(r["fsun"]), _f(r["fsha"]), _f(r["elai"]), _f(r["vai"]),
        _f(r["laisun"]), _f(r["laisha"]),
        _v(r, "solad"), _v(r, "solai"), _v(r, "fabd"), _v(r, "fabi"),
        _v(r, "ftdd"), _v(r, "ftid"), _v(r, "ftii"),
        _v(r, "albgrd"), _v(r, "albgri"), _v(r, "albd"), _v(r, "albi"),
        _v(r, "frevd"), _v(r, "frevi"), _v(r, "fregd"), _v(r, "fregi")))


def ev_twostream(r):
    fab, fre, ftd, fti, gdir, frev, freg, bgap, wgap = rad.twostream(
        _f(r["xl"]), _v(r, "omegas"), _f(r["betads"]), _f(r["betais"]),
        int(r["ib"]), int(r["ic"]), _f(r["cosz"]), _f(r["vai"]), _f(r["fwet"]),
        _f(r["t"]), _v(r, "albgrd"), _v(r, "albgri"), _v(r, "rho"),
        _v(r, "tau"), _f(r["fveg"]), _v(r, "fab_in"), _v(r, "fre_in"),
        _v(r, "ftd_in"), _v(r, "fti_in"), _f(r["gdir_in"]), _v(r, "frev_in"),
        _v(r, "freg_in"), _f(r["bgap_in"]), _f(r["wgap_in"]))
    return [fab[0], fab[1], fre[0], fre[1], ftd[0], ftd[1], fti[0], fti[1],
            gdir, frev[0], frev[1], freg[0], freg[1], bgap, wgap]


_ALB_SC = ["fage", "albold", "tauss", "fsun", "bgap", "wgap"]
_ALB_VE = ["albgrd", "albgri", "albd", "albi", "fabd", "fabi", "ftdd", "ftid",
           "ftii", "frevd", "frevi", "fregd", "fregi", "albsnd", "albsni"]


def ev_albedo(r):
    out = rad.albedo(
        _f(r["tau0"]), _f(r["grain_growth"]), _f(r["extra_growth"]),
        _f(r["dirt_soot"]), _f(r["swemx"]), _v(r, "albsat"), _v(r, "albdry"),
        _v(r, "alblak"), _v(r, "rhol"), _v(r, "rhos"), _v(r, "taul"),
        _v(r, "taus"), _f(r["xl"]), _v(r, "omegas"), _f(r["betads"]),
        _f(r["betais"]), int(r["vegtyp"]), int(r["ist"]), int(r["ice"]),
        int(r["nsoil"]), _f(r["dt"]), _f(r["cosz"]), _f(r["fage_in"]),
        _f(r["elai"]), _f(r["esai"]), _f(r["tg"]), _f(r["tv"]), _f(r["snowh"]),
        _f(r["fsno"]), _f(r["fwet"]), _v(r, "smc", 4), _f(r["sneqvo"]),
        _f(r["sneqv"]), _f(r["qsnow"]), _f(r["fveg"]), _f(r["albold_in"]),
        _f(r["tauss_in"]),
        frevd_in=_v(r, "frevd_in"), frevi_in=_v(r, "frevi_in"),
        fregd_in=_v(r, "fregd_in"), fregi_in=_v(r, "fregi_in"))
    row = [out[k] for k in _ALB_SC]
    for k in _ALB_VE:
        row.extend(out[k])
    return row


LEAVES = {
    "snow_age": (ev_snow_age, ["tauss_out", "fage"]),
    "snowalb_class": (ev_snowalb_class,
                      ["alb", "albsnd1", "albsnd2", "albsni1", "albsni2"]),
    "groundalb": (ev_groundalb, ["albgrd1", "albgrd2", "albgri1", "albgri2"]),
    "surrad": (ev_surrad, ["parsun", "parsha", "sav", "sag", "fsa", "fsr",
                           "fsrv", "fsrg"]),
    "twostream": (ev_twostream,
                  ["fab1", "fab2", "fre1", "fre2", "ftd1", "ftd2", "fti1",
                   "fti2", "gdir", "frev1", "frev2", "freg1", "freg2",
                   "bgap", "wgap"]),
    "albedo": (ev_albedo, _ALB_SC + _pair(*_ALB_VE)),
}

# Survivors that are discharged, with the discharge.  Anything not on this
# list that survives is an unexplained hole and --check fails.
EXPECTED_SURVIVORS = {
    ("snowalb_class", "nband"):
        "NBAND only sizes the zeroing of ALBSND/ALBSNI, and both elements of "
        "both arrays are then assigned unconditionally "
        "(module_sf_noahmplsm.F:3266-3269).  ALBEDO is the only caller and "
        "sets NBAND = 2 unconditionally at line 2822, so no other value is "
        "reachable.  DEAD.",
    ("snowalb_class", "iloc"):
        "ILOC appears only in the declaration block of SNOWALB_CLASS; the "
        "body never references it.  DEAD.",
    ("snowalb_class", "jloc"):
        "JLOC appears only in the declaration block of SNOWALB_CLASS; the "
        "body never references it.  DEAD.",
    ("groundalb", "nsoil"):
        "NSOIL only declares the extent of the SMC dummy; the body reads "
        "SMC(1) and nothing else (line 3315).  DEAD as an arithmetic input.",
    ("groundalb", "nband"):
        "NBAND bounds the DO loop; ALBEDO, the only caller, sets NBAND = 2 "
        "unconditionally at line 2822.  No other value is reachable.",
    ("groundalb", "ice"):
        "ICE is declared INTENT(IN) at line 3293 and never referenced in the "
        "body of GROUNDALB.  DEAD -- verified by grepping the routine.",
    ("groundalb", "iloc"):
        "ILOC appears only in the declaration block.  DEAD.",
    ("groundalb", "jloc"):
        "JLOC appears only in the declaration block.  DEAD.",
    ("surrad", "iloc"):
        "ILOC appears only in the declaration block of SURRAD.  DEAD.",
    ("surrad", "jloc"):
        "JLOC appears only in the declaration block of SURRAD.  DEAD.",
    ("twostream", "vegtyp"):
        "VEGTYP is declared INTENT(IN) at line 3355 and never referenced in "
        "the body of TWOSTREAM.  DEAD.",
    ("twostream", "ist"):
        "IST is declared INTENT(IN) at line 3352 and never referenced in the "
        "body of TWOSTREAM.  DEAD.",
    ("twostream", "iloc"):
        "ILOC appears only in the declaration block.  DEAD.",
    ("twostream", "jloc"):
        "JLOC appears only in the declaration block.  DEAD.",
    ("twostream", "gdir_in"):
        "GDIR is declared INTENT(INOUT) but line 3419 assigns "
        "GDIR = PHI1 + PHI2*COSZI unconditionally, before any read.  Its "
        "entry value can therefore never reach an output.  DEAD as an input.",
    ("albedo", "vegtyp"):
        "ALBEDO forwards VEGTYP only to TWOSTREAM, where it is dead.  DEAD "
        "under the pinned identity.",
    ("albedo", "snowh"):
        "SNOWH is declared INTENT(IN) at line 2839 and never referenced in "
        "the body of ALBEDO.  DEAD.",
    ("albedo", "ice"):
        "ALBEDO forwards ICE only to GROUNDALB, where it is dead.  DEAD.",
    ("albedo", "nsoil"):
        "ALBEDO forwards NSOIL only to GROUNDALB as the SMC extent.  DEAD as "
        "an arithmetic input.",
    ("albedo", "iloc"):
        "ILOC reaches only dead arguments of SNOWALB_CLASS, GROUNDALB and "
        "TWOSTREAM.  DEAD.",
    ("albedo", "jloc"):
        "JLOC reaches only dead arguments of SNOWALB_CLASS, GROUNDALB and "
        "TWOSTREAM.  DEAD.",
    ("albedo", "smc2"):
        "GROUNDALB reads SMC(1) only, so SMC(2:4) cannot influence any "
        "output.  DEAD.",
    ("albedo", "smc3"): "See albedo/smc2.  DEAD.",
    ("albedo", "smc4"): "See albedo/smc2.  DEAD.",
}


def input_columns(leaf, rows, out_cols):
    return [c for c in rows[0] if c != "case" and c not in out_cols]


def run(leaf, verbose):
    ev, out_cols = LEAVES[leaf]
    rows = _rows(leaf)
    want = np.array([[_f(r[c]) for c in out_cols] for r in rows],
                    dtype=np.float32)
    base = np.array([ev(r) for r in rows], dtype=np.float32)
    if not np.array_equal(base.view(np.uint32), want.view(np.uint32)):
        raise SystemExit(f"{leaf}: the UNMUTATED transcription does not "
                         f"reproduce the fixture -- fix that first")

    results = []
    for col in input_columns(leaf, rows, out_cols):
        pinned = rows[0][col]
        constant = all(r[col] == pinned for r in rows)
        mutated = [dict(r, **{col: pinned}) for r in rows]
        got = np.array([ev(r) for r in mutated], dtype=np.float32)
        killed = not np.array_equal(got.view(np.uint32), want.view(np.uint32))
        n_diff = int((got.view(np.uint32) != want.view(np.uint32)).sum())
        results.append((col, killed, n_diff, constant))
        if verbose:
            tag = "KILLED  " if killed else "SURVIVED"
            note = " (constant down the fixture)" if constant and not killed else ""
            print(f"  {tag} {leaf}/{col}  differing lanes={n_diff}{note}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero on an undischarged survivor")
    ap.add_argument("-q", "--quiet", action="store_true")
    ns = ap.parse_args(argv)

    total = killed = 0
    survivors = []
    for leaf in LEAVES:
        if not ns.quiet:
            print(f"{leaf}:")
        for col, k, _n, const in run(leaf, not ns.quiet):
            total += 1
            if k:
                killed += 1
            else:
                survivors.append((leaf, col, const))

    print(f"\nmutants: {total}   killed: {killed}   survived: {len(survivors)}")
    bad = 0
    for leaf, col, const in survivors:
        why = EXPECTED_SURVIVORS.get((leaf, col))
        if why is None:
            print(f"UNDISCHARGED SURVIVOR {leaf}/{col}"
                  f"{'  (constant down the fixture)' if const else ''}")
            bad += 1
        else:
            print(f"discharged {leaf}/{col}: {why}")
    if bad:
        print(f"\n{bad} survivor(s) with no discharge", file=sys.stderr)
    return (1 if bad else 0) if ns.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
