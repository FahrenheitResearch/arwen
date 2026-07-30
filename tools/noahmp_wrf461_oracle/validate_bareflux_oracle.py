#!/usr/bin/env python3
"""Build and validate the BARE_FLUX oracle fixture.

Three modes:

``--build``
    Run the compiled oracle over the case deck and write
    ``gpuwm/data/noahmp/oracle/noahmp-bareflux.csv``.  Requires the oracle
    binary, which requires WSL and the pinned WRF tree; that is why the CSV is
    committed rather than regenerated in CI.

``--check``
    Replay the committed fixture through
    :func:`gpuwm.core.noahmp_bareflux.bare_flux` and report the worst ULP
    distance per output column.  This is what ``tests/test_noahmp_bareflux.py``
    runs; it needs nothing but the repository.

``--mutants``
    Run the mutation study: for each input argument, re-run the fixture with
    that argument replaced by a constant and report whether any output moved.
    An argument whose mutants all reproduce the fixture exactly is one the
    fixture cannot detect being dropped.
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwm.core.fp32_ulp import max_ulp                      # noqa: E402
from gpuwm.core.noahmp_bareflux import bare_flux             # noqa: E402
from gen_bareflux_cases import (                             # noqa: E402
    ARRAY_FIELDS, INT_FIELDS, NSNOW, NSOIL, OUT_FIELDS, REAL_FIELDS,
    build_cases, format_case, hexf, unhexf,
)

FIXTURE = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle",
                       "noahmp-bareflux.csv")

_ARRAY_COLS = [f"{name}{k}" for name in ARRAY_FIELDS
               for k in range(-NSNOW + 1, NSOIL + 1)]
_IN_COLS = ["case", "opt_sfc", "opt_stc"] + INT_FIELDS + REAL_FIELDS + _ARRAY_COLS
_COLS = _IN_COLS + OUT_FIELDS


def _run_oracle(binary: str, deck: str) -> dict[str, list[str]]:
    with open(deck, "r") as handle:
        text = handle.read()
    proc = subprocess.run([binary], input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"oracle failed: {proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        out[parts[0]] = parts[1:]
    return out


def _read_oracle_out(path: str) -> dict[str, list[str]]:
    out = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                out[parts[0]] = parts[1:]
    return out


def cmd_build(args) -> int:
    cases = build_cases()
    deck = args.deck
    with open(deck, "w", newline="\n") as handle:
        handle.write("\n".join(format_case(rec) for rec in cases) + "\n")
    if args.oracle_out:
        # The oracle binary lives under WSL, whose python3 is 3.12 and has no
        # math.fma; the transcription needs 3.13.  So the deck is generated
        # here, run there, and the output fed back in.
        results = _read_oracle_out(args.oracle_out)
    else:
        results = _run_oracle(args.binary, deck)

    rows = []
    for rec in cases:
        name = rec["_name"]
        if name not in results:
            raise RuntimeError(f"oracle produced no row for {name}")
        row = {"case": name, "opt_sfc": 1, "opt_stc": 1}
        for k in INT_FIELDS:
            row[k] = rec[k]
        for k in REAL_FIELDS:
            row[k] = hexf(rec[k])
        for name_arr in ARRAY_FIELDS:
            for j, k in enumerate(range(-NSNOW + 1, NSOIL + 1)):
                row[f"{name_arr}{k}"] = hexf(rec[name_arr][j])
        for k, word in zip(OUT_FIELDS, results[name]):
            row[k] = word
        row["note"] = rec["_note"]
        rows.append(row)

    with open(FIXTURE, "w", newline="\n") as handle:
        handle.write(
            "# WRF v4.6.1 Noah-MP BARE_FLUX bitwise oracle\n"
            "#   tree     <wrf-4.6.1-checkout> (WSL)\n"
            "#   commit   d66e442fccc04111067e29274c9f9eaccc3cef28\n"
            "#   file     phys/module_sf_noahmplsm.F\n"
            "#   sha256   bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282\n"
            "#   patch    private:: -> public:: only, 50 lines, 300 bytes; see\n"
            "#            tools/noahmp_wrf461_oracle/visibility_patch_leaves.py --check\n"
            "#   pristine every row below is bit-identical to the row produced by a\n"
            "#            driver linked against the UNPATCHED object, reached with\n"
            "#            objcopy --globalize-symbol (which leaves .text byte-identical).\n"
            "#            Reproduce: build_bareflux.sh <tree> <work> noopt pristine\n"
            "#   optlevel every row is also bit-identical between the -O0 build and\n"
            "#            WRF's own -O2 build, so no row depends on the optimiser.\n"
            "#   compiler gfortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)\n"
            "#   flags    -w -ffree-form -ffree-line-length-none\n"
            "#            -fconvert=big-endian -frecord-marker=4\n"
            "#            -O2 -ftree-vectorize -funroll-loops  (WRF's own FCOPTIM)\n"
            "#   libc     Ubuntu GLIBC 2.39-0ubuntu8.7 (logf/atanf/powf)\n"
            "#   options  dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1\n"
            "#            opt_inf=1 opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2\n"
            "#            opt_stc=1 opt_rsf=1 opt_soil=1 opt_pedo=1 opt_crop=0\n"
            "#            opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0\n"
            "#            (Registry.EM_COMMON defaults)\n"
            "#   NITERB   5 (DATA NITERB /5/, line 4329 -- not an option)\n"
            "#   layers   NSNOW=3 NSOIL=4; array columns run -2..4\n"
            "# Every REAL is the IEEE-754 binary32 bit pattern, so no decimal\n"
            "# round-trip sits between the fixture and the compiled routine.\n"
        )
        writer = csv.DictWriter(handle, fieldnames=_COLS + ["note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {FIXTURE} ({len(rows)} cases)")
    return 0


def load_fixture(path: str = FIXTURE) -> list[dict]:
    with open(path, "r", newline="") as handle:
        lines = [ln for ln in handle if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def call_row(row: dict, override: dict | None = None) -> dict:
    """Run the transcription on one fixture row; ``override`` replaces inputs."""
    get = dict(row)
    if override:
        get.update(override)
    kw = {k: int(get[k]) for k in INT_FIELDS if k != "iurban"}
    kw["urban_flag"] = bool(int(get["iurban"]))
    for k in REAL_FIELDS:
        kw[k] = unhexf(get[k]) if isinstance(get[k], str) else get[k]
    for name in ARRAY_FIELDS:
        kw[name] = [unhexf(get[f"{name}{k}"]) if isinstance(get[f"{name}{k}"], str)
                    else get[f"{name}{k}"]
                    for k in range(-NSNOW + 1, NSOIL + 1)]
    kw["nsnow"] = NSNOW
    kw["nsoil"] = NSOIL
    kw["opt_sfc"] = int(get["opt_sfc"])
    kw["opt_stc"] = int(get["opt_stc"])
    out = bare_flux(**kw)
    return {
        "tgb_out": out.tgb, "cm_out": out.cm, "ch_out": out.ch,
        "qsfc_out": out.qsfc, "tauxb": out.tauxb, "tauyb": out.tauyb,
        "irb": out.irb, "shb": out.shb, "evb": out.evb, "ghb": out.ghb,
        "t2mb": out.t2mb, "q2b": out.q2b, "ehb2": out.ehb2,
    }


def cmd_check(args) -> int:
    rows = load_fixture(args.fixture)
    worst = {k: 0 for k in OUT_FIELDS}
    bad = []
    for row in rows:
        got = call_row(row)
        for k in OUT_FIELDS:
            want = unhexf(row[k])
            d = max_ulp([got[k]], [want])
            worst[k] = max(worst[k], d)
            if struct.pack("<f", got[k]) != struct.pack("<f", want):
                bad.append((row["case"], k, got[k], want))
    print(f"{len(rows)} cases")
    for k in OUT_FIELDS:
        print(f"  {k:<9s} max_ulp {worst[k]}")
    if bad:
        print(f"{len(bad)} bitwise mismatches; first 10:")
        for case, k, g, w in bad[:10]:
            print(f"  {case} {k}: got 0x{struct.unpack('<I', struct.pack('<f', g))[0]:08X} "
                  f"want 0x{struct.unpack('<I', struct.pack('<f', w))[0]:08X}")
        return 1
    print("all columns bitwise identical to the oracle (max_ulp 0)")
    return 0


# ---------------------------------------------------------------------------
# Mutation study
# ---------------------------------------------------------------------------
MUTANT_CONSTANTS = {
    "real": [0.0, 1.0, -7.5, 12345.0],
    "int": [0, 1, -2, 3],
}

_ELEM_COLS = {name: [f"{name}{k}" for k in range(-NSNOW + 1, NSOIL + 1)]
              for name in ARRAY_FIELDS}


def _targets(per_element: bool):
    """(name, kind, columns) for every mutation target.

    At Fortran-argument granularity an array is one target and every element
    moves together, which is what "one mutant per argument" means.  Passing
    ``per_element`` splits the arrays so the study also reports which
    individual layers the fixture can see.
    """
    out = [(k, "real", [k]) for k in REAL_FIELDS]
    if per_element:
        out += [(col, "real", [col]) for name in ARRAY_FIELDS
                for col in _ELEM_COLS[name]]
    else:
        out += [(name, "real", _ELEM_COLS[name]) for name in ARRAY_FIELDS]
    out += [(k, "int", [k]) for k in INT_FIELDS]
    return out


def _mutant_detected(rows, baseline, columns, value) -> bool:
    for row, base in zip(rows, baseline):
        if all(row[c] == value for c in columns):
            continue  # not actually a mutation for this row
        try:
            got = call_row(row, {c: value for c in columns})
        except (NotImplementedError, ValueError, ZeroDivisionError,
                OverflowError):
            return True
        for k in OUT_FIELDS:
            if struct.pack("<f", got[k]) != struct.pack("<f", base[k]):
                return True
    return False


def mutation_survivors(rows, per_element: bool = False) -> set[str]:
    """Names of arguments no mutant of which perturbs any fixture output."""
    baseline = [call_row(r) for r in rows]
    survivors = set()
    for name, kind, columns in _targets(per_element):
        detected = False
        for const in MUTANT_CONSTANTS[kind]:
            value = hexf(const) if kind == "real" else str(int(const))
            if _mutant_detected(rows, baseline, columns, value):
                detected = True
                break
        if not detected:
            survivors.add(name)
    return survivors


def cmd_mutants(args) -> int:
    rows = load_fixture(args.fixture)
    targets = _targets(args.per_element)
    survivors = mutation_survivors(rows, args.per_element)
    print(f"mutants: {len(targets)}  "
          f"killed: {len(targets) - len(survivors)}  "
          f"survivors: {len(survivors)}")
    print("survivors (fixture cannot detect these being ignored):")
    for name, _, _ in targets:
        if name in survivors:
            print(f"  {name}")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=FIXTURE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--binary")
    p.add_argument("--oracle-out",
                   help="output of a previous run of the oracle binary over "
                        "the same deck (for hosts that cannot exec it)")
    p.add_argument("--deck", required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("check")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("mutants")
    p.add_argument("--per-element", action="store_true",
                   help="split DZSNSO/STC/DF into one target per layer")
    p.set_defaults(func=cmd_mutants)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
