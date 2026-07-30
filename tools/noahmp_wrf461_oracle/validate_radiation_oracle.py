#!/usr/bin/env python3
"""Validate the Noah-MP radiation fixtures and the libm they depend on.

Three independent jobs, each with its own exit condition:

``--structure``   every fixture parses, every hex field is a well-formed
                  binary32 pattern, case names are unique, and the row counts
                  match what run_radiation.F90 declares.

``--libm-sweep``  every glibc entry point the radiation leaves reach
                  (``logf``, ``expf``, ``powf``) is compared bit-for-bit
                  against the *live* glibc symbol through
                  ``libm_probe_radiation.c``, over
                    (a) the exact argument stream replaying all six fixtures
                        produces -- the load-bearing set, and
                    (b) a structured + randomised sweep of the reachable
                        domain, hitting every table subinterval and both arms
                        of every special-case branch.
                  A single differing bit is a failure.  This is not exhaustive
                  over all 2**32 patterns and does not claim to be; the
                  domain and sample count are printed so the bound is on the
                  record.

``--negative-control table``   scales one ``__logf_data`` logc entry by
                  1 + 2**-26 and re-runs the sweep.  It MUST report
                  mismatches, otherwise the sweep is blind.  (Measured: 1980.)

``--negative-control numpy``   swaps logf/expf/powf for numpy's float32
                  equivalents.  MUST report mismatches -- this is the mistake
                  gpuwm/core/noahmp_libm.py exists to prevent.
                  (Measured: 377229.)

``--negative-control unfused`` is a DIAGNOSTIC, not a gate: it replaces every
                  FMA site with a separate multiply and add and reports the
                  count without asserting.  Measured: 0 of 2.09M arguments.
                  Un-fusing moves the binary64 intermediate by ~3e-15
                  relative, ~7 orders of magnitude below binary32 resolution,
                  so it is essentially invisible at the binary32 output.  The
                  fusion choices in noahmp_libm.py therefore rest on the
                  disassembly of the installed libm, not on this sweep.

Must run on the same glibc that produced the fixtures (2.39 here).

On a Windows host, pass ``--wsl``: the transcription needs ``math.fma``
(CPython 3.13+, present on the Windows interpreter) while the ground truth
needs glibc (WSL, whose python3 is 3.12 and has no ``math.fma``).
"""
from __future__ import annotations

import argparse
import csv
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gpuwm.core import noahmp_libm  # noqa: E402
from gpuwm.core import noahmp_radiation as rad  # noqa: E402

ORACLE = REPO / "gpuwm" / "data" / "noahmp" / "oracle"
PROBE_SRC = Path(__file__).resolve().parent / "libm_probe_radiation.c"

EXPECTED_ROWS = {
    "snow_age": 13,
    "snowalb_class": 8,
    "groundalb": 13,
    "surrad": 8,
    "twostream": 24,
    "albedo": 10,
}


def _f(h: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rows(leaf: str):
    with (ORACLE / f"noahmp-radiation-{leaf}.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def check_structure() -> int:
    bad = 0
    for leaf, n in EXPECTED_ROWS.items():
        rs = rows(leaf)
        if len(rs) != n:
            print(f"FAIL {leaf}: {len(rs)} rows, expected {n}")
            bad += 1
        names = [r["case"] for r in rs]
        if len(set(names)) != len(names):
            print(f"FAIL {leaf}: duplicate case names")
            bad += 1
        for r in rs:
            for k, v in r.items():
                if k == "case":
                    continue
                if v.startswith("0x"):
                    if len(v) != 10 or any(c not in "0123456789ABCDEFabcdef"
                                           for c in v[2:]):
                        print(f"FAIL {leaf}:{r['case']}:{k} = {v!r}")
                        bad += 1
                else:
                    int(v)  # integer column
        print(f"ok   {leaf}: {len(rs)} rows, {len(rs[0])} columns")
    return bad


# --------------------------------------------------------------------------
# libm sweep
# --------------------------------------------------------------------------
def _trace_fixture_libm_args():
    """Replay every fixture row with the libm calls instrumented."""
    seen_exp, seen_log, seen_pow = set(), set(), set()
    orig_exp, orig_log, orig_pow = rad._exp, rad._log, rad._pow

    def texp(x):
        seen_exp.add(_u(float(rad.F(x))))
        return orig_exp(x)

    def tlog(x):
        seen_log.add(_u(float(rad.F(x))))
        return orig_log(x)

    def tpow(x, y):
        seen_pow.add((_u(float(rad.F(x))), _u(float(rad.F(y)))))
        return orig_pow(x, y)

    rad._exp, rad._log, rad._pow = texp, tlog, tpow
    try:
        _replay_all()
    finally:
        rad._exp, rad._log, rad._pow = orig_exp, orig_log, orig_pow
    return sorted(seen_exp), sorted(seen_log), sorted(seen_pow)


def _vec(r, base, n=2):
    return tuple(_f(r[f"{base}{i}"]) for i in range(1, n + 1))


def _replay_all():
    for r in rows("snow_age"):
        rad.snow_age(_f(r["tau0"]), _f(r["grain_growth"]), _f(r["extra_growth"]),
                     _f(r["dirt_soot"]), _f(r["swemx"]), _f(r["dt"]), _f(r["tg"]),
                     _f(r["sneqvo"]), _f(r["sneqv"]), _f(r["tauss_in"]))
    for r in rows("snowalb_class"):
        rad.snowalb_class(_f(r["swemx"]), int(r["nband"]), _f(r["qsnow"]),
                          _f(r["dt"]), _f(r["albold"]))
    for r in rows("groundalb"):
        rad.groundalb(_vec(r, "albsat"), _vec(r, "albdry"), _vec(r, "alblak"),
                      int(r["nsoil"]), int(r["nband"]), int(r["ice"]),
                      int(r["ist"]), _f(r["fsno"]), (_f(r["smc1"]),),
                      _vec(r, "albsnd"), _vec(r, "albsni"), _f(r["cosz"]),
                      _f(r["tg"]))
    for r in rows("surrad"):
        rad.surrad(_f(r["mpe"]), _f(r["fsun"]), _f(r["fsha"]), _f(r["elai"]),
                   _f(r["vai"]), _f(r["laisun"]), _f(r["laisha"]),
                   _vec(r, "solad"), _vec(r, "solai"), _vec(r, "fabd"),
                   _vec(r, "fabi"), _vec(r, "ftdd"), _vec(r, "ftid"),
                   _vec(r, "ftii"), _vec(r, "albgrd"), _vec(r, "albgri"),
                   _vec(r, "albd"), _vec(r, "albi"), _vec(r, "frevd"),
                   _vec(r, "frevi"), _vec(r, "fregd"), _vec(r, "fregi"))
    for r in rows("twostream"):
        rad.twostream(_f(r["xl"]), _vec(r, "omegas"), _f(r["betads"]),
                      _f(r["betais"]), int(r["ib"]), int(r["ic"]), _f(r["cosz"]),
                      _f(r["vai"]), _f(r["fwet"]), _f(r["t"]),
                      _vec(r, "albgrd"), _vec(r, "albgri"), _vec(r, "rho"),
                      _vec(r, "tau"), _f(r["fveg"]), _vec(r, "fab_in"),
                      _vec(r, "fre_in"), _vec(r, "ftd_in"), _vec(r, "fti_in"),
                      _f(r["gdir_in"]), _vec(r, "frev_in"), _vec(r, "freg_in"),
                      _f(r["bgap_in"]), _f(r["wgap_in"]))
    for r in rows("albedo"):
        rad.albedo(
            _f(r["tau0"]), _f(r["grain_growth"]), _f(r["extra_growth"]),
            _f(r["dirt_soot"]), _f(r["swemx"]), _vec(r, "albsat"),
            _vec(r, "albdry"), _vec(r, "alblak"), _vec(r, "rhol"),
            _vec(r, "rhos"), _vec(r, "taul"), _vec(r, "taus"), _f(r["xl"]),
            _vec(r, "omegas"), _f(r["betads"]), _f(r["betais"]),
            int(r["vegtyp"]), int(r["ist"]), int(r["ice"]), int(r["nsoil"]),
            _f(r["dt"]), _f(r["cosz"]), _f(r["fage_in"]), _f(r["elai"]),
            _f(r["esai"]), _f(r["tg"]), _f(r["tv"]), _f(r["snowh"]),
            _f(r["fsno"]), _f(r["fwet"]), _vec(r, "smc", 4), _f(r["sneqvo"]),
            _f(r["sneqv"]), _f(r["qsnow"]), _f(r["fveg"]), _f(r["albold_in"]),
            _f(r["tauss_in"]),
            frevd_in=_vec(r, "frevd_in"), frevi_in=_vec(r, "frevi_in"),
            fregd_in=_vec(r, "fregd_in"), fregi_in=_vec(r, "fregi_in"))


def _sweep_domain(rng):
    """Structured + randomised arguments over the reachable domain.

    Reachable ranges, read off the leaves:
      logf : TWOSTREAM's (PHI1+PHI2)/PHI1 and (TMP1+TMP0)/TMP1, both > 1 and
             below ~1e4; swept over [1, 2**14] plus every one of the 16
             __logf_data subintervals, plus the [0.5, 4] band densely.
      expf : SNOW_AGE  arg in [-70, 4]; SNOWALB_CLASS -0.01*DT/3600 in
             [-1e-2, 0]; TWOSTREAM/ALBEDO -H*VAI and -EXT*VAI in [-2e3, 0].
             Swept over [-104, 4], which covers the underflow edge too.
      powf : GROUNDALB MAX(0.01,COSZ)**1.7, base in [0.01, 1], exponent 1.7.
             Swept over base [2**-20, 4] with the exponent held at 1.7f and
             also at a spread of nearby exponents.
    """
    logs, exps, pows = [], [], []

    # logf: every subinterval boundary of the 16-entry table, then dense
    for j in range(17):
        b = (0x3F330000 + j * (1 << 19)) & 0xFFFFFFFF
        logs.append(b)
        logs.append((b + 1) & 0xFFFFFFFF)
        logs.append((b - 1) & 0xFFFFFFFF)
    logs.append(_u(1.0))
    for _ in range(400_000):
        logs.append(_u(2.0 ** rng.uniform(-1.0, 14.0)))
    for _ in range(400_000):
        logs.append(rng.randrange(_u(0.5), _u(4.0)))

    # expf: the whole finite band plus the two special-case edges
    for v in (0.0, -0.0, 88.0, -88.0, 88.7228, -103.9721, -104.0, 88.72284):
        exps.append(_u(v))
    for _ in range(400_000):
        exps.append(_u(rng.uniform(-104.0, 4.0)))
    for _ in range(200_000):
        exps.append(_u(rng.uniform(-1.0e-2, 0.0)))
    for _ in range(200_000):
        exps.append(_u(rng.uniform(-2.0e3, 0.0)))

    # powf: the GROUNDALB call shape, base in the reachable band
    y17 = _u(1.7)
    for _ in range(300_000):
        pows.append((_u(rng.uniform(0.01, 1.0)), y17))
    for _ in range(100_000):
        pows.append((_u(2.0 ** rng.uniform(-20.0, 2.0)), y17))
    for _ in range(100_000):
        pows.append((_u(rng.uniform(0.01, 1.0)), _u(rng.uniform(0.5, 3.0))))
    return logs, exps, pows


class Probe:
    """Runs libm_probe_radiation, natively or through WSL.

    The transcription needs ``math.fma`` (CPython 3.13+) while the *ground
    truth* needs the glibc that built the fixtures.  On a Windows host those
    live in different interpreters, so ``--wsl`` builds and runs the probe
    inside WSL while the comparison stays in the host interpreter.
    """

    def __init__(self, use_wsl: bool):
        self.use_wsl = use_wsl
        self.workdir = Path(tempfile.mkdtemp(prefix="radlibm-"))
        if use_wsl:
            self.remote = "/tmp/radlibm"
            src = _to_wsl(PROBE_SRC)
            self._wsl(f"mkdir -p {self.remote} && gcc -O2 {src} "
                      f"-o {self.remote}/probe -lm")
        else:
            self.exe = self.workdir / "probe"
            subprocess.run(["gcc", "-O2", str(PROBE_SRC), "-o", str(self.exe),
                            "-lm"], check=True)

    @staticmethod
    def _wsl(cmd: str) -> str:
        r = subprocess.run(["wsl.exe", "-e", "bash", "-lc", cmd],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"wsl command failed: {cmd}\n{r.stderr}")
        return r.stdout

    def run(self, lines: list[str]) -> list[int]:
        argf = self.workdir / "args.txt"
        argf.write_text("\n".join(lines) + "\n")
        outf = self.workdir / "out.txt"
        if self.use_wsl:
            self._wsl(f"{self.remote}/probe < {_to_wsl(argf)} > {_to_wsl(outf)}")
        else:
            with argf.open() as fin, outf.open("w") as fout:
                subprocess.run([str(self.exe)], stdin=fin, stdout=fout,
                               check=True)
        return [int(t, 16) for t in outf.read_text().split()]


def _to_wsl(p: Path) -> str:
    s = str(Path(p).resolve())
    if len(s) > 2 and s[1] == ":":
        s = f"/mnt/{s[0].lower()}{s[2:]}"
    return s.replace("\\", "/")


def libm_sweep(seed: int, negative_control: bool, use_wsl: bool) -> int:
    rng = random.Random(seed)
    fx_exp, fx_log, fx_pow = _trace_fixture_libm_args()
    sw_log, sw_exp, sw_pow = _sweep_domain(rng)

    logs = sorted(set(fx_log) | set(sw_log))
    exps = sorted(set(fx_exp) | set(sw_exp))
    pows = sorted(set(fx_pow) | set(sw_pow))

    print(f"fixture argument stream : logf {len(fx_log)}  expf {len(fx_exp)}  "
          f"powf {len(fx_pow)}")
    print(f"total swept             : logf {len(logs)}  expf {len(exps)}  "
          f"powf {len(pows)}")

    restore = None
    if negative_control == "table":
        # Perturb one __logf_data logc entry by 2**-26 relative.  That is far
        # below binary64 noise but above binary32 resolution, so a meaningful
        # share of the arguments landing in subinterval 7 must round
        # differently.  (A 1-ulp-of-double perturbation does NOT work and is
        # not used: 1e-17 relative essentially never flips a binary32 rounding,
        # which is a fact about the control, not about the sweep.)
        tab = list(noahmp_libm._LOGF_TAB)
        invc, logc = tab[7]
        tab[7] = (invc, logc * (1.0 + 2.0 ** -26))
        noahmp_libm._LOGF_TAB = tuple(tab)
        print("negative control 'table': __logf_data logc[7] scaled by 1+2**-26")
    elif negative_control == "numpy":
        # The mistake this whole module exists to prevent: reaching for
        # numpy's float32 transcendentals instead of glibc's algorithm.
        import numpy as _np
        real = (noahmp_libm.logf, noahmp_libm.expf, noahmp_libm.powf)
        noahmp_libm.logf = lambda x: float(_np.log(_np.float32(x)))
        noahmp_libm.expf = lambda x: float(_np.exp(_np.float32(x)))
        noahmp_libm.powf = lambda x, y: float(
            _np.power(_np.float32(x), _np.float32(y)))

        def restore():
            (noahmp_libm.logf, noahmp_libm.expf, noahmp_libm.powf) = real
        print("negative control 'numpy': libm replaced by numpy float32")
    elif negative_control == "unfused":
        # DIAGNOSTIC, not a pass/fail control.  Replaces every FMA site with a
        # separate multiply and add -- the error a reader of the C source
        # rather than the disassembly would make.  It is *expected* to produce
        # ~0 binary32 mismatches: un-fusing perturbs the binary64 intermediate
        # by ~3e-15 relative, seven orders of magnitude below binary32
        # resolution, so it flips a final rounding for roughly 5e-8 of
        # arguments.  Reported so the record shows the fusion choices in
        # noahmp_libm.py rest on the disassembly, not on this sweep.
        real_fma = noahmp_libm._fma
        noahmp_libm._fma = lambda a, b, c: a * b + c

        def restore():
            noahmp_libm._fma = real_fma
        print("DIAGNOSTIC 'unfused': math.fma replaced by a*b + c "
              "(expected to be nearly invisible at binary32; not a gate)")

    probe = Probe(use_wsl)
    bad = 0
    for name, tag, fn, args in (
        ("logf", "l", noahmp_libm.logf, logs),
        ("expf", "e", noahmp_libm.expf, exps),
    ):
        want = probe.run([f"{tag} {a:08x}" for a in args])
        got = [_u(fn(_f(f"0x{a:08x}"))) for a in args]
        diff = [(a, g, w) for a, g, w in zip(args, got, want) if g != w]
        print(f"{name}: {len(args)} args, {len(diff)} mismatches")
        for a, g, w in diff[:5]:
            print(f"    x=0x{a:08x} ours=0x{g:08x} glibc=0x{w:08x}")
        bad += len(diff)

    want = probe.run([f"p {a:08x} {b:08x}" for a, b in pows])
    got = [_u(noahmp_libm.powf(_f(f"0x{a:08x}"), _f(f"0x{b:08x}")))
           for a, b in pows]
    diff = [(ab, g, w) for ab, g, w in zip(pows, got, want) if g != w]
    print(f"powf: {len(pows)} args, {len(diff)} mismatches")
    for (a, b), g, w in diff[:5]:
        print(f"    x=0x{a:08x} y=0x{b:08x} ours=0x{g:08x} glibc=0x{w:08x}")
    bad += len(diff)

    if restore is not None:
        restore()
    if negative_control == "unfused":
        print(f"DIAGNOSTIC 'unfused': {bad} binary32 mismatches "
              f"(not asserted -- see the note in the source)")
        return 0
    if negative_control:
        if bad == 0:
            print("NEGATIVE CONTROL FAILED -- the sweep is blind", file=sys.stderr)
            return 2
        print(f"negative control OK ({bad} mismatches detected)")
        return 0
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--libm-sweep", action="store_true")
    ap.add_argument("--negative-control", choices=["table", "numpy", "unfused"],
                    default=None)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--wsl", action="store_true",
                    help="build and run the glibc probe inside WSL (Windows "
                         "host: math.fma needs CPython 3.13, the ground truth "
                         "needs glibc)")
    ns = ap.parse_args(argv)
    if not (ns.structure or ns.libm_sweep or ns.negative_control):
        ns.structure = ns.libm_sweep = True

    rc = 0
    if ns.structure:
        rc |= check_structure()
    if ns.libm_sweep:
        rc |= libm_sweep(ns.seed, negative_control=None, use_wsl=ns.wsl)
    if ns.negative_control:
        rc |= libm_sweep(ns.seed, negative_control=ns.negative_control,
                         use_wsl=ns.wsl)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
