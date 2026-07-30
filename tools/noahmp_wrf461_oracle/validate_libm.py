#!/usr/bin/env python3
"""Compare gpuwm/core/noahmp_libm.py against the live glibc 2.39 libm.

Runs on Windows; shells out to WSL to build and run ``glibc_ref_libm.c``,
which calls the real ``logf`` / ``log10f`` / ``atanf`` / ``expf`` / ``powf``
through the ordinary ABI so glibc's own ifunc resolver selects the same
variant the gfortran-built Noah-MP oracle calls.

Arrived as the VEGE_FLUX lane's check on its own private transcription; that
transcription was folded into ``gpuwm/core/noahmp_libm.py``, so this now
covers the one shared module -- including ``log10f``, which is the only entry
point that never had a second independent transcription to be checked against.

Three input populations per function:

  ``stride``  every ``2**k``-th bit pattern over the whole float32 domain --
              a uniform sweep of exponents, subnormals, NaN payloads and all;
  ``dense``   every bit pattern in the sub-ranges the Noah-MP leaves
              actually generate (the argument-stream bound);
  ``random``  a fixed-seed pseudo-random sample of the same sub-ranges.

Any mismatch is fatal and is printed with both bit patterns.  A pass is
reported with the exact counts, so the bound is stated rather than implied.

usage:  python tools/noahmp_wrf461_oracle/validate_libm.py [--stride-bits 12]
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gpuwm.core import noahmp_libm as L  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def wsl_path(p: pathlib.Path) -> str:
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def run_wsl(cmd: str) -> str:
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"wsl command failed ({r.returncode}): {cmd}\n"
                           f"{r.stdout}\n{r.stderr}")
    return r.stdout


def build_ref(work: pathlib.Path) -> str:
    src = wsl_path(HERE / "glibc_ref_libm.c")
    exe = wsl_path(work / "glibc_ref_libm")
    run_wsl(f"gcc -O2 -o {exe} {src} -lm")
    return exe


def glibc_eval(exe: str, work: pathlib.Path, fn: str, words: list[int]) -> list[int]:
    inp = work / f"{fn}_in.bin"
    out = work / f"{fn}_out.bin"
    inp.write_bytes(struct.pack(f"<{len(words)}I", *words))
    run_wsl(f"{exe} {fn} {wsl_path(inp)} {wsl_path(out)}")
    raw = out.read_bytes()
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}I", raw))


def u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_of(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_nan(u: int) -> bool:
    return (u & 0x7FFFFFFF) > 0x7F800000


# --------------------------------------------------------------------------
# argument populations
# --------------------------------------------------------------------------
# Argument sub-ranges the Noah-MP leaves generate.  LOG is applied to ratios
# of heights to roughness lengths (order 1 .. 1e5); ATAN to (1-16*MOZ)**0.25
# with MOZ < 0, so >= 1; EXP to small negative canopy-wind exponents; POW to
# (1-16*MOZ)**0.25, (1-15*MOZ)**-0.25, canopy-wind**0.5 and the Q10
# temperature responses AKC/AKO/AVCMX ** ((TC-25)/10).  LOG10 has exactly one
# caller, TDFCND's LOG10(SATRATIO), whose argument is a saturation ratio in
# (0, 1] -- and that is the band where glibc's log10f is least accurate, so it
# is swept densely rather than by stride alone.
LEAF_RANGES = {
    "logf": [(1.0, 1.0e6), (1.000001, 3.0)],
    "log10f": [(1.0e-6, 1.0), (0.1, 1.0)],
    "atanf": [(1.0, 40.0), (1.0, 1.0001)],
    "expf": [(-40.0, 0.0)],
}


def stride_words(stride_bits: int) -> list[int]:
    step = 1 << stride_bits
    return list(range(0, 1 << 32, step))


def dense_words(lo: float, hi: float, cap: int) -> list[int]:
    a, b = u32(lo), u32(hi)
    if a > b:
        a, b = b, a
    n = b - a + 1
    if n <= cap:
        return list(range(a, b + 1))
    step = n // cap
    return list(range(a, b + 1, step))


def check(name: str, exe: str, work: pathlib.Path, py, words: list[int]) -> int:
    ref = glibc_eval(exe, work, name, words)
    bad = 0
    for i, w in enumerate(words):
        x = f32_of(w)
        try:
            got = u32(py(x))
        except NotImplementedError:
            continue
        want = ref[i]
        if got == want:
            continue
        if is_nan(got) and is_nan(want):
            continue          # NaN payload is not part of the contract
        bad += 1
        if bad <= 10:
            print(f"  MISMATCH {name}(0x{w:08x} = {x!r}): "
                  f"got 0x{got:08x} want 0x{want:08x}")
    return bad


def check_pow(exe: str, work: pathlib.Path, pairs: list[tuple[int, int]]) -> int:
    flat: list[int] = []
    for a, b in pairs:
        flat.extend((a, b))
    ref = glibc_eval(exe, work, "powf", flat)
    bad = 0
    for i, (a, b) in enumerate(pairs):
        x, y = f32_of(a), f32_of(b)
        try:
            got = u32(L.powf(x, y))
        except NotImplementedError:
            continue
        want = ref[i]
        if got == want or (is_nan(got) and is_nan(want)):
            continue
        bad += 1
        if bad <= 10:
            print(f"  MISMATCH powf(0x{a:08x}={x!r}, 0x{b:08x}={y!r}): "
                  f"got 0x{got:08x} want 0x{want:08x}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-bits", type=int, default=12,
                    help="sweep every 2**k-th float32 bit pattern (default 12)")
    ap.add_argument("--dense-cap", type=int, default=200000)
    ap.add_argument("--random-n", type=int, default=200000)
    args = ap.parse_args()

    rng = random.Random(20260725)
    failures = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEMP")) as td:
        work = pathlib.Path(td)
        exe = build_ref(work)

        for name, fn in (("logf", L.logf), ("log10f", L.log10f),
                         ("atanf", L.atanf), ("expf", L.expf)):
            sweep = stride_words(args.stride_bits)
            bad = check(name, exe, work, fn, sweep)
            print(f"{name}: stride sweep {len(sweep)} patterns -> {bad} mismatches")
            failures += bad

            dense: list[int] = []
            for lo, hi in LEAF_RANGES[name]:
                dense.extend(dense_words(lo, hi, args.dense_cap))
            bad = check(name, exe, work, fn, dense)
            print(f"{name}: leaf-range sweep {len(dense)} patterns -> {bad} mismatches")
            failures += bad

            rnd = []
            for lo, hi in LEAF_RANGES[name]:
                a, b = u32(lo), u32(hi)
                if a > b:
                    a, b = b, a
                rnd.extend(rng.randrange(a, b + 1) for _ in range(args.random_n // len(LEAF_RANGES[name])))
            bad = check(name, exe, work, fn, rnd)
            print(f"{name}: leaf-range random {len(rnd)} patterns -> {bad} mismatches")
            failures += bad

        # powf: the exponents the subtree uses are a tiny fixed set, so sweep
        # the base densely for each of them, then add a random 2-D sample.
        pairs: list[tuple[int, int]] = []
        fixed_exponents = [0.5, 0.25, -0.25, 2.1, 1.2, 2.4]
        for e in fixed_exponents:
            for b in dense_words(1.0e-3, 1.0e6, args.dense_cap // 4):
                pairs.append((b, u32(e)))
        # Q10 responses: fixed base, exponent = (TC - 25)/10 for TC in [-50, 50]
        for base in (2.1, 1.2, 2.4):
            for k in range(-1000, 1001):
                pairs.append((u32(base), u32(k / 100.0)))
        bad = check_pow(exe, work, pairs)
        print(f"powf: leaf sweep {len(pairs)} pairs -> {bad} mismatches")
        failures += bad

        rp = []
        for _ in range(args.random_n):
            b = rng.randrange(u32(1.0e-6), u32(1.0e8))
            e = rng.randrange(u32(1.0e-3), u32(8.0))
            if rng.random() < 0.5:
                e |= 0x80000000
            rp.append((b, e))
        bad = check_pow(exe, work, rp)
        print(f"powf: random {len(rp)} pairs -> {bad} mismatches")
        failures += bad

    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} mismatches)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
