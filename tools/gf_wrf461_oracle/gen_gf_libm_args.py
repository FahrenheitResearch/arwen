"""Build the argument word list for the GF libm fixture.

Arguments come in two tiers:
1. The LIVE tier: every (alpha, beta) the committed GF oracle actually
   reaches -- up_/dn_ from gf-deep-surface.csv, sh_ from
   gf-shallow-surface.csv -- expanded to the three tgammaf arguments
   alpha, beta, float32(alpha+beta).
2. The SWEEP tier: bit-uniform sweeps over each function's reachable
   domain, so the device transcription is graded far wider than the leaf
   reaches (noahmp-bareflux-libm.csv precedent).

Output: one text file per function, one lowercase hex float32 word per line.
"""

import csv
import sys

import numpy as np

ORACLE = sys.argv[1]
OUT = sys.argv[2]

F = np.float32


def words(path, cols):
    out = []
    with open(path, newline="", encoding="ascii") as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                out.append(F(row[c]))
    return out


live = []
pairs = []
for a, b in (("up_alpha", "up_beta"), ("dn_alpha", "dn_beta")):
    with open(f"{ORACLE}/gf-deep-surface.csv", newline="", encoding="ascii") as fh:
        for row in csv.DictReader(fh):
            al, be = F(row[a]), F(row[b])
            if be > 0:
                pairs.append((al, be))
with open(f"{ORACLE}/gf-shallow-surface.csv", newline="", encoding="ascii") as fh:
    for row in csv.DictReader(fh):
        al, be = F(row["sh_alpha"]), F(row["sh_beta"])
        if be > 0:
            pairs.append((al, be))

for al, be in pairs:
    live += [al, be, F(al + be)]

# The pgamma probe grid from gf-pow-probe.txt, for cross-checking the
# instrument against the committed answer sheet.
probe_words = set()
with open(f"{ORACLE}/gf-pow-probe.txt", encoding="ascii") as fh:
    for line in fh:
        if line.startswith("pgamma "):
            _, a, b, *_ = line.split()
            wa = np.uint32(int(a, 16)).view(F)
            wb = np.uint32(int(b, 16)).view(F)
            probe_words.update([float(wa), float(wb), float(F(wa + wb))])
live += [F(v) for v in sorted(probe_words)]


def bit_sweep(lo, hi, n):
    """n float32 values uniformly spaced in bit space over [lo, hi)."""
    a = np.float32(lo).view(np.uint32)
    b = np.float32(hi).view(np.uint32)
    ws = np.linspace(int(a), int(b), n, endpoint=False, dtype=np.int64)
    return np.unique(ws.astype(np.uint32)).view(np.float32)


def dump(name, values):
    arr = np.unique(np.asarray(values, dtype=np.float32))
    with open(f"{OUT}/gf-libm-args-{name}.txt", "w", encoding="ascii") as fh:
        for w in arr.view(np.uint32):
            fh.write(f"{int(w):08x}\n")
    print(name, arr.size)


# tgammaf: everything the scheme can reach and then some.  tunning in
# [0.2, 0.9], beta in {1.3, 2.5, 4.0} puts alpha in [1.06, 28.01] and
# alpha+beta below 32.1; the sweep starts at 0.25 so all three small-x arms
# of gammaf_positive (x < 0.5, x <= 1.5, x < 2.5) are graded, and runs to
# 36.0 so the overflow edge is covered too.
dump("tgammaf", list(bit_sweep(0.25, 36.0, 65536)) + live)

# lgammaf: gammaf_positive hands it x (x <= 1.5) or x-1 (1.5 < x < 2.5),
# so (0.5, 2.5) covers it; sweep a hair wider.
dump("lgammaf", bit_sweep(0.4, 2.6, 32768))

# expm1f: exp_adj is a sum of x_eps*log(x_adj) and bsum/x_adj -- order 1e-2.
# Sweep [-1, 1] via positive and negative halves in bit space.
pos = bit_sweep(1e-30, 1.0, 8192)
dump("expm1f", list(pos) + [-v for v in pos] + [F(0.0)])

# exp2f: x_adj_log2 * x_adj_frac with x_adj < 36 -> |arg| <= 2.5.  Sweep
# [-4, 4].
pos = bit_sweep(1e-30, 4.0, 8192)
dump("exp2f", list(pos) + [-v for v in pos] + [F(0.0)])

# expf / logf / powf get their grading from the transcribed-in-tree noahmp
# suite already; what GF adds is the pgamma/pbeta/ppowhard rows, which the
# CUDA gate reads straight from gf-pow-probe.txt.  No sweep needed here.
