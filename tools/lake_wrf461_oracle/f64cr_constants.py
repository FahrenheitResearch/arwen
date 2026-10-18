"""Emit the constant block of gpuwm/core/kernels/f64cr.cuh from 400-bit
mpmath values, so every constant there is reproducible.

    python tools/lake_wrf461_oracle/f64cr_constants.py > constants.txt

The header carries the printed block verbatim between its CONSTANTS markers;
tests/test_lake_f64cr.py regenerates it and compares.
"""
from __future__ import annotations

import mpmath
from mpmath.libmp import round_nearest, to_float

mpmath.mp.prec = 400


def split(v, parts):
    out = []
    for _ in range(parts):
        h = to_float(v._mpf_, rnd=round_nearest)  # nearest binary64
        out.append(h)
        v = v - mpmath.mpf(h)
    return out


def h(x):
    if isinstance(x, mpmath.mpf):
        x = to_float(x._mpf_, rnd=round_nearest)
    if x == 0:
        return "0x0p+0"
    return float(x).hex() if x > 0 else "(" + float(x).hex() + ")"


def dd_table(name, values):
    rows = []
    for v in values:
        hi, lo = split(v, 2)
        rows.append(f"    {{{h(hi)},{h(lo)}}}")
    return f"WCR_TABLE wcr_dd {name}[{len(values)}] = {{\n" + ",\n".join(rows) + "\n};"


def block():
    ln2 = mpmath.log(2)
    pio2 = mpmath.pi / 2
    lines = ["// CONSTANTS BEGIN (tools/lake_wrf461_oracle/f64cr_constants.py)"]
    l1, l2, l3 = split(ln2, 3)
    lines.append(f"#define WCR_LN2_1 {h(l1)}\n#define WCR_LN2_2 {h(l2)}\n#define WCR_LN2_3 {h(l3)}")
    lines.append(f"#define WCR_INV_LN2 {h(1 / ln2)}")
    p1, p2, p3 = split(pio2, 3)
    lines.append(f"#define WCR_PIO2_1 {h(p1)}\n#define WCR_PIO2_2 {h(p2)}\n#define WCR_PIO2_3 {h(p3)}")
    lines.append(f"#define WCR_INV_PIO2 {h(1 / pio2)}")
    a, b = split(1 / mpmath.log(10), 2)
    lines.append(f"#define WCR_INV_LN10_1 {h(a)}\n#define WCR_INV_LN10_2 {h(b)}")
    lines.append(f"#define WCR_SQRT2 {h(mpmath.sqrt(2))}")
    # 1/n! for n = 0..29 (expm1 uses 1..11, sin/cos up to 29)
    lines.append(dd_table("wcr_inv_fact", [1 / mpmath.factorial(n) for n in range(30)]))
    # 1/(2n+1) for n = 0..20 (log's atanh series and atan's series)
    lines.append(dd_table("wcr_inv_odd", [mpmath.mpf(1) / (2 * n + 1) for n in range(21)]))
    # 1/n for n = 0..16 (entry 0 unused; log's series near 1)
    lines.append(dd_table("wcr_inv_int", [mpmath.mpf(0)] + [mpmath.mpf(1) / n for n in range(1, 17)]))
    # atan(j/16) for j = 0..16
    lines.append(dd_table("wcr_atan_j16", [mpmath.atan(mpmath.mpf(j) / 16) for j in range(17)]))
    # Fast phase tables.  exp: 2^(j/64), j = 0..63, and ln2/64 in two parts.
    a, b = split(ln2 / 64, 2)
    lines.append(f"#define WCR_LN2_64_1 {h(a)}\n#define WCR_LN2_64_2 {h(b)}")
    lines.append(f"#define WCR_INV_LN2_64 {h(64 / ln2)}")
    lines.append(dd_table("wcr_exp2_j64", [mpmath.mpf(2) ** (mpmath.mpf(j) / 64) for j in range(64)]))
    # log: for m in [0.75, 1.5), j = rint(128 m) in 96..192, inv_j = RN(128 / j)
    # and -log(inv_j) of that binary64 inv_j exactly.
    invs = [to_float((mpmath.mpf(128) / j)._mpf_, rnd=round_nearest) for j in range(96, 193)]
    rows = ",\n".join("    " + h(v) for v in invs)
    lines.append(f"WCR_TABLE double wcr_log_inv[{len(invs)}] = {{\n{rows}\n}};")
    lines.append(dd_table("wcr_log_tab", [-mpmath.log(mpmath.mpf(v)) for v in invs]))
    lines.append("// CONSTANTS END")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(block(), end="")
