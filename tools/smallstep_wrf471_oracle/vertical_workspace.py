"""Expose WRF's t_2ave workspace from an otherwise unchanged native solve.

This is a verification-only witness.  It adds global stores of already
computed native register words.  The oracle runner requires every normal
output word to match the uninstrumented launch before using this workspace.

The witness instruments the module as the default compile sees it: the
opt-in WRF-exact branches, some of which assign the same workspace words,
are resolved away first (gpuwm/verify/default_kernel_source.py), so the
stores land only in the code production runs.
"""
from __future__ import annotations

from gpuwm.verify.default_kernel_source import default_source


def workspace_source(source: str) -> str:
    source = default_source(source)
    prototype = "real cf1, real cf2, real cf3, real rdx, real rdy,"
    if source.count(prototype) != 2:
        raise ValueError("the two native vertical prototypes changed")
    source = source.replace(prototype,
                            "real* __restrict__ oracle_t2, real* __restrict__ oracle_mu,\n                   " + prototype)
    counts = tuple(source.count(marker) for marker in
                   ("real t2_dn =", "real t2_up =", "real muave ="))
    # The selected default body adds one column-parallel copy of both theta
    # assignments and the mass average to the two legacy vertical bodies.
    if counts not in ((2, 2, 2), (3, 3, 3)):
        raise ValueError(f"native vertical workspace layout changed: {counts}")
    for marker, store in (("real t2_dn =", "oracle_t2[c] = t2_dn;"),
                           ("real t2_up =", "oracle_t2[h] = t2_up;")):
        cursor = 0
        for _ in range(source.count(marker)):
            start = source.index(marker, cursor)
            end = source.index(";", start) + 1
            source = source[:end] + "\n        " + store + source[end:]
            cursor = end + len(store) + 9
    marker = "real muave ="
    cursor = 0
    store = "\n    oracle_mu[c] = muts; oracle_mu[st + c] = muave;"
    for _ in range(source.count(marker)):
        start = source.index(marker, cursor)
        end = source.index(";", start) + 1
        source = source[:end] + store + source[end:]
        cursor = end + len(store)
    return source
