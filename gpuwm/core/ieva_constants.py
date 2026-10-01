"""The IEVA column-solve level tiers, importable without CuPy.

``kernels/ieva.cu`` holds each column's tridiagonal (two ``double``
arrays) in a per-thread frame sized by ``IEVA_KMAX``; a configuration
compiles the smallest tier that holds its ``nz + 1`` w levels.  The
launcher (:mod:`gpuwm.core.ieva`) and the local-frame pricing
(:mod:`gpuwm.core.preflight`) both read this one ladder.
"""

from __future__ import annotations

#: ``IEVA_KMAX`` tiers, ascending; the first is the source's ``#ifndef``.
IEVA_LEVEL_TIERS = (65, 129, 257)


def level_tier(nz: int) -> int:
    """The ``IEVA_KMAX`` an ``nz``-level domain compiles at."""
    nz = int(nz)
    for tier in IEVA_LEVEL_TIERS:
        if nz + 1 <= tier:
            return tier
    raise ValueError(
        f"zadvect_implicit: nz={nz} exceeds the implicit column solve's "
        f"{IEVA_LEVEL_TIERS[-1] - 1} half levels")
