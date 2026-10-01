"""Which per-toolchain row a GPU parity table reads, after A146.

The MYNN tables keep one row per (compute capability, NVRTC major.minor)
that reads differently from the table's default (the sm_86 and sm_89
reading).  Before A146 an unlisted toolchain fell back to the default
silently.  That is wrong on Blackwell: NVRTC compiled every float division
by a compile-time constant there as a multiply by the rounded reciprocal
(tools/literal_division_census.py: compute_100 and compute_120, every NVRTC
measured, 12.9 to 13.4), the kernels now spell those divisions
``__fdiv_rn``, and the sm_120 rows were re-recorded after the fix under
NVRTC 13.4 (A167 then measured 12.9.86 reading them word for word and
keyed it to the same rows).  No other architecture's row and nothing
recorded before the fix describes a Blackwell compile under another NVRTC.

So on those architectures a toolchain with no row does not fall back:
an architecture with a row under some other NVRTC FAILS naming this
compiler (it is unmeasured, the Shin-Hong table's rule), and an
architecture with no row at all SKIPS naming itself (a gate that fails for
the card and not the code prevents no breakage; the release card stage
names the skip in its receipt).  Everywhere else the default row stands.

"Those architectures" are every compute capability of major 10 or above,
not a list of the parts someone measured (A167).  The list read only
compute_100 and compute_120, so a B300 (10.3), a Thor (11.0) or a GB10
(12.1) fell through to the sm_86/89 default row as if it were Ampere or
Ada.  MEASURED 2026-10-01 on node-2, CPU only, x / 3.0f and x / 60.0f under
-ftz=true: NVRTC 12.9.86 and 13.4.92 both compile them as a multiply by the
rounded reciprocal for every compute_100, 101, 103, 110, 120 and 121 they
accept (12.9.86 has no compute_110, 13.4.92 no compute_101), and keep
div.rn for compute_75 through compute_90.
"""
from __future__ import annotations

import pytest

#: The lowest compute-capability major whose NVRTC rewrites a constant float
#: division as a reciprocal multiply (Blackwell; see the module docstring).
RECIPROCAL_REWRITE_MIN_MAJOR = 10


def rewrites_constant_division(capability: str) -> bool:
    """Whether NVRTC compiles ``x / C`` as ``x * RN(1/C)`` for this compute
    capability, written as CuPy writes it (``"120"`` for 12.0): every major
    of 10 or above, measured for each Blackwell target NVRTC accepts."""
    return int(capability) // 10 >= RECIPROCAL_REWRITE_MIN_MAJOR


def toolchain() -> tuple[str, tuple[int, ...]]:
    """(compute capability, NVRTC (major, minor)) of the compiling card."""
    import cupy as cp

    return (cp.cuda.Device().compute_capability,
            tuple(cp.cuda.nvrtc.getVersion()))


def toolchain_row(by_toolchain: dict, default, what: str,
                  current: tuple | None = None):
    """``by_toolchain``'s row for this toolchain, or ``default`` where that
    is a measurement of it (see the module docstring)."""
    current = toolchain() if current is None else current
    if current in by_toolchain:
        return by_toolchain[current]
    capability, nvrtc = str(current[0]), tuple(current[1])
    if not rewrites_constant_division(capability):
        return default
    measured = sorted(key for key in by_toolchain if key[0] == capability)
    version = ".".join(str(part) for part in nvrtc)
    if measured:
        pytest.fail(
            f"{what} has no row for NVRTC {version} on sm_{capability}; rows"
            f" are recorded for {measured}.  A146 re-recorded sm_{capability}"
            " after the __fdiv_rn fix under those compilers only, and the"
            " default row is another architecture's.  Measure this compiler"
            " (two processes) and add its row with the attribution, or"
            " compile with a recorded one.  Do NOT fall back.")
    pytest.skip(
        f"{what} has no row for sm_{capability} under any NVRTC (this is"
        f" NVRTC {version}); the default row is another architecture's.  Record"
        " this card's row (two processes) to gate it.")
