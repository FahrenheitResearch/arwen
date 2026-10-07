"""The UW PBL's fixtures of record, and the CPU reference graded on them.

The fixtures (``tests/data/oracles/uwpbl/cases-g35|g44|g61|extra35``, see
PROVENANCE.md there) are WRF v4.7.1's own words: the byte-unmodified
CAMUWPBL sources at gfortran 15.2.0 -O0 on glibc 2.43, six regime families
on 35/44/61 levels (dt 150/60/20 s, four consecutive steps) plus 48
branch-probe columns, each step's exact float32 inputs recorded beside its
outputs.  Pinned here, without a device: the fixture words (sha256), the
saturation table the kernel is given against the one the oracle recorded,
and the binary64 CPU reference ``gpuwm.verify.uwpbl_ref`` against every
output word of every column-step, on a host whose libm is the oracle's.

The product launcher on the card is graded against the same words in
tests/test_uwpbl_launcher_wrf471_parity.py.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.uwpbl_constants import ESTBL
from gpuwm.verify.uwpbl_oracle import (FULL_OUTPUTS, MASS_OUTPUTS,
                                       SURFACE_OUTPUTS, dims, load,
                                       step_arrays)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("UWPBL_ORACLE_DIR")
              or ROOT / "tests/data/oracles/uwpbl")
GRIDS = ("g35", "g44", "g61", "extra35")

#: sha256 of the fixtures of record (tools/uwpbl_wrf471_oracle/build.sh,
#: the oracle host, 2026-09-30).  A regenerated fixture changes these on
#: purpose.
FIXTURE_SHA256 = {
    "cases-g35.bin":
        "a2fd0445d95661e6e2ed43f149a50e8e9d3cfb5281f5e4e69b4d03a56801e855",
    "cases-g44.bin":
        "7b72f35396e601ef1fde89b2d76769b5201285e159998c07151227513c1cbb31",
    "cases-g61.bin":
        "0ecd84e170386790bad0b2381bb06b8081a276035338ed8bfca150210b77a45f",
    "cases-extra35.bin":
        "0f4344b4bc15f7d6383235e719a7b002c81569c2dc9ff83e4328bbeb6891dcbf",
}


def _fixture(grid):
    stem = ORACLE / f"cases-{grid}"
    if not Path(f"{stem}.bin").exists():
        pytest.skip(f"UW PBL oracle fixture {stem}.bin is absent")
    return load(stem)


def test_the_fixtures_are_the_recorded_words():
    if ORACLE != ROOT / "tests/data/oracles/uwpbl":
        pytest.skip("pins apply to the packaged fixtures")
    for name, digest in FIXTURE_SHA256.items():
        data = (ORACLE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_the_saturation_table_is_the_oracles():
    for grid in GRIDS:
        fx = _fixture(grid)
        recorded = fx["const/estbl"]
        assert np.array_equal(recorded.view(np.uint64),
                              np.asarray(ESTBL, np.float64).view(np.uint64))


def _host_libm_is_the_oracles() -> str | None:
    """Why this host's libm is not the oracle's, or None when it is.

    The CPU reference calls Python's ``math.exp/log/pow/cos/acos``, which are
    the C library's.  The oracle linked glibc 2.43 on x86-64, where ifunc
    picks the FMA variants of exp/log/pow; glibc's binary64 exp/log/pow
    (Arm optimized-routines) and cos/acos (IBM, slow paths removed) are the
    same code from glibc 2.28 on.  Another C library, or a CPU without FMA,
    is a different set of functions, and the reference would be graded
    against words its own libm cannot produce.
    """
    import platform
    import sys
    if not sys.platform.startswith("linux") or platform.machine() != "x86_64":
        return f"{sys.platform}/{platform.machine()} is not linux/x86_64"
    libc, version = platform.libc_ver()
    if libc != "glibc":
        return f"the C library is {libc or 'unknown'}, not glibc"
    if tuple(int(x) for x in version.split(".")[:2]) < (2, 28):
        return f"glibc {version} predates 2.28's exp/log/pow and cos/acos"
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="ascii",
                                                errors="replace")
    except OSError:
        return "/proc/cpuinfo is unreadable, so FMA cannot be confirmed"
    if "fma" not in flags.split():
        return "the CPU has no FMA, so glibc selects other exp/log/pow"
    return None


@pytest.mark.parametrize("grid", GRIDS)
def test_the_cpu_reference_is_wrfs_words(grid):
    """The binary64 CPU reference, every output word of every column-step.

    ``gpuwm.verify.uwpbl_ref`` is pure Python float arithmetic (binary64,
    no contraction) transcribing the same Fortran as the kernel, with
    glibc's own transcendentals, so on a glibc x86-64 host it must equal
    the oracle bit for bit, cos and acos included.
    """
    reason = _host_libm_is_the_oracles()
    if reason is not None:
        pytest.skip("the CPU reference is graded only where its libm is "
                    "the oracle's: " + reason)
    from gpuwm.verify.uwpbl_ref.driver import camuwpbl_step

    fx = _fixture(grid)
    ncol, nk, nsteps = dims(fx)
    for step in range(1, nsteps + 1):
        want = step_arrays(fx, step)
        got = camuwpbl_step(want)
        bad = _mismatched_columns(got, want)
        assert not bad, (f"{grid} step {step}: columns {sorted(bad)} "
                         "differ from the oracle")


def _mismatched_columns(got, want):
    bad = set()
    for name in MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS:
        g, w = np.asarray(got[name]), np.asarray(want[name])
        view = np.int32 if w.dtype == np.int32 else np.uint32
        diff = g.view(view) != w.view(view)
        if diff.ndim == 2:
            diff = diff.any(axis=1)
        bad |= set(np.flatnonzero(diff).tolist())
    return bad
