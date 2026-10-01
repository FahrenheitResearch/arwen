"""Workspace geometry, initialization and tile identity gates."""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.physics_inventory import (                       # noqa: E402
    SHINHONG_BLOCK, SHWS_SLOTS, shinhong_workspace_floats,
)

_CU = os.path.join(_ROOT, "gpuwm", "core", "kernels", "shinhong.cu")

#: The CUDA default per-thread stack.  A frame at or under this reserves
#: nothing, because the driver's backing store is
#: ``(frame - default stack) x SMs x threads/SM`` floored at zero.
DEFAULT_STACK_BYTES = 1024


def _source() -> str:
    with open(_CU, encoding="utf-8") as fh:
        return fh.read()


def _define(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", _source(), re.M)
    assert m, f"shinhong.cu has no #define {name}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    """Launcher and kernel must agree, or the kernel writes out of bounds."""
    assert _define("SHWS_SLOTS") == SHWS_SLOTS
    assert _define("SHWS_LANES") == SHINHONG_BLOCK, (
        "SHWS_LANES is the launch block width; shinhong.py must launch at "
        "exactly that block or the lanes alias")


def test_every_workspace_slot_id_is_used_once():
    ids = [int(m) for m in re.findall(r"SHWS_AT\(wsb,\s*(\d+),", _source())]
    assert ids, "no workspace slots found in shinhong.cu"
    assert len(ids) == SHWS_SLOTS, (
        f"shinhong.cu binds {len(ids)} slots but declares SHWS_SLOTS="
        f"{SHWS_SLOTS}")
    assert sorted(ids) == list(range(SHWS_SLOTS)), (
        f"slot ids must be 0..{SHWS_SLOTS - 1} exactly once, got "
        f"{sorted(ids)}")


def test_no_column_array_is_left_on_the_stack():
    """A `real name[SHINHONG_K2]` in the kernel body is the regression."""
    body = _source()
    left = re.findall(r"\breal\s+\w+\s*\[\s*SHINHONG_K2", body)
    assert not left, (
        f"these column arrays are back in the local frame: {left}; that "
        "re-arms the per-context local-memory reservation this workspace "
        "exists to remove")


def test_the_workspace_grows_with_levels_not_with_the_kernels_bound():
    """The extent is a runtime argument, so nz is what it follows."""
    a = shinhong_workspace_floats(49, SHINHONG_BLOCK)
    b = shinhong_workspace_floats(98, SHINHONG_BLOCK)
    assert b > a
    assert a == SHWS_SLOTS * 51 * SHINHONG_BLOCK
    assert b == SHWS_SLOTS * 100 * SHINHONG_BLOCK


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_the_shinhong_frame_stays_under_the_default_stack():
    """The workspace must keep column arrays out of the local frame."""
    import cupy  # noqa: F401
    from gpuwm.core.kernels import load_module

    frame = load_module("shinhong").get_function("shinhong_column").local_size_bytes
    assert frame <= DEFAULT_STACK_BYTES, frame



def _demo_columns(nz, ny, nx, seed=3):
    import cupy as cp
    from gpuwm.core.state import DTYPE

    rng = np.random.default_rng(seed)

    def f3(lo, hi, n=nz):
        return cp.asarray(rng.uniform(lo, hi, (n, ny, nx)).astype(DTYPE))

    def f2(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, (ny, nx)).astype(DTYPE))

    psf = 1.0e5 + rng.uniform(-2e3, 2e3, (ny, nx))
    pif = np.empty((nz + 1, ny, nx), dtype=DTYPE)
    for k in range(nz + 1):
        pif[k] = psf * (1.0 - 0.92 * k / nz)
    pr = np.empty((nz, ny, nx), dtype=DTYPE)
    for k in range(nz):
        pr[k] = 0.5 * (pif[k] + pif[k + 1])
    th = np.empty((nz, ny, nx), dtype=DTYPE)
    base = 290.0 + rng.uniform(-5, 5, (ny, nx))
    for k in range(nz):
        th[k] = base + 3.2 * k / nz * 10.0
    col = dict(u=f3(-12, 12), v=f3(-12, 12), theta=cp.asarray(th),
               qv=f3(1e-4, 1.2e-2), qc=f3(0, 3e-5), qi=f3(0, 2e-5),
               p=cp.asarray(pr), p_interface=cp.asarray(pif),
               exner=cp.asarray((pr / 1.0e5) ** 0.2857).astype(DTYPE),
               dz=f3(20, 250), tke=f3(0.005, 0.5))
    surf = dict(psfc=cp.asarray(psf.astype(DTYPE)), znt=f2(0.01, 0.9),
                ust=f2(0.05, 0.85), hfx=f2(-30, 250), qfx=f2(-1e-5, 2e-4),
                wspd=f2(0.7, 15), br=f2(-2.0, 0.6), psim=f2(0.4, 4.0),
                psih=f2(0.4, 4.0), xland=f2(1.0, 2.0), u10=f2(-8, 8),
                v10=f2(-8, 8), corf=f2(8e-5, 1.2e-4))
    return col, surf


def _bits(out):
    import cupy as cp

    return {k: cp.asnumpy(v).copy() for k, v in out.items()}


def _assert_bitwise(a, b, why):
    for name in a:
        assert a[name].dtype == b[name].dtype
        assert np.array_equal(a[name].view(np.uint32) if
                              a[name].dtype != np.int32 else a[name],
                              b[name].view(np.uint32) if
                              b[name].dtype != np.int32 else b[name]), (
            f"{name} moved: {why}")


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue():
    """No column array may be read before it is written."""
    import cupy as cp
    from gpuwm.core import shinhong as Y

    col, surf = _demo_columns(37, 8, 8)
    real_empty = cp.empty

    def poisoned(value):
        def _empty(shape, dtype=float, *a, **kw):
            arr = real_empty(shape, dtype=dtype, *a, **kw)
            arr.fill(dtype(value) if dtype != cp.int32 else 0)
            return arr
        return _empty

    seen = {}
    for value in (-7.0e30, 3.5e30):
        cp.empty = poisoned(value)
        try:
            seen[value] = _bits(Y.launch_shinhong(**col, **surf, dt=30.0, dx=3000.0, dy=3000.0))
        finally:
            cp.empty = real_empty
    a, b = (seen[v] for v in (-7.0e30, 3.5e30))
    _assert_bitwise(a, b,
                    "the workspace carries residue between uses, so some "
                    "column array is read before it is written")


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [4, 41, 128])
@pytest.mark.parametrize("tke_diag", [0, 1])
def test_tiling_does_not_change_the_answer(nz, tke_diag):
    """A tile boundary must not drop or double-count a column."""
    from gpuwm.core import shinhong as Y

    col, surf = _demo_columns(nz, 12, 12)
    whole = _bits(Y.launch_shinhong(**col, **surf, dt=30.0, dx=3000.0, dy=3000.0, tke_diag=tke_diag))

    original = Y.shinhong_tile_columns
    try:                        # force many tiles, incl. a partial last one
        Y.shinhong_tile_columns = lambda fn, ncol: Y.SHINHONG_BLOCK
        tiled = _bits(Y.launch_shinhong(**col, **surf, dt=30.0, dx=3000.0, dy=3000.0, tke_diag=tke_diag))
    finally:
        Y.shinhong_tile_columns = original
    _assert_bitwise(whole, tiled,
                    "tiling changed the result, so a tile boundary drops "
                    "or double-counts columns")
