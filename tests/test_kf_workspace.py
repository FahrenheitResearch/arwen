"""The gates the Kain-Fritsch column workspace owns.

The breakage each one prevents, named:

1. ``test_the_kf_frame_stays_under_the_default_stack`` -- the whole point.
   ``kf_column``'s column arrays used to live in the per-thread local frame,
   and CUDA sizes ONE per-context local-memory backing store to the widest
   frame in the context times the card's RESIDENT-THREAD CAPACITY, not times
   the occupancy the kernel achieves.  MEASURED on node-1 (RTX 5070 Ti, 70
   SMs x 1,536, sm_120, NVRTC 13.0.48): the 9,216 B frame at nz = 49 took
   840.0 MiB at first launch and never gave it back -- and that was already
   the SPECIALIZED frame; the unspecialized 24,064 B one took 5,738 MiB.
   Anything that puts a column array back on the stack -- a new
   ``float x[KF_KMAX]``, a compiler that stops eliminating the two that
   stayed -- brings that reservation back, and this is the only assertion
   that catches it on a platform with no recorded row (the RTX 3080 among
   them).

2. ``test_the_workspace_slot_map_fits_the_declared_region`` -- kf.cu's slot
   ids are literal because NVRTC has no ``__COUNTER__``.  A duplicate id
   makes two column arrays alias, which is a wrong forecast and not a crash;
   an id past the cap runs off the end of the block's region into the next
   block's, same result.  This reads the ids straight out of the source.

3. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates the workspace and the kernel indexes it.  If the two
   disagree about slot count, lane width or per-slot extent, the kernel
   writes past the allocation.  The LANE WIDTH is the sharp one: kf.cu
   indexes by the thread's lane within a block of ``KFWS_LANES``, so a
   launch at any other block width aliases lanes silently.

4. ``test_the_two_stack_arrays_are_the_measured_pair`` -- ``tv_env`` and
   ``positive_energy`` stay on the stack ON PURPOSE, and the purpose is
   bit-identity, not oversight.  A later cut that "finishes the job" by
   moving them would move output bits (55,360 and 34,560 words of a
   410,624-word grade, MEASURED).

5. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   allocated with ``cp.empty`` and reused across tiles, exactly as the local
   frame was reused across launches.  That is only sound if no column array
   is ever read before it is written.  This fills the workspace with two
   different garbage patterns and asserts the outputs are bit-identical.

6. ``test_the_tiled_launch_matches_the_single_launch`` (GPU) -- the columns
   go in tiles now, offset by ``col0``.  A tile boundary that dropped or
   double-counted a column, or a workspace not re-initialised between tiles,
   shows up here and nowhere else.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core.kf import (                                  # noqa: E402
    KFWS_SLOTS, KF_TILE_BLOCKS_PER_SM, _TPB, kf_workspace_floats,
)

_CU = os.path.join(_ROOT, "gpuwm", "core", "kernels", "kf.cu")

#: The CUDA default per-thread stack.  A frame at or under this reserves
#: nothing, because the driver's backing store is
#: ``(frame - default stack) x SMs x threads/SM`` floored at zero.
DEFAULT_STACK_BYTES = 1024

#: The two arrays kf.cu deliberately leaves on the stack, and the words each
#: one moved when it was put in the workspace instead.  MEASURED on node-1
#: (RTX 5070 Ti, sm_120, NVRTC 13.0.48) by moving each of the 54 arrays in
#: alone with the other 53 left local; every other array moved zero words.
STACK_RESIDENT_ARRAYS = {"tv_env": 55360, "positive_energy": 34560}


def _source() -> str:
    with open(_CU, encoding="utf-8") as fh:
        return fh.read()


def _define(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", _source(), re.M)
    assert m, f"kf.cu has no #define {name}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
def test_the_workspace_slot_map_fits_the_declared_region():
    ids = [int(m.group(1)) for m in re.finditer(
        r"KFWS_AT(?:_I)?\(kfws, (\d+), nz\)", _source())]
    assert ids, "kf_column declares no workspace column arrays"
    assert len(ids) == len(set(ids)), (
        "kf_column reuses a workspace slot id: two column arrays would "
        f"alias.  ids={sorted(ids)}")
    assert max(ids) < KFWS_SLOTS, (
        f"kf_column uses slot {max(ids)} of a {KFWS_SLOTS}-slot region: it "
        "would run into the next block's arrays")
    assert len(ids) == KFWS_SLOTS, (
        f"{len(ids)} slots are declared but KFWS_SLOTS is {KFWS_SLOTS}; a "
        "gap wastes the workspace and a shortfall over-runs it")


def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    assert _define("KFWS_SLOTS") == KFWS_SLOTS
    assert _define("KFWS_LANES") == _TPB, (
        "kf.cu indexes the workspace by lane within a block of KFWS_LANES "
        "threads; a launch at any other block width aliases lanes")
    # One whole block of columns, both sides.
    assert kf_workspace_floats(49, _TPB) == KFWS_SLOTS * 49 * _TPB
    # A partial block still allocates a whole block's region.
    assert kf_workspace_floats(49, 1) == kf_workspace_floats(49, _TPB)
    # The extent is the RUNTIME nz, which is the saving the compile-time
    # frame could not give: KF_KMAX must not appear in the size.
    assert (kf_workspace_floats(98, _TPB)
            == 2 * kf_workspace_floats(49, _TPB))
    assert KF_TILE_BLOCKS_PER_SM >= 1


def test_the_two_stack_arrays_are_the_measured_pair():
    """The two exceptions, pinned by name.

    kf.cu keeps ``tv_env`` and ``positive_energy`` as ``float[KF_KMAX]``
    because the compiler ELIMINATES both -- neither occupies a byte of frame
    when the other 52 arrays do -- and elimination is what lets their
    defining expressions fuse into the expressions that consume them.  Put
    either in the workspace and the store breaks the fusion, and the CAPE
    closure amplifies it into thousands of moved output words.  This pins
    the pair so a later "finish the job" pass has to read that first.
    """
    src = _source()
    declared = set(re.findall(r"^    float ([a-z_]+)\[KF_KMAX\];$",
                              src, re.M))
    assert declared == set(STACK_RESIDENT_ARRAYS), (
        f"kf_column's stack-resident column arrays are {sorted(declared)}, "
        f"not {sorted(STACK_RESIDENT_ARRAYS)}.  Adding one puts the frame "
        "back under the driver's resident-thread pricing; removing one "
        "moves output bits.")
    for name in STACK_RESIDENT_ARRAYS:
        assert f"KfCol {name} = KFWS_AT" not in src


def test_kf_kmax_is_only_a_refusal_ceiling():
    """``KF_KMAX`` may size the two stack arrays and nothing else.

    While it sized all 54, the launcher had to recompile the module per
    level count to keep the frame down.  It does not any more, and the
    breakage a regression here causes is quiet: a workspace slot sized by
    the compile-time bound would over-allocate by 128/nz and the frame
    would start following nz again.
    """
    from gpuwm.core.kf import _KMAX, VERTICAL_LEVEL_BOUNDS

    src = _source()
    assert VERTICAL_LEVEL_BOUNDS == (8, _KMAX)
    assert _define("KF_KMAX") == _KMAX
    uses = [ln.strip() for ln in src.split("\n")
            if re.search(r"(?<![A-Z_])KF_KMAX(?![A-Z_])", ln)
            and not ln.lstrip().startswith(("*", "//", "/*", "#ifndef",
                                            "#define", "#endif"))]
    assert sorted(uses) == sorted([
        "if (nz < 8 || nz > KF_KMAX) return;",
        "float tv_env[KF_KMAX];",
        "float positive_energy[KF_KMAX];",
    ]), uses


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_the_kf_frame_stays_under_the_default_stack():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module

    frame = int(load_module("kf").get_function("kf_column")
                .attributes["local_size_bytes"])
    assert frame <= DEFAULT_STACK_BYTES, (
        f"kf_column compiles a {frame} B per-thread local frame on this "
        "platform.  Over the 1,024 B default stack the driver reserves "
        "(frame - 1024) x SMs x threads/SM of device memory at first launch "
        "and never returns it -- 840.0 MiB on a 70-SM RTX 5070 Ti at the "
        "9,216 B frame this cut removed, and 5,738 MiB at the 24,064 B one "
        "before that.  A column array is back on the stack.")
    del cp


def _kf_batch(copies: int):
    """The parity suite's soundings, INTERLEAVED so neighbouring lanes take
    different branches."""
    import test_kf as tkf

    base = [tkf._sounding(unstable=True), tkf._real74_sounding("unstable"),
            tkf._shallow_sounding(), tkf._guarded_sounding(),
            tkf._real74_sounding("stable"), tkf._noitr_revert_sounding(),
            tkf._tder_suppression_sounding(),
            tkf._sounding(unstable=False)]
    pick = [base[i % len(base)] for i in range(copies)]
    names = ("u", "v", "temperature", "qv", "qc", "pressure", "exner",
             "dz", "w")
    return {n: np.ascontiguousarray(
        np.stack([c[n] for c in pick], axis=1)[:, None], dtype=np.float32)
        for n in names}


def _launch(kernel, dev, out, nz, ny, nx, ws, col0, span, phase):
    from gpuwm.core.kf import load_kf_table, _device_table
    from gpuwm.core.state import DTYPE

    table = load_kf_table()
    kernel(((span + _TPB - 1) // _TPB,), (_TPB,), (
        dev["u"], dev["v"], dev["temperature"], dev["qv"], dev["qc"],
        dev["pressure"], dev["exner"], dev["dz"], dev["w"],
        *_device_table(),
        out["rthcuten"], out["rqvcuten"], out["rqccuten"], out["rqicuten"],
        out["rqrcuten"], out["rqscuten"], out["rainc"], out["triggered"],
        out["cape_before"], out["cape_after"], out["timec"],
        out["nca_seconds"], out["shallow"], out["cloud_base"],
        out["cloud_top"], out["updraft_mass_flux"],
        out["downdraft_mass_flux"], ws,
        DTYPE(table.pressure_top), DTYPE(table.pressure_reciprocal),
        DTYPE(table.thetae_reciprocal), DTYPE(12000.0), DTYPE(60.0),
        DTYPE(300.0), np.int32(phase),
        np.int32(nz), np.int32(ny), np.int32(nx), np.int32(col0)))


def _fresh(nz, ny, nx):
    import cupy as cp

    out = {n: cp.zeros((nz, ny, nx), dtype=cp.float32) for n in (
        "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
        "rqscuten", "updraft_mass_flux", "downdraft_mass_flux")}
    out.update({n: cp.zeros((ny, nx), dtype=cp.float32) for n in (
        "rainc", "cape_before", "cape_after", "timec", "nca_seconds")})
    out.update({n: cp.zeros((ny, nx), dtype=cp.int32) for n in (
        "triggered", "shallow", "cloud_base", "cloud_top")})
    return out


def _bits(out):
    import cupy as cp

    return {n: np.ascontiguousarray(cp.asnumpy(v)).view(np.uint32).copy()
            for n, v in out.items()}


def _differing(a, b):
    return {n: int((a[n].ravel() != b[n].ravel()).sum())
            for n in sorted(a) if (a[n].ravel() != b[n].ravel()).any()}


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import get_kernel

    host = _kf_batch(128)
    nz, ny, nx = host["temperature"].shape
    ncol = ny * nx
    dev = {n: cp.asarray(v) for n, v in host.items()}
    kernel = get_kernel("kf", "kf_column")
    words = kf_workspace_floats(nz, ncol)

    runs = []
    for fill in (np.float32(0.0), np.float32(-7.0e30)):
        ws = cp.full(words, fill, dtype=cp.float32)
        out = _fresh(nz, ny, nx)
        for phase in range(4):
            _launch(kernel, dev, out, nz, ny, nx, ws, 0, ncol, phase)
        cp.cuda.Stream.null.synchronize()
        runs.append(_bits(out))
    assert int(cp.asnumpy(cp.asarray(runs[0]["triggered"])).sum()) > 0, (
        "no column triggered, so this compared two fields of zeros")
    bad = _differing(*runs)
    assert not bad, (
        f"{bad} words move when the workspace starts from different "
        "residue, so some column array is read before it is written and "
        "the result depends on the previous tile")


@pytest.mark.gpu
def test_the_tiled_launch_matches_the_single_launch():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import get_kernel

    host = _kf_batch(200)
    nz, ny, nx = host["temperature"].shape
    ncol = ny * nx
    dev = {n: cp.asarray(v) for n, v in host.items()}
    kernel = get_kernel("kf", "kf_column")

    whole = _fresh(nz, ny, nx)
    ws = cp.empty(kf_workspace_floats(nz, ncol), dtype=cp.float32)
    for phase in range(4):
        _launch(kernel, dev, whole, nz, ny, nx, ws, 0, ncol, phase)

    # A tile that does NOT divide the column count, so the last one is
    # partial: a col0 that walked by the tile instead of by the span would
    # skip columns here.
    tile = 3 * _TPB
    tiled = _fresh(nz, ny, nx)
    ws2 = cp.empty(kf_workspace_floats(nz, tile), dtype=cp.float32)
    for phase in range(4):
        for col0 in range(0, ncol, tile):
            _launch(kernel, dev, tiled, nz, ny, nx, ws2, col0,
                    min(tile, ncol - col0), phase)
    cp.cuda.Stream.null.synchronize()

    a, b = _bits(whole), _bits(tiled)
    assert int(a["triggered"].sum()) > 0
    assert ncol % tile != 0
    bad = _differing(a, b)
    assert not bad, f"tiling moved {bad}"


@pytest.mark.gpu
def test_the_shipped_launcher_tiles_and_frees_its_workspace():
    """``launch_kf`` is the door, so the tiling has to work THROUGH it.

    Its tile comes from the live card's SM count, so a domain wider than one
    tile is what proves the loop runs more than once -- and the workspace it
    allocates must be gone when it returns, or a per-call leak of hundreds
    of MiB accumulates over a forecast.
    """
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.kf import KFPhaseMode, kf_tile_columns, launch_kf

    host = _kf_batch(64)
    nz, ny, nx = host["temperature"].shape
    dev = {n: cp.asarray(v) for n, v in host.items()}
    tile = kf_tile_columns(get_kernel("kf", "kf_column"), ny * nx)
    assert tile >= _TPB

    before = cp.get_default_memory_pool().used_bytes()
    out = launch_kf(dev["u"], dev["v"], dev["temperature"], dev["qv"],
                    dev["qc"], dev["pressure"], dev["exner"], dev["dz"],
                    dev["w"], dx=12000.0, dt=60.0, cudt=300.0,
                    phase_mode=KFPhaseMode.SEPARATE_ICE_SNOW)
    cp.cuda.Stream.null.synchronize()
    assert int(cp.asnumpy(out["triggered"]).sum()) > 0
    held = cp.get_default_memory_pool().used_bytes() - before
    outputs = sum(int(v.nbytes) for v in out.values())
    workspace = kf_workspace_floats(nz, min(tile, ny * nx)) * 4
    # Not `<= outputs`: CuPy's pool rounds every block up, so seventeen
    # output arrays carry a few KiB of slack.  The claim is that the
    # WORKSPACE is gone, and it is three orders of magnitude bigger than
    # that slack, so the two cannot be confused.
    assert held < outputs + workspace // 2, (
        f"launch_kf still holds {held - outputs} B beyond its outputs "
        f"against a {workspace} B workspace; the workspace was not "
        "released")
