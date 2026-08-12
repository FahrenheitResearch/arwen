"""The exchange pack: descriptor round-trip on CPU, digest equality on GPU.

The multi-GPU exchange coalesces every field's halo band for one seam into ONE
contiguous staging buffer, sends it as a single transfer and scatters it on the
far side.  Two things have to be true for that to be worth having, and this
module tests them at two very different prices:

``test_round_trip_*`` (numpy only, no GPU, milliseconds)
    The pack is a set of PITCHED-COPY DESCRIPTORS -- ``rows``, ``width``,
    ``src_pitch``, ``dst_pitch``, offsets -- computed by :func:`_bands`.  Every
    way this can be wrong is an arithmetic error in those five numbers, and an
    arithmetic error does not need a GPU to catch.  The test emulates
    ``cudaMemcpy2D`` in numpy against the descriptors and compares with the
    slice assignment the seam means.  It runs on any machine, so the cheap
    guard is always on.

    Both axes are swept because they use DIFFERENT descriptor shapes: an x band
    is many short runs (one per row), a y band is few long ones (one per
    horizontal slice).  A transposed index survives one and not the other,
    which is exactly why :func:`_bands` documents them separately.

``test_packed_matches_direct`` (2 GPUs, marked ``gpu``)
    Bit-exactness of the packed path against :meth:`exchange_direct`, the
    unpacked reference that moves the same bytes with one transfer per field
    instead of one per seam.  This is the acceptance the pack has to meet:
    SAME BYTES, fewer transfers.  Comparing the packed path against itself
    would prove nothing, which is why the reference implementation exists.

``test_exchange_matches_the_slice_oracle`` (2 GPUs, marked ``gpu``)
    The same exchange against an ORACLE rather than against a twin.  The two
    paths above share one thing -- the band list :class:`_SeamChannel` was
    built with -- so a fault in THAT list is invisible to both of them and to
    the CPU half, which calls :func:`_bands` directly and never sees a
    channel.  MEASURED, on the 4x3090 box: truncating every channel's band
    list moves the run digest (dropping ``u``: 2dc2051c -> e22d7d44) and all
    seven older tests stay green.  So this one fills every carrier with a
    per-band marker, exchanges ONCE, and demands the arrays equal the slice
    assignment the seam plan means -- computed here, from the seams, without
    consulting a band.  An uneven split (nx=97 over two ranks) so the two
    pitches separate, and all three exchange implementations, because
    "packed == direct" cannot fail when both are missing the same field.

Run with ``pytest tilestream/test_exchange_pack.py`` or directly with
``python tilestream/test_exchange_pack.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilestream.multigpu import (  # noqa: E402
    _bands, plan_split, seam_plan,
)

NZ = 5
HALO = 3


# ==========================================================================
# the CPU half: descriptors, emulated
# ==========================================================================

def _inventory(spec, nz: int = NZ) -> dict:
    """A synthetic sub-domain inventory covering every stagger the pack sees.

    Values are a deterministic function of the FLAT INDEX rather than a draw,
    so a copy that lands one row or one column off produces different bytes
    instead of accidentally matching a neighbour.  Two dtypes, because the
    descriptors are all in bytes and an itemsize error would cancel out in a
    single-dtype inventory.
    """
    h, w = int(spec.cny), int(spec.cnx)
    shapes = {
        "mass": (nz, h, w),          # cell centres
        "u": (nz, h, w + 1),         # x face
        "v": (nz, h + 1, w),         # y face
        "w": (nz + 1, h, w),         # z face
        "soil": (4, h, w),           # a non-vertical leading axis
        "psfc": (h, w),              # 2-D, no leading axis at all
    }
    inv = {}
    for k, shp in shapes.items():
        n = int(np.prod(shp))
        dt = np.float64 if k in ("mass", "psfc") else np.float32
        inv[k] = (np.arange(n, dtype=np.float64) * 0.5 + hash(k) % 977
                  ).astype(dt).reshape(shp)
    return inv


def _flat_bytes(arr: np.ndarray) -> np.ndarray:
    """A writable flat uint8 view of a C-contiguous array."""
    assert arr.flags.c_contiguous
    return arr.view(np.uint8).reshape(-1)


def _emulate_pack(bands, src_inv, nbytes: int) -> np.ndarray:
    """``cudaMemcpy2D`` device->staging, in numpy, straight off the descriptors."""
    buf = np.zeros(nbytes, dtype=np.uint8)
    for b in bands:
        s = _flat_bytes(src_inv[b.name])
        for r in range(b.rows):
            src0 = b.src_off + r * b.src_pitch
            dst0 = b.buf_off + r * b.width
            buf[dst0:dst0 + b.width] = s[src0:src0 + b.width]
    return buf


def _emulate_unpack(bands, dst_inv, buf: np.ndarray) -> None:
    """``cudaMemcpy2D`` staging->device, in numpy, straight off the descriptors."""
    for b in bands:
        d = _flat_bytes(dst_inv[b.name])
        for r in range(b.rows):
            src0 = b.buf_off + r * b.width
            dst0 = b.dst_off + r * b.dst_pitch
            d[dst0:dst0 + b.width] = buf[src0:src0 + b.width]


def _reference_with_spec(seam, src_inv, dst_inv, src_spec, dst_spec,
                         nz: int = NZ) -> dict:
    """The oracle, told the grid extents explicitly."""
    from tilestream import gather as _gather

    out = {k: v.copy() for k, v in dst_inv.items()}
    for name, d in out.items():
        s = src_inv[name]
        v = _gather.classify(s.shape, nz, int(src_spec.cny), int(src_spec.cnx),
                             layers_ok=True)
        vd = _gather.classify(d.shape, nz, int(dst_spec.cny),
                              int(dst_spec.cnx), layers_ok=True)
        assert v == vd, f"{name}: {v} vs {vd}"
        sl_s, sl_d = seam.src_sel[v], seam.dst_sel[v]
        if seam.axis == "x":
            d[..., sl_d] = s[..., sl_s]
        else:
            d[..., sl_d, :] = s[..., sl_s, :]
    return out


def _round_trip_case(gx: int, gy: int, nx: int, ny: int, axis: str) -> int:
    """Pack+unpack every seam on ``axis``; assert each equals the oracle."""
    specs = plan_split(nx, ny, HALO, gx=gx, gy=gy)
    seams = [s for s in seam_plan(specs, HALO, nx, ny) if s.axis == axis]
    assert seams, f"no {axis} seams in a {gy}x{gx} split"
    names = list(_inventory(specs[0]).keys())

    for seam in seams:
        src_spec, dst_spec = specs[seam.src_gpu], specs[seam.dst_gpu]
        src_inv = _inventory(src_spec)
        dst_inv = _inventory(dst_spec)
        # perturb the destination so a band that is never written is visible
        for a in dst_inv.values():
            a *= -1.0

        want = _reference_with_spec(seam, src_inv, dst_inv, src_spec, dst_spec)
        bands, nbytes = _bands(seam, src_inv, dst_inv, NZ, src_spec, dst_spec,
                               names)
        assert nbytes == sum(b.nbytes for b in bands) > 0
        buf = _emulate_pack(bands, src_inv, nbytes)
        _emulate_unpack(bands, dst_inv, buf)

        for name in names:
            assert np.array_equal(dst_inv[name], want[name]), (
                f"{axis} seam gpu{seam.src_gpu}->gpu{seam.dst_gpu} field "
                f"{name!r}: packed round trip differs from slice assignment")
    return len(seams)


def test_round_trip_x_seams():
    """x bands: many short runs. A row-pitch error shows up here."""
    assert _round_trip_case(gx=2, gy=1, nx=32, ny=24, axis="x") >= 2


def test_round_trip_y_seams():
    """y bands: few long runs. A slice-pitch error shows up here, not above."""
    assert _round_trip_case(gx=1, gy=2, nx=32, ny=24, axis="y") >= 2


def test_round_trip_two_dimensional_split():
    """Both axes in one plan, which is where the corner rounds live."""
    assert _round_trip_case(gx=2, gy=2, nx=32, ny=24, axis="x") >= 4
    assert _round_trip_case(gx=2, gy=2, nx=32, ny=24, axis="y") >= 4


def test_round_trip_uneven_split():
    """Sub-domains of DIFFERENT width, so src_pitch != dst_pitch.

    Not a corner case for its own sake: on an even split every sub-domain has
    the same extents, the two pitches are equal, and swapping them is a no-op
    that no even-split test can see.  ``nx=33`` over two ranks gives 22 and 23,
    and the pitches separate.  Verified by mutation: swapping the pitches is
    caught here and by nothing above it.
    """
    specs = plan_split(33, 24, HALO, gx=2, gy=1)
    assert len({s.cnx for s in specs}) > 1, "split was even; the case is vacuous"
    assert _round_trip_case(gx=2, gy=1, nx=33, ny=24, axis="x") >= 2


def test_pack_is_one_contiguous_run_per_seam():
    """The whole point: the bands TILE the buffer with no gaps or overlaps.

    If they did not, "one transfer" would be moving padding or, worse, would be
    leaving a field's bytes uninitialised in the middle of the buffer.
    """
    specs = plan_split(32, 24, HALO, gx=2, gy=2)
    seams = seam_plan(specs, HALO, 32, 24)
    names = list(_inventory(specs[0]).keys())
    for seam in seams:
        src_spec, dst_spec = specs[seam.src_gpu], specs[seam.dst_gpu]
        bands, nbytes = _bands(seam, _inventory(src_spec),
                               _inventory(dst_spec), NZ, src_spec, dst_spec,
                               names)
        off = 0
        for b in bands:
            assert b.buf_off == off, "staging buffer has a gap or an overlap"
            off += b.nbytes
        assert off == nbytes


def test_every_exchanged_field_gets_a_band():
    """A field silently dropped from the pack is the ``mup`` bug's shape.

    :doc:`tilestream/SEAM-IN-SURFACE-PRESSURE.md`: an exchange set that quietly
    omits a carrier is bit-exact on a uniform state and wrong on a real one.
    Only sub-2-D fields may be skipped, and only because they are identical on
    every GPU.
    """
    specs = plan_split(32, 24, HALO, gx=2, gy=1)
    seam = seam_plan(specs, HALO, 32, 24)[0]
    inv = _inventory(specs[0])
    inv["column"] = np.arange(NZ, dtype=np.float64)      # vertical-only
    names = list(inv.keys())
    bands, _ = _bands(seam, inv, _inventory(specs[1]) | {"column": inv["column"]},
                      NZ, specs[0], specs[1], names)
    banded = {b.name for b in bands}
    assert banded == set(names) - {"column"}


# ==========================================================================
# the GPU half: the acceptance
# ==========================================================================

def _ngpu() -> int:
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _busy_cards(idle_frac: float = 0.90) -> list:
    """Every visible card that is NOT essentially free, as ``(dev, free_GiB)``.

    The acceptance below is skipped if ANY card on the box is busy, not just
    the two it would use.  That is deliberate and it is the evidence doc's
    rule, not caution for its own sake: on this hardware CONTENTION CHANGES
    RESULTS, not merely timings -- a 3x3 plan that is bit-exact on an idle card
    came out wrong by 3.7e+02 when another process shared the GPU.  A
    bit-exactness test that runs on a contended box can therefore report a
    failure that is about the neighbour, or, worse, be believed when it passes.

    Free memory is the probe rather than ``nvidia-smi --query-compute-apps``
    because that query returns ZERO for a process that has launched but not yet
    created its context.

    A card so full that it will not grant a CUDA CONTEXT counts as busy too,
    and that is not a hypothetical: probing this box while a 22.9 GiB job held
    card 0 raised ``cudaErrorMemoryAllocation`` from ``setDevice`` itself, so a
    probe that only reads free memory crashes before it can report anything.
    A device that refuses a context is the strongest busy signal there is.
    """
    import cupy as cp

    busy = []
    for d in range(_ngpu()):
        try:
            with cp.cuda.Device(d):
                free, total = cp.cuda.runtime.memGetInfo()
        except Exception as exc:                             # noqa: BLE001
            busy.append((d, f"no context: {type(exc).__name__}"))
            continue
        if free < idle_frac * total:
            busy.append((d, round(free / 2**30, 2)))
    return busy


@pytest.mark.gpu
def test_packed_matches_direct():
    """Packed exchange == unpacked reference, bit for bit, and issues fewer transfers.

    The acceptance for the coalesced exchange. ``exchange_direct`` is a
    genuinely separate implementation -- no staging buffer, one pitched copy
    per field -- so equality here is a claim about the arithmetic, not about
    running the same code twice.
    """
    if _ngpu() < 2:
        pytest.skip("needs 2 visible GPUs")
    busy = _busy_cards()
    if busy:
        pytest.skip(f"cards busy, and contention changes RESULTS here: {busy} "
                    "(dev, free GiB)")
    from tilestream import harness as _harness
    from tilestream.multigpu import MultiGPUDomain

    nx = ny = 96
    cfg = _harness.make_config(nx, ny, NZ)
    digests = {}
    counts = {}
    for mode in ("direct", "stream"):
        dom = MultiGPUDomain(
            cfg, ngpu=2, transport="peer",
            state_factory=lambda c: _harness.make_state(c, seed=7))
        counts[mode] = dom.transfer_counts()
        dom.run(3, step_mode="sequential", exchange_mode=mode)
        digests[mode] = dom.hash()
        dom.close()

    assert digests["stream"] == digests["direct"], (
        "the packed exchange is not bit-exact against the unpacked reference: "
        f"{digests['stream']} vs {digests['direct']}")
    assert counts["stream"]["packed"] < counts["stream"]["direct"], (
        "the pack must issue strictly fewer link crossings than the reference")


def _mark(arr, gpu: int, ordinal: int):
    """Fill ``arr`` with values unique to (gpu, field, element).

    Per-band content markers, so an unwritten halo is not merely "different",
    it says WHOSE bytes are sitting in it.  Sub-domain bases are 1e6 apart and
    no sub-domain array here is that large, so a value can only have come from
    the GPU its base names; the arithmetic stays exact in float32 (every
    value < 2**24).
    """
    import cupy as cp

    base = 1.0e6 * (gpu + 1) + 1.0e4 * (ordinal + 1)
    host = base + np.arange(int(arr.size), dtype=np.float64).reshape(arr.shape)
    arr[...] = cp.asarray(host.astype(arr.dtype))
    return cp.asnumpy(arr).copy()


def _oracle(dom, marked: list) -> list:
    """What the seam plan MEANS, as slice assignment on host arrays.

    Built from ``dom.seams`` and the variant classification alone -- never from
    a band -- because the bands are the thing under test.  Rounds run in plan
    order and each round reads a SNAPSHOT taken before it writes, which is the
    ordering the exchange itself has: every channel in a round packs before any
    channel in that round unpacks.
    """
    from tilestream import gather as _gather

    want = [{k: v.copy() for k, v in inv.items()} for inv in marked]
    for phase in sorted({s.phase for s in dom.seams}):
        before = [{k: v.copy() for k, v in inv.items()} for inv in want]
        for seam in [s for s in dom.seams if s.phase == phase]:
            sp = dom.specs[seam.src_gpu]
            for name in dom.exchange_names:
                s = before[seam.src_gpu][name]
                if s.ndim < 2:
                    continue
                d = want[seam.dst_gpu][name]
                v = _gather.classify(s.shape, NZ, int(sp.cny), int(sp.cnx),
                                     layers_ok=True)
                sl_s, sl_d = seam.src_sel[v], seam.dst_sel[v]
                if seam.axis == "x":
                    d[..., sl_d] = s[..., sl_s]
                else:
                    d[..., sl_d, :] = s[..., sl_s, :]
    return want


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["blocking", "stream", "direct"])
def test_exchange_matches_the_slice_oracle(mode):
    """Every exchange path lands the bytes the SEAM PLAN means, not a twin's.

    ``test_packed_matches_direct`` compares two paths that are built from one
    band list, so it cannot see a fault in the list: a channel that silently
    drops a field drops it from both arms and the comparison stays green.
    That is not hypothetical -- it is the mutant this test was written for,
    and the same drop moves the run digest, so the tree was wrong and the
    suite said nothing.  The oracle here is independent of the bands.

    An UNEVEN split (nx=97 over two ranks: 54 and 55 columns) because the two
    pitches are equal on an even one, and one exchange rather than a run,
    because a stepped comparison can only see faults the dynamics preserve --
    the diagnostics p/al/alt are recomputed in the halo every stage and a
    seam that never moves them is invisible to any digest taken after a step.
    """
    if _ngpu() < 2:
        pytest.skip("needs 2 visible GPUs")
    busy = _busy_cards()
    if busy:
        pytest.skip(f"cards busy, and contention changes RESULTS here: {busy} "
                    "(dev, free GiB)")
    import cupy as cp

    from tilestream import harness as _harness
    from tilestream.multigpu import MultiGPUDomain

    cfg = _harness.make_config(97, 96, NZ)
    dom = MultiGPUDomain(
        cfg, ngpu=2, transport="peer",
        state_factory=lambda c: _harness.make_state(c, seed=7))
    try:
        assert len({int(s.cnx) for s in dom.specs}) > 1, (
            "the split came out even; the pitches do not separate and the "
            "case is vacuous")
        marked = [{name: _mark(inv[name], g, i)
                   for i, name in enumerate(dom.names)}
                  for g, inv in enumerate(dom.arrays)]
        want = _oracle(dom, marked)

        {"blocking": dom.exchange_blocking,
         "stream": dom.exchange_stream,
         "direct": dom.exchange_direct}[mode]()
        dom.sync_all()

        for g in range(2):
            for name in dom.names:
                got = cp.asnumpy(dom.arrays[g][name])
                if np.array_equal(got, want[g][name]):
                    continue
                bad = np.argwhere(got != want[g][name])
                first = tuple(int(i) for i in bad[0])
                pytest.fail(
                    f"{mode} exchange left gpu{g} field {name!r} differing "
                    f"from the slice assignment the seam plan means at "
                    f"{len(bad)} of {got.size} points; first at {first}: "
                    f"got {got[first]!r}, expected {want[g][name][first]!r} "
                    f"-- a value near {1.0e6 * (g + 1):.0f} is this GPU's own "
                    "marker and means the halo was never written; a zero "
                    "means the staging buffer was never filled there")
    finally:
        dom.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
