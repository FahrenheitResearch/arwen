"""The slice gate: a decomposed initial condition IS the monolithic one.

Three questions, in the order that makes a failure diagnosable:

1. **Is the plan a decomposition?**  Every point of the domain owned by
   exactly one rank, every rank the same shape.  Cheap, and it fails first
   when the geometry is wrong, so the later failures are never geometry.

2. **Is the slice bit-identical to the monolithic state?**  Build the domain
   once, slice it into ``py*px`` rank windows, scatter the rank INTERIORS back
   and compare the bit patterns.  This is the test that actually proves the
   slice, it needs one card, and it does not care what the numbers are -- so
   it runs on the analytic builder as well as on a real analysis, and a pass
   on the analytic builder is a pass for the transport.

   Compared on bits rather than with a tolerance on purpose.  An off-by-one in
   a smooth field is CLOSE everywhere; ``allclose`` is exactly the instrument
   that cannot see it.

3. **Is the initial condition continuous across the seams?**  Only meaningful
   on a state with horizontal structure, so the geography rung is the one that
   answers it: mean |first difference| across the rank boundaries against the
   same statistic away from them.  A ratio of many times 1 means the slice put
   the wrong rows next to each other.

WHICH CARD EACH RANK IS BUILT ON.  ``DEVICES`` is the same contract
:mod:`tilestream.test_seam_gate` already uses, and it defaults to card 0 for
every rank, which is what this gate has always done.  The slice is a property
of the geometry, so one card gates it completely -- but a run on one card
cannot answer the separate question of whether the reassembly survives a
PHYSICAL device boundary, because there is not one in it.  ``DEVICES=0,1``
round-robins the rank buffers over the cards listed, so rank ``r`` is
allocated, sliced into and read back on ``devices[r % len(devices)]`` and the
gather crosses real card boundaries.

The transfer path is named rather than assumed.  Every rank's arrays come back
through ``cupy.ndarray.get`` -- device-to-host into this process's memory, then
compared on the host -- so the crossing is STAGED THROUGH HOST by construction
and no peer link is involved even where the hardware has one.  That is the
honest description of what this gate exercises; the peer-vs-staged question on
the SEAM transport belongs to :mod:`tilestream.multigpu`, which chooses between
them.

Run directly::

    python -m tilestream.test_decomp_gate            # default 2x2 and 4x2
    NX=256 NY=192 PX=4 PY=2 python -m tilestream.test_decomp_gate
    PLANS=2x1,2x2 DEVICES=0,1,2,3 python -m tilestream.test_decomp_gate
"""
from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import numpy as np

from tilestream import decomp, harness

#: Small enough that the monolithic reference fits on one card with room to
#: spare, non-square so a y/x transposition cannot pass, and divisible by every
#: rank grid under test.
NX = int(os.environ.get("NX", "256"))
NY = int(os.environ.get("NY", "192"))
NZ = int(os.environ.get("NZ", "49"))

#: Rank grids to gate.  ``(1, 1)`` is included because a one-rank "plan" must
#: reduce to the identity -- if it does not, nothing about the multi-rank
#: results is interpretable.
PLANS = [(1, 1), (2, 1), (2, 2), (4, 2)]


def _devices() -> list[int]:
    """Which physical cards the rank buffers are round-robined over.

    Defaults to ``[0]`` -- every rank on card 0, which is what this gate did
    before the knob existed and is a complete gate of the SLICE.  A list
    spreads the ranks, so that the reassembly is a gather from several
    physical devices rather than several windows of one.  Unlike the seam
    gate's ``DEVICES``, the list here need not be as long as the rank count:
    ranks are assigned ``devices[r % len(devices)]``, so one setting covers
    every plan in a sweep.
    """
    raw = os.environ.get("DEVICES")
    if not raw:
        return [0]
    got = [int(v) for v in raw.split(",") if v.strip() != ""]
    if not got:
        raise SystemExit("DEVICES was set but lists no cards")
    return got


def _plans() -> list[tuple[int, int]]:
    raw = os.environ.get("PLANS")
    if raw:
        return [tuple(int(v) for v in p.split("x")) for p in raw.split(",")]
    if os.environ.get("PX") or os.environ.get("PY"):
        return [(int(os.environ.get("PX", "4")), int(os.environ.get("PY", "2")))]
    return PLANS


def cuda_context_pids():
    """Every pid holding a CUDA context on the WHOLE box, or ``None``.

    ``None`` means nvidia-smi could not be read or did not answer in the
    shape expected, which is NOT the same as an idle box and must never be
    reported as one.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    pids = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            return None
    return pids


def cuda_contexts() -> int:
    """CUDA contexts on the WHOLE box, mine included.  ``-1`` if unreadable.

    Recorded next to every bit-exactness verdict because contention on this
    hardware changes RESULTS, not only timings: the same transfer plan that is
    bit-exact on an idle card has been measured wrong by 3.7e+02 when another
    process shared the GPU.  A pass recorded on a contended box is not a pass,
    and the only way to know afterwards is to have written it down.
    """
    pids = cuda_context_pids()
    return -1 if pids is None else len(pids)


def context_verdict(ctx0: int):
    """One end-of-run nvidia-smi sample, and the sentence it supports.

    Returns ``(ctx1, sentence)``.

    THE DEFECT THIS EXISTS FOR.  Every runner used to decide idleness with
    ``max(ctx0, ctx1) > 1``, so ``CUDA CONTEXTS AT END: 1 (start 0)`` printed
    as "Box idle throughout".  The claim happened to be true -- the one
    context was the gate's own -- but nothing on the line said so, and a
    reader cannot tell that from a run where the 1 is a neighbour.  A verdict
    line that asserts more than its own numbers show is the shape of every
    green-on-nothing finding this lane has already fixed elsewhere.

    So the sentence is derived from the PIDS rather than from the count.
    These runners are single-PROCESS -- no multiprocessing, no spawned ranks,
    whatever ``DEVICES`` spreads the rank buffers over -- so "this pid"
    separates the gate's own contexts from everything else exactly, and it
    keeps doing so on several cards at once because every one of those
    contexts belongs to this pid.  ``ctx0`` is read before the gate touches a
    device, which makes any context standing there at the start somebody
    else's; the retrospective question of whose it was cannot be answered
    from a count taken later, so a non-zero start is reported as contention
    rather than explained away.

    One sample, not two: ``ctx1`` and the ownership split come out of the
    same nvidia-smi call, so the number printed and the sentence beside it
    can never describe different moments.
    """
    pids = cuda_context_pids()
    if pids is None or ctx0 < 0:
        return (-1 if pids is None else len(pids),
                "nvidia-smi could not be read, so whether the box was idle "
                "is UNKNOWN and the verdict is provisional.")
    foreign = [p for p in pids if p != os.getpid()]
    if ctx0 > 0 or foreign:
        return (len(pids),
                f"{ctx0} context(s) stood on the box before this gate touched "
                f"a device and {len(foreign)} belong to another process now, "
                "so the box was NOT idle and the verdict is provisional.")
    return (len(pids),
            ("no process holds a CUDA context on the box, so it was idle "
             "throughout and the verdict is clean." if not pids else
             f"all {len(pids)} of them are this gate's own, so the box was "
             "idle throughout and the verdict is clean."))


# --------------------------------------------------------------------------
# rungs
# --------------------------------------------------------------------------

def geography_config(nx: int, ny: int, nz: int):
    """A REAL projection: latitude varies, map factors are not 1, terrain is on.

    The flat periodic default cannot gate a slice.  Its geography is uniform,
    so a rank that gathered the wrong window would hold the same bits as one
    that gathered the right one and the gate would report a pass it did not
    earn.
    """
    return harness.make_config(nx, ny, nz, periodic=True,
                               **harness.GEOGRAPHY_OVERRIDES)


def build_domain(nx: int, ny: int, nz: int, *, physics: bool, seed: int = 4242):
    """``(cfg, state)`` for the monolithic domain of the requested rung."""
    from tilestream import physics_inventory as physinv

    if physics:
        from tilestream import test_gate
        cfg = harness.make_config(
            nx, ny, nz,
            **dict(test_gate.PHYSICS_RUNGS["full+MYNN+Noah-MP"],
                   **harness.GEOGRAPHY_OVERRIDES))
        geo = harness.make_geography(cfg)
        state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
        return cfg, state
    cfg = geography_config(nx, ny, nz)
    geo = harness.make_geography(cfg)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    return cfg, state


def build_rank(cfg, spec, *, physics: bool, seed: int = 4242, device: int = 0):
    """A rank's buffer: the SAME config with only the horizontal extents cut.

    Allocated inside an explicit ``cp.cuda.Device(device)`` context, because
    CuPy's memory pools are per device and an allocation made under the wrong
    current device is the one way a rank can end up somewhere other than where
    the layout says it is.

    Built by the analytic builder and then completely overwritten by the
    slice.  That is not waste -- the builder is what allocates the carriers
    with the right dtypes, the right layered soil/snow shapes and a driver
    attached -- but it IS why the slice has to be total: anything the store
    has no entry for stays analytic, which is why ``install_slice`` refuses a
    partial one.
    """
    import cupy as cp

    tile_cfg = harness.tile_config(cfg, spec.cnx, spec.cny)
    with cp.cuda.Device(device):
        geo = harness.make_geography(tile_cfg)
        state, _drv = harness.make_physics_state(tile_cfg, seed, geography=geo)
    return state


def _array_device(state) -> int | None:
    """The card a built rank's arrays actually landed on, or ``None``.

    Read off an allocation rather than off the request, so the layout printed
    beside the verdict is the one the driver executed.
    """
    from tilestream import physics_inventory as physinv

    for value in physinv.carrier_inventory(state).values():
        dev = getattr(getattr(value, "data", None), "device_id", None)
        if dev is not None:
            return int(dev)
    return None


# --------------------------------------------------------------------------
# the three checks
# --------------------------------------------------------------------------

def check_plan(nx, ny, px, py, halo) -> dict:
    from tilestream import spec as spec_mod

    specs = decomp.rank_specs(nx, ny, px, py, halo)
    # Every stagger, not just mass: the shared-face ownership rule is what
    # decides who writes a u face that two ranks both touch, and a plan can be
    # a perfect decomposition of the mass grid while double-writing every
    # internal u column.
    per_variant = {}
    for variant in ("mass", "u", "v"):
        counts = spec_mod.coverage_counts(specs, ny, nx, variant)
        per_variant[variant] = np.unique(np.asarray(counts)).tolist()
    covered_once = all(v == [1] for v in per_variant.values())
    return dict(specs=specs, covered_once=covered_once,
                coverage_values=per_variant)


def check_slice(nx, ny, nz, px, py, *, physics: bool,
                devices: Sequence[int] = (0,)) -> dict:
    """Slice the monolithic domain into ranks and reassemble it.

    ``devices`` round-robins the rank buffers over physical cards; the layout
    the driver actually produced is returned as ``rank_devices`` so the caller
    prints what happened rather than what was asked for.
    """
    import cupy as cp

    devices = list(devices) or [0]
    cfg, state = build_domain(nx, ny, nz, physics=physics)
    halo = harness.halo_radius(cfg)
    plan = check_plan(nx, ny, px, py, halo)
    specs = plan["specs"]

    store = decomp.store_from_state(state, nz=nz)
    # The monolithic state is freed BEFORE any rank buffer is allocated: the
    # store is on the host, so the domain and the ranks never have to be
    # resident at the same time.  That is what lets this gate run at a domain
    # sized for one card rather than one card divided by the rank count.
    del state
    cp.get_default_memory_pool().free_all_blocks()

    blocks = []
    reports = []
    rank_devices = []
    for r, spec in enumerate(specs):
        dev = devices[r % len(devices)]
        rank = build_rank(cfg, spec, physics=physics, device=dev)
        rank_devices.append(_array_device(rank))
        from tilestream import driver as _driver
        from tilestream import physics_inventory as physinv
        # Slice and read back under the rank's OWN device.  ``.get()`` on an
        # array whose device is not current is legal but takes a slower path,
        # and install_slice's H2D writes must land in that device's context.
        with cp.cuda.Device(dev):
            reports.append(decomp.install_slice(rank, store, spec))
            held = {k: v.get()
                    for k, v in physinv.carrier_inventory(rank).items()
                    if getattr(v, "ndim", 0) >= 2 and hasattr(v, "get")}
            held.update({k: (v.get() if hasattr(v, "get") else np.asarray(v))
                         for k, v in _driver.geography_inventory(rank).items()})
            blocks.append(held)
            del rank
            cp.get_default_memory_pool().free_all_blocks()

    verdict = decomp.assert_slice_faithful(specs, blocks, store)
    verdict["plan"] = plan["covered_once"]
    verdict["coverage_values"] = plan["coverage_values"]
    verdict["store"] = store.summary()
    verdict["halo"] = halo
    verdict["n_broadcast"] = len(reports[0]["broadcast"]) if reports else 0
    verdict["rank_devices"] = rank_devices
    return verdict, store, specs, blocks


def check_seams(store, specs, blocks) -> dict:
    """Continuity of the reassembled initial condition across rank boundaries.

    Surface pressure is the field to look at numerically: it is smooth, it is
    2-D, and it is the field a wrong slice deforms most visibly.  ``psfc`` is
    used when the rung has it and the dycore's own column-mass prognostic
    ``state/mup`` otherwise -- the same quantity's role, and the one whose
    staleness was already measured to drag 23 other carriers with it.
    """
    rebuilt = decomp.reassemble(specs, blocks, store)
    out = {}
    for name in ("fields/psfc", "state/mup", "setup/ht"):
        got = rebuilt.get(name)
        if got is None or got.ndim != 2:
            continue
        out[name] = decomp.seam_statistics(got, specs, nx=store.nx,
                                           ny=store.ny)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    physics = os.environ.get("PHYSICS", "") not in ("", "0", "no")
    devices = _devices()
    ctx0 = cuda_contexts()
    rung = ("full+MYNN+Noah-MP" if physics
            else "geography (real74 projection + terrain)")
    print(f"DOMAIN {NX} x {NY} x {NZ} = {NX * NY * NZ / 1e6:.2f} Mcell   "
          f"rung = {rung}")
    print(f"RANK BUFFERS ON CARD(S) {devices}   "
          + ("one card, so the reassembly crosses no physical device boundary"
             if len(devices) == 1 else
             "ranks round-robin over these cards, so the reassembly gathers "
             "ACROSS physical devices")
          + "\nTRANSFER PATH: device-to-host (cupy .get) per rank, compared on "
            "the host -- STAGED THROUGH HOST by construction, no peer link is "
            "used or claimed here")
    print(f"CUDA CONTEXTS ON THE BOX AT START: {ctx0}"
          + ("  <- CONTENDED, a bit-exactness verdict taken here is NOT a pass"
             if ctx0 > 0 else "  (nothing on the card yet, this gate included)"))
    failures = 0
    rungs_gated = 0
    arrays_gated = 0
    for px, py in _plans():
        t0 = time.perf_counter()
        try:
            verdict, store, specs, blocks = check_slice(
                NX, NY, NZ, px, py, physics=physics, devices=devices)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{py}x{px}  ERROR  {exc}")
            failures += 1
            continue
        ok = verdict["ok"] and verdict["plan"]
        failures += 0 if ok else 1
        rungs_gated += 1
        arrays_gated += int(verdict["checked"])
        print(f"\n--- {py}x{px} = {px * py} ranks "
              f"({decomp.describe_plan(specs, NZ)}) ---")
        laid = verdict.get("rank_devices") or []
        cards = sorted({d for d in laid if d is not None})
        print(f"  rank -> card: "
              + ", ".join(f"r{r}=gpu{d}" for r, d in enumerate(laid))
              + (f"   ({len(cards)} PHYSICAL device(s) crossed)"
                 if len(cards) > 1 else "   (single device, no boundary)"))
        print(f"  {store.summary()}")
        print(f"  coverage: every point owned {verdict['coverage_values']} "
              f"time(s) -> {'PASS' if verdict['plan'] else 'FAIL'}")
        print(f"  BITEXACT: {verdict['checked']} arrays reassembled, "
              f"{len(verdict['mismatched'])} differ -> "
              f"{'PASS' if verdict['ok'] else 'FAIL'}"
              f"   [{time.perf_counter() - t0:.1f} s]")
        for key, info in list(verdict["mismatched"].items())[:6]:
            print(f"    MISMATCH {key}: {info}")
        seams = check_seams(store, specs, blocks)
        for name, st in seams.items():
            for axis in ("x", "y"):
                s = st.get(axis)
                if s is None:
                    continue
                if s["degenerate"]:
                    print(f"  SEAM {name} {axis}: field is horizontally "
                          f"uniform -- carries no seam signal, not gated")
                    continue
                print(f"  SEAM {name} {axis}: across-boundary mean "
                      f"{s['seam_mean']:.6g}, elsewhere {s['bulk_mean']:.6g}, "
                      f"ratio {s['ratio']:.4f}"
                      + ("  PASS" if s["ratio"] < 3.0 else "  FAIL"))
    ctx1, verdict = context_verdict(ctx0)
    print(f"\nCUDA CONTEXTS AT END: {ctx1} (start {ctx0}).  " + verdict)
    # The size goes in the verdict, not only in the rows above it: a rung
    # list that came out empty would otherwise print GATE PASS having
    # reassembled nothing.  Floor of one, on the certify / dual-run template.
    if rungs_gated < 1:
        print("GATE REFUSED -- 0 rank geometries were gated against a floor "
              "of 1, so no verdict is available.")
        return 2
    print("GATE " + ("PASS" if failures == 0 else f"FAIL ({failures})")
          + f" -- {arrays_gated} arrays reassembled over {rungs_gated} rank "
            f"geometries"
          + (f" on {len(devices)} physical cards {devices}, staged through "
             "host" if len(devices) > 1 else " on one card"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
