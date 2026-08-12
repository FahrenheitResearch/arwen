"""THE SHARING GATE: one workspace for every buffer must change no answer.

Run it::

    python -m tilestream.test_share           # from the repository root

:mod:`tilestream.shared_workspace` takes 98.2% of a tile buffer's device
footprint away from the buffer and gives it to the run.  That is a large
saving and it is exactly the shape of change that produces a plausible wrong
answer: nothing crashes, nothing warns, and two tiles quietly integrate
through each other's scratch.  So the saving is worth nothing without this
file.

THE POSITIVE ROWS
-----------------
Every shared configuration is compared to the SAME monolithic reference the
bit-exact gate uses (``test_gate.physics_reference``), over the whole restart
manifest, by SHA-256 per field.  Not close: identical.

THE NEGATIVE ROWS, AND WHAT EACH ONE IS FOR
-------------------------------------------
* ``chain_compute=False`` -- THE CONTROL THIS WORKSTREAM OWES.  Each buffer
  owns a non-blocking stream and nothing orders tile *i*'s ``step`` against
  tile *i+1*'s, so with one shared arena the two write the same bytes while
  both are live.  On a dry 192x192 domain split 3x3, on an RTX 5090, that is
  reproducible and total: ALL NINE carriers differ, four runs out of four,
  and the same configuration with the chain on is bit-exact.

  ON AN RTX 4090 THE SAME RUN IS BIT-EXACT, and the first version of this
  file asserted the mismatch unconditionally and therefore FAILED on the
  4090 -- for the code being safe.  Whether two tiles overlap is a property
  of the card: a 96x96x49 tile already occupies the whole 4090, so the
  second stream's blocks cannot be co-scheduled, while the 5090 has room for
  both.  A control that only watches the ANSWER cannot tell "safe" from "the
  hazard did not happen to fire here", and calling the 4090 safe on that
  evidence is exactly the shape of the six false results this project has
  already produced.

  So the control watches the MECHANISM.  ``run_tiled(timeline=True)`` records
  a CUDA event either side of every tile's ``step`` and reports how many
  tiles began before their predecessor finished.  The rows then assert:

    - with the chain: bit-exact AND zero overlapping steps.  That is the
      safety property, it is checkable on every card, and it is the one that
      licenses the sharing.
    - without the chain: if the timeline shows overlap, the answer MUST have
      changed; if it shows none, the row is reported VACUOUS on this card
      rather than passing quietly.

  At full physics the chain-off configuration matches on both cards, because
  ``dycore.step`` at that rung host-synchronises inside the physics guards --
  RRTMGP's single validation D2H (rrtmgp.py:1620) and the finite checks in
  ``_validated_array`` -- so the host cannot enqueue tile *i+1* until tile
  *i*'s kernels have run.  :func:`main` reports that as a DIAGNOSTIC and
  asserts nothing from it.

* a CARRYING slot forced into the arena -- ``mp_rainnc`` and friends are
  accumulators: they are read, added to, and written, so a second buffer
  sharing their storage destroys the first buffer's total.  The lifetime
  audit in ``gpuwm.core.preflight`` excludes them by name, and this row
  proves that exclusion is doing work rather than being decorative.  It is
  the difference between "the arena is safe" and "the arena's ADMISSION RULE
  is safe".

* poison between tiles -- the inverse control, and the one that must PASS.
  Every arena slot is filled with NaN between tiles; a slot that genuinely
  carried something across the tile boundary turns into NaN and every
  downstream field with it.  Bit-exactness under poisoning is direct
  evidence that nothing live crosses the boundary, as opposed to the
  indirect evidence of "it matched anyway".

WHY THE TILE PLAN IS 4x4 AND NOT 2x1
------------------------------------
A negative control that shares a workspace only needs TWO buffers to be
wrong, but it needs enough tiles for the wrongness to reach the comparison.
The plan here is the gate's own 256x192 four-by-four (FACT 1's geometry), so
16 tiles pass through ``nbuffers`` buffers and every buffer is reused four
times.
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings

import cupy as cp

from tilestream import driver, gather, harness, physics_inventory as physinv
from tilestream import shared_workspace, spec as tspec, vram
from tilestream.test_gate import NZ, SEED, physics_cfg, physics_reference


#: FACT 1's geometry: at the 96x80 gate size a 48x40 tile with halo 16
#: already gathers 83% by 90% of the domain, so a buffer barely differs from
#: the whole domain and a sharing defect has almost nowhere to show.  256x192
#: split 4x4 into 64x48 makes a gathered tile ~40% of the domain and runs 16
#: tiles through the buffers.
NX, NY = 256, 192
TILE_NX, TILE_NY = 64, 48
RUNG = "full+MYNN+Noah-MP"
NSTEPS = 3


def _poison_between_tiles(shared):
    """A ``run_tiled`` progress callback that NaNs every shared backing.

    ``progress`` fires on the host after tile *i*'s scatter has been ENQUEUED,
    so the device may still be working on it, and it fires OUTSIDE the
    ``with stream:`` block, so ``poison``'s fills go to the default stream --
    which a non-blocking tile stream does not synchronise against.  BOTH
    synchronizations are therefore part of the control:

    * before, or the fill lands on top of tile *i*'s kernels while they are
      still reading the arena;
    * after, or the fill lands in the MIDDLE of tile *i+1*'s step.  MEASURED:
      without the trailing synchronize this row raised ``MYNN mass-flux
      inputs must be finite`` on one run of two, which is a harness race
      producing a NaN, not evidence about sharing.

    They make this row slower than the others, which is the correct trade for
    a control.
    """

    def progress(_istep, _itile, _tspec):
        cp.cuda.runtime.deviceSynchronize()
        shared.poison()
        cp.cuda.runtime.deviceSynchronize()

    return progress


def share_case(*, share: bool = True, chain=None, rrtmgp_column_chunk=None,
               mynn_column_chunk=None,
               force_slots: tuple[str, ...] = (), poison_between: bool = False,
               nbuffers: int = 2, parts=("scratch", "dycore", "rrtmgp"),
               nsteps: int = NSTEPS, nx: int = NX, ny: int = NY,
               tile_nx: int = TILE_NX, tile_ny: int = TILE_NY,
               rung: str = RUNG) -> dict:
    """One tiled run against the monolithic reference, with memory numbers.

    ``mynn_column_chunk`` is set for the TILED arm only and restored
    afterwards, so the monolithic reference is always the one the bit-exact
    gate built at the shipped width.  That is the point of the row: a chunk
    width is a capacity knob, and if it were also an answer knob this
    comparison would say so.
    """
    cfg, start, start_scalars, ref_arrays, ref_scalars = physics_reference(
        rung, nx, ny, nsteps, nz=NZ, seed=SEED)
    ref = physinv.field_digests(ref_arrays)
    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)
    tspec.validate_plan(specs, ny, nx)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)

    vram.trim_pool()
    before = vram.device_snapshot()
    previous_chunk = (None if mynn_column_chunk is None
                      else shared_workspace.set_mynn_column_chunk(
                          mynn_column_chunk))
    shared = None
    if share:
        shared = shared_workspace.build(
            tile_cfg,
            scratch="scratch" in parts, dycore="dycore" in parts,
            rrtmgp="rrtmgp" in parts,
            rrtmgp_column_chunk=rrtmgp_column_chunk,
            force_slots=force_slots)

    store = {k: cp.asarray(v) for k, v in start.items()}
    scalars = dict(start_scalars)
    report: dict = {}
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(
                store, cfg, tile_nx, tile_ny, halo=halo, nsteps=nsteps,
                nbuffers=nbuffers, report=report, chain_compute=chain,
                progress=(_poison_between_tiles(shared)
                          if (poison_between and shared is not None)
                          else None),
                inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
                tile_state_factory=lambda tc:
                    driver.make_physics_tile_state(tc, shared=shared),
                scalars=scalars, shared=shared)
        cp.cuda.runtime.deviceSynchronize()
    finally:
        if previous_chunk is not None:
            shared_workspace.set_mynn_column_chunk(previous_chunk)
    elapsed = time.perf_counter() - t0
    peak = vram.device_snapshot()

    got = physinv.field_digests(physinv.carrier_inventory(store))
    differing = sorted(k for k in ref if ref.get(k) != got.get(k))
    record = {
        "bitexact": not differing,
        "carriers": len(ref),
        "differing": differing,
        "tiles": len(specs),
        "seconds": elapsed,
        "chain": report.get("chain_compute"),
        "shared_bytes": report.get("shared_bytes", 0),
        "pool_used": peak["pool_used"] - before["pool_used"],
        "pool_total": peak["pool_total"] - before["pool_total"],
        "scalars_ok": scalars == ref_scalars,
        "radiation_calls": scalars.get("call_counts", {}).get("radiation"),
        "cumulus_calls": scalars.get("call_counts", {}).get("cumulus"),
        "tile_cfg": tile_cfg,
    }
    del store, shared
    vram.trim_pool()
    return record


#: The dry concurrency control's geometry.  192x192 split 3x3 into 64x64 is
#: milestone one's own gate shape, so the reference it is compared against is
#: the one the bit-exact gate already trusts.
DRY_NX = DRY_NY = 192
DRY_TILE = 64


def _dry_tile_factory(shared):
    """A dry tile buffer, optionally drawing from the shared workspaces.

    ``driver.make_tile_state`` builds exactly this state; it is reproduced
    here only because it has no ``shared`` seam of its own -- the dry lane is
    not where the megabytes are, so the product change stayed on the physics
    factory.
    """
    import numpy as np

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DomainState, init_at_rest

    def factory(tile_cfg):
        coord = make_vertical_coord(tile_cfg.nz, hybrid_opt=tile_cfg.hybrid_opt,
                                    etac=tile_cfg.etac)
        base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                               p_surf=tile_cfg.p_surf, ztop=tile_cfg.ztop,
                               terrain_z=None)
        if shared is None:
            return init_at_rest(tile_cfg, coord, base)
        state = DomainState(tile_cfg, scratch_arena=shared.arena,
                            dycore_state_workspace=shared.dycore)
        state.load_base(coord, base)
        return state

    return factory


def dry_case(*, share: bool, chain, nsteps: int = 3, nbuffers: int = 2) -> dict:
    """One dry tiled run against a monolithic dry reference."""
    import numpy as np

    cfg = harness.make_config(DRY_NX, DRY_NY, NZ)
    state = harness.make_state(cfg, SEED)
    start = {k: cp.asnumpy(v).copy()
             for k, v in harness.state_arrays(state).items()}
    harness.run_steps(state, cfg, nsteps)
    ref = {k: cp.asnumpy(v).copy()
           for k, v in harness.state_arrays(state).items()}
    del state
    vram.trim_pool()

    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(DRY_NX, DRY_NY, DRY_TILE, DRY_TILE, halo, True)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    shared = (shared_workspace.build(tile_cfg, rrtmgp=False) if share
              else None)
    store = {k: cp.asarray(v) for k, v in start.items()}
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, DRY_TILE, DRY_TILE, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers, shared=shared,
                         chain_compute=chain, report=report, timeline=True,
                         tile_state_factory=_dry_tile_factory(shared))
    cp.cuda.runtime.deviceSynchronize()
    differing = sorted(k for k in ref
                       if not np.array_equal(cp.asnumpy(store[k]), ref[k]))
    out = {"bitexact": not differing, "differing": differing,
           "carriers": len(ref), "tiles": len(specs),
           "chain": report.get("chain_compute"),
           "overlaps": report.get("overlapping_steps"),
           "shared_bytes": report.get("shared_bytes", 0)}
    del store, shared
    vram.trim_pool()
    return out


#: ``(label, kwargs, expectation)``.  The expectation is not always a
#: bit-exactness verdict, because WHETHER TWO TILES OVERLAP IS A PROPERTY OF
#: THE CARD.  ``"exact"`` and ``"differ"`` are unconditional; ``"hazard"``
#: means "if the timeline shows the steps overlapped, the answer MUST have
#: changed, and if it shows they did not, this card cannot exhibit the defect
#: and the row is reported as vacuous".
#:
#: MEASURED, and the reason this row is written this way: at 96x96x49 tiles,
#: 3x3, nbuffers=2, an RTX 5090 overlaps and all nine carriers differ (four
#: runs of four); an RTX 4090 does NOT overlap and comes out bit-exact.  The
#: first version of this gate asserted the mismatch unconditionally, passed
#: on the 5090, and FAILED on the 4090 -- for the code being safe.
DRY_CASES: tuple[tuple[str, dict, str], ...] = (
    ("dry control: no sharing", dict(share=False, chain=None), "exact"),
    ("dry SAFETY PROPERTY: shared arena WITH the compute chain -- bit-exact "
     "AND zero overlapping steps",
     dict(share=True, chain=True), "exact-and-serial"),
    ("dry NEGATIVE: shared arena, compute chain OFF "
     "(the buffer is reused while still live)",
     dict(share=True, chain=False), "hazard"),
    ("dry NEGATIVE, repeated -- a race that passes once proves nothing",
     dict(share=True, chain=False), "hazard"),
)


def _dry_verdict(rec: dict, expect: str) -> tuple[bool, str]:
    """``(ok, why)`` for one dry row under its own kind of expectation."""
    overlaps = rec.get("overlaps")
    if expect == "exact":
        return rec["bitexact"], "bit-exact over all 9 carriers"
    if expect == "exact-and-serial":
        ok = rec["bitexact"] and overlaps == 0
        return ok, (f"bit-exact over all 9 carriers, {overlaps} overlapping "
                    f"steps")
    if expect == "hazard":
        if overlaps:
            return (not rec["bitexact"],
                    f"{overlaps} overlapping steps and "
                    f"{len(rec['differing'])}/{rec['carriers']} carriers "
                    f"differ")
        return True, ("VACUOUS on this card: the steps did not overlap "
                      "(0 overlapping), so the shared arena was never "
                      "written by two live tiles and the answer is "
                      f"{'bit-exact' if rec['bitexact'] else 'wrong anyway'}")
    raise ValueError(f"unknown dry expectation {expect!r}")


#: ``(label, kwargs, expect_bit_exact)``.
CASES: tuple[tuple[str, dict, bool], ...] = (
    ("baseline: no sharing at all (control that the harness compares)",
     dict(share=False), True),
    ("shared scratch arena only",
     dict(parts=("scratch",)), True),
    ("shared scratch + rebuilt state symbols",
     dict(parts=("scratch", "dycore")), True),
    ("shared scratch + rebuilt + RRTMGP chunk workspace",
     dict(), True),
    ("shared everything, RRTMGP column_chunk 1024",
     dict(rrtmgp_column_chunk=1024), True),
    ("shared everything, MYNN column chunk 4096 (a quarter of the shipped "
     "workspace)",
     dict(mynn_column_chunk=4096), True),
    ("shared everything, nbuffers=3",
     dict(nbuffers=3), True),
    ("shared everything, arena POISONED between every tile",
     dict(poison_between=True), True),
    ("NEGATIVE carrying slot mp_rainnc forced into the shared arena",
     dict(force_slots=("mp_rainnc",)), False),
)

#: Reported, not asserted.  See the module docstring: at this rung the host
#: synchronisations inside physics keep the tiles from ever overlapping, so
#: the answer comes out right for a reason that has nothing to do with the
#: chain.  The dry matrix is where the hazard is demonstrated.
DIAGNOSTIC_CASES: tuple[tuple[str, dict], ...] = (
    ("DIAGNOSTIC full physics, chain OFF -- matches today because physics "
     "host-synchronises inside the step, not because sharing is safe",
     dict(chain=False)),
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}")
    print(f"{NX}x{NY}x{NZ} periodic, {TILE_NX}x{TILE_NY} tiles, "
          f"halo {harness.halo_radius(physics_cfg(RUNG))}, N={NSTEPS}, "
          f"rung {RUNG}")
    print("=" * 78)

    failures: list[str] = []

    print("DRY MATRIX  (192x192x49, 3x3 tiles of 64x64, N=3) -- this is "
          "where the")
    print("concurrency hazard is visible, because a dry step never "
          "host-synchronises")
    print("-" * 78)
    for label, kwargs, expect in DRY_CASES:
        try:
            rec = dry_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        ok, detail = _dry_verdict(rec, expect)
        print(f"  {'PASS' if ok else 'FAIL':4s}  {detail}")
        print(f"        {label}")
        print(f"        chain={rec['chain']}  overlapping steps "
              f"{rec['overlaps']}  shared "
              f"{rec['shared_bytes'] / 2**20:.1f} MiB")
        if not ok:
            failures.append(
                f"{label}: expected {expect}, got "
                f"{'bit-exact' if rec['bitexact'] else 'a mismatch'} with "
                f"{rec['overlaps']} overlapping steps")
    print()

    print("PHYSICS MATRIX")
    print("-" * 78)
    cases = CASES[:4] + CASES[-1:] if quick else CASES
    for label, kwargs, expect in cases:
        try:
            rec = share_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        ok = rec["bitexact"] == expect
        verdict = "PASS" if ok else "FAIL"
        detail = ("bit-exact over all "
                  f"{rec['carriers']} carriers" if rec["bitexact"]
                  else f"{len(rec['differing'])}/{rec['carriers']} differ")
        print(f"  {verdict:4s}  {detail}")
        print(f"        {label}")
        print(f"        chain={rec['chain']}  shared "
              f"{rec['shared_bytes'] / 2**20:.1f} MiB  "
              f"pool_used {rec['pool_used'] / 2**20:.1f} MiB  "
              f"pool_total {rec['pool_total'] / 2**20:.1f} MiB  "
              f"radiation fired {rec['radiation_calls']}x, cumulus "
              f"{rec['cumulus_calls']}x  {rec['seconds']:.2f} s")
        if not rec["bitexact"]:
            print(f"        first differing: {rec['differing'][:6]}")
        if not ok:
            failures.append(
                f"{label}: expected "
                f"{'bit-exact' if expect else 'a MISMATCH'}, got "
                f"{'bit-exact' if rec['bitexact'] else 'a mismatch'}")
        if rec["bitexact"] and not rec["scalars_ok"]:
            failures.append(f"{label}: arrays match, scalar clock does not")

    if not quick:
        print()
        for label, kwargs in DIAGNOSTIC_CASES:
            rec = share_case(**kwargs)
            print(f"  ----  {'bit-exact' if rec['bitexact'] else 'mismatch'} "
                  f"({len(rec['differing'])}/{rec['carriers']} differ)")
            print(f"        {label}")

    print("=" * 78)
    if failures:
        print(f"SHARING GATE FAILED -- {len(failures)} problem(s):")
        for line in failures:
            print(f"  * {line}")
        return 1
    print("SHARING GATE PASSED -- every shared configuration is bit-exact "
          "and every negative control still fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
