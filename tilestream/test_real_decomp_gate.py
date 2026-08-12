"""The real-data slice gate: eight ranks hold ONE analysis, not eight copies.

The synthetic gate (:mod:`tilestream.test_decomp_gate`) proves the transport.
This one proves the thing the transport was built for: that a decomposed run
initialised from a REAL HRRR analysis holds exactly the state a single-GPU run
would have held, and that its initial condition has no seams in it.

Three verdicts, and the first is the one that matters:

1. **BIT-EXACT vs MONOLITHIC.**  The analysis is built once as a single
   domain -- the same ``DomainState`` + ``PhysicsDriver`` the production
   single-GPU runner would step -- then thrown away, then rebuilt as ``P``
   rank windows through :mod:`tilestream.real_init`.  The rank interiors are
   scattered back and compared BIT FOR BIT.  Nothing here is a tolerance:
   either each rank holds the bits the monolithic run holds at the same global
   position, or the slice is wrong.

2. **CONTINUOUS ACROSS THE SEAMS.**  Surface pressure and composite
   reflectivity, measured as mean |first difference| across the rank
   boundaries against the same statistic away from them.  A real analysis has
   a large gradient everywhere, so only the RATIO is interpretable; a correct
   slice sits near 1 because a seam column is an ordinary column with a rank
   boundary drawn on it.

3. **THE ANALYSIS IS ACTUALLY IN THERE.**  Reflectivity and hydrometeor
   maxima, reported.  A slice that is bit-exact against a monolithic state of
   uniform zeros would pass (1) and (2) and be worthless.

Run::

    python -m tilestream.test_real_decomp_gate /path/to/prepared/bundle
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from tilestream import decomp, real_init
from tilestream.test_decomp_gate import (context_verdict,
                                         cuda_contexts)

#: Rank grids to gate, ``(px, py)``.  1x1 is the identity check: if a
#: single-rank "decomposition" is not bit-exact, no multi-rank number means
#: anything.
PLANS = [(1, 1), (2, 1), (2, 2), (4, 2)]


def _plans() -> list[tuple[int, int]]:
    raw = os.environ.get("PLANS")
    if raw:
        return [tuple(int(v) for v in p.split("x")) for p in raw.split(",")]
    return PLANS


def composite_reflectivity(carriers) -> np.ndarray | None:
    """Column-max equivalent reflectivity (dBZ) from the analysed hydrometeors.

    The Stoelinga/RIP sum over rain, snow and graupel.  Present as a check
    that the ANALYSIS is in the state at all -- a bit-exact slice of nothing
    is still nothing -- and as the field a seam is visible in by eye.
    """
    import cupy as cp

    alt = carriers.get("state/alt")
    if alt is None:
        return None
    qr = carriers.get("state/qr")
    qs = carriers.get("state/qs")
    qg = carriers.get("state/qg")
    if qr is None:
        return None
    out = cp.zeros(alt.shape[-2:], dtype=cp.float32)
    for k in range(alt.shape[0]):
        rho = 1.0 / cp.maximum(alt[k], 1e-6)
        zz = 3.630803e9 * cp.power(cp.maximum(qr[k], 0.0) * rho, 1.75)
        if qs is not None:
            zz = zz + 9.80621e8 * cp.power(cp.maximum(qs[k], 0.0) * rho, 1.75)
        if qg is not None:
            zz = zz + 4.33443e10 * cp.power(cp.maximum(qg[k], 0.0) * rho, 1.75)
        out = cp.maximum(out, 10.0 * cp.log10(cp.maximum(zz, 1e-3)))
    return cp.asnumpy(out)


def run_plan(analysis, px: int, py: int, halo: int) -> dict:
    import cupy as cp

    from tilestream import driver as _driver
    from tilestream import physics_inventory as physinv

    nx, ny, nz = analysis.nx, analysis.ny, analysis.nz
    specs = decomp.rank_specs(nx, ny, px, py, halo)

    # -- the monolithic reference, built and then released --------------
    whole = decomp.rank_specs(nx, ny, 1, 1, 0)[0]
    t0 = time.perf_counter()
    _cfg_m, state_m, _drv_m, _rep = real_init.build_rank_state(analysis, whole)
    store = decomp.store_from_state(state_m, nz=nz,
                                    provenance=dict(analysis.provenance))
    ref_cref = composite_reflectivity(physinv.carrier_inventory(state_m))
    build_mono = time.perf_counter() - t0
    del state_m, _drv_m
    cp.get_default_memory_pool().free_all_blocks()

    # -- the ranks -------------------------------------------------------
    t1 = time.perf_counter()
    blocks = []
    grid_reports = []
    for spec in specs:
        _cfg_r, state_r, _drv_r, rep = real_init.build_rank_state(
            analysis, spec)
        grid_reports.append(rep["grid"])
        held = {k: cp.asnumpy(v)
                for k, v in physinv.carrier_inventory(state_r).items()
                if getattr(v, "ndim", 0) >= 2 and hasattr(v, "get")}
        held.update({k: (cp.asnumpy(v) if hasattr(v, "get") else np.asarray(v))
                     for k, v in _driver.geography_inventory(state_r).items()})
        blocks.append(held)
        del state_r, _drv_r
        cp.get_default_memory_pool().free_all_blocks()
    build_ranks = time.perf_counter() - t1

    verdict = decomp.assert_slice_faithful(specs, blocks, store)
    rebuilt = decomp.reassemble(specs, blocks, store)

    seams = {}
    for name in ("fields/psfc", "state/mup", "setup/ht"):
        got = rebuilt.get(name)
        if got is not None and got.ndim == 2:
            seams[name] = decomp.seam_statistics(got, specs, nx=nx, ny=ny)
    if ref_cref is not None:
        seams["composite_reflectivity"] = decomp.seam_statistics(
            ref_cref, specs, nx=nx, ny=ny)

    return dict(specs=specs, verdict=verdict, seams=seams, store=store,
                build_mono=build_mono, build_ranks=build_ranks,
                grid=grid_reports, cref=ref_cref)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m tilestream.test_real_decomp_gate "
              "<prepared-bundle-root>")
        return 2
    from tilestream import harness

    ctx0 = cuda_contexts()
    analysis = real_init.open_bundle(argv[0])
    cfg = analysis.cfg
    halo = harness.halo_radius(cfg)
    print(analysis.summary())
    print(f"cfg: dx={cfg.dx:.0f} m dt={cfg.dt:.0f} s  mp={cfg.mp_physics} "
          f"ra_lw={cfg.ra_lw_physics} ra_sw={cfg.ra_sw_physics} "
          f"cu={cfg.cu_physics} pbl={cfg.bl_pbl_physics} "
          f"sf_surface={cfg.sf_surface_physics} terrain={cfg.terrain_opt} "
          f"ztop={cfg.ztop:.0f}  halo={halo}")
    print(f"domain setup flags (computed over the DOMAIN, re-imposed on every "
          f"rank): {real_init.domain_setup_flags(analysis)}")
    print(f"CUDA CONTEXTS ON THE BOX AT START: {ctx0}"
          + ("   <- CONTENDED; a bit-exactness verdict taken here is NOT a pass"
             if ctx0 > 0
             else "   (nothing on the card yet, this gate included)"))

    failures = 0
    rungs_gated = 0
    arrays_gated = 0
    for px, py in _plans():
        try:
            r = run_plan(analysis, px, py, halo)
        except Exception as exc:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{py}x{px}  ERROR  {exc}")
            failures += 1
            continue
        v = r["verdict"]
        failures += 0 if v["ok"] else 1
        rungs_gated += 1
        arrays_gated += int(v["checked"])
        print(f"\n--- {py}x{px} = {px * py} ranks "
              f"({decomp.describe_plan(r['specs'], analysis.nz)}) ---")
        print(f"  monolithic reference built in {r['build_mono']:.1f} s; "
              f"{px * py} rank windows built in {r['build_ranks']:.1f} s")
        worst = max((g or {}).get("displacement_km", 0.0) for g in r["grid"])
        print(f"  sub-window projection: worst per-rank displacement from the "
              f"domain grid {worst:.6f} km  "
              f"{'PASS' if worst < 1e-6 else 'FAIL'}")
        print(f"  BITEXACT vs MONOLITHIC: {v['checked']} arrays reassembled, "
              f"{len(v['mismatched'])} differ -> "
              f"{'PASS' if v['ok'] else 'FAIL'}")
        for key, info in list(v["mismatched"].items())[:8]:
            print(f"    MISMATCH {key}: {info}")
        for name, st in r["seams"].items():
            for axis in ("x", "y"):
                s = st.get(axis)
                if s is None:
                    continue
                if s["degenerate"]:
                    print(f"  SEAM {name} {axis}: horizontally uniform, "
                          f"carries no seam signal, not gated")
                    continue
                ok = s["ratio"] < 3.0
                failures += 0 if ok else 1
                print(f"  SEAM {name} {axis}: across-boundary mean "
                      f"{s['seam_mean']:.6g}, elsewhere {s['bulk_mean']:.6g}, "
                      f"ratio {s['ratio']:.4f}  {'PASS' if ok else 'FAIL'}")
        cref = r["cref"]
        if cref is not None:
            print(f"  ANALYSIS CONTENT: composite reflectivity "
                  f"{cref.min():.1f}..{cref.max():.1f} dBZ, "
                  f"{100.0 * float((cref > 15.0).mean()):.2f}% of columns "
                  f"above 15 dBZ")

    ctx1, verdict = context_verdict(ctx0)
    print(f"\nCUDA CONTEXTS AT END: {ctx1} (start {ctx0}).  " + verdict)
    if rungs_gated < 1:
        print("REAL-DATA SLICE GATE REFUSED -- 0 rank geometries were gated "
              "against a floor of 1, so no verdict is available.")
        return 2
    print("REAL-DATA SLICE GATE " + ("PASS" if failures == 0
                                     else f"FAIL ({failures})")
          + f" -- {arrays_gated} arrays reassembled over {rungs_gated} rank "
            f"geometries")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
