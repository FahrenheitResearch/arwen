"""SKEPTIC probe for DEFECT 2 (safety observers folded out of the store).

Nothing here is a re-statement of ``tilestream.test_obsfold``.  Every mode
exists because a specific claim in the report is either unproven, proven by a
control that cannot fail, or proven somewhere other than where the shipped
code runs.

    --mode fires      the bit-exact claim, with radiation and cumulus counted
                      by MONKEYPATCHING THE DRIVER (PhysicsDriver._run_radiation
                      / _run_cumulus), not by reading call_counts -- the number
                      the fix itself now supplies.  Also digests the final store
                      and the final resident state.
    --mode missrate   the negative controls, over many seeds, reporting the
                      PER-RUN and PER-SUBSTEP miss rate rather than "it fired".
    --mode price      the fold's marginal cost against a step whose length is
                      also reported, over a window that CROSSES the radiation
                      and cumulus cadences, with the fire counts printed.
    --mode runtimeloop  drives the SHIPPED loop, gpuwm.runtime's own
                      integrate_prepared_case substep block, rather than the
                      copy of it that test_obsfold keeps.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gpuwm.core import streaming
from tilestream import harness, test_join
from tilestream import test_obsfold as T


# --------------------------------------------------------------------------
# independent fire counting
# --------------------------------------------------------------------------

class FireCounter:
    """Counts the ACTUAL radiation / cumulus / PBL invocations.

    ``call_counts`` is a counter the code under test maintains and that the
    fix now also transports; using it to prove the fix fired would be circular.
    This wraps the driver's own private entry points instead, so the number is
    a count of executions.

    On the streamed leg every TILE has its own driver, so one firing model
    step produces ``ntiles`` invocations.  Both the raw count and the set of
    domain-clock times at which they happened are kept, because "16 calls" and
    "one step where all 16 tiles fired" are the same number for very different
    reasons.
    """

    def __init__(self):
        self.radiation = 0
        self.cumulus = 0
        self.pbl = 0
        self.radiation_times: set[float] = set()
        self.cumulus_times: set[float] = set()
        self._saved = None

    def install(self):
        from gpuwm.core import physics as _phys

        counter = self
        cls = _phys.PhysicsDriver
        self._saved = {}

        def wrap(name, field, times=None):
            real = getattr(cls, name)
            self._saved[name] = real

            def patched(self_driver, *a, **kw):
                setattr(counter, field, getattr(counter, field) + 1)
                if times is not None:
                    state = a[1] if len(a) > 1 else kw.get("state")
                    if state is not None:
                        times.add(round(float(state.elapsed_seconds), 6))
                return real(self_driver, *a, **kw)

            setattr(cls, name, patched)

        wrap("_run_radiation", "radiation", self.radiation_times)
        wrap("_run_cumulus", "cumulus", self.cumulus_times)
        return self

    def remove(self):
        from gpuwm.core import physics as _phys

        for name, real in (self._saved or {}).items():
            setattr(_phys.PhysicsDriver, name, real)
        self._saved = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def summary(self, ntiles: int = 1) -> dict:
        return {
            "radiation_calls": self.radiation,
            "cumulus_calls": self.cumulus,
            "radiation_calls_per_tile": self.radiation / max(1, ntiles),
            "cumulus_calls_per_tile": self.cumulus / max(1, ntiles),
            "radiation_model_times": sorted(self.radiation_times),
            "cumulus_model_times": sorted(self.cumulus_times),
        }


def store_digest(store) -> str:
    """SHA-256 over the whole inventory, ``test_join.digest_arrays``'s rules."""
    import hashlib

    per = test_join.digest_arrays(store)
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode())
        h.update(per[name].encode())
    return h.hexdigest()


def state_digest(state) -> str:
    """The same digest, of a resident state, over the SAME carrier names.

    ``physics_inventory.carrier_inventory`` is the inventory the store is
    built from, so the two digests are over the same set by construction --
    this is ``test_join``'s comparator, not a new one.
    """
    from tilestream import physics_inventory as _pi

    return store_digest(_pi.carrier_inventory(state, None))


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def mode_fires(args) -> int:
    """Both legs, clean, with the fires counted from the driver itself."""
    import cupy as cp

    cfg = T.obs_cfg(args.n, args.n, rung=args.rung, dx=args.dx, dt=args.dt)
    print(f"# {args.n}^2 x {cfg.nz} dx={cfg.dx/1000:g} km dt={cfg.dt:g} s "
          f"rung={args.rung!r} tile={args.tile} steps={args.steps}")
    print(f"# radiation every {cfg.radt_minutes*60/cfg.dt:g} steps, cumulus "
          f"every {cfg.cudt_minutes*60/cfg.dt:g}; window is {args.steps}")
    bnd = T.sequential_boundaries(cfg)

    out = {}
    with FireCounter() as fc_r:
        res_r, state = T.run_resident(cfg, args.steps, boundaries=bnd,
                                      poison_step=None)
        out["resident_fires_measured"] = fc_r.summary(1)
        out["resident_digest"] = state_digest(state)
    print(T._report_line("resident", res_r))
    print(f"    call_counts       = {res_r['fires']}")
    print(f"    MEASURED (driver) = {out['resident_fires_measured']}")
    print(f"    digest            = {out['resident_digest']}")
    del state
    cp.get_default_memory_pool().free_all_blocks()

    with FireCounter() as fc_s:
        res_s, stepper = T.run_streamed(cfg, args.steps, boundaries=bnd,
                                        tile=args.tile, poison_step=None,
                                        observer="fold")
        ntiles = int(res_s["tiles"])
        out["streamed_fires_measured"] = fc_s.summary(ntiles)
        out["streamed_digest"] = store_digest(stepper.store)
    print(T._report_line("streamed[fold]", res_s))
    print(f"    call_counts       = {res_s['fires']}   tiles={ntiles}")
    print(f"    MEASURED (driver) = {out['streamed_fires_measured']}")
    print(f"    digest            = {out['streamed_digest']}")

    cmp = T.compare_traces(res_r["trace"], res_s["trace"])
    out.update(resident=res_r, streamed=res_s, compare=cmp)
    rad_r = out["resident_fires_measured"]["radiation_calls"]
    cu_r = out["resident_fires_measured"]["cumulus_calls"]
    rad_s = out["streamed_fires_measured"]["radiation_calls_per_tile"]
    cu_s = out["streamed_fires_measured"]["cumulus_calls_per_tile"]
    print("\n-- THE GATE (skeptic)")
    print(T._line("radiation FIRED on both legs, measured at the driver",
                  rad_r > 0 and rad_s > 0, f"resident {rad_r}, "
                  f"streamed {rad_s:g}/tile"))
    print(T._line("cumulus FIRED on both legs, measured at the driver",
                  cu_r > 0 and cu_s > 0, f"resident {cu_r}, "
                  f"streamed {cu_s:g}/tile"))
    print(T._line("the two legs fired the same number of times",
                  rad_r == rad_s and cu_r == cu_s,
                  f"radiation {rad_r} vs {rad_s:g}; cumulus {cu_r} vs {cu_s:g}"))
    print(T._line("the two legs fired at the same MODEL TIMES",
                  (out["resident_fires_measured"]["radiation_model_times"]
                   == out["streamed_fires_measured"]["radiation_model_times"])
                  and (out["resident_fires_measured"]["cumulus_model_times"]
                       == out["streamed_fires_measured"]["cumulus_model_times"]),
                  f"rad {out['resident_fires_measured']['radiation_model_times']}"
                  f" vs {out['streamed_fires_measured']['radiation_model_times']}"))
    print(T._line("every substep's report is bit-identical",
                  not cmp["mismatches"],
                  f"{len(cmp['mismatches'])} mismatch(es) over "
                  f"{cmp['steps_compared']} substeps x {len(T.FOLD_KEYS)}"))
    for bad in cmp["mismatches"][:8]:
        print(f"      step {bad['step']} {bad['key']}: "
              f"resident={bad['resident']!r} streamed={bad['streamed']!r}")
    moved = len({r["w_max"] for r in res_s["trace"]})
    print(T._line("the fold is ALIVE", moved > 1,
                  f"{moved} distinct w_max over {len(res_s['trace'])}"))
    print(T._line("the two legs' persisted state digests agree",
                  out["resident_digest"] == out["streamed_digest"],
                  "SHA-256 over the whole persisted inventory"))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    del stepper
    cp.get_default_memory_pool().free_all_blocks()
    return 0


def mode_missrate(args) -> int:
    """Every control, over many seeds: how often does it FAIL to fire?"""
    import cupy as cp

    cfg = T.obs_cfg(args.n, args.n, rung=args.rung, dx=args.dx, dt=args.dt)
    print(f"# {args.n}^2 x {cfg.nz} rung={args.rung!r} tile={args.tile} "
          f"steps={args.steps} trials={args.trials}")
    bnd = T.sequential_boundaries(cfg)
    controls = (args.control.split(",") if args.control
                else ["halo", "tileindex", "dropface"])
    rows = []
    for control in controls:
        misses = 0
        substep_misses = 0
        substeps = 0
        for trial in range(args.trials):
            seed = T.SEED + 1000 * (trial + 1)
            res_r, state = T.run_resident(cfg, args.steps, boundaries=bnd,
                                          poison_step=None, seed=seed)
            del state
            cp.get_default_memory_pool().free_all_blocks()
            res_s, stepper = T.run_streamed(
                cfg, args.steps, boundaries=bnd, tile=args.tile,
                poison_step=None, observer="fold", seed=seed, control=control)
            cmp = T.compare_traces(res_r["trace"], res_s["trace"])
            hit_steps = {d["step"] for d in cmp["mismatches"]}
            substeps += cmp["steps_compared"]
            substep_misses += cmp["steps_compared"] - len(hit_steps)
            if not cmp["mismatches"]:
                misses += 1
            rows.append({"control": control, "trial": trial, "seed": seed,
                         "mismatches": len(cmp["mismatches"]),
                         "substeps_hit": len(hit_steps),
                         "substeps": cmp["steps_compared"],
                         "fields": sorted({d["key"] for d in cmp["mismatches"]})})
            print(f"  {control:<10} trial {trial:>3} seed {seed}: "
                  f"{len(cmp['mismatches']):>4} mismatch(es) on "
                  f"{len(hit_steps)}/{cmp['steps_compared']} substeps "
                  f"{'FIRED' if cmp['mismatches'] else '*** MISS ***'}",
                  flush=True)
            del stepper
            cp.get_default_memory_pool().free_all_blocks()
        print(f"\n  CONTROL {control!r}: run miss rate "
              f"{misses}/{args.trials} = {misses/args.trials:.1%}; "
              f"substep miss rate {substep_misses}/{substeps} = "
              f"{substep_misses/max(1,substeps):.1%}\n", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    return 0


def mode_price(args) -> int:
    """The fold's cost, and the step it is a fraction OF, with fires shown."""
    import cupy as cp

    cfg = T.obs_cfg(args.n, args.n, rung=args.rung, dx=args.dx, dt=args.dt)
    print(f"# {args.n}^2 x {cfg.nz} rung={args.rung!r} tile={args.tile} "
          f"steps/leg={args.steps}")
    print(f"# radiation every {cfg.radt_minutes*60/cfg.dt:g} steps, cumulus "
          f"every {cfg.cudt_minutes*60/cfg.dt:g}")
    bnd = T.sequential_boundaries(cfg)
    with FireCounter() as fc:
        res = T.price(cfg, args.steps, args.tile, bnd)
        fires = fc.summary(1)
    ntiles = int(res["tiles"])
    print(json.dumps(res, indent=2, default=str))
    print(f"\n  fires over the WHOLE price run (4 legs of {args.steps}): "
          f"{fires['radiation_calls']} radiation, {fires['cumulus_calls']} "
          f"cumulus invocations over {ntiles} tiles = "
          f"{fires['radiation_calls']/ntiles:g} / "
          f"{fires['cumulus_calls']/ntiles:g} model steps")
    print(f"  radiation model times: {fires['radiation_model_times'][:8]}")
    print(f"  cumulus   model times: {fires['cumulus_model_times'][:8]}")
    print(f"\n  streamed step, fold ON : "
          f"{res['streamed_step_fold_on_s']*1e3:.1f} ms")
    print(f"  streamed step, fold OFF: "
          f"{res['streamed_step_fold_off_s']*1e3:.1f} ms")
    print(f"  fold marginal          : {res['fold_marginal_s']*1e3:.3f} ms "
          f"= {res['fold_share']*100:.2f}% of a step, over {ntiles} tiles")
    print(f"  resident stability_report on the same domain: "
          f"{res['resident_stability_report_s']*1e3:.3f} ms")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"price": res, "fires": fires}, fh, indent=2,
                      default=str)
    return 0


# --------------------------------------------------------------------------
# THE SHIPPED LOOP
# --------------------------------------------------------------------------

def mode_runtimeloop(args) -> int:
    """Run gpuwm.runtime's OWN substep block, not test_obsfold's copy of it.

    ``test_obsfold.RuntimeObservers`` is a hand-transcribed copy of
    runtime.py's per-substep accounting.  It therefore cannot detect a defect
    in the transcription, and it never executes ``streaming.step_health``,
    ``streaming.domain_call_counts`` or ``streaming.domain_swdown_peak`` --
    the three functions the fix actually adds to the shipped path.

    This mode compiles the REAL block out of runtime.py's own source (the
    lines between the substep call and the end of the outer step) and runs it,
    so any NameError, stale-``report`` or missing-key defect in the shipped
    text shows up here.
    """
    import inspect
    import textwrap

    import cupy as cp
    from gpuwm import runtime
    from gpuwm.core import streaming as _streaming

    src = inspect.getsource(runtime.integrate_prepared_case)
    a0 = src.index("            width = cfg.spec_bdy_width")
    a1 = src.index("        # Both of these are the DOMAIN's")
    b0 = src.index("        surface_forcing_updates = _streaming")
    b1 = src.index("        if step_swdown_peak > swdown_peak:")
    substep_block = textwrap.dedent(src[a0:a1])
    outer_block = textwrap.dedent(src[b0:b1])
    print("-- THE SHIPPED PER-SUBSTEP BLOCK, verbatim from gpuwm/runtime.py")
    print(substep_block)
    print("-- THE SHIPPED PER-OUTER-STEP BLOCK")
    print(outer_block)
    print("-- end --\n")
    compiled = compile(substep_block, "runtime.py:substep", "exec")
    compiled_outer = compile(outer_block, "runtime.py:outer", "exec")

    cfg = T.obs_cfg(args.n, args.n, rung=args.rung, dx=args.dx, dt=args.dt)
    bnd = T.sequential_boundaries(cfg)
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(args.tile), tile_ny=int(args.tile),
        nbuffers=2, store="host")
    decision = streaming.decide(cfg, options)
    pk, pj, pi = T.poison_index(cfg)
    poison = None
    if args.poison_step:
        owner = T.owning_tile(cfg, decision, pj, pi)
        assert owner != 0
        poison = T.StorePoison(step=int(args.poison_step), index=(pk, pj, pi),
                               value=float(args.poison_value))
    domain, geo = test_join.build_domain(cfg, seed=T.SEED, boundaries=bnd,
                                         warmup=1)
    build = test_join._make_builder(
        cfg, domain, geo, boundaries=bnd, seam="zeros", snapshot=None,
        geography=True, carry_clock=True, window_tables=True)
    if poison is not None:
        with poison.install():
            stepper = streaming.make_stepper(domain, cfg, options,
                                             decision=decision, build=build)
        poison.store = stepper.store
    else:
        stepper = streaming.make_stepper(domain, cfg, options,
                                         decision=decision, build=build)

    if args.break_fix:
        # THE NEGATIVE CONTROL FOR THE SHIPPED BLOCK.  Put the pre-fix
        # observer back -- stability_report on the resident DomainState --
        # and leave every other line of runtime.py's text exactly as it is.
        # If the poisoned run still raises with this in place, the raise was
        # never caused by the fix and this whole demonstration proves nothing.
        from gpuwm.core.dycore import stability_report as _sr

        _streaming.step_health = (
            lambda stepper, state, cfg, *, boundary_width:
            _sr(state, cfg, boundary_width=boundary_width))
        print("!! NEGATIVE CONTROL: step_health reverted to "
              "stability_report(state, ...)")

    # The exact names runtime.py's block reads.
    env = {
        "cfg": cfg, "integration_cfg": cfg, "state": domain,
        "stepper": stepper, "_streaming": _streaming, "cp": cp,
        "np": np, "nan_free": True, "w_max": 0.0,
        "w_max_boundary_row": None, "boundary_w_max": 0.0,
        "interior_w_max": 0.0, "swdown_peak": -np.inf,
        "swdown_peak_time": 0.0, "forcing_time": 0.0,
        "surface_forcing_updates": 0,
        "dynamics_substeps": 1, "outer_step": 0, "substep": 0,
    }
    raised = None
    trace = []
    for istep in range(1, int(args.steps) + 1):
        if poison is not None:
            poison.begin_step(istep)
        stepper(domain, cfg, refl_10cm_due=False)
        env["outer_step"] = istep - 1
        try:
            exec(compiled, env)
            if not args.skip_outer:
                exec(compiled_outer, env)
            else:
                env["step_swdown_peak"] = 0.0
        except RuntimeError as exc:
            raised = f"substep {istep}: RuntimeError: {exc}"
            break
        except Exception as exc:                       # noqa: BLE001
            raised = f"substep {istep}: {type(exc).__name__}: {exc}"
            break
        trace.append({"step": istep, "nan_free": bool(env["nan_free"]),
                      "w_max": float(env["w_max"]),
                      "surface_forcing_updates":
                          int(env["surface_forcing_updates"]),
                      "swdown_peak": float(env["step_swdown_peak"])})
    print(f"\n-- ran {len(trace)}/{args.steps} substeps through the SHIPPED "
          f"block; raised={raised!r}")
    for row in trace[::max(1, len(trace) // 12 or 1)]:
        print(f"   step {row['step']:>4} nan_free={row['nan_free']} "
              f"w_max={row['w_max']:.6f} rad_updates="
              f"{row['surface_forcing_updates']} "
              f"swdown_peak={row['swdown_peak']:.4f}")
    if trace:
        print(f"   distinct w_max over the run: "
              f"{len({r['w_max'] for r in trace})}")
        print(f"   distinct swdown_peak: "
              f"{len({r['swdown_peak'] for r in trace})}")
        print(f"   surface_forcing_updates moved: "
              f"{trace[0]['surface_forcing_updates']} -> "
              f"{trace[-1]['surface_forcing_updates']}")
    print(f"   store now: {T.store_truth(stepper.store)}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"trace": trace, "raised": raised,
                       "store": T.store_truth(stepper.store)}, fh, indent=2,
                      default=str)
    return 0


MODES = {"fires": mode_fires, "missrate": mode_missrate,
         "price": mode_price, "runtimeloop": mode_runtimeloop}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--nz", type=int, default=T.NZ)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--control", default=None)
    ap.add_argument("--rung", default=T.RUNG)
    ap.add_argument("--dx", type=float, default=T.DX)
    ap.add_argument("--dt", type=float, default=T.DT)
    ap.add_argument("--poison-step", type=int, default=0)
    ap.add_argument("--poison-value", type=float, default=float("nan"))
    ap.add_argument("--skip-outer", action="store_true",
                    help="run only the per-substep block (the nan gate); the "
                         "per-outer-step block needs a physics driver")
    ap.add_argument("--break-fix", action="store_true",
                    help="negative control: revert step_health to the "
                         "pre-fix stability_report(state, ...)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.n % args.tile:
        raise SystemExit(f"tile {args.tile} must divide {args.n}")
    t0 = time.perf_counter()
    rc = MODES[args.mode](args)
    print(f"\n# {args.mode} finished in {time.perf_counter()-t0:.1f} s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
