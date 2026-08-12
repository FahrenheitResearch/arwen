"""SKEPTIC PROBE 2: the controls the nest gate does not carry.

Four questions the gate answers by assertion rather than by measurement:

``--mode instrument``  does ``_require_fired`` actually FAIL when a scheme
                       does not fire?  A refusal that has never been made to
                       refuse is not a control.
``--mode refusals``    do the two REFUSALS the report calls "honest failure"
                       -- a streamed CHILD, and a cross-scheme microphysics
                       edge off a streamed parent -- actually raise?
``--mode vram``        what does streaming the parent COST on the device?
                       ``refresh_from_store`` writes the state's OWN device
                       arrays, so the fix needs the whole prepared parent
                       resident.  Measured on both legs with memGetInfo.
``--mode mutants2``    the mutants probe 1 did not cover.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from tilestream import test_nest as tn


# --------------------------------------------------------------------------

def mode_instrument(size, rung, nsteps):
    """Make ``_require_fired`` refuse, and check ``_cadence``'s ``?``.

    Two deliberate breaks:
      1. a scalars dict in which radiation fired 0 times at a _FULL rung;
      2. a scalars dict with the key ABSENT.
    Both must land in ``failures``.  Then the same two against the rung the
    function is told to skip (``dry``), which must NOT.
    """
    ok = True
    full = dict(call_counts=dict(radiation=3, cumulus=2, ysu=20,
                                 sfclay=20, noah=20),
                microphysics_updates=20, elapsed_seconds=180.0)
    cases = [
        ("all fired, full rung", full, "full(real74)+KF", False),
        ("radiation=0", dict(full, call_counts=dict(full["call_counts"],
                                                    radiation=0)),
         "full(real74)+KF", True),
        ("cumulus=0", dict(full, call_counts=dict(full["call_counts"],
                                                  cumulus=0)),
         "full(real74)+KF", True),
        ("ysu=0", dict(full, call_counts=dict(full["call_counts"], ysu=0)),
         "full(real74)+KF", True),
        ("ysu key ABSENT", dict(full, call_counts={
            k: v for k, v in full["call_counts"].items() if k != "ysu"}),
         "full(real74)+KF", True),
        ("microphysics_updates=0", dict(full, microphysics_updates=0),
         "full(real74)+KF", True),
        ("radiation=0 at the dry rung", dict(
            full, call_counts=dict(full["call_counts"], radiation=0)),
         "dry", False),
    ]
    print("== INSTRUMENT CONTROL: does _require_fired refuse?")
    for label, scalars, rg, must_fail in cases:
        failures = []
        tn._require_fired(scalars, "probe", rg, failures)
        fired = bool(failures)
        good = fired == must_fail
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:34s} rung={rg:16s} "
              f"refused={fired} (wanted {must_fail})"
              + (f"  -> {failures[0][:70]}" if failures else ""))
    print("\n== INSTRUMENT CONTROL: _cadence must print ? for an absent key,"
          " 0 for a measured zero")
    a = tn._cadence(dict(call_counts=dict(radiation=0), elapsed_seconds=1.0))
    b = tn._cadence(dict(call_counts=dict(radiation=1, cumulus=0, ysu=2,
                                          sfclay=2, noah=2),
                         elapsed_seconds=1.0, microphysics_updates=5))
    print(f"  absent keys : {a}")
    print(f"  present keys: {b}")
    ok &= ("cu=?" in a and "pbl=?" in a and "rad=0" in a)
    ok &= ("cu=0" in b and "pbl=2" in b)

    print("\n== INSTRUMENT CONTROL: does the OLD cadence line lie?")
    old = " ".join(
        [f"rad={b_.get('radiation', 0)}" for b_ in
         [dict(radiation=1, cumulus=1, ysu=20, sfclay=20, noah=20)]]
        + ["pbl=" + str(dict(radiation=1, cumulus=1, ysu=20, sfclay=20,
                             noah=20).get("pbl", 0))])
    print(f"  a driver with ysu=20 reported by the OLD code as: {old}")
    ok &= old.endswith("pbl=0")

    print("\n== COMPARE() BLINDNESS: two runs that are both NaN")
    nan = {"state/thp": np.full((4, 4), np.nan, dtype=np.float32)}
    r = tn.compare(nan, dict(nan))
    print(f"  compare(NaN, NaN) -> bitexact={r['bitexact']} "
          f"ndiff={r['ndiff']} nonfinite={r['nonfinite']}")
    print("  leg 3's ok3 is (r1.bitexact and r2.bitexact and feedback_count"
          " equal) -- it does NOT include nonfinite, and _finite_or_raise is"
          " applied to leg 1 only.  So a leg-3 pair that both went NaN would"
          " report PASS with max|d|=0.")
    print(f"\n{'INSTRUMENT CONTROLS PASSED' if ok else 'INSTRUMENT CONTROLS FAILED'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def mode_refusals(size, rung, nsteps):
    """The two refusals must RAISE, not run."""
    import cupy as cp

    from gpuwm.core import streaming
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.dycore import step as dycore_step
    from types import SimpleNamespace

    print("== REFUSAL 1: a STREAMED CHILD must be refused by force()")
    pcfg = tn.parent_cfg(size, rung)
    ccfg = tn.child_cfg(size, rung)
    bnd = tn.build_boundaries(size, rung)
    pstate, _ = tn.build_domain(pcfg, seed=tn.SEED, boundaries=bnd, warmup=1)
    cstate, _ = tn.build_child(ccfg, seed=tn.SEED + 5)
    pclock, cclock = tn.make_clocks(parent_steps=nsteps)
    parent = SimpleNamespace(
        cfg=tn.domain_config(pcfg, grid_id=1, parent_id=0, ratio=1, i0=1, j0=1),
        state=pstate, clock=pclock, parent=None)
    child = SimpleNamespace(
        cfg=tn.domain_config(ccfg, grid_id=2, parent_id=1, ratio=tn.RATIO,
                             i0=size["i_parent_start"],
                             j0=size["j_parent_start"]),
        state=cstate, clock=cclock, parent=parent)
    # publish a FAKE store on the CHILD -- the coupler must refuse
    setattr(cstate, streaming._STORE_ATTR, {"state/mup": cstate.mup})
    coupler = NestCoupler(child, feedback=0)
    from gpuwm.core.state import refresh_model_time
    refresh_model_time(pstate, pclock, kernel_launch=True)
    pclock.prepare_step()
    dycore_step(pstate, pcfg, refl_10cm_due=False)
    pclock.advance()
    refresh_model_time(pstate, pclock, after_step=True)
    ok1 = False
    try:
        coupler.force(child)
        print("  FAIL  a streamed CHILD was FORCED without complaint")
    except RuntimeError as exc:
        ok1 = "STREAMED child" in str(exc)
        print(f"  {'PASS' if ok1 else 'FAIL'}  raised: {str(exc)[:120]}")
    except Exception as exc:                                  # noqa: BLE001
        tn._reraise_if_busy(exc)
        print(f"  ?     raised something else: {type(exc).__name__}: {exc}")
    delattr(cstate, streaming._STORE_ATTR)

    # CONTROL FOR THE REFUSAL: with the store removed it must NOT refuse.
    ok1b = False
    try:
        coupler.force(child)
        ok1b = True
        print("  PASS  control: the same child with no store IS forced")
    except Exception as exc:                                  # noqa: BLE001
        tn._reraise_if_busy(exc)
        print(f"  FAIL  control: refused anyway: {type(exc).__name__}: {exc}")

    del pstate, cstate, coupler, parent, child, bnd
    cp.get_default_memory_pool().free_all_blocks()

    print("\n== REFUSAL 2: cross-scheme microphysics off a STREAMED parent")
    from gpuwm.core import nest as N
    import inspect
    src = inspect.getsource(N.NestCoupler._coupled_parent_field)
    has = ("transition_handles_field" in src and "_is_streamed" in src
           and "unimplemented" in src)
    print(f"  the guard is present in _coupled_parent_field: {has}")
    print("  (it is reachable only with a microphysics_transition configured;"
          " this gate never configures one, so it is UNEXERCISED code)")
    return 0 if (ok1 and ok1b) else 1


# --------------------------------------------------------------------------

def mode_vram(size, rung, nsteps):
    """What does streaming the parent actually cost on the DEVICE?

    ``refresh_from_store`` writes the state's own device arrays.  Those exist
    only because ``attach`` copied OUT of a resident prepared state and
    nothing freed it.  So the streamed leg holds the whole parent resident
    AND a pinned host store AND the tile buffers.  If that is so, streaming a
    parent you intend to nest off buys no device capacity at all, which is
    the entire purpose of the mode.
    """
    import cupy as cp

    print(f"== DEVICE COST  d01 {size['parent_nx']}^2 tile {size['tile']}  "
          f"rung={rung}  {nsteps} steps")
    bnd = tn.build_boundaries(size, rung)
    cp.get_default_memory_pool().free_all_blocks()
    out = {}
    for label, kw in (("leg1 RESIDENT", dict(stream_d01=False, feedback=0)),
                      ("leg2 STREAMED", dict(stream_d01=True, feedback=0))):
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        free0, total = cp.cuda.runtime.memGetInfo()
        peak = {"v": 0}

        # sample the device high-water through the leg by wrapping dycore.step
        from gpuwm.core import dycore
        real = dycore.step

        def spy(*a, **k):
            r = real(*a, **k)
            f, _ = cp.cuda.runtime.memGetInfo()
            peak["v"] = max(peak["v"], free0 - f)
            return r

        dycore.step = spy
        try:
            res = tn.leg(size, rung, nsteps=nsteps, boundaries=bnd, **kw)
        finally:
            dycore.step = real
        cp.cuda.runtime.deviceSynchronize()
        out[label] = peak["v"]
        print(f"  {label:16s} device high-water during the leg: "
              f"{peak['v'] / 2**30:.2f} GiB  (of {total / 2**30:.1f} GiB)")
        del res
        cp.get_default_memory_pool().free_all_blocks()
    a, b = out["leg1 RESIDENT"], out["leg2 STREAMED"]
    print(f"\n  streamed / resident device high-water = {b / max(a, 1):.3f}x")
    print("  A ratio at or above 1.0 means streaming the parent of a nest"
          " saves NO device memory:")
    print("  the coupler's refresh writes state.mup/thp/php/qv/... which are"
          " the resident domain.")
    return 0


# --------------------------------------------------------------------------

class _Mutant2:
    def __init__(self, kind):
        self.kind = kind

    def __enter__(self):
        from gpuwm.core import nest as N

        self._sin = N._sync_in
        kind = self.kind
        if kind == "no_feedback_pull":
            # the FORCE pull asks for exactly 2 attrs; feedback asks for many
            def sin(state, attrs):
                return 0 if len(attrs) > 2 else self._sin(state, attrs)
            N._sync_in = sin
        elif kind == "no_force_pull":
            def sin(state, attrs):
                return 0 if len(attrs) <= 2 else self._sin(state, attrs)
            N._sync_in = sin
        else:
            raise ValueError(kind)
        return self

    def __exit__(self, *exc):
        from gpuwm.core import nest as N

        N._sync_in = self._sin
        return False


def mode_mutants2(size, rung, nsteps):
    import cupy as cp

    print(f"== MUTANTS 2  rung={rung} d01 {size['parent_nx']}^2 {nsteps} steps")
    bnd = tn.build_boundaries(size, rung)
    for kind, fb in (("no_feedback_pull", 1), ("no_force_pull", 1),
                     ("no_force_pull", 0)):
        ref = tn.leg(size, rung, stream_d01=False, feedback=fb, nsteps=nsteps,
                     boundaries=bnd)
        cp.get_default_memory_pool().free_all_blocks()
        with _Mutant2(kind):
            got = tn.leg(size, rung, stream_d01=True, feedback=fb,
                         nsteps=nsteps, boundaries=bnd)
        r1 = tn.compare(ref["d01"], got["d01"])
        r2 = tn.compare(ref["d02"], got["d02"])
        noticed = bool(r1["ndiff"] or r2["ndiff"])
        print(f"  {'NOTICED' if noticed else 'INVISIBLE':9s} fb={fb} {kind:20s}"
              f" d01 {r1['ndiff']}/{r1['ntotal']} max|d|={r1['max_abs']:.4g};"
              f" d02 {r2['ndiff']}/{r2['ntotal']} max|d|={r2['max_abs']:.4g}")
        del ref, got
        cp.get_default_memory_pool().free_all_blocks()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="instrument",
                    choices=["instrument", "refusals", "vram", "mutants2"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--medium", action="store_true")
    ap.add_argument("--rung", default="full(real74)+KF")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args(argv)

    size = (tn.QUICK_SIZE if args.quick else
            (tn.MEDIUM_SIZE if args.medium else tn.FULL_SIZE))
    nsteps = args.steps if args.steps else (6 if args.quick else tn.PARENT_STEPS)
    if args.mode != "instrument":
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"cupy {cp.__version__} "
              f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} "
              f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    try:
        return dict(instrument=mode_instrument, refusals=mode_refusals,
                    vram=mode_vram, mutants2=mode_mutants2)[args.mode](
                        size, args.rung, nsteps)
    except tn.CardTooBusy as exc:
        print(f"INTERRUPTED, not failed: {exc}")
        return 3
    except Exception as exc:                                  # noqa: BLE001
        try:
            tn._reraise_if_busy(exc)
        except tn.CardTooBusy as busy:
            print(f"INTERRUPTED, not failed: {busy}")
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
