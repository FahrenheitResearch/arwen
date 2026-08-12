"""SKEPTIC PROBE for the nest-across-a-streamed-parent fix.

Does not trust ``tilestream.test_nest``'s own cadence line.  Counts every
physics scheme call by wrapping ``PhysicsDriver.compute`` itself, records the
``elapsed_seconds`` each call saw, and reports the counts PER DOMAIN on both
sides of every comparison.  Also reports what ``refresh_from_store`` and
``commit_to_store`` actually FOUND in the store, because both of them
``continue`` past a key the store does not carry -- a silent skip that would
leave a field stale with no error anywhere.

    python -m tilestream.skeptic_probe --mode counts   [--quick|--medium]
    python -m tilestream.skeptic_probe --mode mutants  [--quick|--medium]
    python -m tilestream.skeptic_probe --mode missrate --repeats N
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np

from tilestream import test_nest as tn


# --------------------------------------------------------------------------
# instrument 1: every physics call, counted where it is MADE
# --------------------------------------------------------------------------

class ComputeSpy:
    """Wrap ``PhysicsDriver.compute`` and record one row per call.

    A row is ``(nx, ny, elapsed_at_entry, {scheme: fired})``.  ``nx``/``ny``
    identify the buffer: a resident d01 is the parent shape, a streamed d01
    is a TILE shape (tile + 2*halo), and d02 is the child shape.  That
    distinction is the whole point -- a streamed domain runs its physics on
    tile buffers, so "radiation fired once" on the domain's scalars and
    "radiation fired 16 times" in the process are both true and mean
    different things.
    """

    def __init__(self):
        self.rows: list[tuple] = []

    def __enter__(self):
        from gpuwm.core.physics import PhysicsDriver

        self._real = PhysicsDriver.compute
        spy = self

        def compute(drv, state, *a, **k):
            before = dict(drv.call_counts)
            elapsed = float(state.elapsed_seconds)
            shape = (int(state.thp.shape[-1]), int(state.thp.shape[-2]))
            out = spy._real(drv, state, *a, **k)
            fired = {kk: int(vv - before.get(kk, 0))
                     for kk, vv in drv.call_counts.items()
                     if int(vv - before.get(kk, 0)) > 0}
            mp = int(getattr(drv, "microphysics_updates", 0))
            spy.rows.append((shape, elapsed, fired, mp))
            return out

        PhysicsDriver.compute = compute
        return self

    def __exit__(self, *exc):
        from gpuwm.core.physics import PhysicsDriver

        PhysicsDriver.compute = self._real
        return False

    def by_shape(self):
        agg: dict[tuple, collections.Counter] = {}
        calls: collections.Counter = collections.Counter()
        elapsed: dict[tuple, list] = {}
        for shape, el, fired, _mp in self.rows:
            agg.setdefault(shape, collections.Counter()).update(fired)
            calls[shape] += 1
            elapsed.setdefault(shape, []).append(el)
        return agg, calls, elapsed


# --------------------------------------------------------------------------
# instrument 2: what the store handshake actually MOVED
# --------------------------------------------------------------------------

class SyncSpy:
    """Record every ``refresh_from_store`` / ``commit_to_store`` request.

    Records the attrs ASKED FOR and the attrs actually FOUND, because both
    functions ``continue`` past a key the store does not carry.  A field that
    is asked for and never found is a hole in the fix that raises nothing.
    """

    def __init__(self):
        self.reads: list[tuple[tuple, tuple, int]] = []
        self.writes: list[tuple[tuple, tuple, int]] = []

    def __enter__(self):
        from gpuwm.core import streaming as st

        self._rr, self._cc = st.refresh_from_store, st.commit_to_store
        spy = self

        def refresh(state, attrs):
            store = st.domain_store(state)
            found = () if store is None else tuple(
                a for a in attrs if f"state/{a}" in store)
            n = spy._rr(state, attrs)
            if store is not None:
                spy.reads.append((tuple(attrs), found, n))
            return n

        def commit(state, attrs):
            store = st.domain_store(state)
            found = () if store is None else tuple(
                a for a in attrs if f"state/{a}" in store)
            n = spy._cc(state, attrs)
            if store is not None:
                spy.writes.append((tuple(attrs), found, n))
            return n

        st.refresh_from_store, st.commit_to_store = refresh, commit
        return self

    def __exit__(self, *exc):
        from gpuwm.core import streaming as st

        st.refresh_from_store, st.commit_to_store = self._rr, self._cc
        return False

    def missing(self):
        out = set()
        for asked, found, _n in self.reads + self.writes:
            out |= set(asked) - set(found)
        return sorted(out)


# --------------------------------------------------------------------------
# mode: counts
# --------------------------------------------------------------------------

def _fmt(agg, calls, elapsed, label):
    print(f"    {label}")
    for shape in sorted(agg, reverse=True):
        c = agg[shape]
        els = elapsed[shape]
        print(f"      buffer {shape[0]}x{shape[1]}  compute calls={calls[shape]}"
              f"  rad={c['radiation']} cu={c['cumulus']} ysu={c['ysu']}"
              f" sfclay={c['sfclay']} noah={c['noah']}"
              f"  elapsed {min(els):g}..{max(els):g}"
              f" ({len(set(els))} distinct)")


def mode_counts(size, rung, nsteps):
    """Fire counts on BOTH sides, counted at the call site, plus elapsed."""
    print(f"== INDEPENDENT FIRE COUNTS  rung={rung}  "
          f"d01 {size['parent_nx']}^2 tile {size['tile']}  {nsteps} steps")
    bnd = tn.build_boundaries(size, rung)
    out = {}
    for name, kw in (("leg1 RESIDENT fb=0", dict(stream_d01=False, feedback=0)),
                     ("negctl  STREAMED no-store",
                      dict(stream_d01=True, feedback=0, store_aware=False)),
                     ("leg2 STREAMED fb=0", dict(stream_d01=True, feedback=0)),
                     ("leg3ref RESIDENT fb=1",
                      dict(stream_d01=False, feedback=1)),
                     ("leg3 STREAMED fb=1",
                      dict(stream_d01=True, feedback=1))):
        with ComputeSpy() as spy, SyncSpy() as sync:
            res = tn.leg(size, rung, nsteps=nsteps, boundaries=bnd, **kw)
        agg, calls, els = spy.by_shape()
        out[name] = (res, agg, calls, els, sync)
        print(f"  -- {name}")
        _fmt(agg, calls, els, "physics compute() calls, per buffer shape")
        print(f"      gate's own d01 scalars: {tn._cadence(res['d01_scalars'])}")
        print(f"      gate's own d02 scalars: {tn._cadence(res['d02_scalars'])}")
        if sync.reads or sync.writes:
            rb = sum(n for _a, _f, n in sync.reads)
            wb = sum(n for _a, _f, n in sync.writes)
            print(f"      store handshake: {len(sync.reads)} reads "
                  f"{rb / 2**20:.1f} MiB, {len(sync.writes)} writes "
                  f"{wb / 2**20:.1f} MiB, "
                  f"attrs asked-for but NOT IN STORE: {sync.missing()}")
        import cupy as cp
        del res
        cp.get_default_memory_pool().free_all_blocks()

    # SYMMETRY: the thing the gate never checks.
    print("\n  == SYMMETRY of the DOMAIN-LEVEL fire counts (gate checks only"
          " that they are non-zero) ==")
    ref = out["leg1 RESIDENT fb=0"][0]
    for name in ("negctl  STREAMED no-store", "leg2 STREAMED fb=0"):
        got = out[name][0]
        for dom in ("d01", "d02"):
            a = ref[f"{dom}_scalars"].get("call_counts", {})
            b = got[f"{dom}_scalars"].get("call_counts", {})
            same = a == b
            print(f"    {dom} {name:28s} counts equal to leg1? {same}"
                  f"   leg1={dict(sorted(a.items()))}"
                  f"  got={dict(sorted(b.items()))}")
        print(f"    d01 elapsed leg1={ref['d01_scalars']['elapsed_seconds']} "
              f"vs {name}={got['d01_scalars']['elapsed_seconds']}")
    return 0


# --------------------------------------------------------------------------
# mode: mutants -- break PARTS of the fix and see whether the gate notices
# --------------------------------------------------------------------------

class _Mutant:
    """Break one piece of the fix; everything else stays armed."""

    def __init__(self, name, kind):
        self.name, self.kind = name, kind

    def __enter__(self):
        from gpuwm.core import nest as N
        from gpuwm.core import streaming as st

        self._sin, self._sout = N._sync_in, N._sync_out
        self._din, self._dout = N._DIAGNOSTIC_INPUTS, N._DIAGNOSTIC_OUTPUTS
        kind = self.kind

        if kind == "no_mup_on_force":
            # pull the FIELD but not the coupling mass mup
            def sin(state, attrs):
                return self._sin(state, tuple(a for a in attrs if a != "mup"))
            N._sync_in = sin
        elif kind == "no_field_on_force":
            # pull mup but not the field itself
            def sin(state, attrs):
                return self._sin(state, tuple(a for a in attrs if a == "mup"))
            N._sync_in = sin
        elif kind == "no_diag_inputs":
            N._DIAGNOSTIC_INPUTS = ()
        elif kind == "no_qv_diag_input":
            N._DIAGNOSTIC_INPUTS = tuple(
                a for a in self._din if a != "qv")
        elif kind == "no_diag_outputs":
            N._DIAGNOSTIC_OUTPUTS = ()
        elif kind == "no_commit_kinds":
            # feedback_commit writes only mup back, not the restricted kinds
            def sout(state, attrs):
                if set(attrs) & set(self._dout):
                    return self._sout(state, attrs)      # finalize still fine
                return self._sout(state, ("mup",))
            N._sync_out = sout
        elif kind == "no_commit_mup":
            def sout(state, attrs):
                if set(attrs) & set(self._dout):
                    return self._sout(state, attrs)
                return self._sout(state, tuple(a for a in attrs if a != "mup"))
            N._sync_out = sout
        elif kind == "stale_by_one":
            # the store read happens, but from a ONE-STEP-OLD copy: proves
            # the gate can see a small desync, not only a total one.
            pass
        else:
            raise ValueError(kind)
        self._st = st
        return self

    def __exit__(self, *exc):
        from gpuwm.core import nest as N

        N._sync_in, N._sync_out = self._sin, self._sout
        N._DIAGNOSTIC_INPUTS, N._DIAGNOSTIC_OUTPUTS = self._din, self._dout
        return False


MUTANTS = [
    ("force: pull the field but NOT mup", "no_mup_on_force", 0),
    ("force: pull mup but NOT the field", "no_field_on_force", 0),
    ("feedback: no diagnostic INPUTS pulled", "no_diag_inputs", 1),
    ("feedback: qv not pulled for update_diagnostics", "no_qv_diag_input", 1),
    ("feedback: diagnostic OUTPUTS not committed", "no_diag_outputs", 1),
    ("feedback: restricted kinds not committed", "no_commit_kinds", 1),
    ("feedback: mup not committed", "no_commit_mup", 1),
]


def mode_mutants(size, rung, nsteps):
    """Every mutant MUST make leg 2 or leg 3 stop being bit-exact."""
    import cupy as cp

    print(f"== MUTANTS  rung={rung}  d01 {size['parent_nx']}^2 "
          f"tile {size['tile']}  {nsteps} steps")
    bnd = tn.build_boundaries(size, rung)
    refs = {}
    for fb in (0, 1):
        refs[fb] = tn.leg(size, rung, stream_d01=False, feedback=fb,
                          nsteps=nsteps, boundaries=bnd)
        cp.get_default_memory_pool().free_all_blocks()
    # sanity: the UNMUTATED fix is bit-exact at both feedbacks
    blind = []
    for fb in (0, 1):
        got = tn.leg(size, rung, stream_d01=True, feedback=fb, nsteps=nsteps,
                     boundaries=bnd)
        r1 = tn.compare(refs[fb]["d01"], got["d01"])
        r2 = tn.compare(refs[fb]["d02"], got["d02"])
        print(f"  baseline fb={fb}: d01 ndiff={r1['ndiff']}/{r1['ntotal']} "
              f"d02 ndiff={r2['ndiff']}/{r2['ntotal']} "
              f"nonfinite={r1['nonfinite']}/{r2['nonfinite']}")
        if r1["ndiff"] or r2["ndiff"]:
            print("  !! BASELINE IS NOT BIT-EXACT; mutants below mean nothing")
        del got
        cp.get_default_memory_pool().free_all_blocks()

    for label, kind, fb in MUTANTS:
        try:
            with _Mutant(label, kind):
                got = tn.leg(size, rung, stream_d01=True, feedback=fb,
                             nsteps=nsteps, boundaries=bnd)
            r1 = tn.compare(refs[fb]["d01"], got["d01"])
            r2 = tn.compare(refs[fb]["d02"], got["d02"])
            noticed = bool(r1["ndiff"] or r2["ndiff"])
            print(f"  {'NOTICED' if noticed else 'INVISIBLE':9s} fb={fb}  "
                  f"{label:48s} d01 {r1['ndiff']}/{r1['ntotal']} "
                  f"max|d|={r1['max_abs']:.4g}; d02 {r2['ndiff']}/{r2['ntotal']}"
                  f" max|d|={r2['max_abs']:.4g}"
                  f" nonfinite={r1['nonfinite']}/{r2['nonfinite']}")
            del got
        except Exception as exc:                              # noqa: BLE001
            tn._reraise_if_busy(exc)
            print(f"  RAISED    fb={fb}  {label:48s} "
                  f"{type(exc).__name__}: {exc}")
            noticed = True
        if not noticed:
            blind.append(label)
        cp.get_default_memory_pool().free_all_blocks()
    print()
    if blind:
        print(f"  GATE IS BLIND TO {len(blind)} MUTANT(S): {blind}")
    else:
        print("  every mutant was noticed")
    return 0


# --------------------------------------------------------------------------
# mode: missrate -- how often does the negative control actually fire?
# --------------------------------------------------------------------------

def mode_missrate(size, rung, nsteps, repeats):
    import cupy as cp

    print(f"== MISS RATE of the negative control, {repeats} independent runs "
          f"rung={rung} d01 {size['parent_nx']}^2 {nsteps} steps")
    bnd = tn.build_boundaries(size, rung)
    ref = tn.leg(size, rung, stream_d01=False, feedback=0, nsteps=nsteps,
                 boundaries=bnd)
    cp.get_default_memory_pool().free_all_blocks()
    misses = 0
    d02n, d01n = [], []
    # also: does the ARMED leg reproduce bit-exactness every time?
    armed_fail = 0
    for i in range(int(repeats)):
        bad = tn.leg(size, rung, stream_d01=True, feedback=0, nsteps=nsteps,
                     store_aware=False, boundaries=bnd)
        r1 = tn.compare(ref["d01"], bad["d01"])
        r2 = tn.compare(ref["d02"], bad["d02"])
        fired = (not r1["bitexact"]) or (not r2["bitexact"])
        misses += (not fired)
        d01n.append(r1["ndiff"])
        d02n.append(r2["ndiff"])
        del bad
        cp.get_default_memory_pool().free_all_blocks()
        good = tn.leg(size, rung, stream_d01=True, feedback=0, nsteps=nsteps,
                      boundaries=bnd)
        g1 = tn.compare(ref["d01"], good["d01"])
        g2 = tn.compare(ref["d02"], good["d02"])
        armed_fail += int(not (g1["bitexact"] and g2["bitexact"]))
        del good
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  run {i:3d}: control d01={r1['ndiff']} d02={r2['ndiff']} "
              f"fired={fired} | armed d01={g1['ndiff']} d02={g2['ndiff']} "
              f"bitexact={g1['bitexact'] and g2['bitexact']}", flush=True)
    print(f"\n  negative control MISS RATE: {misses}/{repeats} = "
          f"{100.0 * misses / max(1, repeats):.1f}%")
    print(f"  control d02 ndiff: min={min(d02n)} max={max(d02n)} "
          f"distinct={sorted(set(d02n))}")
    print(f"  control d01 ndiff: distinct={sorted(set(d01n))}")
    print(f"  ARMED leg failed bit-exactness {armed_fail}/{repeats} times")
    return 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="counts",
                    choices=["counts", "mutants", "missrate"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--medium", action="store_true")
    ap.add_argument("--rung", default="full(real74)+KF")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=40)
    args = ap.parse_args(argv)

    import cupy as cp

    size = (tn.QUICK_SIZE if args.quick else
            (tn.MEDIUM_SIZE if args.medium else tn.FULL_SIZE))
    nsteps = args.steps if args.steps else (6 if args.quick else tn.PARENT_STEPS)
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__} "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    try:
        if args.mode == "counts":
            return mode_counts(size, args.rung, nsteps)
        if args.mode == "mutants":
            return mode_mutants(size, args.rung, nsteps)
        return mode_missrate(size, args.rung, nsteps, args.repeats)
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
