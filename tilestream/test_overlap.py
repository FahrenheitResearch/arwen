"""The overlap event chain, proven and BROKEN without a card.

WHAT THIS GATES, AND WHY A GPU GATE CANNOT.  The overlap build moves the
ring sweep's copies onto dedicated streams; what keeps it a forecast is the
event chain :mod:`tilestream.overlap` derives from the ring plan.  The
consolidated GPU gate (``tilestream.test_gate``) proves the DEFAULT chain
bit-exact against a monolithic run -- but it cannot prove the chain is
LOAD-BEARING, because whether an unordered pair of stream operations
actually races is a property of the card and of what else it is running.
The lane already owns the measurement that makes this concrete:
``ring_ordering="submission"`` was bit-exact on an idle card and wrong by
3.7e+02 under contention, hours apart, same code.  A revert-check that
depends on an idle card's scheduler is not a revert-check.

So the adversary here is not a card, it is a SCHEDULER THIS TEST CONTROLS.
The sweep is simulated: every gather / save / patch / step / scatter is a
task on its buffer's copy-in, compute or copy-out stream, ordered by
exactly the dependencies the driver would issue -- same-stream program
order, the structural per-tile edges, and the four plan-shaped wait lists
of :class:`tilestream.overlap.OverlapSchedule`, with CUDA's
wait-takes-the-last-recorded-snapshot semantics reproduced.  The store, the
windows and the ring arena are numpy arrays; the step is a radius-1 stencil
whose halo requirement the plans here satisfy.  An adversarial executor
then runs the tasks in every order the dependencies admit that its
strategies can reach -- issue order, reversed preference, gathers-first,
scatters-first, patches-first, saves-first-patches-last, and seeded random
priorities -- and the result must equal the monolithic reference EXACTLY,
for every strategy, at every plan and buffer count tried, including a
ragged one.

THE REVERT-CHECK, which is the point.  Each wait class is then removed, one
at a time, and the test demands "the digest moves or a control fires" --
resolved per class, because building this gate MEASURED which classes are
which and the answer was not the naive one:

* ``patch_waits`` cut: the CONTROL fires
  (:func:`tilestream.overlap.assert_schedule_covers_hazards` refuses, which
  is what the driver runs at construction under ``overlap="on"``) AND the
  DIGEST MOVES under an adversary.
* ``gather_seam_waits`` cut: control fires AND digest moves -- on a 2x2
  plan at two buffers, where a whole tile's scatter can be starved past
  the next sweep's gathers without blocking them transitively.
* ``scatter_waits`` and ``save_seam_waits`` cut: the control fires, and
  the digest CANNOT move on any plan whose read/write overlap relation is
  symmetric -- this gate PROVES the implication instead of hunting for a
  divergence it showed cannot exist.  A stencil plan reads a window and
  writes an interior, so "j reads what k writes" and "k reads what j
  writes" hold together; that makes ``war_deps`` a subset of
  ``patch_deps``, and the missing scatter-after-gather edge is then
  implied through save -> patch -> step -> scatter, and the missing
  save-after-patch seam edge through gather_seam -> stepped -> ready.
  The two lists are kept in the driver as belt and braces: the checker --
  which fires on their removal, and runs at every construction -- is what
  guards the asymmetric plan on which the implication would not hold.
* the structural cut -- every cross-stream event dropped, the driver's
  ``overlap="unchained"`` -- is invisible to the checker by design, so for
  it the digest alone is the verdict, and it must move.

A contract test then ties the simulation to the driver: the wait loops in
``tilestream/driver.py`` are read out with ``ast`` and each schedule list
must be paired with the event class this module simulated it against --
``patch_waits`` with the save events, ``scatter_waits`` with the gather
events, ``gather_seam_waits`` with the scatter events, ``save_seam_waits``
with the ready events.  The two sides are checked against each other, not
against a restatement of either -- the discipline
``tilestream/test_tile_hook_contract.py`` established, and for the same
reason.

Run with ``pytest tilestream/test_overlap.py`` or directly with
``python -m tilestream.test_overlap`` from the repository root.  numpy
only; no card, no cupy.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilestream import overlap as _overlap          # noqa: E402
from tilestream import rings as _rings              # noqa: E402
from tilestream import spec as _spec                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_PY = os.path.join(HERE, "driver.py")

HALO = 2          # the stencil below has radius 1; 2 leaves margin
SWEEPS = 3        # seam hazards need at least two


# --------------------------------------------------------------------------
# the model problem
# --------------------------------------------------------------------------

def _step(a: np.ndarray) -> np.ndarray:
    """A radius-1 stencil, exact in float64 over small integers.

    ``np.roll`` wraps at the edges of whatever array it is handed, which is
    what the tile kernels do at the edges of the WINDOW; interior cells sit
    at least ``HALO`` from the window edge, so one step per sweep is exact
    there -- the same argument the real halo carries.
    """
    return a + 0.25 * (np.roll(a, 1, 0) + np.roll(a, -1, 0)
                       + np.roll(a, 1, 1) + np.roll(a, -1, 1))


def _reference(store0: np.ndarray, sweeps: int) -> np.ndarray:
    full = store0.copy()
    for _ in range(sweeps):
        full = _step(full)
    return full


# --------------------------------------------------------------------------
# the simulated sweep
# --------------------------------------------------------------------------

class _Task:
    __slots__ = ("idx", "stream", "kind", "tile", "sweep", "deps", "action")

    def __init__(self, idx, stream, kind, tile, sweep, deps, action):
        self.idx = idx
        self.stream = stream
        self.kind = kind
        self.tile = tile
        self.sweep = sweep
        self.deps = deps
        self.action = action


class _Cut:
    """Which dependency class a broken run drops.  ``None`` drops nothing."""

    PLAN_SHAPED = ("patch_waits", "scatter_waits",
                   "gather_seam_waits", "save_seam_waits")

    def __init__(self, name: str | None):
        self.name = name

    def waits(self, sched, attr, itile):
        if self.name == "unchained" or self.name == attr:
            return ()
        return getattr(sched, attr)[itile]

    @property
    def structural(self) -> bool:
        return self.name == "unchained"


def _issue(specs, plan, sched, nbuffers: int, sweeps: int, *,
           defer_seam: bool, cut: _Cut):
    """Every task of ``sweeps`` sweeps, in the driver's own issue order.

    Streams are ``("cin", b)``, ``("comp", b)``, ``("cout", b)`` -- the
    driver's copy-in, compute and copy-out.  Explicit dependencies are
    SNAPSHOTS: the task an event's record most recently landed on at the
    moment the wait is issued, which is CUDA's own wait semantics and the
    reason a seam wait that lands on this sweep's re-recording is
    conservative rather than early.  An event not yet recorded is no
    dependency at all, exactly as an unrecorded CUDA event satisfies a wait
    immediately.
    """
    ntiles = len(specs)
    depth = max(0, nbuffers - 1)
    tasks: list[_Task] = []
    last_on_stream: dict[tuple, _Task] = {}
    ev: dict[tuple[str, int], _Task] = {}
    occupant: list[int | None] = [None] * nbuffers

    # Buffers start POISONED, as the driver's do on their first sweep: a
    # broken chain that lets a step run before any gather filled the buffer
    # must show up as a moved digest, not as a simulator crash.
    cny, cnx = specs[0].cny, specs[0].cnx
    state = {
        "store": None,                       # bound by _execute
        "windows": [np.full((cny, cnx), np.nan) for _ in range(nbuffers)],
        "arena": {b.index: np.zeros((b.height, b.width))
                  for b in plan.bands},
    }

    def emit(stream, kind, tile, sweep, explicit, action):
        deps = []
        pred = last_on_stream.get(stream)
        if pred is not None:
            deps.append(pred)
        deps.extend(t for t in explicit if t is not None)
        task = _Task(len(tasks), stream, kind, tile, sweep, tuple(deps),
                     action)
        tasks.append(task)
        last_on_stream[stream] = task
        return task

    def gather_action(b, tspec):
        def run():
            win = np.empty((tspec.cny, tspec.cnx))
            tspec.apply_gather(state["store"], win, "mass")
            state["windows"][b] = win
        return run

    def save_action(b, itile):
        def run():
            for sv in plan.saves[itile]:
                state["arena"][sv.band.index][...] = \
                    state["windows"][b][sv.ty0:sv.ty0 + sv.h,
                                        sv.tx0:sv.tx0 + sv.w]
        return run

    def patch_action(b, itile):
        def run():
            for p in plan.patches[itile]:
                state["windows"][b][p.ty0:p.ty0 + p.h,
                                    p.tx0:p.tx0 + p.w] = \
                    state["arena"][p.band.index][p.by0:p.by0 + p.h,
                                                 p.bx0:p.bx0 + p.w]
        return run

    def compute_action(b):
        def run():
            state["windows"][b] = _step(state["windows"][b])
        return run

    def scatter_action(b, tspec):
        def run():
            tspec.apply_scatter(state["windows"][b], state["store"], "mass")
        return run

    def issue_gather(itile, tspec, sweep):
        b = itile % nbuffers
        explicit = []
        prev = occupant[b]
        if prev is not None and prev != itile:
            explicit.append(ev.get(("scatter", prev)))
        if defer_seam:
            for k in cut.waits(sched, "gather_seam_waits", itile):
                explicit.append(ev.get(("scatter", k)))
        if cut.structural:
            explicit = []
        g = emit(("cin", b), "G", itile, sweep, explicit,
                 gather_action(b, tspec))
        ev[("gather", itile)] = g

        explicit = [] if cut.structural else [ev.get(("gather", itile))]
        if defer_seam and not cut.structural:
            for m in cut.waits(sched, "save_seam_waits", itile):
                explicit.append(ev.get(("ready", m)))
        s = emit(("cout", b), "S", itile, sweep, explicit,
                 save_action(b, itile))
        ev[("save", itile)] = s

        explicit = []
        if not cut.structural:
            for k in cut.waits(sched, "patch_waits", itile):
                explicit.append(ev.get(("save", k)))
        p = emit(("cin", b), "P", itile, sweep, explicit,
                 patch_action(b, itile))
        ev[("ready", itile)] = p
        occupant[b] = itile

    def issue_compute_scatter(itile, tspec, sweep):
        b = itile % nbuffers
        explicit = ([] if cut.structural
                    else [ev.get(("ready", itile)), ev.get(("save", itile))])
        c = emit(("comp", b), "C", itile, sweep, explicit, compute_action(b))
        ev[("stepped", itile)] = c
        explicit = []
        if not cut.structural:
            explicit.append(ev.get(("stepped", itile)))
            for j in cut.waits(sched, "scatter_waits", itile):
                explicit.append(ev.get(("gather", j)))
        x = emit(("cout", b), "X", itile, sweep, explicit,
                 scatter_action(b, tspec))
        ev[("scatter", itile)] = x

    for sweep in range(sweeps):
        for i in range(min(depth, ntiles)):
            issue_gather(i, specs[i], sweep)
        for itile, tspec in enumerate(specs):
            if depth:
                nxt = itile + depth
                if nxt < ntiles:
                    issue_gather(nxt, specs[nxt], sweep)
            else:
                issue_gather(itile, tspec, sweep)
            issue_compute_scatter(itile, tspec, sweep)
        if not defer_seam:
            # The per-sweep barrier: everything after depends on everything
            # before, which is what the driver's stream syncs +
            # deviceSynchronize mean to the schedule.
            barrier = emit(("barrier",), "B", -1, sweep, list(tasks),
                           lambda: None)
            for stream in list(last_on_stream):
                last_on_stream[stream] = barrier
    return tasks, state


#: ``(name, priority key)``; lower sorts first among READY tasks.
_STRATEGIES = [
    ("issue", lambda t: (t.idx,)),
    ("reversed", lambda t: (-t.idx,)),
    ("gathers-first", lambda t: (0 if t.kind == "G" else 1, t.idx)),
    ("scatters-first", lambda t: (0 if t.kind == "X" else 1, t.idx)),
    ("patches-first", lambda t: (0 if t.kind == "P" else 1, t.idx)),
    ("saves-first-patches-last",
     lambda t: ({"S": 0, "P": 2}.get(t.kind, 1), t.idx)),
]


def _starve_strategies(max_buffers: int = 3):
    """One strategy per stream: run ANYTHING ready before that stream.

    The principled adversary for a missing cross-stream wait: a hazard pair
    with no edge between its streams is exposed by starving the stream the
    missing edge should have gated -- starving a copy-in stream delays its
    gathers past every other tile's scatter (the scatter_waits hazard),
    starving a copy-out stream delays its scatters past the next sweep's
    gathers (the gather_seam hazard).  A strategy naming a stream a small
    plan does not have is inert, not wrong.
    """
    out = []
    for kind in ("cin", "comp", "cout"):
        for b in range(max_buffers):
            stream = (kind, b)

            def key(t, stream=stream):
                return (1 if t.stream == stream else 0, t.idx)

            out.append((f"starve-{kind}{b}", key))
    return out


def _random_strategies(n: int = 4):
    out = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        salt = {"": None}

        def key(t, rng=rng, salt=salt):
            # One stable random priority per task index, per strategy.
            cache = salt.setdefault("keys", {})
            if t.idx not in cache:
                cache[t.idx] = float(rng.random())
            return (cache[t.idx],)

        out.append((f"random-{seed}", key))
    return out


def _execute(tasks, state, store0: np.ndarray, key) -> np.ndarray:
    state["store"] = store0.copy()
    for band in state["arena"].values():
        band[...] = 0.0
    done = [False] * len(tasks)
    remaining = set(range(len(tasks)))
    while remaining:
        ready = [tasks[i] for i in remaining
                 if all(done[d.idx] for d in tasks[i].deps)]
        if not ready:
            raise AssertionError("dependency cycle in the simulated sweep")
        best = min(ready, key=key)
        best.action()
        done[best.idx] = True
        remaining.remove(best.idx)
    return state["store"]


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

def _plans():
    """(label, specs) -- square/exact, ragged, a 2x2 and a two-tile strip."""
    cases = [
        ("12x12 tile 4 (3x3 exact)", 12, 12, 4, 4),
        ("14x12 tile 4 (ragged x)", 14, 12, 4, 4),
        ("8x8 tile 4 (2x2 exact)", 8, 8, 4, 4),
        ("8x4 tile 4 (1x2 strip)", 8, 4, 4, 4),
    ]
    for label, nx, ny, tnx, tny in cases:
        specs = _spec.plan_tiles(nx, ny, tnx, tny, HALO, True)
        yield label, specs


def _store0(ny: int, nx: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(0, 64, size=(ny, nx)).astype(np.float64)


def _run_case(specs, nbuffers: int, *, defer_seam: bool, cut: _Cut,
              strategies) -> list[str]:
    """Strategy names whose result DIFFERS from the monolithic reference."""
    plan = _rings.build_ring_plan(specs, ("mass",))
    sched = _overlap.OverlapSchedule.from_plan(plan)
    ny, nx = specs[0].ny, specs[0].nx
    store0 = _store0(ny, nx)
    want = _reference(store0, SWEEPS)
    moved = []
    for name, key in strategies:
        tasks, state = _issue(specs, plan, sched, nbuffers, SWEEPS,
                              defer_seam=defer_seam, cut=cut)
        got = _execute(tasks, state, store0, key)
        if not np.array_equal(got, want):
            moved.append(name)
    return moved


def _all_strategies():
    return _STRATEGIES + _starve_strategies() + _random_strategies()


# --------------------------------------------------------------------------
# tests: the chain is sufficient
# --------------------------------------------------------------------------

def test_full_chain_is_exact_under_every_adversary():
    """Every plan, buffer count and strategy: digest equals monolithic."""
    for label, specs in _plans():
        for nbuffers in (1, 2, 3):
            for defer in (False, True):
                moved = _run_case(specs, nbuffers, defer_seam=defer,
                                  cut=_Cut(None),
                                  strategies=_all_strategies())
                assert not moved, (
                    f"{label} nbuffers={nbuffers} defer={defer}: the FULL "
                    f"chain diverged under {moved}; the schedule is missing "
                    "an ordering")


def test_schedule_passes_its_own_checker():
    for _label, specs in _plans():
        plan = _rings.build_ring_plan(specs, ("mass",))
        sched = _overlap.OverlapSchedule.from_plan(plan)
        _overlap.assert_schedule_covers_hazards(sched, plan)


# --------------------------------------------------------------------------
# tests: the revert-check -- break the chain, the digest moves AND the
# control fires
# --------------------------------------------------------------------------

def _cut_schedule(sched, attr):
    empty = tuple(() for _ in range(sched.ntiles))
    return _overlap.OverlapSchedule(
        ntiles=sched.ntiles,
        patch_waits=empty if attr == "patch_waits" else sched.patch_waits,
        scatter_waits=(empty if attr == "scatter_waits"
                       else sched.scatter_waits),
        gather_seam_waits=(empty if attr == "gather_seam_waits"
                           else sched.gather_seam_waits),
        save_seam_waits=(empty if attr == "save_seam_waits"
                         else sched.save_seam_waits),
    )


def test_each_cut_wait_class_fires_the_checker():
    """The driver-side control: a schedule missing one class is refused."""
    for label, specs in _plans():
        plan = _rings.build_ring_plan(specs, ("mass",))
        sched = _overlap.OverlapSchedule.from_plan(plan)
        for attr in _Cut.PLAN_SHAPED:
            if not any(getattr(sched, attr)):
                continue          # nothing to cut on this plan
            cut = _cut_schedule(sched, attr)
            try:
                _overlap.assert_schedule_covers_hazards(cut, plan)
            except _overlap.OverlapError:
                continue
            raise AssertionError(
                f"{label}: cutting {attr} was not refused by the checker")


def test_cut_patch_waits_moves_the_digest():
    """RAW on the arena: patch before its producer's save, digest moves."""
    specs = next(s for label, s in _plans() if "3x3" in label)
    moved = _run_case(specs, 2, defer_seam=True, cut=_Cut("patch_waits"),
                      strategies=_all_strategies())
    assert moved, ("cutting patch_waits moved nothing under any adversary; "
                   "the revert-check has lost its teeth")


def test_cut_gather_seam_waits_moves_the_digest():
    """RAW on the store across the seam: the deferred barrier's replacement.

    On the 2x2 plan at two buffers a whole tile's scatter can be delayed
    past the next sweep's first gather without transitively blocking it
    (the occupant wait only reaches the SAME buffer's copy-out stream), so
    cutting the seam list must surface as a moved digest -- and does, under
    the stream-starvation adversaries.  This is also the direct proof that
    the per-sweep barrier the deferred seam removed was load-bearing and
    its event replacement is not decorative.
    """
    specs = next(s for label, s in _plans() if "2x2" in label)
    moved = _run_case(specs, 2, defer_seam=True,
                      cut=_Cut("gather_seam_waits"),
                      strategies=_all_strategies())
    assert moved, ("cutting gather_seam_waits moved nothing under any "
                   "adversary; the seam revert-check has lost its teeth")


def test_belt_and_braces_classes_are_transitively_implied():
    """scatter_waits and save_seam_waits cannot move a symmetric plan.

    Building this gate MEASURED that no adversary moves the digest when
    either list is cut, and the reason is provable rather than a hole in
    the adversary: a stencil plan's read/write overlap relation is
    symmetric, so ``war_deps[k]`` is a subset of ``patch_deps[k]`` and the
    scatter-after-gather edge is implied through
    save -> patch -> step -> scatter; likewise ``save_seam_waits[j]`` is a
    subset of the readers named in ``gather_seam_waits`` and the
    save-after-patch seam edge is implied through
    gather_seam -> stepped -> ready.  This test asserts BOTH inclusions on
    every plan tried, so the day a plan breaks the symmetry the implication
    visibly fails here -- and the driver still carries the explicit waits,
    with :func:`tilestream.overlap.assert_schedule_covers_hazards` (which
    fires on their removal, at every construction) as their revert-guard.
    """
    for label, specs in _plans():
        plan = _rings.build_ring_plan(specs, ("mass",))
        sched = _overlap.OverlapSchedule.from_plan(plan)
        for k in range(plan.ntiles):
            assert set(plan.war_deps[k]) <= set(plan.patch_deps[k]), (
                f"{label}: war_deps[{k}] not implied by patch chain; "
                "scatter_waits is load-bearing on this plan and needs an "
                "adversary case")
        for j in range(plan.ntiles):
            # save(j)@s+1 <- gather(j)@s+1 <- scatter(i)@s <- stepped(i)
            # <- ready(i) = patch(i)@s, valid whenever i is in j's seam
            # list.
            assert (set(sched.save_seam_waits[j])
                    <= set(sched.gather_seam_waits[j])), (
                f"{label}: save_seam_waits[{j}] not implied by the seam "
                "chain; it is load-bearing on this plan and needs an "
                "adversary case")


def test_unchained_moves_the_digest():
    """The structural cut the checker cannot see: digest is the verdict."""
    specs = next(s for label, s in _plans() if "3x3" in label)
    for defer in (False, True):
        moved = _run_case(specs, 2, defer_seam=defer, cut=_Cut("unchained"),
                          strategies=_all_strategies())
        assert moved, (
            f"overlap='unchained' (defer={defer}) was bit-exact under every "
            "adversary; the negative control no longer controls")


# --------------------------------------------------------------------------
# the driver contract: the waits in driver.py pair each schedule list with
# the event class this module simulated it against
# --------------------------------------------------------------------------

#: schedule attribute -> event list its waits must target in driver.py
_PAIRING = {
    "patch_waits": "ev_save",
    "scatter_waits": "ev_gather",
    "gather_seam_waits": "ev_scatter",
    "save_seam_waits": "ev_ready",
}


def _driver_wait_pairing() -> dict[str, set[str]]:
    """``{schedule attr: {event names waited inside its loop}}`` from ast."""
    tree = ast.parse(open(DRIVER_PY, encoding="utf-8").read())
    found: dict[str, set[str]] = {}

    class Walker(ast.NodeVisitor):
        def visit_For(self, node: ast.For):
            it = node.iter
            if (isinstance(it, ast.Subscript)
                    and isinstance(it.value, ast.Attribute)
                    and isinstance(it.value.value, ast.Name)
                    and it.value.value.id == "sched"):
                attr = it.value.attr
                events: set[str] = set()
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "wait_event"
                            and sub.args
                            and isinstance(sub.args[0], ast.Subscript)
                            and isinstance(sub.args[0].value, ast.Name)):
                        events.add(sub.args[0].value.id)
                found.setdefault(attr, set()).update(events)
            self.generic_visit(node)

    Walker().visit(tree)
    return found


def test_driver_waits_each_schedule_list_on_the_simulated_event():
    found = _driver_wait_pairing()
    for attr, event in _PAIRING.items():
        assert attr in found, (
            f"driver.py has no wait loop over sched.{attr}; the schedule "
            "list this module proves necessary is not being waited on")
        assert found[attr] == {event}, (
            f"driver.py waits sched.{attr} on {sorted(found[attr])}, but "
            f"the hazard it orders lives on {event}; the driver and the "
            "simulation have diverged")


def test_driver_runs_the_checker_for_overlap_on():
    """The construction-time control is actually armed."""
    src = open(DRIVER_PY, encoding="utf-8").read()
    assert "assert_schedule_covers_hazards(sched, ring.plan)" in src


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def main() -> int:
    checks = failed = 0
    rows = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        checks += 1
        try:
            fn()
            rows.append(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            rows.append(f"  FAIL  {name}: {exc}")
    print("overlap ordering gate (hermetic, numpy only)")
    print(f"  plans: {', '.join(label for label, _ in _plans())}")
    print(f"  strategies per case: {len(_all_strategies())}, "
          f"sweeps: {SWEEPS}, halo: {HALO}")
    for row in rows:
        print(row)
    print(f"OVERLAP GATE {'FAILED' if failed else 'PASSED'}: "
          f"{checks - failed} of {checks} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
