"""The ``tile_hook`` contract, checked without a GPU and without a real case.

THE CONSOLIDATED GATE CANNOT COVER THIS SEAM.  ``tile_hook`` exists for LATERAL
BOUNDARIES -- the one per-tile input the array transport cannot express -- and
the gate lane is periodic by design, so no gate configuration binds a boundary
and nothing in the gate ever calls the hook.  ``tilestream.test_gate`` was
233 PASS / 0 FAIL for the whole period in which every real-case streamed run
died at its first tile change with

    TypeError: bind() takes 3 positional arguments but 4 were given

because the merged 8x4090 driver moved the contract from
``(tile_state, itile, spec)`` to ``(tile_state, tspec, itile, stream)`` -- arity
AND positional order, ``itile`` going from second to third -- and the only
implementation, :func:`tilestream.realcase.tile_boundary_binder`, was not moved
with it.  What let that through review is that ``driver.py`` stated the contract
TWICE and the two disagreed: the new form beside the call site, the old form in
the docstring that names the binder as its user.

So these tests check the two sides against each other rather than against a
restatement of either:

* :func:`test_driver_call_site_passes_the_documented_four` reads the actual call
  out of ``driver.py`` with :mod:`ast`.
* :func:`test_bind_accepts_the_driver_call_site` and
  :func:`test_bind_takes_the_tile_index_from_the_third_position` drive the REAL
  ``bind`` closure over stub boundaries.  Order is tested, not just arity: a
  callback that accepted four arguments and still read ``itile`` from position
  two would serve every tile its neighbour's boundary, silently.
* :func:`test_no_stale_statement_of_the_contract_survives` fails on the
  second, contradicting docstring that caused this.

numpy only, no GPU, no cupy: the stubbed ``gpuwm.ingest.lateral_bc`` keeps the
attach call out of the picture, which is the only thing in the binder that
needs a device.

Run with ``pytest tilestream/test_tile_hook_contract.py`` or directly with
``python tilestream/test_tile_hook_contract.py``.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilestream import realcase  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_PY = os.path.join(HERE, "driver.py")

# The contract as driver.py states it beside the call site (driver.py:1085).
CONTRACT = ("tile_state", "tspec", "itile", "stream")
# The form it was moved AWAY from.  A tree that still says this anywhere is a
# tree where the next reader can pick the wrong one, which is what happened.
SUPERSEDED = "tile_hook(tile_state, itile, spec)"


class _StubBoundary:
    """Enough of a windowed lateral boundary to build a binder over."""

    def __init__(self, tag):
        self.tag = tag
        self.intervals = ()          # the binder sums bytes over this


@contextlib.contextmanager
def _stub_lateral_bc(record):
    """Replace only ``gpuwm.ingest.lateral_bc``, and put back what was there.

    The binder imports it inside its own body, so the substitution has to be
    live at call time rather than at module import.
    """
    def attach_streaming_lateral_boundaries(tile_state, lb):
        record.append((tile_state, lb))

    names = ("gpuwm", "gpuwm.ingest", "gpuwm.ingest.lateral_bc")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            if sys.modules.get(n) is None:
                sys.modules[n] = types.ModuleType(n)
        sys.modules["gpuwm.ingest.lateral_bc"] = types.ModuleType(
            "gpuwm.ingest.lateral_bc")
        sys.modules["gpuwm.ingest.lateral_bc"].\
            attach_streaming_lateral_boundaries = (
                attach_streaming_lateral_boundaries)
        yield
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def _build_bind(ntiles, record, monkey):
    """The REAL ``bind`` closure, over ``ntiles`` distinguishable stubs."""
    specs = [f"spec{i}" for i in range(ntiles)]
    monkey(realcase, "tile_boundaries",
           lambda _boundaries, spec, _cfg: _StubBoundary(spec))
    with _stub_lateral_bc(record):
        return realcase.tile_boundary_binder(
            object(), specs, object(), verbose=None), specs


class _Monkey:
    """A two-line stand-in so this file runs under bare python as well."""

    def __init__(self):
        self._undo = []

    def __call__(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo.clear()


def _driver_tile_hook_call():
    """The ``tile_hook(...)`` call as it is actually written in driver.py."""
    with open(DRIVER_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=DRIVER_PY)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "tile_hook"]
    assert len(calls) == 1, (
        f"expected exactly one tile_hook call site in driver.py, found "
        f"{len(calls)}")
    return calls[0]


# --------------------------------------------------------------------------
# the driver side
# --------------------------------------------------------------------------

def test_driver_call_site_passes_the_documented_four():
    call = _driver_tile_hook_call()
    assert not call.keywords, "tile_hook is called positionally by contract"
    assert len(call.args) == 4, (
        f"tile_hook call site passes {len(call.args)} arguments, contract "
        f"says {len(CONTRACT)}")
    # arg 0 is the buffer (a subscript, tiles[b]); 1..3 are plain names and
    # are the ones the binder's parameter ORDER has to agree with.
    passed = tuple(a.id for a in call.args[1:] if isinstance(a, ast.Name))
    assert passed == CONTRACT[1:], (
        f"tile_hook call site passes {passed}, contract says {CONTRACT[1:]}")


def test_no_stale_statement_of_the_contract_survives():
    with open(DRIVER_PY, encoding="utf-8") as fh:
        source = fh.read()
    assert SUPERSEDED not in source, (
        f"driver.py still states the superseded contract {SUPERSEDED!r}. "
        "Two disagreeing statements of one contract is what let the binder "
        "fall behind it.")


# --------------------------------------------------------------------------
# the binder side
# --------------------------------------------------------------------------

def test_bind_accepts_the_driver_call_site():
    monkey = _Monkey()
    try:
        bind, _specs = _build_bind(3, [], monkey)
        sig = inspect.signature(bind)
        # Must accept exactly what driver.py:1712 passes.
        sig.bind(object(), "tspec", 0, "stream")
    finally:
        monkey.undo()


def test_bind_takes_the_tile_index_from_the_third_position():
    """Arity alone is not the contract: a four-argument callback that still
    read the index from position two would hand every tile its neighbour's
    boundary and raise nothing at all."""
    monkey = _Monkey()
    record = []
    try:
        bind, specs = _build_bind(3, record, monkey)
        with _stub_lateral_bc(record):
            for itile in (0, 2, 1):
                # a fresh state each time so every call is a first bind and
                # takes the attach path we record
                bind(object(), f"tspec-for-{itile}", itile, "stream")
    finally:
        monkey.undo()

    got = [lb.tag for _state, lb in record]
    assert got == [specs[0], specs[2], specs[1]], (
        f"bind resolved tiles {got}, expected "
        f"{[specs[0], specs[2], specs[1]]} -- the third positional argument "
        "is the tile index")


def test_bind_ignores_the_tspec_it_is_handed():
    """The second positional is the tile's spec and the binder has no use for
    it; passing a different one must not change which boundary is bound."""
    monkey = _Monkey()
    record = []
    try:
        bind, specs = _build_bind(2, record, monkey)
        with _stub_lateral_bc(record):
            bind(object(), "one tspec", 1, "stream")
            bind(object(), "an entirely different tspec", 1, "stream")
    finally:
        monkey.undo()

    tags = [lb.tag for _state, lb in record]
    assert tags == [specs[1], specs[1]], tags


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        else:
            print(f"  PASS  {fn.__name__}")
    print(f"\n{len(TESTS) - failures} PASS / {failures} FAIL")
    sys.exit(1 if failures else 0)
