"""Every legacy-RRTMG construction site, and who fails closed on ozone.

`ra_rrtmg_variant='rrtmg_legacy'` with `o3input = 2` means the child takes
its ozone INTERPOLATED FROM THE PARENT -- WRF evaluates the CAM
climatology on `id == 1` only and passes `o3rad` down.  A child built
without a parent to take it from does not refuse; it silently evaluates a
fresh climatology on its OWN latitudes and reports
`"ozone_routing": "root-climatology"` for a nested domain.  That is a
fail-OPEN, and `o3input = 2` is the RunConfig default.

WHY THIS FILE EXISTS AT ALL, rather than more assertions in
tests/test_rrtmg_legacy_wiring.py: that module imports cupy at module
scope, so conftest's AST auto-marker marks the WHOLE module `gpu` and
`GPUWM_NO_LOCAL_GPU` skips every test in it.  The guard it was pinning
therefore had no CPU-side coverage at all, which is why the 1.8.0 CPU leg
was green while the GPU leg was red.  This module imports no cupy in any
of the three triggering positions (module scope, a fixture, a non-`test_`
helper), so it stays CPU-side.  Importing `gpuwm.runtime` at module scope
is fine: the detector reads this file's own source, not `sys.modules`.

WHY BEHAVIOURAL, rather than `inspect.getsource` string matching: the
previous pin asserted `"radiation_parent is None" in
inspect.getsource(runtime.prepare_child_case)` and went red in 1.8.0 for
a refactor that MOVED the guard into `_child_radiation_adapter` while
preserving it exactly -- and which closed a hole, since the new
mid-run spawn/relocation construction site is guarded by the same code.
The pin could not tell "guard deleted" from "guard relocated".  These
call the guard and watch it fire.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import runtime

REPO = Path(__file__).resolve().parents[1]

#: The refusal both in-memory child routes raise.  Regex-escaped at the
#: use site: the real sentence carries '=' and parentheses.
_REFUSAL = r"requires radiation_parent"


def _legacy_child_cfg(o3input: int = 2):
    """A child RunConfig double resolving to legacy RRTMG on both streams.

    Only the four fields the guard reads before it raises: the scheme
    pair, the variant and the ozone input.  Nothing here touches a
    device, opens a file or allocates a grid.
    """
    return SimpleNamespace(ra_lw_physics=4, ra_sw_physics=4,
                           ra_rrtmg_variant="rrtmg_legacy", o3input=o3input)


def _child_domain(o3input: int = 2, grid_id: int = 2):
    return SimpleNamespace(run=_legacy_child_cfg(o3input), grid_id=grid_id)


# ---------------------------------------------------------------------------
# The guard, called
# ---------------------------------------------------------------------------

def test_a_legacy_child_without_a_radiation_parent_refuses():
    """The construction site fails closed, on CPU, before any asset load."""
    with pytest.raises(ValueError, match=_REFUSAL):
        runtime._child_radiation_adapter(
            None, None, _child_domain(), None, None, None,
            radiation_parent=None)


def test_the_refusal_names_the_domain_it_refused_for():
    """A refusal that does not say WHICH nest is a refusal you cannot act
    on; the tree can carry several children."""
    with pytest.raises(ValueError, match=r"grid_id=7"):
        runtime._child_radiation_adapter(
            None, None, _child_domain(grid_id=7), None, None, None,
            radiation_parent=None)


def test_a_non_rrtmg_child_is_inert_rather_than_refused():
    """NEGATIVE CONTROL, and the one that proves the guard is scoped.

    A child that never asked for RRTMG at all must not be dragged into
    this refusal.  `(1, 1)` resolves away from the legacy pair before the
    guard is consulted, and the adapter returns None -- no radiation
    adapter, no exception.
    """
    cfg = SimpleNamespace(ra_lw_physics=1, ra_sw_physics=1,
                          ra_rrtmg_variant="rrtmg_legacy", o3input=2)
    assert runtime._child_radiation_adapter(
        None, None, SimpleNamespace(run=cfg, grid_id=2), None, None, None,
        radiation_parent=None) is None


def test_the_guard_is_qualified_on_o3input_and_not_unconditional():
    """SECOND NEGATIVE CONTROL: o3input=0 must not raise THIS refusal.

    The `and cfg.o3input == 2` qualifier is verbatim from 29e9af3f5 and
    load-bearing -- o3input=0 uses the legacy-RRTMG wrapper's own O3DATA
    profile and needs no parent, so refusing it would over-specify.

    o3input=0 cannot be asserted to SUCCEED here, because past the guard
    the adapter proceeds to real asset loading with `None` operands.  So
    the assertion is the discriminating one: whatever goes wrong next, it
    must not be the parent-ozone refusal.
    """
    try:
        runtime._child_radiation_adapter(
            None, None, _child_domain(o3input=0), None, None, None,
            radiation_parent=None)
    except Exception as error:      # noqa: BLE001 - any failure but ours
        assert "requires radiation_parent" not in str(error), (
            "o3input=0 needs no parent ozone and must not hit the "
            f"parent-ozone refusal; got: {error}")


# ---------------------------------------------------------------------------
# The sweep: every construction site, classified
# ---------------------------------------------------------------------------

def _construction_sites():
    """(module, enclosing function) for every RRTMGLegacyRadiation(...)."""
    sites = []
    for rel in ("gpuwm/runtime.py", "gpuwm/offline_child_run.py",
                "gpuwm/core/physics.py"):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        owners = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "RRTMGLegacyRadiation"):
                inner = None
                for owner in owners:
                    if owner.lineno <= node.lineno <= owner.end_lineno:
                        if inner is None or owner.lineno > inner.lineno:
                            inner = owner
                sites.append((rel, inner.name if inner else "<module>"))
    return sorted(sites)


def test_every_legacy_construction_site_is_accounted_for():
    """Four sites, named.  A fifth appearing anywhere fails here.

    This is the roster the 1.8.0 audit produced.  It is deliberately a
    whole-set equality rather than a subset check: a new construction
    site is exactly the event that reopens the fail-open, and it must not
    be able to arrive unnoticed.
    """
    assert _construction_sites() == [
        # The fallback inside initialize_physics, reached only when a
        # caller passes no explicit radiation= adapter.  UNGUARDED, and
        # its own comment asserts (does not enforce) that a child must be
        # wired through runtime.prepare_child_case.  Unreachable for a
        # child on the main runtime route -- both child routes pass an
        # explicit radiation= -- but reachable from routes that do not.
        # Tracked; closing it is a cross-route behavioural change.
        ("gpuwm/core/physics.py", "initialize_physics"),
        # The ndown-equivalent offline child.  REFUSES o3input=2: this
        # route stamps parent_id=0 and has no resident parent to take
        # o3rad from, so it cannot be wired, only refused.
        ("gpuwm/offline_child_run.py", "_initialize_child_physics"),
        # The child adapter shared by prepare_child_case and
        # rebuild_child_driver_from_land_state.  GUARDED.
        ("gpuwm/runtime.py", "_child_radiation_adapter"),
        # The ROOT domain.  No ozone parent by design: WRF evaluates the
        # climatology on id == 1, which is this one.
        ("gpuwm/runtime.py", "prepare_real_case"),
    ]


def test_the_offline_child_route_refuses_parent_ozone_it_cannot_supply():
    """The offline route's refusal, pinned as a refusal.

    Source-level because the call needs staged wrfinput/history assets to
    reach; what is pinned is that the o3input==2 branch RAISES rather
    than falling through to the constructor, which is the fail-open this
    closed.
    """
    import inspect

    from gpuwm import offline_child_run

    source = inspect.getsource(offline_child_run._initialize_child_physics)
    guard = source.index("cfg.o3input == 2")
    construct = source.index("RRTMGLegacyRadiation(")
    assert guard < construct, "the ozone refusal must precede construction"
    assert "raise ValueError" in source[guard:construct]
