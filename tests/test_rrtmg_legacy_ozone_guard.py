"""Every legacy-RRTMG construction site, and who fails closed on ozone.

A RESIDENT nest under `ra_rrtmg_variant='rrtmg_legacy'` with `o3input = 2`
takes its ozone INTERPOLATED FROM THE PARENT: WRF runs the climatology
chain on `id == 1` only and hands a nest the root's field through the
`rdf=(p2c)` forcing stream the Registry declares on `o3rad`
(Registry/Registry.EM_COMMON:1264).  A resident child built without a
parent to take it from does not refuse on its own; it silently evaluates
a fresh climatology on its OWN latitudes and reports
`"ozone_routing": "root-climatology"` for a nested domain.  That is a
fail-OPEN, and `o3input = 2` is the RunConfig default.

This file is about the RESIDENT routes.  The OFFLINE child route
(`gpuwm.offline_child_run`) is a different case and is pinned in
tests/test_offline_child_ozone.py: that domain is configured as a WRF root
and stamps `parent_id = 0`, so WRF's own answer for it is the climatology
on its own grid, which it evaluates and declares as
`"ozone_routing": "child-grid-climatology"`.

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


def test_a_non_rrtmg_child_uses_the_common_factory_without_parent_ozone(monkeypatch):
    from gpuwm.core import radiation_composition
    cfg = SimpleNamespace(ra_lw_physics=1, ra_sw_physics=1,
                          ra_rrtmg_variant="rte-rrtmgp", o3input=2)
    seen = {}
    sentinel = object()
    def build(*args, **kwargs):
        seen.update(kwargs)
        return sentinel
    monkeypatch.setattr(radiation_composition, "make_radiation", build)
    result = runtime._child_radiation_adapter(
        SimpleNamespace(start_time="declared", column_chunk=16),
        SimpleNamespace(co2_vmr=None), SimpleNamespace(run=cfg, grid_id=2),
        SimpleNamespace(p_top=5000.), None, None, radiation_parent=None)
    assert result is sentinel
    assert seen["ozone_parent"] is None


def test_the_guard_is_qualified_on_o3input_and_not_unconditional():
    """SECOND NEGATIVE CONTROL: o3input=0 must not raise THIS refusal.

    The `and cfg.o3input == 2` qualifier is verbatim from 29e9af3f5 and
    essential -- o3input=0 uses the legacy-RRTMG wrapper's own O3DATA
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
                "gpuwm/core/physics.py", "gpuwm/core/radiation_composition.py"):
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
    # Root, child, offline and direct initialization all enter one factory.
    assert _construction_sites() == [
        ("gpuwm/core/radiation_composition.py", "engine"),
    ]


def test_the_offline_child_route_is_not_governed_by_this_refusal():
    """The offline route reaches the constructor and names what it built.

    Retired with the defect it was installed for: that route's domain is a
    WRF root, not a nest, so it has a climatology answer of its own and no
    parent to be refused for lacking.  What replaces the refusal is a
    DECLARATION, pinned here so a future edit cannot quietly put the
    resident-nest name back on an offline child, and exercised end to end
    in tests/test_offline_child_ozone.py.
    """
    import inspect

    from gpuwm import offline_child_run
    from gpuwm.core.cam_ozone import ROUTING_CHILD_GRID_CLIMATOLOGY

    source = inspect.getsource(offline_child_run._initialize_child_physics)
    assert "requires radiation_parent" not in source
    assert "needs ozone interpolated from the parent domain" not in source
    assert "ROUTING_CHILD_GRID_CLIMATOLOGY" in source
    assert ROUTING_CHILD_GRID_CLIMATOLOGY == "child-grid-climatology"


def test_a_real_tree_dependency_uses_driver_carrier_without_parent_adapter(monkeypatch):
    from test_cam_ozone import _tree
    from gpuwm.core.cam_ozone import DriverOzoneProvider
    from gpuwm.core import radiation_composition
    exp = _tree()
    seen = {}
    monkeypatch.setattr(radiation_composition, "make_radiation",
                        lambda *args, **kwargs: seen.update(kwargs))
    runtime._child_radiation_adapter(exp, SimpleNamespace(co2_vmr=None),
        exp.domains[1], SimpleNamespace(p_top=5000.), None, None)
    assert isinstance(seen["ozone_parent"], DriverOzoneProvider)
