"""Seam B: a spawn declaration is honored or refused, never dropped.

``gpuwm.experiment.refuse_unrouted_spawn`` existed with zero production
callers -- the guard was written but never mounted, so every route that
neither RESERVES a dormant nest's VRAM nor WATCHES its trigger would
quietly integrate without the nest the user declared.  This pins the
mounting: one real refusal through a front door that is a pure
function, and a source audit over the rest, so a route added later
cannot skip the governance by accident.

The audit is deliberately a source scan rather than a call through each
front door: those routes need prepared caches, sealed manifests and
GPUs to reach their experiment, and the thing worth pinning is that the
guard is PRESENT at admission, which is exactly what the scan sees.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from gpuwm.core.nest_spawn import SpawnConfig

REPO = Path(__file__).resolve().parents[1]

#: Every front door that neither reserves nor watches, and the route
#: label it must name.  Adding a route here without mounting the guard
#: fails; mounting it without listing it here also fails.
GUARDED_FRONT_DOORS = {
    "gpuwm/era5_direct.py": "ERA5-direct prepared-cache",
    "gpuwm/gfs_direct.py": "GFS-direct prepared-cache",
    "gpuwm/mapped_direct.py": "mapped-adapter prepared-cache",
    "gpuwm/prepared_single_domain_forecast.py": "prepared single-domain",
    "gpuwm/prepared_domain_tree_forecast.py": "prepared domain-tree",
    "gpuwm/da/nested_forecast.py": "DA nested free-forecast",
    "gpuwm/native_hierarchy.py": "native hierarchy export",
}


def _spawn_call_route_labels(path: Path) -> list[str]:
    """Every string literal handed to refuse_unrouted_spawn in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    labels: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name != "refuse_unrouted_spawn":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                labels.append(arg.value)
    return labels


@pytest.mark.parametrize("relative,label", sorted(GUARDED_FRONT_DOORS.items()))
def test_every_unrouted_front_door_mounts_the_refusal(relative, label):
    labels = _spawn_call_route_labels(REPO / relative)
    assert labels, (
        f"{relative} is listed as an unrouted route but never calls "
        "refuse_unrouted_spawn; a spawn declaration would be dropped")
    assert any(label in seen for seen in labels), (
        f"{relative} calls refuse_unrouted_spawn but names "
        f"{labels} rather than the expected route {label!r}")


def test_the_audit_list_is_the_whole_set_of_callers():
    """No route may mount the guard without being listed here."""
    found = set()
    for path in sorted(REPO.joinpath("gpuwm").rglob("*.py")):
        if path.name == "experiment.py" and path.parent.name == "gpuwm":
            continue          # the definition itself
        if _spawn_call_route_labels(path):
            found.add(path.relative_to(REPO).as_posix())
    assert found == set(GUARDED_FRONT_DOORS), (
        "the mounted refusals drifted from the audited list; "
        f"unlisted={sorted(found - set(GUARDED_FRONT_DOORS))} "
        f"missing={sorted(set(GUARDED_FRONT_DOORS) - found)}")


# ---------------------------------------------------------------------------
# The one front door that is a pure function: prove the real refusal
# ---------------------------------------------------------------------------

def test_the_da_nested_leg_refuses_a_dormant_child():
    from gpuwm.da import nested_forecast as nf
    from test_da_nested_forecast import _nowcast_experiment

    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))

    # Control: an ordinary child is admitted, unchanged by the guard.
    assert nf.nested_experiment(exp, child).domains[1] is child

    dormant = replace(child, spawn=SpawnConfig(
        trigger="uh", threshold=60.0, earliest_s=0.0, latest_s=3600.0))
    with pytest.raises(ValueError, match="does not implement "
                                         "spawn-triggered"):
        nf.nested_experiment(exp, dormant)


def test_the_refusal_names_the_route_and_the_domain():
    from gpuwm.da import nested_forecast as nf
    from test_da_nested_forecast import _nowcast_experiment

    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    dormant = replace(child, spawn=SpawnConfig(
        trigger="time", at_s=600.0))
    with pytest.raises(ValueError) as err:
        nf.nested_experiment(exp, dormant)
    message = str(err.value)
    assert "DA nested free-forecast" in message
    assert f"d{int(child.grid_id):02d}" in message


# ---------------------------------------------------------------------------
# The run route watches AND lands the newborn: the lift, and its evidence
# ---------------------------------------------------------------------------

def test_the_run_route_no_longer_refuses_a_dormant_nest_for_land_state():
    """The refusal that used to stand here is LIFTED (2026-08-07).

    It named one gap -- a newborn real-data nest had no defined soil/land
    state -- and treated the fix as an open physics decision.  It is not
    open: a nest with no input file of its own is initialized by
    interpolating from the parent, and the surface/soil family goes
    through the Registry's landmask-aware interpolator.  Both now exist on
    this route, so a dormant nest must walk PAST this point.  The stub
    experiment then fails further in, on its own missing attributes, which
    is exactly the control the ordinary-nest test below relies on.
    """
    from types import SimpleNamespace

    from gpuwm import runtime

    from gpuwm.core.streaming import OFF as _STREAMING_OFF

    # ``tiles=OFF`` for the reason given in test_relocation_runner: the
    # front door refuses a [tiles] block at admission, so a double that
    # lacks the attribute a real Experiment always has raises AttributeError
    # before the guard under test is reached.
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, spawn=None),
                 SimpleNamespace(grid_id=2, spawn=object())),
        relocation=SimpleNamespace(enabled=False, follow=None, moves=()),
        tiles=_STREAMING_OFF)
    with pytest.raises(Exception) as err:
        runtime.run_experiment(exp, None, "out-tree-spawn")
    message = str(err.value)
    assert "no defined soil/land state" not in message
    assert "open physics decision" not in message
    # The dormant nest got no further than the same stub failure an
    # ordinary two-domain experiment reaches: the guard is inert now.
    ordinary = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, spawn=None),
                 SimpleNamespace(grid_id=2, spawn=None)),
        relocation=SimpleNamespace(enabled=False, follow=None, moves=()),
        tiles=_STREAMING_OFF)
    with pytest.raises(Exception) as control:
        runtime.run_experiment(ordinary, None, "out-tree-control")
    assert type(err.value) is type(control.value)
    assert message == str(control.value)


def test_the_land_state_machinery_the_lift_claims_really_exists():
    """A lift is only honest if the capability it cites is wired.

    Not a source scan for its own sake: this asserts the operator is
    importable, that the route's spawn preparer actually calls it, and
    that the accounting reaches the birth receipt instead of dying in the
    preparer.
    """
    from gpuwm import runtime
    from gpuwm.core.nest_interp import interp_mask_field
    from gpuwm.ingest.nest_spawn_init import spawn_land_state_from_parent

    assert callable(interp_mask_field)
    assert callable(spawn_land_state_from_parent)
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "spawn_land_state_from_parent(" in source
    assert "rebuild_child_driver_from_land_state(" in source
    preparer = runtime.RealSpawnChildPreparer.__call__
    assert "spawn_land_state_from_parent" in preparer.__code__.co_names
    # The duck-typed receipt seam the runner reads (relocation's idiom).
    assert "last_receipt" in runtime.RealSpawnChildPreparer.__init__.__code__ \
        .co_names
    spawn_init = (Path(runtime.__file__).parent / "ingest"
                  / "nest_spawn_init.py").read_text(encoding="utf-8")
    assert '"land_surface": land_receipt' in spawn_init


def test_the_run_route_still_refuses_a_dormant_nest_with_no_parent():
    """The other direction: the single-domain path walks no tree, so a
    dormant nest there has nothing to be born from and still refuses."""
    from types import SimpleNamespace

    from gpuwm import runtime

    from gpuwm.core.streaming import OFF as _STREAMING_OFF

    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, spawn=object()),),
        relocation=SimpleNamespace(enabled=False, follow=None, moves=()),
        tiles=_STREAMING_OFF)
    with pytest.raises(ValueError, match="single-domain path runs no tree"):
        runtime.run_experiment(exp, None, "out-single-spawn")


def test_the_driver_exists_where_the_lift_claims_it_does():
    """The lift is only honest if the route really wires the runner."""
    from gpuwm import runtime

    assert callable(runtime.walk_spawn_legs)
    assert callable(runtime.build_real_spawn_runner)
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "spawn_runner = build_real_spawn_runner(" in source
    assert "walk_spawn_legs(" in source


def test_the_run_route_admits_an_ordinary_nest():
    """Control: with no spawn table the guard is inert (the call proceeds
    past it and fails later, on the stub, not on this refusal)."""
    from types import SimpleNamespace

    from gpuwm import runtime

    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, spawn=None),
                 SimpleNamespace(grid_id=2, spawn=None)),
        relocation=SimpleNamespace(enabled=False, follow=None, moves=()))
    with pytest.raises(Exception) as err:
        runtime.run_experiment(exp, None, "out-control")
    assert "never born" not in str(err.value)
