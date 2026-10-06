"""Source and memory contracts of the real member coupling component."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_interp import register_nest
from gpuwm.ensemble.batch_kernel import KernelSpec, PointerSpec, generate_batch_source
from gpuwm.ensemble.batch_nesting import nesting_source, member_nest_memory_plan, PreparedMemberNestEdge
from gpuwm.ensemble.batch_state import BatchStateUnsupported
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchStorage


def _source_spec(entry):
    if entry == "nest_bdy_interp1_all_sides":
        member = "cfld nfld west_val west_tend east_val east_tend south_val south_tend north_val north_tend".split()
        shared = [(name, "int32") for name in ("ci_map", "ip_map", "cj_map", "jp_map")]
        shared += [(name, "float32") for name in ("xig", "xjg")]
    else:
        member, shared = ("cfld", "nfld" if entry == "nest_copy_fcn" else "scr"), []
    pointers = tuple(PointerSpec(name, "member") for name in member)
    pointers += tuple(PointerSpec(name, "shared", dtype) for name, dtype in shared)
    return KernelSpec("nest", entry, pointers, ("-std=c++17", "-fmad=false"))


def test_member_sources_retain_the_actual_stock_arithmetic_and_fmad_policy():
    from gpuwm.core import nest_interp
    from gpuwm.core.kernels import _preamble
    original = _preamble() + (Path(nest_interp.__file__).parent / "kernels/nest.cu").read_text()
    assert nesting_source() == original
    for entry in ("nest_bdy_interp1_all_sides", "nest_copy_fcn", "nest_smooth_j", "nest_smooth_i"):
        spec = _source_spec(entry)
        assert generate_batch_source(original, spec, 1) == original
        adapted = generate_batch_source(original, spec, 4)
        assert "__ensemble_member" in adapted
        # The runtime's existing source audit checks the entire untouched
        # arithmetic body; the member prologue only adjusts pointer owners.
        assert spec.options == ("-std=c++17", "-fmad=false")


def _inventory(nx, ny, nz):
    shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx),
              "thp": (nz, ny, nx), "php": (nz + 1, ny, nx), "mup": (ny, nx), "qv": (nz, ny, nx)}
    return SimpleNamespace(cfg=SimpleNamespace(nx=nx, ny=ny, nz=nz, spec_zone=1,
        relax_zone=4, spec_bdy_width=5), storage=SimpleNamespace(specs={
            name: BatchArraySpec(name, shape, "member") for name, shape in shapes.items()}))


def test_edge_memory_counts_private_rolling_tables_and_scratch_shared_geometry_once():
    parent, child = _inventory(39, 37, 6), _inventory(30, 27, 6)
    regs = {stagger: register_nest(nri=3, nrj=3, i_parent_start=5, j_parent_start=5,
        child_nx=30, child_ny=27, parent_nx=39, parent_ny=37,
        stagger="" if stagger == "m" else stagger, wrapper="bdy") for stagger in ("m", "x", "y")}
    fields = ("mu", "u", "v", "w", "t", "ph", "qv")
    plan = member_nest_memory_plan(parent, child, regs, fields)
    assert len([spec for spec in plan.arrays if spec.name.startswith("table:")]) == 8 * len(fields)
    assert all(spec.ownership == "member" for spec in plan.arrays if not spec.name.startswith("geometry:"))
    geometry = [spec for spec in plan.arrays if spec.name.startswith("geometry:")]
    assert len(geometry) == 18 and all(spec.ownership == "shared" for spec in geometry)
    assert {np.dtype(spec.dtype).name for spec in geometry} == {"int32", "float32"}
    # Allocations round complete banks, rather than rounding every slot.
    exact4 = sum((((4 if spec.ownership == "member" else 1) * spec.slab_bytes + 511) // 512) * 512
                 for spec in plan.arrays)
    assert plan.required_bytes(4) == exact4
    assert plan.required_bytes(8) > plan.required_bytes(4)


def test_edge_state_dependency_union_covers_actual_coupled_and_parent_diagnostic_pointer_rows():
    from gpuwm.config import RunConfig
    from gpuwm.experiment import DomainConfig
    from gpuwm.core.preflight import nest_field_kinds
    from gpuwm.ensemble.batch_diagnostics import diagnostic_pointer_fields
    from gpuwm.ensemble.batch_nesting import (prepared_tree_edge_state_fields,
        coupled_field_pointer_names)
    cfg = RunConfig(nx=12, ny=12, nz=8, dx=2250.0, dy=2250.0, ztop=10000.0,
                    dt=9.0, run_seconds=60.0, moist=True, mp_physics=8)
    parent = DomainConfig(1, 0, 1, 1, 1, 1, 60.0, cfg)
    child = DomainConfig(2, 1, 2, 2, 3, 3, 60.0, cfg)
    rows = prepared_tree_edge_state_fields((parent, child))
    reads = {name for kind in nest_field_kinds(cfg) for name in coupled_field_pointer_names(kind)}
    diagnostic = {name for _, name in diagnostic_pointer_fields(moist=True)}
    assert rows[2] == reads
    assert rows[1] == reads | diagnostic
    assert {"u", "v", "w", "thp", "php", "mup", "qv"} <= rows[2]
    assert {"p", "al", "alt", "dphb_resid"} <= rows[1]
    assert not {"u0", "v0", "rw_t", "theta", "rho"} & (rows[1] | rows[2])


def test_scratch_layouts_are_dense_for_mass_scalar_and_each_stagger():
    parent, child = _inventory(39, 37, 6), _inventory(30, 27, 6)
    fields = ("mu", "u", "v", "w", "t", "ph", "qv")
    plan = member_nest_memory_plan(parent, child, {}, fields)
    expected = {
        "parent": {(1, 37, 39), (6, 37, 40), (6, 38, 39), (7, 37, 39), (6, 37, 39)},
        "child": {(1, 27, 30), (6, 27, 31), (6, 28, 30), (7, 27, 30), (6, 27, 30)},
    }
    for endpoint, shapes in expected.items():
        specs = [spec for spec in plan.arrays if spec.name.startswith(f"nest:{endpoint}_scratch:")]
        assert {spec.shape for spec in specs} == shapes
        # W/ph and t/qv reuse only their identical layouts. Equal element
        # counts alone are insufficient to share a dense kernel operand.
        assert len(specs) == len(shapes)
    for members in (1, 2, 4, 8):
        storage = BatchStorage(plan, members, array_module=np,
                               available_bytes=plan.required_bytes(members))
        for name, value in storage.arrays.items():
            if name.startswith("nest:"):
                assert value.flags.c_contiguous
                assert value.shape == (members, *storage.specs[name].shape)
                assert value.strides[0] == storage.specs[name].slab_bytes


def _rebind_fixture():
    from test_nest_coupler import _clock
    parents = [_clock(1, 0, step_ticks=9, dt=9, advanced=True) for _ in range(2)]
    children = [_clock(2, 1, step_ticks=3, dt=3) for _ in range(2)]
    for parent, child in zip(parents, children, strict=True):
        parent.step_ticks, parent.dt_fp32, parent.ticks = 12, np.float32(12), 12
        child.step_ticks, child.dt_fp32 = 4, np.float32(4)
        parent.adaptive_state = child.adaptive_state = {"started": True}
    edge = object.__new__(PreparedMemberNestEdge)
    edge.members, edge.fields, edge._tick_den = 2, ("mu", "u"), 1
    edge.parent = SimpleNamespace(cfg=SimpleNamespace(grid_id=1),
        nodes=tuple(SimpleNamespace(clock=clock) for clock in parents))
    edge.child = SimpleNamespace(cfg=SimpleNamespace(grid_id=2, spec_zone=1, relax_zone=4, spec_bdy_width=5),
        nodes=tuple(SimpleNamespace(clock=clock) for clock in children))
    edge.parent_dt_fp32, edge.parent_interval_ticks = np.float32(9), 9
    edge._clock_contract = (edge.parent_dt_fp32.tobytes(), 9)
    edge._clock_roster = tuple((id(parent), id(child)) for parent, child in zip(parents, children, strict=True))
    edge._prepared_feedback, edge.interval_rebind_count = None, 0
    edge.force_count, edge.feedback_count, edge.generation, edge.valid = 7, 5, 11, True
    token = {"generation": 0}
    def bindings():
        if token["generation"]:
            raise BatchStateUnsupported("original backing changed during compile")
    edge._require_bindings = bindings
    edge._parent_views = {name: object() for name in edge.fields}
    edge._child_views = {name: object() for name in edge.fields}
    edge.tables = {name: {"untouched": np.arange(7, dtype=np.float32)} for name in edge.fields}
    edge.registrations, edge.geometry = {"m": object(), "x": object()}, {"m": {}, "x": {}}
    edge._forces = [(object(), object(), object()) for _ in edge.fields]
    return edge, parents, children, token


def _rebind_words(edge, children):
    return (edge._clock_contract, id(edge._forces), tuple(tuple(id(item) for item in row) for row in edge._forces),
            edge.interval_rebind_count, edge.force_count, edge.feedback_count, edge.generation, edge.valid,
            tuple(clock.dtbc_fp32.tobytes() for clock in children),
            tuple(value.tobytes() for sides in edge.tables.values() for value in sides.values()))


@pytest.mark.parametrize("failure", ["compile", "backing", "clock"])
def test_rebind_preparation_failure_keeps_every_prior_launch_and_table_atomic(monkeypatch, failure):
    import gpuwm.ensemble.batch_nesting as module
    edge, parents, children, token = _rebind_fixture()
    before, calls = _rebind_words(edge, children), []
    error = RuntimeError("interpolation launch preparation failed")
    def prepare(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["parent_dt_fp32"].tobytes() == np.float32(12).tobytes()
        if len(calls) == 2:
            if failure == "compile":
                raise error
            if failure == "backing":
                token["generation"] = 1
            else:
                parents[0].ticks += 1
        return lambda: pytest.fail("preparation launched numerical FORCE")
    monkeypatch.setattr(module, "prepare_member_bdy_interp1", prepare)
    with pytest.raises(RuntimeError if failure == "compile" else BatchStateUnsupported) as caught:
        edge.rebind_parent_interval(parent_clocks=parents, child_clocks=children)
    if failure == "compile":
        assert caught.value is error
    assert len(calls) == 2 and _rebind_words(edge, children) == before


@pytest.mark.parametrize("change", ["lattice", "lead", "parent_fp32", "child_fp32", "alias", "owner",
                                    "divergent", "feedback", "integer", "calendar"])
def test_rebind_refuses_unproved_clock_authority_before_any_launch_preparation(monkeypatch, change):
    import gpuwm.ensemble.batch_nesting as module
    edge, parents, children, _ = _rebind_fixture()
    before = _rebind_words(edge, children)
    monkeypatch.setattr(module, "prepare_member_bdy_interp1", lambda *args, **kwargs: pytest.fail("invalid clock compiled a launch"))
    if change == "lattice":
        parents[1].tick_den = children[1].tick_den = 2
    elif change == "lead":
        parents[1].ticks += 1
    elif change == "parent_fp32":
        parents[1].dt_fp32 = np.nextafter(np.float32(12), np.float32(13))
    elif change == "child_fp32":
        children[1].dt_fp32 = np.nextafter(np.float32(4), np.float32(5))
    elif change == "alias":
        parents[1] = parents[0]
    elif change == "owner":
        edge.parent.nodes[1].clock = _rebind_fixture()[1][0]
    elif change == "divergent":
        parents[1].step_ticks, parents[1].ticks, parents[1].dt_fp32 = 15, 15, np.float32(15)
    elif change == "feedback":
        edge._prepared_feedback = object()
    elif change == "integer":
        parents[1].step_ticks = 12.0
    else:
        parents[1].run_ticks = 11
    with pytest.raises((BatchStateUnsupported, ValueError)):
        edge.rebind_parent_interval(parent_clocks=parents, child_clocks=children)
    assert _rebind_words(edge, children) == before


def test_rebind_only_replaces_interpolation_args_and_repeated_interval_is_noop(monkeypatch):
    import gpuwm.ensemble.batch_nesting as module
    edge, parents, children, _ = _rebind_fixture()
    old = tuple(edge._forces)
    tables, geometry, views = edge.tables, edge.geometry, (edge._parent_views, edge._child_views)
    prepared = []
    def prepare(*args, **kwargs):
        result = lambda: None
        prepared.append(result)
        return result
    monkeypatch.setattr(module, "prepare_member_bdy_interp1", prepare)
    assert edge.rebind_parent_interval(parent_clocks=parents, child_clocks=children)
    assert edge._clock_contract == (np.float32(12).tobytes(), 12)
    assert edge.interval_rebind_count == 1 and not edge.valid
    assert (edge.force_count, edge.feedback_count, edge.generation) == (7, 5, 11)
    assert all(row[:2] == prior[:2] and row[2] is replacement
               for row, prior, replacement in zip(edge._forces, old, prepared, strict=True))
    assert edge.tables is tables and edge.geometry is geometry
    assert (edge._parent_views, edge._child_views) == views
    assert not edge.rebind_parent_interval(parent_clocks=parents, child_clocks=children)
    assert len(prepared) == 2 and edge.interval_rebind_count == 1
