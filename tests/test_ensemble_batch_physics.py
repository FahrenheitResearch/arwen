"""Physics column allocation and immutable-load source audits without CUDA."""

from pathlib import Path

import numpy as np
import pytest

from gpuwm.ensemble.batch_physics import (ColumnField, column_memory_plan,
    remap_shared_column_loads, ring_memory_plan, ysu_output_plan, _thompson_base_source,
    physics_coupling_source)


def test_column_plan_prices_member_private_packed_words():
    fields = (ColumnField("theta", (49, 150, 150)), ColumnField("soil", (4, 150, 150)),
              ColumnField("category", (150, 150), "int32"))
    plan = column_memory_plan(fields)
    rows = {r["name"]: r for r in plan.inventory(10)}
    assert rows["physics:theta"]["ownership"] == "member"
    assert rows["physics:theta"]["shape"] == (10, 49, 150, 150)
    assert rows["physics:category"]["dtype"] == np.dtype("int32").str


def test_ysu_plan_prices_full_one_launch_workspace():
    from gpuwm.core.physics_inventory import ysu_workspace_floats
    plan = ysu_output_plan(nz=49, ny=150, nx=150, members=10)
    rows = {r["name"]: r for r in plan.inventory(10)}
    assert rows["ysu:workspace"]["shape"] == (ysu_workspace_floats(49, 225000),)
    assert rows["ysu:workspace"]["ownership"] == "shared"
    assert rows["ysu:du"]["ownership"] == "member"
    assert rows["ysu:cloudflg"]["dtype"] == np.dtype("int32").str
    assert plan.required_bytes(10) > 10**9


def test_shared_grid_load_audit_changes_only_integer_indices():
    source = "float v = xland[col]; float w = xland[col] * coefficient;"
    got = remap_shared_column_loads(source, ("xland",), member_columns=15)
    assert got == "float v = xland[(col) % 15]; float w = xland[(col) % 15] * coefficient;"
    with pytest.raises(ValueError, match="written"):
        remap_shared_column_loads("xland[idx] = value;", ("xland",), member_columns=15)
    with pytest.raises(ValueError, match="unsupported"):
        remap_shared_column_loads("float a = xland[col + 1];", ("xland",), member_columns=15)


@pytest.mark.parametrize("module,pointers", [
    ("ysu", ("xland",)), ("sfclay", ("xland", "lakemask")),
    ("noah", ("ivgtyp", "isltyp", "xland_a", "shdmin_a", "shdmax_a", "tmn_a", "snoalb_a", "embck_a")),
])
def test_default_physics_sources_have_only_audited_shared_field_loads(module, pointers):
    path = Path(__file__).parents[1] / "gpuwm" / "core" / "kernels" / f"{module}.cu"
    original = path.read_text()
    transformed = remap_shared_column_loads(original, pointers, member_columns=22500)
    assert transformed != original
    for pointer in pointers:
        assert f"{pointer}[(" in transformed


def test_compact_ring_ledger_prices_member_saves_and_shared_descriptor():
    plan = ring_memory_plan({"theta": 49, "rain": 1, "h_diabatic": 49},
                            ny=150, nx=150, width=1, zero_fields=("h_diabatic",))
    rows = {r["name"]: r for r in plan.inventory(10)}
    assert rows["physics:ring:theta"]["shape"] == (10, 49, 596)
    assert rows["physics:ring:rain"]["shape"] == (10, 1, 596)
    assert "physics:ring:h_diabatic" not in rows
    assert rows["physics:ring:table"]["shape"] == (3, 4)


def test_shared_real_terrain_thompson_loads_change_only_integer_indices():
    path = Path(__file__).parents[1] / "gpuwm" / "core" / "kernels" / "thompson.cu"
    original = path.read_text()
    changed = _thompson_base_source(original, member_columns=22500)
    assert "thb[thb_full ? k * __ensemble_member_columns + idx % __ensemble_member_columns : k]" in changed
    assert "phb[phb_full ? (k + 1) * __ensemble_member_columns + idx % __ensemble_member_columns : k + 1]" in changed
    assert changed.count("__fadd_rn") == original.count("__fadd_rn")
    assert changed.count("__fmul_rn") == original.count("__fmul_rn")


def test_member_coupling_source_keeps_all_float_operations_and_local_faces():
    import ast
    path = Path(__file__).parents[1] / "gpuwm" / "core" / "tendency_coupling.py"
    tree = ast.parse(path.read_text())
    original = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "_SOURCE" for target in node.targets))
    changed = physics_coupling_source(original, members=10, ny=150, nx=150)
    for intrinsic in ("__fadd_rn", "__fmul_rn", "__fdiv_rn"):
        assert changed.count(intrinsic) == original.count(intrinsic)
    assert "const long long j = packed_j % __ensemble_member_y" in changed
    assert "const long long i = cell % nx" in changed
    assert "const long long g = packed_g % (__ensemble_member_y + 1)" in changed
    assert "(k * ny + member * __ensemble_member_y) * nx" in changed
