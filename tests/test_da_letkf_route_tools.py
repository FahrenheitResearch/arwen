"""The LETKF route A/B and shadow tools: arithmetic and seam behaviour on CPU."""
import json
import sys
import types

import numpy as np

from tools.da_letkf_device_ab import build_case, compare


def test_compare_reports_max_rms_and_support():
    ref = {"a": np.zeros((2, 1, 2, 2))}
    ref["a"][:, 0, 0, 0] = [1.0, -1.0]
    new = {"a": ref["a"].copy()}
    new["a"][0, 0, 0, 0] += 1e-12
    row = compare(new, ref)["a"]
    assert row["max_abs_diff"] == (1.0 + 1e-12) - 1.0
    assert row["rms_diff"] > 0 and not row["identical_bytes"]
    assert row["zero_support_equal"]
    assert row["max_abs_increment"] == 1.0
    same = compare(ref, ref)["a"]
    assert same["identical_bytes"] and same["max_abs_diff"] == 0.0


def test_build_case_tiny_shape_is_a_valid_analysis_input():
    prior, batches, grid, loc = build_case(members=4, shape=(3, 12, 13),
                                           radars=2, fields=("u", "v"))
    assert set(prior) == {"u", "v"} and prior["u"].shape == (4, 3, 12, 13)
    assert grid.geodesic and grid.nz == 3
    names = [b.name for b in batches]
    assert names[:2] == ["vr_0", "vr_1"] and "reflectivity" in names
    assert all(b.window is not None for b in batches[:2])


def test_shadow_returns_the_production_result_and_records_the_shadow(
        tmp_path, monkeypatch):
    from gpuwm.da import letkf
    from gpuwm.da import radar_assimilation as ra
    from tools import da_letkf_route_shadow as shadow

    fake_cp = types.SimpleNamespace(
        get_default_memory_pool=lambda: types.SimpleNamespace(
            free_all_blocks=lambda: None))
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    produced = {"x": np.ones((2, 1, 1, 2))}
    shadowed = {"x": np.ones((2, 1, 1, 2)) + 1e-9}

    def attempt(solver, prior, batches, geometry, config, *, namespace,
                storage, supports_staging, progress, diagnostics):
        diagnostics.active_points = 7
        return ({k: v.copy() for k, v in
                 (produced if storage == "prod" else shadowed).items()},
                diagnostics, 0.0, 0.0)

    monkeypatch.setattr(ra, "_analysis_attempt", attempt)
    original = ra._execute_analysis
    try:
        shadow.install(tmp_path, "prod", "shadow")
        config = types.SimpleNamespace(analysis_fields=("x",))
        batch = types.SimpleNamespace(name="b")
        diag = letkf.LetkfDiagnostics()
        inc, d, _s, _u, storage, attempts = ra._execute_analysis(
            letkf.analyze, {"x": None}, [batch], None, config,
            namespace=fake_cp, device="cuda", diagnostics=diag)
    finally:
        ra._execute_analysis = original
    assert storage == "prod" and attempts[0]["storage"] == "prod"
    assert inc["x"].tobytes() == produced["x"].tobytes()
    assert diag.active_points == 7
    row = json.loads((tmp_path / "call-001.json").read_text())
    assert row["production"]["route"] == "prod"
    assert row["shadow"]["route"] == "shadow"
    assert abs(row["difference"]["x"]["max_abs_diff"] - 1e-9) < 1e-15
