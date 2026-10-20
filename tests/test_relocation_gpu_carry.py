"""GPU moving carry must preserve the CPU donor rule and every carried word."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("classes", ["mixed", "water", "fallback"])
def test_gpu_donors_and_physics_carry_match_cpu_words(classes):
    from gpuwm.core.nest_relocation import Placement, plan_relocation, transplant_overlap
    from gpuwm.core.physics_continuation import _capture, shift_continuation, shift_carriers
    from gpuwm.ingest.relocation_init import donor_fill_plan, overlap_mask_for_plan

    plan = plan_relocation(
        placement_from=Placement(2, 10, 10),
        placement_to=Placement(2, 12, 9, generation=1),
        parent_grid_ratio=3, child_nx=30, child_ny=24)
    mask = overlap_mask_for_plan(plan, (24, 30))
    land = np.indices(mask.shape).sum(axis=0) % 2
    if classes == "water":
        land[:] = 0
    elif classes == "fallback":
        land[:] = ~mask
    cpu = donor_fill_plan(overlap_mask=mask, landmask=land)
    gpu = donor_fill_plan(overlap_mask=mask, landmask=land, device=True)
    np.testing.assert_array_equal(gpu.donor_flat, cpu.donor_flat)
    assert gpu.counts == cpu.counts

    # Include signed zero and NaN payloads: numerical equality alone cannot
    # prove a pure transport preserved all words.
    words = np.arange(3 * 24 * 30, dtype=np.uint32).reshape(3, 24, 30)
    words.ravel()[::13] = 0x80000000
    words.ravel()[::17] = 0x7fc12345
    value = words.view(np.float32)
    original = cp.asarray(value)
    captured = _capture(original)
    reference = {"held/rate": value, "mp_rainnc": value[0]}
    resident = {"held/rate": captured, "mp_rainnc": captured[0]}
    strip = {"mp_rainnc": np.full((24, 30), 4., np.float32)}
    expected = shift_continuation(reference, plan, strip=strip, ring=2)
    actual = shift_continuation(resident, plan, strip=strip, ring=2)
    for name in expected:
        assert isinstance(actual[name], cp.ndarray)
        np.testing.assert_array_equal(actual[name].get().view(np.uint32),
                                      expected[name].view(np.uint32))
    expected = shift_carriers({"glw": value}, plan, cpu)
    actual = shift_carriers({"glw": captured}, plan, gpu)
    np.testing.assert_array_equal(actual["glw"].get().view(np.uint32),
                                  expected["glw"].view(np.uint32))
    targets = []
    for source, empty in ((value, np.zeros_like(value)),
                          (captured, cp.zeros_like(captured))):
        transplant_overlap(source_state=SimpleNamespace(thp=source),
                           target_state=SimpleNamespace(thp=empty), plan=plan,
                           attrs=("thp",))
        targets.append(empty.get() if hasattr(empty, "get") else empty)
    np.testing.assert_array_equal(targets[0].view(np.uint32), targets[1].view(np.uint32))
    captured[...] = 7.
    np.testing.assert_array_equal(original.get().view(np.uint32), words)


def test_gpu_nearest_donor_keeps_first_index_on_ties():
    from gpuwm.ingest.relocation_init import _nearest_donor
    targets = np.array([[1, 1], [2, 2], [0, 0]], dtype=np.float64)
    donors = np.array([[1, 0], [0, 1], [3, 2], [2, 3]], dtype=np.float64)
    np.testing.assert_array_equal(_nearest_donor(targets, donors, device=True),
                                  _nearest_donor(targets, donors))


def test_gpu_land_rebuild_keeps_surface_reservoirs_independent(monkeypatch):
    from test_nest_window_gpu import _real_window_case
    from gpuwm.runtime import PreparedTreeRelocationChildPreparer
    from tilestream import test_nest_executor as executor

    initialize, moved, parent = _real_window_case(monkeypatch, complete_statics=True)
    initialized = initialize(moved, parent)
    shape = (moved.run.ny, moved.run.nx)
    land = {"tsk": cp.full(shape, 301., cp.float32),
            "xice": cp.full(shape, .2, cp.float32)}
    exp = executor.experiment(parent.cfg.run, moved.run, steps=2)
    preparer = PreparedTreeRelocationChildPreparer(exp=exp, model=SimpleNamespace())
    preparer._rebuild_driver(initialized, moved, parent, land)
    fields = initialized.state.physics.fields
    fields["tsk"][...] = 302.
    fields["xice"][...] = .3
    np.testing.assert_array_equal(fields["sst"].get(), np.full(shape, 301., np.float32))
    np.testing.assert_array_equal(fields["xicem"].get(), np.full(shape, .2, np.float32))
