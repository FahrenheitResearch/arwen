"""Regression checks for the upheld base4 landing findings."""
import os
import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def test_analysis_reflectivity_uses_domain_shape_for_slab_template():
    from gpuwm.core.refl import analysis_refl_10cm
    from gpuwm.io.wrfout import WrfoutWriter
    state = SimpleNamespace(qv=np.ones((3, 1, 8), np.float32))
    kwargs = ({"shape": (3, 75, 8)} if "shape" in inspect.signature(analysis_refl_10cm).parameters else {})
    refl = analysis_refl_10cm(state, **kwargs)
    writer = SimpleNamespace(nz=3, ny=75, nx=8)
    assert WrfoutWriter._dims_for(writer, "REFL_10CM", refl.shape) == (
        "Time", "bottom_top", "south_north", "west_east")
    assert np.count_nonzero(refl.view(np.uint32)) == 0


def test_store_analysis_uses_host_arrays_without_reading_slab_carriers():
    from gpuwm.core.refl import analysis_refl_10cm
    class PresenceOnly:
        def __getattr__(self, name):
            pytest.fail("store analysis read the stale slab carrier")
    state = SimpleNamespace(qv=PresenceOnly(), _streamed_domain=object())
    field = analysis_refl_10cm(state, shape=(3, 75, 8))
    assert isinstance(field, np.ndarray) and field.shape == (3, 75, 8)
    assert not np.any(field)


def test_tick_zero_disk_projection_includes_reflectivity():
    from gpuwm import disk_budget
    from test_run_disk_budget import _exp
    exp = _exp()
    receipt = disk_budget.projected_run_bytes(
        exp, keep_checkpoints=1, fetch=None, chain=None, render=False)
    for domain, row in zip(exp.domains, receipt["domains"], strict=True):
        assert row["history_bytes"] == (
            row["history_frames"] * disk_budget.history_frame_bytes(domain.run))


def test_acoustic_initialization_marker_is_restart_infrastructure():
    from gpuwm.io.restart import classify_state_attr, state_manifest
    state = SimpleNamespace(_wrf_acoustic_muts_initialized=True)
    assert classify_state_attr("_wrf_acoustic_muts_initialized") == "infra"
    assert state_manifest(state) == {}


def test_slab_analysis_writes_full_domain_rust_artifact(tmp_path):
    import netCDF4
    from gpuwm.prepared_single_domain_forecast import _consume_due_native_refl_10cm
    from gpuwm.io.wrfout import WrfoutWriter
    state = SimpleNamespace(qv=np.ones((3, 1, 8), np.float32),
                            physics=SimpleNamespace(mp_physics=8))
    def forbidden_consume(state):
        pytest.fail("analysis tried to consume a microphysics stash")
    kwargs = ({"shape": (3, 75, 8)} if "shape" in inspect.signature(
        _consume_due_native_refl_10cm).parameters else {})
    reflected = _consume_due_native_refl_10cm(state, 0, forbidden_consume, **kwargs)
    frame = {"T": np.zeros((3, 75, 8), np.float32), "REFL_10CM": reflected}
    path = tmp_path / "store-analysis.nc"
    with WrfoutWriter(path, nx=8, ny=75, nz=3, dx=3000, dy=3000,
        field_schema=frame, global_attrs={"START_DATE": "2024-01-01_00:00:00", "DT": 15.}) as writer:
        writer.write_frame("2024-01-01_00:00:00", frame)
    with netCDF4.Dataset(path) as dataset:
        field = np.asarray(dataset["REFL_10CM"][0])
        assert field.shape == (3, 75, 8)
        assert np.count_nonzero(field.view(np.uint32)) == 0


@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1", reason="strict process")
@pytest.mark.parametrize("physics", [False, True])
def test_strict_iau_fold_keeps_all_dynamics_and_scalar_rates(physics):
    from gpuwm.da.iau import IauStepTendencies
    from gpuwm.core.dycore import _fold_physics_wrf
    z = np.zeros((2, 3, 4), np.float32)
    slots = {k: z.copy() for k in (
        "smag_ru", "smag_rv", "smag_rw", "smag_rth", "smag_rph", "smag_rqv")}
    state = SimpleNamespace(scratch=lambda shape, name: slots[name])
    specs = [(z, None, None, None, None, k, "") for k in slots]
    base = (SimpleNamespace(ru=z+10, rv=z+20, rw=z+30, rtheta=z+40,
                            scalar_for=lambda name: z+60 if name == "qv" else None)
            if physics else None)
    slow = {k: z+i for i, k in enumerate(("ru_t", "rv_t", "rw_t", "rth_t", "rph_t"), 1)}
    pt = IauStepTendencies(base, slow, {"qv": z+6})
    _fold_physics_wrf(state, specs, pt)
    stage = SimpleNamespace(**{k: z.copy() for k in slow})
    pt.strict_stage_tendencies().add_to_slow(stage)
    assert [float(getattr(stage, k).flat[0]) for k in slow] == [1, 2, 3, 4, 5]
    expected = [10, 20, 30, 40, 0, 66] if physics else [0, 0, 0, 0, 0, 6]
    assert [float(v.flat[0]) for v in slots.values()] == expected


def test_bookkeeping_observation_compiles_actual_default_kernels():
    from conftest import requires_gpu
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("needs the card queue")
    import cupy as cp
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("needs GPU")
    from gpuwm.verify.smallstep_bookkeeping_oracle import _observed_module
    module = _observed_module()
    for name in ("small_step_init_uv", "small_step_init_column",
                 "small_step_finish_uv", "small_step_finish_column"):
        module.get_function(name)
