from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants
from gpuwm.core import microphysics
from gpuwm.core import nssl2_runtime as runtime
from gpuwm.core import preflight


_REGISTRY_NAMES = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
)
_PRECIPITATION_SLOTS = (
    ("rainnc", "mp_rainnc"),
    ("rainncv", "mp_rainncv"),
    ("snownc", "mp_snownc"),
    ("snowncv", "mp_snowncv"),
    ("graupelnc", "mp_graupelnc"),
    ("graupelncv", "mp_graupelncv"),
    ("hailnc", "mp_hailnc"),
    ("hailncv", "mp_hailncv"),
    ("sr", "mp_sr"),
)
_RAW_RATES = ("rqrcuten", "rqscuten", "rqicuten", "rqccuten")


class _State:
    def __init__(self, cfg):
        shape = (cfg.nz, cfg.ny, cfg.nx)
        full_shape = (cfg.nz + 1, cfg.ny, cfg.nx)
        surface = (cfg.ny, cfg.nx)
        self._scratch = {}
        self.p = np.full(shape, 85000.0, dtype=np.float32)
        self.alt = np.full(shape, 0.8, dtype=np.float32)
        self.thp = np.linspace(0.5, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
        self.h_diabatic = np.full(shape, -7.0, dtype=np.float32)
        self.thb = np.linspace(299.0, 301.0, cfg.nz, dtype=np.float32)
        self.phb = (
            np.arange(cfg.nz + 1, dtype=np.float32)
            * np.float32(125.0 * 9.81)
        )
        self.php = np.zeros(full_shape, dtype=np.float32)
        self.w = np.linspace(-0.2, 0.2, np.prod(full_shape), dtype=np.float32).reshape(full_shape)
        for index, name in enumerate(_REGISTRY_NAMES, start=1):
            setattr(
                self,
                name,
                np.full(shape, index * 1.0e-6, dtype=np.float32),
            )
        self.effc = np.full(shape, 91.0, dtype=np.float32)
        self.effi = np.full(shape, 92.0, dtype=np.float32)
        self.effs = np.full(shape, 93.0, dtype=np.float32)
        precipitation = {}
        for index, (component, slot) in enumerate(_PRECIPITATION_SLOTS, start=1):
            value = np.full(surface, index * 0.25, dtype=np.float32)
            self._scratch[slot] = value
            precipitation[component] = value
        rates = {}
        for index, name in enumerate(_RAW_RATES, start=1):
            value = np.full(shape, index * 1.0e-8, dtype=np.float32)
            self._scratch[f"cu_{name}"] = value
            rates[name] = value
        self.physics = SimpleNamespace(
            state=self,
            mp_physics=18,
            microphysics_updates=0,
            microphysics=SimpleNamespace(**precipitation),
            cu_rates=rates,
            refl_10cm=None,
        )

    def scratch(self, shape, name):
        value = self._scratch.get(name)
        if value is None:
            value = np.zeros(shape, dtype=np.float32)
            self._scratch[name] = value
        elif value.shape != tuple(shape):
            raise ValueError(f"scratch shape mismatch for {name}")
        return value


def _cfg(*, cu_physics=1):
    return RunConfig(
        nx=2,
        ny=1,
        nz=3,
        dx=1000.0,
        dy=1000.0,
        ztop=3000.0,
        dt=1.0,
        run_seconds=1.0,
        moist=True,
        mp_physics=18,
        cu_physics=cu_physics,
    )


@pytest.fixture
def numpy_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "cp", np)
    monkeypatch.setattr(microphysics, "cp", np)


def test_runtime_adapter_binds_exact_state_authorities_and_finishes_once(
        monkeypatch, numpy_runtime):
    cfg = _cfg()
    state = _State(cfg)
    original_theta = state.thb[:, None, None] + state.thp.copy()
    original_thp = state.thp.copy()
    captured = {}
    events = []
    finish_calls = []
    stash_calls = []
    actual_finish = microphysics.moist_physics_finish

    def finish(*args):
        finish_calls.append(args)
        actual_finish(*args)

    def stash(actual_state, reflectivity):
        stash_calls.append((actual_state, reflectivity))
        assert actual_state.physics.refl_10cm is None
        actual_state.physics.refl_10cm = reflectivity

    monkeypatch.setattr(runtime, "moist_physics_finish", finish)
    monkeypatch.setattr(runtime, "stash_refl_10cm", stash)

    def coordinator(
            rho, dz, registry, precipitation, hooks, dt_s, **kwargs):
        captured.update(
            rho=rho,
            dz=dz,
            registry=registry,
            precipitation=precipitation,
            hooks=hooks,
            dt_s=dt_s,
            kwargs=kwargs,
        )
        workspace = object()
        hooks.fused_gs(workspace)
        hooks.nucond(workspace)
        # Sedimentation receives pre-process absolute temperature.  The same
        # scratch is not refreshed for radar until final QVEXCESS returns.
        np.testing.assert_array_equal(
            kwargs["temperature_k"], original_theta * captured["pii"])
        hooks.qv_excess(workspace)
        expected_t = (original_theta + np.float32(1.5)) * captured["pii"]
        np.testing.assert_array_equal(kwargs["temperature_k"], expected_t)
        kwargs["refl_10cm"][...] = np.float32(37.5)
        kwargs["re_cloud_m"][...] = np.float32(12.0e-6)
        kwargs["re_ice_m"][...] = np.float32(30.0e-6)
        kwargs["re_snow_m"][...] = np.float32(100.0e-6)
        hooks.moist_physics_finish(workspace)
        return workspace

    monkeypatch.setattr(runtime, "run_nssl2_production_step", coordinator)

    def fused(_workspace, fields):
        events.append(("fused", fields))
        captured["pii"] = fields.pii.copy()
        fields.theta[...] += np.float32(0.25)

    def nucond(_workspace, fields):
        events.append(("nucond", fields))
        fields.theta[...] += np.float32(0.5)

    def qv_excess(_workspace, fields):
        events.append(("qv_excess", fields))
        fields.theta[...] += np.float32(0.75)

    diagnostics = runtime.apply_nssl2_production(
        state,
        cfg,
        1.0,
        runtime.NSSL2RuntimeHooks(
            fused_gs=fused,
            nucond=nucond,
            qv_excess=qv_excess,
        ),
        output_due=True,
        radiation_due=True,
    )

    assert [name for name, _ in events] == ["fused", "nucond", "qv_excess"]
    assert events[0][1] is events[1][1] is events[2][1]
    fields = events[0][1]
    np.testing.assert_array_equal(fields.theta, original_theta + np.float32(1.5))
    np.testing.assert_array_equal(fields.rho, np.float32(1.0) / state.alt)
    np.testing.assert_array_equal(
        fields.pii,
        np.power(state.p / np.float32(100000.0), np.float32(287.0 / 1004.5)),
    )
    expected_dz = np.diff(
        state.phb / np.float32(constants.G))[:, None, None]
    np.testing.assert_array_equal(
        fields.dz, np.broadcast_to(expected_dz, fields.dz.shape))
    assert fields.pressure is state.p
    assert fields.w is state.w
    assert captured["rho"] is fields.rho
    assert captured["dz"] is fields.dz
    assert captured["dt_s"] == 1.0

    registry = captured["registry"]
    for name in _REGISTRY_NAMES:
        assert getattr(registry, name) is getattr(state, name)
    precipitation = captured["precipitation"]
    for component, slot in _PRECIPITATION_SLOTS:
        value = state._scratch[slot]
        assert getattr(precipitation, component) is value
        assert getattr(diagnostics, component) is value

    kwargs = captured["kwargs"]
    assert kwargs["first_step"] is True
    assert kwargs["cu_used"] is True
    for argument, rate_name in (
            ("qrcuten", "rqrcuten"),
            ("qscuten", "rqscuten"),
            ("qicuten", "rqicuten"),
            ("qccuten", "rqccuten")):
        assert kwargs[argument] is state.physics.cu_rates[rate_name]
    assert kwargs["output_due"] is True
    assert kwargs["radiation_due"] is True

    # One finish, one metre->micron conversion, and one stash. A second
    # conversion would make these values 1e6 times too large.
    assert len(finish_calls) == 1
    np.testing.assert_array_equal(state.thp, original_thp + np.float32(1.5))
    np.testing.assert_array_equal(state.h_diabatic, np.float32(1.5))
    np.testing.assert_array_equal(state.effc, np.float32(12.0))
    np.testing.assert_array_equal(state.effi, np.float32(30.0))
    np.testing.assert_array_equal(state.effs, np.float32(100.0))
    assert len(stash_calls) == 1
    assert state.physics.refl_10cm is stash_calls[0][1]
    np.testing.assert_array_equal(state.physics.refl_10cm, np.float32(37.5))
    # The outer PhysicsDriver acceptance remains the only counter authority.
    assert state.physics.microphysics_updates == 0


@pytest.mark.parametrize("failure_stage", ["fused_gs", "qv_excess"])
def test_earlier_hook_failure_has_no_finish_radius_conversion_or_refl_stash(
        monkeypatch, numpy_runtime, failure_stage):
    cfg = _cfg()
    state = _State(cfg)
    original_thp = state.thp.copy()
    state._scratch["refl_t"] = np.full(
        state.p.shape, -123.0, dtype=np.float32)
    radii_before = tuple(value.copy() for value in (
        state.effc, state.effi, state.effs))
    finish_calls = []
    stash_calls = []

    monkeypatch.setattr(
        runtime, "moist_physics_finish",
        lambda *args: finish_calls.append(args))
    monkeypatch.setattr(
        runtime, "stash_refl_10cm",
        lambda *args: stash_calls.append(args))

    def coordinator(_rho, _dz, _registry, _precipitation, hooks, _dt, **_kwargs):
        hooks.fused_gs(object())
        hooks.nucond(object())
        hooks.qv_excess(object())
        pytest.fail("coordinator advanced after failed runtime hook")

    monkeypatch.setattr(runtime, "run_nssl2_production_step", coordinator)

    def fused(_workspace, fields):
        if failure_stage == "fused_gs":
            raise RuntimeError("runtime stage failed")
        fields.theta[...] += np.float32(0.01)

    def nucond(_workspace, fields):
        fields.theta[...] += np.float32(0.02)

    def qv_excess(_workspace, fields):
        if failure_stage == "qv_excess":
            raise RuntimeError("runtime stage failed")
        fields.theta[...] += np.float32(0.03)

    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=fused,
        nucond=nucond,
        qv_excess=qv_excess,
    )
    with pytest.raises(RuntimeError, match="runtime stage failed"):
        runtime.apply_nssl2_production(
            state, cfg, 1.0, hooks,
            output_due=True, radiation_due=True)

    assert finish_calls == []
    assert stash_calls == []
    assert state.physics.refl_10cm is None
    np.testing.assert_array_equal(
        state._scratch["refl_t"],
        (state.thb[:, None, None] + original_thp)
        * np.power(state.p / constants.P0, constants.RCP),
    )
    for actual, expected in zip((state.effc, state.effi, state.effs),
                                radii_before):
        np.testing.assert_array_equal(actual, expected)


def test_missing_hooks_and_stale_refl_fail_before_prep_or_coordinator(
        monkeypatch, numpy_runtime):
    cfg = _cfg()
    state = _State(cfg)
    h_before = state.h_diabatic.copy()
    scratch_before = set(state._scratch)
    coordinator_calls = []
    monkeypatch.setattr(
        runtime, "run_nssl2_production_step",
        lambda *args, **kwargs: coordinator_calls.append((args, kwargs)))

    with pytest.raises(
            runtime.NSSL2ProductionConfigurationError, match="fused_gs"):
        runtime.apply_nssl2_production(
            state,
            cfg,
            1.0,
            runtime.NSSL2RuntimeHooks(
                nucond_qvexcess=lambda workspace, fields: None),
        )
    np.testing.assert_array_equal(state.h_diabatic, h_before)
    assert set(state._scratch) == scratch_before

    state.physics.refl_10cm = object()
    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
        nucond_qvexcess=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
    )
    with pytest.raises(
            runtime.NSSL2ProductionConfigurationError,
            match="stash was not consumed"):
        runtime.apply_nssl2_production(
            state, cfg, 1.0, hooks, output_due=True)
    np.testing.assert_array_equal(state.h_diabatic, h_before)
    assert set(state._scratch) == scratch_before
    assert coordinator_calls == []


@pytest.mark.parametrize("broken", ["precipitation-copy", "kf-rate-copy"])
def test_noncanonical_driver_aliases_fail_before_state_mutation(
        monkeypatch, numpy_runtime, broken):
    cfg = _cfg()
    state = _State(cfg)
    if broken == "precipitation-copy":
        state.physics.microphysics.rainnc = \
            state.physics.microphysics.rainnc.copy()
        match = "canonical scratch slot mp_rainnc"
    else:
        state.physics.cu_rates["rqrcuten"] = \
            state.physics.cu_rates["rqrcuten"].copy()
        match = "raw-rate authority rqrcuten"
    h_before = state.h_diabatic.copy()
    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
        nucond_qvexcess=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
    )
    with pytest.raises(runtime.NSSL2ProductionConfigurationError, match=match):
        runtime.apply_nssl2_production(state, cfg, 1.0, hooks)
    np.testing.assert_array_equal(state.h_diabatic, h_before)


def test_first_call_comes_only_from_microphysics_update_counter(
        monkeypatch, numpy_runtime):
    cfg = _cfg(cu_physics=0)
    state = _State(cfg)
    state.physics.cu_rates = None
    state.physics.microphysics_updates = 4
    observed = {}

    def coordinator(_rho, _dz, _registry, _precipitation, hooks, _dt, **kwargs):
        observed.update(kwargs)
        workspace = object()
        hooks.fused_gs(workspace)
        hooks.nucond_qvexcess(workspace)
        hooks.moist_physics_finish(workspace)

    monkeypatch.setattr(runtime, "run_nssl2_production_step", coordinator)
    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
        nucond_qvexcess=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.02)),
    )
    runtime.apply_nssl2_production(state, cfg, 1.0, hooks)
    assert observed["first_step"] is False
    assert observed["cu_used"] is False
    assert not (set(observed) & {"qrcuten", "qscuten", "qicuten", "qccuten"})


def test_non_output_call_prepares_sedimentation_temperature_without_refl_stash(
        monkeypatch, numpy_runtime):
    cfg = _cfg(cu_physics=0)
    state = _State(cfg)
    state.physics.cu_rates = None
    shape = state.p.shape
    state._scratch["refl_t"] = np.full(shape, -17.0, dtype=np.float32)
    state._scratch["refl_10cm"] = np.full(shape, -23.0, dtype=np.float32)
    stash_calls = []
    monkeypatch.setattr(
        runtime, "stash_refl_10cm", lambda *args: stash_calls.append(args))

    def coordinator(_rho, _dz, _registry, _precipitation, hooks, _dt, **kwargs):
        assert kwargs["output_due"] is False
        np.testing.assert_array_equal(
            kwargs["temperature_k"],
            (state.thb[:, None, None] + state.thp)
            * np.power(state.p / constants.P0, constants.RCP),
        )
        assert kwargs["refl_10cm"] is None
        workspace = object()
        hooks.fused_gs(workspace)
        hooks.nucond_qvexcess(workspace)
        # Eventual selector cadence: radii are refreshed on every MP call.
        kwargs["re_cloud_m"][...] = np.float32(12.0e-6)
        kwargs["re_ice_m"][...] = np.float32(30.0e-6)
        kwargs["re_snow_m"][...] = np.float32(100.0e-6)
        hooks.moist_physics_finish(workspace)

    monkeypatch.setattr(runtime, "run_nssl2_production_step", coordinator)
    hooks = runtime.NSSL2RuntimeHooks(
        fused_gs=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.01)),
        nucond_qvexcess=lambda workspace, fields: fields.theta.__iadd__(
            np.float32(0.02)),
    )
    runtime.apply_nssl2_production(
        state, cfg, 1.0, hooks,
        output_due=False, radiation_due=True)

    assert stash_calls == []
    assert state.physics.refl_10cm is None
    assert np.all(state._scratch["refl_t"] > 0.0)
    np.testing.assert_array_equal(state._scratch["refl_10cm"], -23.0)


def test_mp18_preflight_registers_exact_runtime_scratch():
    slots = preflight.scratch_slot_registry(_cfg())
    expected = {
        "mp_th", "mp_rho", "mp_pii", "mp_dz8w", "mp_z8w",
        "nssl2_nucond_ss",
        "refl_t", "refl_10cm",
    }
    assert expected <= set(slots)
    assert slots["mp_th"] == (3, 1, 2)
    assert slots["mp_z8w"] == (4, 1, 2)
    assert slots["refl_10cm"] == (3, 1, 2)
    for name in expected - {"refl_10cm"}:
        assert preflight.scratch_slot_uses_arena(name)
    assert not preflight.scratch_slot_uses_arena("refl_10cm")


def test_global_mp18_selector_requires_persistent_binding():
    state = SimpleNamespace(qv=np.ones((1,), dtype=np.float32))
    with pytest.raises(
            runtime.NSSL2ProductionConfigurationError,
            match="PhysicsDriver"):
        microphysics.apply(state, _cfg(), 1.0)


@pytest.mark.gpu
@requires_gpu
def test_gpu_runtime_boundary_executes_real_transport_nucond_and_diagnostics():
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.core.state import DomainState

    cfg = RunConfig(
        nx=2,
        ny=2,
        nz=3,
        dx=1000.0,
        dy=1000.0,
        ztop=3000.0,
        dt=1.0,
        run_seconds=1.0,
        moist=True,
        mp_physics=18,
    )
    state = DomainState(cfg)
    shape = state.p.shape
    state.thb[...] = cp.asarray([299.0, 300.0, 301.0], dtype=cp.float32)
    state.phb[...] = cp.asarray(
        [0.0, 125.0, 250.0, 375.0], dtype=cp.float32) * cp.float32(9.81)
    state.thp[...] = cp.float32(0.0)
    state.php[...] = cp.float32(0.0)
    state.p[...] = cp.float32(85000.0)
    state.alt[...] = cp.float32(0.8)
    state.w[...] = cp.float32(0.0)
    state.qv[...] = cp.float32(0.005)
    state.qc[...] = cp.float32(2.0e-5)
    state.qr[...] = cp.float32(2.0e-6)
    state.qi[...] = cp.float32(1.0e-6)
    state.qs[...] = cp.float32(2.0e-6)
    state.qg[...] = cp.float32(1.0e-6)
    state.qh[...] = cp.float32(5.0e-7)
    state.qndrop[...] = cp.float32(1.0e8)
    state.qnr[...] = cp.float32(1.0e5)
    state.qni[...] = cp.float32(1.0e5)
    state.qns[...] = cp.float32(1.0e4)
    state.qng[...] = cp.float32(1.0e4)
    state.qnh[...] = cp.float32(1.0e3)
    state.qnn[...] = cp.float32(4.0e8)
    state.qvolg[...] = state.qg / cp.float32(500.0)
    state.qvolh[...] = state.qh / cp.float32(900.0)
    driver = initialize_physics(state, cfg)
    hook_events = []

    def fused_gs(workspace, fields):
        hook_events.append("fused_gs")
        workspace.field("qc")[...] += cp.float32(1.0e-9)
        fields.theta[...] += cp.float32(0.01)

    def nucond_qvexcess(workspace, fields):
        hook_events.append("nucond_qvexcess")
        launch_nucond(
            fields.theta,
            fields.rho,
            fields.pressure,
            fields.pii,
            fields.w,
            workspace.field("qv"),
            workspace.field("qc"),
            workspace.field("qr"),
            workspace.field("qi"),
            workspace.field("qs"),
            workspace.field("qndrop"),
            workspace.field("qnr"),
            workspace.field("qni"),
            workspace.field("qns"),
            workspace.field("qnn"),
            1.0,
            concentration_space=True,
        )

    diagnostics = runtime.apply_nssl2_production(
        state,
        cfg,
        1.0,
        runtime.NSSL2RuntimeHooks(
            fused_gs=fused_gs,
            nucond_qvexcess=nucond_qvexcess,
        ),
        output_due=True,
        radiation_due=True,
    )
    driver.accept_microphysics(diagnostics)
    cp.cuda.Stream.null.synchronize()

    assert hook_events == ["fused_gs", "nucond_qvexcess"]
    assert driver.microphysics_updates == 1
    assert driver.refl_10cm is not None
    assert consume_refl_10cm(state).shape == shape
    assert driver.refl_10cm is None
    for name in (*_REGISTRY_NAMES, "h_diabatic", "effc", "effi", "effs"):
        value = getattr(state, name)
        assert bool(cp.isfinite(value).all()), name
    for name in _REGISTRY_NAMES:
        assert bool((getattr(state, name) >= cp.float32(0.0)).all()), name
    micron = np.float32(1.0e6)
    assert float(cp.min(state.effc).get()) >= float(np.float32(2.51e-6) * micron)
    assert float(cp.min(state.effi).get()) >= float(np.float32(10.01e-6) * micron)
    assert float(cp.min(state.effs).get()) >= float(np.float32(25.0e-6) * micron)
    assert bool((diagnostics.sr >= cp.float32(0.0)).all())
    assert bool((diagnostics.sr <= cp.float32(1.0)).all())
