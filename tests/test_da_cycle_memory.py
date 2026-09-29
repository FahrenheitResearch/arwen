"""The prepared DA cycle's device lifetimes and its fit decision, on CPU.

The driver runs its trajectories one after another on one card.  Each
trajectory's state, physics driver and model have to be gone before the
next trajectory is wired, or a domain whose one trajectory fits the card
runs out of memory building its second.  And the cycle has to decide
whether its largest trajectory fits before the first upload, once, and
refuse with the sizes when it does not.

These cells run the driver's own ``cycle`` with the device work stood in
by host fakes: the fakes hand out the owners a trajectory holds, keep only
weak references to them, and check at every ``wire`` that the previous
trajectory's owners are dead.
"""

from __future__ import annotations

import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_da_nested_forecast import _nowcast_experiment


class _Owner:
    """A stand-in for one device owner; weak-referenceable, holds nothing."""

    def __init__(self, kind: str, **fields):
        self.kind = kind
        self.__dict__.update(fields)


class _Clock:
    tick_den = 1

    def __init__(self):
        self.ticks = 0
        self.step_count = 0

    @property
    def elapsed_seconds(self) -> float:
        return float(self.ticks) / self.tick_den


def _fake_cupy():
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    module = types.ModuleType("cupy")
    module.get_default_memory_pool = lambda: pool
    module.get_default_pinned_memory_pool = lambda: pool
    module.asarray = np.asarray
    module.abs = np.abs
    module.cuda = SimpleNamespace(
        Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)))
    return module


def _drive(monkeypatch, tmp_path, *, free_bytes=1 << 50, members=2,
           legs=2):
    """Run ``cycle`` over ``legs`` free legs with host fakes.

    Returns ``(events, report_path)``.  ``events`` records, in order, the
    admission's device read and every restore, and each restore records
    how many earlier owners were still alive when it was called.
    """
    from gpuwm.core import clock as clock_module
    from gpuwm.core import health as health_module
    from gpuwm.core import model as model_module
    from gpuwm.core import preflight as preflight_module
    from gpuwm.da import obsop as obsop_module
    from gpuwm.da import perturb as perturb_module
    from gpuwm.da import treatment as treatment_module
    from gpuwm.ensemble import member as member_module
    from gpuwm.ingest import hrrr_physics as physics_module
    from gpuwm.ingest import lateral_bc as lateral_bc_module
    from gpuwm.ingest import prepared_cache as cache_module
    from gpuwm.io import restart as restart_module
    import gpuwm.prepared_single_domain_forecast as psdf
    import gpuwm.runtime as runtime_module
    from tools import da_cycle_prepared as driver

    exp = _nowcast_experiment()
    cfg = exp.root.run
    events: list = []
    owners: list = []

    def remember(obj):
        owners.append((obj.kind, weakref.ref(obj)))
        return obj

    def alive() -> list:
        return [kind for kind, ref in owners if ref() is not None]

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(psdf, "preflight_prepared_forecast",
                        lambda **_: SimpleNamespace(
                            experiment=exp, forcing_hours=(0, 1),
                            proof={}, prepared_cache_path=tmp_path / "cache",
                            cache_identity=None, static={},
                            landuse_identity=None, grid=None,
                            boundary_interval_seconds=3600))

    def free_and_total(device=None):
        events.append(("free", None))
        return int(free_bytes), int(free_bytes)

    monkeypatch.setattr(preflight_module, "device_free_and_total_bytes",
                        free_and_total)
    monkeypatch.setattr(preflight_module, "local_memory_profile_from_device",
                        lambda cp: None)

    shape = (cfg.nz, cfg.ny, cfg.nx)

    def restore(*_args, **_kwargs):
        events.append(("restore", alive()))
        state = remember(_Owner(
            "state", c1h=np.ones(cfg.nz), c2h=np.zeros(cfg.nz),
            dnw=np.ones(cfg.nz), mub2d=np.ones(shape[1:])))
        return SimpleNamespace(initial_result=SimpleNamespace(state=state),
                               met=None, surface=None)

    monkeypatch.setattr(cache_module, "restore_prepared_cache", restore)
    monkeypatch.setattr(physics_module, "initialize_prepared_physics",
                        lambda *a, **k: remember(_Owner("driver", fields={})))
    monkeypatch.setattr(runtime_module, "declared_constant_glw",
                        lambda exp: None)
    monkeypatch.setattr(lateral_bc_module, "bind_lateral_boundary_clock",
                        lambda state, clock: None)
    monkeypatch.setattr(clock_module, "resolve_clock",
                        lambda *a, **k: SimpleNamespace(
                            clocks=lambda: {1: _Clock()}))
    monkeypatch.setattr(clock_module, "build_schedule", lambda *a, **k: None)

    class _Node:
        def __init__(self, dc, grid, state, clock, *rest):
            self.cfg, self.grid, self.state, self.clock = dc, grid, state, clock

    class _Model:
        def __init__(self, root, nodes, schedule, _history, fingerprint):
            self.root = root
            self.nodes_by_grid_id = nodes
            self._pool_trim_policy = {"release_unused_blocks": False}
            owners.append(("model", weakref.ref(self)))

    def execute(model, **_):
        model.root.clock.ticks += 60

    monkeypatch.setattr(model_module, "DomainNode", _Node)
    monkeypatch.setattr(model_module, "ExperimentState", _Model)
    monkeypatch.setattr(model_module, "ModelRuntimeStatus",
                        lambda: SimpleNamespace())
    monkeypatch.setattr(model_module, "execute_experiment", execute)
    monkeypatch.setattr(health_module, "StateHealthValidator",
                        lambda state: SimpleNamespace(
                            validate=lambda phase: SimpleNamespace(ok=True)))
    monkeypatch.setattr(perturb_module, "apply_perturbations",
                        lambda state, seed, cfg: {})
    monkeypatch.setattr(member_module, "refresh_diagnostics",
                        lambda state, **_: None)
    monkeypatch.setattr(obsop_module, "simulated_reflectivity",
                        lambda state, cfg: np.zeros(shape, np.float32))
    monkeypatch.setattr(treatment_module, "verify_treatment",
                        lambda enabled, analyses: {})

    def write_restart(model, directory, *, valid_time):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "gpuwmrst_d01.npz"
        path.write_bytes(b"set")
        return path

    monkeypatch.setattr(driver, "write_leg_restart", write_restart)
    monkeypatch.setattr(driver, "restore_leg_restart",
                        lambda model, path, *, expected_seconds:
                        SimpleNamespace(elapsed_ticks=expected_seconds,
                                        tick_den=1))
    monkeypatch.setattr(driver, "restart_domain_ids", lambda path: (1,))
    monkeypatch.setattr(restart_module, "tree_restart_members",
                        lambda path: {1: Path(path)})

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "da_cycle_prepared", "--prepared-root", str(tmp_path / "prepared"),
        "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
        "--prepared-content-sha256", "0" * 64,
        "--physics-profile", "test", "--run-seconds", "900",
        "--history-interval-seconds", "900", "--members", str(members),
        "--free-legs", str(legs), "--leg-seconds", "60",
        "--out", str(out)])
    events.append(("cycle-exit", driver.main()))
    return events, out / "cycle-report.json"


def test_each_trajectory_is_released_before_the_next_is_wired(monkeypatch,
                                                              tmp_path):
    events, _report = _drive(monkeypatch, tmp_path)
    restores = [still for kind, still in events if kind == "restore"]
    # Two legs of a control and two members.
    assert len(restores) == 6
    for index, still_alive in enumerate(restores):
        assert still_alive == [], (
            f"restore {index} ran beside the previous trajectory's "
            f"{still_alive}")


def test_the_fit_is_decided_once_before_the_first_upload(monkeypatch,
                                                         tmp_path):
    import json

    events, report = _drive(monkeypatch, tmp_path)
    kinds = [kind for kind, _ in events]
    assert kinds.count("free") == 1
    assert kinds.index("free") < kinds.index("restore")
    admission = json.loads(report.read_text(encoding="utf-8"))[
        "memory_admission"]
    assert admission["fits"] is True
    assert admission["perturbation_bytes"] > 0
    assert admission["observation_bytes"] == 0
    assert admission["required_bytes"] <= admission["budget_bytes"]


def test_a_cycle_that_cannot_fit_is_refused_with_its_sizes_before_upload(
        monkeypatch, tmp_path):
    import json

    with pytest.raises(SystemExit) as refusal:
        _drive(monkeypatch, tmp_path, free_bytes=1 << 20)
    message = str(refusal.value)
    report = json.loads((tmp_path / "out" / "cycle-report.json").read_text(
        encoding="utf-8"))["memory_admission"]
    assert report["fits"] is False
    assert f"{report['required_bytes']:,} bytes" in message
    assert f"{report['free_bytes']:,} bytes" in message
    assert f"{report['forecast_resident_bytes']:,} bytes" in message
    assert f"{report['perturbation_bytes']:,} bytes" in message
    assert "before the first upload" in message


def test_the_admission_is_the_forecast_envelope_when_nothing_is_added():
    from gpuwm.core import preflight
    from gpuwm.da.cycle_admission import price_cycle

    exp = _nowcast_experiment()
    price = price_cycle(exp, forcing_intervals=1, observation_points=0,
                        perturbation_bytes=0)
    estimate = preflight.estimate_experiment(exp, forcing_intervals=1)
    assert price.required_bytes == estimate.peak_envelope_bytes


def test_observations_add_and_a_perturbation_competes_with_the_step():
    from gpuwm.da.cycle_admission import price_cycle

    exp = _nowcast_experiment()
    bare = price_cycle(exp, forcing_intervals=1, observation_points=0,
                       perturbation_bytes=0)
    points = 10_000_000
    observed = price_cycle(exp, forcing_intervals=1,
                           observation_points=points, perturbation_bytes=0)
    assert observed.observation_bytes == 5 * points
    assert observed.required_bytes > bare.required_bytes
    small = price_cycle(exp, forcing_intervals=1, observation_points=0,
                        perturbation_bytes=1)
    assert small.required_bytes == bare.required_bytes
    large = price_cycle(exp, forcing_intervals=1, observation_points=0,
                        perturbation_bytes=bare.forecast_step_bytes * 4)
    assert large.required_bytes > bare.required_bytes


def test_a_child_is_priced_with_its_own_scratch():
    from gpuwm.core import preflight
    from gpuwm.da import nested_forecast as nf
    from gpuwm.da.cycle_admission import price_cycle

    exp = _nowcast_experiment()
    child = nf.nest_domain_config(exp, nf.NestGeometry(ratio=3, nx=126,
                                                       ny=126))
    nested = nf.nested_experiment(exp, child)
    price = price_cycle(nested, forcing_intervals=1, observation_points=0,
                        perturbation_bytes=0)
    shared = preflight.estimate_experiment(nested, forcing_intervals=1)
    assert price.domains == 2
    assert price.forecast_resident_bytes == sum(
        domain.resident_bytes for domain in shared.domains
    ) + shared.k_tables_bytes
    assert price.required_bytes > shared.peak_envelope_bytes


def test_the_draw_census_holds_the_spectrum_multiply():
    from gpuwm.da import perturb

    config = perturb.PerturbationConfig.from_mapping({
        "dx_km": 1.0, "dy_km": 1.0, "rim_width": 5,
        "fields": [{"name": "theta", "amplitude": 1.0,
                    "length_scale_km": 20.0}]})
    nz, ny, nx = 55, 1024, 1792
    points = nz * ny * nx
    spectrum = nz * ny * (nx // 2 + 1)
    # The four objects alive at the spectrum multiply, at the compute
    # width of 8 bytes: 2,828,165,120 bytes on this grid.
    assert 8 * points + 5 * 8 * spectrum == 2_828_165_120
    working = perturb.device_working_bytes(config, (nz, ny, nx))
    assert working >= 2_828_165_120
    host = perturb.PerturbationConfig.from_mapping({
        "dx_km": 1.0, "dy_km": 1.0, "rim_width": 5, "fft_host": True,
        "fields": [{"name": "theta", "amplitude": 1.0,
                    "length_scale_km": 20.0}]})
    assert perturb.device_working_bytes(host, (nz, ny, nx)) < working
