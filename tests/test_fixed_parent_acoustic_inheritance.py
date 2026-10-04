"""Fixed-clock checkpoint producers retain the choice a derived child inherits.

The checkpoint loop, both serializers, history reader, SINT and downscale
planner run here. NumPy state fixtures replace only CUDA stepping and observers.
"""

from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from conftest import requires_netcdf_bridge
from gpuwm import runtime
from gpuwm.acoustic_adaptation import (
    acoustic_receipt, adapt_experiment_to_terrain, readings_from_static,
    steepest_slope,
)
from gpuwm.case_data import load_experiment_case
from gpuwm.cli import main as cli_main
from gpuwm.core import health, streaming, uh_diag
from gpuwm.core.model import restart_identity_payload
from gpuwm.downscale import downscale_plan_path
from gpuwm.io import restart
from gpuwm.offline_child import (
    OfflineChildPlacement, adapt_child_acoustics, child_epssm_is_auto,
    interpolate_parent_initial_state, resolve_child_run_config,
)
from test_case_data import _EXPERIMENT_TOML, make_case_toml
from test_child_acoustic_adaptation import _Domain, _Experiment
from test_offline_child import _history
from test_restart import _fill_setup, _shim_driver_state
from tilestream import physics_inventory


def _history_frame(path, valid, terrain):
    """Legal four-level input with a mild parent slope and a steep SINT child."""
    source_path = path.parent / "two-level-fixture.nc"
    _history(source_path, valid, mp=1, nx=36, ny=36)
    with netCDF4.Dataset(source_path) as source, netCDF4.Dataset(path, "w") as target:
        target.setncatts({key: source.getncattr(key) for key in source.ncattrs()})
        for name, dimension in source.dimensions.items():
            target.createDimension(name, {"bottom_top": 4,
                                         "bottom_top_stag": 5}.get(
                                             name, len(dimension)))
        for name, variable in source.variables.items():
            attrs = {key: variable.getncattr(key) for key in variable.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            output = target.createVariable(
                name, variable.datatype, variable.dimensions,
                **({"fill_value": fill} if fill is not None else {}))
            output.setncatts(attrs)
            value = np.asarray(variable[:])
            for dimension, indices in (("bottom_top", [0, 0, 1, 1]),
                                       ("bottom_top_stag", [0, 0, 1, 1, 2])):
                if dimension in variable.dimensions:
                    value = np.take(value, indices,
                                    axis=variable.dimensions.index(dimension))
            if name == "ZNW":
                value = np.linspace(1., 0., 5, dtype=np.float32).reshape(value.shape)
            if name == "ZNU":
                value = np.asarray([.875, .625, .375, .125],
                                   dtype=np.float32).reshape(value.shape)
            output[:] = value
        target.DX = target.DY = 3000.
        target.GRID_ID = 1
        target.variables["HGT"][0] = terrain
        target.variables["XLAT"][0] = np.broadcast_to(
            (35. + np.arange(36) * .01)[:, None], (36, 36))
        target.variables["XLONG"][0] = np.broadcast_to(
            (-97. + np.arange(36) * .01)[None, :], (36, 36))
    source_path.unlink()


class _CpuStreamedDomain(streaming.StreamedDomain):
    """Real store checkpoint transport with CPU-only clock advancement."""

    def __call__(self, state, cfg, **kwargs):
        assert state is self.state
        self.scalars["elapsed_seconds"] += cfg.dt


def _parent_checkpoint(tmp_path, monkeypatch, producer, choice, *, resumed=False):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    spelling = {"omitted": "", "auto": 'epssm = "auto"\n',
                "explicit-low": "epssm = 0.1\n",
                "explicit-safe": "epssm = 0.5\n", "legacy": ""}[choice]
    exp, data = load_experiment_case(make_case_toml(
        parent_dir, experiment=_EXPERIMENT_TOML + spelling))
    assert exp.root.run.use_adaptive_time_step is False
    assert (1 in exp.auto_epssm) is (choice in {"omitted", "auto", "legacy"})
    assert restart_identity_payload(exp) == restart_identity_payload(
        replace(exp, auto_epssm=()))
    cfg = replace(
        exp.root.run, nx=36, ny=36, nz=4, eta_levels=None,
        dx=3000., dy=3000., dt=15., ztop=9000., run_seconds=600.,
        output_interval_s=600., restart_interval_s=600.,
        time_step_sound=4, mp_physics=1, hybrid_opt=2, etac=.2,
        terrain_opt=1, moist=True, specified=True, spec_bdy_width=3,
        spec_zone=1, relax_zone=2)
    state, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries
    state.lateral_boundaries = build_lateral_boundaries(
        [{"u": state.u.copy()}, {"u": state.u.copy()}], [0., 1200.],
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    state._lateral_boundary_device = SimpleNamespace(rolling=False, clock=None)
    driver.fields["swdown"] = np.zeros((36, 36), dtype=np.float32)
    terrain = np.broadcast_to(np.where(np.arange(36) < 18, 200., 1370.),
                              (36, 36)).astype(np.float32).copy()
    reading = steepest_slope(terrain, cfg.dx, cfg.dy)
    assert reading.slope == pytest.approx(.39)
    prepared_exp = _Experiment((_Domain(1, cfg),), auto_epssm=exp.auto_epssm)
    adapted, receipt = adapt_experiment_to_terrain(
        prepared_exp, readings_from_static(prepared_exp, {1: {"HGT_M": terrain}}))
    assert adapted.domains[0].run is cfg
    assert not receipt[0].adapted and not receipt[0].offcentering_raised

    def resident_step(current, configuration, **kwargs):
        current.elapsed_seconds += configuration.dt

    cp = SimpleNamespace(
        ndarray=np.ndarray, asnumpy=np.asarray, asarray=np.asarray, max=np.max,
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "gpuwm.core.dycore",
                        SimpleNamespace(step=resident_step))
    monkeypatch.setattr(health, "StateHealthValidator", lambda _: SimpleNamespace(
        require_healthy=lambda **kwargs: None))
    monkeypatch.setattr(streaming, "stability_observer", lambda _: lambda *a, **kw: {
        "nan": False, "w_max": 0., "w_argmax": 0,
        "boundary_w_max": 0., "interior_w_max": 0., "cfl": 0.})
    monkeypatch.setattr(runtime, "apply_single_domain_pbl_cadence", lambda *a: None)
    monkeypatch.setattr(runtime, "trajectory_digest_enabled", lambda: False)
    monkeypatch.setattr(uh_diag, "reset_up_heli_max", lambda *a: None)
    monkeypatch.setattr(runtime, "_reset_streamed_up_heli_max", lambda *a: None)

    def history(prepared, output_dir, valid_time, **kwargs):
        path = output_dir / f"wrfout_d01_{valid_time:%Y-%m-%d_%H_%M_%S}"
        _history_frame(path, valid_time, terrain)
        return path

    monkeypatch.setattr(runtime, "write_case_output", history)
    stepper = resident_step
    reseeds = []
    if producer == "streamed":
        store = {key: value.copy() for key, value in
                 physics_inventory.carrier_manifest(state).items()}
        stepper = _CpuStreamedDomain(
            SimpleNamespace(store=store, cfg=cfg, tiles=[state],
                            reseed_clock=lambda value: reseeds.append(dict(value))), None,
            state=state, scalars=physics_inventory.carrier_scalars(state),
            host_store=False)
    labels = None if choice == "legacy" else exp.auto_epssm
    output_dir = parent_dir / "run"
    # A prepared case carries its grid; the runtime hands it to the
    # ensemble member hook on every single-domain run.
    prepared = SimpleNamespace(
        cfg=cfg, grid=None, initial_result=SimpleNamespace(
            state=state, initial_perturbation=None))
    if choice == "legacy":
        # The pre-label producer's API writes the same numeric checkpoint.
        summary = runtime.integrate_prepared_case(
            output_dir, prepared, start_time=exp.start_time,
            output_title="checkpoint inheritance fixture", domain_id=1,
            auto_epssm=labels, stepper=stepper)
    else:
        # Keep the outer door and checkpoint call real. Preparation supplies
        # manufactured state because this gate measures serialization and
        # inheritance, independently of forcing downloads or CUDA dynamics.
        from gpuwm.ingest import preflight
        exp = replace(exp, domains=(replace(exp.root, run=cfg,
                                           history_interval_s=600.),),
                      run_seconds=600., restart_interval_s=600.)
        times = (exp.start_time, exp.start_time + timedelta(seconds=600))
        catalog = SimpleNamespace(valid_times=times, excluded_valid_times=())
        monkeypatch.setattr(preflight, "build_input_catalog", lambda _: catalog)
        monkeypatch.setattr(runtime, "forcing_snapshots", lambda *a: {})
        monkeypatch.setattr(runtime, "forcing_schedule", lambda *a: times)
        monkeypatch.setattr(runtime, "prepare_experiment_case", lambda *a, **kw: prepared)
        monkeypatch.setattr(streaming, "make_stepper", lambda *a, **kw: stepper)
        if producer == "member":
            from gpuwm.ensemble.member import run_member
            outcome = run_member(
                base_config=parent_dir / "case.toml", member_dir=output_dir,
                index=0, seed=1, perturbation="none",
                prepare=lambda _: (exp, data, prepared))
            assert outcome.sim_seconds == 600.
            assert outcome.wrfout_count == 2
            summary = SimpleNamespace(completed_seconds=outcome.sim_seconds,
                wrfout_paths=tuple(sorted(output_dir.glob("wrfout_d01_*"))))
        else:
            summary = runtime.run_experiment(exp, data, output_dir)
    assert summary.completed_seconds == 600.
    assert len(summary.wrfout_paths) == 2
    checkpoint = output_dir / restart.restart_filename(
        exp.start_time + timedelta(seconds=600), "d01")
    header = restart.read_restart_header(checkpoint)
    assert header.get(restart.AUTO_EPSSM_HEADER_KEY, False) is (
        choice in {"omitted", "auto"})
    assert header["config"]["epssm"] == cfg.epssm
    if resumed:
        assert choice in {"omitted", "auto"} and producer != "member"
        first_frames = summary.wrfout_paths
        first_header = header
        # A reset clock and sentinel prove the actual reader restored this
        # endpoint. Clock-only fixture stepping cannot remove the sentinel.
        state.elapsed_seconds = 0.
        state.thp.fill(17.)
        if producer == "streamed":
            stepper.scalars["elapsed_seconds"] = 0.
            stepper.store["state/thp"].fill(17.)
        cfg = replace(cfg, run_seconds=1200.)
        prepared.cfg = cfg
        exp = replace(exp, run_seconds=1200.,
                      domains=(replace(exp.root, run=cfg),))
        times = (exp.start_time, exp.start_time + timedelta(seconds=1200))
        catalog.valid_times = times
        summary = runtime.run_experiment(exp, data, output_dir, restart=checkpoint)
        assert summary.completed_seconds == 1200.
        assert len(summary.wrfout_paths) == 1
        checkpoint = output_dir / restart.restart_filename(
            exp.start_time + timedelta(seconds=1200), "d01")
        header = restart.read_restart_header(checkpoint)
        assert header[restart.AUTO_EPSSM_HEADER_KEY] is True
        assert header["elapsed_seconds"] == 1200.
        assert header["config"]["epssm"] == first_header["config"]["epssm"]
        assert header["physics_setup"] == first_header["physics_setup"]
        actual = (stepper.store["state/thp"]
                  if producer == "streamed" else state.thp)
        assert not actual.any(), "checkpoint restoration left the preparation sentinel"
        if producer == "streamed":
            assert reseeds[-1]["elapsed_seconds"] == 600.
        return checkpoint, (*first_frames, *summary.wrfout_paths), cfg
    return checkpoint, summary.wrfout_paths, cfg


@requires_netcdf_bridge
@pytest.mark.parametrize("producer", ["resident", "streamed"])
@pytest.mark.parametrize("choice", [
    "omitted", "auto", "explicit-low", "explicit-safe", "legacy",
])
@pytest.mark.parametrize("footprint", ["steep", "flat"])
def test_fixed_parent_checkpoint_to_downscale_keeps_the_acoustic_choice(
        tmp_path, monkeypatch, capsys, producer, choice, footprint):
    _assert_downscale_inheritance(
        tmp_path, monkeypatch, capsys, producer, choice, footprint)


@requires_netcdf_bridge
@pytest.mark.parametrize("choice", ["omitted", "auto", "explicit-low"])
def test_ensemble_parent_checkpoint_keeps_the_choice_for_a_steep_child(
        tmp_path, monkeypatch, capsys, choice):
    _assert_downscale_inheritance(
        tmp_path, monkeypatch, capsys, "member", choice, "steep")


@requires_netcdf_bridge
@pytest.mark.parametrize("producer", ["resident", "streamed"])
def test_resumed_fixed_parent_rewrites_the_label_for_a_steep_child(
        tmp_path, monkeypatch, capsys, producer):
    _assert_downscale_inheritance(
        tmp_path, monkeypatch, capsys, producer, "omitted", "steep", resumed=True)


def _assert_downscale_inheritance(
        tmp_path, monkeypatch, capsys, producer, choice, footprint, *, resumed=False):
    checkpoint, frames, parent_cfg = _parent_checkpoint(
        tmp_path, monkeypatch, producer, choice, resumed=resumed)
    i = 18 if footprint == "steep" else 8
    with netCDF4.Dataset(frames[0]) as dataset:
        point = (float(dataset.variables["XLAT"][0, 12, i]),
                 float(dataset.variables["XLONG"][0, 12, i]))
    outdir = tmp_path / "child"
    result = cli_main([
        "downscale", str(frames[0].parent), "--parent-domain", "1",
        "--parent-restart", str(checkpoint), "--point", f"{point[0]},{point[1]}",
        "--ratio", "3", "--child-size", "18,18",
        "--output-interval-seconds", "300", "--out", str(outdir),
        "--dry-run", "--render-products", "none",
    ])
    captured = capsys.readouterr()
    refused = footprint == "steep" and choice in {"explicit-low", "legacy"}
    assert result == (2 if refused else 0), captured.err
    if refused:
        assert "epssm 0.1 is set explicitly" in captured.err
        assert "set d02's epssm to at least 0.2" in captured.err
        assert not downscale_plan_path(outdir, dry_run=True).exists()
        return

    plan = json.loads(downscale_plan_path(outdir, dry_run=True).read_text())
    child_path = Path(plan["child_config"])
    child_cfg = resolve_child_run_config(child_path)
    automatic = choice in {"omitted", "auto"}
    assert child_epssm_is_auto(child_path) is automatic
    assert child_cfg.epssm == parent_cfg.epssm
    placement = OfflineChildPlacement(
        parent_nx=36, parent_ny=36, child_nx=18, child_ny=18,
        parent_grid_ratio=3,
        i_parent_start=plan["placement"]["i_parent_start"],
        j_parent_start=plan["placement"]["j_parent_start"])
    input_bytes = child_path.read_bytes()
    config_bytes = json.dumps(asdict(child_cfg), sort_keys=True).encode()
    child, acoustic = adapt_child_acoustics(
        child_cfg, child_config_path=child_path, frame_path=frames[0],
        placement=placement)
    assert acoustic_receipt([acoustic]) == plan["acoustic_substeps"]
    initial = interpolate_parent_initial_state(
        frames[0], placement, source_mp_physics=1, child_cfg=child_cfg)
    exp = _Experiment((_Domain(2, child_cfg),), auto_epssm=((2,) if automatic else ()))
    prepared, receipts = adapt_experiment_to_terrain(exp, readings_from_static(exp, {2: {
        "HGT_M": initial.fields["HGT"],
        "MAPFAC_U": initial.fields["MAPFAC_U"],
        "MAPFAC_V": initial.fields["MAPFAC_V"],
    }}))
    assert child == prepared.domains[0].run
    assert acoustic_receipt([acoustic]) == acoustic_receipt(receipts)
    if footprint == "steep":
        assert acoustic.reading.slope == pytest.approx(.5633333, abs=1.e-6)
        assert (child.epssm, child.time_step_sound) == (
            (.5, 4) if choice == "explicit-safe" else (.2, 6))
    else:
        assert acoustic.reading.slope == 0.
        assert child is child_cfg
        assert json.dumps(asdict(child), sort_keys=True).encode() == config_bytes
    assert child_path.read_bytes() == input_bytes
    assert not outdir.exists()
