"""Short forecasts exercise live timesteps through the production executor."""
from dataclasses import replace
from datetime import datetime
import json
import time

import pytest

from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp,cu,spectral,first_alarm", [
    (6, 16, False, False), (6, 6, False, False), (6, 0, False, False),
    (16, 0, False, False), (1, 0, True, False), (6, 16, False, True),
])
def test_adaptive_physics_forecast(tmp_path, mp, cu, spectral, first_alarm):
    import cupy as cp
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.model import execute_experiment
    from gpuwm.experiment import (
        DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig)
    from gpuwm.spectral_ops.config import from_mapping
    from gpuwm.verify.cases import wk82
    from gpuwm.verify.cases.nest_ideal_common import (
        assemble_idealized_tree, consume_history_reflectivity)

    snow = mp in (6, 16) and not cu
    initial_dt, maximum_dt = (120, 240) if snow else (6, 12)
    seconds, interval, spacing = (2700., 900., 30000.) if snow else (180., 60., 3000.)
    if first_alarm:
        initial_dt, maximum_dt, spacing = 90, 90, 10000.
    cfg = replace(
        wk82.default_config(), nx=24, ny=24, nz=32, dx=spacing, dy=spacing,
        dt=60. if snow else 6., run_seconds=seconds,
        output_interval_s=interval,
        mp_physics=mp, cu_physics=16 if cu == 6 else cu,
        ntiedtke_tiedtke_closure=cu == 6, cudt_minutes=0.,
        use_adaptive_time_step=True, starting_time_step=initial_dt,
        min_time_step=1, max_time_step=maximum_dt, max_step_increase_pct=20,
        step_to_output_time=True)
    domain = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=interval, run=cfg, time_step=int(cfg.dt))
    config = from_mapping({
        "mode": "apply", "boundary": "tapered", "edge_taper_cells": 4,
        "cadence_steps": 3, "receipt_directory": str(tmp_path / "receipts"),
        "scalar": [{"field": "thp", "diffusion": {
            "order": 3, "reference_wavelength_m": 18000.,
            "e_fold_time_s": 450.}}],
    }) if spectral else None
    exp = ExperimentConfig(
        name="adaptive_physics", start_time=datetime(2000, 1, 1),
        run_seconds=seconds, vertical=VerticalConfig((), 0., 1, .2),
        projection=ProjectionConfig("lambert", 35., -97., 30., 60., -97.),
        restart_interval_s=0., domains=(domain,), spectral_numerics=config)
    coord = make_vertical_coord(cfg.nz)
    sounding = (lambda z: 250. + .003 * z) if snow else (lambda z: wk82.wk82_sounding(z)[0])
    base = make_base_state(coord, sounding,
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    if snow:
        from gpuwm.core.moist import init_moist_balanced
        state = init_moist_balanced(cfg, coord, base, lambda z: z * 0. + 1.e-4)
    else:
        state = wk82.build(cfg, coord, base)
    # Seed frozen precipitation so both snow-ratio validators see a wet column.
    if snow:
        state.qs[0:4] = cp.float32(1.e-5)
    model = assemble_idealized_tree(exp, state)
    steps, history, scheme_steps = [], [], []

    def snapshot(_model, node, ticks):
        consume_history_reflectivity(node, ticks)
        history.append(node.clock.elapsed_seconds)

    def observe(**row):
        steps.append(row)
        if cu:
            scalars = state.physics.cumulus_callable._pipeline[1].scalars
            scheme_steps.append({key: float(scalars[key])
                                 for key in ("dt", "delt", "ztmst")})
        elif snow:
            scheme_steps.append(state.physics._wsm6_minor_loops)

    started = time.perf_counter()
    execute_experiment(model, experiment=exp, history_handler=snapshot,
                       step_observer=observe, validate_state=True)
    cp.cuda.Stream.null.synchronize()
    print(json.dumps({"mp": mp, "cu": cu, "spectral": spectral,
                      "steps": steps, "history": history,
                      "wall_seconds": time.perf_counter() - started}))
    assert model.root.clock.elapsed_seconds == seconds
    assert state.elapsed_seconds == seconds
    assert history == [0., interval, 2 * interval, seconds]
    if not first_alarm:
        assert len({row["dt"] for row in steps}) > 2
    if snow:
        from gpuwm.core.physics import _wsm6_minor_loop_count
        expected = [_wsm6_minor_loop_count(row["dt"]) for row in steps]
        assert len(set(expected)) > 1
        assert scheme_steps == expected
        assert float(state.physics.microphysics.rainnc.max()) > 0.
    for name in ("u", "v", "w", "thp", "qv", "qc", "qr", "qi", "qs", "qg"):
        value = getattr(state, name, None)
        if value is not None:
            assert bool(cp.isfinite(value).all()), name
    if cu:
        assert state.physics.call_counts["cumulus"] == len(steps)
        for row, scalars in zip(steps, scheme_steps, strict=True):
            assert list(scalars.values()) == pytest.approx([row["dt"]] * 3)
    if spectral:
        receipts = sorted((tmp_path / "receipts").rglob("*.json"))
        assert len(receipts) == len(steps) // 3
        for index, path in enumerate(receipts):
            receipt = json.loads(path.read_text())
            window = steps[3 * index:3 * index + 3]
            assert receipt["dt_s"] * 3 == pytest.approx(sum(row["dt"] for row in window))
