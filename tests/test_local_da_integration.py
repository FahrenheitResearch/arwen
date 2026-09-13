"""Integration seams, using real checkpoint readers and the CPU analysis."""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def small_grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid
    return TargetGrid.from_projection(LambertGrid(ref_lat=35., ref_lon=-97.,
        truelat1=33., truelat2=37., stand_lon=-97., dx=3000., dy=3000.,
        e_we=6, e_sn=6), z_w=np.array([0., 1500., 4000.]), name='local-fixture')


def states_on_disk(tmp_path, count=3):
    output = {}
    for index in range(count):
        root = tmp_path / f'member_{index:03d}'
        root.mkdir()
        fields = dict(u=np.full((2, 5, 6), 10. + index), v=np.ones((2, 6, 5)),
            w=np.zeros((3, 5, 5)), thp=np.full((2, 5, 5), index * .1),
            qv=np.full((2, 5, 5), .004),
            p=np.broadcast_to(np.array([90000., 70000.])[:, None, None], (2, 5, 5)))
        path = root / 'gpuwmrst_d01_end.npz'
        np.savez(path, **{'state/' + k: v for k, v in fields.items()})
        output[index] = {'member_dir': str(root)}
    return output


def test_native_backend_point_analysis_uses_applied_cadence(tmp_path, monkeypatch):
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm.local_da import build_plan, utc
    from test_local_da_plan import request, price, availability
    # Inside the test, so the shared observation package skips exactly the
    # tests that need it instead of emptying the file at collection.
    pytest.importorskip('arwen_global.obs_table')
    from test_local_da_obs_tables import row
    import tomllib
    plan = build_plan(request(scale=2), availability=availability, price=price)
    backend = PreparedBackend(plan, tmp_path)
    # Use the generated experiment through its own reader.
    from gpuwm.experiment import load_experiment
    (tmp_path / 'experiment.toml').write_text(plan['configuration']['experiment'])
    backend.exp = load_experiment(tmp_path / 'experiment.toml')
    backend.mp_physics = backend.exp.root.run.mp_physics
    backend.grid = small_grid()
    backend.setup = {'thb': np.array([300., 310.])}
    lat, lon = backend.grid.lat[2, 2], backend.grid.lon[2, 2]
    report = row(latitude_deg=float(lat), longitude_deg=float(lon), variable='wind_u_m_s', value=12., error=2.)
    monkeypatch.setattr(backend, '_surface_for_members', lambda _: {})
    monkeypatch.setattr(backend, '_observation_window', lambda *a: dict(rows=[report], surface=[], radar=None, cwp=None, receipts=[]))
    inc, receipt = backend.assimilate(0, states_on_disk(tmp_path, plan['selected']['members']))
    assert set(inc) == set(range(plan['selected']['members']))
    assert np.any(inc[0]['u'] != 0.)
    assert receipt['point_observations'][0]['error_inflation'] == plan['cadence_settings']['applied']['error_inflation']
    assert receipt['point_observations'][0]['counts']['accepted'] == 1


def test_cadence_inflates_point_sigma_and_background_gate():
    pytest.importorskip('arwen_global.obs_table')
    from test_local_da_obs_tables import adapt, row
    batches, receipt = adapt([row(value=299.)], error_inflation=3.)
    assert receipt['counts']['accepted'] == 1
    np.testing.assert_array_equal(batches[0].errors[batches[0].mask], 6.)
    with pytest.raises(ValueError, match='inflation'):
        adapt([row()], error_inflation=.5)


def test_member_prepared_factory_bypasses_case_catalog(tmp_path, monkeypatch):
    from gpuwm import ensemble
    from gpuwm.ensemble.member import run_member
    import gpuwm.case_data as case_data
    from gpuwm.experiment import load_experiment
    from gpuwm.local_da import build_plan
    from test_local_da_plan import request, price, availability
    plan = build_plan(request(), availability=availability, price=price)
    config = tmp_path / 'experiment.toml'
    config.write_text(plan['configuration']['experiment'])
    exp = load_experiment(config)
    state = SimpleNamespace(thp=np.zeros((2, 3, 3)), elapsed_seconds=0.)
    prepared = SimpleNamespace(initial_result=SimpleNamespace(state=state))
    calls = []
    def integrate(outdir, item, **kwargs):
        assert item is prepared
        state.thp += 1.
        state.elapsed_seconds = kwargs['run_seconds']
        calls.append(kwargs)
        return SimpleNamespace(wrfout_paths=(), completed_seconds=kwargs['run_seconds'])
    runtime = SimpleNamespace(single_domain=lambda exp: exp.root, integrate_prepared_case=integrate)
    monkeypatch.setitem(sys.modules, 'gpuwm.runtime', runtime)
    import gpuwm
    monkeypatch.setattr(gpuwm, 'runtime', runtime, raising=False)
    monkeypatch.setitem(sys.modules, 'gpuwm.io.wrfout', SimpleNamespace(quarantine_orphan_wrfouts=lambda p: None))
    monkeypatch.setattr(case_data, 'load_experiment_case', lambda *a: pytest.fail('catalog was used'))
    result = run_member(base_config=config, member_dir=tmp_path / 'member', index=0,
        seed=123, perturbation='none', run_seconds=900.,
        prepare=lambda p: (exp, SimpleNamespace(output_title='fixture', output_domain='d01'), prepared))
    assert result.sim_seconds == 900. and len(calls) == 1
    assert result.initial_state_sha256 != result.final_state_sha256


def test_accelerator_staging_preserves_observation_window(tmp_path, monkeypatch):
    import gpuwm.da.radar_assimilation as owner
    from gpuwm.da.letkf import GriddedObs, Localization
    grid = small_grid()
    states = states_on_disk(tmp_path)
    checkpoints = {i: Path(v['member_dir']) / 'gpuwmrst_d01_end.npz' for i, v in states.items()}
    shape = (2, 5, 5)
    batch = GriddedObs(name='point-fixture', values=np.ones(shape), errors=np.ones(shape),
        simulated=np.ones((3, *shape)), mask=np.ones(shape, bool), window=(0, 4, 0, 4))
    fake = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray,
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    monkeypatch.setattr(owner, 'resolve_solve_device', lambda _: ('cuda', 'mock staging'))
    seen = []
    def solve(prior, batches, geometry, cfg, diagnostics):
        seen.extend(b.window for b in batches)
        return {key: np.zeros_like(value) for key, value in prior.items()}
    monkeypatch.setattr(owner, 'analyze', solve)
    config = owner.RadarAssimilationConfig(localization=Localization(1000., 1000.),
        rtps_alpha=0., velocity=False, reflectivity=False, analysis_fields=('thp',), solve_device='cuda')
    owner.assimilate_radar_grid(checkpoints, None, grid, config,
        extra_obs=[batch], extra_obs_provenance={'source': 'fixture'})
    assert seen == [(0, 4, 0, 4)]


def test_accelerator_staging_carries_every_field_a_batch_declares(tmp_path, monkeypatch):
    """The class of defect, not the one instance of it.

    The window was dropped because the device rebuild named its fields one
    by one and the window was not on the list.  This asks the general
    question instead: after staging, every field of the dataclass is
    either the array that moved to the device or the value it arrived
    with, so the next field added to GriddedObs cannot go missing quietly.
    """
    from dataclasses import fields as dataclass_fields
    import gpuwm.da.radar_assimilation as owner
    from gpuwm.da.letkf import GriddedObs, Localization
    grid = small_grid()
    states = states_on_disk(tmp_path)
    checkpoints = {i: Path(v['member_dir']) / 'gpuwmrst_d01_end.npz' for i, v in states.items()}
    shape = (2, 5, 5)
    localization = Localization(2000., 2000.)
    batch = GriddedObs(name='point-fixture', values=np.ones(shape), errors=np.ones(shape),
        simulated=np.ones((3, *shape)), mask=np.ones(shape, bool),
        localization=localization, window=(0, 4, 0, 4))
    moved = {'values', 'errors', 'simulated', 'mask'}
    fake = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray,
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    monkeypatch.setattr(owner, 'resolve_solve_device', lambda _: ('cuda', 'mock staging'))
    staged = []
    def solve(prior, batches, geometry, cfg, diagnostics):
        staged.extend(batches)
        return {key: np.zeros_like(value) for key, value in prior.items()}
    monkeypatch.setattr(owner, 'analyze', solve)
    config = owner.RadarAssimilationConfig(localization=localization,
        rtps_alpha=0., velocity=False, reflectivity=False, analysis_fields=('thp',), solve_device='cuda')
    owner.assimilate_radar_grid(checkpoints, None, grid, config,
        extra_obs=[batch], extra_obs_provenance={'source': 'fixture'})
    assert len(staged) == 1
    names = [field.name for field in dataclass_fields(GriddedObs)]
    assert moved < set(names), 'the staged field list no longer matches the dataclass'
    for name in names:
        before, after = getattr(batch, name), getattr(staged[0], name)
        if name in moved:
            np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
        else:
            assert after == before, (
                f"staging dropped or changed {name}, which it does not move"
                " to the device")
