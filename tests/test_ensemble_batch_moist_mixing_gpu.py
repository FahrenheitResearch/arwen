"""Complete held moist mixing versus independent original fixed helpers."""
from types import MappingProxyType

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _inputs(count, *, terrain=False, mapped=False, boundary='periodic', km_opt=4,
            diff6=0, exempt=False, clock_dt=0.):
    from gpuwm.config import RunConfig
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import DomainState
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.moist import moist_species
    from gpuwm.ensemble.batch_state import PreparedHostMember
    from gpuwm.ensemble.batch_moist_mixing import workspace_specs
    cfg = RunConfig(nx=13, ny=12, nz=5, dx=3000., dy=2700., ztop=9000.,
        dt=3., run_seconds=30., terrain_opt=int(terrain), hybrid_opt=2,
        moist=True, mp_physics=8, bl_pbl_physics=1, sf_sfclay_physics=1,
        sf_surface_physics=2, km_opt=km_opt, diff_opt=2,
        diff_6th_opt=diff6, diff_6th_factor=0.2, diff_6th_slopeopt=1,
        moist_mix6_off=exempt, clock_dt=clock_dt,
        specified=boundary == 'specified', open_x=boundary == 'open')
    coords = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt)
    j, i = np.indices((cfg.ny, cfg.nx))
    topo = 60. + 20. * np.sin(i * 0.8) * np.cos(j * 0.6) if terrain else None
    base = make_base_state(coords, lambda z: 300. + 0.002 * z, cfg.p_surf, cfg.ztop, terrain_z=topo)
    shapes = state_array_shapes(cfg)
    specs = workspace_specs(cfg, has_msf=mapped)
    result = []
    for member in range(count):
        state = DomainState(cfg, array_module=np)
        state.load_base(coords, base)
        for name, origin in (('u', 4.), ('v', -1.5), ('w', 0.04),
                             ('thp', 0.3), ('php', 1.), ('mup', 1.7)):
            value = getattr(state, name)
            index = np.arange(value.size).reshape(value.shape)
            value[...] = origin + 0.09 * member + 0.08 * np.sin(index * 0.7 + member)
            getattr(state, name + '0')[...] = value + np.float32(0.005)
        for name in moist_species(state):
            value = getattr(state, name)
            index = np.arange(value.size).reshape(value.shape)
            scale = 1e5 if name in ('nr', 'ni') else (0.008 if name == 'qv' else 3e-4)
            value[...] = scale * (1. + 0.05 * np.sin(index * 0.31 + member))
            # This exposes a binding to live qv instead of the required qv0.
            getattr(state, name + '0')[...] = value * np.float32(1.03)
        if mapped:
            state.set_map_coriolis(msft=1.01 + 0.002 * i + 0.001 * j,
                msfu=1.01 + 0.002 * np.indices((cfg.ny, cfg.nx + 1))[1],
                msfv=1.01 + 0.001 * np.indices((cfg.ny + 1, cfg.nx))[0])
        update_diagnostics(state, cfg.hypsometric_opt)
        arrays = {name: getattr(state, name).copy() for name in shapes}
        arrays.update({spec.name: np.zeros(spec.shape, spec.dtype) for spec in specs})
        controls = {'physics', 'lateral_boundaries', '_scratch', '_scratch_arena',
                    '_host_setup_state', '_phb_host', 'p_perturbation'}
        scalars = {name: value for name, value in vars(state).items() if name not in shapes and name not in controls}
        clock = {'ticks': 0, 'step_ticks': 3, 'tick_den': 1, 'run_ticks': 30,
                 'step_count': 0, 'dt_fp32': np.float32(3), 'dtbc_fp32': np.float32(0)}
        result.append(PreparedHostMember(cfg, arrays, scalars, clock, phb_host=state._phb_host.copy()))
    return tuple(result)


def _admit(inputs, *, share=True):
    import cupy as cp
    from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    from gpuwm.ensemble.batch_moist_mixing import workspace_specs, required_scratch_slots
    cfg = inputs[0].cfg
    return BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=2**30,
        shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys())) if share else (),
        extra_specs=workspace_specs(cfg, has_msf=inputs[0].scalars['has_msf']),
        scratch_slots=required_scratch_slots(cfg))


def _same(cp, actual, expected, label):
    a, b = cp.asnumpy(actual).view(np.uint32), cp.asnumpy(expected).view(np.uint32)
    different = a != b
    if np.any(different):
        positions = np.argwhere(different)[:12]
        raise AssertionError({'label': label, 'different_words': int(np.count_nonzero(different)),
            'first12': [{'coordinate': row.tolist(), 'actual': f'{int(a[tuple(row)]):08x}',
                        'expected': f'{int(b[tuple(row)]):08x}'} for row in positions]})


CASES = (
    (False, False, 'periodic', 4, 0, False, 0.),
    (False, True, 'specified', 4, 0, False, 0.),
    (True, True, 'specified', 4, 0, False, 0.),
    (True, False, 'open', 4, 2, False, 0.),
    (True, True, 'specified', 4, 2, False, 0.),
    (False, True, 'periodic', 4, 2, False, 0.),
    (False, False, 'specified', 1, 2, False, 0.),
    (True, True, 'periodic', 1, 2, False, 0.),
    (True, True, 'specified', 4, 2, True, 0.),
    (True, True, 'specified', 4, 2, False, 12.),
)


@pytest.mark.parametrize('count', (1, 4, 10, 20))
@pytest.mark.parametrize('terrain,mapped,boundary,km_opt,diff6,exempt,clock_dt', CASES)
def test_full_moist_fixed_helper_preserves_all_held_and_workspaces(
        count, terrain, mapped, boundary, km_opt, diff6, exempt, clock_dt):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_mixing import prepare_fixed_tendencies
    from test_ensemble_batch_mixing_gpu import _scalar_state
    inputs = _inputs(count, terrain=terrain, mapped=mapped, boundary=boundary,
                     km_opt=km_opt, diff6=diff6, exempt=exempt, clock_dt=clock_dt)
    batch = _admit(inputs)
    before = {name: cp.asnumpy(batch.storage.arrays[name]).view(np.uint32).tobytes()
              for name in state_array_shapes(batch.cfg)}
    launch = prepare_fixed_tendencies(batch)
    launch()
    cp.cuda.get_current_stream().synchronize()
    for member, source in enumerate(inputs):
        scalar = _scalar_state(source)
        dycore.prepare_fixed_tendencies(scalar, source.cfg)
        original = dycore.fixed_scalar_tendencies(scalar, source.cfg)
        assert original is not None
        for name, expected in original.items():
            _same(cp, batch.scratch_member_view('smag_r' + name, member), expected, (member, name))
        for slot, expected in scalar._scratch.items():
            if slot.startswith(('smag_', 'diff6_')):
                _same(cp, batch.scratch_member_view(slot, member), expected, (member, slot))
    for name, expected in before.items():
        assert cp.asnumpy(batch.storage.arrays[name]).view(np.uint32).tobytes() == expected, name


@pytest.mark.parametrize('count', (1, 4, 20))
def test_moist_scalar_private_static_fields_preserve_original_words(count):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.ensemble.batch_mixing import prepare_fixed_tendencies
    from test_ensemble_batch_mixing_gpu import _scalar_state
    inputs = _inputs(count, terrain=True, mapped=True, boundary='specified', diff6=2)
    batch = _admit(inputs, share=False)
    prepare_fixed_tendencies(batch)()
    for member, source in enumerate(inputs):
        scalar = _scalar_state(source)
        dycore.prepare_fixed_tendencies(scalar, source.cfg)
        for name, expected in dycore.fixed_scalar_tendencies(scalar, source.cfg).items():
            _same(cp, batch.scratch_member_view('smag_r' + name, member), expected, (member, name))


def test_standalone_scalar_mixing_requires_current_metric_coefficients():
    from gpuwm.ensemble.batch_moist_mixing import prepare_scalar_fixed_tendencies
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    batch = _admit(_inputs(4, mapped=True))
    scalar = prepare_scalar_fixed_tendencies(batch)
    assert isinstance(scalar.tendencies, MappingProxyType)
    with pytest.raises(BatchStateUnsupported, match='Km/Kh'):
        scalar()


def test_prepared_moist_fixed_submission_allocates_no_device_storage():
    import cupy as cp
    from gpuwm.ensemble.batch_mixing import prepare_fixed_tendencies
    batch = _admit(_inputs(4, terrain=True, mapped=True, boundary='specified', diff6=2))
    launch = prepare_fixed_tendencies(batch)
    launch()
    cp.cuda.get_current_stream().synchronize()
    requests = []
    original = cp.cuda.get_allocator()
    def observe(size):
        requests.append(int(size))
        return original(size)
    with cp.cuda.using_allocator(observe):
        launch()
        cp.cuda.get_current_stream().synchronize()
    assert not requests
