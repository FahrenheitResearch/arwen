"""Member scalar transport and sumflux words against original scalar calls."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


class _Tendencies:
    def __init__(self, fields):
        self.fields = fields

    def scalar_for(self, name):
        return self.fields.get(name)


def _pack(count, *, mapped, boundary, sources, vorder=3):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import DomainState
    from gpuwm.ensemble import batch_moist
    from gpuwm.ensemble.batch_state import BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES
    from gpuwm.ensemble.batch_storage import BatchArraySpec
    from test_ensemble_batch_bigstep_gpu import _scalar_state
    cfg = RunConfig(nx=11, ny=11, nz=5, dx=3000., dy=3000., ztop=9000., dt=3., run_seconds=6.,
        moist=True, mp_physics=8, km_opt=1, diff_opt=2,
        specified=boundary == 'specified', open_x=boundary == 'open', v_sca_adv_order=vorder)
    shapes = state_array_shapes(cfg)
    specs = batch_moist.workspace_specs(cfg)
    if sources:
        specs += tuple(BatchArraySpec(prefix + name, (cfg.nz, cfg.ny, cfg.nx), 'member')
            for prefix, names in (('transport_physics_', ('qv', 'qc', 'nr')),
                                  ('transport_fixed_', ('qv', 'qs')))
            for name in names)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), cfg.p_surf, cfg.ztop)
    inputs = []
    for member in range(count):
        host = DomainState(cfg, array_module=np)
        host.load_base(coord, base)
        host.mup[...] = np.float32(2. + 0.04 * member)
        host.mup0[...] = np.float32(1.7 + 0.025 * member)
        if mapped:
            j, i = np.indices((cfg.ny, cfg.nx))
            host.set_map_coriolis(msft=1.01 + 0.001 * j + 0.0003 * i,
                msfu=1.02 + np.full((cfg.ny, cfg.nx + 1), 0.001),
                msfv=1.03 + np.full((cfg.ny + 1, cfg.nx), 0.001))
        from gpuwm.core.moist import moist_species
        for name in moist_species(host):
            scale = 1e5 if name in ('nr', 'ni') else 1e-3
            sample = np.arange(host.qv.size).reshape(host.qv.shape)
            value = scale * (0.6 + 0.3 * np.sin(sample + member))
            value[(sample + member) % 5 == 0] = 0
            getattr(host, name)[...] = value
            getattr(host, name + '0')[...] = value
        arrays = {name: getattr(host, name).copy() for name in shapes}
        for spec in specs:
            arrays[spec.name] = np.zeros(spec.shape, spec.dtype)
            if spec.name.startswith(('transport_physics_', 'transport_fixed_')):
                index = np.arange(np.prod(spec.shape)).reshape(spec.shape)
                scale = 10. if spec.name.endswith('nr') else 0.02
                arrays[spec.name][...] = scale * np.cos(index + 0.2 * member)
        controls = {'physics', 'lateral_boundaries', '_scratch', '_scratch_arena', '_host_setup_state', '_phb_host'}
        scalars = {name: value for name, value in vars(host).items() if name not in shapes and name not in controls}
        scratch = {}
        for slot, shape in (('rk_ru_m', shapes['u']), ('rk_rv_m', shapes['v']), ('rk_ww_m', shapes['w'])):
            index = np.arange(np.prod(shape)).reshape(shape)
            scratch[slot] = (1300. * np.sin(index + 0.5 * member)).astype(np.float32)
        scratch['rk_ww_m'][0] = scratch['rk_ww_m'][-1] = 0
        clock = {'ticks': 0, 'step_ticks': 3, 'tick_den': 1, 'run_ticks': 6,
                 'step_count': 0, 'dt_fp32': np.float32(3), 'dtbc_fp32': np.float32(0)}
        inputs.append(PreparedHostMember(cfg, arrays, scalars, clock, scratch=scratch,
                                        phb_host=host._phb_host))
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & shapes.keys()))
    slots = batch_moist.required_scratch_slots(cfg)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=2**30,
        shared_fields=shared, extra_specs=specs, scratch_slots=slots)
    references = [_scalar_state(replace(member, arrays={name: member.arrays[name] for name in shapes}))
                  for member in inputs]
    return batch, references, inputs


def _same(cp, actual, expected, label):
    a, b = cp.asnumpy(actual).view(np.uint32), cp.asnumpy(expected).view(np.uint32)
    different = a != b
    assert not np.any(different), (label, int(np.count_nonzero(different)), np.argwhere(different)[:12].tolist())


@pytest.mark.parametrize('count', (1, 4, 10, 20))
@pytest.mark.parametrize('mapped', (False, True))
@pytest.mark.parametrize('boundary', ('periodic', 'specified', 'open'))
@pytest.mark.parametrize('sources', (False, True))
def test_three_scalar_stages_match_all_original_species_and_live_scratch(count, mapped, boundary, sources, vorder=3):
    import cupy as cp
    from gpuwm.core import moist as original
    from gpuwm.ensemble import batch_moist
    batch, references, inputs = _pack(count, mapped=mapped, boundary=boundary, sources=sources, vorder=vorder)
    names = original.moist_species(batch)
    physics = _Tendencies({name: batch.storage.arrays['transport_physics_' + name]
                          for name in ('qv', 'qc', 'nr')}) if sources else None
    fixed = {name: batch.storage.arrays['transport_fixed_' + name] for name in ('qv', 'qs')} if sources else None
    fluxes = tuple(batch.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
    for stage, (dt_eff, final) in enumerate(((1., False), (1.5, False), (3., True))):
        advance = batch_moist.prepare_scalars_stage(batch, batch.cfg, *fluxes, dt_eff, final,
            physics_tendencies=physics, fixed_tendencies=fixed)
        advance()
        for member, reference in enumerate(references):
            p = _Tendencies({name: cp.asarray(inputs[member].arrays['transport_physics_' + name])
                            for name in ('qv', 'qc', 'nr')}) if sources else None
            f = {name: cp.asarray(inputs[member].arrays['transport_fixed_' + name])
                 for name in ('qv', 'qs')} if sources else None
            scalar_flux = tuple(reference.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
            original.advance_scalars_stage(reference, batch.cfg, *scalar_flux, dt_eff, final,
                physics_tendencies=p, fixed_tendencies=f)
            for name in names:
                _same(cp, batch.member_view(name, member), getattr(reference, name), (stage, member, name))
            for slot, expected in reference._scratch.items():
                if slot.startswith(('moist_', 'pd_')):
                    _same(cp, batch.scratch_member_view(slot, member), expected, (stage, member, slot))


@pytest.mark.parametrize('count', (1, 4))
@pytest.mark.parametrize('boundary', ('periodic', 'specified'))
def test_order_five_scalar_stages_match_all_member_words(count, boundary):
    test_three_scalar_stages_match_all_original_species_and_live_scratch(
        count, True, boundary, True, vorder=5)


@pytest.mark.parametrize('count', (1, 4, 10, 20))
def test_sumflux_zero_accumulate_and_finish_match_original_all_words(count):
    import cupy as cp
    from gpuwm.core.dycore import _prepare_sumflux_launch
    from gpuwm.ensemble.batch_moist import prepare_sumflux_launch
    batch, references, _ = _pack(count, mapped=False, boundary='periodic', sources=False)
    targets = tuple(batch.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
    sources = (batch.u_pp, batch.v_pp, batch.ww_pp)
    for arrays in (sources, (batch.u, batch.v, batch.w)):
        for ordinal, array in enumerate(arrays):
            values = np.arange(array.size, dtype=np.float32).reshape(array.shape)
            array[...] = cp.asarray(np.sin(values + ordinal))
    for member, reference in enumerate(references):
        for name in ('u_pp', 'v_pp', 'ww_pp', 'u', 'v', 'w'):
            getattr(reference, name)[...] = batch.member_view(name, member)
    operations = [('zero_sumflux', (), 0), ('accumulate_sumflux', sources, 0),
                  ('accumulate_sumflux', sources, 0), ('finish_sumflux', (batch.u, batch.v, batch.w), 2)]
    for entry, source, nsub in operations:
        prepare_sumflux_launch(batch, entry, targets, source, nsub)()
        for member, reference in enumerate(references):
            scalar_targets = tuple(reference.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
            scalar_sources = (() if not source else tuple(getattr(reference, name) for name in
                (('u', 'v', 'w') if entry == 'finish_sumflux' else ('u_pp', 'v_pp', 'ww_pp'))))
            _prepare_sumflux_launch(entry, scalar_targets, scalar_sources, nsub)()
            for ordinal, expected in enumerate(scalar_targets):
                _same(cp, targets[ordinal][member], expected, (entry, member, ordinal))


@pytest.mark.parametrize('count', (1, 4, 10, 20))
@pytest.mark.parametrize('mapped', (False, True))
@pytest.mark.parametrize('supply_more', (False, True))
def test_supplied_scalar_tables_preserve_held_and_recomputed_stage_words(count, mapped, supply_more):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core import moist as original
    from gpuwm.ingest import lateral_bc as lateral
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables
    from gpuwm.ensemble.batch_moist import prepare_scalars_stage
    batch, references, _ = _pack(count, mapped=mapped, boundary='specified', sources=False)
    cfg = batch.cfg
    boundaries = []
    for member, reference in enumerate(references):
        snapshot = dict(lateral.domain_boundary_snapshot(reference))
        if supply_more:
            coupled_mass = (reference.c1h[:, None, None]
                            * (reference.mub2d + reference.mup)[None]
                            + reference.c2h[:, None, None])
            for name in ('qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ni'):
                snapshot[name] = cp.asnumpy(coupled_mass * getattr(reference, name)).astype(np.float64)
        after = {name: value * (1. + 0.001 * (member + 1)) + 0.01
                 for name, value in snapshot.items()}
        boundaries.append(lateral.build_lateral_boundaries([snapshot, after], [0., 90.],
            spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone))
    tables = MemberBoundaryTables.from_prepared(boundaries, cfg, array_module=cp, available_bytes=2**30)
    clock = SimpleNamespace(elapsed_seconds=4.25, dt_fp32=np.float32(cfg.dt),
                            dtbc_launch_fp32=np.float32(7.25))
    for member, reference in enumerate(references):
        lateral.attach_lateral_boundaries(reference, boundaries[member])
        reference._lateral_boundary_device.clock = clock
    fluxes = tuple(batch.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
    for stage, (dt_eff, final) in enumerate(((1., False), (1.5, False), (3., True))):
        prepare_scalars_stage(batch, cfg, *fluxes, dt_eff, final, apply_relax=stage == 0,
                              tables=tables, clock=clock)()
        for member, reference in enumerate(references):
            scalar_flux = tuple(reference.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
            original.advance_scalars_stage(reference, cfg, *scalar_flux, dt_eff, final, apply_relax=stage == 0)
            for name in original.moist_species(batch):
                _same(cp, batch.member_view(name, member), getattr(reference, name), ('tables', stage, member, name))
            for slot, expected in reference._scratch.items():
                if slot.startswith(('moist_', 'pd_')) or slot == 'lbc_qv_held':
                    _same(cp, batch.scratch_member_view(slot, member), expected, ('tables', stage, member, slot))


@pytest.mark.parametrize('count', (1, 4, 20))
@pytest.mark.parametrize('field_count', (3, 9))
@pytest.mark.parametrize('width', (1, 2))
def test_flow_dependent_member_perimeters_match_original_raw_words(count, field_count, width):
    import cupy as cp
    from gpuwm.ensemble.batch_moist import prepare_flow_dependent_boundaries
    from gpuwm.ingest.lateral_bc import apply_flow_dependent_boundaries
    batch, references, _ = _pack(count, mapped=False, boundary='specified', sources=False)
    names = ('qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ni', 'qv', 'qc0')[:field_count]
    words = np.array([0, 0x80000000, 0x7FA12345, 0x7FC56789, 0x7F800000,
                      0xFF800000, 0x3F800000, 0xBF800000], dtype=np.uint32)
    for ordinal, name in enumerate(names):
        target = getattr(batch, name)
        index = (np.arange(target.size) + ordinal).reshape(target.shape) % len(words)
        target[...] = cp.asarray(words[index].view(np.float32))
        for member, reference in enumerate(references):
            getattr(reference, name)[...] = batch.member_view(name, member)
    u_flux, v_flux = (batch.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m'))
    prepare_flow_dependent_boundaries(batch, tuple(getattr(batch, name) for name in names),
                                     u_flux, v_flux, width, inflow_value=0.25)()
    for member, reference in enumerate(references):
        apply_flow_dependent_boundaries(tuple(getattr(reference, name) for name in names),
            reference.existing_scratch('rk_ru_m'), reference.existing_scratch('rk_rv_m'),
            width, inflow_value=0.25)
        for name in names:
            _same(cp, batch.member_view(name, member), getattr(reference, name), ('flow', member, name))
