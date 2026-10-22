"""Runoff allocation contracts only; no fake hydrologic numerical models."""
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm.hydro.memory import ArrayMemory, HydroMemoryPlan, MemoryContext, StageMemory
from gpuwm.hydro import cfe, pet
from gpuwm.hydro.noahowp import factory
from gpuwm.hydro.noahowp.runtime import schema


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'cupy', None)  # planning must never allocate/compile/read values
    ids = ['cat-b', 'cat-a']
    context = MemoryContext('standalone', 'strict', 3600, duration_seconds=86400)
    forcing = StageMemory((ArrayMemory('fields', (8, 2), 'float32', checkpointed=True),),
        json.dumps(dict(catchment_ids=ids, time_origin='2023-01-01T00:00:00Z')))
    settings = {'stages': [dict(name='forcing', factory='gpuwm.hydro.forcing_inputs:prepare_series'),
                          dict(name='pet', factory='gpuwm.hydro.pet:prepare'),
                          dict(name='runoff', factory='gpuwm.hydro.cfe:prepare')]}
    return tmp_path, ids, context, forcing, settings


def pet_options():
    return dict(forcing_component='forcing', parameters={key: 1 for key in pet.PARAMETER_NAMES}, method=5)


def cfe_packet(path, ids):
    vectors = {*cfe.OFFSET_NAMES[:4], 'smc_profile', 'soil_layer_depths', 'delta_soil_layer_depths'}
    shapes = {'fields': [len(cfe.FIELD_NAMES), len(ids)],
              'options': [len(cfe.OPTION_NAMES), len(ids)],
              'offsets': [len(cfe.OFFSET_NAMES), len(ids)], **{key: [3] for key in vectors}}
    packet = dict(schema='cfe.initial.pack/v1', source_pin=cfe.SOURCE_PIN,
        field_names=cfe.FIELD_NAMES, option_names=cfe.OPTION_NAMES,
        catchments=ids, time_origin='2023-01-01T00:00:00Z', period_seconds=3600,
        buffers=[dict(name=key, shape=value, dtype='i32' if key in ('options', 'offsets') else 'f64',
            bytes=__import__('math').prod(value)*(4 if key in ('options', 'offsets') else 8),
            sha256='a'*64, path='payload-not-present') for key, value in shapes.items()])
    path.write_text(json.dumps(packet))
    return packet


def cfe_options(path, ids):
    return dict(manifest=str(path), forcing_component='forcing', pet_component='pet',
                identity=dict(catchment_ids=ids, time_origin='2023-01-01T00:00:00Z'),
                initial_step=0, time_step_fraction=1)


def test_cfe_pet_full_prepared_rosters_and_independent_members(prepared):
    path, ids, context, forcing, settings = prepared
    pet_plan = pet.prepare_memory(settings=settings, options=pet_options(), context=context,
                                  components={'forcing': forcing})
    manifest = path/'cfe.json'
    cfe_packet(manifest, ids)
    runoff = cfe.prepare_memory(settings=settings, options=cfe_options(manifest, ids),
                               context=context, components={'forcing': forcing, 'pet': pet_plan})
    stages = (('forcing', forcing), ('pet', pet_plan), ('runoff', runoff))
    one = HydroMemoryPlan(context, stages, ('unpriced compiled kernels',), 'settings').inventory()
    three = HydroMemoryPlan(context, stages, ('unpriced compiled kernels',), 'settings').inventory(members=3)
    assert three['required_bytes'] == one['required_bytes']*3
    assert not one['complete']
    arrays = {a.name: a for a in runoff.arrays}
    assert arrays['rainrate_mm_s'].alias_of == 'forcing/fields'
    assert arrays['rainrate_mm_s'].offset_bytes == 7*2*4
    assert arrays['potential_et_m_s'].alias_of == 'pet/potential_et_m_s'
    assert arrays['flux_Qout_m'].alias_of == 'runoff/fields'
    assert not arrays['flux_Qout_m'].checkpointed
    assert arrays['status'].checkpointed and arrays['status'].status
    assert runoff.non_pool_bytes is None and pet_plan.non_pool_bytes is None
    assert 'prepare/parameter_table' in {a.name for a in pet_plan.arrays}
    assert json.loads(runoff.evidence)['payload_sha256']['fields'] == 'a'*64


@pytest.mark.parametrize('defect', ['axis', 'bytes', 'seal', 'epoch'])
def test_cfe_refuses_inconsistent_prepared_metadata(prepared, defect):
    path, ids, context, forcing, settings = prepared
    manifest = path/'cfe.json'
    packet = cfe_packet(manifest, ids)
    if defect == 'axis': packet['catchments'] = ids[::-1]
    if defect == 'bytes': packet['buffers'][0]['bytes'] += 8
    if defect == 'seal': packet['buffers'][0]['sha256'] = 'unsealed'
    if defect == 'epoch': packet['time_origin'] = '2022-12-31T00:00:00Z'
    manifest.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        cfe.prepare_memory(settings=settings, options=cfe_options(manifest, ids),
                           context=context, components={'forcing': forcing, 'pet': None})


def test_missing_source_remains_partial_without_guessed_allocation(prepared):
    path, ids, context, forcing, settings = prepared
    manifest = path/'cfe.json'
    cfe_packet(manifest, ids)
    runoff = cfe.prepare_memory(settings=settings, options=cfe_options(manifest, ids),
        context=context, components={'forcing': None, 'pet': None})
    assert len(runoff.gaps) == 2
    assert 'rainrate_mm_s' not in {a.name for a in runoff.arrays}
    assert 'potential_et_m_s' not in {a.name for a in runoff.arrays}


def test_source_axis_mismatch_is_refused(prepared):
    path, ids, context, forcing, settings = prepared
    manifest = path/'cfe.json'
    cfe_packet(manifest, ids)
    wrong = StageMemory(forcing.arrays, json.dumps(dict(catchment_ids=ids[::-1])))
    with pytest.raises(ValueError, match='order'):
        cfe.prepare_memory(settings=settings, options=cfe_options(manifest, ids), context=context,
                           components={'forcing': wrong, 'pet': None})


def noah_packet(path, ids):
    metadata = schema()
    raw = Path(factory.__file__).with_name('schema.json').read_bytes()
    packet = dict(schema='arwen.noahowp.initial.v1', source_revision=metadata['source_revision'],
        schema_sha256=hashlib.sha256(raw).hexdigest(), parameter_sha256='b'*64,
        columns=[dict(id=i) for i in ids], files={})
    for tag, count in metadata['slab_fields'].items():
        packet['files'][tag] = dict(dtype=tag, shape=[count, len(ids)],
            bytes=count*len(ids)*(8 if tag == 'f64' else 4), sha256='c'*64, path='not-read')
    path.write_text(json.dumps(packet))
    return packet


def test_noah_typed_slabs_views_forcing_and_hidden_bmi_workspace(prepared):
    path, ids, context, forcing, settings = prepared
    manifest = path/'noah.json'
    noah_packet(manifest, ids)
    settings['stages'] = settings['stages'][:1] + [dict(name='lsm', factory='gpuwm.hydro.noahowp.factory:prepare_slab')]
    value = factory.prepare_slab_memory(settings=settings, options=dict(initial_manifest=str(manifest),
        forcing_component='forcing', time_origin='2023-01-01T00:00:00Z', fabric_sha256='d'*64),
        context=context, components={'forcing': forcing})
    row = HydroMemoryPlan(context, (('forcing', forcing), ('lsm', value)), ('resources',), 's').inventory()
    arrays = {a.name: a for a in value.arrays}
    assert arrays['domain.dt'].alias_of == 'lsm/slab/f32'
    assert arrays['domain.dt'].checkpointed
    assert arrays['workspace/bmi_converted'].shape == (3, len(ids))
    assert not arrays['workspace/bmi_converted'].checkpointed
    assert arrays['sticky_status'].status and not arrays['status'].status
    assert arrays['sticky_status'].status_name == 'lsm'
    assert next(a for a in row['arrays'] if a['name'] == 'lsm/sticky_status')['status_name'] == 'lsm/lsm'
    assert not any('distinct status-name binding' in gap for gap in value.gaps)
    assert len([key for key in row['backings'] if key.startswith('lsm/slab/')]) == 3
    assert len([a for a in value.arrays if a.name.startswith('forcing_input/')]) == 8
    assert json.loads(value.evidence)['catchment_ids'] == ids


def test_noah_opaque_alternate_inputs_are_named_gap(prepared):
    _, _, context, _, settings = prepared
    value = factory.prepare_slab_memory(settings=settings, options=dict(
        slabs={tag: 'source:' + tag for tag in schema()['slab_fields']}),
        context=context, components={})
    assert not value.arrays and value.non_pool_bytes is None
    assert 'input paths' in value.gaps[0]


@pytest.fixture
def noah_component(prepared):
    _, ids, context, _, _ = prepared
    metadata = schema()
    typed = {'f32': 'float32', 'i32': 'int32', 'f64': 'float64'}
    source_arrays = tuple(ArrayMemory(tag, (count, len(ids)), typed[tag])
                          for tag, count in metadata['slab_fields'].items())
    # The supplied fields are views into the provider's actual typed slabs.
    source_arrays += tuple(ArrayMemory(f['name'],
        (len(ids), f['count']) if f['bounds'] else (len(ids),), typed[f['dtype']],
        alias_of='input/' + f['dtype'], offset_bytes=f['offset']*len(ids)*(8 if f['dtype'] == 'f64' else 4))
        for f in metadata['fields'])
    source = StageMemory(source_arrays, json.dumps(dict(catchment_ids=ids, source_sha256='1'*64)))
    settings = dict(stages=[dict(name='input'), dict(name='noah')])
    options = dict(parameter_sha256='b'*64, fabric_sha256='d'*64,
                   time_origin='2023-01-01T00:00:00Z')
    return ids, context, source, settings, options


@pytest.mark.parametrize('kind', ['slabs', 'fields'])
def test_noah_component_paths_price_actual_backings_and_copy_lifetimes(noah_component, kind):
    ids, context, source, settings, options = noah_component
    roster = schema()['slab_fields'] if kind == 'slabs' else [f['name'] for f in schema()['fields']]
    options[kind] = {key: 'input:' + key for key in roster}
    result = factory.prepare_slab_memory(settings=settings, options=options, context=context,
                                         components={'input': source})
    arrays = {a.name: a for a in result.arrays}
    packed = kind == 'slabs'
    for tag in schema()['slab_fields']:
        assert arrays['slab/' + tag].alias_of == ('input/' + tag if packed else None)
        assert arrays['slab/' + tag].checkpointed
    assert arrays['domain.dt'].alias_of == ('input/f32' if packed else 'noah/slab/f32')
    setup = [a for a in result.arrays if a.name.startswith('prepare/input/')]
    assert len(setup) == (0 if packed else len(schema()['fields']))
    assert all(a.lifetime == 'prepare' and not a.checkpointed for a in setup)
    assert not arrays['workspace/bmi_converted'].checkpointed
    inventory = HydroMemoryPlan(context, (('input', source), ('noah', result)),
                               result.gaps, 's').inventory()
    assert sum(k.startswith('noah/slab/') for k in inventory['backings']) == (0 if packed else 3)
    assert sum(a.checkpointed for a in result.arrays) == len(schema()['fields']) + 5
    evidence = json.loads(result.evidence)
    assert evidence['catchment_ids'] == ids and evidence['manifest_sha256'] is None
    assert '1'*64 in evidence['sources']['input']
    assert not inventory['complete'] and result.non_pool_bytes is None


@pytest.mark.parametrize('defect', ['shape', 'roster', 'parameter'])
def test_noah_component_metadata_mismatches_are_refused(noah_component, defect):
    from dataclasses import replace
    ids, context, source, settings, options = noah_component
    options['slabs'] = {tag: 'input:' + tag for tag in schema()['slab_fields']}
    if defect == 'shape':
        source = replace(source, arrays=tuple(replace(a, shape=(1, 2)) if a.name == 'f32' else a for a in source.arrays))
    if defect == 'roster': del options['slabs']['i32']
    if defect == 'parameter': options['parameter_sha256'] = 'not-sealed'
    with pytest.raises(ValueError):
        factory.prepare_slab_memory(settings=settings, options=options, context=context,
                                   components={'input': source})


def test_noah_missing_packed_provider_never_invents_an_allocation(noah_component):
    ids, context, source, settings, options = noah_component
    options.update(catchment_ids=ids, slabs={tag: 'absent:' + tag for tag in schema()['slab_fields']})
    result = factory.prepare_slab_memory(settings=settings, options=options, context=context,
                                         components={'absent': None})
    assert not any(a.name.startswith('slab/') for a in result.arrays)
    assert not any(a.name.startswith('domain.') for a in result.arrays)
    assert sum('absent:' in gap for gap in result.gaps) == 3


@pytest.fixture
def gather(prepared):
    from gpuwm.hydro import runoff
    _, ids, context, _, _ = prepared
    settings = {'stages': [dict(name='network'), dict(name='source'), dict(name='areas'),
                            dict(name='gather', factory='gpuwm.hydro.runoff:prepare')]}
    network = StageMemory((ArrayMemory('catchment_areasqkm', (2,), 'float64'),
        ArrayMemory('catchment_to_node', (2,), 'int32')),
        json.dumps(dict(catchment_ids=ids, node_ids=['r-1', 'r-2', 'r-3'])))
    source = StageMemory((ArrayMemory('fields', (4, 2), 'float64', checkpointed=True),
        ArrayMemory('depth', (2,), 'float64', alias_of='source/fields', offset_bytes=16)),
        json.dumps(dict(catchment_ids=ids, period_seconds=3600)))
    components = dict(network=network, source=source)
    area = runoff.prepare_areas_memory(settings=settings, options=dict(network_component='network'),
                                      context=context, components=components)
    components['areas'] = area
    options = dict(source_component='source', depth_field='depth', network_component='network',
                   area_component='areas', area_array='areasqkm', area_units='km2')
    return runoff, settings, context, components, options


def test_gather_aliases_checkpoint_roster_and_setup_reservation(gather):
    module, settings, context, components, options = gather
    value = module.prepare_memory(settings=settings, options=options, context=context, components=components)
    arrays = {a.name: a for a in value.arrays}
    assert arrays['depth'].alias_of == 'source/fields' and arrays['depth'].offset_bytes == 16
    assert arrays['area'].alias_of == 'network/catchment_areasqkm'
    assert arrays['catchment_to_node'].alias_of == 'network/catchment_to_node'
    assert arrays['offsets'].shape == (4,)
    assert {a.name for a in value.arrays if a.checkpointed} == {
        'depth', 'area', 'catchment_to_node', 'offsets', 'indices', 'routing_lateral',
        'routing_discharge_m3_s', 'routing_volume_m3', 'status'}
    assert sum(a.alias_of is None and a.lifetime == 'resident' for a in value.arrays) == 6
    assert len([a for a in value.arrays if a.lifetime == 'prepare']) == 6
    area = components['areas']
    assert area.non_pool_bytes == 0 and not area.arrays[0].checkpointed
    assert value.non_pool_bytes is None
    assert any('internal workspace' in gap for gap in value.gaps)
    assert not any('distinct status-name binding' in gap for gap in value.gaps)
    plan = HydroMemoryPlan(context, (*components.items(), ('gather', value)), value.gaps, 's')
    one, three = plan.inventory(), plan.inventory(members=3)
    assert three['required_bytes'] == 3*one['required_bytes']
    assert not one['complete']
    assert next(a for a in one['arrays'] if a['name'] == 'gather/status')['status_name'] == 'gather/runoff'
    assert len(one['backings']) == 15  # 3 provider roots + 6 retained + 6 setup
    evidence = json.loads(value.evidence)
    assert evidence['period_seconds'] == 3600
    assert len(evidence['kernel_sha256']) == 64


@pytest.mark.parametrize('defect', ['map_dtype', 'source_period'])
def test_gather_refuses_inconsistent_metadata(gather, defect):
    module, settings, context, components, options = gather
    key = {'source_axis': 'source', 'area_axis': 'areas', 'map_dtype': 'network',
           'source_period': 'source', 'area_shape': 'areas'}[defect]
    original = components[key]
    evidence = json.loads(original.evidence)
    arrays = original.arrays
    if defect == 'source_period': evidence['period_seconds'] = -1
    if defect == 'map_dtype':
        arrays = (arrays[0], ArrayMemory('catchment_to_node', (2,), 'int64'))
    components[key] = StageMemory(arrays, json.dumps(evidence))
    with pytest.raises(ValueError):
        module.prepare_memory(settings=settings, options=options, context=context, components=components)


def test_gather_missing_axes_dependencies_and_schedule_stay_partial(gather):
    module, settings, context, components, options = gather
    components['network'] = None
    missing = module.prepare_memory(settings=settings, options=options, context=context, components=components)
    assert not missing.arrays and 'network allocation inventory' in missing.gaps[0]
    assert not module.prepare_areas_memory(settings=settings, options=options,
        context=context, components=components).arrays
    components['network'] = StageMemory((), '{"catchment_ids": ["a"]}')
    missing = module.prepare_memory(settings=settings, options=options, context=context, components=components)
    assert not missing.arrays and 'axes' in missing.gaps[0]
