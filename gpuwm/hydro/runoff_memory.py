"""Prepared runoff allocation metadata; no CUDA imports or numerical reads.

Payload SHA declarations bind plans; existing Rust loaders still verify bytes.
Kernel/compiler and generated validation allocations remain explicit gaps.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

from .memory import ArrayMemory, StageMemory

_DTYPES = {'f32': ('float32', 4), 'f64': ('float64', 8), 'i32': ('int32', 4)}


def _seal(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('runoff prepared metadata requires a SHA256 seal')
    return value


def _ids(values):
    if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
        raise ValueError('runoff memory requires unique ordered catchment IDs')
    return list(values)


def _evidence(context, **pins):
    return json.dumps(dict(schema='arwen.hydro.runoff-memory/v1',
                           context=asdict(context), ownership='independent member on selected parent device', **pins), sort_keys=True)


def _stage(settings, components):
    return settings['stages'][len(components)]['name']


def _borrow(settings, components, reference, name, ids, gaps, *, checkpointed=True):
    component, sep, field = reference.partition(':')
    if not sep:
        raise ValueError('runoff input must name component:array')
    source = components.get(component)
    if source is None:
        gaps.append(f'{reference}: prior stage allocation metadata is unavailable')
        return None
    array = next((a for a in source.arrays if a.name == field), None)
    # The series stage publishes eight views of its stable output slab.
    row = next((r for r in settings.get('stages', ()) if r['name'] == component), {})
    if array is None and row.get('factory') == 'gpuwm.hydro.forcing_inputs:prepare_series':
        from .forcing import FIELDS
        slab = next((a for a in source.arrays if a.name == 'fields'), None)
        if slab is not None and field in FIELDS and slab.shape == (8, len(ids)):
            array = ArrayMemory(field, (len(ids),), slab.dtype,
                alias_of=slab.alias_of or f'{component}/fields',
                offset_bytes=slab.offset_bytes + FIELDS.index(field) * len(ids) *
                (8 if slab.dtype == 'float64' else 4))
    if array is None:
        gaps.append(f'{reference}: source field shape/dtype/backing is undeclared')
        return None
    try:
        pins = json.loads(source.evidence)
    except (ValueError, TypeError):
        pins = {}
    if not isinstance(pins, dict) or pins.get('catchment_ids') is None:
        gaps.append(f'{reference}: source ordered catchment axis is undeclared')
    elif pins['catchment_ids'] != ids:
        raise ValueError(f'{reference}: runoff source catchment order differs')
    return ArrayMemory(name, array.shape, array.dtype,
        alias_of=array.alias_of or f'{component}/{array.name}',
        offset_bytes=array.offset_bytes, checkpointed=checkpointed)


def cfe_memory(*, settings, options, context, components):
    from .cfe import FIELD_NAMES, OPTION_NAMES, OFFSET_NAMES, SOURCE_PIN
    raw = Path(options['manifest']).read_bytes()
    packet = json.loads(raw)
    binding = options['identity']
    ids = _ids(binding['catchment_ids'])
    n = len(ids)
    if (packet['schema'] not in ('cfe.initial.pack/v1', 'cfe.capture.pack/v1')
            or tuple(packet['field_names']) != FIELD_NAMES
            or tuple(packet['option_names']) != OPTION_NAMES
            or packet.get('source_pin', SOURCE_PIN) != SOURCE_PIN):
        raise ValueError('CFE memory packet differs from the native schema')
    if packet['catchments'] != ids:
        raise ValueError('CFE memory packet catchment order differs')
    if not binding.get('time_origin'):
        raise ValueError('CFE memory needs the prepared time origin')
    if packet['schema'] == 'cfe.initial.pack/v1' and packet['time_origin'] != binding['time_origin']:
        raise ValueError('CFE memory initial epoch differs')
    period = float(packet['period_seconds'])
    if not math.isfinite(period) or period <= 0:
        raise ValueError('CFE memory needs a positive period')
    expected = {'fields', 'options', 'offsets', *OFFSET_NAMES[:4],
                'smc_profile', 'soil_layer_depths', 'delta_soil_layer_depths'}
    buffers = packet['buffers']
    if len(buffers) != len(expected) or {b['name'] for b in buffers} != expected:
        raise ValueError('CFE memory requires each prepared native buffer once')
    arrays, pins, shapes = [], {}, {}
    for b in buffers:
        dtype, width = _DTYPES['i32' if b['name'] in ('options', 'offsets') else 'f64']
        shape = tuple(b['shape'])
        if (b['dtype'] != ('i32' if dtype == 'int32' else 'f64')
                or any(type(v) is not int or v < 0 for v in shape)
                or b['bytes'] != math.prod(shape)*width):
            raise ValueError('CFE memory buffer metadata differs')
        shapes[b['name']] = shape
        pins[b['name']] = _seal(b['sha256'])
        arrays.append(ArrayMemory(b['name'], shape, dtype, checkpointed=True))
    if (shapes['fields'] != (len(FIELD_NAMES), n)
            or shapes['options'] != (len(OPTION_NAMES), n)
            or shapes['offsets'] != (len(OFFSET_NAMES), n)
            or any(len(shapes[key]) != 1 for key in expected - {'fields', 'options', 'offsets'})
            or not shapes['smc_profile'] == shapes['soil_layer_depths'] == shapes['delta_soil_layer_depths']):
        raise ValueError('CFE memory native buffer extents differ')
    arrays.extend(ArrayMemory(name, (n,), dtype, checkpointed=True, status=name == 'status')
                  for name, dtype in (('status', 'int32'), ('current_time_step', 'int32'),
                     ('time_step_fraction', 'float64'), ('nwm_ponded_depth_m', 'float64')))
    gaps = []
    source = options.get('precipitation_component', options.get('forcing_component'))
    units = options.get('precipitation_units', 'mm/s')
    rain_name = {'mm/s': 'rainrate_mm_s', 'm/s': 'precipitation_m_s', 'mm/h': 'precipitation_mm_h'}[units]
    for reference, name, dtypes in (
        (f"{source}:{options.get('precipitation_field', 'RAINRATE')}", rain_name, ('float32', 'float64')),
        (options['pet_component'] + ':potential_et_m_s', 'potential_et_m_s', ('float64',))):
        a = _borrow(settings, components, reference, name, ids, gaps)
        if a is not None:
            if a.shape != (n,) or a.dtype not in dtypes:
                raise ValueError('CFE memory forcing extent/dtype differs')
            arrays.append(a)
    own = _stage(settings, components)
    arrays.extend(ArrayMemory(name, (n,), 'float64', alias_of=f'{own}/fields',
                  offset_bytes=i*n*8) for i, name in enumerate(FIELD_NAMES))
    if period != context.interval_seconds or settings.get('after_schedule'):
        gaps.append('CFE effective advance schedule needs its resolved period/advance roster')
    return StageMemory(tuple(arrays), _evidence(context, manifest_sha256=hashlib.sha256(raw).hexdigest(),
        payload_sha256=pins, catchment_ids=ids, time_origin=binding['time_origin'],
        period_seconds=period, options=options,
        sources={k: v.evidence for k, v in components.items() if v is not None}), gaps=tuple(gaps))


def pet_memory(*, settings, options, context, components):
    from .pet import PARAMETER_NAMES, _INPUTS
    source = components.get(options['forcing_component'])
    if source is None:
        return None
    try:
        ids = _ids(json.loads(source.evidence)['catchment_ids'])
    except (ValueError, KeyError, TypeError, AttributeError):
        return StageMemory((), _evidence(context, source=source.evidence),
                           gaps=('PET source ordered catchment axis is undeclared',))
    n = len(ids)
    rows = options.get('catchment_parameters')
    configured = [r['parameters'] for r in rows] if rows is not None else [options['parameters']]
    if (any(set(r) != set(PARAMETER_NAMES) for r in configured)
            or rows is not None and [r['catchment_id'] for r in rows] != ids):
        raise ValueError('PET memory parameter roster differs from forcing')
    arrays, gaps = [], []
    for key in _INPUTS:
        a = _borrow(settings, components, options['forcing_component'] + ':' + key,
                    'forcing/' + key, ids, gaps)
        if a is not None:
            if a.shape != (n,) or a.dtype not in ('float32', 'float64'):
                raise ValueError('PET memory forcing shape/dtype differs')
            arrays.append(a)
    arrays.extend((ArrayMemory('parameters', (n, 12), 'float64', checkpointed=True),
        ArrayMemory('methods', (n,), 'int32', checkpointed=True),
        ArrayMemory('potential_et_m_s', (n,), 'float64', checkpointed=True),
        ArrayMemory('diagnostics', (n, 6), 'float64', checkpointed=True),
        ArrayMemory('status', (n,), 'int32', checkpointed=True, status=True),
        ArrayMemory('prepare/parameter_table', (n if rows is not None else 1, 12),
                    'float64', lifetime='prepare'),
        ArrayMemory('prepare/finite_mask', (n, 12), 'bool', lifetime='prepare')))
    gaps.append('PET generated validation/reduction temporary reservation is unpriced')
    return StageMemory(tuple(arrays), _evidence(context, catchment_ids=ids,
        options=options, source=source.evidence), gaps=tuple(gaps))


def _noah_component_inputs(settings, options, context, components, metadata):
    """Borrow packed state, or retain setup inputs while new state is copied."""
    packed = options.get('slabs') is not None
    references = options['slabs' if packed else 'fields']
    required = (set(metadata['slab_fields']) if packed else
                {field['name'] for field in metadata['fields']})
    if set(references) != required:
        raise ValueError('Noah-OWP memory component input roster differs')
    ids = options.get('catchment_ids')
    if ids is None:
        for reference in references.values():
            component, sep, _ = reference.partition(':')
            if not sep:
                raise ValueError('Noah-OWP memory input must name component:array')
            source = components.get(component)
            if source is not None:
                try:
                    ids = json.loads(source.evidence)['catchment_ids']
                    break
                except (ValueError, TypeError, KeyError):
                    pass
    if ids is None:
        return None, (), ('Noah-OWP component input paths need prepared ordered catchment axes',)
    ids = _ids(ids)
    n = len(ids)
    specs = ({tag: ((count, n), _DTYPES[tag][0])
              for tag, count in metadata['slab_fields'].items()} if packed else
             {f['name']: (((n, f['count']) if f['bounds'] else (n,)), _DTYPES[f['dtype']][0])
              for f in metadata['fields']})
    arrays, gaps = [], []
    for key, reference in references.items():
        name = 'slab/' + key if packed else 'prepare/input/' + key
        value = _borrow(settings, components, reference, name, ids, gaps,
                        checkpointed=packed)
        if value is None:
            continue
        if (value.shape, value.dtype) != specs[key]:
            raise ValueError('Noah-OWP memory component input extent/dtype differs')
        if not packed:
            # These arrays are read during setup, then only the packed copies
            # survive in PreparedNoahOWP. Upstream storage stays independently
            # retained by its owning stage.
            from dataclasses import replace
            value = replace(value, lifetime='prepare')
        arrays.append(value)
    return ids, tuple(arrays), tuple(gaps)


def noahowp_memory(*, settings, options, context, components):
    from .noahowp.runtime import schema
    metadata = schema()
    if sum(options.get(key) is not None for key in ('initial_manifest', 'slabs', 'fields')) != 1:
        raise ValueError('Noah-OWP memory requires exactly one initial input path')
    schema_raw = Path(__file__).with_name('noahowp').joinpath('schema.json').read_bytes()
    arrays, pins, gaps = [], {}, []
    manifest_sha = None
    if options.get('initial_manifest') is not None:
        raw = Path(options['initial_manifest']).read_bytes()
        manifest_sha = hashlib.sha256(raw).hexdigest()
        packet = json.loads(raw)
        if (packet.get('schema') != 'arwen.noahowp.initial.v1'
            or packet.get('source_revision') != metadata['source_revision']
            or packet.get('schema_sha256') != hashlib.sha256(schema_raw).hexdigest()):
            raise ValueError('Noah-OWP memory packet schema/source differs')
        ids = _ids([c['id'] for c in packet['columns']])
        n = len(ids)
        if options.get('catchment_ids', ids) != ids:
            raise ValueError('Noah-OWP memory catchment order differs')
        parameter_sha = _seal(packet['parameter_sha256'])
        if options.get('parameter_sha256', parameter_sha) != parameter_sha:
            raise ValueError('Noah-OWP memory parameter identity differs')
        for tag, count in metadata['slab_fields'].items():
            b = packet['files'][tag]
            dtype, width = _DTYPES[tag]
            if b['shape'] != [count, n] or b['bytes'] != count*n*width or b['dtype'] != tag:
                raise ValueError('Noah-OWP memory slab extent/dtype differs')
            pins[tag] = _seal(b['sha256'])
            arrays.append(ArrayMemory('slab/' + tag, (count, n), dtype, checkpointed=True))
    else:
        ids, inputs, missing = _noah_component_inputs(settings, options, context, components, metadata)
        if ids is None:
            return StageMemory((), _evidence(context, options=options), gaps=missing)
        n = len(ids)
        parameter_sha = _seal(options['parameter_sha256'])
        arrays.extend(inputs)
        gaps.extend(missing)
        if options.get('fields') is not None:
            arrays.extend(ArrayMemory('slab/' + tag, (count, n), _DTYPES[tag][0], checkpointed=True)
                          for tag, count in metadata['slab_fields'].items())
    _seal(options['fabric_sha256'])
    from datetime import datetime
    datetime.fromisoformat(options['time_origin'])
    own = _stage(settings, components)
    slabs = {a.name: a for a in arrays if a.name.startswith('slab/')}
    for f in metadata['fields']:
        slab = slabs.get('slab/' + f['dtype'])
        if slab is None:
            # Missing provider metadata must not invent a new packed backing.
            continue
        dtype, width = _DTYPES[f['dtype']]
        shape = (n, f['count']) if f['bounds'] else (n,)
        arrays.append(ArrayMemory(f['name'], shape, dtype,
            alias_of=slab.alias_of or f"{own}/{slab.name}",
            offset_bytes=slab.offset_bytes + f['offset']*n*width,
            checkpointed=True))
    arrays.extend((ArrayMemory('status', (n,), 'int32', checkpointed=True),
        ArrayMemory('sticky_status', (n,), 'int32', checkpointed=True, status=True, status_name='lsm'),
        ArrayMemory('workspace/bmi_converted', (3, n), 'float32'),
        ArrayMemory('prepare/dt_compare', (n,), 'bool', lifetime='prepare'),
        ArrayMemory('prepare/epoch_compare', (n,), 'bool', lifetime='prepare')))
    gaps.append('Noah-OWP generated validation/reduction reservation is unpriced')
    if 'forcing_component' in options:
        from .forcing import FIELDS
        for field in FIELDS:
            a = _borrow(settings, components, options['forcing_component'] + ':' + field,
                        'forcing_input/' + field, ids, gaps)
            if a is not None:
                if a.shape != (n,) or a.dtype != 'float32':
                    raise ValueError('Noah-OWP memory requires float32 forcing vectors')
                arrays.append(a)
    return StageMemory(tuple(arrays), _evidence(context, manifest_sha256=manifest_sha,
        payload_sha256=pins, schema_sha256=hashlib.sha256(schema_raw).hexdigest(),
        parameter_sha256=parameter_sha, catchment_ids=ids, options=options,
        sources={k: v.evidence for k, v in components.items() if v is not None}), gaps=tuple(gaps))


def _axes(source, context, options):
    """Axes are an owner contract, never inferred from a payload's values."""
    try:
        pins = json.loads(source.evidence)
        ids = _ids(pins['catchment_ids'])
        nodes = _ids(pins['node_ids'])
    except (ValueError, KeyError, TypeError):
        return None, StageMemory((), _evidence(context, options=options,
            source=source.evidence), gaps=('runoff network ordered catchment/node axes are undeclared',))
    if max(len(ids), len(nodes)) >= 2147483647:
        raise ValueError('runoff memory axes must fit positive int32')
    return (ids, nodes), None


def areas_memory(*, settings, options, context, components):
    source = components.get(options['network_component'])
    if source is None:
        return StageMemory((), _evidence(context, options=options), gaps=(
            'runoff area provider needs the prepared network allocation inventory',))
    axes, missing = _axes(source, context, options)
    if missing is not None:
        return missing
    ids, nodes = axes
    gaps = []
    area = _borrow(settings, components,
        options['network_component'] + ':' + options.get('array', 'catchment_areasqkm'),
        'areasqkm', ids, gaps, checkpointed=False)
    if area is not None and (area.shape != (len(ids),) or area.dtype != 'float64'):
        raise ValueError('runoff area provider requires a float64 catchment vector')
    # This prepare-only resource retains a view, allocates nothing and has no kernel.
    return StageMemory(() if area is None else (area,), _evidence(context,
        catchment_ids=ids, node_ids=nodes, units='km2', options=options,
        source=source.evidence), non_pool_bytes=0, gaps=tuple(gaps))


def routing_memory(*, settings, options, context, components):
    network = components.get(options['network_component'])
    if network is None:
        return StageMemory((), _evidence(context, options=options), gaps=(
            'runoff gather needs the prepared network allocation inventory',))
    axes, missing = _axes(network, context, options)
    if missing is not None:
        return missing
    ids, nodes = axes
    n, m = len(ids), len(nodes)
    if options['area_units'] not in ('m2', 'km2'):
        raise ValueError('runoff area units must explicitly be m2 or km2')
    arrays, gaps = [], []
    for reference, name, dtypes in (
            (options['source_component'] + ':' + options['depth_field'], 'depth', ('float32', 'float64')),
            (options['area_component'] + ':' + options['area_array'], 'area', ('float64',)),
            (options['network_component'] + ':catchment_to_node', 'catchment_to_node', ('int32',))):
        array = _borrow(settings, components, reference, name, ids, gaps)
        if array is not None:
            if array.shape != (n,) or array.dtype not in dtypes:
                raise ValueError('runoff gather input extent/dtype differs')
            arrays.append(array)
    arrays.extend((ArrayMemory('offsets', (m + 1,), 'int64', checkpointed=True),
        ArrayMemory('indices', (n,), 'int32', checkpointed=True),
        ArrayMemory('routing_lateral', (m,), 'float32', checkpointed=True),
        ArrayMemory('routing_discharge_m3_s', (m,), 'float64', checkpointed=True),
        ArrayMemory('routing_volume_m3', (m,), 'float64', checkpointed=True),
        ArrayMemory('status', (m,), 'int32', checkpointed=True, status=True, status_name='runoff'),
        ArrayMemory('prepare/arange', (n,), 'int32', lifetime='prepare'),
        ArrayMemory('prepare/keys', (2, n), 'int32', lifetime='prepare'),
        ArrayMemory('prepare/lexsort_indices', (n,), 'int64', lifetime='prepare'),
        ArrayMemory('prepare/bincount', (m,), 'int64', lifetime='prepare'),
        ArrayMemory('prepare/validation_masks', (7, n), 'bool', lifetime='prepare'),
        ArrayMemory('prepare/validation_results', (2,), 'bool', lifetime='prepare')))
    source = components.get(options['source_component'])
    try:
        period = json.loads(source.evidence)['period_seconds']
    except (AttributeError, ValueError, KeyError, TypeError):
        period = None
    if period is None:
        gaps.append('runoff producer period is undeclared')
    elif type(period) not in (int, float) or not math.isfinite(period) or period <= 0:
        raise ValueError('runoff producer period must be finite and positive')
    if period != context.interval_seconds or settings.get('after_schedule'):
        gaps.append('runoff effective advance schedule needs its resolved producer period/advance roster')
    gaps.append('runoff lexsort/bincount/cumsum/validation internal workspace reservation is unpriced')
    return StageMemory(tuple(arrays), _evidence(context, catchment_ids=ids, node_ids=nodes,
        period_seconds=period, options=options,
        kernel_sha256=hashlib.sha256(Path(__file__).with_name('runoff.cu').read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(__file__).with_name('runoff.py').read_bytes()).hexdigest(),
        sources={k: v.evidence for k, v in components.items() if v is not None}), gaps=tuple(gaps))
