"""Resolve a WPS metgrid directory through the existing namelist translator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from gpuwm.ingest.metem import met_em_series, parse_met_em_name
from gpuwm.netcdf_bridge import open_dataset


@dataclass(frozen=True)
class MetemRun:
    directory: Path
    namelist_input: Path
    paths: dict[int, tuple[Path, ...]]
    metadata: dict[int, object]
    experiment: object
    toml_text: str
    substitution_report: object
    interval_seconds: int
    controls: dict


def _metadata(path):
    """Read layout authority without decoding atmospheric payloads."""
    with open_dataset(path) as ds:
        attrs = dict(ds.global_attributes)
        for name in ('GRID_ID', 'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START',
                     'PARENT_GRID_RATIO'):
            if name not in attrs and name.lower() in attrs:
                attrs[name] = attrs[name.lower()]
        for name in ('GRID_ID', 'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START',
                     'PARENT_GRID_RATIO', 'MAP_PROJ'):
            value = attrs.get(name)
            if value is None or not np.isfinite(value) or float(value) != int(value):
                raise ValueError(f'{path.name}: missing or noninteger layout attribute {name}')
        _, time = parse_met_em_name(path)
        return SimpleNamespace(path=path, global_attributes=attrs,
            nx=len(ds.dimensions['west_east']), ny=len(ds.dimensions['south_north']),
            start_date=time.strftime('%Y-%m-%d_%H:%M:%S'),
            variables={name:tuple(var.shape) for name,var in ds.variables.items()})


def resolve_metem_run(directory, *, rrtmg_variant=None):
    from gpuwm.experiment import build_experiment
    from gpuwm.namelist_import import import_namelists, parse_namelist
    from gpuwm.wrfinput_door import synthesize_wps_namelist, require_preserved_wrf_selectors
    import tomllib

    directory = Path(directory).resolve()
    namelist = directory/'namelist.input'
    if not directory.is_dir() or not namelist.is_file():
        raise ValueError('--met-em DIR requires met_em.d0*.nc and the producing namelist.input in DIR')
    ids = set()
    for path in directory.glob('met_em.d*'):
        domain, _ = parse_met_em_name(path)
        ids.add(int(domain[1:]))
    if not ids or ids != set(range(1, max(ids)+1)):
        raise ValueError('met_em domain files must include d01 and every consecutive child domain')
    paths = {gid:met_em_series(directory, f'd{gid:02d}') for gid in sorted(ids)}
    metadata = {gid:_metadata(files[0]) for gid,files in paths.items()}
    root_attrs = metadata[1].global_attributes
    for gid, item in metadata.items():
        for key in ('MMINLU', 'NUM_LAND_CAT'):
            if key not in item.global_attributes:
                raise ValueError(f'{item.path.name}: missing land-use identity {key}')
        if item.global_attributes['GRID_ID'] != gid:
            raise ValueError(f'{item.path.name}: grid_id disagrees with filename')
    with TemporaryDirectory(prefix='gpuwm-metgrid-') as temporary:
        wps_path = Path(temporary)/'namelist.wps'
        wps_path.write_text(synthesize_wps_namelist(metadata), encoding='utf-8')
        text, report = import_namelists(wps_path, namelist, rrtmg_variant=rrtmg_variant,
            landuse_identity={key:root_attrs[key] for key in ('MMINLU', 'NUM_LAND_CAT')},
            metgrid_initialization=True)
    require_preserved_wrf_selectors(report)
    exp = build_experiment(tomllib.loads(text), source=str(namelist))
    if tuple(domain.grid_id for domain in exp.domains) != tuple(paths):
        raise ValueError('namelist max_dom differs from the met_em domain inventory')
    parsed = parse_namelist(namelist)
    interval = parsed['time_control'].get('interval_seconds', [10800])[0]
    if isinstance(interval, bool) or not np.isfinite(interval) or interval <= 0 or int(interval) != interval:
        raise ValueError('interval_seconds must be a positive integer')
    for domain in exp.domains:
        files = paths[domain.grid_id]
        times = [parse_met_em_name(path)[1] for path in files]
        if not times or times[0] != exp.start_time or (domain.parent_id == 0 and len(times) < 2):
            raise ValueError(f'd{domain.grid_id:02d}: met_em must start at the namelist start and include a later boundary time')
        if any((b-a).total_seconds() != interval for a,b in zip(times,times[1:])):
            raise ValueError(f'd{domain.grid_id:02d}: met_em times must follow interval_seconds={interval}')
        if domain.parent_id == 0 and times[-1] < exp.start_time + timedelta(seconds=exp.run_seconds):
            raise ValueError(f'd{domain.grid_id:02d}: met_em stops at {times[-1]}; supply forcing through the requested end')
        item = metadata[domain.grid_id]
        if (domain.run.nx, domain.run.ny) != (item.nx, item.ny):
            raise ValueError(f'd{domain.grid_id:02d}: namelist dimensions differ from met_em')
    return MetemRun(directory, namelist, paths, metadata, exp, text, report, int(interval), parsed)


def metgrid_initialization_controls(case, run, *, cfg=None):
    """Resolve WRF's actual preparation branches from controls and field flags."""
    from gpuwm.ingest.real import DECLARED_ANALYZED_HYDROMETEORS
    from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    if cfg is not None:
        check_analyzed_scalar_capability(case.attributes, cfg)
    values = run.controls.get('domains', {})
    requested = values.get('sfcp_to_sfcp', [False])
    use_sh = values.get('use_sh_qv', [False])
    for name, value in (('sfcp_to_sfcp', requested), ('use_sh_qv', use_sh)):
        if len(value) != 1 or not isinstance(value[0], bool):
            raise ValueError(f'{name} requires one logical value')
    # WRF real.F:1213 forces sfcprs2 for an ascending (hybrid) input
    # pressure stack. Use the file's actual order, before native sorting.
    ascending = bool(np.all(np.diff(case.snapshot.levels_hpa) > 0))
    sfcp = requested[0] or ascending
    operation = 'sfcprs2' if sfcp else 'sfcprs3'
    flags = ('PSFC', 'SOILHGT') if sfcp else ('PSFC', 'SOILHGT', 'SLP')
    for flag in flags:
        if case.attributes.get('FLAG_'+flag) != 1:
            raise ValueError(f'{case.path.name}: {operation} requires FLAG_{flag}=1')
    if not sfcp and 'PMSL' not in case.snapshot.fields:
        raise ValueError(f'{case.path.name}: sfcprs3 requires the declared PMSL field')
    if 'num_metgrid_levels' in values and values['num_metgrid_levels'] != [case.geometry['num_metgrid_levels']]:
        raise ValueError('num_metgrid_levels differs from the file vertical dimension')
    return {'sfcp_to_sfcp':sfcp, 'use_sh_qv':use_sh[0],
            'analyzed_species':tuple(name for name in DECLARED_ANALYZED_HYDROMETEORS if name in case.snapshot.fields),
            'analyzed_surface_fields':tuple(name for name in DECLARED_ANALYZED_HYDROMETEORS if name in case.snapshot.fields),
            'analyzed_number_fields':tuple(name for name in METGRID_NUMBER_FIELDS if name in case.snapshot.fields)}


def check_analyzed_scalar_capability(attributes, cfg):
    """Refuse only flagged state that the selected active package would lose."""
    from gpuwm.core.nest_fields import nest_field_kinds
    active = set(nest_field_kinds(cfg))
    from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    for name in METGRID_NUMBER_FIELDS:
        if attributes.get('FLAG_'+name, 0) not in (0, 1):
            raise ValueError(f'FLAG_{name} must be 0 or 1')
    aliases = {'QNWFA':('nwfa',), 'QNIFA':('nifa',), 'QNBCA':('nbca',)}
    for name, fields in aliases.items():
        value = attributes.get('FLAG_'+name, 0)
        if value not in (0, 1):
            raise ValueError(f'FLAG_{name} must be 0 or 1')
        if value == 1 and active.intersection(fields):
            raise ValueError(f'FLAG_{name}=1 declares analyzed state used by this microphysics package; native metgrid initialization does not yet interpolate {name}. Run WRF real.exe and use gpuwm run --wrfinput DIR to preserve that state')


def metgrid_analysis_shapes(metadata):
    """The actual fields shared initialize_real uploads, from Rust inventory."""
    attrs, variables = metadata.global_attributes, metadata.variables
    specific = attrs.get('FLAG_SH') == 1
    shapes = {}
    pairs = [('TT','T2'),('GHT',None),('UU','U10'),('VV','V10')]
    pairs += [('SPECHUMD','Q2'),('PRES',None)] if specific else [('RH','RH2')]
    from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    pairs += [(name,name+'_SFC') for name in ('QC','QR','QI','QS','QG','QH') if attrs.get('FLAG_'+name) == 1]
    pairs += [(name,name+'_SFC') for name in METGRID_NUMBER_FIELDS if attrs.get('FLAG_'+name) == 1]
    for name, surface in pairs:
        shape = variables.get(name)
        if shape is None or len(shape) != 4 or shape[0] != 1 or shape[1] < 3:
            raise ValueError(f'{metadata.path.name}: {name} needs Time/level/y/x inventory for native initialization')
        shapes[name] = (shape[1]-1,*shape[2:])
        if surface is not None: shapes[surface] = tuple(shape[2:])
    for name in ('PSFC','SOILHGT', *(('PMSL',) if attrs.get('FLAG_SLP') == 1 else ())):
        shape=variables.get(name)
        if shape != (1,metadata.ny,metadata.nx):
            raise ValueError(f'{metadata.path.name}: {name} requires the mass-grid surface inventory')
        shapes[name] = shape[1:]
    return shapes


def metgrid_memory_admission(run, exp):
    """Cold-device admission prices preparation independently of forecast tiling."""
    from gpuwm.core.preflight import (device_memory_probe_subprocess, device_memory_probe_reason,
        profile_from_device_probe, estimate_phases)
    inventory = {gid:metgrid_analysis_shapes(item) for gid,item in run.metadata.items()}
    for domain in exp.domains:
        check_analyzed_scalar_capability(run.metadata[domain.grid_id].global_attributes, domain.run)
    probe = device_memory_probe_subprocess()
    phases = estimate_phases(exp,source='met_em',forcing_interval_seconds=run.interval_seconds,
        ingest_forcing_interval_seconds=run.interval_seconds,forcing_intervals=len(run.paths[1])-1,
        analysis_shapes_by_domain=inventory,sequential_domains=True,
        profile=profile_from_device_probe(probe))
    free = None if probe is None else int(probe['free_bytes'])
    if free is not None and phases.peak_envelope_bytes > free:
        raise ValueError(f"{phases.verdict(free)}. Reduce the domain/level count and regenerate met_em, or select a GPU with enough memory. Forecast tiling does not reduce native initialization memory")
    if phases.streamed is not None and phases.streamed.host_budget_bytes is not None and phases.streamed.host_bytes > phases.streamed.host_budget_bytes:
        raise ValueError('the streamed forecast host store exceeds available RAM; reduce the requested domain or use a machine with enough host memory')
    return {'analysis_shapes_by_domain':inventory, 'forecast_peak_bytes':phases.forecast_envelope_bytes,
        'preparation_peak_bytes':phases.ingest_envelope_bytes,'available_device_bytes':free,
        'device_probe_note':None if probe is not None else device_memory_probe_reason(),
        'retained_boundary_intervals':len(run.paths[1])-1,'sequential_domain_preparation':True}
