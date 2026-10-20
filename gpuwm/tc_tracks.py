"""Source-neutral tropical cyclone tracks from completed fields.

Atmospheric reads, unit conversion and interpolation run in rw_tc_fields.
Earth2Studio runs only in GPUWM_EARTH2_PYTHON, never in the CLI process.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TABLE = Path(__file__).with_name('authorities') / 'tc-track-fields.v1.json'
SCHEMA = 'gpuwm.tc-tracks.v1'
MANIFEST = 'gpuwm.tc-tracks-manifest.v1'
COLUMNS = 'member_id,track_id,weight,valid,lead_hours,lat,lon,mslp_pa,max_wind10_ms,flags'.split(',')


def utc(value):
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError(f'time needs an explicit UTC offset: {value}')
    return d.astimezone(timezone.utc)


def stamp(d):
    return d.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for p in files:
        if path.is_dir():
            h.update(str(p.relative_to(path)).encode() + b'\0')
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binary(name):
    """Resolve native tracker tools through the engine's normal artifact ladder."""
    from gpuwm import bridges, rustwx
    env = 'GPUWM_' + name.upper()
    candidate = (rustwx.find_renderer() if name == 'rw_wrfbatch' else
                 bridges.find_artifact(env, bridges.executable_name(name),
                                       crate_relative='tools/rustwx'))
    if candidate is None:
        path = shutil.which(name)
        candidate = bridges.accept_resolved(Path(path)) if path else None
    if candidate is None:
        raise ValueError(f'{name} is required; build tools/rustwx or install engine bridges '
                         f'({env} remains an optional override)')
    return str(candidate)


def run(cmd, **kw):
    return subprocess.run([str(x) for x in cmd], check=True, text=True, **kw)


def inventory(path):
    return json.loads(run([binary('rw_tc_fields'), 'inspect', path], capture_output=True).stdout)


def cf_seconds(values, units, absolute):
    unit, sep, origin = units.partition(' since ')
    scale = {'seconds': 1, 'second': 1, 's': 1, 'hours': 3600, 'hour': 3600,
             'days': 86400, 'minutes': 60, 'nanoseconds': 1e-9}.get(unit)
    if scale is None:
        raise ValueError(f'unsupported CF time units: {units}')
    base = 0
    if absolute:
        if not sep:
            raise ValueError(f'absolute time has no CF epoch: {units}')
        text = origin.strip().replace(' UTC', '+00:00')
        d = datetime.fromisoformat(text.replace('Z', '+00:00'))
        base = d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp()
    return [base + float(v) * scale for v in values]


def producer_init(metadata):
    """A producer's init_time attribute. CF and our exporters write UTC; a value with no offset is UTC
    (ml-export of ArWen history writes 2024-09-24T12:00:00), never local time."""
    value = metadata.get('attributes', {}).get('init_time')
    if not value:
        return None
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return stamp(d if d.tzinfo else d.replace(tzinfo=timezone.utc))


def expand_file(member, path):
    """Expand explicit initialization and ensemble axes without collapsing identity."""
    metadata = inventory(path)
    inv = metadata['variables']
    t = inv['time']
    times = cf_seconds(t['values'], t['attrs']['units'], True)
    leads_name = next((n for n in ('lead_time', 'prediction_timedelta')
                       if n in inv and inv[n]['dims'] != t['dims']), None)
    leads = cf_seconds(inv[leads_name]['values'], inv[leads_name]['attrs']['units'], False) if leads_name else [0]
    lead_dim = inv[leads_name]['dims'][0] if leads_name else None
    ensemble_arrays = [v for v in inv.values() if 'ensemble' in v['dims']]
    nens = ensemble_arrays[0]['shape'][ensemble_arrays[0]['dims'].index('ensemble')] if ensemble_arrays else 1
    has_ensemble = bool(ensemble_arrays)
    out = []
    for ti, init in enumerate(times if leads_name else times[:1]):
        for ei in range(nens):
            m = dict(member)
            suffix = (f'-init{ti}' if leads_name and len(times) > 1 else '') + (f'-ens{ei:03d}' if has_ensemble else '')
            m['member_id'] = str(member['member_id']) + suffix
            m['init'] = member.get('init') or producer_init(metadata) or stamp(datetime.fromtimestamp(init, timezone.utc))
            m['ancestry'] = member.get('ancestry', {})
            frames = []
            for fi, offset in enumerate(leads if leads_name else times):
                select = {t['dims'][0]: ti if leads_name else fi}
                if lead_dim:
                    select[lead_dim] = fi
                if has_ensemble:
                    select['ensemble'] = ei
                valid = init + offset if leads_name else offset
                frames.append({'path': str(path), 'valid': stamp(datetime.fromtimestamp(valid, timezone.utc)), 'select': select})
            m['frames'] = frames
            if 'weight' in m and suffix:
                m['weight'] /= nens * (len(times) if leads_name else 1)
            m.pop('file', None)
            out.append(m)
    return out


def normalize_manifest(doc, root=Path('.')):
    if doc.get('schema') != MANIFEST:
        raise ValueError(f'expected {MANIFEST}')
    members = []
    for original in doc.get('members', []):
        m = dict(original)
        for key in ('member_id', 'family', 'source'):
            if not m.get(key):
                raise ValueError(f'member needs {key}')
        m['member_id'] = str(m['member_id'])
        if m['family'] not in ('ai', 'arwen', 'hex', 'analysis'):
            raise ValueError(f'unknown family {m["family"]}')
        m.setdefault('ancestry', {})
        if 'file' in m:
            p = (root / m['file']).resolve()
            members.extend(expand_file(m, p))
        else:
            if not m.get('frames') or not m.get('init'):
                raise ValueError('member needs file or frames plus init')
            m['frames'] = [dict(f, path=str((root / f['path']).resolve()), valid=stamp(utc(f['valid']))) for f in m['frames']]
            members.append(m)
    if not members or len({m['member_id'] for m in members}) != len(members):
        raise ValueError('manifest needs nonempty, unique member IDs')
    families = {f: [m for m in members if m['family'] == f] for f in {m['family'] for m in members}}
    for family, roster in families.items():
        present = [m for m in roster if 'weight' in m]
        if present and len(present) != len(roster):
            raise ValueError(f'{family}: supply all weights or none within a family')
        for m in roster:
            m.setdefault('weight', 1 / len(roster))
            if not math.isfinite(m['weight']) or m['weight'] < 0:
                raise ValueError('weights must be finite and nonnegative')
            m['init'] = stamp(utc(m['init']))
            m['frames'].sort(key=lambda f: f['valid'])
            valid = [f['valid'] for f in m['frames']]
            if len(valid) != len(set(valid)):
                raise ValueError(f'{m["member_id"]}: duplicate valid times')
            if any(utc(v) < utc(m['init']) for v in valid):
                raise ValueError('valid time precedes initialization')
    return dict(schema=MANIFEST, members=members,
                weight_policy='Defaults sum to 1 within each family. No combined cross-family probability is implied.')


def shortcut(path, work):
    path = Path(path).resolve()
    if path.is_dir() and not (path / '.zgroup').exists():
        stores = sorted(path.glob('*.zarr'))
        if not stores:
            export = work / 'ml-export'
            run([sys.executable, '-m', 'gpuwm', 'ml-export', path, '--out', export,
                 '--domains', 'd01', '--levels', '850,500,400,300,250,200', '--grid', 'latlon:0.25',
                 '--names', 'era5', '--variables', 'geopotential,temperature,u_component_of_wind,v_component_of_wind,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure', '--threads', str(os.cpu_count() or 1)])
            stores = sorted(export.glob('*.zarr'))
        if len(stores) != 1:
            raise ValueError('--input run needs one domain; use a manifest to name domains explicitly')
        path = stores[0]
    meta = inventory(path)
    if meta.get('format') == 'grib2':
        return normalize_manifest(dict(schema=MANIFEST, members=[dict(member_id='member-000', family='arwen',
            source='arwen-grib2', init=meta['init'], vocabulary='arwen-grib2', ancestry={},
            frames=[dict(path=str(path), valid=meta['valid'])])]))
    vocab = 'era5-zarr' if path.is_dir() else 'earth2studio'
    family = 'arwen' if path.is_dir() else 'ai'
    source = meta.get('attributes', {}).get('source') or path.name.split('-20')[0]
    return normalize_manifest({'schema': MANIFEST, 'members': [dict(member_id='member-000', family=family,
                               source=source, file=str(path), vocabulary=vocab, ancestry=meta.get('attributes', {}))]})


def coordinate(s, scale):
    return float(s[:-1]) * scale * (-1 if s[-1] in 'SW' else 1)


def best_track(path, storm=None):
    table = json.loads(TABLE.read_text())['best_track_formats']
    lines = Path(path).read_text().splitlines()
    fmt = 'atcf' if any(', BEST,' in l or ',BEST,' in l for l in lines[:10]) else 'hurdat2'
    row = table[fmt]
    fixes = {}
    current = None
    storms = set()
    for line in lines:
        f = [x.strip() for x in line.split(row['delimiter'])]
        if fmt == 'hurdat2' and f and f[0].startswith(('AL', 'EP', 'CP')):
            current = f[0]
            storms.add(current)
            continue
        if fmt == 'hurdat2':
            if not current or (storm and current != storm):
                continue
            if not storm and len(storms) > 1:
                raise ValueError('HURDAT2 has multiple storms; use --storm-id ALnnYYYY')
            d = datetime.strptime(f[row['date']] + f[row['hhmm']], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
        else:
            if len(f) < 10 or f[row['filter_column']] != row['filter_value']:
                continue
            current = f[0] + f[1].zfill(2) + f[row['time']][:4]
            if storm and storm != current:
                continue
            minute = int(f[row['minute']] or 0)
            d = datetime.strptime(f[row['time']], '%Y%m%d%H').replace(tzinfo=timezone.utc) + timedelta(minutes=minute)
        pressure = float(f[row['pressure_hpa']])
        wind = float(f[row['wind_knots']])
        fixes[stamp(d)] = dict(valid=stamp(d), lat=coordinate(f[row['lat']], row['coordinate_scale']),
                              lon=coordinate(f[row['lon']], row['coordinate_scale']),
                              mslp_pa=pressure*100 if pressure > 0 else None,
                              max_wind10_ms=wind*0.514444 if wind >= 0 else None,
                              flags=['landfall'] if fmt == 'hurdat2' and f[row['record_identifier']] == 'L' else [])
    if not fixes:
        raise ValueError('best-track file has no selected fixes')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), format=fmt, storm_id=storm or current,
                wind_definition='NHC maximum 1-minute sustained wind; differs from model instantaneous 10 m wind',
                fixes=sorted(fixes.values(), key=lambda f: f['valid']))


def distance(a, b):
    la, lb = math.radians(a['lat']), math.radians(b['lat'])
    dl = math.radians(b['lon'] - a['lon'])
    h = math.sin((lb-la)/2)**2 + math.cos(la)*math.cos(lb)*math.sin(dl/2)**2
    return 12742 * math.asin(min(1, math.sqrt(h)))


def scores(members, tracks, best):
    limit = json.loads(TABLE.read_text())['verification']['association_max_distance_km']
    results = {}
    for m in members:
        ts = [t for t in tracks if t['member_id'] == m['member_id']]
        rows = []
        for b in best['fixes']:
            at_time = [(t, f) for t in ts for f in t['fixes'] if f['valid'] == b['valid']]
            if not at_time:
                continue
            finite = [(t, f) for t, f in at_time if f['lat'] is not None and distance(f,b) <= limit]
            if not finite:
                rows.append(dict(valid=b['valid'], centre_distance_km=None, pressure_error_hpa=None, track_id=None))
            else:
                # Verification association is explicit and never alters raw tracks.
                t, f = min(finite, key=lambda pair: distance(pair[1], b))
                rows.append(dict(valid=b['valid'], centre_distance_km=distance(f,b),
                                 pressure_error_hpa=(f['mslp_pa']-b['mslp_pa'])/100 if f['mslp_pa'] is not None and b['mslp_pa'] is not None else None, track_id=t['track_id']))
        results[m['member_id']] = dict(weight=m['weight'], association='nearest detected candidate within configured radius at each common valid time; raw candidates retained',
                                       association_max_distance_km=limit,
                                       detections=sum(r['centre_distance_km'] is not None for r in rows),
                                       misses=sum(r['centre_distance_km'] is None for r in rows), common_valid_times=rows)
    return results


def missing(valid, init):
    return dict(valid=valid, lead_hours=(utc(valid)-utc(init)).total_seconds()/3600,
                lat=None, lon=None, mslp_pa=None, max_wind10_ms=None, flags=['missing'])


def write_csv(path, tracks):
    with Path(path).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=COLUMNS)
        w.writeheader()
        for t in tracks:
            for f in t['fixes']:
                w.writerow(dict(member_id=t['member_id'], track_id=t['track_id'], weight=t['weight'],
                                **{k: json.dumps(f[k]) if k == 'flags' else f[k] for k in COLUMNS[3:]}))


def tempest(member, frames, config, work, workers=1):
    nc = work / 'tempest.nc'
    req = dict(frames=[dict(directory=f['directory'], unix_hours=utc(f['valid']).timestamp()/3600) for f in frames], fields=config['fields'], out=str(nc))
    request = work / 'tempest-request.json'
    dump(request,req)
    stages=[]
    start=time.monotonic();run([binary('rw_tc_fields'),'tempest',request]);stages.append(dict(stage='tempest_write',seconds=time.monotonic()-start,parallelism='1 thread; bounded native NetCDF staging'))
    nodes=work/'nodes.txt';stitched=work/'stitched.txt'
    grid=json.loads((Path(frames[0]['directory'])/'grid.json').read_text())
    regional=grid['route']!='global_exact_lattice'
    detect=[os.environ.get('GPUWM_DETECTNODES') or 'DetectNodes','--in_data',nc,'--out',nodes,*config['detect']]
    if regional:detect+=['--regional']
    # StitchNodes prepends the two structured-grid index columns itself.
    stitch=[os.environ.get('GPUWM_STITCHNODES') or 'StitchNodes','--in',nodes,'--out',stitched,'--in_fmt','lon,lat,slp,wind','--caltype','standard',*config['stitch']]
    # Detect each disjoint time group concurrently, then stitch the complete
    # chronologically ordered node stream once. A single-member file benefits
    # from the worker budget just as a many-member manifest does.
    slots=min(workers,len(frames))
    groups=[frames[i*len(frames)//slots:(i+1)*len(frames)//slots] for i in range(slots)]
    def detect_group(item):
        i,group=item
        part=work/f'nodes-{i:04d}.txt'
        cmd=[str(x) for x in detect]
        cmd[cmd.index('--out')+1]=str(part)
        pattern='^('+ '|'.join(utc(f['valid']).strftime('%Y-%m-%d %H:%M:%S') for f in group) +')$'
        cmd+=['--timefilter',pattern]
        start=time.monotonic()
        logpath=work/f'detect-{i:04d}.log'
        with logpath.open('w') as log:run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if 'EXCEPTION (' in logpath.read_text():
            raise ValueError(f'TempestExtremes detect failed; see {logpath}')
        return part,dict(stage='detect_time_group',seconds=time.monotonic()-start,parallelism=f'1 process in {slots} concurrent member workers',command=cmd)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=slots) as pool:parts=list(pool.map(detect_group,enumerate(groups)))
    with nodes.open('w') as stream:
        for part,stage in parts:
            stream.write(part.read_text());stages.append(stage)
    stages.append(dict(stage='detect_all_times',seconds=time.monotonic()-start,parallelism=f'{slots} concurrent processes; disjoint valid times'))
    for name,cmd in [('stitch',stitch)]:
        start=time.monotonic()
        with (work/f'{name}.log').open('w') as log:run(cmd,stdout=log,stderr=subprocess.STDOUT)
        # These upstream tools catch their own exceptions and can exit zero.
        backend_log=(work/f'{name}.log').read_text()
        if 'EXCEPTION (' in backend_log or 'do not have matching --in_fmt entries' in backend_log:
            raise ValueError(f'TempestExtremes {name} failed; see {work/name}.log')
        stages.append(dict(stage=name,seconds=time.monotonic()-start,parallelism='1 process per member; short complete-path association',command=[str(x) for x in cmd]))
    bytime={f['valid']:i for i,f in enumerate(frames)}; tracks=[];fixes=None
    for line in stitched.read_text().splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='start':
            fixes=[missing(fr['valid'],member['init']) for fr in frames]
            tracks.append(dict(track_id=f'{member["member_id"]}-{len(tracks):03d}',member_id=member['member_id'],weight=member['weight'],fixes=fixes))
        elif fixes is not None:
            # StitchNodes output: i, j, lon, lat, slp, wind, year, month, day, hour.
            valid=stamp(datetime(int(f[-4]),int(f[-3]),int(f[-2]),tzinfo=timezone.utc)+timedelta(hours=float(f[-1])))
            if valid not in bytime:raise ValueError(f'TempestExtremes produced unexpected valid {valid}')
            fix=fixes[bytime[valid]];fix.update(lat=float(f[3]),lon=(float(f[2])+180)%360-180,mslp_pa=float(f[4]),max_wind10_ms=float(f[5]),flags=[])
            edge=min(abs(fix['lat']-min(grid['lat'])),abs(max(grid['lat'])-fix['lat']))*111.195
            edge=min(edge,min(abs(((fix['lon']-x+180)%360)-180) for x in (grid['lon'][0],grid['lon'][-1]))*111.195*math.cos(math.radians(fix['lat'])))
            if regional and edge<=config['edge_radius_km']:fix['flags'].append('edge')
    if not tracks:tracks=[dict(track_id=member['member_id']+'-none',member_id=member['member_id'],weight=member['weight'],fixes=[missing(fr['valid'],member['init']) for fr in frames])]
    return dict(tracks=tracks,stages=stages,route='regional TempestExtremes --regional' if regional else 'global TempestExtremes',peak_vram_mib=0)


def execute(args):
    import tempfile
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.tc-work-', dir=out))
    try:
        return execute_work(args, out, work)
    except Exception as error:
        dump(out / 'failure-receipt.json', dict(error=str(error), work=str(work)))
        raise
    finally:
        # Remove owned atmospheric staging on both success and failure.
        deleted = []
        for p in sorted(work.rglob('*')):
            if p.is_file() and (p.suffix in ('.f32', '.u8', '.nc') or any(a.endswith('.zarr') for a in p.parts)):
                deleted.append(dict(path=str(p), bytes=p.stat().st_size))
                p.unlink()
        dump(out / 'deleted-files.json', deleted)


def execute_work(args, out, work):
    begin=time.monotonic();stages=[]
    table=json.loads(TABLE.read_text());config=table['backends'][args.backend]
    manifest=normalize_manifest(json.loads(Path(args.manifest).read_text()),Path(args.manifest).resolve().parent) if args.manifest else shortcut(args.input,work)
    dump(out/'manifest.json',manifest)
    members=manifest['members'];jobs=[];memberframes=[]
    slots=min(args.workers, max(1,sum(len(m['frames']) for m in members)))
    threads=max(1,(os.cpu_count() or 1)//slots)
    for mi,m in enumerate(members):
        mdir=work/f'member-{mi:04d}';mdir.mkdir();frames=[]
        for fi,frame in enumerate(m['frames']):
            d=mdir/f'frame-{fi:04d}';r=dict(path=frame['path'],select=frame.get('select',{}),out=str(d),vocabulary=table['vocabularies'][frame.get('vocabulary',m.get('vocabulary','earth2studio'))],fields=config['fields'])
            req=mdir/f'extract-{fi:04d}.json';dump(req,r)
            jobs.append(req);frames.append(dict(frame,directory=str(d)))
        memberframes.append((m,frames,mdir))
    start=time.monotonic()
    def extract(req):
        env=dict(os.environ,RAYON_NUM_THREADS=str(threads))
        with req.with_suffix('.log').open('w') as log:run([binary('rw_tc_fields'),'extract',req],env=env,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=slots) as pool:list(pool.map(extract,jobs))
    stages.append(dict(stage='extract_regrid_write',seconds=time.monotonic()-start,parallelism=f'{slots} processes x {threads} Rust threads'))
    backend_results=[]
    if args.backend=='tempestextremes':
        member_slots=min(args.workers,len(memberframes))
        time_workers=max(1,args.workers//member_slots)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            backend_results=list(pool.map(lambda item:tempest(*item[:2],config,item[2],time_workers),memberframes))
    else:
        interpreter=os.environ.get('GPUWM_EARTH2_PYTHON')
        if not interpreter:raise ValueError('GPUWM_EARTH2_PYTHON must name the Earth2Studio interpreter')
        for m,frames,mdir in memberframes:
            req=mdir/'gpu-request.json';result=mdir/'gpu-result.json'
            dump(req,dict(member=m,frames=frames,backend=args.backend,config=config,out=str(result)))
            env=dict(os.environ)
            extra=env.get('GPUWM_EARTH2_EXTRA_SITE')
            if extra:env['PYTHONPATH']=extra+os.pathsep+env.get('PYTHONPATH','')
            gpu_log = mdir/'gpu.log'
            try:
                with gpu_log.open('w') as log:run([interpreter,Path(__file__).with_name('tc_tracks_gpu.py'),req],env=env,stdout=log,stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as error:
                raise ValueError(f'{args.backend} tracker exited {error.returncode}; log {gpu_log}\n'
                                 + gpu_log.read_text(errors='replace')[-4000:]) from error
            backend_results.append(json.loads(result.read_text()))
    tracks=[t for r in backend_results for t in r['tracks']]
    for r in backend_results:stages.extend(r['stages'])
    paths=sorted({f['path'] for m in members for f in m['frames']})
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:hashes=dict(zip(paths,pool.map(sha,paths)))
    stages.append(dict(stage='input_hash',seconds=time.monotonic()-start,parallelism=f'{min(args.workers,len(paths))} active workers'))
    code_hashes = {name: sha(path) for name, path in {
        'orchestrator': Path(__file__), 'gpu_driver': Path(__file__).with_name('tc_tracks_gpu.py'),
        'native_reader': Path(binary('rw_tc_fields')), 'renderer': Path(binary('rw_wrfbatch'))}.items()}
    root = Path(__file__).resolve().parents[1]
    revision = 'sha256:' + hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    if (root / '.git').exists():
        revision = run(['git','-C',root,'rev-parse','HEAD'],capture_output=True).stdout.strip()
    tracker_config=dict(config,code_sha256=code_hashes,field_authority_sha256=sha(TABLE),cadence_hours={m['member_id']:[(utc(b['valid'])-utc(a['valid'])).total_seconds()/3600 for a,b in zip(m['frames'],m['frames'][1:])] for m in members})
    document=dict(schema=SCHEMA,tracker=dict(backend=args.backend,config=tracker_config,config_sha256=hashlib.sha256(json.dumps(tracker_config,sort_keys=True).encode()).hexdigest(),code_revision=revision),
                  members=[dict(**{k:m[k] for k in ('member_id','family','source','init','weight','ancestry')},inputs=[dict(path=p,sha256=hashes[p]) for p in sorted({f['path'] for f in m['frames']})]) for m in members],tracks=tracks)
    renderer_parallelism = f"Rayon pool; RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset (native default)')}"
    if args.best_track:
        document['best_track']=best_track(args.best_track,args.storm_id)
        document['scores']=scores(document['members'],tracks,document['best_track'])
    dump(out/'tracks.json',document);write_csv(out/'tracks.csv',tracks)
    receipt=dict(schema='gpuwm.tc-tracks-receipt.v1',command=[sys.executable,'-m','gpuwm',*sys.argv[1:]],code_revision=revision,backend=args.backend,config_sha256=document['tracker']['config_sha256'],input_sha256=hashes,
                 routes=[r['route'] for r in backend_results],route_reason='GPU trackers require exact 721x1440 0.25 degree global lattice; regional fields use an explicit no-signal mask. TempestExtremes uses regional topology when cropped.',
                 stages=stages,total_seconds=time.monotonic()-begin,peak_vram_mib=max(r['peak_vram_mib'] for r in backend_results),number_of_fixes=sum(f['lat'] is not None for t in tracks for f in t['fixes']),
                 backend_receipts=[{k:v for k,v in r.items() if k not in ('tracks','stages')} for r in backend_results],scores=document.get('scores'),weight_policy=manifest['weight_policy'],wind_definition='Model instantaneous maximum 10 m wind; NHC 1-minute sustained wind is a different quantity.',landfall_error=None)
    if args.best_track:
        from gpuwm.rustwx import renderer_env
        start=time.monotonic()
        run([binary('rw_wrfbatch'),'--tc-track-landfalls',out/'tracks.json',out/'landfalls.json'],env=renderer_env())
        landfalls=json.loads((out/'landfalls.json').read_text())
        comparisons=[]
        for b in document['best_track']['fixes']:
            if 'landfall' not in b['flags']:continue
            for m in members:
                choices=[f for f in landfalls['crossings'] if f['member_id']==m['member_id'] and distance(f,b)<1000]
                f=min(choices,key=lambda f:abs((utc(f['valid'])-utc(b['valid'])).total_seconds())) if choices else None
                comparisons.append(dict(member_id=m['member_id'],best=b,model=f,distance_km=distance(f,b) if f else None,time_error_hours=(utc(f['valid'])-utc(b['valid'])).total_seconds()/3600 if f else None))
        receipt['landfall_error']=dict(method=landfalls['method'],comparisons=comparisons)
        stages.append(dict(stage='landfall',seconds=time.monotonic()-start,parallelism=renderer_parallelism))
    if args.render:
        from gpuwm.rustwx import renderer_env
        png=out/'d01'/'tc-track-plume'/members[0]['init'][:10]/'track-plume.png';png.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();run([binary('rw_wrfbatch'),'--tc-track-plume',out/'tracks.json',png],env=renderer_env())
        stages.append(dict(stage='render',seconds=time.monotonic()-start,parallelism=renderer_parallelism))
        chart=png.with_name('intensity.png')
        start=time.monotonic();intensity_chart(document,chart)
        stages.append(dict(stage='intensity_chart',seconds=time.monotonic()-start,parallelism='1 thread; small plotted track table'))
    receipt['total_seconds']=time.monotonic()-begin
    dump(out/'receipt.json',receipt)
    print(out/'tracks.json')
    return 0


def intensity_chart(document, path):
    """Intensity series only. Weather maps always use rw_wrfbatch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for track in document['tracks']:
        fixes = track['fixes']
        times = [utc(f['valid']) for f in fixes]
        label = f"{track['member_id']} {track['track_id']} w={track['weight']:.4g}"
        for ax, key, scale in ((axes[0], 'mslp_pa', .01), (axes[1], 'max_wind10_ms', 1)):
            ax.plot(times, [f[key]*scale if f[key] is not None else float('nan') for f in fixes], marker='.', label=label)
    best = document.get('best_track', {}).get('fixes', [])
    if best:
        for ax, key, scale in ((axes[0], 'mslp_pa', .01), (axes[1], 'max_wind10_ms', 1)):
            ax.plot([utc(f['valid']) for f in best], [f[key]*scale if f[key] is not None else float('nan') for f in best], color='black', label='NHC best track')
    axes[0].set_ylabel('Central pressure (hPa)')
    axes[1].set_ylabel('Wind (m/s)')
    axes[1].set_xlabel('Valid time (UTC)')
    axes[0].set_title('Completed-output cyclone intensity')
    axes[1].set_title('Model instantaneous 10 m maximum; NHC 1-minute sustained maximum')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def register_cli(subparsers):
    p=subparsers.add_parser('tc-tracks',help='Track tropical cyclones in completed member output')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--manifest');group.add_argument('--input')
    p.add_argument('--out',required=True);p.add_argument('--backend',choices=('wuduan','vitart','tempestextremes'),default='wuduan')
    p.add_argument('--render',action='store_true',help='Draw the plume with rw_wrfbatch');p.add_argument('--best-track');p.add_argument('--storm-id');p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    p.set_defaults(func=main)


def main(args):
    try:
        if args.workers<1:raise ValueError('--workers must be positive')
        return execute(args)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print(f'gpuwm tc-tracks: {e}',file=sys.stderr)
        return 2
