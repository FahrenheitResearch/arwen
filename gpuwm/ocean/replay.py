"""Replay orchestration. Rust reads files; CUDA interpolates/diagnoses all fields."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, time, math, hashlib
from .door import read_config, load_prepared, prepare_coupler, upload, download, render_ocean

def utc(t):return datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def run(config,forcing,outdir,*,proof=False,render=False,research_data_licence=""):
    import cupy as cp
    import numpy as np
    from .coupling import DeviceCoupler,_CUDA
    start=time.perf_counter();cfg=read_config(config)
    if proof:
        grid,bathy,prepared=load_prepared(cfg['prepared'])
        coupler=DeviceCoupler(grid,bathy,interval=cfg['interval_seconds'])
    else:
        coupler=prepare_coupler(cfg);grid,bathy=coupler.grid,coupler.bathymetry
    # A request points to one or more native extraction directories, allowing
    # wind reuse without duplicating a sibling's acquisition or editing it.
    forcing=Path(forcing).resolve();request=json.loads(forcing.read_text())
    licence=request.get('licence','')
    if licence not in ('CC0-1.0','public-domain') and (not licence or licence!=research_data_licence):raise ValueError('forcing data require --research-data-licence '+(licence or '<missing licence>'))
    streams=request.get('streams',[str(forcing.parent)])
    fields={};times=None;metadata=[]
    for stream in streams:
        stream=(forcing.parent/stream).resolve();m=json.loads((stream/'manifest.json').read_text())
        if m['schema']!='gpuwm.ocean.forcing/v1':raise ValueError('forcing extraction schema mismatch')
        if times is None:times=m['times'];axes=(m['latitude'],m['longitude'])
        elif times!=m['times'] or axes!=(m['latitude'],m['longitude']):raise ValueError('forcing time/grid axes differ across streams')
        for name,row in m['arrays'].items():
            if name in fields:raise ValueError('duplicate forcing plane '+name)
            if row['shape']!=[len(times),grid.ny,grid.nx]:raise ValueError('forcing and ocean grid differ')
            path=stream/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('forcing hash mismatch '+name)
            fields[name]=upload(path,row['shape'])
        metadata.append(m)
    if not times or len(times)<2:raise ValueError('replay requires at least two forcing times')
    if set(('u10','v10','psfc','temperature','dewpoint'))-fields.keys():raise ValueError('replay requires u10 v10 psfc temperature dewpoint')
    # Coordinate identity is checked on device once, before stepping. Comparing
    # sizes alone would silently place the same-shaped forcing on another coast.
    lat=cp.asarray(axes[0],dtype=cp.float32);lon=cp.asarray(axes[1],dtype=cp.float32)
    if not bool(cp.allclose(grid.latitude,lat[:,None],atol=1.e-5,rtol=0.)) or not bool(cp.allclose(grid.longitude,lon[None,:],atol=1.e-5,rtol=0.)):
        raise ValueError('forcing coordinates differ from prepared ocean grid')
    out=Path(outdir).resolve()
    if out.exists() and any(out.iterdir()):out=out.with_name(out.name+'-'+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=True)
    scratch={k:cp.empty(grid.shape,dtype=cp.float32) for k in ('u10','v10','psfc','temperature','qv','ustar','wind_speed','stress')}
    from gpuwm.certify.kernel_manifest import record_module
    module=cp.RawModule(code=_CUDA,options=())
    record_module('gpuwm.ocean.coupling:device',source=_CUDA,options=(),module=module)
    kernel=module.get_function('replay');magnitude=module.get_function('stress_mag')
    n=grid.nx*grid.ny;blocks=((n+255)//256,);threads=(256,)
    frames=[];selection={0,len(times)-1}
    # The strongest wind time is selected from on-device output diagnostics,
    # after all steps. Keep regular output frames; render only three below.
    output_frames=[];saved=[];max_wind=-1.;strongest=None
    cp.cuda.get_current_stream().synchronize();upload_wall=time.perf_counter()-start
    loop_start=time.perf_counter();device_start,device_end=cp.cuda.Event(),cp.cuda.Event();device_start.record()
    def diagnose(record,alpha,t):
        coupler.surface_roughness(coupler.z0,scratch['ustar']) if coupler.forcing is not None else None
        kernel(blocks,threads,(*(fields[k] for k in ('u10','v10','psfc','temperature','dewpoint')),coupler.z0,
            *(scratch[k] for k in ('u10','v10','psfc','temperature','qv','ustar','wind_speed')),
            np.float32(alpha),np.int32(record),np.int32(n)))
        coupler.handoff(**{k:scratch[k] for k in ('u10','v10','psfc','ustar','temperature','qv')},time_seconds=t)
    def save(t,index):
        nonlocal max_wind,strongest
        magnitude(blocks,threads,(coupler.forcing.tau_x,coupler.forcing.tau_y,scratch['stress'],np.int32(n)))
        quantities={k:scratch[k] for k in ('wind_speed','stress')};quantities['psfc']=coupler.forcing.psfc
        if coupler.waves is not None:
            o=coupler.waves.outputs;quantities.update(hs=o.hs,tp=o.tp,direction=o.direction)
        if coupler.surge is not None:
            o=coupler.surge.outputs;quantities.update(eta=o.eta,inundation=o.inundation)
        frame={'valid_time':utc(t),'fields':{}}
        stats={}
        # The native surge integrator accumulates these counters on device.
        # Inspect them at output cadence so finite-value recovery inside a
        # kernel cannot conceal a failed integration in the final receipt.
        if coupler.surge is not None and hasattr(coupler.surge,'diagnostics'):
            names=('nonfinite','negative_depth','cfl','invalid_forcing')
            counters={name:int(coupler.surge.diagnostics[i]) for i,name in enumerate(names)}
            stats['surge_diagnostics']=counters
            if any(counters.values()):raise RuntimeError(f'surge integration invalid at {utc(t)}: {counters}')
        for name,array in quantities.items():
            if not bool(cp.all(cp.isfinite(array))):raise RuntimeError(f'nonfinite {name} at {utc(t)}')
            stats[name]={'min':float(cp.min(array)),'max':float(cp.max(array))}
            target=out/'planes'/f'{index:04d}-{name}.bin';download(array,target);frame['fields'][name]=str(target)
        # These reductions occur only at the existing output cadence. Keep
        # roughness evidence in the receipt without a host read per ocean step.
        if coupler.waves is not None:
            for name,array in (('roughness_m',coupler.z0),('charnock',coupler.alpha),('ustar',scratch['ustar'])):
                if not bool(cp.all(cp.isfinite(array))):raise RuntimeError(f'nonfinite {name} at {utc(t)}')
                stats[name]={'min':float(cp.min(array)),'max':float(cp.max(array))}
        if stats['wind_speed']['max']>max_wind:max_wind=stats['wind_speed']['max'];strongest=index
        output_frames.append(frame);saved.append({'valid_time':utc(t),'statistics':stats})
        progress={'schema':'gpuwm.ocean.progress/v1','valid_time':utc(t),'coupling_steps':coupler.steps,
                  'wall_seconds':time.perf_counter()-start,'statistics':stats}
        temporary=out/'progress.json.new';temporary.write_text(json.dumps(progress,indent=2))
        temporary.replace(out/'progress.json')
        print(json.dumps(progress),flush=True)
    diagnose(0,0.,times[0]);save(times[0],0)
    handoffs=1
    for record in range(len(times)-1):
        t=times[record];end=times[record+1]
        while t<end-1.e-8:
            dt=min(coupler.interval,end-t)
            diagnose(record,(t-times[record])/(end-times[record]),t)
            coupler.advance(dt);handoffs+=1;t+=dt
        diagnose(record,1.,end);handoffs+=1;save(end,record+1)
    device_end.record();device_end.synchronize();loop_wall=time.perf_counter()-loop_start
    selection.add(strongest)
    for i in sorted(selection):frames.append(output_frames[i])
    frames[0]['fields'].update({k:str(Path(cfg['prepared'])/f'{k}.bin') for k in ('elevation','wet')})
    products={'schema':'gpuwm.ocean.products/v1','out_dir':str(out/'products'),'domain':f'ocean-{grid.ny}x{grid.nx}',
        'source_label':request.get('source_label','Atmosphere-ocean coupling'),
        'nx':grid.nx,'ny':grid.ny,'latitude':str(Path(cfg['prepared'])/'latitude.bin'),'longitude':str(Path(cfg['prepared'])/'longitude.bin'),'wet':str(Path(cfg['prepared'])/'wet.bin'),
        'rows':str(Path(__file__).with_name('rows.json')),'frames':frames}
    product_request=out/'products-request.json';product_request.write_text(json.dumps(products,indent=2))
    render_wall=0.
    if render:
        t=time.perf_counter();render_ocean(product_request);render_wall=time.perf_counter()-t
    receipt={'schema':'gpuwm.ocean.run-receipt/v1','mode':'forcing-proof' if proof else 'coupled-solvers',
        'solvers':[] if proof else cfg['solvers'],'grid':[grid.ny,grid.nx],'start':utc(times[0]),'end':utc(times[-1]),
        'forcing_handoffs':handoffs,'coupling_steps':coupler.steps,'interval_seconds':coupler.interval,
        'forcing_diagnosis':'winds/pressure/temperature/dewpoint; CUDA neutral 10m log layer ustar with current wave roughness','forcing_data_licence':licence,
        'roughness_feedback':not proof and coupler.waves is not None,'vertical_datum':bathy.datum,
        'wall_seconds':{'upload_and_validation':upload_wall,'device_replay_and_output':loop_wall,'render':render_wall,'total':time.perf_counter()-start},
        'parallelism':{'replay':'CUDA 256 threads/block','per_step_host_field_transforms':0,'per_step_host_field_copies':0,'output_cadence_seconds':'forcing record interval'},
        'cuda_event_elapsed_seconds':cp.cuda.get_elapsed_time(device_start,device_end)/1000,
        'allocated_cuda_bytes':cp.get_default_memory_pool().total_bytes(),'frames':saved,'product_request':str(product_request),
        'source_manifests':[m['input_sha256'] for m in metadata]}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='frames'}))
    return out
