"""Foreground native WRF CLI observer; preserves the existing analysis/IAU door.
SPDX-License-Identifier: Apache-2.0

Usage: python forecast-budget-adapter.py --receipt out.json --ledger-source file.py
  --masks masks.npy --mask-metadata masks.json -- <original CLI args --_worker>
Use --baseline instead of mask arguments for the matched native digest control.
"""
import argparse, contextlib, datetime, hashlib, importlib.util, json, sys, time
from pathlib import Path


class PblMassLedger:
    """Read-only YSU producer accounting in dry and moist pressure measures.

    Coupled source is sampled before composition can alias the PBL slot.
    Compare it with the final scalar physics ledger before calling it realized.
    The pressure-source minus surface-flux residual tests the YSU column solve;
    it is not an atmospheric water source and must never be added to the budget.
    """
    def __init__(self, state, mass_reduce, surface_reduce):
        self.state = state
        self.mass_reduce, self.surface_reduce = mass_reduce, surface_reduce
        self.values = {}
        self.calls = 0

    def sample(self, driver, cfg, rates, coupled, atmosphere):
        import cupy as cp
        if (cfg.bl_pbl_physics != 1 or cfg.bldt != 0 or atmosphere is None
                or (driver.urban is not None and driver.urban.pbl_terms is not None)):
            raise ValueError('PBL mass observer requires every-step YSU without BEP')
        s = self.state
        names = ('qv', 'qc', 'qi', 'qr', 'qs', 'qg')
        zero = cp.zeros_like(s.qv, dtype=cp.float64)
        raw = cp.stack([rates.get('d'+name, zero) for name in names]).astype(cp.float64)
        src = cp.stack([zero if coupled.scalar_for(name) is None else
                        coupled.scalar_for(name) for name in names]).astype(cp.float64)
        c = (s.c1h[:,None,None]*s.total_mu()[None]+s.c2h[:,None,None]).astype(cp.float64)
        # Match the producer's FP32 pressure subtraction, then reduce in FP64.
        p = atmosphere['p_interface']
        dp = (p[:-1]-p[1:]).astype(cp.float64)
        moist_c = dp/(-s.dnw[:,None,None].astype(cp.float64))
        rows = {'coupled_source': self.mass_reduce(src)*cfg.dt,
                'raw_dry_source': self.mass_reduce(c*raw)*cfg.dt,
                'ysu_pressure_source': self.mass_reduce(moist_c*raw)*cfg.dt,
                'pressure_measure_difference': self.mass_reduce((moist_c-c)*raw)*cfg.dt,
                'surface_qfx_hfx': self.surface_reduce(cp.stack(
                    [driver.fields[k] for k in ('qfx','hfx')]).astype(cp.float64))*cfg.dt}
        for name, value in rows.items():
            if name not in self.values:
                self.values[name] = cp.zeros_like(value)
            self.values[name] += value
        self.calls += 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--ledger-source',type=Path,required=True)
    parser.add_argument('--scalar-ledger-source',type=Path,
                        help='optional final-RK transport/source accounting module')
    parser.add_argument('--scalar-transport',action='store_true',
                        help='also shadow the PD limiter for directional transport budgets')
    parser.add_argument('--scalar-mass',action='store_true',
                        help='also account for water in the conserved dry-pressure coordinate')
    parser.add_argument('--pbl-mass',action='store_true',
                        help='also separate YSU pressure metric and coupled moisture source')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--masks',type=Path)
    parser.add_argument('--mask-metadata',type=Path)
    parser.add_argument('--qualification-model-masks',action='store_true')
    parser.add_argument('native',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.scalar_transport and args.scalar_ledger_source is None:
        parser.error('--scalar-transport requires --scalar-ledger-source')
    if args.scalar_mass and args.scalar_ledger_source is None:
        parser.error('--scalar-mass requires --scalar-ledger-source')
    if args.pbl_mass and not args.scalar_mass:
        parser.error('--pbl-mass requires --scalar-mass')
    native=args.native[1:] if args.native[:1]==['--'] else args.native
    if '--_worker' not in native:
        parser.error('require --_worker: a fresh child would lose the observer')
    if args.receipt.exists():parser.error('receipt already exists')
    if not args.baseline and not args.qualification_model_masks and not (args.masks and args.mask_metadata):
        parser.error('observed forecasts require fixed masks and provenance metadata')
    from gpuwm import wrfinput_forecast as door
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm.core import dycore,physics
    rec={'status':'running','native_arguments':native,'baseline':args.baseline,
         'stages':[],'frames':{},'errors':[], 'budget_weight':'initial (1/alt)*dz',
         'adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'pbl':'raw rates before coupling; not realized RK increments',
         'fields':['theta','qv','qc','qi','qr','qs','qg'],
         'installed_sources':{m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (door,tree,dycore,physics)},
         'bands':['initial_AGL<1500m','1500-3000m','3000-6000m','>=6000m'],
         'ledger_file_sha256':hashlib.sha256(args.ledger_source.read_bytes()).hexdigest()}
    capture={};stack=contextlib.ExitStack();old_run=tree.run_prepared_tree
    old_mp=dycore.apply_microphysics;old_pbl=physics.PhysicsDriver._couple_pbl_slot
    def now():return datetime.datetime.now(datetime.UTC).isoformat()
    @contextlib.contextmanager
    def stage(name,parallelism):
        a=time.perf_counter();row={'name':name,'start_utc':now(),'parallelism':parallelism}
        try:yield
        finally:
            row.update(end_utc=now(),wall_s=time.perf_counter()-a);rec['stages'].append(row)
    def save():
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(rec,indent=2)+'\n')
    def snapshot(s):
        return cp.stack([s.thb+s.thp,s.qv,s.qc,s.qi,s.qr,s.qs,s.qg]).astype(cp.float64)
    def reduce(a):
        return cp.stack([cp.stack([cp.sum(a*capture['weight']*b*m,axis=(1,2,3))/cp.maximum(m.sum(),1)
            for b in capture['bands']]) for m in capture['masks']])
    def dry_c(s):
        return (s.c1h[:,None,None]*s.total_mu()[None]+s.c2h[:,None,None]).astype(cp.float64)
    def water(s):
        return cp.stack([s.qv,s.qc,s.qi,s.qr,s.qs,s.qg]).astype(cp.float64)
    def mass_reduce(a):
        return cp.stack([cp.stack([cp.sum(a*capture['dry_weight']*b*m,axis=(1,2,3))/cp.sum(capture['area']*m)
            for b in capture['bands']]) for m in capture['masks']])
    def mp(s,cfg,dt,**kw):
        if s is not capture.get('state'):return old_mp(s,cfg,dt,**kw)
        before=snapshot(s)
        before_cq=dry_c(s)*water(s) if args.scalar_mass else None
        result=old_mp(s,cfg,dt,**kw)
        if args.scalar_mass:
            capture['mass_mp']+=mass_reduce(dry_c(s)*water(s)-before_cq)
        capture['mp']+=reduce(snapshot(s)-before);rec['mp_calls']+=1
        return result
    def pbl(self,cfg,rates,state=None,**kw):
        if capture.get('state') is self.state:
            a=cp.zeros_like(capture['initial'])
            for j,k in ((0,'dtheta'),(1,'dqv'),(2,'dqc'),(3,'dqi')):
                if rates.get(k) is not None:a[j]=rates[k]
            capture['pbl']+=reduce(a)*cfg.dt;rec['pbl_calls']+=1
            for i,m in enumerate(capture['masks']):
                capture['flux'][i]+=cp.stack([cp.sum(self.fields[k]*m)/cp.maximum(m.sum(),1)*cfg.dt
                    for k in ('qfx','hfx')])
        result=old_pbl(self,cfg,rates,state,**kw)
        if args.pbl_mass and capture.get('state') is self.state:
            capture['pbl_mass'].sample(self,cfg,rates,result,kw.get('atmosphere'))
        return result
    def bootstrap(**kw):
        s=kw['node'].state;cfg=kw['node'].cfg.run
        capture.update(state=s,cfg=cfg,start=kw['node'].clock.elapsed_seconds,grid_id=kw['node'].cfg.grid_id)
        rec['config']={k:getattr(cfg,k,None) for k in ('dt','dx','dy','mp_physics','bl_pbl_physics','bldt','km_opt','diff_opt','ysu_version','cu_physics','shcu_physics','sf_urban_physics')}
        if cfg.mp_physics!=6 and not (cfg.mp_physics==8 and (args.baseline or args.qualification_model_masks)):
            raise ValueError('scoring observer requires WSM6; Thompson is qualification-only')
        if args.baseline:return
        with stage('observer_setup','GPU; 1 CPU submission thread; mask validation is required output'):
            if cfg.bldt!=0:raise ValueError('raw PBL integration requires every-step PBL')
            zi=(s.phb+s.php)/cp.float32(9.81);z=.5*(zi[1:]+zi[:-1])-zi[0]
            capture['weight']=((zi[1:]-zi[:-1])/s.alt).astype(cp.float64)
            bands=cp.ascontiguousarray((z>=1500).astype(cp.uint8)+(z>=3000).astype(cp.uint8)+(z>=6000).astype(cp.uint8))
            capture['bands']=[bands==i for i in range(4)]
            if args.qualification_model_masks:
                from gpuwm.da.obsop import simulated_reflectivity
                class View:
                    def __init__(self,state):self.state=state;self.buffers={}
                    def __getattr__(self,k):return getattr(self.state,k)
                    def scratch(self,shape,slot,dtype=None):
                        dtype=cp.dtype(cp.float32 if dtype is None else dtype);key=(tuple(shape),slot,dtype.str)
                        if key not in self.buffers:self.buffers[key]=cp.empty(shape,dtype)
                        return self.buffers[key]
                comp=simulated_reflectivity(View(s),cfg).max(0)
                interior=cp.zeros_like(comp,dtype=cp.bool_);interior[5:-5,5:-5]=True
                masks=cp.stack([interior,interior&(comp<20),interior&(comp>=20)&(comp<35),interior&(comp>=40)])
                rec['mask_labels']=['interior','model_initial_lt20','model_initial_20_35','model_initial_ge40']
                rec['mask_provenance']='qualification MODEL masks, not MRMS'
            else:
                meta=json.loads(args.mask_metadata.read_text())
                declared=meta.get('input_authority_sha256',{})
                if not declared or any(rec['input_authority_sha256'].get(k)!=v for k,v in declared.items()):
                    raise ValueError('mask metadata must match native input-authority hashes')
                raw=args.masks.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=meta['mask_sha256']:raise ValueError('mask hash differs')
                masks=cp.load(args.masks,allow_pickle=False)
                if masks.dtype!=cp.bool_ or masks.ndim!=3 or masks.shape[1:]!=s.qv.shape[1:]:
                    raise ValueError('mask file requires bool (regions,ny,nx) on native grid')
                if len(meta['labels'])!=len(masks):raise ValueError('mask labels differ')
                rec['mask_labels']=meta['labels'];rec['mask_provenance']=meta
            rec['mask_counts']=masks.sum(axis=(1,2)).get().tolist()
            if args.qualification_model_masks:
                keep=[i for i,n in enumerate(rec['mask_counts']) if n]
                masks=masks[keep]
                rec['mask_labels']=[rec['mask_labels'][i] for i in keep]
                rec['mask_counts']=[rec['mask_counts'][i] for i in keep]
            elif min(rec['mask_counts'])==0:raise ValueError('empty fixed diagnostic region')
            capture['masks']=masks;capture['initial']=snapshot(s)
            if args.scalar_mass:
                from gpuwm.core.constants import G
                capture['area']=s.cell_area_weight()
                capture['dry_weight']=-s.dnw[:,None,None].astype(cp.float64)/G*capture['area'][None]
                capture['initial_cq']=dry_c(s)*water(s)
                capture['mass_mp']=cp.zeros((len(masks),4,6),cp.float64)
                rec['mass_convention']='C*q endpoints, C=c1h*mu+c2h; -dnw/G and engine physical area; divide by fixed region area'
                rec['mass_fields']=['qv','qc','qi','qr','qs','qg']
                if args.pbl_mass:
                    def surface_reduce(a):
                        return cp.stack([cp.sum(a*capture['area']*m,axis=(1,2))/cp.sum(capture['area']*m) for m in capture['masks']])
                    capture['pbl_mass']=PblMassLedger(s,mass_reduce,surface_reduce)
                    rec['pbl_mass_convention']='Producer coupled rates, raw dry-pressure rates, YSU moist-pressure rates; area-weighted surface flux; all dt-integrated; compare coupled source to final scalar physics before attribution'
            shape=(len(masks),4,7)
            capture['mp']=cp.zeros(shape,cp.float64);capture['pbl']=cp.zeros(shape,cp.float64)
            capture['flux']=cp.zeros((len(masks),2),cp.float64)
            spec=importlib.util.spec_from_file_location('explicit_budget_ledger',args.ledger_source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            capture['ledger']=module.ProcessLedger(bands) if cfg.mp_physics==6 else None
            rec.update(mp_calls=0,pbl_calls=0,ledger_labels=module.LABELS,
                       wsm6_ledger_attached=capture['ledger'] is not None)
            if capture['ledger'] is not None:
                stack.enter_context(capture['ledger'].observe())
                rec.update(kernel_source_sha256=capture['ledger'].source_sha256,
                    instrumented_source_sha256=capture['ledger'].instrumented_sha256)
            if args.scalar_ledger_source is not None:
                spec=importlib.util.spec_from_file_location('explicit_scalar_ledger',args.scalar_ledger_source)
                scalar_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar_module)
                capture['scalar']=scalar_module.ScalarProcessLedger(s,reduce,transport=args.scalar_transport,
                    mass_reduce=mass_reduce if args.scalar_mass else None)
                stack.enter_context(capture['scalar'].observe())
                rec.update(scalar_terms=scalar_module.TERMS,
                    scalar_mixing_terms=scalar_module.MIXING_TERMS,
                    scalar_transport_terms=scalar_module.TRANSPORT_TERMS,
                    scalar_transport_source_sha256=capture['scalar'].transport_source_sha256,
                    scalar_installed_source_sha256=capture['scalar'].source_sha256,
                    scalar_ledger_file_sha256=hashlib.sha256(args.scalar_ledger_source.read_bytes()).hexdigest())
                if args.scalar_mass:rec['scalar_mass_terms']=scalar_module.MASS_TERMS
            dycore.apply_microphysics=mp;physics.PhysicsDriver._couple_pbl_slot=pbl
    def observe(*,grid_id,model_seconds,**event):
        if args.baseline or grid_id!=capture.get('grid_id'):return
        try:
            elapsed=float(model_seconds)-capture['start'];rec['last_lead_seconds']=elapsed
            if elapsed not in (15.,60.,300.,900.):return
            with stage('sample_'+str(int(elapsed)), 'GPU reductions; compact required output download'):
                s=capture['state'];w=.5*(s.w[1:]+s.w[:-1]);wmax=w.max(0)
                compact=None
                if capture['ledger'] is not None:
                    ledger=capture['ledger'].values
                    compact=cp.stack([ledger[:,:,m].sum(axis=2)/m.sum() for m in capture['masks']])
                row={'net':reduce(snapshot(s)-capture['initial']).get().tolist(),
                     'microphysics':capture['mp'].get().tolist(),'pbl_raw_integral':capture['pbl'].get().tolist(),
                     'surface_qfx_hfx_integral':capture['flux'].get().tolist(),
                     'wsm6_process_ledger':None if compact is None else compact.get().tolist(),
                     'updraft_columns_1_5_10_max':cp.stack([cp.stack([cp.sum((wmax>=x)&m) for x in (1.,5.,10.)]+[cp.max(cp.where(m,wmax,-cp.inf))]) for m in capture['masks']]).get().tolist()}
                if 'scalar' in capture:
                    row.update(scalar_processes={k:v.get().tolist() for k,v in capture['scalar'].values.items()},
                               scalar_calls=dict(capture['scalar'].calls),
                               scalar_mixing={k:v.get().tolist() for k,v in capture['scalar'].mixing_values.items()},
                               scalar_mixing_calls={**capture['scalar'].mixing_calls, 'sixth_order':dict(capture['scalar'].mixing_calls['sixth_order'])},
                               scalar_mixing_config=capture['scalar'].mixing_config,
                               scalar_transport={k:v.get().tolist() for k,v in capture['scalar'].transport_values.items()},
                               scalar_transport_calls=dict(capture['scalar'].transport_calls))
                    if args.scalar_mass:
                        row.update(mass_net=mass_reduce(dry_c(s)*water(s)-capture['initial_cq']).get().tolist(),
                                   mass_mp=capture['mass_mp'].get().tolist(),
                                   mass_scalar={k:v.get().tolist() for k,v in capture['scalar'].mass_values.items()},
                                   mass_mixing={k:v.get().tolist() for k,v in capture['scalar'].mass_mixing_values.items()},
                                   mass_transport={k:v.get().tolist() for k,v in capture['scalar'].mass_transport_values.items()})
                rec['frames'][str(int(elapsed))]=row
                if args.pbl_mass:
                    row['pbl_mass']={k:v.get().tolist() for k,v in capture['pbl_mass'].values.items()}
                    row['pbl_mass_calls']=capture['pbl_mass'].calls
        except Exception as exc:rec['errors'].append(repr(exc))
    def run(inputs,**kw):
        if rec.get('native_calls'):raise ValueError('one member process per diagnostic receipt')
        rec['native_calls']=1;rec['input_authority_sha256']=dict(inputs.authority_sha256)
        base=kw.get('initialization');previous=kw.get('ensemble_bootstrap')
        if base is None:raise ValueError('native initialization must be explicit')
        class Proxy:
            def __getattr__(self,k):return getattr(base,k)
            def step_observer(self,**event):
                original=getattr(base,'step_observer',None)
                if original is not None:original(**event)
                observe(**event)
        def handoff(**event):
            result=None if previous is None else previous(**event)
            if result is not None:raise ValueError('external bootstrap took over the run')
            bootstrap(**event)
        kw.update(initialization=Proxy(),ensemble_bootstrap=handoff)
        result=old_run(inputs,**kw)
        rec['final_state_digest']=result['final_state_digest']
        rec['native_status']=result['status'];rec['analysis_bundle']=result.get('analysis_bundle')
        rec['native_timing_seconds']=result.get('timing_seconds')
        if rec['errors']:raise RuntimeError(rec['errors'])
        if not args.baseline and (not rec['mp_calls'] or not rec['frames']):raise RuntimeError('no observed WSM6 steps')
        return result
    try:
        # Import CUDA only inside the externally granted foreground process.
        import cupy as cp
        tree.run_prepared_tree=run
        with stage('native_cli','GPU native forecast; 1 CPU CLI/restore/receipt driver'):
            status=door.main(native)
        if status or rec.get('native_calls')!=1:raise RuntimeError('native CLI did not complete one member: '+str(status))
        rec['status']='pass'
    except BaseException as exc:
        rec.update(status='failed',error=repr(exc));raise
    finally:
        stack.close();tree.run_prepared_tree=old_run
        dycore.apply_microphysics=old_mp;physics.PhysicsDriver._couple_pbl_slot=old_pbl
        save()
    return 0


if __name__=='__main__':raise SystemExit(main())
