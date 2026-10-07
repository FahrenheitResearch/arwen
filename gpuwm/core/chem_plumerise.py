"""Freitas plume driver and profile cache, for table-selected emitted rows.

WRF f52c197e chem/module_plumerise1.F:224-292 and emissions_driver.F:686-698.
GSL 3e6660c6 physics/smoke_dust/module_plumerise.F90:118-166 and
rrfs_smoke_wrapper.F90:326-327. This process moves no transported mass; the
fire process applies the cached injection and its ledger owns the emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from gpuwm.core.chem_context import ChemAllocation
from gpuwm.core.chem_plumerise_cache import PlumeParameters, plume_call_due, distribute_cached

KEY = 'plumerise.freitas'
LEDGER = 'emitted'
REQUIRES = ('t_phy','p_phy','rho','qv','u_phy','v_phy','z','z_at_w')
KERNEL_MODULES = ('chem_plumerise',)
#: Workspace words per fire column: PL_WS_ROWS (kernels/chem_plumerise.cu)
#: rows of 202 words -- the transcription's 114 state rows, the heat table,
#: the plume tops and the routines' 202-word scratch.
PLUME_WORKSPACE_WORDS = 131 * 202
#: Fire columns per launch (bounds the workspace at 2048 x 106 KB).
PLUME_BATCH_COLUMNS = 2048
ALLOCATES = (
    ChemAllocation('plume_k_min','2d','int32'),
    ChemAllocation('plume_k_max','2d','int32'),
    ChemAllocation('plume_flam_frac','2d'),
    ChemAllocation('plume_top','2d',output_name='PLUME_TOP',units='m',
                   description='Maximum Freitas plume top above ground'),
)
# No 3-D array is kept.  The fire process redistributes each hour's emission
# over the cached bounds k_min, k_max and flam_frac (GSL module_plumerise.F90:
# 158-166, apply_cached), so a per-column unit injection profile would never
# be read; the landuse arm's per-group cache is not allocated because that arm
# is refused at run time (step).  Each was nz*ny*nx float32 on every smoke
# domain: 108 MB apiece at 896 x 512 x 59.


@lru_cache(maxsize=1)
def fire_group_properties():
    """Pinned WRF heat-flux rows; mean_fct and firesize remain explicit inputs."""
    path=Path(__file__).resolve().parent.parent/'data/chem/plumerise/fire_groups.v1.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    return tuple((float(r['heat_min']),float(r['heat_max']),bool(r['single_heat_pass']))
                 for r in data['groups'])


def _device():
    if os.environ.get('GPUWM_NO_LOCAL_GPU','') not in ('','0'):
        raise RuntimeError('GPUWM_NO_LOCAL_GPU forbids plume launch; launching would use the protected local device')
    import cupy
    return cupy


def _validate_met(met):
    cp=_device()
    if any(name not in met for name in REQUIRES):
        raise ValueError('Freitas needs all declared met fields; missing thermodynamics or geometry gives undefined plume buoyancy')
    shape=met['t_phy'].shape
    if len(shape)!=3 or not 16<=shape[0]<=200:
        raise ValueError('Freitas supports 16..200 model levels; outside this range the 200-level source state or 16-plane group cache is overrun')
    nz,ny,nx=shape
    for name in REQUIRES:
        a=met[name]; expected=(nz+1,ny,nx) if name=='z_at_w' else shape
        if not isinstance(a,cp.ndarray) or a.dtype!=cp.float32 or a.shape!=expected or not a.flags.c_contiguous:
            raise ValueError(f'plume met {name} needs contiguous float32 {expected}; other storage is misaddressed by the column kernel')
    return shape


def launch(met, ebu_in, *, arm='frp', frp_inst=None, kpbl=None,
           uspdavg2d=None, hpbl2d=None, mean_fct=None, firesize=None,
           parameters=PlumeParameters(), audit=False):
    """Gather fire columns, run one thread per column, and return packed results.

    FRP is already diurnally scaled by the fire process. The cap is applied
    here. Selection uses >= frp_min; non-fire columns are never launched or
    scattered. Landuse inputs have shape (4,ny,nx), emissions (nrows,ny,nx).
    """
    cp=_device(); nz,ny,nx=_validate_met(met); nxy=ny*nx
    if arm not in ('frp','landuse'):
        raise ValueError('plume arm must be frp or landuse; another arm applies incorrect fire physics')
    if (not isinstance(ebu_in,cp.ndarray) or ebu_in.dtype!=cp.float32 or ebu_in.ndim!=3
            or ebu_in.shape[1:]!=(ny,nx) or not ebu_in.flags.c_contiguous):
        raise ValueError('plume emissions need contiguous rows_2d float32; invalid row or grid strides inject into the wrong cells')
    nrows=ebu_in.shape[0]
    if nrows<1:
        raise ValueError('plume needs an emitted row vector; an empty vector loses the fire driver emission gate')
    def plane(value,name,dtype=cp.float32):
        if not isinstance(value,cp.ndarray) or value.dtype!=dtype or value.shape!=(ny,nx) or not value.flags.c_contiguous:
            raise ValueError(f'plume {name} needs contiguous {dtype} 2d input; invalid storage reads outside the column')
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f'plume {name} must be finite; undefined values corrupt plume time stepping')
        return value
    prop=cp.zeros((12,nxy),dtype=cp.float32)
    if arm=='frp':
        frp=plane(frp_inst,'frp_inst')
        if bool(cp.any(frp<0)):
            raise ValueError('FRP must be nonnegative; negative burnt area makes the plume radius undefined')
        prop[0]=cp.minimum(frp.ravel(),np.float32(parameters.frp_max))
        prop[9]=plane(kpbl,'kpbl',cp.int32).ravel()
        prop[10]=plane(uspdavg2d,'uspdavg2d').ravel()
        prop[11]=plane(hpbl2d,'hpbl2d').ravel()
        fire=prop[0]>=np.float32(parameters.frp_min)
        if bool(cp.any(((kpbl<1)|(kpbl>nz))&fire.reshape(ny,nx))):
            raise ValueError('kpbl must address a model level; an invalid windy-PBL override writes outside the injection column')
    else:
        for value,name in [(mean_fct,'mean_fct'),(firesize,'firesize')]:
            if (not isinstance(value,cp.ndarray) or value.dtype!=cp.float32
                    or value.shape!=(4,ny,nx) or not value.flags.c_contiguous):
                raise ValueError(f'landuse {name} needs four contiguous float32 planes; missing groups apply the wrong fire-property arm')
            if bool(cp.any(~cp.isfinite(value))) or bool(cp.any(value<0)):
                raise ValueError(f'landuse {name} must be finite and nonnegative; invalid fractions or areas produce undefined fire heating')
        prop[1:5]=mean_fct.reshape(4,nxy); prop[5:9]=firesize.reshape(4,nxy)
        sf=((mean_fct[0]+mean_fct[1])+mean_fct[2])+mean_fct[3]
        ss=((firesize[0]+firesize[1])+firesize[2])+firesize[3]
        fire=(sf.ravel()>=np.float32(parameters.landuse_gate))&(ss.ravel()>=np.float32(parameters.landuse_gate))&(cp.max(ebu_in,axis=0).ravel()!=0)
        if bool(cp.any((mean_fct>=np.float32(parameters.landuse_gate))&(firesize<=0)&fire.reshape(1,ny,nx))):
            raise ValueError('each active landuse group needs positive fire area; zero area divides by a zero plume radius')
    if bool(cp.any(~cp.isfinite(ebu_in))) or bool(cp.any(ebu_in<0)):
        raise ValueError('emitted rows must be finite and nonnegative; invalid incoming emissions corrupt the emitted mass ledger')
    columns=cp.flatnonzero(fire).astype(cp.int32); nf=int(columns.size)
    if not nf: return dict(columns=columns)
    selected={k:met[k].reshape(met[k].shape[0],nxy)[:,columns] for k in REQUIRES}
    if any(bool(cp.any(~cp.isfinite(a))) for a in selected.values()):
        raise ValueError('fire-column meteorology must be finite; undefined thermodynamics corrupt plume time stepping')
    if (bool(cp.any(selected['p_phy']<=0)) or bool(cp.any(selected['t_phy']<=0))
            or bool(cp.any(selected['rho']<=0)) or bool(cp.any(selected['qv']<0))):
        raise ValueError('fire-column pressure, temperature and density must be positive and vapor nonnegative; invalid thermodynamics make buoyancy undefined')
    if (bool(cp.any(cp.diff(selected['z_at_w'],axis=0)<=0))
            or bool(cp.any(cp.diff(selected['z'],axis=0)<=0))
            or bool(cp.any(selected['z'][-1]-selected['z_at_w'][0]<=1950))):
        raise ValueError('fire columns need increasing heights and a top above 1950 m AGL; invalid geometry overruns interpolation or indexes below the 20-level damping layer')
    packed=cp.zeros((nf,8,nz+1),dtype=cp.float32)
    for slot,name in enumerate(REQUIRES):
        packed[:,slot,:selected[name].shape[0]]=selected[name].T
    incoming=cp.ascontiguousarray(ebu_in.reshape(nrows,nxy)[:,columns].T)
    props=cp.ascontiguousarray(prop[:,columns].T)
    p=np.zeros(20,dtype=np.float32)
    p[:7]=parameters.frp_min,parameters.frp_wthreshold,parameters.zpbl_threshold,parameters.uspd_threshold,parameters.frp_max,parameters.alpha,parameters.wind_eff_opt
    for g,(low,high,single) in enumerate(fire_group_properties()): p[7+2*g:9+2*g]=low,high; p[16+g]=single
    if parameters.alpha<=0 or parameters.wind_eff_opt not in (0,1):
        raise ValueError('plume alpha must be positive and wind option 0 or 1; invalid entrainment divides by zero or selects undefined wind physics')
    output=cp.zeros((nf,nrows,nz),dtype=cp.float32); bounds=cp.zeros((nf,2),dtype=cp.int32)
    diagnostics=cp.zeros((nf,10),dtype=cp.float32); iterations=cp.zeros((nf,8),dtype=cp.int32)
    profiles=cp.empty((nf,7,200),dtype=cp.float32) if audit else None
    status=cp.empty(nf,dtype=cp.int32)
    groups=cp.zeros((nf,16),dtype=cp.float32)
    from gpuwm.core.chem_plumerise_cache import plume_function
    kernel=plume_function('freitas_columns')
    # Each fire column's plume state is a PLUME_WORKSPACE_WORDS slice of a
    # global workspace (kernels/chem_plumerise.cu, freitas_columns): as local
    # memory it was a 93 KB frame CUDA reserves for every resident thread of
    # the card.  Columns run in batches of at most PLUME_BATCH_COLUMNS so the
    # workspace is bounded (2048 x 106 KB = 217 MB) however many fires burn.
    batch=min(nf,PLUME_BATCH_COLUMNS)
    workspace=cp.empty(batch*PLUME_WORKSPACE_WORDS,dtype=cp.float32)
    params=cp.asarray(p)
    for first in range(0,nf,batch):
        count=min(batch,nf-first)
        kernel(((count+31)//32,),(32,),
            (packed,incoming,props,params,output,bounds,diagnostics,
             profiles if profiles is not None else np.uint64(0),groups,iterations,status,
             np.int32(nf),np.int32(nz),np.int32(nrows),np.int32(0 if arm=='frp' else 1),
             workspace,np.int32(first),np.int32(count)))
    del workspace
    if bool(cp.any(status)):
        raise ValueError('computed plume injection exceeds model levels; scattering would write outside the emission column')
    return dict(columns=columns,ebu=output,bounds=bounds,diagnostics=diagnostics,
                profiles=profiles,groups=groups,steps=iterations,packed_met=packed,properties=props)


def scatter(result,k_min,k_max,flam_frac,plume_top,injection=None):
    """Scatter only selected columns, preserving every non-fire cell.

    ``injection`` (nz, ny, nx), when given, receives the unit-emission
    profile of each fire column; the process keeps none (see ALLOCATES).
    """
    columns=result['columns']
    if not columns.size: return
    k_min.ravel()[columns]=result['bounds'][:,0]
    k_max.ravel()[columns]=result['bounds'][:,1]
    flam_frac.ravel()[columns]=result['diagnostics'][:,0]
    plume_top.ravel()[columns]=result['diagnostics'][:,9]
    if injection is None: return
    # Process launch uses unit row emissions; the cache is a unit profile.
    injection.reshape(injection.shape[0],-1)[:,columns]=result['ebu'][:,0,:].T


def rows(table): return table.rows_for(KEY)


def refusal(cfg):
    """Why this configuration cannot run plumerise.freitas, or None.

    The landuse arm (WRF-Chem plumerise_driver) sizes each fire from
    per-column flaming fractions and fire sizes (mean_fct, firesize;
    module_plumerise1.F:224-275) that prep_chem_sources files carry and no
    source row in this build supplies, so its plume would size fires from
    nothing.  Refused here, at configuration, rather than at the plume's
    first due step after a whole preparation.
    """
    if getattr(cfg,'plume_fire_properties','frp')=='landuse':
        return ("plume_fire_properties = 'landuse' reads per-column flaming "
                "fractions and fire sizes (mean_fct, firesize; WRF-Chem "
                "module_plumerise1.F:224-275) that no source row in this "
                "build supplies, so the plume would size fires it has no data "
                "for; run the frp arm")
    return None


def _validate_config(cfg):
    if getattr(cfg,'scale_fire_emiss',False):
        raise ValueError('MOZART scale_fire_emiss is unavailable; scaling requires a MOZART mechanism and would misallocate emitted species without it')
    arm=getattr(cfg,'plume_fire_properties','frp')
    if arm not in ('frp','landuse'):
        raise ValueError('plume_fire_properties must be frp or landuse; another arm applies incorrect fire heating')


def rebuilt_mid_run(k_max, ktau):
    """Unwritten plume columns need a profile before the next cadence beat.

    A relocated fresh strip has zero bounds beside the carried overlap.
    Waiting for the regular plume cadence would inject fresh-strip fires
    into the lowest two levels. A restored written cache has bounds >= 2.
    """
    return int(ktau)>2 and bool((k_max==0).any())


def init(ctx):
    """Initialize only empty columns; preserve every restored valid cache."""
    _validate_config(ctx.cfg)
    if not rows(ctx.table): return
    cp=_device()
    chem=getattr(getattr(ctx,'state',None),'chem',None)
    empty=ctx.diag['plume_k_max']==0
    if chem is not None and rebuilt_mid_run(ctx.diag['plume_k_max'],getattr(ctx.clock,'ktau',1)):
        chem.plume_rebuilt_columns=empty
    ctx.diag['plume_k_min'][empty]=1
    ctx.diag['plume_k_max'][empty]=2
    ctx.diag['plume_flam_frac'][empty]=0.
    ctx.diag['plume_top'][empty]=0.


def step(ctx,dt,ktau):
    """Cache injection only; fire emissions remain the fire process's work.

    The fire process publishes model-grid inputs in ctx.diag['fire_plume_inputs'].
    SourceFrames.at may supply individual inputs through source-variable role
    bindings; emission rows select sources. No source name selects physics.
    """
    active=rows(ctx.table)
    if not active: return
    _validate_config(ctx.cfg)
    arm=getattr(ctx.cfg,'plume_fire_properties','frp')
    clock=ctx.clock
    if clock is None:
        raise ValueError('plume needs the chem clock; absent time schedules incorrect fire injection updates')
    defaults=PlumeParameters()
    frequency=getattr(ctx.cfg,'plumerisefire_frq',defaults.frp_cadence_minutes if arm=='frp' else defaults.landuse_cadence_minutes)
    interval=getattr(ctx.cfg,'stepfirepl',None)
    if interval is None and arm=='landuse' and dt>0:
        # WRF chemics_init.F:646,650 uses positive NINT and then MAX(1,...).
        count=np.float32(np.float32(frequency)*np.float32(60)/np.float32(dt))
        interval=max(1,int(np.floor(float(count)+.5)))
    # GSL wrapper.F90:234,297 forms its r4 clock from ktau*dt. WRF uses
    # ChemClock's r8 start-of-step curr_secs, so the two clocks differ.
    cadence_secs=float(np.float32(np.float32(ktau)*np.float32(dt))) if arm=='frp' else clock.curr_secs
    chem=getattr(getattr(ctx,'state',None),'chem',None)
    rebuilt=getattr(chem,'plume_rebuilt_columns',None)
    due=plume_call_due(arm,cadence_secs,ktau,frequency,
                      adaptive=getattr(ctx.cfg,'use_adaptive_time_step',False),dt=dt,
                      stepfirepl=interval)
    if rebuilt is None and not due:
        return
    if chem is not None: chem.plume_rebuilt_columns=None
    if arm=='landuse':
        # WRF-Chem's arm reads per-column mean_fct/firesize from
        # prep_chem_sources files (gpuwm/data/chem/plumerise/LANDUSE_INPUTS.md);
        # no source row supplies them, and a land-cover category cannot.
        raise ValueError("plume_fire_properties='landuse' needs per-column flaming fractions and fire sizes (mean_fct, firesize) that no source row in this build supplies; deriving them from land cover would invent the fire, so run the frp arm")
    inputs=ctx.diag.get('fire_plume_inputs')
    if inputs is None:
        from gpuwm.core.chem_fire import plume_inputs
        inputs=plume_inputs(ctx,ktau)
    if inputs is None and ctx.frames is not None:
        inputs=frame_inputs(ctx,ctx.diag.get('fire_valid_time',clock.curr_secs))
    if inputs is None:
        raise ValueError('fire_plume_inputs is not produced; plume injection would use absent FRP or landuse/PBL source fields')
    cp=_device(); met={name:ctx.met(name) for name in REQUIRES}
    ny,nx=met['t_phy'].shape[1:]
    unit=cp.ones((1,ny,nx),dtype=cp.float32)
    inputs=dict(inputs)
    inputs={name:(value if name=='parameters' else cp.asarray(value,dtype=cp.int32 if name=='kpbl' else cp.float32)) for name,value in inputs.items()}
    only_rebuilt = rebuilt is not None and not due
    if only_rebuilt and arm=='frp':
        inputs['frp_inst']=cp.where(rebuilt,inputs['frp_inst'],cp.float32(0))
    emitted=inputs.pop('ebu_in',None)
    if emitted is not None:
        unit[:,cp.max(emitted,axis=0)==0]=0.
    result=launch(met,unit,arm=arm,**inputs)
    scatter(result,ctx.diag['plume_k_min'],ctx.diag['plume_k_max'],
            ctx.diag['plume_flam_frac'],ctx.diag['plume_top'])
    if arm=='frp':
        inactive=inputs['frp_inst']<np.float32(inputs.get('parameters',defaults).frp_min)
    else:
        m=inputs['mean_fct']; f=inputs['firesize']
        inactive=((((m[0]+m[1])+m[2])+m[3])<np.float32(defaults.landuse_gate))|((((f[0]+f[1])+f[2])+f[3])<np.float32(defaults.landuse_gate))
        if emitted is not None: inactive|=cp.max(emitted,axis=0)==0
    retired=inactive&(ctx.diag['plume_k_max']>2)
    if only_rebuilt:
        retired &= rebuilt
    ctx.diag['plume_k_min'][retired]=1; ctx.diag['plume_k_max'][retired]=2
    ctx.diag['plume_flam_frac'][retired]=0.; ctx.diag['plume_top'][retired]=0.
    # emission.fire (after this process in CHEM_STEP_ORDER) redistributes
    # the hour's emission over the new bounds when it sees this step here.
    ctx.state.chem.plume_ran_ktau=ktau
    ctx.state.chem.plume_ran_columns=rebuilt if only_rebuilt else None


def frame_inputs(ctx,valid_time):
    """Resolve table-declared process input roles through SourceFrames.at.

    A source variable may carry {process: KEY, input: <launcher parameter>}.
    No product name, species name or species slot determines the route.
    """
    active=rows(ctx.table)
    sources={e['source'] for row in active for e in row.emissions if e.get('vertical')=='plumerise'}
    result={}
    for source in sorted(sources):
        row=ctx.table.sources[source]
        for field,binding in row.variables.items():
            if binding.get('process')!=KEY: continue
            role=binding.get('input')
            if role not in ('frp_inst','kpbl','uspdavg2d','hpbl2d','mean_fct','firesize','ebu_in'):
                raise ValueError(f'plume frame input role {role} is not in the launcher contract; binding an unrelated source field applies incorrect fire physics')
            if role in result:
                raise ValueError(f'plume input {role} has multiple frame bindings; unaggregated sources would double fire heating or mix incompatible column properties')
            result[role]=ctx.frames.at(source,field,valid_time)
    if not result: return None
    return result


def apply_cached(ctx,ebu_in,ebu,columns=None):
    """Reapply cached bounds with source-order arithmetic to a new row vector."""
    cp=_device()
    nr,nz,ny,nx=ebu.shape; nxy=ny*nx
    if columns is None:
        columns=cp.flatnonzero(cp.any(ebu_in!=0.,axis=0).ravel()).astype(cp.int32)
    arm=getattr(ctx.cfg,'plume_fire_properties','frp')
    if arm=='frp':
        return distribute_cached(columns,ctx.diag['plume_k_min'],ctx.diag['plume_k_max'],
                ctx.diag['plume_flam_frac'],ebu_in,ctx.met('z_at_w'),ebu)
    if not columns.size: return
    cache=ctx.diag['plume_group_cache']
    # The same validator establishes device type, grid, contiguity and height
    # bounds for the redistribution route without launching a plume solve.
    if (cache.shape!=(nz,ny,nx) or cache.dtype!=cp.float32 or not cache.flags.c_contiguous
            or ebu_in.shape!=(nr,ny,nx) or ebu_in.dtype!=cp.float32
            or ebu.dtype!=cp.float32 or not ebu.flags.c_contiguous or not ebu_in.flags.c_contiguous):
        raise ValueError('landuse cache and emission arrays must share contiguous float32 grid storage; mismatched layouts write outside emission columns')
    if columns.dtype!=cp.int32 or columns.ndim!=1 or not columns.flags.c_contiguous:
        raise ValueError('landuse column offsets need a contiguous int32 vector; other words are invalid emission addresses')
    if bool(cp.any((columns<0)|(columns>=nxy))) or cp.unique(columns).size!=columns.size:
        raise ValueError('landuse column offsets must be unique and inside the grid; invalid offsets race or write outside emission columns')
    z=ctx.met('z_at_w')
    if z.shape!=(nz+1,ny,nx) or z.dtype!=cp.float32 or not z.flags.c_contiguous:
        raise ValueError('landuse redistribution needs contiguous 3d_w float32 heights; other storage misaddresses layer thickness')
    if any(cp.shares_memory(ebu,a) for a in (cache,ebu_in,z,columns)):
        raise ValueError('landuse emission output cannot alias inputs; writes would overwrite later group bounds or emission values')
    gathered=cache.reshape(nz,nxy)[:16,columns]
    lo=gathered[0::4]; hi=gathered[1::4]; frac=gathered[2::4]
    active=frac>=np.float32(1.e-6)
    if (bool(cp.any(~cp.isfinite(gathered))) or bool(cp.any(frac<0))
            or bool(cp.any(active&((lo<1)|(hi<lo)|(hi>=nz))))):
        raise ValueError('landuse cached groups need finite fractions and valid inclusive bounds; corrupt cache divides by zero or writes outside model levels')
    heights=z.reshape(nz+1,nxy)[:,columns]
    incoming=ebu_in.reshape(nr,nxy)[:,columns]
    if (bool(cp.any(~cp.isfinite(heights))) or bool(cp.any(cp.diff(heights,axis=0)<=0))
            or bool(cp.any(~cp.isfinite(incoming))) or bool(cp.any(incoming<0))):
        raise ValueError('landuse heights must increase and emitted rows be finite/nonnegative; invalid thickness or input produces undefined emitted mass')
    from gpuwm.core.chem_plumerise_cache import plume_function
    kernel=plume_function('ebu_distribute_landuse')
    kernel(((columns.size+63)//64,),(64,),
           (columns,cache,ebu_in,z,ebu,np.int32(columns.size),np.int32(nz),np.int32(nxy),np.int32(nr)))
