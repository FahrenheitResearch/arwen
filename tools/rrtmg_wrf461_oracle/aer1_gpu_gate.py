"""GPU hash gate against the WRF column oracle. Run only in the card queue."""
from pathlib import Path
import hashlib,inspect,json,os,sys,tempfile
import numpy as np
import cupy as cp
from gpuwm.core import rrtmg_aerosol_optics as aer
from gpuwm.core import rrtmg_legacy_device as prep
from gpuwm.core import rrtmg_sw as sw
from gpuwm.core.rrtmg_legacy import _adapter_kernel
from gpuwm.certify.kernel_manifest import kernel_manifest


def run(out):
    root=Path(__file__).parent
    wanted=json.loads((root/'aer1_reference.json').read_text())['columns']
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        path=Path(tmp)/'columns.txt'
        # The deterministic input generator is shared with the Fortran driver.
        import subprocess
        subprocess.run([sys.executable,str(root/'aer1_make_columns.py'),str(path)],check=True)
        lines=iter(path.read_text().splitlines())
        nc,nz=map(int,next(lines).split())
        tables=sw.load_sw_tables(aer_opt=1)
        engine=sw.CudaSW(tables)
        mismatches=[]; checks=[]
        def check(cid,name,value):
            array=cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value,np.float32)
            array=np.asarray(array,np.float32)
            digest=hashlib.sha256(array.copy(order='C').tobytes()).hexdigest()
            pin=wanted[str(cid)][name]['sha256']
            checks.append({'column':cid,'field':name,'words':int(array.size),'sha256':digest,'match':digest==pin})
            if digest!=pin:mismatches.append({'column':cid,'field':name,'got':digest,'want':pin})
        for cid in range(1,nc+1):
            header=list(map(int,next(lines).split()))
            scalars=np.asarray(next(lines).split(),np.float32)
            layer=np.asarray([next(lines).split() for _ in range(nz)],np.float32)
            interface=np.asarray([next(lines).split() for _ in range(nz+1)],np.float32)
            julian,gmt,xtime,radt,declin,solcon,coszen,lat,lon,albedo,tsk,xland,xice,snow,obscur=scalars
            monthly=aer.aer1_climatology_device(cp.asarray([lat],cp.float32),cp.asarray([lon],cp.float32),tables.aer1_climatology)
            aerod,aod,timed=aer.aer1_time_pressure_device(monthly,julian,cp.asarray(layer[:,0][None,:]),cp.asarray(interface[:,0][None,:]),tables.aer1_climatology)
            check(cid,'aer/monthly',cp.asnumpy(monthly[0]).transpose(1,0,2).reshape(12,72,order='F'))
            check(cid,'aer/timed',timed[0])
            check(cid,'aer/aerod',np.concatenate((cp.asnumpy(aerod[0]),np.zeros((1,6),np.float32))))
            check(cid,'aer/AODTOT',aod[0])
            if coszen<=0:
                check(cid,'wrf/gsw',np.float32(0));check(cid,'wrf/swddir',np.float32(0))
                check(cid,'wrf/rthratensw',np.zeros(nz,np.float32));check(cid,'driver/SWDOWN',np.float32(0))
                continue
            optics=aer.aer1_sw_optics_device(aerod,nz+1,tables)
            for name,value in zip(('ztaua','zasya','zomga'),optics):
                check(cid,'spcin/'+name,cp.asnumpy(value[0]).T)
            columns=('p3d','t3d','dz8w','pi3d','rho3d','qv3d','qc3d','qr3d','qi3d','qs3d','qg3d','cldfra3d','re_cloud','re_ice','re_snow','o33d')
            kw={name:cp.asarray(layer[:,k][None,:]) for k,name in enumerate(columns) if name not in ('pi3d','rho3d')}
            kw.update(p8w=cp.asarray(interface[:,0][None,:]),t8w=cp.asarray(interface[:,1][None,:]))
            for name,val in (('tsk',tsk),('albedo',albedo),('xland',xland),('xice',xice),('snow',snow),('xlat',lat),('xcoszen',coszen),('obscur',obscur)):
                kw[name]=cp.asarray([val],cp.float32)
            kw.update(solcon=solcon,icloud=1,warm_rain=False,cldovrlp=2,idcor=0,o3input=2,
                      has_reqc=1,has_reqi=1,has_reqs=1,yr=1974,julian=julian,mp_physics=8,sf_surface_physics=2)
            ps=prep.swrad_prep_batch_device(**kw)
            ps['adjes']=np.full(1,ps['adjes'],np.float32)
            for name,value in ps.items():
                key='entry/'+name
                if key in wanted[str(cid)]:
                    if isinstance(value,cp.ndarray) and value.ndim==3:value=value.transpose(2,0,1)
                    check(cid,key,value)
            accepted=inspect.signature(engine.rrtmg_sw_batched_device).parameters
            ins={key:value for key,value in ps.items() if key in accepted}
            result=engine.rrtmg_sw_batched_device(**ins,aer_opt=1,aerosol=optics,mcica_layout='column')
            gsw=cp.zeros(1,cp.float32);rth=cp.zeros((nz,1),cp.float32)
            _adapter_kernel('rla_sw_out_grid')((1,),(128,),
              (np.int32(1),np.int32(nz),np.int32(nz+1),result['swdflx'],result['swuflx'],result['swhr'],
               cp.asarray(layer[:,3,None]),np.float32(86400),np.int64(1),cp.asarray([0],cp.int64),rth,gsw))
            swdown=cp.empty(1,cp.float32)
            _adapter_kernel('rla_swdown')((1,),(128,),
              (np.int64(1),gsw,cp.asarray([albedo],cp.float32),swdown))
            check(cid,'wrf/gsw',gsw);check(cid,'driver/SWDOWN',swdown)
            check(cid,'wrf/swddir',result['swdkdir'][0,0])
            check(cid,'wrf/rthratensw',rth[:,0])
            engine.release_scratch()
    summary={'columns':nc,'mode':'strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default',
      'passed':not mismatches,'mismatches':mismatches,'checks':checks,
      'max_ulp':0 if not mismatches else None,'per_field_tolerance':0,
      'kernel_manifest':kernel_manifest()}
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('columns','mode','passed','max_ulp')} | {'mismatch_count':len(mismatches),'first_mismatch':mismatches[:1]}))
    return 0 if not mismatches else 1


if __name__=='__main__':
    raise SystemExit(run(Path(sys.argv[1])))
