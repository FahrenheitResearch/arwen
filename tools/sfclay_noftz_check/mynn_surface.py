from pathlib import Path
import sys, json, struct, subprocess, hashlib, os
import numpy as np
from tools.mynn_sfclay_wrf461_column_oracle import columns as c, compare as cmp
L = Path('/work/pool-sfclay-noftz-mynn-mm5-eta')
kind, action = sys.argv[1:3]
configs = c.CONFIGS + tuple((f'new_{i}_{spp}',i,1,dx,first,seeded,12,spp)
    for i,dx,first,seeded in [(0,1.0,1,True),(1,1e6,2000000,False),(2,250.0,1,True),(3,1e7,100,False)] for spp in (0,1))
c.CONFIGS = cmp.CONFIGS = configs
if kind == 'subnormal':
    rows = [(f'qv_bits_{bits}_{water}',c._column(qv1=np.array([bits],np.uint32).view(np.float32)[0], qsfc=0., xland=water, hfx=0.,qfx=0.)) for bits in [1,2,4095,0x7fffff,0x800000,0x800001] for water in [1.,2.]]
else:
    rng=np.random.default_rng(7102026)
    rows=[]
    for k in range(768):
        t=float(rng.uniform(195,325)); dz=float(10**rng.uniform(0.0,2.7)); p=float(rng.uniform(20000,105000))
        water=2. if k%3==0 else 1.
        q=float(10**rng.uniform(-10,-1.3))
        if k%7==0:
            q=float(0.622*611.2*np.exp(17.67*(t-273.15)/(t-29.65))/p)
        col=c._column(u1=float(rng.uniform(-85,85)),v1=float(rng.uniform(-65,65)),u2=float(rng.uniform(-90,90)),v2=float(rng.uniform(-70,70)),t1=t,tsk=float(rng.uniform(max(190,t-40),min(340,t+40))),qv1=q,p1=p,psfc=p+float(rng.uniform(50,1200)),dz1=dz,dz2=float(rng.uniform(1,500)),pblh=float(rng.uniform(0,6500)),mavail=float(rng.uniform(0,1)),xland=water,snowh=float(rng.choice([0,.09999999,.1,.10000001,2.])),znt=min(float(10**rng.uniform(-9,.3)),dz*.05),hfx=float(rng.uniform(-800,2500)),qfx=float(rng.uniform(-.001,.002)),ust=float(rng.uniform(.001,3)),ustm=float(rng.uniform(.001,3)),mol=float(rng.uniform(-20,20)),qsfc=float(rng.uniform(0,.05)))
        rows.append((f'new_random_{k:04d}',col))
    for boundary in [7.,13.]:
        for dz in [np.nextafter(np.float32(2*boundary),np.float32(-np.inf)),np.float32(2*boundary),np.nextafter(np.float32(2*boundary),np.float32(np.inf))]:
            for t in [np.nextafter(np.float32(273.15),np.float32(-np.inf)),np.float32(273.15),np.nextafter(np.float32(273.15),np.float32(np.inf))]:
                for water in [1.,1.4999999,1.5,2.]:
                    rows.append((f'boundary_{len(rows)}',c._column(dz1=dz,tsk=t,xland=water,snowh=.1)))
directory=L/kind
if action=='prepare':
    directory.mkdir(exist_ok=True)
    arrays={f:np.array([row[f] for _,row in rows],np.float32) for f in c.FIELDS}
    blob=c.columns_bytes(arrays)
    (directory/'columns.bin').write_bytes(blob)
    (directory/'configs.txt').write_text(c.configs_text())
    (directory/'column-names.txt').write_text('\n'.join(name for name,_ in rows)+'\n')
    (L/'receipts'/f'{kind}-inputs.json').write_text(json.dumps(dict(ncol=len(rows),configs=configs,seed=7102026,sha256=hashlib.sha256(blob).hexdigest(),names=[name for name,_ in rows]),indent=1)+'\n')
    subprocess.run([str(L/'oracles/mynn/o2/run_columns'),str(directory/'columns.bin'),str(directory/'configs.txt'),str(directory/'oracle.bin')],check=True)
    print(kind,len(rows),'prepared')
else:
    actual_hash=hashlib.sha256()
    independent=dict(compared=0,mismatched=0,nonfinite_got=0,nonfinite_ref=0,max_ulp=0,max_abs=0.)
    grade=cmp._grade
    def audited(got,want):
        for name,ref in want.items():
            g=np.asarray(got[name],np.float32).reshape(ref.shape)
            a=g.view(np.uint32); b=ref.view(np.uint32)
            actual_hash.update(a.tobytes())
            order=lambda x: np.where(x&0x80000000,0xffffffff-x,x+np.uint32(0x80000000)).astype(np.int64)
            u=np.abs(order(a)-order(b))
            independent['compared']+=g.size; independent['mismatched']+=int(np.count_nonzero(a!=b))
            independent['nonfinite_got']+=int(np.count_nonzero(~np.isfinite(g)))
            independent['nonfinite_ref']+=int(np.count_nonzero(~np.isfinite(ref)))
            independent['max_ulp']=max(independent['max_ulp'],int(u.max()))
            independent['max_abs']=max(independent['max_abs'],float(np.max(np.abs(g.astype(float)-ref.astype(float)))))
        return grade(got,want)
    cmp._grade=audited
    if kind=='subnormal':
        # Keep the actual first-call words as a compact witness, independent
        # of the claim harness's ULP routine.
        audit_grade=cmp._grade
        witnesses=[]
        def witnessed(got,want):
            if not witnesses:
                witnesses.append({name:dict(got=float(np.asarray(got[name])[0]),want=float(ref[0]),got_bits=hex(int(np.asarray(got[name],np.float32).view(np.uint32)[0])),want_bits=hex(int(ref.view(np.uint32)[0]))) for name,ref in want.items()})
            return audit_grade(got,want)
        cmp._grade=witnessed
    receipt=cmp.run(directory/'oracle.bin',directory)
    receipt['summary']=cmp.summarize(receipt)
    independent['actual_sha256']=actual_hash.hexdigest()
    receipt['coverage']=cmp.coverage(directory/'oracle.bin',directory)
    receipt['independent_audit']=independent
    receipt['oracle_sha256']=hashlib.sha256((directory/'oracle.bin').read_bytes()).hexdigest()
    if kind=='subnormal': receipt['first_call_column0']=witnesses[0]
    mode='strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default'
    (L/'receipts'/f'{kind}-{mode}.json').write_text(json.dumps(dict(summary=receipt['summary'],independent_audit=independent,first_call_column0=receipt.get('first_call_column0')),indent=1)+'\n')
    print(json.dumps(dict(kind=kind,mode=mode,summary=receipt['summary'],independent_audit=independent),indent=1))
    sys.exit(bool(receipt['summary']['mismatched_values']))
