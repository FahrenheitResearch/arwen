"""Compare every real-run false-arm field before and after the scalar port.
Run under the GPU mutex with L pointing to the lane, its false prepared
input in real-prepared, and its reviewed predecessor worktree in off-base.
"""
import os,sys,subprocess,json,shlex,hashlib
from pathlib import Path
import numpy as np

def main():
    r=Path(os.environ['L']);rows=[]
    for mode in ('default','strict'):
        checkpoints=[]; commands=[]
        for phase,code in (('before','off-base'),('after','src')):
            t=r/code;env=dict(os.environ,T=str(t),PYTHONPATH=str(t)+':'+str(t/'tests'))
            for key in ('GPUWM_WRF_EXACT','GPUWM_WRF_EXACT_ADVECTION'):env.pop(key,None)
            if mode=='strict':env.update(GPUWM_WRF_EXACT='1',GPUWM_WRF_EXACT_ADVECTION='1')
            out=r/f'off-real/{mode}/{phase}'
            door=[sys.executable,'-m','gpuwm.cli','sim',str(r/'real-prepared'),'--experiment-config',str(r/'real-prepared/experiment.toml'),'--wps-namelist',str(r/'real-prepared/namelist.wps'),'--outdir',str(out),'--print-command']
            p=subprocess.run(door,env=env,cwd=t,text=True,capture_output=True,check=True)
            (r/f'receipts/off-real-{mode}-{phase}-command.log').write_text(p.stdout+p.stderr)
            line=next(line for line in p.stdout.splitlines() if ' -m gpuwm.prepared_single_domain_forecast ' in line);cmd=shlex.split(line);print(shlex.join(cmd),flush=True)
            existing=list(out.rglob("gpuwmrst_d01_*.npz"))
            if existing: print("REUSE COMPLETE CHECKPOINT",existing,flush=True)
            else: subprocess.run(cmd,env=env,cwd=t,check=True)
            files=list(out.rglob('gpuwmrst_d01_*.npz'));assert len(files)==1,files;checkpoints.append(files[0]);commands.append(cmd)
        with np.load(checkpoints[0],allow_pickle=False) as a,np.load(checkpoints[1],allow_pickle=False) as b:
            assert a.files==b.files;fields={}
            for k in a.files:
                if k=='__gpuwm_restart_header__':continue
                assert a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes(),(mode,k)
                fields[k]={'shape':list(a[k].shape),'dtype':str(a[k].dtype),'bytes':a[k].nbytes,'sha256':hashlib.sha256(a[k].tobytes()).hexdigest()}
            assert np.isfinite(a['fields/qke']).all()
        rows.append({'mode':mode,'status':'PASS','fields':len(fields),'seconds':3600,'commands':commands,'full_field_hashes':fields})
    (r/'receipts/off-real-hashes.json').write_text(json.dumps({
        'before': subprocess.check_output(['git', '-C', str(r/'off-base'), 'rev-parse', 'HEAD'], text=True).strip(),
        'after': subprocess.check_output(['git', '-C', str(r/'src'), 'rev-parse', 'HEAD'], text=True).strip(),
        'readings': rows}, indent=2)+'\n')
    print('OFF_REAL_HASHES_PASS',[(x['mode'],x['fields']) for x in rows],flush=True)


if __name__ == "__main__":
    main()
