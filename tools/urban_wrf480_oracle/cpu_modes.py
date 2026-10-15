"""Checked source-boundary and selector matrix on the CPU lane box."""
from pathlib import Path
import argparse,json,subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('lane',type=Path);a=p.parse_args();l=a.lane.resolve()
    root=Path(__file__).resolve().parents[2];out=l/'modes';out.mkdir(exist_ok=True)
    cases=[]
    for tt in (0,1):
        for ch in (1,2):
            for fvg in (0.,.2,1.):
                for profile in ('0.07 0.07 0.07 0','0.2 0.2 0.2 0','0.46 0.46 0.46 0.46'):
                    cases.append(dict(tt=tt,ch_scheme=ch,fvg=fvg,profile=profile))
    for key,values in {'boundr':(1,2),'ahoption':(0,1),'alhoption':(0,1),'imp_scheme':(1,2),'iri_scheme':(0,1),'groption':(0,1)}.items():
        for value in values:cases.append(dict(tt=0,ch_scheme=2,fvg=.2,profile='0.2 0.2 0.2 0',**{key:value}))
    results=[]
    keys=('ch_scheme','ts_scheme','ahoption','alhoption','imp_scheme','iri_scheme','groption')
    for i,c in enumerate(cases):
        label=f'mode-{i:03d}';prefix=out/label;fractions=' '.join([str(c['fvg'])]*3)
        cmd=[str(l/'defined/run_rounded'),'nlcd',label,str(prefix),*[str(c.get(k,0 if k=='groption' else -9)) for k in keys],'.5',str(c.get('boundr',-9)),'1',str(c['tt']),fractions,c['profile']]
        with prefix.with_suffix('.reference.log').open('w') as f:q=subprocess.run(cmd,cwd=l/'defined',stdout=f,stderr=subprocess.STDOUT)
        row={'case':c,'reference_command':cmd,'reference_exit':q.returncode}
        if q.returncode==0:
            grade=[str(root/'tools/urban_wrf480_oracle/grade_full.py'),'--listing',str(l/'urban480-final.cuh'),'--csv',str(prefix)+'.csv','--tables',str(l/'sources/v4.8.0/run'),'--out',str(prefix)+'.json','--tt',str(c['tt']),'--fvg',str(c['fvg']),'--fgr','.5']
            for key in keys:
                if key in c:grade+=['--switch',f'{key}={c[key]}']
            if 'boundr' in c:
                for key in ('boundr','boundb','boundg'):grade+=['--switch',f'{key}={c["boundr"]}']
            row['grade_command']=['python',*grade]
            with prefix.with_suffix('.port.log').open('w') as f:q=subprocess.run(row['grade_command'],stdout=f,stderr=subprocess.STDOUT)
            row['grade_exit']=q.returncode
            if prefix.with_suffix('.json').is_file():row['grade']=json.loads(prefix.with_suffix('.json').read_text())
        results.append(row)
    receipt={'cases':results,'failed_cases':sum(r['reference_exit']!=0 or r.get('grade_exit',1)!=0 for r in results)}
    (l/'cpu-modes.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'cases':len(results),'failed_cases':receipt['failed_cases']}));raise SystemExit(bool(receipt['failed_cases']))


if __name__=='__main__':main()
