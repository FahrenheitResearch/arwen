"""New source masks and layers through the actual Rust boundary, no lane grader."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
    import numpy as np
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    from tools.wps_bitmap_oracle.compare_words import compare
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--bridge',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    backend=CpuPreprocessBackend(a.bridge)
    records=[]
    rng=np.random.default_rng(702431)
    for nl,ny,nx in ((1,2,2),(3,11,13),(7,5,7),(50,2,3),(137,3,5)):
        root=a.out/f'n{nl}-y{ny}-x{nx}'
        root.mkdir(exist_ok=True)
        source=rng.uniform(0,1e12,(nl,ny,nx)).astype('<f4')
        valid=rng.integers(0,2,source.shape,dtype='u1')
        # A valid negative-zero neighborhood around one truly masked cell.
        source[0]=np.array([0x80000000],dtype='<u4').view('<f4')[0]
        valid[0]=1
        valid[0,0,0]=0
        xx,yy=np.meshgrid(np.arange((nx-1)*4+1,dtype='f4')/4,np.arange((ny-1)*4+1,dtype='f4')/4)
        coords=np.column_stack((xx.ravel(),yy.ravel())).astype('<f4')
        source.tofile(root/'source.bin')
        valid.tofile(root/'valid.bin')
        coords.tofile(root/'coords.bin')
        subprocess.run([str(a.oracle),str(root/'source.bin'),str(root/'valid.bin'),
                        str(root/'coords.bin'),str(root/'reference.bin'),str(nx),str(ny),str(nl),str(len(coords))],check=True)
        actual,counts=backend.missing_value_chain(source,valid,yy,xx,workers=2)
        actual.tofile(root/'actual.bin')
        result=compare(root/'reference.bin',root/'actual.bin',actual.size)
        record={'shape':[nl,ny,nx],'targets':len(coords),'counts':counts.sum(axis=0).tolist(),'comparison':result}
        records.append(record)
        print(json.dumps(record),flush=True)
    print('SOURCE',__import__('gpuwm.ingest.cpu_backend',fromlist=['CpuPreprocessBackend']).__file__,flush=True)
    (a.out/'summary.json').write_text(json.dumps(records,indent=2))
    return int(any(r['comparison']['mismatches'] for r in records))

if __name__=='__main__':
    raise SystemExit(main())
