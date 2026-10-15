"""Compare the real cloud kernel with WRF opt_cloud on 64 columns."""
from pathlib import Path
import argparse,gzip,struct,json
import numpy as np
import cupy as cp
from gpuwm.core.kernels import get_kernel
from gpuwm.core.radiation_tables import radiation_table_words
from check_columns import read_columns,compare_fields

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 fixture=Path(__file__).parent/'fixtures'
 raw=gzip.decompress((fixture/'cloud-inputs.bin.gz').read_bytes())
 if struct.unpack('<4i',raw[:16])!=(0x47524335,64,32,4):raise ValueError('cloud input contract')
 refpath=a.out.with_suffix('.ref.bin');refpath.write_bytes(gzip.decompress((fixture/'defined-reference.bin.gz').read_bytes()))
 reference,_=read_columns(refpath);fields=tuple(reference)[7:]
 candidate={name:[] for name in fields};at=16;n=32;c=16
 table=radiation_table_words('goddard-cloud')
 assert table is radiation_table_words('goddard-cloud')
 kernel=get_kernel('goddard','goddard_cloud_columns')
 for batch in range(4):
  dz=np.frombuffer(raw,dtype='<f4',offset=at,count=c*n).copy();at+=c*n*4
  q=np.frombuffer(raw,dtype='<f4',offset=at,count=c*n*6).copy();at+=c*n*6*4
  re=np.frombuffer(raw,dtype='<f4',offset=at,count=c*n*6).copy();at+=c*n*6*4
  out=cp.empty(c*n*(3*11+3*10),dtype=cp.float32)
  kernel((4,),(128,),(cp.asarray(dz),cp.asarray(q),cp.asarray(re),table,out,np.int32(n),np.int32(c)))
  words=out.get().view('<u4');off=0
  for name in fields:
   size=c*n*(11 if name.startswith('sw_') else 10)
   candidate[name].append(words[off:off+size].reshape((-1,c)).T.copy());off+=size
 candidate={name:np.concatenate(parts) for name,parts in candidate.items()}
 if at!=len(raw):raise ValueError('extra cloud input words')
 from gpuwm.wrf_exact import ENABLED
 passed,result=compare_fields({name:reference[name] for name in fields},candidate,ENABLED)
 a.out.write_text(json.dumps({'pass':passed,'strict':ENABLED,'columns':64,'fields':result},indent=2)+'\n')
 print('cloud pass',passed,'strict',ENABLED)
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
