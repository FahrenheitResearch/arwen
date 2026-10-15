"""Run the real shared-loader kernels against the 64-column WRF fixture."""
from pathlib import Path
import argparse,gzip,struct
import numpy as np
import cupy as cp
from gpuwm.core.kernels import get_kernel_int_defines
from gpuwm.wrf_exact import ENABLED

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mutate-band',type=int,choices=(1,2));a=p.parse_args()
 fixture=Path(__file__).parent/'fixtures'/'inputs.bin.gz'
 raw=gzip.decompress(fixture.read_bytes())
 if struct.unpack('<4i',raw[:16])!=(0x47524935,64,32,4):raise ValueError('input contract')
 n=32;c=16;ni=c*((n+1)+4*n+3+10+1);no=c*n*(3*11+3*10)
 defines=(('GDD_MAX_LEVELS',32),)
 if a.mutate_band:defines+=(('GDD_MUTATE_BAND',a.mutate_band),)
 kernel=get_kernel_int_defines('goddard','goddard_bands_columns',defines)
 at=16
 with a.out.open('wb') as out:
  out.write(struct.pack('<5i',0x47524435,64,n,4,64))
  for batch in range(4):
   inputs=np.frombuffer(raw,dtype='<f4',count=ni,offset=at).copy();at+=ni*4
   flags=np.frombuffer(raw,dtype='<i4',count=3*c,offset=at).copy();at+=3*c*4
   optics=np.frombuffer(raw,dtype='<f4',count=no,offset=at).copy();at+=no*4
   di=cp.asarray(inputs);df=cp.asarray(flags);do=cp.asarray(optics)
   output=cp.empty(c*(6*(n+1)+4),dtype=cp.float32)
   kernel((1,),(16,),(di,df,do,output,np.int32(n),np.int32(c)))
   cp.cuda.get_current_stream().synchronize()
   out.write(output.get().astype('<f4',copy=False).tobytes())
   # The band stage takes optics as inputs. It does not qualify optics ingestion.
   out.write(optics.tobytes())
 if at!=len(raw):raise ValueError('extra input words')
 print('columns=64 bands=2 strict='+str(ENABLED)+' output='+str(a.out))
if __name__=='__main__':main()
