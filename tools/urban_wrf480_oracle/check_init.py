"""Grade every source-initialized old and new single-layer Registry field."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from gpuwm.verify.urban_oracle import load
from gpuwm.core.urban_tables import load_urban_params,urban_category_set
from gpuwm.core.urban_state import urban_var_init_host,option_spec


def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
    results=[];total=0;diffs=[]
    def convert(array):return array.T if array.ndim==2 else array.transpose(1,2,0)
    for name,ref in load(a.oracle).items():
        lcz=int(name[-1]);params=load_urban_params(1,lcz,generation='wrf_480')
        got=urban_var_init_host(option=1,use_wudapt_lcz=lcz,params=params,categories=urban_category_set(isurban=13),
            ivgtyp=convert(ref['ivgtyp']),tsk=convert(ref['tsk_in']),tslb=convert(ref['tslb_in']),tmn=convert(ref['tmn_in']),smois=convert(ref['smois_in']),
            frc_urb2d=convert(ref['frc_in']),num_urban_hi=15,nz=4,spec=option_spec(1,generation='wrf_480'))
        fields={}
        for key,value in got.items():
            if key not in ref:continue
            want=convert(ref[key]);untouched=want==np.float32(-7)
            mask=~untouched;actual=value[mask];expected=want[mask]
            bad=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32));total+=actual.size
            fields[key]={'words':actual.size,'differences':len(bad),'reference_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'port_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'untouched_slots':int(untouched.sum())}
            for index in bad:diffs.append({'case':name,'field':key,'index':int(index),'got':int(actual.view(np.uint32)[index]),'want':int(expected.view(np.uint32)[index])})
        results.append({'case':name,'fields':fields})
    result={'words':total,'differences':len(diffs),'cells':diffs,'cases':results};a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'words':total,'differences':len(diffs)}));raise SystemExit(bool(diffs))


if __name__=='__main__':main()
