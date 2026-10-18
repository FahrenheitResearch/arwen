"""CPU precursor comparison. This never substitutes for GPU qualification."""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
from tools.satmedmf_ccpp_oracle.check import Oracle,columns,compare,advance

def pack(x):
    n=x["u"].shape[0]
    return np.asfortranarray(np.concatenate([v.reshape(n,-1,order="F") for v in x.values()],axis=1))

def unpack(output,kpbl,n,k,nt=5):
    out={};offset=0
    for key,size,shape in [("du",k,(n,k)),("dv",k,(n,k)),("tdt",k,(n,k)),
       ("rtg",nt*k,(n,k,nt)),("hpbl",1,(n,)),("dkt",k,(n,k)),("dku",k,(n,k)),("sflux",4,(n,4))]:
        out[key]=np.asfortranarray(output[:,offset:offset+size].reshape(shape,order="F"))
        offset+=size
    out["kpbl"]=kpbl
    return out

def run(reference,hostlib,out):
    oracle=Oracle(reference)
    table=np.empty(7501,dtype=np.float64)
    oracle.library.satmedmf_table_oracle(ctypes.c_void_p(table.ctypes.data))
    lib=ctypes.CDLL(str(Path(hostlib).resolve()))
    fn=lib.satmedmf_host
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double]+[ctypes.c_void_p]*5
    fn.restype=None
    x=columns();n,k=x["u"].shape;results=[]
    for step in range(3):
        packed=pack(x)
        output=np.zeros((n,10*k+5),order="F")
        kpbl=np.zeros(n,dtype=np.int32);highwater=np.zeros(n,dtype=np.int32)
        fn(n,k,5,60.0,*[ctypes.c_void_p(v.ctypes.data) for v in [packed,output,kpbl,table,highwater]])
        expected=oracle(x)
        receipt=compare(expected,unpack(output,kpbl,n,k))
        results.append({"step":step,"fields":receipt,"workspace_highwater":int(highwater.max())})
        print(step,{a:v for a,v in receipt.items() if not v["pass"]},flush=True)
        x=advance(x,expected,60.0)
    data={"verdict":"PASS" if all(all(v["pass"] for v in r["fields"].values()) for r in results) else "FAIL",
          "stage":"CPU precursor only","columns":n,"levels":k,"steps":results}
    Path(out).write_text(json.dumps(data,indent=2)+"\n")
    return data

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference");p.add_argument("host");p.add_argument("out")
    a=p.parse_args();run(a.reference,a.host,a.out)
