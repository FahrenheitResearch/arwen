"""Locate the first assigned scalar divergence without tolerance."""
from pathlib import Path
import argparse,json


def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('port',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
    def read(path):
        return [s.split()[1:] for s in path.read_text().splitlines() if s.startswith('TRACE ')]
    ref,port=read(a.reference),read(a.port)
    result={'reference_scalars':len(ref),'port_scalars':len(port),'first':None}
    for i,(x,y) in enumerate(zip(ref,port)):
        if x[1]!=y[1] or float(x[2])!=float(y[2]):
            result['first']={'index':i,'reference':x,'port':y,'reference_context':ref[max(0,i-3):i+4],'port_context':port[max(0,i-3):i+4]}
            break
    a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
