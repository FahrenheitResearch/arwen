"""Package unchanged native stream words and portable build metadata."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument("native",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    shapes=dict(latitude=(32,),units=(2,32),coords=(6,32),nearest=(2,2,32))
    fields={};streams={}
    for name,shape in shapes.items():
        raw=(args.native/(name+".bin")).read_bytes()
        fields[name]=np.frombuffer(raw,dtype='<f4').reshape(shape)
        streams[name]=hashlib.sha256(raw).hexdigest()
    target=args.output/'native.npz';np.savez_compressed(target,**fields)
    receipt=dict(schema='sfire-native-geographic-units-v1',sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 streams_sha256=streams,extraction=json.loads((args.native/'extraction.json').read_text()),
                 build=json.loads((args.native/'build.json').read_text()))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
