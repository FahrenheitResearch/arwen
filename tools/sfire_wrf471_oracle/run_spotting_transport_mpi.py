"""Actual MPI rendezvous transport of native typed firebrand packets."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from mpi4py import MPI

from gpuwm.core.sfire_spotting import exchange_neighbor_packets


def run(destination):
    comm=MPI.COMM_WORLD
    rank,size=comm.Get_rank(),comm.Get_size()
    if size!=9:
        raise ValueError("the native eight-neighbor topology control requires nine ranks")
    x,y=rank%3,rank//3
    directions=((-1,0),(1,0),(0,1),(0,-1),(-1,1),(1,1),(-1,-1),(1,-1))
    neighbors=np.asarray([(y+dy)*3+x+dx if 0<=x+dx<3 and 0<=y+dy<3 else -1
                          for dx,dy in directions],np.int32)
    records=[]
    for count in (0,2,20000):
        packets=[]
        for edge,neighbor in enumerate(neighbors):
            n=count if neighbor>=0 else 0
            real=np.full((8,n),rank+.375,np.float32)
            integer=np.full((3,n),rank+17,np.int32)
            packets.append((real,integer))
        received=exchange_neighbor_packets(comm,packets,neighbors,return_device=False)
        words=0
        for neighbor,(real,integer) in zip(neighbors,received):
            expected=count if neighbor>=0 else 0
            assert real.shape==(8,expected) and integer.shape==(3,expected)
            assert np.all(real==np.float32(neighbor+.375))
            assert np.all(integer==np.int32(neighbor+17))
            words+=real.size+integer.size
        records.append(dict(rank=rank,brands_per_edge=count,received_words=words,
                            different_words=0,payload_bytes_per_edge=count*44))
    all_records=comm.gather(records,root=0)
    if rank==0:
        flattened=[row for group in all_records for row in group]
        root=Path(__file__).resolve().parents[2]
        paths=("gpuwm/core/sfire_spotting.py","tools/sfire_wrf471_oracle/run_spotting_transport_mpi.py")
        result=dict(ranks=size,records=flattened,graded_words=sum(row["received_words"] for row in flattened),
            different_words=0,mpi_library=MPI.Get_library_version().strip(),
            source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths})
        Path(destination).write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({key:result[key] for key in ("ranks","graded_words","different_words")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("destination")
    run(parser.parse_args().destination)
