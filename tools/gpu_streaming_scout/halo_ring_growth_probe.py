"""Halo ring growth over a METIS decomposition (CPU only, seconds).

Answers "how big does one partition get at K halo rings, and how much
redundant compute does that buy" for a hex mesh.  Decides the (P, K) pair
for any partition-streaming or multi-device plan.

    python halo_ring_growth_probe.py GRAPH_INFO "PART.{P}" 2,4,8,16 128 out.json

GRAPH_INFO is an MPAS ``*.graph.info`` (1-based cell adjacency).  The second
argument is a format template for the matching ``*.graph.info.part.P`` files.
Output JSON carries, per P and per ring depth 0..K: ``max_frac`` (largest
partition as a fraction of the global mesh -- what must fit the card) and
``redundancy`` (sum of locals over global -- the wasted-compute multiplier).

Measured for task #260; see Downloads/GPUWM-HEX-SINGLE-CARD-STREAMING.md.
"""

import sys, json
import numpy as np

def load_graph(path):
    raw = open(path).read().split("\n")
    n = int(raw[0].split()[0])
    counts = np.empty(n, dtype=np.int64); rows=[]
    for i in range(n):
        nb = np.fromstring(raw[i+1], dtype=np.int64, sep=" ") - 1
        counts[i]=nb.size; rows.append(nb)
    offs = np.zeros(n+1, dtype=np.int64); np.cumsum(counts, out=offs[1:])
    return n, offs, np.concatenate(rows)

def neigh(F, offs, flat):
    c = offs[F+1]-offs[F]
    tot = int(c.sum())
    if tot==0: return np.zeros(0,dtype=np.int64)
    starts = np.repeat(offs[F], c)
    base = np.repeat(np.cumsum(c)-c, c)
    idx = starts + (np.arange(tot, dtype=np.int64) - base)
    return flat[idx]

def rings(owned, offs, flat, n, K):
    seen = np.zeros(n, dtype=bool); seen[owned]=True
    F = owned; cum=[owned.size]; tot=owned.size
    for k in range(1,K+1):
        cand = np.unique(neigh(F, offs, flat))
        cand = cand[cand>=0]
        fresh = cand[~seen[cand]]
        seen[fresh]=True
        tot += fresh.size; cum.append(tot); F=fresh
        if fresh.size==0:
            cum.extend([tot]*(K-k)); break
    return cum

graph, tpl, Plist, K = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4])
n, offs, flat = load_graph(graph)
res={"graph":graph.split("/")[-1],"n_cells":n,"by_P":{}}
for P in Plist:
    try: part=np.loadtxt(tpl.format(P=P), dtype=np.int64)
    except Exception as e: res["by_P"][str(P)]={"error":str(e)[:120]}; continue
    per=[rings(np.flatnonzero(part==p), offs, flat, n, K) for p in range(P)]
    M=np.array(per, dtype=np.int64)  # (P, K+1)
    res["by_P"][str(P)]={"owned_max":int(M[:,0].max()),
        "rings":{str(k):{"max_local":int(M[:,k].max()),
                          "max_frac":round(float(M[:,k].max())/n,5),
                          "redundancy":round(float(M[:,k].sum())/n,4)} for k in range(K+1)}}
    print(json.dumps({"P":P,"ok":True}), flush=True)
open(sys.argv[5],"w").write(json.dumps(res))
print("WROTE "+sys.argv[5])
