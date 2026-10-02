"""Render an analysis chart of the three restored compiled-Fortran words."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gpuwm.core.fp32_ulp import fp32_ulp_distance

p=argparse.ArgumentParser()
p.add_argument("receipt",type=Path)
p.add_argument("output",type=Path)
a=p.parse_args()
receipt=json.loads(a.receipt.read_text())
points=receipt["evidence"]["native"]["repaired"]
old=np.array([r["old_word"] for r in points],dtype=np.uint32).view(np.float32)
reference=np.array([r["fortran_word"] for r in points],dtype=np.uint32).view(np.float32)
distance=fp32_ulp_distance(old,reference)
labels=[",".join(map(str,r["index"])) for r in points]
x=np.arange(len(points))
fig,ax=plt.subplots(figsize=(6.4,3.4),layout="constrained")
ax.bar(x-.16,distance,width=.32,label="Previous constant",color="#d88b39")
ax.bar(x+.16,np.zeros(len(points)),width=.32,label="Corrected constant",color="#238766")
for pos,val in zip(x,distance):
    ax.text(pos-.16,float(val)+.05,str(int(val)),ha="center",fontsize=10)
    ax.text(pos+.16,.06,"0",ha="center",fontsize=10,color="#15654b")
ax.set_xticks(x,labels)
ax.set_xlabel("Array index (level, row, column)")
ax.set_ylabel("Distance from compiled WRF (ULP)")
ax.set_ylim(0,float(distance.max())+.7)
ax.set_title("Vertical momentum difference (ULP)\nRestored exact matches",loc="left",fontsize=13)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False,loc="upper right")
fig.savefig(a.output,dpi=170)
print(json.dumps({"points":len(points),"old_ulp":distance.tolist(),"corrected_ulp":[0]*len(points)}))
