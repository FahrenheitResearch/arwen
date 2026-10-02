"""Plot numerical comparison receipts, without weather-field rendering."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def draw(initial,fixed,output):
    before=json.loads(Path(initial).read_text())
    after=json.loads(Path(fixed).read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),layout="constrained")
    labels=[f"Initial\n{before['cases']} cases",f"Fixed\n{after['cases']} cases"]
    values=[100*before['different_words']/before['words'],100*after['different_words']/after['words']]
    colors=["#ba4040","#207887"]
    axes[0].bar(labels,values,color=colors,width=.52)
    axes[0].set(title="Sixth-order tendency words",ylabel="Different from Fortran (%)",ylim=(0,27))
    for x,v,row in zip(range(2),values,(before,after)):
        axes[0].text(x,v+.8,f"{row['different_words']:,} / {row['words']:,}",ha="center",fontsize=9)
    axes[1].bar(["Initial HFX","Fixed HFX"],[9955411,1],color=colors,width=.52)
    axes[1].set_yscale("log")
    axes[1].set(title="Prescribed-flux HFX diagnostic",ylabel="Maximum ULP distance",ylim=(.5,5e7))
    for x,v in enumerate((9955411,1)):
        axes[1].text(x,v*1.45,f"{v:,}",ha="center",fontsize=9)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Compiled WRF v4.7.1 comparisons",fontsize=14)
    fig.savefig(output,dpi=180)
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("initial",type=Path);p.add_argument("fixed",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();draw(a.initial,a.fixed,a.output)
