"""Apply the complete series to a lane-owned index of the full engine base.

The engine-base object directory is borrowed read-only. No checkout, commit,
or write in that repository is performed. All new objects/index files stay in
the owned verification repository. This is a packaging check, not physics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

ENGINE_BASE="4d1079249ad0cc2553a9632ac9ef267553913dd1"
SLIM_BASE="3836f68c4fa77bd60ecf527bb408ee1832b8e359"


def run(args):
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True)
    if p.returncode:raise RuntimeError(f"{args!r}\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-repo",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    source=Path(__file__).resolve().parents[2]
    a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)
    repo=a.work/"repo";series=a.work/"series"
    if repo.exists():raise ValueError("verification repository already exists")
    objects=run(["git","-C",a.base_repo,"rev-parse","--path-format=absolute","--git-path","objects"])
    run(["git","init","-q",repo])
    (repo/".git/objects/info/alternates").write_text(objects+"\n")
    commands=[["git","-C",repo,"read-tree",ENGINE_BASE]]
    run(commands[-1])
    run(["git","-C",source,"format-patch",f"{SLIM_BASE}..HEAD","-o",series])
    for patch in sorted(series.glob("*.patch")):
        commands.append(["git","-C",repo,"apply","--cached",patch])
        run(commands[-1])
    names=run(["git","-C",source,"diff","--name-only",SLIM_BASE,"HEAD"]).splitlines()
    rows=[]
    for name in names:
        expected=run(["git","-C",source,"ls-tree","HEAD",name]).split()[2]
        actual=run(["git","-C",repo,"ls-files","-s","--",name]).split()[1]
        if expected!=actual:raise ValueError(f"applied blob differs: {name}")
        rows.append({"path":name,"blob":actual})
    receipt={"engine_base":ENGINE_BASE,"slim_base":SLIM_BASE,
             "head":run(["git","-C",source,"rev-parse","HEAD"]),
             "patches":len(list(series.glob("*.patch"))),"matching_blobs":rows,
             "commands":commands,"base_repo_modified":False}
    a.receipt.write_text(json.dumps(receipt,indent=2,default=str))
    print(f"Applied {receipt['patches']} patches to full-base index; {len(rows)} final blobs match.")


if __name__=="__main__":main()
