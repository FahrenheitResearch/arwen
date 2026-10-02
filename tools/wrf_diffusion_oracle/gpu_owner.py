"""Reserve a short shared oracle job without overlapping an exclusive hold."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
from pathlib import Path
import re
import sys

def blockers(owner,lane,now=None):
    now=now or datetime.now(timezone.utc)
    latest={}
    for line in Path(owner).read_text().splitlines():
        fields=line.split()
        if len(fields)<3: continue
        if fields[2] in ("start","release"): latest[fields[0]]=line
    blocked=[]
    for name,line in latest.items():
        if name==lane or line.split()[2]=="release": continue
        if "shared" in line.lower() and "exclusive" not in line.lower() and not re.search(r"\bcut\b",line,re.I): continue
        pid=re.search(r"\bpid\s+(\d+)",line)
        bound=re.search(r"\bbounded\s+(\d+)\s+min",line)
        try:
            start=datetime.fromisoformat(line.split()[1].replace("Z","+00:00"))
            expired=bool(bound and now>start+timedelta(minutes=int(bound[1])))
        except ValueError:
            expired=False
        gone=bool(pid and not Path("/proc",pid[1]).exists())
        if not (expired and gone): blocked.append(name)
    return blocked

def reserve(owner,lane,pid,minutes,description):
    owner=Path(owner)
    with (owner.parent/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        active=blockers(owner,lane)
        if active:
            print("Exclusive GPU reservation prevents overlapping this oracle job: "+", ".join(active),file=sys.stderr)
            return 3
        with owner.open("a") as output:
            output.write(f"{lane} {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} start pid {pid} bounded {minutes} min ({description}; shared)\n")
    return 0

def release(owner,lane):
    owner=Path(owner)
    with (owner.parent/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with owner.open("a") as output:
            output.write(f"{lane} {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} release\n")
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("owner",type=Path);p.add_argument("lane")
    p.add_argument("pid",type=int);p.add_argument("minutes",type=int);p.add_argument("description")
    p.add_argument("--release",action="store_true")
    a=p.parse_args();raise SystemExit(release(a.owner,a.lane) if a.release else reserve(a.owner,a.lane,a.pid,a.minutes,a.description))
