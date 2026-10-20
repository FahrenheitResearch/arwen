"""One exclusive vegetation check with explicit model PID ancestry."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root)
def contexts():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(p.split(',')[0]),row=p) for p in s.strip().splitlines() if p.strip()]
def descendants(pid):
 pairs={int(p):int(parent) for p,parent in (line.split() for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())}
 owned={pid};changed=True
 while changed:
  before=len(owned);owned.update(p for p,pp in pairs.items() if pp in owned);changed=len(owned)>before
 return sorted(owned)
r={'before':contexts(),'samples':[],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
assert not r['before'],r['before']
cmd=['timeout','--signal=TERM','--kill-after=10','180','bash','tools/native_step/run_vegetation.sh','--numerics','fast']
with open('receipts/vegetation-fast.log','w') as log:
 process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 r['controller_child_pid']=process.pid;r['argv']=cmd
 while process.poll() is None:
  # Context PIDs are checked after ancestry and before ancestry again to avoid
  # a just-started child being misclassified from a stale process snapshot.
  rows=contexts();owned=descendants(process.pid)
  r['samples'].append({'contexts':rows,'owned_pids':owned,'foreign':[x for x in rows if x['pid'] not in owned]})
  time.sleep(.5)
r.update(returncode=process.returncode,after=contexts())
Path('receipts/vegetation-fast-contexts.json').write_text(json.dumps(r,indent=2)+'\n')
assert process.returncode==0,r
assert not r['after'] and not any(x['foreign'] for x in r['samples']),r
print(json.dumps({'returncode':process.returncode,'samples':len(r['samples']),'foreign':0}))
