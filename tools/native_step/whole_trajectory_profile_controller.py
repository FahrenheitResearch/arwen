"""Separate CUDA-only profiles under one exclusive monitored lease."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
def contexts():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(p.split(',')[0]),row=p) for p in s.strip().splitlines() if p.strip()]
def descendants(pid):
 pairs={int(p):int(parent) for p,parent in (line.split() for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())}
 owned={pid};changed=True
 while changed:
  before=len(owned);owned.update(p for p,pp in pairs.items() if pp in owned);changed=len(owned)>before
 return sorted(owned)
pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'tools/native_step/measure_whole_trajectory.py',Path(__file__).resolve(),root/'receipts/local-binding.json',root/'receipts/libawstep-value.so']}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':[],'source':pins,'status':'NOT_ACCEPTED'}
try:
 for arm in ('profile',):
  entry={'arm':arm,'before':contexts(),'samples':[]};r['arms'].append(entry);assert not entry['before']
  cmd=['timeout','--signal=TERM','--kill-after=10','240','bash','tools/native_step/profile_whole_trajectory.sh'];entry['argv']=cmd
  with (root/'receipts/whole-trajectory-profile-driver.log').open('w') as log:
   process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
   entry['controller_child_pid']=process.pid
   while process.poll() is None:
    rows=contexts();owned=descendants(process.pid)
    entry['samples'].append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_s':time.monotonic(),'contexts':rows,'owned_pids':owned,'foreign':[x for x in rows if x['pid'] not in owned]});time.sleep(.5)
  entry.update(returncode=process.returncode,after=contexts())
  assert process.returncode==0 and not entry['after'] and not any(x['foreign'] for x in entry['samples']),entry
 r['status']='EMPTY_ENDPOINTS_NO_SAMPLED_FOREIGN_CONTEXT'
finally:
 (root/'receipts/whole-trajectory-profile-contexts.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['python3','tools/native_step/reduce_whole_trajectory_profile.py'],cwd=root,check=True)
