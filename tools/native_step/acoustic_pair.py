"""Run only inside the canonical exclusive GPU lease."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
root=Path(__file__).resolve().parents[2]
os.chdir(root)
def contexts():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return p.stdout.strip().splitlines()
evidence={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':contexts(),'samples':[],'runs':[]}
assert not evidence['before'],evidence['before']
try:
 for mode in ('strict','fast'):
  cmd=['timeout','--signal=TERM','--kill-after=10','240','bash','tools/native_step/run_acoustic.sh','--numerics',mode]
  started=time.perf_counter()
  with (root/f'receipts/acoustic-{mode}.log').open('w') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while process.poll() is None:
    evidence['samples'].append({'mode':mode,'monotonic_s':time.monotonic(),'contexts':contexts()})
    time.sleep(1)
  evidence['runs'].append({'mode':mode,'argv':cmd,'returncode':process.returncode,'wall_s':time.perf_counter()-started})
  print(json.dumps(evidence['runs'][-1]),flush=True)
  if process.returncode:break
finally:
 evidence['after']=contexts()
 (root/'receipts/acoustic-pair-contexts.json').write_text(json.dumps(evidence,indent=2)+'\n')
assert all(x['returncode']==0 for x in evidence['runs']),evidence['runs']
assert len(evidence['runs'])==2
assert not evidence['after'],evidence['after']
