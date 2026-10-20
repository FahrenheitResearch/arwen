"""Differential CPU predicate check against six pinned original verdicts."""
import ast,copy,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from deferred_annotations import SITES,PINS,validate_sources
ROOT=Path(__file__).resolve().parents[2];validate_sources(ROOT);started=time.perf_counter()
rows=[]
for site,rule in SITES.items():
 filename,name,line=site;path=ROOT/'gpuwm/core'/filename
 nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and (n.name,n.lineno)==(name,line)];assert len(nodes)==1
 ns={'np':np,'name':'re_cloud'};exec(compile(ast.Module(body=[copy.deepcopy(nodes[0])],type_ignores=[]),str(path),'exec'),ns);verdict=ns[name]
 if rule==2:
  patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7f800001,0xff800001]
  threshold=np.float32(.001)
  values=list(np.array(patterns,np.uint32).view(np.float32))+[np.nextafter(threshold,np.float32(-np.inf)),threshold,np.nextafter(threshold,np.float32(np.inf))]
  typed=[np.asarray(x,np.float32) for x in values]
 elif name=='_cldprmc_verdict':typed=[np.asarray([x],np.int32) for x in (0,1,-1,7,2147483647,-2147483648)]
 else:typed=[np.asarray(x,np.bool_) for x in (False,True)]
 for value in typed:
  with np.errstate(invalid='ignore'):
   flat=value.ravel().astype(np.float64);part=flat.astype(value.dtype).reshape(value.shape)
  expected=None
  try:verdict(np.asarray(part))
  except Exception as e:expected={'type':type(e).__name__,'text':str(e)}
  bits=int(flat.view(np.uint64)[0]);rows.append({'site':site,'rule':rule,'dtype':value.dtype.str,'input_hex':value.tobytes().hex(),'binary64_bits':f'{bits:016x}','expected_failure':expected is not None,'exception':expected})
stream=''.join(f"{r['rule']} {r['binary64_bits']}\n" for r in rows)
cmd=[str(ROOT/'receipts/test-numeric-rules'),'--stream'];proc=subprocess.run(cmd,input=stream.encode('ascii'),capture_output=True,check=True);actual=[bool(int(x)) for x in proc.stdout.decode('ascii').split()];assert len(actual)==len(rows)  # a byte pipe (release lint T4)
for row,got in zip(rows,actual):row['native_failure']=got;assert got==row['expected_failure'],row
assert 'cupy' not in sys.modules,'CPU comparison initialized CuPy'
receipt={'status':'PASS','cases':len(rows),'original_verdicts':len(SITES),'differences':0,'rows':rows,'source':PINS,'gpu_calls':0,'wall_s':time.perf_counter()-started,'command':cmd,'implementation_sha256':hashlib.sha256((ROOT/'gpuwm/native_step/recorder.c').read_bytes()).hexdigest(),'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT/'receipts/deferred-predicates-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','source')}))
