"""Adversarial controls for classification of retained validator state."""
import copy,hashlib,io,json,runpy,sys
from contextlib import redirect_stdout
from pathlib import Path
from receipt_io import receipt_bytes
from unittest.mock import patch
root=Path(__file__).resolve().parents[2]
script=root/'tools/native_step/compare_boundary_retirement.py'
files=[root/'receipts'/n for n in ('boundary-trajectory-strict-python-01.json','boundary-trajectory-strict-native-01.json','boundary-trajectory-strict-comparison-01.json','boundary-health-strict-comparison-01.json')]
base={p:json.loads(receipt_bytes(p)) for p in files};result=[]

def run_case(name,change,expected):
 docs=copy.deepcopy(base);change(docs)
 payloads={p:json.dumps(v).encode() for p,v in docs.items()}
 docs[files[3]]['receipts']={str(p.relative_to(root)):hashlib.sha256(payloads[p]).hexdigest() for p in files[:2]}
 payloads[files[3]]=json.dumps(docs[files[3]]).encode()
 written=[]
 def read_bytes(path):return payloads[path]
 def read_text(path,*args,**kwargs):return payloads[path].decode()
 original_read_bytes=Path.read_bytes;original_read_text=Path.read_text
 def rb(path):return read_bytes(path) if path in payloads else original_read_bytes(path)
 def rt(path,*args,**kwargs):return read_text(path) if path in payloads else original_read_text(path,*args,**kwargs)
 caught=None
 with patch.object(Path,'read_bytes',rb),patch.object(Path,'read_text',rt),patch.object(Path,'exists',lambda p:False),patch.object(Path,'write_text',lambda p,s,*a,**k:written.append(s)),patch.object(sys,'argv',[str(script),'--attempt','01','--numerics','strict']),redirect_stdout(io.StringIO()):
  try:runpy.run_path(str(script),run_name='__main__')
  except AssertionError as error:caught=repr(error)
 assert (caught is None)==expected,(name,caught)
 result.append(dict(case=name,accepted=caught is None,expected_accept=expected,error=caught,output_count=len(written)))

def model_corruption(docs):
 rows=docs[files[1]]['endpoints']['400']['arrays']
 row=next(x for x in rows if x['name']=='root.model.root.state.physics.pbl_tendencies.ru');row['sha256']='0'*64

def cached_corruption(docs):
 row=next(x for x in docs[files[0]]['endpoints']['400']['arrays'] if x['name']=='root.validators.1.fields[21].values');row['sha256']='0'*64

def pointer_corruption(docs):
 row=next(x for x in docs[files[1]]['endpoints']['400']['arrays'] if x['name']=='root.model.root.state._scratch.integration_health_field_ptr');row['pointer_targets'][21]=['root.model.root.state.physics.pbl_tendencies.rv']

def unknown_host(docs):docs[files[2]]['endpoints']['400']['host_differences']['root.model.root.state.elapsed_seconds']={'python':1,'native':2}
def invalid_counter(docs):docs[files[2]]['endpoints']['400']['host_differences']['root.model.root.state.physics.radiation_callable._cuda_sw._scratch_by_stream.0.allocations']['native']=-1
run_case('recorded exact scoped comparison',lambda d:None,True)
for name,mutator in [('current held state changed',model_corruption),('old validator snapshot changed',cached_corruption),('descriptor changed to other current field',pointer_corruption),('new semantic host difference',unknown_host),('invalid allocation counter',invalid_counter)]:run_case(name,mutator,False)
def retired_corruption(docs):
 row=next(x for x in docs[files[1]]['endpoints']['406']['arrays'] if x['name']=='root.validators.1.fields[21].values');row['sha256']='0'*64
run_case('retired native cache changed',retired_corruption,False)
output=root/'receipts/boundary-retirement-comparison-cpu.json' ;assert not output.exists();output.write_text(json.dumps(dict(status='PASS',cases=result),indent=2)+'\n');print(json.dumps(dict(status='PASS',controls=len(result),refusals=sum(not r['accepted'] for r in result))))
