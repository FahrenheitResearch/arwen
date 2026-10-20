"""New actual-comparator negative controls: absent host fields are not null."""
import datetime,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'tools/native_step/compare_whole_trajectory.py'
before=source.read_text()
needle="if k not in wall_names and k not in compile_names and l['host'].get(k)!=r['host'].get(k)"
replacement="if k not in wall_names and k not in compile_names and (k not in l['host'] or k not in r['host'] or l['host'][k]!=r['host'][k])"
assert before.count(needle)==1
after=before.replace(needle,replacement)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':sys.argv,'source_before_sha256':hashlib.sha256(before.encode()).hexdigest(),'source_after_sha256':hashlib.sha256(after.encode()).hexdigest(),'cases':[],'gpu_runs':0,'completed_controls_repeated':0,'deleted_files':[]}
stage=Path(tempfile.mkdtemp(prefix='host-presence-',dir=ROOT/'receipts'));script=stage/'tools/native_step/compare_whole_trajectory.py';script.parent.mkdir(parents=True);(stage/'receipts').mkdir()
original=ROOT/'receipts/referee-wall-intake/packet/receipts'
inputs={arm:json.loads((original/f'whole-trajectory-wall-fast-{arm}.json').read_text()) for arm in ('python','native')}
try:
 for label,key in [('model null','root.new_semantic_null'),('unknown diagnostic null','root.model._step_log._compile_announcement.unknown_future_null')]:
  assert all(key not in v['final']['host'] for v in inputs.values())
  inputs['native']['final']['host'][key]=None
  for arm,value in inputs.items():(stage/f'receipts/whole-trajectory-wall-fast-{arm}.json').write_text(json.dumps(value))
  row={'case':label,'added_key':key}
  for version,code,expected in [('before',before,0),('after',after,1)]:
   script.write_text(code)
   command=[sys.executable,'-B',str(script),'--prefix','whole-trajectory-wall','--numerics','fast']
   result=subprocess.run(command,capture_output=True,text=True)
   result_path=stage/'receipts/whole-trajectory-wall-fast-comparison.json';comparison=json.loads(result_path.read_text())
   row[version]={'argv':command,'returncode':result.returncode,'status':comparison['status'],'host_differences':comparison['host_differences'],'stderr':result.stderr,'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest()}
   assert result.returncode==expected,row
  receipt['cases'].append(row);del inputs['native']['final']['host'][key]
 source.write_text(after);receipt['status']='TWO_NEW_NULL_PRESENCE_CASES_FIXED'
finally:
 for p in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if p.is_file():
   receipt['deleted_files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});p.unlink()
  elif p.is_dir():p.rmdir()
 stage.rmdir()
 (ROOT/'receipts/host-presence-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt.get('status'),'cases':len(receipt['cases']),'deleted_files':len(receipt['deleted_files']),'deleted_bytes':sum(x['bytes'] for x in receipt['deleted_files'])}))
