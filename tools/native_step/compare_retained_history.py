"""Compare new full history against retained native comparator base hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('forecast',type=Path);p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
refs={x.name[:-5]:x for x in a.reference.glob('wrfout_*.json')}
files={x.name:x for x in (a.forecast/'wrfout').glob('wrfout_*') if x.is_file() and x.suffix!='.json'}
assert refs and refs.keys()==files.keys()
r=dict(reference=str(a.reference),forecast=str(a.forecast),method='new typed-byte SHA256 against retained base side of native identity comparator',files=[],typed_words=0,mismatched_variables=0,schema_differences=[])
for name in sorted(refs):
 ref=json.loads(refs[name].read_text());assert ref['schema']=='gpuwm.identity-native.v1'
 assert '/AC-base-fast-warm-local/forecast/' in ref['argv'][-2]
 expected={x['base']['name']:x['base'] for x in ref['variables']}
 row=dict(name=name,reference_file_sha256=hashlib.sha256(refs[name].read_bytes()).hexdigest(),file_sha256=hashlib.file_digest(files[name].open('rb'),'sha256').hexdigest(),variables=[])
 with Dataset(files[name]) as f:
  f.set_auto_maskandscale(False)
  if f.variables.keys()!=expected.keys():r['schema_differences'].append(name+': variables')
  for key in sorted(set(f.variables)&set(expected)):
   x=np.ascontiguousarray(f[key][:]);e=expected[key];digest=hashlib.sha256(x.tobytes()).hexdigest()
   schema=list(x.shape)==e['shape'] and x.dtype==np.dtype(e['dtype']) and x.nbytes==e['bytes']
   if not schema:r['schema_differences'].append(name+': '+key)
   equal=schema and digest==e['sha256'];r['typed_words']+=x.size;r['mismatched_variables']+=int(not equal)
   row['variables'].append(dict(name=key,shape=list(x.shape),dtype=str(x.dtype),words=int(x.size),base_sha256=e['sha256'],driver_sha256=digest,equal=equal))
 r['files'].append(row)
r['status']='PASS' if r['mismatched_variables']==0 and not r['schema_differences'] else 'FAIL'
if r['status']=='PASS':r['different_words']=0
else:r['different_words']=None
r['limitation']='Original base raw arrays were deleted by their owner; mismatched hashes cannot give word counts or magnitudes.'
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='files'}))
