"""Record exact-chain constructor and compiler census through their real owners.

Run with GPUWM_NO_LOCAL_GPU=1 GPUWM_WRF_EXACT=0. No device is opened.
The output records actual descriptors, compiler sites and allocation itemization.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,pprint,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
def load(name):
 path=ROOT/'tests'/('test_'+name+'.py')
 spec=importlib.util.spec_from_file_location('census_'+name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def span(text,node):
 lines=text.splitlines(keepends=True)
 return sum(map(len,lines[:node.lineno-1]))+node.col_offset,sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--write',action='store_true');a=ap.parse_args();receipt={}
 h=load('health');from gpuwm.core import physics,state
 physics.cp=state.cp=np;physics.DTYPE=state.DTYPE=np.float32
 cfg=h._inventory_config();s=state.DomainState(cfg);h._attach_real_driver(s,cfg,physics,np)
 fields=h.collect_state_fields(s,backend='gpu');names=[f.name for f in fields]
 receipt['health']={'names':names,'dtypes':{f.name:str(f.values.dtype) for f in fields},'added_names':sorted(set(names)-set(h._REAL_DRIVER_INVENTORY))}
 if a.write:
  p=ROOT/'tests/test_health.py';text=p.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_REAL_DRIVER_INVENTORY' for t in n.targets));start,end=span(text,node.value)
  text=text[:start]+'tuple("""\n'+'\n'.join(' '.join(names[i:i+6]) for i in range(0,len(names),6))+'\n""".split())'+text[end:];p.write_text(text)
 alloc=load('physics_allocation_inventory');live=alloc._live_inventory();new=sorted(set(live)-set(alloc._PHYSICS_ALLOCATION_INVENTORY))
 assert set(new) <= {'gpuwm/core/physics.py','gpuwm/core/wsm6_default.py'},new
 assert all(live[k]==v for k,v in alloc._PHYSICS_ALLOCATION_INVENTORY.items())
 receipt['new_allocation_rows']={k:live[k] for k in new}
 if a.write:
  p=ROOT/'tests/test_physics_allocation_inventory.py';text=p.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_PHYSICS_ALLOCATION_INVENTORY' for t in n.targets));_,end=span(text,node.value)
  extra='\n    # Physics constructor allocations are fixed domain/soil/history carriers,\n    # priced by physics_inventory_census. Default WSM6 delegates its kernels\n    # to the common loader and has no per-call allocation.\n'
  for k in new:extra+='    '+repr(k)+': '+repr(live[k])+',\n'
  p.write_text(text[:end-1]+extra+text[end-1:])
 km=load('kernel_manifest');sites={'rawmodule':[],'direct_nvrtc':[]}
 for rel in km.SITE_FILES:
  tree=ast.parse((ROOT/rel).read_text())
  for kind,test in [('rawmodule',km._is_rawmodule),('direct_nvrtc',km._is_nvrtc)]:
   sites[kind]+=[{'path':rel,'line':n.lineno,'expression':ast.unparse(n)} for n in ast.walk(tree) if test(n)]
 receipt['compiler_sites']=sites
 if a.write:
  p=ROOT/'tests/test_kernel_manifest.py';text=p.read_text()
  for name,kind in [('EXPECTED_SITE_COUNT','rawmodule'),('EXPECTED_NVRTC_SITE_COUNT','direct_nvrtc')]:text,n=re.subn(r'(?m)^'+name+r' = \d+$',name+' = '+str(len(sites[kind])),text);assert n==1
  p.write_text(text)
 pre=load('preflight');exp=pre.experiment_from_run_config(pre.load_config(pre.CONFIG_D01),pre.datetime(1974,4,3,12));est=pre.pf.estimate_experiment(exp);d=est.domains[0]
 cats={c:d.category_bytes(c) for c in ('state','physics','scratch','lbc','nest','transient')};receipt['preflight']={'categories':cats,'resident_bytes':d.resident_bytes,'items':[repr(item) for item in d.items] if hasattr(d,'items') else [],'previous_resident_bytes':1523343280,'added_resident_bytes':d.resident_bytes-1523343280}
 if a.write:
  p=ROOT/'tests/test_preflight.py';text=p.read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_estimate_domain_itemization_pins');assertion=next(n for n in ast.walk(fn) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and getattr(n.test.left,'id',None)=='by_cat');start,end=span(text,assertion.test.comparators[0]);text=text[:start]+pprint.pformat(cats,sort_dicts=False)+text[end:]
  text=text.replace('assert d01.resident_bytes == 1523343280','assert d01.resident_bytes == '+str(d.resident_bytes)).replace('assert payload["domains"]["d01"]["resident_bytes"] == 1523343280','assert payload["domains"]["d01"]["resident_bytes"] == '+str(d.resident_bytes))
  text=text.replace('ratio = est.domains[0].resident_bytes / measured','assert est.domains[0].resident_bytes == '+str(d.resident_bytes)+'\n    # Compare the recorded pre-intake residency to its historical measurement.\n    # New history carriers are separately priced above; this is no new pool measurement.\n    ratio = 1523343280 / measured')
  p.write_text(text)
 a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'health_fields':len(names),'new_allocation_rows':receipt['new_allocation_rows'],'compiler_sites':{k:len(v) for k,v in sites.items()},'preflight':receipt['preflight']},indent=2))
if __name__=='__main__':main()
