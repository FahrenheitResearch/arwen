"""New AC port checks: exact source preservation, child binding and conditional guards. CPU only."""
import ast,ctypes as C,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from radiation_capture import RecordedRadiation, transformed_method
from radiation_deferred import Rule,Guard,DeferredNativeChecks
source=ROOT/'gpuwm/core/rrtmg_legacy.py';raw=source.read_bytes();tree=transformed_method(raw);compile(tree,str(source),'exec')
original=ast.parse(raw);cls=next(x for x in original.body if isinstance(x,ast.ClassDef) and x.name=='RRTMGLegacyRadiation');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__call__')
# Exactly preserve this branch's device vertical route and child snapshot path.
checks=[]
for line in (1922,1923,2001):
 a=next(x for x in ast.walk(method) if getattr(x,'lineno',0)==line and isinstance(x,(ast.Assign,ast.If)))
 b=next(x for x in ast.walk(tree) if getattr(x,'lineno',0)==line and isinstance(x,type(a)))
 if line==2001:a,b=ast.Module(body=a.body,type_ignores=[]),ast.Module(body=b.body,type_ignores=[])
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 checks.append('unchanged_original_AST_'+str(line))
try:transformed_method(raw+b'\n')
except RuntimeError:checks.append('changed_source_refuses')
else:raise AssertionError('changed source accepted')
obj=RecordedRadiation.__new__(RecordedRadiation);obj.cp=np;obj.cfg=NS(nx=3,ny=2,nz=4)
source_array=np.zeros((4,2,3),np.float32);provider=lambda:None
owner=NS(mode='parent-interpolated');driver=NS(cam_ozone=owner,call_counts={'cam_ozone':1},o3rad=source_array)
obj.state=NS(physics=driver,fnm=np.ones(4,np.float32),fnp=np.ones(4,np.float32));obj.vertical=(obj.state.fnm,obj.state.fnp);obj.provider=provider;obj.adapter=NS(_ozone_provider=provider);obj.parent_ozone=source_array
assert obj._parent_ozone() is source_array;obj.check_bindings();checks.append('child_device_binding_admitted')
def refusal(label,mutate,restore,phrase):
 mutate()
 try:obj.check_bindings()
 except RuntimeError as e:assert phrase in str(e);checks.append(label)
 else:raise AssertionError(label)
 finally:restore()
refusal('changed_carrier_refuses',lambda:setattr(driver,'o3rad',source_array.copy()),lambda:setattr(driver,'o3rad',source_array),'carrier identity')
refusal('missing_force_refuses',lambda:driver.call_counts.update(cam_ozone=0),lambda:driver.call_counts.update(cam_ozone=1),'before parent')
refusal('host_provider_route_refuses',lambda:setattr(owner,'mode','root'),lambda:setattr(owner,'mode','parent-interpolated'),'custom host')
refusal('provider_rebind_refuses',lambda:setattr(obj.adapter,'_ozone_provider',lambda:None),lambda:setattr(obj.adapter,'_ozone_provider',provider),'provider identity')
refusal('vertical_rebind_refuses',lambda:setattr(obj.state,'fnm',obj.state.fnm.copy()),lambda:setattr(obj.state,'fnm',obj.vertical[0]),'weight binding')
lib=C.CDLL(str(ROOT/'receipts/libradiation-composed-host.so'));fn=lib.host_numeric_guard;fn.argtypes=[C.c_void_p,C.c_size_t];fn.restype=C.c_int
words=(C.c_double*2)(1.0,0.0);count=C.c_uint64(0);rules=(Rule*2)(Rule(0,1,0,0,5,C.addressof(count)),Rule(1,1,0,0,0,None));fr=C.c_uint64();fw=C.c_uint64();g=Guard(C.addressof(words),2,C.addressof(rules),2,C.addressof(fr),C.addressof(fw))
for value,rc in [(0,0),(5,0),(6,-1007),(0,0)]:
 count.value=value;assert fn(C.byref(g),C.sizeof(g))==rc
 assert fr.value==(0 if rc else 2**64-1);checks.append('active_count_'+str(value)+'_rc_'+str(rc))
words[1]=1.;assert fn(C.byref(g),C.sizeof(g))==-1007 and fr.value==1;checks.append('inactive_first_still_checks_later_rule')
# The Python collector assigns the same live count to only new chunk verdicts.
d=DeferredNativeChecks.__new__(DeferredNativeChecks);sink=[1];activity={};d._sinks=[(sink,activity)]
with d.active_chunk(count,5):sink.extend([2,3])
assert activity=={1:(count,5),2:(count,5)};checks.append('active_chunk_scope_exact')
result={'status':'PASS','scope':__doc__,'checks':checks,'count':len(checks),'gpu_calls':0,'source_sha256':hashlib.sha256(raw).hexdigest(),'generated_sha256':hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),'helper_sha256':hashlib.sha256((ROOT/'receipts/libradiation-composed-host.so').read_bytes()).hexdigest(),'command':'CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -B tools/native_step/check_radiation_port.py'}
(ROOT/'receipts/radiation-port-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
