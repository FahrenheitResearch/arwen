import ctypes as c,hashlib,json
from pathlib import Path
r=Path(__file__).parent
m=c.CDLL('libm.so.6');ours=c.CDLL(str(r/'table_identity.so'));ours.g64_table.restype=c.c_void_p
base=min(int(l.split('-')[0],16)-int(l.split()[2],16) for l in open('/proc/self/maps') if 'libm.so.6' in l)
result=[]
for f,name,off,n in [(0,'exp',0xe6130,256*8),(1,'log',0xe7270,128*16),(2,'pow',0xe82c8,128*32)]:
 a=c.string_at(ours.g64_table(f),n);b=c.string_at(base+off,n)
 result.append({'function':name,'installed_offset':hex(off),'bytes':n,'mismatching_bytes':sum(x!=y for x,y in zip(a,b)),'arm_sha256':hashlib.sha256(a).hexdigest(),'installed_sha256':hashlib.sha256(b).hexdigest()})
print(json.dumps(result,indent=2))
raise SystemExit(any(x['mismatching_bytes'] for x in result))
