import ctypes,struct
m=ctypes.CDLL('libm.so.6')
b=min(int(l.split('-')[0],16)-int(l.split()[2],16) for l in open('/proc/self/maps') if 'libm.so.6' in l)
for p,n in [(0xe6080,8),(0xe71e0,18),(0xe8280,9)]:
 print(hex(p))
 for i,v in enumerate(struct.unpack('<'+'d'*n,ctypes.string_at(b+p,n*8))):print(i,v.hex())
