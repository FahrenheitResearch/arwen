import ctypes,struct
m=ctypes.CDLL('libm.so.6');p=m.pow;p.argtypes=[ctypes.c_double]*2;p.restype=ctypes.c_double
V=[0x7ff0000000000001,0x7ff8000000000123,0xfff8000000000456]
def f(u):return struct.unpack('d',struct.pack('Q',u))[0]
def u(x):return hex(struct.unpack('Q',struct.pack('d',x))[0])
for x in V:
 for y in V:print(hex(x),hex(y),u(p(f(x),f(y))))
