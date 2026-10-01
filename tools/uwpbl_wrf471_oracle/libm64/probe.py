import ctypes,struct,json
m=ctypes.CDLL('libm.so.6')
base=min(int(l.split('-')[0],16)-int(l.split()[2],16) for l in open('/proc/self/maps') if 'libm.so.6' in l)
for n in ['exp','log','pow','cos','acos','__exp_finite','__log_finite','__pow_finite']:
 try:p=ctypes.cast(getattr(m,n),ctypes.c_void_p).value
 except AttributeError:continue
 print(n,hex(p-base))
# public wrappers have resolved PLT GOT slots. Read them after one call.
for n,args,plt in [('exp',[1.],0x125d0),('log',[2.],0x124e0),('pow',[2.,3.],0x12510),('acos',[0.],0x124c0)]:
 f=getattr(m,n);f.argtypes=[ctypes.c_double]*len(args);f.restype=ctypes.c_double;f(*args)
 b=ctypes.string_at(base+plt,16)
 i=b.find(b'\xff\x25')
 if i>=0:
  slot=base+plt+i+6+struct.unpack('<i',b[i+2:i+6])[0]
  target=ctypes.c_void_p.from_address(slot).value
  print(n,'resolved_core',hex(target-base),'plt',hex(plt),'got',hex(slot-base))
