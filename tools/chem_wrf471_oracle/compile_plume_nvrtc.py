"""Compile the assembled plume source with NVRTC without loading CUDA.

Calls libnvrtc directly through ctypes. No CUDA runtime, context, driver,
device enumeration or module loading occurs. PTX is a compile-only receipt.
"""
import argparse
import ctypes as c
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('library'); p.add_argument('source'); p.add_argument('output'); p.add_argument('--arch',default='compute_89')
a=p.parse_args(); nv=c.CDLL(a.library)
def fn(name,types):
    f=getattr(nv,name); f.argtypes=types; f.restype=c.c_int; return f
create=fn('nvrtcCreateProgram',[c.POINTER(c.c_void_p),c.c_char_p,c.c_char_p,c.c_int,c.c_void_p,c.c_void_p])
compile=fn('nvrtcCompileProgram',[c.c_void_p,c.c_int,c.POINTER(c.c_char_p)])
logsize=fn('nvrtcGetProgramLogSize',[c.c_void_p,c.POINTER(c.c_size_t)])
getlog=fn('nvrtcGetProgramLog',[c.c_void_p,c.c_void_p])
ptxsize=fn('nvrtcGetPTXSize',[c.c_void_p,c.POINTER(c.c_size_t)])
getptx=fn('nvrtcGetPTX',[c.c_void_p,c.c_void_p])
destroy=fn('nvrtcDestroyProgram',[c.POINTER(c.c_void_p)])
program=c.c_void_p(); status=create(c.byref(program),Path(a.source).read_bytes(),b'chem_plumerise.cu',0,None,None)
assert status==0,status
try:
    flags=[b'--std=c++17',('--gpu-architecture='+a.arch).encode(),b'--fmad=false']
    status=compile(program,len(flags),(c.c_char_p*len(flags))(*flags))
    n=c.c_size_t(); logsize(program,c.byref(n)); buf=c.create_string_buffer(n.value); getlog(program,buf)
    log=buf.value.decode(); Path(a.output+'.log').write_text(log)
    for line in log.splitlines():
        if 'error:' in line: print(line,flush=True)
    if status: raise RuntimeError('NVRTC rejected the plume source; uncompilable device physics cannot be integrated')
    ptxsize(program,c.byref(n)); buf=c.create_string_buffer(n.value); getptx(program,buf)
    Path(a.output).write_bytes(buf.value)
    print('compile-only PTX bytes',len(buf.value),flush=True)
finally: destroy(c.byref(program))
