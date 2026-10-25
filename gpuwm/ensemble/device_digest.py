# SPDX-License-Identifier: Apache-2.0
"""Device XXH64 tree digest over logical C-order bytes, seed zero.

Each leaf is at most 1 MiB. The root hashes the little-endian uint64 leaf
digests. Only the 8-byte root leaves the card. This is an integrity checksum,
not a cryptographic signature. Empty input has one empty leaf.
"""
from __future__ import annotations

BLOCK_BYTES = 1 << 20
SCHEMA = 'xxh64-tree-1MiB'

_SOURCE = r'''
typedef unsigned long long U64;
const U64 P1=11400714785074694791ULL, P2=14029467366897019727ULL;
const U64 P3=1609587929392839161ULL, P4=9650029242287828579ULL;
const U64 P5=2870177450012600261ULL;
__device__ U64 rot(U64 x,int r) { return (x<<r)|(x>>(64-r)); }
__device__ U64 rd64(const unsigned char *p) {
  U64 x=0; for(int i=0;i<8;i++) x|=((U64)p[i])<<(i*8); return x;
}
__device__ unsigned int rd32(const unsigned char *p) {
  unsigned int x=0; for(int i=0;i<4;i++) x|=((unsigned int)p[i])<<(i*8); return x;
}
__device__ U64 round64(U64 a,U64 x) { return rot(a+x*P2,31)*P1; }
__device__ U64 merge64(U64 a,U64 v) { return (a^round64(0,v))*P1+P4; }
extern "C" __global__ void hash_blocks(const unsigned char *src,U64 bytes,U64 block_bytes,U64 *out) {
  U64 start=(U64)blockIdx.x*block_bytes;
  U64 n=bytes>start ? min(bytes-start,block_bytes) : 0;
  const unsigned char *p=src+start;
  int lane=threadIdx.x;
  U64 v=lane==0 ? P1+P2 : lane==1 ? P2 : lane==2 ? 0 : 0-P1;
  if(lane<4) {
    for(U64 off=0;off+32<=n;off+=32) v=round64(v,rd64(p+off+lane*8));
  }
  U64 v0=__shfl_sync(0xffffffff,v,0), v1=__shfl_sync(0xffffffff,v,1);
  U64 v2=__shfl_sync(0xffffffff,v,2), v3=__shfl_sync(0xffffffff,v,3);
  if(lane!=0) return;
  U64 h;
  if(n>=32) {
    h=rot(v0,1)+rot(v1,7)+rot(v2,12)+rot(v3,18);
    h=merge64(h,v0); h=merge64(h,v1); h=merge64(h,v2); h=merge64(h,v3);
  } else h=P5;
  h+=n;
  U64 off=(n/32)*32;
  for(;off+8<=n;off+=8) { h^=round64(0,rd64(p+off)); h=rot(h,27)*P1+P4; }
  if(off+4<=n) { h^=(U64)rd32(p+off)*P1; h=rot(h,23)*P2+P3; off+=4; }
  for(;off<n;off++) { h^=(U64)p[off]*P5; h=rot(h,11)*P1; }
  h^=h>>33; h*=P2; h^=h>>29; h*=P3; h^=h>>32;
  out[blockIdx.x]=h;
}
'''
_KERNEL = None


def tree_digest(array):
    """Hash a device array on the current stream; return a 16-digit root.

    Noncontiguous inputs are gathered in device memory to logical C order.
    Native member banks are contiguous, so they need no field-sized scratch.
    The tiny root read synchronizes this stream, never the whole device.
    """
    import cupy as cp
    import numpy as np
    global _KERNEL
    if not isinstance(array, cp.ndarray):
        raise TypeError('device digest requires a CUDA array')
    if array.device.id != cp.cuda.runtime.getDevice():
        raise ValueError('digest array is on another CUDA device')
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SOURCE, 'hash_blocks')
    packed = cp.ascontiguousarray(array)
    count = max(1, (packed.nbytes + BLOCK_BYTES - 1) // BLOCK_BYTES)
    leaves = cp.empty(count, dtype=cp.uint64)
    root = cp.empty(1, dtype=cp.uint64)
    _KERNEL((count,), (32,), (packed, np.uint64(packed.nbytes), np.uint64(BLOCK_BYTES), leaves))
    _KERNEL((1,), (32,), (leaves, np.uint64(leaves.nbytes), np.uint64(leaves.nbytes), root))
    return f'{int(root.get()[0]):016x}'
