# s is supplied by assemble.py.
s=re.sub(r'(\w+) (\w+) = \{\.([fu])\s*=\s*([^;]+)\};',r'\1 \2; \2.\3 = \4;',s)
s=re.sub(r'\{\.lo\s*=\s*([^,{}]+),\s*\.hi\s*=\s*([^,{}]+),\s*\.ex\s*=\s*([^,{}]+),\s*\.sgn\s*=\s*([^{}]+)\}',r'{{\1,\2,\3,\4}}',s)
s=s.replace('''typedef union {
  struct {
    u128 r;
    int64_t _ex;
    uint64_t _sgn;
  };
  struct {
    uint64_t lo;
    uint64_t hi;
    int64_t ex;
    uint64_t sgn;
  };
} dint64_t;''','''typedef union {
  struct {uint64_t lo,hi; int64_t ex; uint64_t sgn;};
  struct {u128 r; int64_t _ex; uint64_t _sgn;};
} dint64_t;''')
s=s.replace('typedef unsigned __int128 u128;', '''// Unsigned arithmetic modulo 2^128 using two limbs. No NVRTC flag needed.
struct u128 {
  uint64_t lo,hi;
  __device__ u128() = default;
  __device__ constexpr u128(uint64_t l):lo(l),hi(0){}
  __device__ constexpr u128(uint64_t l,uint64_t h):lo(l),hi(h){}
  __device__ operator uint64_t() const {return lo;}
  __device__ u128 operator+(u128 b)const{uint64_t l=lo+b.lo;return u128(l,hi+b.hi+(l<lo));}
  __device__ u128 operator-(u128 b)const{return u128(lo-b.lo,hi-b.hi-(lo<b.lo));}
  __device__ u128 operator*(u128 b)const{return u128(lo*b.lo,__umul64hi(lo,b.lo)+lo*b.hi+hi*b.lo);}
  __device__ u128 operator|(u128 b)const{return u128(lo|b.lo,hi|b.hi);}
  __device__ u128 operator>>(uint64_t k)const{if(!k)return *this;if(k>=128)return u128(0);if(k>=64)return u128(hi>>(k-64));return u128((lo>>k)|(hi<<(64-k)),hi>>k);}
  __device__ u128 operator<<(uint64_t k)const{if(!k)return *this;if(k>=128)return u128(0);if(k>=64)return u128(0,lo<<(k-64));return u128(lo<<k,(hi<<k)|(lo>>(64-k)));}
  __device__ u128& operator+=(u128 b){*this=*this+b;return *this;}
  __device__ bool operator<(u128 b)const{return hi<b.hi||(hi==b.hi&&lo<b.lo);}
  __device__ bool operator>(u128 b)const{return b<*this;}
  __device__ bool operator==(u128 b)const{return hi==b.hi&&lo==b.lo;}
};
''')
s=s.replace('B >> k : 0','B >> k : u128(0)')
s=s.replace('c[0] < u','u128(c[0]) < u')
s=s.replace('''typedef union {
  struct {uint64_t lo,hi; int64_t ex; uint64_t sgn;};
  struct {u128 r; int64_t _ex; uint64_t _sgn;};
} dint64_t;''','''typedef struct {uint64_t lo,hi; int64_t ex; uint64_t sgn;} dint64_t;
__device__ inline u128 get128(const dint64_t *a){return u128(a->lo,a->hi);}
__device__ inline void set128(dint64_t *a,u128 v){a->lo=v.lo;a->hi=v.hi;}
''')
s=re.sub(r'\{\{([^{}\n]+,[^{}\n]+,[^{}\n]+,[^{}\n]+)\}\}',r'{\1}',s)
s=re.sub(r'r->r \+= ([^;]+);',r'set128(r,get128(r) + (\1));',s)
s=re.sub(r'r->r = ([^;]+);',r'set128(r,\1);',s)
s=re.sub(r'(\w+)->r\b',r'get128(\1)',s)
s=s.replace('__device__ u128 operator>>(uint64_t k)const','template<class K> __device__ u128 operator>>(K k)const').replace('__device__ u128 operator<<(uint64_t k)const','template<class K> __device__ u128 operator<<(K k)const')
s=s.replace('__device__ u128& operator+=', 'template<class V> __device__ u128 operator+(V v)const{return *this+u128((uint64_t)v);}\n  __device__ u128& operator+=')
s=s.replace('__device__ __forceinline__ double __math_invalid(double x){return __ddiv_rn((__dsub_rn(x,x)),(__dsub_rn(x,x)));}', '__device__ __forceinline__ double __math_invalid(double x){uint64_t u=asuint64(x);return asdouble((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL ? u|0x8000000000000ULL : 0xfff8000000000000ULL);}')
# Lift function-local CORE-MATH tables into device readonly global storage.
for ns in ['cos','acos']:
 start=s.index('namespace g64_'+ns+' {');end=s.index('\n}\n',start)
 # namespace text ends at final standalone brace, not at function endings.
 end=s.index('namespace g64_acos {',start) if ns=='cos' else len(s)
 chunk=s[start:end];glob=[];at=0
 while True:
  m=re.search(r'\bstatic const double\s+(\w+)([^=;]*)=',chunk[at:])
  if not m:break
  a=at+m.start();b=at+m.end();name=m[1];shape=m[2].strip();depth=0;j=b
  while j<len(chunk):
   if chunk[j]=='{':depth+=1
   if chunk[j]=='}':depth-=1
   if chunk[j]==';' and depth==0:break
   j+=1
  value=chunk[b:j].strip();unique='g64_table_'+str(len(glob))
  glob.append('__device__ const double '+unique+shape+' = '+value+';')
  repl='const auto &'+name+' = '+unique+';'
  chunk=chunk[:a]+repl+chunk[j+1:];at=a+len(repl)
 chunk=chunk.replace('namespace g64_'+ns+' {','namespace g64_'+ns+' {\n'+'\n'.join(glob),1)
 s=s[:start]+chunk+s[end:]
