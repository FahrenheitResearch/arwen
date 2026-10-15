"""Compile the emitted physics for CPU grading on the lane box."""
from pathlib import Path
import argparse,json,subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('listing',type=Path);a=p.parse_args()
    listing=a.listing.resolve();root=Path(__file__).resolve().parents[2]
    cpp=listing.with_suffix('.host.cpp')
    cpp.write_text(f'''#include "{root}/gpuwm/core/kernels/urban_ucm_host.cpp"
static inline int __float_as_int(float x) {{return ucm_host_bits<int>(x);}}
static inline long long __double2ll_rn(double x) {{return std::llrint(x);}}
static inline double __fma_rn(double a,double b,double c) {{return std::fma(a,b,c);}}
static inline double __dsqrt_rn(double x) {{return std::sqrt(x);}}
static inline unsigned long long __umul64hi(unsigned long long a,unsigned long long b) {{return (unsigned long long)(((__uint128_t)a*b)>>64);}}
#include "{root}/gpuwm/core/kernels/glibc_trig_flt32.cuh"
#include "{root}/gpuwm/core/kernels/urban_ucm480_support.cuh"
#include "{listing}"
#include "{listing.with_suffix('.bridge.cpp')}"
''')
    command=['g++','-std=c++17','-O0','-g','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',str(cpp),'-o',str(listing.with_suffix('.so'))]
    q=subprocess.run(command,text=True,capture_output=True)
    listing.with_suffix('.compile.log').write_text(q.stdout+q.stderr)
    listing.with_suffix('.compile.json').write_text(json.dumps({'command':command,'exit':q.returncode},indent=2))
    print('COMPILE_EXIT',q.returncode)
    print((q.stdout+q.stderr)[:6000])
    raise SystemExit(q.returncode)


if __name__=='__main__':main()
