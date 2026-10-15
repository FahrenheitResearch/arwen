"""Compile the complete new surface unit with the existing CPU shims."""
from pathlib import Path
import argparse,subprocess,json


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];kdir=root/'gpuwm/core/kernels'
    source=a.output.with_suffix('.cpp')
    source.write_text(f'''#include "{kdir}/urban_ucm_host.cpp"
static inline int __float_as_int(float x) {{return ucm_host_bits<int>(x);}}
static inline long long __double2ll_rn(double x) {{return std::llrint(x);}}
static inline double __fma_rn(double a,double b,double c) {{return std::fma(a,b,c);}}
static inline double __dsqrt_rn(double x) {{return std::sqrt(x);}}
static inline unsigned long long __umul64hi(unsigned long long a,unsigned long long b) {{return (unsigned long long)(((__uint128_t)a*b)>>64);}}
#include "{kdir}/glibc_trig_flt32.cuh"
#include "{kdir}/urban_ucm480_support.cuh"
#include "{kdir}/urban_ucm480.cuh"
#include "{kdir}/urban_ucm480.cu"
''')
    cmd=['g++','-std=c++17','-O0','-g','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(source),'-o',str(a.output)]
    q=subprocess.run(cmd,text=True,capture_output=True)
    a.output.with_suffix('.log').write_text(q.stdout+q.stderr)
    a.output.with_suffix('.json').write_text(json.dumps({'command':cmd,'exit':q.returncode},indent=2))
    print('COUPLED_COMPILE_EXIT',q.returncode);print((q.stdout+q.stderr)[:8000])
    raise SystemExit(q.returncode)


if __name__=='__main__':main()
