# WRF NSSL GAMMA_DP/GAMMA_SP (its six-term Lanczos series, evaluated with the
# host libm double log/exp that the gfortran-built WRF calls) for the
# diagnostic constants the WOOF kernels carry; prints IEEE bits.  Run on the
# oracle box.
import math, numpy as np
f=np.float32
cof=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5]
stp=2.5066282746310005
def gdp(xx):
    x=xx; y=x; tmp=x+5.5
    tmp=(x+0.5)*math.log(tmp)-tmp
    ser=1.000000000190015
    for c in cof:
        y=y+1.0; ser=ser+c/y
    return math.exp(tmp+math.log(stp*ser/x))
def gsp(xx): return f(gdp(float(f(xx))))
b=lambda v: hex(np.array(v,dtype=np.float32).view(np.uint32))
snu=f(-0.8); one=f(1)
gsnow1=gsp(snu+one); gsnow73=gsp(snu+f(7.0/3.0))
gams1=gsp(f(2)+snu); gams2=gsp(one+snu)
fs=f(f(f(one+snu)*gsp(one+snu))/gsp(f(5.0/3.0)+snu))
gamc1=gsp(f(2)); fc=f(f(f(1)*gsp(f(1)))/gsp(f(5.0/3.0)))
print('gsnow1',b(gsnow1),gsnow1,'gsnow73',b(gsnow73),gsnow73)
print('gams1',b(gams1),gams1,'gams2',b(gams2),gams2,'factor_s',b(fs),fs)
print('gamc1',b(gamc1),gamc1,'factor_c',b(fc),fc, 'gami1', b(gsp(f(2))))
print('woof snow consts', b(f(0.91816872358322144)), b(f(4.5908436775207520)), b(f(0.83693999052047729)), b(f(4.590843677520752)), b(f(0.8877619504928589)), b(f(1.1077321767807007)))
