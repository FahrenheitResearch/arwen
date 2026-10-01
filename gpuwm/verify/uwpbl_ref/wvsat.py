"""WRF module_cam_wv_saturation.F:82-98,235-298,512-639,757-774."""
import math
from gpuwm.core.uwpbl_constants import ESTBL, ESTBL_TMIN as TMIN, ESTBL_TMAX as TMAX, ESTBL_TTRICE as TTRICE, PCF, EPSILO, LATVAP, LATICE, RH2O, CPAIR, TMELT
from .fortran import fmin, fmax, aint

def _div(a, b):
    # Fortran IEEE division, including a zero denominator.
    if b == 0.0:
        return math.nan if a == 0.0 else math.copysign(math.inf, a * math.copysign(1.0, b))
    return a / b

def estblf(td):
    e = fmax(fmin(td, TMAX), TMIN)
    i = int(e-TMIN)+1
    ai = aint(e-TMIN)
    return (TMIN+ai-e+1.0)*ESTBL[i-1]-(TMIN+ai-e)*ESTBL[i]

def _sat(t, p):
    es = estblf(t)
    qs = fmin(1.0, _div(EPSILO*es, p-(1.0-EPSILO)*es))
    if qs < 0.0:
        qs = 1.0
        es = p
    return es, qs

def aqsat(t, p, es, qs, kstart, kend):
    for k in range(kstart, kend+1):
        es[k], qs[k] = _sat(t[k], p[k])

def fqsatd(t, p):
    es, qs = _sat(t, p)
    trinv = 1.0/TTRICE
    tc = t-TMELT
    lflg = tc >= -TTRICE and tc < 0.0
    weight = fmin(-(tc*trinv), 1.0)
    hlatsb = LATVAP+weight*LATICE
    hlatvp = LATVAP-2369.0*tc
    hltalt = hlatsb if t < TMELT else hlatvp
    tterm = PCF[0]+tc*(PCF[1]+tc*(PCF[2]+tc*(PCF[3]+tc*PCF[4]))) if lflg else 0.0
    desdt = hltalt*es/(RH2O*t*t)+tterm*trinv
    gam = _div(hltalt*qs*p*desdt, CPAIR*es*(p-(1.0-EPSILO)*es))
    if qs == 1.0:
        gam = 0.0
    return 1, es, qs, gam

def vqsatd(t, p, es, qs, gam, length):
    for i in range(1, length+1):
        _, es[i], qs[i], gam[i] = fqsatd(t[i], p[i])
