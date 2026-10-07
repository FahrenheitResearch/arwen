"""Float32 references for both pinned Freitas variants and GSL injection.

Pinned ccpp-physics 3e6660c6df54e95a0871e990c2294dd397ae3860:
https://github.com/ufs-community/ccpp-physics/blob/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust/module_plumerise.F90#L143-L166
Indices passed here retain Fortran's one-based convention; kp2 is exclusive.
"""
from __future__ import annotations

import numpy as np


def frp_driver_override(frp_inst, plume_k_min, plume_k_max, kpbl,
                        uspdavg2d, hpbl2d, *, frp_min=np.float32(1.e7),
                        frp_wthreshold=np.float32(1.e9),
                        zpbl_lim=np.float32(2.e3), uspd_lim=np.float32(5.),
                        wind_eff_opt=1):
    """GSL module_plumerise.F90:143-154, including positive NINT ties."""
    f = np.float32
    k1, k2 = int(plume_k_min), int(plume_k_max)
    if f(frp_inst) <= f(frp_min):
        fraction = f(0.)
    elif (f(frp_inst) <= f(frp_wthreshold)
          and f(uspdavg2d) >= f(uspd_lim) and f(hpbl2d) > f(zpbl_lim)
          and wind_eff_opt == 1):
        k1 = 2
        # kpbl is positive. Fortran NINT rounds a half away from zero.
        k2 = max(3, int(np.floor(f(f(int(kpbl)) / f(3.)) + f(.5))))
        fraction = f(.85)
    else:
        fraction = f(.9)
    return k1, k2, fraction


def ebu_distribute(kp1, kp2, flam_frac, ebu_in, z_at_w):
    """GSL module_plumerise.F90:158-162 with each operation rounded to f4."""
    f = np.float32
    z = np.asarray(z_at_w, dtype=np.float32)
    if z.ndim != 1 or not (1 <= kp1 < kp2 <= len(z)):
        raise ValueError("injection bounds must address distinct w levels; "
                         "invalid bounds read outside the column or divide by zero")
    dz = f(z[kp2 - 1] - z[kp1 - 1])
    if not np.isfinite(z).all() or not np.all(np.diff(z) > f(0.)):
        raise ValueError("w levels must be finite and strictly increasing; "
                         "invalid thickness gives negative or undefined emissions")
    out = np.zeros(len(z) - 1, dtype=np.float32)
    for k in range(kp1 - 1, kp2 - 1):
        out[k] = f(f(f(f(flam_frac) * f(ebu_in)) * f(z[k + 1] - z[k])) / dz)
    out[0] = f(f(f(1.) - f(flam_frac)) * f(ebu_in))
    return out


def reference_column(case, arm, *, namespace=None):
    """One model column, with zero state followed by deliberate imm carry.

    Inputs are the names published by the two Fortran drivers. Returned
    diagnostic profiles use the plume's 200-level grid; emission profiles
    use the variable model grid. Species are independent emitted-row vectors.
    ``namespace`` replaces this module's arithmetic helpers by name; the
    maintainer audit (tools/chem_wrf471_oracle/audit_plume_cpu.py) passes
    JIT-compiled copies of the same statements.  The reference itself needs
    only NumPy.
    """
    ns=globals() if namespace is None else namespace
    if arm not in ('frp','landuse'):
        raise ValueError('Freitas arm must be frp or landuse; another arm applies incorrect fire physics')
    prefix='gsl' if arm=='frp' else 'wrf'
    nz=int(case['nz']); s=np.zeros((STATE_SIZE,202),dtype=np.float32)
    def field(name): return s[STATE_SLOTS[name]]
    cp=f32(1004.5); rd=f32(287.); rcp=fdiv(rd,cp)
    # WRF module_plumerise1.F:262-266, including left-to-right theta prep.
    for k in range(1,nz+1):
        pi=fmul(cp,ns['fpow'](fdiv(case['p_phy'][k-1],f32(100000.)),rcp))
        field('picon')[k]=pi
        field('thtcon')[k]=fmul(fdiv(case['t_phy'][k-1],pi),cp)
    for dest,key in [('ucon','u_phy'),('vcon','v_phy'),('rvcon','qv_in')]:
        field(dest)[1:nz+1]=case[key]
    for dest,key in [('zcon','z'),('zzcon','z_at_w')]:
        field(dest)[1:nz+1]=np.asarray(case[key][:nz],dtype=np.float32)-f32(case['z_at_w'][0])
    if prefix=='wrf': kmt=ns['wrf_get_env_condition'](s,1,nz,0,1)
    else: kmt=ns['gsl_get_env_condition'](s,1,nz,0,1,f32(9.81),cp,rd,fdiv(cp,rd))
    incoming=np.asarray(case['ebu_in'],dtype=np.float32)
    from pathlib import Path
    import json
    groups=json.loads((Path(__file__).resolve().parent.parent/'data/chem/plumerise/fire_groups.v1.json').read_text())['groups']
    heat=np.zeros((5,202),dtype=np.float32)
    for g,entry in enumerate(groups,1): heat[g,1:3]=entry['heat_min'],entry['heat_max']
    out=np.zeros((nz,len(incoming)),dtype=np.float32)
    tops=np.zeros((2,4),dtype=np.float32); steps=np.zeros((2,4),dtype=np.int32)
    solver_k1=np.zeros(4,dtype=np.int32); solver_k2=solver_k1.copy()
    if arm=='frp':
        frp=f32(case['frp_inst'])
        k1=1; k2=2
        if frp>=f32(1.e7):
            for imm in (1,2):
                area=fmax(f32(1.e4),fmul(word(0x39dc3373 if imm==1 else 0x3a4c78ea),frp))
                ns['gsl_get_fire_properties'](s,imm,area,frp)
                top=ns['gsl_makeplume'](s,kmt,f32(0.),0,imm,0,f32(.05))
                tops[imm-1,0]=top; steps[imm-1,0]=int(field('steps')[0])
            tt=np.zeros(202,dtype=np.float32); tt[1:3]=tops[:,0]
            k1,k2=ns['gsl_set_flam_vert'](s,tt,0,0,200,field('zzcon'))
            solver_k1[0]=k1; solver_k2[0]=k2
        k1,k2,fraction=frp_driver_override(frp,k1,k2,case['kpbl'],case['uspdavg2d'],case['hpbl2d'])
        for r,e in enumerate(incoming): out[:,r]=ebu_distribute(k1,k2,fraction,e,case['z_at_w'])
        extra=dict(k_min=np.int32(k1),k_max=np.int32(k2),flam_frac=fraction)
    else:
        fractions=case['mean_fct']; sizes=case['firesize']
        out[0]=incoming
        # WRF driver:224-244 gates occur before preparing or calling the solver.
        sf=fadd(fadd(fadd(fractions[0],fractions[1]),fractions[2]),fractions[3])
        ss=fadd(fadd(fadd(sizes[0],sizes[1]),sizes[2]),sizes[3])
        if sf>=f32(1.e-6) and ss>=f32(1.e-6) and np.max(incoming)!=f32(0.):
            for group in range(1,5):
                if fractions[group-1]<f32(1.e-6): continue
                tt=np.zeros(202,dtype=np.float32)
                for imm in (1,2):
                    ns['wrf_get_fire_properties'](s,imm,group,f32(sizes[group-1]),f32(0.),heat)
                    if groups[group-1]['single_heat_pass'] and imm==2:
                        tt[2]=tt[1]; tt[1]=field('zzcon')[1]
                    else:
                        tt[imm]=ns['wrf_makeplume'](s,kmt,f32(0.),0,imm)
                        steps[imm-1,group-1]=int(field('steps')[0])
                tops[:,group-1]=tt[1:3]
                k1,k2=ns['wrf_set_flam_vert'](s,tt,0,0,200,field('zzcon'),
                    s[STATE_SLOTS['w_vmd']:STATE_SLOTS['w_vmd']+3],
                    s[STATE_SLOTS['vmd']:STATE_SLOTS['vmd']+3])
                solver_k1[group-1]=k1; solver_k2[group-1]=k2
                dz=fsub(field('zzcon')[k2+1],field('zzcon')[k1]); dzi=fdiv(f32(1.),dz)
                for k in range(k1,k2+1):
                    for r,e in enumerate(incoming): out[k-1,r]=fadd(out[k-1,r],fmul(fmul(fractions[group-1],e),dzi))
            for k in range(1,nz):
                out[k]=np.asarray([fmul(e,fsub(case['z_at_w'][k+1],case['z_at_w'][k])) for e in out[k]],dtype=np.float32)
        else:
            s.fill(0.)
        extra={}
    result=dict(ebu=out,ztopmax=tops,steps=steps,solver_k_min=solver_k1,solver_k_max=solver_k2,**extra)
    for name in ('w','t','qv','qc','qh','qi','radius'): result[name]=field(name)[1:201].copy()
    return result

# BEGIN FREITAS SOLVER TRANSCRIPTION
# WRF f52c197ed39d12e087d02c50f412d90d418f6186 chem/module_chem_plumerise_scalar.F.
# GSL 3e6660c6df54e95a0871e990c2294dd397ae3860 physics/smoke_dust/module_smoke_plumerise.F90.
# Generated by tools/chem_wrf471_oracle/transcribe_plume.py; no reassociation.
from gpuwm.core.noahmp_libm import expf as fexp, powf as _portable_powf, sqrtf as fsqrt
f32 = np.float32

def word(value):
    return np.uint32(value).view(np.float32)

def fpow(x,y):
    # glibc's invalid negative-base/noninteger power returns 0xffc00000.
    # WRF fallpart:1690 computes this for a slightly negative intermediate
    # density in some cases; VTC is unused, but preserve the observed word.
    if x<0 and np.isfinite(x) and np.isfinite(y) and y!=np.trunc(y): return word(0xffc00000)
    return _portable_powf(x,y)

def fadd(a,b): return f32(f32(a)+f32(b))
def fsub(a,b): return f32(f32(a)-f32(b))
def fmul(a,b): return f32(f32(a)*f32(b))
def fdiv(a,b): return f32(f32(a)/f32(b))
def fabs(a): return f32(abs(f32(a)))
def fmin(a,b): return np.minimum(f32(a),f32(b))
def fmax(a,b): return np.maximum(f32(a),f32(b))
def fsum(a):
    total=f32(0.)
    for value in a: total=fadd(total,value)
    return total

def fpowi(a,n):
    # libgcc __powisf2: square-and-multiply, then reciprocal for negative n.
    a=f32(a); power=abs(n); result=a if power&1 else f32(1.)
    power >>= 1
    while power:
        a=fmul(a,a)
        if power&1: result=fmul(result,a)
        power >>= 1
    return fdiv(f32(1.),result) if n<0 else result

STATE_SLOTS = {'w': 0, 't': 1, 'qv': 2, 'qc': 3, 'qh': 4, 'qi': 5, 'sc': 6, 'vth': 7, 'vti': 8, 'rho': 9, 'txs': 10, 'est': 11, 'qsat': 12, 'qpas': 13, 'qtotal': 14, 'wc': 15, 'wt': 16, 'tt': 17, 'qvt': 18, 'qct': 19, 'qht': 20, 'qit': 21, 'sct': 22, 'dzm': 23, 'dzt': 24, 'zm': 25, 'zt': 26, 'vctr1': 27, 'vctr2': 28, 'vt3dc': 29, 'vt3df': 30, 'vt3dk': 31, 'vt3dg': 32, 'scr1': 33, 'pke': 34, 'the': 35, 'thve': 36, 'thee': 37, 'pe': 38, 'te': 39, 'qvenv': 40, 'rhe': 41, 'dne': 42, 'sce': 43, 'ucon': 44, 'vcon': 45, 'wcon': 46, 'thtcon': 47, 'rvcon': 48, 'picon': 49, 'tmpcon': 50, 'dncon': 51, 'prcon': 52, 'zcon': 53, 'zzcon': 54, 'scon': 55, 'dz': 56, 'dqsdz': 57, 'visc': 58, 'viscosity': 59, 'tstpf': 60, 'n': 61, 'nm1': 62, 'l': 63, 'advw': 64, 'advt': 65, 'advv': 66, 'advc': 67, 'advh': 68, 'advi': 69, 'cvh': 70, 'cvi': 71, 'adiabat': 72, 'wbar': 73, 'alast': 74, 'vhrel': 75, 'virel': 76, 'zsurf': 77, 'zbase': 78, 'ztop': 79, 'lbase': 80, 'area': 81, 'rsurf': 82, 'alpha': 83, 'radius': 84, 'heating': 85, 'fmoist': 86, 'bload': 87, 'dt': 88, 'time': 89, 'tdur': 90, 'mintime': 91, 'mdur': 92, 'maxtime': 93, 'w_vmd': 94, 'vmd': 97, 'upe': 100, 'vpe': 101, 'vel_e': 102, 'vel_p': 103, 'rad_p': 104, 'vel_t': 105, 'rad_t': 106, 'ztop_': 107, 'steps': 108, 'steps1': 109, 'steps2': 110, 'top1': 111, 'top2': 112, 'initialized': 113}
STATE_SIZE = 114

# chem/module_chem_plumerise_scalar.F:212-287.
def wrf_get_env_condition(s, k1, k2, kmt, wind_eff):
    k = 0
    kcon = 0
    klcl = 0
    nk = 0
    nkmid = 0
    i = 0
    znz = f32(0.)
    themax = f32(0.)
    tlll = f32(0.)
    plll = f32(0.)
    rlll = f32(0.)
    zlll = f32(0.)
    dzdd = f32(0.)
    dzlll = f32(0.)
    tlcl = f32(0.)
    plcl = f32(0.)
    dzlcl = f32(0.)
    dummy = f32(0.)
    n_setgrid = 0
    if (n_setgrid == 0):
        n_setgrid = int(1)
        wrf_set_grid(s)
    znz = f32(s[53][k2])
    k = 200
    while k >= 1:
        if (s[26][k] < znz):
            break
        k += -1
    kmt = int(min(k, 199))
    nk = int(((k2 - k1) + 1))
    wrf_htint(s, nk, s[44], s[53], kmt, s[100], s[26])
    wrf_htint(s, nk, s[45], s[53], kmt, s[101], s[26])
    wrf_htint(s, nk, s[47], s[53], kmt, s[35], s[26])
    wrf_htint(s, nk, s[48], s[53], kmt, s[40], s[26])
    k = 1
    while k <= kmt:
        s[40][k] = f32(fmax(s[40][k], word(0x322bcc77)))
        k += 1
    s[34][1] = f32(s[49][1])
    k = 1
    while k <= kmt:
        s[36][k] = f32(fmul(s[35][k], fadd(word(0x3f800000), fmul(word(0x3f1c28f6), s[40][k]))))
        k += 1
    k = 2
    while k <= kmt:
        s[34][k] = f32(fsub(s[34][(k - 1)], fdiv(fmul(word(0x419cf5c3), fsub(s[26][k], s[26][(k - 1)])), fadd(s[36][k], s[36][(k - 1)]))))
        k += 1
    k = 1
    while k <= kmt:
        s[39][k] = f32(fdiv(fmul(s[35][k], s[34][k]), word(0x447b2000)))
        s[38][k] = f32(fmul(fpow(fdiv(s[34][k], word(0x447b2000)), word(0x405fffff)), word(0x47c35000)))
        s[42][k] = f32(fdiv(s[38][k], fmul(fmul(word(0x438f8000), s[39][k]), fadd(word(0x3f800000), fmul(word(0x3f1c28f6), s[40][k])))))
        s[102][k] = f32(fsqrt(fadd(fpowi(s[100][k], 2), fpowi(s[101][k], 2))))
        k += 1
    if (wind_eff < 1):
        for _a in range(1, (kmt)+1):
            s[102][_a] = f32(word(0x00000000))
    k = 1
    while k <= kmt:
        s[38][k] = f32(fmul(s[38][k], word(0x3a83126f)))
        k += 1
    return kmt
    return kmt

# chem/module_chem_plumerise_scalar.F:294-322.
def wrf_set_grid(s):
    k = 0
    mzp = 0
    s[56, 0] = f32(word(0x42c80000))
    mzp = int(200)
    s[26][1] = f32(s[77, 0])
    s[25][1] = f32(s[77, 0])
    s[26][2] = f32(fadd(s[26][1], fmul(word(0x3f000000), s[56, 0])))
    s[25][2] = f32(fadd(s[25][1], s[56, 0]))
    k = 3
    while k <= mzp:
        s[26][k] = f32(fadd(s[26][(k - 1)], s[56, 0]))
        s[25][k] = f32(fadd(s[25][(k - 1)], s[56, 0]))
        k += 1
    k = 1
    while k <= (mzp - 1):
        s[23][k] = f32(fdiv(word(0x3f800000), fsub(s[26][(k + 1)], s[26][k])))
        k += 1
    s[23][mzp] = f32(s[23][(mzp - 1)])
    k = 2
    while k <= mzp:
        s[24][k] = f32(fdiv(word(0x3f800000), fsub(s[25][k], s[25][(k - 1)])))
        k += 1
    s[24][1] = f32(fdiv(fmul(s[24][2], s[24][2]), s[24][3]))
    return

# chem/module_chem_plumerise_scalar.F:328-410.
def wrf_set_flam_vert(s, ztopmax, k1, k2, nkp, zzcon, w_vmd, vmd):
    imm = 0
    k = 0
    k_lim = np.zeros(202, dtype=np.int32)
    w_thresold = f32(0.)
    xxx = f32(0.)
    k_initial = 0
    k_final = 0
    ko = 0
    kk4 = 0
    kl = 0
    imm = 1
    while imm <= 2:
        k = 1
        while k <= (nkp - 1):
            if (zzcon[k] > ztopmax[imm]):
                break
            k += 1
        k_lim[imm] = int(k)
        imm += 1
    k1 = int(max(3, k_lim[1]))
    k2 = int(max(3, k_lim[2]))
    if (k2 < k1):
        k2 = int(k1)
    w_thresold = f32(word(0x3f800000))
    imm = 1
    while imm <= 2:
        for _a in range(1, (nkp)+1):
            vmd[imm, _a] = f32(word(0x00000000))
        xxx = f32(word(0x00000000))
        k_initial = int(0)
        k_final = int(0)
        ko = (nkp - 10)
        while ko >= 2:
            if (w_vmd[imm, ko] < w_thresold):
                ko += -1
                continue
            if (k_final == 0):
                k_final = int(ko)
            if (fsub(w_vmd[imm, ko], word(0x3f800000)) > w_vmd[imm, (ko - 1)]):
                k_initial = int(ko)
                break
            ko += -1
        if ((k_final > 0) and (k_initial > 0)):
            k_initial = int(int(fmul((k_final + k_initial), word(0x3f000000))))
            kk4 = int(((k_final - k_initial) + 2))
            ko = 1
            while ko <= (kk4 - 1):
                kl = int(((ko + k_initial) - 1))
                vmd[imm, kl] = f32(fmul(fdiv(fmul(word(0x40c00000), f32(ko)), fpowi(f32(kk4), 2)), fsub(word(0x3f800000), fdiv(f32(ko), f32(kk4)))))
                ko += 1
            if (fsum(vmd[imm, 1:(nkp)+1]) != word(0x3f800000)):
                xxx = f32(fdiv(fsub(word(0x3f800000), fsum(vmd[imm, 1:(nkp)+1])), f32(((k_final - k_initial) + 1))))
                ko = k_initial
                while ko <= k_final:
                    vmd[imm, ko] = f32(fadd(vmd[imm, ko], xxx))
                    ko += 1
        imm += 1
    return k1, k2

# chem/module_chem_plumerise_scalar.F:417-535.
def wrf_get_fire_properties(s, imm, iveg_ag, burnt_area, std_burnt_area, heat_flux):
    moist = 0
    i = 0
    icount = 0
    bfract = f32(0.)
    effload = f32(0.)
    heat = f32(0.)
    hinc = f32(0.)
    heat_fluxw = f32(0.)
    s[81, 0] = f32(burnt_area)
    heat_fluxw = f32(fmul(heat_flux[iveg_ag, imm], word(0x447a0000)))
    s[92, 0] = int(53)
    s[87, 0] = f32(word(0x41200000))
    moist = int(10)
    s[93, 0] = int((int(s[92, 0]) + 2))
    heat = f32(word(0x4b933f50))
    s[83, 0] = f32(word(0x3d4ccccd))
    s[93, 0] = int((int(s[93, 0]) * 60))
    s[82, 0] = f32(fsqrt(fdiv(s[81, 0], word(0x40490fd0))))
    s[86, 0] = f32(fdiv(moist, word(0x42c80000)))
    i = 1
    while i <= 200:
        s[85][i] = f32(word(0x38d1b717))
        i += 1
    s[90, 0] = f32(fmul(int(s[92, 0]), word(0x42700000)))
    bfract = f32(word(0x3f800000))
    effload = f32(fmul(s[87, 0], bfract))
    icount = int(1)
    if (int(s[92, 0]) > 200):
        raise ValueError('plume source duration exceeds heating storage')
    while (icount <= int(s[92, 0])):
        s[85][icount] = f32(fmul(heat_fluxw, word(0x3f0ccccd)))
        icount = int((icount + 1))
    if (0 != 1):
        hinc = f32(fdiv(s[85][1], word(0x40800000)))
        s[85][1] = f32(word(0x3dcccccd))
        s[85][2] = f32(hinc)
        s[85][3] = f32(fmul(word(0x40000000), hinc))
        s[85][4] = f32(fmul(word(0x40400000), hinc))
    else:
        if (imm == 1):
            hinc = f32(fdiv(s[85][1], word(0x40800000)))
            s[85][1] = f32(word(0x3dcccccd))
            s[85][2] = f32(hinc)
            s[85][3] = f32(fmul(word(0x40000000), hinc))
            s[85][4] = f32(fmul(word(0x40400000), hinc))
        else:
            hinc = f32(fdiv(fsub(s[85][1], fmul(fmul(heat_flux[iveg_ag, (imm - 1)], word(0x447a0000)), word(0x3f0ccccd))), word(0x40800000)))
            s[85][1] = f32(fadd(fmul(fmul(heat_flux[iveg_ag, (imm - 1)], word(0x447a0000)), word(0x3f0ccccd)), word(0x3dcccccd)))
            s[85][2] = f32(fadd(s[85][1], hinc))
            s[85][3] = f32(fadd(s[85][2], hinc))
            s[85][4] = f32(fadd(s[85][3], hinc))
    return

# chem/module_chem_plumerise_scalar.F:600-831.
def wrf_makeplume(s, kmt, ztopmax, ixx, imm):
    varn = ''
    izprint = 0
    iconv = 0
    itime = 0
    k = 0
    kk = 0
    kkmax = 0
    deltak = 0
    ilastprint = 0
    nrectotal = 0
    i_micro = 0
    n_sub_step = 0
    wmax = f32(0.)
    rmaxtime = f32(0.)
    es = f32(0.)
    esat = f32(0.)
    heat = f32(0.)
    dt_save = f32(0.)
    cixx = ''
    delz_thresold = word(0x42c80000)
    s[108, 0] = 0
    s[60, 0] = f32(word(0x40000000))
    s[59, 0] = f32(word(0x43fa0000))
    nrectotal = int(150)
    s[91, 0] = int(1)
    ztopmax = f32(word(0x00000000))
    s[79, 0] = f32(word(0x00000000))
    s[89, 0] = f32(word(0x00000000))
    s[88, 0] = f32(word(0x3f800000))
    wmax = f32(word(0x3f800000))
    kkmax = int(10)
    deltak = int(20)
    ilastprint = int(0)
    s[63, 0] = int(1)
    wrf_initial(s, kmt)
    izprint = int(0)
    rmaxtime = f32(f32(int(s[93, 0])))
    while (s[89, 0] <= rmaxtime):
        s[62, 0] = int(min(kmt, (kkmax + deltak)))
        s[88, 0] = f32(fmin(word(0x40a00000), fdiv(fsub(s[25][2], s[25][1]), fmul(s[60, 0], wmax))))
        s[89, 0] = f32(fadd(s[89, 0], s[88, 0]))
        s[108, 0] += 1
        s[91, 0] = int((1 + (int(s[89, 0]) // 60)))
        wmax = f32(word(0x3f800000))
        wrf_tend0_plumerise(s)
        s[63, 0] = int(1)
        wrf_lbound(s)
        wrf_vel_advectc_plumerise(s, int(s[62, 0]), s[15], s[16], s[9], s[23])
        wrf_scl_advectc_plumerise(s, 'sc', int(s[62, 0]))
        wrf_scl_misc(s, int(s[62, 0]))
        wrf_scl_dyn_entrain(s, int(s[62, 0]), 200, s[73, 0], s[0], s[72, 0], s[83, 0], s[84], s[17], s[1], s[39], s[18], s[2], s[40], s[19], s[3], s[20], s[4], s[21], s[5], s[102], s[103], s[105], s[104], s[106])
        wrf_damp_grav_wave(s, 1, int(s[62, 0]), deltak, s[88, 0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40])
        dt_save = f32(s[88, 0])
        n_sub_step = int(3)
        s[88, 0] = f32(fdiv(s[88, 0], f32(n_sub_step)))
        i_micro = 1
        while i_micro <= n_sub_step:
            wrf_fallpart(s, int(s[62, 0]))
            s[63, 0] = 2
            while int(s[63, 0]) <= (int(s[62, 0]) - 1):
                s[73, 0] = f32(fmul(word(0x3f000000), fadd(s[0][int(s[63, 0])], s[0][(int(s[63, 0]) - 1)])))
                es = f32(wrf_esat_pr(s, s[1][int(s[63, 0])]))
                s[12][int(s[63, 0])] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][int(s[63, 0])], es)))
                s[11][int(s[63, 0])] = f32(es)
                s[9][int(s[63, 0])] = f32(fdiv(fmul(word(0x4559bccd), s[38][int(s[63, 0])]), s[1][int(s[63, 0])]))
                if (s[0][int(s[63, 0])] >= word(0x00000000)):
                    s[57, 0] = f32(fdiv(fsub(s[12][(int(s[63, 0]) + 1)], s[12][(int(s[63, 0]) - 1)]), fsub(s[26][(int(s[63, 0]) + 1)], s[26][(int(s[63, 0]) - 1)])))
                else:
                    s[57, 0] = f32(fdiv(fsub(s[12][(int(s[63, 0]) + 1)], s[12][(int(s[63, 0]) - 1)]), fsub(s[26][(int(s[63, 0]) + 1)], s[26][(int(s[63, 0]) - 1)])))
                wrf_waterbal(s)
                s[63, 0] += 1
            i_micro += 1
        s[88, 0] = f32(dt_save)
        wrf_visc_w(s, int(s[62, 0]), deltak, kmt)
        wrf_update_plumerise(s, int(s[62, 0]), 's')
        wrf_hadvance_plumerise(s, 1, int(s[62, 0]), s[88, 0], s[15], s[16], s[0], int(s[91, 0]))
        wrf_buoyancy_plumerise(s, int(s[62, 0]), s[1], s[39], s[2], s[40], s[4], s[5], s[3], s[16], s[33])
        wrf_entrainment(s, int(s[62, 0]), s[0], s[16], s[84], s[83, 0])
        wrf_update_plumerise(s, int(s[62, 0]), 'w')
        wrf_hadvance_plumerise(s, 2, int(s[62, 0]), s[88, 0], s[15], s[16], s[0], int(s[91, 0]))
        k = 2
        while k <= int(s[62, 0]):
            es = f32(wrf_esat_pr(s, s[1][k]))
            s[12][k] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][k], es)))
            s[11][k] = f32(es)
            s[10][k] = f32(fsub(s[1][k], s[39][k]))
            s[9][k] = f32(fdiv(fmul(word(0x4559bccd), s[38][k]), s[1][k]))
            if (fabs(s[15][k]) > wmax):
                wmax = f32(fabs(s[15][k]))
            k += 1
        wrf_damp_grav_wave(s, 2, int(s[62, 0]), deltak, s[88, 0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40])
        k = 2
        while k <= int(s[62, 0]):
            s[84][k] = f32(s[104][k])
            k += 1
        kk = int(1)
        while (s[0][kk] > word(0x3f800000)):
            kk = int((kk + 1))
            s[79, 0] = f32(s[25][kk])
        s[107][int(s[91, 0])] = f32(s[79, 0])
        ztopmax = f32(fmax(s[79, 0], ztopmax))
        kkmax = int(max(kk, kkmax))
        if (int(s[91, 0]) > 10):
            if (fabs(fsub(s[107][int(s[91, 0])], s[107][(int(s[91, 0]) - 10)])) < delz_thresold):
                k = 2
                while k <= int(s[62, 0]):
                    s[94:97][imm, k] = f32(s[0][k])
                    k += 1
                break
    if imm == 1:
        s[111, 0] = ztopmax
    else:
        s[112, 0] = ztopmax
    if imm == 1:
        s[109, 0] = s[108, 0]
    else:
        s[110, 0] = s[108, 0]
    return ztopmax
    return ztopmax

# chem/module_chem_plumerise_scalar.F:841-865.
def wrf_burn(s, eflux, water):
    if (s[89, 0] > s[90, 0]):
        eflux = f32(word(0x358637bd))
        water = f32(word(0x00000000))
        return eflux, water
    else:
        eflux = f32(s[85][int(s[91, 0])])
        water = f32(fdiv(fmul(fmul(eflux, fdiv(s[88, 0], word(0x4b933f50))), fadd(word(0x3f000000), s[86, 0])), word(0x3f0ccccd)))
        water = f32(fmul(water, word(0x447a0000)))
    return eflux, water
    return eflux, water

# chem/module_chem_plumerise_scalar.F:885-964.
def wrf_lbound(s):
    es = f32(0.)
    esat = f32(0.)
    eflux = f32(0.)
    water = f32(0.)
    pres = f32(0.)
    c1 = f32(0.)
    c2 = f32(0.)
    f = f32(0.)
    zv = f32(0.)
    denscor = f32(0.)
    xwater = f32(0.)
    s[4][1] = f32(s[4][2])
    s[5][1] = f32(s[5][2])
    s[3][1] = f32(word(0x00000000))
    eflux, water = wrf_burn(s, eflux, water)
    pres = f32(fmul(s[38][1], word(0x447a0000)))
    c1 = f32(fdiv(word(0x40a00000), fmul(word(0x40c00000), s[83, 0])))
    c2 = f32(fmul(word(0x3f666666), s[83, 0]))
    f = f32(fdiv(eflux, fmul(fmul(pres, word(0x447b2666)), word(0x40490fd0))))
    f = f32(fmul(fmul(word(0x452ff46e), f), s[81, 0]))
    zv = f32(fmul(c1, s[82, 0]))
    s[0][1] = f32(fdiv(fmul(c1, fpow(fmul(c2, f), word(0x3eaaaaab))), fpow(zv, word(0x3eaaaaab))))
    denscor = f32(fdiv(fdiv(fdiv(fmul(c1, f), word(0x411ced67)), fpow(fmul(c2, f), word(0x3eaaaaab))), fpow(zv, word(0x3fd55555))))
    s[1][1] = f32(fdiv(s[39][1], fsub(word(0x3f800000), denscor)))
    s[15][1] = f32(s[0][1])
    s[103][1] = f32(word(0x00000000))
    s[104][1] = f32(s[82, 0])
    s[7][1] = f32(word(0xc0800000))
    s[8][1] = f32(word(0xc0400000))
    s[10][1] = f32(fsub(s[1][1], s[39][1]))
    s[58][1] = f32(s[59, 0])
    s[9][1] = f32(fdiv(fmul(word(0x4559bccd), s[38][1]), s[1][1]))
    xwater = f32(fdiv(water, fmul(fmul(s[0][1], s[88, 0]), s[9][1])))
    s[2][1] = f32(fadd(xwater, s[40][1]))
    es = f32(wrf_esat_pr(s, s[1][1]))
    s[11][1] = f32(es)
    s[12][1] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][1], es)))
    if (s[2][1] > s[12][1]):
        s[3][1] = f32(fadd(fsub(s[2][1], s[12][1]), s[3][1]))
        s[2][1] = f32(s[12][1])
    wrf_waterbal(s)
    return

# chem/module_chem_plumerise_scalar.F:972-1029.
def wrf_initial(s, kmt):
    isub = 0
    k = 0
    n1 = 0
    n2 = 0
    n3 = 0
    lbuoy = 0
    itmp = 0
    isubm1 = 0
    xn1 = f32(0.)
    xi = f32(0.)
    es = f32(0.)
    esat = f32(0.)
    s[61, 0] = int(kmt)
    k = 1
    while k <= int(s[61, 0]):
        s[10][k] = f32(word(0x00000000))
        s[0][k] = f32(word(0x00000000))
        s[1][k] = f32(s[39][k])
        s[15][k] = f32(word(0x00000000))
        s[16][k] = f32(word(0x00000000))
        s[2][k] = f32(s[40][k])
        s[7][k] = f32(word(0x00000000))
        s[8][k] = f32(word(0x00000000))
        s[4][k] = f32(word(0x00000000))
        s[5][k] = f32(word(0x00000000))
        s[3][k] = f32(word(0x00000000))
        es = f32(wrf_esat_pr(s, s[1][k]))
        s[11][k] = f32(es)
        s[12][k] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][k], es)))
        s[9][k] = f32(fdiv(fmul(word(0x4559bccd), s[38][k]), s[1][k]))
        s[103][k] = f32(word(0x00000000))
        s[104][k] = f32(word(0x00000000))
        k += 1
    s[84][1] = f32(s[82, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[84][k] = f32(fadd(s[84][(k - 1)], fmul(fmul(word(0x3f99999a), s[83, 0]), fsub(s[26][k], s[26][(k - 1)]))))
        k += 1
    s[84][1] = f32(s[82, 0])
    s[104][1] = f32(s[82, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[84][k] = f32(fadd(s[84][(k - 1)], fmul(fmul(word(0x3f99999a), s[83, 0]), fsub(s[26][k], s[26][(k - 1)]))))
        s[104][k] = f32(s[84][k])
        k += 1
    s[58][1] = f32(s[59, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[58][k] = f32(fmax(word(0x3a83126f), fsub(s[58][(k - 1)], fdiv(fmul(word(0x3f800000), s[59, 0]), word(0x43480000)))))
        k += 1
    wrf_lbound(s)
    return

# chem/module_chem_plumerise_scalar.F:1034-1050.
def wrf_damp_grav_wave(s, ifrom, nm1, deltak, dt, zt, zm, w, t, tt, qv, qh, qi, qc, te, pe, qvenv):
    dummy = np.zeros(202, dtype=np.float32)
    if (ifrom == 1):
        wrf_friction(s, ifrom, nm1, deltak, dt, zt, zm, t, tt, te)
        return
    for _a in range(1, (200)+1):
        dummy[_a] = f32(word(0x00000000))
    if (ifrom == 2):
        wrf_friction(s, ifrom, nm1, deltak, dt, zt, zm, w, dummy, dummy)
    return

# chem/module_chem_plumerise_scalar.F:1055-1085.
def wrf_friction(s, ifrom, nm1, deltak, dt, zt, zm, var1, vart, var2):
    k = 0
    nfpt = 0
    kf = 0
    zmkf = f32(0.)
    ztop = f32(0.)
    distim = f32(0.)
    c1 = f32(0.)
    c2 = f32(0.)
    kf = int((nm1 - int(deltak)))
    zmkf = f32(zm[kf])
    ztop = f32(zm[nm1])
    distim = f32(fmin(fmul(word(0x40400000), dt), word(0x42700000)))
    c1 = f32(fdiv(word(0x3f800000), fmul(distim, fsub(ztop, zmkf))))
    c2 = f32(fmul(dt, c1))
    if (ifrom == 1):
        k = nm1
        while k >= 2:
            if (zt[k] <= zmkf):
                k += -1
                continue
            vart[k] = f32(fadd(vart[k], fmul(fmul(c1, fsub(zt[k], zmkf)), fsub(var2[k], var1[k]))))
            k += -1
    elif (ifrom == 2):
        k = nm1
        while k >= 2:
            if (zt[k] <= zmkf):
                k += -1
                continue
            var1[k] = f32(fadd(var1[k], fmul(fmul(c2, fsub(zt[k], zmkf)), fsub(var2[k], var1[k]))))
            k += -1
    return

# chem/module_chem_plumerise_scalar.F:1091-1121.
def wrf_vel_advectc_plumerise(s, m1, wc, wt, rho, dzm):
    k = 0
    flxw = np.zeros(202, dtype=np.float32)
    dn0 = np.zeros(202, dtype=np.float32)
    c1z = f32(0.)
    for _a in range(1, (m1)+1):
        dn0[_a] = f32(fmul(rho[_a], word(0x3a83126f)))
    flxw[1] = f32(fmul(wc[1], dn0[1]))
    k = 2
    while k <= (m1 - 1):
        flxw[k] = f32(fmul(fmul(wc[k], word(0x3f000000)), fadd(dn0[k], dn0[(k + 1)])))
        k += 1
    c1z = f32(word(0x3f000000))
    k = 2
    while k <= (m1 - 2):
        wt[k] = f32(fadd(wt[k], fmul(fdiv(fmul(c1z, dzm[k]), fadd(dn0[k], dn0[(k + 1)])), fadd(fsub(fmul(fadd(flxw[k], flxw[(k - 1)]), fadd(wc[k], wc[(k - 1)])), fmul(fadd(flxw[k], flxw[(k + 1)]), fadd(wc[k], wc[(k + 1)]))), fmul(fmul(fsub(flxw[(k + 1)], flxw[(k - 1)]), word(0x40000000)), wc[k])))))
        k += 1
    return

# chem/module_chem_plumerise_scalar.F:1127-1147.
def wrf_hadvance_plumerise(s, iac, m1, dt, wc, wt, wp, mintime):
    k = 0
    dummy = np.zeros(202, dtype=np.float32)
    eps = f32(0.)
    eps = f32(word(0x3e4ccccd))
    if (mintime == 1):
        eps = f32(word(0x3f000000))
    wrf_predict_plumerise(s, m1, wc, wp, wt, dummy, iac, fmul(word(0x40000000), dt), eps)
    return

# chem/module_chem_plumerise_scalar.F:1152-1193.
def wrf_predict_plumerise(s, npts, ac, ap, fa, af, iac, dtlp, epsu):
    m = 0
    if (iac == 1):
        m = 1
        while m <= npts:
            ac[m] = f32(fadd(ac[m], fmul(epsu, fsub(ap[m], fmul(word(0x40000000), ac[m])))))
            m += 1
        return
    elif (iac == 2):
        m = 1
        while m <= npts:
            af[m] = f32(ap[m])
            ap[m] = f32(fadd(ac[m], fmul(epsu, af[m])))
            m += 1
    m = 1
    while m <= npts:
        ac[m] = f32(af[m])
        m += 1
    return

# chem/module_chem_plumerise_scalar.F:1198-1228.
def wrf_buoyancy_plumerise(s, m1, t, te, qv, qvenv, qh, qi, qc, wt, scr1):
    k = 0
    tv = f32(0.)
    tve = f32(0.)
    qwtotl = f32(0.)
    umgamai = f32(0.)
    umgamai = f32(word(0x3f2aaaab))
    k = 2
    while k <= (m1 - 1):
        tv = f32(fdiv(fmul(t[k], fadd(word(0x3f800000), fdiv(qv[k], word(0x3f1f3b64)))), fadd(word(0x3f800000), qv[k])))
        tve = f32(fdiv(fmul(te[k], fadd(word(0x3f800000), fdiv(qvenv[k], word(0x3f1f3b64)))), fadd(word(0x3f800000), qvenv[k])))
        qwtotl = f32(fadd(fadd(qh[k], qi[k]), qc[k]))
        scr1[k] = f32(fmul(fmul(word(0x411ccccd), umgamai), fsub(fdiv(fsub(tv, tve), tve), qwtotl)))
        k += 1
    k = 2
    while k <= (m1 - 2):
        wt[k] = f32(fadd(wt[k], fmul(word(0x3f000000), fadd(scr1[k], scr1[(k + 1)]))))
        k += 1

# chem/module_chem_plumerise_scalar.F:1234-1273.
def wrf_entrainment(s, m1, w, wt, radius, alpha):
    k = 0
    dmdtm = f32(0.)
    wbar = f32(0.)
    radius_bar = f32(0.)
    umgamai = f32(0.)
    dyn_entr = f32(0.)
    umgamai = f32(word(0x3f2aaaab))
    k = 2
    while k <= (m1 - 1):
        wbar = f32(w[k])
        radius_bar = f32(fmul(word(0x3f000000), fadd(radius[k], radius[(k - 1)])))
        dmdtm = f32(fdiv(fmul(fmul(fmul(umgamai, word(0x40000000)), alpha), fabs(wbar)), radius_bar))
        wt[k] = f32(fsub(wt[k], fmul(dmdtm, fabs(wbar))))
        dyn_entr = f32(fdiv(fmul(word(0x3ea2f96b), fabs(fsub(fadd(fsub(s[103][k], s[102][k]), s[103][(k - 1)]), s[102][(k - 1)]))), radius_bar))
        wt[k] = f32(fsub(wt[k], fmul(dyn_entr, fabs(wbar))))
        k += 1

# chem/module_chem_plumerise_scalar.F:1279-1401.
def wrf_scl_advectc_plumerise(s, varn, mzp):
    dtlto2 = f32(0.)
    k = 0
    dtlto2 = f32(fmul(word(0x3f000000), s[88, 0]))
    s[29][1] = f32(fmul(fmul(fmul(fadd(s[0][1], s[15][1]), dtlto2), s[9][1]), word(0x3a83126f)))
    s[30][1] = f32(fmul(fmul(fmul(word(0x3f000000), fadd(s[0][1], s[15][1])), dtlto2), s[23][1]))
    k = 2
    while k <= mzp:
        s[29][k] = f32(fmul(fmul(fmul(fmul(fadd(s[0][k], s[15][k]), dtlto2), word(0x3f000000)), fadd(s[9][k], s[9][(k + 1)])), word(0x3a83126f)))
        s[30][k] = f32(fmul(fmul(fmul(fadd(s[0][k], s[15][k]), dtlto2), word(0x3f000000)), s[23][k]))
        k += 1
    k = 1
    while k <= mzp:
        s[27][k] = f32(fmul(fsub(s[26][(k + 1)], s[25][k]), s[23][k]))
        s[28][k] = f32(fmul(fsub(s[25][k], s[26][k]), s[23][k]))
        s[31][k] = f32(fdiv(s[24][k], fmul(s[9][k], word(0x3a83126f))))
        k += 1
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[1][_a])
    wrf_fa_zc_plumerise(s, mzp, s[1], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[1], s[33], s[17], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[2][_a])
    wrf_fa_zc_plumerise(s, mzp, s[2], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[2], s[33], s[18], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[3][_a])
    wrf_fa_zc_plumerise(s, mzp, s[3], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[3], s[33], s[19], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[5][_a])
    wrf_fa_zc_plumerise(s, mzp, s[5], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[5], s[33], s[21], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[4][_a])
    wrf_fa_zc_plumerise(s, mzp, s[4], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[4], s[33], s[20], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[103][_a])
    wrf_fa_zc_plumerise(s, mzp, s[103], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[103], s[33], s[105], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[104][_a])
    wrf_fa_zc_plumerise(s, mzp, s[104], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[104], s[33], s[106], s[88, 0])
    return
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[6][_a])
    wrf_fa_zc_plumerise(s, mzp, s[6], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    wrf_advtndc_plumerise(s, mzp, s[6], s[33], s[22], s[88, 0])
    return

# chem/module_chem_plumerise_scalar.F:1407-1449.
def wrf_fa_zc_plumerise(s, m1, scp, scr1, vt3dc, vt3df, vt3dg, vt3dk, vctr1, vctr2):
    k = 0
    dfact = f32(0.)
    dfact = f32(word(0x3f000000))
    k = 1
    while k <= (m1 - 1):
        vt3dg[k] = f32(fmul(vt3dc[k], fadd(fadd(fmul(vctr1[k], scr1[k]), fmul(vctr2[k], scr1[(k + 1)])), fmul(vt3df[k], fsub(scr1[k], scr1[(k + 1)])))))
        k += 1
    k = 1
    while k <= (m1 - 1):
        if (vt3dc[k] > word(0x00000000)):
            if (fmul(vt3dg[k], vt3dk[k]) > fmul(dfact, scr1[k])):
                vt3dg[k] = f32(fmul(vt3dc[k], scr1[k]))
        elif (vt3dc[k] < word(0x00000000)):
            if (fmul((-vt3dg[k]), vt3dk[(k + 1)]) > fmul(dfact, scr1[(k + 1)])):
                vt3dg[k] = f32(fmul(vt3dc[k], scr1[(k + 1)]))
        k += 1
    k = 2
    while k <= (m1 - 1):
        scr1[k] = f32(fadd(scr1[k], fmul(vt3dk[k], fadd(fsub(vt3dg[(k - 1)], vt3dg[k]), fmul(scp[k], fsub(vt3dc[k], vt3dc[(k - 1)]))))))
        k += 1
    return

# chem/module_chem_plumerise_scalar.F:1454-1463.
def wrf_advtndc_plumerise(s, m1, scp, sca, sct, dtl):
    k = 0
    dtli = f32(0.)
    dtli = f32(fdiv(word(0x3f800000), dtl))
    k = 2
    while k <= (m1 - 1):
        sct[k] = f32(fadd(sct[k], fmul(fsub(sca[k], scp[k]), dtli)))
        k += 1
    return

# chem/module_chem_plumerise_scalar.F:1469-1476.
def wrf_tend0_plumerise(s):
    for _a in range(1, (int(s[62, 0]))+1):
        s[16][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[17][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[18][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[19][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[20][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[21][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[105][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[106][_a] = f32(word(0x00000000))

# chem/module_chem_plumerise_scalar.F:1484-1515.
def wrf_scl_misc(s, m1):
    k = 0
    dmdtm = f32(0.)
    k = 2
    while k <= (m1 - 1):
        s[73, 0] = f32(fmul(word(0x3f000000), fadd(s[0][k], s[0][(k - 1)])))
        s[72, 0] = f32(fdiv(fmul((-s[73, 0]), word(0x411cf5c3)), word(0x447b0000)))
        dmdtm = f32(fdiv(fmul(fmul(word(0x40000000), s[83, 0]), fabs(s[73, 0])), s[84][k]))
        s[17][k] = f32(fsub(fadd(s[17][k], s[72, 0]), fmul(dmdtm, fsub(s[1][k], s[39][k]))))
        s[18][k] = f32(fsub(s[18][k], fmul(dmdtm, fsub(s[2][k], s[40][k]))))
        s[19][k] = f32(fsub(s[19][k], fmul(dmdtm, s[3][k])))
        s[20][k] = f32(fsub(s[20][k], fmul(dmdtm, s[4][k])))
        s[21][k] = f32(fsub(s[21][k], fmul(dmdtm, s[5][k])))
        s[105][k] = f32(fsub(s[105][k], fmul(dmdtm, fsub(s[103][k], s[102][k]))))
        s[106][k] = f32(fadd(s[106][k], fmul(fmul(fmul(word(0x3f000000), dmdtm), word(0x3f99999a)), s[84][k])))
        k += 1

# chem/module_chem_plumerise_scalar.F:1521-1584.
def wrf_scl_dyn_entrain(s, m1, nkp, wbar, w, adiabat, alpha, radius, tt, t, te, qvt, qv, qvenv, qct, qc, qht, qh, qit, qi, vel_e, vel_p, vel_t, rad_p, rad_t):
    k = 0
    dmdtm = f32(0.)
    k = 2
    while k <= (m1 - 1):
        rad_t[k] = f32(fadd(rad_t[k], fdiv(fabs(fsub(vel_e[k], vel_p[k])), word(0x40490ff9))))
        dmdtm = f32(fdiv(fmul(word(0x3f22f96b), fabs(fsub(vel_e[k], vel_p[k]))), radius[k]))
        vel_t[k] = f32(fsub(vel_t[k], fmul(dmdtm, fsub(vel_p[k], vel_e[k]))))
        tt[k] = f32(fsub(tt[k], fmul(dmdtm, fsub(t[k], te[k]))))
        qvt[k] = f32(fsub(qvt[k], fmul(dmdtm, fsub(qv[k], qvenv[k]))))
        qct[k] = f32(fsub(qct[k], fmul(dmdtm, qc[k])))
        qht[k] = f32(fsub(qht[k], fmul(dmdtm, qh[k])))
        qit[k] = f32(fsub(qit[k], fmul(dmdtm, qi[k])))
        k += 1

# chem/module_chem_plumerise_scalar.F:1591-1627.
def wrf_visc_w(s, m1, deltak, kmt):
    k = 0
    m2 = 0
    dz1t = f32(0.)
    dz1m = f32(0.)
    dz2t = f32(0.)
    dz2m = f32(0.)
    d2wdz = f32(0.)
    d2tdz = f32(0.)
    d2qvdz = f32(0.)
    d2qhdz = f32(0.)
    d2qcdz = f32(0.)
    d2qidz = f32(0.)
    d2scdz = f32(0.)
    d2vel_pdz = f32(0.)
    d2rad_dz = f32(0.)
    m2 = int(min(m1, kmt))
    k = 2
    while k <= (m2 - 1):
        dz1t = f32(fmul(word(0x3f000000), fsub(s[26][(k + 1)], s[26][(k - 1)])))
        dz2t = f32(fdiv(s[58][k], fmul(dz1t, dz1t)))
        dz1m = f32(fmul(word(0x3f000000), fsub(s[25][(k + 1)], s[25][(k - 1)])))
        dz2m = f32(fdiv(s[58][k], fmul(dz1m, dz1m)))
        d2wdz = f32(fmul(fadd(fsub(s[0][(k + 1)], fmul(2, s[0][k])), s[0][(k - 1)]), dz2m))
        d2tdz = f32(fmul(fadd(fsub(s[1][(k + 1)], fmul(2, s[1][k])), s[1][(k - 1)]), dz2t))
        d2qvdz = f32(fmul(fadd(fsub(s[2][(k + 1)], fmul(2, s[2][k])), s[2][(k - 1)]), dz2t))
        d2qhdz = f32(fmul(fadd(fsub(s[4][(k + 1)], fmul(2, s[4][k])), s[4][(k - 1)]), dz2t))
        d2qcdz = f32(fmul(fadd(fsub(s[3][(k + 1)], fmul(2, s[3][k])), s[3][(k - 1)]), dz2t))
        d2qidz = f32(fmul(fadd(fsub(s[5][(k + 1)], fmul(2, s[5][k])), s[5][(k - 1)]), dz2t))
        d2vel_pdz = f32(fmul(fadd(fsub(s[103][(k + 1)], fmul(2, s[103][k])), s[103][(k - 1)]), dz2t))
        d2rad_dz = f32(fmul(fadd(fsub(s[104][(k + 1)], fmul(2, s[104][k])), s[104][(k - 1)]), dz2t))
        s[16][k] = f32(fadd(s[16][k], d2wdz))
        s[17][k] = f32(fadd(s[17][k], d2tdz))
        s[18][k] = f32(fadd(s[18][k], d2qvdz))
        s[19][k] = f32(fadd(s[19][k], d2qcdz))
        s[20][k] = f32(fadd(s[20][k], d2qhdz))
        s[21][k] = f32(fadd(s[21][k], d2qidz))
        s[105][k] = f32(fadd(s[105][k], d2vel_pdz))
        s[106][k] = f32(fadd(s[106][k], d2rad_dz))
        k += 1

# chem/module_chem_plumerise_scalar.F:1635-1667.
def wrf_update_plumerise(s, m1, varn):
    k = 0
    if (varn == 'w'):
        k = 2
        while k <= (m1 - 1):
            s[0][k] = f32(fadd(s[0][k], fmul(s[16][k], s[88, 0])))
            k += 1
        return
    else:
        k = 2
        while k <= (m1 - 1):
            s[1][k] = f32(fadd(s[1][k], fmul(s[17][k], s[88, 0])))
            s[2][k] = f32(fadd(s[2][k], fmul(s[18][k], s[88, 0])))
            s[3][k] = f32(fadd(s[3][k], fmul(s[19][k], s[88, 0])))
            s[4][k] = f32(fadd(s[4][k], fmul(s[20][k], s[88, 0])))
            s[5][k] = f32(fadd(s[5][k], fmul(s[21][k], s[88, 0])))
            s[2][k] = f32(fmax(word(0x00000000), s[2][k]))
            s[3][k] = f32(fmax(word(0x00000000), s[3][k]))
            s[4][k] = f32(fmax(word(0x00000000), s[4][k]))
            s[5][k] = f32(fmax(word(0x00000000), s[5][k]))
            s[103][k] = f32(fadd(s[103][k], fmul(s[105][k], s[88, 0])))
            s[104][k] = f32(fadd(s[104][k], fmul(s[106][k], s[88, 0])))
            k += 1

# chem/module_chem_plumerise_scalar.F:1673-1729.
def wrf_fallpart(s, m1):
    k = 0
    vtc = f32(0.)
    dfhz = f32(0.)
    dfiz = f32(0.)
    dz1 = f32(0.)
    k = 2
    while k <= (m1 - 1):
        vtc = f32(fmul(word(0x40a36fb7), fpow(s[9][k], word(0x3e000000))))
        s[7][k] = f32(word(0xc0800000))
        s[75, 0] = f32(fadd(s[0][k], s[7][k]))
        s[70][k] = f32(fadd(word(0x3fcccccd), fmul(word(0x3a156c0d), fpow(fabs(s[75, 0]), word(0x3fc00000)))))
        s[8][k] = f32(word(0xc0400000))
        s[76, 0] = f32(fadd(s[0][k], s[8][k]))
        s[71][k] = f32(fadd(word(0x3fcccccd), fdiv(fmul(word(0x3a156c0d), fpow(fabs(s[76, 0]), word(0x3fc00000))), word(0x3f400000))))
        if (s[75, 0] >= word(0x00000000)):
            dfhz = f32(fdiv(fmul(s[4][k], fsub(fmul(s[9][k], s[7][k]), fmul(s[9][(k - 1)], s[7][(k - 1)]))), s[9][(k - 1)]))
        else:
            dfhz = f32(fdiv(fmul(s[4][k], fsub(fmul(s[9][(k + 1)], s[7][(k + 1)]), fmul(s[9][k], s[7][k]))), s[9][k]))
        if (s[76, 0] >= word(0x00000000)):
            dfiz = f32(fdiv(fmul(s[5][k], fsub(fmul(s[9][k], s[8][k]), fmul(s[9][(k - 1)], s[8][(k - 1)]))), s[9][(k - 1)]))
        else:
            dfiz = f32(fdiv(fmul(s[5][k], fsub(fmul(s[9][(k + 1)], s[8][(k + 1)]), fmul(s[9][k], s[8][k]))), s[9][k]))
        dz1 = f32(fsub(s[25][k], s[25][(k - 1)]))
        s[20][k] = f32(fsub(s[20][k], fdiv(dfhz, dz1)))
        s[21][k] = f32(fsub(s[21][k], fdiv(dfiz, dz1)))
        k += 1

# chem/module_chem_plumerise_scalar.F:1818-1835.
def wrf_waterbal(s):
    if (s[3][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[3][int(s[63, 0])] = f32(word(0x00000000))
    if (s[4][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[4][int(s[63, 0])] = f32(word(0x00000000))
    if (s[5][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[5][int(s[63, 0])] = f32(word(0x00000000))
    wrf_evaporate(s)
    wrf_sublimate(s)
    wrf_glaciate(s)
    wrf_melt(s)
    wrf_convert(s)
    return

# chem/module_chem_plumerise_scalar.F:1843-2048.
def wrf_evaporate(s):
    evhdt = f32(0.)
    evidt = f32(0.)
    evrate = f32(0.)
    evap = f32(0.)
    sd = f32(0.)
    quant = f32(0.)
    dividend = f32(0.)
    divisor = f32(0.)
    devidt = f32(0.)
    sd = f32(fsub(s[12][int(s[63, 0])], s[2][int(s[63, 0])]))
    if (sd == word(0x00000000)):
        return
    evhdt = f32(word(0x00000000))
    evidt = f32(word(0x00000000))
    evrate = f32(fabs(fmul(s[73, 0], s[57, 0])))
    evap = f32(fmul(evrate, s[88, 0]))
    if (sd <= word(0x00000000)):
        if (evap >= fabs(sd)):
            s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
            s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
            s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
            return
        else:
            s[3][int(s[63, 0])] = f32(fadd(s[3][int(s[63, 0])], evap))
            s[2][int(s[63, 0])] = f32(fsub(s[2][int(s[63, 0])], evap))
            s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(evap, word(0x451ba0a4))))
            return
    else:
        if (evap <= s[3][int(s[63, 0])]):
            if (sd <= evap):
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                return
            else:
                sd = f32(fsub(sd, evap))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evap))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evap, word(0x451ba0a4))))
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], evap))
        else:
            if (sd <= s[3][int(s[63, 0])]):
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                return
            else:
                sd = f32(fsub(sd, s[3][int(s[63, 0])]))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[3][int(s[63, 0])]))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[3][int(s[63, 0])], word(0x451ba0a4))))
                s[3][int(s[63, 0])] = f32(word(0x00000000))
        if (s[4][int(s[63, 0])] > word(0x2edbe6ff)):
            quant = f32(fmul(fsub(fsub(s[12][int(s[63, 0])], s[3][int(s[63, 0])]), s[2][int(s[63, 0])]), s[9][int(s[63, 0])]))
            evhdt = f32(fdiv(fmul(fmul(fmul(s[88, 0], word(0x3a0e9b39)), quant), fpow(fmul(s[4][int(s[63, 0])], s[9][int(s[63, 0])]), word(0x3f266666))), s[9][int(s[63, 0])]))
            if (evhdt <= s[4][int(s[63, 0])]):
                if (sd <= evhdt):
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], sd))
                    s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                    return
                else:
                    sd = f32(fsub(sd, evhdt))
                    s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evhdt))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evhdt, word(0x451ba0a4))))
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], evhdt))
            else:
                if (sd <= s[4][int(s[63, 0])]):
                    s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], sd))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                    return
                else:
                    sd = f32(fsub(sd, s[4][int(s[63, 0])]))
                    s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[4][int(s[63, 0])]))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[4][int(s[63, 0])], word(0x451ba0a4))))
                    s[4][int(s[63, 0])] = f32(word(0x00000000))
        if (s[5][int(s[63, 0])] <= word(0x2edbe6ff)):
            return
        dividend = f32(fmul(fmul(fmul(fpow(fdiv(word(0x49742400), s[9][int(s[63, 0])]), word(0x3ef33333)), fsub(fdiv(sd, s[12][int(s[63, 0])]), 1)), fpow(s[5][int(s[63, 0])], word(0x3f066666))), word(0x3f90a3d7)))
        divisor = f32(fadd(word(0x492ae600), fdiv(word(0x4a7a3e80), fmul(word(0x41200000), s[11][int(s[63, 0])]))))
        devidt = f32(fdiv(fmul((-s[71][int(s[63, 0])]), dividend), divisor))
        evidt = f32(fmul(devidt, s[88, 0]))
        if (evidt <= s[5][int(s[63, 0])]):
            if (sd <= evidt):
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], sd))
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x45306b59))))
                return
            else:
                sd = f32(fsub(sd, evidt))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evidt))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evidt, word(0x45306b59))))
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], evidt))
        else:
            if (sd <= s[5][int(s[63, 0])]):
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], sd))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x45306b59))))
                return
            else:
                sd = f32(fsub(sd, s[5][int(s[63, 0])]))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[5][int(s[63, 0])]))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[5][int(s[63, 0])], word(0x45306b59))))
                s[5][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2059-2122.
def wrf_convert(s):
    accrete = f32(0.)
    con = f32(0.)
    q = f32(0.)
    h = f32(0.)
    bc1 = f32(0.)
    bc2 = f32(0.)
    total = f32(0.)
    if (s[1][int(s[63, 0])] <= word(0x4386a666)):
        return
    if (s[3][int(s[63, 0])] == word(0x00000000)):
        return
    accrete = f32(word(0x00000000))
    con = f32(word(0x00000000))
    q = f32(fmul(s[9][int(s[63, 0])], s[3][int(s[63, 0])]))
    h = f32(fmul(s[9][int(s[63, 0])], s[4][int(s[63, 0])]))
    if (s[4][int(s[63, 0])] > word(0x00000000)):
        accrete = f32(fmul(fmul(word(0x3baa64c3), q), fpow(h, word(0x3f600000))))
    if (1 != 0):
        con = f32(fdiv(fmul(fmul(fmul(q, q), q), word(0x3e158106)), fmul(word(0x42700000), fadd(fmul(fmul(word(0x40a00000), q), word(0x3e158106)), word(0x4564c000)))))
    else:
        con = f32(fmax(word(0x00000000), fmul(word(0x3a83126f), fsub(q, word(0x3f000000)))))
    total = f32(fdiv(fmul(fadd(con, accrete), s[88, 0]), s[9][int(s[63, 0])]))
    if (total < s[3][int(s[63, 0])]):
        s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], total))
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], total))
        return
    else:
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], s[3][int(s[63, 0])]))
        s[3][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2130-2232.
def wrf_convert2(s):
    ka = f32(0.)
    keins = f32(0.)
    kzwei = f32(0.)
    kdrei = f32(0.)
    vt = f32(0.)
    a = f32(0.)
    b = f32(0.)
    c = f32(0.)
    d = f32(0.)
    con = f32(0.)
    accrete = f32(0.)
    total = f32(0.)
    y = np.zeros(202, dtype=np.float32)
    roh = f32(0.)
    a = f32(word(0x00000000))
    b = f32(word(0x00000000))
    y[1] = f32(s[1][int(s[63, 0])])
    y[4] = f32(s[0][int(s[63, 0])])
    y[2] = f32(s[3][int(s[63, 0])])
    y[3] = f32(s[4][int(s[63, 0])])
    y[5] = f32(s[84][int(s[63, 0])])
    roh = f32(fmul(s[9][int(s[63, 0])], word(0x3a83126f)))
    ka = f32(word(0x3a03126f))
    if (y[1] < word(0x43811333)):
        keins = f32(word(0x3a6bedfa))
        kzwei = f32(word(0x3baa64c3))
        kdrei = f32(word(0x41763d71))
    else:
        keins = f32(word(0x3ac49ba6))
        kzwei = f32(word(0x3be410b6))
        kdrei = f32(word(0x413947ae))
    vt = f32(fmul((-kdrei), fpow(fdiv(y[3], roh), word(0x3e000000))))
    if (y[4] > word(0x00000000)):
        if True:
            a = f32(fdiv(fmul(fmul(fmul(fdiv(1, y[4]), y[2]), y[2]), word(0x447a0000)), fmul(word(0x42700000), fadd(word(0x40a00000), fdiv(word(0x425b999a), fmul(fmul(y[2], word(0x447a0000)), word(0x3f800000)))))))
        else:
            if (y[2] > fmul(ka, roh)):
                a = f32(fmul(fdiv(keins, y[4]), fsub(y[2], fmul(ka, roh))))
    else:
        a = f32(word(0x00000000))
    if (y[4] > word(0x00000000)):
        b = f32(fmul(fmul(fmul(fdiv(kzwei, fsub(y[4], vt)), fmax(word(0x00000000), y[2])), fpow(fmax(word(0x3a83126f), roh), word(0xbf600000))), fpow(fmax(word(0x00000000), y[3]), word(0x3f600000))))
    else:
        b = f32(word(0x00000000))
    con = f32(a)
    accrete = f32(b)
    total = f32(fdiv(fmul(fadd(con, accrete), fdiv(1, s[23][int(s[63, 0])])), roh))
    if (total < s[3][int(s[63, 0])]):
        s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], total))
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], total))
        return
    else:
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], s[3][int(s[63, 0])]))
        s[3][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2249-2292.
def wrf_sublimate(s):
    dtsubh = f32(0.)
    dividend = f32(0.)
    divisor = f32(0.)
    subl = f32(0.)
    dtsubh = f32(word(0x00000000))
    if (s[1][int(s[63, 0])] > word(0x4386a666)):
        return
    if (s[2][int(s[63, 0])] <= s[12][int(s[63, 0])]):
        return
    dividend = f32(fmul(fmul(fmul(fpow(fdiv(word(0x49742400), s[9][int(s[63, 0])]), word(0x3ef33333)), fsub(fdiv(s[2][int(s[63, 0])], s[12][int(s[63, 0])]), 1)), fpow(s[5][int(s[63, 0])], word(0x3f066666))), word(0x3f90a3d7)))
    divisor = f32(fadd(word(0x492ae600), fdiv(word(0x4a7a3e80), fmul(word(0x41200000), s[11][int(s[63, 0])]))))
    dtsubh = f32(fabs(fdiv(dividend, divisor)))
    subl = f32(fmul(dtsubh, s[88, 0]))
    if (subl < s[2][int(s[63, 0])]):
        s[2][int(s[63, 0])] = f32(fsub(s[2][int(s[63, 0])], subl))
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], subl))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(subl, word(0x45306b59))))
        return
    else:
        s[5][int(s[63, 0])] = f32(s[2][int(s[63, 0])])
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(s[2][int(s[63, 0])], word(0x45306b59))))
        s[2][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2305-2345.
def wrf_glaciate(s):
    dfrzh = f32(0.)
    dfrzh = f32(word(0x00000000))
    if (s[4][int(s[63, 0])] <= word(0x00000000)):
        return
    if (s[2][int(s[63, 0])] < s[12][int(s[63, 0])]):
        return
    if (s[1][int(s[63, 0])] > word(0x4386a666)):
        return
    dfrzh = f32(fmul(fmul(s[88, 0], word(0x3ccccccd)), s[4][int(s[63, 0])]))
    if (dfrzh < s[4][int(s[63, 0])]):
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], dfrzh))
        s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], dfrzh))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(word(0x43a655ad), dfrzh)))
        return
    else:
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], s[4][int(s[63, 0])]))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(word(0x43a655ad), s[4][int(s[63, 0])])))
        s[4][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2356-2391.
def wrf_melt(s):
    dtmelt = f32(0.)
    dtmelt = f32(word(0x00000000))
    if (s[5][int(s[63, 0])] <= word(0x00000000)):
        return
    if (s[1][int(s[63, 0])] < word(0x43888000)):
        return
    dtmelt = f32(fmul(fmul(fmul(fmul(fmul(s[88, 0], fdiv(word(0x401147ae), s[9][int(s[63, 0])])), s[71][int(s[63, 0])]), fsub(s[1][int(s[63, 0])], word(0x43888000))), fpow(fmul(fmul(s[9][int(s[63, 0])], s[5][int(s[63, 0])]), word(0x358637bd)), word(0x3f066666))), word(0x3f90705d)))
    if (dtmelt < s[5][int(s[63, 0])]):
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], dtmelt))
        s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], dtmelt))
        s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(word(0x43a6228f), dtmelt)))
        return
    else:
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], s[5][int(s[63, 0])]))
        s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(word(0x43a6228f), s[5][int(s[63, 0])])))
        s[5][int(s[63, 0])] = f32(word(0x00000000))
    return

# chem/module_chem_plumerise_scalar.F:2396-2433.
def wrf_htint(s, nzz1, vctra, eleva, nzz2, vctrb, elevb):
    l = 0
    k = 0
    kk = 0
    wt = f32(0.)
    l = int(1)
    k = 1
    while k <= nzz2:
        while True:
            if ((elevb[k] < eleva[1]) or ((elevb[k] >= eleva[l]) and (elevb[k] <= eleva[(l + 1)]))):
                wt = f32(fdiv(fsub(elevb[k], eleva[l]), fsub(eleva[(l + 1)], eleva[l])))
                vctrb[k] = f32(fadd(vctra[l], fmul(fsub(vctra[(l + 1)], vctra[l]), wt)))
                break
            elif (elevb[k] > eleva[nzz1]):
                wt = f32(fdiv(fsub(elevb[k], eleva[nzz1]), fsub(eleva[(nzz1 - 1)], eleva[nzz1])))
                vctrb[k] = f32(fadd(vctra[nzz1], fmul(fsub(vctra[(nzz1 - 1)], vctra[nzz1]), wt)))
                break
            l = int((l + 1))
            if (l == nzz1):
                kk = 1
                while kk <= l:
                    kk += 1
                raise ValueError('plume source guard: invalid duration or interpolation bounds')
        k += 1

# chem/module_chem_plumerise_scalar.F:2440-2462.
def wrf_esat_pr(s, tem):
    esat_pr = f32(0.)
    temc = f32(0.)
    esatm = f32(0.)
    temc = f32(fsub(tem, word(0x4388a666)))
    if (temc <= word(0xc2200000)):
        esatm = f32(fmul(word(0x40c39168), fexp(fdiv(fmul(word(0x41b45604), temc), fadd(temc, word(0x4388bd71))))))
        esat_pr = f32(fdiv(esatm, word(0x41200000)))
        return esat_pr
    esatm = f32(fmul(word(0x40c39653), fexp(fdiv(fmul(fsub(word(0x4195d4fe), fdiv(temc, word(0x43634ccd))), temc), fadd(temc, word(0x4380ef5c))))))
    esat_pr = f32(fdiv(esatm, word(0x41200000)))
    return esat_pr
    return esat_pr

# physics/smoke_dust/module_smoke_plumerise.F90:132-207.
def gsl_get_env_condition(s, k1, k2, kmt, wind_eff, g, cp, rgas, cpor):
    k = 0
    kcon = 0
    klcl = 0
    nk = 0
    nkmid = 0
    i = 0
    znz = f32(0.)
    themax = f32(0.)
    tlll = f32(0.)
    plll = f32(0.)
    rlll = f32(0.)
    zlll = f32(0.)
    dzdd = f32(0.)
    dzlll = f32(0.)
    tlcl = f32(0.)
    plcl = f32(0.)
    dzlcl = f32(0.)
    dummy = f32(0.)
    errmsg = ''
    errflg = 0
    if (not bool(s[113, 0])):
        gsl_set_grid(s)
    znz = f32(s[53][k2])
    pass
    k = 200
    while k >= 1:
        if (s[26][k] < znz):
            pass
            break
        k += -1
    if (errflg != 0):
        pass
        return kmt
    kmt = int(min(k, 199))
    nk = int(((k2 - k1) + 1))
    gsl_htint(s, nk, s[44], s[53], kmt, s[100], s[26])
    if (errflg != 0):
        return kmt
    gsl_htint(s, nk, s[45], s[53], kmt, s[101], s[26])
    if (errflg != 0):
        return kmt
    gsl_htint(s, nk, s[47], s[53], kmt, s[35], s[26])
    if (errflg != 0):
        return kmt
    gsl_htint(s, nk, s[48], s[53], kmt, s[40], s[26])
    if (errflg != 0):
        return kmt
    k = 1
    while k <= kmt:
        s[40][k] = f32(fmax(s[40][k], word(0x322bcc77)))
        k += 1
    s[34][1] = f32(s[49][1])
    k = 1
    while k <= kmt:
        s[36][k] = f32(fmul(s[35][k], fadd(word(0x3f800000), fmul(word(0x3f1c28f6), s[40][k]))))
        k += 1
    k = 2
    while k <= kmt:
        s[34][k] = f32(fsub(s[34][(k - 1)], fdiv(fmul(fmul(g, word(0x40000000)), fsub(s[26][k], s[26][(k - 1)])), fadd(s[36][k], s[36][(k - 1)]))))
        k += 1
    k = 1
    while k <= kmt:
        s[39][k] = f32(fdiv(fmul(s[35][k], s[34][k]), cp))
        s[38][k] = f32(fmul(fpow(fdiv(s[34][k], cp), cpor), word(0x47c35000)))
        s[42][k] = f32(fdiv(s[38][k], fmul(fmul(rgas, s[39][k]), fadd(word(0x3f800000), fmul(word(0x3f1c28f6), s[40][k])))))
        s[102][k] = f32(fsqrt(fadd(fpowi(s[100][k], 2), fpowi(s[101][k], 2))))
        k += 1
    if (wind_eff < 1):
        for _a in range(1, (kmt)+1):
            s[102][_a] = f32(word(0x00000000))
    k = 1
    while k <= kmt:
        s[38][k] = f32(fmul(s[38][k], word(0x3a83126f)))
        k += 1
    return kmt
    return kmt

# physics/smoke_dust/module_smoke_plumerise.F90:214-245.
def gsl_set_grid(s):
    k = 0
    mzp = 0
    s[56, 0] = f32(word(0x42c80000))
    mzp = int(200)
    s[26][1] = f32(s[77, 0])
    s[25][1] = f32(s[77, 0])
    s[26][2] = f32(fadd(s[26][1], fmul(word(0x3f000000), s[56, 0])))
    s[25][2] = f32(fadd(s[25][1], s[56, 0]))
    k = 3
    while k <= mzp:
        s[26][k] = f32(fadd(s[26][(k - 1)], s[56, 0]))
        s[25][k] = f32(fadd(s[25][(k - 1)], s[56, 0]))
        k += 1
    k = 1
    while k <= (mzp - 1):
        s[23][k] = f32(fdiv(word(0x3f800000), fsub(s[26][(k + 1)], s[26][k])))
        k += 1
    s[23][mzp] = f32(s[23][(mzp - 1)])
    k = 2
    while k <= mzp:
        s[24][k] = f32(fdiv(word(0x3f800000), fsub(s[25][k], s[25][(k - 1)])))
        k += 1
    s[24][1] = f32(fdiv(fmul(s[24][2], s[24][2]), s[24][3]))
    s[113, 0] = True
    return

# physics/smoke_dust/module_smoke_plumerise.F90:251-283.
def gsl_set_flam_vert(s, ztopmax, k1, k2, nkp, zzcon):
    imm = 0
    k = 0
    k_lim = np.zeros(202, dtype=np.int32)
    imm = 1
    while imm <= 2:
        k = 1
        while k <= (nkp - 1):
            if (zzcon[k] > ztopmax[imm]):
                break
            k += 1
        k_lim[imm] = int(k)
        imm += 1
    k1 = int(min(max(4, k_lim[1]), 51))
    k2 = int(min(51, k_lim[2]))
    if (k2 <= k1):
        k2 = int((k1 + 1))
    return k1, k2

# physics/smoke_dust/module_smoke_plumerise.F90:290-391.
def gsl_get_fire_properties(s, imm, burnt_area, frp):
    moist = 0
    i = 0
    icount = 0
    bfract = f32(0.)
    effload = f32(0.)
    heat = f32(0.)
    hinc = f32(0.)
    heat_fluxw = f32(0.)
    errflg = 0
    errmsg = ''
    s[81, 0] = f32(burnt_area)
    heat_fluxw = f32(fdiv(fmul(word(0x3f6147ae), fdiv(frp, s[81, 0])), word(0x3f0ccccd)))
    s[92, 0] = int(53)
    s[87, 0] = f32(word(0x41200000))
    moist = int(10)
    s[93, 0] = int((int(s[92, 0]) + 2))
    heat = f32(word(0x4b933f50))
    s[93, 0] = int((int(s[93, 0]) * 60))
    s[82, 0] = f32(fsqrt(fdiv(s[81, 0], word(0x40490fd0))))
    s[86, 0] = f32(fdiv(moist, word(0x42c80000)))
    i = 1
    while i <= 200:
        s[85][i] = f32(word(0x38d1b717))
        i += 1
    s[90, 0] = f32(fmul(int(s[92, 0]), word(0x42700000)))
    bfract = f32(word(0x3f800000))
    effload = f32(fmul(s[87, 0], bfract))
    icount = int(1)
    if (int(s[92, 0]) > 200):
        pass
        pass
        return
    while (icount <= int(s[92, 0])):
        s[85][icount] = f32(fmul(heat_fluxw, word(0x3f0ccccd)))
        icount = int((icount + 1))
    if (1 != 1):
        hinc = f32(fdiv(s[85][1], word(0x40800000)))
        s[85][1] = f32(word(0x3dcccccd))
        s[85][2] = f32(hinc)
        s[85][3] = f32(fmul(word(0x40000000), hinc))
        s[85][4] = f32(fmul(word(0x40400000), hinc))
    else:
        hinc = f32(fdiv(s[85][1], word(0x40800000)))
        if (imm == 1):
            s[85][1] = f32(word(0x3dcccccd))
            s[85][2] = f32(hinc)
            s[85][3] = f32(fmul(word(0x40000000), hinc))
            s[85][4] = f32(fmul(word(0x40400000), hinc))
        else:
            s[85][2] = f32(fadd(s[85][1], hinc))
            s[85][3] = f32(fadd(s[85][2], hinc))
            s[85][4] = f32(fadd(s[85][3], hinc))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:455-702.
def gsl_makeplume(s, kmt, ztopmax, ixx, imm, mpiid, alpha):
    varn = ''
    izprint = 0
    iconv = 0
    itime = 0
    k = 0
    kk = 0
    kkmax = 0
    deltak = 0
    ilastprint = 0
    nrectotal = 0
    i_micro = 0
    n_sub_step = 0
    wmax = f32(0.)
    rmaxtime = f32(0.)
    es = f32(0.)
    esat = f32(0.)
    heat = f32(0.)
    dt_save = f32(0.)
    cixx = ''
    delz_thresold = word(0x42c80000)
    dtknt = 0
    s[108, 0] = 0
    s[60, 0] = f32(word(0x40000000))
    s[59, 0] = f32(word(0x43fa0000))
    nrectotal = int(150)
    dtknt = int(0)
    s[91, 0] = int(1)
    ztopmax = f32(word(0x00000000))
    s[79, 0] = f32(word(0x00000000))
    s[89, 0] = f32(word(0x00000000))
    s[88, 0] = f32(word(0x3f800000))
    wmax = f32(word(0x3f800000))
    kkmax = int(10)
    deltak = int(20)
    ilastprint = int(0)
    s[63, 0] = int(1)
    gsl_initial(s, kmt, alpha)
    izprint = int(0)
    rmaxtime = f32(f32(int(s[93, 0])))
    while (s[89, 0] <= rmaxtime):
        s[62, 0] = int(min(kmt, (kkmax + deltak)))
        s[88, 0] = f32(fmax(word(0x3c23d70a), fmin(word(0x40a00000), fdiv(fsub(s[25][2], s[25][1]), fmul(s[60, 0], wmax)))))
        dtknt = int((dtknt + 1))
        s[89, 0] = f32(fadd(s[89, 0], s[88, 0]))
        s[108, 0] += 1
        s[91, 0] = int((1 + (int(s[89, 0]) // 60)))
        wmax = f32(word(0x3f800000))
        gsl_tend0_plumerise(s)
        s[63, 0] = int(1)
        gsl_lbound(s, alpha)
        gsl_vel_advectc_plumerise(s, int(s[62, 0]), s[15], s[16], s[9], s[23])
        gsl_scl_advectc_plumerise(s, 'sc', int(s[62, 0]))
        gsl_scl_misc(s, int(s[62, 0]), alpha)
        gsl_scl_dyn_entrain(s, int(s[62, 0]), 200, s[73, 0], s[0], s[72, 0], alpha, s[84], s[17], s[1], s[39], s[18], s[2], s[40], s[19], s[3], s[20], s[4], s[21], s[5], s[102], s[103], s[105], s[104], s[106])
        gsl_damp_grav_wave(s, 1, int(s[62, 0]), deltak, s[88, 0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40])
        dt_save = f32(s[88, 0])
        n_sub_step = int(3)
        s[88, 0] = f32(fdiv(s[88, 0], f32(n_sub_step)))
        i_micro = 1
        while i_micro <= n_sub_step:
            gsl_fallpart(s, int(s[62, 0]))
            s[63, 0] = int(2)
            while (int(s[63, 0]) <= (int(s[62, 0]) - 1)):
                s[73, 0] = f32(fmul(word(0x3f000000), fadd(s[0][int(s[63, 0])], s[0][(int(s[63, 0]) - 1)])))
                es = f32(gsl_esat_pr(s, s[1][int(s[63, 0])]))
                s[12][int(s[63, 0])] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][int(s[63, 0])], es)))
                s[11][int(s[63, 0])] = f32(es)
                s[9][int(s[63, 0])] = f32(fdiv(fmul(word(0x4559bccd), s[38][int(s[63, 0])]), s[1][int(s[63, 0])]))
                if (s[0][int(s[63, 0])] >= word(0x00000000)):
                    s[57, 0] = f32(fdiv(fsub(s[12][(int(s[63, 0]) + 1)], s[12][(int(s[63, 0]) - 1)]), fsub(s[26][(int(s[63, 0]) + 1)], s[26][(int(s[63, 0]) - 1)])))
                else:
                    s[57, 0] = f32(fdiv(fsub(s[12][(int(s[63, 0]) + 1)], s[12][(int(s[63, 0]) - 1)]), fsub(s[26][(int(s[63, 0]) + 1)], s[26][(int(s[63, 0]) - 1)])))
                gsl_waterbal(s)
                s[63, 0] = int((int(s[63, 0]) + 1))
            i_micro += 1
        s[88, 0] = f32(dt_save)
        gsl_visc_w(s, int(s[62, 0]), deltak, kmt)
        gsl_update_plumerise(s, int(s[62, 0]), 's')
        gsl_hadvance_plumerise(s, 1, int(s[62, 0]), s[88, 0], s[15], s[16], s[0], int(s[91, 0]))
        gsl_buoyancy_plumerise(s, int(s[62, 0]), s[1], s[39], s[2], s[40], s[4], s[5], s[3], s[16], s[33])
        gsl_entrainment(s, int(s[62, 0]), s[0], s[16], s[84], alpha)
        gsl_update_plumerise(s, int(s[62, 0]), 'w')
        gsl_hadvance_plumerise(s, 2, int(s[62, 0]), s[88, 0], s[15], s[16], s[0], int(s[91, 0]))
        k = 2
        while k <= int(s[62, 0]):
            es = f32(gsl_esat_pr(s, s[1][k]))
            s[12][k] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][k], es)))
            s[11][k] = f32(es)
            s[10][k] = f32(fsub(s[1][k], s[39][k]))
            s[9][k] = f32(fdiv(fmul(word(0x4559bccd), s[38][k]), s[1][k]))
            if (fabs(s[15][k]) > wmax):
                wmax = f32(fabs(s[15][k]))
            k += 1
        gsl_damp_grav_wave(s, 2, int(s[62, 0]), deltak, s[88, 0], s[26], s[25], s[0], s[1], s[17], s[2], s[4], s[5], s[3], s[39], s[38], s[40])
        k = 2
        while k <= int(s[62, 0]):
            s[84][k] = f32(s[104][k])
            k += 1
        kk = int(1)
        while (s[0][kk] > word(0x3f800000)):
            kk = int((kk + 1))
            s[79, 0] = f32(s[25][kk])
        s[107][int(s[91, 0])] = f32(s[79, 0])
        ztopmax = f32(fmax(s[79, 0], ztopmax))
        kkmax = int(max(kk, kkmax))
        if (int(s[91, 0]) > 10):
            if (fabs(fsub(s[107][int(s[91, 0])], s[107][(int(s[91, 0]) - 10)])) < delz_thresold):
                break
    if imm == 1:
        s[111, 0] = ztopmax
    else:
        s[112, 0] = ztopmax
    if imm == 1:
        s[109, 0] = s[108, 0]
    else:
        s[110, 0] = s[108, 0]
    return ztopmax
    return ztopmax

# physics/smoke_dust/module_smoke_plumerise.F90:710-738.
def gsl_burn(s, eflux, water):
    if (s[89, 0] > s[90, 0]):
        eflux = f32(word(0x358637bd))
        water = f32(word(0x00000000))
        return eflux, water
    else:
        eflux = f32(s[85][int(s[91, 0])])
        water = f32(fdiv(fmul(fmul(eflux, fdiv(s[88, 0], word(0x4b933f50))), fadd(word(0x3f000000), s[86, 0])), word(0x3f0ccccd)))
        water = f32(fmul(water, word(0x447a0000)))
    return eflux, water
    return eflux, water

# physics/smoke_dust/module_smoke_plumerise.F90:758-842.
def gsl_lbound(s, alpha):
    es = f32(0.)
    esat = f32(0.)
    eflux = f32(0.)
    water = f32(0.)
    pres = f32(0.)
    c1 = f32(0.)
    c2 = f32(0.)
    f = f32(0.)
    zv = f32(0.)
    denscor = f32(0.)
    xwater = f32(0.)
    s[4][1] = f32(s[4][2])
    s[5][1] = f32(s[5][2])
    s[3][1] = f32(word(0x00000000))
    eflux, water = gsl_burn(s, eflux, water)
    pres = f32(fmul(s[38][1], word(0x447a0000)))
    c1 = f32(fdiv(word(0x40a00000), fmul(word(0x40c00000), alpha)))
    c2 = f32(fmul(word(0x3f666666), alpha))
    f = f32(fdiv(eflux, fmul(fmul(pres, word(0x447b2666)), word(0x40490fd0))))
    f = f32(fmul(fmul(word(0x452ff46e), f), s[81, 0]))
    zv = f32(fmul(c1, s[82, 0]))
    s[0][1] = f32(fdiv(fmul(c1, fpow(fmul(c2, f), word(0x3eaaaaab))), fpow(zv, word(0x3eaaaaab))))
    denscor = f32(fdiv(fdiv(fdiv(fmul(c1, f), word(0x411ced67)), fpow(fmul(c2, f), word(0x3eaaaaab))), fpow(zv, word(0x3fd55555))))
    s[1][1] = f32(fdiv(s[39][1], fsub(word(0x3f800000), denscor)))
    s[15][1] = f32(s[0][1])
    s[103][1] = f32(word(0x00000000))
    s[104][1] = f32(s[82, 0])
    s[7][1] = f32(word(0xc0800000))
    s[8][1] = f32(word(0xc0400000))
    s[10][1] = f32(fsub(s[1][1], s[39][1]))
    s[58][1] = f32(s[59, 0])
    s[9][1] = f32(fdiv(fmul(word(0x4559bccd), s[38][1]), s[1][1]))
    xwater = f32(fdiv(water, fmax(word(0x1e3ce508), fmul(fmul(s[0][1], s[88, 0]), s[9][1]))))
    s[2][1] = f32(fadd(xwater, s[40][1]))
    es = f32(gsl_esat_pr(s, s[1][1]))
    s[11][1] = f32(es)
    s[12][1] = f32(fdiv(fmul(word(0x3f1f3b64), es), fmax(word(0x1e3ce508), fsub(s[38][1], es))))
    if (s[2][1] > s[12][1]):
        s[3][1] = f32(fadd(fsub(s[2][1], s[12][1]), s[3][1]))
        s[2][1] = f32(s[12][1])
    gsl_waterbal(s)
    return

# physics/smoke_dust/module_smoke_plumerise.F90:850-913.
def gsl_initial(s, kmt, alpha):
    isub = 0
    k = 0
    n1 = 0
    n2 = 0
    n3 = 0
    lbuoy = 0
    itmp = 0
    isubm1 = 0
    xn1 = f32(0.)
    xi = f32(0.)
    es = f32(0.)
    esat = f32(0.)
    s[61, 0] = int(kmt)
    k = 1
    while k <= int(s[61, 0]):
        s[10][k] = f32(word(0x00000000))
        s[0][k] = f32(word(0x00000000))
        s[1][k] = f32(s[39][k])
        s[15][k] = f32(word(0x00000000))
        s[16][k] = f32(word(0x00000000))
        s[2][k] = f32(s[40][k])
        s[7][k] = f32(word(0x00000000))
        s[8][k] = f32(word(0x00000000))
        s[4][k] = f32(word(0x00000000))
        s[5][k] = f32(word(0x00000000))
        s[3][k] = f32(word(0x00000000))
        es = f32(gsl_esat_pr(s, s[1][k]))
        s[11][k] = f32(es)
        s[12][k] = f32(fdiv(fmul(word(0x3f1f3b64), es), fsub(s[38][k], es)))
        s[9][k] = f32(fdiv(fmul(word(0x4559bccd), s[38][k]), s[1][k]))
        s[103][k] = f32(word(0x00000000))
        s[104][k] = f32(word(0x00000000))
        k += 1
    s[84][1] = f32(s[82, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[84][k] = f32(fadd(s[84][(k - 1)], fmul(fmul(word(0x3f99999a), alpha), fsub(s[26][k], s[26][(k - 1)]))))
        k += 1
    s[84][1] = f32(s[82, 0])
    s[104][1] = f32(s[82, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[84][k] = f32(fadd(s[84][(k - 1)], fmul(fmul(word(0x3f99999a), alpha), fsub(s[26][k], s[26][(k - 1)]))))
        s[104][k] = f32(s[84][k])
        k += 1
    s[58][1] = f32(s[59, 0])
    k = 2
    while k <= int(s[61, 0]):
        s[58][k] = f32(fmax(word(0x3a83126f), fsub(s[58][(k - 1)], fdiv(fmul(word(0x3f800000), s[59, 0]), word(0x43480000)))))
        k += 1
    gsl_lbound(s, alpha)
    return

# physics/smoke_dust/module_smoke_plumerise.F90:918-934.
def gsl_damp_grav_wave(s, ifrom, nm1, deltak, dt, zt, zm, w, t, tt, qv, qh, qi, qc, te, pe, qvenv):
    dummy = np.zeros(202, dtype=np.float32)
    if (ifrom == 1):
        gsl_friction(s, ifrom, nm1, deltak, dt, zt, zm, t, tt, te)
        return
    for _a in range(1, (200)+1):
        dummy[_a] = f32(word(0x00000000))
    if (ifrom == 2):
        gsl_friction(s, ifrom, nm1, deltak, dt, zt, zm, w, dummy, dummy)
    return

# physics/smoke_dust/module_smoke_plumerise.F90:939-969.
def gsl_friction(s, ifrom, nm1, deltak, dt, zt, zm, var1, vart, var2):
    k = 0
    nfpt = 0
    kf = 0
    zmkf = f32(0.)
    ztop = f32(0.)
    distim = f32(0.)
    c1 = f32(0.)
    c2 = f32(0.)
    kf = int((nm1 - int(deltak)))
    zmkf = f32(zm[kf])
    ztop = f32(zm[nm1])
    distim = f32(fmin(fmul(word(0x40400000), dt), word(0x42700000)))
    c1 = f32(fdiv(word(0x3f800000), fmul(distim, fsub(ztop, zmkf))))
    c2 = f32(fmul(dt, c1))
    if (ifrom == 1):
        k = nm1
        while k >= 2:
            if (zt[k] <= zmkf):
                k += -1
                continue
            vart[k] = f32(fadd(vart[k], fmul(fmul(c1, fsub(zt[k], zmkf)), fsub(var2[k], var1[k]))))
            k += -1
    elif (ifrom == 2):
        k = nm1
        while k >= 2:
            if (zt[k] <= zmkf):
                k += -1
                continue
            var1[k] = f32(fadd(var1[k], fmul(fmul(c2, fsub(zt[k], zmkf)), fsub(var2[k], var1[k]))))
            k += -1
    return

# physics/smoke_dust/module_smoke_plumerise.F90:975-1005.
def gsl_vel_advectc_plumerise(s, m1, wc, wt, rho, dzm):
    k = 0
    flxw = np.zeros(202, dtype=np.float32)
    dn0 = np.zeros(202, dtype=np.float32)
    c1z = f32(0.)
    for _a in range(1, (m1)+1):
        dn0[_a] = f32(fmul(rho[_a], word(0x3a83126f)))
    flxw[1] = f32(fmul(wc[1], dn0[1]))
    k = 2
    while k <= (m1 - 1):
        flxw[k] = f32(fmul(fmul(wc[k], word(0x3f000000)), fadd(dn0[k], dn0[(k + 1)])))
        k += 1
    c1z = f32(word(0x3f000000))
    k = 2
    while k <= (m1 - 2):
        wt[k] = f32(fadd(wt[k], fmul(fdiv(fmul(c1z, dzm[k]), fadd(dn0[k], dn0[(k + 1)])), fadd(fsub(fmul(fadd(flxw[k], flxw[(k - 1)]), fadd(wc[k], wc[(k - 1)])), fmul(fadd(flxw[k], flxw[(k + 1)]), fadd(wc[k], wc[(k + 1)]))), fmul(fmul(fsub(flxw[(k + 1)], flxw[(k - 1)]), word(0x40000000)), wc[k])))))
        k += 1
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1011-1031.
def gsl_hadvance_plumerise(s, iac, m1, dt, wc, wt, wp, mintime):
    k = 0
    dummy = np.zeros(202, dtype=np.float32)
    eps = f32(0.)
    eps = f32(word(0x3e4ccccd))
    if (mintime == 1):
        eps = f32(word(0x3f000000))
    gsl_predict_plumerise(s, m1, wc, wp, wt, dummy, iac, fmul(word(0x40000000), dt), eps)
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1036-1077.
def gsl_predict_plumerise(s, npts, ac, ap, fa, af, iac, dtlp, epsu):
    m = 0
    if (iac == 1):
        m = 1
        while m <= npts:
            ac[m] = f32(fadd(ac[m], fmul(epsu, fsub(ap[m], fmul(word(0x40000000), ac[m])))))
            m += 1
        return
    elif (iac == 2):
        m = 1
        while m <= npts:
            af[m] = f32(ap[m])
            ap[m] = f32(fadd(ac[m], fmul(epsu, af[m])))
            m += 1
    m = 1
    while m <= npts:
        ac[m] = f32(af[m])
        m += 1
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1082-1112.
def gsl_buoyancy_plumerise(s, m1, t, te, qv, qvenv, qh, qi, qc, wt, scr1):
    k = 0
    tv = f32(0.)
    tve = f32(0.)
    qwtotl = f32(0.)
    umgamai = f32(0.)
    umgamai = f32(word(0x3f2aaaab))
    k = 2
    while k <= (m1 - 1):
        tv = f32(fdiv(fmul(t[k], fadd(word(0x3f800000), fdiv(qv[k], word(0x3f1f3b64)))), fadd(word(0x3f800000), qv[k])))
        tve = f32(fdiv(fmul(te[k], fadd(word(0x3f800000), fdiv(qvenv[k], word(0x3f1f3b64)))), fadd(word(0x3f800000), qvenv[k])))
        qwtotl = f32(fadd(fadd(qh[k], qi[k]), qc[k]))
        scr1[k] = f32(fmul(fmul(word(0x411ccccd), umgamai), fsub(fdiv(fsub(tv, tve), tve), qwtotl)))
        k += 1
    k = 2
    while k <= (m1 - 2):
        wt[k] = f32(fadd(wt[k], fmul(word(0x3f000000), fadd(scr1[k], scr1[(k + 1)]))))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1118-1158.
def gsl_entrainment(s, m1, w, wt, radius, alpha):
    k = 0
    dmdtm = f32(0.)
    wbar = f32(0.)
    radius_bar = f32(0.)
    umgamai = f32(0.)
    dyn_entr = f32(0.)
    umgamai = f32(word(0x3f2aaaab))
    k = 2
    while k <= (m1 - 1):
        wbar = f32(w[k])
        radius_bar = f32(fmul(word(0x3f000000), fadd(radius[k], radius[(k - 1)])))
        dmdtm = f32(fdiv(fmul(fmul(fmul(umgamai, word(0x40000000)), alpha), fabs(wbar)), radius_bar))
        wt[k] = f32(fsub(wt[k], fmul(dmdtm, fabs(wbar))))
        dyn_entr = f32(fdiv(fmul(word(0x3ea2f96b), fabs(fsub(fadd(fsub(s[103][k], s[102][k]), s[103][(k - 1)]), s[102][(k - 1)]))), radius_bar))
        wt[k] = f32(fsub(wt[k], fmul(dyn_entr, fabs(wbar))))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1164-1287.
def gsl_scl_advectc_plumerise(s, varn, mzp):
    dtlto2 = f32(0.)
    k = 0
    dtlto2 = f32(fmul(word(0x3f000000), s[88, 0]))
    s[29][1] = f32(fmul(fmul(fmul(fadd(s[0][1], s[15][1]), dtlto2), s[9][1]), word(0x3a83126f)))
    s[30][1] = f32(fmul(fmul(fmul(word(0x3f000000), fadd(s[0][1], s[15][1])), dtlto2), s[23][1]))
    k = 2
    while k <= mzp:
        s[29][k] = f32(fmul(fmul(fmul(fmul(fadd(s[0][k], s[15][k]), dtlto2), word(0x3f000000)), fadd(s[9][k], s[9][(k + 1)])), word(0x3a83126f)))
        s[30][k] = f32(fmul(fmul(fmul(fadd(s[0][k], s[15][k]), dtlto2), word(0x3f000000)), s[23][k]))
        k += 1
    k = 1
    while k <= mzp:
        s[27][k] = f32(fmul(fsub(s[26][(k + 1)], s[25][k]), s[23][k]))
        s[28][k] = f32(fmul(fsub(s[25][k], s[26][k]), s[23][k]))
        s[31][k] = f32(fdiv(s[24][k], fmul(s[9][k], word(0x3a83126f))))
        k += 1
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[1][_a])
    gsl_fa_zc_plumerise(s, mzp, s[1], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[1], s[33], s[17], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[2][_a])
    gsl_fa_zc_plumerise(s, mzp, s[2], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[2], s[33], s[18], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[3][_a])
    gsl_fa_zc_plumerise(s, mzp, s[3], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[3], s[33], s[19], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[5][_a])
    gsl_fa_zc_plumerise(s, mzp, s[5], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[5], s[33], s[21], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[4][_a])
    gsl_fa_zc_plumerise(s, mzp, s[4], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[4], s[33], s[20], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[103][_a])
    gsl_fa_zc_plumerise(s, mzp, s[103], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[103], s[33], s[105], s[88, 0])
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[104][_a])
    gsl_fa_zc_plumerise(s, mzp, s[104], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[104], s[33], s[106], s[88, 0])
    return
    for _a in range(1, (200)+1):
        s[33][_a] = f32(s[6][_a])
    gsl_fa_zc_plumerise(s, mzp, s[6], s[33], s[29], s[30], s[32], s[31], s[27], s[28])
    gsl_advtndc_plumerise(s, mzp, s[6], s[33], s[22], s[88, 0])
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1293-1334.
def gsl_fa_zc_plumerise(s, m1, scp, scr1, vt3dc, vt3df, vt3dg, vt3dk, vctr1, vctr2):
    k = 0
    dfact = f32(0.)
    dfact = f32(word(0x3f000000))
    k = 1
    while k <= (m1 - 1):
        vt3dg[k] = f32(fmul(vt3dc[k], fadd(fadd(fmul(vctr1[k], scr1[k]), fmul(vctr2[k], scr1[(k + 1)])), fmul(vt3df[k], fsub(scr1[k], scr1[(k + 1)])))))
        k += 1
    k = 1
    while k <= (m1 - 1):
        if (vt3dc[k] > word(0x00000000)):
            if (fmul(vt3dg[k], vt3dk[k]) > fmul(dfact, scr1[k])):
                vt3dg[k] = f32(fmul(vt3dc[k], scr1[k]))
        elif (vt3dc[k] < word(0x00000000)):
            if (fmul((-vt3dg[k]), vt3dk[(k + 1)]) > fmul(dfact, scr1[(k + 1)])):
                vt3dg[k] = f32(fmul(vt3dc[k], scr1[(k + 1)]))
        k += 1
    k = 2
    while k <= (m1 - 1):
        scr1[k] = f32(fadd(scr1[k], fmul(vt3dk[k], fadd(fsub(vt3dg[(k - 1)], vt3dg[k]), fmul(scp[k], fsub(vt3dc[k], vt3dc[(k - 1)]))))))
        k += 1
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1339-1348.
def gsl_advtndc_plumerise(s, m1, scp, sca, sct, dtl):
    k = 0
    dtli = f32(0.)
    dtli = f32(fdiv(word(0x3f800000), dtl))
    k = 2
    while k <= (m1 - 1):
        sct[k] = f32(fadd(sct[k], fmul(fsub(sca[k], scp[k]), dtli)))
        k += 1
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1353-1362.
def gsl_tend0_plumerise(s):
    for _a in range(1, (int(s[62, 0]))+1):
        s[16][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[17][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[18][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[19][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[20][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[21][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[105][_a] = f32(word(0x00000000))
    for _a in range(1, (int(s[62, 0]))+1):
        s[106][_a] = f32(word(0x00000000))

# physics/smoke_dust/module_smoke_plumerise.F90:1370-1402.
def gsl_scl_misc(s, m1, alpha):
    k = 0
    dmdtm = f32(0.)
    k = 2
    while k <= (m1 - 1):
        s[73, 0] = f32(fmul(word(0x3f000000), fadd(s[0][k], s[0][(k - 1)])))
        s[72, 0] = f32(fdiv(fmul((-s[73, 0]), word(0x411cf5c3)), word(0x447b0000)))
        dmdtm = f32(fdiv(fmul(fmul(word(0x40000000), alpha), fabs(s[73, 0])), s[84][k]))
        s[17][k] = f32(fsub(fadd(s[17][k], s[72, 0]), fmul(dmdtm, fsub(s[1][k], s[39][k]))))
        s[18][k] = f32(fsub(s[18][k], fmul(dmdtm, fsub(s[2][k], s[40][k]))))
        s[19][k] = f32(fsub(s[19][k], fmul(dmdtm, s[3][k])))
        s[20][k] = f32(fsub(s[20][k], fmul(dmdtm, s[4][k])))
        s[21][k] = f32(fsub(s[21][k], fmul(dmdtm, s[5][k])))
        s[105][k] = f32(fsub(s[105][k], fmul(dmdtm, fsub(s[103][k], s[102][k]))))
        s[106][k] = f32(fadd(s[106][k], fmul(fmul(fmul(word(0x3f000000), dmdtm), word(0x3f99999a)), s[84][k])))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1408-1471.
def gsl_scl_dyn_entrain(s, m1, nkp, wbar, w, adiabat, alpha, radius, tt, t, te, qvt, qv, qvenv, qct, qc, qht, qh, qit, qi, vel_e, vel_p, vel_t, rad_p, rad_t):
    k = 0
    dmdtm = f32(0.)
    k = 2
    while k <= (m1 - 1):
        rad_t[k] = f32(fadd(rad_t[k], fdiv(fabs(fsub(vel_e[k], vel_p[k])), word(0x40490ff9))))
        dmdtm = f32(fdiv(fmul(word(0x3f22f96b), fabs(fsub(vel_e[k], vel_p[k]))), radius[k]))
        vel_t[k] = f32(fsub(vel_t[k], fmul(dmdtm, fsub(vel_p[k], vel_e[k]))))
        tt[k] = f32(fsub(tt[k], fmul(dmdtm, fsub(t[k], te[k]))))
        qvt[k] = f32(fsub(qvt[k], fmul(dmdtm, fsub(qv[k], qvenv[k]))))
        qct[k] = f32(fsub(qct[k], fmul(dmdtm, qc[k])))
        qht[k] = f32(fsub(qht[k], fmul(dmdtm, qh[k])))
        qit[k] = f32(fsub(qit[k], fmul(dmdtm, qi[k])))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1478-1525.
def gsl_visc_w(s, m1, deltak, kmt):
    k = 0
    m2 = 0
    dz1t = f32(0.)
    dz1m = f32(0.)
    dz2t = f32(0.)
    dz2m = f32(0.)
    d2wdz = f32(0.)
    d2tdz = f32(0.)
    d2qvdz = f32(0.)
    d2qhdz = f32(0.)
    d2qcdz = f32(0.)
    d2qidz = f32(0.)
    d2scdz = f32(0.)
    d2vel_pdz = f32(0.)
    d2rad_dz = f32(0.)
    printed = False
    m2 = int(min(m1, kmt))
    k = 2
    while k <= (m2 - 1):
        dz1t = f32(fmul(word(0x3f000000), fsub(s[26][(k + 1)], s[26][(k - 1)])))
        dz2t = f32(fdiv(s[58][k], fmul(dz1t, dz1t)))
        dz1m = f32(fmul(word(0x3f000000), fsub(s[25][(k + 1)], s[25][(k - 1)])))
        dz2m = f32(fdiv(s[58][k], fmul(dz1m, dz1m)))
        d2wdz = f32(fmul(fadd(fsub(s[0][(k + 1)], fmul(2, s[0][k])), s[0][(k - 1)]), dz2m))
        d2tdz = f32(fmul(fadd(fsub(s[1][(k + 1)], fmul(2, s[1][k])), s[1][(k - 1)]), dz2t))
        d2qvdz = f32(fmul(fadd(fsub(s[2][(k + 1)], fmul(2, s[2][k])), s[2][(k - 1)]), dz2t))
        d2qhdz = f32(fmul(fadd(fsub(s[4][(k + 1)], fmul(2, s[4][k])), s[4][(k - 1)]), dz2t))
        d2qcdz = f32(fmul(fadd(fsub(s[3][(k + 1)], fmul(2, s[3][k])), s[3][(k - 1)]), dz2t))
        d2qidz = f32(fmul(fadd(fsub(s[5][(k + 1)], fmul(2, s[5][k])), s[5][(k - 1)]), dz2t))
        d2vel_pdz = f32(fmul(fadd(fsub(s[103][(k + 1)], fmul(2, s[103][k])), s[103][(k - 1)]), dz2t))
        d2rad_dz = f32(fmul(fadd(fsub(s[104][(k + 1)], fmul(2, s[104][k])), s[104][(k - 1)]), dz2t))
        s[16][k] = f32(fadd(s[16][k], d2wdz))
        s[17][k] = f32(fadd(s[17][k], d2tdz))
        s[18][k] = f32(fadd(s[18][k], d2qvdz))
        s[19][k] = f32(fadd(s[19][k], d2qcdz))
        s[20][k] = f32(fadd(s[20][k], d2qhdz))
        s[21][k] = f32(fadd(s[21][k], d2qidz))
        s[105][k] = f32(fadd(s[105][k], d2vel_pdz))
        s[106][k] = f32(fadd(s[106][k], d2rad_dz))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1533-1573.
def gsl_update_plumerise(s, m1, varn):
    k = 0
    if (varn == 'w'):
        k = 2
        while k <= (m1 - 1):
            s[0][k] = f32(fadd(s[0][k], fmul(s[16][k], s[88, 0])))
            k += 1
        return
    else:
        k = 2
        while k <= (m1 - 1):
            s[1][k] = f32(fadd(s[1][k], fmul(s[17][k], s[88, 0])))
            s[2][k] = f32(fadd(s[2][k], fmul(s[18][k], s[88, 0])))
            s[3][k] = f32(fadd(s[3][k], fmul(s[19][k], s[88, 0])))
            s[4][k] = f32(fadd(s[4][k], fmul(s[20][k], s[88, 0])))
            s[5][k] = f32(fadd(s[5][k], fmul(s[21][k], s[88, 0])))
            s[2][k] = f32(fmax(word(0x00000000), s[2][k]))
            s[3][k] = f32(fmax(word(0x00000000), s[3][k]))
            s[4][k] = f32(fmax(word(0x00000000), s[4][k]))
            s[5][k] = f32(fmax(word(0x00000000), s[5][k]))
            s[103][k] = f32(fadd(s[103][k], fmul(s[105][k], s[88, 0])))
            s[104][k] = f32(fadd(s[104][k], fmul(s[106][k], s[88, 0])))
            k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1579-1639.
def gsl_fallpart(s, m1):
    k = 0
    vtc = f32(0.)
    dfhz = f32(0.)
    dfiz = f32(0.)
    dz1 = f32(0.)
    k = 2
    while k <= (m1 - 1):
        vtc = f32(fmul(word(0x40a36fb7), fpow(s[9][k], word(0x3e000000))))
        s[7][k] = f32(word(0xc0800000))
        s[75, 0] = f32(fadd(s[0][k], s[7][k]))
        s[70][k] = f32(fadd(word(0x3fcccccd), fmul(word(0x3a156c0d), fpow(fabs(s[75, 0]), word(0x3fc00000)))))
        s[8][k] = f32(word(0xc0400000))
        s[76, 0] = f32(fadd(s[0][k], s[8][k]))
        s[71][k] = f32(fadd(word(0x3fcccccd), fdiv(fmul(word(0x3a156c0d), fpow(fabs(s[76, 0]), word(0x3fc00000))), word(0x3f400000))))
        if (s[75, 0] >= word(0x00000000)):
            dfhz = f32(fdiv(fmul(s[4][k], fsub(fmul(s[9][k], s[7][k]), fmul(s[9][(k - 1)], s[7][(k - 1)]))), s[9][(k - 1)]))
        else:
            dfhz = f32(fdiv(fmul(s[4][k], fsub(fmul(s[9][(k + 1)], s[7][(k + 1)]), fmul(s[9][k], s[7][k]))), s[9][k]))
        if (s[76, 0] >= word(0x00000000)):
            dfiz = f32(fdiv(fmul(s[5][k], fsub(fmul(s[9][k], s[8][k]), fmul(s[9][(k - 1)], s[8][(k - 1)]))), s[9][(k - 1)]))
        else:
            dfiz = f32(fdiv(fmul(s[5][k], fsub(fmul(s[9][(k + 1)], s[8][(k + 1)]), fmul(s[9][k], s[8][k]))), s[9][k]))
        dz1 = f32(fsub(s[25][k], s[25][(k - 1)]))
        s[20][k] = f32(fsub(s[20][k], fdiv(dfhz, dz1)))
        s[21][k] = f32(fsub(s[21][k], fdiv(dfiz, dz1)))
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:1644-1667.
def gsl_waterbal(s):
    if (s[3][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[3][int(s[63, 0])] = f32(word(0x00000000))
    if (s[4][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[4][int(s[63, 0])] = f32(word(0x00000000))
    if (s[5][int(s[63, 0])] <= word(0x2edbe6ff)):
        s[5][int(s[63, 0])] = f32(word(0x00000000))
    gsl_evaporate(s)
    gsl_sublimate(s)
    gsl_glaciate(s)
    gsl_melt(s)
    gsl_convert(s)
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1675-1881.
def gsl_evaporate(s):
    evhdt = f32(0.)
    evidt = f32(0.)
    evrate = f32(0.)
    evap = f32(0.)
    sd = f32(0.)
    quant = f32(0.)
    dividend = f32(0.)
    divisor = f32(0.)
    devidt = f32(0.)
    sd = f32(fsub(s[12][int(s[63, 0])], s[2][int(s[63, 0])]))
    if (sd == word(0x00000000)):
        return
    evhdt = f32(word(0x00000000))
    evidt = f32(word(0x00000000))
    evrate = f32(fabs(fmul(s[73, 0], s[57, 0])))
    evap = f32(fmul(evrate, s[88, 0]))
    if (sd <= word(0x00000000)):
        if (evap >= fabs(sd)):
            s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
            s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
            s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
            return
        else:
            s[3][int(s[63, 0])] = f32(fadd(s[3][int(s[63, 0])], evap))
            s[2][int(s[63, 0])] = f32(fsub(s[2][int(s[63, 0])], evap))
            s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(evap, word(0x451ba0a4))))
            return
    else:
        if (evap <= s[3][int(s[63, 0])]):
            if (sd <= evap):
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                return
            else:
                sd = f32(fsub(sd, evap))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evap))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evap, word(0x451ba0a4))))
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], evap))
        else:
            if (sd <= s[3][int(s[63, 0])]):
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], sd))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                return
            else:
                sd = f32(fsub(sd, s[3][int(s[63, 0])]))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[3][int(s[63, 0])]))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[3][int(s[63, 0])], word(0x451ba0a4))))
                s[3][int(s[63, 0])] = f32(word(0x00000000))
        if (s[4][int(s[63, 0])] > word(0x2edbe6ff)):
            quant = f32(fmul(fsub(fsub(s[12][int(s[63, 0])], s[3][int(s[63, 0])]), s[2][int(s[63, 0])]), s[9][int(s[63, 0])]))
            evhdt = f32(fdiv(fmul(fmul(fmul(s[88, 0], word(0x3a0e9b39)), quant), fpow(fmul(s[4][int(s[63, 0])], s[9][int(s[63, 0])]), word(0x3f266666))), s[9][int(s[63, 0])]))
            if (evhdt <= s[4][int(s[63, 0])]):
                if (sd <= evhdt):
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], sd))
                    s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                    return
                else:
                    sd = f32(fsub(sd, evhdt))
                    s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evhdt))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evhdt, word(0x451ba0a4))))
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], evhdt))
            else:
                if (sd <= s[4][int(s[63, 0])]):
                    s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                    s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], sd))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x451ba0a4))))
                    return
                else:
                    sd = f32(fsub(sd, s[4][int(s[63, 0])]))
                    s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[4][int(s[63, 0])]))
                    s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[4][int(s[63, 0])], word(0x451ba0a4))))
                    s[4][int(s[63, 0])] = f32(word(0x00000000))
        if (s[5][int(s[63, 0])] <= word(0x2edbe6ff)):
            return
        dividend = f32(fmul(fmul(fmul(fpow(fdiv(word(0x49742400), s[9][int(s[63, 0])]), word(0x3ef33333)), fsub(fdiv(sd, s[12][int(s[63, 0])]), 1)), fpow(s[5][int(s[63, 0])], word(0x3f066666))), word(0x3f90a3d7)))
        divisor = f32(fadd(word(0x492ae600), fdiv(word(0x4a7a3e80), fmul(word(0x41200000), s[11][int(s[63, 0])]))))
        devidt = f32(fdiv(fmul((-s[71][int(s[63, 0])]), dividend), divisor))
        evidt = f32(fmul(devidt, s[88, 0]))
        if (evidt <= s[5][int(s[63, 0])]):
            if (sd <= evidt):
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], sd))
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x45306b59))))
                return
            else:
                sd = f32(fsub(sd, evidt))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], evidt))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(evidt, word(0x45306b59))))
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], evidt))
        else:
            if (sd <= s[5][int(s[63, 0])]):
                s[2][int(s[63, 0])] = f32(s[12][int(s[63, 0])])
                s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], sd))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(sd, word(0x45306b59))))
                return
            else:
                sd = f32(fsub(sd, s[5][int(s[63, 0])]))
                s[2][int(s[63, 0])] = f32(fadd(s[2][int(s[63, 0])], s[5][int(s[63, 0])]))
                s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(s[5][int(s[63, 0])], word(0x45306b59))))
                s[5][int(s[63, 0])] = f32(word(0x00000000))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1890-1958.
def gsl_convert(s):
    accrete = f32(0.)
    con = f32(0.)
    q = f32(0.)
    h = f32(0.)
    bc1 = f32(0.)
    bc2 = f32(0.)
    total = f32(0.)
    if (s[1][int(s[63, 0])] <= word(0x4386a666)):
        return
    if (s[3][int(s[63, 0])] == word(0x00000000)):
        return
    accrete = f32(word(0x00000000))
    con = f32(word(0x00000000))
    q = f32(fmul(s[9][int(s[63, 0])], s[3][int(s[63, 0])]))
    h = f32(fmul(s[9][int(s[63, 0])], s[4][int(s[63, 0])]))
    if (s[4][int(s[63, 0])] > word(0x00000000)):
        accrete = f32(fmul(fmul(word(0x3baa64c3), q), fpow(h, word(0x3f600000))))
    if (1 != 0):
        con = f32(fdiv(fmul(fmul(fmul(q, q), q), word(0x3e158106)), fmul(word(0x42700000), fadd(fmul(fmul(word(0x40a00000), q), word(0x3e158106)), word(0x4564c000)))))
    else:
        con = f32(fmax(word(0x00000000), fmul(word(0x3a83126f), fsub(q, word(0x3f000000)))))
    total = f32(fdiv(fmul(fadd(con, accrete), s[88, 0]), s[9][int(s[63, 0])]))
    if (total < s[3][int(s[63, 0])]):
        s[3][int(s[63, 0])] = f32(fsub(s[3][int(s[63, 0])], total))
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], total))
        return
    else:
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], s[3][int(s[63, 0])]))
        s[3][int(s[63, 0])] = f32(word(0x00000000))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:1966-2012.
def gsl_sublimate(s):
    dtsubh = f32(0.)
    dividend = f32(0.)
    divisor = f32(0.)
    subl = f32(0.)
    dtsubh = f32(word(0x00000000))
    if (s[1][int(s[63, 0])] > word(0x4386a666)):
        return
    if (s[2][int(s[63, 0])] <= s[12][int(s[63, 0])]):
        return
    dividend = f32(fmul(fmul(fmul(fpow(fdiv(word(0x49742400), s[9][int(s[63, 0])]), word(0x3ef33333)), fsub(fdiv(s[2][int(s[63, 0])], s[12][int(s[63, 0])]), 1)), fpow(s[5][int(s[63, 0])], word(0x3f066666))), word(0x3f90a3d7)))
    divisor = f32(fadd(word(0x492ae600), fdiv(word(0x4a7a3e80), fmul(word(0x41200000), s[11][int(s[63, 0])]))))
    dtsubh = f32(fabs(fdiv(dividend, divisor)))
    subl = f32(fmul(dtsubh, s[88, 0]))
    if (subl < s[2][int(s[63, 0])]):
        s[2][int(s[63, 0])] = f32(fsub(s[2][int(s[63, 0])], subl))
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], subl))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(subl, word(0x45306b59))))
        return
    else:
        s[5][int(s[63, 0])] = f32(s[2][int(s[63, 0])])
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(s[2][int(s[63, 0])], word(0x45306b59))))
        s[2][int(s[63, 0])] = f32(word(0x00000000))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:2025-2065.
def gsl_glaciate(s):
    dfrzh = f32(0.)
    dfrzh = f32(word(0x00000000))
    if (s[4][int(s[63, 0])] <= word(0x00000000)):
        return
    if (s[2][int(s[63, 0])] < s[12][int(s[63, 0])]):
        return
    if (s[1][int(s[63, 0])] > word(0x4386a666)):
        return
    dfrzh = f32(fmul(fmul(s[88, 0], word(0x3ccccccd)), s[4][int(s[63, 0])]))
    if (dfrzh < s[4][int(s[63, 0])]):
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], dfrzh))
        s[4][int(s[63, 0])] = f32(fsub(s[4][int(s[63, 0])], dfrzh))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(word(0x43a655ad), dfrzh)))
        return
    else:
        s[5][int(s[63, 0])] = f32(fadd(s[5][int(s[63, 0])], s[4][int(s[63, 0])]))
        s[1][int(s[63, 0])] = f32(fadd(s[1][int(s[63, 0])], fmul(word(0x43a655ad), s[4][int(s[63, 0])])))
        s[4][int(s[63, 0])] = f32(word(0x00000000))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:2076-2111.
def gsl_melt(s):
    dtmelt = f32(0.)
    dtmelt = f32(word(0x00000000))
    if (s[5][int(s[63, 0])] <= word(0x00000000)):
        return
    if (s[1][int(s[63, 0])] < word(0x43888000)):
        return
    dtmelt = f32(fmul(fmul(fmul(fmul(fmul(s[88, 0], fdiv(word(0x401147ae), s[9][int(s[63, 0])])), s[71][int(s[63, 0])]), fsub(s[1][int(s[63, 0])], word(0x43888000))), fpow(fmul(fmul(s[9][int(s[63, 0])], s[5][int(s[63, 0])]), word(0x358637bd)), word(0x3f066666))), word(0x3f90705d)))
    if (dtmelt < s[5][int(s[63, 0])]):
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], dtmelt))
        s[5][int(s[63, 0])] = f32(fsub(s[5][int(s[63, 0])], dtmelt))
        s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(word(0x43a6228f), dtmelt)))
        return
    else:
        s[4][int(s[63, 0])] = f32(fadd(s[4][int(s[63, 0])], s[5][int(s[63, 0])]))
        s[1][int(s[63, 0])] = f32(fsub(s[1][int(s[63, 0])], fmul(word(0x43a6228f), s[5][int(s[63, 0])])))
        s[5][int(s[63, 0])] = f32(word(0x00000000))
    return

# physics/smoke_dust/module_smoke_plumerise.F90:2116-2155.
def gsl_htint(s, nzz1, vctra, eleva, nzz2, vctrb, elevb):
    errmsg = ''
    errflg = 0
    l = 0
    k = 0
    kk = 0
    wt = f32(0.)
    l = int(1)
    k = 1
    while k <= nzz2:
        while True:
            if ((elevb[k] < eleva[1]) or ((elevb[k] >= eleva[l]) and (elevb[k] <= eleva[(l + 1)]))):
                wt = f32(fdiv(fsub(elevb[k], eleva[l]), fsub(eleva[(l + 1)], eleva[l])))
                vctrb[k] = f32(fadd(vctra[l], fmul(fsub(vctra[(l + 1)], vctra[l]), wt)))
                break
            elif (elevb[k] > eleva[nzz1]):
                wt = f32(fdiv(fsub(elevb[k], eleva[nzz1]), fsub(eleva[(nzz1 - 1)], eleva[nzz1])))
                vctrb[k] = f32(fadd(vctra[nzz1], fmul(fsub(vctra[(nzz1 - 1)], vctra[nzz1]), wt)))
                break
            l = int((l + 1))
            if (l == nzz1):
                kk = 1
                while kk <= l:
                    kk += 1
                pass
                pass
        k += 1

# physics/smoke_dust/module_smoke_plumerise.F90:2162-2185.
def gsl_esat_pr(s, tem):
    esat_pr = f32(0.)
    temc = f32(0.)
    esatm = f32(0.)
    temc = f32(fsub(tem, word(0x4388a666)))
    if (temc <= word(0xc2200000)):
        esatm = f32(fmul(word(0x40c39168), fexp(fdiv(fmul(word(0x41b45604), temc), fadd(temc, word(0x4388bd71))))))
        esat_pr = f32(fdiv(esatm, word(0x41200000)))
        return esat_pr
    esatm = f32(fmul(word(0x40c39653), fexp(fdiv(fmul(fsub(word(0x4195d4fe), fdiv(temc, word(0x43634ccd))), temc), fadd(temc, word(0x4380ef5c))))))
    esat_pr = f32(fdiv(esatm, word(0x41200000)))
    return esat_pr
    return esat_pr
