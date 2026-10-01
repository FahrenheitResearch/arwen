// Generated literal transcription. F: comments refer to module_cam_bl_eddy_diff.F.
// Constant expression subtrees are folded in source grouping to exact hex words.
#ifndef UWPBL_ZISOCL_CUH
#define UWPBL_ZISOCL_CUH
#define uw_cpair R8(UW_CPAIR)
#define uw_rair R8(UW_RAIR)
#define uw_zvir R8(UW_ZVIR)
#define uw_latvap R8(UW_LATVAP)
#define uw_latice R8(UW_LATICE)
#define uw_g R8(UW_GRAVIT)
#define uw_vk R8(UW_KARMAN)
#define uw_b123 R8(UW_B123)
#define uw_latsub (uw_latvap + uw_latice)
#define uw_ccon ((R8(0x1.1p+3) * R8(0x1.999999999999ap-4)) * uw_vk)

__device__ R8 uw_compute_cubic(R8 a, R8 b, R8 c) {
    R8 qq;
    R8 rr;
    R8 dd;
    R8 theta;
    R8 aa;
    R8 bb;
    R8 x1;
    R8 x2;
    R8 x3;
    qq = ((uw_sq(a) - (R8(0x1.8000000000000p+1) * b)) / R8(0x1.2000000000000p+3)); // F:4038 qq = (a**2-3._r8*b)/9._r8
    rr = ((((R8(0x1.0000000000000p+1) * uw_powi(a, 3)) - ((R8(0x1.2000000000000p+3) * a) * b)) + (R8(0x1.b000000000000p+4) * c)) / R8(0x1.b000000000000p+5)); // F:4039 rr = (2._r8*a**3 - 9._r8*a*b + 27._r8*c)/54._r8
    dd = (uw_sq(rr) - uw_powi(qq, 3)); // F:4041 dd = rr**2 - qq**3
    if ((dd <= R8(0x0.0p+0))) { // F:4042 if( dd .le. 0._r8 ) then
        theta = uw_acos((rr / uw_pow(qq, R8(0x1.8000000000000p+0)))); // F:4043 theta = acos(rr/qq**(3._r8/2._r8))
        x1 = ((-((R8(0x1.0000000000000p+1) * uw_sqrt(qq)) * uw_cos((theta / R8(0x1.8000000000000p+1))))) - (a / R8(0x1.8000000000000p+1))); // F:4044 x1 = -2._r8*sqrt(qq)*cos(theta/3._r8) - a/3._r8
        x2 = ((-((R8(0x1.0000000000000p+1) * uw_sqrt(qq)) * uw_cos(((theta + R8(0x1.921fb00000000p+2)) / R8(0x1.8000000000000p+1))))) - (a / R8(0x1.8000000000000p+1))); // F:4045 x2 = -2._r8*sqrt(qq)*cos((theta+2._r8*3.141592)/3._r8) - a/3._r8
        x3 = ((-((R8(0x1.0000000000000p+1) * uw_sqrt(qq)) * uw_cos(((theta - R8(0x1.921fb00000000p+2)) / R8(0x1.8000000000000p+1))))) - (a / R8(0x1.8000000000000p+1))); // F:4046 x3 = -2._r8*sqrt(qq)*cos((theta-2._r8*3.141592)/3._r8) - a/3._r8
        return uw_max(uw_max(uw_max(x1, x2), x3), R8(0x1.47ae147ae147bp-7)); // F:4047 compute_cubic = max(max(max(x1,x2),x3),xmin)
    } else {
        if ((rr >= R8(0x0.0p+0))) { // F:4050 if( rr .ge. 0._r8 ) then
            aa = (-uw_pow((uw_sqrt((uw_sq(rr) - uw_powi(qq, 3))) + rr), R8(0x1.5555555555555p-2))); // F:4051 aa = -(sqrt(rr**2-qq**3)+rr)**(1._r8/3._r8)
        } else {
            aa = uw_pow((uw_sqrt((uw_sq(rr) - uw_powi(qq, 3))) - rr), R8(0x1.5555555555555p-2)); // F:4053 aa =  (sqrt(rr**2-qq**3)-rr)**(1._r8/3._r8)
        }
        if ((aa == R8(0x0.0p+0))) { // F:4055 if( aa .eq. 0._r8 ) then
            bb = R8(0x0.0p+0); // F:4056 bb = 0._r8
        } else {
            bb = (qq / aa); // F:4058 bb = qq/aa
        }
        return uw_max(((aa + bb) - (a / R8(0x1.8000000000000p+1))), R8(0x1.47ae147ae147bp-7)); // F:4060 compute_cubic = max((aa+bb)-a/3._r8,xmin)
    }
}
__device__ void uw_exacol(int pver, V ri, R8 bflxs, R8 minpblh, V zi, VI ktop, VI kbase, int& ncvfin, Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    int k;
    int ncv;
    R8 rimaxentr;
    V riex = ws.r8((pver + 1));
    ncvfin = 0; // F:3114 ncvfin(i) = 0
    for (ncv = 1; ncv <= ncvmax; ncv += 1) { // F:3115 do ncv = 1, ncvmax
        ktop(ncv) = 0; // F:3116 ktop(i,ncv)  = 0
        kbase(ncv) = 0; // F:3117 kbase(i,ncv) = 0
    }
    rimaxentr = R8(0x0.0p+0); // F:3125 rimaxentr = 0._r8
    for (int uw_k = 2; uw_k <= pver; ++uw_k) { // F:3129 riex(2:pver) = ri(i,2:pver)
        riex(uw_k) = ri(uw_k);
    }
    riex((pver + 1)) = (rimaxentr - bflxs); // F:3134 riex(pver+1) = rimaxentr - bflxs(i)
    ncv = 0; // F:3136 ncv = 0
    k = (pver + 1); // F:3137 k   = pver + 1
    while ((k > 2)) { // F:3139 do while ( k .gt. ntop_turb + 1 )
        if ((riex(k) < rimaxentr)) { // F:3144 if( riex(k) .lt. rimaxentr ) then
            ncv = (ncv + 1); // F:3148 ncv = ncv + 1
            kbase(ncv) = uw_imin((k + 1), (pver + 1)); // F:3153 kbase(i,ncv) = min(k+1,pver+1)
            while (((riex(k) < rimaxentr) && (k > 2))) { // F:3157 do while( riex(k) .lt. rimaxentr .and. k .gt. ntop_turb + 1 )
                k = (k - 1); // F:3158 k = k - 1
            }
            ktop(ncv) = k; // F:3164 ktop(i,ncv) = k
        } else {
            k = (k - 1); // F:3170 k = k - 1
        }
    }
    ncvfin = ncv; // F:3176 ncvfin(i) = ncv
    return; // F:3180 return
}
__device__ void uw_zisocl(int pver, V z, V zi, V n2, V s2, V bprod, V sprod, R8 bflxs, R8 tkes, int& ncvfin, VI kbase, VI ktop, VL belongcv, V ricl, V ghcl, V shcl, V smcl, V lbrk, V wbrk, V ebrk, bool& extend, bool& extend_up, bool& extend_dn, Ws& ws) {
    const int ncvmax = pver;
    const int nbot_turb = pver;
    WsMark uw_mark(ws);
    bool bottom;
    int ncv;
    int incv;
    int k;
    int kb;
    int kt;
    int ncvinit;
    int cntu;
    int cntd;
    int kbinc;
    int ktinc;
    R8 wint;
    R8 dwinc;
    R8 dw_surf;
    R8 dzinc;
    R8 gh;
    R8 sh;
    R8 sm;
    R8 gh_surf;
    R8 sh_surf;
    R8 sm_surf;
    R8 l2n2;
    R8 l2s2;
    R8 dl2n2;
    R8 dl2s2;
    R8 dl2n2_surf;
    R8 dl2s2_surf;
    R8 lint;
    R8 dlint;
    R8 dlint_surf;
    R8 lbulk;
    R8 lz;
    R8 ricll;
    R8 trma;
    R8 trmb;
    R8 trmc;
    R8 det;
    R8 zbot;
    R8 l2rat;
    R8 gg;
    R8 tunlramp;
    for (k = 1; k <= ncvmax; k += 1) { // F:3308 do k = 1, ncvmax
        ricl(k) = R8(0x0.0p+0); // F:3309 ricl(i,k) = 0._r8
        ghcl(k) = R8(0x0.0p+0); // F:3310 ghcl(i,k) = 0._r8
        shcl(k) = R8(0x0.0p+0); // F:3311 shcl(i,k) = 0._r8
        smcl(k) = R8(0x0.0p+0); // F:3312 smcl(i,k) = 0._r8
        lbrk(k) = R8(0x0.0p+0); // F:3313 lbrk(i,k) = 0._r8
        wbrk(k) = R8(0x0.0p+0); // F:3314 wbrk(i,k) = 0._r8
        ebrk(k) = R8(0x0.0p+0); // F:3315 ebrk(i,k) = 0._r8
    }
    extend = false; // F:3317 extend    = .false.
    extend_up = false; // F:3318 extend_up = .false.
    extend_dn = false; // F:3319 extend_dn = .false.
    ncv = 1; // F:3325 ncv = 1
    while ((ncv <= ncvfin)) { // F:3327 do while( ncv .le. ncvfin(i) )
        ncvinit = ncv; // F:3329 ncvinit = ncv
        cntu = 0; // F:3330 cntu    = 0
        cntd = 0; // F:3331 cntd    = 0
        kb = kbase(ncv); // F:3332 kb      = kbase(i,ncv)
        kt = ktop(ncv); // F:3333 kt      = ktop(i,ncv)
        lbulk = (zi(kt) - zi(kb)); // F:3359 lbulk      = zi(i,kt) - zi(i,kb)
        dlint_surf = R8(0x0.0p+0); // F:3360 dlint_surf = 0._r8
        dl2n2_surf = R8(0x0.0p+0); // F:3361 dl2n2_surf = 0._r8
        dl2s2_surf = R8(0x0.0p+0); // F:3362 dl2s2_surf = 0._r8
        dw_surf = R8(0x0.0p+0); // F:3363 dw_surf    = 0._r8
        if ((kb == (pver + 1))) { // F:3364 if( kb .eq. pver+1 ) then
            if ((bflxs > R8(0x0.0p+0))) { // F:3366 if( bflxs(i) .gt. 0._r8 ) then
                gg = ((((R8(0x1.0000000000000p-1) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, R8(0x1.8000000000000p+0))); // F:3373 gg    = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                gh = (gg / (R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))); // F:3374 gh    = gg/(alph5-gg*alph3)
                gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3376 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
                sh = (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))); // F:3377 sh    = alph5/(1._r8+alph3*gh)
                sm = (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh))); // F:3378 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                ricll = uw_min((-((sm / sh) * (bprod((pver + 1)) / sprod((pver + 1))))), R8(0x1.851eb851eb852p-3)); // F:3379 ricll = min(-(sm/sh)*(bprod(i,pver+1)/sprod(i,pver+1)),ricrit)
                dlint_surf = z(pver); // F:3387 dlint_surf = z(i,pver)
                dl2n2_surf = (-(((uw_vk * uw_sq(z(pver))) * bprod((pver + 1))) / (sh * uw_sqrt(tkes)))); // F:3388 dl2n2_surf = -vk*(z(i,pver)**2)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                dl2s2_surf = (((uw_vk * uw_sq(z(pver))) * sprod((pver + 1))) / (sm * uw_sqrt(tkes))); // F:3389 dl2s2_surf =  vk*(z(i,pver)**2)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                dw_surf = ((tkes / R8(0x1.7333333333333p+2)) * z(pver)); // F:3390 dw_surf    = (tkes(i)/b1)*z(i,pver)
            } else {
                lbulk = (zi(kt) - z(pver)); // F:3396 lbulk = zi(i,kt) - z(i,pver)
            }
        }
        lint = dlint_surf; // F:3406 lint = dlint_surf
        l2n2 = dl2n2_surf; // F:3407 l2n2 = dl2n2_surf
        l2s2 = dl2s2_surf; // F:3408 l2s2 = dl2s2_surf
        wint = dw_surf; // F:3409 wint = dw_surf
        l2n2 = R8(0x0.0p+0); // F:3411 l2n2 = 0._r8
        l2s2 = R8(0x0.0p+0); // F:3412 l2s2 = 0._r8
        if ((kt < (kb - 1))) { // F:3421 if( kt .lt. kb - 1 ) then
            for (k = (kb - 1); k >= (kt + 1); k += -1) { // F:3423 do k = kb - 1, kt + 1, -1
                tunlramp = R8(0x1.051eb851eb852p-3); // F:3426 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3434 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z((k - 1)) - z(k)); // F:3439 dzinc = z(i,k-1) - z(i,k)
                l2n2 = (l2n2 + (((lz * lz) * n2(k)) * dzinc)); // F:3440 l2n2  = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2(k)) * dzinc)); // F:3441 l2s2  = l2s2 + lz*lz*s2(i,k)*dzinc
                lint = (lint + dzinc); // F:3442 lint  = lint + dzinc
            }
            ricll = uw_min((l2n2 / uw_max(l2s2, R8(0x1.19799812dea11p-40))), R8(0x1.851eb851eb852p-3)); // F:3451 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((R8(0x1.a8f03ff93f46ap+7) * ricll) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - (R8(-0x1.11f3168d8b188p+2) * ricll)))); // F:3452 trma  = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * R8(-0x1.466dc5d638866p+5)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ricll)) + R8(0x1.1cc63f141205cp-1)))); // F:3453 trmb  = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll; // F:3454 trmc  = ricll
            det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:3455 det   = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + uw_sqrt(det)) / R8(0x1.0000000000000p+1)) / trma); // F:3456 gh    = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3458 gh    = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))); // F:3459 sh    = alph5/(1._r8+alph3*gh)
            sm = (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh))); // F:3460 sm    = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = ((wint - (sh * l2n2)) + (sm * l2s2)); // F:3461 wint  = wint - sh*l2n2 + sm*l2s2
        } else {
            lint = dlint_surf; // F:3472 lint = dlint_surf
            l2n2 = dl2n2_surf; // F:3473 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf; // F:3474 l2s2 = dl2s2_surf
            wint = dw_surf; // F:3475 wint = dw_surf
        }
        l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3499 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
        l2s2 = uw_min(l2s2, ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sm))); // F:3500 l2s2 =  min( l2s2, tkemax*lint/(b1*sm))
        extend = false; // F:3530 extend = .false.
        tunlramp = R8(0x1.051eb851eb852p-3); // F:3535 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
        lz = uw_pow((uw_pow((uw_vk * zi(kt)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3543 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
        dzinc = (z((kt - 1)) - z(kt)); // F:3549 dzinc = z(i,kt-1)-z(i,kt)
        dl2n2 = (((lz * lz) * n2(kt)) * dzinc); // F:3550 dl2n2 = lz*lz*n2(i,kt)*dzinc
        dl2s2 = (((lz * lz) * s2(kt)) * dzinc); // F:3551 dl2s2 = lz*lz*s2(i,kt)*dzinc
        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3552 dwinc = -sh*dl2n2 + sm*dl2s2
        while ((((-dl2n2) > (-((R8(-0x1.47ae147ae147bp-5) * l2n2) / R8(0x1.0a3d70a3d70a4p+0)))) && ((kt - 1) > 1))) { // F:3563 do while ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) .and. kt-1 .gt. ntop_turb )
            lint = (lint + dzinc); // F:3574 lint = lint + dzinc
            l2n2 = (l2n2 + dl2n2); // F:3575 l2n2 = l2n2 + dl2n2
            l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3576 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
            l2s2 = (l2s2 + dl2s2); // F:3577 l2s2 = l2s2 + dl2s2
            wint = (wint + dwinc); // F:3578 wint = wint + dwinc
            kt = (kt - 1); // F:3582 kt        = kt - 1
            extend = true; // F:3583 extend    = .true.
            extend_up = true; // F:3584 extend_up = .true.
            if ((kt == 1)) { // F:3585 if( kt .eq. ntop_turb ) then
                *ws.err = 3590; return; // F:3590 stop
            }
            ktinc = (kbase(((ncv + cntu) + 1)) - 1); // F:3599 ktinc = kbase(i,ncv+cntu+1) - 1
            if ((kt == ktinc)) { // F:3601 if( kt .eq. ktinc ) then
                for (k = (kbase(((ncv + cntu) + 1)) - 1); k >= (ktop(((ncv + cntu) + 1)) + 1); k += -1) { // F:3603 do k = kbase(i,ncv+cntu+1) - 1, ktop(i,ncv+cntu+1) + 1, -1
                    tunlramp = R8(0x1.051eb851eb852p-3); // F:3606 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3614 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z((k - 1)) - z(k)); // F:3620 dzinc = z(i,k-1)-z(i,k)
                    dl2n2 = (((lz * lz) * n2(k)) * dzinc); // F:3621 dl2n2 = lz*lz*n2(i,k)*dzinc
                    dl2s2 = (((lz * lz) * s2(k)) * dzinc); // F:3622 dl2s2 = lz*lz*s2(i,k)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3623 dwinc = -sh*dl2n2 + sm*dl2s2
                    lint = (lint + dzinc); // F:3625 lint = lint + dzinc
                    l2n2 = (l2n2 + dl2n2); // F:3626 l2n2 = l2n2 + dl2n2
                    l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3627 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2); // F:3628 l2s2 = l2s2 + dl2s2
                    wint = (wint + dwinc); // F:3629 wint = wint + dwinc
                }
                kt = ktop(((ncv + cntu) + 1)); // F:3633 kt        = ktop(i,ncv+cntu+1)
                ncvfin = (ncvfin - 1); // F:3634 ncvfin(i) = ncvfin(i) - 1
                cntu = (cntu + 1); // F:3635 cntu      = cntu + 1
            }
            tunlramp = R8(0x1.051eb851eb852p-3); // F:3643 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = uw_pow((uw_pow((uw_vk * zi(kt)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3651 lz = ( (vk*zi(i,kt))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z((kt - 1)) - z(kt)); // F:3657 dzinc = z(i,kt-1)-z(i,kt)
            dl2n2 = (((lz * lz) * n2(kt)) * dzinc); // F:3658 dl2n2 = lz*lz*n2(i,kt)*dzinc
            dl2s2 = (((lz * lz) * s2(kt)) * dzinc); // F:3659 dl2s2 = lz*lz*s2(i,kt)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3660 dwinc = -sh*dl2n2 + sm*dl2s2
        }
        if ((cntu > 0)) { // F:3670 if( cntu .gt. 0 ) then
            for (incv = 1; incv <= (ncvfin - ncv); incv += 1) { // F:3671 do incv = 1, ncvfin(i) - ncv
                kbase((ncv + incv)) = kbase(((ncv + cntu) + incv)); // F:3672 kbase(i,ncv+incv) = kbase(i,ncv+cntu+incv)
                ktop((ncv + incv)) = ktop(((ncv + cntu) + incv)); // F:3673 ktop(i,ncv+incv)  = ktop(i,ncv+cntu+incv)
            }
        }
        if ((kb != (pver + 1))) { // F:3681 if( kb .ne. pver + 1 ) then
            tunlramp = R8(0x1.051eb851eb852p-3); // F:3686 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
            lz = uw_pow((uw_pow((uw_vk * zi(kb)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3694 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
            dzinc = (z((kb - 1)) - z(kb)); // F:3700 dzinc = z(i,kb-1)-z(i,kb)
            dl2n2 = (((lz * lz) * n2(kb)) * dzinc); // F:3701 dl2n2 = lz*lz*n2(i,kb)*dzinc
            dl2s2 = (((lz * lz) * s2(kb)) * dzinc); // F:3702 dl2s2 = lz*lz*s2(i,kb)*dzinc
            dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3703 dwinc = -sh*dl2n2 + sm*dl2s2
            while ((((-dl2n2) > (-((R8(-0x1.47ae147ae147bp-5) * l2n2) / R8(0x1.0a3d70a3d70a4p+0)))) && (kb != (pver + 1)))) { // F:3716 do while( ( -dl2n2 .gt. (-rinc*l2n2/(1._r8-rinc)) ) .and.(kb.ne.pver+1))
                lint = (lint + dzinc); // F:3721 lint = lint + dzinc
                l2n2 = (l2n2 + dl2n2); // F:3722 l2n2 = l2n2 + dl2n2
                l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3723 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                l2s2 = (l2s2 + dl2s2); // F:3724 l2s2 = l2s2 + dl2s2
                wint = (wint + dwinc); // F:3725 wint = wint + dwinc
                kb = (kb + 1); // F:3729 kb        =  kb + 1
                extend = true; // F:3730 extend    = .true.
                extend_dn = true; // F:3731 extend_dn = .true.
                kbinc = 0; // F:3744 kbinc = 0
                if ((ncv > 1)) { // F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                    kbinc = (ktop((ncv - 1)) + 1); // F:3745 if( ncv .gt. 1 ) kbinc = ktop(i,ncv-1) + 1
                }
                if ((kb == kbinc)) { // F:3746 if( kb .eq. kbinc ) then
                    for (k = (ktop((ncv - 1)) + 1); k <= (kbase((ncv - 1)) - 1); k += 1) { // F:3748 do k =  ktop(i,ncv-1) + 1, kbase(i,ncv-1) - 1
                        tunlramp = R8(0x1.051eb851eb852p-3); // F:3751 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                        lz = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3759 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                        dzinc = (z((k - 1)) - z(k)); // F:3765 dzinc = z(i,k-1)-z(i,k)
                        dl2n2 = (((lz * lz) * n2(k)) * dzinc); // F:3766 dl2n2 = lz*lz*n2(i,k)*dzinc
                        dl2s2 = (((lz * lz) * s2(k)) * dzinc); // F:3767 dl2s2 = lz*lz*s2(i,k)*dzinc
                        dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3768 dwinc = -sh*dl2n2 + sm*dl2s2
                        lint = (lint + dzinc); // F:3770 lint = lint + dzinc
                        l2n2 = (l2n2 + dl2n2); // F:3771 l2n2 = l2n2 + dl2n2
                        l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3772 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                        l2s2 = (l2s2 + dl2s2); // F:3773 l2s2 = l2s2 + dl2s2
                        wint = (wint + dwinc); // F:3774 wint = wint + dwinc
                    }
                    kb = kbase((ncv - 1)); // F:3781 kb        = kbase(i,ncv-1)
                    ncv = (ncv - 1); // F:3782 ncv       = ncv - 1
                    ncvfin = (ncvfin - 1); // F:3783 ncvfin(i) = ncvfin(i) -1
                    cntd = (cntd + 1); // F:3784 cntd      = cntd + 1
                }
                if ((kb == (pver + 1))) { // F:3792 if( kb .eq. pver + 1 ) then
                    if ((bflxs > R8(0x0.0p+0))) { // F:3794 if( bflxs(i) .gt. 0._r8 ) then
                        gg = ((((R8(0x1.0000000000000p-1) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, R8(0x1.8000000000000p+0))); // F:3796 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                        gh_surf = (gg / (R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))); // F:3797 gh_surf = gg/(alph5-gg*alph3)
                        gh_surf = uw_min(uw_max(gh_surf, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3799 gh_surf = min(max(gh_surf,-3.5334_r8),0.0233_r8)
                        sh_surf = (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh_surf))); // F:3800 sh_surf = alph5/(1._r8+alph3*gh_surf)
                        sm_surf = (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh_surf)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh_surf))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh_surf))); // F:3801 sm_surf = (alph1 + alph2*gh_surf)/(1._r8+alph3*gh_surf)/(1._r8+alph4*gh_surf)
                        dlint_surf = z(pver); // F:3804 dlint_surf = z(i,pver)
                        dl2n2_surf = (-(((uw_vk * uw_pow(z(pver), R8(0x1.0000000000000p+1))) * bprod((pver + 1))) / (sh_surf * uw_sqrt(tkes)))); // F:3805 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh_surf*sqrt(tkes(i)))
                        dl2s2_surf = (((uw_vk * uw_pow(z(pver), R8(0x1.0000000000000p+1))) * sprod((pver + 1))) / (sm_surf * uw_sqrt(tkes))); // F:3806 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm_surf*sqrt(tkes(i)))
                        dw_surf = ((tkes / R8(0x1.7333333333333p+2)) * z(pver)); // F:3807 dw_surf = (tkes(i)/b1)*z(i,pver)
                    } else {
                        dlint_surf = R8(0x0.0p+0); // F:3809 dlint_surf = 0._r8
                        dl2n2_surf = R8(0x0.0p+0); // F:3810 dl2n2_surf = 0._r8
                        dl2s2_surf = R8(0x0.0p+0); // F:3811 dl2s2_surf = 0._r8
                        dw_surf = R8(0x0.0p+0); // F:3812 dw_surf = 0._r8
                    }
                    lint = (lint + dlint_surf); // F:3821 lint = lint + dlint_surf
                    l2n2 = (l2n2 + dl2n2_surf); // F:3822 l2n2 = l2n2 + dl2n2_surf
                    l2n2 = (-uw_min((-l2n2), ((R8(0x1.4000000000000p+4) * lint) / (R8(0x1.7333333333333p+2) * sh)))); // F:3823 l2n2 = -min(-l2n2, tkemax*lint/(b1*sh))
                    l2s2 = (l2s2 + dl2s2_surf); // F:3824 l2s2 = l2s2 + dl2s2_surf
                    wint = (wint + dw_surf); // F:3825 wint = wint + dw_surf
                } else {
                    tunlramp = R8(0x1.051eb851eb852p-3); // F:3830 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                    lz = uw_pow((uw_pow((uw_vk * zi(kb)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3838 lz = ( (vk*zi(i,kb))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                    dzinc = (z((kb - 1)) - z(kb)); // F:3844 dzinc = z(i,kb-1)-z(i,kb)
                    dl2n2 = (((lz * lz) * n2(kb)) * dzinc); // F:3845 dl2n2 = lz*lz*n2(i,kb)*dzinc
                    dl2s2 = (((lz * lz) * s2(kb)) * dzinc); // F:3846 dl2s2 = lz*lz*s2(i,kb)*dzinc
                    dwinc = ((-(sh * dl2n2)) + (sm * dl2s2)); // F:3847 dwinc = -sh*dl2n2 + sm*dl2s2
                }
            }
            if (((kb == (pver + 1)) && (ncv != 1))) { // F:3853 if( (kb.eq.pver+1) .and. (ncv.ne.1) ) then
                *ws.err = 3858; return; // F:3858 stop
            }
        }
        if ((cntd > 0)) { // F:3869 if( cntd .gt. 0 ) then
            for (incv = 1; incv <= (ncvfin - ncv); incv += 1) { // F:3870 do incv = 1, ncvfin(i) - ncv
                kbase((ncv + incv)) = kbase((ncvinit + incv)); // F:3871 kbase(i,ncv+incv) = kbase(i,ncvinit+incv)
                ktop((ncv + incv)) = ktop((ncvinit + incv)); // F:3872 ktop(i,ncv+incv)  = ktop(i,ncvinit+incv)
            }
        }
        if ((wint < R8(0x1.47ae147ae147bp-7))) { // F:3878 if( wint .lt. 0.01_r8 ) then
            wint = R8(0x1.47ae147ae147bp-7); // F:3879 wint = 0.01_r8
        }
        if (extend) { // F:3892 if( extend ) then
            ktop(ncv) = kt; // F:3894 ktop(i,ncv)  = kt
            kbase(ncv) = kb; // F:3895 kbase(i,ncv) = kb
            lbulk = (zi(kt) - zi(kb)); // F:3901 lbulk      = zi(i,kt) - zi(i,kb)
            dlint_surf = R8(0x0.0p+0); // F:3902 dlint_surf = 0._r8
            dl2n2_surf = R8(0x0.0p+0); // F:3903 dl2n2_surf = 0._r8
            dl2s2_surf = R8(0x0.0p+0); // F:3904 dl2s2_surf = 0._r8
            dw_surf = R8(0x0.0p+0); // F:3905 dw_surf    = 0._r8
            if ((kb == (pver + 1))) { // F:3906 if( kb .eq. pver + 1 ) then
                if ((bflxs > R8(0x0.0p+0))) { // F:3907 if( bflxs(i) .gt. 0._r8 ) then
                    gg = ((((R8(0x1.0000000000000p-1) * uw_vk) * z(pver)) * bprod((pver + 1))) / uw_pow(tkes, R8(0x1.8000000000000p+0))); // F:3909 gg = 0.5_r8*vk*z(i,pver)*bprod(i,pver+1)/(tkes(i)**(3._r8/2._r8))
                    gh = (gg / (R8(0x1.65aee631f8a09p-1) - (gg * R8(-0x1.15694467381d8p+5)))); // F:3910 gh = gg/(alph5-gg*alph3)
                    gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3912 gh = min(max(gh,-3.5334_r8),0.0233_r8)
                    sh = (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))); // F:3913 sh = alph5/(1._r8+alph3*gh)
                    sm = (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh))); // F:3914 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
                    dlint_surf = z(pver); // F:3917 dlint_surf = z(i,pver)
                    dl2n2_surf = (-(((uw_vk * uw_pow(z(pver), R8(0x1.0000000000000p+1))) * bprod((pver + 1))) / (sh * uw_sqrt(tkes)))); // F:3918 dl2n2_surf = -vk*(z(i,pver)**2._r8)*bprod(i,pver+1)/(sh*sqrt(tkes(i)))
                    dl2s2_surf = (((uw_vk * uw_pow(z(pver), R8(0x1.0000000000000p+1))) * sprod((pver + 1))) / (sm * uw_sqrt(tkes))); // F:3919 dl2s2_surf =  vk*(z(i,pver)**2._r8)*sprod(i,pver+1)/(sm*sqrt(tkes(i)))
                    dw_surf = ((tkes / R8(0x1.7333333333333p+2)) * z(pver)); // F:3920 dw_surf    = (tkes(i)/b1)*z(i,pver)
                } else {
                    lbulk = (zi(kt) - z(pver)); // F:3922 lbulk = zi(i,kt) - z(i,pver)
                }
            }
            lint = dlint_surf; // F:3925 lint = dlint_surf
            l2n2 = dl2n2_surf; // F:3926 l2n2 = dl2n2_surf
            l2s2 = dl2s2_surf; // F:3927 l2s2 = dl2s2_surf
            wint = dw_surf; // F:3928 wint = dw_surf
            l2n2 = R8(0x0.0p+0); // F:3930 l2n2 = 0._r8
            l2s2 = R8(0x0.0p+0); // F:3931 l2s2 = 0._r8
            for (k = (kt + 1); k <= (kb - 1); k += 1) { // F:3940 do k = kt + 1, kb - 1
                tunlramp = R8(0x1.051eb851eb852p-3); // F:3942 tunlramp = 0.5_r8*(1._r8+ctunl)*tunl
                lz = uw_pow((uw_pow((uw_vk * zi(k)), R8(-0x1.8000000000000p+1)) + uw_pow((tunlramp * lbulk), R8(-0x1.8000000000000p+1))), R8(-0x1.5555555555555p-2)); // F:3950 lz = ( (vk*zi(i,k))**(-cleng) + (tunlramp*lbulk)**(-cleng) )**(-1._r8/cleng)
                dzinc = (z((k - 1)) - z(k)); // F:3955 dzinc = z(i,k-1) - z(i,k)
                lint = (lint + dzinc); // F:3956 lint = lint + dzinc
                l2n2 = (l2n2 + (((lz * lz) * n2(k)) * dzinc)); // F:3957 l2n2 = l2n2 + lz*lz*n2(i,k)*dzinc
                l2s2 = (l2s2 + (((lz * lz) * s2(k)) * dzinc)); // F:3958 l2s2 = l2s2 + lz*lz*s2(i,k)*dzinc
            }
            ricll = uw_min((l2n2 / uw_max(l2s2, R8(0x1.19799812dea11p-40))), R8(0x1.851eb851eb852p-3)); // F:3961 ricll = min(l2n2/max(l2s2,ntzero),ricrit)
            trma = ((R8(0x1.a8f03ff93f46ap+7) * ricll) + (R8(0x1.7333333333333p+3) * (R8(-0x1.174bc6a7ef9dbp+2) - (R8(-0x1.11f3168d8b188p+2) * ricll)))); // F:3962 trma = alph3*alph4*ricll+2._r8*b1*(alph2-alph4*alph5*ricll)
            trmb = ((ricll * R8(-0x1.466dc5d638866p+5)) + (R8(0x1.7333333333333p+3) * ((-(R8(0x1.65aee631f8a09p-1) * ricll)) + R8(0x1.1cc63f141205cp-1)))); // F:3963 trmb = ricll*(alph3+alph4)+2._r8*b1*(-alph5*ricll+alph1)
            trmc = ricll; // F:3964 trmc = ricll
            det = uw_max(((trmb * trmb) - ((R8(0x1.0000000000000p+2) * trma) * trmc)), R8(0x0.0p+0)); // F:3965 det = max(trmb*trmb-4._r8*trma*trmc,0._r8)
            gh = ((((-trmb) + uw_sqrt(det)) / R8(0x1.0000000000000p+1)) / trma); // F:3966 gh = (-trmb + sqrt(det))/2._r8/trma
            gh = uw_min(uw_max(gh, R8(-0x1.c4467381d7dbfp+1)), R8(0x1.7dbf487fcb924p-6)); // F:3968 gh = min(max(gh,-3.5334_r8),0.0233_r8)
            sh = (R8(0x1.65aee631f8a09p-1) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))); // F:3969 sh = alph5 / (1._r8+alph3*gh)
            sm = (((R8(0x1.1cc63f141205cp-1) + (R8(-0x1.174bc6a7ef9dbp+2) * gh)) / (R8(0x1.0000000000000p+0) + (R8(-0x1.15694467381d8p+5) * gh))) / (R8(0x1.0000000000000p+0) + (R8(-0x1.88240b780346ep+2) * gh))); // F:3970 sm = (alph1 + alph2*gh)/(1._r8+alph3*gh)/(1._r8+alph4*gh)
            wint = uw_max(((wint - (sh * l2n2)) + (sm * l2s2)), R8(0x1.47ae147ae147bp-7)); // F:3974 wint = max( wint - sh*l2n2 + sm*l2s2, 0.01_r8 )
        }
        lbrk(ncv) = lint; // F:3982 lbrk(i,ncv) = lint
        wbrk(ncv) = (wint / lint); // F:3983 wbrk(i,ncv) = wint/lint
        ebrk(ncv) = (R8(0x1.7333333333333p+2) * wbrk(ncv)); // F:3984 ebrk(i,ncv) = b1*wbrk(i,ncv)
        ebrk(ncv) = uw_min(ebrk(ncv), R8(0x1.4000000000000p+4)); // F:3985 ebrk(i,ncv) = min(ebrk(i,ncv),tkemax)
        ricl(ncv) = ricll; // F:3986 ricl(i,ncv) = ricll
        ghcl(ncv) = gh; // F:3987 ghcl(i,ncv) = gh
        shcl(ncv) = sh; // F:3988 shcl(i,ncv) = sh
        smcl(ncv) = sm; // F:3989 smcl(i,ncv) = sm
        ncv = (ncv + 1); // F:3996 ncv = ncv + 1
    }
    for (ncv = (ncvfin + 1); ncv <= ncvmax; ncv += 1) { // F:4004 do ncv = ncvfin(i) + 1, ncvmax
        ktop(ncv) = 0; // F:4005 ktop(i,ncv)  = 0
        kbase(ncv) = 0; // F:4006 kbase(i,ncv) = 0
    }
    for (k = 1; k <= (pver + 1); k += 1) { // F:4014 do k = 1, pver + 1
        belongcv(k) = false; // F:4015 belongcv(i,k) = .false.
    }
    for (ncv = 1; ncv <= ncvfin; ncv += 1) { // F:4018 do ncv = 1, ncvfin(i)
        for (k = ktop(ncv); k <= kbase(ncv); k += 1) { // F:4019 do k = ktop(i,ncv), kbase(i,ncv)
            belongcv(k) = true; // F:4020 belongcv(i,k) = .true.
        }
    }
    return; // F:4024 return
}
#endif
