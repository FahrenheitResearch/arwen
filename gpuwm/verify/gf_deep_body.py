"""``CUP_gf`` itself: module_cu_gf_deep.F:39-1868, one column, ``imid = 0``.

Split out of :mod:`gpuwm.verify.gf_deep_ref` only because the routine is
1500 lines of Fortran and the 15 procedures it calls are useful on their own
(``CUP_gf_sh`` reaches seven of them).  Everything here is float32 in WRF's
association order; see the sibling module's docstring for the precision rules
and for the one place -- ``tgammaf`` -- where this port is not bitwise.

Stage names in the returned dict are the oracle's column names in
``gpuwm/data/gf/oracle/gf-deep-levels.csv`` and ``gf-deep-surface.csv``, so a
gate can address any boundary in the routine without re-deriving it.
"""

from __future__ import annotations

import numpy as np

from gpuwm.verify.gf_deep_ref import (
    _HALF,
    _Z0,
    _Z1,
    _a,
    _maxloc,
    BETAJB,
    C1,
    CP,
    F,
    FLUXTUNE,
    FRH_THRESH,
    G,
    MAXENS3,
    PGCD,
    RH_THRESH,
    R_V,
    XLV,
    _TINY32,
    cup_dd_edt,
    cup_dd_moisture,
    cup_env,
    cup_env_clev,
    cup_forcing_ens_3d,
    cup_kbcon,
    cup_maximi,
    cup_minimi,
    cup_output_ens_3d,
    cup_up_aa0,
    cup_up_aa1bl,
    cup_up_moisture,
    get_cloud_bc,
    get_inversion_layers,
    get_lateral_massflux,
    get_zu_zd_pdf_fim,
    _powf,
    rates_up_pdf_deep,
)

__all__ = ["cup_gf_column"]


def cup_gf_column(
    *, zo, t, q, z1, tn, qo, po, psur, us, vs, rho, hfx, qfx, xland, dx, omeg,
    kpbl, ccn, dtime, mconv=0.0, csum=0, ichoice=0, dicycle=1, xmbs_in=0.0,
    fzu_up=None, fzu_dn=None,
):
    """Run one column through the deep cloud model.

    Inputs are ``gf_ref.gf_driver_prep``'s prepared column: ``(nz,)`` float32
    per-level arrays and float32 per-column scalars.  ``mconv`` is accepted
    for signature fidelity and then ignored, exactly as WRF ignores it --
    ``CUP_gf:1660`` zeroes it and rebuilds it on the cloud grid before its
    only use.
    """
    nz = int(np.asarray(t).shape[0])
    ktf = nz
    kte = nz

    def lift(x):
        out = _a(nz)
        out[1:] = np.asarray(x, dtype=F)
        return out

    zo = lift(zo)
    t = lift(t)
    q = lift(q)
    tn = lift(tn)
    qo = lift(qo)
    po = lift(po)
    us = lift(us)
    vs = lift(vs)
    rho = lift(rho)
    omeg = lift(omeg)
    z1 = F(z1)
    psur = F(psur)
    hfx = F(hfx)
    qfx = F(qfx)
    xland = F(xland)
    dx = F(dx)
    ccn = F(ccn)
    dtime = F(dtime)
    xmbs_in = F(xmbs_in)
    kpbl = int(kpbl)
    R = {}

    ierr = 0
    kbcon = 0
    ktop = 0
    k22 = 0
    jmin = 0

    # ---- :359-406 : w*, and the temperature/moisture excesses --------------
    flux_tun = FLUXTUNE
    lambau = F(2.0)
    pgcon = _Z0
    ztexec = _Z0
    zqexec = _Z0
    # `0.608*t(i,1)*qfx(i)/xlv` is four left-associated operations, not
    # `(0.608*t)*(qfx/xlv)`.  Both spellings happen to agree on all 216
    # columns of this fixture; the source-faithful one is what ships, and
    # CUP_gf_sh:301 is the identical line.
    buo_flux = F(
        F(F(hfx / CP) + F(F(F(F(0.608) * t[1]) * qfx) / XLV)) / rho[1]
    )
    zws = max(_Z0, F(F(F(F(F(flux_tun * F(0.41)) * buo_flux) * zo[2]) * G) / t[1]))
    if zws > _TINY32:
        zws = F(F(1.2) * _powf(zws, F(0.3333)))
        ztexec = max(F(F(flux_tun * hfx) / F(F(rho[1] * zws) * CP)), _Z0)
        zqexec = max(F(F(F(flux_tun * qfx) / XLV) / F(rho[1] * zws)), _Z0)
    zws = max(
        _Z0, F(F(F(F(F(flux_tun * F(0.41)) * buo_flux) * zo[kpbl]) * G) / t[kpbl])
    )
    zws = F(F(1.2) * _powf(zws, F(0.3333)))
    zws = F(zws * rho[kpbl])
    R["zws"] = zws
    R["ztexec"] = ztexec
    R["zqexec"] = zqexec

    # ---- :409-433 ----------------------------------------------------------
    cap_maxs = F(75.0)
    closure_n = F(16.0)
    cap_max = cap_maxs
    cap_max_increment = F(20.0)
    xland1 = int(xland + F(0.0001))
    if xland > F(1.5) or xland < F(0.5):
        xland1 = 0
        cap_max_increment = F(20.0)
    else:
        if ztexec > _Z0:
            cap_max = F(cap_max + F(25.0))
        if ztexec < _Z0:
            cap_max = F(cap_max - F(25.0))
    R["cap_max"] = cap_max
    R["xland1"] = xland1

    # ---- :455-471 : entrainment rate, radius, and sig ----------------------
    c1d = _a(nz)
    entr_rate = F(F(7.0e-5) - F(min(F(20.0), F(csum)) * F(3.0e-6)))
    if xland1 == 0:
        entr_rate = F(7.0e-5)
    radius = F(F(0.2) / entr_rate)
    frh = min(_Z1, F(F(F(F(F(3.14) * radius) * radius) / dx) / dx))
    if frh > FRH_THRESH:
        frh = FRH_THRESH
        radius = F(np.sqrt(F(F(F(frh * dx) * dx) / F(3.14))))
        entr_rate = F(F(0.2) / radius)
    sig = F(F(_Z1 - frh) * F(_Z1 - frh))
    sig_thresh = F(F(_Z1 - FRH_THRESH) * F(_Z1 - FRH_THRESH))
    R["entr_rate"] = entr_rate
    R["sig"] = sig
    R["sig_thresh"] = sig_thresh

    # ---- :480-556 ----------------------------------------------------------
    cnvwt = _a(nz)
    zuo = _a(nz)
    zdo = _a(nz)
    cupclw = _a(nz)
    z = zo.copy()
    xz = zo.copy()
    cd = np.full(nz + 1, F(1.0e-9), dtype=F)
    cd[0] = _Z0
    cdd = np.full(nz + 1, F(1.0e-9), dtype=F)
    cdd[0] = _Z0
    hcdo = _a(nz)
    qrcdo = _a(nz)
    dellaqc = _a(nz)
    edtmax = _Z1
    edtmin = F(0.1)
    depth_min = F(1000.0)
    kbmax = 1
    kdet = 1
    aa0 = _Z0
    aa1 = _Z0
    kstabm = ktf - 1
    ierr2 = 0
    ierr3 = 0
    zkbmax = F(4000.0)
    zcutdown = F(4000.0)
    z_detr = F(1000.0)
    pr_ens = np.zeros(MAXENS3 + 1, dtype=F)
    start_level = kte
    pmin_lev = 1

    # ---- :561-582 : cup_env x2, cup_env_clev x2 ----------------------------
    qes, he, hes = cup_env(z, t, q, po, nz)
    qeso, heo, heso = cup_env(zo, tn, qo, po, nz)
    (qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup) = (
        cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, nz)
    )
    (
        qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup
    ) = cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, nz)
    R.update(
        qes=qes, he=he, hes=hes, qeso=qeso, heo=heo, heso=heso,
        qes_cup=qes_cup, q_cup=q_cup, he_cup=he_cup, hes_cup=hes_cup,
        gamma_cup1=gamma_cup.copy(), t_cup=t_cup, qeso_cup=qeso_cup,
        qo_cup=qo_cup, heo_cup=heo_cup, heso_cup=heso_cup, zo_cup=zo_cup,
        po_cup=po_cup, gammao_cup=gammao_cup, tn_cup=tn_cup,
    )

    # ---- :583-615 ----------------------------------------------------------
    u_cup = _a(nz)
    v_cup = _a(nz)
    u_cup[1] = us[1]
    v_cup[1] = vs[1]
    for k in range(2, ktf + 1):
        u_cup[k] = F(_HALF * F(us[k - 1] + us[k]))
        v_cup[k] = F(_HALF * F(vs[k - 1] + vs[k]))
    R["u_cup"] = u_cup
    R["v_cup"] = v_cup
    for k in range(1, ktf + 1):
        if zo_cup[k] > F(zkbmax + z1):
            kbmax = k
            break
    for k in range(1, ktf + 1):
        if zo_cup[k] > F(z_detr + z1):
            kdet = k
            break
    R["kbmax"] = kbmax

    # ---- :621-633 : k22 ----------------------------------------------------
    k22 = _maxloc(heo_cup, 2, kbmax + 2)
    if k22 >= kbmax:
        ierr = 2
        ktop = 0
        k22 = 0
        kbcon = 0
    R["k22_0"] = k22

    # ---- :638-644 ----------------------------------------------------------
    hkb = _Z0
    hkbo = _Z0
    if ierr == 0:
        x_add = F(F(XLV * zqexec) + F(CP * ztexec))
        hkb = get_cloud_bc(he_cup, k22, x_add)
        hkbo = get_cloud_bc(heo_cup, k22, x_add)
    R["hkb0"] = hkb
    R["hkbo0"] = hkbo

    # ---- :648-653 : cup_kbcon ---------------------------------------------
    kbcon, k22, hkbo, ierr, _ = cup_kbcon(
        cap_inc=cap_max_increment, iloop=1, k22=k22, he_cup=heo_cup,
        hes_cup=heso_cup, hkb=hkbo, ierr=ierr, kbmax=kbmax, p_cup=po_cup,
        cap_max=cap_max, ztexec=ztexec, zqexec=zqexec, z_cup=z_cup,
        entr_rate=entr_rate, heo=heo, nz=nz,
    )
    R["kbcon_1"] = kbcon
    R["k22_1"] = k22
    R["ierr_1"] = ierr

    # ---- :657-659 ----------------------------------------------------------
    kstabi = cup_minimi(heso_cup, kbcon, kstabm, ierr)
    R["kstabi"] = kstabi
    R["kstabm"] = kstabm

    # ---- :660-685 ----------------------------------------------------------
    frh_kb = _Z0
    if ierr == 0:
        frh_kb = min(F(qo_cup[kbcon] / qeso_cup[kbcon]), _Z1)
        if frh_kb >= RH_THRESH and sig <= sig_thresh:
            ierr = 231
        else:
            for k in range(kbcon + 1, ktf + 1):
                if F(po[kbcon] - po[k]) > F(150.0):
                    pmin_lev = k
                    break
            start_level = k22
            x_add = F(F(XLV * zqexec) + F(CP * ztexec))
            hkb = get_cloud_bc(he_cup, k22, x_add)
    R["frh_kb"] = frh_kb
    R["pmin_lev"] = pmin_lev

    # ---- :693-726 ----------------------------------------------------------
    if kstabi < kbcon:
        kbcon = 1
        ierr = 42
    entr_rate_2d = np.full(nz + 1, entr_rate, dtype=F)
    entr_rate_2d[0] = _Z0
    if ierr == 0:
        kbcon = max(2, kbcon)
        for k in range(1, ktf + 1):
            f = min(F(qo_cup[k] / qeso_cup[k]), _Z1)
            entr_rate_2d[k] = F(entr_rate * F(F(1.3) - f))
    R["entr2d_a"] = entr_rate_2d.copy()
    R["start_level"] = start_level

    # ---- :737-738 : rates_up_pdf -------------------------------------------
    zuo, ktop, ktopdby, kbcon, ierr, pdfinfo = rates_up_pdf_deep(
        ktop=ktop, ierr=ierr, p_cup=po_cup, entr_rate_2d=entr_rate_2d,
        hkbo=hkbo, heo=heo, heso_cup=heso_cup, z_cup=zo_cup, kstabi=kstabi,
        k22=k22, kbcon=kbcon, csum=csum, nz=nz, ktf=ktf, fzu_override=fzu_up,
    )
    R["zu_pdf"] = zuo.copy()
    R["ktop_pdf"] = ktop
    R["ktopdby"] = ktopdby
    R["kbcon_2"] = kbcon
    R["ierr_2"] = ierr
    R["up_tun"] = pdfinfo.get("tunning", _Z0)
    R["up_alpha"] = pdfinfo.get("alpha", _Z0)
    R["up_beta"] = pdfinfo.get("beta", _Z0)
    R["up_fzu"] = pdfinfo.get("fzu", _Z0)
    R["up_kbadj"] = pdfinfo.get("kb_adj", 0)
    R["up_kklev"] = pdfinfo.get("kklev", 0)
    R["up_kfinal"] = pdfinfo.get("kfinalzu", 0)

    # ---- :743-763 ----------------------------------------------------------
    zu = _a(nz)
    xzu = _a(nz)
    if ierr == 0:
        if k22 > 1:
            zuo[1:k22] = _Z0
        for k in range(k22, ktop + 1):
            xzu[k] = zuo[k]
            zu[k] = zuo[k]
        zuo[ktop + 1 :] = _Z0

    # ---- :767-770 : get_lateral_massflux -----------------------------------
    upme, upmd, upmeu, upmdu, cd, entr_rate_2d = get_lateral_massflux(
        ierr=ierr, ktop=ktop, zo_cup=zo_cup, zuo=zuo, cd=cd,
        entr_rate_2d=entr_rate_2d, kbcon=kbcon, k22=k22, lambau=lambau,
        nz=nz, ktf=ktf,
    )

    # ---- :777-852 : the in-cloud updraft -----------------------------------
    uc = _a(nz)
    vc = _a(nz)
    hc = _a(nz)
    dby = _a(nz)
    hco = _a(nz)
    dbyo = _a(nz)
    dbyt = _a(nz)
    if ierr == 0:
        for k in range(1, start_level + 1):
            uc[k] = u_cup[k]
            vc[k] = v_cup[k]
        for k in range(1, start_level):
            hc[k] = he_cup[k]
            hco[k] = heo_cup[k]
        hc[start_level] = hkb
        hco[start_level] = hkbo
    ktopkeep = 0
    if ierr == 0:
        ktopkeep = ktop
        for k in range(start_level + 1, ktop + 1):
            denom = F(F(zuo[k - 1] - F(_HALF * upmd[k - 1])) + upme[k - 1])
            if denom < F(1.0e-8):
                ierr = 51
                break
            du = F(F(zu[k - 1] - F(_HALF * upmd[k - 1])) + upme[k - 1])
            duu = F(F(zu[k - 1] - F(_HALF * upmdu[k - 1])) + upmeu[k - 1])
            hc[k] = F(
                F(
                    F(F(hc[k - 1] * zu[k - 1]) - F(F(_HALF * upmd[k - 1]) * hc[k - 1]))
                    + F(upme[k - 1] * he[k - 1])
                )
                / du
            )
            uc[k] = F(
                F(
                    F(
                        F(F(uc[k - 1] * zu[k - 1]) - F(F(_HALF * upmdu[k - 1]) * uc[k - 1]))
                        + F(upmeu[k - 1] * us[k - 1])
                    )
                    - F(
                        F(F(pgcon * _HALF) * F(zu[k] + zu[k - 1]))
                        * F(u_cup[k] - u_cup[k - 1])
                    )
                )
                / duu
            )
            vc[k] = F(
                F(
                    F(
                        F(F(vc[k - 1] * zu[k - 1]) - F(F(_HALF * upmdu[k - 1]) * vc[k - 1]))
                        + F(upmeu[k - 1] * vs[k - 1])
                    )
                    - F(
                        F(F(pgcon * _HALF) * F(zu[k] + zu[k - 1]))
                        * F(v_cup[k] - v_cup[k - 1])
                    )
                )
                / duu
            )
            dby[k] = F(hc[k] - hes_cup[k])
            hco[k] = F(
                F(
                    F(F(hco[k - 1] * zuo[k - 1]) - F(F(_HALF * upmd[k - 1]) * hco[k - 1]))
                    + F(upme[k - 1] * heo[k - 1])
                )
                / denom
            )
            dbyo[k] = F(hco[k] - heso_cup[k])
            dz = F(zo_cup[k + 1] - zo_cup[k])
            dbyt[k] = F(dbyt[k - 1] + F(dbyo[k] * dz))
        for k in range(ktop - 1, kbcon - 1, -1):
            if dbyo[k] > _Z0:
                ktopkeep = k + 1
                break
        ktop = ktopkeep
    R["ktop_dbyt"] = ktop
    R["ierr_3"] = ierr

    # ---- :854-881 ----------------------------------------------------------
    if ierr == 0:
        for k in range(ktop + 1, ktf + 1):
            hc[k] = hes_cup[k]
            uc[k] = u_cup[k]
            vc[k] = v_cup[k]
            hco[k] = heso_cup[k]
            dby[k] = _Z0
            dbyo[k] = _Z0
            zu[k] = _Z0
            zuo[k] = _Z0
            cd[k] = _Z0
            entr_rate_2d[k] = _Z0
            upme[k] = _Z0
            upmd[k] = _Z0
        if ktop < kbcon + 2:
            ierr = 5
            ktop = 0
    R.update(
        entr2d_b=entr_rate_2d, cd=cd, upme=upme, upmd=upmd, upmeu=upmeu,
        upmdu=upmdu, hc=hc, uc=uc, vc=vc, hco=hco, dby=dby, dbyo=dbyo,
        dbyt=dbyt,
    )

    # ---- :882-896 : kzdown --------------------------------------------------
    kzdown = 0
    if ierr == 0:
        zktop = F(F(zo_cup[ktop] - z1) * F(0.6))
        zktop = min(F(zktop + z1), F(zcutdown + z1))
        for k in range(1, ktf + 1):
            if zo_cup[k] > zktop:
                kzdown = min(k, kstabi - 1)
                break
    R["kzdown"] = kzdown

    # ---- :900-941 : jmin ---------------------------------------------------
    jmin = cup_minimi(heso_cup, k22, kzdown, ierr)
    if ierr == 0:
        jmini = jmin
        keep_going = True
        while keep_going:
            keep_going = False
            if jmini - 1 < kdet:
                kdet = jmini - 1
            if jmini >= ktop - 1:
                jmini = ktop - 2
            ki = jmini
            hcdo[ki] = heso_cup[ki]
            dh = _Z0
            for k in range(ki - 1, 0, -1):
                hcdo[k] = heso_cup[jmini]
                dz = F(zo_cup[k + 1] - zo_cup[k])
                dh = F(dh + F(dz * F(hcdo[k] - heso_cup[k])))
                if dh > _Z0:
                    jmini -= 1
                    if jmini > 5:
                        keep_going = True
                    else:
                        ierr = 9
                        break
        jmin = jmini
        if jmini <= 5:
            ierr = 4

    # ---- :946-954 ----------------------------------------------------------
    if ierr == 0:
        if jmin - 1 < kdet:
            kdet = jmin - 1
        if F(-zo_cup[kbcon] + zo_cup[ktop]) < depth_min:
            ierr = 6
    R["kdet"] = kdet
    R["ierr_4"] = ierr

    # ---- :960-1082 : the downdraft ------------------------------------------
    zdo = _a(nz)
    cdd = _a(nz)
    ddme = _a(nz)
    ddmd = _a(nz)
    ddmeu = _a(nz)
    ddmdu = _a(nz)
    hcdo = heso_cup.copy()
    ucd = u_cup.copy()
    vcd = v_cup.copy()
    dbydo = _a(nz)
    mentrd_rate_2d = np.full(nz + 1, entr_rate, dtype=F)
    mentrd_rate_2d[0] = _Z0
    beta = max(F(0.02), F(F(0.05) - F(F(csum) * F(0.0015))))
    if xland1 == 0:
        edtmax = max(F(0.1), F(F(0.4) - F(F(csum) * F(0.015))))
    bud = _Z0
    dnpdf = {}
    if ierr == 0:
        cdd[1 : jmin + 1] = F(1.0e-9)
        cdd[jmin] = _Z0
        zdo, dnpdf = get_zu_zd_pdf_fim(
            draft="DOWN", p=po_cup, kb=kdet, kt=jmin, kpbli=kpbl, csum=csum,
            zubeg=_Z0, nz=nz, ktf=ktf, fzu_override=fzu_dn,
        )
        skip = False
        if zdo[jmin] < F(1.0e-8):
            zdo[jmin] = _Z0
            jmin -= 1
            if zdo[jmin] < F(1.0e-8):
                ierr = 876
                skip = True
        if not skip:
            kpeak = _maxloc(zdo, 1, nz)
            for ki in range(jmin, kpeak - 1, -1):
                dzo = F(zo_cup[ki + 1] - zo_cup[ki])
                ddmd[ki] = F(F(cdd[ki] * dzo) * zdo[ki + 1])
                ddme[ki] = F(F(zdo[ki] - zdo[ki + 1]) + ddmd[ki])
                if ddme[ki] < _Z0:
                    ddme[ki] = _Z0
                    ddmd[ki] = F(zdo[ki + 1] - zdo[ki])
                    if zdo[ki + 1] > _Z0:
                        cdd[ki] = F(ddmd[ki] / F(dzo * zdo[ki + 1]))
                if zdo[ki + 1] > _Z0:
                    mentrd_rate_2d[ki] = F(ddme[ki] / F(dzo * zdo[ki + 1]))
            mentrd_rate_2d[1] = _Z0
            for ki in range(kpeak - 1, 0, -1):
                dzo = F(zo_cup[ki + 1] - zo_cup[ki])
                ddme[ki] = F(F(mentrd_rate_2d[ki] * dzo) * zdo[ki + 1])
                ddmd[ki] = F(F(zdo[ki + 1] + ddme[ki]) - zdo[ki])
                if ddmd[ki] < _Z0:
                    ddmd[ki] = _Z0
                    ddme[ki] = F(zdo[ki] - zdo[ki + 1])
                    if zdo[ki + 1] > _Z0:
                        mentrd_rate_2d[ki] = F(ddme[ki] / F(dzo * zdo[ki + 1]))
                if zdo[ki + 1] > _Z0:
                    cdd[ki] = F(ddmd[ki] / F(dzo * zdo[ki + 1]))
            # the c1d quadratic, computed and then overwritten by the constant
            # c1 on the very next line (:1031-1033) -- kept because it is what
            # WRF evaluates, not because it survives.
            for k in range(kbcon + 1, ktop):
                c1d[k] = C1
            for k in range(2, jmin + 2):
                ddmeu[k - 1] = F(ddme[k - 1] + F(lambau * ddmd[k - 1]))
                ddmdu[k - 1] = F(ddmd[k - 1] + F(lambau * ddmd[k - 1]))
            dbydo[jmin] = F(hcdo[jmin] - heso_cup[jmin])
            bud = F(dbydo[jmin] * F(zo_cup[jmin + 1] - zo_cup[jmin]))
            for ki in range(jmin, 0, -1):
                dzo = F(zo_cup[ki + 1] - zo_cup[ki])
                h_entr = F(
                    _HALF * F(heo[ki] + F(_HALF * F(hco[ki] + hco[ki + 1])))
                )
                denu = F(F(zdo[ki + 1] - F(_HALF * ddmdu[ki])) + ddmeu[ki])
                deno = F(F(zdo[ki + 1] - F(_HALF * ddmd[ki])) + ddme[ki])
                ucd[ki] = F(
                    F(
                        F(
                            F(F(ucd[ki + 1] * zdo[ki + 1]) - F(F(_HALF * ddmdu[ki]) * ucd[ki + 1]))
                            + F(ddmeu[ki] * us[ki])
                        )
                        - F(F(pgcon * zdo[ki + 1]) * F(us[ki + 1] - us[ki]))
                    )
                    / denu
                )
                vcd[ki] = F(
                    F(
                        F(
                            F(F(vcd[ki + 1] * zdo[ki + 1]) - F(F(_HALF * ddmdu[ki]) * vcd[ki + 1]))
                            + F(ddmeu[ki] * vs[ki])
                        )
                        - F(F(pgcon * zdo[ki + 1]) * F(vs[ki + 1] - vs[ki]))
                    )
                    / denu
                )
                hcdo[ki] = F(
                    F(
                        F(F(hcdo[ki + 1] * zdo[ki + 1]) - F(F(_HALF * ddmd[ki]) * hcdo[ki + 1]))
                        + F(ddme[ki] * h_entr)
                    )
                    / deno
                )
                dbydo[ki] = F(hcdo[ki] - heso_cup[ki])
                bud = F(bud + F(dbydo[ki] * dzo))
    if bud > _Z0:
        ierr = 7
    # jmin, only now: :990-997 walks it down one level when the downdraft
    # profile is empty at its own originating level.
    R["jmin"] = jmin
    R.update(
        zdo=zdo, cdd=cdd, ddme=ddme, ddmd=ddmd, ddmeu=ddmeu, ddmdu=ddmdu,
        mentrd2d=mentrd_rate_2d, hcdo=hcdo, ucd=ucd, vcd=vcd, dbydo=dbydo,
        c1d=c1d, bud=bud, beta=beta, edtmax=edtmax, ierr_5=ierr,
    )
    R["dn_tun"] = dnpdf.get("tunning", _Z0)
    R["dn_alpha"] = dnpdf.get("alpha", _Z0)
    R["dn_beta"] = dnpdf.get("beta", _Z0)
    R["dn_fzu"] = dnpdf.get("fzu", _Z0)
    R["dn_kbadj"] = dnpdf.get("kb_adj", 0)

    # ---- :1086-1090 : cup_dd_moisture --------------------------------------
    qcdo, qrcdo, pwdo, pwevo, bu, ierr, _ = cup_dd_moisture(
        ierr=ierr, zd=zdo, hcd=hcdo, hes_cup=heso_cup, qes_cup=qeso_cup,
        q_cup=qo_cup, z_cup=zo_cup, dd_massentr=ddme, dd_massdetr=ddmd,
        jmin=jmin, gamma_cup=gammao_cup, q=qo, nz=nz,
    )
    R.update(qcdo=qcdo, qrcdo=qrcdo, pwdo=pwdo, pwevo=pwevo, bu=bu)

    # ---- :1102-1107 : cup_up_moisture --------------------------------------
    qco, qrco, pwo, clw_all, pwavo, psum, psumh, ierr = cup_up_moisture(
        ierr=ierr, z_cup=zo_cup, p_cup=p_cup, kbcon=kbcon, ktop=ktop,
        dby=dbyo, xland1=xland1, q=qo, gamma_cup=gammao_cup, zu=zuo,
        qes_cup=qeso_cup, k22=k22, qe_cup=qo_cup, zqexec=zqexec, ccn=ccn,
        rho=rho, c1d=c1d, t=tn_cup, up_massentr=upme, up_massdetr=upmd,
        nz=nz,
    )
    R.update(
        qco=qco, qrco=qrco, pwo=pwo, clw_all=clw_all, pwavo=pwavo,
        psum=psum, psumh=psumh,
    )

    # ---- :1109-1117 ---------------------------------------------------------
    if ierr == 0:
        dp = F(F(100.0) * F(po_cup[1] - po_cup[2]))
        for k in range(2, ktop + 1):
            cupclw[k] = qrco[k]
            cnvwt[k] = F(F(F(zuo[k] * cupclw[k]) * G) / dp)
    R["cupclw"] = cupclw
    R["cnvwt"] = cnvwt

    # ---- :1121-1136 : cup_up_aa0 x2 ----------------------------------------
    aa0 = cup_up_aa0(
        z=z, zu=zu, dby=dby, gamma_cup=gamma_cup, t_cup=t_cup, kbcon=kbcon,
        ktop=ktop, ierr=ierr, ktf=ktf,
    )
    aa1 = cup_up_aa0(
        z=zo, zu=zuo, dby=dbyo, gamma_cup=gammao_cup, t_cup=tn_cup,
        kbcon=kbcon, ktop=ktop, ierr=ierr, ktf=ktf,
    )
    if ierr == 0 and aa1 == _Z0:
        ierr = 17
    R.update(aa0=aa0, aa1=aa1, ierr_6=ierr)

    # ---- :1141-1203 : the diurnal-cycle closure ----------------------------
    aa1_bl = _Z0
    xf_dicycle = _Z0
    tau_ecmwf = _Z0
    tau_bl = _Z0
    umean = _Z0
    wmean = F(7.0)
    if ierr == 0:
        tau_ecmwf = F(F(zo_cup[ktopdby] - zo_cup[kbcon]) / wmean)
        tau_ecmwf = F(
            tau_ecmwf * F(F(1.0061) + F(F(1.23e-2) * F(dx / F(1000.0))))
        )
        if xland1 == 0:
            umean = F(
                F(2.0)
                + F(
                    np.sqrt(
                        F(
                            F(2.0)
                            * F(
                                F(
                                    F(F(us[1] * us[1]) + F(vs[1] * vs[1]))
                                    + F(us[kbcon] * us[kbcon])
                                )
                                + F(vs[kbcon] * vs[kbcon])
                            )
                        )
                    )
                )
            )
            tau_bl = F(F(zo_cup[kbcon] - z1) / umean)
        else:
            tau_bl = F(F(zo_cup[ktopdby] - zo_cup[kbcon]) / wmean)
    t_star = F(4.0)
    aa1_bl = cup_up_aa1bl(
        t=t, tn=tn, q=q, qo=qo, dtime=dtime, z=zo_cup, kbcon=kbcon,
        ierr=ierr, ktf=ktf,
    )
    if ierr == 0:
        if F(zo_cup[kbcon] - z1) > zo[min(kte, kpbl + 1)]:
            aa1_bl = _Z0
        else:
            aa1_bl = max(_Z0, F(F(aa1_bl / t_star) * tau_bl))
    axx = aa1
    R.update(tau_ecmwf=tau_ecmwf, tau_bl=tau_bl, aa1_bl=aa1_bl, umean=umean)

    # ---- :1297-1305 : cup_dd_edt -------------------------------------------
    edt, edtc = cup_dd_edt(
        ierr=ierr, us=us, vs=vs, z=zo, ktop=ktop, kbcon=kbcon, p=po,
        pwav=pwavo, pwev=pwevo, edtmax=edtmax, edtmin=edtmin, ktf=ktf,
    )
    edto = edtc if ierr == 0 else _Z0
    R.update(edt=edt, edtc1=edtc, edto=edto)

    # ---- :1369-1495 : the della fields -------------------------------------
    dellu = _a(nz)
    dellv = _a(nz)
    dellah = _a(nz)
    dellaq = _a(nz)
    dellat = _a(nz)
    if ierr == 0:
        dp = F(F(100.0) * F(po_cup[1] - po_cup[2]))
        dellu[1] = F(
            F(
                F(
                    PGCD
                    * F(
                        F(F(edto * zdo[2]) * ucd[2])
                        - F(F(edto * zdo[2]) * u_cup[2])
                    )
                )
                * G
            )
            / dp
        )
        dellv[1] = F(
            F(
                F(
                    PGCD
                    * F(
                        F(F(edto * zdo[2]) * vcd[2])
                        - F(F(edto * zdo[2]) * v_cup[2])
                    )
                )
                * G
            )
            / dp
        )
        for k in range(2, ktop + 1):
            dp = F(F(100.0) * F(po_cup[k] - po_cup[k + 1]))
            dellu[k] = F(
                F(
                    -F(
                        F(
                            F(zuo[k + 1] * F(uc[k + 1] - u_cup[k + 1]))
                            - F(zuo[k] * F(uc[k] - u_cup[k]))
                        )
                        * G
                    )
                    / dp
                )
                + F(
                    F(
                        F(
                            F(
                                F(zdo[k + 1] * F(ucd[k + 1] - u_cup[k + 1]))
                                - F(zdo[k] * F(ucd[k] - u_cup[k]))
                            )
                            * G
                        )
                        / dp
                    )
                    * F(edto * PGCD)
                )
            )
            dellv[k] = F(
                F(
                    -F(
                        F(
                            F(zuo[k + 1] * F(vc[k + 1] - v_cup[k + 1]))
                            - F(zuo[k] * F(vc[k] - v_cup[k]))
                        )
                        * G
                    )
                    / dp
                )
                + F(
                    F(
                        F(
                            F(
                                F(zdo[k + 1] * F(vcd[k + 1] - v_cup[k + 1]))
                                - F(zdo[k] * F(vcd[k] - v_cup[k]))
                            )
                            * G
                        )
                        / dp
                    )
                    * F(edto * PGCD)
                )
            )
    if ierr == 0:
        dp = F(F(100.0) * F(po_cup[1] - po_cup[2]))
        dellah[1] = F(
            F(
                F(
                    F(F(edto * zdo[2]) * hcdo[2])
                    - F(F(edto * zdo[2]) * heo_cup[2])
                )
                * G
            )
            / dp
        )
        dellaq[1] = F(
            F(
                F(
                    F(F(edto * zdo[2]) * qcdo[2])
                    - F(F(edto * zdo[2]) * qo_cup[2])
                )
                * G
            )
            / dp
        )
        g_rain = F(F(F(_HALF * F(pwo[1] + pwo[2])) * G) / dp)
        e_dn = F(F(F(F(F(-_HALF) * F(pwdo[1] + pwdo[2])) * G) / dp) * edto)
        dellaq[1] = F(F(dellaq[1] + e_dn) - g_rain)
        for k in range(2, ktop + 1):
            dp = F(F(100.0) * F(po_cup[k] - po_cup[k + 1]))
            dellah[k] = F(
                F(
                    -F(
                        F(
                            F(zuo[k + 1] * F(hco[k + 1] - heo_cup[k + 1]))
                            - F(zuo[k] * F(hco[k] - heo_cup[k]))
                        )
                        * G
                    )
                    / dp
                )
                + F(
                    F(
                        F(
                            F(
                                F(zdo[k + 1] * F(hcdo[k + 1] - heo_cup[k + 1]))
                                - F(zdo[k] * F(hcdo[k] - heo_cup[k]))
                            )
                            * G
                        )
                        / dp
                    )
                    * edto
                )
            )
            detup = upmd[k]
            dz = F(zo_cup[k] - zo_cup[k - 1])
            if k < ktop:
                dellaqc[k] = F(
                    F(F(F(F(zuo[k] * c1d[k]) * qrco[k]) * dz) / dp) * G
                )
            if k == ktop:
                dellaqc[k] = F(
                    F(
                        F(F(detup * _HALF) * F(qrco[k + 1] + qrco[k]))
                        * G
                    )
                    / dp
                )
            g_rain = F(F(F(_HALF * F(pwo[k] + pwo[k + 1])) * G) / dp)
            e_dn = F(
                F(F(F(F(-_HALF) * F(pwdo[k] + pwdo[k + 1])) * G) / dp) * edto
            )
            c_up = F(
                F(
                    dellaqc[k]
                    + F(
                        F(F(zuo[k + 1] * qrco[k + 1]) - F(zuo[k] * qrco[k])) * G
                    )
                    / dp
                )
                + g_rain
            )
            dellaq[k] = F(
                F(
                    F(
                        F(
                            -F(
                                F(
                                    F(zuo[k + 1] * F(qco[k + 1] - qo_cup[k + 1]))
                                    - F(zuo[k] * F(qco[k] - qo_cup[k]))
                                )
                                * G
                            )
                            / dp
                        )
                        + F(
                            F(
                                F(
                                    F(
                                        F(zdo[k + 1] * F(qcdo[k + 1] - qo_cup[k + 1]))
                                        - F(zdo[k] * F(qcdo[k] - qo_cup[k]))
                                    )
                                    * G
                                )
                                / dp
                            )
                            * edto
                        )
                    )
                    - c_up
                )
                + e_dn
            )
    R.update(
        dellu=dellu, dellv=dellv, dellah=dellah, dellaq=dellaq,
        dellaqc=dellaqc,
    )

    # ---- :1500-1524 : the mbdt-perturbed state -----------------------------
    mbdt = F(0.1)
    xaa0_ens = _Z0
    xhe = _a(nz)
    xq = _a(nz)
    xt = _a(nz)
    if ierr == 0:
        for k in range(1, ktf + 1):
            xhe[k] = F(F(dellah[k] * mbdt) + heo[k])
            xq[k] = max(F(1.0e-16), F(F(dellaq[k] * mbdt) + qo[k]))
            dellat[k] = F(F(_Z1 / CP) * F(dellah[k] - F(XLV * dellaq[k])))
            xt[k] = F(F(dellat[k] * mbdt) + tn[k])
            xt[k] = max(F(190.0), xt[k])
        xhe[ktf] = heo[ktf]
        xq[ktf] = qo[ktf]
        xt[ktf] = tn[ktf]
    R.update(dellat=dellat, xq=xq, xt=xt)

    # ---- :1528-1539 ---------------------------------------------------------
    # cup_env's ``he`` is written under ``itest .le. 0``, and itest is -1, so
    # the call OVERWRITES the xhe built at :1508 with 9.81z+1004T+2.5e6q on
    # the perturbed state.  Both spellings are live in WRF; this one wins.
    #
    # cup_env writes nothing when ierr /= 0 (its outputs are intent(out) and
    # stay whatever the previous column left), while cup_env_clev zeroes all
    # eight of its outputs UNCONDITIONALLY before the ierr guard (:2325-2336).
    # So the _cup arrays are defined for every column and qes/he/hes are not.
    #
    # And the third call's 13th argument is ``po_cup``, not a fresh array: the
    # perturbed-state clev OVERWRITES the environment's own ``po_cup``.  It
    # rebuilds it from the same ``po``, so the words are identical where
    # ierr == 0 -- but where ierr /= 0 the unconditional zeroing wins and
    # ``po_cup`` comes back all zeros.  A port that keeps the environment's
    # copy disagrees with WRF on 156 of the 216 columns.
    if ierr == 0:
        xqes, xhe, xhes = cup_env(xz, xt, xq, po, nz)
        (xqes_cup, xq_cup, xhe_cup, xhes_cup, _xz_cup, po_cup, gamma_cup,
         xt_cup) = cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, nz)
    else:
        xqes = _a(nz)
        xhes = _a(nz)
        xqes_cup = _a(nz)
        xq_cup = _a(nz)
        xhe_cup = _a(nz)
        xhes_cup = _a(nz)
        gamma_cup = _a(nz)
        xt_cup = _a(nz)
        po_cup = _a(nz)
    R["po_cup"] = po_cup
    R.update(
        xhe=xhe, xqes=xqes, xhes=xhes, xqes_cup=xqes_cup, xq_cup=xq_cup,
        xhe_cup=xhe_cup, xhes_cup=xhes_cup, gamma_cupx=gamma_cup,
        xt_cup=xt_cup,
    )

    # ---- :1546-1578 ---------------------------------------------------------
    xhc = _a(nz)
    xdby = _a(nz)
    xhkb = _Z0
    if ierr == 0:
        x_add = F(F(XLV * zqexec) + F(CP * ztexec))
        xhkb = get_cloud_bc(xhe_cup, k22, x_add)
        for k in range(1, start_level):
            xhc[k] = xhe_cup[k]
        xhc[start_level] = xhkb
        for k in range(start_level + 1, ktop + 1):
            xhc[k] = F(
                F(
                    F(F(xhc[k - 1] * xzu[k - 1]) - F(F(_HALF * upmd[k - 1]) * xhc[k - 1]))
                    + F(upme[k - 1] * xhe[k - 1])
                )
                / F(F(xzu[k - 1] - F(_HALF * upmd[k - 1])) + upme[k - 1])
            )
            xdby[k] = F(xhc[k] - xhes_cup[k])
        for k in range(ktop + 1, ktf + 1):
            xhc[k] = xhes_cup[k]
            xdby[k] = _Z0
    R.update(xhc=xhc, xdby=xdby, xhkb=xhkb)

    # ---- :1583-1623 ---------------------------------------------------------
    xaa0 = cup_up_aa0(
        z=xz, zu=xzu, dby=xdby, gamma_cup=gamma_cup, t_cup=xt_cup,
        kbcon=kbcon, ktop=ktop, ierr=ierr, ktf=ktf,
    )
    if ierr == 0:
        xaa0_ens = xaa0
        for k in range(1, ktop + 1):
            for n in range(1, MAXENS3 + 1):
                # (pr_ens + pwo) + edto*pwdo -- the accumulator absorbs pwo
                # first and the downdraft term second, which is not the same
                # float32 as adding the two terms together beforehand.
                pr_ens[n] = F(
                    F(pr_ens[n] + pwo[k]) + F(edto * pwdo[k])
                )
        if pr_ens[7] < F(1.0e-6):
            ierr = 18
            pr_ens[:] = _Z0
        for n in range(1, MAXENS3 + 1):
            if pr_ens[n] < F(1.0e-5):
                pr_ens[n] = _Z0
    R.update(xaa0=xaa0, pr7=pr_ens[7], ierr_7=ierr)

    # ---- :1633-1654 : the ierr2 / ierr3 cap probes -------------------------
    ierr2 = ierr
    ierr3 = ierr
    k22x = cup_maximi(heo_cup, 2, kbmax, ierr)
    kbconx, k22x2, hkbo, ierr2, _ = cup_kbcon(
        cap_inc=cap_max_increment, iloop=2, k22=k22x, he_cup=heo_cup,
        hes_cup=heso_cup, hkb=hkbo, ierr=ierr2, kbmax=kbmax, p_cup=po_cup,
        cap_max=cap_max, ztexec=ztexec, zqexec=zqexec, z_cup=z_cup,
        entr_rate=entr_rate, heo=heo, nz=nz,
    )
    kbconx, k22x, hkbo, ierr3, _ = cup_kbcon(
        cap_inc=cap_max_increment, iloop=3, k22=k22x2, he_cup=heo_cup,
        hes_cup=heso_cup, hkb=hkbo, ierr=ierr3, kbmax=kbmax, p_cup=po_cup,
        cap_max=cap_max, ztexec=ztexec, zqexec=zqexec, z_cup=z_cup,
        entr_rate=entr_rate, heo=heo, nz=nz,
    )
    R.update(k22x=k22x, kbconx=kbconx, ierr2=ierr2, ierr3=ierr3)

    # ---- :1659-1666 : mconv on the cloud grid, with the DEEP g -------------
    mconv2 = _Z0
    if ierr == 0:
        for k in range(1, ktop + 1):
            dq = F(qo_cup[k + 1] - qo_cup[k])
            mconv2 = F(mconv2 + F(F(omeg[k] * dq) / G))
    R["mconv2"] = mconv2

    # ---- :1667-1674 ---------------------------------------------------------
    xf_ens, forcing, xf_dicycle, closure_n = cup_forcing_ens_3d(
        closure_n=closure_n, xland1=xland1, aa0=aa0, aa1=aa1, xaa0=xaa0_ens,
        mbdt=mbdt, dtime=dtime, ierr=ierr, ierr2=ierr2, ierr3=ierr3, axx=axx,
        mconv=mconv2, p_cup=po_cup, ktop=ktop, omeg=omeg, zd=zdo, k22=k22,
        zu=zuo, pr_ens=pr_ens, edt=edto, kbcon=kbcon, ichoice=ichoice,
        dicycle=dicycle, tau_ecmwf=tau_ecmwf, aa1_bl=aa1_bl, nz=nz,
    )
    R.update(xf_ens_pre=xf_ens.copy(), forcing=forcing, closure_n=closure_n)
    R["xf_dicycle"] = xf_dicycle

    # ---- :1715-1723 ---------------------------------------------------------
    outt, outq, outqc, pre, xmb, xf_ens, ierr = cup_output_ens_3d(
        xf_ens=xf_ens, ierr=ierr, dellat=dellat, dellaq=dellaq,
        dellaqc=dellaqc, zu=zuo, pw=pwo, ktop=ktop, edt=edto, pwd=pwdo,
        p_cup=po_cup, pr_ens=pr_ens, sig=sig, closure_n=closure_n,
        xmbs_in=xmbs_in, dicycle=dicycle, xf_dicycle=xf_dicycle, nz=nz,
    )
    R.update(
        outt_o=outt.copy(), outq_o=outq.copy(), outqc_o=outqc.copy(),
        xf_ens=xf_ens, xmb=xmb,
    )

    # ---- :1724-1743 ---------------------------------------------------------
    outu = _a(nz)
    outv = _a(nz)
    xmb_out = _Z0
    if ierr == 0 and pre > _Z0:
        pre = max(pre, _Z0)
        xmb_out = xmb
        for k in range(1, ktop + 1):
            outu[k] = F(dellu[k] * xmb)
            outv[k] = F(dellv[k] * xmb)
    elif ierr != 0 or pre == _Z0:
        ktop = 0
        outt[:] = _Z0
        outq[:] = _Z0
        outqc[:] = _Z0
        outu[:] = _Z0
        outv[:] = _Z0

    # ---- :1803-1821 : dissipative heating ----------------------------------
    if ierr == 0:
        dts = _Z0
        fpi = _Z0
        for k in range(1, ktop + 1):
            dp = F(F(po_cup[k] - po_cup[k + 1]) * F(100.0))
            dts = F(
                dts
                - F(
                    F(F(F(outu[k] * us[k]) + F(outv[k] * vs[k])) * dp) / G
                )
            )
            fpi = F(
                fpi
                + F(
                    np.sqrt(F(F(outu[k] * outu[k]) + F(outv[k] * outv[k]))) * dp
                )
            )
        if fpi > _Z0:
            for k in range(1, ktop + 1):
                fp = F(
                    np.sqrt(F(F(outu[k] * outu[k]) + F(outv[k] * outv[k]))) / fpi
                )
                outt[k] = F(outt[k] + F(F(F(fp * dts) * G) / CP))
    R.update(
        outt=outt, outq=outq, outqc=outqc, outu=outu, outv=outv, pre=pre,
        xmb_out=xmb_out, ktop=ktop, kbcon=kbcon, k22=k22, ierr=ierr,
        zuo=zuo, zu=zu,
    )

    # ---- the get_inversion_layers capture, for the shallow port ------------
    dtempdz, k_inv, clamped = get_inversion_layers(
        ierr=R["ierr_6"], p_cup=p_cup, t_cup=t_cup, z_cup=z_cup,
        kstart=kbcon, kend=kstabi, nz=nz, ktf=ktf,
    )
    R.update(dtempdz=dtempdz, k_inv=k_inv, kinv_clamped=int(clamped))
    return R
