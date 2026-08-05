"""``CUP_gf_sh``: module_cu_gf_sh.F:58-936, one column, float32.

The shallow arm of Grell-Freitas, and not a reduced copy of the deep one.  It
shares seven procedures with ``module_cu_gf_deep`` -- ``cup_env``,
``cup_env_clev``, ``get_cloud_bc``, ``cup_kbcon``, ``cup_minimi``,
``cup_up_aa0``, ``get_lateral_massflux`` -- plus ``rates_up_pdf`` and
``get_inversion_layers``, and everything else about it is different:

* no downdraft at all, so no ``cup_dd_*``, no ``edt``, no ``pwev``;
* no closure ensemble -- three closures, averaged, against the deep arm's
  sixteen;
* **no scale awareness.**  ``sig = (1-frh)^2`` lives only in
  ``cup_output_ens_3d`` on the deep/mid path.  ``CUP_gf_sh`` has no ``dx``
  argument, and none of the fourteen fields GFDRV hands it depends on grid
  spacing, so its answer cannot move with dx.  The oracle proves that rather
  than assuming it (``gf-shallow-consistency.csv``, column
  ``ndiff_words_vs_dx1``), which is why its capture is keyed by case alone;
* ``mbdt = .5`` against the deep arm's ``.1``, and the perturbed state is
  built from ``dellah``/``dellaq`` directly rather than from a mass-flux
  ensemble;
* ``entr_rate = 9.e-5`` flat, against the deep arm's csum- and dx-dependent
  ``7.e-5``;
* ``ktop`` comes from ``get_inversion_layers``' 800 hPa slot, or from 200 mb
  above ``kbcon`` -- never from a buoyancy integral;
* ``cup_kbcon`` runs at ``iloop = 5``, its own branch: ``kbcon`` starts at
  ``k22`` rather than ``k22+1``, two levels of negative buoyancy are allowed
  instead of one, ``plus`` is a flat 150 mb, and a ``cap_max`` above 200 mb
  re-references the depth to ``cap_max`` itself;
* the updraft profile is ``get_zu_zd_pdf_fim``'s ``SH2`` branch, ``beta =
  2.5`` against ``UP``'s 1.3, and it does NOT zero ``zu`` below ``kb_adj``.

Two traps a reading of the Fortran gets wrong, both found by the fixture.

``k22`` is off by one, on purpose-as-shipped.  ``k22(i) = maxloc(HEO_CUP(i,
2:kbmax(i)), 1)`` (:373) is a MAXLOC over an array SECTION, so it returns the
position WITHIN ``2:kbmax``, 1-based -- and WRF uses that position as an
absolute level index without adding the section's offset.  WRF's ``k22`` is
therefore one level below the argmax of ``heo_cup``.  A port that "fixes" it
to ``maxloc + 1`` disagrees with WRF on every column where the two differ.

``po_cup`` and ``gamma_cup`` come back ZEROED on rejected columns.  The
perturbed-state ``cup_env_clev`` (:760) is handed ``po_cup`` and
``gamma_cup`` themselves, not fresh arrays, and the routine zeroes its
outputs BEFORE its ``ierr`` guard (module_cu_gf_deep.F:2324-2334).  Nothing
downstream reads either on a rejected column, so it is invisible to WRF and
visible to a bitwise capture.

The one non-bitwise value is the same one the deep arm has: ``fzu``, through
``tgammaf``.  See :mod:`gpuwm.verify.gf_deep_ref`.

Stage names in the returned dict are the oracle's own column names in
``gpuwm/data/gf/oracle/gf-shallow-levels.csv`` and
``gf-shallow-surface.csv``.
"""

from __future__ import annotations

import numpy as np

from gpuwm.verify.gf_deep_ref import (
    _HALF,
    _Z0,
    _Z1,
    _a,
    _maxloc,
    F,
    _TINY32,
    cup_env,
    cup_env_clev,
    cup_kbcon,
    cup_minimi,
    cup_up_aa0,
    get_cloud_bc,
    get_inversion_layers,
    get_lateral_massflux,
    _powf,
    rates_up_pdf_shallow,
)

__all__ = ["cup_gf_sh_column", "SH_G", "SH_CP", "SH_XLV", "SH_C0", "SH_C1"]

# module_cu_gf_sh.F:48-54.  Declared separately from the deep module's set and
# numerically equal to it -- which is NOT true of the driver's GFS set, where
# three of four differ.  They are spelled out here rather than imported so
# that a future WRF release changing one of them changes one place.
SH_C1 = F(0.0)          # c1_shal
SH_G = F(9.81)
SH_CP = F(1004.0)
SH_XLV = F(2.5e6)
SH_RV = F(461.0)
SH_C0 = F(0.001)        # c0_shal
SH_FLUXTUNE = F(1.5)


def cup_gf_sh_column(
    *, zo, t, q, z1, tn, qo, po, psur, dhdt, kpbl, rho, hfx, qfx, xland,
    dtime, ichoice=0, fzu_override=None,
):
    """Run one column through the shallow cloud model.

    Arguments are named for ``CUP_gf_sh``'s own dummies, so ``t``/``q`` are
    GFDRV's ``t2d``/``q2d`` and ``tn``/``qo`` are its ``tshall``/``qshall``
    (module_cu_gf_wrfdrv.F:508) -- the boundary-layer-forced state, not the
    fully forced one the deep arm gets.
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
    dhdt = lift(dhdt)
    rho = lift(rho)
    z1 = F(z1)
    psur = F(psur)
    hfx = F(hfx)
    qfx = F(qfx)
    xland = F(xland)
    dtime = F(dtime)
    kpbl = int(kpbl)
    R = {}

    ierr = 0
    kbcon = 0
    ktop = 0
    k22 = 0

    # ---- :241-256 ----------------------------------------------------------
    start_level = 0
    flux_tun = SH_FLUXTUNE
    ktopx = 0
    xland1 = int(xland + F(0.001))
    if xland > F(1.5) or xland < F(0.5):
        xland1 = 0
    pre = _Z0
    xmb_out = _Z0
    cap_max_increment = F(25.0)
    ierrc = " "
    entr_rate = F(9.0e-5)

    # ---- :265-277 ----------------------------------------------------------
    up_massentro = _a(nz)
    up_massdetro = _a(nz)
    z = zo.copy()
    xz = zo.copy()
    qrco = _a(nz)
    pwo = _a(nz)
    cd = np.zeros(nz + 1, dtype=F)
    cd[1:] = F(_Z1 * entr_rate)
    dellaqc = _a(nz)
    cupclw = _a(nz)
    cnvwt = _a(nz)
    zuo = _a(nz)
    zu = _a(nz)
    xzu = _a(nz)
    outt = _a(nz)
    outq = _a(nz)
    outqc = _a(nz)

    # ---- :287-298 ----------------------------------------------------------
    cap_maxs = F(125.0)
    kbmax = 1
    aa0 = _Z0
    aa1 = _Z0
    cap_max = cap_maxs
    ztexec = _Z0
    zqexec = _Z0
    zws = _Z0

    # ---- :299-319 : the convective-scale velocity --------------------------
    buo_flux = F(
        F(F(hfx / SH_CP) + F(F(F(F(0.608) * t[1]) * qfx) / SH_XLV)) / rho[1]
    )
    zws = max(
        _Z0, F(F(F(F(F(flux_tun * F(0.41)) * buo_flux) * zo[2]) * SH_G) / t[1])
    )
    if zws > _TINY32:
        zws = F(F(1.2) * _powf(zws, F(0.3333)))
        ztexec = max(F(F(flux_tun * hfx) / F(F(rho[1] * zws) * SH_CP)), _Z0)
        zqexec = max(F(F(F(flux_tun * qfx) / SH_XLV) / F(rho[1] * zws)), _Z0)
    zws = max(
        _Z0,
        F(F(F(F(F(flux_tun * F(0.41)) * buo_flux) * zo[kpbl]) * SH_G) / t[kpbl]),
    )
    zws = F(F(1.2) * _powf(zws, F(0.3333)))
    zws = F(zws * rho[kpbl])
    R.update(buo_flux=buo_flux, zws=zws, ztexec=ztexec, zqexec=zqexec,
             xland1=xland1, entr_rate=entr_rate)

    zkbmax = F(3000.0)

    # ---- :328-349 : the two environments -----------------------------------
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
        z_cup=z_cup, p_cup=p_cup, gamma_cup0=gamma_cup.copy(), t_cup=t_cup,
        qeso_cup=qeso_cup, qo_cup=qo_cup, heo_cup=heo_cup, heso_cup=heso_cup,
        zo_cup=zo_cup, po_cup0=po_cup.copy(), gammao_cup=gammao_cup,
        tn_cup=tn_cup,
    )

    # ---- :350-363 : kbmax ---------------------------------------------------
    if ierr == 0:
        for k in range(1, ktf + 1):
            if zo_cup[k] > F(zkbmax + z1):
                kbmax = k
                break
        kbmax = min(kbmax, ktf // 2)
    R["kbmax"] = kbmax

    # ---- :370-383 : cap_max and k22 ----------------------------------------
    # The cap collapse is OUTSIDE the ierr guard, so it happens on rejected
    # columns too.  And see this module's docstring for the MAXLOC offset.
    if kpbl > 3:
        cap_max = po_cup[kpbl]
    if ierr == 0:
        # MAXLOC over heo_cup(2:kbmax) is the position INSIDE the section, so
        # the absolute level is that position + 1 -- and WRF does not add it.
        k22 = _maxloc(heo_cup, 2, kbmax) - 1 if kbmax >= 2 else 0
        k22 = max(2, k22)
        if k22 > kbmax:
            ierr = 2
            ierrc = "could not find k22"
            ktop = 0
            k22 = 0
            kbcon = 0
    R["cap_max"] = cap_max
    R["k22_0"] = k22

    # ---- :387-393 -----------------------------------------------------------
    hkb = _Z0
    hkbo = _Z0
    if ierr == 0:
        x_add = F(F(SH_XLV * zqexec) + F(SH_CP * ztexec))
        hkb = get_cloud_bc(he_cup, k22, x_add)
        hkbo = get_cloud_bc(heo_cup, k22, x_add)
    R["hkb0"] = hkb
    R["hkbo0"] = hkbo

    # ---- :396-400 -----------------------------------------------------------
    dbyo = _a(nz)

    # ---- :402-407 : cup_kbcon at iloop = 5 ---------------------------------
    kbcon, k22, hkbo, ierr, kbc_msg = cup_kbcon(
        cap_inc=cap_max_increment, iloop=5, k22=k22, he_cup=heo_cup,
        hes_cup=heso_cup, hkb=hkbo, ierr=ierr, kbmax=kbmax, p_cup=po_cup,
        cap_max=cap_max, ztexec=ztexec, zqexec=zqexec, z_cup=z_cup,
        entr_rate=entr_rate, heo=heo, nz=nz,
    )
    if kbc_msg is not None:
        ierrc = kbc_msg
    R.update(kbcon_1=kbcon, k22_1=k22, hkbo_1=hkbo, ierr_1=ierr)

    # ---- :409-414 -----------------------------------------------------------
    kstabi = cup_minimi(heso_cup, kbcon, kbmax, ierr)
    dtempdz, k_inv, kinv_clamped = get_inversion_layers(
        ierr=ierr, p_cup=p_cup, t_cup=t_cup, z_cup=z_cup, kstart=kbcon,
        kend=kstabi, nz=nz, ktf=ktf,
    )
    R.update(kstabi=kstabi, dtempdz=dtempdz, k_inv=k_inv,
             kstabi_oob=int(kinv_clamped))

    # ---- :417-449 : the entrainment profile and the first ktop -------------
    entr_rate_2d = np.zeros(nz + 1, dtype=F)
    entr_rate_2d[1:] = entr_rate
    if ierr == 0:
        start_level = k22
        x_add = F(F(SH_XLV * zqexec) + F(SH_CP * ztexec))
        hkb = get_cloud_bc(he_cup, k22, x_add)
        if kbcon > ktf - 4:
            ierr = 231
        for k in range(1, ktf + 1):
            frh = F(F(2.0) * min(F(qo_cup[k] / qeso_cup[k]), _Z1))
            entr_rate_2d[k] = F(entr_rate * F(F(2.3) - frh))
            cd[k] = entr_rate_2d[k]
        ktop = 1
        if k_inv[1] > 0 and F(po_cup[kbcon] - po_cup[k_inv[1]]) < F(200.0):
            ktop = k_inv[1]
        else:
            for k in range(kbcon + 1, ktf + 1):
                if F(po_cup[kbcon] - po_cup[k]) > F(200.0):
                    ktop = k
                    break
    R.update(start_level=start_level, hkb_2=hkb, ierr_231=ierr, ktop_0=ktop,
             entr2d_a=entr_rate_2d.copy(), cd_a=cd.copy())

    # ---- :451-452 : the normalised mass-flux profile -----------------------
    zuo, ktop, kbcon, ierr, pdf = rates_up_pdf_shallow(
        ktop=ktop, ierr=ierr, p_cup=po_cup, entr_rate_2d=entr_rate_2d,
        z_cup=zo_cup, kpbl=kpbl, k22=k22, kbcon=kbcon, nz=nz, ktf=ktf,
        fzu_override=fzu_override,
    )
    if ierr == 0:
        ktopx = ktop
    R.update(zu_pdf=zuo.copy(), ktop_pdf=ktop, kbcon_2=kbcon, ierr_2=ierr,
             sh_tun=pdf["tunning"], sh_alpha=pdf["alpha"], sh_beta=pdf["beta"],
             sh_fzu=pdf["fzu"], sh_kbadj=pdf["kb_adj"],
             sh_kfinal=pdf["kfinalzu"])

    # ---- :453-486 -----------------------------------------------------------
    if ierr == 0:
        if k22 > 1:
            zuo[1:k22] = _Z0
            zu[1:k22] = _Z0
            xzu[1:k22] = _Z0
        for k in range(_maxloc(zuo, 1, nz), ktop + 1):
            if zuo[k] < F(1.0e-6):
                ktop = k - 1
                break
        for k in range(k22, ktop + 1):
            xzu[k] = zuo[k]
            zu[k] = zuo[k]
        zuo[ktop + 1 : ktf + 1] = _Z0
        zu[ktop + 1 : ktf + 1] = _Z0
        xzu[ktop + 1 : ktf + 1] = _Z0
        k22 = max(2, k22)
    R.update(zuo_b=zuo.copy(), ktop_3=ktop, k22_3=k22)

    # ---- :490-493 : lateral mass flux, without the momentum limb -----------
    up_massentro, up_massdetro, _, _, cd, entr_rate_2d = get_lateral_massflux(
        ierr=ierr, ktop=ktop, zo_cup=zo_cup, zuo=zuo, cd=cd,
        entr_rate_2d=entr_rate_2d, kbcon=kbcon, k22=k22, nz=nz, ktf=ktf,
    )
    # :4312-4315 -- up_massentr/up_massdetr are a straight copy, total here.
    up_massentr = up_massentro
    up_massdetr = up_massdetro

    # ---- :495-514 -----------------------------------------------------------
    hc = _a(nz)
    qco = _a(nz)
    qrco = _a(nz)
    dby = _a(nz)
    hco = _a(nz)
    dbyo = _a(nz)
    dbyt = _a(nz)
    qaver = _Z0
    ki = 0
    if ierr == 0:
        hc[1:start_level] = he_cup[1:start_level]
        hco[1:start_level] = heo_cup[1:start_level]
        hc[start_level] = hkb
        hco[start_level] = hkbo

    # ---- :517-611 : the in-cloud updraft ------------------------------------
    if ierr == 0:
        for k in range(start_level + 1, ktop + 1):
            hc[k] = F(
                F(
                    F(hc[k - 1] * zu[k - 1])
                    - F(F(_HALF * up_massdetr[k - 1]) * hc[k - 1])
                    + F(up_massentr[k - 1] * he[k - 1])
                )
                / F(
                    F(zu[k - 1] - F(_HALF * up_massdetr[k - 1]))
                    + up_massentr[k - 1]
                )
            )
            dby[k] = max(_Z0, F(hc[k] - hes_cup[k]))
            hco[k] = F(
                F(
                    F(hco[k - 1] * zuo[k - 1])
                    - F(F(_HALF * up_massdetro[k - 1]) * hco[k - 1])
                    + F(up_massentro[k - 1] * heo[k - 1])
                )
                / F(
                    F(zuo[k - 1] - F(_HALF * up_massdetro[k - 1]))
                    + up_massentro[k - 1]
                )
            )
            dbyo[k] = F(hco[k] - heso_cup[k])
            dz = F(zo_cup[k + 1] - zo_cup[k])
            dbyt[k] = F(dbyt[k - 1] + F(dbyo[k] * dz))
        ki = _maxloc(dbyt, 1, nz)
        if ktop > ki + 1:
            ktop = ki + 1
            zuo[ktop + 1 : ktf + 1] = _Z0
            zu[ktop + 1 : ktf + 1] = _Z0
            cd[ktop + 1 : ktf + 1] = _Z0
            up_massdetro[ktop] = zuo[ktop]
            up_massentro[ktop : ktf + 1] = _Z0
            up_massdetro[ktop + 1 : ktf + 1] = _Z0
            entr_rate_2d[ktop + 1 : ktf + 1] = _Z0
        if ktop < kbcon + 1:
            ierr = 5
            ierrc = "ktop is less than kbcon+1"
        elif ktop > ktf - 2:
            ierr = 5
            ierrc = "ktop is larger than ktf-2"
    R.update(ki_dbyt=ki)

    if ierr == 0:
        qaver = get_cloud_bc(qo_cup, k22)
        qaver = F(qaver + zqexec)
        qco[1:start_level] = qo_cup[1:start_level]
        qco[start_level] = qaver
        for k in range(start_level + 1, ktop + 1):
            trash = F(
                qeso_cup[k]
                + F(
                    F(F(_Z1 / SH_XLV) * F(gammao_cup[k] / F(_Z1 + gammao_cup[k])))
                    * dbyo[k]
                )
            )
            trash2 = qco[k - 1]
            qco[k] = F(
                F(
                    F(trash2 * F(zuo[k - 1] - F(F(0.5) * up_massdetr[k - 1])))
                    + F(up_massentr[k - 1] * qo[k - 1])
                )
                / F(
                    F(zuo[k - 1] - F(_HALF * up_massdetr[k - 1]))
                    + up_massentr[k - 1]
                )
            )
            if qco[k] >= trash:
                dz = F(z_cup[k] - z_cup[k - 1])
                qrco[k] = F(
                    F(qco[k] - trash) / F(_Z1 + F(F(SH_C0 + SH_C1) * dz))
                )
                pwo[k] = F(F(F(SH_C0 * dz) * qrco[k]) * zuo[k])
                qco[k] = F(trash + qrco[k])
            else:
                qrco[k] = _Z0
            cupclw[k] = qrco[k]
        R["qco_a"] = qco.copy()
        for k in range(k22 + 1, ktop + 1):
            dp = F(F(100.0) * F(po_cup[k] - po_cup[k + 1]))
            cnvwt[k] = F(F(F(zuo[k] * cupclw[k]) * SH_G) / dp)
            qco[k] = F(qco[k] - qrco[k])
        for k in range(ktop + 1, ktf):
            hc[k] = hes_cup[k]
            hco[k] = heso_cup[k]
            qco[k] = qeso_cup[k]
            qrco[k] = _Z0
            dby[k] = _Z0
            dbyo[k] = _Z0
            zu[k] = _Z0
            xzu[k] = _Z0
            zuo[k] = _Z0
    else:
        R["qco_a"] = qco.copy()
    R.update(ktop_4=ktop, ierr_4=ierr, qaver=qaver, hc=hc, hco=hco, dby=dby,
             dbyo=dbyo, dbyt=dbyt, qrco=qrco, pwo=pwo, cupclw=cupclw, qco=qco,
             cnvwt=cnvwt, upme=up_massentro, upmd=up_massdetro, cd_b=cd,
             entr2d_b=entr_rate_2d)

    # ---- :615-630 : the cloud work functions -------------------------------
    aa0 = cup_up_aa0(
        z=z, zu=zu, dby=dby, gamma_cup=gamma_cup, t_cup=t_cup, kbcon=kbcon,
        ktop=ktop, ierr=ierr, ktf=ktf,
    )
    aa1 = cup_up_aa0(
        z=zo, zu=zuo, dby=dbyo, gamma_cup=gammao_cup, t_cup=tn_cup,
        kbcon=kbcon, ktop=ktop, ierr=ierr, ktf=ktf,
    )
    if ierr == 0 and aa1 <= _Z0:
        ierr = 17
        ierrc = "cloud work function zero"
    R.update(aa0=aa0, aa1=aa1, ierr_5=ierr)

    # ---- :639-720 : the dellas ----------------------------------------------
    dellah = _a(nz)
    dellaq = _a(nz)
    if ierr == 0:
        for k in range(k22, ktop + 1):
            entup = up_massentro[k]
            detup = up_massdetro[k]
            dp = F(F(100.0) * F(po_cup[k] - po_cup[k + 1]))
            dellah[k] = F(
                F(
                    -F(
                        F(zuo[k + 1] * F(hco[k + 1] - heo_cup[k + 1]))
                        - F(zuo[k] * F(hco[k] - heo_cup[k]))
                    )
                    * SH_G
                )
                / dp
            )
            dz = F(zo_cup[k + 1] - zo_cup[k])
            if k < ktop:
                dellaqc[k] = F(
                    F(F(F(F(zuo[k] * SH_C1) * qrco[k]) * dz) / dp) * SH_G
                )
            else:
                dellaqc[k] = F(F(F(detup * qrco[k]) * SH_G) / dp)
            c_up = F(
                dellaqc[k]
                + F(
                    F(
                        F(zuo[k + 1] * qrco[k + 1]) - F(zuo[k] * qrco[k])
                    )
                    * SH_G
                )
                / dp
            )
            dellaq[k] = F(
                F(
                    F(
                        F(
                            -F(
                                F(zuo[k + 1] * F(qco[k + 1] - qo_cup[k + 1]))
                                - F(zuo[k] * F(qco[k] - qo_cup[k]))
                            )
                            * SH_G
                        )
                        / dp
                    )
                    - c_up
                )
                - F(F(F(_HALF * F(pwo[k] + pwo[k + 1])) * SH_G) / dp)
            )
    R.update(dellah=dellah, dellaq=dellaq, dellaqc=dellaqc)

    # ---- :725-746 : the mbdt-perturbed state --------------------------------
    mbdt = F(0.5)
    dellat = _a(nz)
    xhe = _a(nz)
    xq = _a(nz)
    xt = _a(nz)
    if ierr == 0:
        for k in range(1, ktf + 1):
            xhe[k] = F(F(dellah[k] * mbdt) + heo[k])
            xq[k] = max(
                F(1.0e-16), F(F(F(dellaq[k] + dellaqc[k]) * mbdt) + qo[k])
            )
            dellat[k] = F(
                F(_Z1 / SH_CP) * F(dellah[k] - F(SH_XLV * dellaq[k]))
            )
            xt[k] = F(
                F(F(F(F(-dellaqc[k]) * SH_XLV) / SH_CP) + dellat[k]) * mbdt
                + tn[k]
            )
            xt[k] = max(F(190.0), xt[k])
        xhe[ktf] = heo[ktf]
        xq[ktf] = qo[ktf]
        xt[ktf] = tn[ktf]
    R.update(dellat=dellat, xhe=xhe, xq=xq, xt=xt)

    # ---- :749-810 : the perturbed static control ----------------------------
    # cup_env leaves xqes/xhes untouched on a rejected column (it has no
    # zeroing pass), while cup_env_clev zeroes all eight of its outputs before
    # the ierr guard -- and its 13th and 15th actuals here are po_cup and
    # gamma_cup themselves.  Both effects are reproduced.
    #
    # And cup_env's third output is `he`, which here is `xhe` -- so the call
    # OVERWRITES the perturbed moist static energy built at :731 rather than
    # leaving it, because the guard is `itest .le. 0` and itest is -1.  The
    # deep arm has the identical trap at CUP_gf:1508.
    xqes = _a(nz)
    xhes = _a(nz)
    xqes_cup = _a(nz)
    xq_cup = _a(nz)
    xhe_cup = _a(nz)
    xhes_cup = _a(nz)
    xz_cup = _a(nz)
    xt_cup = _a(nz)
    if ierr == 0:
        xqes, xhe, xhes = cup_env(xz, xt, xq, po, nz)
        (
            xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup,
            xt_cup,
        ) = cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, nz)
    else:
        po_cup = _a(nz)
        gamma_cup = _a(nz)
    R.update(xhe=xhe, xqes=xqes, xhes=xhes, xqes_cup=xqes_cup, xq_cup=xq_cup,
             xhe_cup=xhe_cup, xhes_cup=xhes_cup, gamma_cupx=gamma_cup,
             xt_cup=xt_cup, po_cupx=po_cup)

    xhc = _a(nz)
    xdby = _a(nz)
    xhkb = _Z0
    if ierr == 0:
        x_add = F(F(SH_XLV * zqexec) + F(SH_CP * ztexec))
        xhkb = get_cloud_bc(xhe_cup, k22, x_add)
        xhc[1:start_level] = xhe_cup[1:start_level]
        xhc[start_level] = xhkb
        xzu[1 : ktf + 1] = zuo[1 : ktf + 1]
        for k in range(start_level + 1, ktop + 1):
            xhc[k] = F(
                F(
                    F(xhc[k - 1] * xzu[k - 1])
                    - F(F(_HALF * up_massdetro[k - 1]) * xhc[k - 1])
                    + F(up_massentro[k - 1] * xhe[k - 1])
                )
                / F(
                    F(xzu[k - 1] - F(_HALF * up_massdetro[k - 1]))
                    + up_massentro[k - 1]
                )
            )
            xdby[k] = F(xhc[k] - xhes_cup[k])
        for k in range(ktop + 1, ktf + 1):
            xhc[k] = xhes_cup[k]
            xdby[k] = _Z0
            xzu[k] = _Z0
    xaa0 = cup_up_aa0(
        z=xz, zu=xzu, dby=xdby, gamma_cup=gamma_cup, t_cup=xt_cup,
        kbcon=kbcon, ktop=ktop, ierr=ierr, ktf=ktf,
    )
    R.update(xhkb=xhkb, xaa0=xaa0, xhc=xhc, xdby=xdby, xzu=xzu)

    # ---- :817-874 : the shallow closure and the tendencies ------------------
    xmb = _Z0
    xmbmax = _Z0
    xkshal = _Z0
    blqe = _Z0
    trash = _Z0
    xff = [_Z0, _Z0, _Z0]
    if ierr == 0:
        xmbmax = F(1.0)
        xkshal = F(F(xaa0 - aa1) / mbdt)
        if xkshal <= _Z0 and xkshal > F(F(-0.01) * mbdt):
            xkshal = F(F(-0.01) * mbdt)
        if xkshal > _Z0 and xkshal < F(1.0e-2):
            xkshal = F(1.0e-2)
        xff[0] = max(_Z0, F(F(-F(aa1 - aa0)) / F(xkshal * dtime)))
        xff[1] = F(F(0.03) * zws)
        for k in range(1, kpbl + 1):
            blqe = F(
                blqe
                + F(F(F(F(100.0) * dhdt[k]) * F(po_cup[k] - po_cup[k + 1])) / SH_G)
            )
        trash = max(F(hc[kbcon] - he_cup[kbcon]), F(1.0e1))
        xff[2] = max(_Z0, F(blqe / trash))
        xff[2] = min(xmbmax, xff[2])
        xmb = F(F(F(xff[0] + xff[1]) + xff[2]) / F(3.0))
        xmb = min(xmbmax, xmb)
        if ichoice > 0:
            xmb = min(xmbmax, xff[ichoice - 1])
        if xmb <= _Z0:
            ierr = 21
            ierrc = "21"
    if ierr != 0:
        k22 = 0
        kbcon = 0
        ktop = 0
        xmb = _Z0
        outt[:] = _Z0
        outq[:] = _Z0
        outqc[:] = _Z0
    else:
        xmb_out = xmb
        pre = _Z0
        for k in range(2, ktop + 1):
            outt[k] = F(dellat[k] * xmb)
            outq[k] = F(dellaq[k] * xmb)
            outqc[k] = F(dellaqc[k] * xmb)
            pre = F(pre + F(pwo[k] * xmb))
    R.update(xkshal=xkshal, xff1=xff[0], xff2=xff[1], xff3=xff[2], blqe=blqe,
             trash_kb=trash, xmbmax=xmbmax, xmb=xmb, ierr_6=ierr, k22=k22,
             kbcon=kbcon, ktop=ktop, ierr=ierr, xmb_out=xmb_out, pre=pre,
             outt=outt, outq=outq, outqc=outqc, zuo=zuo, ierrc=ierrc,
             ktopx=ktopx)
    return R
