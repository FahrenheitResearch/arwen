"""``GFDRV`` end to end: module_cu_gf_wrfdrv.F:1-844, one column, float32.

:mod:`gpuwm.verify.gf_ref` is the driver's *input* half -- the column
preparation at :383-492, bitwise since session two.  This module closes the
loop: it calls the two cloud models and then applies the driver's *output*
half, so that GFDRV in gives GFDRV out with nothing left implicit.

What the output half actually is, once the arms WRF cannot reach are removed
(``imid_gf`` is a parameter 0 at :69, so ``outtm``/``outqm``/``outqcm``/
``outum``/``outvm``/``cupclwm``/``pretm`` are identically zero and
``cutenm`` never leaves zero):

* three gating scalars, not one.  ``cutens`` starts at 1 and is knocked to 0
  by ``ishallow_g3 == 0`` (:331) OR by ``xmbs <= 0`` (:525); ``cuten`` is 1
  if and only if the deep arm precipitated (:727), and the ``else`` limb
  ZEROES ``kbcon`` and ``ktop`` -- so a deep cloud that formed and did not
  rain is erased from the diagnostics as well as from the tendencies;
* ``RTHCUTEN`` is a potential-temperature tendency and the others are not:
  the sum is divided by the Exner function (:747), and by nothing else;
* the ``t2d < 258`` split (:820-840) sends the SAME condensate number to
  ``RQICUTEN`` or ``RQCCUTEN`` and the same in-cloud water to ``GDC2`` or
  ``GDC``, level by level.  Note ``RQCCUTEN`` and ``GDC``/``GDC2`` are
  written TWICE -- once unconditionally at :813-815 and again inside the
  split -- and the second write wins on every level;
* ``PRATEC`` is gated by an OR over all three precipitation rates (:800), so
  a column where only the shallow arm rained still gets ``RAINCV``;
* ``HTOP``/``HBOT`` start at ``real(kte)``/``real(kts)`` -- note the
  crossing: ``HBOT`` is initialised to the TOP index and ``HTOP`` to the
  bottom one -- and move only under that same OR, by ``ktop + .001``.

Two things a port has to get in the right order, both measured rather than
reasoned about.  ``neg_check`` runs on the shallow tendencies BEFORE the deep
scheme is called (:527), so the deep arm sees a shallow ``prets`` that has
already been rescaled.  And ``cutens`` is decided from ``xmbs`` BEFORE that
``neg_check`` (:525), i.e. from the mass flux, not from the rescaled
precipitation.

Precision: everything here is float32 in WRF's association order.  The
driver's mixed-precision expressions are all in the input half; the output
half is uniformly real(4).
"""

from __future__ import annotations

import numpy as np

from gpuwm.verify.gf_deep_body import cup_gf_column
from gpuwm.verify.gf_deep_ref import F, neg_check
from gpuwm.verify.gf_ref import gf_driver_prep
from gpuwm.verify.gf_shallow_ref import cup_gf_sh_column

__all__ = ["gf_driver_output", "gfdrv_column"]

_Z0 = F(0.0)
_Z1 = F(1.0)
_TCRIT_SPLIT = F(258.0)


def gf_driver_output(
    *, pi, t2d, dt, ishallow, outt, outq, outqc, outu, outv, cupclw, pret,
    ktop, kbcon, outts, outqs, outqcs, cupclws, prets, xmbs, k22s, kbcons,
    ktops, nz,
):
    """module_cu_gf_wrfdrv.F:296-301, :328-331 and :713-840, one column.

    ``outt``..``pret`` are the deep arm's tendencies AFTER ``neg_check``;
    ``outts``..``prets`` are the shallow arm's, likewise.  ``kbcon`` and
    ``ktop`` are modified in place by the ``cuten`` gate and returned.
    """
    pi = np.asarray(pi, dtype=F)
    t2d = np.asarray(t2d, dtype=F)
    dt = F(dt)

    # The mid-level arm, spelled out as the zeros it is.  Keeping the terms
    # rather than dropping them is not decoration: `x + 0.0` is not the
    # identity on a negative zero, and outts can be one.
    zl = np.zeros(nz, dtype=F)
    outtm = outqm = outqcm = outum = outvm = cupclwm = zl
    pretm = _Z0
    cutenm = _Z0

    # :328-331 -- cutens is on unless the shallow arm is off ...
    cutens = _Z1
    if ishallow == 0:
        cutens = _Z0
    # ... or produced no mass flux (:525).  Decided from xmbs, before
    # neg_check touches prets.
    if ishallow == 1 and xmbs <= _Z0:
        cutens = _Z0

    # :724-742
    ktop_deep = int(ktop)
    cuten = _Z0
    if pret > _Z0:
        cuten = _Z1
    else:
        cuten = _Z0
        kbcon = 0
        ktop = 0

    # :743-754
    rthcuten = np.empty(nz, dtype=F)
    rqvcuten = np.empty(nz, dtype=F)
    dudt = np.empty(nz, dtype=F)
    dvdt = np.empty(nz, dtype=F)
    for k in range(nz):
        rthcuten[k] = F(
            F(F(F(cutens * outts[k]) + F(cutenm * outtm[k])) + F(cuten * outt[k]))
            / pi[k]
        )
        rqvcuten[k] = F(
            F(F(cuten * outq[k]) + F(cutens * outqs[k])) + F(cutenm * outqm[k])
        )
        dudt[k] = F(F(outum[k] * cutenm) + F(outu[k] * cuten))
        dvdt[k] = F(F(outvm[k] * cutenm) + F(outv[k] * cuten))

    # :799-808.  HBOT starts at the TOP index and HTOP at the bottom one.
    hbot = F(nz)
    htop = F(1.0)
    pratec = _Z0
    raincv = _Z0
    if pret > _Z0 or pretm > _Z0 or prets > _Z0:
        pratec = F(F(F(cuten * pret) + F(cutenm * pretm)) + F(cutens * prets))
        raincv = F(pratec * dt)
        if F(ktop) > htop:
            htop = F(F(ktop) + F(0.001))
        if F(kbcon) < hbot:
            hbot = F(F(kbcon) + F(0.001))

    # :810-840.  The unconditional write at :813-815 is superseded on every
    # level by the 258 K split below it; both are spelled out because the
    # first is what runs when RQICUTEN is absent, and WRF's own caller always
    # passes it.
    rqccuten = np.empty(nz, dtype=F)
    rqicuten = np.empty(nz, dtype=F)
    gdc = np.empty(nz, dtype=F)
    gdc2 = np.empty(nz, dtype=F)
    for k in range(nz):
        qc = F(F(outqcm[k] + outqcs[k]) + F(outqc[k] * cuten))
        cw = F(F(cupclwm[k] + cupclws[k]) + F(cupclw[k] * cuten))
        if t2d[k] < _TCRIT_SPLIT:
            rqicuten[k] = qc
            rqccuten[k] = _Z0
            gdc2[k] = cw
            gdc[k] = _Z0
        else:
            rqicuten[k] = _Z0
            rqccuten[k] = qc
            gdc[k] = cw
            gdc2[k] = _Z0

    return dict(
        rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqccuten,
        rqicuten=rqicuten, dudt_phy=dudt, dvdt_phy=dvdt, gdc=gdc, gdc2=gdc2,
        pratec=pratec, raincv=raincv, htop=htop, hbot=hbot,
        ktop_deep=ktop_deep, kbcon=int(kbcon), ktop=int(ktop),
        cuten=cuten, cutens=cutens,
        k22_shallow=int(k22s) if ishallow == 1 else 0,
        kbcon_shallow=int(kbcons) if ishallow == 1 else 0,
        ktop_shallow=int(ktops) if ishallow == 1 else 0,
        xmb_shallow=F(xmbs) if ishallow == 1 else _Z0,
    )


def gfdrv_column(
    *, u, v, w, t, qv, p, pi, rho, dz8w, p8w, rthften, rqvften, rthraten,
    rthblten, rqvblten, ht, hfx, qfx, xland, kpbl, dt, dx, ishallow,
    ichoice=0, fzu_up=None, fzu_dn=None, fzu_sh=None,
):
    """One column through the whole of GFDRV, in the driver's own order.

    Preparation (:383-492) -> ``CUP_gf_sh`` -> ``neg_check('shallow')`` ->
    ``cup_gf`` -> ``neg_check('deep')`` -> the output algebra (:713-840).

    ``ktop_deep`` is written twice by the driver -- at :720 inside the
    shallow-diagnostics block and again unconditionally at :726 -- with the
    same value both times, so the port writes it once.
    """
    nz = int(np.asarray(t).shape[0])

    def col(x):
        return np.asarray(x, dtype=F).reshape(1, -1)

    def sca(x):
        return np.asarray([x], dtype=F)

    prep = gf_driver_prep(
        u=col(u), v=col(v), w=col(w), t=col(t), qv=col(qv), p=col(p),
        pi=col(pi), rho=col(rho), dz8w=col(dz8w), p8w=col(p8w),
        rthften=col(rthften), rqvften=col(rqvften), rthraten=col(rthraten),
        rthblten=col(rthblten), rqvblten=col(rqvblten), ht=sca(ht),
        hfx=sca(hfx), qfx=sca(qfx), xland=sca(xland),
        kpbl=np.asarray([kpbl], dtype=np.int32), dt=dt, dx=dx,
    )
    P = {k: (v[0] if np.asarray(v).ndim else v) for k, v in prep.items()}
    kpbli = int(P["kpbli"])

    # ---- the shallow arm (:504-530) ----------------------------------------
    zero = np.zeros(nz + 1, dtype=F)
    sh = None
    outts = zero.copy()
    outqs = zero.copy()
    outqcs = zero.copy()
    outus = zero.copy()
    outvs = zero.copy()
    cupclws = np.zeros(nz, dtype=F)
    prets = _Z0
    xmbs = _Z0
    k22s = kbcons = ktops = 0
    if ishallow == 1:
        sh = cup_gf_sh_column(
            zo=P["zo"], t=P["t2d"], q=P["q2d"], z1=P["ter11"], tn=P["tshall"],
            qo=P["qshall"], po=P["p2d"], psur=P["psur"], dhdt=P["dhdt"],
            kpbl=kpbli, rho=P["rhoi"], hfx=P["hfxi"], qfx=P["qfxi"],
            xland=P["xlandi"], dtime=dt, ichoice=0, fzu_override=fzu_sh,
        )
        outts = sh["outt"].copy()
        outqs = sh["outq"].copy()
        outqcs = sh["outqc"].copy()
        cupclws = sh["cupclw"][1:].copy()
        prets = sh["pre"]
        xmbs = sh["xmb_out"]
        k22s, kbcons, ktops = sh["k22"], sh["kbcon"], sh["ktop"]
        prets, _ = neg_check(
            "shallow", dt, outqs, outts, outus, outvs, outqcs, prets, nz
        )

    # ---- the deep arm (:626-711) -------------------------------------------
    dp = cup_gf_column(
        zo=P["zo"], t=P["t2d"], q=P["q2d"], z1=P["ter11"], tn=P["tn"],
        qo=P["qo"], po=P["p2d"], psur=P["psur"], us=P["us"], vs=P["vs"],
        rho=P["rhoi"], hfx=P["hfxi"], qfx=P["qfxi"], xland=P["xlandi"],
        dx=P["dxi"], omeg=P["omeg"], kpbl=kpbli, ccn=P["ccn"], dtime=dt,
        mconv=P["mconv"], ichoice=ichoice, xmbs_in=xmbs,
        fzu_up=fzu_up, fzu_dn=fzu_dn,
    )
    outt = dp["outt"].copy()
    outq = dp["outq"].copy()
    outqc = dp["outqc"].copy()
    outu = dp["outu"].copy()
    outv = dp["outv"].copy()
    pret = dp["pre"]
    pret, _ = neg_check("deep", dt, outq, outt, outu, outv, outqc, pret, nz)

    out = gf_driver_output(
        pi=pi, t2d=P["t2d"], dt=dt, ishallow=ishallow,
        outt=outt[1:], outq=outq[1:], outqc=outqc[1:], outu=outu[1:],
        outv=outv[1:], cupclw=dp["cupclw"][1:], pret=pret,
        ktop=dp["ktop"], kbcon=dp["kbcon"],
        outts=outts[1:], outqs=outqs[1:], outqcs=outqcs[1:],
        cupclws=cupclws, prets=prets, xmbs=xmbs, k22s=k22s, kbcons=kbcons,
        ktops=ktops, nz=nz,
    )
    out["prep"] = P
    out["deep"] = dp
    out["shallow"] = sh
    out["pret"] = pret
    out["prets"] = prets
    # The two schemes' tendencies AFTER neg_check and BEFORE the gating --
    # the boundary run_cup_gf.F90 captures, and the one that separates a
    # composition error from GFDRV's own mixed-precision preparation.
    out["outt"] = outt[1:]
    out["outq"] = outq[1:]
    out["outqc"] = outqc[1:]
    out["outu"] = outu[1:]
    out["outv"] = outv[1:]
    out["outts"] = outts[1:]
    out["outqs"] = outqs[1:]
    out["outqcs"] = outqcs[1:]
    return out
