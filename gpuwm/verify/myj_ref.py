"""CPU float32 references for the WRF v4.6.1 MYJ PBL / Eta surface layer.

Transcribed line by line from the byte-frozen reference bundle sources

* ``phys/module_bl_myjpbl.F``  (MYJPBL: MIXLEN, PRODQ2, DIFCOF, VDIFQ,
  VDIFH, VDIFV) -- Janjic (2002) NCEP Office Note 437, Mellor-Yamada
  level 2.5 as extended by Janjic; and
* ``phys/module_sf_myjsfc.F``  (MYJSFC / SFCDIF / MYJSFCINIT) -- the Eta
  similarity surface layer with Janjic (1994) viscous sublayer over water
  and the Zilitinkevich thermal-roughness fix over land.

Both column functions take gpuwm's bottom-up ``(nz,)`` columns and flip to
MYJ's own top-down 1-based layout internally, exactly as the Fortran
drivers flip WRF's bottom-up arrays (module_bl_myjpbl.F:364-383,
module_sf_myjsfc.F:237-259).  All arithmetic is ``np.float32`` with the
Fortran's expression shapes.  Transcendentals are NumPy's float32
``log``/``exp``/``sqrt``/``atan``; pinning them to a specific libm word
(the ``noahmp_libm`` treatment) is deliberately deferred to the MYJ
oracle campaign -- NO oracle comparison against the WRF Fortran has been
run yet, and nothing in this module claims one.

WRF quirks preserved on purpose (cited where they occur):

* ``CT`` is identically zero in ARW v4.6.1: MYJSFC zeroes it every call
  (module_sf_myjsfc.F:206-211) and SFCDIF's countergradient block is
  commented out (:816-825, ``CT=0.``), so MIXLEN's ``DTH(K)+CT`` fix
  (module_bl_myjpbl.F:845-850) and VDIFH's ``RKCT`` term add exactly zero.
  Transcribed anyway so the seam is WRF's.
* MYJPBL recomputes QSFC over water from ``PQ0SEA`` and over land from
  ``ELFLX`` (module_bl_myjpbl.F:547-564); QSFC/QZ0/THZ0 here follow the
  Eta convention (specific humidity), exactly as the shared WRF fields do
  under this scheme pairing.
* The Fortran's ``IVGTYP/ISURBAN/IZ0TLND`` SFCDIF arguments are dead: the
  Chen-Zhang CZIL block is commented out and ``CZIL=0.1`` is hard-coded
  (module_sf_myjsfc.F:689-697), so the reference takes none of them.

DECLARED DIVERGENCES FROM WRF (one, stated here and again where it lives):

* The interface-height column is GROUND-relative, not sea-level relative:
  WRF seeds ``ZINT(KTE+1)=HT`` (module_bl_myjpbl.F:312,
  module_sf_myjsfc.F:162) and gpuwm seeds 0.  HT cancels exactly in real
  arithmetic and only approximately in float32; gpuwm's column is the more
  accurate one.  Full declaration and the measurement that bounds it:
  :func:`_interface_heights`.
"""

from __future__ import annotations

import numpy as np

# The similarity-function tables live in core (NumPy-only) because the CUDA
# launcher must reach them without importing the verification tree; verify
# importing core is the allowed direction (gpuwm/core/state.py's
# sase_limits precedent).  Re-exported here so the reference's callers keep
# one import site.
from gpuwm.core.myjsfc_tables import KZTM, KZTM2, build_psi_tables

F = np.float32

# ---------------------------------------------------------------------------
# share/module_model_constants.F values consumed by both modules.
G = F(9.81)
R_D = F(287.0)
R_V = F(461.6)
CP = F(F(7.0) * R_D / F(2.0))
XLV = F(2.5e6)
XLS = F(2.85e6)
EP_1 = F(R_V / R_D - F(1.0))
P608 = F(R_V / R_D - F(1.0))          # rvovrd - 1.
PQ0 = F(379.90516)
A2C = F(17.2693882)                    # module_model_constants a2
A3C = F(273.16)
A4C = F(35.86)
EPSQ2 = F(0.2)
P1000MB = F(1.0e5)

# ---------------------------------------------------------------------------
# module_bl_myjpbl.F:25-124 parameters (stepwise float32, the Fortran's
# compile-time single-precision folding).
VKARMAN = F(0.4)
CAPA = F(R_D / CP)
RLIVWV = F(XLS / XLV)
ELOCP = F(F(2.72e6) / CP)
EPS1 = F(1.0e-12)
EPS2 = F(0.0)
EPSL = F(0.32)
EPSRU = F(1.0e-7)
EPSRS = F(1.0e-7)
EPSTRB = F(1.0e-24)
FH_PBL = F(1.01)
ALPH = F(0.30)
BETA = F(F(1.0) / F(273.0))
EL0MAX = F(1000.0)
EL0MIN = F(1.0)
ELFC = F(F(0.23) * F(0.5))
PRT = F(1.0)
A1 = F(0.659888514560862645)
A2X = F(0.6574209922667784586)
B1 = F(11.87799326209552761)
B2 = F(7.226971804046074028)
C1 = F(0.000830955950095854396)
ELZ0 = F(0.0)
ESQ = F(5.0)
SEAFC = F(0.98)
PQ0SEA = F(PQ0 * SEAFC)
BTG = F(BETA * G)
RB1 = F(F(1.0) / B1)

# module_bl_myjpbl.F:62-92 (stepwise, left-to-right with parentheses).
ADNH = F(F(F(F(F(F(9.0) * A1) * A2X) * A2X)
           * F(F(F(12.0) * A1) + F(F(3.0) * B2))) * F(BTG * BTG))
ADNM = F(F(F(F(F(F(18.0) * A1) * A1) * A2X) * F(B2 - F(F(3.0) * A2X))) * BTG)
ANMH = F(F(F(F(F(-9.0) * A1) * A2X) * A2X) * F(BTG * BTG))
ANMM = F(F(F(F(F(-3.0) * A1) * A2X)
           * F(F(F(F(F(3.0) * A2X) + F(F(F(3.0) * B2) * C1))
                 + F(F(F(18.0) * A1) * C1)) - B2)) * BTG)
BDNH = F(F(F(F(3.0) * A2X) * F(F(F(7.0) * A1) + B2)) * BTG)
BDNM = F(F(F(6.0) * A1) * A1)
BEQH = F(F(F(F(A2X * B1) * BTG)
           + F(F(F(F(3.0) * A2X) * F(F(F(7.0) * A1) + B2)) * BTG)))
BEQM = F(F(F(F(-A1) * B1) * F(F(1.0) - F(F(3.0) * C1)))
         + F(F(F(6.0) * A1) * A1))
BNMH = F(F(-A2X) * BTG)
BNMM = F(A1 * F(F(1.0) - F(F(3.0) * C1)))
BSHH = F(F(F(F(F(9.0) * A1) * A2X) * A2X) * BTG)
BSHM = F(F(F(F(F(18.0) * A1) * A1) * A2X) * C1)
BSMH = F(F(F(F(F(-3.0) * A1) * A2X)
           * F(F(F(F(F(3.0) * A2X) + F(F(F(3.0) * B2) * C1))
                 + F(F(F(12.0) * A1) * C1)) - B2)) * BTG)
CESH = A2X
CESM = F(A1 * F(F(1.0) - F(F(3.0) * C1)))

# module_bl_myjpbl.F:98-101 free equilibrium term.
AEQH = F(F(F(F(F(F(F(9.0) * A1) * A2X) * A2X) * B1) * F(BTG * BTG))
         + F(F(F(F(F(F(9.0) * A1) * A2X) * A2X)
               * F(F(F(F(12.0) * A1)) + F(F(3.0) * B2))) * F(BTG * BTG)))
AEQM = F(F(F(F(F(F(F(3.0) * A1) * A2X) * B1)
            * F(F(F(F(F(3.0) * A2X) + F(F(F(3.0) * B2) * C1))
                  + F(F(F(18.0) * A1) * C1)) - B2)) * BTG)
         + F(F(F(F(F(F(18.0) * A1) * A1) * A2X)
               * F(B2 - F(F(3.0) * A2X))) * BTG))

# module_bl_myjpbl.F:107-124 forbidden area / near isotropy.
REQU = F(-AEQH / AEQM)
EPSGH = F(1.0e-9)
EPSGM = F(REQU * EPSGH)
UBRYL = F(F(F(F(F(F(F(F(F(18.0) * REQU) * A1) * A1) * A2X) * B2) * C1) * BTG)
          + F(F(F(F(F(F(F(9.0) * A1) * A2X) * A2X) * B2) * F(BTG * BTG))))
UBRYL = F(UBRYL / F(F(REQU * ADNM) + ADNH))
UBRY = F(F(F(1.0) + EPSRS) * UBRYL)
UBRY3 = F(F(3.0) * UBRY)
AUBH = F(F(F(F(F(F(F(27.0) * A1) * A2X) * A2X) * B2) * F(BTG * BTG))
         - F(ADNH * UBRY3))
AUBM = F(F(F(F(F(F(F(54.0) * A1) * A1) * A2X) * B2) * F(C1 * BTG))
         - F(ADNM * UBRY3))
BUBH = F(F(F(F(F(F(9.0) * A1) * A2X) + F(F(F(3.0) * A2X) * B2)) * BTG)
         - F(BDNH * UBRY3))
BUBM = F(F(F(F(F(18.0) * A1) * A1) * C1) - F(BDNM * UBRY3))
CUBR = F(F(1.0) - UBRY3)
RCUBR = F(F(1.0) / CUBR)

# ---------------------------------------------------------------------------
# module_sf_myjsfc.F:23-55 parameters.
ITRMX = 5
ELOCP_SFC = ELOCP
GOCP02 = F(F(G / CP) * F(2.0))
GOCP10 = F(F(G / CP) * F(10.0))
EPSU2_SFC = F(1.0e-6)
EPSUST_SFC = F(1.0e-9)
EPSZT = F(1.0e-28)
A2S = F(17.2693882)
A3S = F(273.16)
A4S = F(35.86)
EXCML = F(0.0001)
EXCMS = F(0.0001)
GLKBR = F(10.0)
GLKBS = F(30.0)
PI = F(3.1415926)
QVISC = F(2.1e-5)
RIC = F(0.505)
SMALL = F(0.35)
SQPR = F(0.84)
SQSC = F(0.84)
SQVISC = F(258.2)
TVISC = F(2.1e-5)
USTC = F(0.7)
USTR = F(0.225)
VISC = F(1.5e-5)
FH_SFC = F(1.01)
WWST = F(1.2)
ZTFC = F(1.0)
CZIV = F(SMALL * GLKBS)
GRRS = F(GLKBR / GLKBS)
RTVISC = F(F(1.0) / TVISC)
RVISC = F(F(1.0) / VISC)
ZQRZT = F(SQSC / SQPR)
FZQ1 = F(F(RTVISC * QVISC) * ZQRZT)
FZQ2 = F(F(RTVISC * QVISC) * ZQRZT)
FZT1 = F(F(RVISC * TVISC) * SQPR)
FZT2 = F(F(F(CZIV * GRRS) * TVISC) * SQPR)
FZU1 = F(CZIV * VISC)
PIHF = F(F(0.5) * PI)
RQVISC = F(F(1.0) / QVISC)
USTFC = F(F(0.018) / G)
WWST2 = F(WWST * WWST)

__all__ = [
    "EPSQ2", "F", "KZTM", "KZTM2", "myjsfc_psi_tables",
    "np_myjpbl_column", "np_myjsfc_column",
]

#: The MYJSFCINIT table set (module_sf_myjsfc.F:1174-1299), built once in
#: :mod:`gpuwm.core.myjsfc_tables` and shared with the CUDA launcher.
myjsfc_psi_tables = build_psi_tables


def _interface_heights(dz, ht=F(0.0)):
    """MYJ's ``ZINT`` column, top-down, as both modules build it.

    Returns ``zh`` of length ``nz+1`` where ``zh[m]`` is the top interface
    of top-down layer ``m`` (0-based) and ``zh[nz]`` is the ground.  The
    recursion is WRF's, ``ZINT(I,K,J)=ZINT(I,K+1,J)+DZ(I,KFLIP,J)`` with
    ``KFLIP=KTE+1-K`` (module_bl_myjpbl.F:313-320,
    module_sf_myjsfc.F:163-170).

    DELIBERATE DIVERGENCE, gpuwm goes its own way (float32 precision):
    WRF seeds that recursion with the TERRAIN HEIGHT --
    ``ZINT(I,KTE+1,J)=HT(I,J)`` (module_bl_myjpbl.F:312,
    module_sf_myjsfc.F:162) -- so every interface height it carries is
    height above SEA LEVEL.  gpuwm seeds it with ``ht=0.0`` by default, so
    the column is height above GROUND.  Every consumer in both modules
    reads only DIFFERENCES of these heights (``Z(K)-Z(K+1)``,
    ``Z(K)-Z(K+2)``, ``ZHK(LPBL)-ZHK(LMH+1)``, ``ZSL=0.5*(Z(1)-Z(2))``),
    so HT cancels EXACTLY in real arithmetic -- but not in float32, where
    subtracting two numbers offset by kilometres loses low-order bits the
    ground-relative column keeps.  gpuwm's answer is the more accurate
    one, which is why it ships; declaring it is the standing rule, since
    an undeclared divergence is what is forbidden and this term WILL show
    up in the ULP table when the MYJ oracle campaign runs.

    ``ht`` is not plumbed through the shipped launchers or either CUDA
    translation unit -- production always runs the ``ht=0`` column.  The
    parameter exists so the divergence is MEASURABLE rather than merely
    asserted, and
    ``tests/test_myj_port.py::test_the_dropped_terrain_height_cancels_in_float32``
    drives it over five columns crossed with terrain at 1523.7314,
    2987.3129 and 4411.0837 m.  MEASURED THERE: KPBL identical in every
    case; non-tendency fields at most 69 ULP apart, attained on ``lh``
    over water where the move is 1.645e-05 W m-2; at most 5.75e-06 in
    relative terms, attained on ``qfx`` rather than on ``lh`` -- the two
    peak on different fields, so the test measures them separately and
    neither may be divided out of the other; tendency rows at most 2.05
    quanta of the source field's ULP over dt.
    """
    dz = np.asarray(dz, dtype=F)
    nz = dz.shape[0]
    zh = np.zeros(nz + 1, dtype=F)
    zh[nz] = F(ht)
    for m in range(nz - 1, -1, -1):
        zh[m] = F(zh[m + 1] + dz[nz - 1 - m])
    return zh


def _table(psi: np.ndarray, zeta, ztmin, dzeta):
    """One table interpolation (module_sf_myjsfc.F:583-588 pattern)."""
    rz = F(F(zeta - ztmin) / dzeta)
    k = int(rz)
    rdzt = F(rz - F(float(k)))
    k = min(k, KZTM2)
    k = max(k, 0)
    return F(F(F(psi[k + 1] - psi[k]) * rdzt) + psi[k])


def np_myjsfc_column(
        *, dz, tke, u1, v1, t1, th1, qv1, qc1, p1, psfc,
        tsk, qsfc, thz0, qz0, uz0, vz0, ust, znt, z0base,
        akhs, akms, xland, mavail, itimestep, tables, ht=F(0.0)):
    """One MYJSFC surface column (module_sf_myjsfc.F:66-358 + SFCDIF).

    ``dz``/``tke`` are full bottom-up float32 columns; the remaining state
    is float32 scalars.  ``qv1`` is a mixing ratio (WRF ``qv_curr``);
    ``tke`` is TKE (``Q2K = 2*TKE``, :246).  Returns the complete inout +
    output set as a dict of float32 scalars (plus ``pblh``).  ``lowlyr=1``
    (ARW sigma, MYJSFCINIT:1117-1124).

    ``ht`` is the interface-height origin (module_sf_myjsfc.F:162-170).  It
    defaults to gpuwm's DECLARED ground-relative 0 rather than WRF's
    terrain height; :func:`_interface_heights` carries the declaration and
    names the test that measures it.
    """
    dz = np.asarray(dz, dtype=F)
    tke = np.asarray(tke, dtype=F)
    nz = dz.shape[0]
    ntsd = int(itimestep)
    zh = _interface_heights(dz, ht)
    # :186-211  first-step seeding + CT.
    ust = F(ust)
    if ntsd == 1:
        ust = F(0.1)
    ct = F(0.0)
    seamask = F(F(xland) - F(1.0))
    thsk = F(F(tsk) / F(F(psfc) / P1000MB) ** CAPA)
    # Q2 profile in myj layout for the PBLH scan (:246,:263-277).
    q2m = np.empty(nz, dtype=F)
    for m in range(nz):
        q2m[m] = F(F(2.0) * tke[nz - 1 - m])
    lmh = nz          # Fortran LMH (1-based)
    lpbl = lmh
    for kf in range(lmh - 1, 0, -1):      # Fortran K=LMH-1..1
        if q2m[kf - 1] <= F(EPSQ2 * FH_SFC):
            lpbl = kf
            break
    else:
        lpbl = 1
    pblh = F(zh[lpbl - 1] - zh[lmh])      # PBLH=ZHK(LPBL)-ZHK(LMH+1) (:277)
    # Lowest-layer state (:284-294).
    plow = F(p1)
    tlow = F(t1)
    thlow = F(th1)
    qlow = F(F(qv1) / F(F(1.0) + F(qv1)))
    cwmlow = F(qc1)
    thelow = F(F(F(cwmlow * F(-ELOCP_SFC / tlow)) + F(1.0)) * thlow)
    ulow = F(u1)
    vlow = F(v1)
    zsl = F(F(zh[lmh - 1] - zh[lmh]) * F(0.5))
    apesfc = F(F(F(psfc) / P1000MB) ** CAPA)
    if ntsd == 1:
        tz0 = F(tsk)                       # :296-305 (ARW branch)
    else:
        tz0 = F(F(thz0) * apesfc)

    # ---------------- SFCDIF (:361-1056) --------------------------------
    t = tables
    psim1, psih1 = t["psim1"], t["psih1"]
    psim2, psih2 = t["psim2"], t["psih2"]
    dzeta1, dzeta2 = t["dzeta1"], t["dzeta2"]
    ztmin1, ztmax1 = t["ztmin1"], t["ztmax1"]
    ztmin2, ztmax2 = t["ztmin2"], t["ztmax2"]
    fh01, fh02 = t["fh01"], t["fh02"]

    qs = F(qsfc)
    thz0 = F(thz0)
    qz0 = F(qz0)
    uz0 = F(uz0)
    vz0 = F(vz0)
    z0 = F(znt)
    akhs = F(akhs)
    akms = F(akms)

    rdz = F(F(1.0) / zsl)
    cxchl = F(EXCML * rdz)
    cxchs = F(EXCMS * rdz)
    btgx = F(G / thlow)
    elfc = F(VKARMAN * btgx)
    if pblh > F(1000.0):
        btgh = F(btgx * pblh)
    else:
        btgh = F(btgx * F(1000.0))

    wstar2 = F(0.0)
    if seamask > F(0.5):
        # ---- sea points (:443-636) ----
        for _ in range(ITRMX):
            z0 = F(max(F(F(USTFC * ust) * ust), F(1.59e-5)))
            if ust < USTC:
                if ust < USTR:
                    if ntsd == 1:
                        akms = cxchs
                        akhs = cxchs
                        qs = qlow
                    zu = F(F(FZU1 * F(np.sqrt(np.sqrt(
                        F(F(z0 * ust) * RVISC))))) / ust)
                    wght = F(F(akms * zu) * RVISC)
                    rwgh = F(wght / F(wght + F(1.0)))
                    uz0 = F(F(F(ulow * rwgh) + uz0) * F(0.5))
                    vz0 = F(F(F(vlow * rwgh) + vz0) * F(0.5))
                    zt = F(FZT1 * zu)
                    zq = F(FZQ1 * zt)
                    wghtt = F(F(akhs * zt) * RTVISC)
                    wghtq = F(F(akhs * zq) * RQVISC)
                    if ntsd > 1:
                        thz0 = F(F(F(F(F(wghtt * thlow) + thsk)
                                     / F(wghtt + F(1.0))) + thz0) * F(0.5))
                        qz0 = F(F(F(F(F(wghtq * qlow) + qs)
                                    / F(wghtq + F(1.0))) + qz0) * F(0.5))
                    else:
                        thz0 = F(F(F(wghtt * thlow) + thsk)
                                 / F(wghtt + F(1.0)))
                        qz0 = F(F(F(wghtq * qlow) + qs)
                                / F(wghtq + F(1.0)))
                if USTR <= ust < USTC:
                    zu = z0
                    uz0 = F(0.0)
                    vz0 = F(0.0)
                    zt = F(F(FZT2 * F(np.sqrt(np.sqrt(
                        F(F(z0 * ust) * RVISC))))) / ust)
                    zq = F(FZQ2 * zt)
                    wghtt = F(F(akhs * zt) * RTVISC)
                    wghtq = F(F(akhs * zq) * RQVISC)
                    if ntsd > 1:
                        thz0 = F(F(F(F(F(wghtt * thlow) + thsk)
                                     / F(wghtt + F(1.0))) + thz0) * F(0.5))
                        qz0 = F(F(F(F(F(wghtq * qlow) + qs)
                                    / F(wghtq + F(1.0))) + qz0) * F(0.5))
                    else:
                        thz0 = F(F(F(wghtt * thlow) + thsk)
                                 / F(wghtt + F(1.0)))
                        qz0 = F(F(F(wghtq * qlow) + qs)
                                / F(wghtq + F(1.0)))
            else:
                zu = z0
                uz0 = F(0.0)
                vz0 = F(0.0)
                zt = z0
                thz0 = thsk
                zq = z0
                qz0 = qs
            tem = F(F(tlow + tz0) * F(0.5))
            thm = F(F(thelow + thz0) * F(0.5))
            a = F(thm * P608)
            b = F(F(F(F(ELOCP_SFC / tem) - F(1.0)) - P608) * thm)
            dthv = F(F(F(F(thelow - thz0)
                         * F(F(F(F(qlow + qz0) + cwmlow)
                               * F(F(0.5) * P608)) + F(1.0)))
                       + F(F(F(qlow - qz0) + cwmlow) * a))
                     + F(cwmlow * b))
            du2 = F(max(F(F(F(ulow - uz0) * F(ulow - uz0))
                          + F(F(vlow - vz0) * F(vlow - vz0))), EPSU2_SFC))
            rib = F(F(F(btgx * dthv) * zsl) / du2)
            zslu = F(zsl + zu)
            zslt = F(zsl + zt)
            rzsu = F(zslu / zu)
            rzst = F(zslt / zt)
            rlogu = F(np.log(rzsu))
            rlogt = F(np.log(rzst))
            rlmo = F(F(F(elfc * akhs) * dthv) / F(F(ust * ust) * ust))
            zetalu = F(zslu * rlmo)
            zetalt = F(zslt * rlmo)
            zetau = F(zu * rlmo)
            zetat = F(zt * rlmo)
            zetalu = F(min(max(zetalu, ztmin1), ztmax1))
            zetalt = F(min(max(zetalt, ztmin1), ztmax1))
            zetau = F(min(max(zetau, F(ztmin1 / rzsu)), F(ztmax1 / rzsu)))
            zetat = F(min(max(zetat, F(ztmin1 / rzst)), F(ztmax1 / rzst)))
            psmz = _table(psim1, zetau, ztmin1, dzeta1)
            psmzl = _table(psim1, zetalu, ztmin1, dzeta1)
            simm = F(F(psmzl - psmz) + rlogu)
            pshz = _table(psih1, zetat, ztmin1, dzeta1)
            pshzl = _table(psih1, zetalt, ztmin1, dzeta1)
            simh = F(F(F(pshzl - pshz) + rlogt) * fh01)
            ustark = F(ust * VKARMAN)
            akms = F(max(F(ustark / simm), cxchs))
            akhs = F(max(F(ustark / simh), cxchs))
            if dthv <= F(0.0):
                wstar2 = F(WWST2 * F(abs(F(F(F(btgh * akhs) * dthv)))
                                     ** F(F(2.0) / F(3.0))))
            else:
                wstar2 = F(0.0)
            ust = F(max(F(np.sqrt(F(akms * F(np.sqrt(F(du2 + wstar2)))))),
                        EPSUST_SFC))
    else:
        # ---- land points (:644-806) ----
        if ntsd == 1:
            qs = qlow
        zu = z0
        uz0 = F(0.0)
        vz0 = F(0.0)
        zt = F(zu * ZTFC)
        thz0 = thsk
        zq = zt
        qz0 = qs
        tem = F(F(tlow + tz0) * F(0.5))
        thm = F(F(thelow + thz0) * F(0.5))
        a = F(thm * P608)
        b = F(F(F(F(ELOCP_SFC / tem) - F(1.0)) - P608) * thm)
        dthv = F(F(F(F(thelow - thz0)
                     * F(F(F(F(qlow + qz0) + cwmlow)
                           * F(F(0.5) * P608)) + F(1.0)))
                   + F(F(F(qlow - qz0) + cwmlow) * a))
                 + F(cwmlow * b))
        du2 = F(max(F(F(F(ulow - uz0) * F(ulow - uz0))
                      + F(F(vlow - vz0) * F(vlow - vz0))), EPSU2_SFC))
        rib = F(F(F(btgx * dthv) * zsl) / du2)
        zslu = F(zsl + zu)
        rzsu = F(zslu / zu)
        rlogu = F(np.log(rzsu))
        zslt = F(zsl + zu)       # u,v and t are at the same level (:685)
        czil = F(0.1)            # :697 (Chen-Zhang block commented out)
        zilfc = F(F(F(-czil) * VKARMAN) * SQVISC)
        czetmax = F(10.0)
        if dthv > F(0.0):
            if rib < RIC:
                zzil = F(zilfc * F(F(1.0) + F(F(F(rib / RIC) * F(rib / RIC))
                                              * czetmax)))
            else:
                zzil = F(zilfc * F(F(1.0) + czetmax))
        else:
            zzil = zilfc
        for _ in range(ITRMX):
            # Zilitinkevich fix (:733): the v4.6.1 form uses Z0BASE.
            zt = F(max(F(F(np.exp(F(zzil * F(np.sqrt(F(ust * F(z0base)))))))
                        * F(z0base)), EPSZT))
            rzst = F(zslt / zt)
            rlogt = F(np.log(rzst))
            rlmo = F(F(F(elfc * akhs) * dthv) / F(F(ust * ust) * ust))
            zetalu = F(zslu * rlmo)
            zetalt = F(zslt * rlmo)
            zetau = F(zu * rlmo)
            zetat = F(zt * rlmo)
            zetalu = F(min(max(zetalu, ztmin2), ztmax2))
            zetalt = F(min(max(zetalt, ztmin2), ztmax2))
            zetau = F(min(max(zetau, F(ztmin2 / rzsu)), F(ztmax2 / rzsu)))
            zetat = F(min(max(zetat, F(ztmin2 / rzst)), F(ztmax2 / rzst)))
            psmz = _table(psim2, zetau, ztmin2, dzeta2)
            psmzl = _table(psim2, zetalu, ztmin2, dzeta2)
            simm = F(F(psmzl - psmz) + rlogu)
            pshz = _table(psih2, zetat, ztmin2, dzeta2)
            pshzl = _table(psih2, zetalt, ztmin2, dzeta2)
            simh = F(F(F(pshzl - pshz) + rlogt) * fh02)
            ustark = F(ust * VKARMAN)
            akms = F(max(F(ustark / simm), cxchl))
            akhs = F(max(F(ustark / simh), cxchl))
            if dthv <= F(0.0):
                wstar2 = F(WWST2 * F(abs(F(F(F(btgh * akhs) * dthv)))
                                     ** F(F(2.0) / F(3.0))))
            else:
                wstar2 = F(0.0)
            ust = F(max(F(np.sqrt(F(akms * F(np.sqrt(F(du2 + wstar2)))))),
                        EPSUST_SFC))
    ct = F(0.0)   # :816-825, countergradient block commented out.
    rlmo_out = rlmo

    # ---- 2m / 10m diagnostics (:829-1023) ----
    umflx = F(akms * F(ulow - uz0))
    vmflx = F(akms * F(vlow - vz0))
    hsflx = F(akhs * F(thlow - thz0))
    hlflx = F(akhs * F(qlow - qz0))
    zu10 = F(zu + F(10.0))
    zt02 = F(zt + F(2.0))
    zt10 = F(zt + F(10.0))
    rlnu10 = F(np.log(F(zu10 / zu)))
    rlnt02 = F(np.log(F(zt02 / zt)))
    rlnt10 = F(np.log(F(zt10 / zt)))
    ztau10 = F(zu10 * rlmo_out)
    ztat02 = F(zt02 * rlmo_out)
    ztat10 = F(zt10 * rlmo_out)
    if seamask > F(0.5):
        ztau10 = F(min(max(ztau10, ztmin1), ztmax1))
        ztat02 = F(min(max(ztat02, ztmin1), ztmax1))
        ztat10 = F(min(max(ztat10, ztmin1), ztmax1))
        psm10 = _table(psim1, ztau10, ztmin1, dzeta1)
        simm10 = F(F(psm10 - psmz) + rlnu10)
        psh02 = _table(psih1, ztat02, ztmin1, dzeta1)
        simh02 = F(F(F(psh02 - pshz) + rlnt02) * fh01)
        psh10 = _table(psih1, ztat10, ztmin1, dzeta1)
        simh10 = F(F(F(psh10 - pshz) + rlnt10) * fh01)
        akms10 = F(max(F(ustark / simm10), cxchs))
        akhs02 = F(max(F(ustark / simh02), cxchs))
        akhs10 = F(max(F(ustark / simh10), cxchs))
    else:
        ztau10 = F(min(max(ztau10, ztmin2), ztmax2))
        ztat02 = F(min(max(ztat02, ztmin2), ztmax2))
        ztat10 = F(min(max(ztat10, ztmin2), ztmax2))
        psm10 = _table(psim2, ztau10, ztmin2, dzeta2)
        simm10 = F(F(psm10 - psmz) + rlnu10)
        psh02 = _table(psih2, ztat02, ztmin2, dzeta2)
        simh02 = F(F(F(psh02 - pshz) + rlnt02) * fh02)
        psh10 = _table(psih2, ztat10, ztmin2, dzeta2)
        simh10 = F(F(F(psh10 - pshz) + rlnt10) * fh02)
        akms10 = F(max(F(ustark / simm10), cxchl))
        akhs02 = F(max(F(ustark / simh02), cxchl))
        akhs10 = F(max(F(ustark / simh10), cxchl))
    u10 = F(F(umflx / akms10) + uz0)
    v10 = F(F(vmflx / akms10) + vz0)
    th02 = F(F(hsflx / akhs02) + thz0)
    if ((thlow > thz0 and (th02 < thz0 or th02 > thlow))
            or (thlow < thz0 and (th02 > thz0 or th02 < thlow))):
        th02 = F(thz0 + F(F(F(2.0) * rdz) * F(thlow - thz0)))
    th10 = F(F(hsflx / akhs10) + thz0)
    if ((thlow > thz0 and (th10 < thz0 or th10 > thlow))
            or (thlow < thz0 and (th10 > thz0 or th10 < thlow))):
        th10 = F(thz0 + F(F(F(10.0) * rdz) * F(thlow - thz0)))
    q02 = F(F(hlflx / akhs02) + qz0)
    q10 = F(F(hlflx / akhs10) + qz0)
    term1 = F(F(-0.068283) / tlow)
    pshltr = F(F(psfc) * F(np.exp(term1)))
    u10e = u10
    v10e = v10
    if seamask < F(0.5):
        # Equivalent z0 for shelter winds over land (:988-1020).
        zuuz = F(min(F(zu * F(0.50)), F(0.18)))
        zu = F(max(F(zu * F(0.35)), zuuz))
        zu10 = F(zu + F(10.0))
        rzsu_l = F(zu10 / zu)
        rlnu10 = F(np.log(rzsu_l))
        zetau_l = F(zu * rlmo_out)
        ztau10 = F(zu10 * rlmo_out)
        ztau10 = F(min(max(ztau10, ztmin2), ztmax2))
        zetau_l = F(min(max(zetau_l, F(ztmin2 / rzsu_l)),
                        F(ztmax2 / rzsu_l)))
        psm10 = _table(psim2, ztau10, ztmin2, dzeta2)
        simm10 = F(F(psm10 - psmz) + rlnu10)
        ekms10 = F(max(F(ustark / simm10), cxchl))
        u10e = F(F(umflx / ekms10) + uz0)
        v10e = F(F(vmflx / ekms10) + vz0)
    # ---- WRF driver arrays (:1029-1053) ----
    rlow = F(plow / F(R_D * tlow))
    chs = akhs
    chs2 = akhs02
    cqs2 = akhs02
    hfx = F(F(F(-rlow) * CP) * hsflx)
    qfx = F(F(F(-rlow) * hlflx) * F(mavail))
    flx_lh = F(XLV * qfx)
    flhc = F(F(rlow * CP) * akhs)
    flqc = F(F(rlow * akhs) * F(mavail))
    qgh = F(F(F(F(F(F(1.0) - seamask) * PQ0) + F(seamask * PQ0SEA)) / plow)
            * F(np.exp(F(F(A2S * F(tlow - A3S)) / F(tlow - A4S)))))
    qgh = F(qgh / F(F(1.0) - qgh))       # convert to mixing ratio (:1041)
    cpm = F(CP * F(F(1.0) + F(F(0.8) * qlow)))
    if seamask > F(0.5):
        qs = F(F(PQ0SEA / F(psfc))
               * F(np.exp(F(F(A2S * F(F(tsk) - A3S)) / F(F(tsk) - A4S)))))
        qs = F(qs / F(F(1.0) - qs))
    # ---- shelter supersaturation removal (module_sf_myjsfc.F:326-349) ----
    tshltr = th02          # SFCDIF's TH02 dummy is MYJSFC's TSHLTR.
    qshltr = q02           # SFCDIF's Q02 dummy is MYJSFC's QSHLTR.
    rapa = apesfc
    th02p = tshltr
    th10p = th10
    th2_grid = tshltr      # TH02(I,J)=TSHLTR(I,J) (:329)
    rapa02 = F(rapa - F(GOCP02 / th02p))
    rapa10 = F(rapa - F(GOCP10 / th10p))
    t02p = F(th02p * rapa02)
    t10p = F(th10p * rapa10)
    t2_grid = F(th2_grid * apesfc)      # :337
    rcap = F(F(1.0) / CAPA)
    p02p = F(F(rapa02 ** rcap) * P1000MB)
    p10p = F(F(rapa10 ** rcap) * P1000MB)
    qs02 = F(F(PQ0 / p02p)
             * F(np.exp(F(F(A2C * F(t02p - A3C)) / F(t02p - A4C)))))
    qs10 = F(F(PQ0 / p10p)
             * F(np.exp(F(F(A2C * F(t10p - A3C)) / F(t10p - A4C)))))
    if qshltr > qs02:
        qshltr = qs02
    if q10 > qs10:
        q10 = qs10
    q02_grid = F(qshltr / F(F(1.0) - qshltr))   # :349, 2m mixing ratio
    return {
        "ust": ust, "znt": z0, "thz0": thz0, "qz0": qz0,
        "uz0": uz0, "vz0": vz0, "qsfc": qs, "akhs": akhs, "akms": akms,
        "rmol": rlmo_out, "ct": ct, "pblh": pblh, "rib": rib,
        "chs": chs, "chs2": chs2, "cqs2": cqs2,
        "hfx": hfx, "qfx": qfx, "lh": flx_lh,
        "flhc": flhc, "flqc": flqc, "qgh": qgh, "cpm": cpm,
        "u10": u10, "v10": v10, "t2": t2_grid, "th2": th2_grid,
        "tshltr": tshltr, "th10": th10, "q2": q02_grid,
        "qshltr": qshltr, "q10": q10, "pshltr": pshltr,
        "u10e": u10e, "v10e": v10e,
    }


# ---------------------------------------------------------------------------
# MYJPBL column pieces.  All arrays are in myj top-down 0-based layout
# (python index m = Fortran K-1); ``zh`` has nz+1 entries with zh[nz]=0.
# ---------------------------------------------------------------------------

def _mixlen(lmh, u, v, t, the, q, cwm, q2, zh, ct):
    """module_bl_myjpbl.F:771-963 (MIXLEN)."""
    nzm = lmh - 1
    lpbl = lmh
    for kf in range(lmh - 1, 0, -1):
        if q2[kf - 1] <= F(EPSQ2 * FH_PBL):
            lpbl = kf
            break
    else:
        lpbl = 1
    pblh = F(zh[lpbl] - zh[lmh])          # PBLH=Z(LPBL+1)-Z(LMH+1) (:834)
    q1 = np.zeros(lmh, dtype=F)
    dth = np.empty(nzm, dtype=F)
    for m in range(nzm):
        dth[m] = F(the[m] - the[m + 1])
    for kf in range(lmh - 2, 0, -1):      # K=LMH-2..1
        if dth[kf - 1] > F(0.0) and dth[kf] <= F(0.0):
            dth[kf - 1] = F(dth[kf - 1] + ct)
            break
    ct = F(0.0)
    gm = np.empty(nzm, dtype=F)
    gh = np.empty(nzm, dtype=F)
    for m in range(nzm):
        rdz = F(F(2.0) / F(zh[m] - zh[m + 2]))
        # ((...)*RDZ)*RDZ, the Fortran's own left-to-right chain (:856).
        gml = F(F(F(F(F(u[m] - u[m + 1]) * F(u[m] - u[m + 1]))
                    + F(F(v[m] - v[m + 1]) * F(v[m] - v[m + 1])))
                  * rdz) * rdz)
        gm[m] = F(max(gml, EPSGM))
        tem = F(F(t[m] + t[m + 1]) * F(0.5))
        thm = F(F(the[m] + the[m + 1]) * F(0.5))
        a = F(thm * P608)
        b = F(F(F(F(ELOCP / tem) - F(1.0)) - P608) * thm)
        ghl = F(F(F(F(dth[m]
                      * F(F(F(F(F(q[m] + q[m + 1])
                                + cwm[m]) + cwm[m + 1])
                            * F(F(0.5) * P608)) + F(1.0)))
                    + F(F(F(F(q[m] - q[m + 1]) + cwm[m]) - cwm[m + 1]) * a))
                  + F(F(cwm[m] - cwm[m + 1]) * b)) * rdz)
        if abs(ghl) <= EPSGH:
            ghl = EPSGH
        gh[m] = ghl
    lmxl = lmh
    elm = np.empty(nzm, dtype=F)
    for m in range(nzm):
        gml = gm[m]
        ghl = gh[m]
        if ghl >= EPSGH:
            if F(gml / ghl) <= REQU:
                elm[m] = EPSL
                lmxl = m + 1
            else:
                aubr = F(F(F(AUBM * gml) + F(AUBH * ghl)) * ghl)
                bubr = F(F(BUBM * gml) + F(BUBH * ghl))
                qol2st = F(F(F(F(-0.5) * bubr)
                             + F(np.sqrt(F(F(F(bubr * bubr) * F(0.25))
                                           - F(aubr * CUBR))))) * RCUBR)
                eloq2x = F(F(1.0) / qol2st)
                elm[m] = F(max(F(np.sqrt(F(eloq2x * q2[m]))), EPSL))
        else:
            aden = F(F(F(ADNM * gml) + F(ADNH * ghl)) * ghl)
            bden = F(F(BDNM * gml) + F(BDNH * ghl))
            qol2un = F(F(F(-0.5) * bden)
                       + F(np.sqrt(F(F(F(bden * bden) * F(0.25)) - aden))))
            eloq2x = F(F(1.0) / F(qol2un + EPSRU))
            elm[m] = F(max(F(np.sqrt(F(eloq2x * q2[m]))), EPSL))
    if elm[lmh - 2] == EPSL:
        lmxl = lmh
    blmx = F(zh[lmxl - 1] - zh[lmh])
    mixht = blmx
    for m in range(lpbl - 1, lmh):
        q1[m] = F(np.sqrt(q2[m]))
    szq = F(0.0)
    sq = F(0.0)
    for m in range(nzm):
        qdzl = F(F(q1[m] + q1[m + 1]) * F(zh[m + 1] - zh[m + 2]))
        szq = F(F(F(F(F(zh[m + 1] + zh[m + 2]) - zh[lmh]) - zh[lmh])
                  * qdzl) + szq)
        sq = F(qdzl + sq)
    el0 = F(min(F(F(F(ALPH * szq) * F(0.5)) / sq), EL0MAX))
    el0 = F(max(el0, EL0MIN))
    lpblm = max(lpbl - 1, 1)
    el = np.zeros(nzm, dtype=F)
    rel = np.zeros(nzm, dtype=F)
    for m in range(lpblm):
        el[m] = F(min(F(F(zh[m] - zh[m + 2]) * ELFC), elm[m]))
        rel[m] = F(el[m] / elm[m])
    if lpbl < lmh:
        for m in range(lpbl - 1, lmh - 1):
            vkrmz = F(F(zh[m + 1] - zh[lmh]) * VKARMAN)
            el[m] = F(min(F(vkrmz / F(F(vkrmz / el0) + F(1.0))), elm[m]))
            rel[m] = F(el[m] / elm[m])
    for m in range(lpbl, lmh - 2):        # K=LPBL+1..LMH-2
        srel = F(min(F(F(F(F(rel[m - 1] + rel[m + 1]) * F(0.5)) + rel[m])
                       * F(0.5)), rel[m]))
        el[m] = F(max(F(srel * elm[m]), EPSL))
    return gm, gh, el, pblh, lpbl, lmxl, ct, mixht


def _prodq2(lmh, dtturbl, ustar, gm, gh, el, q2):
    """module_bl_myjpbl.F:967-1167 (PRODQ2), in place on el/q2."""
    for m in range(lmh - 1):
        gml = gm[m]
        ghl = gh[m]
        aequ = F(F(F(AEQM * gml) + F(AEQH * ghl)) * ghl)
        bequ = F(F(BEQM * gml) + F(BEQH * ghl))
        eqol2 = F(F(F(-0.5) * bequ)
                  + F(np.sqrt(F(F(F(bequ * bequ) * F(0.25)) - aequ))))
        if ((F(gml + F(ghl * ghl)) <= EPSTRB)
                or (ghl >= EPSGH and F(gml / ghl) <= REQU)
                or (eqol2 <= EPS2)):
            q2[m] = EPSQ2
            el[m] = EPSL
        else:
            anum = F(F(F(ANMM * gml) + F(ANMH * ghl)) * ghl)
            bnum = F(F(BNMM * gml) + F(BNMH * ghl))
            aden = F(F(F(ADNM * gml) + F(ADNH * ghl)) * ghl)
            bden = F(F(BDNM * gml) + F(BDNH * ghl))
            cden = F(1.0)
            arhs = F(F(-F(F(anum * bden) - F(bnum * aden))) * F(2.0))
            brhs = F(F(-anum) * F(4.0))
            crhs = F(F(-bnum) * F(2.0))
            dloq1 = F(el[m] / F(np.sqrt(q2[m])))
            eloq21 = F(F(1.0) / eqol2)
            eloq11 = F(np.sqrt(eloq21))
            eloq31 = F(eloq21 * eloq11)
            eloq41 = F(eloq21 * eloq21)
            eloq51 = F(eloq21 * eloq31)
            rden1 = F(F(1.0) / F(F(F(aden * eloq41) + F(bden * eloq21))
                                 + cden))
            rhsp1 = F(F(F(F(F(arhs * eloq51) + F(brhs * eloq31))
                          + F(crhs * eloq11)) * rden1) * rden1)
            eloq12 = F(eloq11 + F(F(dloq1 - eloq11)
                                  * F(np.exp(F(rhsp1 * dtturbl)))))
            eloq12 = F(max(eloq12, EPS1))
            eloq22 = F(eloq12 * eloq12)
            eloq32 = F(eloq22 * eloq12)
            eloq42 = F(eloq22 * eloq22)
            eloq52 = F(eloq22 * eloq32)
            rden2 = F(F(1.0) / F(F(F(aden * eloq42) + F(bden * eloq22))
                                 + cden))
            rhs2 = F(F(F(-F(F(anum * eloq42) + F(bnum * eloq22))) * rden2)
                     + RB1)
            rhsp2 = F(F(F(F(F(arhs * eloq52) + F(brhs * eloq32))
                          + F(crhs * eloq12)) * rden2) * rden2)
            rhst2 = F(rhs2 / rhsp2)
            eloq13 = F(F(eloq12 - rhst2)
                       + F(F(F(rhst2 + dloq1) - eloq12)
                           * F(np.exp(F(rhsp2 * dtturbl)))))
            eloq13 = F(max(eloq13, EPS1))
            eloqn = eloq13
            if eloqn > EPS1:
                q2[m] = F(F(el[m] * el[m]) / F(eloqn * eloqn))
                q2[m] = F(max(q2[m], EPSQ2))
                if q2[m] == EPSQ2:
                    el[m] = EPSL
            else:
                q2[m] = EPSQ2
                el[m] = EPSL
    # Lower boundary (:1164), left-to-right: (B1**(2./3.)*USTAR)*USTAR.
    q2[lmh - 1] = F(max(F(F(F(B1 ** F(F(2.0) / F(3.0))) * F(ustar))
                          * F(ustar)), EPSQ2))


def _difcof(lmh, gm, gh, el, q2, zh):
    """module_bl_myjpbl.F:1172-1325 (DIFCOF)."""
    akm = np.zeros(lmh - 1, dtype=F)
    akh = np.zeros(lmh - 1, dtype=F)
    for m in range(lmh - 1):
        ell = el[m]
        eloq2 = F(F(ell * ell) / q2[m])
        eloq4 = F(eloq2 * eloq2)
        gml = gm[m]
        ghl = gh[m]
        aden = F(F(F(ADNM * gml) + F(ADNH * ghl)) * ghl)
        bden = F(F(BDNM * gml) + F(BDNH * ghl))
        cden = F(1.0)
        besm = F(BSMH * ghl)
        besh = F(F(BSHM * gml) + F(BSHH * ghl))
        rden = F(F(1.0) / F(F(F(aden * eloq4) + F(bden * eloq2)) + cden))
        esm = F(F(F(besm * eloq2) + CESM) * rden)
        esh = F(F(F(besh * eloq2) + CESH) * rden)
        rdz = F(F(2.0) / F(zh[m] - zh[m + 2]))
        q1l = F(np.sqrt(q2[m]))
        elqdz = F(F(ell * q1l) * rdz)
        akm[m] = F(elqdz * esm)
        akh[m] = F(elqdz * esh)
    return akm, akh


def _vdifq(lmh, dtdif, q2, el, zh):
    """module_bl_myjpbl.F:1330-1406 (VDIFQ), in place on q2."""
    esqhf = F(F(0.5) * ESQ)
    n = lmh - 2
    dtoz = np.empty(n, dtype=F)
    akq = np.empty(n, dtype=F)
    cr = np.empty(n, dtype=F)
    cm = np.empty(n, dtype=F)
    rsq2 = np.empty(n, dtype=F)
    for m in range(n):
        dtoz[m] = F(F(dtdif + dtdif) / F(zh[m] - zh[m + 2]))
        akq[m] = F(F(F(F(np.sqrt(F(F(q2[m] + q2[m + 1]) * F(0.5))))
                      * F(el[m] + el[m + 1])) * esqhf)
                   / F(zh[m + 1] - zh[m + 2]))
        cr[m] = F(F(-dtoz[m]) * akq[m])
    cm[0] = F(F(dtoz[0] * akq[0]) + F(1.0))
    rsq2[0] = q2[0]
    for m in range(1, n):
        cf = F(F(F(-dtoz[m]) * akq[m - 1]) / cm[m - 1])
        cm[m] = F(F(F(F(-cr[m - 1]) * cf)
                    + F(F(akq[m - 1] + akq[m]) * dtoz[m])) + F(1.0))
        rsq2[m] = F(F(F(-rsq2[m - 1]) * cf) + q2[m])
    dtozs = F(F(dtdif + dtdif) / F(zh[lmh - 2] - zh[lmh]))
    akqs = F(F(F(F(np.sqrt(F(F(q2[lmh - 2] + q2[lmh - 1]) * F(0.5))))
                * F(el[lmh - 2] + ELZ0)) * esqhf)
             / F(zh[lmh - 1] - zh[lmh]))
    cf = F(F(F(-dtozs) * akq[n - 1]) / cm[n - 1])
    q2[lmh - 2] = F(F(F(F(dtozs * akqs) * q2[lmh - 1])
                      - F(rsq2[n - 1] * cf) + q2[lmh - 2])
                    / F(F(F(F(akq[n - 1] + akqs) * dtozs)
                          - F(cr[n - 1] * cf)) + F(1.0)))
    for m in range(n - 1, -1, -1):
        q2[m] = F(F(F(F(-cr[m]) * q2[m + 1]) + rsq2[m]) / cm[m])


def _vdifh(dtdif, lmh, lpbl, sz0, rkhs, clow, cts, species, nspec,
           rkh, zh, rho):
    """module_bl_myjpbl.F:1411-1519 (VDIFH), in place on species.

    ``species`` is ``(nspec, lmh)``; ``sz0/clow/cts`` are ``(nspec,)``.
    """
    n = lmh - 1
    dtoz = np.empty(n, dtype=F)
    cr = np.empty(n, dtype=F)
    cm = np.empty(n, dtype=F)
    rkct = np.zeros((nspec, n), dtype=F)
    rss = np.empty((nspec, n), dtype=F)
    for m in range(n):
        dtoz[m] = F(dtdif / F(zh[m] - zh[m + 1]))
        cr[m] = F(F(-dtoz[m]) * rkh[m])
        if (m + 1) < lpbl:
            for s in range(nspec):
                rkct[s, m] = F(0.0)
        else:
            rkhz = F(rkh[m] * F(zh[m] - zh[m + 2]))
            for s in range(nspec):
                rkct[s, m] = F(F(rkhz * cts[s]) * F(0.5))
    rhok = rho[0]
    cm[0] = F(F(dtoz[0] * rkh[0]) + rhok)
    for s in range(nspec):
        rss[s, 0] = F(F(F(-rkct[s, 0]) * dtoz[0])
                      + F(species[s, 0] * rhok))
    for m in range(1, n):
        dtozl = dtoz[m]
        cf = F(F(F(-dtozl) * rkh[m - 1]) / cm[m - 1])
        rhok = rho[m]
        cm[m] = F(F(F(F(-cr[m - 1]) * cf)
                    + F(F(rkh[m - 1] + rkh[m]) * dtozl)) + rhok)
        for s in range(nspec):
            rss[s, m] = F(F(F(F(-rss[s, m - 1]) * cf)
                            + F(F(rkct[s, m - 1] - rkct[s, m]) * dtozl))
                          + F(species[s, m] * rhok))
    dtozs = F(dtdif / F(zh[lmh - 1] - zh[lmh]))
    rkhh = rkh[n - 1]
    cf = F(F(F(-dtozs) * rkhh) / cm[n - 1])
    cmb = F(cr[n - 1] * cf)
    rhok = rho[lmh - 1]
    for s in range(nspec):
        rkss = F(rkhs * clow[s])
        cmsb = F(F(F(-cmb) + F(F(rkhh + rkss) * dtozs)) + rhok)
        rssb = F(F(F(F(-rss[s, n - 1]) * cf) + F(rkct[s, n - 1] * dtozs))
                 + F(species[s, lmh - 1] * rhok))
        species[s, lmh - 1] = F(F(F(F(dtozs * rkss) * sz0[s]) + rssb)
                                / cmsb)
    for m in range(n - 1, -1, -1):
        rcml = F(F(1.0) / cm[m])
        for s in range(nspec):
            species[s, m] = F(F(F(F(-cr[m]) * species[s, m + 1])
                                + rss[s, m]) * rcml)


def _vdifv(lmh, dtdif, uz0, vz0, rkms, u, v, rkm, zh, rho):
    """module_bl_myjpbl.F:1613-1689 (VDIFV), in place on u/v."""
    n = lmh - 1
    dtoz = np.empty(n, dtype=F)
    cr = np.empty(n, dtype=F)
    cm = np.empty(n, dtype=F)
    rsu = np.empty(n, dtype=F)
    rsv = np.empty(n, dtype=F)
    for m in range(n):
        dtoz[m] = F(dtdif / F(zh[m] - zh[m + 1]))
        cr[m] = F(F(-dtoz[m]) * rkm[m])
    rhok = rho[0]
    cm[0] = F(F(dtoz[0] * rkm[0]) + rhok)
    rsu[0] = F(u[0] * rhok)
    rsv[0] = F(v[0] * rhok)
    for m in range(1, n):
        dtozl = dtoz[m]
        cf = F(F(F(-dtozl) * rkm[m - 1]) / cm[m - 1])
        rhok = rho[m]
        cm[m] = F(F(F(F(-cr[m - 1]) * cf)
                    + F(F(rkm[m - 1] + rkm[m]) * dtozl)) + rhok)
        rsu[m] = F(F(F(-rsu[m - 1]) * cf) + F(u[m] * rhok))
        rsv[m] = F(F(F(-rsv[m - 1]) * cf) + F(v[m] * rhok))
    dtozs = F(dtdif / F(zh[lmh - 1] - zh[lmh]))
    rkmh = rkm[n - 1]
    cf = F(F(F(-dtozs) * rkmh) / cm[n - 1])
    rhok = rho[lmh - 1]
    rcmvb = F(F(1.0) / F(F(F(F(rkmh + rkms) * dtozs)
                          - F(cr[n - 1] * cf)) + rhok))
    dtozak = F(dtozs * rkms)
    u[lmh - 1] = F(F(F(F(dtozak * uz0) - F(rsu[n - 1] * cf))
                     + F(u[lmh - 1] * rhok)) * rcmvb)
    v[lmh - 1] = F(F(F(F(dtozak * vz0) - F(rsv[n - 1] * cf))
                     + F(v[lmh - 1] * rhok)) * rcmvb)
    for m in range(n - 1, -1, -1):
        rcml = F(F(1.0) / cm[m])
        u[m] = F(F(F(F(-cr[m]) * u[m + 1]) + rsu[m]) * rcml)
        v[m] = F(F(F(F(-cr[m]) * v[m + 1]) + rsv[m]) * rcml)


def np_myjpbl_column(
        *, dz, u, v, t, th, exner, qv, qc, qi, p, psfc, tke,
        ust, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0,
        xland, sice, snow, akhs, akms, elflx, ct, dtturbl, ht=F(0.0)):
    """One MYJPBL column (module_bl_myjpbl.F:131-766).

    All column inputs are bottom-up float32 ``(nz,)`` arrays (WRF-up), the
    scheme flips internally.  ``qv`` is a mixing ratio; ``qi=None`` runs
    the no-ice arm (``flQI=.FALSE.``, NSPEC=3).  ``tke`` is TKE_MYJ;
    ``elflx`` is the latent heat flux (WRF binds ``ELFLX=lh``,
    module_pbl_driver.F:1457).  Returns bottom-up tendencies plus updated
    state, mirroring the WRF INOUT set.

    ``ht`` is the interface-height origin (module_bl_myjpbl.F:312-320).  It
    defaults to gpuwm's DECLARED ground-relative 0 rather than WRF's
    terrain height; :func:`_interface_heights` carries the declaration and
    names the test that measures it.
    """
    dz = np.asarray(dz, dtype=F)
    nz = dz.shape[0]
    lmh = nz
    flqi = qi is not None
    nspec = 4 if flqi else 3
    rdtturbl = F(F(1.0) / F(dtturbl))
    dtdif = F(dtturbl)

    def flip(a):
        return np.asarray(a, dtype=F)[::-1].copy()

    zh = _interface_heights(dz, ht)
    uk = flip(u)
    vk = flip(v)
    tk = flip(t)
    thk = flip(th)
    qvk_mix = flip(qv)
    qcwk = flip(qc)
    qcik = flip(qi) if flqi else None
    pk = flip(p)
    exnerk = flip(exner)
    cwm = qcwk.copy()
    if flqi:
        cwm = np.array([F(cwm[m] + qcik[m]) for m in range(nz)], dtype=F)
    the = np.array(
        [F(F(F(cwm[m] * F(-ELOCP / tk[m])) + F(1.0)) * thk[m])
         for m in range(nz)], dtype=F)
    qk = np.array([F(qvk_mix[m] / F(F(1.0) + qvk_mix[m]))
                   for m in range(nz)], dtype=F)
    q2 = np.array([F(F(2.0) * F(tke[nz - 1 - m])) for m in range(nz)],
                  dtype=F)

    # ---- setup_integration (:344-469) ----
    ct_l = F(ct)
    gm, gh, el, pblh, lpbl, lmxl, ct_l, mixht = _mixlen(
        lmh, uk, vk, tk, the, qk, cwm, q2, zh, ct_l)
    _prodq2(lmh, F(dtturbl), F(ust), gm, gh, el, q2)
    kpbl = nz - lpbl + 1                  # KPBL=KTE-LPBL+1 (:421)
    # DIFCOF's T argument feeds only its commented-out inversion block
    # (module_bl_myjpbl.F:1275-1322), so the transcription drops it.
    akm, akh = _difcof(lmh, gm, gh, el, q2, zh)
    # EXCH_H publication (:438-444), WRF-up index.
    exch_h = np.zeros(nz, dtype=F)
    for k_up in range(nz - 1):
        kflip = nz - 1 - k_up             # Fortran KFLIP=KTE-K
        deltaz = F(F(0.5) * F(zh[kflip - 1] - zh[kflip + 1]))
        exch_h[k_up] = F(akh[kflip - 1] * deltaz)
    _vdifq(lmh, dtdif, q2, el, zh)
    tke_new = np.empty(nz, dtype=F)
    el_out = np.zeros(nz, dtype=F)        # EL_MYJ, WRF-up, level KTE stays 0
    for k_up in range(nz):
        kflip = nz - k_up                 # Fortran KFLIP=KTE+1-K (1-based)
        q2[kflip - 1] = F(max(q2[kflip - 1], EPSQ2))
        tke_new[k_up] = F(F(0.5) * q2[kflip - 1])
        if kflip < nz:
            el_out[k_up] = el[kflip - 1]
    # ---- surface theta (:474-479) ----
    thsk = F(F(tsk) * F(F(F(1.0e5) / F(psfc)) ** CAPA))
    # ---- main_integration, heat/moisture (:493-652) ----
    rhok = np.array(
        [F(pk[m] / F(F(R_D * tk[m])
                     * F(F(F(1.0) + F(P608 * qk[m])) - cwm[m])))
         for m in range(nz)], dtype=F)
    akhk = np.array(
        [F(F(akh[m] * F(0.5)) * F(rhok[m] + rhok[m + 1]))
         for m in range(nz - 1)], dtype=F)
    seamask = F(F(xland) - F(1.0))
    thz0_new = F(F(F(F(1.0) - seamask) * thsk) + F(seamask * F(thz0)))
    akhs_dens = F(F(akhs) * rhok[nz - 1])
    qsfc_new = F(qsfc)
    if seamask < F(0.5):
        qfc1 = F(F(XLV * F(chklowq)) * akhs_dens)
        if F(snow) > F(0.0) or F(sice) > F(0.5):
            qfc1 = F(qfc1 * RLIVWV)
        if qfc1 > F(0.0):
            qlow = qk[nz - 1]
            qsfc_new = F(qlow + F(F(elflx) / qfc1))
    else:
        exnsfc = F(F(F(1.0e5) / F(psfc)) ** CAPA)
        qsfc_new = F(F(PQ0SEA / F(psfc))
                     * F(np.exp(F(F(A2C * F(thsk - F(A3C * exnsfc)))
                                  / F(thsk - F(A4C * exnsfc))))))
    qz0_new = F(F(F(F(1.0) - seamask) * qsfc_new) + F(seamask * F(qz0)))
    # Species stack (:569-585).
    sz0 = np.zeros(nspec, dtype=F)
    sz0[0] = thz0_new
    sz0[1] = qz0_new
    clow = np.zeros(nspec, dtype=F)
    clow[0] = F(1.0)
    clow[1] = F(chklowq)
    cts = np.zeros(nspec, dtype=F)
    cts[0] = ct_l                          # zero in ARW (MIXLEN zeroed it)
    species = np.empty((nspec, nz), dtype=F)
    species[0] = the
    species[1] = qk
    species[2] = qcwk
    if flqi:
        species[3] = qcik
    _vdifh(dtdif, lmh, lpbl, sz0, akhs_dens, clow, cts, species, nspec,
           akhk, zh, rhok)
    the_new = species[0]
    qk_new = species[1]
    qcwk_new = species[2]
    cwm_new = qcwk_new.copy()
    if flqi:
        qcik_new = species[3]
        cwm_new = np.array([F(cwm_new[m] + qcik_new[m])
                            for m in range(nz)], dtype=F)
    rthblten = np.empty(nz, dtype=F)
    rqvblten = np.empty(nz, dtype=F)
    rqcblten = np.empty(nz, dtype=F)
    rqiblten = np.zeros(nz, dtype=F)
    for k_up in range(nz):
        m = nz - 1 - k_up
        thold = F(th[k_up])
        ape = F(F(1.0) / F(exner[k_up]))
        thnew = F(the_new[m] + F(F(cwm_new[m] * ELOCP) * ape))
        rthblten[k_up] = F(F(thnew - thold) * rdtturbl)
        qold = F(F(qv[k_up]) / F(F(1.0) + F(qv[k_up])))
        dqdt = F(F(qk_new[m] - qold) * rdtturbl)
        rqvblten[k_up] = F(dqdt / F(F(F(1.0) - qk_new[m])
                                    * F(F(1.0) - qk_new[m])))
        rqcblten[k_up] = F(F(qcwk_new[m] - F(qc[k_up])) * rdtturbl)
        if flqi:
            rqiblten[k_up] = F(F(qcik_new[m] - F(qi[k_up])) * rdtturbl)
    # ---- main_integration, momentum (:714-759) ----
    akmk = np.array(
        [F(F(akm[m] * F(rhok[m] + rhok[m + 1])) * F(0.5))
         for m in range(nz - 1)], dtype=F)
    akms_dens = F(F(akms) * rhok[nz - 1])
    uk_new = uk.copy()
    vk_new = vk.copy()
    _vdifv(lmh, dtdif, F(uz0), F(vz0), akms_dens, uk_new, vk_new,
           akmk, zh, rhok)
    rublten = np.empty(nz, dtype=F)
    rvblten = np.empty(nz, dtype=F)
    for k_up in range(nz):
        m = nz - 1 - k_up
        rublten[k_up] = F(F(uk_new[m] - F(u[k_up])) * rdtturbl)
        rvblten[k_up] = F(F(vk_new[m] - F(v[k_up])) * rdtturbl)
    return {
        "rublten": rublten, "rvblten": rvblten, "rthblten": rthblten,
        "rqvblten": rqvblten, "rqcblten": rqcblten, "rqiblten": rqiblten,
        "tke": tke_new, "el": el_out, "exch_h": exch_h,
        "pblh": pblh, "kpbl": np.int32(kpbl), "mixht": mixht,
        "thz0": thz0_new, "qz0": qz0_new, "qsfc": qsfc_new, "ct": ct_l,
    }
