"""Probe transform for WRF v4.8.0 ``phys/module_mp_fast_sbm.F``.

Takes the pinned pristine source, applies the one-line access transform
(``transform_access.py``) and then inserts CALL lines to the external probe
subroutines in ``fsbm_probe.F90`` at named anchors inside ``FAST_SBM`` and
``COAL_BOTT_NEW``.  Nothing else is edited: the probe routines are called
through an implicit interface, so no USE line is needed, and they never write
to their arguments.  The script refuses unless every anchor's key line is at
its pinned line number with its pinned content (and, where the key text
repeats, the pinned occurrence with its pinned neighbours), and unless the
diff between the access-transformed text and the output is exactly the
inserted CALL lines.

Anchors (pristine line numbers; the access transform shifts lines after 3862
up by one, which the script accounts for):

========================  =====  =================================================
name                      line   where
========================  =====  =================================================
entry                     4370   after the t_new / t_old loop (entry conversion done)
forcing                   4633   after each substep's forcing update (DEL2IN)
nucl_before               4656   before JERNUCL01_KS
nucl_after                4664   after JERNUCL01_KS
cond1_before              4715   before ONECOND1
cond1_after               4722   after ONECOND1
coal_before               4774   before COAL_BOTT_NEW
coal_after                4779   after COAL_BOTT_NEW
spont_before              4829   before Spont_Rain_BreakUp
spont_after               4829   after Spont_Rain_BreakUp
fall_drop_before/after    4907   around FALFLUXHUCM_Z for drops
fall_snow_before/after    4928   around FALFLUXHUCM_Z for snow
fall_dense_before/after   4952   around FALFLUXHUCM_Z for hail or graupel
diag                      5265   after the diagnostics and surface precipitation loop
xxx_before                8375   before coll_xxx_lwf (drop-drop) in COAL_BOTT_NEW
xxx_after                 8375   after coll_xxx_lwf
breakup                   8436   after label 11: breakup finished, with its final ndiv
========================  =====  =================================================

Cold anchors (milestone 2). Three sit outside FAST_SBM and COAL_BOTT_NEW:
inside JERNUCL01_KS (ICE_NUCL is private to its module, so these are its
only observation) and inside ONECOND2 and ONECOND3 at their JERSUPSAT_KS
calls.

========================  =====  =================================================
name                      line   where
========================  =====  =================================================
homfrz                    3108   before the homogeneous freezing heat (JERNUCL01_KS)
icenucl_before/after      3126   around ICE_NUCL (JERNUCL01_KS)
prepass                   4512   after the ice pre-pass (DSUPICE_XYZ, cloud base)
nucl_ice                  4664   after JERNUCL01_KS, before crystals move to snow
dispatch                  4711   before the ONECOND1/2/3 regime dispatch
cond2_before/after        4730   around ONECOND2
cond3_before/after        4748   around ONECOND3
coal_ice_before/after     4774   around COAL_BOTT_NEW (ice spectra and fractions)
freez_before/after        4795   around FREEZ
melt_before/after         4805   around J_W_MELT
merge                     4826   after the merges, before spontaneous breakup
snowbrk                   4836   before the snow-breakup guard
exit                      5276   before the exit conversion and its clips
supsat_c2_21              7460   before JERSUPSAT_KS in ONECOND2 (KCOND 21)
supsat_c2_20              7543   before JERSUPSAT_KS in ONECOND2 (KCOND 20)
supsat_c3                 8057   before JERSUPSAT_KS in ONECOND3
coal_cold_before          8440   before the cold block of COAL_BOTT_NEW
cc01..cc12                8452+  before each collision call of the cold block
hm                        8553   after the Hallett-Mossop splinters
coal_cold_after           8568   after the cold block
rime                      8604   before label 61, after the riming heat
========================  =====  =================================================

Usage::

    python3 instrument_probes.py --in module_mp_fast_sbm.F \\
        --out module_mp_fast_sbm_probe.F --diff transform_probe.diff \\
        --anchors probe_anchors.json

Standard library only.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transform_access import (  # noqa: E402
    DELETED_LINE_NUMBER,
    TransformRefusal,
    access_transform,
)

CALL_PATTERN = re.compile(r"^\s*CALL fsbm_probe_[a-z0-9_]+\(.*\)$")


@dataclass(frozen=True)
class Anchor:
    name: str
    line: int                 # pristine line number of the key line
    key: str                  # exact content of the key line, stripped
    where: str                # "before" or "after"
    call: str                 # the inserted statement, without indentation
    offset: int = 0           # for "after": insert after line + offset
    context: tuple = field(default_factory=tuple)  # ((offset, stripped content), ...)
    occurrence: int = 0       # 0: key must be unique; n: n-th occurrence


FALL_KEY = "call FALFLUXHUCM_Z(ffx_z,VRX,RHOCGS_z,PCGS_z,ZCGS_z,DT,kts,kte,nkr)"
FALL_ARGS = "itimestep,i,j,ffx_z,VRX,rhocgs_z,pcgs_z,zcgs_z,kts,kte)"
NUCL_ARGS = ("itimestep,I,K,J,FF1IN,FCCN,TT,QQ,DEL1IN,DEL2IN,Is_This_CloudBase,"
             "KZ_Cloud_Base(I,J),W_Stag_My,rhocgs(I,K,J),pcgs(I,K,J))")
COND1_ARGS = ("itimestep,I,K,J,FF1R,TT,QQ,DEL1IN,DEL2IN,DIV1,DIV2,DTCOND,"
              "VR1_Z(:,K),pcgs(I,K,J),rhocgs(I,K,J))")
COAL_ARGS = "itimestep,I,K,J,FF1R,TT,QQ,DEL1IN,DEL2IN,PP,rhocgs(I,K,J),dt_coll)"
XXX_ARGS = "itimestep,Iin,Kin,Jin,G1,CWLL,XL_MG,PP)"
SPONT_KEY = ("CALL Spont_Rain_BreakUp (DT ,FF1R_D, XL_D, Prob, Gain_Var_New, "
             "NND, NKR, ikr_spon_break)")

# ---------------------------------------------------------------- cold anchors
JERSUPSAT_KEY = "CALL JERSUPSAT_KS(DEL1_d,DEL2_d,DEL1N,DEL2N, &"
SUPSAT_ARGS = ("Itimestep,Iin,Kin,Jin,KCOND,DEL1_d,DEL2_d,RW_d,PW_d,RI_d,PI_d,DTT,"
               "ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)")
COND2_ARGS = ("itimestep,I,K,J,FF2R,FF3R,FF4R,FF5R,TT,QQ,DEL1IN,DEL2IN,DIV1,DIV2,DTCOND,"
              "pcgs(I,K,J),rhocgs(I,K,J),ISYM2,ISYM3,ISYM4,ISYM5)")
COND3_ARGS = ("itimestep,I,K,J,FF1R,FF2R,FF3R,FF4R,FF5R,TT,QQ,DEL1IN,DEL2IN,DIV1,DIV2,"
              "DTCOND,pcgs(I,K,J),rhocgs(I,K,J),ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)")
COND_END_KEY = ",ISYM1,ISYM2,ISYM3,ISYM4,ISYM5,I,J,K,W(i,k,j),DX,Itimestep)"
COALICE_ARGS = "itimestep,I,K,J,FF3R,FF4R,FF5R,FLIQFR_SD,FLIQFR_GD,FLIQFR_HD,FRIMFR_SD)"
PHASE_ARGS = "itimestep,I,K,J,FF1R,FF2R,FF3R,FF4R,FF5R,T_NEW(I,K,J),rhocgs(I,K,J))"
COLD_ARGS = ("itimestep,Iin,Kin,Jin,g1,g3,g4,g5,rf3,tt,alwc,prdkrn,icol_drop,icol_snow,"
             "icol_graupel,icol_hail,PP,cwls,cwsl,cwlh,cwhl,cwss,cwlg,cwgl)")
ICENUCL_KEY = "CALL ICE_NUCL (PSI2,xi,SUP2,TT,DSUPICE_XYZ,SUP2_OLD,ICEMAX,NKR,COL)"
ICENUCL_ARGS = "Iin,Kin,Jin,PSI2,SUP2,TT,DSUPICE_XYZ,SUP2_OLD)"
# The twelve collision calls of the cold block, in source order (probe ids 1-12).
CC_SITES = (
    (8452, "call coll_xyz_lwf(g1,g3,g5,rf1,rf3,rf5,cwls,xl_mg,xs_mg, &"),
    (8455, "call coll_xyz_lwf(g1,g3,g4,rf1,rf3,rf4,cwls,xl_mg,xs_mg, &"),
    (8462, "call coll_xyx_lwf(g3,g1,rf3,rf1,cwsl,xs_mg,xl_mg, &"),
    (8466, "call coll_xyz_lwf(g3,g1,g5,rf3,rf1,rf5,cwsl,xs_mg,xl_mg, &"),
    (8469, "call coll_xyz_lwf(g3,g1,g4,rf3,rf1,rf4,cwsl,xs_mg,xl_mg, &"),
    (8485, "call coll_xyy_lwf(g1,g4,rf1,rf4,cwlg,xl_mg,xg_mg, &"),
    (8495, "call coll_xyx_lwf(g4,g1,rf4,rf1,cwgl,xg_mg,xl_mg, &"),
    (8501, "call coll_xyz_lwf(g1,g4,g5,rf1,rf4,rf5,cwlg,xl_mg,xg_mg, &"),
    (8512, "call coll_xyz_lwf(g4,g1,g5,rf4,rf1,rf5,cwgl,xg_mg,xl_mg, &"),
    (8524, "call coll_xyy_lwf(g1,g5,rf1,rf5,cwlh,xl_mg,xh_mg, &"),
    (8534, "call coll_xyx_lwf(g5,g1,rf5,rf1,cwhl,xh_mg,xl_mg, &"),
    (8563, "call coll_xxx_lwf(g3,rf3,cwss,xs_mg,chucm,ima,prdkrn,nkr)"),
)

COLD_ANCHORS: tuple[Anchor, ...] = (
    Anchor("homfrz", 3108, "ARG_1 = 334.0D0*SUM_ICE*RORI", "before",
           "CALL fsbm_probe_homfrz(Iin,Kin,Jin,TT,SUM_ICE,RORI)",
           context=((-7, "IF(TPC <= -38.0D0) THEN"),)),
    Anchor("icenucl_before", 3126, ICENUCL_KEY, "before",
           "CALL fsbm_probe_icenucl(0," + ICENUCL_ARGS),
    Anchor("icenucl_after", 3126, ICENUCL_KEY, "after",
           "CALL fsbm_probe_icenucl(1," + ICENUCL_ARGS),
    Anchor("prepass", 4512, "ENDIF", "after",
           "CALL fsbm_probe_prepass(itimestep,DSUPICE_XYZ,DTFREEZ_XYZ,KZ_Cloud_Base,t_old,"
           "qv_old,pcgs,I_START,I_END,J_START,J_END,ims,ime,kms,kme,jms,jme,its,ite,kts,"
           "kte,jts,jte)",
           occurrence=75,
           context=((-1, "END DO"), (-2, "END DO"), (-3, "END DO"), (2, "do j = jts,jte"))),
    Anchor("nucl_ice", 4664, ",I,J,K)", "after",
           "CALL fsbm_probe_nuclice(itimestep,I,K,J,FF2IN,TT,QQ,DEL1IN,DEL2IN,SUP2_OLD,"
           "DSUPICE_XYZ(I,K,J))",
           context=((-8, "CALL JERNUCL01_KS(FF1IN,FF2IN,FCCN \t\t  &"),)),
    Anchor("dispatch", 4711,
           "IF(ISYM1==1 .AND. ((TT-273.15)>-0.187 .OR.(sum(ISYM2)==0 .AND. &", "before",
           "CALL fsbm_probe_dispatch(itimestep,I,K,J,TT,ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)"),
    Anchor("cond2_before", 4730, "CALL ONECOND2(TT,QQ,PP,rhocgs(I,K,J) &", "before",
           "CALL fsbm_probe_cond2(0," + COND2_ARGS),
    Anchor("cond2_after", 4740, COND_END_KEY, "after",
           "CALL fsbm_probe_cond2(1," + COND2_ARGS, occurrence=1,
           context=((-10, "CALL ONECOND2(TT,QQ,PP,rhocgs(I,K,J) &"),)),
    Anchor("cond3_before", 4748, "CALL ONECOND3(TT,QQ,PP,rhocgs(I,K,J) &", "before",
           "CALL fsbm_probe_cond3(0," + COND3_ARGS),
    Anchor("cond3_after", 4759, COND_END_KEY, "after",
           "CALL fsbm_probe_cond3(1," + COND3_ARGS, occurrence=2,
           context=((-11, "CALL ONECOND3(TT,QQ,PP,rhocgs(I,K,J) &"),)),
    Anchor("coal_ice_before", 4774,
           "CALL COAL_BOTT_NEW (FF1R,FF2R,FF3R,     \t\t\t\t    &", "before",
           "CALL fsbm_probe_coalice(0," + COALICE_ARGS),
    Anchor("coal_ice_after", 4779, "I,J,K,Itimestep,CollEff_out)", "after",
           "CALL fsbm_probe_coalice(1," + COALICE_ARGS),
    Anchor("freez_before", 4795, "CALL FREEZ &", "before",
           "CALL fsbm_probe_phase(1,0," + PHASE_ARGS),
    Anchor("freez_after", 4799, "KRFREEZ,ICEMAX,NKR)", "after",
           "CALL fsbm_probe_phase(1,1," + PHASE_ARGS,
           context=((-4, "CALL FREEZ &"),)),
    Anchor("melt_before", 4805,
           "CALL J_W_MELT(FF1R,XL,FF2R,XI,FF3R,XS,FF4R,XG,FF5R,XH, &", "before",
           "CALL fsbm_probe_phase(2,0," + PHASE_ARGS),
    Anchor("melt_after", 4806, "T_NEW(I,K,J),DT,rhocgs(I,K,J),COL,ICEMAX,NKR)", "after",
           "CALL fsbm_probe_phase(2,1," + PHASE_ARGS),
    Anchor("merge", 4826,
           "IF (Spont_Rain_BreakUp_On == 1 .AND. (SUM(FF1R) > 43.0*1.0D-30) )THEN", "before",
           "CALL fsbm_probe_merge(itimestep,I,K,J,FF1R,FF2R,FF3R,FF4R,FF5R)"),
    Anchor("snowbrk", 4836,
           "IF (Snow_BreakUp_On == 1 .and. ICEPROCS == 1 .and. sum(FF3R(KR_SNOW_MIN:NKR))"
           "> (NKR-KR_SNOW_MIN)*1.0D-30)THEN", "before",
           "CALL fsbm_probe_snowbrk(itimestep,I,K,J,FF3R,KR_SNOW_MIN,KR_SNOW_MAX)"),
    Anchor("exit", 5276, "if (conserv)then", "before",
           "CALL fsbm_probe_exit(itimestep,chem_new,n_chem,qi,qs,qg,qni,qns,qng,SNOWNC,"
           "SNOWNCV,GRAUPELNC,GRAUPELNCV,ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte)",
           occurrence=2, context=((1, "DO j = jts,jte"), (-2, "end do"))),
    Anchor("supsat_c2_21", 7460, JERSUPSAT_KEY, "before",
           "CALL fsbm_probe_supsat(21," + SUPSAT_ARGS, occurrence=3,
           context=((-6, "DEL1_d = DEL1"),)),
    Anchor("supsat_c2_20", 7543, JERSUPSAT_KEY, "before",
           "CALL fsbm_probe_supsat(20," + SUPSAT_ARGS, occurrence=4,
           context=((-6, "DEL1_d = DEL1"),)),
    Anchor("supsat_c3", 8057, JERSUPSAT_KEY, "before",
           "CALL fsbm_probe_supsat(3," + SUPSAT_ARGS, occurrence=5,
           context=((-6, "DEL1_d = DEL1"),)),
    Anchor("coal_cold_before", 8440, "if(tt <= 273.15)then", "before",
           "CALL fsbm_probe_coldcoal(0," + COLD_ARGS),
) + tuple(
    Anchor(f"cc{n:02d}", line, key, "before",
           f"CALL fsbm_probe_cc({n},itimestep,Iin,Kin,Jin)")
    for n, (line, key) in enumerate(CC_SITES, start=1)
) + (
    Anchor("hm", 8553,
           "g3(1)=g3(1)+conc_icempl*xs_mg(1)/col ! [KSS] >> FAST-sbm has small snow as IC",
           "after",
           "CALL fsbm_probe_hm(itimestep,Iin,Kin,Jin,tt,conc_old,conc_new,dconc,conc_icempl)"),
    Anchor("coal_cold_after", 8568, "endif ! if tt <= 273.15", "after",
           "CALL fsbm_probe_coldcoal(1," + COLD_ARGS),
    Anchor("rime", 8604, "61   continue", "before",
           "CALL fsbm_probe_rime(itimestep,Iin,Kin,Jin,tt,t_new,deldrop,cont_init_drop,"
           "cont_fin_drop,cont_init_ice,cont_fin_ice,DEL1in,DEL2in)"),
)

ANCHORS: tuple[Anchor, ...] = (
    Anchor("entry", 4367, "t_old(i,k,j) = th_old(i,k,j)*pi_phy(i,k,j)", "after",
           "CALL fsbm_probe_entry(itimestep,chem_new,n_chem,t_new,t_old,qv,qv_old,"
           "pcgs,rhocgs,zcgs,ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte)",
           offset=3, context=((1, "END DO"), (2, "END DO"), (3, "END DO"))),
    Anchor("forcing", 4633, "DEL2IN = EW1N/ES2N-1.0", "after",
           "CALL fsbm_probe_forcing(itimestep,I,K,J,IKL,TT,QQ,DEL1IN,DEL2IN,DIV1,DIV2,"
           "DTCOND,DEL_T,DEL_Q)",
           context=((-8, "TT = TT + DEL_T"),)),
    Anchor("nucl_before", 4656, "CALL JERNUCL01_KS(FF1IN,FF2IN,FCCN \t\t  &", "before",
           "CALL fsbm_probe_nucl(0," + NUCL_ARGS),
    Anchor("nucl_after", 4664, ",I,J,K)", "after",
           "CALL fsbm_probe_nucl(1," + NUCL_ARGS,
           context=((-8, "CALL JERNUCL01_KS(FF1IN,FF2IN,FCCN \t\t  &"),)),
    Anchor("cond1_before", 4715, "CALL ONECOND1(TT,QQ,PP,rhocgs(I,K,J) &", "before",
           "CALL fsbm_probe_cond1(0," + COND1_ARGS),
    Anchor("cond1_after", 4722,
           ",ISYM2,ISYM3,ISYM4,ISYM5,I,J,K,W(i,k,j),DX,Itimestep)", "after",
           "CALL fsbm_probe_cond1(1," + COND1_ARGS,
           context=((-7, "CALL ONECOND1(TT,QQ,PP,rhocgs(I,K,J) &"),)),
    Anchor("coal_before", 4774,
           "CALL COAL_BOTT_NEW (FF1R,FF2R,FF3R,     \t\t\t\t    &", "before",
           "CALL fsbm_probe_coal(0," + COAL_ARGS),
    Anchor("coal_after", 4779, "I,J,K,Itimestep,CollEff_out)", "after",
           "CALL fsbm_probe_coal(1," + COAL_ARGS),
    Anchor("spont_before", 4829, SPONT_KEY, "before",
           "CALL fsbm_probe_spont(0,itimestep,I,K,J,FF1R_D,ikr_spon_break)"),
    Anchor("spont_after", 4829, SPONT_KEY, "after",
           "CALL fsbm_probe_spont(1,itimestep,I,K,J,FF1R_D,ikr_spon_break)",
           context=((1, "FF1R(:) = FF1R_D(:)"),)),
    Anchor("fall_drop_before", 4907, FALL_KEY, "before",
           "CALL fsbm_probe_fall(0,1," + FALL_ARGS, occurrence=1,
           context=((-7, "vrx(k,:)=vr1_z3D(:,i,k,j)"),)),
    Anchor("fall_drop_after", 4907, FALL_KEY, "after",
           "CALL fsbm_probe_fall(1,1," + FALL_ARGS, occurrence=1),
    Anchor("fall_snow_before", 4928, FALL_KEY, "before",
           "CALL fsbm_probe_fall(0,2," + FALL_ARGS, occurrence=2,
           context=((-7, "vrx(k,:)=vr3_z3D(:,i,k,j)"),)),
    Anchor("fall_snow_after", 4928, FALL_KEY, "after",
           "CALL fsbm_probe_fall(1,2," + FALL_ARGS, occurrence=2),
    Anchor("fall_dense_before", 4952, FALL_KEY, "before",
           "CALL fsbm_probe_fall(0,3," + FALL_ARGS, occurrence=3,
           context=((-8, "vrx(k,:) = vr4_z3D(:,i,k,j)"),)),
    Anchor("fall_dense_after", 4952, FALL_KEY, "after",
           "CALL fsbm_probe_fall(1,3," + FALL_ARGS, occurrence=3),
    Anchor("diag", 5264, "! cycle by J", "after",
           "CALL fsbm_probe_diag(itimestep,qc,qr,qnc,qnr,qna,qv,th_phy,RAINNC,RAINNCV,"
           "VR1_Z,ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte)",
           offset=1, context=((1, "END DO"), (-1, "END DO"))),
    Anchor("xxx_before", 8375,
           "call coll_xxx_lwf (G1,fl1,CWLL,XL_MG,CHUCM,IMA,1.d0,NKR)", "before",
           "CALL fsbm_probe_xxx(0," + XXX_ARGS),
    Anchor("xxx_after", 8375,
           "call coll_xxx_lwf (G1,fl1,CWLL,XL_MG,CHUCM,IMA,1.d0,NKR)", "after",
           "CALL fsbm_probe_xxx(1," + XXX_ARGS),
    Anchor("breakup", 8436, "11   continue", "after",
           "CALL fsbm_probe_brk(itimestep,Iin,Kin,Jin,icol_drop,icol_drop_brk,ndiv,"
           "G1,FF1R,break_drop_per)",
           context=((-1, ""), (-2, "end if"))),
) + COLD_ANCHORS


def _access_number(pristine_line: int) -> int:
    return pristine_line - 1 if pristine_line > DELETED_LINE_NUMBER else pristine_line


def probe_transform(pristine: bytes) -> tuple[bytes, bytes, str, dict]:
    """Return (access bytes, probe bytes, probe diff text, summary)."""

    access, _, access_summary = access_transform(pristine)
    lines = access.decode("ascii").split("\n")
    stripped = [ln.strip() for ln in lines]
    inserts: dict[int, list[tuple[str, str]]] = {}  # 0-based index -> [(where, text)]
    records = []
    for anchor in ANCHORS:
        n = _access_number(anchor.line)
        idx = n - 1
        if stripped[idx] != anchor.key:
            raise TransformRefusal(
                f"anchor {anchor.name}: line {anchor.line} is {stripped[idx]!r}, "
                f"expected {anchor.key!r}")
        hits = [i for i, s in enumerate(stripped) if s == anchor.key]
        if anchor.occurrence == 0:
            if hits != [idx]:
                raise TransformRefusal(
                    f"anchor {anchor.name}: key is not unique ({len(hits)} hits)")
        elif len(hits) < anchor.occurrence or hits[anchor.occurrence - 1] != idx:
            raise TransformRefusal(
                f"anchor {anchor.name}: key is not occurrence {anchor.occurrence}")
        for off, content in anchor.context:
            if stripped[idx + off] != content:
                raise TransformRefusal(
                    f"anchor {anchor.name}: context line {anchor.line + off} is "
                    f"{stripped[idx + off]!r}, expected {content!r}")
        if anchor.where == "before":
            at = idx
            indent_src = lines[idx]
        elif anchor.where == "after":
            at = idx + anchor.offset
            indent_src = lines[at]
        else:
            raise TransformRefusal(f"anchor {anchor.name}: bad position")
        indent = indent_src[: len(indent_src) - len(indent_src.lstrip())]
        if not indent:
            indent = "  "
        text = indent + anchor.call
        if not CALL_PATTERN.match(text):
            raise TransformRefusal(f"anchor {anchor.name}: not a CALL line")
        inserts.setdefault(at, []).append((anchor.where, text))
        records.append({"name": anchor.name, "pristine_line": anchor.line,
                        "access_line": n, "where": anchor.where,
                        "offset": anchor.offset, "call": anchor.call})

    out: list[str] = []
    for i, line in enumerate(lines):
        for where, text in inserts.get(i, []):
            if where == "before":
                out.append(text)
        out.append(line)
        for where, text in inserts.get(i, []):
            if where == "after":
                out.append(text)
    out_text = "\n".join(out)
    access_text = "\n".join(lines)

    diff_lines = list(difflib.unified_diff(
        access_text.splitlines(keepends=True), out_text.splitlines(keepends=True),
        fromfile="a/module_mp_fast_sbm_access.F",
        tofile="b/module_mp_fast_sbm_probe.F", n=0))
    body = [d for d in diff_lines if not d.startswith(("---", "+++", "@@"))]
    removed = [d for d in body if d.startswith("-")]
    added = [d[1:].rstrip("\n") for d in body if d.startswith("+")]
    if removed:
        raise TransformRefusal(f"probe transform removed lines: {removed}")
    if len(added) != len(ANCHORS) or not all(CALL_PATTERN.match(a) for a in added):
        raise TransformRefusal("probe diff is not exactly the inserted CALL lines")
    # Removing the inserted lines must give back the access text byte for byte.
    inserted = {text for group in inserts.values() for _, text in group}
    back = "\n".join(line for line in out if line not in inserted)
    if back != access_text:
        raise TransformRefusal("removing the CALL lines does not restore the input")

    probe_bytes = out_text.encode("ascii")
    for rec in records:
        rec["probe_line"] = next(
            i + 1 for i, line in enumerate(out) if line.strip() == rec["call"])
    summary = {
        "transform": "probe",
        "input": "access-transformed source",
        "access": access_summary,
        "output_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "inserted_lines": len(added),
        "removed_lines": 0,
        "all_inserted_lines_are_calls": True,
        "anchors": records,
    }
    return access, probe_bytes, "".join(diff_lines), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, type=Path,
                        help="pristine LF module_mp_fast_sbm.F")
    parser.add_argument("--out", dest="dst", required=True, type=Path)
    parser.add_argument("--diff", dest="diff", required=True, type=Path)
    parser.add_argument("--anchors", dest="anchors", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _, probe, diff, summary = probe_transform(args.src.read_bytes())
    except TransformRefusal as exc:
        print(f"instrument_probes: REFUSED: {exc}", file=sys.stderr)
        return 2
    args.dst.write_bytes(probe)
    args.diff.write_bytes(diff.encode("ascii"))
    args.anchors.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "anchors"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
