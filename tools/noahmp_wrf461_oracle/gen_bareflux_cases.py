#!/usr/bin/env python3
"""Emit the BARE_FLUX oracle case deck.

Every case is written as IEEE-754 binary32 hex words so that the bits the
compiled WRF routine sees are exactly the bits recorded in the fixture.

Each case carries a note saying which branch of BARE_FLUX / SFCDIF1 / ESAT it
is there to bind.  ``--explain`` prints that table.
"""

from __future__ import annotations

import argparse
import struct
import sys

NSNOW = 3
NSOIL = 4

REAL_FIELDS = [
    "dt", "sag", "lwdn", "ur", "uu", "vv", "sfctmp", "thair", "qair", "eair",
    "rhoair", "snowh", "zlvl", "zpd", "z0m", "fsno", "emg", "rsurf", "lathea",
    "gamma", "rhsur", "q2", "pahb", "dx", "dz8w", "qc", "psfc", "sfcprs",
    "tgb", "cm", "ch", "qsfc",
]
ARRAY_FIELDS = ["dzsnso", "stc", "df"]
INT_FIELDS = ["isnow", "ivgtyp", "iloc", "jloc", "iurban"]

OUT_FIELDS = [
    "tgb_out", "cm_out", "ch_out", "qsfc_out", "tauxb", "tauyb", "irb",
    "shb", "evb", "ghb", "t2mb", "q2b", "ehb2",
]


def hexf(x: float) -> str:
    return "%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


def unhexf(word: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word, 16)))[0]


BASE = dict(
    dt=90.0, sag=400.0, lwdn=350.0, ur=5.0, uu=4.0, vv=3.0,
    sfctmp=295.0, thair=296.0, qair=0.008, eair=1200.0, rhoair=1.15,
    snowh=0.0, zlvl=20.0, zpd=0.0, z0m=0.05, fsno=0.0, emg=0.95,
    rsurf=80.0, lathea=2.5104e6, gamma=66.0, rhsur=0.7, q2=0.008,
    pahb=0.0, dx=1000.0, dz8w=40.0, qc=0.0, psfc=98000.0, sfcprs=98000.0,
    tgb=298.0, cm=0.01, ch=0.01, qsfc=0.0085,
    isnow=0, ivgtyp=16, iloc=3, jloc=7, iurban=0,
    dzsnso=[0.05, 0.10, 0.15, 0.10, 0.30, 0.60, 1.00],
    stc=[265.0, 268.0, 271.0, 283.0, 285.0, 287.0, 289.0],
    df=[0.30, 0.35, 0.40, 1.20, 1.30, 1.40, 1.50],
)


def case(name: str, note: str, **over):
    rec = {k: (list(v) if isinstance(v, list) else v) for k, v in BASE.items()}
    for k, v in over.items():
        if k not in rec:
            raise KeyError(k)
        rec[k] = v
    rec["_name"] = name
    rec["_note"] = note
    return rec


def build_cases() -> list[dict]:
    cases = [
        case("bf01_unstable_day",
             "SFCDIF1 MOZ<0 unstable branch (logf+atanf+powf); ESAT water "
             "(T>0); RAMB/RAHB take the 1/(CM*UR) side of MAX; T2MB/Q2B "
             "normal branch; no snow reset; urban_flag false"),
        case("bf02_stable_night",
             "SFCDIF1 MOZ>=0 stable branch (FMNEW=-5*MOZ); ESAT water",
             sag=0.0, lwdn=280.0, tgb=289.0, sfctmp=295.0, rhsur=0.9,
             eair=1500.0),
        case("bf03_ice_branch",
             "ESAT ice branch (T<=0 -> ESATI/DSATI) at both ESAT call sites",
             sag=60.0, lwdn=230.0, sfctmp=258.0, tgb=256.0, eair=120.0,
             qair=0.0009, rhoair=1.35, lathea=2.844e6, gamma=74.0,
             stc=[248.0, 250.0, 252.0, 257.0, 259.0, 261.0, 263.0]),
        case("bf04_snow_reset",
             "OPT_STC==1 snow reset: SNOWH>0.05 .AND. TGB>TFRZ -> TGB=TFRZ "
             "and GHB recomputed from the residual; ISNOW=-2 so DF/DZSNSO/STC "
             "are indexed at -1",
             snowh=0.55, isnow=-2, sag=700.0, lwdn=330.0, sfctmp=278.0,
             tgb=274.0, fsno=0.9, emg=0.98, rsurf=50.0, rhsur=1.0,
             stc=[268.0, 270.0, 272.0, 274.0, 276.0, 278.0, 280.0]),
        case("bf05_lowwind",
             "UR->0: SFCDIF1 |TMP1|<=MPE guard fires; RAMB/RAHB huge; "
             "EHB2<1.0E-5 so T2MB=TGB and Q2B=QSFC",
             ur=1.0e-6, uu=8.0e-7, vv=6.0e-7),
        case("bf06_highwind_ramb_floor",
             "CM*UR>1 and CH*UR>1 so RAMB/RAHB take the 1.0 side of MAX",
             ur=50.0, uu=40.0, vv=30.0, z0m=1.5, zlvl=20.0),
        case("bf07_urban",
             "parameters%urban_flag TRUE -> Q2B forced to QSFC",
             iurban=1, ivgtyp=13, z0m=0.8, emg=0.97),
        case("bf08_tdc_clamp_high",
             "TDC upper clamp: TGB-TFRZ>50 so T saturates at +50 C",
             tgb=340.0, sfctmp=320.0, sag=900.0, lwdn=450.0, eair=4000.0,
             qair=0.02, rhoair=1.05, rhsur=0.4,
             stc=[300.0, 302.0, 304.0, 330.0, 328.0, 326.0, 324.0]),
        case("bf09_tdc_clamp_low",
             "TDC lower clamp: TGB-TFRZ<-50 so T saturates at -50 C; ice",
             tgb=200.0, sfctmp=210.0, sag=5.0, lwdn=90.0, eair=1.0,
             qair=1.0e-5, rhoair=1.6, lathea=2.844e6, gamma=74.0,
             rhsur=0.99,
             stc=[195.0, 197.0, 199.0, 205.0, 207.0, 209.0, 211.0]),
        case("bf10_degenerate_cmfm",
             "Degenerate roughness: (ZLVL-ZPD)/Z0M ~ 1 so TMPCM,TMPCH ~ 0 and "
             "the |CMFM|<=MPE / |CHFH|<=MPE guards fire",
             zlvl=10.0, zpd=0.0, z0m=9.99999),
        case("bf11_degenerate_z0m_2m",
             "Degenerate 2 m roughness: (2+Z0M)/Z0M ~ 1 so TMPCM2,TMPCH2 ~ 0 "
             "and the FH2=MIN(FH2,0.9*TMPCH2) clamp binds at a tiny bound, "
             "which drives EHB2.  NOTE: this does NOT reach the "
             "|CM2FM2|<=MPE / |CH2FH2|<=MPE guards -- gcov confirms both stay "
             "untaken -- and nothing can, through BARE_FLUX: CM2FM2 feeds "
             "only the commented-out CH2 formula, and CH2FH2 feeds only CH2, "
             "which BARE_FLUX declares, passes as INTENT(OUT) and never reads",
             z0m=2.5e5, zlvl=1.0e6, zpd=0.0, ur=12.0),
        case("bf12_fh_clamp",
             "Strongly unstable with a short log profile so FH=MIN(FH,0.9*"
             "TMPCH) and FM=MIN(FM,0.9*TMPCM) clamps bind",
             z0m=4.0, zlvl=10.0, sag=950.0, lwdn=420.0, tgb=330.0,
             sfctmp=295.0, ur=0.6, uu=0.5, vv=0.33, rsurf=400.0, rhsur=0.35),
        case("bf13_pahb",
             "Non-zero advected precipitation heat PAHB enters B and the "
             "snow-reset residual",
             pahb=125.0, snowh=0.30, isnow=-1, sag=520.0, sfctmp=276.0,
             tgb=272.0, fsno=0.6,
             stc=[266.0, 269.0, 271.0, 272.5, 274.0, 276.0, 278.0]),
        case("bf14_isnow_shallow",
             "ISNOW=-1 with SNOWH below the 0.05 reset threshold: the "
             "SNOWH>0.05 guard is exercised on its false side with snow present",
             isnow=-1, snowh=0.03, sag=300.0, sfctmp=280.0, tgb=282.0,
             fsno=0.4),
        case("bf15_zpd_offset",
             "Non-zero ZPD so (ZLVL-ZPD) differs from ZLVL in all four LOG "
             "arguments",
             zpd=6.5, zlvl=25.0, z0m=0.9, ur=3.2, uu=-2.0, vv=-2.5,
             sag=610.0, tgb=305.0),
        case("bf16_negative_wind_components",
             "UU,VV negative so TAUXB,TAUYB change sign",
             uu=-4.0, vv=-3.0),
        case("bf17_libm_logf_1",
             "Randomised column on which glibc logf and a correctly-rounded "
             "float32 log disagree: substituting FP64-log-then-round-once "
             "for logf moves at least one output bit",
             isnow=-1,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("4389F256"),
             lwdn=unhexf("4389D206"),
             ur=unhexf("400AE7E3"),
             uu=unhexf("C100FE65"),
             vv=unhexf("C167F1A0"),
             sfctmp=unhexf("43981E24"),
             thair=unhexf("43940000"),
             qair=unhexf("3C8C597B"),
             eair=unhexf("45045327"),
             rhoair=unhexf("3F671E93"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("40C5B26E"),
             zpd=unhexf("40350F41"),
             z0m=unhexf("3F83E5B9"),
             fsno=unhexf("3EFEBCC8"),
             emg=unhexf("3F75A242"),
             rsurf=unhexf("41B4E7AC"),
             lathea=unhexf("4A2D9580"),
             gamma=unhexf("425E2D5C"),
             rhsur=unhexf("3EC49BA4"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("4771FBF7"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("438FEE5C"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3F806F56"), unhexf("3DCF5D35"), unhexf("3F8DDAD4"), unhexf("3F4D9F8A"), unhexf("3F96D99C"), unhexf("3F2A2F4E"), unhexf("3F3FCDA6")],
             stc=[unhexf("439940BD"), unhexf("43794427"), unhexf("4396F9A2"), unhexf("43991862"), unhexf("4380DA76"), unhexf("4391AC4F"), unhexf("439725EF")],
             df=[unhexf("40142321"), unhexf("3F9A1AE9"), unhexf("3F98E479"), unhexf("3F717A0E"), unhexf("3EF1E697"), unhexf("3F6E78DA"), unhexf("3F8CA3FF")]),
        case("bf18_libm_logf_2",
             "Randomised column on which glibc logf and a correctly-rounded "
             "float32 log disagree: substituting FP64-log-then-round-once "
             "for logf moves at least one output bit",
             isnow=0,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("4409AE43"),
             lwdn=unhexf("439EA451"),
             ur=unhexf("4175049B"),
             uu=unhexf("419226A8"),
             vv=unhexf("40EECDC7"),
             sfctmp=unhexf("43687A21"),
             thair=unhexf("43940000"),
             qair=unhexf("3CAF4C0E"),
             eair=unhexf("44221A87"),
             rhoair=unhexf("3F349C74"),
             snowh=unhexf("3CA3D70A"),
             zlvl=unhexf("40B469CC"),
             zpd=unhexf("4016622B"),
             z0m=unhexf("3E273CB2"),
             fsno=unhexf("3F609242"),
             emg=unhexf("3F7DFC0B"),
             rsurf=unhexf("4428E936"),
             lathea=unhexf("4A193900"),
             gamma=unhexf("426FFDC0"),
             rhsur=unhexf("3F03F7F3"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("479DB5AE"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("438001DC"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3DA45B2E"), unhexf("3F9592E3"), unhexf("3DB140CA"), unhexf("3F137411"), unhexf("3E642808"), unhexf("3F894392"), unhexf("3F84AF8A")],
             stc=[unhexf("4386915F"), unhexf("4395FE3E"), unhexf("43993466"), unhexf("438D04EA"), unhexf("4392C373"), unhexf("43946E72"), unhexf("438FB098")],
             df=[unhexf("3FA9620C"), unhexf("400B4B1D"), unhexf("3F161EB8"), unhexf("400C1246"), unhexf("400E7ABA"), unhexf("3FF4E3D8"), unhexf("40085832")]),
        case("bf19_libm_atanf_1",
             "Randomised column on which glibc atanf and a correctly-rounded "
             "float32 atan disagree; SFCDIF1 calls ATAN only on the MOZ<0 "
             "branch",
             isnow=-1,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("4405561A"),
             lwdn=unhexf("43E40B65"),
             ur=unhexf("401FBA94"),
             uu=unhexf("41903C2B"),
             vv=unhexf("C198C853"),
             sfctmp=unhexf("438D149D"),
             thair=unhexf("43940000"),
             qair=unhexf("3BF56181"),
             eair=unhexf("4538D409"),
             rhoair=unhexf("3F78D080"),
             snowh=unhexf("3ECCCCCD"),
             zlvl=unhexf("408C2E77"),
             zpd=unhexf("3F8F90FC"),
             z0m=unhexf("3F1A89E4"),
             fsno=unhexf("3F58AEF1"),
             emg=unhexf("3F77B192"),
             rsurf=unhexf("43559FF8"),
             lathea=unhexf("4A2D9580"),
             gamma=unhexf("428F29A9"),
             rhsur=unhexf("3F2376A4"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("4303F640"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("47963FA2"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("439BA4AC"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3F8BE374"), unhexf("3DDB7866"), unhexf("3F5F8418"), unhexf("3F30C2D5"), unhexf("3F956466"), unhexf("3F08BBCB"), unhexf("3E625700")],
             stc=[unhexf("438B2230"), unhexf("4375A2DA"), unhexf("439607B0"), unhexf("437D58A8"), unhexf("4391F19F"), unhexf("4376112B"), unhexf("437B1891")],
             df=[unhexf("3FF97479"), unhexf("3FF083EA"), unhexf("3F2131CA"), unhexf("3F8F7A9F"), unhexf("3F872684"), unhexf("3F6DDE3A"), unhexf("3F469B9F")]),
        case("bf20_libm_atanf_2",
             "Randomised column on which glibc atanf and a correctly-rounded "
             "float32 atan disagree; SFCDIF1 calls ATAN only on the MOZ<0 "
             "branch",
             isnow=-2,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("43562D22"),
             lwdn=unhexf("439E4409"),
             ur=unhexf("40C83260"),
             uu=unhexf("C1767DC2"),
             vv=unhexf("419ADEA1"),
             sfctmp=unhexf("438E11E9"),
             thair=unhexf("43940000"),
             qair=unhexf("3B07FBD3"),
             eair=unhexf("4542ED2E"),
             rhoair=unhexf("3F6C2315"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("41A87884"),
             zpd=unhexf("401E7474"),
             z0m=unhexf("3E11CEF5"),
             fsno=unhexf("3F0EAD0E"),
             emg=unhexf("3F738B20"),
             rsurf=unhexf("4119E361"),
             lathea=unhexf("4A193900"),
             gamma=unhexf("42A1B5FB"),
             rhsur=unhexf("3E80BB63"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("47891F93"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("43962E57"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3F4BEF9A"), unhexf("3F3DC0FC"), unhexf("3F525CA7"), unhexf("3F976CB6"), unhexf("3E64E1BD"), unhexf("3D26C732"), unhexf("3EC4BF96")],
             stc=[unhexf("437B792E"), unhexf("437A0882"), unhexf("438984D0"), unhexf("438EE066"), unhexf("438D7C00"), unhexf("43800029"), unhexf("43812E08")],
             df=[unhexf("40005965"), unhexf("3F215C75"), unhexf("400BB7AB"), unhexf("3F2C01C0"), unhexf("3F20013D"), unhexf("3F906B89"), unhexf("4003B8FA")]),
        case("bf21_libm_powf_1",
             "Randomised column on which glibc powf(x,0.25) disagrees with "
             "both a correctly-rounded x**0.25 and the sqrt(sqrt(x)) "
             "shortcut",
             isnow=-1,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("43F86F3F"),
             lwdn=unhexf("43AB3DFF"),
             ur=unhexf("3F850B6B"),
             uu=unhexf("418E9269"),
             vv=unhexf("C08247A0"),
             sfctmp=unhexf("43987CA5"),
             thair=unhexf("43940000"),
             qair=unhexf("3C873F3E"),
             eair=unhexf("43C153E4"),
             rhoair=unhexf("3F5C59B9"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("4144ABF1"),
             zpd=unhexf("400B05E8"),
             z0m=unhexf("3C39ABD4"),
             fsno=unhexf("3EBB343A"),
             emg=unhexf("3F7BC044"),
             rsurf=unhexf("43873448"),
             lathea=unhexf("4A2D9580"),
             gamma=unhexf("4256D953"),
             rhsur=unhexf("3F12BBF9"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("47BA9AD1"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("439FCF57"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3F4270CC"), unhexf("3E8FE869"), unhexf("3F415006"), unhexf("3F986E2A"), unhexf("3F9196CA"), unhexf("3F20DB2F"), unhexf("3EEABCFE")],
             stc=[unhexf("438149E7"), unhexf("439A6C4B"), unhexf("438404D1"), unhexf("437F221A"), unhexf("43900FA7"), unhexf("43840575"), unhexf("437B372C")],
             df=[unhexf("401AFA3B"), unhexf("3FE5D42F"), unhexf("3FA88430"), unhexf("3F842C47"), unhexf("3E77F274"), unhexf("401C97A5"), unhexf("3FF8B968")]),
        case("bf22_libm_powf_2",
             "Randomised column on which glibc powf(x,0.25) disagrees with "
             "both a correctly-rounded x**0.25 and the sqrt(sqrt(x)) "
             "shortcut",
             isnow=-1,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("4419FBDD"),
             lwdn=unhexf("43B7DF40"),
             ur=unhexf("4009953F"),
             uu=unhexf("C16413A5"),
             vv=unhexf("41909A81"),
             sfctmp=unhexf("43989C1C"),
             thair=unhexf("43940000"),
             qair=unhexf("3BCEF8A3"),
             eair=unhexf("44AB9C16"),
             rhoair=unhexf("3F50DC9A"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("4041289B"),
             zpd=unhexf("3C73463C"),
             z0m=unhexf("3DCFF847"),
             fsno=unhexf("3F08BC50"),
             emg=unhexf("3F7AD93E"),
             rsurf=unhexf("4117D204"),
             lathea=unhexf("4A193900"),
             gamma=unhexf("42780E64"),
             rhsur=unhexf("3EBC1492"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("476C2ADD"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("43ABCFB2"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3F48C8C1"), unhexf("3F1B5133"), unhexf("3F056D35"), unhexf("3ED1BC43"), unhexf("3EC501BD"), unhexf("3F468BA8"), unhexf("3F6EE6D9")],
             stc=[unhexf("4380206B"), unhexf("4399D19E"), unhexf("438F1788"), unhexf("4396410D"), unhexf("438245A7"), unhexf("4396DFEA"), unhexf("438BEAC4")],
             df=[unhexf("3FC773E4"), unhexf("40198F89"), unhexf("3F7FD793"), unhexf("3F12780E"), unhexf("3FBDDDD1"), unhexf("3FBF5F9E"), unhexf("3F91BCC5")]),
        case("bf23_mozsgn_reset_1",
             "MOZ changes sign twice across the five stability iterations, "
             "so MOZSGN reaches 2 and SFCDIF1 zeroes MOZ/FM/FH/MOZ2/FM2/FH2",
             isnow=0,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("44045856"),
             lwdn=unhexf("439468D5"),
             ur=unhexf("3E848716"),
             uu=unhexf("41406FB0"),
             vv=unhexf("C121D809"),
             sfctmp=unhexf("439731AA"),
             thair=unhexf("43940000"),
             qair=unhexf("3B8A56D2"),
             eair=unhexf("44644FDD"),
             rhoair=unhexf("3FA70EB0"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("419A2B71"),
             zpd=unhexf("3FD328D4"),
             z0m=unhexf("40038C18"),
             fsno=unhexf("3E9A8A82"),
             emg=unhexf("3F6383F5"),
             rsurf=unhexf("3FC7BCE4"),
             lathea=unhexf("4A193900"),
             gamma=unhexf("42A59A21"),
             rhsur=unhexf("3F1187DD"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("00000000"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("47AD362B"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("4385F2A3"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3E133290"), unhexf("3F21C43D"), unhexf("3F13F74C"), unhexf("3F604E5F"), unhexf("3F0D0A8D"), unhexf("3EEC6E22"), unhexf("3F3D12BA")],
             stc=[unhexf("43823261"), unhexf("43817B6F"), unhexf("4391925F"), unhexf("43817C38"), unhexf("4398FD1C"), unhexf("438E9CC2"), unhexf("439962C5")],
             df=[unhexf("3FFD519A"), unhexf("3FBCE3F2"), unhexf("3E983C24"), unhexf("3FB920AC"), unhexf("3F271D18"), unhexf("3FBDB11D"), unhexf("3E8F027D")]),
        case("bf24_mozsgn_reset_2",
             "MOZ changes sign twice across the five stability iterations, "
             "so MOZSGN reaches 2 and SFCDIF1 zeroes MOZ/FM/FH/MOZ2/FM2/FH2",
             isnow=-1,
             ivgtyp=16,
             iloc=3,
             jloc=7,
             iurban=0,
             dt=unhexf("42B40000"),
             sag=unhexf("4414C10D"),
             lwdn=unhexf("43C82EC9"),
             ur=unhexf("3F43ABD6"),
             uu=unhexf("C137AF37"),
             vv=unhexf("40C7A30C"),
             sfctmp=unhexf("439CFBAC"),
             thair=unhexf("43940000"),
             qair=unhexf("3CA6DF22"),
             eair=unhexf("4329C173"),
             rhoair=unhexf("3FA3DD9D"),
             snowh=unhexf("00000000"),
             zlvl=unhexf("4102E152"),
             zpd=unhexf("400B223D"),
             z0m=unhexf("3FFF6E76"),
             fsno=unhexf("3F09B989"),
             emg=unhexf("3F7B5174"),
             rsurf=unhexf("412FDC0A"),
             lathea=unhexf("4A2D9580"),
             gamma=unhexf("42966ECF"),
             rhsur=unhexf("3EC407F3"),
             q2=unhexf("3C03126F"),
             pahb=unhexf("42AB6501"),
             dx=unhexf("447A0000"),
             dz8w=unhexf("42200000"),
             qc=unhexf("00000000"),
             psfc=unhexf("47863655"),
             sfcprs=unhexf("47BF6800"),
             tgb=unhexf("43974232"),
             cm=unhexf("3C23D70A"),
             ch=unhexf("3C23D70A"),
             qsfc=unhexf("3C0B4396"),
             dzsnso=[unhexf("3EF33CC2"), unhexf("3F57A12D"), unhexf("3F99F355"), unhexf("3FAC7B5D"), unhexf("3FBDCEE6"), unhexf("3FBDE2BB"), unhexf("3EFAAA34")],
             stc=[unhexf("43714BBC"), unhexf("43840145"), unhexf("4395EC80"), unhexf("4399CE50"), unhexf("43971299"), unhexf("438E7FAB"), unhexf("43707CD9")],
             df=[unhexf("401B7F18"), unhexf("3F672A52"), unhexf("3F7BEEC8"), unhexf("3FB3C858"), unhexf("3F21DAB8"), unhexf("3FE5B64D"), unhexf("4008F835")]),
        case("bf25_ehb2_threshold",
             "UR tuned so EHB2 lands at 3.0e-6, between 1.0e-6 and the "
             "1.0E-5 cut-off, which is what pins the cut-off value rather "
             "than merely which side of it a case falls on",
             ur=unhexf("368BFBB2"), uu=unhexf("365FF91E"),
             vv=unhexf("3627FAD6")),
        case("bf26_snowh_just_above_threshold",
             "SNOWH = 0.0500001, one step above the snow-reset threshold, so "
             "the reset fires; paired with bf27 this pins the 0.05 constant "
             "to within a float32 step",
             snowh=unhexf("3D4CCCE8"), isnow=-1, sag=700.0, lwdn=330.0,
             sfctmp=278.0, tgb=274.0, emg=0.98, rsurf=50.0, rhsur=1.0,
             stc=[268.0, 270.0, 272.0, 274.0, 276.0, 278.0, 280.0]),
        case("bf27_snowh_just_below_threshold",
             "SNOWH = 0.0499999, one step below the snow-reset threshold, so "
             "the reset does not fire although everything else matches bf26",
             snowh=unhexf("3D4CCCB2"), isnow=-1, sag=700.0, lwdn=330.0,
             sfctmp=278.0, tgb=274.0, emg=0.98, rsurf=50.0, rhsur=1.0,
             stc=[268.0, 270.0, 272.0, 274.0, 276.0, 278.0, 280.0]),
    ]
    return cases


def format_case(rec: dict) -> str:
    parts = [rec["_name"], "1", "1"]
    parts += [str(rec[k]) for k in INT_FIELDS]
    parts += [hexf(rec[k]) for k in REAL_FIELDS]
    for name in ARRAY_FIELDS:
        vals = rec[name]
        if len(vals) != NSNOW + NSOIL:
            raise ValueError(f"{name} must have {NSNOW + NSOIL} entries")
        parts += [hexf(v) for v in vals]
    return " ".join(parts)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="-")
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args(argv)

    cases = build_cases()
    if args.explain:
        for rec in cases:
            print(f"{rec['_name']}: {rec['_note']}")
        return 0

    lines = [format_case(rec) for rec in cases]
    text = "\n".join(lines) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", newline="\n") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
