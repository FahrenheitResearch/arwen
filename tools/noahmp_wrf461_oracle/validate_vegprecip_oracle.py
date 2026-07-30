#!/usr/bin/env python3
"""Fixture generator and acceptance gate for the Noah-MP PHENOLOGY /
PRECIP_HEAT bitwise port.

Sub-commands
------------
``--emit-cases {phen,prcp,negctl}``
    Write oracle driver input lines to stdout.  Nothing else in the project
    knows the case list; it lives here so a reviewer can read the branch each
    case is there to bind.

``--oracle PATH``
    Run the compiled oracle over the case list and write the two fixture CSVs
    under ``gpuwm/data/noahmp/oracle/``.  Every REAL crosses the boundary as an
    IEEE-754 binary32 bit pattern, so no decimal rounding can enter.

``--validate``
    Recompute every fixture row with :mod:`gpuwm.core.noahmp_vegprecip` and
    report the worst ULP distance per output column.  Non-zero is a failure.

``--mutants``
    The mutation study: for each argument of each leaf, build a mutant that
    ignores that argument and report whether the fixture detects it.

``--libm-sweep-dump N`` / ``--libm-sweep-check FILE``
    Independent re-verification of :mod:`gpuwm.core.noahmp_libm` (which this
    port depends on but does not own).  ``--libm-sweep-dump`` prints the
    arguments this port actually feeds ``expf``/``powf``, plus a randomized
    sweep, as hex; a C program on a glibc host answers them; ``--check``
    compares the answers against the transcription.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import struct
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # <repo>/tools/<dir> -> <repo>
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpuwm.core import noahmp_vegprecip as vp          # noqa: E402
from gpuwm.core.fp32_ulp import max_ulp                 # noqa: E402
from gpuwm.core.noahmp_libm import expf, f32, powf      # noqa: E402

FIXTURE_DIR = os.path.join(_ROOT, "gpuwm", "data", "noahmp", "oracle")
PHEN_CSV = os.path.join(FIXTURE_DIR, "noahmp-vegprecip-phenology.csv")
PRCP_CSV = os.path.join(FIXTURE_DIR, "noahmp-vegprecip-precip_heat.csv")


def h(x: float) -> str:
    """8-digit hex of the binary32 pattern of ``x``."""
    return f"{struct.unpack('<I', struct.pack('<f', x))[0]:08X}"


def uh(s: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(s, 16)))[0]


# ==========================================================================
# Vegetation parameter rows
#
# The MPTABLE rows are the USGS 27-class block of
# ``run/MPTABLE.TBL`` in the pinned WRF tree (&noahmp_usgs_parameters).  The
# column index is the USGS vegetation category, which is also the VEGTYP the
# case passes.  ISWATER=16, ISBARREN=19, ISICE=24, ISURBAN=1 come from the
# same namelist block.
#
# Rows whose name starts with SYN_ are synthetic.  They exist because no
# MPTABLE row can put SAI or LAI strictly inside (0, 0.05), so no MPTABLE row
# can bind the ``SAI < 0.05`` / ``LAI < 0.05`` tests with a *nonzero* value --
# and a test that only ever sees an argument already equal to its replacement
# is not bound at all.  Each SYN_ row is otherwise an MPTABLE row.
# ==========================================================================
VEG = {
    # name: (hvt, hvb, tmin, ch2op, urban, laim[12], saim[12])
    "USGS01_URBAN": (15.0, 1.00, 0.0, 0.1, True,
                     [0.0] * 12, [0.0] * 12),
    "USGS07_GRASSLAND": (1.00, 0.05, 273.0, 0.1, False,
                         [0.4, 0.5, 0.6, 0.7, 1.2, 3.0, 3.5, 1.5, 0.7, 0.6, 0.5, 0.4],
                         [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 1.3, 1.1, 0.4, 0.4, 0.4]),
    "USGS08_SHRUBLAND": (1.10, 0.10, 273.0, 0.1, False,
                         [0.0, 0.0, 0.2, 0.6, 1.5, 2.3, 2.3, 1.7, 0.6, 0.2, 0.0, 0.0],
                         [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.7, 0.3, 0.2]),
    "USGS11_DECID_BROADLEAF": (16.0, 11.5, 273.0, 0.1, False,
                               [0.0, 0.0, 0.3, 1.2, 3.0, 4.7, 4.5, 3.4, 1.2, 0.3, 0.0, 0.0],
                               [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 1.2, 1.6, 1.4, 0.6, 0.4]),
    "USGS12_DECID_NEEDLE": (18.0, 7.00, 268.0, 0.1, False,
                            [0.0, 0.0, 0.0, 0.6, 1.2, 2.0, 2.6, 1.7, 1.0, 0.5, 0.2, 0.0],
                            [0.3, 0.3, 0.3, 0.4, 0.4, 0.7, 1.3, 1.2, 1.0, 0.8, 0.6, 0.5]),
    "USGS16_WATER": (0.00, 0.00, 0.0, 0.1, False, [0.0] * 12, [0.0] * 12),
    "USGS19_BARREN": (0.00, 0.00, 0.0, 0.1, False, [0.0] * 12, [0.0] * 12),
    "USGS20_HERB_TUNDRA": (0.50, 0.10, 268.0, 0.1, False,
                           [0.2, 0.3, 0.3, 0.4, 0.6, 1.5, 1.7, 0.8, 0.4, 0.3, 0.2, 0.2],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.5, 0.3, 0.2]),
    "USGS23_BARE_TUNDRA": (0.50, 0.10, 268.0, 0.1, False, [0.0] * 12, [0.0] * 12),
    "USGS24_SNOW_ICE": (0.00, 0.00, 0.0, 0.1, False, [0.0] * 12, [0.0] * 12),
    # synthetic, see the block comment above
    "SYN_SAI_SUBTHRESH": (16.0, 11.5, 273.0, 0.1, False, [2.0] * 12, [0.03] * 12),
    "SYN_LAI_SUBTHRESH": (16.0, 11.5, 273.0, 0.1, False, [0.03] * 12, [0.30] * 12),
}

ISWATER, ISBARREN, ISICE = 16, 19, 24

# --------------------------------------------------------------------------
# PHENOLOGY cases.  "binds" names the decision each case is here to exercise.
# Every case carries lai_in / sai_in = 9.99 / 8.88 so that any leak of the
# INTENT(INOUT) input value into an output is unmissable.
# --------------------------------------------------------------------------
PHEN_CASES = [
    # id, veg, vegtyp, yearlen, julian, lat, snowh, tv, binds
    ("PH01_NH_MIDYEAR", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7, 0.02, 290.0,
     "LAT>=0; interior IT1/IT2; MAX(SNOWH-HVB,0)=0; HVT>1 so no EXP; TV>TMIN"),
    ("PH02_SH_WRAP", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, -0.7, 0.02, 290.0,
     "LAT<0 so DAY=MOD(JULIAN+YEARLEN/2,YEARLEN); IT2>12 wraps to 1"),
    ("PH03_IT1_UNDERFLOW", "USGS07_GRASSLAND", 7, 365, 0.0, 0.7, 0.05, 280.0,
     "JULIAN=0 so IT1=0 wraps to 12; HVT==1.0 exactly so the EXP branch is "
     "live with SNOWH<SNOWHC"),
    ("PH04_SNOW_OVER_SNOWHC", "USGS07_GRASSLAND", 7, 365, 200.0, 0.7, 2.0, 280.0,
     "EXP branch with SNOWH>SNOWHC so MIN picks SNOWHC and FB=1; ESAI<0.05 "
     "then ELAI forced to 0"),
    ("PH05_WATER", "USGS16_WATER", 16, 365, 100.0, 0.7, 0.5, 271.0,
     "VEGTYP==ISWATER; HVT==0 so the EXP branch is off and MAX(1e-6,HVT-HVB) "
     "takes the floor"),
    ("PH06_BARREN", "USGS19_BARREN", 19, 365, 100.0, 0.7, 0.5, 271.0,
     "VEGTYP==ISBARREN"),
    ("PH07_ICE", "USGS24_SNOW_ICE", 24, 365, 100.0, 0.7, 0.5, 271.0,
     "VEGTYP==ISICE"),
    ("PH08_URBAN", "USGS01_URBAN", 1, 365, 100.0, 0.7, 0.5, 271.0,
     "parameters%urban_flag"),
    ("PH09_IGS_OFF", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7, 0.02, 260.0,
     "TV<TMIN so IGS=0"),
    ("PH10_LEAP_YEAR", "USGS12_DECID_NEEDLE", 12, 366, 365.9, 0.7, 0.10, 275.0,
     "YEARLEN=366; IT1=12 and IT2 wraps"),
    ("PH11_EXP_PARTIAL_BURIAL", "USGS20_HERB_TUNDRA", 20, 365, 200.3, 0.7, 0.15, 275.0,
     "EXP branch with SNOWH<SNOWHC and a partial FB"),
    ("PH12_ZERO_TABLE_EXP", "USGS23_BARE_TUNDRA", 23, 365, 100.0, 0.7, 0.30, 275.0,
     "all-zero LAIM/SAIM with the EXP branch live"),
    ("PH13_SAI_SUBTHRESH", "SYN_SAI_SUBTHRESH", 11, 365, 181.5, 0.7, 0.02, 290.0,
     "SAI in (0,0.05) so SAI:=0, then SAI==0.0 forces LAI:=0"),
    ("PH14_LAI_SUBTHRESH", "SYN_LAI_SUBTHRESH", 11, 365, 181.5, 0.7, 0.02, 290.0,
     "LAI in (0,0.05) with SAI>=0.05, binding the first LAI disjunct alone"),
    ("PH15_PARTIAL_BURIAL", "USGS11_DECID_BROADLEAF", 11, 365, 250.0, 0.7, 13.0, 290.0,
     "MAX(SNOWH-HVB,0)>0 and MIN picks SNOWH-HVB"),
    ("PH16_FULL_BURIAL", "USGS11_DECID_BROADLEAF", 11, 365, 250.0, 0.7, 20.0, 290.0,
     "MIN picks HVT-HVB so FB=1 and both ELAI and ESAI collapse"),
    ("PH17_ESAI_SUBTHRESH", "USGS20_HERB_TUNDRA", 20, 365, 15.0, 0.7, 0.15, 275.0,
     "ESAI in (0,0.05) so ESAI:=0 while ELAI>=0.05, binding the ESAI==0 "
     "disjunct of the ELAI test on its own"),
    ("PH18_SH_EARLY", "USGS08_SHRUBLAND", 8, 365, 10.0, -0.7, 0.0, 271.0,
     "southern hemisphere with a small JULIAN; HVT>1 so no EXP; SNOWH=0"),
    # ---------------------------------------------------------------------
    # The four cases below exist because the mutation study proved the ones
    # above cannot detect VEGTYP, ISWATER, ISBARREN, ISICE or URBAN_FLAG being
    # ignored: in USGS the water/barren/ice/urban rows carry all-zero LAIM and
    # SAIM, so the zeroing they trigger is not observable.  These pair a
    # *vegetated* parameter row with a category index that points at it, which
    # is exactly what a different land-use dataset does -- MODIS-IGBP puts
    # ISWATER at 17, ISBARREN at 16, ISICE at 15 and ISURBAN at 13 while USGS
    # puts them at 16, 19, 24 and 1.  The index is data, not a constant.
    ("PH19_WATER_INDEX_HITS_VEG", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0,
     "VEGTYP==ISWATER on a row with nonzero LAIM/SAIM, so the zeroing is "
     "observable", (11, 19, 24, False)),
    ("PH20_BARREN_INDEX_HITS_VEG", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0,
     "VEGTYP==ISBARREN on a vegetated row", (16, 11, 24, False)),
    ("PH21_ICE_INDEX_HITS_VEG", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0,
     "VEGTYP==ISICE on a vegetated row", (16, 19, 11, False)),
    ("PH22_URBAN_FLAG_ON_VEG", "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0,
     "parameters%urban_flag true on a vegetated row", (16, 19, 24, True)),
]

# Negative controls: never written to the fixture.  They exist to show what
# the dead branches would have produced, which is what makes "dead" a
# measurement rather than a claim.
PHEN_NEGCTL = [
    ("NEGCTL_DVEG7_INPUT_LAI", 7, "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0, 2.5, 0.7,
     "dveg=7 reads the INTENT(INOUT) LAI instead of the table; proves both "
     "that the block is a different function and that under dveg=4 the input "
     "LAI/SAI are dead"),
    ("NEGCTL_DVEG4_INPUT_LAI", 4, "USGS11_DECID_BROADLEAF", 11, 365, 181.5, 0.7,
     0.02, 290.0, 2.5, 0.7,
     "same inputs at dveg=4: the table overwrites LAI/SAI"),
]

# --------------------------------------------------------------------------
# PRECIP_HEAT cases
# --------------------------------------------------------------------------
# id, ist, dt, uu, vv, elai, esai, fveg, bdfall, rain, snow, fp,
# canliq, canice, tv, sfctmp, tg, ch2op, binds
PRCP_CASES = [
    ("PR01_CAP_LIMITED", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 5.0e-4, 0.0, 1.0,
     0.10, 0.0, 280.0, 281.0, 279.0, 0.1,
     "LSAI>0; rain MIN picks the capacity side; CANICE==0 so FWET comes from "
     "liquid; 0<FVEG<1"),
    ("PR02_RATE_LIMITED", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 1.0e-4, 0.0, 1.0,
     0.0, 0.0, 280.0, 281.0, 279.0, 0.1,
     "rain MIN picks FVEG*RAIN*FP"),
    ("PR03_OVERFULL_CANOPY", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 5.0e-4, 0.0, 1.0,
     0.50, 0.0, 280.0, 281.0, 279.0, 0.1,
     "CANLIQ>MAXLIQ makes the capacity negative so MAX(QINTR,0) binds"),
    ("PR04_BURIED_WITH_STORE", 1, 90.0, 2.0, 1.0, 0.0, 0.0, 0.0, 100.0, 4.0e-4,
     2.0e-4, 1.0, 0.20, 0.30, 275.0, 276.0, 274.0, 0.1,
     "LSAI==0 with CANLIQ>0 and CANICE>0: both buried-canopy dumps; FVEG<=0 "
     "so PAHB absorbs PAHG; FWET base is 0 so powf(0,0.667) is exercised"),
    ("PR05_FVEG_ONE", 1, 90.0, 2.0, 1.0, 2.0, 0.4, 1.0, 100.0, 2.0e-4, 1.0e-4, 1.0,
     0.05, 0.02, 272.0, 274.0, 271.0, 0.1,
     "FVEG>=1 so PAHB:=0; CANICE>0 so FWET comes from snow; TV>270.15 so FT>0"),
    ("PR06_BURIED_EMPTY", 1, 90.0, 2.0, 1.0, 0.0, 0.0, 0.5, 100.0, 3.0e-4, 2.0e-4, 1.0,
     0.0, 0.0, 275.0, 276.0, 274.0, 0.1,
     "LSAI==0 with empty stores: the CANLIQ>0 and CANICE>0 tests both fail"),
    ("PR07_FT_FLOOR", 1, 90.0, 3.0, 4.0, 2.0, 0.4, 0.8, 100.0, 0.0, 1.0e-4, 1.0,
     0.0, 0.50, 265.0, 266.0, 264.0, 0.1,
     "TV<270.15 so MAX(0,(TV-270.15)/1.87e5) takes the floor; RAIN==0"),
    ("PR08_ICEDRIP_MIN_STORE", 1, 3600.0, 40.0, 30.0, 2.0, 0.4, 0.8, 100.0, 0.0,
     1.0e-4, 1.0, 0.0, 1.00, 300.0, 299.0, 298.0, 0.1,
     "ICEDRIP MIN picks CANICE/DT+QINTS"),
    ("PR09_ICEDRIP_MIN_RATE", 1, 90.0, 40.0, 30.0, 3.0, 0.5, 0.8, 100.0, 0.0,
     1.0e-4, 1.0, 0.0, 20.0, 300.0, 299.0, 298.0, 0.1,
     "ICEDRIP MIN picks CANICE*(FV+FT); CANICE>MAXSNO so MAX(QINTS,0) binds "
     "and FWET>1 so MIN(FWET,1) binds"),
    ("PR10_TINY_CANOPY", 1, 90.0, 2.0, 1.0, 1.0e-6, 0.0, 0.8, 100.0, 5.0e-4, 0.0, 1.0,
     0.0, 0.0, 280.0, 281.0, 279.0, 0.1,
     "MAXLIQ<1e-6 so MAX(MAXLIQ,1e-6) takes the floor in FWET"),
    ("PR11_LAKE_ABOVE_FREEZING", 2, 90.0, 2.0, 1.0, 2.0, 0.4, 0.8, 100.0, 2.0e-4,
     3.0e-4, 1.0, 0.05, 0.02, 275.0, 276.0, 280.0, 0.1,
     "IST==2 and TG>TFRZ so QSNOW and SNOWHIN are zeroed"),
    ("PR12_LAKE_BELOW_FREEZING", 2, 90.0, 2.0, 1.0, 2.0, 0.4, 0.8, 100.0, 2.0e-4,
     3.0e-4, 1.0, 0.05, 0.02, 271.0, 272.0, 270.0, 0.1,
     "IST==2 with TG<TFRZ: the lake test fails on its second conjunct"),
    ("PR13_PAH_CLIP_MIXED", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 1.0e-2, 5.0e-3,
     1.0, 0.05, 0.05, 275.0, 280.0, 285.0, 0.1,
     "PAHV clipped at +20, PAHG and PAHB clipped at -20"),
    ("PR14_PAH_CLIP_REVERSED", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 1.0e-2,
     5.0e-3, 1.0, 0.05, 0.05, 285.0, 275.0, 270.0, 0.1,
     "PAHV clipped at -20, PAHG and PAHB clipped at +20"),
    ("PR15_LAKE_AT_FREEZING", 2, 90.0, 2.0, 1.0, 2.0, 0.4, 0.8, 100.0, 2.0e-4,
     3.0e-4, 1.0, 0.05, 0.02, 271.0, 272.0, 273.16, 0.1,
     "IST==2 with TG exactly TFRZ: strict > must not fire"),
    ("PR16_QINTS_RATE_LIMITED", 1, 90.0, 2.0, 1.0, 2.0, 0.4, 0.5, 100.0, 0.0,
     2.0e-4, 0.5, 0.0, 0.01, 275.0, 276.0, 274.0, 0.1,
     "snow MIN picks FVEG*SNOW*FP; FP<1"),
    ("PR17_DENSE_SNOWFALL", 1, 90.0, 2.0, 1.0, 2.0, 0.4, 0.8, 250.0, 1.0e-4,
     4.0e-4, 1.0, 0.02, 0.03, 274.0, 275.0, 272.0, 0.1,
     "a different BDFALL exercises the 0.27+46/BDFALL factor and SNOWHIN"),
    ("PR18_NO_PRECIP", 1, 90.0, 0.0, 0.0, 2.0, 0.4, 0.8, 100.0, 0.0, 0.0, 1.0,
     0.03, 0.0, 275.0, 276.0, 274.0, 0.1,
     "RAIN==SNOW==0 and UU==VV==0: SQRT(0), and every rate collapses"),
    ("PR19_CAP_EXACTLY_FULL", 1, 90.0, 2.0, 1.0, 3.0, 0.5, 0.8, 100.0, 5.0e-4, 0.0,
     1.0, 0.28, 0.0, 280.0, 281.0, 279.0, 0.1,
     "CANLIQ==MAXLIQ exactly, so the capacity term is a signed zero and the "
     "MIN/MAX tie-breaking rule is observable"),
]

PHEN_OUT_COLS = ["lai", "sai", "elai", "esai", "igs", "fb"]
PRCP_OUT_COLS = [
    "canliq", "canice", "qintr", "qdripr", "qthror", "qints", "qdrips",
    "qthros", "pahv", "pahg", "pahb", "qrain", "qsnow", "snowhin", "fwet",
    "cmc",
]


# ==========================================================================
# case -> driver line
# ==========================================================================
def phen_indices(case):
    """(iswater, isbarren, isice, urban_flag) for a case.

    Defaults to the USGS namelist values and the parameter row's own urban
    flag; a case may override all four with a trailing 10th field.
    """
    hvt, hvb, tmin, ch2op, urban, laim, saim = VEG[case[1]]
    if len(case) >= 10 and case[9] is not None:
        return case[9]
    return (ISWATER, ISBARREN, ISICE, urban)


def phen_line(case, dveg=4, lai_in=9.99, sai_in=8.88, caseid=None):
    cid, vegname, vegtyp, yearlen, julian, lat, snowh, tv = case[:8]
    hvt, hvb, tmin, _ch2op, _urban, laim, saim = VEG[vegname]
    iswater, isbarren, isice, urban = phen_indices(case)
    words = [
        "PHEN", caseid or cid,
        str(dveg), "0", str(vegtyp), str(yearlen), "1",
        str(iswater), str(isbarren), str(isice), "1" if urban else "0",
        h(snowh), h(tv), h(lat), h(julian), h(280.0), h(lai_in), h(sai_in),
        h(hvt), h(hvb), h(tmin),
    ]
    words += [h(v) for v in laim]
    words += [h(v) for v in saim]
    return " ".join(words)


def prcp_line(case):
    (cid, ist, dt, uu, vv, elai, esai, fveg, bdfall, rain, snow, fp,
     canliq, canice, tv, sfctmp, tg, ch2op) = case[:18]
    words = [
        "PRCP", cid, "3", "5", "11", str(ist),
        h(dt), h(uu), h(vv), h(elai), h(esai), h(fveg), h(bdfall),
        h(rain), h(snow), h(fp), h(canliq), h(canice), h(tv), h(sfctmp),
        h(tg), h(ch2op), h(vp.TFRZ),
    ]
    return " ".join(words)


def emit_cases(which: str) -> str:
    lines = []
    if which in ("phen", "all"):
        lines += [phen_line(c) for c in PHEN_CASES]
    if which in ("prcp", "all"):
        lines += [prcp_line(c) for c in PRCP_CASES]
    if which == "negctl":
        for row in PHEN_NEGCTL:
            cid, dveg, vegname, vegtyp, yearlen, julian, lat, snowh, tv, li, si, _ = row
            fake = (cid, vegname, vegtyp, yearlen, julian, lat, snowh, tv)
            lines.append(phen_line(fake, dveg=dveg, lai_in=li, sai_in=si))
    return "\n".join(lines) + "\n"


def run_oracle(binary: str, payload: str) -> dict[str, list[str]]:
    # ``binary`` is a command line, not just a path, so that a Windows host can
    # drive a Linux-built oracle through a launcher (e.g. "wsl -e /tmp/.../run").
    proc = subprocess.run(
        binary.split(), input=payload, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"oracle exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    out = {}
    for line in proc.stdout.strip().split("\n"):
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out


# ==========================================================================
# fixture writing
# ==========================================================================
def write_fixtures(binary: str) -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    phen_out = run_oracle(binary, emit_cases("phen"))
    with open(PHEN_CSV, "w", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([
            "case", "dveg", "croptype", "vegtyp", "yearlen", "pgs",
            "iswater", "isbarren", "isice", "urban_flag", "veg_row",
            "snowh", "tv", "lat", "julian", "troot", "lai_in", "sai_in",
            "hvt", "hvb", "tmin",
        ] + [f"laim{i:02d}" for i in range(1, 13)]
          + [f"saim{i:02d}" for i in range(1, 13)]
          + PHEN_OUT_COLS + ["binds"])
        for case in PHEN_CASES:
            cid, vegname, vegtyp, yearlen, julian, lat, snowh, tv = case[:8]
            binds = case[8]
            hvt, hvb, tmin, _c, _urban, laim, saim = VEG[vegname]
            iswater, isbarren, isice, urban = phen_indices(case)
            row = [
                cid, 4, 0, vegtyp, yearlen, 1,
                iswater, isbarren, isice, 1 if urban else 0, vegname,
                h(snowh), h(tv), h(lat), h(julian), h(280.0), h(9.99), h(8.88),
                h(hvt), h(hvb), h(tmin),
            ]
            row += [h(v) for v in laim]
            row += [h(v) for v in saim]
            row += phen_out[cid]
            row.append(binds)
            w.writerow(row)

    prcp_out = run_oracle(binary, emit_cases("prcp"))
    with open(PRCP_CSV, "w", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([
            "case", "iloc", "jloc", "vegtyp", "ist",
            "dt", "uu", "vv", "elai", "esai", "fveg", "bdfall",
            "rain", "snow", "fp", "canliq_in", "canice_in",
            "tv", "sfctmp", "tg", "ch2op",
        ] + PRCP_OUT_COLS + ["binds"])
        for case in PRCP_CASES:
            (cid, ist, dt, uu, vv, elai, esai, fveg, bdfall, rain, snow, fp,
             canliq, canice, tv, sfctmp, tg, ch2op) = case[:18]
            binds = case[18]
            row = [cid, 3, 5, 11, ist,
                   h(dt), h(uu), h(vv), h(elai), h(esai), h(fveg), h(bdfall),
                   h(rain), h(snow), h(fp), h(canliq), h(canice),
                   h(tv), h(sfctmp), h(tg), h(ch2op)]
            row += prcp_out[cid]
            row.append(binds)
            w.writerow(row)

    print(f"wrote {PHEN_CSV} ({len(PHEN_CASES)} rows)")
    print(f"wrote {PRCP_CSV} ({len(PRCP_CASES)} rows)")


# ==========================================================================
# validation
# ==========================================================================
def load_phen():
    with open(PHEN_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def load_prcp():
    with open(PRCP_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def phen_call(row, fn=vp.phenology, **override):
    kw = dict(
        dveg=int(row["dveg"]),
        vegtyp=int(row["vegtyp"]),
        croptype=int(row["croptype"]),
        snowh=uh(row["snowh"]),
        tv=uh(row["tv"]),
        lat=uh(row["lat"]),
        yearlen=int(row["yearlen"]),
        julian=uh(row["julian"]),
        lai=uh(row["lai_in"]),
        sai=uh(row["sai_in"]),
        troot=uh(row["troot"]),
        pgs=int(row["pgs"]),
        iswater=int(row["iswater"]),
        isbarren=int(row["isbarren"]),
        isice=int(row["isice"]),
        urban_flag=bool(int(row["urban_flag"])),
        hvt=uh(row["hvt"]),
        hvb=uh(row["hvb"]),
        tmin=uh(row["tmin"]),
        laim=[uh(row[f"laim{i:02d}"]) for i in range(1, 13)],
        saim=[uh(row[f"saim{i:02d}"]) for i in range(1, 13)],
    )
    kw.update(override)
    return fn(**kw)


def prcp_call(row, fn=vp.precip_heat, **override):
    kw = dict(
        iloc=int(row["iloc"]),
        jloc=int(row["jloc"]),
        vegtyp=int(row["vegtyp"]),
        ist=int(row["ist"]),
        dt=uh(row["dt"]),
        uu=uh(row["uu"]),
        vv=uh(row["vv"]),
        elai=uh(row["elai"]),
        esai=uh(row["esai"]),
        fveg=uh(row["fveg"]),
        bdfall=uh(row["bdfall"]),
        rain=uh(row["rain"]),
        snow=uh(row["snow"]),
        fp=uh(row["fp"]),
        canliq=uh(row["canliq_in"]),
        canice=uh(row["canice_in"]),
        tv=uh(row["tv"]),
        sfctmp=uh(row["sfctmp"]),
        tg=uh(row["tg"]),
        ch2op=uh(row["ch2op"]),
    )
    kw.update(override)
    return fn(**kw)


def validate(verbose: bool = True) -> int:
    bad = 0
    for label, rows, cols, caller in (
        ("PHENOLOGY", load_phen(), PHEN_OUT_COLS, phen_call),
        ("PRECIP_HEAT", load_prcp(), PRCP_OUT_COLS, prcp_call),
    ):
        worst = {c: 0 for c in cols}
        nbit = 0
        for row in rows:
            got = caller(row)
            for name in cols:
                g = getattr(got, name)
                w = uh(row[name])
                u = max_ulp([g], [w])
                worst[name] = max(worst[name], u)
                if h(g) != row[name]:
                    nbit += 1
                    bad += 1
                    print(f"  MISMATCH {label} {row['case']} {name}: "
                          f"got {h(g)} ({g!r}) want {row[name]} ({w!r})")
        if verbose:
            print(f"{label}: {len(rows)} rows, "
                  f"max_ulp {max(worst.values()) if worst else 0}, "
                  f"{nbit} non-bit-identical lanes")
            print("  per-column max_ulp: "
                  + ", ".join(f"{c}={worst[c]}" for c in cols))
    return bad


# ==========================================================================
# mutation study
# ==========================================================================
def _phen_mutant(argname):
    """A phenology that ignores ``argname`` by pinning it to a fixed value."""
    NEUTRAL = {
        "vegtyp": 0, "snowh": 0.0, "tv": 300.0, "lat": 0.0, "yearlen": 365,
        "julian": 0.0, "lai": 0.0, "sai": 0.0, "troot": 0.0, "pgs": 0,
        "iswater": -1, "isbarren": -1, "isice": -1, "urban_flag": False,
        "hvt": 0.0, "hvb": 0.0, "tmin": 0.0,
        "laim": [0.0] * 12, "saim": [0.0] * 12,
    }

    def mutant(**kw):
        kw[argname] = NEUTRAL[argname]
        return vp.phenology(**kw)
    return mutant


def _prcp_mutant(argname):
    NEUTRAL = {
        "iloc": 0, "jloc": 0, "vegtyp": 0, "ist": 1, "dt": 1.0, "uu": 0.0,
        "vv": 0.0, "elai": 0.0, "esai": 0.0, "fveg": 0.5, "bdfall": 1.0,
        "rain": 0.0, "snow": 0.0, "fp": 0.0, "canliq": 0.0, "canice": 0.0,
        "tv": 273.0, "sfctmp": 273.0, "tg": 273.0, "ch2op": 0.05,
    }
    # fveg and ch2op are pinned to legal interior values rather than 0 so that
    # the mutant is killed by producing a different number, not by dividing by
    # a zero MAXLIQ.  A mutant that only ever raises is weaker evidence.

    def mutant(**kw):
        kw[argname] = NEUTRAL[argname]
        return vp.precip_heat(**kw)
    return mutant


def mutation_study() -> int:
    survivors = []
    print("=== mutation study: one mutant per argument, argument ignored ===")

    phen_rows = load_phen()
    for arg in ("vegtyp", "snowh", "tv", "lat", "yearlen", "julian", "lai",
                "sai", "troot", "pgs", "iswater", "isbarren", "isice",
                "urban_flag", "hvt", "hvb", "tmin", "laim", "saim"):
        killed_by = None
        for row in phen_rows:
            try:
                got = phen_call(row, fn=_phen_mutant(arg))
            except Exception:
                killed_by = row["case"] + " (raised)"
                break
            if any(h(getattr(got, c)) != row[c] for c in PHEN_OUT_COLS):
                killed_by = row["case"]
                break
        status = f"killed by {killed_by}" if killed_by else "SURVIVED"
        print(f"  PHENOLOGY   drop {arg:<12s} -> {status}")
        if killed_by is None:
            survivors.append(("PHENOLOGY", arg))

    prcp_rows = load_prcp()
    for arg in ("iloc", "jloc", "vegtyp", "ist", "dt", "uu", "vv", "elai",
                "esai", "fveg", "bdfall", "rain", "snow", "fp", "canliq",
                "canice", "tv", "sfctmp", "tg", "ch2op"):
        killed_by = None
        for row in prcp_rows:
            try:
                got = prcp_call(row, fn=_prcp_mutant(arg))
            except Exception:
                killed_by = row["case"] + " (raised)"
                break
            if any(h(getattr(got, c)) != row[c] for c in PRCP_OUT_COLS):
                killed_by = row["case"]
                break
        status = f"killed by {killed_by}" if killed_by else "SURVIVED"
        print(f"  PRECIP_HEAT drop {arg:<12s} -> {status}")
        if killed_by is None:
            survivors.append(("PRECIP_HEAT", arg))

    print()
    print(f"survivors: {len(survivors)}")
    for leaf, arg in survivors:
        print(f"  {leaf}.{arg}")
    return len(survivors)


# ==========================================================================
# libm sweep -- independent re-verification of the inherited transcription
# ==========================================================================
def libm_sweep_dump(n: int) -> None:
    """Print 'E <hex>' / 'P <hexbase> <hexexp>' lines for a C driver to answer."""
    seen_e, seen_p = [], []
    for row in load_prcp():
        r = prcp_call(row)
        # the arguments the leaf actually forms
        elai, esai = uh(row["elai"]), uh(row["esai"])
        lsai = f32(elai + esai)
        if lsai > 0.0:
            fveg, ch2op = uh(row["fveg"]), uh(row["ch2op"])
            dt, rain, snow = uh(row["dt"]), uh(row["rain"]), uh(row["snow"])
            bdfall = uh(row["bdfall"])
            maxliq = f32(f32(fveg * ch2op) * lsai)
            maxsno = f32(f32(f32(fveg * f32(6.6))
                             * f32(f32(0.27) + f32(f32(46.0) / bdfall))) * lsai)
            seen_e.append(f32(-f32(f32(rain * dt) / maxliq)))
            seen_e.append(f32(-f32(f32(snow * dt) / maxsno)))
        seen_p.append(r.fwet)
    for row in load_phen():
        hvt, snowh = uh(row["hvt"]), uh(row["snowh"])
        if 0.0 < hvt <= 1.0:
            seen_e.append(f32(-f32(snowh / f32(0.2))))

    rng = random.Random(20260725)
    for _ in range(n):
        seen_e.append(f32(-80.0 + 80.0 * rng.random()))
        seen_p.append(f32(rng.random()))

    lines = [f"E {h(v)}" for v in seen_e]
    lines += [f"P {h(v)} {h(f32(0.667))}" for v in seen_p]
    sys.stdout.write("\n".join(lines) + "\n")


def libm_sweep_check(path: str) -> int:
    bad = 0
    total = 0
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            total += 1
            if parts[0] == "E":
                got, want = expf(uh(parts[1])), uh(parts[2])
            else:
                got, want = powf(uh(parts[1]), uh(parts[2])), uh(parts[3])
            if h(got) != h(want):
                bad += 1
                if bad <= 5:
                    print(f"  libm mismatch: {line.strip()} -> {h(got)}")
    print(f"libm sweep: {total} arguments, {bad} mismatches against live glibc")
    return bad


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-cases", choices=["phen", "prcp", "all", "negctl"])
    ap.add_argument("--oracle")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--libm-sweep-dump", type=int)
    ap.add_argument("--libm-sweep-check")
    args = ap.parse_args(argv)

    rc = 0
    if args.emit_cases:
        sys.stdout.write(emit_cases(args.emit_cases))
    if args.oracle:
        write_fixtures(args.oracle)
    if args.validate:
        rc |= 1 if validate() else 0
    if args.mutants:
        mutation_study()
    if args.libm_sweep_dump:
        libm_sweep_dump(args.libm_sweep_dump)
    if args.libm_sweep_check:
        rc |= 1 if libm_sweep_check(args.libm_sweep_check) else 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
