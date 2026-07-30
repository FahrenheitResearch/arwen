#!/usr/bin/env python3
"""Mutation study for the Noah-MP NOAHMP_SFLX composition.

``max_ulp 0`` on four columns is not evidence that a composition is right.  A
composition's whole job is plumbing -- forwarding an argument, holding a value
across a call, taking a branch -- and plumbing is easy to get wrong in ways
that four well-behaved columns cannot see.  A sibling lane in this project
reached ``max_ulp 0`` on 29 columns and then found that 13 of 14 argument-drop
mutants still reproduced its pinned CSV.

Three families of mutant are generated against ``gpuwm/core/noahmp_sflx.py``:

*argument mutants*
    One per argument :func:`sflx_pre`, :func:`sflx_post` and :func:`error`
    actually consume, plus one per ``SflxParameters`` component the composition
    reads for itself.  The mutant overwrites the argument with a fixed,
    physically plausible value at the top of the routine, so the routine still
    runs but can no longer see what the caller passed.

*structure mutants*
    NOAHMP_SFLX's own thirty statements and ERROR's nine, each broken in the
    one way a careless transcription would break it: a dropped term, a flipped
    comparison, a reordered sum, a loop bound taken from the wrong variable, a
    MIN written where WRF wrote MAX.

*constant mutants*
    One per ``_f(<literal>)`` site, perturbed by a relative 1e-3 -- large
    enough that FP32 cancellation cannot swallow it, small enough that no
    branch flips for the wrong reason.

Every mutant is run through ``tests/test_noahmp_sflx.py``, which is both the
whole-column gate and ERROR's own oracle.  Survivors are printed; each one has
to be argued *unreachable*, not merely listed, and a survivor that is not in
``EXPECTED_SURVIVORS`` fails the run.

Usage::

    python3 mutation_study_sflx.py [--quick] [--filter SUBSTRING]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "gpuwm" / "core" / "noahmp_sflx.py"
TEST = REPO / "tests" / "test_noahmp_sflx.py"

# The statement immediately after which an override can be injected without
# disturbing the transcription: everything above each anchor is validation or
# type coercion.
ANCHOR_PRE = "    isnow = int(col.isnow)"
ANCHOR_POST = "    ist = int(ist)"
ANCHOR_ERR = "    dzsnso = np.asarray(dzsnso, dtype=np.float32)"

SCALAR = "_f(0.137)"

ARG_MUTANTS: list[tuple[str, str, str]] = []      # (label, anchor, statement)


def _scalars(anchor, names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append((n, anchor, f"{n} = {value}"))


def _arrays(anchor, names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append(
            (n, anchor,
             f"{n} = np.full(np.shape({n}), {value}, dtype=np.float32)"))


# --- sflx_pre ---------------------------------------------------------------
_scalars(ANCHOR_PRE, [
    "lat", "julian", "cosz", "dt", "dx", "dz8w", "shdmax", "sfctmp", "sfcprs",
    "psfc", "uu", "vv", "q2", "qc", "soldn", "lwdn", "prcpconv", "prcpnonc",
    "prcpshcv", "tbot", "co2air", "o2air", "foln", "zlvl", "albold", "sneqvo",
    "tah", "eah", "canliq", "canice", "tv", "tg", "qsfc", "lai", "sai", "cm",
    "ch", "tauss", "wa", "acc_ssoil",
])
_arrays(ANCHOR_PRE, ["zsoil"])
ARG_MUTANTS.append(("yearlen", ANCHOR_PRE, "yearlen = 366"))
ARG_MUTANTS.append(("col.zsnso", ANCHOR_PRE,
                    "col = _mut_col(col, 'zsnso', -0.137)"))
ARG_MUTANTS.append(("col.stc", ANCHOR_PRE,
                    "col = _mut_col(col, 'stc', 281.37)"))
ARG_MUTANTS.append(("col.snowh", ANCHOR_PRE, "col = _mut_col(col, 'snowh', 0.137)"))
ARG_MUTANTS.append(("col.sneqv", ANCHOR_PRE, "col = _mut_col(col, 'sneqv', 13.7)"))
ARG_MUTANTS.append(("col.isnow", ANCHOR_PRE, "isnow = 0"))
ARG_MUTANTS.append(("smc(pre)", ANCHOR_PRE,
                    "smc = np.full(np.shape(smc), _f(0.237), dtype=np.float32)"))
ARG_MUTANTS.append(("parameters.nroot", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'nroot', 2)"))
ARG_MUTANTS.append(("parameters.isbarren", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'isbarren', 99)"))
ARG_MUTANTS.append(("parameters.urban_flag", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'urban_flag', True)"))
ARG_MUTANTS.append(("parameters.ch2op", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'ch2op', _f(0.137))"))
ARG_MUTANTS.append(("parameters.hvt", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'hvt', _f(13.7))"))
ARG_MUTANTS.append(("parameters.hvb", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'hvb', _f(0.137))"))
ARG_MUTANTS.append(("parameters.tmin", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'tmin', _f(200.0))"))
ARG_MUTANTS.append(("parameters.laim", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'laim',"
                    " np.full(12, _f(1.37), dtype=np.float32))"))
ARG_MUTANTS.append(("parameters.saim", ANCHOR_PRE,
                    "parameters = _mut_p(parameters, 'saim',"
                    " np.full(12, _f(0.37), dtype=np.float32))"))

# --- sflx_post --------------------------------------------------------------
_scalars(ANCHOR_POST, ["dt"])
_arrays(ANCHOR_POST, ["zsoil", "ficeold", "acc_etrani"])
_scalars(ANCHOR_POST, ["wa", "wslake", "acc_qinsur", "acc_qseva",
                       "acc_dwater", "acc_prcp", "acc_ecan", "acc_etran",
                       "acc_edir"])
ARG_MUTANTS.append(("en.ponding", ANCHOR_POST, "en = _mut_e(en, 'ponding', 1.37)"))
ARG_MUTANTS.append(("en.latheag", ANCHOR_POST,
                    "en = _mut_e(en, 'latheag', 2.5104e6)"))
ARG_MUTANTS.append(("en.fgev", ANCHOR_POST, "en = _mut_e(en, 'fgev', 13.7)"))
ARG_MUTANTS.append(("en.btrani", ANCHOR_POST,
                    "en = _mut_e(en, 'btrani', (0.37,) * col.nsoil)"))
ARG_MUTANTS.append(("en.imelt", ANCHOR_POST,
                    "en = _mut_e(en, 'imelt', (2,) * (col.nsnow + col.nsoil))"))
ARG_MUTANTS.append(("en.frozen_ground", ANCHOR_POST,
                    "en = _mut_e(en, 'frozen_ground', False)"))
ARG_MUTANTS.append(("en.frozen_canopy", ANCHOR_POST,
                    "en = _mut_e(en, 'frozen_canopy', False)"))
ARG_MUTANTS.append(("pre.fveg", ANCHOR_POST, "pre = _mut_x(pre, 'fveg', 0.37)"))
ARG_MUTANTS.append(("pre.beg_wb", ANCHOR_POST,
                    "pre = _mut_x(pre, 'beg_wb', 137.0)"))
ARG_MUTANTS.append(("pre.prcp", ANCHOR_POST, "pre = _mut_x(pre, 'prcp', 1.37e-4)"))
ARG_MUTANTS.append(("pre.swdown", ANCHOR_POST,
                    "pre = _mut_x(pre, 'swdown', 137.0)"))
ARG_MUTANTS.append(("pre.qrain", ANCHOR_POST, "pre = _mut_x(pre, 'qrain', 1.37e-4)"))
ARG_MUTANTS.append(("pre.qsnow", ANCHOR_POST, "pre = _mut_x(pre, 'qsnow', 1.37e-4)"))
ARG_MUTANTS.append(("pre.snowhin", ANCHOR_POST,
                    "pre = _mut_x(pre, 'snowhin', 1.37e-6)"))
ARG_MUTANTS.append(("pre.bdfall", ANCHOR_POST, "pre = _mut_x(pre, 'bdfall', 113.7)"))
ARG_MUTANTS.append(("pre.elai", ANCHOR_POST, "pre = _mut_x(pre, 'elai', 1.37)"))
ARG_MUTANTS.append(("pre.esai", ANCHOR_POST, "pre = _mut_x(pre, 'esai', 0.37)"))
ARG_MUTANTS.append(("pre.canliq", ANCHOR_POST, "pre = _mut_x(pre, 'canliq', 0.137)"))
ARG_MUTANTS.append(("pre.canice", ANCHOR_POST, "pre = _mut_x(pre, 'canice', 0.137)"))
ARG_MUTANTS.append(("pre.dzsnso", ANCHOR_POST,
                    "pre = _mut_x(pre, 'dzsnso',"
                    " np.full(np.shape(pre.dzsnso), _f(0.37), dtype=np.float32))"))

# --- error ------------------------------------------------------------------
_scalars(ANCHOR_ERR, [
    "swdown", "fsa", "fsr", "fira", "fsh", "fcev", "fgev", "fctr", "ssoil",
    "sav", "sag", "beg_wb", "canliq", "canice", "sneqv", "wa", "prcp", "ecan",
    "etran", "edir", "runsrf", "runsub", "dt", "qtldrn", "pah", "firr",
    "canhs", "irmirate", "irfirate", "acc_dwater", "acc_prcp", "acc_ecan",
    "acc_etran", "acc_edir",
])
_arrays(ANCHOR_ERR, ["smc", "dzsnso"])
ARG_MUTANTS.append(("error/ist", ANCHOR_ERR, "ist = 1"))
ARG_MUTANTS.append(("error/calculate_soil", ANCHOR_ERR,
                    "calculate_soil = True"))


# ---------------------------------------------------------------------------
# structure mutants
# ---------------------------------------------------------------------------
STRUCT_MUTANTS: list[tuple[str, str, str]] = [
    # -- :827-833  DZSNSO ---------------------------------------------------
    ("827 loop starts at 1, not ISNOW+1",
     "    for iz in range(isnow + 1, nsoil + 1):                          # :827",
     "    for iz in range(1, nsoil + 1):                          # :827"),
    ("828 top-layer special case dropped",
     "        if iz == isnow + 1:                                         # :828",
     "        if False:                                         # :828"),
    ("829 top layer keeps ZSNSO's sign",
     "            DZ[iz] = _f(-ZS[iz])                                    # :829",
     "            DZ[iz] = _f(ZS[iz])                                    # :829"),
    ("831 difference taken the other way",
     "            DZ[iz] = _f(ZS[iz - 1] - ZS[iz])                        # :831",
     "            DZ[iz] = _f(ZS[iz] - ZS[iz - 1])                        # :831"),
    # -- :837-839  TROOT ----------------------------------------------------
    ("839 TROOT divides by ZSOIL(1), not ZSOIL(NROOT)",
     "    denom = _f(-zsoil[nroot - 1])",
     "    denom = _f(-zsoil[0])"),
    # -- :845-847  BEG_WB ---------------------------------------------------
    ("845 BEG_WB drops the aquifer term",
     "        beg_wb = _f(_f(_f(_f(canliq) + _f(canice)) + _f(col.sneqv))\n"
     "                    + _f(wa))                                       # :845",
     "        beg_wb = _f(_f(_f(canliq) + _f(canice)) + _f(col.sneqv))     # :845"),
    ("847 BEG_WB layer term loses the 1000",
     "            beg_wb = _f(beg_wb\n"
     "                        + _f(_f(smc[iz - 1] * DZ[iz]) * _THOUSAND))  # :847",
     "            beg_wb = _f(beg_wb + _f(smc[iz - 1] * DZ[iz]))  # :847"),
    ("844 BEG_WB computed for a lake column too",
     "    if int(ist) == 1:                                               # :844",
     "    if True:                                               # :844"),
    # -- :863-875  FVEG -----------------------------------------------------
    ("863 FVEG takes SHDFAC's place from SHDMAX",
     "    fveg = _f(shdmax)                                               # :863",
     "    fveg = _f(0.5)                                               # :863"),
    ("864 FVEG floor becomes a ceiling",
     "    if fveg <= _FVEG_FLOOR:                                         # :864",
     "    if fveg >= _FVEG_FLOOR:                                         # :864"),
    ("874 ISBARREN zeroing dropped",
     "    if parameters.urban_flag or vegtyp == parameters.isbarren:      # :874",
     "    if parameters.urban_flag:      # :874"),
    ("874 urban zeroing dropped",
     "    if parameters.urban_flag or vegtyp == parameters.isbarren:      # :874\n"
     "        fveg = _ZERO",
     "    if vegtyp == parameters.isbarren:      # :874\n"
     "        fveg = _ZERO"),
    ("875 bare-canopy zeroing dropped",
     "    if _f(elai + esai) == _ZERO:                                    # :875",
     "    if False:                                    # :875"),
    # -- :979-984  SICE, QVAP, QDEW, EDIR ------------------------------------
    ("979 SICE clamp dropped",
     "        sice[iz] = _fmax(_ZERO, _f(smc[iz] - col.sh2o[iz]))          # :979",
     "        sice[iz] = _f(smc[iz] - col.sh2o[iz])          # :979"),
    ("979 SICE difference reversed",
     "        sice[iz] = _fmax(_ZERO, _f(smc[iz] - col.sh2o[iz]))          # :979",
     "        sice[iz] = _fmax(_ZERO, _f(col.sh2o[iz] - smc[iz]))          # :979"),
    ("980 SNEQVO taken before WATER, not after ENERGY",
     "    sneqvo = _f(col.sneqv)                                           # :980",
     "    sneqvo = _f(en.sneqvo)                                           # :980"),
    ("982 QVAP loses its clamp",
     "    qvap = _fmax(ratio, _ZERO)                                       # :982",
     "    qvap = ratio                                       # :982"),
    ("983 QDEW loses its absolute value",
     "    qdew = abs(_fmin(ratio, _ZERO))                                  # :983",
     "    qdew = _fmin(ratio, _ZERO)                                  # :983"),
    ("984 EDIR is the sum, not the difference",
     "    edir = _f(qvap - qdew)                                           # :984",
     "    edir = _f(qvap + qdew)                                           # :984"),
    ("982 latent heat taken from the canopy leg",
     "    ratio = _f(_f(en.fgev) / _f(en.latheag))",
     "    ratio = _f(_f(en.fgev) / _f(en.latheav))"),
    # -- :1067-1076  the tail ------------------------------------------------
    ("1067 snow clamp becomes AND",
     "    if snowh <= _SNOW_EPS or sneqv <= _SNOW_EPS:                     # :1067",
     "    if snowh <= _SNOW_EPS and sneqv <= _SNOW_EPS:                     # :1067"),
    ("1067 snow clamp gate becomes <",
     "    if snowh <= _SNOW_EPS or sneqv <= _SNOW_EPS:                     # :1067",
     "    if snowh < _SNOW_EPS or sneqv < _SNOW_EPS:                     # :1067"),
    ("1072 albedo sentinel gate becomes > 0",
     "    if pre.swdown != _ZERO:                                          # :1072",
     "    if pre.swdown > _ZERO:                                          # :1072"),
    ("1073 albedo divides SWDOWN by FSR",
     "        albedo = _f(_f(en.fsr) / pre.swdown)                         # :1073",
     "        albedo = _f(pre.swdown / _f(en.fsr))                         # :1073"),
    ("1073 albedo uses FSA, not FSR",
     "        albedo = _f(_f(en.fsr) / pre.swdown)                         # :1073",
     "        albedo = _f(_f(en.fsa) / pre.swdown)                         # :1073"),
    ("1062 urban QSFC override applied unconditionally",
     "    if parameters.urban_flag:                                        # :1062",
     "    if True:                                        # :1062"),
    # -- ERROR ---------------------------------------------------------------
    ("1638 ERRSW adds FSA and FSR to SWDOWN",
     "    errsw = _f(swdown - _f(fsa + fsr))",
     "    errsw = _f(swdown + _f(fsa + fsr))"),
    ("1641 shortwave gate uses the energy tolerance",
     "    if abs(errsw) > ERRSW_TOL:                                      # :1641",
     "    if abs(errsw) > _f(100.0):                                      # :1641"),
    ("1662 ERRENG drops CANHS",
     "    sink = _f(sink + _f(canhs))",
     "    sink = _f(sink + _ZERO)"),
    ("1662 ERRENG drops FIRR",
     "    sink = _f(sink + _f(firr))",
     "    sink = _f(sink + _ZERO)"),
    ("1662 ERRENG subtracts PAH",
     "    erreng = _f(_f(_f(_f(sav) + _f(sag)) - sink) + _f(pah))",
     "    erreng = _f(_f(_f(_f(sav) + _f(sag)) - sink) - _f(pah))"),
    ("1665 energy gate widened",
     "    if abs(erreng) > ERRENG_TOL:                                    # :1665",
     "    if abs(erreng) > _f(100.0):                                    # :1665"),
    ("1696 lake branch taken for soil columns",
     "    if int(ist) == 1:                                               # :1696  soil",
     "    if int(ist) == 2:                                               # :1696  soil"),
    ("1697 END_WB drops SNEQV",
     "        end_wb = _f(_f(_f(_f(canliq) + _f(canice)) + _f(sneqv)) + _f(wa))",
     "        end_wb = _f(_f(_f(canliq) + _f(canice)) + _f(wa))"),
    ("1702 ACC_DWATER adds END_WB instead of the difference",
     "        acc_dwater = _f(acc_dwater + _f(end_wb - beg_wb))            # :1702",
     "        acc_dwater = _f(acc_dwater + _f(end_wb + beg_wb))            # :1702"),
    ("1703 ACC_PRCP not scaled by DT",
     "        acc_prcp = _f(acc_prcp + _f(prcp * dt))                      # :1703",
     "        acc_prcp = _f(acc_prcp + prcp)                      # :1703"),
    ("1706 ACC_EDIR overwritten instead of accumulated",
     "        acc_edir = _f(acc_edir + _f(edir * dt))                      # :1706",
     "        acc_edir = _f(edir * dt)                      # :1706"),
    ("1709 IRFIRATE not scaled by 1000",
     "            inner = _f(acc_prcp + _f(_f(irfirate) * _THOUSAND))       # :1709",
     "            inner = _f(acc_prcp + _f(irfirate))       # :1709"),
    ("1709 IRMIRATE subtracted, not added",
     "            inner = _f(inner + _f(_f(irmirate) * _THOUSAND))",
     "            inner = _f(inner - _f(_f(irmirate) * _THOUSAND))"),
    ("1710 RUNSUB dropped from the residual",
     "            inner = _f(inner - runsub)",
     "            inner = _f(inner - _ZERO)"),
    ("1710 QTLDRN dropped from the residual",
     "            inner = _f(inner - qtldrn)",
     "            inner = _f(inner - _ZERO)"),
    ("1710 ERRWAT is the sum, not the difference",
     "            errwat = _f(acc_dwater - inner)",
     "            errwat = _f(acc_dwater + inner)"),
    ("1713 water gate widened",
     "            if abs(errwat) > ERRWAT_TOL:                             # :1713",
     "            if abs(errwat) > _f(100.0):                             # :1713"),
    ("1708 calculate_soil gate inverted",
     "        if calculate_soil:                                           # :1708",
     "        if not calculate_soil:                                           # :1708"),
    ("1734 lake branch reports the residual instead of zero",
     "        errwat = _ZERO                                               # :1734",
     "        errwat = _f(1.0)                                               # :1734"),
]


#: Mutants that change no value NOAHMP_SFLX can ever produce.  Killing one of
#: these would require the port to be *wrong*, so their survival is a property
#: of WRF and not a weakness of the fixture.
EQUIVALENT: dict[str, str] = {
    "arg/lai":
        "PHENOLOGY assigns LAI from the monthly table at 1301 before any read, "
        "on every DVEG in {1,3,4}.  The only reader of the entry value is the "
        "DVEG 7/8/9 'use input LAI' block at 1310-1316, which the pinned "
        "identity does not admit and gpuwm.core.noahmp_vegprecip refuses.",
    "arg/sai":
        "SAI, for the same reason as LAI: PHENOLOGY assigns it at 1302.",
    "arg/en.frozen_ground":
        "FROZEN_GROUND gates WATER's SICE(1) exchange at 6145-6153, whose only "
        "effect is `SICE(1) += (QSDEW-QSEVA)*DT/(DZ(1)*1000)`.  On the one "
        "column of the four that has FROZEN_GROUND true (snowpack_frozen_soil) "
        "SNEQV is 50 mm, so QSNSUB takes all of QVAP at 6128 and QSEVA is "
        "exactly 0.0; QDEW is 0.0 so QSNFRO takes it and QSDEW is exactly 0.0.  "
        "The increment is exactly zero, and the gate is unobservable.  "
        "gpuwm/data/noahmp/oracle/noahmp-water.csv constrains the same "
        "statement on a column where it is not.",
    "arg/ficeold":
        "FICEOLD reaches SNOWWATER's COMPACT and is read only by the melt "
        "metamorphism term, inside `IF (IMELT(J) == 1 .AND. ...)`.  Three of "
        "the four columns have ISNOW = 0, so COMPACT's loop is empty; the "
        "fourth is accumulating at SFCTMP = 263 K and no layer is melting.  "
        "The WATER lane added wat_melting_pack_ficeold to noahmp-water.csv "
        "precisely because its own columns had the same hole.",
    "arg/en.imelt":
        "IMELT reaches the same COMPACT gate as FICEOLD and is inert for the "
        "same reason; see that entry.",
}

#: Mutants that DO change behaviour, on states the four pinned columns never
#: reach.  These are not equivalences: each is a measured gap in what
#: noahmp-sflx.csv can constrain, and each names the column that would close
#: it.  They are listed so the gap is a recorded fact rather than a silent one.
UNCOVERED: dict[str, str] = {
    "struct/844 BEG_WB computed for a lake column too":
        "Every noahmp-sflx.csv column is IST = 1.  Closing it needs an IST = 2 "
        "column; noahmp-sflx-error.csv covers ERROR's own lake branch but "
        "BEG_WB is computed in NOAHMP_SFLX, above ERROR.",
    "struct/874 ISBARREN zeroing dropped":
        "The only column that reaches 874 is bare_thin_snow_melt, VEGTYP = 16, "
        "and MPTABLE gives category 16 LAIM and SAIM identically zero.  "
        "PHENOLOGY therefore returns ELAI = ESAI = 0 and 875 zeroes FVEG "
        "anyway, so 874 and 875 mask each other.  Closing it needs a barren or "
        "urban column with a non-zero monthly LAI.",
    "struct/874 urban zeroing dropped":
        "No column is urban.  noahmp-energy.csv case urban_snowfree exercises "
        "the flag inside ENERGY, but ENERGY's own FVEG is an input, so that "
        "case says nothing about 874.",
    "struct/875 bare-canopy zeroing dropped":
        "The converse of the 874 entry: on the one column where ELAI+ESAI is "
        "zero, VEGTYP == ISBARREN has already zeroed FVEG.",
    "arg/parameters.isbarren":
        "Same cluster: with category 16's LAIM identically zero, moving "
        "ISBARREN moves neither PHENOLOGY's zeroing nor 874's.",
    "const/line185/0.05":
        "_FVEG_FLOOR.  bare_thin_snow_melt is the only column with "
        "SHDMAX <= 0.05, and its FVEG is zeroed by 874/875 two lines later, so "
        "the floor's value is never observable.",
    "struct/979 SICE clamp dropped":
        "SMC >= SH2O on every layer of every column at 979, so MAX(0, .) is "
        "the identity.  PHASECHANGE is what maintains that, which makes the "
        "clamp defensive; the fixture cannot show it is needed.",
    "struct/982 QVAP loses its clamp":
        "FGEV is 25.93, 2.87, 0.13 and 8.56 W/m2 on the four columns, all "
        "positive, so the ground never dews and QVAP == FGEV/LATHEAG on all of "
        "them.  Closing this, and the two entries below, needs one column with "
        "FGEV < 0.",
    "struct/983 QDEW loses its absolute value":
        "QDEW is exactly 0.0 on all four columns; see the QVAP entry.",
    "struct/984 EDIR is the sum, not the difference":
        "EDIR = QVAP - QDEW and QDEW is 0.0 on all four columns; see the QVAP "
        "entry.",
    "struct/982 latent heat taken from the canopy leg":
        "LATHEAV == LATHEAG bit for bit on all four columns -- 2.5104e6 on "
        "three and 2.844e6 on snowpack_frozen_soil.  They differ only when "
        "FROZEN_CANOPY and FROZEN_GROUND disagree, i.e. TV and TG straddle "
        "TFRZ, which no column here does.",
    "struct/1067 snow clamp becomes AND":
        "The four columns leave (SNOWH, SNEQV) at (0, 0), (0, 0), (0.2, 50) "
        "and (0.0164, 4.098): never one above the 1e-6 threshold and the other "
        "below, which is the only state that separates OR from AND.",
    "struct/1067 snow clamp gate becomes <":
        "No column leaves SNOWH or SNEQV exactly at 1e-6, so <= and < agree.",
    "const/line186/1e-06":
        "_SNOW_EPS, for the same reason as the two 1067 entries: every column "
        "is far from the threshold on both sides.",
    "struct/1072 albedo sentinel gate becomes > 0":
        "SWDOWN is 820, 0, 190 and 610 W/m2, so `/= 0` and `> 0` can only "
        "differ on a negative SWDOWN.  ATM emits either 0.0 (1152-1156) or "
        "SOLDN, so that needs a negative downward shortwave forcing.",
}

EXPECTED_SURVIVORS: dict[str, str] = {**EQUIVALENT, **UNCOVERED}


_HELPER = '''

# --- mutation-study helpers, appended by mutation_study_sflx.py -------------
def _mut_p(p, name, value):
    import copy
    q = copy.copy(p)
    object.__setattr__(q, name, value)
    return q


def _mut_e(e, name, value):
    import copy
    q = copy.copy(e)
    object.__setattr__(q, name, value)
    return q


def _mut_x(x, name, value):
    import copy
    q = copy.copy(x)
    object.__setattr__(q, name, value)
    return q


def _mut_col(c, name, value):
    q = c.copy()
    cur = getattr(q, name)
    if isinstance(cur, np.ndarray):
        setattr(q, name, np.full(cur.shape, np.float32(value),
                                 dtype=np.float32))
    else:
        setattr(q, name, np.float32(value))
    return q
'''


def build_arg_mutant(text: str, anchor: str, statement: str) -> str:
    if text.count(anchor) != 1:
        raise SystemExit(f"anchor is not unique: {anchor!r}")
    return text.replace(anchor, anchor + "\n    " + statement, 1) + _HELPER


def build_struct_mutant(text: str, needle: str, replacement: str) -> str:
    if needle not in text:
        raise SystemExit(f"structure mutant needle not found:\n{needle}")
    if text.count(needle) != 1:
        raise SystemExit(f"structure mutant needle is not unique:\n{needle}")
    return text.replace(needle, replacement, 1)


_LITERAL = re.compile(r"_f\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)")


def constant_sites(text: str):
    out = []
    for m in _LITERAL.finditer(text):
        value = float(m.group(1))
        if value == 0.0:
            continue
        line = text.count("\n", 0, m.start()) + 1
        out.append((line, m.start(), m.end(), value))
    return out


def build_const_mutant(text: str, site) -> str:
    _line, start, end, value = site
    return text[:start] + f"_f({value * 1.001!r})" + text[end:]


def run_tests(quick: bool) -> bool:
    cmd = [sys.executable, "-m", "pytest", str(TEST), "-q", "--no-header",
           "-p", "no:cacheprovider", "-p", "no:randomly"]
    if quick:
        cmd.append("-x")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return proc.returncode == 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--filter", default="")
    args = ap.parse_args(argv)

    original = SOURCE.read_text()
    if not run_tests(args.quick):
        raise SystemExit("the unmutated port already fails; fix that first")

    survivors: list[str] = []
    total = 0
    try:
        for label, anchor, statement in ARG_MUTANTS:
            name = f"arg/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            SOURCE.write_text(build_arg_mutant(original, anchor, statement))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}", flush=True)
            else:
                print(f"killed    {name}", flush=True)

        for label, needle, replacement in STRUCT_MUTANTS:
            name = f"struct/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            SOURCE.write_text(build_struct_mutant(original, needle, replacement))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}", flush=True)
            else:
                print(f"killed    {name}", flush=True)

        for site in constant_sites(original):
            name = f"const/line{site[0]}/{site[3]!r}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            SOURCE.write_text(build_const_mutant(original, site))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}", flush=True)
            else:
                print(f"killed    {name}", flush=True)
    finally:
        SOURCE.write_text(original)

    equivalent = [s for s in survivors if s in EQUIVALENT]
    uncovered = [s for s in survivors if s in UNCOVERED]
    print(f"\n{total - len(survivors)} of {total} mutants killed; "
          f"{len(equivalent)} equivalent, {len(uncovered)} not covered by the "
          "four pinned columns")
    unexpected = [s for s in survivors if s not in EXPECTED_SURVIVORS]
    # A filtered run did not attempt most mutants, so it cannot say which
    # entries have gone stale.  Reporting them anyway would be noise that
    # trains the reader to ignore the section.
    stale = ([] if args.filter
             else [s for s in EXPECTED_SURVIVORS if s not in survivors])
    for s in equivalent:
        print(f"equivalent mutant  {s}\n    {EQUIVALENT[s]}")
    for s in uncovered:
        print(f"NOT COVERED, and this is a gap not an equivalence  {s}\n"
              f"    {UNCOVERED[s]}")
    if unexpected:
        print("UNEXPECTED survivors, each of which must be argued unreachable:")
        for s in unexpected:
            print(f"  {s}")
    if stale:
        print("stale EXPECTED_SURVIVORS entries (these are now killed):")
        for s in stale:
            print(f"  {s}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
