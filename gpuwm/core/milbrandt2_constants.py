# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   The GAMMLN log-gamma coefficients and their evaluation, taken from WRF
#   v4.6.1 phys/module_mp_thompson.F:5325-5347, which preserves the notice
#   immediately above and below that routine:
#
#       (C) Copr. 1986-92 Numerical Recipes Software 2.02
#
#   ArWen's position on this material -- taken from WRF, reproduced in
#   order to match WRF bit for bit, and the standard Lanczos g=5 n=6
#   coefficient set -- is set out in
#   licenses/NOTICE-Numerical-Recipes.txt.  The notice is preserved here
#   because WRF preserved it.
# ======================================================================
"""Milbrandt-Yau 2-moment (WRF ``mp_physics=9``) first-call constants.

WRF v4.6.1 ``phys/module_mp_milbrandt2mom.F`` computes its SAVE-attributed
distribution constants inside ``mp_milbrandt2mom_main`` on every call
(:1257-1438, the ``if (.TRUE.)`` block that the source comments as
"need only to be computed once per model integration").  Every one of
those values depends only on compile-time ``parameter`` declarations --
the shape parameters ``alpha_x``/``MUc``, the fall-speed pairs
``afx``/``bfx``, and the mass-diameter pairs ``cmx``/``dmx`` -- so gpuwm
hoists the whole block here, evaluates it once at import, and hands the
kernel a read-only float32 vector.  ``idt = 1./dt`` (:1269) is the single
dt-dependent entry of WRF's SAVE list and is therefore computed inside
the kernel from the launch dt instead of living in this table.

FIDELITY NOTES
  * ``gamma`` here is the scheme's OWN gamma (:160-195): the Numerical
    Recipes Lanczos form deliberately truncated to FOUR series terms
    (``do j=1,4``, :184), evaluated in float64 and returned as float32
    exactly as the Fortran function result conversion does.  Using a true
    gamma would silently reconstitute the accuracy the scheme's author
    removed; the 4-term values are what every WRF mp=9 run integrates
    with.
  * Each assignment chains in float32 (``np.float32`` arithmetic) in the
    Fortran statement order, because the SAVE variables are default REAL
    and the expressions round at every operation.
  * ``snowSpherical = .false.`` (:1174) and ``CCNtype = 2`` (driver
    :3615, continental) are WRF's hard-coded settings for mp=9; the
    dependent branches (:1285-1302, :1323-1331) are resolved here the
    same way.

The exported ``CK`` vector is indexed by the ``CK_INDEX`` mapping; the
CUDA kernel mirrors those indices with ``#define`` rows so its body keeps
the Fortran spelling (``GC13``, ``ckQr1``, ...).  The order is
load-bearing: kernels/milbrandt2.cu is generated against it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# Milbrandt-Yau option 9 carries hail mass beside graupel and a number
# moment for EVERY one of the six hydrometeors -- the WRF driver's
# CASE(MILBRANDT2MOM) arm binds qnc/qnr/qni/qns/qng/qnh
# (module_microphysics_driver.F:1857-1862) and Registry.EM_COMMON declares
# all six in the ``scalar`` package.  ``nc`` is transported, unlike
# Morrison's diagnostic droplet number, because the scheme prognoses it:
# NccnFNC activation writes it at :3014 and evaporation depletes it at
# :3022.  The tuples live here, beside the scheme's constant table and
# away from any kernel import, because the offline child's transport
# inventory and the online forcing table both read them and the offline
# door answers on a box with no working CuPy; gpuwm.core.moist re-exports
# them under the same names.
MY2_MASS_SPECIES = ("qi", "qs", "qg", "qh")
MY2_NUMBER_SPECIES = ("nc", "nr", "ni", "ns", "ng", "nh")
MY2_SPECIES = MY2_MASS_SPECIES + MY2_NUMBER_SPECIES

F = np.float32

# ---------------------------------------------------------------------------
# The scheme's own gamma (module_mp_milbrandt2mom.F:160-195): Lanczos with
# stp/cof from Numerical Recipes but the series truncated to FOUR terms.
# ---------------------------------------------------------------------------

_LANCZOS_COF = (76.18009172947146, -86.50532032941677, 24.01409824083091,
                -1.231739572450155, 0.1208650973866179e-2,
                -0.5395239384953e-5)
_LANCZOS_STP = 2.5066282746310005


def gamma_my2(xx: float) -> np.float32:
    """The scheme's 4-term Lanczos gamma, float64 core, float32 result."""
    x = float(F(xx))
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.000000000190015
    for j in range(4):                       # :184  do j=1,4  (NOT 6)
        y += 1.0
        ser += _LANCZOS_COF[j] / y
    return F(math.exp(tmp + math.log(_LANCZOS_STP * ser / x)))


def gammln_my2(xx: float) -> np.float32:
    """The scheme's 6-term ln(gamma) (:233-265); float32 result."""
    x = float(F(xx))
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.000000000190015
    for j in range(6):
        y += 1.0
        ser += _LANCZOS_COF[j] / y
    return F(tmp + math.log(_LANCZOS_STP * ser / x))


def _build() -> dict[str, np.float32]:
    g = gamma_my2
    c: dict[str, np.float32] = {}

    # -- fixed parameters entering the derived chain (:1026-1078) --
    MUc = F(3.0)
    alpha_c = F(1.0)
    alpha_r = F(0.0)
    alpha_i = F(0.0)
    alpha_s = F(0.0)
    alpha_g = F(0.0)
    alpha_h = F(0.0)
    No_r_SM = F(1.0e7)
    No_g_SM = F(4.0e6)
    No_h_SM = F(1.0e5)
    afr, bfr = F(149.100), F(0.5000)
    afi, bfi = F(71.340), F(0.6635)
    afs, bfs = F(11.720), F(0.4100)
    afg, bfg = F(19.300), F(0.3700)
    afh, bfh = F(206.890), F(0.6384)
    deg = F(400.0)
    deh = F(900.0)
    dei = F(500.0)
    dew = F(1000.0)
    desFix = F(100.0)
    Dso = F(125.0e-6)
    dmr = F(3.0)
    dmi = F(3.0)
    dmg = F(3.0)
    dmh = F(3.0)
    mgo = F(1.6e-10)
    fdielec = F(4.464)
    CHLC = F(0.2501e7)
    CHLF = F(0.334e6)
    CPD = F(0.100546e4)
    PI = F(0.314159265359e1)
    thrd = F(1.0) / F(3.0)
    sixth = F(0.5) * thrd

    # -- the :1259-1302 preamble --
    c["PI2"] = PI * F(2.0)
    c["PIov4"] = F(0.25) * PI
    c["PIov6"] = PI * sixth
    c["CHLS"] = CHLC + CHLF
    c["LCP"] = CHLC / CPD
    c["LFP"] = CHLF / CPD
    c["iCHLF"] = F(1.0) / CHLF
    c["LSP"] = c["LCP"] + c["LFP"]
    c["ck5"] = F(4098.170) * c["LCP"]
    c["ck6"] = F(5806.485) * c["LSP"]
    c["imgo"] = F(1.0) / mgo
    c["idew"] = F(1.0) / dew
    c["idei"] = F(1.0) / dei
    c["ideg"] = F(1.0) / deg
    c["ideh"] = F(1.0) / deh

    c["cmr"] = c["PIov6"] * dew
    c["icmr"] = F(1.0) / c["cmr"]
    c["cmi"] = F(440.0)
    c["icmi"] = F(1.0) / c["cmi"]
    c["cmg"] = c["PIov6"] * deg
    c["icmg"] = F(1.0) / c["cmg"]
    c["cmh"] = c["PIov6"] * deh
    c["icmh"] = F(1.0) / c["cmh"]

    c["cms_D3"] = c["PIov6"] * desFix
    # snowSpherical = .false. (:1174) selects Brandes et al. 2007 (:1290)
    cms = F(0.1597)
    dms = F(2.078)
    c["cms"] = cms
    c["dms"] = dms
    c["icms"] = F(1.0) / cms
    c["idms"] = F(1.0) / dms
    mso = cms * F(np.power(Dso, dms))
    c["mso"] = mso
    c["imso"] = F(1.0) / mso
    eds = cms / c["PIov6"]
    fds = dms - F(3.0)
    c["eds"] = eds
    c["fds"] = fds
    # :1302 -- fds /= -1 and not snowSpherical
    c["GS50"] = g(F(1.0) + fds + alpha_s)

    # -- Cloud (:1305-1321) --
    iMUc = F(1.0) / MUc
    c["iMUc"] = iMUc
    c["GC1"] = g(alpha_c + F(1.0))
    c["iGC1"] = F(1.0) / c["GC1"]
    c["GC2"] = g(alpha_c + F(1.0) + F(3.0) * iMUc)
    c["GC3"] = g(alpha_c + F(1.0) + F(6.0) * iMUc)
    c["GC4"] = g(alpha_c + F(1.0) + F(9.0) * iMUc)
    c["GC11"] = g(F(1.0) * iMUc + F(1.0) + alpha_c)
    c["GC12"] = g(F(2.0) * iMUc + F(1.0) + alpha_c)
    c["GC5"] = g(F(1.0) + alpha_c)
    c["iGC5"] = F(1.0) / c["GC5"]
    c["GC6"] = g(F(1.0) + alpha_c + F(1.0) * iMUc)
    c["GC7"] = g(F(1.0) + alpha_c + F(2.0) * iMUc)
    c["GC8"] = g(F(1.0) + alpha_c + F(3.0) * iMUc)
    c["GC13"] = g(F(3.0) * iMUc + F(1.0) + alpha_c)
    c["GC14"] = g(F(4.0) * iMUc + F(1.0) + alpha_c)
    c["GC15"] = g(F(5.0) * iMUc + F(1.0) + alpha_c)
    c["icexc9"] = F(1.0) / (c["GC2"] * c["iGC1"] * c["PIov6"] * dew)
    # CCNtype = 2 (driver :3615): continental 1
    c["N_c_SM"] = F(2.0e8)

    # -- Rain (:1334-1354) --
    c["cexr1"] = F(1.0) + alpha_r + dmr + bfr
    c["cexr2"] = F(1.0) + alpha_r + dmr
    c["GR17"] = g(F(2.5) + alpha_r + F(0.5) * bfr)
    c["GR31"] = g(F(1.0) + alpha_r)
    c["iGR31"] = F(1.0) / c["GR31"]
    c["GR32"] = g(F(2.0) + alpha_r)
    c["GR33"] = g(F(3.0) + alpha_r)
    c["GR34"] = g(F(4.0) + alpha_r)
    c["iGR34"] = F(1.0) / c["GR34"]
    c["GR35"] = g(F(5.0) + alpha_r)
    c["GR36"] = g(F(6.0) + alpha_r)
    c["GR37"] = g(F(7.0) + alpha_r)
    c["GR50"] = F(np.power(No_r_SM * c["GR31"], F(0.75)))
    c["cexr5"] = F(2.0) + alpha_r
    c["cexr6"] = F(2.5) + alpha_r + F(0.5) * bfr
    c["cexr9"] = c["cmr"] * c["GR34"] * c["iGR31"]
    c["icexr9"] = F(1.0) / c["cexr9"]
    c["cexr3"] = F(1.0) + bfr + alpha_r
    c["cexr4"] = F(1.0) + alpha_r
    c["ckQr1"] = afr * g(F(1.0) + alpha_r + dmr + bfr) / g(
        F(1.0) + alpha_r + dmr)
    c["ckQr2"] = afr * g(F(1.0) + alpha_r + bfr) * c["GR31"]
    c["ckQr3"] = afr * g(F(7.0) + alpha_r + bfr) / c["GR37"]

    # -- Ice (:1357-1374) --
    c["GI4"] = g(alpha_i + dmi + bfi)
    c["GI6"] = g(F(2.5) + bfi * F(0.5) + alpha_i)
    c["GI11"] = g(F(1.0) + bfi + alpha_i)
    c["GI20"] = g(F(0.0) + bfi + F(1.0) + alpha_i)
    c["GI21"] = g(F(1.0) + bfi + F(1.0) + alpha_i)
    c["GI22"] = g(F(2.0) + bfi + F(1.0) + alpha_i)
    c["GI31"] = g(F(1.0) + alpha_i)
    c["iGI31"] = F(1.0) / c["GI31"]
    c["GI32"] = g(F(2.0) + alpha_i)
    c["GI33"] = g(F(3.0) + alpha_i)
    c["GI34"] = g(F(4.0) + alpha_i)
    c["GI35"] = g(F(5.0) + alpha_i)
    c["GI36"] = g(F(6.0) + alpha_i)
    c["GI40"] = g(F(1.0) + alpha_i + dmi)
    c["icexi9"] = F(1.0) / (c["cmi"] * g(F(1.0) + alpha_i + dmi)
                            * c["iGI31"])
    c["ckQi1"] = afi * g(F(1.0) + alpha_i + dmi + bfi) / c["GI40"]
    c["ckQi2"] = afi * c["GI11"] * c["iGI31"]
    c["ckQi4"] = F(1.0) / (c["cmi"] * c["GI40"] * c["iGI31"])

    # -- Snow (:1377-1399) --
    c["cexs1"] = F(2.5) + F(0.5) * bfs + alpha_s
    c["cexs2"] = F(1.0) + alpha_s + dms
    c["icexs2"] = F(1.0) / c["cexs2"]
    c["GS09"] = g(F(2.5) + bfs * F(0.5) + alpha_s)
    c["GS11"] = g(F(1.0) + bfs + alpha_s)
    c["GS12"] = g(F(2.0) + bfs + alpha_s)
    c["GS13"] = g(F(3.0) + bfs + alpha_s)
    c["GS31"] = g(F(1.0) + alpha_s)
    c["iGS31"] = F(1.0) / c["GS31"]
    c["GS32"] = g(F(2.0) + alpha_s)
    c["GS33"] = g(F(3.0) + alpha_s)
    c["GS34"] = g(F(4.0) + alpha_s)
    c["iGS34"] = F(1.0) / c["GS34"]
    c["GS35"] = g(F(5.0) + alpha_s)
    c["GS36"] = g(F(6.0) + alpha_s)
    c["GS40"] = g(F(1.0) + alpha_s + dms)
    c["iGS40"] = F(1.0) / c["GS40"]
    c["iGS20"] = F(1.0) / (c["GS40"] * c["iGS31"] * cms)
    c["ckQs1"] = afs * g(F(1.0) + alpha_s + dms + bfs) * c["iGS40"]
    c["ckQs2"] = afs * c["GS11"] * c["iGS31"]
    c["GS40_D3"] = g(F(1.0) + alpha_s + F(3.0))
    c["iGS20_D3"] = F(1.0) / (c["GS40_D3"] * c["iGS31"] * c["cms_D3"])
    c["rfact_FvFm"] = (c["PIov6"] * c["icms"]
                       * g(F(4.0) + bfs + alpha_s)
                       / g(F(1.0) + dms + bfs + alpha_s))

    # -- Graupel (:1402-1419) --
    c["GG09"] = g(F(2.5) + F(0.5) * bfg + alpha_g)
    c["GG11"] = g(F(1.0) + bfg + alpha_g)
    c["GG12"] = g(F(2.0) + bfg + alpha_g)
    c["GG13"] = g(F(3.0) + bfg + alpha_g)
    c["GG31"] = g(F(1.0) + alpha_g)
    c["iGG31"] = F(1.0) / c["GG31"]
    c["GG32"] = g(F(2.0) + alpha_g)
    c["GG33"] = g(F(3.0) + alpha_g)
    c["GG34"] = g(F(4.0) + alpha_g)
    c["iGG34"] = F(1.0) / c["GG34"]
    c["GG35"] = g(F(5.0) + alpha_g)
    c["GG36"] = g(F(6.0) + alpha_g)
    c["GG40"] = g(F(1.0) + alpha_g + dmg)
    c["iGG99"] = F(1.0) / (c["GG40"] * c["iGG31"] * c["cmg"])
    c["GG50"] = F(np.power(No_g_SM * c["GG31"], F(0.75)))
    c["ckQg1"] = afg * g(F(1.0) + alpha_g + dmg + bfg) / c["GG40"]
    c["ckQg2"] = afg * c["GG11"] * c["iGG31"]
    c["ckQg4"] = F(1.0) / (c["cmg"] * c["GG40"] * c["iGG31"])

    # -- Hail (:1422-1436) --
    c["GH09"] = g(F(2.5) + bfh * F(0.5) + alpha_h)
    c["GH11"] = g(F(1.0) + bfh + alpha_h)
    c["GH12"] = g(F(2.0) + bfh + alpha_h)
    c["GH13"] = g(F(3.0) + bfh + alpha_h)
    c["GH31"] = g(F(1.0) + alpha_h)
    c["iGH31"] = F(1.0) / c["GH31"]
    c["GH32"] = g(F(2.0) + alpha_h)
    c["GH33"] = g(F(3.0) + alpha_h)
    c["iGH34"] = F(1.0) / g(F(4.0) + alpha_h)
    c["GH40"] = g(F(1.0) + alpha_h + dmh)
    c["iGH99"] = F(1.0) / (c["GH40"] * c["iGH31"] * c["cmh"])
    c["GH50"] = F(np.power(No_h_SM * c["GH31"], F(0.75)))
    c["ckQh1"] = afh * g(F(1.0) + alpha_h + dmh + bfh) / c["GH40"]
    c["ckQh2"] = afh * c["GH11"] * c["iGH31"]
    c["ckQh4"] = F(1.0) / (c["cmh"] * c["GH40"] * c["iGH31"])

    # -- calcDiag reflectivity factors (:3385-3396), parameter-only, f32 --
    c["cxr"] = c["icmr"] * c["icmr"]
    c["cxi"] = F(1.0) / fdielec * c["icmr"] * c["icmr"]
    c["Gzr"] = ((F(6.0) + alpha_r) * (F(5.0) + alpha_r)
                * (F(4.0) + alpha_r)
                / ((F(3.0) + alpha_r) * (F(2.0) + alpha_r)
                   * (F(1.0) + alpha_r)))
    c["Gzi"] = ((F(6.0) + alpha_i) * (F(5.0) + alpha_i)
                * (F(4.0) + alpha_i)
                / ((F(3.0) + alpha_i) * (F(2.0) + alpha_i)
                   * (F(1.0) + alpha_i)))
    # snowSpherical = .false. -> the dms=2 branch (:3392-3394)
    c["Gzs"] = ((F(4.0) + alpha_s) * (F(3.0) + alpha_s)
                / ((F(2.0) + alpha_s) * (F(1.0) + alpha_s)))
    c["Gzg"] = ((F(6.0) + alpha_g) * (F(5.0) + alpha_g)
                * (F(4.0) + alpha_g)
                / ((F(3.0) + alpha_g) * (F(2.0) + alpha_g)
                   * (F(1.0) + alpha_g)))
    c["Gzh"] = ((F(6.0) + alpha_h) * (F(5.0) + alpha_h)
                * (F(4.0) + alpha_h)
                / ((F(3.0) + alpha_h) * (F(2.0) + alpha_h)
                   * (F(1.0) + alpha_h)))
    return c


#: name -> float32 value, in the fixed emission order the kernel indexes.
CONSTANTS: dict[str, np.float32] = _build()

#: The kernel-facing order.  Three translation units #define names as
#: ``ck[index]`` against their ``const float* ck`` parameter:
#: kernels/milbrandt2.cu's whole block is generated from this tuple, and
#: kernels/nest_microphysics.cu and kernels/milbrandt2_zet.cu hand-type
#: the subsets they read.  An inserted constant that is not regenerated
#: shifts every macro after it and NOTHING raises, so
#: :func:`verify_ck_alias_defines` holds all three equal to
#: :data:`CK_INDEX` at IMPORT, and
#: tests/test_milbrandt2_contract.py holds the generated block equal to
#: the generator and the checked set equal to the units that index.
CK_ORDER: tuple[str, ...] = tuple(CONSTANTS)

#: name -> index into the CK vector.
CK_INDEX: dict[str, int] = {name: i for i, name in enumerate(CK_ORDER)}


def ck_vector() -> np.ndarray:
    """The read-only float32 constants vector, in CK_ORDER."""
    return np.asarray([CONSTANTS[name] for name in CK_ORDER],
                      dtype=np.float32)


def cuda_define_block() -> str:
    """Emit the #define block for kernels/milbrandt2.cu (dev utility)."""
    lines = [f"#define {name} ck[{i}]" for i, name in enumerate(CK_ORDER)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The positional-index guard, at import
# ---------------------------------------------------------------------------
#
# THE BREAKAGE.  Three translation units read this vector by POSITION
# through hand-typed ``#define <alias> ck[<n>]`` rows.  Insert a constant
# in the middle of ``_build()`` without regenerating them and every macro
# after the insertion point reads a different number: no compiler
# complains, no launch fails, and the run produces wrong weather.  Only
# kernels/milbrandt2.cu had a guard, and only because its whole block is
# generated -- ``cuda_define_block() in source`` covers a block that is
# emitted, not one that is typed.  The mixed nest edge's ``MY2E_`` rows
# and the lifted Z block's ``my2z_`` rows are hand-typed SUBSETS, which no
# generated-block comparison can reach.
#
# So the rows themselves are checked, and at IMPORT rather than in a test:
# the two subset units are reached by a forecast and by the radar
# observation operator, and neither runs pytest first.
_CK_ALIAS_DEFINE = re.compile(
    r"^[ \t]*#define[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]+ck\[(\d+)\]",
    re.MULTILINE)

#: The prefix each unit may qualify a Fortran name with, before the empty
#: one, so a prefixed alias is resolved as itself rather than as a bare
#: name that happens to share a tail.  A unit may qualify or not; what it
#: may not do is name a constant this table has never heard of.
_CK_ALIAS_PREFIXES: tuple[str, ...] = ("MY2E_", "my2z_", "")

#: Every translation unit that indexes the CK vector positionally.  A
#: fourth one joins this tuple, and
#: ``tests/test_milbrandt2_contract.py`` fails if one appears in the
#: kernels directory and does not.
CK_ALIAS_TRANSLATION_UNITS: tuple[str, ...] = (
    "milbrandt2.cu", "milbrandt2_zet.cu", "nest_microphysics.cu",
)


def _ck_name_for_alias(alias: str) -> str | None:
    for prefix in _CK_ALIAS_PREFIXES:
        if alias.startswith(prefix):
            candidate = alias[len(prefix):]
            if candidate in CK_INDEX:
                return candidate
    return None


def verify_ck_alias_defines(kernel_dir=None) -> dict[str, int]:
    """Hold every ``#define <alias> ck[<n>]`` row equal to CK_INDEX.

    Returns unit name -> rows checked.  A unit that is not on disk is
    skipped and is absent from the result: the standalone preparation
    wheel stages this table (pure numpy) and none of these kernels, so
    there is nothing there to check and nothing there to launch either.
    """

    directory = (Path(__file__).resolve().parent / "kernels"
                 if kernel_dir is None else Path(kernel_dir))
    checked: dict[str, int] = {}
    for unit in CK_ALIAS_TRANSLATION_UNITS:
        path = directory / unit
        if not path.is_file():
            continue
        rows = 0
        for alias, index in _CK_ALIAS_DEFINE.findall(
                path.read_text(encoding="utf-8")):
            name = _ck_name_for_alias(alias)
            if name is None:
                raise RuntimeError(
                    f"{path}: '#define {alias} ck[{index}]' names no "
                    "constant in gpuwm.core.milbrandt2_constants.CK_ORDER. "
                    "An alias is the Fortran name, optionally prefixed by "
                    f"one of {[p for p in _CK_ALIAS_PREFIXES if p]}; a row "
                    "that resolves to no name is a row nothing can check, "
                    "and an unchecked positional index is wrong weather "
                    "with no error anywhere. Spell the constant's own "
                    "name, or add it to _build() if it is genuinely new")
            expected = CK_INDEX[name]
            if int(index) != expected:
                raise RuntimeError(
                    f"{path}: '#define {alias} ck[{index}]' reads the "
                    f"wrong constant -- {name} is at index {expected} in "
                    "CK_ORDER. A constant was inserted into "
                    "gpuwm.core.milbrandt2_constants._build() and this "
                    "unit was not regenerated with it, so every macro from "
                    "the insertion point on reads a different number and "
                    "the scheme integrates against constants nobody chose. "
                    "Regenerate the block -- cuda_define_block() emits "
                    "milbrandt2.cu's; the hand-typed subsets take the same "
                    "indices -- and re-pin the source digests")
            rows += 1
        checked[unit] = rows
    return checked


#: Checked at import, on every route that can reach these kernels.
CK_ALIAS_ROWS_CHECKED: dict[str, int] = verify_ck_alias_defines()


if __name__ == "__main__":
    for name in CK_ORDER:
        print(f"{name} = {CONSTANTS[name]!r}")
    print(f"-- {len(CK_ORDER)} constants --")


_CK_DEVICE = None


def ck_vector_device():
    """:func:`ck_vector` uploaded once per process, as a read-only FP32 plane.

    The cache lives HERE, in the pure-table module, rather than beside the
    scheme: the mixed nest edge into mp=9 needs the same vector and is
    staged into the standalone RW-WPS preparation wheel, which deliberately
    carries no forecast scheme (``tools/build_rw_wps_release.py``
    ``_CORE_MODULES``).  Importing the scheme for a constant vector put an
    unresolvable internal import in that wheel and its staging gate refused
    the build.  One cache, one upload, both callers.

    CuPy is imported inside the call because this module is otherwise pure
    numpy and is imported on CPU-only routes.
    """
    global _CK_DEVICE
    if _CK_DEVICE is None:
        import cupy as cp

        _CK_DEVICE = cp.asarray(ck_vector())
    return _CK_DEVICE
