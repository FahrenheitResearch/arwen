"""The fast SBM port's record of the pinned WRF file: routines, findings, rules.

The pinned file is WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (9,053 lines, LF
SHA-256 ``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1 and v4.7.1).  A copy, LF, is in the tree at
``tools/fast_sbm_wrf480_oracle/wrf/phys/module_mp_fast_sbm.F`` so the tests
read the artifact itself; its hash is checked against the harness pin in
``tools/fast_sbm_wrf480_oracle/sources.sha256`` before anything is read.
WRF is NCAR/UCAR public domain; the notice is in ``NOTICE`` and in
``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The scheme is FSBM-2 by Jacob
Shpund and co-authors (see the package docstring).

What is here, all plain data plus the functions that check it:

``ROUTINES``
    Every SUBROUTINE and FUNCTION in the file (45, the preprocessor stub
    ``SBM_fast`` included), with its module and lines, and for each the
    parts the program ports, each with the item that owns it (milestone 1
    or milestone 2 of the fast-sbm program) and, once built, the host
    function, or the stated reason a part is not ported.
    ``routine_map_problems`` lists what is wrong with a map against the
    parsed file, and the tests hold that list empty.
``FINDINGS``
    Every catalogued WRF behaviour the port reproduces or corrects: D1 to
    D18 and S1 to S3, with lines, what WRF does, its consequence, what the
    port does and the ``FsbmBehavior`` field that selects it.
``DEAD_PATHS``
    Code the 33-bin path never reaches or values it computes and never
    reads.  None of it is ported; each has its line here.
The contract rules
    ``MASS_GRID_MERGES`` with ``mass_grid_mismatches``: the crystal, snow,
    graupel and hail mass grids equal the drop grid, which is what makes
    WRF's bin-by-bin merges conserve mass.  ``GRIDS_ARE_TABLES``: ``FREEZ``
    and ``J_W_MELT`` take the mass grids as dummies (``J_W_MELT`` declares
    them ``INTENT(INOUT)``, :6645) and never assign them, so the host
    passes the loaded tables and never copies them per call.  The 09-13/14
    port's generator had moved these grids into per-call scratch through
    that ``INTENT(INOUT)``; this rule is its "fix 2" made part of the
    contract.

The cold routines' module-table needs (``Routine.module_reads``) come from
the 09-13/14 port's call graph and are checked here against the file:
``FREEZ``, ``J_W_MELT`` and ``ICE_NUCL`` read no module variable;
``ONECOND2`` and ``ONECOND3`` read only ``FR_LIM`` and ``FRH_LIM``, which
nothing assigns and which feed a local nothing reads (:2421-2425);
``COAL_BOTT_NEW`` reads the mass grids, their mg forms, ``chucm``, ``ima``,
``PKIJ``, ``QKJ`` and the kernel families ``Kernals_KS`` fills, all of which
milestone 1 loads and gates.  ``FAST_SBM``'s own cold parts (the ice
pre-pass, the dispatch, the ice fall) add none: every module variable it
mentions that no milestone 1 table carries is a working array it sets
before use or state of the polarimetric operator
(``FAST_SBM_UNCARRIED_READS``).  No table work is left for the cold half.

Importing this module reads nothing; the functions take the file's text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

#: LF SHA-256 of the pinned file (fast-sbm design 3.1).
PINNED_SOURCE_SHA256 = (
    "fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e")
#: Its line count.
PINNED_SOURCE_LINES = 9053
#: Its path in the tree, relative to the repository root.
PINNED_SOURCE_PATH = "tools/fast_sbm_wrf480_oracle/wrf/phys/module_mp_fast_sbm.F"

#: The items of the fast-sbm program that own ported parts.
M1_ITEMS = {
    "fsbm-bins-tables": "bin state and packaged tables",
    "fsbm-fortran-oracle": "the Fortran oracle harness and fixtures",
    "fsbm-derived-tables": "the derived tables of FAST_HUCMINIT",
    "fsbm-condensation": "warm diffusional growth and evaporation",
    "fsbm-nucleation": "droplet activation and the aerosol column",
    "fsbm-collision-breakup": "collision-coalescence and breakup",
    "fsbm-sedimentation": "sedimentation and surface precipitation",
    "fsbm-warm-column": "the warm FAST_SBM call",
}
M2_ITEMS = {
    "fsbm-cold-contract": "behaviour fields, tile inputs, routine map",
    "fsbm-cold-oracle": "cold sweeps, refusals, graupel build, KC-2",
    "fsbm-ice-nucleation-phase": "ice nucleation, freezing, melting",
    "fsbm-deposition": "ice and mixed-phase diffusional growth",
    "fsbm-cold-collision": "liquid-ice and ice-ice collisions",
    "fsbm-fall-exit": "fall, surface precipitation and exit of every species",
    "fsbm-cold-column": "the full FAST_SBM call and the cold column gate",
}
ITEMS = {**M1_ITEMS, **M2_ITEMS}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

Lines = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Part:
    """A part of a routine: the lines, what they do, and either the item
    that owns them (with the host function once it is built, a dotted
    name under ``gpuwm.core.fast_sbm``) or the reason they are not
    ported."""

    lines: Lines
    what: str
    owner: str | None = None
    host: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Routine:
    """One SUBROUTINE or FUNCTION of the pinned file.  ``module_reads``,
    where recorded, is the set of non-PARAMETER module variables the
    routine reads (lower case)."""

    name: str
    kind: str
    module: str
    lines: tuple[int, int]
    parts: tuple[Part, ...]
    module_reads: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Finding:
    """A catalogued WRF behaviour.  ``anchor`` is text that appears in the
    first cited range of the pinned file, so a citation that drifts fails
    a test.  ``field`` is the ``FsbmBehavior`` field that selects WRF's
    behaviour or the port's, ``None`` when there is none."""

    id: str
    lines: Lines
    anchor: str
    what: str
    consequence: str
    port: str
    field: str | None
    owner: str


@dataclass(frozen=True)
class DeadPath:
    """Code the 33-bin path never reaches, or a value it never reads."""

    key: str
    lines: Lines
    anchor: str
    what: str


def _p(lines, what, owner=None, host=None, reason=None) -> Part:
    if isinstance(lines[0], int):
        lines = (tuple(lines),)
    return Part(lines=tuple(tuple(r) for r in lines), what=what, owner=owner,
                host=host, reason=reason)


_NEVER_CALLED = "never called: {caller}, and nothing in the file calls it"

# ---------------------------------------------------------------------------
# The routine map
# ---------------------------------------------------------------------------

ROUTINES: tuple[Routine, ...] = (
    Routine("SBM_fast", "subroutine", "module_mp_fast_sbm", (4, 7), (
        _p((4, 7), "a stub",
           reason="the preprocessor stub compiled only when BUILD_SBM_FAST "
                  "is not 1 (:1-22); the scheme is the #else branch"),)),
    # module_mp_SBM_BreakUp (:96-444)
    Routine("Spontanous_Init", "subroutine", "module_mp_SBM_BreakUp",
            (120, 189), (
                _p((120, 189), "spontaneous breakup probabilities and gains",
                   owner="fsbm-derived-tables",
                   host="derived.spontaneous_init"),)),
    Routine("Spont_Rain_BreakUp", "subroutine", "module_mp_SBM_BreakUp",
            (205, 300), (
                _p((205, 300), "spontaneous breakup of large drops",
                   owner="fsbm-collision-breakup",
                   host="collision.spont_rain_breakup"),)),
    Routine("BreakUp_Snow", "subroutine", "module_mp_SBM_BreakUp",
            (302, 442), (
                _p((302, 442), "snow breakup",
                   reason="FAST_SBM calls it only IF (KR_SNOW_MAX <= NKR) "
                          "(:4841) and KR_SNOW_MAX is 35 on the 33-bin grid "
                          "(:110), so it never runs; its temperature "
                          "argument TT_r is never set (:4841).  The "
                          "cold-column item pins that the guard never calls "
                          "it (dead path breakup_snow)"),)),
    # module_mp_SBM_Collision (:447-1691)
    Routine("coll_xyy_lwf", "subroutine", "module_mp_SBM_Collision",
            (463, 650), (
                _p((463, 650), "drops collected by graupel (:8485) and by "
                               "hail (:8524)",
                   owner="fsbm-cold-collision"),)),
    Routine("coll_xxx_lwf", "subroutine", "module_mp_SBM_Collision",
            (652, 827), (
                _p((652, 827), "self-collection by Bott's flux method: drops "
                               "(:8375); the cold block's snow call (:8563) "
                               "is the cold-collision item's part of "
                               "COAL_BOTT_NEW",
                   owner="fsbm-collision-breakup",
                   host="collision.coll_xxx_lwf"),)),
    Routine("coll_xyx_lwf", "subroutine", "module_mp_SBM_Collision",
            (829, 1030), (
                _p((829, 1030), "ice collecting drops into itself: snow "
                                "(:8462), graupel (:8495), hail (:8534)",
                   owner="fsbm-cold-collision"),)),
    Routine("coll_xyz_lwf", "subroutine", "module_mp_SBM_Collision",
            (1032, 1236), (
                _p((1032, 1236), "drop-snow collisions into the dense ice "
                                 "(:8452-8469)",
                   owner="fsbm-cold-collision"),)),
    Routine("coll_xxy_lwf", "subroutine", "module_mp_SBM_Collision",
            (1238, 1360), (
                _p((1238, 1360), "self-collection into another species",
                   reason=_NEVER_CALLED.format(
                       caller="COAL_BOTT_NEW imports it (:8271) and never "
                              "calls it")),)),
    Routine("INTERPOL_SE", "subroutine", "module_mp_SBM_Collision",
            (1362, 1393), (
                _p((1362, 1393), "the linear table interpolation modkrn_KS "
                                 "CASE(11) calls (:1488)",
                   owner="fsbm-cold-collision"),)),
    Routine("modkrn_KS", "subroutine", "module_mp_SBM_Collision",
            (1395, 1503), (
                _p((1416, 1417), "the return above 0 C before the output is "
                                 "set (D6), all the warm path reaches",
                   owner="fsbm-collision-breakup",
                   host="collision.coal_bott_warm"),
                _p(((1395, 1420), (1481, 1503)),
                   "CASE(11), the case COAL_BOTT_NEW selects (:8320), at or "
                   "below 0 C",
                   owner="fsbm-cold-collision"),
                _p((1421, 1479), "CASE(1)",
                   reason="never selected: the CASE(1) call at :8319 is "
                          "commented out and the live call at :8320 passes "
                          "11 (dead path modkrn_case1)"),
            )),
    Routine("coll_breakup_KS", "subroutine", "module_mp_SBM_Collision",
            (1505, 1634), (
                _p((1505, 1634), "collisional breakup of drops",
                   owner="fsbm-collision-breakup",
                   host="collision.coll_breakup"),)),
    Routine("courant_bott_KS", "subroutine", "module_mp_SBM_Collision",
            (1636, 1689), (
                _p((1636, 1689), "Bott's target bins and Courant numbers",
                   owner="fsbm-derived-tables", host="derived.courant_bott"),
            )),
    # module_mp_SBM_Auxiliary (:1694-3010)
    Routine("POLYSVP", "function", "module_mp_SBM_Auxiliary", (1716, 1742), (
        _p((1716, 1742), "saturation vapour pressure over water and ice",
           owner="fsbm-condensation", host="condensation.polysvp"),)),
    Routine("JERRATE_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (1744, 1921), (
                _p((1744, 1921), "growth rates of drops (IN = 1)",
                   owner="fsbm-condensation",
                   host="condensation.jerrate_drops"),
                _p((1744, 1921), "growth rates of the three crystal habits, "
                                 "snow and the dense ice, with the nskin "
                                 "rule at :1788-1794",
                   owner="fsbm-deposition"),
            )),
    Routine("JERTIMESC_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (1925, 1964), (
                _p((1925, 1964), "the relaxation time of drops",
                   owner="fsbm-condensation", host="condensation.jertimesc"),
                _p((1925, 1964), "the relaxation time of each ice species",
                   owner="fsbm-deposition"),
            )),
    Routine("JERSUPSAT_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (1966, 2330), (
                _p((1966, 2330), "every branch as WRF has it, the "
                                 "mixed-phase ones with S1 to S3 included",
                   owner="fsbm-condensation", host="condensation.jersupsat"),
                _p(((2196, 2313),), "the rederived small-ratio forms of the "
                                    "corrected preset (S1, S2)",
                   owner="fsbm-deposition"),
            )),
    Routine("JERDFUN_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (2334, 2442), (
                _p((2334, 2442), "the drop remap (ITYPE 1)",
                   owner="fsbm-condensation", host="condensation.jerdfun"),
                _p((2334, 2442), "the ice remaps (ITYPE 2 to 5)",
                   owner="fsbm-deposition"),
            )),
    Routine("JERNEWF_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (2444, 2757), (
                _p((2444, 2757), "the Kovetz-Olund remap",
                   owner="fsbm-condensation", host="condensation.jernewf"),)),
    Routine("JERDFUN_NEW_KS", "subroutine", "module_mp_SBM_Auxiliary",
            (2759, 2834), (
                _p((2759, 2834), "the drop remap ONECOND1 returns",
                   owner="fsbm-condensation",
                   host="condensation.jerdfun_new"),)),
    Routine("Relaxation_Time", "subroutine", "module_mp_SBM_Auxiliary",
            (2836, 3008), (
                _p((2836, 3008), "relaxation times",
                   reason=_NEVER_CALLED.format(
                       caller="module_mp_fast_sbm imports it (:3848) and "
                              "never calls it")),)),
    # module_mp_SBM_Nucleation (:3013-3840)
    Routine("JERNUCL01_KS", "subroutine", "module_mp_SBM_Nucleation",
            (3037, 3192), (
                _p(((3037, 3099), (3184, 3192)), "drop activation",
                   owner="fsbm-nucleation", host="nucleation.jernucl01_warm"),
                _p((3100, 3111), "homogeneous freezing at -38 C (D14)",
                   owner="fsbm-ice-nucleation-phase"),
                _p((3113, 3182), "crystal nucleation with its vapour sink "
                                 "and heating",
                   owner="fsbm-ice-nucleation-phase"),
            )),
    Routine("WATER_NUCLEATION", "subroutine", "module_mp_SBM_Nucleation",
            (3194, 3330), (
                _p((3194, 3330), "drop activation from the aerosol",
                   owner="fsbm-nucleation",
                   host="nucleation.water_nucleation"),)),
    Routine("ICE_NUCL", "subroutine", "module_mp_SBM_Nucleation",
            (3340, 3475), (
                _p((3340, 3475), "Meyers et al. (1992) deposition and "
                                 "condensation-freezing nucleation",
                   owner="fsbm-ice-nucleation-phase"),),
            module_reads=()),
    Routine("Cloud_Base_Super", "subroutine", "module_mp_SBM_Nucleation",
            (3479, 3570), (
                _p((3479, 3570), "activation at cloud base (Pinsky et al. "
                                 "2012)",
                   owner="fsbm-nucleation",
                   host="nucleation.cloud_base_super"),)),
    Routine("supmax_COEFF", "subroutine", "module_mp_SBM_Nucleation",
            (3572, 3733), (
                _p((3572, 3733), "the maximum supersaturation coefficients",
                   owner="fsbm-nucleation", host="nucleation.supmax_coeff"),
            )),
    Routine("LogNormal_modes_Aerosol", "subroutine",
            "module_mp_SBM_Nucleation", (3735, 3838), (
                _p((3735, 3838), "the lognormal aerosol spectra",
                   owner="fsbm-derived-tables",
                   host="derived.lognormal_modes_aerosol"),)),
    # module_mp_fast_sbm (:3843-9052)
    Routine("FAST_SBM", "subroutine", "module_mp_fast_sbm", (4001, 5322), (
        _p(((4001, 4312), (4351, 4395), (4514, 4559), (4578, 4666),
            (4675, 4701), (4710, 4723), (4764, 4790), (4823, 4832),
            (4846, 4914), (4961, 5071), (5266, 5322)),
           "the warm path: entry, cloud base, the per-point warm "
           "processes, drop fall, diagnostics, drop precipitation, exit",
           owner="fsbm-warm-column", host="column.fast_sbm_warm_call"),
        _p(((4396, 4512), (4560, 4577), (4667, 4674), (4702, 4709),
            (4724, 4763), (4791, 4822), (4833, 4845)),
           "the ice pre-pass (DSUPICE_XYZ, D15, D17), the dense-ice slot "
           "of the entry, the merge of nucleated crystals into snow, the "
           "ice flags, the ice and mixed-phase dispatch, FREEZ, J_W_MELT, "
           "the merges and the snow-breakup guard, in the full call",
           owner="fsbm-cold-column"),
        _p(((4915, 4960), (4974, 5055), (5072, 5121), (5276, 5320)),
           "snow and dense-ice fall, the ice diagnostics, surface snow "
           "and dense ice, the exit and its clip for every species",
           owner="fsbm-fall-exit"),
        _p((5122, 5265), "the polarimetric forward operator",
           reason="left to milestone M6 of the fast-sbm design (the "
                  "polarimetric operator on bins); it changes no state"),
        _p((4313, 4349), "the aerosol lateral boundary reset",
           reason="compiled only with DM_PARALLEL, so serial WRF has none; "
                  "left to the engine wiring, M5 (dead path "
                  "mpi_aerosol_boundary)"),
    )),
    Routine("FALFLUXHUCM_Z", "subroutine", "module_mp_fast_sbm",
            (5324, 5382), (
                _p((5324, 5382), "the upstream fall of one species",
                   owner="fsbm-sedimentation", host="sedimentation.falflux"),
                _p((5364, 5373), "the layer-depth divisor (D8)",
                   owner="fsbm-fall-exit"),
            ),
            module_reads=()),
    Routine("FAST_HUCMINIT", "subroutine", "module_mp_fast_sbm",
            (5384, 6236), (
                _p((5384, 6043), "the READs of SBM_input_33 (the "
                                 "DM_PARALLEL broadcasts do nothing "
                                 "serially)",
                   owner="fsbm-bins-tables", host="tables.load_tables"),
                _p((6070, 6236), "the derived tables: mg grids, Bott "
                                 "categories, drop radii, aerosol spectra, "
                                 "BREAKINIT_KS, kernels, breakup init",
                   owner="fsbm-derived-tables", host="derived.build_derived"),
                _p((6044, 6069), "the scattering tables of the polarimetric "
                                 "operator",
                   reason="left to milestone M6 with the operator"),
            )),
    Routine("Kernals_KS", "subroutine", "module_mp_fast_sbm", (6238, 6541), (
        _p((6238, 6541), "every kernel family at a pressure",
           owner="fsbm-derived-tables", host="derived.kernels_at_pressure"),
    )),
    Routine("ckern_z", "function", "module_mp_fast_sbm", (6544, 6558), (
        _p((6544, 6558), "kernel interpolation in pressure",
           owner="fsbm-derived-tables", host="derived._ckern_z"),)),
    Routine("FREEZ", "subroutine", "module_mp_fast_sbm", (6560, 6636), (
        _p((6560, 6636), "immediate freezing (Bigg 1953)",
           owner="fsbm-ice-nucleation-phase"),),
            module_reads=()),
    Routine("J_W_MELT", "subroutine", "module_mp_fast_sbm", (6638, 6746), (
        _p((6638, 6746), "melting at constant time scales",
           owner="fsbm-ice-nucleation-phase"),),
            module_reads=()),
    Routine("ONECOND1", "subroutine", "module_mp_fast_sbm", (6748, 7143), (
        _p((6748, 7143), "warm diffusional growth and evaporation",
           owner="fsbm-condensation", host="condensation.onecond1"),),
            module_reads=("fr_lim", "frh_lim")),
    Routine("ONECOND2", "subroutine", "module_mp_fast_sbm", (7145, 7714), (
        _p((7145, 7714), "ice-only diffusional growth",
           owner="fsbm-deposition"),),
            module_reads=("fr_lim", "frh_lim")),
    Routine("ONECOND3", "subroutine", "module_mp_fast_sbm", (7716, 8261), (
        _p((7716, 8261), "mixed-phase diffusional growth",
           owner="fsbm-deposition"),),
            module_reads=("fr_lim", "frh_lim")),
    Routine("COAL_BOTT_NEW", "subroutine", "module_mp_fast_sbm",
            (8263, 8642), (
                _p(((8263, 8436), (8604, 8611), (8639, 8642)),
                   "the drop collisions and breakup, the warm path",
                   owner="fsbm-collision-breakup",
                   host="collision.coal_bott_warm"),
                _p(((8437, 8570), (8572, 8602), (8603, 8640)),
                   "the cold block, Hallett-Mossop, the riming heat and "
                   "the exit checks (deldrop :8598, the ice spectra, Del_T "
                   ":8636)",
                   owner="fsbm-cold-collision"),
            ),
            module_reads=("chucm", "cwgl", "cwhl", "cwlg", "cwlh", "cwll",
                          "cwls", "cwsl", "cwss", "ima", "pkij", "qkj", "xg",
                          "xg_mg", "xh", "xh_mg", "xi", "xl", "xl_mg", "xs",
                          "xs_mg")),
    Routine("BREAKINIT_KS", "subroutine", "module_mp_fast_sbm",
            (8644, 8780), (
                _p((8644, 8780), "ECOALMASSM and BRKWEIGHT",
                   owner="fsbm-derived-tables", host="derived.build_derived"),
                _p((8644, 8780), "the READs of PKIJ and QKJ (:8725, :8755)",
                   owner="fsbm-bins-tables", host="tables.load_tables"),
            )),
    Routine("ecoalmass", "function", "module_mp_fast_sbm", (8784, 8804), (
        _p((8784, 8804), "coalescence efficiency by mass (D10)",
           owner="fsbm-derived-tables", host="derived._ecoalmass"),)),
    Routine("ecoaldiam", "function", "module_mp_fast_sbm", (8807, 8860), (
        _p((8807, 8860), "coalescence efficiency by diameter",
           owner="fsbm-derived-tables", host="derived._ecoal_diam"),)),
    Routine("ecoalLowList", "function", "module_mp_fast_sbm", (8863, 8908), (
        _p((8863, 8908), "Low and List (1982a) efficiency",
           owner="fsbm-derived-tables", host="derived._ecoal_low_list"),)),
    Routine("ecoalOchs", "function", "module_mp_fast_sbm", (8912, 8951), (
        _p((8912, 8951), "Beard and Ochs (1995) efficiency",
           owner="fsbm-derived-tables", host="derived._ecoal_ochs"),)),
    Routine("COLLENERGY", "subroutine", "module_mp_fast_sbm", (8955, 9021), (
        _p((8955, 9021), "collision kinetic and surface energies",
           owner="fsbm-derived-tables", host="derived._collenergy"),)),
    Routine("vTBeard", "function", "module_mp_fast_sbm", (9026, 9048), (
        _p((9026, 9048), "the drop fall speed lookup (D10)",
           owner="fsbm-derived-tables", host="derived._vt_beard"),)),
)


# ---------------------------------------------------------------------------
# The findings
# ---------------------------------------------------------------------------

FINDINGS: tuple[Finding, ...] = (
    Finding("D1", ((4523, 4523), (5064, 5064)), "VR1_Z(1:nkr,K)",
            "VR1_Z has no i, j index and is overwritten at every point; the "
            "surface precipitation reads it after the loop, and so do snow "
            "(:5075) and the dense ice (:5093, :5107)",
            "every column's RAINNC uses the fall-speed pressure factor of the "
            "tile's last column",
            "corrected: each column's own factor",
            "rainnc_velocity", "fsbm-warm-column"),
    Finding("D2", ((5058, 5121),), "RAINNCV(I,J) = 0.0",
            "RAINNC is fall speed times the lowest-level spectrum after "
            "sedimentation, times dt, for every species",
            "surface precipitation does not equal the column water lost",
            "corrected: the bottom-face outflow",
            "rainnc_source", "fsbm-warm-column"),
    Finding("D3", ((3321, 3327), (3305, 3313)), "PSI1(KR)+DROPCONCN(KR)/DX",
            "WATER_NUCLEATION adds the activated drops and takes no vapour "
            "and releases no heat for them; the RMASSL_NUCL it forms is not "
            "the activated mass and is discarded",
            "liquid water created from nothing at every activation",
            "corrected: the sink and heating as the ice branch applies them",
            "nucleation_water_sink", "fsbm-nucleation"),
    Finding("D4", ((5285, 5285),), "qc(i,k,j)+qr(i,k,j).lt.1.e-13",
            "drop bins zeroed where qc + qr < 1e-13 at exit",
            "tiny water loss",
            "kept in both presets and counted in the budget",
            None, "fsbm-warm-column"),
    Finding("D5", ((4620, 4620),), "DTCOND = min(DT-DTNEW,DTCOND)",
            "DTCOND is not reset per point and carries a last-place drift "
            "across points",
            "ULP-level differences between serial and parallel orders",
            "corrected: reset per point",
            "dtcond_carry", "fsbm-warm-column"),
    Finding("D6", ((1417, 1417),), "if (tc > 0.0) return",
            "modkrn_KS returns before setting its output above 0 C",
            "none: the value feeds only an unused diagnostic there",
            "recorded",
            None, "fsbm-collision-breakup"),
    Finding("D7", ((8423, 8423),), "break_drop_per > 1.001",
            "the breakup retry doubles its substep count without a limit",
            "a possible endless loop",
            "both presets cap it and refuse by name",
            "breakup_retry_cap", "fsbm-collision-breakup"),
    Finding("D8", ((5364, 5373),), "ZCGS(K+1)",
            "sedimentation divides by the distance between level centres, "
            "not the layer depth",
            "column water not conserved exactly on stretched grids",
            "both presets keep WRF's divisor until the fall-exit item builds "
            "the layer depth; the warm call refuses 'layer_depth' until then",
            "fall_depth", "fsbm-fall-exit"),
    Finding("D9", ((4387, 4387),), "K_found .eqv. .FALSE.",
            "the cloud-base test parses as (a .and. b .and. K_found) .eqv. "
            "(.FALSE. .and. ...)",
            "the cloud-base activation branch runs at the wrong level or "
            "not at all",
            "corrected: the first level the code's comment describes",
            "cloud_base_rule", "fsbm-warm-column"),
    Finding("D10", ((8795, 8795), (6100, 6100), (9041, 9045)),
            "PI=3.1415927d0",
            "ecoalmass forms a diameter with pi 3.1415927 while DROPRADII "
            "uses 3.141593, so vTBeard hands every bin-centre drop the next "
            "bin's fall speed",
            "272 of the 1024 ECOALMASSM entries change, 44 by more than 10 "
            "percent",
            "corrected: each bin's own fall speed",
            "fall_speed_lookup", "fsbm-derived-tables"),
    Finding("D11", ((5291, 5291),), "qs(i,k,j).lt.1.e-13",
            "the exit zeroes every snow bin when qs (bins 19-33) is below "
            "1e-13; small ice (qi, bins 1-18) is not counted",
            "nucleated and small ice is deleted at the end of every call "
            "unless large snow exists",
            "corrected: the clip keyed on qi + qs",
            "snow_clip_key", "fsbm-fall-exit"),
    Finding("D12", ((6683, 6734),), "meltrate = 0.5/50.",
            "J_W_MELT removes rate x dt of a bin with no limit",
            "a bin goes negative once dt exceeds 100 s in the middle bands "
            "or 176 s in the large ones",
            "corrected: the bin keeps exp(-rate dt) of itself",
            "melt_step", "fsbm-ice-nucleation-phase"),
    Finding("D13", ((6627, 6627), (6741, 6741), (8590, 8590), (3108, 3108)),
            "333.*SUM_ICE/RO",
            "the fusion heat is 333 in FREEZ and J_W_MELT, 320 in riming and "
            "334 in homogeneous freezing; the scheme's AL2 - AL1 is 334",
            "liquid-ice static energy does not close",
            "corrected: AL2 - AL1 at every site",
            "fusion_heat", "fsbm-ice-nucleation-phase"),
    Finding("D14", ((3094, 3111),), "IF(SUP1>0.0D0 .AND. TPC>T_NUCL_DROP_MIN)",
            "homogeneous freezing at -38 C runs only inside the "
            "water-supersaturated branch",
            "supercooled drops below -38 C survive in subsaturated air",
            "corrected: every point at or below -38 C; the cold-column item "
            "measures both on KC-2 and records the decision",
            "homogeneous_freezing", "fsbm-ice-nucleation-phase"),
    Finding("D15", ((4405, 4410), (4504, 4505)), "IF(T_OLD(I,K,J).GE.238.15",
            "w_stag, the scalar the DSUPICE_XYZ term multiplies, is set only "
            "for 238.15 <= T_old < 274.15 K",
            "below 238.15 K the ice-nucleation forcing uses the w_stag of the "
            "most recent point, in the pre-pass's (j, i, k) order, with "
            "238.15 <= T_old < 274.15; undefined if no such point came "
            "earlier in the call",
            "wrf_v480 defines the undefined first read as the point's own "
            "w_stag; corrected: every point its own w",
            "dsupice_w", "fsbm-cold-column"),
    Finding("D16", ((4744, 4744), (4724, 4725)), "IF (TT > 233.15)THEN",
            "the dispatch has no branch for mixed phase at or below 233.15 K, "
            "or for ice without drops above -0.187 C",
            "no diffusional growth or loss at those points",
            "measured by the cold-column item on KC-2; the decision and any "
            "behaviour field land with that evidence",
            None, "fsbm-cold-column"),
    Finding("D17", ((4411, 4417), (4497, 4503)), "UX=25.*(U(I,K,J)+U(I+1,K,J)",
            "the pre-pass takes UX and VX as the four-point average of U and "
            "V over (I..I+1, J..J+1), and the point's own wind at I = IDE-1 "
            "or J = JDE-1; the winds solve_em hands FAST_SBM are the "
            "mass-point u_phy and v_phy (module_big_step_utilities_em.F "
            ":4857-4858)",
            "the horizontal DSUPICE_XYZ forcing (:4504) multiplies centred "
            "differences at (I, J) by a wind at (I+1/2, J+1/2), half a cell "
            "off diagonally; the copy at :4411-4417 feeds only DTFREEZ_XYZ, "
            "which nothing reads",
            "corrected: each point's own mass-point wind",
            "dsupice_wind", "fsbm-cold-column"),
    Finding("D18", ((4351, 4360), (4262, 4265), (4362, 4370), (5267, 5274)),
            "th_old(i,k,j)=th_phy(i,k,j)",
            "the pre-pass reads T_OLD and QV_OLD one column beyond the tile "
            "(I_START..I_END and J_START..J_END, :4262-4265, filled from "
            "th_old and qv_old at :4362-4370); every tile writes th_old and "
            "qv_old back on its own columns at exit (:5267-5274), and at "
            "itimestep 1 over its halo too (:4351-4360); WRF's "
            "microphysics driver calls FAST_SBM once per tile, in a loop "
            "over tiles that OpenMP may run in parallel "
            "(module_microphysics_driver.F :867-870, v4.6.1)",
            "DSUPICE_XYZ at a tile's edge depends on the order of the tiles: "
            "a tile called after its neighbour reads the neighbour's "
            "post-call th_old and qv_old, one called before it reads the "
            "pre-call values, and under OpenMP the reads race the writes.  "
            "Across MPI patches the halo exchange before the call "
            "(HALO_EM_SBM, Registry.EM_COMMON :3423) supplies pre-call "
            "values",
            "both presets define it as MPI does: every halo value is the "
            "neighbours' state before the call, and the itimestep 1 reset "
            "changes only this call's T_OLD and QV_OLD, never the returned "
            "tile's halo or a neighbour",
            None, "fsbm-cold-column"),
    Finding("S1", ((2252, 2256), (2286, 2290)),
            "ALFA=PI*(1+(X/PI)/2.0D0",
            "branches 8 and 9 of JERSUPSAT_KS expand ALFA = sqrt(PI**2 + X) "
            "as PI + X/2 where PI + X/(2 PI) is right",
            "every output 2.5e-5 from the exact solution on the oracle's "
            "branch 8 case",
            "corrected: the forms rederived",
            "supsat_small_ratio", "fsbm-deposition"),
    Finding("S2", ((2202, 2202),), "IF (GAMA < 0.5*2.0d-10)",
            "branch 7 replaces any GAMA below 1e-10 with 5e-14, and GAMA is "
            "negative whenever both phases are present",
            "the slow mode stops decaying over the step, an error at most "
            "1e-6",
            "corrected: GAMA keeps its sign",
            "supsat_small_ratio", "fsbm-deposition"),
    Finding("S3", ((2196, 2313),), "in case DABS(GAMA*DT)<=1.0D-6",
            "branches 7 to 9 carry DEL1INT to first order in GAMA only",
            "the difference is proportional to G31, zero without forcing, "
            "so zero on every WRF call",
            "recorded",
            None, "fsbm-condensation"),
)


# ---------------------------------------------------------------------------
# Dead on the 33-bin path: recorded, never ported
# ---------------------------------------------------------------------------

DEAD_PATHS: tuple[DeadPath, ...] = (
    DeadPath("breakup_snow", ((4836, 4845), (110, 111)),
             "KR_SNOW_MIN:NKR",
             "BreakUp_Snow: KR_SNOW_MAX is 35 and NKR is 33, so the call at "
             ":4841 never runs, although the guard's sum over the empty "
             "section FF3R(34:33) passes (0 > -1e-30).  Its TT_r is never "
             "set, and the FLIQFR_SD and FRIMFR_SD carry-over across points "
             "reaches only this call and the polarimetric block"),
    DeadPath("coll_xxy_lwf", ((1238, 1360),), "coll_xxy_lwf",
             "coll_xxy_lwf: imported by COAL_BOTT_NEW (:8271), never called"),
    DeadPath("modkrn_case1", ((1421, 1479),), "CASE(1)",
             "modkrn_KS CASE(1): the live call selects CASE(11) (:8320); "
             "its TTCOAL input is therefore never read"),
    DeadPath("relaxation_time", ((2836, 3008),), "Relaxation_Time",
             "Relaxation_Time: imported at :3848, never called"),
    DeadPath("demott", ((3027, 3030),), "DE_MOTT",
             "DeMott ice nucleation: ice_nucl_method 1 is named at :3029; "
             "the file has no code for it and the method is 0"),
    DeadPath("fr_lim", ((3997, 3997), (2421, 2425)), "FR_LIM(NKR), FRH_LIM(NKR)",
             "FR_LIM and FRH_LIM: nothing assigns them; ONECOND1, ONECOND2 "
             "and ONECOND3 pass them to JERDFUN_KS, which copies them into "
             "FR_LIM_KR and never reads it"),
    DeadPath("dtfreez_xyz", ((4436, 4439), (4423, 4423)), "DTFREEZ_XYZ",
             "DTFREEZ_XYZ is computed and never read; its T_OLD(I+1) read at "
             ":4423 (and J+1 at :4430) on a one-column or one-row domain "
             "falls outside the filled halo"),
    DeadPath("unused_kernel_families", ((6238, 6541),), "cwsg",
             "the crystal kernel families (cwli, cwil, cwii, cwis, cwsi, "
             "cwig, cwih) and snow-graupel (cwsg): Kernals_KS fills them and "
             "no collision reads them, because FF2R is zero before every "
             "collision (:4671-4672) and no snow-graupel call exists"),
    DeadPath("onecond2_crystal_branch", ((4724, 4725),), "sum(ISYM2)>1",
             "the crystal term of the ONECOND2 and ONECOND3 dispatch: "
             "sum(ISYM2) is 0 because FF2R is emptied into snow before it "
             "(:4667-4674)"),
    DeadPath("drop_dense_other_build", ((8476, 8537),), "icol_graupel == 1",
             "drop-graupel in the hail build and drop-hail in the graupel "
             "build: the entry leaves the other dense slot at zero "
             "(:4560-4577) and nothing fills it before COAL_BOTT_NEW"),
    DeadPath("graupel_to_hail", ((8482, 8482), (3908, 3908)),
             "alwc < alcr_g",
             "the graupel-to-hail branch (:8497-8513): ALCR_G is 100 g/m3, "
             "so it needs 100 g/m3 of liquid water"),
    DeadPath("mpi_aerosol_boundary", ((4313, 4349),), "DM_PARALLEL",
             "the aerosol lateral boundary reset exists only in MPI builds; "
             "left to the engine wiring (M5)"),
    DeadPath("unused_constants", ((4090, 4099), (3889, 3892), (3905, 3905)),
             "ANC1,ANC2,PARSP",
             "ANC1, ANC2, PARSP, PEPL, PEPI, PERL, PERI, TAUR1, TAUR2, EPS_R1, "
             "EPS_R2, EPSF1, EPSF3 and EPSF4 are set by the DATA statement "
             "and never read; DX_BOUND, ICETURB, LIQTURB and KRBREAK are "
             "never read; the module C1_MEY (0.00033) and C2_MEY reach "
             "ONECOND1 to 3 only for their error prints, and ICE_NUCL uses "
             "its own C1_MEY = 1.0D-3 (:3355); DEL_BB, DEL_BBN and DEL_BBR "
             "(:4210-4212) are never read"),
    DeadPath("sr_never_set", ((4014, 4014),), "SNOWNCV,GRAUPELNC,GRAUPELNCV,SR",
             "SR, the frozen fraction of the surface precipitation, is an "
             "argument FAST_SBM never assigns; the port's result has no SR"),
)


# ---------------------------------------------------------------------------
# FAST_SBM's own module reads
# ---------------------------------------------------------------------------

#: The module variables ``FAST_SBM`` mentions (``module_variable_reads``)
#: that no milestone 1 table carries (``FsbmTables``, ``FsbmDerived``,
#: ``FsbmKernels``), each with a line that mentions it and what it is.
#: Every one is a working array the call sets before it reads it, or state
#: of the polarimetric operator (:5122-5265, milestone M6).  So the cold
#: parts of ``FAST_SBM`` read no table milestone 1 does not load; a new
#: name here would be a table the cold half has to add.
FAST_SBM_UNCARRIED_READS = {
    "xs_d": (4197, "a copy of the snow grid for BreakUp_Snow (dead path "
                   "breakup_snow)"),
    "ncond": (4204, "the condensation substep count, set per call"),
    "ncoll": (4205, "the collision substep count, set per call"),
    "fccn": (4543, "the point's aerosol spectrum, copied from chem_new"),
    "fliqfr_sd": (4770, "liquid fraction of snow, zeroed before each "
                        "COAL_BOTT_NEW; read only by BreakUp_Snow and the "
                        "polarimetric operator"),
    "fliqfr_gd": (4771, "liquid fraction of graupel, zeroed likewise"),
    "fliqfr_hd": (4772, "liquid fraction of hail, zeroed likewise"),
    "frimfr_sd": (4773, "rimed fraction of snow, zeroed likewise"),
    "ff1r_d": (4827, "the drop spectrum copy of Spont_Rain_BreakUp, and "
                     "the operator's drops (:5151)"),
    "xl_d": (4828, "the drop grid copy of Spont_Rain_BreakUp"),
    "ff3r_d": (4839, "the snow spectrum copy of the snow-breakup guard, "
                     "and the operator's snow (:5159)"),
    "ff2r_d": (5137, "the polarimetric operator's crystals"),
    "ro2bl_d": (5144, "the polarimetric operator's crystal densities"),
    "ff4r_d": (5168, "the polarimetric operator's graupel"),
    "ff5r_d": (5171, "the polarimetric operator's hail"),
    "rhocgs_d": (5185, "the polarimetric operator's air density"),
    "t_new_d": (5186, "the polarimetric operator's temperature"),
    "icloud": (5189, "the polarimetric operator's cloud flag"),
    "bin_log": (5197, "log10 of the bin masses, set by FAST_HUCMINIT "
                      "(:5446) and read only by the polarimetric operator"),
}


# ---------------------------------------------------------------------------
# Contract rules
# ---------------------------------------------------------------------------

#: The bin-by-bin merges that conserve mass only while every ice mass grid
#: equals the drop grid: nucleated crystals into snow (:4670), crystals into
#: snow (:4811), graupel into hail (:4815) or hail into graupel (:4818).
MASS_GRID_MERGES = (4670, 4811, 4815, 4818)
#: The mass grids of the merged species, as ``FsbmTables`` names them.
MERGED_MASS_GRIDS = ("xi", "xs", "xg", "xh")
#: The routines that take the mass grids as dummies and never assign them.
GRIDS_ARE_TABLES = ("FREEZ", "J_W_MELT")
#: The mass-grid dummies of those routines.
GRID_DUMMIES = ("xl", "xi", "xs", "xg", "xh")


def mass_grid_mismatches(tables) -> tuple[str, ...]:
    """The ice mass grids of ``tables`` that are not bit for bit the drop
    grid ``xl``: ``"xi[:, 0]"`` to ``"xi[:, 2]"`` for the three habits,
    ``"xs"``, ``"xg"``, ``"xh"``.  Empty for the packaged tables, in either
    precision view.  A non-empty answer means WRF's bin-by-bin merges
    (``MASS_GRID_MERGES``) would create or destroy mass."""
    xl = np.asarray(tables.xl)
    out = []
    xi = np.asarray(tables.xi)
    for habit in range(xi.shape[1]):
        if xi[:, habit].dtype != xl.dtype or not np.array_equal(
                xi[:, habit], xl):
            out.append(f"xi[:, {habit}]")
    for name in ("xs", "xg", "xh"):
        grid = np.asarray(getattr(tables, name))
        if grid.dtype != xl.dtype or not np.array_equal(grid, xl):
            out.append(name)
    return tuple(out)


# ---------------------------------------------------------------------------
# Reading the Fortran
# ---------------------------------------------------------------------------

class ParsedRoutine(NamedTuple):
    """A routine as the file declares it: its name as written, ``kind``
    (``"subroutine"`` or ``"function"``), the enclosing module (as written,
    ``None`` outside any) and its header and END lines."""

    name: str
    kind: str
    module: str | None
    first: int
    last: int


_HEADER = re.compile(
    r"^\s*(?:(?:double\s+precision|real|integer|logical)\s+)?"
    r"(subroutine|function)\s+([a-z_][a-z0-9_]*)", re.I)
_END = re.compile(r"^\s*end\s*(subroutine|function)\b", re.I)
_MODULE = re.compile(r"^\s*module\s+([a-z_][a-z0-9_]*)\s*$", re.I)
_END_MODULE = re.compile(r"^\s*end\s*module\b", re.I)
_CONTAINS = re.compile(r"^\s*contains\s*$", re.I)


def strip_comment(line: str) -> str:
    """``line`` without its ``!`` comment (quotes respected)."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "!":
            break
        out.append(ch)
    return "".join(out)


def parse_routines(text: str) -> list[ParsedRoutine]:
    """Every SUBROUTINE and FUNCTION header of ``text`` with its END line.

    Headers are matched after comments are removed, in any case, with an
    optional ``double precision``, ``real``, ``integer`` or ``logical``
    result type; a CALL, a comment or an ``END SUBROUTINE`` is not a
    header.  A header without its END, or an END without a header, raises
    ``ValueError``.
    """
    out: list[ParsedRoutine] = []
    module = None
    open_: tuple[str, str, str | None, int] | None = None
    for number, raw in enumerate(text.split("\n"), 1):
        line = strip_comment(raw)
        if _MODULE.match(line):
            module = _MODULE.match(line).group(1)
            continue
        if _END_MODULE.match(line):
            module = None
            continue
        if _END.match(line):
            if open_ is None:
                raise ValueError(f"END {_END.match(line).group(1)} at line "
                                 f"{number} closes no routine")
            out.append(ParsedRoutine(open_[0], open_[1], open_[2], open_[3],
                                     number))
            open_ = None
            continue
        match = _HEADER.match(line)
        if match:
            if open_ is not None:
                raise ValueError(f"{match.group(2)} at line {number} opens "
                                 f"inside {open_[0]} (line {open_[3]})")
            open_ = (match.group(2), match.group(1).lower(), module, number)
    if open_ is not None:
        raise ValueError(f"{open_[0]} at line {open_[3]} has no END")
    return out


def statements(text: str, first: int, last: int) -> list[tuple[int, str]]:
    """The Fortran statements of lines ``first``..``last`` (one-based), as
    ``(first line, text)``: comments and preprocessor lines removed and
    ``&`` continuations joined."""
    lines = text.split("\n")
    out: list[tuple[int, str]] = []
    buffer = ""
    start = None
    for number in range(first, last + 1):
        raw = lines[number - 1]
        if raw.lstrip().startswith("#"):
            continue
        piece = strip_comment(raw).strip()
        if not piece:
            continue
        if buffer and piece.startswith("&"):
            piece = piece[1:].strip()
        if start is None:
            start = number
        if piece.endswith("&"):
            buffer += " " + piece[:-1]
            continue
        out.append((start, (buffer + " " + piece).strip()))
        buffer = ""
        start = None
    return out


_OPERATORS = re.compile(
    r"\.(and|or|not|eqv|neqv|eq|ne|lt|le|gt|ge|true|false)\.", re.I)
_NUMBER = re.compile(
    r"(?<![a-z_0-9])(\d+\.\d*|\.\d+|\d+)([ed][+-]?\d+)?(_[a-z0-9_]+)?", re.I)
_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_IDENT = re.compile(r"[a-z_][a-z0-9_]*", re.I)
_TYPE = re.compile(
    r"^(real|integer|logical|double\s*precision|character|complex)\b", re.I)


def identifiers(statement: str) -> set[str]:
    """The names a statement mentions, lower case (literals, strings and
    dotted operators removed)."""
    text = _STRING.sub(" ", statement)
    text = _OPERATORS.sub(" ", text)
    text = _NUMBER.sub(" ", text)
    return {m.group(0).lower() for m in _IDENT.finditer(text)}


def _split_top(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return parts


def declaration(statement: str) -> tuple[list[str], bool] | None:
    """``(names, is_parameter)`` for a type declaration, else ``None``."""
    match = _TYPE.match(statement)
    if not match:
        return None
    rest = statement[match.end():]
    if "::" in rest:
        attributes, entities = rest.split("::", 1)
    else:
        stripped = rest.lstrip()
        if stripped.startswith("("):
            depth = 0
            for index, ch in enumerate(stripped):
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0:
                    break
            attributes, entities = stripped[:index + 1], stripped[index + 1:]
        elif stripped.startswith("*"):
            star = re.match(r"\*\s*\d+", stripped)
            attributes, entities = stripped[:star.end()], stripped[star.end():]
        else:
            attributes, entities = "", stripped
        if re.match(r"\s*function\b", entities, re.I):
            return None
    names = []
    for entity in _split_top(entities):
        found = _IDENT.search(entity)
        if found:
            names.append(found.group(0).lower())
    return names, "parameter" in attributes.lower()


def _routine(text: str, name: str) -> ParsedRoutine:
    for routine in parse_routines(text):
        if routine.name.lower() == name.lower():
            return routine
    raise KeyError(name)


def module_variables(text: str, module: str) -> tuple[set[str], set[str]]:
    """``(variables, parameters)`` declared in the specification part of
    ``module`` (the lines between its MODULE statement and CONTAINS), lower
    case.  The stub branch at the top of the file is skipped: a module is
    taken at its last MODULE statement."""
    lines = text.split("\n")
    start = None
    for number, raw in enumerate(lines, 1):
        match = _MODULE.match(strip_comment(raw))
        if match and match.group(1).lower() == module.lower():
            start = number
    if start is None:
        raise KeyError(module)
    end = next(number for number in range(start + 1, len(lines) + 1)
               if _CONTAINS.match(strip_comment(lines[number - 1])))
    variables: set[str] = set()
    parameters: set[str] = set()
    for _, statement in statements(text, start + 1, end - 1):
        found = declaration(statement)
        if found:
            (parameters if found[1] else variables).update(found[0])
    return variables, parameters


def module_variable_reads(text: str, name: str) -> tuple[str, ...]:
    """The non-PARAMETER variables of its own module that routine ``name``
    mentions, lower case and sorted: names in its body that are neither a
    dummy argument nor declared inside it."""
    routine = _routine(text, name)
    body = statements(text, routine.first, routine.last)
    header = body[0][1]
    local = {routine.name.lower()}
    arguments = re.search(r"\((.*)\)", header)
    if arguments:
        local |= {a.strip().lower() for a in arguments.group(1).split(",")
                  if a.strip()}
    mentioned: set[str] = set()
    for _, statement in body[1:-1]:
        found = declaration(statement)
        if found:
            local.update(found[0])
        mentioned |= identifiers(statement)
    variables, _ = module_variables(text, routine.module)
    return tuple(sorted((mentioned & variables) - local))


_ASSIGNMENT = re.compile(r"^([a-z_][a-z0-9_]*)\s*(\([^=]*\))?\s*=(?!=)", re.I)
_LOGICAL_IF = re.compile(r"^if\s*\(", re.I)


def assignment_targets(text: str, name: str) -> set[str]:
    """The names routine ``name`` assigns to (``x = ...`` and ``x(...) =
    ...``, also after a one-line logical IF), lower case."""
    routine = _routine(text, name)
    out: set[str] = set()
    for _, statement in statements(text, routine.first + 1, routine.last - 1):
        body = statement
        if _LOGICAL_IF.match(body):
            depth = 0
            for index, ch in enumerate(body):
                depth += ch == "("
                depth -= ch == ")"
                if ch == ")" and depth == 0:
                    body = body[index + 1:].strip()
                    break
        if declaration(body):
            continue
        match = _ASSIGNMENT.match(body)
        if match:
            out.add(match.group(1).lower())
    return out


def code_lines(text: str, first: int, last: int) -> list[int]:
    """The lines of ``first``..``last`` (one-based) that hold code: not
    blank, not only a comment, not a preprocessor line."""
    lines = text.split("\n")
    return [n for n in range(first, last + 1)
            if strip_comment(lines[n - 1]).strip()
            and not lines[n - 1].lstrip().startswith("#")]


def routine_map_problems(parsed: Iterable[ParsedRoutine],
                         routines: Iterable[Routine] = ROUTINES,
                         items: dict[str, str] | None = None,
                         text: str | None = None) -> list[str]:
    """What is wrong with ``routines`` as a map of ``parsed``: a routine of
    the file with no entry, an entry for no routine, a name, kind, module or
    line range that differs, a routine with no part, a part with neither an
    owner nor a reason (or both), an owner that is no item, a part outside
    its routine's lines, and a milestone 1 part without a host.  Given the
    file's ``text``, also every code line of a routine (``code_lines``) that
    no part covers, so a part dropped from a routine that keeps others
    fails too.  Empty means every line of every routine in the file has an
    owner or a stated reason."""
    items = ITEMS if items is None else items
    problems: list[str] = []
    by_name = {}
    for routine in routines:
        key = routine.name.lower()
        if key in by_name:
            problems.append(f"{routine.name}: listed twice")
        by_name[key] = routine
    seen = set()
    for found in parsed:
        key = found.name.lower()
        seen.add(key)
        routine = by_name.get(key)
        if routine is None:
            problems.append(f"{found.name} (:{found.first}-{found.last}) has "
                            f"neither an owner nor a reason")
            continue
        if routine.name != found.name:
            problems.append(f"{found.name}: the map writes {routine.name!r}")
        if routine.kind != found.kind:
            problems.append(f"{found.name}: kind {routine.kind!r}, the file "
                            f"says {found.kind!r}")
        if routine.module != (found.module or ""):
            problems.append(f"{found.name}: module {routine.module!r}, the "
                            f"file says {found.module!r}")
        if routine.lines != (found.first, found.last):
            problems.append(f"{found.name}: lines {routine.lines}, the file "
                            f"says {(found.first, found.last)}")
        if not routine.parts:
            problems.append(f"{found.name}: no part has an owner or a "
                            f"reason")
        for part in routine.parts:
            label = f"{found.name} part {part.what!r}"
            if (part.owner is None) == (part.reason is None):
                problems.append(f"{label}: needs exactly one of an owner and "
                                f"a reason")
            if part.reason is not None and not part.reason.strip():
                problems.append(f"{label}: the reason is empty")
            if part.owner is not None and part.owner not in items:
                problems.append(f"{label}: owner {part.owner!r} is no item")
            if part.owner in M1_ITEMS and not part.host:
                problems.append(f"{label}: a milestone 1 part names no host "
                                f"function")
            for first, last in part.lines:
                if not found.first <= first <= last <= found.last:
                    problems.append(f"{label}: lines {first}-{last} lie "
                                    f"outside {found.first}-{found.last}")
        if text is not None:
            covered = {n for part in routine.parts
                       for first, last in part.lines
                       for n in range(first, last + 1)}
            bare = [n for n in code_lines(text, found.first, found.last)
                    if n not in covered]
            if bare:
                spans, start = [], bare[0]
                for prev, n in zip(bare, bare[1:] + [None]):
                    if n != prev + 1:
                        spans.append(f"{start}-{prev}" if prev > start
                                     else f"{start}")
                        start = n
                problems.append(f"{found.name}: code lines {', '.join(spans)} "
                                f"lie in no part")
    for key, routine in by_name.items():
        if key not in seen:
            problems.append(f"{routine.name}: in the map but not in the file")
    return problems
