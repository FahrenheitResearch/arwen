"""Restore WRF real.exe fields and boundary tables through the Rust reader.

File geometry and physics inventory are validated before GPU allocation.
This module adapts state only; the shared forecast runner owns integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


from gpuwm import netcdf_bridge
import numpy as np

# Every science-field entry below has one explicit gpuwm consumer.  The
# importer is intentionally closed-world: variables outside these inventories
# are errors, rather than being hidden in a catch-all auxiliary bucket.
REQUIRED_WRFINPUT = (
    "U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "T_INIT",
    "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD", "QRAIN",
    "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE", "QNSNOW",
    "QNGRAUPEL", "HGT", "FNM", "FNP", "RDNW", "RDN", "DNW",
    "DN", "ZNU", "ZNW", "C1H", "C2H", "C1F", "C2F", "C3H",
    "C4H", "C3F", "C4F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    "F", "E", "SINALPHA", "COSALPHA", "LANDMASK", "LU_INDEX",
    "ISLTYP", "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SNOW",
    "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX", "PSFC", "T2",
    "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
    "CF1", "CF2", "CF3",
)

ALIASES = {
    "XICE": ("XICE", "SEAICE"),
    "ALBBCK": ("ALBBCK", "ALBEDO"),
    "LAI": ("LAI", "LAI12M"),
    "P_TOP": ("P_TOP",),
}

MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNRAIN": "nr", "QNICE": "ni", "QNSNOW": "ns",
    "QNGRAUPEL": "ng", "QNCLOUD": "nc",
    # mp_physics=28 (Thompson aerosol-aware).  QNCLOUD is already above --
    # Morrison declares the same Registry name -- but for mp=28 it stops
    # being an optional diagnosed field and becomes a REQUIRED prognostic
    # (see REQUIRED_QNCLOUD_MICROPHYSICS below).  QNWFA/QNIFA are new here; WRF
    # declares them as scalars with the input stream in their IO string
    # (Registry/registry.new3d_wif:87-90, ``i0rhusdf=(bdy_interp:dt)``), so
    # real.exe writes them into wrfinput exactly as it writes QNRAIN.
    "QNWFA": "nwfa", "QNIFA": "nifa",
    # mp_physics=16 (WDM6).  QNCLOUD and QNRAIN are already above and carry
    # WDM6's prognostic nc/nr unchanged; QNCCN is new here.  It maps to
    # ``nn``, WDM6's own state name for the CCN reservoir -- NOT to NSSL's
    # ``qnn``, which is the same WRF variable in a different scheme's state
    # and is why the two maps stay separate rather than merging.
    "QNCCN": "nn",
    # mp_physics=50 (P3 one-category).  QICE/QNICE/QNRAIN are already above
    # and carry P3's single ice mass and its two number moments unchanged;
    # QIR and QIB are new here.  They are the rime mass and rime volume P3
    # carries INSTEAD of a snow/graupel/hail split, so without a name for
    # them this closed-world importer cannot read a P3 wrfinput at all --
    # ``read_wrfinput`` rejects the file as carrying "unmapped WRF
    # variable(s)" before any inventory check is reached.
    "QIR": "qir", "QIB": "qib",
    # mp_physics=9 (Milbrandt-Yau double-moment, seven class).  QNCLOUD,
    # QNRAIN, QNICE, QNSNOW and QNGRAUPEL are already above and carry
    # Milbrandt's five prognostic number moments unchanged; QHAIL and
    # QNHAIL are new here.  Both names ALSO appear in NSSL_MOISTURE_MAP
    # below, and they map to different state there -- NSSL's number
    # moments are ``qnh``/``qng``/... while Milbrandt's are the same
    # ``nh``/``ng``/... family Morrison and WDM6 use
    # (gpuwm/core/state.py: the mp=9 arm allocates
    # ``qh, nc, nr, ni, ns, ng, nh``).  Same WRF Registry name, two
    # schemes, two state targets: that is exactly why the two maps stay
    # separate rather than merging.
    "QHAIL": "qh", "QNHAIL": "nh",
}

# NSSL reuses several WRF Registry names carried by Morrison, but the state
# names are scheme-native (for example QNRAIN -> qnr, not nr).  Keep this as
# a distinct map so restoration cannot choose a target by whichever attribute
# happens to exist on DomainState.
NSSL_MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
}
ALL_MOISTURE_WRFINPUT = frozenset(MOISTURE_MAP) | frozenset(
    NSSL_MOISTURE_MAP)

# ``real.exe`` writes a physics-package-specific moisture inventory.  The
# first three mass species are active in every gpuwm moist configuration;
# WSM6 adds the three ice-category masses, while Morrison additionally owns
# four transported number moments.  Native option-18 NSSL defaults add hail,
# five two-moment number fields, predicted CCN, and graupel/hail volume.  Its
# Registry aliases overlap Morrison but map to distinct scheme-native state.
# QNCLOUD is a documented optional Morrison restart field (WRF's matched
# default diagnoses cloud number).
BASE_MOISTURE_WRFINPUT = ("QVAPOR", "QCLOUD", "QRAIN")
ICE_MASS_WRFINPUT = ("QICE", "QSNOW", "QGRAUP")
MORRISON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL")
MORRISON_OPTIONAL_MOISTURE_WRFINPUT = ("QNCLOUD",)
THOMPSON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE")
#: mp_physics=28 adds the prognostic droplet number and the two aerosol
#: number tracers to classic Thompson's two moments.  All three are
#: transported scalars in WRF's own Registry (Registry.EM_COMMON:3036's
#: ``scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca``; the Registry also writes qnbca,
#: but Thompson uses it only at wif_input_opt=2 and otherwise returns zero,
#: module_mp_thompson.F:3983-3988) and all
#: three carry the wrfinput stream in their IO strings
#: (Registry.EM_COMMON:542 for QNCLOUD, registry.new3d_wif:87-90 for
#: QNWFA/QNIFA).  Unlike Morrison's QNCLOUD -- which WRF diagnoses and gpuwm
#: therefore treats as optional -- an mp=28 wrfinput that lacked QNCLOUD
#: would start every column at nc = 0, and WRF's terminal clamp
#: (module_mp_thompson.F:3976) would silently hold it at 2/rho rather than
#: raising.  Required, not optional.
THOMPSON_AEROSOL_NUMBER_WRFINPUT = ("QNRAIN", "QNICE", "QNCLOUD",
                                    "QNWFA", "QNIFA")
#: mp_physics=16 (WDM6).  ``Registry.EM_COMMON:3031`` declares the package
#: as ``moist:qv,qc,qr,qi,qs,qg;scalar:qnn,qnc,qnr``, so WDM6's wrfinput is
#: WSM6's six masses plus exactly three transported numbers -- the CCN
#: reservoir QNCCN (:539), the cloud droplet number QNCLOUD (:541) and the
#: rain number QNRAIN (:533).  All three are prognostic, none is diagnosed,
#: so all three are REQUIRED: a WDM6 wrfinput without QNCLOUD would start
#: every column at nc = 0 and WDM6's own slope floor would hold it there,
#: which is the mp=28 hazard in the note above, not Morrison's optional
#: field.
WDM6_NUMBER_WRFINPUT = ("QNCCN", "QNCLOUD", "QNRAIN")
#: The schemes for which QNCLOUD is a required prognostic rather than the
#: optional diagnosed Morrison field.  Consulted by
#: :func:`_restore_active_moisture`, whose "missing is tolerated" branch
#: keys on ``MORRISON_OPTIONAL_MOISTURE_WRFINPUT`` alone and would otherwise
#: extend Morrison's exemption to mp=28.  16 belongs for the same reason:
#: WDM6 PREDICTS the droplet number (Registry.EM_COMMON:3031's
#: ``scalar:qnn,qnc,qnr``), so QNCLOUD is the prognostic its double-moment
#: warm rain is built on, not Morrison's diagnosed convenience field.
#: Renamed off the mp28 spelling when 16 joined -- a set with two members
#: named for one of them is how the next scheme gets forgotten.
#: 9 belongs for the same reason 16 does: Milbrandt-Yau PREDICTS the
#: droplet number (its ``scalar:qnc,...`` package), so QNCLOUD is a
#: prognostic there, not Morrison's diagnosed convenience field.
REQUIRED_QNCLOUD_MICROPHYSICS = frozenset({9, 16, 28})
#: mp_physics=50 (P3 one-category).  ``Registry.EM_COMMON:3038`` declares
#: the package as ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` -- ONE ice
#: mass, and no snow and no graupel anywhere in the package, because P3
#: predicts rime mass and rime volume in place of a snow/graupel/hail
#: split.  QICE is therefore added through this tuple rather than through
#: ``ICE_MASS_WRFINPUT``: putting 50 in the ice-mass branch would demand a
#: QSNOW and a QGRAUP that a P3 wrfinput never contains and that a P3
#: ``DomainState`` has no field to receive (gpuwm/core/state.py:464-478).
#: All four scalars are transported prognostics, and every one carries the
#: wrfinput stream in its Registry IO string -- ``i0rhusdf=(bdy_interp:dt)``
#: on QNICE (Registry.EM_COMMON:523), QNRAIN (:533), QIR (:555) and QIB
#: (:557) -- so real.exe writes all four exactly as it writes Morrison's
#: QNRAIN.  None of them is diagnosed, so unlike Morrison there is no
#: optional member here, and a missing one does not raise inside P3, it
#: silently rewrites the analysis:
#:   * a zero QNICE beside a real QICE is clamped to ``nsmall`` = 1.e-16
#:     (module_mp_p3.F:231, :2573) and the mean-mass ice diameter built
#:     from it (:2578) then indexes the lookup table (:2582) in its
#:     largest-particle bin for the whole column;
#:   * a zero QIR/QIB sends ``calc_bulkRhoRime`` (:6784, called at :2580)
#:     down its ``bi_rim < 1.e-15`` branch, which hard-zeros both and
#:     returns rho_rime = 0 (:6812-6814), declaring every ice particle in
#:     the restored state unrimed -- P3's one distinguishing prognostic,
#:     silently set to nothing.
#: Both are finite, bounded and wrong: the mp=28 terminal-clamp hazard in
#: the note above, in P3's own numbers.
#: mp_physics=9 (Milbrandt-Yau double-moment, seven class).
#: ``Registry.EM_COMMON`` declares the package as
#: ``moist:qv,qc,qr,qi,qs,qg,qh;scalar:qnc,qnr,qni,qns,qng,qnh`` -- WSM6's
#: six masses plus HAIL, and a number moment for every one of the six
#: hydrometeors.  All six numbers are prognostic and none is diagnosed, so
#: all six are REQUIRED, QNCLOUD included: WRF's own driver binds
#: qnc/qnr/qni/qns/qng/qnh INOUT to ``mp_milbrandt2mom_driver``
#: (module_microphysics_driver.F:1857-1862), which is the same argument
#: that puts 16 and 28 in REQUIRED_QNCLOUD_MICROPHYSICS above.  QHAIL is a
#: MASS, so it is listed here rather than folded into ICE_MASS_WRFINPUT,
#: which is shared with the six-species schemes that have no hail at all.
MILBRANDT_MOISTURE_WRFINPUT = (
    "QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNHAIL",
)
P3_MOISTURE_WRFINPUT = ("QICE", "QNICE", "QNRAIN", "QIR", "QIB")
NSSL_MOISTURE_WRFINPUT = (
    "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
)

PHYSICS_FIELD_ALIASES = {
    "landmask": ("LANDMASK",), "xland": ("XLAND",),
    "tsk": ("TSK",), "pblh": ("PBLH",), "ivgtyp": ("IVGTYP", "LU_INDEX"),
    "isltyp": ("ISLTYP",), "vegfra": ("VEGFRA",), "tmn": ("TMN",),
    "xice": ("XICE", "SEAICE"), "swdown": ("SWDOWN",), "glw": ("GLW",),
    "snow": ("SNOW",), "snowh": ("SNOWH",), "smois": ("SMOIS",),
    "tslb": ("TSLB",), "sh2o": ("SH2O",), "psfc": ("PSFC",),
    "t2": ("T2",), "q2": ("Q2",), "th2": ("TH2",),
    "u10": ("U10",), "v10": ("V10",), "snoalb": ("SNOALB",),
    "albbck": ("ALBBCK", "ALBEDO"), "lai": ("LAI",),
    "shdmin": ("SHDMIN",), "shdmax": ("SHDMAX",),
    "ust": ("UST",), "znt": ("ZNT",), "hfx": ("HFX",),
    "qfx": ("QFX",), "lh": ("LH",), "grdflx": ("GRDFLX",),
    # RUC Registry input state -> the existing LSMRUC INOUT carriers.
    "acrunoff": ("ACRUNOFF",), "rhosnf": ("RHOSNF",),
    "snowfallac": ("SNOWFALLAC",), "soilt1": ("SOILT1",),
    # Stock v4.6.1 names this field qke; real.exe products also use QKE.
    "qke": ("qke", "QKE"),
}

RUC_INPUT_FIELDS = frozenset({"ACRUNOFF", "RHOSNF", "SNOWFALLAC", "SOILT1"})
MYNN_QKE_INPUT_FIELDS = frozenset(PHYSICS_FIELD_ALIASES["qke"])

# Optional restart-state fields have explicit consumers in
# ``restore_domain_state``.  They are permitted when present but are not
# synthesized when absent.
OPTIONAL_WRFINPUT = (
    "H_DIABATIC", "RAINNC", "RAINC", "QNCLOUD", "SST", "CANWAT", "LAKEMASK",
    "QNWFA2D", "QNIFA2D",
)

# These are the only non-science variable records allowed through the reader.
# ``Times`` is character metadata and is not copied into ``RestoredDomain.raw``.
EXPLICIT_AUXILIARY_WRFINPUT = (
    "Times", "XTIME", "ITIMESTEP",
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
)

_MASS_3D_DIMS = ("bottom_top", "south_north", "west_east")
_MASS_2D_DIMS = ("south_north", "west_east")
_U_3D_DIMS = ("bottom_top", "south_north", "west_east_stag")
_V_3D_DIMS = ("bottom_top", "south_north_stag", "west_east")
_W_3D_DIMS = ("bottom_top_stag", "south_north", "west_east")
_U_2D_DIMS = ("south_north", "west_east_stag")
_V_2D_DIMS = ("south_north_stag", "west_east")
_SOIL_DIMS = ("soil_layers_stag", "south_north", "west_east")
_WRFINPUT_GEOMETRY_DIMENSIONS = frozenset({
    "bottom_top", "bottom_top_stag", "south_north", "south_north_stag",
    "west_east", "west_east_stag", "soil_layers_stag",
})

# Field-specific staggering is checked while bytes are read, before any GPU
# state exists.  The dimension names are part of the contract as well as the
# resulting shape, so a truncated variable cannot borrow an unrelated
# dimension of the same length.
WRFINPUT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    **{name: _MASS_3D_DIMS for name in (
        "T", "T_INIT", "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD",
        "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE",
        "QNSNOW", "QNGRAUPEL", "QNCLOUD", "QHAIL", "QNDROP",
        "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL", "H_DIABATIC",
        "QIR", "QIB", "QNWFA", "QNIFA", "QNBCA", "qke", "QKE", "qke_adv",
    )},
    "U": _U_3D_DIMS, "V": _V_3D_DIMS, "W": _W_3D_DIMS,
    "PH": _W_3D_DIMS, "PHB": _W_3D_DIMS,
    **{name: _MASS_2D_DIMS for name in (
        "MU", "MUB", "HGT", "MAPFAC_M", "F", "E", "SINALPHA",
        "COSALPHA", "LANDMASK", "LU_INDEX", "ISLTYP", "TSK", "TMN",
        "SNOW", "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX",
        "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
        "XICE", "SEAICE", "ALBBCK", "ALBEDO", "LAI", "LAI12M",
        "SWDOWN", "GLW", "QNWFA2D", "QNIFA2D",
        "PBLH", "UST", "ZNT", "HFX", "QFX", "LH", "GRDFLX", "RAINNC",
        "RAINC", "XLAT", "XLONG", "SST", "CANWAT", "LAKEMASK",
        "ACRUNOFF", "RHOSNF", "SNOWFALLAC", "SOILT1",
    )},
    "MAPFAC_U": _U_2D_DIMS, "MAPFAC_V": _V_2D_DIMS,
    "XLAT_U": _U_2D_DIMS, "XLONG_U": _U_2D_DIMS,
    "XLAT_V": _V_2D_DIMS, "XLONG_V": _V_2D_DIMS,
    **{name: _SOIL_DIMS for name in ("TSLB", "SMOIS", "SH2O")},
    **{name: ("bottom_top",) for name in (
        "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "ZNU",
        "C1H", "C2H", "C3H", "C4H",
    )},
    **{name: ("bottom_top_stag",) for name in (
        "ZNW", "C1F", "C2F", "C3F", "C4F",
    )},
    **{name: () for name in (
        "P_TOP", "CF1", "CF2", "CF3", "XTIME", "ITIMESTEP",
    )},
}


def _mapped_wrfinput_names() -> set[str]:
    names = set(REQUIRED_WRFINPUT) | set(OPTIONAL_WRFINPUT)
    names.update(alias for aliases in ALIASES.values() for alias in aliases)
    names.update(ALL_MOISTURE_WRFINPUT)
    names.update(
        alias for aliases in PHYSICS_FIELD_ALIASES.values() for alias in aliases)
    return names


MAPPED_WRFINPUT = frozenset(_mapped_wrfinput_names())
# Registry writes QNBCA for mp28 even when wif_input_opt != 2. In that
# selection Thompson sets its local and returned BC aerosol to zero
# (module_mp_thompson.F:1807-1811,3983-3988). Retain and validate the input
# without claiming a prognostic black-carbon consumer exists in ArWen.
INACTIVE_AEROSOL_WRFINPUT = frozenset({"QNBCA"})
# Registry.EM_COMMON:3168 allocates qke_adv for every MYNN run. The three
# read/write sites in module_bl_mynn.F:837-841,861-864,1442-1445 are gated
# by bl_mynn_tkeadvect. Retain and validate it only under the explicitly
# selected inactive operation; it is not an alias for the live qke carrier.
INACTIVE_MYNN_WRFINPUT = frozenset({"qke_adv"})
ALLOWED_WRFINPUT = (MAPPED_WRFINPUT | frozenset(EXPLICIT_AUXILIARY_WRFINPUT)
                   | INACTIVE_AEROSOL_WRFINPUT | INACTIVE_MYNN_WRFINPUT)

#: Standard real.exe wrfinput variables the restored model does not consume
#: (F20 conformance defines exactly what is restored; everything else is
#: skipped).  Enumerated explicitly â€” first contact with the production
#: handoff inputs (registered identity e2c6fdf4...) surfaced these â€” so an
#: unanticipated variable still fails loudly instead of being silently
#: ignored.  THM/P_HYD are present-but-unconsumed under the registered
#: use_theta_m=0 restoration; SST/land-use climatology fields are superseded
#: by the mapped surface restoration set.
IGNORED_WRFINPUT = frozenset({
    "BATHYMETRY_FLAG", "CFN", "CFN1", "CLAT", "CLDFRA", "CPLMASK",
    "DTS", "DTSEPS", "DZS", "EROD", "FCX", "FNDALBSI", "FNDICEDEPTH",
    "FNDSNOWH", "FNDSNOWSI", "FNDSOILW", "FRC_URB2D", "GCX", "GOT_VAR_SSO",
    "LAKEFLAG", "LAKE_DEPTH", "LAKE_DEPTH_FLAG", "LANDUSEF",
    "LAT_LL_D", "LAT_LL_T", "LAT_LL_U", "LAT_LL_V", "LAT_LR_D", "LAT_LR_T",
    "LAT_LR_U", "LAT_LR_V", "LAT_UL_D", "LAT_UL_T", "LAT_UL_U", "LAT_UL_V",
    "LAT_UR_D", "LAT_UR_T", "LAT_UR_U", "LAT_UR_V", "LON_LL_D", "LON_LL_T",
    "LON_LL_U", "LON_LL_V", "LON_LR_D", "LON_LR_T", "LON_LR_U", "LON_LR_V",
    "LON_UL_D", "LON_UL_T", "LON_UL_U", "LON_UL_V", "LON_UR_D", "LON_UR_T",
    "LON_UR_U", "LON_UR_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX",
    "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV", "O3_GFS_DU", "P00",
    "PC", "PCB", "P_HYD", "P_STRAT", "QV_BASE", "RDX", "RDY", "RESM",
    "SAVE_TOPO_FROM_REAL", "SHDAVG", "SMCREL", "SNOWC", "SOILCBOT",
    "SOILCTOP", "SR", "STEP_NUMBER", "T00", "THIS_IS_AN_IDEAL_RUN",
    "THM", "TISO", "TLP", "TLP_STRAT", "TOPOSLPX", "TOPOSLPY", "T_BASE",
    "UOCE", "U_BASE", "U_FRAME", "VAR", "VAR_SSO", "VOCE", "V_BASE",
    "V_FRAME", "WATER_DEPTH", "ZETATOP", "ZS", "Z_BASE",
})


# ==========================================================================
# The scheme matrix: which real.exe products this door restores, by name.
# ==========================================================================
#
# A closed-world importer that refuses with "unmapped WRF variable(s):
# ['QAOLI', 'QICE2', ...]" has told the reader nothing they can act on.
# The names in that list are the hydrometeor inventory of ONE WRF
# microphysics package, and the file's own ``MP_PHYSICS`` global attribute
# says which -- ``real.exe`` writes it into every wrfinput it produces.
# So the door reads the attribute FIRST and refuses by scheme name with a
# remedy, and the variable-inventory check below it becomes the second
# line of defence rather than the message a user sees.

#: WRF ``mp_physics`` -> (scheme name, the wrfinput hydrometeor inventory
#: this door restores for it).  Membership here is the door's microphysics
#: contract: exactly the values :func:`_active_moisture_inventory` builds
#: an inventory for, in one table so the refusal and the reader cannot
#: disagree about what is supported.
SUPPORTED_MICROPHYSICS: Mapping[int, str] = MappingProxyType({
    0: "moisture off (dry)",
    1: "Kessler warm rain",
    6: "WSM6",
    8: "Thompson",
    9: "Milbrandt-Yau double-moment (seven class)",
    10: "Morrison double-moment",
    16: "WDM6 double-moment warm rain",
    18: "NSSL double-moment",
    28: "Thompson aerosol-aware",
    50: "P3 one-category",
})

#: ``mp_physics`` values that are real WRF v4.6.1 packages this port does
#: NOT restore, each with the reason and the remedy.  Every one of these
#: writes a wrfinput whose hydrometeor names ArWen has no state field for,
#: so restoring it would mean dropping prognostics on the floor.  The
#: three P3 siblings recite ``gpuwm.config``'s own refusal text, which is
#: where their missing physics is argued in detail.
UNSUPPORTED_MICROPHYSICS: Mapping[int, str] = MappingProxyType({
    2: "Lin et al.",
    3: "WSM3",
    4: "WSM5",
    5: "Ferrier (Eta)",
    7: "Goddard 4-ice",
    11: "CAM 5.1",
    13: "SBU-YLin",
    14: "WDM5",
    17: "NSSL 2-moment 4-ice",
    19: "NSSL 1-moment",
    21: "NSSL 1-moment lfo",
    22: "NSSL 2-moment, no hail",
    24: "WSM7",
    26: "WDM7",
    30: "HUJI spectral bin (fast)",
    32: "HUJI spectral bin (full)",
    40: "Morrison + CESM aerosol",
    51: "P3 with prognostic droplet number",
    52: "P3, two ice categories",
    53: "P3 one category, three-moment ice",
    55: "P3 multi-category with liquid fraction",
    56: "P3 multi-category, three-moment ice",
})

#: WRF ``sf_surface_physics`` -> (scheme name, the soil-layer count the
#: scheme's wrfinput carries).  This is the LAND-SURFACE half of the
#: contract and it is checked separately from microphysics, because the
#: two fail for different reasons and a reader fixing one wants to be told
#: about the other in the same breath.  The counts are WRF's own
#: (module_check_a_mundo.F's num_soil_layers table); the accepted set is
#: :data:`gpuwm.config.LAND_SURFACE_SCHEMES`.
WRF_LAND_SURFACE_SCHEMES: Mapping[int, tuple[str, int]] = MappingProxyType({
    0: ("no land-surface model", 4),
    1: ("5-layer thermal diffusion (slab)", 5),
    2: ("Noah LSM", 4),
    3: ("RUC LSM", 9),
    4: ("Noah-MP LSM", 4),
    5: ("CLM4", 10),
    7: ("Pleim-Xiu LSM", 2),
    8: ("SSiB", 3),
})


def _integral_attribute(value) -> int | None:
    """A NetCDF global attribute as an int, or None if it is not one.

    WRF writes these as NC_INT, but gpuwm decodes NetCDF through the Rust
    bridge, which promotes every numeric type to f64 -- so the same
    ``MP_PHYSICS = 6`` arrives as ``6`` from netCDF4 and ``6.0`` from the
    bridge.  Accepting an integral float here is what makes the scheme
    check read the same fact through either reader; accepting a
    NON-integral one would let ``6.5`` be scored as WSM6.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else None
    if isinstance(value, np.ndarray) and value.size == 1:
        return _integral_attribute(value.reshape(-1)[0])
    return None


def _named_microphysics(mp_physics: int) -> str:
    """``6`` -> ``"mp_physics=6 (WSM6)"``; unknown values stay bare."""
    name = SUPPORTED_MICROPHYSICS.get(
        mp_physics, UNSUPPORTED_MICROPHYSICS.get(mp_physics))
    return (f"mp_physics={mp_physics}"
            if name is None else f"mp_physics={mp_physics} ({name})")


def supported_microphysics_sentence() -> str:
    """The accepted microphysics menu, as one sentence, for every refusal."""
    return "this door restores " + ", ".join(
        f"{value} ({SUPPORTED_MICROPHYSICS[value]})"
        for value in sorted(SUPPORTED_MICROPHYSICS))


def supported_land_surface_sentence() -> str:
    """The accepted land-surface menu, as one sentence, for every refusal."""
    from gpuwm.config import LAND_SURFACE_SCHEMES
    return "this door restores " + ", ".join(
        f"{value} ({WRF_LAND_SURFACE_SCHEMES[value][0]})"
        for value in sorted(LAND_SURFACE_SCHEMES))


def unsupported_microphysics_refusal(mp_physics: int) -> str:
    """One sentence naming the scheme, why it is refused, and the remedy."""
    if mp_physics in UNSUPPORTED_MICROPHYSICS:
        head = (
            f"{_named_microphysics(mp_physics)} is a WRF package this port "
            "does not carry, so its wrfinput hydrometeor inventory has no "
            "ArWen state to be restored into")
    else:
        head = (
            f"mp_physics={mp_physics} is not a WRF microphysics package "
            "this door recognises")
    return (
        f"{head}. {supported_microphysics_sentence()}. "
        "Remedy: re-run real.exe with one of those mp_physics values in "
        "namelist.input -- real.exe writes a DIFFERENT set of moisture "
        "variables per scheme, so an existing wrfinput cannot be "
        "reinterpreted under another one.")


def unsupported_land_surface_refusal(sf_surface_physics: int) -> str:
    """One sentence naming the land-surface scheme and the remedy."""
    entry = WRF_LAND_SURFACE_SCHEMES.get(sf_surface_physics)
    if entry is None:
        head = (f"sf_surface_physics={sf_surface_physics} is not a WRF "
                "land-surface scheme this door recognises")
        soil = ""
    else:
        head = (f"sf_surface_physics={sf_surface_physics} ({entry[0]}) is a "
                "WRF land-surface scheme this port does not carry")
        soil = (f" Its wrfinput carries {entry[1]} soil layers and the soil "
                "state that scheme writes, which is not the state ArWen's "
                "land-surface drivers read.")
    return (
        f"{head}.{soil} {supported_land_surface_sentence()}. "
        "Remedy: re-run real.exe with one of those sf_surface_physics "
        "values in namelist.input -- the soil geometry and the soil "
        "fields real.exe writes are chosen by this setting, so an "
        "existing wrfinput cannot be reinterpreted under another one.")


def check_supported_schemes(attributes: Mapping[str, object], *,
                            source: str) -> dict[str, int]:
    """Refuse a wrfinput whose physics packages this door cannot restore.

    ``attributes`` is a wrfinput's global attribute mapping.  Returns the
    two resolved scheme ids on success.

    BOTH halves are reported together.  A reader who has just been told
    their microphysics is unported will change it, re-run ``real.exe`` for
    an hour, and be told about their land-surface scheme; one refusal
    listing both is one round trip instead of two.
    """
    from gpuwm.config import LAND_SURFACE_SCHEMES

    problems = []
    resolved = {}
    for key, accepted, refusal in (
            ("MP_PHYSICS", set(SUPPORTED_MICROPHYSICS),
             unsupported_microphysics_refusal),
            ("SF_SURFACE_PHYSICS", set(LAND_SURFACE_SCHEMES),
             unsupported_land_surface_refusal)):
        if key not in attributes:
            # An absent attribute is not agreement.  real.exe writes both
            # into every wrfinput it produces; a file without them was not
            # written by real.exe, or was rewritten by something that
            # dropped them, and either way this door cannot establish
            # which physics the moisture inventory belongs to.
            problems.append(
                f"{source} has no {key} global attribute, so the physics "
                "package that produced it cannot be established.  Every "
                "real.exe wrfinput carries one; a file that does not was "
                "not written by real.exe or was rewritten by a tool that "
                "dropped it.")
            continue
        value = _integral_attribute(attributes[key])
        if value is None:
            problems.append(
                f"{source} global attribute {key}={attributes[key]!r} is "
                "not an integer scheme id")
            continue
        resolved[key] = value
        if value not in accepted:
            problems.append(f"{source}: {refusal(value)}")
    if problems:
        raise ValueError("\n".join(problems))
    return resolved
@dataclass(frozen=True)
class RestoredDomain:
    path: Path
    raw: Mapping[str, np.ndarray]
    dimensions: Mapping[str, int]
    global_attributes: Mapping[str, object]
    mapped_variables: tuple[str, ...]
    auxiliary_variables: tuple[str, ...]

    def wrf_frame(self) -> dict[str, np.ndarray]:
        """CPU inverse of the restored atmospheric mapping."""
        raw = self.raw
        theta_prime = np.asarray(
            (raw["T"] + np.float32(300.0)) - raw["T_INIT"],
            dtype=np.float32)
        return {
            "U": raw["U"].copy(), "V": raw["V"].copy(),
            "W": raw["W"].copy(),
            "T": np.asarray(raw["T_INIT"] + theta_prime
                            - np.float32(300.0), dtype=np.float32),
            "PH": raw["PH"].copy(), "MU": raw["MU"].copy(),
            "PHB": raw["PHB"].copy(), "MUB": raw["MUB"].copy(),
            "QVAPOR": raw["QVAPOR"].copy(),
        }


def _read_numeric(variable) -> np.ndarray:
    value = np.ma.asarray(variable[...])
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"WRF input variable {variable.name} contains masked data")
    array = np.asarray(value)
    if variable.dimensions and variable.dimensions[0] == "Time":
        if array.shape[0] != 1:
            raise ValueError(
                f"WRF input {variable.name} must carry exactly one Time record")
        array = array[0]
    if array.dtype.kind not in "iufb":
        raise TypeError(f"WRF input {variable.name} is not numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"WRF input {variable.name} contains non-finite values")
    # ``np.ascontiguousarray`` promotes a zero-dimensional value to ``(1,)``.
    # WRF writes Registry scalars as ``(Time,)`` records, so preserve the
    # scalar produced by removing that singleton Time dimension.
    return np.array(array, copy=True, order="C")

def _explicit_wrfinput_dimensions(
        dimensions: Mapping[str, int]) -> Mapping[str, int]:
    """Validate a caller-pinned, non-N5S WRF domain geometry.

    Requiring the complete seven-dimension contract keeps an arbitrary file
    from defining its own expected geometry and thereby making a truncated
    but self-consistent handoff look valid.
    """
    if not isinstance(dimensions, Mapping):
        raise TypeError("expected_dimensions must be a mapping")
    names = set(dimensions)
    missing = sorted(_WRFINPUT_GEOMETRY_DIMENSIONS - names)
    extra = sorted(names - _WRFINPUT_GEOMETRY_DIMENSIONS)
    if missing or extra:
        raise ValueError(
            "explicit WRF geometry dimension inventory mismatch: "
            f"missing={missing}, extra={extra}")
    normalized = {}
    for name in sorted(_WRFINPUT_GEOMETRY_DIMENSIONS):
        value = dimensions[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0):
            raise ValueError(
                f"explicit WRF geometry {name} must be a positive integer")
        normalized[name] = int(value)
    for mass, staggered in (
            ("bottom_top", "bottom_top_stag"),
            ("south_north", "south_north_stag"),
            ("west_east", "west_east_stag")):
        if normalized[staggered] != normalized[mass] + 1:
            raise ValueError(
                f"explicit WRF geometry {staggered} must equal {mass} + 1")
    return MappingProxyType(normalized)


def _active_moisture_inventory(cfg) -> tuple[frozenset[str], frozenset[str]]:
    """Return required/allowed wrfinput moisture names for ``cfg``.

    ``cfg=None`` is the registered N5S compatibility path and retains the
    historical Morrison contract verbatim.
    """
    if cfg is None:
        required = frozenset(
            BASE_MOISTURE_WRFINPUT + ICE_MASS_WRFINPUT
            + MORRISON_NUMBER_WRFINPUT)
        return required, required | frozenset(
            MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if not hasattr(cfg, "moist") or not hasattr(cfg, "mp_physics"):
        raise TypeError("cfg must expose moist and mp_physics")
    if not isinstance(cfg.moist, (bool, np.bool_)):
        raise TypeError("cfg.moist must be boolean")
    if (isinstance(cfg.mp_physics, (bool, np.bool_))
            or not isinstance(cfg.mp_physics, (int, np.integer))):
        raise TypeError("cfg.mp_physics must be an integer")
    moist = bool(cfg.moist)
    mp_physics = int(cfg.mp_physics)
    if mp_physics not in SUPPORTED_MICROPHYSICS:
        # NAMED, with a remedy.  This used to read "unsupported active
        # wrfinput mp_physics=55", which tells a reader the number they
        # already typed and nothing else.
        raise ValueError(unsupported_microphysics_refusal(mp_physics))
    if not moist:
        if mp_physics != 0:
            raise ValueError(
                f"mp_physics={mp_physics} requires cfg.moist=True")
        return frozenset(), frozenset()
    required = BASE_MOISTURE_WRFINPUT
    # 16 belongs with the ice-carrying set: wdm6scheme's moist inventory
    # (Registry.EM_COMMON:3031) is qv,qc,qr,qi,qs,qg -- WSM6's, character
    # for character.  WDM6 is double-moment in the WARM half only.
    if mp_physics in (6, 8, 9, 10, 16, 18, 28):
        required += ICE_MASS_WRFINPUT
    if mp_physics == 9:
        # Milbrandt-Yau: WSM6's six masses (added just above) plus hail
        # and six number moments.  See MILBRANDT_MOISTURE_WRFINPUT.
        required += MILBRANDT_MOISTURE_WRFINPUT
    if mp_physics == 16:
        required += WDM6_NUMBER_WRFINPUT
    if mp_physics == 8:
        required += THOMPSON_NUMBER_WRFINPUT
    elif mp_physics == 28:
        # Thompson aerosol-aware: classic Thompson's two moments plus the
        # prognostic droplet number and the two aerosol tracers.
        required += THOMPSON_AEROSOL_NUMBER_WRFINPUT
    elif mp_physics == 10:
        required += MORRISON_NUMBER_WRFINPUT
    elif mp_physics == 18:
        required += NSSL_MOISTURE_WRFINPUT
    elif mp_physics == 50:
        # P3 one-category.  Its whole inventory beyond the three base
        # masses arrives in one tuple because its ice mass does NOT come
        # from ``ICE_MASS_WRFINPUT`` -- see P3_MOISTURE_WRFINPUT above.
        # QSNOW and QGRAUP therefore stay out of ``allowed`` as well as out
        # of ``required``, so a six-species wrfinput handed to a P3 config
        # is refused as carrying inactive moisture rather than restored
        # with two frozen species dropped on the floor.
        required += P3_MOISTURE_WRFINPUT
    allowed = frozenset(required)
    if mp_physics == 10:
        allowed |= frozenset(MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    return frozenset(required), allowed


def _active_moisture_map(cfg) -> Mapping[str, str]:
    """Return the exact WRF-name -> DomainState-name map for ``cfg``."""
    _, allowed = _active_moisture_inventory(cfg)
    if cfg is None or int(cfg.mp_physics) != 18:
        candidates = MOISTURE_MAP
    else:
        candidates = NSSL_MOISTURE_MAP
    return MappingProxyType({
        wrf_name: state_name
        for wrf_name, state_name in candidates.items()
        if wrf_name in allowed
    })


def _validate_wrfinput_geometry(name: str, variable,
                                expected_extents: Mapping[str, int],
                                value: np.ndarray) -> None:
    expected_dimensions = WRFINPUT_DIMENSIONS.get(name)
    if expected_dimensions is None:
        raise ValueError(f"WRF input variable {name} has no mapped geometry")
    actual_dimensions = tuple(variable.dimensions)
    if actual_dimensions[:1] == ("Time",):
        actual_dimensions = actual_dimensions[1:]
    try:
        expected_shape = tuple(
            expected_extents[dim] for dim in expected_dimensions)
    except KeyError as exc:
        raise ValueError(
            f"pinned WRF geometry has no dimension {exc.args[0]} for "
            f"WRF input {name}") from exc
    if actual_dimensions != expected_dimensions or value.shape != expected_shape:
        raise ValueError(
            f"WRF input {name} shape mismatch: expected pinned {expected_shape} "
            f"on {expected_dimensions}, got {value.shape} on {actual_dimensions}")


def read_wrfinput(path: str | Path, *, require_complete: bool = True,
                  expected_dimensions: Mapping[str, int],
                  cfg=None, check_schemes: bool = True,
                  ) -> RestoredDomain:
    """Read one wrfinput file without importing CuPy.

    ``expected_dimensions`` pins all seven WRF extents the caller expects,
    so an arbitrary file cannot define its own geometry and thereby make a
    truncated but self-consistent handoff look valid.  ``cfg`` selects the
    exact active hydrometeor inventory; ``cfg=None`` retains the historical
    Morrison contract the registered N5S case is scored against.

    ``check_schemes`` reads the file's own ``MP_PHYSICS`` and
    ``SF_SURFACE_PHYSICS`` global attributes and refuses an unported
    package BY NAME before any variable inventory is consulted (see
    :func:`check_supported_schemes`).  It is the front door's default and
    is off only for the registered verification fixtures, which are
    synthesised without global attributes.
    """
    path = Path(path)
    expected_extents = _explicit_wrfinput_dimensions(expected_dimensions)
    required_moisture, allowed_moisture = _active_moisture_inventory(cfg)
    # Foreign input: WRF real.exe's own wrfinput, decoded field by
    # field through the Rust bridge. Times is validated by file identity.
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        if check_schemes:
            # BEFORE the inventory check below, on purpose.  An unported
            # package's hydrometeor names would otherwise surface as
            # "unmapped WRF variable(s): ['QAOLI', 'QICE2', ...]", which
            # names ten symbols and not the one fact -- the scheme -- that
            # a reader can act on.
            check_supported_schemes(attrs, source=str(path))
        unknown = sorted(
            set(dataset.variables) - ALLOWED_WRFINPUT - IGNORED_WRFINPUT)
        if unknown:
            # Reached only when the scheme attributes said the package is
            # supported and the file still carries names this door has no
            # state for -- a genuinely unrecognised variable set.  Say
            # which scheme claimed them, so the disagreement is visible.
            claimed = _integral_attribute(attrs.get("MP_PHYSICS"))
            claim = ("" if claimed is None else
                     f" The file declares {_named_microphysics(claimed)},"
                     " whose inventory this door does map, so these names"
                     " are outside every scheme it knows.")
            raise ValueError(
                f"{path} has unmapped WRF variable(s): {unknown}.{claim}")
        raw = {}
        for name, variable in dataset.variables.items():
            if name == "Times" or name in IGNORED_WRFINPUT:
                continue
            value = _read_numeric(variable)
            _validate_wrfinput_geometry(
                name, variable, expected_extents, value)
            raw[name] = value
    _validate_supplied_physics_fields(raw, cfg, attrs)
    if "QNBCA" in raw and int(getattr(cfg, "wif_input_opt", 0)) == 2:
        raise NotImplementedError(
            "QNBCA is supplied with wif_input_opt=2, but the black-carbon "
            "state/physics consumer is not implemented")
    present_moisture = set(raw) & ALL_MOISTURE_WRFINPUT
    extra_moisture = sorted(present_moisture - allowed_moisture)
    if extra_moisture:
        raise ValueError(
            f"{path} has inactive WRF moisture variable(s) for the active "
            f"physics: {extra_moisture}")
    if require_complete:
        non_moisture_required = (
            set(REQUIRED_WRFINPUT) - ALL_MOISTURE_WRFINPUT)
        missing = sorted(name for name in non_moisture_required
                         if name not in raw)
        missing.extend(sorted(required_moisture - present_moisture))
        if int(getattr(cfg, "mp_physics", 0)) == 28:
            missing.extend(name for name in ("QNWFA2D", "QNIFA2D")
                           if name not in raw)
        missing.extend(sorted(
            name for name, alternatives in ALIASES.items()
            if not any(alias in raw for alias in alternatives)))
        if missing:
            raise ValueError(f"{path} is missing mapped WRF variable(s): {missing}")
    mapped = MAPPED_WRFINPUT & set(raw)
    auxiliary = (set(EXPLICIT_AUXILIARY_WRFINPUT) - {"Times"}) & set(raw)
    return RestoredDomain(
        path=path, raw=MappingProxyType(raw),
        dimensions=MappingProxyType(dimensions),
        global_attributes=MappingProxyType(attrs),
        mapped_variables=tuple(sorted(mapped)),
        auxiliary_variables=tuple(sorted(auxiliary)))


def _validate_supplied_physics_fields(raw, cfg, attributes):
    """Every newly admitted physics field has a selected, named consumer."""
    surface = (getattr(cfg, "sf_surface_physics", None) if cfg is not None
               else _integral_attribute(attributes.get("SF_SURFACE_PHYSICS")))
    pbl = (getattr(cfg, "bl_pbl_physics", None) if cfg is not None
           else _integral_attribute(attributes.get("BL_PBL_PHYSICS")))
    ruc = RUC_INPUT_FIELDS & raw.keys()
    if ruc and surface != 3:
        raise ValueError(
            f"WRF physics input {sorted(ruc)} requires the RUC "
            "sf_surface_physics=3 consumer")
    mynn = (MYNN_QKE_INPUT_FIELDS | INACTIVE_MYNN_WRFINPUT) & raw.keys()
    if mynn and pbl != 5:
        raise ValueError(
            f"WRF physics input {sorted(mynn)} requires the MYNN "
            "bl_pbl_physics=5 consumer")
    if "qke" in raw and "QKE" in raw and not np.array_equal(raw["qke"], raw["QKE"]):
        raise ValueError("WRF physics input qke and QKE contain conflicting aliases")
    if bool(getattr(cfg, "bl_mynn_tkeadvect", False)):
        raise NotImplementedError(
            "bl_mynn_tkeadvect=True requires the qke_adv transport and "
            "MYNN feedback operation, which is not implemented")
    if "qke_adv" in raw and getattr(cfg, "bl_mynn_tkeadvect", None) is not False:
        raise ValueError(
            "WRF qke_adv can be retained only with an explicit "
            "bl_mynn_tkeadvect=False selection; active qke_adv transport "
            "and MYNN feedback are not implemented")

def _expect_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"WRF {name} shape {value.shape} != expected {expected}")


def _first(raw: Mapping[str, np.ndarray], names: Sequence[str], *,
           required: bool = True):
    for name in names:
        if name in raw:
            return raw[name]
    if required:
        raise ValueError(f"WRF input is missing every alias in {tuple(names)}")
    return None


def _restore_active_moisture(state, raw: Mapping[str, np.ndarray], cfg,
                             array_module) -> None:
    """Restore the scheme-native moisture fields and their RK ``*0`` copies."""
    # The QNCLOUD exemption is Morrison's alone (WRF diagnoses cloud number
    # there).  For mp=28 QNCLOUD is the prognostic droplet number the whole
    # scheme turns on, and a wrfinput without it must fail here rather than
    # start every column at WRF's 2/rho terminal clamp floor
    # (module_mp_thompson.F:3976), which is finite, bounded and wrong.
    optional_moisture = frozenset(MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if (cfg is not None
            and int(cfg.mp_physics) in REQUIRED_QNCLOUD_MICROPHYSICS):
        optional_moisture = frozenset()
    # Surface emission tendencies are inputs too, and must be installed
    # before microphysics_init tests whether to fill a synthetic profile.
    for wrf_name, state_name in (("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        target = getattr(state, state_name, None)
        if target is not None and wrf_name in raw:
            _expect_shape(wrf_name, raw[wrf_name], target.shape)
            target[...] = array_module.asarray(
                raw[wrf_name], dtype=array_module.float32)
    state_names = []
    for wrf_name, state_name in _active_moisture_map(cfg).items():
        target = getattr(state, state_name, None)
        if target is None:
            raise ValueError(
                f"DomainState lacks active mp_physics={cfg.mp_physics} "
                f"field {state_name} for WRF {wrf_name}")
        state_names.append(state_name)
        if wrf_name in raw:
            _expect_shape(wrf_name, raw[wrf_name], target.shape)
            target[...] = array_module.asarray(
                raw[wrf_name], dtype=array_module.float32)
        elif wrf_name not in optional_moisture:
            raise ValueError(
                f"WRF input lacks active mp_physics={cfg.mp_physics} "
                f"field {wrf_name}")

    if cfg is not None and int(cfg.mp_physics) == 16:
        # WRF module_mp_wdm6.F:220-227 replaces the complete cold-start
        # CCN reservoir with ccn_conc on its first step, even when wrfinput
        # carries a nonzero QNCCN. Preserve that initialization after the
        # file restore, before saving the RK-beginning copies. Checkpoint
        # restoration has its separate transport and does not enter here.
        state.nn[...] = array_module.float32(cfg.wdm6_ccn_conc)

    # DomainState owns RK-beginning copies only for prognostic fields.  Sync
    # every active one that exists, including all ten NSSL-only fields.
    for state_name in dict.fromkeys(state_names):
        source = getattr(state, state_name)
        initial = getattr(state, f"{state_name}0", None)
        if initial is not None:
            initial[...] = source


def wrf_coordinate_and_base(restored):
    """Reconstruct shared setup objects from file values, without regeneration."""
    from gpuwm.core.grid import VerticalCoord, BaseState

    raw = restored.raw
    names = ('znw', 'znu', 'dnw', 'rdnw', 'dn', 'rdn', 'fnp', 'fnm',
             'c1f', 'c2f', 'c3f', 'c4f', 'c1h', 'c2h', 'c3h', 'c4h')
    top = float(np.asarray(raw['P_TOP']).item())
    coord = VerticalCoord(**{name: raw[name.upper()] for name in names},
                          hybrid_opt=int(restored.global_attributes['HYBRID_OPT']),
                          etac=float(restored.global_attributes['ETAC']), p_top=top)
    # WRF module_initialize_real.F:3785/5106 writes T_INIT minus t0.
    # DomainState.thb is absolute base potential temperature.
    theta_base = np.asarray(raw['T_INIT'], np.float32) + np.float32(300.0)
    base = BaseState(raw['MUB'], top, raw['PB'], raw['ALB'], theta_base,
                     raw['PHB'], raw['HGT'])
    return coord, base


def restore_domain_state(restored: RestoredDomain, cfg, *, scratch_arena=None,
                         dycore_state_workspace=None, radiation=None,
                         radiation_start_time=None, radiation_latitude=None,
                         radiation_longitude=None):
    """Upload one CPU-restored domain into a production ``DomainState``."""
    import cupy as cp
    from gpuwm.core.state import DomainState

    raw = restored.raw
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    expected = {
        "U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx),
        "W": (nz + 1, ny, nx), "T": (nz, ny, nx),
        "PH": (nz + 1, ny, nx), "MU": (ny, nx),
        "PHB": (nz + 1, ny, nx), "MUB": (ny, nx),
        "T_INIT": (nz, ny, nx), "P": (nz, ny, nx),
        "PB": (nz, ny, nx), "AL": (nz, ny, nx),
        "ALB": (nz, ny, nx),
    }
    for name, shape in expected.items():
        _expect_shape(name, raw[name], shape)
    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    state = DomainState(cfg, **state_kwargs)
    coord, base = wrf_coordinate_and_base(restored)
    state.load_base(coord, base)

    for wrf_name, state_name in (("U", "u"), ("V", "v"), ("W", "w"),
                                 ("PH", "php"), ("MU", "mup")):
        getattr(state, state_name)[...] = cp.asarray(raw[wrf_name], dtype=cp.float32)
    state.thp[...] = cp.asarray(
        (raw["T"].astype(np.float32) + np.float32(300.0))
        - base.thb, dtype=cp.float32)
    state.p[...] = cp.asarray(raw["P"] + raw["PB"], dtype=cp.float32)
    state.al[...] = cp.asarray(raw["AL"], dtype=cp.float32)
    state.alt[...] = cp.asarray(raw["AL"] + raw["ALB"], dtype=cp.float32)
    for name in ("CF1", "CF2", "CF3"):
        source = np.asarray(raw[name]).reshape(-1)
        _expect_shape(name, source, (1,))
        setattr(state, name.lower(), cp.float32(source[0]))
    state.set_map_coriolis(
        raw["MAPFAC_M"], raw["MAPFAC_U"], raw["MAPFAC_V"], raw["F"],
        raw["E"], sina=raw["SINALPHA"], cosa=raw["COSALPHA"])
    _restore_active_moisture(state, raw, cfg, cp)
    if "H_DIABATIC" in raw:
        state.h_diabatic[...] = cp.asarray(raw["H_DIABATIC"], dtype=cp.float32)

    for current, initial in (
            ("u", "u0"), ("v", "v0"), ("w", "w0"),
            ("thp", "thp0"), ("php", "php0"), ("mup", "mup0")):
        source = getattr(state, current, None)
        target = getattr(state, initial, None)
        if source is not None and target is not None:
            target[...] = source

    return state


def initialize_wrfinput_physics(state, restored, cfg, *, radiation=None,
                               radiation_start_time=None, radiation_latitude=None,
                               radiation_longitude=None, landuse=None,
                               constant_glw_wm2=None, cam_ozone=None):
    """Initialize physics from the file's actual surface and land categories."""
    _validate_supplied_physics_fields(restored.raw, cfg, restored.global_attributes)
    import cupy as cp
    from gpuwm.core.physics import initialize_physics

    raw = restored.raw
    xice = _first(raw, ALIASES["XICE"])
    albbck = _first(raw, ALIASES["ALBBCK"])
    lai = _first(raw, ALIASES["LAI"])
    swdown = _first(raw, ("SWDOWN",), required=False)
    glw = (constant_glw_wm2 if constant_glw_wm2 is not None
           else _first(raw, ("GLW",), required=False))
    pblh = _first(raw, ("PBLH",), required=False)
    # real.exe leaves soil columns 0.0 at water points (WRF never reads
    # them there; it uses SST/TSK).  gpuwm's health gate bounds TSLB
    # globally, so fill water columns with TSK â€” the same values WRF's
    # own surface init uses over water â€” before restoration.
    landmask = np.asarray(raw["LANDMASK"])
    tslb = np.array(raw["TSLB"], copy=True)
    water = landmask < 0.5
    if np.any(water):
        tslb[:, water] = np.broadcast_to(
            np.asarray(raw["TSK"])[water], tslb[:, water].shape)
    raw = {**raw, "TSLB": tslb}  # alias loop below re-reads raw (immutable)
    driver = initialize_physics(
        state, cfg, landmask=raw["LANDMASK"], tsk=raw["TSK"],
        landuse=landuse, xland=raw.get("XLAND"),
        landuse_dataset=str(restored.global_attributes["MMINLU"]),
        sst=raw.get("SST", raw["TSK"]),
        soil_temperature=tslb, soil_moisture=raw["SMOIS"],
        liquid_moisture=raw["SH2O"], ivgtyp=raw["LU_INDEX"],
        isltyp=raw["ISLTYP"], vegfra=raw["VEGFRA"], tmn=raw["TMN"],
        xice=xice, snow=raw["SNOW"], snow_depth=raw["SNOWH"],
        swdown=(0.0 if swdown is None else swdown),
        # These diagnostics are not wrfinput fields for this Registry.  WRF
        # v4.6.1 phy_init initializes all three to zero before the first
        # radiation/PBL calls, so preserve that start-of-run convention.
        glw=glw,
        pblh=(0.0 if pblh is None else pblh),
        radiation=radiation,
        radiation_start_time=radiation_start_time,
        radiation_latitude=radiation_latitude,
        radiation_longitude=radiation_longitude,
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
    for field in driver.fields:
        if field == "glw" and constant_glw_wm2 is not None:
            continue
        aliases = PHYSICS_FIELD_ALIASES.get(field, (field.upper(),))
        value = _first(raw, aliases, required=False)
        if value is not None:
            if value.shape != driver.fields[field].shape:
                raise ValueError(
                    f"WRF physics field {aliases[0]} shape {value.shape} != "
                    f"gpuwm {field} shape {driver.fields[field].shape}")
            driver.fields[field][...] = cp.asarray(
                value, dtype=driver.fields[field].dtype)
    driver.fields["albbck"][...] = cp.asarray(albbck, dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    if "RAINNC" in raw:
        driver.microphysics.rainnc[...] = cp.asarray(raw["RAINNC"], dtype=cp.float32)
    if driver.rainc is not None and "RAINC" in raw:
        driver.rainc[...] = cp.asarray(raw["RAINC"], dtype=cp.float32)
    return driver

def _decode_times(variable) -> tuple[datetime, ...]:
    data = np.ma.asarray(variable[...])
    if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
        raise ValueError(f"WRF time variable {variable.name} contains masked data")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 19:
        raise ValueError(
            f"WRF time variable {variable.name} must have shape (records, 19), "
            f"got {data.shape}")
    values = []
    for row in data:
        try:
            text = b"".join(np.asarray(row, dtype="S1").tolist()).decode(
                "ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"WRF time variable {variable.name} is not ASCII") from exc
        try:
            value = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                f"WRF time variable {variable.name} has invalid timestamp "
                f"{text!r}") from exc
        if value.strftime("%Y-%m-%d_%H:%M:%S") != text:
            raise ValueError(
                f"WRF time variable {variable.name} timestamp is not canonical: "
                f"{text!r}")
        values.append(value)
    return tuple(values)


_WRFBDY_FIELDS = {
    "u": ("U", "bottom_top", "south_north", "west_east_stag"),
    "v": ("V", "bottom_top", "south_north_stag", "west_east"),
    "theta": ("T", "bottom_top", "south_north", "west_east"),
    "phi": ("PH", "bottom_top_stag", "south_north", "west_east"),
    "mu": ("MU", None, "south_north", "west_east"),
    "qv": ("QVAPOR", "bottom_top", "south_north", "west_east"),
}


def _wrfbdy_side_table(variable, index: int, side_name: str) -> np.ndarray:
    """Transpose WRF ``(width,z,side)`` into gpuwm's side convention."""
    value = np.ma.asarray(variable[index])
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"wrfbdy_d01 {variable.name} contains masked data")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        axes = ((1, 2, 0) if side_name in ("west", "east")
                else (1, 0, 2))
        value = np.transpose(value, axes)
    elif value.ndim == 2:
        value = (np.transpose(value, (1, 0))[None]
                 if side_name in ("west", "east") else value[None])
    else:
        raise ValueError(
            f"wrfbdy_d01 {variable.name} must be a 2-D or 3-D side table")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"wrfbdy_d01 {variable.name} contains non-finite data")
    return np.ascontiguousarray(value)


def _boundary_strip(field: np.ndarray, side_name: str,
                    width: int) -> np.ndarray:
    if side_name == "west":
        return field[..., :width]
    if side_name == "east":
        return field[..., -width:][..., ::-1]
    if side_name == "south":
        return field[..., :width, :]
    return field[..., -width:, :][..., ::-1, :]


def _mass_from_boundary_sides(initial_mu: np.ndarray,
                              sides: Mapping[str, np.ndarray]) -> np.ndarray:
    """Restore a boundary-frame MU field from gpuwm-oriented side tables."""
    mu = np.asarray(initial_mu, dtype=np.float32).copy()
    width = sides["west"].shape[-1]
    for distance in range(width):
        mu[:, distance] = sides["west"][0, :, distance]
        mu[:, -1 - distance] = sides["east"][0, :, distance]
        mu[distance, :] = sides["south"][0, distance, :]
        mu[-1 - distance, :] = sides["north"][0, distance, :]
    return mu


def _wrf_and_gpuwm_mass_weights(restored: RestoredDomain,
                                mu: np.ndarray
                                ) -> tuple[dict[str, np.ndarray],
                                           dict[str, np.ndarray]]:
    """Return source-WRF and target-gpuwm FP32 dry-mass weights.

    ``real_em.F`` calls ``couple`` before packing wrfbdy.  At mass points
    that routine retains separate perturbation/base products, whereas
    gpuwm's producer first forms total column mass.  U/V additionally use
    WRF's one-sided physical faces and staggered mass averages.  Keeping the
    two expression trees distinct makes read-time normalization deterministic
    instead of relying on algebraic equivalence across FP32 roundoff.
    """
    raw = restored.raw
    required = {
        "MUB", "C1H", "C2H", "C1F", "C2F",
        "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"wrfinput lacks wrfbdy coupling variable(s): {missing}")
    mub = np.asarray(raw["MUB"], dtype=np.float32)
    if mu.shape != mub.shape:
        raise ValueError(
            f"wrfbdy MU frame {mu.shape} != wrfinput MUB {mub.shape}")
    c1h = np.asarray(raw["C1H"], dtype=np.float32)[:, None, None]
    c2h = np.asarray(raw["C2H"], dtype=np.float32)[:, None, None]
    c1f = np.asarray(raw["C1F"], dtype=np.float32)[:, None, None]
    c2f = np.asarray(raw["C2F"], dtype=np.float32)[:, None, None]

    # WRF v4.6.1 module_big_step_utilities_em.F:423-506,576-606.
    wrf_muu = np.empty((mu.shape[0], mu.shape[1] + 1), dtype=np.float32)
    wrf_muv = np.empty((mu.shape[0] + 1, mu.shape[1]), dtype=np.float32)
    wrf_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (mu[:, 1:] + mu[:, :-1]
                           + mub[:, 1:] + mub[:, :-1]), dtype=np.float32)
    wrf_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (mu[1:, :] + mu[:-1, :]
                           + mub[1:, :] + mub[:-1, :]), dtype=np.float32)
    wrf_muu[:, 0] = np.asarray(mu[:, 0] + mub[:, 0], dtype=np.float32)
    wrf_muu[:, -1] = np.asarray(mu[:, -1] + mub[:, -1], dtype=np.float32)
    wrf_muv[0, :] = np.asarray(mu[0, :] + mub[0, :], dtype=np.float32)
    wrf_muv[-1, :] = np.asarray(mu[-1, :] + mub[-1, :], dtype=np.float32)
    wrf_half = np.asarray(
        c1h * mu[None] + (c1h * mub[None] + c2h), dtype=np.float32)
    wrf_full = np.asarray(
        c1f * mu[None] + (c1f * mub[None] + c2f), dtype=np.float32)

    # gpuwm lateral_bc._coupled_device_fields: total MU is formed first.
    total = np.asarray(mub + mu, dtype=np.float32)
    gpu_muu = np.empty_like(wrf_muu)
    gpu_muv = np.empty_like(wrf_muv)
    gpu_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (total[:, 1:] + total[:, :-1]),
        dtype=np.float32)
    gpu_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (total[1:, :] + total[:-1, :]),
        dtype=np.float32)
    gpu_muu[:, 0], gpu_muu[:, -1] = total[:, 0], total[:, -1]
    gpu_muv[0, :], gpu_muv[-1, :] = total[0, :], total[-1, :]
    gpu_half = np.asarray(c1h * total[None] + c2h, dtype=np.float32)
    gpu_full = np.asarray(c1f * total[None] + c2f, dtype=np.float32)

    wrf = {
        "u": np.asarray(c1h * wrf_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * wrf_muv[None] + c2h, dtype=np.float32),
        "theta": wrf_half, "phi": wrf_full, "qv": wrf_half,
    }
    gpuwm = {
        "u": np.asarray(c1h * gpu_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * gpu_muv[None] + c2h, dtype=np.float32),
        "theta": gpu_half, "phi": gpu_full, "qv": gpu_half,
    }
    if any(not np.all(np.isfinite(weight)) or np.any(weight <= 0.0)
           for weight in (*wrf.values(), *gpuwm.values())):
        raise ValueError("wrfinput produces invalid wrfbdy dry-mass weights")
    return wrf, gpuwm


def _normalize_wrfbdy_endpoint(
        restored: RestoredDomain, gpu_name: str, side_name: str,
        coupled: np.ndarray, width: int,
        wrf_weights: Mapping[str, np.ndarray],
        gpuwm_weights: Mapping[str, np.ndarray]) -> np.ndarray:
    """Decouple one WRF endpoint and recouple it in gpuwm producer order."""
    if gpu_name == "mu":
        return np.asarray(coupled, dtype=np.float32)
    wrf_weight = _boundary_strip(
        _boundary_mass_weight(wrf_weights, gpu_name), side_name, width)
    gpuwm_weight = _boundary_strip(
        _boundary_mass_weight(gpuwm_weights, gpu_name), side_name, width)
    if np.array_equal(wrf_weight, gpuwm_weight):
        # Avoid a lossy divide/multiply round trip when the expression trees
        # already agree bitwise (normally U/V).
        return np.asarray(coupled, dtype=np.float32)
    primitive = np.asarray(coupled / wrf_weight, dtype=np.float32)
    if gpu_name in ("u", "v"):
        map_name = "MAPFAC_U" if gpu_name == "u" else "MAPFAC_V"
        map_factor = _boundary_strip(
            np.asarray(restored.raw[map_name], dtype=np.float32)[None],
            side_name, width)[0]
        primitive = np.asarray(primitive * map_factor[None], dtype=np.float32)
        return np.asarray(
            np.asarray(primitive * gpuwm_weight, dtype=np.float32)
            / map_factor[None], dtype=np.float32)
    return np.asarray(primitive * gpuwm_weight, dtype=np.float32)


# module_model_constants.F declares R_v and R_d as default REAL. The
# real.exe initial/boundary writer divides those FP32 constants; rounding
# the FP64 ratio instead changes one ULP and breaks its THM pairing proof.
_WRF_RVOVRD = np.float32(461.6)/np.float32(287.0)


def _moist_theta_time_law(restored, side_name, tables, width,
                          wrf_weights, gpuwm_weights):
    """Convert source THM/QV/MU interpolation into a dry-theta time law.

    A and Q are WRF's coupled perturbation THM and QV. M is WRF's mass
    weight, G is the shared dycore's weight. In exact arithmetic the target
    is G*(A-300*a*Q)/(M+a*Q), with a=Rv/Rd. All five source quantities are
    linear in time, making this a quadratic/linear rational function.
    Its derivative is supplied to spec_bdytend, not an endpoint secant.
    """
    from gpuwm.ingest.lateral_bc import RationalTimeLaw, SideBoundary
    a = float(_WRF_RVOVRD)
    A, Ad = (np.asarray(x, np.float64) for x in tables["theta"][side_name])
    Q, Qd = (np.asarray(x, np.float64) for x in tables["qv"][side_name])
    _, mud = tables["mu"][side_name]
    M = np.asarray(_boundary_strip(wrf_weights["theta"], side_name, width), np.float64)
    G = np.asarray(_boundary_strip(gpuwm_weights["theta"], side_name, width), np.float64)
    Md = np.asarray(restored.raw["C1H"], np.float64)[:, None, None] * np.asarray(mud, np.float64)
    Gd = Md
    H, Hd = A-300.0*a*Q, Ad-300.0*a*Qd
    denominator = M+a*Q
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("wrfbdy moist-theta conversion requires positive finite mass and moisture denominator")
    rate = (Md+a*Qd)/denominator
    value = G*H/denominator
    tendency = (Gd*H+G*Hd)/denominator-value*rate
    return SideBoundary(value, tendency, RationalTimeLaw(Gd*Hd/denominator, rate))


def _boundary_mass_weight(weights, name):
    """Only declared mass-grid scalar fields share water-vapour coupling."""
    from gpuwm.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    return weights["qv" if name in COUPLED_SCALAR_STATE_FIELDS else name]


def _check_initial_boundary_pair(restored, tables, width, *, layouts=None):
    """Compare the first boundary values with the file that initializes them.

    Reconstruct WRF's own FP32 couple operation, before normalization into
    ArWen's coupling order. Never replace MU and then use the replacement as
    proof that an unrelated boundary file belongs to this initial condition.
    """
    layouts = _WRFBDY_FIELDS if layouts is None else layouts
    raw = restored.raw
    wrf, _ = _wrf_and_gpuwm_mass_weights(restored, np.asarray(raw["MU"], dtype=np.float32))
    for key, (name, *_dimensions) in layouts.items():
        field = np.asarray(raw[name], dtype=np.float32)
        if key == "theta" and int(restored.global_attributes["USE_THETA_M"]) == 1:
            # real.exe writes initial T dry, but couples its runtime moist
            # theta into T_B*. Reconstruct the actual writer's representation.
            factor = np.float32(1.0) + _WRF_RVOVRD * np.asarray(raw["QVAPOR"], np.float32)
            field = np.asarray(factor * (field + np.float32(300.0))
                               - np.float32(300.0), dtype=np.float32)
        if key == "mu":
            field = field[None]
        else:
            field = np.asarray(
                field * _boundary_mass_weight(wrf, key), dtype=np.float32)
            if key in ("u", "v"):
                field = np.asarray(field / np.asarray(raw["MAPFAC_" + name], dtype=np.float32)[None], dtype=np.float32)
        for side, (actual, _tendency) in tables[key].items():
            expected = _boundary_strip(field, side, width)
            actual = np.asarray(actual, dtype=np.float32)
            # Permit independent WRF builds' final FP32 rounding, without
            # treating a changed boundary field as a new initial state.
            tolerance = 4.0 * np.abs(np.spacing(expected)) + 1e-6
            if actual.shape != expected.shape or np.any(np.abs(actual - expected) > tolerance):
                raise ValueError(f"wrfbdy {name} {side} does not match initial {restored.path.name}; use the pair produced by the same real.exe run")


def _check_boundary_identity(dataset, restored):
    for name in ("GRID_ID", "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2",
                 "STAND_LON", "CEN_LAT", "CEN_LON", "HYBRID_OPT", "ETAC", "USE_THETA_M"):
        if name not in dataset.ncattrs() or name not in restored.global_attributes:
            raise ValueError(f"wrfbdy/wrfinput pair lacks {name} identity")
        found = dataset.getncattr(name)
        expected = restored.global_attributes[name]
        if not np.isfinite([found, expected]).all() or float(found) != float(expected):
            raise ValueError(f"wrfbdy {name}={found} differs from wrfinput {expected}")
    if int(dataset.getncattr("USE_THETA_M")) not in (0, 1):
        raise ValueError("wrfbdy USE_THETA_M must identify dry (0) or moist (1) theta")


def read_wrfbdy(path: str | Path, *, run_seconds: float,
                restored: RestoredDomain,
                forcing_interval_seconds: float,
                spec_bdy_width: int = 5,
                first_interval_only: bool = False, cfg=None):
    """Build coupled gpuwm d01 boundary intervals from real.exe tables.

    ``cfg`` carries the actual selected scalar boundary operation. The
    public WRF door always supplies it; its absence retains the original
    dry/vapour inventory for compatibility with verification callers.

    ``spec_bdy_width`` is the caller's ``&bdy_control spec_bdy_width``; the
    file's own ``bdy_width`` dimension must equal it.  It was pinned at the
    registered N5S value of 5 while this was a verification importer, which
    would have refused any user who set it otherwise.

    ``first_interval_only`` keeps a run inside the FIRST forcing interval
    -- the registered N5S contract, which is scored only there.  A front
    door leaves it False and consumes every validated record, which is what
    lets ``run_seconds`` reach the full ``wrfbdy`` coverage.  It used to be
    inferred from the forcing interval being six hours, which meant a user
    with ordinary six-hourly GFS forcing silently inherited the
    verification case's ceiling.
    """
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
    )

    if not isinstance(restored, RestoredDomain):
        raise TypeError("read_wrfbdy requires the restored d01 wrfinput")
    if (isinstance(forcing_interval_seconds, (bool, np.bool_))
            or not isinstance(
                forcing_interval_seconds,
                (int, float, np.integer, np.floating))
            or not np.isfinite(float(forcing_interval_seconds))
            or float(forcing_interval_seconds) <= 0.0
            or not float(forcing_interval_seconds).is_integer()):
        raise ValueError(
            "forcing_interval_seconds must be a positive whole number")
    forcing_interval = int(forcing_interval_seconds)
    from gpuwm.boundary_fields import external_scalar_fields
    layouts = dict(_WRFBDY_FIELDS)
    if cfg is not None:
        by_state = {state: wrf for wrf, state in _active_moisture_map(cfg).items()}
        for name in external_scalar_fields(cfg):
            layouts[name] = (by_state[name], "bottom_top", "south_north", "west_east")
    field_names = {name: layout[0] for name, layout in layouts.items()}
    with netcdf_bridge.open_dataset(path) as dataset:
        _check_boundary_identity(dataset, restored)
        if "Times" not in dataset.variables:
            raise ValueError("wrfbdy_d01 has no Times variable")
        if "bdy_width" not in dataset.dimensions:
            raise ValueError("wrfbdy_d01 has no bdy_width dimension")
        width = len(dataset.dimensions["bdy_width"])
        if width != int(spec_bdy_width):
            raise ValueError(
                f"{path} bdy_width {width} != the requested "
                f"spec_bdy_width {int(spec_bdy_width)}; real.exe writes "
                "the boundary tables at the width &bdy_control asked for, "
                "so this pair came from two different namelists")
        times = _decode_times(dataset.variables["Times"])
        if not times:
            raise ValueError("wrfbdy_d01 has no records")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("wrfbdy_d01 records must increase")
        next_name = ("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        if len(times) == 1:
            # real.exe's standard single-interval product: one record of
            # boundary values at thisbdytime plus _BT tendencies valid to
            # nextbdytime.  The interval loop below already extrapolates the
            # last record by the registered cadence; here we only verify the
            # file's own metadata pins that same interval.
            if next_name not in dataset.variables:
                raise ValueError(
                    "single-record wrfbdy_d01 needs nextbdytime metadata")
            next_times = _decode_times(dataset.variables[next_name])
            expected_next = times[0] + timedelta(
                seconds=forcing_interval)
            if len(next_times) != 1 or next_times[0] != expected_next:
                raise ValueError(
                    "single-record wrfbdy_d01 nextbdytime does not pin the "
                    f"requested {forcing_interval}-second forcing interval")
        elif next_name in dataset.variables:
            # WRF products encountered in the wild use either one metadata
            # record for the final endpoint or one next-time record per Times
            # record.  Accept only those two explicit encodings.
            next_times = _decode_times(dataset.variables[next_name])
            expected_all = tuple(
                value + timedelta(seconds=forcing_interval)
                for value in times)
            expected_final = (expected_all[-1],)
            if next_times not in (expected_all, expected_final):
                raise ValueError(
                    "multi-record wrfbdy_d01 nextbdytime does not match "
                    f"Times plus {forcing_interval} seconds")
        required_records = []
        for gpu_name, wrf_name in field_names.items():
            _, zdim, ydim, xdim = layouts[gpu_name]
            for side_name, suffix in (
                    ("west", "XS"), ("east", "XE"),
                    ("south", "YS"), ("north", "YE")):
                value_name = f"{wrf_name}_B{suffix}"
                tendency_name = f"{wrf_name}_BT{suffix}"
                for record_name in (value_name, tendency_name):
                    if record_name not in dataset.variables:
                        raise ValueError(f"wrfbdy_d01 is missing {record_name}")
                    records = dataset.variables[record_name].shape[0]
                    if records != len(times):
                        raise ValueError(
                            f"wrfbdy_d01 {record_name} record count mismatch: "
                            f"expected {len(times)}, got {records}")
                if (dataset.variables[value_name].shape
                        != dataset.variables[tendency_name].shape):
                    raise ValueError(
                        f"wrfbdy_d01 {value_name}/{tendency_name} shape "
                        "mismatch")
                side_dim = ydim if side_name in ("west", "east") else xdim
                expected_dimensions = (["Time", "bdy_width"]
                                       + ([] if zdim is None else [zdim])
                                       + [side_dim])
                for record_name in (value_name, tendency_name):
                    actual_dimensions = list(dataset.variables[record_name].dimensions)
                    if actual_dimensions != expected_dimensions:
                        raise ValueError(
                            f"wrfbdy_d01 {record_name} dimensions "
                            f"{tuple(actual_dimensions)} != expected real.exe "
                            f"{tuple(expected_dimensions)}")
                required_records.append(
                    (gpu_name, wrf_name, side_name, value_name, tendency_name))
        origin = times[0]
        if origin.strftime("%Y-%m-%d_%H:%M:%S") != str(restored.global_attributes.get("START_DATE")):
            raise ValueError("wrfbdy first Times record differs from wrfinput START_DATE")
        starts = [(time - origin).total_seconds() for time in times]
        forcing_intervals = [
            later - earlier for earlier, later in zip(starts, starts[1:])]
        if any(interval != forcing_interval
               for interval in forcing_intervals):
            observed = sorted({int(value) for value in forcing_intervals})
            raise ValueError(
                f"{path} must use the requested {forcing_interval}-second "
                f"forcing cadence; its Times records step by {observed} "
                "seconds")
        first_interval_end = (starts[1] if len(starts) > 1
                              else starts[0] + forcing_interval)
        coverage_end = (first_interval_end if first_interval_only
                        else starts[-1] + forcing_interval)
        if not 0.0 < float(run_seconds) <= coverage_end:
            # BOTH numbers, before any integration.  A run that outlives
            # its lateral forcing does not fail -- it keeps extrapolating
            # the last tendency and produces a finite, bounded, wrong
            # forecast, which is the exact failure mode this door exists
            # to make impossible.
            scope = ("the FIRST boundary interval" if first_interval_only
                     else f"{len(times)} boundary record(s)")
            raise ValueError(
                f"requested run of {float(run_seconds):.0f} s exceeds the "
                f"{coverage_end:.0f} s of lateral forcing in {path} "
                f"({scope} at a {forcing_interval}-second cadence, the "
                "last one extrapolated by its own _BT tendency for one "
                "further interval).  Shorten the run to at most "
                f"{coverage_end:.0f} s, or re-run real.exe over a longer "
                "window so wrfbdy_d01 carries more records.")
        intervals = []
        for index, start in enumerate(starts):
            if index + 1 < len(starts):
                end = starts[index + 1]
            else:
                end = start + forcing_interval
            raw_tables = {gpu_name: {} for gpu_name in field_names}
            for (gpu_name, wrf_name, side_name, value_name,
                 tendency_name) in required_records:
                value = _wrfbdy_side_table(
                    dataset.variables[value_name], index, side_name)
                tendency = _wrfbdy_side_table(
                    dataset.variables[tendency_name], index, side_name)
                raw_tables[gpu_name][side_name] = (value, tendency)
            if index == 0:
                _check_initial_boundary_pair(restored, raw_tables, width, layouts=layouts)
            duration = float(end - start)
            mu_start_sides = {
                side_name: pair[0]
                for side_name, pair in raw_tables["mu"].items()}
            mu_end_sides = {
                side_name: np.asarray(
                    pair[0] + np.float32(duration) * pair[1],
                    dtype=np.float32)
                for side_name, pair in raw_tables["mu"].items()}
            mu_start = _mass_from_boundary_sides(
                restored.raw["MU"], mu_start_sides)
            mu_end = _mass_from_boundary_sides(
                restored.raw["MU"], mu_end_sides)
            wrf_start, gpuwm_start = _wrf_and_gpuwm_mass_weights(
                restored, mu_start)
            wrf_end, gpuwm_end = _wrf_and_gpuwm_mass_weights(
                restored, mu_end)
            side_tables = {gpu_name: {} for gpu_name in field_names}
            for gpu_name, sides in raw_tables.items():
                for side_name, (value, tendency) in sides.items():
                    if (gpu_name == "theta" and
                            int(restored.global_attributes["USE_THETA_M"]) == 1):
                        side_tables[gpu_name][side_name] = _moist_theta_time_law(
                            restored, side_name, raw_tables, width,
                            wrf_start, gpuwm_start)
                        continue
                    future = np.asarray(
                        value + np.float32(duration) * tendency,
                        dtype=np.float32)
                    normalized = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, value, width,
                        wrf_start, gpuwm_start)
                    normalized_future = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, future, width,
                        wrf_end, gpuwm_end)
                    target_tendency = (
                        normalized_future.astype(np.float64)
                        - normalized.astype(np.float64)) / duration
                    side_tables[gpu_name][side_name] = SideBoundary(
                        normalized.astype(np.float64), target_tendency)
            fields = {
                gpu_name: FieldBoundary(**side_tables[gpu_name])
                for gpu_name in field_names
            }
            intervals.append(BoundaryInterval(start, end, fields))
    return LateralBoundaries(tuple(intervals), width, 1, width - 1)


# ==========================================================================
# The front door's own instruments: cheap, CPU-only, and refusing FIRST.
# ==========================================================================


@dataclass(frozen=True)
class WrfinputMetadata:
    """Everything a door needs from a wrfinput without decoding a field.

    Read from the file's dimensions and global attributes only, so it is
    cheap enough to run on every domain before anything is restored.  The
    checks that use it -- scheme support, geometry agreement with
    ``namelist.input`` -- must all fire before the first byte of U is
    decoded, or a mismatch costs the user a full read of an 87 MB file to
    be told the grid is the wrong shape.
    """

    path: Path
    grid_id: int
    dimensions: Mapping[str, int]
    mp_physics: int
    sf_surface_physics: int
    start_date: str
    global_attributes: Mapping[str, object]

    @property
    def nx(self) -> int:
        return self.dimensions["west_east"]

    @property
    def ny(self) -> int:
        return self.dimensions["south_north"]

    @property
    def nz(self) -> int:
        return self.dimensions["bottom_top"]

    @property
    def soil_layers(self) -> int:
        return self.dimensions["soil_layers_stag"]

    def pinned_dimensions(self) -> Mapping[str, int]:
        """The seven-extent geometry contract, as the file declares it.

        Handed to :func:`read_wrfinput` as ``expected_dimensions`` ONLY
        after :func:`check_grid_agreement` has confirmed the file's own
        claim against the namelist.  Skipping that step and pinning the
        geometry to the file would make the pin vacuous -- the file would
        be checked against itself.
        """
        return MappingProxyType({
            name: int(self.dimensions[name])
            for name in sorted(_WRFINPUT_GEOMETRY_DIMENSIONS)})


def read_wrfinput_metadata(path: str | Path) -> WrfinputMetadata:
    """Read one wrfinput's geometry and physics identity, no fields."""
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim)
                      for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name)
                 for name in dataset.ncattrs()}
    missing = sorted(_WRFINPUT_GEOMETRY_DIMENSIONS - set(dimensions))
    if missing:
        raise ValueError(
            f"{path} is missing WRF grid dimension(s) {missing}; a "
            "real.exe wrfinput declares all seven "
            f"({sorted(_WRFINPUT_GEOMETRY_DIMENSIONS)})")
    resolved = check_supported_schemes(attrs, source=str(path))
    grid_id = _integral_attribute(attrs.get("GRID_ID"))
    if grid_id is None:
        raise ValueError(
            f"{path} has no integer GRID_ID global attribute, so which "
            "domain of the nest hierarchy it initializes cannot be "
            "established")
    return WrfinputMetadata(
        path=path, grid_id=grid_id,
        dimensions=MappingProxyType(
            {name: int(value) for name, value in dimensions.items()}),
        mp_physics=resolved["MP_PHYSICS"],
        sf_surface_physics=resolved["SF_SURFACE_PHYSICS"],
        start_date=str(attrs.get("START_DATE", "")),
        global_attributes=MappingProxyType(attrs))


def check_grid_agreement(metadata: WrfinputMetadata, cfg, *,
                         source: str) -> None:
    """Refuse a wrfinput/namelist pair that disagrees about the grid.

    ``real.exe`` and ArWen are being handed the SAME ``namelist.input``,
    so nx/ny/nz and the soil geometry must match.  When they do not, one
    of the two files is from a different experiment, and every field that
    follows would either raise deep inside a shape check or -- worse, for
    the 1-D vertical coefficient arrays -- silently broadcast.  Both
    numbers are named for every axis that disagrees, in ONE refusal, so a
    reader is not walked through four axes one re-run at a time.
    """
    from gpuwm.config import soil_layer_count

    rows = [
        ("west_east (nx)", metadata.nx, int(cfg.nx)),
        ("south_north (ny)", metadata.ny, int(cfg.ny)),
        ("bottom_top (nz)", metadata.nz, int(cfg.nz)),
        ("soil_layers_stag", metadata.soil_layers, soil_layer_count(cfg)),
    ]
    bad = [(axis, found, wanted) for axis, found, wanted in rows
           if found != wanted]
    if not bad:
        return
    lines = "\n".join(
        f"  {axis}: {metadata.path.name} has {found}, "
        f"{source} resolves to {wanted}"
        for axis, found, wanted in bad)
    raise ValueError(
        f"{metadata.path} and {source} disagree about the grid:\n{lines}\n"
        "Entering at wrfinput means real.exe already baked the domain "
        "layout and the vertical grid, so the namelist cannot change "
        "them: use the namelist.input that produced these files, or "
        "re-run real.exe with this namelist.")


@dataclass(frozen=True)
class BoundaryCoverage:
    """How far in time one ``wrfbdy`` file can force a run."""

    path: Path
    times: tuple[datetime, ...]
    forcing_interval_seconds: int
    spec_bdy_width: int
    coverage_seconds: float

    @property
    def start(self) -> datetime:
        return self.times[0]

    @property
    def end(self) -> datetime:
        return self.times[0] + timedelta(seconds=self.coverage_seconds)


def wrfbdy_coverage(path: str | Path) -> BoundaryCoverage:
    """Read a ``wrfbdy``'s time coverage without decoding a boundary table.

    This is the instrument behind "run length is capped by wrfbdy".  It
    opens the file for its ``Times`` records, its ``bdy_width`` and its
    ``nextbdytime`` metadata only -- no side table is touched -- so a run
    whose length outstrips its forcing is refused at the front door, in
    milliseconds, rather than after every boundary array has been
    decoded and uploaded.
    """
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        if "Times" not in dataset.variables:
            raise ValueError(f"{path} has no Times variable")
        if "bdy_width" not in dataset.dimensions:
            raise ValueError(f"{path} has no bdy_width dimension")
        width = len(dataset.dimensions["bdy_width"])
        times = _decode_times(dataset.variables["Times"])
        if not times:
            raise ValueError(f"{path} has no records")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"{path} records must increase")
        next_name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
        next_times = (_decode_times(dataset.variables[next_name])
                      if next_name in dataset.variables else ())
    if len(times) > 1:
        steps = {int((later - earlier).total_seconds())
                 for earlier, later in zip(times, times[1:])}
        if len(steps) != 1:
            raise ValueError(
                f"{path} Times records do not step by one uniform forcing "
                f"interval: {sorted(steps)} seconds")
        interval = steps.pop()
    elif next_times:
        # real.exe's standard single-interval product declares its own
        # cadence in the nextbdytime metadata, and that is the only place
        # a one-record file states it.
        interval = int((next_times[-1] - times[0]).total_seconds())
    else:
        raise ValueError(
            f"{path} carries a single Times record and no nextbdytime "
            "metadata, so the forcing interval it covers cannot be "
            "established; a real.exe wrfbdy always writes one")
    if interval <= 0:
        raise ValueError(
            f"{path} declares a non-positive forcing interval of "
            f"{interval} seconds")
    return BoundaryCoverage(
        path=path, times=times, forcing_interval_seconds=interval,
        spec_bdy_width=width,
        coverage_seconds=float(
            (times[-1] - times[0]).total_seconds() + interval))


def check_boundary_coverage(coverage: BoundaryCoverage,
                            run_seconds: float, *, source: str) -> None:
    """Refuse, BEFORE any integration, a run longer than its forcing."""
    if 0.0 < float(run_seconds) <= coverage.coverage_seconds:
        return
    raise ValueError(
        f"{source} requests a run of {float(run_seconds):.0f} s but "
        f"{coverage.path} carries only {coverage.coverage_seconds:.0f} s "
        f"of lateral forcing ({len(coverage.times)} record(s) from "
        f"{coverage.start:%Y-%m-%d_%H:%M:%S} at a "
        f"{coverage.forcing_interval_seconds}-second cadence, valid "
        f"through {coverage.end:%Y-%m-%d_%H:%M:%S}).  Entering at "
        "wrfinput caps run length at wrfbdy coverage: shorten the run to "
        f"at most {coverage.coverage_seconds:.0f} s, or re-run real.exe "
        "over a longer window.")


#: What nest relocation needs that a ``wrfinput`` handoff does not carry.
#:
#: VERIFIED 2026-09-04 against gpuwm/ingest/relocation_init.py,
#: gpuwm/ingest/nest_init.py and gpuwm/core/nest_relocation.py: none of
#: the three so much as mentions ``wrfinput`` or ``wrfbdy``.  A relocated
#: child's statics are rebuilt for the new footprint from the nest's own
#: static source at nest resolution and its atmosphere is filled from the
#: LIVE PARENT, so nothing on that path reopens an initial-condition file
#: after t=0 and the capability survives this entry point intact.
#:
#: It is not free, though, and the one condition is structural:
#: :func:`gpuwm.ingest.relocation_init.real_relocation_initializer`
#: requires EXACTLY ONE statics source -- a geography ``catalog`` or a
#: sealed-corridor ``statics_builder`` -- and refuses with neither.  A
#: wrfinput handoff supplies neither, because real.exe's statics are baked
#: into the wrfinput at the ORIGINAL footprint and say nothing about
#: ground the nest has not stood on yet.  So a wrfinput-entry run that
#: wants a moving nest must additionally declare a geography root.
RELOCATION_REQUIREMENT = (
    "nest relocation needs additional static coverage: the relocation path "
    "rebuilds a moved nest's statics from the geography source at nest "
    "resolution and takes its atmosphere from the live parent, and never "
    "reopens wrfinput/wrfbdy after t=0.  It does need a statics source of "
    "its own -- real_relocation_initializer takes exactly one of a "
    "geography catalog or a sealed statics_builder and refuses with "
    "neither -- and a wrfinput carries statics only for the footprint "
    "real.exe wrote, so a moving-nest run entering here must also declare "
    "a geography root. This adapter does not yet bind that future coverage."
)


def format_scheme_matrix() -> str:
    """The accepted/refused scheme matrix, for the door's own --help."""
    from gpuwm.config import LAND_SURFACE_SCHEMES

    lines = ["  microphysics (MP_PHYSICS) restored:"]
    for value in sorted(SUPPORTED_MICROPHYSICS):
        lines.append(f"    {value:>3d}  {SUPPORTED_MICROPHYSICS[value]}")
    lines.append("  microphysics refused by name (with a remedy):")
    lines.append("    " + ", ".join(
        str(value) for value in sorted(UNSUPPORTED_MICROPHYSICS)))
    lines.append("  land surface (SF_SURFACE_PHYSICS) restored:")
    for value in sorted(LAND_SURFACE_SCHEMES):
        name, soil = WRF_LAND_SURFACE_SCHEMES[value]
        lines.append(f"    {value:>3d}  {name} ({soil} soil layers)")
    lines.append("  land surface refused by name (with a remedy):")
    lines.append("    " + ", ".join(
        f"{value} ({WRF_LAND_SURFACE_SCHEMES[value][0]})"
        for value in sorted(set(WRF_LAND_SURFACE_SCHEMES)
                            - set(LAND_SURFACE_SCHEMES))))
    return "\n".join(lines)


__all__ = [
    "BoundaryCoverage", "RELOCATION_REQUIREMENT", "RestoredDomain",
    "SUPPORTED_MICROPHYSICS", "UNSUPPORTED_MICROPHYSICS",
    "WRF_LAND_SURFACE_SCHEMES", "WrfinputMetadata",
    "check_boundary_coverage", "check_grid_agreement",
    "check_supported_schemes", "format_scheme_matrix", "read_wrfbdy",
    "read_wrfinput", "read_wrfinput_metadata", "restore_domain_state",
    "supported_land_surface_sentence", "supported_microphysics_sentence",
    "unsupported_land_surface_refusal", "unsupported_microphysics_refusal",
    "wrfbdy_coverage",
]
