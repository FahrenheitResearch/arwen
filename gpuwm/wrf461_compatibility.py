"""Declarative WRF v4.6.1 physics-combination authority.

The tables in this module are a transcription of WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  They deliberately separate
WRF's verdict from ArWen implementation readiness: a WRF-legal tuple may
still have a named ArWen structural blocker, but ArWen must never describe
that blocker as a WRF incompatibility.

Only the schemes ported by this release are represented.  Keeping the
cross-product dimensions and citations here makes admission a table lookup,
not a collection of subtly different front-door conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Iterator, Mapping


WRF_VERSION = "v4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"


class WRFVerdict(str, Enum):
    """WRF's disposition for one represented physics tuple."""

    LEGAL = "legal"
    FATAL = "fatal-in-physics-init"
    LEGAL_RECONFIGURED = "legal-with-silent-reconfiguration"
    NOT_EXPRESSIBLE = "not-expressible-in-wrf-v4.6.1"


@dataclass(frozen=True)
class WRFCitation:
    """One source anchor carried by an authority-table cell."""

    path: str
    lines: str
    law: str

    @property
    def anchor(self) -> str:
        return f"{self.path}:{self.lines}"


@dataclass(frozen=True)
class WRFCompatibilityCell:
    """One fully cited cell of the represented cross-product."""

    mp_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    sf_surface_physics: int
    radiation: str
    cu_physics: int
    verdict: WRFVerdict
    citations: tuple[WRFCitation, ...]
    silent_reconfiguration: str | None = None


MP_OPTIONS = (1, 6, 8, 10, 18, 28)
PBL_OPTIONS = (0, 1, 5)
SURFACE_LAYER_OPTIONS = (0, 1, 5, 91)
LAND_SURFACE_OPTIONS = (0, 2, 3, 4)
RADIATION_OPTIONS = (
    "off",
    "dudhia-shortwave",
    "rrtmg-rte-rrtmgp",
    "rrtmg-legacy",
    "analytic",
)
CUMULUS_OPTIONS = (0, 1)

MATRIX_CELL_COUNT = (
    len(MP_OPTIONS)
    * len(PBL_OPTIONS)
    * len(SURFACE_LAYER_OPTIONS)
    * len(LAND_SURFACE_OPTIONS)
    * len(RADIATION_OPTIONS)
    * len(CUMULUS_OPTIONS)
)


_MP_CITATION = MappingProxyType({
    1: WRFCitation(
        "Registry/Registry.EM_COMMON", "3015",
        "the Kessler package binds mp_physics=1 and allocates qv/qc/qr"),
    6: WRFCitation(
        "Registry/Registry.EM_COMMON", "3021",
        "the WSM6 package binds mp_physics=6"),
    8: WRFCitation(
        "Registry/Registry.EM_COMMON", "3024",
        "the Thompson package binds mp_physics=8"),
    10: WRFCitation(
        "Registry/Registry.EM_COMMON", "3026",
        "the Morrison two-moment package binds mp_physics=10"),
    18: WRFCitation(
        "Registry/Registry.EM_COMMON", "3033",
        "the NSSL two-moment package binds mp_physics=18"),
    28: WRFCitation(
        "Registry/Registry.EM_COMMON", "3036",
        "the aerosol-aware Thompson package binds mp_physics=28"),
})

_LAND_SURFACE_CITATION = MappingProxyType({
    0: WRFCitation(
        "Registry/Registry.EM_COMMON", "3144",
        "the no-LSM package binds sf_surface_physics=0"),
    2: WRFCitation(
        "Registry/Registry.EM_COMMON", "3146",
        "the Noah LSM package binds sf_surface_physics=2"),
    3: WRFCitation(
        "Registry/Registry.EM_COMMON", "3147",
        "the RUC LSM package binds sf_surface_physics=3"),
    4: WRFCitation(
        "Registry/Registry.EM_COMMON", "3149",
        "the Noah-MP package binds sf_surface_physics=4"),
})

_CUMULUS_CITATION = MappingProxyType({
    0: WRFCitation(
        "Registry/Registry.EM_COMMON", "3189",
        "the no-cumulus package binds cu_physics=0"),
    1: WRFCitation(
        "Registry/Registry.EM_COMMON", "3190",
        "the Kain-Fritsch package binds cu_physics=1"),
})

_RADIATION_CITATION = MappingProxyType({
    "off": WRFCitation(
        "phys/module_physics_init.F", "2230-2345,2348-2445",
        "the LW and SW initialization SELECT CASE blocks accept zero through "
        "their no-action default paths"),
    "dudhia-shortwave": WRFCitation(
        "Registry/Registry.EM_COMMON", "3117",
        "the Dudhia shortwave package binds ra_sw_physics=1; longwave zero "
        "uses the no-action initialization path"),
    "rrtmg-rte-rrtmgp": WRFCitation(
        "Registry/Registry.EM_COMMON", "3109,3120",
        "WRF's bundled RRTMG packages bind the 4/4 LW/SW selectors; ArWen's "
        "RTE+RRTMGP adapter is an explicitly receipted implementation of "
        "those same WRF selectors"),
    "rrtmg-legacy": WRFCitation(
        "Registry/Registry.EM_COMMON", "3109,3120",
        "WRF's bundled RRTMG packages bind the 4/4 LW/SW selectors; ArWen's "
        "legacy variant ports that implementation directly"),
    "analytic": WRFCitation(
        "Registry/Registry.EM_COMMON", "3107-3125",
        "WRF v4.6.1 registers no ra_lw_physics=90 or ra_sw_physics=90 "
        "package; analytic radiation is ArWen-specific"),
})

_SOIL_LAYER_CITATION = WRFCitation(
    "share/module_check_a_mundo.F", "3548-3563",
    "set_physics_rconfigs silently sets the WRF soil-layer count: no-LSM=5, "
    "Noah=4, Noah-MP=4, and an invalid RUC count=6 (RUC 6 or 9 is retained)")


def _pair(
    verdict: WRFVerdict, path: str, lines: str, law: str
) -> tuple[WRFVerdict, WRFCitation]:
    return verdict, WRFCitation(path, lines, law)


# This is the complete PBL/surface-layer compatibility law intersecting the
# ported selectors.  PBL-off executes no PBL initializer and therefore accepts
# every represented surface layer.  YSU requires isfc=1, which revised/classic
# MM5 set.  MYNN PBL accepts isfc in {5,1,2}; the represented values 5,1,91
# initialize isfc to 5,1,1 respectively.
PBL_SURFACE_LAYER_AUTHORITY: Mapping[
    tuple[int, int], tuple[WRFVerdict, WRFCitation]
] = MappingProxyType({
    (0, 0): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3697-3915",
        "bl_pbl_physics=0 selects no PBL initialization branch"),
    (0, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3697-3915",
        "revised MM5 sets isfc=1; PBL-off selects no PBL initializer"),
    (0, 5): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3213-3219,3697-3915",
        "MYNN surface sets isfc=5; PBL-off selects no PBL initializer"),
    (0, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3697-3915",
        "classic MM5 sets isfc=1; PBL-off selects no PBL initializer"),
    (1, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3699-3701",
        "YSU fatals unless the initialized surface-layer class is isfc=1"),
    (1, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3699-3701",
        "revised MM5 sets isfc=1, satisfying YSU"),
    (1, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3699-3701",
        "MYNN surface sets isfc=5, and YSU fatals unless isfc=1"),
    (1, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3699-3701",
        "classic MM5 sets isfc=1, satisfying YSU"),
    (5, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3837-3839",
        "MYNN PBL fatals unless the initialized surface class is 5, 1, or 2"),
    (5, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3837-3839",
        "revised MM5 sets isfc=1, which MYNN PBL explicitly accepts"),
    (5, 5): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3213-3219,3837-3839",
        "MYNN surface sets isfc=5, which MYNN PBL explicitly accepts"),
    (5, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3837-3839",
        "classic MM5 sets isfc=1, which MYNN PBL explicitly accepts"),
})


def pbl_surface_layer_verdict(
    bl_pbl_physics: int, sf_sfclay_physics: int
) -> tuple[WRFVerdict, WRFCitation]:
    """Return the exact WRF table entry for one represented pairing."""

    try:
        return PBL_SURFACE_LAYER_AUTHORITY[
            (int(bl_pbl_physics), int(sf_sfclay_physics))]
    except KeyError as exc:
        raise ValueError(
            "pair is outside the WRF v4.6.1 ported-set authority table: "
            f"bl_pbl_physics={bl_pbl_physics}, "
            f"sf_sfclay_physics={sf_sfclay_physics}") from exc


def compatibility_cell(
    *,
    mp_physics: int,
    bl_pbl_physics: int,
    sf_sfclay_physics: int,
    sf_surface_physics: int,
    radiation: str,
    cu_physics: int,
) -> WRFCompatibilityCell:
    """Build one cited cell without applying an ArWen structural gate."""

    pair_verdict, pair_citation = pbl_surface_layer_verdict(
        bl_pbl_physics, sf_sfclay_physics)
    try:
        citations = (
            _MP_CITATION[int(mp_physics)],
            pair_citation,
            _LAND_SURFACE_CITATION[int(sf_surface_physics)],
            _RADIATION_CITATION[str(radiation)],
            _CUMULUS_CITATION[int(cu_physics)],
            _SOIL_LAYER_CITATION,
        )
    except KeyError as exc:
        raise ValueError(
            f"tuple axis {exc.args[0]!r} is outside the represented "
            "WRF v4.6.1 matrix") from exc

    silent_reconfiguration = None
    if pair_verdict is WRFVerdict.FATAL:
        verdict = WRFVerdict.FATAL
    elif radiation == "analytic":
        verdict = WRFVerdict.NOT_EXPRESSIBLE
    elif int(sf_surface_physics) == 0:
        # ArWen's no-LSM state deliberately carries an empty four-level soil
        # dimension for byte identity.  WRF overwrites the namelist value to
        # five even though its no-LSM package allocates no soil prognostics.
        verdict = WRFVerdict.LEGAL_RECONFIGURED
        silent_reconfiguration = "WRF sets num_soil_layers=5 for no-LSM"
    else:
        verdict = WRFVerdict.LEGAL

    return WRFCompatibilityCell(
        mp_physics=int(mp_physics),
        bl_pbl_physics=int(bl_pbl_physics),
        sf_sfclay_physics=int(sf_sfclay_physics),
        sf_surface_physics=int(sf_surface_physics),
        radiation=str(radiation),
        cu_physics=int(cu_physics),
        verdict=verdict,
        citations=citations,
        silent_reconfiguration=silent_reconfiguration,
    )


def iter_compatibility_matrix() -> Iterator[WRFCompatibilityCell]:
    """Yield the complete represented cross-product (:data:`MATRIX_CELL_COUNT`
    cells; 2,880 once mp_physics=28 joined the microphysics axis)."""

    for mp, pbl, sfclay, lsm, radiation, cumulus in product(
        MP_OPTIONS,
        PBL_OPTIONS,
        SURFACE_LAYER_OPTIONS,
        LAND_SURFACE_OPTIONS,
        RADIATION_OPTIONS,
        CUMULUS_OPTIONS,
    ):
        yield compatibility_cell(
            mp_physics=mp,
            bl_pbl_physics=pbl,
            sf_sfclay_physics=sfclay,
            sf_surface_physics=lsm,
            radiation=radiation,
            cu_physics=cumulus,
        )


__all__ = [
    "CUMULUS_OPTIONS",
    "LAND_SURFACE_OPTIONS",
    "MATRIX_CELL_COUNT",
    "MP_OPTIONS",
    "PBL_OPTIONS",
    "PBL_SURFACE_LAYER_AUTHORITY",
    "RADIATION_OPTIONS",
    "SURFACE_LAYER_OPTIONS",
    "WRF_COMMIT",
    "WRF_VERSION",
    "WRFCitation",
    "WRFCompatibilityCell",
    "WRFVerdict",
    "compatibility_cell",
    "iter_compatibility_matrix",
    "pbl_surface_layer_verdict",
]
