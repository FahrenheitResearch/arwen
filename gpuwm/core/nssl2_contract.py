"""WRF v4.6.1 NSSL two-moment (``mp_physics=18``) contract.

This module is deliberately CPU-only.  It pins the unified NSSL selector,
resolved default mode, transported state, radiation diagnostics, and numerical
setup before the complete CUDA driver is admitted.  The authority is NCAR WRF
v4.6.1 commit ``d66e442fccc04111067e29274c9f9eaccc3cef28``:

* ``Registry/Registry.EM_COMMON:2410-2425,3033,3049-3056``;
* ``share/module_check_a_mundo.F:3377-3465``;
* ``phys/module_physics_init.F:4615-4652``; and
* ``phys/module_mp_nssl_2mom.F``.

Since WRF v4.6 the former NSSL IDs are compatibility spellings of option 18.
The target here is the native option-18 default: two moments, hail, predicted
CCN, and predicted graupel/hail volume (density), without the optional sixth
moments.  Importing this contract does not make the scheme executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


MP_PHYSICS = 18
WRF_REFERENCE_VERSION = "v4.6.1"
WRF_REFERENCE_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
CONTRACT_ID = "wrf-v4.6.1-nssl-mp18-two-moment-hail-ccn-density-v1"

# Registry packages nssl_2mom + nssl2mconc + nssl_hail + nssl_ccn_opt
# + nssl_hailvol after module_check_a_mundo resolves all -1 selectors.
BASE_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
TWO_MOMENT_NUMBER_FIELDS = ("qndrop", "qnr", "qni", "qns", "qng")
HAIL_MASS_FIELD = "qh"
HAIL_NUMBER_FIELD = "qnh"
PREDICTED_CCN_FIELD = "qnn"
GRAUPEL_VOLUME_FIELD = "qvolg"
HAIL_VOLUME_FIELD = "qvolh"
SIXTH_MOMENT_FIELDS = ("qzr", "qzg", "qzh")

RADIATION_EFFECTIVE_RADIUS_FIELDS = ("re_cloud", "re_ice", "re_snow")
REFLECTIVITY_FIELDS = ("refl_10cm",)
PRECIPITATION_FIELDS = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "hailnc", "hailncv",
)
HAIL_DIAGNOSTIC_FIELDS = ("hail_maxk1", "hail_max2d")

# Registry defaults are the runtime authority.  In particular, v4.6.1's
# Registry says 900 kg m-3 for nssl_rho_qhl although README.NSSLmp still says
# 800; module_physics_init forwards the Registry value into nssl_2mom_init.
WRF_NAMELIST_DEFAULTS: Mapping[str, int | float] = MappingProxyType({
    "nssl_cccn": 0.5e9,
    "nssl_alphah": 0.0,
    "nssl_alphahl": 1.0,
    "nssl_cnoh": 4.0e5,
    "nssl_cnohl": 4.0e4,
    "nssl_cnor": 8.0e5,
    "nssl_cnos": 3.0e6,
    "nssl_rho_qh": 500.0,
    "nssl_rho_qhl": 900.0,
    "nssl_rho_qs": 100.0,
    "nssl_icdx": 6,
    "nssl_icdxhl": 6,
    "nssl_2moment_on": -1,
    "nssl_hail_on": -1,
    "nssl_ccn_on": -1,
    "nssl_density_on": -1,
    "nssl_3moment": 0,
})


@dataclass(frozen=True)
class NSSL2Mode:
    """Selectors after WRF's option-18 consistency pass.

    ``density_moments`` follows the Registry package selector: 0 is fixed
    density, 1 predicts graupel volume, and 2 predicts graupel and hail volume.
    ``sixth_moments`` is 0, 1 (rain/graupel), or 2 (rain/graupel/hail).
    """

    two_moment: bool
    hail: bool
    predicted_ccn: bool
    density_moments: int
    sixth_moments: int

    @property
    def transported_fields(self) -> tuple[str, ...]:
        fields = list(BASE_MASS_FIELDS)
        if self.hail:
            fields.append(HAIL_MASS_FIELD)
        if self.two_moment:
            fields.extend(TWO_MOMENT_NUMBER_FIELDS)
            if self.hail:
                fields.append(HAIL_NUMBER_FIELD)
        if self.predicted_ccn:
            fields.append(PREDICTED_CCN_FIELD)
        if self.density_moments >= 1:
            fields.append(GRAUPEL_VOLUME_FIELD)
        if self.density_moments >= 2:
            fields.append(HAIL_VOLUME_FIELD)
        if self.sixth_moments >= 1:
            fields.extend(SIXTH_MOMENT_FIELDS[:2])
        if self.sixth_moments >= 2:
            fields.append(SIXTH_MOMENT_FIELDS[2])
        return tuple(fields)

    @property
    def radiation_fields(self) -> tuple[str, ...]:
        return RADIATION_EFFECTIVE_RADIUS_FIELDS if self.two_moment else ()


def _selector(name: str, value: int, allowed: tuple[int, ...]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer selector, got {value!r}")
    if value not in allowed:
        options = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of {options}, got {value}")
    return value


def resolve_nssl2_mode(
        *, mp_physics: int = MP_PHYSICS, nssl_2moment_on: int = -1,
        nssl_hail_on: int = -1, nssl_ccn_on: int = -1,
        nssl_density_on: int = -1, nssl_3moment: int = 0,
        ) -> NSSL2Mode:
    """Apply WRF v4.6.1's native option-18 default resolution.

    The public ``nssl_3moment`` switch is 0/1.  WRF rewrites 1 to its internal
    package value 2 when hail is enabled, allocating qzh as well as qzr/qzg.
    Deprecated scheme IDs are intentionally not accepted here; an importer
    must canonicalize them explicitly rather than disguising a legacy setup.
    """
    if mp_physics != MP_PHYSICS:
        raise ValueError(
            f"NSSL-2 contract requires mp_physics={MP_PHYSICS}, got "
            f"{mp_physics}")
    two = _selector("nssl_2moment_on", nssl_2moment_on, (-1, 0, 1))
    hail = _selector("nssl_hail_on", nssl_hail_on, (-1, 0, 1))
    ccn = _selector("nssl_ccn_on", nssl_ccn_on, (-1, 0, 1))
    density = _selector("nssl_density_on", nssl_density_on, (-1, 0, 1, 2))
    three = _selector("nssl_3moment", nssl_3moment, (0, 1))

    two = 1 if two < 0 else two
    hail = 1 if hail < 0 else hail
    ccn = 1 if ccn < 0 else ccn
    density = (2 if hail else 1) if density < 0 else density
    if density == 2 and not hail:
        raise ValueError(
            "nssl_density_on=2 predicts hail volume and requires "
            "nssl_hail_on=1")
    if three and not two:
        raise ValueError(
            "nssl_3moment=1 requires nssl_2moment_on=1")

    return NSSL2Mode(
        two_moment=bool(two),
        hail=bool(hail),
        predicted_ccn=bool(ccn),
        density_moments=density,
        sixth_moments=(2 if hail else 1) if three else 0,
    )


DEFAULT_MODE = resolve_nssl2_mode()
DEFAULT_RESTART_FIELDS = DEFAULT_MODE.transported_fields


__all__ = [
    "BASE_MASS_FIELDS",
    "CONTRACT_ID",
    "DEFAULT_MODE",
    "DEFAULT_RESTART_FIELDS",
    "GRAUPEL_VOLUME_FIELD",
    "HAIL_DIAGNOSTIC_FIELDS",
    "HAIL_MASS_FIELD",
    "HAIL_NUMBER_FIELD",
    "HAIL_VOLUME_FIELD",
    "MP_PHYSICS",
    "NSSL2Mode",
    "PRECIPITATION_FIELDS",
    "PREDICTED_CCN_FIELD",
    "RADIATION_EFFECTIVE_RADIUS_FIELDS",
    "REFLECTIVITY_FIELDS",
    "SIXTH_MOMENT_FIELDS",
    "TWO_MOMENT_NUMBER_FIELDS",
    "WRF_NAMELIST_DEFAULTS",
    "WRF_REFERENCE_COMMIT",
    "WRF_REFERENCE_VERSION",
    "resolve_nssl2_mode",
]
