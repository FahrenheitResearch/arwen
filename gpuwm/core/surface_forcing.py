"""Persistent ARW surface-forcing contracts shared by RUC and Noah-MP.

WRF v4.6.1's ARW surface driver passes the microphysics accumulations
``RAINNCV``, ``SNOWNCV``, ``GRAUPELNCV`` and ``HAILNCV`` separately, plus
the convective ``RAINCV``/``RAINSHV`` accumulations.  Those are amounts in
millimetres accumulated since the preceding surface call; each LSM performs
its own WRF-prescribed conversion to rates.

Keeping these six values named prevents the old ``RAINBL``/``SR`` seam from
silently erasing hydrometeor identity before either LSM sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SURFACE_PRECIPITATION_FIELD_NAMES = {
    "rain_convective": "surface_raincv",
    "rain_nonconvective": "surface_rainncv",
    "rain_shallow_convective": "surface_rainshv",
    "snow_nonconvective": "surface_snowncv",
    "graupel_nonconvective": "surface_graupelncv",
    "hail_nonconvective": "surface_hailncv",
}

SURFACE_PRECIPITATION_FIELDS = tuple(
    SURFACE_PRECIPITATION_FIELD_NAMES.values())


@dataclass(frozen=True)
class SurfacePrecipitationForcing:
    """Six WRF ARW precipitation accumulations, each in mm."""

    rain_convective: object
    rain_nonconvective: object
    rain_shallow_convective: object
    snow_nonconvective: object
    graupel_nonconvective: object
    hail_nonconvective: object

    @classmethod
    def from_fields(
            cls, fields: Mapping[str, object]) -> "SurfacePrecipitationForcing":
        return cls(**{
            member: fields[field]
            for member, field in SURFACE_PRECIPITATION_FIELD_NAMES.items()
        })

    def clear(self) -> None:
        """Consume every accumulation exactly once after a surface call."""
        for member in SURFACE_PRECIPITATION_FIELD_NAMES:
            getattr(self, member)[...] = 0.0


def noahmp_six_precipitation_rates(
        total_precipitation, snow_ratio,
        forcing: SurfacePrecipitationForcing, dt: float, *, arrays):
    """Transcribe Noah-MP driver's six-rate ARW branch.

    Source: WRF v4.6.1 ``phys/noahmp/drivers/wrf/
    module_sf_noahmpdrv.F:776-786``.  Inputs are accumulated millimetres;
    outputs are the six ``PRCP*`` rates in mm s-1.  ``PRCPOTHR`` preserves
    WRF's residual: precipitation present in ``RAINBL`` but absent from the
    three liquid-rate inputs is added to nonconvective rain, and its
    ``SR`` fraction is additionally reported as snow.
    """
    xp = arrays
    dt32 = xp.float32(dt)
    # Preserve the source's six divisions.  Replacing them with one
    # reciprocal and multiplies differs by an FP32 ULP for ordinary DT.
    prcpconv = forcing.rain_convective / dt32
    prcpnonc = forcing.rain_nonconvective / dt32
    prcpshcv = forcing.rain_shallow_convective / dt32
    prcpsnow = forcing.snow_nonconvective / dt32
    prcpgrpl = forcing.graupel_nonconvective / dt32
    prcphail = forcing.hail_nonconvective / dt32
    prcp = total_precipitation / dt32
    prcpothr = xp.maximum(
        xp.float32(0.0), prcp - prcpconv - prcpnonc - prcpshcv)
    prcpnonc = prcpnonc + prcpothr
    prcpsnow = prcpsnow + snow_ratio * prcpothr
    return {
        "prcpconv": prcpconv.astype(xp.float32),
        "prcpnonc": prcpnonc.astype(xp.float32),
        "prcpshcv": prcpshcv.astype(xp.float32),
        "prcpsnow": prcpsnow.astype(xp.float32),
        "prcpgrpl": prcpgrpl.astype(xp.float32),
        "prcphail": prcphail.astype(xp.float32),
    }


__all__ = [
    "SURFACE_PRECIPITATION_FIELDS",
    "SURFACE_PRECIPITATION_FIELD_NAMES",
    "SurfacePrecipitationForcing",
    "noahmp_six_precipitation_rates",
]
