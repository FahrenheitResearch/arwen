"""WRF Morrison dense-ice constants selected by ``morr_rimed_ice``.

WRF v4.6.1 Registry ``Registry.EM_COMMON:2663-2666`` defaults the scalar
``&physics`` option to 1 (hail).  ``MORR_TWO_MOMENT_INIT`` selects AG/BG and
RHOG at ``phys/module_mp_morr_two_moment.F:337-382`` and derives CG from
RHOG at :404-411.  Keep that branch in one CPU-only module so the model,
reflectivity diagnostic, and independent NumPy mirror consume one selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RimedIceConstants:
    """Morrison fall-speed and mass-size constants for dense ice."""

    ag: float
    bg: float
    rhog: float

    @property
    def cg(self) -> float:
        """Mass-size coefficient ``CG = RHOG * PI / 6`` (WRF :410)."""
        return self.rhog * math.pi / 6.0


_GRAUPEL = RimedIceConstants(ag=19.3, bg=0.37, rhog=400.0)
_HAIL = RimedIceConstants(ag=114.5, bg=0.5, rhog=900.0)


def rimed_ice_constants(morr_rimed_ice: int = 1) -> RimedIceConstants:
    """Return WRF's graupel (0) or hail (1) Morrison constants."""
    if morr_rimed_ice == 0:
        return _GRAUPEL
    if morr_rimed_ice == 1:
        return _HAIL
    raise ValueError(
        "morr_rimed_ice must be 0 (graupel) or 1 (hail, WRF default), "
        f"got {morr_rimed_ice!r}.")


__all__ = ["RimedIceConstants", "rimed_ice_constants"]
