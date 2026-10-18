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


#: The unit's driver scratch (gpuwm/core/morrison.py apply; CPU-only here
#: so the VRAM registry can price it without importing CuPy): the WORK
#: stack's plane count (kernels/morrison.cu MO_NWORK), the plane holding
#: the scheme's temperature (MO_T3D, WRF's T1D after the final loop), the
#: (nz, ny, nx) preparation slots, the chunked WORK slot and the int32
#: column slot.
NWORK = 21
WORK_T3D = 12
SCRATCH_VOLUME_SLOTS = ("morr_theta", "morr_rho", "morr_pii", "morr_dz")
SCRATCH_WORK_SLOT = "morr_work"
SCRATCH_INT_SURFACE_SLOTS = ("morr_ltrue",)


def work_chunk_columns(ncol: int) -> int:
    """Columns per Morrison call in a forecast step: ceil(ncol / NWORK).

    The WORK stack holds NWORK (nz, columns) planes for the columns of one
    call.  Sized for the whole domain it is NWORK volumes, the breakage
    this chunking prevents: carried at full size, the WRF transcription's
    WORK stack added 19 volumes of VRAM to the door-default microphysics
    (+186 MB on the d01 itemization fixture), and domains the card had
    held stopped fitting.  Chunked, WORK is at most one volume plus
    NWORK * nz values, so the scheme's scratch matches the kernel it
    replaced.  WRF's columns are independent: a column's words do not
    depend on its chunk (tests/test_morrison_regime_oracle.py grades the
    chunked call against WRF).
    """
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError(f"ncol must be positive, got {ncol}")
    return -(-ncol // NWORK)


def work_slot_shape(nz: int, ncol: int) -> tuple[int]:
    """The flat ``morr_work`` scratch shape for a domain's chunked call."""
    return (NWORK * int(nz) * work_chunk_columns(ncol),)


__all__ = ["RimedIceConstants", "rimed_ice_constants", "NWORK", "WORK_T3D",
           "SCRATCH_VOLUME_SLOTS", "SCRATCH_WORK_SLOT",
           "SCRATCH_INT_SURFACE_SLOTS", "work_chunk_columns",
           "work_slot_shape"]
