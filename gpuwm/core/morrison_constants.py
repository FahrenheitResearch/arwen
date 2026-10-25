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


#: The WORK stack a forecast step may hold whole before it is chunked: 128 MiB.
#: Below this, ceil(ncol / NWORK)-column calls bought no memory worth having
#: and cost the step a launch per chunk of three kernels that each fill a
#: few dozen blocks: the u2 user shape (156 x 136 x 50, GFS 3 km) ran its
#: microphysics as 21 calls of ~1,010 columns, 12.7 ms a step on the card
#: against 0.37 ms for 2.8.7's single launch, and the whole 48 h forecast
#: took 1.24x its 2.8.7 twin (WOOF 1.1 final canary, 547 s against 440 s).
#: A domain whose whole stack (NWORK volumes) is larger keeps WRF's
#: one-volume chunks: at 256 MiB the 200 x 251 x 49 d01 itemization fixture
#: (207 MB) took the whole stack again, 597 -> 793 MB of scratch, which is
#: the breakage the chunking exists to prevent (domains the card had held
#: stopped fitting).  The u2 shape's whole stack is 89 MB.
WORK_FLOOR_BYTES = 128 * 1024 * 1024


def work_chunk_columns(ncol: int, nz: int | None = None) -> int:
    """Columns per Morrison call in a forecast step.

    ceil(ncol / NWORK), or the whole domain in one call when ``nz`` is
    given (the forecast's own calls) and the domain's whole WORK stack,
    NWORK volumes, fits :data:`WORK_FLOOR_BYTES`; never more than ``ncol``.

    The WORK stack holds NWORK (nz, columns) planes for the columns of one
    call.  Sized for the whole domain it is NWORK volumes, the breakage
    this chunking prevents: carried at full size, the WRF transcription's
    WORK stack added 19 volumes of VRAM to the door-default microphysics
    (+186 MB on the d01 itemization fixture), and domains the card had
    held stopped fitting.  Chunked, WORK is at most one volume plus
    NWORK * nz values, or at most the 128 MiB floor for a small domain
    taken in one call, so a large domain keeps its one-volume scratch and a
    small one takes one launch.  WRF's columns are independent: a column's words do not
    depend on its chunk (tests/test_morrison_regime_oracle.py grades the
    chunked call against WRF).
    """
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError(f"ncol must be positive, got {ncol}")
    chunk = -(-ncol // NWORK)
    if nz is not None:
        nz = int(nz)
        if nz < 1:
            raise ValueError(f"nz must be positive, got {nz}")
        if NWORK * nz * ncol * 4 <= WORK_FLOOR_BYTES:
            # The whole domain's WORK stack fits the floor: one call.  A
            # larger domain keeps WRF-sized one-volume chunks; raising its
            # chunk to a 256 MiB floor added up to 256 MiB of scratch per domain
            # (the d01 itemization fixture 597 -> 793 MB, and the headline
            # domain stopped fitting its 16 GB card resident).
            chunk = ncol
    return min(chunk, ncol)


def work_slot_shape(nz: int, ncol: int) -> tuple[int]:
    """The flat ``morr_work`` scratch shape for a domain's chunked call."""
    return (NWORK * int(nz) * work_chunk_columns(ncol, nz),)


__all__ = ["RimedIceConstants", "rimed_ice_constants", "NWORK", "WORK_T3D",
           "SCRATCH_VOLUME_SLOTS", "SCRATCH_WORK_SLOT",
           "SCRATCH_INT_SURFACE_SLOTS", "WORK_FLOOR_BYTES", "work_chunk_columns",
           "work_slot_shape"]
