"""The warm FAST_SBM call: the host float64 column driver of the fast SBM.

Transcribed from WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1), routine ``FAST_SBM``, lines 4001-5322, on its warm
path:

* :4204-4212, the call constants (``NCOND``, ``NCOLL``, ``DTCOND``,
  ``DTCOLL``);
* :4214-4262, the entry conversion (through ``state.bins_to_distributions``);
* :4264-4276, ``pcgs``, ``rhocgs`` and ``zcgs``;
* :4281-4302, the first-step aerosol initialization (through the processes'
  ``initial_aerosol``, which binds the nucleation item's
  ``initial_aerosol_column``);
* :4351-4368, ``th_old``/``qv_old`` on the first step, ``T_new`` and
  ``T_old``;
* :4373-4395, the cloud-base level of each column, including the operator
  precedence WRF actually compiles (finding D9 of the oracle manifest,
  ``cloud_base_level_wrf``, which evaluates the nucleation item's
  ``cloud_base_level``);
* :4514-4852, the per-point loop in WRF's loop order (j, k, i): the fall
  speed correction (:4518-4531), the negative-bin zeroing (:4532-4577), the
  ``T_old > 193.15 K`` gate, the supersaturations at the old and advected
  states (:4582-4604), the gate on supersaturation or hydrometeors (:4606),
  the ``NCOND`` forcing substeps with ``DIFFU`` (:4610-4637), droplet
  nucleation (:4639-4676), the ``ISYM`` flags and the ``ONECOND1`` regime
  (:4678-4721), one ``COAL_BOTT_NEW`` step (:4766-4787), the write-back of
  T and qv (:4789-4790), the spontaneous breakup gate (:4826-4832) and the
  theta update (:4848);
* :4891-4913, drop sedimentation per column;
* :4974-5055, the bulk diagnostics (through ``state.bulk_diagnostics``);
* :5058-5076, the surface precipitation of the drops;
* :5266-5274, the ``th_old``/``qv_old`` update;
* :5276-5320, the exit conversion with its 1e-13 clip (through
  ``state.distributions_to_bins``).

The process routines themselves (``JERNUCL01_KS``, ``ONECOND1``,
``Kernals_KS`` and ``COAL_BOTT_NEW``, ``Spont_Rain_BreakUp``,
``FALFLUXHUCM_Z`` and the fall-speed correction) come through the
``FsbmProcesses`` protocol.  ``default_processes()`` binds the milestone 1
host functions of the derived-tables (C), condensation (D), nucleation (E),
collision-breakup (F) and sedimentation (G) items, by the names the fast-sbm
design fixes in its section 4.2; tests substitute stand-ins.

WRF is NCAR/UCAR public domain; the notice is in ``NOTICE`` and in
``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The scheme is FSBM-2 by Jacob
Shpund and co-authors, from Alexander Khain's Hebrew University Cloud Model
(see the package docstring and Shpund et al. 2019).

Scope and refusals
------------------
This is the warm path only.  The call refuses, before it changes anything
(it works on copies and never writes to the caller's tile), any tile with a
point whose ``T_old`` or ``T_new`` is at or below 273.15 K, and any nonzero
snow or dense-ice bin.  It refuses as well, first, any non-finite value in
the tile (``th``, ``qv``, ``th_old``, ``qv_old`` as the call uses them,
``w``, ``xland``, ``rainnc`` and every bin): each gate of ``FAST_SBM`` is a
comparison a NaN fails, so a NaN ``th_old`` or ``w`` would pass the cold
refusal, skip its point's processes without a word, and the exit would
write a finite ``th_old`` over it.  A substep temperature that reaches 273.15 K inside
the call is refused the same way.  Each refusal names the milestone 3
processes WRF would run there and this reference does not: ``FREEZ``
(immediate freezing, :4794), the ice branch of ``JERNUCL01_KS``
(``ICE_NUCL``, :3113-3180), ``ONECOND2`` and ``ONECOND3`` (:4723-4763),
the cold collision block of ``COAL_BOTT_NEW`` (:8445-8570), ``J_W_MELT``
(:4804) and the snow and dense-ice sedimentation (:4915-4964).

Behaviour
---------
``FsbmBehavior`` (``constants.py``) selects WRF's findings or their
corrections: D1 (``rainnc_velocity``) and D2 (``rainnc_source``) act here
in the surface precipitation, D3 (``nucleation_water_sink``) inside the
nucleation process, D5 (``dtcond_carry``) here in the substep length, D7
(``breakup_retry_cap``) inside the collision process, D9
(``cloud_base_rule``) here in the cloud-base detection, and D10
(``fall_speed_lookup``) in the coalescence efficiencies of the derived
tables.  D4, the exit clip, is kept in both presets and its mass is reported
(``FsbmWarmResult.clipped``).

D8: the sedimentation divides by the level spacing, WRF's divisor, and the
call refuses a behaviour whose ``fall_depth`` asks for the layer depth
until the fall-exit item builds it, so no result carries WRF's column water
under the other rule's label.  The behaviour's other cold-half fields (D11
to D15, D17, S1 and S2) act only on the cold path this call refuses.

The tile's cold inputs (``COLD_TILE_FIELDS``: the winds, grid spacing,
domain bounds, halo and the snow and dense-ice accumulations) are the
contract of the cold call; this call never reads them and returns them as
they came.

D10: ``derived`` carries the fall-speed lookup its ``ECOALMASSM`` was built
with (``build_derived`` takes it from the behaviour).  The call refuses a
derived table whose lookup is not the behaviour's, before it changes
anything: otherwise a ``corrected`` run on WRF's table (or the reverse)
would carry one behaviour's collision kernels under the other's label, as
the milestone 1 evidence did before this check.  The default processes also
refuse a derived table that does not say which lookup it holds.

D9: the cloud-base test at :4387 parses as ``(a .and. b .and. K_found)
.eqv. (.FALSE. .and. ...)``, so ``KZ_Cloud_Base`` ends at the highest level
that is not supersaturated with w above 10 cm/s.  The ``wrf_v480`` preset
reproduces it; the ``corrected`` preset takes the test the code's comment
describes (the lowest level above the second that is supersaturated, with
w above 10 cm/s and height below 3 km), as the fast-sbm design's D9 row
rules, so the ``Cloud_Base_Super`` activation branch (Pinsky et al. 2012)
runs at the cloud base.  The call's ``cloud_base_rule`` argument overrides
the behaviour's rule for measurement.

Numerics follow the rest of the port: float64 throughout, WRF's literals
exactly, WRF's operation order, and sequential sums where the Fortran sums
(``_seqsum``).  ``RAINNC`` and ``RAINNCV`` accumulate bin by bin as WRF's
do (the processes take the running total as ``start``).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol

import numpy as np

from gpuwm.core.fast_sbm.constants import (
    AA1_MY,
    AA2_MY,
    BB1_MY,
    BB2_MY,
    CLOUD_BASE_RULES as _CLOUD_BASE_RULES,
    COL,
    DenseIce,
    FsbmBehavior,
    FsbmRefusal,
    N_BIN_SCALARS,
    NCOLL,
    NCOND,
    NKR,
    NKR_AEROSOL,
)
from gpuwm.core.fast_sbm.state import (
    CHEM_SLICES,
    CLIP_DEFAULT,
    FsbmColumnBins,
    FsbmDistributions,
    bins_to_distributions,
    bulk_diagnostics,
    distributions_to_bins,
)

__all__ = [
    "CLOUD_BASE_RULES",
    "COLD_INCREMENT_PROCESSES",
    "COLD_PROCESSES",
    "COLD_TILE_FIELDS",
    "CoalStep",
    "CondensationStep",
    "FallStep",
    "FsbmIncrement",
    "FsbmPoint",
    "FsbmProcesses",
    "FsbmResult",
    "FsbmTally",
    "FsbmTile",
    "FsbmWarmResult",
    "HALO_FIELDS",
    "HALO_FILLS",
    "HALO_NEIGHBOURS",
    "HALO_NONE",
    "HALO_TILE_IS_DOMAIN",
    "INCREMENT_PROCESSES",
    "NucleationStep",
    "TALLY_CLASSES",
    "TALLY_DESTINATIONS",
    "TALLY_PAIRS",
    "WARM_INCREMENT_PROCESSES",
    "WrfWarmProcesses",
    "check_cold_inputs",
    "cloud_base_level_wrf",
    "default_processes",
    "fast_sbm_warm_call",
    "liquid_mixing_ratio",
    "whole_domain_tile",
]

#: The milestone 3 processes a cold point or an ice bin would need, named in
#: every refusal of the warm call.
COLD_PROCESSES = (
    "FREEZ (immediate freezing, :4794)",
    "ICE_NUCL (the ice branch of JERNUCL01_KS, :3113-3180)",
    "ONECOND2 and ONECOND3 (ice and mixed-phase growth, :4723-4763)",
    "the cold collision block of COAL_BOTT_NEW (:8445-8570)",
    "J_W_MELT (melting, :4804)",
    "snow and dense-ice sedimentation (:4915-4964)",
)

#: Cloud-base rules (``constants.CLOUD_BASE_RULES``): ``"wrf_v480"`` is the
#: test as WRF compiles it (D9); ``"first_level"`` the test its comment
#: describes.
CLOUD_BASE_RULES = _CLOUD_BASE_RULES

#: The freezing threshold WRF keys FREEZ and the cold collision block on.
_T_FREEZE = 273.15
#: ``IF (T_OLD(I,K,J).GT.193.15)`` (:4582).
_T_OLD_GATE = 193.15
#: ``IF ( TT >= 233.15 )`` before COAL_BOTT_NEW (:4769).
_T_COLL = 233.15
#: The ``DIFFU`` switch-off temperature (:4635).
_T_DIFFU = 265.0
#: The hydrometeor gate (:4606) and the spontaneous breakup gate (:4826).
_HYDRO_GATE = 1.0e-20
_SPONT_GATE = 43.0 * 1.0e-30
#: The ice ISYM thresholds (:4697-4704).
_ISYM_ICE = 1.0e-10


# ---------------------------------------------------------------------------
# Tile, points, results
# ---------------------------------------------------------------------------

class FsbmPoint(NamedTuple):
    """A grid point as WRF indexes it inside one call: one-based ``i``, ``j``,
    ``k`` within the tile, the call's ``itimestep`` and the condensation
    substep (``IKL``, 1..NCOND) where there is one.  ``k`` is ``None`` for a
    whole-column record (sedimentation, surface precipitation)."""

    itimestep: int
    i: int
    j: int
    k: int | None
    substep: int | None = None


#: How a tile's halo was filled (``FsbmTile.halo_fill``), by code:
#: 0 ``"none"``, not supplied; 1 ``"neighbours"``, the caller's values from
#: the domain around the tile, which must be the neighbours' state before
#: this call (finding D18: WRF's result there depends on the order of its
#: tiles, and the port defines it as the MPI halo exchange gives it);
#: 2 ``"tile_is_domain"``, built by
#: ``whole_domain_tile`` for a tile that is the whole domain, where every
#: horizontal branch of the ice pre-pass takes its edge form.
HALO_FILLS = ("none", "neighbours", "tile_is_domain")
HALO_NONE, HALO_NEIGHBOURS, HALO_TILE_IS_DOMAIN = range(len(HALO_FILLS))

#: The tile fields the ice pre-pass reads one column beyond the tile
#: (``FsbmTile.halo_*``).  ``th`` and ``qv`` are there because the first
#: step sets ``th_old`` and ``qv_old`` from them over the halo too
#: (:4351-4360).
HALO_FIELDS = ("th", "qv", "th_old", "qv_old", "p", "pii")

#: The tile fields that are scalars (``FsbmTile.copy``).
_TILE_SCALARS = ("ni", "dx", "dy", "ide", "jde", "its", "jts", "halo_fill")

#: The inputs only the cold call reads.  Their defaults mean "not
#: supplied": an empty array, a zero spacing or bound, halo code 0.  The
#: warm call never reads them and returns them unchanged.
COLD_TILE_FIELDS = ("u", "v", "dx", "dy", "ide", "jde", "its", "jts",
                    *(f"halo_{name}" for name in HALO_FIELDS), "halo_fill",
                    "snownc", "graupelnc")


def _not_supplied() -> np.ndarray:
    return np.zeros((0, 0))


def _not_supplied_1d() -> np.ndarray:
    return np.zeros(0)


@dataclass
class FsbmTile:
    """A tile of columns in WRF's order.

    Columns are stored flat, ``c = (j - 1) * ni + (i - 1)``, so a single row
    (``nj = 1``) is simply the columns in ``i`` order.  Per column and
    level, ``(ncol, nz)``: potential temperature ``th`` (K), ``qv``,
    ``th_old``, ``qv_old`` (kg/kg), pressure ``p`` (Pa), Exner ``pii``, dry
    air density ``rho`` (kg/m3) and layer depth ``dz`` (m, WRF's ``dz8w``).
    ``w`` is the vertical velocity on the ``nz + 1`` full levels (m/s).  The
    advected bins are ``ff1``, ``ff5``, ``ff6`` ``(ncol, nz, 33)`` and
    ``ff8`` ``(ncol, nz, 43)`` in WRF's stored form (see ``state``).
    ``xland`` is WRF's land flag (1 is land), ``rainnc`` the accumulated
    surface precipitation (mm).

    The cold call reads more (``COLD_TILE_FIELDS``), all of it for the ice
    pre-pass of :4372-4512 except the two accumulations:

    ``u``, ``v``
        The horizontal wind WRF hands ``FAST_SBM`` (m/s), over the tile and
        one column beyond it on the +i and +j sides, ``((nj + 1) * (ni +
        1), nz)``, flat as ``e = (j - 1) * (ni + 1) + (i - 1)`` for tile
        indices ``i`` in 1..ni+1 and ``j`` in 1..nj+1 (``wind_index``).
        WRF's ``solve_em`` passes the mass-point winds ``u_phy`` and
        ``v_phy`` here (``module_big_step_utilities_em.F`` :4857-4858, WRF
        v4.6.1 and v4.7.1), and the pre-pass averages ``U(I..I+1,
        J..J+1)`` (:4412-4413, :4498-4499), a wind half a cell off
        diagonally (finding D17).  The tile carries the winds as WRF hands
        them; ``FsbmBehavior.dsupice_wind`` chooses WRF's average
        (``"wrf_v480"``, which reads the +i and +j column) or each point's
        own wind (``"own_point"``).
    ``dx``, ``dy``
        The grid spacing (m), ``DXHUCM = 100 DX`` (:4259-4260).
    ``ide``, ``jde``, ``its``, ``jts``
        The domain and the tile's place in it, in WRF's convention: mass
        points run 1..``ide``-1 and 1..``jde``-1, and the tile's first
        column is (``its``, ``jts``).  The pre-pass switches its horizontal
        differences at ``I = 1``, ``I = IDE-1``, ``J = 1`` and ``J =
        JDE-1`` (:4411-4503).
    ``halo_th``, ``halo_qv``, ``halo_th_old``, ``halo_qv_old``,
    ``halo_p``, ``halo_pii``
        The six fields of ``HALO_FIELDS`` over the tile and one column on
        each side, ``((nj + 2) * (ni + 2), nz)``, flat as ``e = j * (ni + 2)
        + i`` for tile indices ``i`` in 0..ni+1 and ``j`` in 0..nj+1
        (``halo_index``).  The interior repeats the tile's own fields and
        must equal them.  The pre-pass reads ``T_OLD``, ``QV_OLD`` and
        ``pcgs`` at (i+-1, j) and (i, j+-1) (:4418-4488), never at a
        corner, and never outside the domain except in the dead
        ``DTFREEZ_XYZ`` reads at :4423 and :4430 (dead path
        ``dtfreez_xyz``); the first step sets ``th_old``, ``qv_old`` from
        ``th``, ``qv`` over the halo as well (:4351-4360).  The halo is
        the neighbours' state before this call, whatever order WRF would
        have called the tiles in (finding D18); the first-step reset
        changes only this call's ``T_OLD`` and ``QV_OLD``, never the halo
        the call returns.
    ``halo_fill``
        How the halo was filled, a code of ``HALO_FILLS``.
    ``snownc``, ``graupelnc``
        The accumulated surface snow and dense ice (mm), ``(ncol,)``, which
        ``FAST_SBM`` accumulates beside ``rainnc`` (:5082-5119).

    ``check_cold_inputs`` refuses a tile the cold call could not run.
    Every field is numeric, so a tile compares and copies field by field.
    """

    th: np.ndarray
    qv: np.ndarray
    th_old: np.ndarray
    qv_old: np.ndarray
    p: np.ndarray
    pii: np.ndarray
    rho: np.ndarray
    dz: np.ndarray
    w: np.ndarray
    ff1: np.ndarray
    ff5: np.ndarray
    ff6: np.ndarray
    ff8: np.ndarray
    xland: np.ndarray
    rainnc: np.ndarray
    ni: int | None = None
    u: np.ndarray = field(default_factory=_not_supplied)
    v: np.ndarray = field(default_factory=_not_supplied)
    dx: float = 0.0
    dy: float = 0.0
    ide: int = 0
    jde: int = 0
    its: int = 0
    jts: int = 0
    halo_th: np.ndarray = field(default_factory=_not_supplied)
    halo_qv: np.ndarray = field(default_factory=_not_supplied)
    halo_th_old: np.ndarray = field(default_factory=_not_supplied)
    halo_qv_old: np.ndarray = field(default_factory=_not_supplied)
    halo_p: np.ndarray = field(default_factory=_not_supplied)
    halo_pii: np.ndarray = field(default_factory=_not_supplied)
    halo_fill: int = HALO_NONE
    snownc: np.ndarray = field(default_factory=_not_supplied_1d)
    graupelnc: np.ndarray = field(default_factory=_not_supplied_1d)

    @property
    def ncol(self) -> int:
        return int(np.shape(self.th)[0])

    @property
    def nz(self) -> int:
        return int(np.shape(self.th)[1])

    @property
    def row_length(self) -> int:
        return self.ncol if self.ni is None else int(self.ni)

    @property
    def nrows(self) -> int:
        return self.ncol // self.row_length

    def wind_index(self, i: int, j: int) -> int:
        """Row of ``u`` and ``v`` for tile indices ``i`` in 1..ni+1 and
        ``j`` in 1..nj+1 (one-based, ``i = ni + 1`` is the column beyond
        the tile)."""
        return (j - 1) * (self.row_length + 1) + (i - 1)

    def halo_index(self, i: int, j: int) -> int:
        """Row of the ``halo_*`` arrays for tile indices ``i`` in 0..ni+1
        and ``j`` in 0..nj+1 (one-based, 0 and ni+1 are the halo)."""
        return j * (self.row_length + 2) + i

    @property
    def halo_fill_name(self) -> str:
        code = int(self.halo_fill)
        return HALO_FILLS[code] if 0 <= code < len(HALO_FILLS) else \
            f"unknown ({code})"

    def copy(self) -> "FsbmTile":
        """A deep copy: every array a new float64 array, the scalars as
        they are (``check_cold_inputs`` judges their types)."""
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in _TILE_SCALARS:
                values[name] = value
            else:
                values[name] = np.array(value, dtype=np.float64, copy=True)
        return FsbmTile(**values)

    def column_bins(self, c: int) -> FsbmColumnBins:
        return FsbmColumnBins(ff1=self.ff1[c], ff5=self.ff5[c],
                              ff6=self.ff6[c], ff8=self.ff8[c])

    def chem(self) -> np.ndarray:
        """The bins in WRF's ``chem_new`` order, ``(ncol, nz, 142)``."""
        out = np.empty((self.ncol, self.nz, N_BIN_SCALARS))
        for name, cols in CHEM_SLICES.items():
            out[:, :, cols] = getattr(self, name)
        return out

    @classmethod
    def from_chem(cls, chem, **fields) -> "FsbmTile":
        """Build a tile whose bins come from a ``(ncol, nz, 142)`` array in
        ``chem_new`` order; every other field is passed by name."""
        chem = np.asarray(chem, dtype=np.float64)
        bins = {name: chem[:, :, cols].copy()
                for name, cols in CHEM_SLICES.items()}
        return cls(**fields, **bins)


#: The process names of the warm call's increments.
WARM_INCREMENT_PROCESSES = (
    "negative_zeroing", "forcing", "nucleation", "onecond1", "coal_bott",
    "spont_breakup", "sedimentation", "exit_clip")
#: The process names the cold call adds: ice and mixed-phase growth
#: (:4730, :4748), immediate freezing (:4795), melting (:4805), the merge of
#: crystals into snow and of the two dense-ice slots (:4809-4821), and
#: homogeneous freezing where the behaviour runs it outside
#: ``JERNUCL01_KS`` (D14, ``"any_saturation"``).  Nucleation keeps its
#: name: WRF's one ``JERNUCL01_KS`` call holds the drop, homogeneous and
#: ice branches, and the crystals it makes join snow at once (:4667-4674).
COLD_INCREMENT_PROCESSES = (
    "onecond2", "onecond3", "freez", "j_w_melt", "merge",
    "homogeneous_freezing")
INCREMENT_PROCESSES = WARM_INCREMENT_PROCESSES + COLD_INCREMENT_PROCESSES


@dataclass(frozen=True)
class FsbmIncrement:
    """One process call's change at one point (or one column).

    ``d_ff1`` is the change of the drop spectrum in the scheme-internal form
    (``f``, 1/(g cm3)): ``(33,)`` at a point, ``(nz, 33)`` for a column
    process.  ``d_fccn`` is the aerosol change (cm-3 per ln radius) where the
    process touches aerosol.  ``d_t`` (K) and ``d_qv`` (g/g) are the changes
    of the point's working temperature and vapour.  ``process`` is one of
    ``INCREMENT_PROCESSES``: the warm call's ``"negative_zeroing"``,
    ``"forcing"``, ``"nucleation"``, ``"onecond1"``, ``"coal_bott"``,
    ``"spont_breakup"``, ``"sedimentation"`` and ``"exit_clip"``, and the
    cold call's ``COLD_INCREMENT_PROCESSES``.

    The cold increments carry the ice spectra in the same form, ``None``
    where the process leaves them alone: ``d_ff2`` the crystals of the
    three habits ``(33, 3)`` (columns, plates, dendrites; ``FF2R``, which
    ``FAST_SBM`` empties into snow within the point), ``d_ff3`` snow,
    ``d_ff4`` graupel and ``d_ff5`` hail (each ``(33,)``, or ``(nz, 33)``
    for a column process).
    """

    process: str
    point: FsbmPoint
    d_ff1: np.ndarray | None
    d_fccn: np.ndarray | None
    d_t: float
    d_qv: float
    d_ff2: np.ndarray | None = None
    d_ff3: np.ndarray | None = None
    d_ff4: np.ndarray | None = None
    d_ff5: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.process not in INCREMENT_PROCESSES:
            raise FsbmRefusal("FsbmIncrement", f"process {self.process!r} is "
                              f"not one of {INCREMENT_PROCESSES}")


#: The collision pairs a tally can record, keyed by name: the pair's
#: class (design 7.1: liquid-liquid, liquid-ice, ice-ice), its first and
#: second species (WRF's ``x`` and ``y`` arguments) and the kernel family
#: of the call.  ``drop_drop`` is milestone 1's ``coll_xxx_lwf`` call
#: (:8375); the others are the cold block of ``COAL_BOTT_NEW``
#: (:8440-8568): drops with snow (:8452-8471), with graupel (:8485-8513)
#: and with hail (:8524-8535), and snow with snow (:8563).  The crystal
#: pairs are absent: ``FF2R`` is zero before every collision (:4671-4672).
TALLY_PAIRS = {
    "drop_drop": ("liquid_liquid", "drop", "drop", "cwll"),
    "drop_snow": ("liquid_ice", "drop", "snow", "cwls"),
    "snow_drop": ("liquid_ice", "snow", "drop", "cwsl"),
    "drop_graupel": ("liquid_ice", "drop", "graupel", "cwlg"),
    "graupel_drop": ("liquid_ice", "graupel", "drop", "cwgl"),
    "drop_hail": ("liquid_ice", "drop", "hail", "cwlh"),
    "hail_drop": ("liquid_ice", "hail", "drop", "cwhl"),
    "snow_snow": ("ice_ice", "snow", "snow", "cwss"),
}
#: The tally classes of design 7.1.
TALLY_CLASSES = ("liquid_liquid", "liquid_ice", "ice_ice")

#: Where each pair's collected mass can go (``FsbmTally.destination``), by
#: the collision routine WRF calls: ``coll_xxx_lwf`` and ``coll_xyx_lwf``
#: keep it in the first species, ``coll_xyy_lwf`` puts it in the second,
#: ``coll_xyz_lwf`` in a third.  ``drop_snow`` always makes dense ice, the
#: run's class (``coll_xyz_lwf`` into hail or graupel, :8451-8457);
#: ``snow_drop`` keeps it in snow while ALWC < ALCR (:8461-8463) and makes
#: the run's dense ice otherwise (:8465-8471); drops with graupel make
#: graupel while ALWC < ALCR_G and hail otherwise (:8482-8513, the second
#: branch dead path ``graupel_to_hail``); drops with hail make hail
#: (:8524-8535).
TALLY_DESTINATIONS = {
    "drop_drop": ("drop",),
    "drop_snow": ("graupel", "hail"),
    "snow_drop": ("snow", "graupel", "hail"),
    "drop_graupel": ("graupel", "hail"),
    "graupel_drop": ("graupel", "hail"),
    "drop_hail": ("hail",),
    "hail_drop": ("hail",),
    "snow_snow": ("snow",),
}


@dataclass(frozen=True)
class FsbmTally:
    """The mass one collision call took per pair of bins, at one point.

    ``pair`` is a key of ``TALLY_PAIRS``.  ``taken_x[a-1, b-1]`` is the
    mass taken from bin ``a`` of the pair's first species by its collisions
    with bin ``b`` of the second, in the units of Bott's ``g`` (``3 x^2 f
    1e3``, :8325-8331), after WRF's clips, as milestone 1's ``coll_xxx_lwf``
    tally defines it.  For a self-collection (``drop_drop``,
    ``snow_snow``) that is the whole record and ``taken_y`` is ``None``; for
    two species ``taken_y[a-1, b-1]`` is the mass taken from bin ``b`` of
    the second species by the same collisions.  Each is ``(33, 33)``.  The
    emulator lane splits the liquid rows at its partition radius itself.

    ``destination`` is the species the collected mass went to, one of the
    pair's ``TALLY_DESTINATIONS``: ``"drop"``, ``"snow"``, ``"graupel"``
    or ``"hail"``.  For ``snow_drop`` it follows from the point's ALWC,
    and for ``drop_snow`` and ``snow_drop`` the dense species is the run's
    class, which ``FsbmResult`` checks against its ``dense_ice``.  A tally
    closes against the mass moved only with its destination: the taken
    mass leaves the pair's species and arrives in this one.
    """

    pair: str
    point: FsbmPoint
    taken_x: np.ndarray
    taken_y: np.ndarray | None
    destination: str

    def __post_init__(self) -> None:
        where = "FsbmTally"
        if self.pair not in TALLY_PAIRS:
            raise FsbmRefusal(where, f"pair {self.pair!r} is not one of "
                              f"{tuple(TALLY_PAIRS)}")
        _, first, second, _ = TALLY_PAIRS[self.pair]
        if np.shape(self.taken_x) != (NKR, NKR):
            raise FsbmRefusal(where, f"taken_x has shape "
                              f"{np.shape(self.taken_x)}, expected "
                              f"({NKR}, {NKR})")
        if (first == second) != (self.taken_y is None):
            raise FsbmRefusal(where, f"pair {self.pair!r} "
                              + ("is a self-collection and takes no taken_y"
                                 if first == second else
                                 "is two species and needs taken_y"))
        if self.taken_y is not None and np.shape(self.taken_y) != (NKR, NKR):
            raise FsbmRefusal(where, f"taken_y has shape "
                              f"{np.shape(self.taken_y)}, expected "
                              f"({NKR}, {NKR})")
        allowed = TALLY_DESTINATIONS[self.pair]
        if not isinstance(self.destination, str) or \
                self.destination not in allowed:
            raise FsbmRefusal(where, f"destination {self.destination!r} is "
                              f"not where pair {self.pair!r} puts its mass, "
                              f"one of {allowed}")

    @property
    def tally_class(self) -> str:
        return TALLY_PAIRS[self.pair][0]


@dataclass
class FsbmWarmResult:
    """What one call returns (``FsbmResult`` names it for both calls).

    ``tile`` is the state after the call (a new object; the input tile is
    never modified).  ``rainncv`` is this call's surface precipitation (mm)
    as the behaviour's D1 and D2 settings compute it; ``outflow_mm`` is the
    drop mass that left through the bottom face during sedimentation (mm,
    both presets, NaN when the sedimentation process reports no outflow).
    ``bulk`` holds the diagnostics of :4974-5055, each ``(ncol, nz)``.
    ``kz_cloud_base`` is WRF's one-based cloud-base level per column.
    ``nucleated`` is the drop mass mixing ratio created by activation during
    the call and ``nucleation_vapour`` the vapour change it caused (zero
    under WRF's D3), ``clipped`` the drop mass mixing ratio the exit clip
    zeroed (D4), all ``(ncol, nz)`` in g/g.  ``increments`` holds every
    process call's change when ``record_increments`` was set.

    The cold half adds: ``snowncv`` and ``graupelncv``, this call's surface
    snow and dense ice (mm, ``(ncol,)``; WRF's ``SNOWNCV`` and
    ``GRAUPELNCV``, :5059-5119), whose accumulations are ``tile.snownc`` and
    ``tile.graupelnc``; the ice diagnostics ``qi``, ``qs``, ``qg`` and
    their numbers ``qni``, ``qns``, ``qng`` (properties over ``bulk``);
    and ``tallies``, the per-pair collection records (``FsbmTally``) of the
    liquid-liquid, liquid-ice and ice-ice collisions when they were asked
    for.  The warm call has no snow or dense ice: its ``snowncv`` and
    ``graupelncv`` are exact zeros, as WRF's sums over zero bins are, and it
    records no tallies.

    ``dense_ice`` is the dense-ice option (``constants.DenseIce``) the call
    ran with; its ``identity`` is what the run records.  The cold call
    takes the option as a ``DenseIce`` from ``constants.dense_ice(name)``,
    never WRF's ``hail_opt`` integer, and records it here; every process
    item of milestone 2 takes the same ``DenseIce``.  The warm call leaves
    it ``None``: it refuses every dense-ice bin, so it ran neither class.
    A result with tallies names its option, and each tally's dense species
    is that option's (``drop_snow`` and ``snow_drop`` make the run's dense
    ice, and the other class's pairs are dead, ``drop_dense_other_build``).
    """

    tile: FsbmTile
    rainncv: np.ndarray
    outflow_mm: np.ndarray
    bulk: dict[str, np.ndarray]
    kz_cloud_base: np.ndarray
    nucleated: np.ndarray
    nucleation_vapour: np.ndarray
    clipped: np.ndarray
    increments: list[FsbmIncrement] | None = None
    dtcond_last: float = field(default=float("nan"))
    snowncv: np.ndarray | None = None
    graupelncv: np.ndarray | None = None
    tallies: list[FsbmTally] | None = None
    dense_ice: DenseIce | None = None

    def __post_init__(self) -> None:
        where = "FsbmResult"
        if self.dense_ice is not None and not isinstance(self.dense_ice,
                                                         DenseIce):
            raise FsbmRefusal(where, f"dense_ice must be a DenseIce "
                                     f"(constants.dense_ice(name)), got "
                                     f"{self.dense_ice!r}")
        if self.tallies is None:
            return
        if self.dense_ice is None:
            raise FsbmRefusal(where, "a result with tallies must name its "
                                     "dense_ice: drop_snow and snow_drop "
                                     "put their mass in the run's dense ice")
        dense = self.dense_ice.name
        other = "graupel" if dense == "hail" else "hail"
        for tally in self.tallies:
            _, first, second, _ = TALLY_PAIRS[tally.pair]
            if other in (first, second):
                raise FsbmRefusal(where, f"tally {tally.pair!r} at "
                                         f"{tally.point} meets {other}, which "
                                         f"the {dense} build leaves empty "
                                         f"(:4560-4577)")
            if tally.pair in ("drop_snow", "snow_drop") and \
                    tally.destination == other:
                raise FsbmRefusal(where, f"tally {tally.pair!r} at "
                                         f"{tally.point} sends its mass to "
                                         f"{other}, but the run's dense ice "
                                         f"is {dense} (:8451-8471)")

    @property
    def qi(self) -> np.ndarray:
        """Cloud ice, snow bins 1-18 (kg/kg, :5011-5015)."""
        return self.bulk["qi"]

    @property
    def qs(self) -> np.ndarray:
        """Snow, bins 19-33 (kg/kg, :5016-5021)."""
        return self.bulk["qs"]

    @property
    def qg(self) -> np.ndarray:
        """The dense-ice class, hail or graupel (kg/kg, :5024-5040)."""
        return self.bulk["qg"]

    @property
    def qni(self) -> np.ndarray:
        """Number of cloud ice (#/kg)."""
        return self.bulk["qni"]

    @property
    def qns(self) -> np.ndarray:
        """Number of snow (#/kg)."""
        return self.bulk["qns"]

    @property
    def qng(self) -> np.ndarray:
        """Number of the dense-ice class (#/kg)."""
        return self.bulk["qng"]


#: The result of either call.
FsbmResult = FsbmWarmResult


# ---------------------------------------------------------------------------
# What the processes hand back
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NucleationStep:
    """``JERNUCL01_KS`` on the warm path: the new drop spectrum (``FF1IN``),
    aerosol (``FCCN``), temperature, vapour and supersaturations."""

    ff1: np.ndarray
    fccn: np.ndarray
    tt: float
    qq: float
    del1in: float
    del2in: float


@dataclass(frozen=True)
class CondensationStep:
    """``ONECOND1``: the new drop spectrum, temperature, vapour and
    supersaturations (``DIV1``/``DIV2`` are not changed by the routine)."""

    ff1: np.ndarray
    tt: float
    qq: float
    del1in: float
    del2in: float


@dataclass(frozen=True)
class CoalStep:
    """``COAL_BOTT_NEW`` on the warm path."""

    ff1: np.ndarray
    tt: float
    qq: float
    del1in: float
    del2in: float


@dataclass(frozen=True)
class FallStep:
    """``FALFLUXHUCM_Z`` for one column: ``ffx`` after the fall (the WRF
    argument form, ``f / rhocgs``, ``(nz, 33)``) and the per-bin outflow
    through the bottom face as the sedimentation item defines it (``None``
    when unknown)."""

    ffx: np.ndarray
    outflow: np.ndarray | None


class FsbmProcesses(Protocol):
    """The process routines the warm call reaches, keyword arguments only.

    ``point`` locates each call (``FsbmPoint``).  Arrays are float64; a
    process must not modify its inputs.
    """

    def velocities(self, *, vr1: np.ndarray, p_cgs: np.ndarray) -> np.ndarray:
        """Drop fall speeds per level, ``VR1 * sqrt(1e6 / pcgs)``, (nz, 33)."""

    def initial_aerosol(self, *, derived: Any, rho_cgs: np.ndarray,
                        is_land: bool) -> np.ndarray:
        """The first-step aerosol of one column (:4281-4296), (nz, 43)."""

    def nucleation(self, *, point: FsbmPoint, ff1in: np.ndarray,
                   fccn: np.ndarray, tt: float, qq: float, rho_cgs: float,
                   p_cgs: float, del1in: float, del2in: float, w_stag: float,
                   is_cloud_base: int, kz_cloud_base: int, tables: Any,
                   derived: Any, behavior: FsbmBehavior) -> NucleationStep:
        """``JERNUCL01_KS`` on the warm path."""

    def onecond1(self, *, point: FsbmPoint, ff1r: np.ndarray, tt: float,
                 qq: float, p_cgs: float, rho_cgs: float, vr1: np.ndarray,
                 del1in: float, del2in: float, div1: float, div2: float,
                 dtcond: float, tables: Any) -> CondensationStep:
        """``ONECOND1``."""

    def coal(self, *, point: FsbmPoint, ff1r: np.ndarray, tt: float,
             qq: float, del1in: float, del2in: float, p_cgs: float,
             rho_cgs: float, dt_coll: float, tables: Any, derived: Any,
             behavior: FsbmBehavior) -> CoalStep:
        """``COAL_BOTT_NEW`` (with its ``Kernals_KS`` call) on the warm path."""

    def spont(self, *, point: FsbmPoint, ff1r: np.ndarray, dt: float,
              tables: Any, derived: Any) -> np.ndarray:
        """``Spont_Rain_BreakUp``: the new drop spectrum."""

    def falflux(self, *, point: FsbmPoint, ffx: np.ndarray, v: np.ndarray,
                rho_cgs: np.ndarray, zcgs: np.ndarray, dt: float) -> FallStep:
        """``FALFLUXHUCM_Z`` for the drops of one column."""

    def rainnc_level1(self, *, point: FsbmPoint, ff_bottom: np.ndarray,
                      v_bottom: np.ndarray, tables: Any, dt: float,
                      start: float) -> float:
        """WRF's surface precipitation (:5058-5076) in mm, accumulated bin by
        bin onto ``start`` as WRF accumulates onto RAINNC and RAINNCV."""

    def rainnc_outflow(self, *, point: FsbmPoint, outflow: np.ndarray,
                       tables: Any, start: float) -> float:
        """The bottom-face outflow as surface precipitation in mm,
        accumulated onto ``start`` the same way."""


# ---------------------------------------------------------------------------
# The default binding: the milestone 1 functions of items C, D, E, F and G
# ---------------------------------------------------------------------------

_ITEMS = {
    "derived": "fsbm-derived-tables",
    "condensation": "fsbm-condensation",
    "nucleation": "fsbm-nucleation",
    "collision": "fsbm-collision-breakup",
    "sedimentation": "fsbm-sedimentation",
}


def _call(fn: Callable, **available):
    """Call ``fn`` with the arguments it declares, taken by name.

    The design (4.2) fixes the argument names; items may add keyword
    arguments, which are passed when the column has a value of that name
    and left to their defaults otherwise.  A required argument the column
    cannot supply is an error naming it.
    """
    sig = inspect.signature(fn)
    args: list = []
    kwargs: dict = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in available:
            if param.kind == param.POSITIONAL_ONLY:
                args.append(available[name])
            else:
                kwargs[name] = available[name]
        elif param.default is param.empty:
            raise TypeError(
                f"{fn.__module__}.{fn.__qualname__} needs argument {name!r}, "
                f"which the warm column does not supply (it supplies "
                f"{sorted(available)})")
    return fn(*args, **kwargs)


def _pick(result, names: tuple[str, ...], what: str, default=inspect.Parameter.empty):
    """The first of ``names`` present on ``result`` (attribute or key)."""
    for name in names:
        if isinstance(result, dict) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    if default is not inspect.Parameter.empty:
        return default
    raise TypeError(f"{what} result {type(result).__name__} has none of "
                    f"{names}")


class WrfWarmProcesses:
    """The default ``FsbmProcesses``: the host functions of items C to G.

    Construction imports ``gpuwm.core.fast_sbm.{derived, condensation,
    nucleation, collision, sedimentation}`` and raises ``ImportError``
    naming the items whose modules are missing.  Collision kernels depend
    only on the pressure and the collision step, so they are computed once
    per pair and reused (``kernels``); the result is the same as WRF's call
    to ``Kernals_KS`` at every ``COAL_BOTT_NEW``.
    """

    def __init__(self) -> None:
        import importlib

        modules = {}
        missing = []
        for name, item in _ITEMS.items():
            try:
                modules[name] = importlib.import_module(
                    f"gpuwm.core.fast_sbm.{name}")
            except ImportError as exc:
                if exc.name != f"gpuwm.core.fast_sbm.{name}":
                    raise
                missing.append(f"gpuwm/core/fast_sbm/{name}.py ({item})")
        if missing:
            raise ImportError(
                "the warm column's default processes need "
                + ", ".join(missing)
                + "; these land with the other wave 2 items of the fast-sbm "
                  "program")
        self._derived = modules["derived"]
        self._cond = modules["condensation"]
        self._nucl = modules["nucleation"]
        self._coll = modules["collision"]
        self._sed = modules["sedimentation"]
        self._kernel_cache: dict[tuple[float, float], Any] = {}
        self._kernel_owner: tuple[Any, Any] | None = None

    # -- tables and kernels ------------------------------------------------
    def kernels(self, *, p_cgs: float, dt_coll: float, tables, derived):
        """``kernels_at_pressure`` of item C, cached per (pressure, step)
        for the tables and derived tables it was last called with."""
        owner = self._kernel_owner
        if owner is None or owner[0] is not tables or owner[1] is not derived:
            self._kernel_cache.clear()
            self._kernel_owner = (tables, derived)
        key = (float(p_cgs), float(dt_coll))
        found = self._kernel_cache.get(key)
        if found is None:
            found = _call(self._derived.kernels_at_pressure, tables=tables,
                          derived=derived, p_cgs=float(p_cgs),
                          dt_coll=float(dt_coll))
            if len(self._kernel_cache) > 4096:
                self._kernel_cache.clear()
            self._kernel_cache[key] = found
        return found

    # -- the protocol ------------------------------------------------------
    def velocities(self, *, vr1, p_cgs):
        return np.asarray(_call(self._sed.corrected_velocities, vr=vr1,
                                p_cgs=p_cgs), dtype=np.float64)

    def initial_aerosol(self, *, derived, rho_cgs, is_land):
        return np.asarray(_call(self._nucl.initial_aerosol_column,
                                fccnr_con=derived.fccnr_con,
                                fccnr_mar=derived.fccnr_mar,
                                rho_cgs=rho_cgs, is_land=is_land),
                          dtype=np.float64)

    def nucleation(self, *, point, ff1in, fccn, tt, qq, rho_cgs, p_cgs,
                   del1in, del2in, w_stag, is_cloud_base, kz_cloud_base,
                   tables, derived, behavior):
        res = _call(self._nucl.jernucl01_warm, psi1=ff1in, fccnr=fccn,
                    xl=tables.xl, tt=tt, qq=qq, ror=rho_cgs, pp=p_cgs,
                    sup1=del1in, sup2=del2in, w=w_stag,
                    is_cloud_base=bool(is_cloud_base), rccn=derived.rccn,
                    col=COL, behavior=behavior, tables=tables,
                    derived=derived, dropradii=getattr(derived, "dropradii",
                                                       None))
        what = "jernucl01_warm"
        return NucleationStep(
            ff1=np.asarray(_pick(res, ("psi1", "ff1", "ff1in"), what),
                           dtype=np.float64),
            fccn=np.asarray(_pick(res, ("fccnr", "fccn"), what),
                            dtype=np.float64),
            tt=float(_pick(res, ("tt",), what, tt)),
            qq=float(_pick(res, ("qq",), what, qq)),
            del1in=float(_pick(res, ("sup1", "del1in", "del1n"), what,
                               del1in)),
            del2in=float(_pick(res, ("sup2", "del2in", "del2n"), what,
                               del2in)))

    def onecond1(self, *, point, ff1r, tt, qq, p_cgs, rho_cgs, vr1, del1in,
                 del2in, div1, div2, dtcond, tables):
        res = _call(self._cond.onecond1, tt=tt, qq=qq, pp=p_cgs,
                    ror=rho_cgs, vr1=vr1, del1n=del1in, del2n=del2in,
                    ff1=ff1r, tables=tables, dtcond=dtcond, div1=div1,
                    div2=div2, psi1=ff1r.copy(), col=COL)
        what = "onecond1"
        return CondensationStep(
            ff1=np.asarray(_pick(res, ("ff1", "psi1", "ff1r"), what),
                           dtype=np.float64),
            tt=float(_pick(res, ("tt", "tpn"), what)),
            qq=float(_pick(res, ("qq", "qpn"), what)),
            del1in=float(_pick(res, ("del1n", "del1in", "del1"), what)),
            del2in=float(_pick(res, ("del2n", "del2in", "del2"), what)))

    def coal(self, *, point, ff1r, tt, qq, del1in, del2in, p_cgs, rho_cgs,
             dt_coll, tables, derived, behavior):
        lookup = getattr(derived, "fall_speed_lookup", None)
        if lookup != behavior.fall_speed_lookup:
            raise FsbmRefusal(
                "WrfWarmProcesses.coal",
                f"the derived tables' fall_speed_lookup is {lookup!r} and the "
                f"behaviour's is {behavior.fall_speed_lookup!r} (D10); the "
                f"collision kernels are built from derived.ecoalmassm, so "
                f"this collision step would run on efficiencies the "
                f"behaviour did not choose (build the table with "
                f"build_derived(tables, dt, behavior=...))")
        kern = self.kernels(p_cgs=p_cgs, dt_coll=dt_coll, tables=tables,
                            derived=derived)
        res = _call(self._coll.coal_bott_warm, ff1r=ff1r, tt=tt, qq=qq,
                    pp=p_cgs, rho=rho_cgs, dt_coll=dt_coll, tables=tables,
                    derived=derived, kernels=kern, behavior=behavior,
                    del1in=del1in, del2in=del2in)
        what = "coal_bott_warm"
        if isinstance(res, np.ndarray):
            return CoalStep(ff1=np.asarray(res, dtype=np.float64), tt=tt,
                            qq=qq, del1in=del1in, del2in=del2in)
        return CoalStep(
            ff1=np.asarray(_pick(res, ("ff1r", "ff1"), what),
                           dtype=np.float64),
            tt=float(_pick(res, ("tt",), what, tt)),
            qq=float(_pick(res, ("qq",), what, qq)),
            del1in=float(_pick(res, ("del1in", "del1n"), what, del1in)),
            del2in=float(_pick(res, ("del2in", "del2n"), what, del2in)))

    def spont(self, *, point, ff1r, dt, tables, derived):
        res = _call(self._coll.spont_rain_breakup, dt=dt, ff1r=ff1r,
                    xl=tables.xl, prob=derived.prob,
                    gain_var_new=derived.gain_var_new,
                    ikr_spon_break=derived.ikr_spon_break,
                    nnd=getattr(derived, "nnd", None))
        if not isinstance(res, np.ndarray):
            res = _pick(res, ("ff1r", "ff1"), "spont_rain_breakup")
        return np.asarray(res, dtype=np.float64)

    def falflux(self, *, point, ffx, v, rho_cgs, zcgs, dt):
        res = _call(self._sed.falflux, ffx=ffx, v=v, rho_cgs=rho_cgs,
                    zcgs=zcgs, dt=dt)
        what = "falflux"
        out = _pick(res, ("outflow",), what, None)
        return FallStep(
            ffx=np.asarray(_pick(res, ("ffx",), what), dtype=np.float64),
            outflow=None if out is None else np.asarray(out,
                                                        dtype=np.float64))

    def rainnc_level1(self, *, point, ff_bottom, v_bottom, tables, dt,
                      start):
        return float(_call(self._sed.surface_precip_level1,
                           ff_bottom=ff_bottom, v_bottom=v_bottom,
                           x=tables.xl, ro1bl=tables.ro1bl, col=COL, dt=dt,
                           start=start))

    def rainnc_outflow(self, *, point, outflow, tables, start):
        return float(_call(self._sed.surface_precip_outflow, outflow=outflow,
                           x=tables.xl, ro1bl=tables.ro1bl, col=COL,
                           start=start))


def default_processes() -> WrfWarmProcesses:
    """The processes of the other milestone 1 items (see ``WrfWarmProcesses``)."""
    return WrfWarmProcesses()


# ---------------------------------------------------------------------------
# Helpers that keep WRF's arithmetic
# ---------------------------------------------------------------------------

def _seqsum(values) -> float:
    """Fortran ``SUM`` at -O0: left to right, never pairwise."""
    total = 0.0
    for v in np.asarray(values, dtype=np.float64).ravel().tolist():
        total += v
    return total


def _supersaturation(tt: float, qq: float, pp: float):
    """:4588-4604 and :4625-4633: ``DIV1``, ``DEL1``, ``DIV2``, ``DEL2``."""
    es1n = AA1_MY * math.exp(-BB1_MY / tt)
    es2n = AA2_MY * math.exp(-BB2_MY / tt)
    ew1n = qq * pp / (0.622 + 0.378 * qq)
    return (ew1n / es1n, ew1n / es1n - 1.0, ew1n / es2n, ew1n / es2n - 1.0)


def liquid_mixing_ratio(f, xl, rho_cgs: float) -> float:
    """Drop mass mixing ratio (g/g) of a spectrum as ONECOND1 forms it,
    ``(sum f x x) * 3 COL / rho`` (:7106-7112)."""
    f = np.asarray(f, dtype=np.float64)
    return _seqsum(f * np.asarray(xl) * np.asarray(xl)) * (3.0 * COL) \
        * (1.0 / rho_cgs)


def _w_stag(w_col: np.ndarray, k: int, nz: int) -> float:
    """``w_stag_my`` in cm/s (:4380-4384, :4651-4655); ``k`` zero-based."""
    if k < nz - 1:
        return 50.0 * (w_col[k] + w_col[k + 1])
    return 100.0 * w_col[k]


def cloud_base_level_wrf(t_new, qv, p_cgs, w_full, zcgs, *,
                         rule: str = "wrf_v480") -> int:
    """``KZ_Cloud_Base`` of one column (:4373-4395), one-based.

    ``rule="wrf_v480"`` evaluates the test at :4387 as gfortran parses it:
    ``.EQV.`` binds more loosely than ``.AND.``, so the condition is
    ``(SUP_WATER > 0 .and. w > 10 cm/s .and. K_found) .eqv. (.FALSE. .and.
    ...)``, true exactly when ``(SUP_WATER > 0 .and. w > 10 .and. K_found)``
    is false.  ``K_found`` becomes true at level 1, so the result is the
    highest level that is not supersaturated with w above 10 cm/s (finding
    D9 of the oracle manifest).  ``rule="first_level"`` is the test the
    code describes: the lowest level above the second that is
    supersaturated, with w above 10 cm/s and height below 3 km; 0 when there
    is none, as WRF's initial ``KZ_Cloud_Base = 0``.

    One transcription serves both items: this evaluates the nucleation
    item's ``cloud_base_level`` (zero based, ``None`` for no level) with the
    rule given and returns WRF's one-based level.
    """
    from gpuwm.core.fast_sbm.nucleation import cloud_base_level

    if rule not in CLOUD_BASE_RULES:
        raise FsbmRefusal("cloud_base_level_wrf",
                          f"rule {rule!r} is not one of {CLOUD_BASE_RULES}")
    kz = cloud_base_level(t_new, qv, p_cgs, w_full, zcgs, rule=rule)
    return 0 if kz is None else kz + 1


#: Tile fields checked for non-finite values; ``th_old`` and ``qv_old``
#: after the first step's reset (:4351-4360), since WRF ignores what the
#: tile held there on step 1.
_FINITE_FIELDS = ("th", "qv", "th_old", "qv_old", "w", "xland", "rainnc",
                  "ff1", "ff5", "ff6", "ff8")


def _check_finite(state: "FsbmTile", where: str) -> None:
    """Refuse any NaN or infinity in the tile before a process runs.

    Every gate of the call is a comparison: the cold refusal (``T <=
    273.15``), the ``T_old > 193.15`` gate (:4582), the supersaturation and
    hydrometeor gate (:4606), the cloud-base test (:4387, w above 10 cm/s),
    the spontaneous breakup gate (:4826) and the land test of the first-step
    aerosol (:4281).  A NaN fails each of them, so the point would be
    skipped or carried on as if it were valid, and the exit writes
    ``th_old`` and ``qv_old`` from the call's own state (:5266-5274), which
    leaves no trace of a NaN that came in there.
    """
    ni = state.row_length
    for name in _FINITE_FIELDS:
        values = np.asarray(getattr(state, name), dtype=np.float64)
        bad = np.argwhere(~np.isfinite(values))
        if not bad.size:
            continue
        index = tuple(int(v) for v in bad[0])
        c = index[0]
        place = f"column i={c % ni + 1}, j={c // ni + 1}"
        if len(index) > 1:
            place += f", level k={index[1] + 1}"
        if len(index) > 2:
            place += f", bin {index[2] + 1}"
        raise FsbmRefusal(
            where, f"tile.{name} is {float(values[index])!r} at {place}; a "
                   f"non-finite value fails every comparison FAST_SBM gates "
                   f"on (the cold refusal, T_old > 193.15 K at :4582, the "
                   f"supersaturation and hydrometeor gate at :4606, the "
                   f"cloud-base test at :4387, the breakup gate at :4826), "
                   f"so its point would be skipped or carried on without a "
                   f"refusal, and the exit would write th_old and qv_old "
                   f"over it (:5266-5274)")


def _check_supplied_cold_inputs_finite(tile: "FsbmTile", where: str) -> None:
    """The warm call never reads the cold inputs and returns them as they
    came; it still refuses a non-finite one, as every entry point refuses
    NaN and infinity, rather than hand it back as valid state."""
    for name in COLD_TILE_FIELDS:
        values = np.asarray(getattr(tile, name), dtype=np.float64)
        bad = np.argwhere(~np.isfinite(values))
        if bad.size:
            index = tuple(int(v) for v in bad[0])
            raise FsbmRefusal(where, f"tile.{name}{list(index)} is "
                                     f"{float(values[index])!r}; the call "
                                     f"would return it unchanged as valid "
                                     f"state")


def _refuse_cold(where: str, what: str) -> FsbmRefusal:
    return FsbmRefusal(
        where, f"{what}; the warm reference refuses rather than skip "
               f"{'; '.join(COLD_PROCESSES)} (milestone 3)")


def _check_tile(tile: FsbmTile) -> None:
    where = "fast_sbm_warm_call"
    ncol, nz = np.shape(tile.th)
    shapes = {
        "th": (ncol, nz), "qv": (ncol, nz), "th_old": (ncol, nz),
        "qv_old": (ncol, nz), "p": (ncol, nz), "pii": (ncol, nz),
        "rho": (ncol, nz), "dz": (ncol, nz), "w": (ncol, nz + 1),
        "ff1": (ncol, nz, NKR), "ff5": (ncol, nz, NKR),
        "ff6": (ncol, nz, NKR), "ff8": (ncol, nz, NKR_AEROSOL),
        "xland": (ncol,), "rainnc": (ncol,),
    }
    for name, shape in shapes.items():
        got = np.shape(getattr(tile, name))
        if got != shape:
            raise FsbmRefusal(where, f"tile.{name} has shape {got}, expected "
                              f"{shape}")
    if nz < 2:
        raise FsbmRefusal(where, "a column needs at least two levels")
    ni = tile.row_length
    if ni < 1 or ncol % ni:
        raise FsbmRefusal(where, f"{ncol} columns do not fill rows of {ni}")
    for name in ("p", "pii", "rho", "dz"):
        if not np.all(np.asarray(getattr(tile, name)) > 0.0):
            raise FsbmRefusal(where, f"tile.{name} must be positive")


# ---------------------------------------------------------------------------
# The cold inputs of a tile
# ---------------------------------------------------------------------------

def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_))


def _finite_array(tile: FsbmTile, name: str, shape: tuple[int, ...],
                  where: str) -> np.ndarray:
    values = np.asarray(getattr(tile, name), dtype=np.float64)
    if values.size == 0:
        raise FsbmRefusal(where, f"tile.{name} is not supplied; the cold "
                                 f"call needs it with shape {shape}")
    if values.shape != shape:
        raise FsbmRefusal(where, f"tile.{name} has shape {values.shape}, "
                                 f"expected {shape}")
    bad = np.argwhere(~np.isfinite(values))
    if bad.size:
        index = tuple(int(v) for v in bad[0])
        raise FsbmRefusal(where, f"tile.{name}{list(index)} is "
                                 f"{float(values[index])!r}; a non-finite "
                                 f"cold input would pass into the ice "
                                 f"pre-pass's differences (DSUPICE_XYZ, "
                                 f":4504) or the accumulations without a "
                                 f"refusal")
    return values


def check_cold_inputs(tile: FsbmTile) -> None:
    """Refuse a tile whose cold inputs the cold call could not use.

    ``_check_tile``'s shape rules come first.  Then, naming the field, a
    ``FsbmRefusal`` for: an input left at its "not supplied" default
    (``COLD_TILE_FIELDS``); an array of the wrong shape or with a
    non-finite value; a spacing that is not positive; a tile that does not
    lie inside its domain (``its >= 1``, ``its + ni - 1 <= ide - 1``, and
    the same in j); a halo whose interior differs from the tile's own
    fields (two values for one point, and the pre-pass would difference
    the wrong one); a halo pressure or Exner that is not positive; an
    unknown halo code, or ``"tile_is_domain"`` on a tile that is not the
    whole domain.
    """
    where = "check_cold_inputs"
    _check_tile(tile)
    ncol, nz = tile.ncol, tile.nz
    ni, nj = tile.row_length, tile.nrows
    _finite_array(tile, "u", ((nj + 1) * (ni + 1), nz), where)
    _finite_array(tile, "v", ((nj + 1) * (ni + 1), nz), where)
    for name in ("dx", "dy"):
        value = getattr(tile, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise FsbmRefusal(where, f"tile.{name} must be a number, got "
                                     f"{value!r}")
        if not math.isfinite(float(value)) or not float(value) > 0.0:
            raise FsbmRefusal(where, f"tile.{name} is {value!r}; the grid "
                                     f"spacing (m) must be positive "
                                     f"(DXHUCM = 100 DX, :4259)")
    for name in ("ide", "jde", "its", "jts"):
        if not _is_int(getattr(tile, name)):
            raise FsbmRefusal(where, f"tile.{name} must be an int, got "
                                     f"{getattr(tile, name)!r}")
    ide, jde, its, jts = (int(tile.ide), int(tile.jde), int(tile.its),
                          int(tile.jts))
    if its < 1 or jts < 1:
        raise FsbmRefusal(where, f"the tile's first column (its={its}, "
                                 f"jts={jts}) is not supplied or lies "
                                 f"outside the domain; WRF's indices start "
                                 f"at 1")
    if its + ni - 1 > ide - 1 or jts + nj - 1 > jde - 1:
        raise FsbmRefusal(where, f"the tile's columns {its}..{its + ni - 1} "
                                 f"x {jts}..{jts + nj - 1} do not lie inside "
                                 f"the domain's mass points 1..{ide - 1} x "
                                 f"1..{jde - 1} (ide={ide}, jde={jde})")
    shape = ((nj + 2) * (ni + 2), nz)
    for name in HALO_FIELDS:
        halo = _finite_array(tile, f"halo_{name}", shape, where)
        own = np.asarray(getattr(tile, name), dtype=np.float64)
        for c in range(ncol):
            i, j = c % ni + 1, c // ni + 1
            row = halo[tile.halo_index(i, j)]
            if not np.array_equal(row, own[c]):
                k = int(np.argmax(row != own[c]))
                raise FsbmRefusal(
                    where, f"tile.halo_{name} at tile column i={i}, j={j}, "
                           f"level k={k + 1} is {row[k]!r} and tile.{name} "
                           f"there is {own[c, k]!r}; the halo's interior "
                           f"must repeat the tile")
        if name in ("p", "pii") and not np.all(halo > 0.0):
            raise FsbmRefusal(where, f"tile.halo_{name} must be positive")
    code = tile.halo_fill
    if not _is_int(code) or not 1 <= int(code) < len(HALO_FILLS):
        raise FsbmRefusal(where, f"tile.halo_fill is {code!r} "
                                 f"({tile.halo_fill_name}); a supplied halo "
                                 f"says how it was filled, one of "
                                 f"{dict(enumerate(HALO_FILLS))} other than 0")
    if int(code) == HALO_TILE_IS_DOMAIN and (
            (ide - 1, jde - 1, its, jts) != (ni, nj, 1, 1)):
        raise FsbmRefusal(where, f"tile.halo_fill says the tile is the whole "
                                 f"domain, but the domain is "
                                 f"{ide - 1} x {jde - 1} and the tile "
                                 f"{ni} x {nj} starts at ({its}, {jts})")
    for name in ("snownc", "graupelnc"):
        _finite_array(tile, name, (ncol,), where)


def whole_domain_tile(tile: FsbmTile, *, dx: float, dy: float, u, v,
                      snownc=None, graupelnc=None) -> FsbmTile:
    """The tile as a whole domain, with the cold inputs filled: a helper for
    single-column and stand-alone callers.

    Returns a copy of ``tile`` whose domain is the tile itself (``ide =
    ni + 1``, ``jde = nj + 1``, ``its = jts = 1``), so every horizontal
    branch of the ice pre-pass takes its edge form: the differences at
    :4422-4435, :4442-4493 use the point itself where a neighbour would lie
    outside the domain, and the wind average at :4411-4417, :4497-4503 (D17,
    ``dsupice_wind="wrf_v480"``) uses the point's own wind at the last
    column and row.  With that domain the
    pre-pass reads no halo value (the reads outside it, ``T_OLD`` at
    :4423 and :4430 on a one-column or one-row domain, feed
    ``DTFREEZ_XYZ``, which nothing reads).  The halo is still filled, by
    copying each edge column
    outward, so the tile passes ``check_cold_inputs``, and ``halo_fill``
    records ``"tile_is_domain"``.

    ``u`` and ``v`` are the tile's own winds, ``(ncol, nz)`` m/s, extended
    to the +i and +j column by the same copy.  ``dx`` and ``dy`` are the
    spacing in m.  ``snownc`` and ``graupelnc`` default to the tile's own
    when supplied and to zeros otherwise.
    """
    where = "whole_domain_tile"
    _check_tile(tile)
    out = tile.copy()
    ncol, nz = out.ncol, out.nz
    ni, nj = out.row_length, out.nrows
    winds = {}
    for name, value in (("u", u), ("v", v)):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (ncol, nz):
            raise FsbmRefusal(where, f"{name} has shape {array.shape}, "
                                     f"expected the tile's ({ncol}, {nz})")
        wide = np.empty(((nj + 1) * (ni + 1), nz))
        for j in range(1, nj + 2):
            for i in range(1, ni + 2):
                c = (min(j, nj) - 1) * ni + (min(i, ni) - 1)
                wide[out.wind_index(i, j)] = array[c]
        winds[name] = wide
    out.u, out.v = winds["u"], winds["v"]
    out.dx, out.dy = float(dx), float(dy)
    out.ide, out.jde, out.its, out.jts = ni + 1, nj + 1, 1, 1
    for name in HALO_FIELDS:
        own = np.asarray(getattr(out, name), dtype=np.float64)
        halo = np.empty(((nj + 2) * (ni + 2), nz))
        for j in range(0, nj + 2):
            for i in range(0, ni + 2):
                c = (min(max(j, 1), nj) - 1) * ni + (min(max(i, 1), ni) - 1)
                halo[out.halo_index(i, j)] = own[c]
        setattr(out, f"halo_{name}", halo)
    out.halo_fill = HALO_TILE_IS_DOMAIN
    for name, given in (("snownc", snownc), ("graupelnc", graupelnc)):
        if given is not None:
            setattr(out, name, np.array(given, dtype=np.float64, copy=True))
        elif np.size(getattr(out, name)) == 0:
            setattr(out, name, np.zeros(ncol))
    check_cold_inputs(out)
    return out


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def fast_sbm_warm_call(tile: FsbmTile, itimestep: int, dt: float, tables,
                       derived, *,
                       behavior: FsbmBehavior = FsbmBehavior.corrected(),
                       processes: FsbmProcesses | None = None,
                       record_increments: bool = False,
                       trace: Callable[[str, FsbmPoint, dict], None]
                       | None = None,
                       cloud_base_rule: str | None = None,
                       ) -> FsbmWarmResult:
    """One warm call of WRF's ``FAST_SBM`` for a tile of columns.

    ``tile`` is not modified; the state after the call is
    ``result.tile``.  ``itimestep`` is WRF's one-based step counter (step 1
    initializes the aerosol and sets ``th_old``/``qv_old``).  ``tables`` is
    ``load_tables("float64")``; ``derived`` the derived tables of item C
    (anything with ``fccnr_con``, ``fccnr_mar``, ``rccn``, ``prob``,
    ``gain_var_new``, ``nnd`` and ``ikr_spon_break`` serves the default
    processes).  ``trace``, when given, is called as ``trace(event, point,
    values)`` with the values WRF's probe anchors see: ``"entry"``,
    ``"cloud_base"``, ``"forcing"``, ``"isym"``, ``"fall"`` and ``"diag"``.
    ``cloud_base_rule`` is ``behavior.cloud_base_rule`` (D9) unless given.
    A ``derived`` that names a ``fall_speed_lookup`` other than
    ``behavior.fall_speed_lookup`` is refused (D10): build it with
    ``build_derived(tables, dt, behavior=behavior)``.  A behaviour whose
    ``fall_depth`` is not ``"level_spacing"`` is refused (D8): the
    layer-depth fall is not built yet.  The tile's cold inputs
    (``COLD_TILE_FIELDS``) are not read and come back unchanged; a
    non-finite one is refused.  The result's ``snowncv`` and ``graupelncv``
    are zeros.
    """
    where = "fast_sbm_warm_call"
    if isinstance(itimestep, bool) or int(itimestep) != itimestep \
            or itimestep < 1:
        raise FsbmRefusal(where, f"itimestep must be a positive int, got "
                          f"{itimestep!r}")
    itimestep = int(itimestep)
    dt = float(dt)
    if not dt > 0.0:
        raise FsbmRefusal(where, f"dt must be positive, got {dt!r}")
    if getattr(tables, "precision", "float64") != "float64":
        raise FsbmRefusal(where, "the host reference runs on the float64 "
                          "table views (load_tables('float64'))")
    if not isinstance(behavior, FsbmBehavior):
        raise FsbmRefusal(where, f"behavior must be an FsbmBehavior, got "
                          f"{type(behavior).__name__}")
    if cloud_base_rule is None:
        cloud_base_rule = behavior.cloud_base_rule
    if cloud_base_rule not in CLOUD_BASE_RULES:
        raise FsbmRefusal(where, f"cloud_base_rule {cloud_base_rule!r} is "
                          f"not one of {CLOUD_BASE_RULES}")
    lookup = getattr(derived, "fall_speed_lookup", None)
    if lookup is not None and lookup != behavior.fall_speed_lookup:
        raise FsbmRefusal(
            where, f"derived was built with fall_speed_lookup {lookup!r} and "
                   f"the behaviour asks for {behavior.fall_speed_lookup!r} "
                   f"(D10): the collision step would run on the other "
                   f"behaviour's coalescence efficiencies under this one's "
                   f"label; build it with build_derived(tables, dt, "
                   f"behavior=behavior)")
    # Guard for D8, retired by the fall-exit item when it builds the
    # layer-depth fall: until then the sedimentation process divides by the
    # level spacing whatever the behaviour says, so a "layer_depth" run
    # would report WRF's column water under the corrected label.
    if behavior.fall_depth != "level_spacing":
        raise FsbmRefusal(
            where, f"fall_depth {behavior.fall_depth!r} (D8) is not built: "
                   f"the sedimentation this call runs divides by the level "
                   f"spacing (FALFLUXHUCM_Z, :5364-5373), so the result would "
                   f"carry WRF's divisor under this behaviour's label")
    _check_tile(tile)
    _check_supplied_cold_inputs_finite(tile, where)
    if processes is None:
        processes = default_processes()

    state = tile.copy()
    ncol, nz = state.ncol, state.nz
    ni = state.row_length
    nj = ncol // ni
    xl = np.asarray(tables.xl, dtype=np.float64)

    # Snow and dense ice are milestone 3: refuse any nonzero bin.
    for name, label in (("ff5", "snow"), ("ff6", "dense-ice")):
        bad = np.argwhere(np.asarray(getattr(state, name)) != 0.0)
        if bad.size:
            c, k, b = (int(v) for v in bad[0])
            raise _refuse_cold(
                where, f"{label} bin {b + 1} at column i={c % ni + 1}, "
                       f"j={c // ni + 1}, level k={k + 1} is "
                       f"{getattr(state, name)[c, k, b]!r}, not zero")

    # :4264-4276
    pcgs = state.p * 10.0
    rhocgs = state.rho * 0.001
    zcgs = np.empty((ncol, nz))
    z_full = np.zeros(ncol)
    for k in range(nz):
        zcgs[:, k] = z_full + 0.5 * state.dz[:, k] * 100.0
        z_full = z_full + state.dz[:, k] * 100.0

    # :4351-4368
    if itimestep == 1:
        state.th_old = state.th.copy()
        state.qv_old = state.qv.copy()
    _check_finite(state, where)
    t_new = state.th * state.pii
    t_old = state.th_old * state.pii
    for name, temps in (("T_old", t_old), ("T_new", t_new)):
        cold = np.argwhere(temps <= _T_FREEZE)
        if cold.size:
            c, k = (int(v) for v in cold[0])
            raise _refuse_cold(
                where, f"{name} = {temps[c, k]!r} K at column "
                       f"i={c % ni + 1}, j={c // ni + 1}, level k={k + 1} "
                       f"is at or below {_T_FREEZE} K")

    # :4214-4262, entry conversion
    ff1r = np.empty((ncol, nz, NKR))
    ff3r = np.empty((ncol, nz, NKR))
    ff5r = np.empty((ncol, nz, NKR))
    fccn = np.empty((ncol, nz, NKR_AEROSOL))
    for c in range(ncol):
        dist = bins_to_distributions(state.column_bins(c), rhocgs[c], tables)
        ff1r[c], ff3r[c], ff5r[c], fccn[c] = (dist.ff1r, dist.ff3r,
                                              dist.ff5r, dist.fccn)
    # :4281-4296, first-step aerosol
    if itimestep == 1:
        for c in range(ncol):
            fccn[c] = processes.initial_aerosol(
                derived=derived, rho_cgs=rhocgs[c].copy(),
                is_land=bool(state.xland[c] == 1.0))

    qv = state.qv.copy()
    qv_old = state.qv_old
    if trace is not None:
        for c in range(ncol):
            for k in range(nz):
                trace("entry", FsbmPoint(itimestep, c % ni + 1, c // ni + 1,
                                         k + 1),
                      {"ff1": ff1r[c, k].copy(), "ff5": ff3r[c, k].copy(),
                       "ff6": ff5r[c, k].copy(), "fccn": fccn[c, k].copy(),
                       "t_new": t_new[c, k], "t_old": t_old[c, k],
                       "qv": qv[c, k], "qv_old": qv_old[c, k],
                       "pcgs": pcgs[c, k], "rhocgs": rhocgs[c, k],
                       "zcgs": zcgs[c, k]})

    # :4373-4395, cloud base per column (ICEPROCS = 1)
    kz_cloud_base = np.zeros(ncol, dtype=np.int64)
    for c in range(ncol):
        kz_cloud_base[c] = cloud_base_level_wrf(
            t_new[c], qv[c], pcgs[c], state.w[c], zcgs[c],
            rule=cloud_base_rule)
        if trace is not None:
            trace("cloud_base", FsbmPoint(itimestep, c % ni + 1,
                                          c // ni + 1, None),
                  {"kz_cloud_base": int(kz_cloud_base[c])})

    # :4518-4531, fall speeds per point (VR1_Z3D)
    vel = np.empty((ncol, nz, NKR))
    for c in range(ncol):
        vel[c] = processes.velocities(vr1=np.asarray(tables.vr1,
                                                     dtype=np.float64),
                                      p_cgs=pcgs[c].copy())

    dtcond = dt / float(NCOND)        # :4206, carried across points (D5)
    dt_coll = dt / float(NCOLL)       # :4207-4208
    increments: list[FsbmIncrement] | None = [] if record_increments else None
    nucleated = np.zeros((ncol, nz))
    nucl_vapour = np.zeros((ncol, nz))
    th_new = state.th.copy()
    rho_col = rhocgs

    def record(process, point, f_before, f_after, t_before, t_after,
               q_before, q_after, a_before=None, a_after=None):
        if increments is None:
            return
        increments.append(FsbmIncrement(
            process=process, point=point,
            d_ff1=None if f_before is None else f_after - f_before,
            d_fccn=None if a_before is None else a_after - a_before,
            d_t=t_after - t_before, d_qv=q_after - q_before))

    # :4514-4852, the per-point loop in WRF's order (j, k, i)
    for jrow in range(nj):
        for k in range(nz):
            for irow in range(ni):
                c = jrow * ni + irow
                base = FsbmPoint(itimestep, irow + 1, jrow + 1, k + 1)
                vr1z = vel[c, k]
                f_entry = ff1r[c, k]
                FF1R = np.where(f_entry < 0.0, 0.0, f_entry)
                FCCN = np.where(fccn[c, k] < 0.0, 0.0, fccn[c, k])
                FF3R = np.where(ff3r[c, k] < 0.0, 0.0, ff3r[c, k])
                FF5R = np.where(ff5r[c, k] < 0.0, 0.0, ff5r[c, k])
                FF4R = np.zeros(NKR)
                if increments is not None and (
                        np.any(f_entry < 0.0) or np.any(fccn[c, k] < 0.0)):
                    record("negative_zeroing", base, f_entry, FF1R, 0.0, 0.0,
                           0.0, 0.0, fccn[c, k], FCCN)
                tnew_pt = t_new[c, k]
                qv_pt = qv[c, k]
                if t_old[c, k] > _T_OLD_GATE:
                    TT = t_old[c, k]
                    QQ = qv_old[c, k]
                    if QQ <= 0.0:
                        QQ = 1.0e-10
                    PP = pcgs[c, k]
                    TTA = t_new[c, k]
                    QQA = qv[c, k]
                    if QQA <= 0.0:
                        QQA = 1.0e-10
                    DIV1, DEL1IN, DIV2, DEL2IN = _supersaturation(TT, QQ, PP)
                    DIV3, DEL1AD, DIV4, DEL2AD = _supersaturation(TTA, QQA,
                                                                  PP)
                    hydro = ((_seqsum(FF1R) + _seqsum(FF3R)) + _seqsum(FF4R)) \
                        + _seqsum(FF5R)
                    if DEL1AD > 0.0 or DEL2AD > 0.0 or hydro > _HYDRO_GATE:
                        DEL_T = (TTA - TT) / float(NCOND)
                        DEL_Q = (QQA - QQ) / float(NCOND)
                        DIFFU = 1
                        if DIV1 == DIV3:
                            DIFFU = 0
                        if DIV2 == DIV4:
                            DIFFU = 0
                        if not behavior.dtcond_carry:
                            dtcond = dt / float(NCOND)
                        DTNEW = 0.0
                        for ikl in range(1, NCOND + 1):
                            dtcond = min(dt - DTNEW, dtcond)
                            DTNEW = DTNEW + dtcond
                            if DIFFU == 0:
                                continue
                            sub = base._replace(substep=ikl)
                            t0, q0 = TT, QQ
                            TT = TT + DEL_T
                            QQ = QQ + DEL_Q
                            DIV1, DEL1IN, DIV2, DEL2IN = _supersaturation(
                                TT, QQ, PP)
                            record("forcing", sub, None, None, t0, TT, q0, QQ)
                            if trace is not None:
                                trace("forcing", sub, {
                                    "ikl": ikl, "tt": TT, "qq": QQ,
                                    "del1in": DEL1IN, "del2in": DEL2IN,
                                    "div1": DIV1, "div2": DIV2,
                                    "dtcond": dtcond, "del_t": DEL_T,
                                    "del_q": DEL_Q})
                            if TT <= _T_FREEZE:
                                raise _refuse_cold(
                                    where, f"substep {ikl} temperature "
                                           f"{TT!r} K at {sub} is at or "
                                           f"below {_T_FREEZE} K")
                            if DIV1 > DIV2 and TT <= _T_DIFFU:
                                DIFFU = 0
                            if DIFFU != 1:
                                continue
                            if DEL1IN > 0.0 or DEL2IN > 0.0:
                                FF1IN = FF1R.copy()
                                is_cb = 1 if (
                                    kz_cloud_base[c] == k + 1
                                    and COL * _seqsum(FF1IN * xl) < 5.0) \
                                    else 0
                                w_stag = _w_stag(state.w[c], k, nz)
                                t0, q0 = TT, QQ
                                a0 = FCCN
                                res = processes.nucleation(
                                    point=sub, ff1in=FF1IN, fccn=FCCN.copy(),
                                    tt=TT, qq=QQ, rho_cgs=rho_col[c, k],
                                    p_cgs=pcgs[c, k], del1in=DEL1IN,
                                    del2in=DEL2IN, w_stag=w_stag,
                                    is_cloud_base=is_cb,
                                    kz_cloud_base=int(kz_cloud_base[c]),
                                    tables=tables, derived=derived,
                                    behavior=behavior)
                                f_new = np.asarray(res.ff1, dtype=np.float64)
                                nucleated[c, k] += (
                                    liquid_mixing_ratio(f_new, xl,
                                                        rho_col[c, k])
                                    - liquid_mixing_ratio(FF1R, xl,
                                                          rho_col[c, k]))
                                nucl_vapour[c, k] += res.qq - q0
                                record("nucleation", sub, FF1R, f_new, t0,
                                       res.tt, q0, res.qq, a0,
                                       np.asarray(res.fccn, dtype=np.float64))
                                FF1R = f_new
                                FCCN = np.asarray(res.fccn, dtype=np.float64)
                                TT, QQ = res.tt, res.qq
                                DEL1IN, DEL2IN = res.del1in, res.del2in
                            FMAX1 = max(0.0, float(np.max(FF1R)))
                            ISYM1 = 1 if FMAX1 > 0.0 else 0
                            ISYM3 = 1 if float(np.max(FF3R)) > _ISYM_ICE else 0
                            ISYM4 = 1 if float(np.max(FF4R)) > _ISYM_ICE else 0
                            ISYM5 = 1 if float(np.max(FF5R)) > _ISYM_ICE else 0
                            no_ice = ISYM3 == 0 and ISYM4 == 0 and ISYM5 == 0
                            warm = (TT - 273.15) > -0.187 or no_ice
                            if trace is not None:
                                trace("isym", sub, {
                                    "isym1": ISYM1, "isym3": ISYM3,
                                    "isym4": ISYM4, "isym5": ISYM5,
                                    "onecond1": bool(ISYM1 == 1 and warm)})
                            if ISYM1 == 1 and warm:
                                if TT <= _T_FREEZE:
                                    raise _refuse_cold(
                                        where, f"temperature {TT!r} K before "
                                               f"ONECOND1 at {sub} is at or "
                                               f"below {_T_FREEZE} K")
                                t0, q0 = TT, QQ
                                res = processes.onecond1(
                                    point=sub, ff1r=FF1R.copy(), tt=TT, qq=QQ,
                                    p_cgs=pcgs[c, k], rho_cgs=rho_col[c, k],
                                    vr1=vr1z.copy(), del1in=DEL1IN,
                                    del2in=DEL2IN, div1=DIV1, div2=DIV2,
                                    dtcond=dtcond, tables=tables)
                                f_new = np.asarray(res.ff1, dtype=np.float64)
                                record("onecond1", sub, FF1R, f_new, t0,
                                       res.tt, q0, res.qq)
                                FF1R = f_new
                                TT, QQ = res.tt, res.qq
                                DEL1IN, DEL2IN = res.del1in, res.del2in
                            elif not no_ice:   # unreachable after the refusals
                                raise _refuse_cold(where, f"ice at {sub}")
                        for _ in range(NCOLL):
                            if TT >= _T_COLL:
                                if TT <= _T_FREEZE:
                                    raise _refuse_cold(
                                        where, f"temperature {TT!r} K before "
                                               f"COAL_BOTT_NEW at {base} is "
                                               f"at or below {_T_FREEZE} K")
                                t0, q0 = TT, QQ
                                res = processes.coal(
                                    point=base, ff1r=FF1R.copy(), tt=TT,
                                    qq=QQ, del1in=DEL1IN, del2in=DEL2IN,
                                    p_cgs=PP, rho_cgs=rho_col[c, k],
                                    dt_coll=dt_coll, tables=tables,
                                    derived=derived, behavior=behavior)
                                f_new = np.asarray(res.ff1, dtype=np.float64)
                                record("coal_bott", base, FF1R, f_new, t0,
                                       res.tt, q0, res.qq)
                                FF1R = f_new
                                TT, QQ = res.tt, res.qq
                                DEL1IN, DEL2IN = res.del1in, res.del2in
                        tnew_pt = TT
                        qv_pt = QQ
                # :4792-4822: FREEZ below 273.15 K; J_W_MELT and the merge of
                # the ice classes do nothing without ice.
                if tnew_pt < _T_FREEZE:
                    raise _refuse_cold(where, f"T_new {tnew_pt!r} K at {base} "
                                              f"is below {_T_FREEZE} K")
                # :4826-4832
                if _seqsum(FF1R) > _SPONT_GATE:
                    f_new = np.asarray(processes.spont(
                        point=base, ff1r=FF1R.copy(), dt=dt, tables=tables,
                        derived=derived), dtype=np.float64)
                    record("spont_breakup", base, FF1R, f_new, tnew_pt,
                           tnew_pt, qv_pt, qv_pt)
                    FF1R = f_new
                # :4848 and the write-back (:4850-4886)
                t_new[c, k] = tnew_pt
                qv[c, k] = qv_pt
                th_new[c, k] = tnew_pt / state.pii[c, k]
                ff1r[c, k] = FF1R
                fccn[c, k] = FCCN
                ff3r[c, k] = FF3R
                ff5r[c, k] = FF5R

    # :4891-4913, drop sedimentation per column
    outflow_mm = np.full(ncol, np.nan)
    outflows: list[np.ndarray | None] = [None] * ncol
    for jrow in range(nj):
        for irow in range(ni):
            c = jrow * ni + irow
            colpt = FsbmPoint(itimestep, irow + 1, jrow + 1, None)
            rho_c = rhocgs[c][:, None]
            ffx = ff1r[c] / rho_c
            if trace is not None:
                trace("fall", colpt, {"ffx": ffx.copy(), "vrx": vel[c].copy(),
                                      "rhocgs": rhocgs[c].copy(),
                                      "pcgs": pcgs[c].copy(),
                                      "zcgs": zcgs[c].copy()})
            fall = processes.falflux(point=colpt, ffx=ffx.copy(),
                                     v=vel[c].copy(),
                                     rho_cgs=rhocgs[c].copy(),
                                     zcgs=zcgs[c].copy(), dt=dt)
            f_new = np.asarray(fall.ffx, dtype=np.float64) * rho_c
            record("sedimentation", colpt, ff1r[c], f_new, 0.0, 0.0, 0.0,
                   0.0)
            ff1r[c] = f_new
            outflows[c] = fall.outflow
            if fall.outflow is not None:
                outflow_mm[c] = processes.rainnc_outflow(
                    point=colpt, outflow=fall.outflow, tables=tables,
                    start=0.0)

    # :4974-5055, diagnostics
    dists = [FsbmDistributions(ff1r=ff1r[c], ff3r=ff3r[c],
                               ff4r=np.zeros((nz, NKR)), ff5r=ff5r[c],
                               fccn=fccn[c]) for c in range(ncol)]
    bulks = [bulk_diagnostics(dists[c], rhocgs[c], tables)
             for c in range(ncol)]
    bulk = {name: np.stack([getattr(b, name) for b in bulks])
            for name in ("qc", "qr", "qi", "qs", "qg", "qnc", "qnr", "qni",
                         "qns", "qng", "qna")}

    # :5058-5076, surface precipitation of the drops (D1, D2)
    # RAINNC and RAINNCV accumulate bin by bin, onto the running total and
    # onto zero (:5065-5070); snow and dense ice add exact zeros here.
    rainncv = np.zeros(ncol)
    rainnc = state.rainnc.copy()
    last = ncol - 1                      # VR1_Z(:,1) after the loop (D1)
    for jrow in range(nj):
        for irow in range(ni):
            c = jrow * ni + irow
            colpt = FsbmPoint(itimestep, irow + 1, jrow + 1, None)
            if behavior.rainnc_source == "level1_after_fall":
                v_c = vel[last if behavior.rainnc_velocity == "tile_last"
                          else c, 0]
                for start, sink in ((float(rainnc[c]), rainnc),
                                    (0.0, rainncv)):
                    sink[c] = processes.rainnc_level1(
                        point=colpt, ff_bottom=ff1r[c, 0].copy(),
                        v_bottom=v_c.copy(), tables=tables, dt=dt,
                        start=start)
            else:
                if outflows[c] is None:
                    raise FsbmRefusal(
                        where, "rainnc_source 'bottom_outflow' needs the "
                               "sedimentation process to report its outflow")
                rainncv[c] = outflow_mm[c]
                rainnc[c] = processes.rainnc_outflow(
                    point=colpt, outflow=outflows[c], tables=tables,
                    start=float(rainnc[c]))
    if trace is not None:
        for c in range(ncol):
            trace("diag", FsbmPoint(itimestep, c % ni + 1, c // ni + 1, None),
                  {**{name: bulk[name][c].copy() for name in bulk},
                   "qv": qv[c].copy(), "th": th_new[c].copy(),
                   "rainnc": rainnc[c], "rainncv": rainncv[c],
                   "vr1_z_level1_after_loop": vel[last, 0].copy()})

    # :5266-5274 and the exit conversion with the clip (:5276-5320, D4)
    clipped = np.zeros((ncol, nz))
    out = state
    out.th = th_new
    out.qv = qv
    out.th_old = th_new.copy()
    out.qv_old = qv.copy()
    out.rainnc = rainnc
    for c in range(ncol):
        unclipped = distributions_to_bins(dists[c], rhocgs[c], tables,
                                          bulks[c], clip=None)
        bins = distributions_to_bins(dists[c], rhocgs[c], tables, bulks[c],
                                     clip=CLIP_DEFAULT)
        gone = np.any(unclipped.ff1 != bins.ff1, axis=1)
        clipped[c, gone] = np.array([_seqsum(row)
                                     for row in unclipped.ff1[gone]])
        if increments is not None and np.any(gone):
            d = np.zeros((nz, NKR))
            d[gone] = -ff1r[c][gone]
            increments.append(FsbmIncrement(
                process="exit_clip",
                point=FsbmPoint(itimestep, c % ni + 1, c // ni + 1, None),
                d_ff1=d, d_fccn=None, d_t=0.0, d_qv=0.0))
        out.ff1[c], out.ff5[c], out.ff6[c], out.ff8[c] = (bins.ff1, bins.ff5,
                                                          bins.ff6, bins.ff8)

    # :5059-5060 and :5072-5121: SNOWNCV and GRAUPELNCV start at zero and
    # add zero bins on the warm path; SNOWNC and GRAUPELNC stay as they came.
    return FsbmWarmResult(tile=out, rainncv=rainncv, outflow_mm=outflow_mm,
                          bulk=bulk, kz_cloud_base=kz_cloud_base,
                          nucleated=nucleated,
                          nucleation_vapour=nucl_vapour, clipped=clipped,
                          increments=increments, dtcond_last=dtcond,
                          snowncv=np.zeros(ncol), graupelncv=np.zeros(ncol))
