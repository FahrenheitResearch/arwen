"""Named step-local device workspace for the MYNN PBL solver.

Every device array the MYNN column solver needs between its own kernels is
declared here, once, as a named slot with an exact shape formula in
``(ncol, nz)``.  Nothing in :mod:`gpuwm.core.mynn_pbl_gpu` allocates: it
draws views from a holder built here, and on the runtime path that holder
is backed by :meth:`gpuwm.core.state.DomainState.scratch`, which is the
mechanism ``preflight.scratch_slot_registry`` prices and the shared arena
shares between domains.

Why this module exists at all
-----------------------------
The solver used to allocate its whole working set on every call.  Measured
on the RTX 5090 at ``nz = 49`` by counting every request that reached the
CuPy device allocator during one carried ``mynn_pbl_step``:

  ==========  =========================  ==========  ================
  columns     bytes allocated per step   per column  pool allocations
  ==========  =========================  ==========  ================
  25,600      1,332.9 MiB                54,595      484
  102,400     5,331.6 MiB                54,595      484
  360,000     18,743.8 MiB               54,595      484
  ==========  =========================  ==========  ================

360,000 is the d04 nest of ``configs/real74_4dom``.  None of those arrays
appeared in ``physics_array_shapes`` or in the scratch registry, so the
preflight's estimate for a MYNN nest was not merely low, it was blind: what
it could not see outweighed what it could.  On a card with no ECC that is
worse than having no preflight, because the headroom check returns a
reassuring number.

The same measurement through this module reports **54.1 bytes per column**,
18.6 MiB per step at the d04 width -- a factor of 1,008 -- and every byte of
what remains resident is a named slot the registry prices.

Two properties of the declaration matter and are enforced by tests:

* **Bounded.** Slot shapes are written against a *column chunk*, not
  against ``ny * nx``, and ``mynn_pbl_runtime`` walks the domain in chunks
  of that width.  :data:`MYNN_PBL_COLUMN_CHUNK` is a ceiling, so a 600x600
  nest and a 501x501 nest declare byte-for-byte the same workspace and a
  domain narrower than the chunk declares less, never more.  A workspace
  that grew with the domain would price honestly and still not fit.
* **Write-before-read, or constant.** Every slot is either overwritten by
  the kernel that owns it before anything reads it, or it is one of the
  explicitly listed constant-zero feeds WRF passes to a system this lane
  has switched off.  :meth:`MynnPblScratch.poison` fills the first group
  with NaNs, and ``tests/test_mynn_pbl_scratch.py`` uses it to show that a
  misclassified slot changes the forecast.  The constant-zero group is
  never poisoned and never arena-aliased, following the
  ``physics_dry_qv``/``physics_dry_qc`` precedent in the lifetime audit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

import cupy as cp

from gpuwm.core.state import DTYPE

#: Default columns per MYNN call.  Measured on the RTX 5090 at nz = 49 over
#: the 360,000-column d04 batch: a fresh input batch per row, one
#: ``initflag=1`` call, two warm-up steps, five timed steps, median.  Every
#: width produced the same carried-state hash, including the unchunked call:
#:
#:   ==========  ===========  ===========  ==============
#:   chunk       us / column  s / step     workspace MiB
#:   ==========  ===========  ===========  ==============
#:   2,048            4.4642       1.6071          505.2
#:   4,096            2.5064       0.9023          606.7
#:   8,192            1.5609       0.5619          809.7
#:   16,384           1.4929       0.5374         1215.6
#:   24,576           1.4563       0.5243         1621.6
#:   49,152           1.6716       0.6018         2839.4
#:   360,000          2.1621       0.7784        18242.8
#:   ==========  ===========  ===========  ==============
#:
#: "workspace MiB" is every live slot including the six full-width returned
#: tendency fields, which are 423.4 MiB of it at this nest and do not shrink
#: with the chunk.
#:
#: The plateau is 8k-25k wide.  Below it the walk is kernel-launch bound --
#: at 2,048 columns it is 176 launches of a kernel doing 2,048 columns of
#: work -- and above it each ``(chunk, nz)`` array leaves L2, which is the
#: same effect that made the unchunked call cost 45% more per column than
#: the plateau.  16,384 sits in the plateau, is a power of two, and holds
#: the chunk-bounded part of the workspace under 800 MiB at nz = 49.
MYNN_PBL_COLUMN_CHUNK = 16384

#: Group slots holding ``count`` arrays of one ``(chunk, nz)`` layer each.
_LAYER_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}
#: Group slots holding ``count`` arrays of one ``(chunk, nz + 1)`` face each.
_FACE_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}
#: Group slots holding ``count`` arrays of one ``(chunk,)`` column each.
_COLUMN_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {}


def _layers(slot: str, names: Sequence[str]) -> str:
    _LAYER_GROUPS[slot] = (len(names), tuple(names))
    return slot


def _faces(slot: str, names: Sequence[str]) -> str:
    _FACE_GROUPS[slot] = (len(names), tuple(names))
    return slot


def _columns(slot: str, names: Sequence[str]) -> str:
    _COLUMN_GROUPS[slot] = (len(names), tuple(names))
    return slot


# --- driver assembly (mynn_pbl_gpu.mynn_bl_driver_cuda) --------------------
SLOT_PREP = _layers("mynn_pbl_prep",
                    ("qv1", "sqw", "thl", "thetav", "qke_seed"))
SLOT_ZW = _faces("mynn_pbl_zw", ("zw",))
SLOT_SURFACE = _columns("mynn_pbl_surface",
                        ("flt", "fltv", "flq", "flqv", "flqc", "th_sfc",
                         "rmol", "zet", "pmz", "phh"))
SLOT_DELT = _columns("mynn_pbl_delt", ("delt",))
SLOT_DISS_HEAT = _layers("mynn_pbl_diss_heat", ("diss_heat",))
SLOT_EXCHANGE = _layers("mynn_pbl_exchange", ("exch_m", "exch_h"))

# --- GET_PBLH / SCALE_AWARE -----------------------------------------------
SLOT_PBLH = _columns("mynn_pbl_pblh", ("zi", "psig_bl", "psig_shcu"))

# --- mym_level2 (adjacent-level pairs) ------------------------------------
SLOT_LEVEL2_PAIRS = _layers(
    "mynn_pbl_level2_pairs",
    ("dz", "u", "v", "thl", "thetav", "qw", "ql", "vt", "vq",
     "dz_prev", "u_prev", "v_prev", "thl_prev", "thetav_prev",
     "qw_prev", "ql_prev", "vt_prev", "vq_prev"))
SLOT_LEVEL2_OUT = _layers("mynn_pbl_level2_out",
                          ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh"))

# --- mym_length -----------------------------------------------------------
SLOT_MIXLENGTH = _layers("mynn_pbl_mixlength", ("el", "qkw"))
SLOT_MIXLENGTH_WORK = _layers("mynn_pbl_mixlength_work",
                              tuple(f"w{k}" for k in range(5)))

# --- mym_turbulence -------------------------------------------------------
SLOT_LEVEL2_FULL = _layers("mynn_pbl_level2_full",
                           ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh"))
SLOT_TURBULENCE = _layers("mynn_pbl_turbulence",
                          ("dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt",
                           "pdq", "pdc"))

# --- mym_predict ----------------------------------------------------------
SLOT_PREDICT = _layers("mynn_pbl_predict", ("qke", "tsq", "qsq", "cov"))
SLOT_PREDICT_WORK = _layers("mynn_pbl_predict_work",
                            tuple(f"w{k}" for k in range(10)))

# --- mym_condensation -----------------------------------------------------
SLOT_CONDENSATION = _layers("mynn_pbl_condensation",
                            ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm"))

# --- mym_initialize -------------------------------------------------------
SLOT_INITIALIZE = _layers("mynn_pbl_initialize",
                          ("el", "qke", "tsq", "qsq", "cov", "sm", "sh"))
#: The kernel packs qkw, qtke, thetaw, elBLavg, dlu, dld, dtl, dqw, dtv,
#: gm, gh, pdk, pdt, pdq, pdc into one buffer; see
#: ``_INITIALIZE_SCRATCH_VECTORS`` in mynn_pbl_gpu.
SLOT_INITIALIZE_WORK = _layers("mynn_pbl_initialize_work",
                               tuple(f"v{k}" for k in range(15)))

# --- DMP_mf ---------------------------------------------------------------
SLOT_PLUME_LAYER = _layers("mynn_pbl_plume_layer",
                           ("edmf_a", "edmf_w", "edmf_qt", "edmf_thl",
                            "edmf_ent", "edmf_qc", "qc_bl", "cldfra_bl",
                            "vt", "vq"))
SLOT_PLUME_FACE = _faces("mynn_pbl_plume_face",
                         ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc",
                          "s_awu", "s_awv"))
SLOT_PLUME_COLUMN = _columns("mynn_pbl_plume_column",
                             ("maxwidth", "ztop", "maxmf"))
#: 8 plume vectors x 8 plumes x (nz + 1) faces per column.  The single
#: largest slot in the manifest: 12,800 bytes per column at nz = 49, which
#: is 23.4% of the 54,595 bytes per column the solver used to allocate.
SLOT_PLUME_WORK = _faces("mynn_pbl_plume_work",
                         tuple(f"p{k}" for k in range(64)))
#: 3 work vectors + 8 per-plume vectors, layer sized.
SLOT_PLUME_SCRATCH = _layers("mynn_pbl_plume_scratch",
                             tuple(f"w{k}" for k in range(11)))

# --- mynn_tendencies ------------------------------------------------------
SLOT_TENDENCY = _layers("mynn_pbl_tendency",
                        ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs",
                         "dozone", "thl"))
SLOT_TENDENCY_WORK = _layers("mynn_pbl_tendency_work",
                             ("dtz", "rhoinv", "delp", "a", "b", "c", "d",
                              "cpw", "dpw", "sqv2", "sqc2", "sqi2", "sqs2"))
SLOT_TENDENCY_FACE = _faces("mynn_pbl_tendency_face", ("khdz", "kmdz"))

# --- constant-zero feeds --------------------------------------------------
# WRF passes these to systems this lane's pinned identity switches off.  No
# kernel writes them; they must read as zero on every call, so they are
# excluded from poisoning and from arena aliasing.
SLOT_ZERO_LAYER = _layers("mynn_pbl_zero_layer", ("zero", "snow"))
SLOT_ZERO_FACE = _faces("mynn_pbl_zero_face", ("zero",))
SLOT_PLUME_ZERO_LAYER = _layers(
    "mynn_pbl_plume_zero_layer",
    ("sub_thl", "sub_sqv", "sub_u", "sub_v", "det_thl", "det_sqv",
     "det_sqc", "det_u", "det_v"))
SLOT_PLUME_ZERO_FACE = _faces(
    "mynn_pbl_plume_zero_face",
    ("s_awqke", "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa",
     "s_awqnbca"))
SLOT_TENDENCY_ZERO = _layers("mynn_pbl_tendency_zero",
                             ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"))

# --- the wrapper's column staging (mynn_pbl_runtime) ----------------------
#: ``mynnedmf_wrapper_run`` transposes ``(nz, ny, nx)`` fields into column
#: batches.  Doing that at full width allocated 27 arrays of ``ny * nx``
#: columns per step -- 1,682.3 MiB at the d04 nest, on top of the solver's
#: own working set and equally invisible to the preflight.  Staged one chunk
#: at a time these are bounded and priced like everything else.
MYNN_PBL_STAGE_LAYERS = (
    "dz", "u", "v", "w", "th", "p", "exner", "rho", "tk",
    "qv", "qc", "qi", "qs", "sqv", "sqc", "sqi", "sqs",
    "qke", "tsq", "qsq", "cov", "el", "sh", "sm",
    "qc_bl", "qi_bl", "cldfra_bl",
)
SLOT_STAGE_LAYER = _layers("mynn_pbl_stage_layer", MYNN_PBL_STAGE_LAYERS)
SLOT_STAGE_DX = _columns("mynn_pbl_stage_dx", ("dx",))
#: gpuwm has no ocean-current coupling, so WRF's UOCE/VOCE stay at their
#: Registry default of zero; one constant-zero column serves both.
SLOT_ZERO_COLUMN = _columns("mynn_pbl_zero_column", ("zero",))

#: Slots that hold zeros for the whole run.  Read before written by
#: construction, so they are correctness-critical to leave alone.
MYNN_PBL_CONSTANT_ZERO_SLOTS = frozenset((
    SLOT_ZERO_LAYER, SLOT_ZERO_FACE, SLOT_PLUME_ZERO_LAYER,
    SLOT_PLUME_ZERO_FACE, SLOT_TENDENCY_ZERO, SLOT_ZERO_COLUMN,
))

#: int32 slots.  ``ScratchArena`` is float32-only, so these are priced by
#: the registry and allocated per state rather than shared.
MYNN_PBL_INDEX_SLOTS: dict[str, tuple[str, ...]] = {
    "mynn_pbl_kpbl": ("kpbl",),
    "mynn_pbl_pblh_index": ("kzi",),
    "mynn_pbl_plume_index": ("ktop",),
}

#: One int32 word per validated array.  ``_tendency_device_arrays`` used to
#: evaluate ``bool(cp.isfinite(a).all())`` per array, which allocated a full
#: ``(ncol, nz)`` bool temporary and forced a device synchronisation for each
#: one.  The reductions now write one of these persistent words per array
#: and the whole group is read back in a single copy, so a predicate group
#: costs one synchronisation instead of one per array.
#:
#: 64 words because the widest group is ``mynn_tendencies``' 50 validated
#: inputs; ``_flag_mask`` reads a longer group a block at a time rather than
#: silently truncating it.
MYNN_PBL_FLAG_SLOTS = ("mynn_pbl_validity_flags",)
_FLAG_WORDS = 64


def mynn_pbl_scratch_shapes(chunk: int, nz: int) -> dict[str, tuple[int, ...]]:
    """Flat float32 slot shapes for one MYNN call of ``chunk`` columns.

    Flat because the holder hands out reshaped contiguous prefixes, exactly
    as :meth:`gpuwm.core.state.ScratchArena.view` does, and because the
    oracle callers of the individual leaves pass shapes the driver never
    uses (``mym_level2`` is launched on adjacent-level pairs, one element
    shorter than a layer).
    """
    chunk = int(chunk)
    nz = int(nz)
    if chunk < 1 or nz < 1:
        raise ValueError(f"MYNN scratch needs chunk >= 1 and nz >= 1, got "
                         f"{chunk} and {nz}")
    shapes: dict[str, tuple[int, ...]] = {}
    for slot, (count, _names) in _LAYER_GROUPS.items():
        shapes[slot] = (count * chunk * nz,)
    for slot, (count, _names) in _FACE_GROUPS.items():
        shapes[slot] = (count * chunk * (nz + 1),)
    for slot, (count, _names) in _COLUMN_GROUPS.items():
        shapes[slot] = (count * chunk,)
    return shapes


def mynn_pbl_index_shapes(chunk: int, nz: int) -> dict[str, tuple[int, ...]]:
    """Flat int32 slot shapes for one MYNN call of ``chunk`` columns."""
    chunk = int(chunk)
    return {slot: (len(names) * chunk,)
            for slot, names in MYNN_PBL_INDEX_SLOTS.items()}


def mynn_pbl_flag_shapes() -> dict[str, tuple[int, ...]]:
    """Flat int32 validity-flag slot shapes (independent of size)."""
    return {slot: (_FLAG_WORDS,) for slot in MYNN_PBL_FLAG_SLOTS}


def mynn_pbl_scratch_bytes(chunk: int, nz: int) -> int:
    """Total device bytes one MYNN workspace occupies."""
    total = sum(shape[0] for shape in
                mynn_pbl_scratch_shapes(chunk, nz).values()) * 4
    total += sum(shape[0] for shape in
                 mynn_pbl_index_shapes(chunk, nz).values()) * 4
    total += sum(shape[0] for shape in mynn_pbl_flag_shapes().values()) * 4
    return int(total)


#: The six A-grid tendency fields ``mynn_pbl_step`` returns.  Full ``(nz, ny,
#: nx)`` because ``couple_ysu_tendencies`` consumes whole fields; they were
#: six unpriced per-step allocations before, 423.4 MiB at the d04 nest.
MYNN_PBL_TENDENCY_FIELDS = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")


def mynn_pbl_tendency_field_shapes(nz: int, ny: int, nx: int
                                   ) -> dict[str, tuple[int, ...]]:
    """Slot shapes for the returned A-grid tendency fields."""
    return {f"mynn_pbl_out_{name}": (int(nz), int(ny), int(nx))
            for name in MYNN_PBL_TENDENCY_FIELDS}


def mynn_pbl_slot_names() -> tuple[str, ...]:
    """Every slot name this module declares, float32 then int32 then flags."""
    return (*sorted(_LAYER_GROUPS), *sorted(_FACE_GROUPS),
            *sorted(_COLUMN_GROUPS), *sorted(MYNN_PBL_INDEX_SLOTS),
            *MYNN_PBL_FLAG_SLOTS)


class MynnPblScratch:
    """Views onto the declared MYNN slots for one batch of columns.

    ``from_state`` is the runtime path: every slot comes from
    ``DomainState.scratch``, so the registry prices it and the shared arena
    can back it.  ``standalone`` is for the oracle leaves, which are called
    directly on four-column fixtures and have no domain; it allocates each
    slot once, on first use, and holds it for the life of the holder.
    """

    def __init__(self, buffers: Mapping[str, cp.ndarray],
                 *, chunk: int, nz: int, lazy: bool = False):
        self._buffers = dict(buffers)
        self._chunk = int(chunk)
        self._nz = int(nz)
        self._lazy = bool(lazy)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_state(cls, state, chunk: int, nz: int) -> "MynnPblScratch":
        """Draw every declared slot from ``DomainState.scratch``."""
        buffers = {}
        for slot, shape in mynn_pbl_scratch_shapes(chunk, nz).items():
            buffers[slot] = state.scratch(shape, slot)
        for slot, shape in mynn_pbl_index_shapes(chunk, nz).items():
            buffers[slot] = state.scratch(shape, slot, dtype=cp.int32)
        for slot, shape in mynn_pbl_flag_shapes().items():
            buffers[slot] = state.scratch(shape, slot, dtype=cp.int32)
        return cls(buffers, chunk=chunk, nz=nz)

    @classmethod
    def standalone(cls, chunk: int, nz: int) -> "MynnPblScratch":
        """A self-owned workspace for callers with no ``DomainState``.

        Slots are allocated on first request rather than up front, because
        the oracle leaves exercise one routine at a time and a four-column
        fixture has no reason to pay for the plume block.
        """
        return cls({}, chunk=chunk, nz=nz, lazy=True)

    # -- access ------------------------------------------------------------

    @property
    def chunk(self) -> int:
        return self._chunk

    @property
    def nz(self) -> int:
        return self._nz

    def _backing(self, slot: str, values: int, dtype) -> cp.ndarray:
        buf = self._buffers.get(slot)
        if buf is None:
            if not self._lazy:
                raise KeyError(
                    f"MYNN scratch slot {slot!r} was not provided; a "
                    f"state-backed workspace declares every slot up front")
            buf = cp.zeros((values,), dtype=dtype)
            self._buffers[slot] = buf
        elif buf.size < values:
            if self._lazy:
                buf = cp.zeros((values,), dtype=dtype)
                self._buffers[slot] = buf
            else:
                raise ValueError(
                    f"MYNN scratch slot {slot!r} holds {buf.size} values, "
                    f"this call needs {values}; the declared workspace is "
                    f"{self._chunk} columns at nz={self._nz}")
        if buf.dtype != np.dtype(dtype):
            raise TypeError(f"MYNN scratch slot {slot!r} is {buf.dtype}, "
                            f"requested {np.dtype(dtype)}")
        return buf

    def group(self, slot: str, names: Iterable[str], shape) -> dict:
        """One contiguous sub-array of ``shape`` per name, in order."""
        names = tuple(names)
        shape = tuple(int(extent) for extent in shape)
        unit = 1
        for extent in shape:
            unit *= extent
        buf = self._backing(slot, len(names) * unit, DTYPE)
        return {name: buf[index * unit:(index + 1) * unit].reshape(shape)
                for index, name in enumerate(names)}

    def one(self, slot: str, shape) -> cp.ndarray:
        """The first (and usually only) array in a slot."""
        shape = tuple(int(extent) for extent in shape)
        unit = 1
        for extent in shape:
            unit *= extent
        return self._backing(slot, unit, DTYPE)[:unit].reshape(shape)

    def index(self, slot: str, shape) -> cp.ndarray:
        """An int32 array from an index slot."""
        shape = tuple(int(extent) for extent in shape)
        unit = 1
        for extent in shape:
            unit *= extent
        return self._backing(slot, unit, cp.int32)[:unit].reshape(shape)

    def flags(self, slot: str = MYNN_PBL_FLAG_SLOTS[0]) -> cp.ndarray:
        """The persistent int32 validity words, zeroed for this use."""
        buf = self._backing(slot, _FLAG_WORDS, cp.int32)
        buf[...] = 0
        return buf

    # -- the write-before-read lever ---------------------------------------

    def poison(self) -> None:
        """NaN-fill every slot that a kernel is required to overwrite.

        The constant-zero feeds are deliberately skipped: nothing writes
        them and the pinned identity requires them to read as zero.  A slot
        that belongs in that set but is not listed there will surface here
        as a NaN in the forecast, which is the point.
        """
        for slot, buf in self._buffers.items():
            if slot in MYNN_PBL_CONSTANT_ZERO_SLOTS:
                continue
            if buf.dtype == np.dtype(DTYPE):
                buf.fill(DTYPE(np.nan))
            else:
                buf.fill(-2147483647)


__all__ = [
    "MYNN_PBL_COLUMN_CHUNK",
    "MYNN_PBL_CONSTANT_ZERO_SLOTS",
    "MYNN_PBL_FLAG_SLOTS",
    "MYNN_PBL_INDEX_SLOTS",
    "MYNN_PBL_STAGE_LAYERS",
    "MYNN_PBL_TENDENCY_FIELDS",
    "MynnPblScratch",
    "SLOT_STAGE_DX",
    "SLOT_STAGE_LAYER",
    "SLOT_ZERO_COLUMN",
    "mynn_pbl_tendency_field_shapes",
    "SLOT_CONDENSATION",
    "SLOT_DELT",
    "SLOT_DISS_HEAT",
    "SLOT_EXCHANGE",
    "SLOT_INITIALIZE",
    "SLOT_INITIALIZE_WORK",
    "SLOT_LEVEL2_FULL",
    "SLOT_LEVEL2_OUT",
    "SLOT_LEVEL2_PAIRS",
    "SLOT_MIXLENGTH",
    "SLOT_MIXLENGTH_WORK",
    "SLOT_PBLH",
    "SLOT_PLUME_COLUMN",
    "SLOT_PLUME_FACE",
    "SLOT_PLUME_LAYER",
    "SLOT_PLUME_SCRATCH",
    "SLOT_PLUME_WORK",
    "SLOT_PLUME_ZERO_FACE",
    "SLOT_PLUME_ZERO_LAYER",
    "SLOT_PREDICT",
    "SLOT_PREDICT_WORK",
    "SLOT_PREP",
    "SLOT_SURFACE",
    "SLOT_TENDENCY",
    "SLOT_TENDENCY_FACE",
    "SLOT_TENDENCY_WORK",
    "SLOT_TENDENCY_ZERO",
    "SLOT_TURBULENCE",
    "SLOT_ZERO_FACE",
    "SLOT_ZERO_LAYER",
    "mynn_pbl_flag_shapes",
    "mynn_pbl_index_shapes",
    "mynn_pbl_scratch_bytes",
    "mynn_pbl_scratch_shapes",
    "mynn_pbl_slot_names",
]
