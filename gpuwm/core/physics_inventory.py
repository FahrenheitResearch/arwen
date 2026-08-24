"""What a configuration's physics ALLOCATES and PUBLISHES -- no runtime.

Configuration predicates and name tables only, split out of
:mod:`gpuwm.core.physics` because that module's body imports cupy and
these answers must not cost a GPU runtime to ask.  The VRAM estimator
(:func:`gpuwm.core.preflight.physics_array_shapes`) is the consumer that
forced the split: ``gpuwm domain`` -- a command that integrates nothing
on a card -- reached ``import cupy`` through this metadata and refused
every CPU-only install, with ``--card`` declared and the answer already
in hand (measured by the 2.5.0 persona walks).

Everything here is re-exported by :mod:`gpuwm.core.physics`, whose
docstrings carry the WRF line-number authority; import from either, but
from HERE when the caller must stay importable without cupy.  The
runtime-free property is held by
``tests/test_wizard_sizing_authority.py``, which runs the wizard in an
interpreter where cupy does not resolve.
"""

from __future__ import annotations

from gpuwm.config import RunConfig, SASE_PBL_SCHEME, radiation_enabled
from gpuwm.core.wdm6_constants import WDM6_NUMBER_SPECIES

#: HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (cfg.hmix_k_diag): the (momentum,
#: scalar) history names each horizontal mixing producer publishes under.
#:
#: Named for the PRODUCER, not for the diagnostic, because that is the
#: whole point of the field.  ``km_opt = 4`` publishes WRF's own Registry
#: names for the 2-D Smagorinsky viscosities it computes; SASE publishes
#: scheme-qualified names for its governed horizontal diffusivity.  Both
#: are m2 s-1 on the mass grid and both are the coefficient of the same
#: down-gradient horizontal flux, so a run that removes one producer and
#: installs the other can be compared field to field on the channel it
#: swapped -- which is the measurement that decides whether "SASE
#: supplies the mixing the km_opt operator would otherwise apply" is
#: true, rather than leaving it as an assertion.
_HMIX_K_DIAG_NAMES: dict[str, tuple[str, str]] = {
    "smagorinsky": ("XKMH", "XKHH"),
    "sase": ("SASE_KMH", "SASE_KHH"),
}


def hmix_k_diag_names(cfg: RunConfig) -> tuple[str, ...]:
    """The horizontal-K history names this configuration can publish.

    EMPTY when the run has no horizontal mixing producer at all (the
    acknowledged ``km_opt = 0`` control).  Deliberately empty rather than
    a pair of zero fields: an absent variable cannot be misread as a
    measured zero, and "this file has no horizontal viscosity variable"
    is the strongest available statement that this run ran no horizontal
    mixing operator.
    """
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        return _HMIX_K_DIAG_NAMES["sase"]
    if cfg.km_opt == 4:
        return _HMIX_K_DIAG_NAMES["smagorinsky"]
    return ()


def physics_enabled(cfg: RunConfig) -> bool:
    """Whether any non-timesplit physics scheme is configured."""
    return bool(radiation_enabled(cfg) or cfg.sf_sfclay_physics
                or cfg.sf_surface_physics or cfg.bl_pbl_physics
                or cfg.cu_physics)


def physics_driver_required(cfg: RunConfig) -> bool:
    """Whether setup must attach a persistent :class:`PhysicsDriver`.

    Microphysics is advanced after RK3 rather than through the non-timesplit
    tendency path selected by :func:`physics_enabled`.  It still needs the
    driver for accumulated precipitation and the output-due REFL_10CM
    handoff, so an mp-only domain must receive the same persistent attachment
    as a domain with radiation, surface, PBL, or cumulus physics.
    """
    return bool(cfg.mp_physics or physics_enabled(cfg))


def physics_retains_ysu_output(cfg: RunConfig) -> bool:
    """Whether the raw YSU output dict crosses a model-step boundary.

    Positive ``bldt`` keeps the historical diagnostic object untouched.
    At ``bldt == 0`` every configured PBL call is immediately consumed by
    :meth:`PhysicsDriver._run_ysu`, so retaining the raw rates duplicates the
    coupled PBL tendencies without serving a later reader.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt > 0.0)


def physics_reuses_pbl_composition(cfg: RunConfig) -> bool:
    """Whether the composed tendency target can be the fresh PBL stack.

    This is deliberately narrower than ``stepbl == 1``: every positive-bldt
    configuration retains the historical allocation path.  With active YSU
    and literal ``bldt == 0``, ``_run_ysu`` replaces the PBL stack before
    every composition, so radiation/cumulus can be accumulated into it once
    without corrupting a value needed by the next step.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt == 0.0
                and (radiation_enabled(cfg) or cfg.cu_physics))


#: ``mp_physics`` values whose WRF Registry ``moist`` package carries QI,
#: and therefore the values for which the PBL driver returns an ``rqi``
#: tendency the run must hold.  THREE sites read it and all three must
#: agree, or the ``gpuwm check --alloc`` measurement stops covering true
#: runtime residency: physics.py's ``_pbl_optional_tendency_components``
#: (what the run allocates), ``preflight.physics_array_shapes`` (what the
#: estimate prices) and ``preflight._materialize_physics`` (what the
#: measurement constructs).  They used to be three literal tuples, and
#: mp=16 reached production having moved only two of them.
PBL_RQI_MICROPHYSICS = (6, 8, 9, 10, 16, 18, 28, 50)


def microphysics_scratch_slots(
        mp_physics: int) -> tuple[tuple[str, str], ...]:
    """Driver diagnostic component -> canonical persistent scratch slot.

    ``mp_physics=28`` shares mp=8's row, and the authority for that is WRF's
    own driver arm rather than mp=8's spelling: ``CASE (THOMPSONAERO)``
    (``phys/module_microphysics_driver.F:1029``) calls ``mp_gt_driver`` with
    RAINNC (:1085), RAINNCV (:1086), SNOWNC (:1087), SNOWNCV (:1088),
    GRAUPELNC (:1089), GRAUPELNCV (:1090) and SR (:1091) and with NO hail
    argument -- the identical seven ``CASE (THOMPSON)`` binds at :1253-:1259.
    gpuwm's aerosol adapter writes the same seven canonical scratch slots
    (``gpuwm/core/microphysics_aerosol.py:263-269``), so the driver aliases
    them here instead of allocating a private zero-filled copy set.
    """
    if mp_physics == 1:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_kessler_sr"))
    if mp_physics in (6, 8, 10, 16, 28):
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"))
    if mp_physics in (9, 18):
        # Milbrandt-Yau shares NSSL's nine-slot row because its WRF driver
        # arm binds the same nine: RAINNC/RAINNCV (:1868-1869),
        # SNOWNC/SNOWNCV (:1870-1871), HAILNC/HAILNCV (:1872-1873),
        # GRAUPELNC/GRAUPELNCV (:1874-1875) and SR (:1876) in
        # module_microphysics_driver.F's CASE (MILBRANDT2MOM).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"),
                ("hailnc", "mp_hailnc"),
                ("hailncv", "mp_hailncv"),
                ("sr", "mp_sr"))
    if mp_physics == 50:
        # P3 one-category.  FIVE slots, not mp=6/8's seven: the driver arm
        # ``CASE (P3_1CATEGORY)`` binds RAINNC/RAINNCV/SR/SNOWNC/SNOWNCV and
        # NO graupel argument (module_microphysics_driver.F:1590-1595),
        # because P3 has a single ice category whose rime fraction spans
        # what other schemes split into snow, graupel and hail.  Listing
        # graupel here would allocate a canonical accumulator that stayed
        # zero forever and let output claim a graupel field P3 never has.
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"))
    return ()


# ---------------------------------------------------------------------------
# Scheme output/state NAME tables, hoisted from their runtime modules
# (sfclay, mynn_sfclay, mynn_pbl_runtime -- each imports cupy at module
# scope) so the estimator can price their allocations without a GPU
# runtime.  Each runtime module re-imports its table, so there is still
# exactly one spelling of every inventory.
# ---------------------------------------------------------------------------

SFCLAY_OUTPUTS = (
    "znt", "ust", "mol", "hfx", "qfx", "qsfc", "zol", "regime",
    "psim", "psih", "fm", "fh", "lh", "u10", "v10", "th2", "t2",
    "q2", "chs", "chs2", "cqs2", "flhc", "flqc", "qgh", "rmol",
    "wspd", "br", "gz1oz0", "cpm", "ck", "cka", "cd", "cda",
)

MYNN_SURFACE_OUTPUTS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm",
    # module_sf_mynn.F:436 declares ZNT INTENT(INOUT); whichever leaf the
    # isftcflx arm selects (:635/:641/:643/:647) rewrites it on every water
    # column and the new value must persist into the next step.
    "znt",
)

#: WRF Registry ``mynnscheme`` prognostic/carried 3-D state, in the spelling
#: ``module_pbl_driver.F`` binds.  ``el_pbl``/``sh3d``/``sm3d`` keep WRF's
#: names rather than the solver's ``el``/``sh``/``sm`` so the restart
#: manifest and wrfout read the same identifiers WRF writes.
MYNN_PBL_STATE_3D = (
    "qke", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d",
    "qc_bl", "qi_bl", "cldfra_bl",
)

#: Per-column plume diagnostics the wrapper exports (``:1698-1699``).
MYNN_PBL_DIAGNOSTICS_2D = ("maxwidth", "maxmf", "ztop_plume")

#: Integer per-column diagnostic; kept apart because it is int32.
MYNN_PBL_DIAGNOSTICS_INT_2D = ("ktop_plume",)


# Hoisted from gpuwm.core.microphysics (module-scope cupy) for the
# same reason as the tables above: the preflight scratch registry
# prices these snapshot slots on installs with no GPU runtime.
def spec_zone_ring_save_slots(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Preflight-registry helper: every ``mp_ring_save_*`` snapshot slot the
    ring guard creates for this config, with its exact shape.

    Mirrors :func:`_capture_spec_zone_ring` and
    :func:`spec_zone_ring_slices`: one slot per (captured array, non-empty
    ring edge).  Empty when the guard is off (periodic/open, sz = 0, or no
    microphysics).  Consumed by
    ``gpuwm.core.preflight.scratch_slot_registry`` so the completeness and
    allocation gates see the family with true sizes.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return {}
    sz = int(cfg.spec_zone)
    if sz <= 0 or cfg.mp_physics == 0 or not cfg.moist:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    fields = ["thp", "qv", "qc", "qr"]
    if cfg.mp_physics in (6, 8, 9, 10, 16, 28):
        fields += ["qi", "qs", "qg"]
    if cfg.mp_physics == 8:
        fields += ["nr", "ni"]
    if cfg.mp_physics == 9:
        # Milbrandt-Yau: hail mass beside graupel plus all six numbers.
        fields += ["qh", "nc", "nr", "ni", "ns", "ng", "nh"]
    if cfg.mp_physics == 16:
        # WDM6's three transported moments (gpuwm/core/wdm6.py).  nn is in
        # the set because the scheme WRITES it -- rain that fully evaporates
        # returns its number to the CCN reservoir (module_mp_wdm6.F:
        # 1249-1252) and cloud that fully evaporates does the same
        # (:1990-1994) -- so an unguarded ring would keep the change.
        fields += list(WDM6_NUMBER_SPECIES)
    if cfg.mp_physics == 10:
        fields += ["nc", "nr", "ni", "ns", "ng"]
    if cfg.mp_physics == 28:
        # The exact mp=28 moment set (gpuwm/core/state.py's mp==28 arm and
        # gpuwm/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES).  nwfa2d/nifa2d
        # are NOT here: WRF passes them INTENT(IN) and no microphysics
        # kernel writes them, so the guard has nothing to restore.
        fields += ["nr", "ni", "nc", "nwfa", "nifa"]
    if cfg.mp_physics == 50:
        # P3 one-category: one ice mass with its rime mass/volume, two
        # number moments, and the two previous-step carriers.  No qs/qg
        # and no effs -- P3 does not have those species (state.py mp==50).
        fields += ["qi", "qir", "qib", "ni", "nr", "th_old", "qv_old",
                   "effc", "effi"]
    if cfg.mp_physics in (6, 8, 9, 10, 16, 28):
        fields += ["effc", "effi", "effs"]
    if cfg.mp_physics == 10:
        fields += ["effr"]
    # Every scheme (Kessler's rain-only fallback included) stashes due
    # reflectivity into the same persistent refl_10cm slot, and the guard
    # captures that slot unconditionally once it exists -- so the family
    # must be enumerated for every scheme or the first due-reflectivity
    # call would allocate unbudgeted mp_ring_save_refl_10cm_* buffers
    # behind the allocation gate.
    volume_slots = ["refl_10cm"]
    if cfg.mp_physics == 1:
        surface_slots = ["mp_rainnc", "mp_rainncv", "mp_kessler_sr"]
    elif cfg.mp_physics == 50:
        # P3's wrapper writes RAINNC/RAINNCV/SNOWNC/SNOWNCV/SR and nothing
        # else (module_mp_p3.F:894-898): there is no graupel category and
        # the driver arm passes no GRAUPELNC (:1590-1595).
        surface_slots = ["mp_rainnc", "mp_rainncv", "mp_snownc",
                         "mp_snowncv", "mp_sr"]
    else:
        surface_slots = ["mp_rainnc", "mp_rainncv", "mp_snownc",
                         "mp_snowncv", "mp_graupelnc", "mp_graupelncv",
                         "mp_sr"]
        if cfg.mp_physics in (9, 18):
            # The two hail-bearing schemes.  Both bind HAILNC/HAILNCV in
            # WRF's driver (module_microphysics_driver.F:1841-1842 for
            # mp=9), both are written by clipped tiles only, and both must
            # therefore be captured -- mp=18's pair was previously outside
            # the guard's slot family, so its ring HAILNC accumulated where
            # WRF's is exactly zero.
            surface_slots += ["mp_hailnc", "mp_hailncv"]
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    edge_dims = ((n_lo, nx), (ny - n_hi, nx),
                 (n_hi - n_lo, e_lo), (n_hi - n_lo, nx - e_hi))
    slots: dict[str, tuple[int, ...]] = {}
    for key in fields + volume_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (nz, rows, cols)
    for key in surface_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (rows, cols)
    return slots


# ---------------------------------------------------------------------------
# The YSU per-thread column workspace, priced without a runtime
# ---------------------------------------------------------------------------
# These live HERE and not in :mod:`gpuwm.core.ysu` for the reason this
# module exists: ysu.py imports cupy at module scope, and the VRAM
# estimator prices the YSU workspace on installs with no GPU runtime
# (``gpuwm domain --card`` on a CuPy-less box reached ``import cupy``
# through exactly this pricing and refused, caught by the publish test
# job's replay before the 2.5.3 tag).  ysu.py re-exports them, so the
# launcher and the kernel-pin test keep their one authority.
#
# YSUWS_SLOTS must match ysu.cu's YSUWS_SLOTS / YSUWS_LANES;
# tests/test_ysu_workspace.py re-derives them from the .cu source and
# fails if either side moves alone.
YSUWS_SLOTS = 18

#: Launch block, and the tile's granularity.  ysu.cu indexes the
#: workspace by the thread's lane within its block, so a tile is always a
#: whole number of blocks.  Pinned against the kernel's ``YSUWS_LANES``.
YSU_BLOCK = 32

#: Blocks per SM the tile is sized for.  MEASURED, not assumed -- see
#: docs/kernel_local_memory_bounds.md for the sweep this came from.  The
#: device query in :func:`gpuwm.core.ysu.ysu_tile_columns` only ever
#: lowers it.
YSU_TILE_BLOCKS_PER_SM = 16


def ysu_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: ysu.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so
    the unit of allocation is one block's region, not one column's.

    The per-slot extent is ``nz + 1`` rather than the kernel's compile-time
    ``KMAX``: ``zq`` is the one array indexed at ``nz``, and unlike the
    compile-time frame this is allocated when ``nz`` is known.  A 49-level
    run therefore holds 50 levels of arrays where the frame had to hold
    128.
    """
    blocks = (int(columns) + YSU_BLOCK - 1) // YSU_BLOCK
    return blocks * YSUWS_SLOTS * (int(nz) + 1) * YSU_BLOCK


__all__ = [
    "spec_zone_ring_save_slots",
    "MYNN_PBL_DIAGNOSTICS_2D", "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_PBL_STATE_3D", "MYNN_SURFACE_OUTPUTS",
    "PBL_RQI_MICROPHYSICS", "SFCLAY_OUTPUTS",
    "YSUWS_SLOTS", "YSU_BLOCK", "YSU_TILE_BLOCKS_PER_SM",
    "hmix_k_diag_names",
    "microphysics_scratch_slots", "physics_driver_required",
    "physics_enabled", "physics_retains_ysu_output",
    "physics_reuses_pbl_composition", "ysu_workspace_floats",
]
