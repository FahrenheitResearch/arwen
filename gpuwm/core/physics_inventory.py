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


#: The one PBL selector whose driver path (``PhysicsDriver._run_ysu``, the
#: ``1`` row of the PBL dispatch table in gpuwm/core/physics.py) creates the
#: retained ``last_ysu`` output dict.
YSU_PBL_SCHEME = 1


def physics_retains_ysu_output(cfg: RunConfig) -> bool:
    """Whether the raw YSU output dict crosses a model-step boundary.

    Positive ``bldt`` keeps the historical diagnostic object untouched.
    Only YSU creates that object: MYJ, MYNN, Shin-Hong and SASE leave
    ``last_ysu`` None (SASE's raw rates and diagnostics have their own
    canonical buffers), so pricing it for them over-counts about ten
    arrays and can refuse a run that fits at the margin.
    At ``bldt == 0`` every configured PBL call is immediately consumed by
    :meth:`PhysicsDriver._run_ysu`, so retaining the raw rates duplicates the
    coupled PBL tendencies without serving a later reader.
    """
    return bool(cfg.bl_pbl_physics == YSU_PBL_SCHEME and cfg.bldt > 0.0)


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


#: The raw PBL fields needed to repeat mass coupling after a grid move.
PBL_RAW_RATE_NAMES = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")
PBL_SHARED_FORCING = {"dtheta": "gf_rthblten", "dqv": "gf_rqvblten"}


def pbl_raw_rate_names(cfg: RunConfig) -> tuple[str, ...]:
    """Rates held only when a PBL call can be skipped after relocation."""
    if not cfg.bl_pbl_physics or cfg.bldt == 0.0:
        return ()
    names = (PBL_RAW_RATE_NAMES if cfg.mp_physics in PBL_RQI_MICROPHYSICS
             else PBL_RAW_RATE_NAMES[:-1])
    # SASE returns a physical half-level vertical acceleration too.
    # Relocation couples it onto z faces using the transplanted mass.
    return names + (("dw",) if cfg.bl_pbl_physics == SASE_PBL_SCHEME else ())


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
    # USTM is Registry.EM_COMMON:1954 state ("U* IN SIMILARITY THEORY
    # WITHOUT VCONV"), unconditional and restart-carried, and SFCLAY1D
    # writes it beside UST on every column (module_sf_sfclay.F:799-804).
    "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol", "regime",
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

#: The Eta similarity surface layer's (``sf_sfclay_physics = 2``) WRF INOUT
#: state, in the argument order of the ``myjsfc_column`` kernel that
#: gpuwm/core/myjsfc.py launches (hoisted from there, which imports cupy).
MYJ_SFCLAY_INOUT = ("ust", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc",
                    "akhs", "akms")

#: The Eta layer's pure outputs, in the same kernel's argument order.
#: ``rib`` is the field Noah reads as its bulk Richardson number, the same
#: slot the MM5 surface layers fill through ``br``
#: (gpuwm/core/physics.py::_run_noah).
MYJ_SFCLAY_OUTPUTS = ("rmol", "ct", "pblh", "rib", "chs", "chs2", "cqs2",
                      "hfx", "qfx", "lh", "flhc", "flqc", "qgh", "cpm",
                      "u10", "v10", "t2", "th2", "tshltr", "th10", "q2",
                      "qshltr", "q10", "pshltr", "u10e", "v10e")

#: Every persistent 2-D surface field ``initialize_physics`` holds for the
#: Eta layer, in allocation order: the kernel roster above plus the four
#: Registry fields its driver path reads or publishes (``ustm``, ``wspd``,
#: ``ch``, ``mixht``) and ``z0base``, the background roughness seeded from
#: the cold-start ZNT.  The allocation and the VRAM estimate
#: (:func:`gpuwm.core.preflight.physics_field_names_2d`) both read THIS
#: tuple; while the estimate restated nothing for the selector, 17 of these
#: planes were allocated and never priced.
MYJ_SFCLAY_FIELDS_2D = tuple(dict.fromkeys((
    *MYJ_SFCLAY_INOUT, *MYJ_SFCLAY_OUTPUTS,
    "ustm", "wspd", "ch", "mixht", "z0base")))

#: MYJ's carried 3-D PBL state (``bl_pbl_physics = 2``): WRF's TKE_MYJ and
#: EL_MYJ, allocated once by ``initialize_physics`` for this selector only.
#: Read by the allocation and by
#: :func:`gpuwm.core.preflight.physics_array_shapes`, so the two stay one
#: list; the estimate used to omit both, 2*nz planes per domain.
MYJ_PBL_STATE_3D = ("tke_myj", "el_myj")

#: The UW moist-turbulence PBL's (``bl_pbl_physics = 9``) persistent fields,
#: allocated by ``initialize_physics`` for that selector only and priced by
#: the VRAM estimate from these same tuples.  Interface fields carry nz+1
#: levels, as WRF's kms:kme arrays do for this scheme.
#:
#: * carried, (nz+1): WRF's EXCH_M / EXCH_H through kte+1, read back as
#:   kvm_in/kvh_in on every step after the first
#:   (module_bl_camuwpbl_driver.F:436-441, :551-556);
#: * carried, (ny, nx): the residual surface stress TAURESX2D/TAURESY2D
#:   (driver.F:465-470, :621-622);
#: * published, (nz+1): TKE_PBL, TURBTYPE3D, SMAW3D (driver.F:742-748);
#: * published, (ny, nx): TPERT2D, QPERT2D, WPERT2D (driver.F:754-756);
#: * held, (nz): the radiation step's CLDFRA, which WRF's radiation driver
#:   writes on due steps and the scheme reads on every PBL step
#:   (module_radiation_driver.F:1309-1332, module_pbl_driver.F:1940).
UWPBL_STATE_FULL = ("uw_kvm", "uw_kvh")
UWPBL_STATE_2D = ("tauresx2d", "tauresy2d")
UWPBL_DIAGNOSTICS_FULL = ("tke_pbl", "turbtype3d", "smaw3d")
UWPBL_DIAGNOSTICS_2D = ("tpert2d", "qpert2d", "wpert2d")
UWPBL_HELD_3D = ("uw_cldfra",)

#: The UW launcher's per-call output roster (gpuwm/core/uwpbl.py
#: ``uwpbl_step``), held HERE so the launcher allocates from it and the
#: preflight prices from it: one copy, nothing to drift.  Mass-level (nz)
#: float32 tendencies; the interface (nz+1) outputs are
#: :data:`UWPBL_DIAGNOSTICS_FULL`; the surface (ny, nx) float32 outputs,
#: plus the int32 ``kpbl2d`` (WRF's KPBL, one-based).
UWPBL_MASS_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                      "rqcblten", "rqiblten", "rqniblten")
UWPBL_SURFACE_OUTPUTS = ("tpert2d", "qpert2d", "wpert2d", "pblh2d")

#: Threads per block of the UW column launch.
UWPBL_BLOCK = 64

#: CAM's automatic arrays, per column, in binary64 and int32 slots per
#: interface level (nk + 1).  Every allocation in the device transcription
#: is taken at the top of its routine, unconditionally (uwpbl_eddy.cuh,
#: uwpbl_caleddy.cuh, uwpbl_zisocl.cuh, uwpbl_vdiff.cuh and the column
#: driver), so the high-water mark is a function of the level count alone.
#: MEASURED 2026-09-30 with a counting build of the same headers over every
#: column of the oracle fixtures: 109 * (nk + 1) - 74 binary64 slots and
#: 5 * (nk + 1) - 2 int32 slots at nk = 35, 44 and 61, the branch-probe
#: columns included.  The coefficients are those slopes with the negative
#: intercepts dropped, so the pool is never short; the device still
#: reports an overflow and the launcher refuses it by name.
UWPBL_R8_SLOTS_PER_LEVEL = 109
UWPBL_I4_SLOTS_PER_LEVEL = 5

#: Device bytes one launch's two pools may hold before the domain is walked
#: in column chunks.  At 50 levels one column costs 45.5 KB, so a chunk is
#: about 17,000 columns.
UWPBL_WORKSPACE_BUDGET_BYTES = 768 * 1024 * 1024


def uwpbl_workspace_slots(nk: int) -> tuple[int, int]:
    """``(r8_slots, i4_slots)`` one column of ``nk`` mass levels needs."""
    levels = int(nk) + 1
    return (UWPBL_R8_SLOTS_PER_LEVEL * levels,
            UWPBL_I4_SLOTS_PER_LEVEL * levels)


def uwpbl_chunk_columns(nk: int, ncols: int,
                        budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES
                        ) -> int:
    """Columns per UW launch so the two pools fit ``budget_bytes``."""
    r8, i4 = uwpbl_workspace_slots(nk)
    per_column = 8 * r8 + 4 * i4
    chunk = max(1, int(budget_bytes) // per_column)
    if chunk >= UWPBL_BLOCK:
        chunk = chunk // UWPBL_BLOCK * UWPBL_BLOCK
    return int(min(chunk, max(int(ncols), 1)))


def uwpbl_workspace_bytes(nk: int, ncols: int,
                          budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES
                          ) -> int:
    """Device bytes the UW launcher's pools hold for one call.

    The pools are sized to one chunk and reused across the domain's
    chunks, plus the one int32 overflow word.
    """
    r8, i4 = uwpbl_workspace_slots(nk)
    chunk = uwpbl_chunk_columns(nk, ncols, budget_bytes)
    return int(chunk * (8 * r8 + 4 * i4) + 4)


# Hoisted from gpuwm.core.microphysics (module-scope cupy) for the
# same reason as the tables above: the preflight scratch registry
# prices these snapshot slots on installs with no GPU runtime.
def ring_guard_row(mp_physics: int) -> dict[str, list[str]]:
    """The registry's ``consumers.ring_guard`` row for ``mp_physics``.

    ``state_fields`` are the 3-D state arrays the guard captures around a
    microphysics call; ``surface_slots`` the (ny, nx) accumulator and
    diagnostic slots the scheme writes.  A scheme without a row is refused
    by name: pricing a ring the guard would then capture differently is
    exactly the under-budgeting the allocation gate exists to stop.
    """

    from gpuwm.physics_registry import consumer_rows_by_selector

    row = consumer_rows_by_selector("microphysics", "ring_guard").get(
        int(mp_physics))
    if not isinstance(row, dict):
        raise ValueError(
            f"mp_physics={mp_physics} has no consumers.ring_guard row in "
            "gpuwm/physics_registry_v2.json, so its specified-zone ring "
            "cannot be priced; give the option its row in "
            "tools/build_registry.py and regenerate the registry")
    return {"state_fields": list(row["state_fields"]),
            "surface_slots": list(row["surface_slots"])}


def ring_guard_state_fields() -> tuple[str, ...]:
    """The union, over every implemented scheme, of the ring guard's rows.

    The family ``gpuwm.core.microphysics`` captures around a call, in row
    order: presence guards at the capture site keep a scheme unaffected by
    another scheme's names, so the union is what lets a scheme's registry
    moment names be captured at all.

    Hoisted here for the same reason :func:`ring_guard_row` was -- that
    module imports cupy at module scope, so on a card-free install (a CPU
    runner, a preflight-only install) the union could not be read at all,
    and the every-scheme coverage that says the captured family and the
    priced family are ONE family could not be checked anywhere.
    """

    from gpuwm.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "ring_guard")
    names: list[str] = []
    for mp in sorted(rows):
        if not isinstance(rows[mp], dict):
            continue
        for name in rows[mp]["state_fields"]:
            if name not in names:
                names.append(name)
    return tuple(names)


def spec_zone_ring_slices(ny: int, nx: int, sz: int):
    """Non-overlapping index tuples covering exactly the ring WRF's
    clipped microphysics tiles exclude.

    WRF (1-based, ide/jde staggered ends): tiles run
    ``its = ids+sz .. ide-1-sz``, ``jts = jds+sz .. jde-1-sz``
    (solve_em.F:3631-3639, :4040-4048;
    module_microphysics_driver.F:870-879).  On gpuwm's 0-based (ny, nx)
    mass grid the surviving tile is ``sz .. nx-1-sz`` x ``sz .. ny-1-sz``
    and the excluded ring is ``i < sz or i > nx-1-sz or j < sz or
    j > ny-1-sz``.  The leading Ellipsis makes each tuple apply to
    (ny, nx) and (nz, ny, nx) arrays alike.  Degenerate domains
    (``2*sz >= ny`` or ``nx`` -- WRF's clip leaves an empty tile) are
    covered without overlap.

    Hoisted from :mod:`gpuwm.core.microphysics` (module-scope cupy), which
    re-exports it: a moving nest's host-side accumulation carry
    (:mod:`gpuwm.core.physics_continuation`) reads the same ring and must
    import on an install with no GPU runtime.
    """
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    return (
        (Ellipsis, slice(0, n_lo), slice(None)),          # south rows
        (Ellipsis, slice(n_hi, ny), slice(None)),         # north rows
        (Ellipsis, slice(n_lo, n_hi), slice(0, e_lo)),    # west columns
        (Ellipsis, slice(n_lo, n_hi), slice(e_hi, nx)),   # east columns
    )


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
    # The scheme's snapshot family comes from the REGISTRY's row for it
    # (``consumers.ring_guard``, tools/build_registry.py), the same row
    # gpuwm.core.microphysics captures the union of, so the priced family
    # and the captured family cannot drift from each other.  The if-chain
    # this replaces priced the condensate and radii arm for
    # (6, 8, 9, 10, 16, 28) and not for 18, so an NSSL-2 nested run was
    # under-priced and its nine number/volume moments were left advancing
    # in the ring WRF's clipped tiles never touch.
    ring = ring_guard_row(int(cfg.mp_physics))
    fields = list(ring["state_fields"])
    # Every scheme (Kessler's rain-only fallback included) stashes due
    # reflectivity into the same persistent refl_10cm slot, and the guard
    # captures that slot unconditionally once it exists -- so the family
    # must be enumerated for every scheme or the first due-reflectivity
    # call would allocate unbudgeted mp_ring_save_refl_10cm_* buffers
    # behind the allocation gate.
    volume_slots = ["refl_10cm"]
    surface_slots = list(ring["surface_slots"])
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


# These constants mirror the kernel geometry for runtime-free memory pricing.
SHWS_SLOTS = 50
SHINHONG_BLOCK = 32
SHINHONG_TILE_BLOCKS_PER_SM = 16


def shinhong_workspace_floats(nz: int, columns: int) -> int:
    """Floats for whole blocks with 1-based levels and a top sentinel."""
    blocks = (int(columns) + SHINHONG_BLOCK - 1) // SHINHONG_BLOCK
    return blocks * SHWS_SLOTS * (int(nz) + 2) * SHINHONG_BLOCK


__all__ = [
    "spec_zone_ring_save_slots", "spec_zone_ring_slices",
    "MYJ_PBL_STATE_3D", "MYJ_SFCLAY_FIELDS_2D", "MYJ_SFCLAY_INOUT",
    "UWPBL_STATE_FULL", "UWPBL_STATE_2D", "UWPBL_DIAGNOSTICS_FULL",
    "UWPBL_DIAGNOSTICS_2D", "UWPBL_HELD_3D", "UWPBL_MASS_OUTPUTS",
    "UWPBL_SURFACE_OUTPUTS", "UWPBL_BLOCK", "UWPBL_R8_SLOTS_PER_LEVEL",
    "UWPBL_I4_SLOTS_PER_LEVEL", "UWPBL_WORKSPACE_BUDGET_BYTES",
    "uwpbl_workspace_slots", "uwpbl_chunk_columns", "uwpbl_workspace_bytes",
    "MYJ_SFCLAY_OUTPUTS",
    "MYNN_PBL_DIAGNOSTICS_2D", "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_PBL_STATE_3D", "MYNN_SURFACE_OUTPUTS",
    "PBL_RQI_MICROPHYSICS", "SFCLAY_OUTPUTS",
    "SHWS_SLOTS", "SHINHONG_BLOCK", "SHINHONG_TILE_BLOCKS_PER_SM",
    "shinhong_workspace_floats",
    "YSUWS_SLOTS", "YSU_BLOCK", "YSU_TILE_BLOCKS_PER_SM",
    "hmix_k_diag_names",
    "microphysics_scratch_slots", "physics_driver_required",
    "physics_enabled", "physics_retains_ysu_output",
    "physics_reuses_pbl_composition", "ysu_workspace_floats",
]
