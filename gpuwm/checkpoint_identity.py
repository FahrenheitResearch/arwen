"""The name a checkpoint gives every scheme, and the gate that asks first.

A restart file records the identity of every scheme that integrated it, so
a resume onto a different implementation fails BEFORE it restores rather
than continuing a trajectory that is not the one it claims.  The tables
below are those identities, and the gate below them is the question
``validate_run_config`` asks at plan review: can the checkpoint writer name
everything this configuration selected?

WHY THEY LIVE IN A TOP-LEVEL MODULE rather than beside the writer in
:mod:`gpuwm.io.restart`, which is where they were written and where every
reader in the tree still spells them.  The gate is called from
:func:`gpuwm.config.validate_run_config` -- every door's plan review --
and ``gpuwm/config.py`` ships in distributions that stage no forecast
executor and therefore no ``gpuwm/io`` at all: the standalone RW-WPS
preprocessing project stages ``gpuwm/*.py`` and an explicit handful of
``gpuwm/io`` modules, and ``restart.py`` is deliberately not one of them
(it reaches the CUDA side).  A gate imported out of ``gpuwm.io.restart``
is therefore unresolvable in a preparation-only install -- the plan review
every front door runs would raise ``ModuleNotFoundError`` on a
configuration it should simply have accepted.

Answering that with an exception in the packaging tables would have left
the same import waiting to be reached at run time.  These are pure
configuration facts -- integers, strings and dicts -- so the module that
holds them imports nothing from ``gpuwm.io``, nothing from CuPy, and
resolves its two configuration helpers inside the one function that needs
them.  It stages, and imports, wherever ``gpuwm/config.py`` does.

:mod:`gpuwm.io.restart` imports every name here and re-exports it under
the spelling the tree already uses, so there is ONE copy of each table and
``gpuwm.io.restart.MICROPHYSICS_ALGORITHM_IDENTITIES`` is that copy.
"""


from __future__ import annotations


def drop_default_diffusion_selectors(values: dict) -> None:
    """Preserve pre-selector identity bytes for the unchanged metric form."""
    if values.get("diff_opt", 2) == 2 and values.get("mix_full_fields", True) is True:
        values.pop("diff_opt", None)
        values.pop("mix_full_fields", None)


def drop_default_spp_selectors(values: dict) -> None:
    """Keep newly added SPP selectors absent when disabled.

    spp_lsm predates the SPP consumer port and must keep its existing echo.
    """
    for name in ("spp_conv", "spp_pbl"):
        if values.get(name, 0) == 0:
            values.pop(name, None)

#: THE table of output-only RunConfig switches: each one decides what a
#: forecast WRITES, never what it integrates, so it may differ between
#: the run that wrote a checkpoint and the run that resumes it, and
#: between the run that prepared a cache and the run that reads it.
#: Every place that asks "does this switch change the model" reads this
#: one set: the member restart walk and the configuration digest in
#: :mod:`gpuwm.io.restart` (re-exported there under this name), the tree
#: restart identity comparison beside them, and the prepared-cache
#: identity in :mod:`gpuwm.ingest.prepared_cache`.  Two copies of this
#: list drifted once already: the prepared cache knew only the first
#: member, so switching any of the other three on refused an unchanged
#: cache and forced a second preparation of identical arrays.
#:
#: Membership is a proof, not a label: the switch reads model state and
#: writes only its own diagnostic buffers.
#:
#: * ``nwp_diagnostics`` -- inertness pinned by tests/test_uh_lifecycle.py;
#:   its accumulator payloads are tolerant in both directions (missing in
#:   the file restores as zeros with a note; present under a
#:   diagnostics-off resume is dropped with a note).
#: * ``tke_budget`` -- the accumulator writes only its own scratch
#:   (tests/test_tke_budget.py).
#: * ``sase_flux_diag`` -- four history buffers filled from arrays the
#:   SASE step already holds (tests/test_sase_gpu.py, every prognostic
#:   byte-identical with the switch on).
#: * ``hmix_k_diag`` -- two history buffers copied from the eddy
#:   viscosity the run's own mixing producer computed; nothing reads them
#:   back (tests/test_sase_gpu.py, every prognostic byte-identical with
#:   the switch on, under both producers).
#:
#: A newly switched-on diagnostic that a positive PBL cadence carries
#: between steps restores as zeros, its cold value, and is refilled by
#: the next due step.  The physics selectors beside them
#: (``sase_moist_n2``, ``sase_stable_dissipation``,
#: ``sase_additive_dissipation``) move the trajectory and are NOT here.
CONFIG_DIAGNOSTIC_FIELDS = frozenset(
    {"nwp_diagnostics", "tke_budget", "sase_flux_diag", "hmix_k_diag",
     "surface_energy_diag"})

#: Versioned semantic identities.  These are deliberately explicit instead
#: of inferred from scheme numbers: a trajectory-changing implementation or
#: policy change must advance its tag, causing an incompatible restart to
#: fail before restore.  Asset bytes and resolved per-run values are bound
#: by the writer in :mod:`gpuwm.io.restart`.
#:
#: WHAT DOES NOT GET A ROW, so the next reader does not add one and call
#: it completeness.  A row exists for an input CONFIG CANNOT PROVE: a
#: resolved callable, packaged asset bytes, a resolved cadence, a policy
#: chosen at setup.  Every selector config DOES prove is already bound,
#: twice: the writer's ``configuration_sha256`` hashes the whole RunConfig
#: minus the run-length and diagnostic fields, and the resume path
#: compares every field by name before it compares that hash -- so a
#: closure change refuses with the field named, not with a digest
#: mismatch.  ``km_opt`` and its constants (c_k, c_s, khdif, kvdif,
#: mix_isotropic, mix_full_fields, diff_opt, diff_6th_opt) are the worked
#: example: none is excluded from the fingerprint, none has a packaged
#: asset or a resolved callable, and giving the turbulence closure a table
#: here would add a SIXTH authority over the five registry rows that
#: already describe it -- the drift surface these tables exist to avoid --
#: while invalidating every v2 checkpoint on disk, since any added key
#: moves the physics-setup fingerprint.  The residual a row would close
#: (same selector, differently transcribed kernels) is the whole dycore's,
#: not km_opt's, and it is closed by ONE string,
#: :data:`DYCORE_MIXING_ALGORITHM_IDENTITY` below, not a per-option table.
MICROPHYSICS_ALGORITHM_IDENTITIES = {
    0: "disabled",
    # "real-clamp-product" (2.8.8) on every row whose scheme finishes
    # through gpuwm.core.microphysics.moist_physics_finish (mp=1, 6, 9, 10,
    # 16, 18, 28, 50): the finish now clamps the theta increment at WRF's
    # REAL product float32(mp_tend_lim) * float32(dt) rather than the double
    # product rounded once, which classic Thompson (mp=8) already did.  For
    # some mp_tend_lim, dt pairs (HRRR's 0.07 K/s at dt = 12 s) the two
    # bounds differ by one float32 unit, so every clamped cell's thp and
    # h_diabatic move, and a checkpoint written before the change may not
    # continue under it.
    1: "kessler-warm-rain-v2-real-clamp-product",
    6: "wsm6-single-moment-six-class-wrf-v4.6.1-v2-real-clamp-product",
    # v5 (2.8.8): rain collecting graupel reads the one slab WRF builds
    # (thompson_racg_index; WRF v4.6.1 subscripts it at idx_bg1=5, a
    # declared divergence), and the finish clamps at the REAL product
    # mp_tend_lim*dt.  Both move mp=8 trajectories, so a v4 checkpoint is
    # refused before restore instead of resuming onto them.
    8: ("classic-thompson-wrf-v4.6.1-v5-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity-rain-density-condensation-history-"
        "one-slab-rain-graupel-real-clamp-product"),
    # Milbrandt-Yau two-moment (WRF v4.6.1 MILBRANDT2MOM).  Named at the
    # mp=8/28/50 granularity -- the trajectory-defining configuration,
    # not the scheme name --
    # because that is what makes an incompatible resume fail BEFORE
    # restore.  "six-category-2mom" is the change everything else follows
    # from: hail is a category of its own beside graupel and every one of
    # the six carries a number moment, which the WRF driver binds as
    # qnc/qnr/qni/qns/qng/qnh (module_microphysics_driver.F:1857-1862) and
    # gpuwm transports as MY2_SPECIES (gpuwm/core/moist.py).  The four
    # tokens after it are the switches the WRF driver HARD-CODES, so they
    # are part of this port's identity rather than of its configuration
    # (each is pinned with its line in
    # gpuwm.config.MILBRANDT2_FIXED_IDENTITY): "ccntype2" is the
    # continental CCN spectrum (module_mp_milbrandt2mom.F:3615),
    # "meyers-contact-nucl" is prim_ice_nucl=1 (:1175), "nonspherical-snow"
    # is snow_spherical=.false. (:1174), and "full-sedimentation" is the
    # precipDiag/sedi/warmphase/autoconv/icephase/snow block all left on
    # (:3618-3623).  A build that flipped any of them would integrate a
    # different trajectory while staying finite, so it advances this tag
    # rather than resuming onto it.
    9: ("milbrandt-yau-wrf-v4.6.1-v2-six-category-2mom-ccntype2-"
        "meyers-contact-nucl-nonspherical-snow-full-sedimentation-"
        "real-clamp-product"),
    # Range-safe freezing, retained cleanup vapor and in-range number
    # preservation change subsequent tendencies even from finite inputs.
    10: ("morrison-two-moment-v4-kf-number-seeding-finite-freezing-"
         "final-vapor-in-range-number-real-clamp-product"),
    # WDM6 (WRF v4.6.1 WDM6SCHEME, Registry/Registry.EM_COMMON:3031).  Named
    # at the mp=8/28 granularity -- the trajectory-defining pieces, not the
    # scheme name.  "prognostic-nc-nr-ccn" is the change everything else
    # follows from (module_mp_wdm6.F carries qnn/qnc/qnr as scalars);
    # "gamma-mu1-rain" names the rain PSD whose intercept is diagnosed from
    # nr rather than fixed (:2251-2261); "ccn-activation" names the
    # supersaturation activation that moves mass and number out of the
    # reservoir (:1951-1969); "xland-autoconversion" names the per-column
    # maritime/continental threshold (:607-614), which makes the LAND MASK
    # part of this scheme's trajectory identity in a way no other gpuwm
    # microphysics has been; "ccn-conc-init" records that the CCN reservoir
    # starts from the namelist constant fill (:220-227) rather than an
    # ingested aerosol field, so a future ingest must advance this tag
    # instead of silently resuming onto it.  v4 (A144): "zero-rate-rain-no-
    # evaporation" names the rain condensation cap that no longer turns a
    # zero rate (rain with no number) into evaporation, which changes the
    # vapour, ice and theta trajectory wherever such rain meets
    # subsaturated air (docs/wdm6_oracle_known_deltas.md, section 6).
    16: ("wdm6-double-moment-warm-rain-wrf-v4.6.1-v5-prognostic-nc-nr-ccn-"
         "gamma-mu1-rain-ccn-activation-xland-autoconversion-ccn-conc-init-"
         "conservative-rain-interface-flux-bounded-transport-time-"
         "zero-rate-rain-no-evaporation-real-clamp-product"),
    18: ("nssl-two-moment-state-transport-v2-process-boundary-fail-loud-"
         "real-clamp-product"),
    # Thompson AEROSOL-AWARE (WRF v4.6.1 THOMPSONAERO,
    # Registry/Registry.EM_COMMON:3036).  Named at the granularity the mp=8
    # row uses -- the trajectory-defining pieces, not the scheme name --
    # because that is what makes an incompatible resume fail BEFORE restore.
    # "prognostic-nc" is the change everything else follows from
    # (module_mp_thompson.F:1795-1812 freezes nc1d at entry and :3972-4021
    # applies the single terminal ncten/nwfaten/nifaten clamp); "nwfa-nifa"
    # names the two transported aerosol tracers; "ccn-activate-table" names
    # the tnccn_act asset the activation reads
    # (:5102-5108); "demott-koop" names the ice-nucleation pair that replaces
    # classic Cooper (iceDeMott called at :2574/:2623, iceKoop at :2637;
    # the functions themselves at :5447 and :5521); "scavenging" names the
    # six aerosol wet-removal rates; "surface-emission" names the unclamped
    # nwfa2d/nifa2d injection mp_gt_driver applies AFTER the terminal clamp
    # (:1310-1327), which is a real ordering choice a reimplementation could
    # get wrong while leaving every bound intact.  "synthetic-aerosol-init"
    # records that this build's aerosol profile comes from thompson_init's
    # fill (:493-551) and not from a WIF metgrid stream: a future
    # wif_input_opt ingest is a DIFFERENT initial condition and must advance
    # this tag rather than silently resume onto it.
    # v2 (2.8.8): "wrf-arithmetic" names the generation graded word for
    # word on the 157-column WRF oracle.  Every EXP, LOG, LOG10 and ** takes
    # WOOF's own libm words (kernels/thompson_aerosol_libm.cuh), the cold
    # and warm process rates follow WRF's operation order, snow and graupel
    # accumulation, fallout and final latent heating take WRF's terms, and
    # a level at exactly 273.15 K no longer takes the wet-bulb melting
    # branch.  "rain-graupel-slab1" names the read of rain collecting
    # graupel from the one graupel-density slab WRF allocates, where 2.8.7
    # reproduced WRF's out-of-bounds index 5.  Each moves the moments,
    # theta and aerosol from finite inputs in both thompson_version
    # generations (they share these kernels), so a 2.8.7 (v1) checkpoint
    # may not continue under v2.  "real-clamp-product" names the finish
    # clamp at WRF's REAL product (above mp=1); it joins the v2 string
    # rather than opening a v3 because v2 has not shipped, and the string
    # still changes, so a checkpoint an earlier 2.8.8 build wrote under the
    # double-product clamp is refused like a 2.8.7 one.
    28: ("thompson-aerosol-aware-wrf-v4.6.1-v2-prognostic-nc-nwfa-nifa-"
         "ccn-activate-table-demott-koop-scavenging-surface-emission-"
         "synthetic-aerosol-init-wrf-arithmetic-rain-graupel-slab1-"
         "real-clamp-product"),
    # P3 (WRF v4.6.1 P3_1CATEGORY, Registry.EM_COMMON:3038).  Named at the
    # granularity the mp=8/28 rows use -- the trajectory-defining
    # configuration, not the scheme name -- because that is what makes an
    # incompatible resume fail BEFORE restore.  "1cat" and "2mom-ice" name
    # the nCat=1 / log_3momentIce=.false. build (module_mp_p3.F:1043-1050);
    # "specified-nc" names log_predictNc=.false., which is the difference
    # between this row and the unported mp=51; "diagnosed-ssat" names
    # log_predictSsat=.false., the branch that makes th_old/qv_old the
    # cross-step carriers this restart serializes (:2325-2337); "rime-mass-
    # volume-transported" records that qir/qib advect with qi
    # (gpuwm/core/moist.py::P3_SPECIES) -- a build that stopped
    # transporting them would integrate a DIFFERENT trajectory while
    # staying finite, which is exactly the silent resume this string is
    # here to refuse.
    50: ("p3-one-category-wrf-v4.6.1-v2-2mom-ice-specified-nc-"
         "diagnosed-ssat-rime-mass-volume-transported-real-clamp-product"),
}
from gpuwm.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
MICROPHYSICS_ALGORITHM_IDENTITIES.update({
    _s.mp_id: _s.algorithm_identity for _s in _NAMED_MP_SCHEMES.values()})

SURFACE_LAYER_ALGORITHM_IDENTITIES = {
    0: "disabled",
    # v2 (2.8.8): the sfclay unit compiles without FMA contraction and
    # keeps denormal values instead of flushing them to zero
    # (gpuwm.core.kernels.module_options) on WOOF's own float32
    # log/exp/pow/atan; with isfflx = 0 it keeps the
    # incoming CHS, CHS2 and CQS2 as WRF does instead of recomputing them;
    # and an overflowing scalar-roughness exponential stops before FH goes
    # infinite.  Each changes the exchange coefficients and fluxes the next
    # step reads, so a 2.8.7 (v1) checkpoint may not continue under v2.
    1: "revised-mm5-surface-layer-v2-no-fma-denormals-kept-flux-off-"
       "keeps-chs-finite-scalar-roughness",
    # The Eta similarity surface layer.  The identity binds the WRF version
    # whose byte-frozen module_sf_myjsfc.F the port transcribes AND the
    # similarity tables it interpolates: MYJSFCINIT builds PSIM/PSIH by
    # accumulating ZETA in float32, so a different table construction is a
    # different scheme even at the same WRF version, and a checkpoint may
    # not resume across one.
    # v2 (2.8.8): that is what happened.  The tables are now built with
    # WOOF's own float32 log/atan/exp (gpuwm/core/myjsfc_tables.py), which
    # moved 2,277 of 10,001 PSIM entries by up to 16 ULP and 1,382 PSIH
    # entries, and the column compiles without FMA contraction or a flush
    # of denormals to zero, on WOOF's libm words.  A 2.8.7 (v1) checkpoint
    # may not continue under v2.
    2: "eta-similarity-surface-layer-wrf-v4.6.1-v2-myjsfcinit-tables-"
       "woof-libm-no-fma-denormals-kept",
    # v2 (2.8.8): no FMA contraction, denormals kept rather than flushed to
    # zero, WOOF's libm words (kernels/mynn_libm.cuh), and the first-step
    # seed through the same unit.  It changes UST, MOL, QSFC and the fluxes
    # the next step reads, so a 2.8.7 (v1) checkpoint may not continue
    # under v2.
    5: "mynn-surface-layer-wrf-v4.6.1-v2-no-fma-denormals-kept",
    # v2 (2.8.8): the classic MM5 column (kernels/sfclay_classic.cuh) is in
    # the sfclay unit and takes its arithmetic (no FMA, denormals kept), so
    # a 2.8.7 (v1) checkpoint may not continue under v2.
    91: "classic-mm5-surface-layer-v2-no-fma-denormals-kept",
}
#: WRF v4.7.1 lsm_mosaic; bound only for the enabled tile path.
NOAH_MOSAIC_ALGORITHM_IDENTITY = "noah-mosaic-wrf-v4.7.1-v1"

LAND_SURFACE_ALGORITHM_IDENTITIES = {
    0: "disabled",
    2: "noah-lsm-v2-post-sflx-chs2-source-water-lake-skin",
    # v2 (2.8.5): a LAKEMASK column is bypassed only when the lake model is
    # selected (module_sf_ruclsm.F:824, lakemodel==1 .and. lakemask==1).
    # v1 bypassed every LAKEMASK column, so with sf_lake_physics = 0 a lake
    # column was never advanced; it now runs RUC's water branch.  That
    # changes the surface state and fluxes of every RUC run with lake
    # cells, so a v1 checkpoint may not continue under v2.
    # v3 (2.8.5): SOILPROP's soil-water diffusivity and conductivity take
    # the WRF v4.0-4.5 normalisation by default (ruc_soilprop = "wrf_45",
    # gpuwm/core/ruc_tier.py); v2 ran the v4.6.1 form over total porosity,
    # which moves 2.5 to 8 times more water up into a dry top soil level.
    # That changes the soil water and surface fluxes of every RUC run, so
    # a v2 checkpoint may not continue under v3.
    # v4: the post-SFCTMP irrigation takes the WRF v4.0-4.5 crop-fraction-
    # scaled floor by default (ruc_irrigation = "wrf_45",
    # gpuwm/core/ruc_mosaic.py), whatever mosaic_lu says; v3 ran the v4.6.1
    # relaxation under mosaic_lu = 1 and nothing without it.  That changes
    # the root-zone soil water of every RUC run with cropland, so a v3
    # checkpoint may not continue under v4.
    # v5: the snow scheme takes the WRF v4.0-4.5 form the operational
    # RAP/HRRR branch carries by default (ruc_snow = "wrf_45",
    # gpuwm/core/ruc_tier.py); v4 ran the v4.6.1 snow conductivity, cover,
    # melt and albedo.  That changes the snow pack and the surface over it
    # in every RUC run with snow, so a v4 checkpoint may not continue under
    # v5.
    # v6 restores generic irrigation and snow to wrf_461. Unpublished v4/v5
    # headers omitted wrf_45, so interpreting those missing selectors as the
    # restored defaults would change their continuing trajectory. The new
    # identity refuses those checkpoints before restoring live forecast arrays.
    # v7 (2.8.8): every EXP, LOG, LOG10, TANH and REAL**REAL of the column
    # takes WOOF's float32 words and the soil-resistance COS glibc's cosf
    # (kernels/ruc.cu), where v6 rounded float64 stand-ins once; the
    # sea-ice blend and skin reset run only under fractional_seaice = 1, as
    # in WRF (kernels/ruc_fused_driver.cuh), where v6 ran them on every ice
    # cell; and SFCEVP accumulates QFX*dt once, not WRF's duplicated twice.
    # The first two change soil water, soil ice, fluxes and the ice-cell
    # surface every step, so a 2.8.7 (v6) checkpoint may not continue
    # under v7.  (The 2.8.8 RUC initialization carry and the real.exe soil
    # moisture floor act at cold start only and need no identity.)
    3: "ruc-lsm-wrf-v4.6.1-v7-woof-libm-fractional-seaice-gate",
    4: "noahmp-lsm-wrf-v4.6.1-v1",
}
PBL_ALGORITHM_IDENTITIES = {
    0: "disabled",
    # v2 (2.8.8): YSU compiles without FMA contraction on WOOF's float32
    # pow/exp (gfk_pow/gfk_exp, kernels/ysu.cu), reads theta as WRF's
    # driver hands it (th*pi/pi), makes the bulk Richardson sign test in
    # double, and takes WRF's surface-drag form on every topo_wind = 0
    # column.  Each moves the exchange coefficients and tendencies, so a
    # 2.8.7 (v1) checkpoint may not continue under v2.
    1: "ysu-wrf-v4.6.1-v2-no-fma-woof-libm-driver-theta-double-ri-"
       "wrf-surface-drag",
    # The exact driver changes the former rounded-length generation's
    # diffusivities and tendencies in both stock and fork forms. A checkpoint
    # from that generation cannot promise the same continuation here.
    # v3 was opened in 2.8.8 and has never shipped, so it also names the
    # rest of that release's MYNN generation: the gsd_41 fork's driver
    # arithmetic and length-1 formulation, and the moisture variance that
    # now keeps denormals rather than flushing them (kernels/mynn_pbl.cu).
    # Every 2.8.7 (v2) checkpoint is refused by this one bump; a v4 would
    # refuse nothing more that a release wrote.
    5: "mynn-edmf-pbl-wrf-v4.6.1-v3-exact-driver",
    # Adding a scheme means adding its row, not relaxing the check.  The
    # identity binds the WRF version whose byte-frozen module_bl_shinhong.F
    # the certified CPU authority transcribes (max ULP 0, both arms); a
    # future re-transcription against a different WRF advances the suffix
    # rather than silently resuming onto this one.
    # MYJ carries genuinely prognostic state -- TKE_MYJ is read as 2*TKE at
    # the top of every call and rewritten at the bottom, and the Eta surface
    # layer's PBLH scan reads it too -- so a resume that dropped it would
    # continue a different boundary layer while staying finite.  The
    # identity binds the WRF version the port transcribes; a
    # re-transcription against another WRF advances the suffix rather than
    # silently resuming onto this one.
    2: "myj-pbl-wrf-v4.6.1-v1-mellor-yamada-2.5-janjic",
    # The UW moist-turbulence PBL carries prognostic state across steps --
    # the interface diffusivities it reads back as kvm_in/kvh_in and the
    # residual surface stress -- and its saturation table is WRF's own
    # words (gpuwm/core/uwpbl_constants.py).  The identity binds the WRF
    # version the port transcribes, v4.7.1.
    9: "uw-moist-turbulence-pbl-wrf-v4.7.1-v1",
    # v2 (2.8.8): Shin-Hong compiles without FMA contraction on WOOF's
    # float32 pow/exp (kernels/shinhong.cu), which moved its exchange
    # coefficient by up to 8 ULP, so a 2.8.7 (v1) checkpoint may not
    # continue under v2.
    11: "shinhong-pbl-wrf-v4.6.1-v2-no-fma-woof-libm",
    # SASE carries no WRF version in its identity because there is no WRF
    # scheme it transcribes.  What the identity DOES have to bind is the
    # closure's constant registry: sase_config_id() is a SHA-256 over
    # every registered coefficient, so a checkpoint written under one set
    # of constants cannot be resumed under another -- which is the whole
    # job of this table.
    900: "sase-experimental-v1",
}
#: sf_surface_physics -> the ``PhysicsDriver`` attribute holding that
#: scheme's packed parameter bundle, and the packaged-asset roles whose
#: bytes it was built from.  A land-surface scheme with no row here cannot
#: be restart-identified: a checkpoint that omitted its parameters would
#: resume against a silently different table set.  Adding a scheme means
#: adding its row, not relaxing the check.
LAND_SURFACE_PARAMETER_SOURCES = {
    2: ("noah_params", ("noah_vegparm", "noah_soilparm",
                        "noah_genparm", "noah_landuse")),
    # RUC reads the RUC SECTIONS of the same three files Noah reads --
    # VEGPARM's MODI-RUC/USGS-RUC blocks and SOILPARM's STAS-RUC block -- so
    # the asset roles are shared while the bundle object is not.  LANDUSE.TBL
    # is absent: gpuwm.core.ruc never opens it, because RUC's roughness,
    # albedo and emissivity come from its own VEGPARM rows.
    3: ("ruc_params", ("noah_vegparm", "noah_soilparm", "noah_genparm")),
    4: ("noahmp_params", ("noahmp_mptable", "noahmp_soilparm",
                          "noahmp_genparm")),
}
LONGWAVE_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "wrf-v4.6.1-rrtm-longwave-v1",
    4: "rte-rrtmgp-v1",
    90: "analytic-clear-sky-v1",
}
SHORTWAVE_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "wrf-v4.6.1-dudhia-shortwave-v1",
    4: "rte-rrtmgp-v1",
    90: "analytic-clear-sky-v1",
}
#: Above-model optical-column policy is separate from the gas/RTE algorithm
#: identity because changing the cap changes model-top fluxes while retaining
#: the same packaged coefficient tables and solver.
LONGWAVE_ABOVE_ATMOSPHERE_POLICIES = {
    0: "not-applicable-radiation-disabled",
    1: "wrf-v4.6.1-rrtm-deltap-4mb-buffer-layers",
    4: "wrf-v4.6.1-lw-4hpa-sw-half-ptop-clear-cap-to-rte-floor-v1",
    90: "not-applicable-analytic-surface-flux-proxy",
}
SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES = {
    0: "not-applicable-radiation-disabled",
    1: "not-applicable-dudhia-model-column-only",
    4: "wrf-v4.6.1-lw-4hpa-sw-half-ptop-clear-cap-to-rte-floor-v1",
    90: "not-applicable-analytic-surface-flux-proxy",
}
# Backward-compatible names for the historical coupled selections.  New
# identity code records each component independently; these aliases keep
# readers/tests that inspect a 4/4 or 90/90 setup source-compatible.
RADIATION_ALGORITHM_IDENTITIES = {
    key: LONGWAVE_ALGORITHM_IDENTITIES[key] for key in (0, 4, 90)}
RADIATION_ABOVE_ATMOSPHERE_POLICIES = {
    key: LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[key] for key in (0, 4, 90)}
#: The urban canopy models (``sf_urban_physics``).  Bound into the
#: checkpoint header only when one runs, so every existing header is
#: unchanged; the string names the WRF version whose routine each CUDA
#: column is graded against.
URBAN_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "urban-slucm-wrf-v4.7.1-v1",
    2: "urban-bep-wrf-v4.7.1-v1",
    3: "urban-bep-bem-wrf-v4.7.1-v1",
}

CUMULUS_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "kain-fritsch-v3-wrf-phase-energy-feedback",
    # The corrected-k22 identity IS the shipped algorithm (owner ruling);
    # a restart written under it must never resume under a WRF-faithful
    # build, which would be a different identity string.
    3: "grell-freitas-wrf461-gfdrv-corrected-k22-v1",
    # New Tiedtke, cu_ntiedtke/cumastrn as shipped in WRF v4.6.1.
    #
    # BUMP THE -v1 IF THE DRIVER SEAM MOVES, not only if the kernels do.
    # This port computes PRATEC at max_ulp == 0 and deliberately does not
    # hand it to the driver (docs/ntiedtke/PORT-RECORD.md section 38), so every
    # checkpoint written under this identity carries cu_pratec == 0 by
    # construction.  An implementation that delivered it would give the
    # same slot a different meaning, and that is exactly the kind of
    # cross-resume this string exists to refuse.
    16: "new-tiedtke-wrf461-cumastrn-v1",
}

#: The implementation of the dycore's once-per-step mixing: WRF's diff_opt
#: = 2 package (km_opt 1 to 4, gpuwm/core/dycore.py) and the sixth-order
#: filter (kernels/diff6.cu).  Configuration proves which closure runs and
#: with which constants (``configuration_sha256``); it cannot prove how the
#: operators are transcribed, and 2.8.8 changed exactly that under the same
#: selectors: the default build now compiles them with WRF's operation
#: order and IEEE arithmetic (no FMA, no flush to zero, IEEE division and
#: square root, gpuwm.core.kernels.DIFFUSION_OPTIONS), km_opt = 1 runs WRF's
#: isotropic_km through the shared package, a PBL-off open boundary keeps
#: its surface-flux ring, and the sixth-order filter accumulates in place
#: within WRF's loop bounds.  Millions of tendency words moved on WRF's
#: fixtures, so a checkpoint written before 2.8.8 under any of these would
#: continue a trajectory that is not the one it claims.
#:
#: Recorded only for a domain that runs one of these operators
#: (:func:`dycore_mixing_identity`), the way the urban row is recorded only
#: when an urban model runs: a run without mixing (the default km_opt = 1
#: with khdif = kvdif = 0 under a PBL scheme, and no sixth order) writes
#: exactly the header it wrote before 2.8.8 and resumes across the
#: release, because nothing it integrates moved.  A further change to the
#: mixing operators advances this tag.
DYCORE_MIXING_ALGORITHM_IDENTITY = (
    "wrf-v4.6.1-diff-opt2-mixing-and-sixth-order-v1-wrf-order-ieee")


def dycore_mixing_identity(cfg) -> str | None:
    """The mixing identity this domain's checkpoint records, or ``None``.

    ``None`` exactly when the dycore builds no mixing tendency at all:
    :func:`gpuwm.config.wrf_mixing_package_active` is false and the
    sixth-order filter is off -- the two predicates
    ``gpuwm.core.dycore.prepare_fixed_tendencies`` reads as
    ``include_smag`` and ``include_diff6``.
    """
    from gpuwm.config import wrf_mixing_package_active

    if wrf_mixing_package_active(cfg) or int(
            getattr(cfg, "diff_6th_opt", 0)) > 0:
        return DYCORE_MIXING_ALGORITHM_IDENTITY
    return None


#: The implementation of the implicit upper damping layer (damp_opt = 3,
#: inside WRF's advance_w).  2.8.8 forms its strength ``dampmag =
#: dts*dampcoef`` as WRF's one float32 product in the default build
#: (gpuwm.core.acoustic.damp_magnitude); 2.8.7 formed it as a double product
#: rounded once, which is a different word for sound steps such as 4.5 s
#: with dampcoef 0.2, and then every damped w and geopotential word moved
#: by up to 2,401 ULP against WRF.  The selectors did not change, so a
#: 2.8.7 checkpoint of a damp_opt = 3 run would continue a trajectory that
#: is not the one it claims.
#:
#: Recorded only for a domain that runs the damper
#: (:func:`upper_damping_identity`), the way the mixing row is: a run with
#: damp_opt = 0 (the default) writes the header it wrote before 2.8.8 and
#: resumes across the release.  A further change to the damper advances
#: this tag.
UPPER_DAMPING_ALGORITHM_IDENTITY = (
    "wrf-v4.6.1-damp-opt3-advance-w-float32-dampmag-v1")


def upper_damping_identity(cfg) -> str | None:
    """The upper-damping identity this domain's checkpoint records, or
    ``None`` when the implicit damper is off (damp_opt != 3), the one
    predicate :func:`gpuwm.core.acoustic.damp_magnitude` reads."""
    if int(getattr(cfg, "damp_opt", 0)) == 3:
        return UPPER_DAMPING_ALGORITHM_IDENTITY
    return None


#: PLAN REVIEW: every identity table the checkpoint writer resolves from
#: CONFIGURATION alone, as (table name, config attribute, label) rows.
#: These are the lookups :func:`physics_setup_identity` and the helpers it
#: calls perform on the way to a checkpoint whose only inputs are config
#: values -- which is exactly why the answer can be demanded before the
#: run starts instead of an hour into it.  Tables are NAMED rather than
#: captured so a row resolves through the module (monkeypatchable, and
#: comparable against the names the writer's own source reads:
#: tests/test_checkpoint_scheme_identity_gate.py derives the writer's set
#: from source rather than restating this list).  Radiation is absent and
#: handled by :func:`_radiation_identity_table_rows`: its selector is a
#: resolved (lw, sw) PAIR whose branch decides which tables are read at
#: all.
_CONFIGURED_IDENTITY_TABLES = (
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", "mp_physics", "microphysics"),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", "sf_sfclay_physics",
     "surface layer"),
    ("LAND_SURFACE_ALGORITHM_IDENTITIES", "sf_surface_physics",
     "land surface"),
    # A land-surface scheme is named TWICE on the way to a checkpoint:
    # once for its algorithm, and once for the packed parameter bundle
    # _land_surface_parameters_identity resolves through
    # LAND_SURFACE_PARAMETER_SOURCES -- which raises the same
    # "has no parameter-bundle row" RestartManifestError, from the same
    # point in the run, as the microphysics lookup that lost a forecast.
    # A scheme added to LAND_SURFACE_ALGORITHM_IDENTITIES and not to
    # LAND_SURFACE_PARAMETER_SOURCES reproduces that defect exactly, so
    # both tables are asked about here rather than only the one the first
    # instance of the defect happened to be in.
    ("LAND_SURFACE_PARAMETER_SOURCES", "sf_surface_physics",
     "land-surface parameter bundle"),
    ("PBL_ALGORITHM_IDENTITIES", "bl_pbl_physics", "PBL"),
    ("CUMULUS_ALGORITHM_IDENTITIES", "cu_physics", "cumulus"),
    # gpuwm/io/restart.py binds the urban canopy model a checkpoint was
    # written under; the table carries 0 ("disabled") too, so asking at
    # every value refuses nothing the writer would write.
    ("URBAN_ALGORITHM_IDENTITIES", "sf_urban_physics", "urban canopy"),
)

#: Rows of :data:`_CONFIGURED_IDENTITY_TABLES` the writer reaches only for
#: SOME values of their selector, as (table name, attribute) -> predicate
#: on the resolved id.  Asking outside the predicate would refuse a run
#: the writer would have written without complaint, which is the one way
#: a plan-review gate can be worse than no gate.
_CONDITIONAL_IDENTITY_TABLES = {
    # No land surface runs and no parameter bundle is resolved:
    # physics_setup_identity's ``sf_surface_physics != 0`` arm.
    ("LAND_SURFACE_PARAMETER_SOURCES", "sf_surface_physics"):
        lambda key: key != 0,
}


def _identity_table(name: str) -> dict:
    """The named module-level identity table, resolved at CALL time.

    Late resolution is what lets the gate see a table a test replaced and
    what lets the rows above be plain names.
    """
    return globals()[name]


def _radiation_identity_table_rows(cfg):
    """The radiation identity tables THIS configuration is named through.

    Mirrors the branch in :func:`_radiation_setup_identity` exactly,
    because which tables a radiation setup is looked up in is decided by
    the setup: a coupled selection resolves through the RADIATION_* pair
    (algorithm AND above-atmosphere policy), a mixed one through the
    LONGWAVE_*/SHORTWAVE_* pairs, and legacy RRTMG through module
    constants -- no table, so nothing to ask about.  Returned as
    (table name, config attribute, scheme id, label) rows.
    """
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY,
                                      rrtmg_variant)

    lw_id, sw_id = radiation_scheme_ids(cfg)
    if ((lw_id, sw_id) == (4, 4)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        return ()
    if (lw_id == sw_id
            and lw_id in _identity_table("RADIATION_ALGORITHM_IDENTITIES")):
        return (("RADIATION_ALGORITHM_IDENTITIES", "ra_lw_physics", lw_id,
                 "radiation"),
                ("RADIATION_ABOVE_ATMOSPHERE_POLICIES", "ra_lw_physics",
                 lw_id, "radiation above-atmosphere policy"))
    return (("LONGWAVE_ALGORITHM_IDENTITIES", "ra_lw_physics", lw_id,
             "longwave radiation"),
            ("LONGWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_lw_physics", lw_id,
             "longwave above-atmosphere policy"),
            ("SHORTWAVE_ALGORITHM_IDENTITIES", "ra_sw_physics", sw_id,
             "shortwave radiation"),
            ("SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_sw_physics", sw_id,
             "shortwave above-atmosphere policy"))


def unidentifiable_checkpoint_schemes(cfg) -> list[str]:
    """Which of this configuration's schemes the checkpoint writer cannot name.

    One entry per unnameable selection, in table order, each spelling the
    label, the id and the table that has no row for it.  Empty is the
    answer for every configuration whose checkpoints can be written.
    """
    gaps: list[str] = []

    def _report(message: str) -> None:
        if message not in gaps:
            gaps.append(message)

    for name, attribute, label in _CONFIGURED_IDENTITY_TABLES:
        value = getattr(cfg, attribute, None)
        try:
            key = int(value)
        except (TypeError, ValueError):
            # One attribute can feed two tables; say it once.
            _report(f"{label} {attribute}={value!r} is not a scheme id")
            continue
        predicate = _CONDITIONAL_IDENTITY_TABLES.get((name, attribute))
        if predicate is not None and not predicate(key):
            continue
        if key not in _identity_table(name):
            _report(
                f"{label} scheme {key} ({attribute}) has no row in the "
                f"checkpoint identity table")
    for name, attribute, key, label in _radiation_identity_table_rows(cfg):
        if key not in _identity_table(name):
            _report(
                f"{label} scheme {key} ({attribute}) has no row in the "
                f"checkpoint identity table")
    return gaps


def require_identifiable_checkpoint_schemes(cfg) -> None:
    """Refuse, AT PLAN REVIEW, a run whose checkpoints could not be written.

    THE CONCRETE BREAKAGE (measured 2026-09-10 07:39Z, single-domain ERA5
    forecast, mp_physics=9): every scheme a checkpoint names is looked up
    in one of the tables above, and a scheme with no row raises
    :class:`RestartManifestError` from ``_scheme_algorithm``.  That lookup
    happens inside ``write_restart``, which happens at the FIRST restart
    interval -- so a configuration the loader accepts and the forecast
    integrates dies at its first checkpoint, taking the 59 minutes of
    forecast it had already produced with it.  Nothing before that point
    can tell the operator, because nothing before that point asks the
    question.  This asks it while the answer is still free.

    Called from :func:`gpuwm.config.validate_run_config`, which every door
    runs before any device work, so it needs no flag and no opt-in: a run
    that would not have been checkpointable is refused where a
    configuration error belongs, and a run that is checkpointable never
    sees it.

    ``ValueError``, deliberately, and not the ``RestartManifestError``
    the writer's own lookup raises: this is a CONFIGURATION refusal, and
    every door already renders a ValueError from the invariant battery as
    a user-facing refusal with exit code 2 (gpuwm/cli.py) rather than a
    traceback.  The manifest error stays exactly where it was, as the
    backstop for a state that reaches a write unnameable.
    """
    gaps = unidentifiable_checkpoint_schemes(cfg)
    if not gaps:
        return
    detail = "; ".join(gaps)
    raise ValueError(
        f"this configuration cannot be checkpointed: {detail}. A restart "
        "file records the identity of every scheme that integrated it, so "
        "an unnamed scheme fails the write at the first restart interval "
        "and discards the forecast produced up to it. Add the identity "
        "row for the scheme in gpuwm/checkpoint_identity.py -- do not "
        "relax the check, which is what makes an incompatible resume "
        "fail before it "
        "restores.")


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  The tables above are the source
# tools/build_registry.py copies each option's restart_algorithm_identity
# row from; this holds the tables to the registry's implemented inventory so
# a scheme that lands in the registry without its identity row fails THIS
# import -- the suite, not a forecast at its first restart interval.  Only
# gpuwm.physics_registry is reached, which imports nothing from gpuwm and
# no device library, so the module stays importable wherever config.py is.
def _require_agreement_with_the_registry() -> None:
    import os

    from gpuwm.physics_registry import (
        REGISTRY_REBUILD_ENV, component_options, require_registry_agreement)

    require_registry_agreement(
        "gpuwm.checkpoint_identity.MICROPHYSICS_ALGORITHM_IDENTITIES",
        "microphysics", MICROPHYSICS_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "gpuwm.checkpoint_identity.SURFACE_LAYER_ALGORITHM_IDENTITIES",
        "surface_layer", SURFACE_LAYER_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "gpuwm.checkpoint_identity.LAND_SURFACE_ALGORITHM_IDENTITIES",
        "land_surface", LAND_SURFACE_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "gpuwm.checkpoint_identity.LAND_SURFACE_PARAMETER_SOURCES",
        "land_surface", LAND_SURFACE_PARAMETER_SOURCES,
        cited_absences={0: (
            "no land surface runs and no parameter bundle is resolved: "
            "physics_setup_identity's sf_surface_physics != 0 arm")})
    require_registry_agreement(
        "gpuwm.checkpoint_identity.PBL_ALGORITHM_IDENTITIES",
        "pbl", PBL_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "gpuwm.checkpoint_identity.CUMULUS_ALGORITHM_IDENTITIES",
        "cumulus", CUMULUS_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "gpuwm.checkpoint_identity.URBAN_ALGORITHM_IDENTITIES",
        "urban", URBAN_ALGORITHM_IDENTITIES)
    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    # Radiation selects on a (lw, sw) PAIR, so the id-set helper does not
    # apply; every implemented option's resolved pair must be nameable in
    # both spectra's tables.  The legacy aggregate selector (-1, -1)
    # resolves to 4/4 through ra_physics=4.
    for option_id, option in component_options("radiation").items():
        lw = int(option["selectors"]["ra_lw_physics"])
        sw = int(option["selectors"]["ra_sw_physics"])
        if (lw, sw) == (-1, -1):
            lw, sw = 4, 4
        for value, table, name in (
                (lw, LONGWAVE_ALGORITHM_IDENTITIES,
                 "LONGWAVE_ALGORITHM_IDENTITIES"),
                (lw, LONGWAVE_ABOVE_ATMOSPHERE_POLICIES,
                 "LONGWAVE_ABOVE_ATMOSPHERE_POLICIES"),
                (sw, SHORTWAVE_ALGORITHM_IDENTITIES,
                 "SHORTWAVE_ALGORITHM_IDENTITIES"),
                (sw, SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES,
                 "SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES")):
            if value not in table:
                raise RuntimeError(
                    f"radiation option {option_id!r} resolves to "
                    f"({lw}, {sw}) and gpuwm.checkpoint_identity.{name} has "
                    f"no row {value}; a checkpoint of it would fail at the "
                    "first restart interval")


_require_agreement_with_the_registry()
