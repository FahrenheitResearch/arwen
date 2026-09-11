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
#: not km_opt's: no dycore option of any kind carries an identity string.
#: If that binding is ever wanted it is ONE dycore-wide identity beside
#: PHYSICS_DRIVER_ALGORITHM_IDENTITY, opened at a release boundary.
MICROPHYSICS_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "kessler-warm-rain-v1",
    6: "wsm6-single-moment-six-class-wrf-v4.6.1-v1",
    8: ("classic-thompson-wrf-v4.6.1-experimental-v3-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity"),
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
    9: ("milbrandt-yau-wrf-v4.6.1-v1-six-category-2mom-ccntype2-"
        "meyers-contact-nucl-nonspherical-snow-full-sedimentation"),
    10: "morrison-two-moment-v2-kf-number-seeding",
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
    # instead of silently resuming onto it.
    16: ("wdm6-double-moment-warm-rain-wrf-v4.6.1-v1-prognostic-nc-nr-ccn-"
         "gamma-mu1-rain-ccn-activation-xland-autoconversion-ccn-conc-init"),
    18: "nssl-two-moment-state-transport-v1-process-boundary-fail-loud",
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
    28: ("thompson-aerosol-aware-wrf-v4.6.1-v1-prognostic-nc-nwfa-nifa-"
         "ccn-activate-table-demott-koop-scavenging-surface-emission-"
         "synthetic-aerosol-init"),
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
    50: ("p3-one-category-wrf-v4.6.1-v1-2mom-ice-specified-nc-"
         "diagnosed-ssat-rime-mass-volume-transported"),
}
SURFACE_LAYER_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "revised-mm5-surface-layer-v1",
    # The Eta similarity surface layer.  The identity binds the WRF version
    # whose byte-frozen module_sf_myjsfc.F the port transcribes AND the
    # similarity tables it interpolates: MYJSFCINIT builds PSIM/PSIH by
    # accumulating ZETA in float32, so a different table construction is a
    # different scheme even at the same WRF version, and a checkpoint may
    # not resume across one.
    2: "eta-similarity-surface-layer-wrf-v4.6.1-v1-myjsfcinit-tables",
    5: "mynn-surface-layer-wrf-v4.6.1-v1",
    91: "classic-mm5-surface-layer-v1",
}
LAND_SURFACE_ALGORITHM_IDENTITIES = {
    0: "disabled",
    2: "noah-lsm-v2-post-sflx-chs2-source-water-lake-skin",
    3: "ruc-lsm-wrf-v4.6.1-v1",
    4: "noahmp-lsm-wrf-v4.6.1-v1",
}
PBL_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "ysu-v1",
    5: "mynn-edmf-pbl-wrf-v4.6.1-v1",
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
    11: "shinhong-pbl-wrf-v4.6.1-v1",
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
