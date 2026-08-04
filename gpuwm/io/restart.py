"""wrfrst-style full-state restart files with a bit-identical contract.

``write_restart`` serializes every cross-step model field — prognostics,
physics/soil/snow surface state, accumulators, held slow tendencies, the
KF driver persistence, and the model clock — so that ``restore_restart``
into a freshly prepared process continues the trajectory FP32-bit-exactly
(Phase 4 Task 8 gate: 6 h + restart + 6 h == uninterrupted 12 h on every
state field and accumulator).

Format: NumPy NPZ (an uncompressed zip of raw little-endian ``.npy``
members).  Chosen over NetCDF for exactness and auditability: the ``.npy``
payload is the array's memory bytes, so FP32 values (including negative
zeros, denormals, and NaN payloads) round-trip bit-exactly with no
writer-side dimension/type mapping in between, and any zip tool can audit
the members.  The wrfout NetCDF writer keeps its WRF-tooling role; restart
files are a model-internal exchange, not a product.

Completeness is ENFORCED, not hoped for: ``write_restart`` walks every
``DomainState`` attribute, every ``DomainState._scratch`` slot, and every
``PhysicsDriver`` attribute and raises :class:`RestartManifestError` for
anything not explicitly classified below as serialized, rebuilt, setup, or
infrastructure.  Adding model state without updating this manifest fails
the next restart write (and the CPU manifest tests) loudly.

Classification argument (audit2 restart findings, adjudicated here):

* SERIALIZED — read across step boundaries and not reconstructable:
  prognostics (+ Morrison moments and effective radii, which feed the NEXT
  radiation call), ``h_diabatic`` (WRF ``rdu``, Registry.EM_COMMON:1389 —
  re-zeroing drops one step of retained heating), the km_opt=2 prognostic
  SGS TKE carrier ``tke`` (WRF ``r``, Registry.EM_COMMON:312 — a developed
  turbulence field with no reconstruction route), the live microphysics
  accumulators in scratch (``mp_*``; the driver's diagnostic dataclass aliases
  this canonical set), the KF driver persistence (``cu_*`` scratch,
  W0AVG), the held coupled tendencies (mu-coupled at their historical due
  step — recoupling at restore is NOT bit-identical), the surface/Noah
  ``fields`` dict (UST/MOL/ZNT/QSFC/HFX/QFX/PBLH/SH2O/SNOTIME/ALBEDO/EMISS
  and the rest of WRF's r-flagged surface block), ``_pending_rainbl``,
  ``microphysics_updates`` (behavior-gating counter), and the clock.
* REBUILT — overwritten before every read: the RK time-t copies (written
  from prognostics at each ``dycore.step`` entry), the slow-tendency slots
  (zeroed each RK stage), the acoustic perturbations (reseeded by
  ``_init_small_steps`` each stage; ``ww_pp`` rediagnosed every substep),
  and the per-call scratch work buffers (Morrison/Kessler prep, refl,
  advection, diffusion, LBC helpers).  The driver's one-frame
  ``refl_10cm`` handoff is also rebuilt: normal output consumes it before
  any same-step restart write, and a resume never rewrites the boundary
  frame.  The bit-identity gate is the proof of this list: a missed
  cross-step dependence diverges the trajectory.
  The driver ``microphysics`` dataclass is rebuilt as aliases of the
  serialized ``scratch/mp_*`` arrays; v2 files carrying both historical
  copies are accepted only when those copies compare byte-for-byte equal.
* SETUP — deterministic from config + ingest (base state, coordinates,
  map factors, LBC tables) or resolved while physics is initialized
  (radiation calendar/grid/gases/ozone, Noah parameters, scheme policies,
  and coefficient assets): rebuilt by the normal preparation path and
  VALIDATED against SHA-256 fingerprints stored in the header, so a restart
  into a different setup fails loudly instead of drifting.  The dynamics
  fingerprint covers the attached lateral-boundary FORCING tables (every
  interval's side values and tendencies, byte-level).  The physics identity
  separately carries explicit versioned algorithm/policy names plus actual
  active-asset byte digests; config IDs alone are not treated as proof of
  identical physics.  The resident LBC device tables are re-uploaded by
  ``attach_lateral_boundaries`` during preparation; ``restore_restart``
  requires that attachment to exist and restores the clock LAST, after
  attach reset it to zero (audit: attach rewinds every physics calendar).
* McICA carries no RNG state: the subcolumn generator's seeds are pure
  functions of the column pressures and the fixed permuteseed
  (kernels/rrtmgp_mcica.cu:35-47, mirrored by
  ``npref.np_mcica_maxran_masks`` — proven there against WRF's kissvec),
  so radiation is call-time stateless and nothing is serialized for it.

Cupy is imported lazily so the module (and the CPU manifest/roundtrip
tests) stay importable without a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gpuwm.config import radiation_scheme_ids
from gpuwm.supervisor import fsync_file
from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY,
                                  RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant)
from gpuwm.core.nssl2_contract import (
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_MODE as NSSL2_DEFAULT_MODE,
    DEFAULT_RESTART_FIELDS as NSSL2_DEFAULT_RESTART_FIELDS,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT as NSSL2_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as NSSL2_WRF_REFERENCE_VERSION,
)
from gpuwm.state_serialization_contract import (
    LATERAL_BOUNDARY_PREFIX_SCHEMA,
    STATE_SERIALIZED_ATTRS,
    STATE_SETUP_ARRAYS,
    STATE_SETUP_SCALARS,
    lateral_boundary_prefix_identity as _lateral_boundary_prefix_identity,
    setup_core_fingerprint as _shared_setup_core_fingerprint,
    setup_fingerprint as _shared_setup_fingerprint,
)

#: Bump on any change to the key layout or classification tables; restores
#: reject unknown versions instead of guessing.  v2 pinned the lateral-
#: boundary forcing tables.  v3 removes duplicate driver/microphysics members;
#: the driver now aliases the serialized scratch/mp_* accumulator set.  v4
#: binds the resolved physics/radiation setup and every active packaged data
#: asset, including an explicit above-atmosphere radiation policy.  v5 adds
#: KF's independently held ice/snow rates and coupled snow tendency.  The reader
#: retains a byte-equality-checked v2 array-layout shim, but an old unbound v2
#: file is rejected because the v4 identity header is mandatory.
RESTART_FORMAT_VERSION = 5
READABLE_RESTART_FORMAT_VERSIONS = frozenset({2, RESTART_FORMAT_VERSION})

#: MP18 extends the existing v5 physics-identity object rather than creating
#: another archive format.  This nested contract is independently versioned:
#: an old/aliased MP18 payload cannot be mistaken for the canonical Registry
#: transport even though the outer NPZ layout remains v5.
NSSL2_RESTART_CONTRACT_VERSION = 1
NSSL2_RESTART_PROGNOSTICS = NSSL2_DEFAULT_RESTART_FIELDS
NSSL2_RESTART_AUXILIARY_STATE = ("h_diabatic",)
NSSL2_RESTART_PRECIPITATION_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
    "mp_sr",
)
# The generic Morrison spellings and NSSL's historical Fortran slab argument
# names are deliberately not accepted as checkpoint identities.  Registry
# names above are the sole durable public contract.
NSSL2_LEGACY_RESTART_ALIASES = frozenset({
    "state/nc", "state/nr", "state/ni", "state/ns", "state/ng",
    "state/ccw", "state/crw", "state/cci", "state/csw", "state/chw",
    "state/chl", "state/cn", "state/vhw", "state/vhl",
    "driver/microphysics/rainnc", "driver/microphysics/rainncv",
    "driver/microphysics/snownc", "driver/microphysics/snowncv",
    "driver/microphysics/graupelnc", "driver/microphysics/graupelncv",
    "driver/microphysics/hailnc", "driver/microphysics/hailncv",
    "driver/microphysics/sr",
})

_HEADER_KEY = "__gpuwm_restart_header__"

#: RunConfig fields that may legitimately differ between the writing and
#: the resuming run (forecast length / output & restart cadence).  Every
#: other field must match exactly or the restore fails.
CONFIG_RUN_LENGTH_FIELDS = frozenset({
    "run_seconds", "output_interval_s", "restart_interval_s"})

#: Trajectory-inert diagnostic toggles, restart-boundary-adjustable exactly
#: like the run-length fields: flipping one cannot change the model
#: trajectory (nwp_diagnostics inertness is pinned by
#: tests/test_uh_lifecycle.py), and a header written before the knob
#: existed simply lacks the key.  The accumulator payloads themselves stay
#: tolerant in both directions (missing in file -> zeroed with a note;
#: present in file under a diagnostics-off resume -> dropped with a note).
#: ``tke_budget`` joins it on the same argument: the accumulator reads
#: model state and writes only its own diagnostic scratch, so flipping it
#: at a restart boundary cannot change the trajectory (inertness pinned by
#: tests/test_tke_budget.py).
#:
#: ``sase_flux_diag`` joins on the same argument: it allocates four
#: history buffers and fills them from arrays the SASE step already
#: holds, reading no prognostic and writing none, so resuming with the
#: diagnostic newly switched on continues the SAME model.  Its two
#: siblings ``sase_moist_n2``, ``sase_stable_dissipation`` and
#: ``sase_additive_dissipation`` are deliberately NOT here -- all three
#: are physics selectors that move the trajectory, and a physics
#: selector may not change under a resume.
CONFIG_DIAGNOSTIC_FIELDS = frozenset(
    {"nwp_diagnostics", "tke_budget", "sase_flux_diag"})

#: An opt-in tree-checkpoint contract for restart-extend orchestration.  It
#: admits exactly one setup change: appending future root LBC intervals after
#: the sealed interval inventory recorded in the checkpoint.  Ordinary
#: restart readers never consult this marker and retain exact setup matching.
SEALED_FORCING_EXTENSION_MODE = "sealed-prefix-v1"

#: Root external-LBC clock semantic identity (Davies clock bind,
#: 2026-07-28).  Which dtbc the root's external Davies consumers took is
#: invisible to the config echo, the setup fingerprint (tables only), and
#: the physics identity, yet it changes every downstream trajectory: WRF's
#: post-increment recurrence (dyn_em/solve_em.F:371-372, reset at
#: share/mediation_integrate.F:1522) versus the retired one-step-lagged
#: elapsed-based calculation.  Specified-domain headers therefore record
#: the semantic that INTEGRATED the checkpoint; restores require it to
#: match the resuming state's binding mode, and a header without the key
#: is a pre-epoch file (legacy semantics) that fails closed under bound
#: production code.
ROOT_EXTERNAL_LBC_CLOCK_IDENTITY = "wrf-postincrement-v1"
ROOT_EXTERNAL_LBC_CLOCK_LEGACY = "legacy-elapsed-v0"

#: Versioned semantic identities.  These are deliberately explicit instead
#: of inferred from scheme numbers: a trajectory-changing implementation or
#: policy change must advance its tag, causing an incompatible restart to fail
#: before restore.  Asset bytes and resolved per-run values are bound below.
PHYSICS_SETUP_SCHEMA_VERSION = 2
PHYSICS_DRIVER_ALGORITHM_IDENTITY = \
    "gpuwm-physics-driver-v3-kf-phase-energy-pre-mp-expiry"
MICROPHYSICS_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "kessler-warm-rain-v1",
    6: "wsm6-single-moment-six-class-wrf-v4.6.1-v1",
    8: ("classic-thompson-wrf-v4.6.1-experimental-v3-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity"),
    10: "morrison-two-moment-v2-kf-number-seeding",
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
}

#: The mp_physics=28 state a restart may NEVER drop.  Every name here is
#: already in ``STATE_SERIALIZED_ATTRS`` and therefore already written by the
#: generic loop; this tuple exists so the *absence* of one is a refusal
#: rather than a silent, finite, wrong resume.  That failure mode is real and
#: specific to this scheme: WRF's terminal clamps (module_mp_thompson.F:
#: 3976-3982) hold nwfa/nifa at their floors and nc at 2/rho rather than
#: raising, so a checkpoint that lost the aerosol state would restore, run,
#: stay bounded, produce no NaN and no health trip -- and integrate a
#: measurably different (aerosol-inert) forecast.  ``nwfa2d``/``nifa2d`` are
#: included because they are cross-step CONSTANTS that nothing in the
#: forecast rewrites: thompson_init derives nwfa2d once at :509-510 and only
#: in the "no initial CCN" branch, so a resume that dropped them would run
#: forever with zero surface aerosol emission.  WRF agrees they belong in the
#: restart stream: Registry.EM_COMMON:492-493 gives QNWFA2D/QNIFA2D the IO
#: string ``i01{17}rhdu``, whose ``r`` is the restart stream.
THOMPSON_AEROSOL_RESTART_STATE = ("nc", "nwfa", "nifa")
THOMPSON_AEROSOL_RESTART_SURFACE_STATE = ("nwfa2d", "nifa2d")
SURFACE_LAYER_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "revised-mm5-surface-layer-v1",
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
}
RRTMGP_TRACE_GAS_POLICY_IDENTITY = \
    "rfmip-experiment-zero-plus-date-policy-and-overrides-v1"
#: Distinct restart identity for the exact port of WRF v4.6.1's bundled
#: legacy RRTMG (RunConfig.ra_rrtmg_variant = "rrtmg_legacy" on the 4/4
#: pair).  Deliberately NOT "rte-rrtmgp-v1": a restart written under one
#: 4/4 implementation must refuse to resume under the other.
RRTMG_LEGACY_LW_ALGORITHM_IDENTITY = "wrf-v4.6.1-rrtmg-legacy-lw-v1"
RRTMG_LEGACY_SW_ALGORITHM_IDENTITY = "wrf-v4.6.1-rrtmg-legacy-sw-v1"
#: Legacy RRTMG extends the model column with WRF's own Cavallo buffer
#: layers (deltap = 4 mb), like RRTM option 1 but with RRTMG's tables.
RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY = \
    "wrf-v4.6.1-rrtmg-deltap-4mb-buffer-layers-v1"

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
PHYSICS_ASSET_PATHS = {
    "rrtmgp_gas_lw": Path("data/rrtmgp/rrtmgp-gas-lw-g256.nc"),
    "rrtmgp_gas_sw": Path("data/rrtmgp/rrtmgp-gas-sw-g224.nc"),
    "rrtmgp_cloud_lw": Path("data/rrtmgp/rrtmgp-clouds-lw-bnd.nc"),
    "rrtmgp_cloud_sw": Path("data/rrtmgp/rrtmgp-clouds-sw-bnd.nc"),
    "rrtmgp_rfmip": Path("data/rrtmgp/rfmip-clear-sky-inputs.nc"),
    "wrf_rrtm_data": Path("data/wrf_radiation/RRTM_DATA"),
    "wrf_rrtmg_lw_data": Path("data/wrf_radiation/RRTMG_LW_DATA"),
    "wrf_rrtmg_lw_statics": Path("data/wrf_radiation/rrtmg_lw_statics.npz"),
    "wrf_rrtmg_sw_data": Path("data/wrf_radiation/RRTMG_SW_DATA"),
    "wrf_ozone_data": Path("data/wrf_radiation/ozone.formatted"),
    "wrf_ozone_lat": Path("data/wrf_radiation/ozone_lat.formatted"),
    "wrf_ozone_plev": Path("data/wrf_radiation/ozone_plev.formatted"),
    "noah_vegparm": Path("data/noah_tables/VEGPARM.TBL"),
    "noah_soilparm": Path("data/noah_tables/SOILPARM.TBL"),
    "noah_genparm": Path("data/noah_tables/GENPARM.TBL"),
    "noah_landuse": Path("data/noah_tables/LANDUSE.TBL"),
    # Noah-MP's three tables, whose bytes gpuwm/core/noahmp_mynn_contract.py
    # already pins against the WRF v4.6.1 tree.
    "noahmp_mptable": Path("data/noahmp/MPTABLE.TBL"),
    "noahmp_soilparm": Path("data/noahmp/SOILPARM.TBL"),
    "noahmp_genparm": Path("data/noahmp/GENPARM.TBL"),
    "kf_lutab": Path("data/kf_lutab/kf_lutab.npz"),
}

# --------------------------------------------------------------------------
# DomainState attribute classification.
# --------------------------------------------------------------------------

#: Cross-step state serialized under ``state/<name>`` (skipped when the
#: attribute is None for the active configuration).  p/al/alt are
#: recomputed from the prognostics at each step entry, but serializing the
#: end-of-step EOS diagnostics keeps the restored object bit-equal to the
#: live one for any pre-step consumer (e.g. output frames).
#: Overwritten before every read (see the module docstring's argument).
STATE_REBUILT_ATTRS = frozenset({
    # RK time-t copies: dycore.step writes them from the prognostics first.
    "u0", "v0", "w0", "thp0", "php0", "mup0",
    "qv0", "qc0", "qr0", "qi0", "qs0", "qg0", "nr0", "ni0", "ns0", "ng0",
    "qh0", "qndrop0", "qnr0", "qni0", "qns0", "qng0", "qnh0",
    "qnn0", "qvolg0", "qvolh0",
    # km_opt=2's TKE time-t copy, written from the serialized ``tke``
    # carrier by dycore.step before any reader (core/dycore.py:2186-2187),
    # exactly like thp0 and the moist time-t copies above.  The carrier
    # itself is SERIALIZED (state_serialization_contract.py).
    "tke0",
    # mp_physics=28 (Thompson aerosol-aware) RK time-t copies.  nc0 exists
    # only for mp=28: mp=10 allocates nc but does not transport it and so
    # has no nc0 (gpuwm/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES).
    "nc0", "nwfa0", "nifa0",
    # Slow-tendency slots, zeroed at the top of every RK stage.
    "ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t",
    # Acoustic perturbations, reseeded by _init_small_steps each stage
    # (ww_pp is rediagnosed by advance_mu_th on every substep).
    "u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp",
    "p_pp", "p_pp_old", "ww_pp", "al_pp",
})

#: Deterministic setup arrays covered by the header fingerprint.
#: Setup scalars folded into the fingerprint alongside the arrays.
#: Machinery: handled by dedicated sections (scratch, physics, clock) or
#: rebuilt by attach/prepare (LBC device mirrors, host caches).
STATE_INFRA_ATTRS = frozenset({
    "_scratch", "_scratch_arena", "_phb_host", "_dz_min",
    "_host_setup_state",
    "physics", "lateral_boundaries", "_lateral_boundary_device",
    "elapsed_seconds", "_nest_restart_classification",
})

# --------------------------------------------------------------------------
# DomainState._scratch slot classification.
# --------------------------------------------------------------------------

#: Persistent read-modify-write scratch: the canonical microphysics
#: accumulators (the kernels update them in place and the driver aliases them)
#: and the KF driver persistence
#: (NCA timers, PRATEC/RAINCV, stored per-column rates, RAINC).
SERIALIZED_SCRATCH_SLOTS = frozenset({
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_sr", "mp_kessler_sr",
    "mp_hailnc", "mp_hailncv",
    "cu_rainc", "cu_nca", "cu_pratec", "cu_raincv",
    "cu_rthcuten", "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten",
    "cu_rqrcuten", "cu_rqscuten",
    # WRF UP_HELI_MAX (Registry IO "rh02" -- the r is this row).  A running
    # max is a read-modify-write accumulator exactly like the mp_* totals;
    # a checkpoint written before the slot existed restores it zeroed with
    # a note, never a refusal (gpuwm/core/uh_diag.py owns the lifecycle).
    "up_heli_max",
})

#: Per-call work buffers overwritten before every read.  ``mp_``/``cu_``
#: rebuilds are EXACT names only — a future accumulator slot under those
#: prefixes must be classified explicitly instead of silently dropping.
REBUILT_SCRATCH_SLOTS = frozenset({
    "mp_th", "mp_rho", "mp_pii", "mp_z", "mp_dz8w", "mp_z8w",
    "mp_thompson_temperature",
    "mp_thompson_frozen_reference_density",
    "mp_thompson_frozen_reference_temperature",
    "mp_thompson_rain_reference_density",
    "mp_thompson_snow_melt_marker",
    "mp_thompson_graupel_melt_marker",
    "mp_thompson_snow_velocity_boost",
    # WRF's private classic-graupel number exists only across one
    # output-due Thompson call and is finalized/consumed by REFL_10CM.
    "mp_thompson_graupel_number_shadow",
    # mp_physics=28 (Thompson aerosol-aware).  Listed as EXACT names, not a
    # new "mp_thompson_aero_" prefix, because the prefix rule above exists
    # precisely to stop a future accumulator from being silently dropped --
    # and three of these ARE accumulators.  They are nonetheless rebuilt,
    # not serialized: WRF zeroes ncten/nwfaten/nifaten at the top of every
    # column call (module_mp_thompson.F:1679-1681) and applies them once
    # before returning (:3972-4021), so nothing in them survives a call
    # boundary, let alone a restart.  The entry snapshots are likewise
    # re-frozen from state at every call entry (:1795-1848).  Serializing
    # any of them would be the bug: a restored non-zero tendency would be
    # applied to state a second time.
    "mp_thompson_aero_ncten",
    "mp_thompson_aero_nwfaten",
    "mp_thompson_aero_nifaten",
    "mp_thompson_aero_entry_density",
    "mp_thompson_aero_nwfa_entry_m3",
    "mp_thompson_aero_nifa_entry_m3",
    "mp_thompson_aero_tau1_density",
    "mp_thompson_aero_nwfa_work_m3",
    "mp_thompson_aero_qc_entry",
    "mp_thompson_aero_ni_entry",
    "mp_thompson_aero_rc_entry",
    "mp_thompson_aero_nc_entry_m3",
    "mp_thompson_aero_nu_c_entry",
    "mp_thompson_aero_l_qc_entry",
    "mp_thompson_aero_condensation_rate",
    "nssl2_driver_state", "nssl2_driver_surface_export",
    "nssl2_driver_ignored_accumulator",
    "nssl2_fused_temperature", "nssl2_primary_ice_target",
    "nssl2_nucond_ss",
    # The DA reflectivity operator's own dry-air density and diagnosed
    # temperature (gpuwm/da/obsop.py:_nssl_reflectivity).  Both are filled
    # in full at the top of one H(x) call — rho from 1/alt, T from theta
    # and Exner — and consumed by the shared NSSL diagnostic inside that
    # same call, so no restart boundary can fall between the write and the
    # read and neither carries anything across a step.
    "da_nssl_rho", "da_nssl_t",
    "cu_expiring",
})

REBUILT_SCRATCH_PREFIXES = (
    "rk_", "adv_", "smag_", "diff_", "diff6_", "acoustic_", "openbc_",
    "moist_", "pd_", "morr_", "wsm6_", "refl_", "physics_", "lbc_",
    "integration_health_", "nest_",
    # UP_HELI_MAX per-step work planes (column UH + use_column flags),
    # overwritten by every launch; the accumulator itself is the exact
    # serialized name above, deliberately NOT under this prefix.
    "uh_diag_",
    # Spec-zone ring-guard snapshots live only between the capture and
    # restore inside ONE microphysics.apply call (core/microphysics.py);
    # their contents are dead at any restart boundary.
    "mp_ring_save_",
    # The km_opt=2 TKE budget's own slots (gpuwm/core/tke_budget.py).  The
    # per-term 3-D fields and the mu totals are rewritten inside every step
    # before anything reads them.  The slab accumulator and its step counter
    # ARE carried across steps, but they are a report-only diagnostic window
    # the caller drains and resets -- never trajectory state -- so a resumed
    # run restarts the current window rather than continuing it, and the
    # drained receipt records the step count it actually covered.
    "tke_budget_",
    # MYNN's declared workspace (gpuwm/core/mynn_pbl_scratch.py).  Every one
    # of these is rebuilt inside the call that reads it; the scheme's carried
    # state is the ten 3-D ``fields`` arrays, which are serialized as fields
    # and are deliberately not scratch.
    "mynn_pbl_",
)

# --------------------------------------------------------------------------
# PhysicsDriver attribute classification.
# --------------------------------------------------------------------------

#: Held coupled slow tendencies.  Serialized COUPLED, exactly as held: the
#: cumulus/radiation arrays were mu-coupled with total_mu() at their
#: historical due/expiry step, and mu has evolved since — recoupling
#: restored rates at restore time is not bitwise identical (audit).
DRIVER_TENDENCY_ATTRS = ("pbl_tendencies", "radiation_tendencies",
                         "cumulus_tendencies")
TENDENCY_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi",
                       "rqs")
TENDENCY_REQUIRED_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc")

MICROPHYSICS_COMPONENTS = ("rainnc", "rainncv", "sr", "snownc", "snowncv",
                           "graupelnc", "graupelncv", "hailnc", "hailncv")
MICROPHYSICS_REQUIRED_COMPONENTS = ("rainnc", "rainncv", "sr")

#: Driver attributes serialized into the file/header.
DRIVER_SERIALIZED_ATTRS = frozenset({
    "pbl_tendencies", "radiation_tendencies", "cumulus_tendencies",
    "rthratenlw", "rthratensw", "_pending_rainbl",
    "microphysics_updates", "call_counts", "ysu_nan_guard_fires",
    "fields",
})

#: Driver attributes rebuilt by initialize_physics from config/tables, or
#: aliases of serialized storage: ``rainc``/``cu_nca``/``cu_pratec``/
#: ``cu_raincv``/``cu_rates`` reference the ``cu_*`` scratch slots (data
#: restored in place through the scratch pool, so the aliases stay live),
#: ``sfclay_result`` aliases the ``fields`` arrays (restored in place —
#: never rebound — so SFCLAY's seven WRF-inout fields stay coupled),
#: active-scheme ``microphysics`` aliases serialized ``mp_*`` scratch
#: (mp=0 rebuilds its all-zero output placeholder), ``tendencies`` is
#: recomposed by every ``compute()``, and ``last_ysu`` is refreshed before
#: any consumer.  ``refl_10cm`` is an
#: ephemeral output handoff: the due microphysics call rebuilds it, output
#: consumes it once, and restart-resume does not reproduce the boundary
#: frame (PROVENANCE.md D2).  The WSM6 SR roundoff limit, ULP count, and
#: minor-loop count are deterministic functions of the resolved
#: ``mp_physics`` and ``dt`` configuration and are rebuilt with the driver.
DRIVER_REBUILT_ATTRS = frozenset({
    # SASE: the active flag and the kernel-module tuple are re-derived
    # from the resumed RunConfig at driver init; the ledger is a
    # per-step diagnostic replaced before any consumer reads it; the
    # flux-diagnostic buffers are output-only and refilled by the first
    # step after the resume.
    "sase_active", "last_sase_ledger", "sase_flux_diag",
    "sase_nan_guard_fires",
    # The horizontal eddy-viscosity diagnostic, on the SAME terms as the
    # flux-diagnostic buffers beside it: output-only, never read back by
    # the physics, and refilled by the first step after a resume (the
    # SASE half in the closure slot, the Smagorinsky half at the next
    # output).  A resumed run's first frame therefore carries the
    # post-resume step's viscosity, which is the value that step used.
    "hmix_k_diag",
    # OLR (TOA outgoing longwave) on the SAME terms: output-only, never
    # read back by the physics, and refilled by the next due radiation
    # call after the resume.  WRF's own row is restart-carried
    # (Registry.EM_COMMON:1839 flags it ``rh``) and gpuwm's is not, which
    # is a deliberate scope choice rather than an oversight: adding it to
    # the archive changes the v5 key layout and would reject every
    # checkpoint already on disk, which is not a price a diagnostic
    # nothing consumes gets to charge.  The visible consequence is that a
    # resumed run publishes zeros for OLR until its next radiation call,
    # exactly as a cold-started run does before its first one.
    "olr",
    "state", "sfclay_result", "mynn_sfclay_result",
    "mynn_sfclay_sea_result", "noah_params",
    # Selector-value -> runner-method receipt, re-resolved from the resumed
    # RunConfig by PhysicsDriver.__init__ (the config fingerprint already
    # binds the selector values themselves).
    "scheme_dispatch",
    "radiation_callable", "cumulus_callable",
    "ra_physics", "ra_lw_physics", "ra_sw_physics",
    "radiation_active", "cu_physics", "mp_physics", "surface_enabled",
    # Noah LSM option selectors: plain cfg-derived scalars reconstructed at
    # driver init, exactly like the radiation and cumulus selectors above.
    "noah_usemonalb", "noah_rdlai2d", "noah_opt_thcnd",
    # Noah-MP: the parsed tables and the solar geometry are rebuilt by
    # initialize_physics from the packaged assets and the caller's
    # start-time/latitude/longitude, and BOTH are bound into the checkpoint
    # header by _land_surface_parameters_identity -- so a resume against
    # different tables or a different date/latitude is refused rather than
    # silently continuing a different trajectory.  The four-layer thickness
    # vector is a module constant.  The per-call column census is a receipt
    # the next call overwrites.
    "noahmp_params", "noahmp_geometry", "noahmp_soil_thickness_m",
    "last_noahmp_census",
    # RUC: the parsed tables and the nine-level geometry are rebuilt by
    # initialize_physics from the same packaged assets, and the bundle is
    # bound into the checkpoint header by _land_surface_parameters_identity,
    # so a resume against different table bytes is refused.  Unlike Noah-MP,
    # RUC reads no solar geometry at all -- it takes GSW and GLW as forcing --
    # so there is no second identity to bind.  The per-call census is a
    # receipt the next call overwrites.
    "ruc_params", "last_ruc_census",
    "tendencies", "last_ysu", "refl_10cm", "microphysics",
    "nssl2_binding",
    "bldt_seconds", "stepbl", "radt_minutes", "cudt_minutes",
    "stepra", "stepcu", "radt_seconds", "cudt_seconds",
    "rainc", "cu_nca", "cu_pratec", "cu_raincv", "cu_expiring",
    "_cu_expiry_pending",
    "cu_rates", "_sr_roundoff_upper", "_sr_roundoff_max_ulps",
    "_wsm6_minor_loops",
})

#: Scheme-callable state classification.  The walk covers DIRECT array
#: attributes, every value of dict-valued attributes, and ONE level of
#: object-container attributes; anything array-bearing outside these
#: allowlists fails the write.  Deeper nesting is out of walk scope by
#: design — a container holding arrays must itself be classified here.
#:
#: Arrays: the cumulus adapter's W0AVG is restart state (WRF
#: Registry.EM_COMMON:1575 r-flags it); the radiation constants are
#: setup-time (lat/lon grids, ozone climatology).
CUMULUS_CALLABLE_ARRAYS = frozenset({"w0avg"})
RADIATION_CALLABLE_ARRAYS = frozenset({
    "latitude_deg", "longitude_deg", "_ozone_logp", "_ozone_vmr",
    # Legacy-RRTMG adapter: _ozone_lat_interp is setup state (a
    # deterministic construction-time interpolation of the packaged CAM
    # climatology onto the static latitude grid); _o33d_grid is
    # SERIALIZED state -- WRF's O3RAD is a restart-carried field (rdf),
    # and a child domain's first post-restore radiation call consumes
    # the parent's retained o33d BEFORE the parent's next radiation
    # cadence tick, so rebuild-on-resume would orphan it (and break
    # resumed-vs-uninterrupted bit equality).
    "_ozone_lat_interp", "_o33d_grid"})
#: Containers CLASSIFIED as acceptable, deliberately (review F2 — no
#: silent blind spots): the RRTMGP gas/cloud table objects are
#: rebuild-on-load (module-level ``lru_cache`` loads of packaged
#: k-distribution/cloud-optics data — deterministic, never mutated per
#: call; their lazy ``_device`` mirrors likewise), and the KF adapter's
#: ``_history_state`` is a back-reference to the DomainState itself,
#: whose arrays the state walk already covers.
RADIATION_CALLABLE_CONTAINERS = frozenset({
    "lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables",
    "chunk_workspace",
    # Legacy-RRTMG adapter containers, all rebuild-on-load: _C (the LW
    # coefficient dict) and _sw_tables/_cuda_sw/_ozone_climo are
    # digest-checked deterministic loads of packaged assets performed at
    # construction (never mutated per call); _night_outputs is a
    # per-radiation-call product fully rebuilt before every consumption.
    "_C", "_sw_tables", "_cuda_sw", "_ozone_climo", "_night_outputs",
    # _ozone is the gpuwm.ingest.wrf_ozone MODULE reference (its globals
    # include cached climatology arrays); modules are code, not state.
    "_ozone"})
CUMULUS_CALLABLE_CONTAINERS = frozenset({"_history_state"})


class RestartManifestError(RuntimeError):
    """Model state exists that the restart manifest does not classify."""


class RestartMismatchError(ValueError):
    """The restart file does not match the resuming configuration/setup."""


def producer_identity() -> dict[str, str]:
    """Which build wrote this file.

    A checkpoint or a history file that has been separated from the run's
    logs -- archived, mailed, or found in a directory a year later -- can
    otherwise say nothing about the code that produced it.  The version is
    the installed distribution's, not a hand-maintained constant, because
    the hand-maintained constant is exactly what went stale for four
    releases; and the restart format version rides along because a reader
    that cannot parse the payload still needs to know why.
    """
    from gpuwm import DISTRIBUTION_NAME, __version__

    return {
        "distribution": DISTRIBUTION_NAME,
        "version": __version__,
        "restart_format_version": str(RESTART_FORMAT_VERSION),
    }


def _admissible_elapsed_seconds(value, where: str) -> float:
    """Model-clock seconds that arithmetic can survive.

    ``float(value)`` alone admits ``NaN`` -- Python's json writes and reads
    the bare token by default -- and admits a negative clock, and both pass
    every identity check a restart makes before poisoning cadence and
    resume arithmetic downstream of them.  MP18 already refused both; this
    is the same refusal for every format and both directions.
    """
    # ``bool`` is an ``int`` and ``float("600")`` succeeds, so neither a
    # flag nor a numeric string may pass as a clock: the header field is
    # written as a JSON number and anything else is a malformed header.
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         np.integer,
                                                         np.floating)):
        raise RestartManifestError(
            f"{where} elapsed_seconds must be a real number, got {value!r}")
    try:
        elapsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"{where} elapsed_seconds must be a real number, "
            f"got {value!r}") from exc
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RestartManifestError(
            f"{where} elapsed_seconds must be finite and non-negative, "
            f"got {elapsed!r}")
    return elapsed


def _nssl2_restart_contract_identity() -> dict[str, object]:
    """Return the exact versioned MP18 state carried by restart v5."""
    return {
        "schema_version": NSSL2_RESTART_CONTRACT_VERSION,
        "physics_contract_id": NSSL2_CONTRACT_ID,
        "wrf_reference": {
            "version": NSSL2_WRF_REFERENCE_VERSION,
            "commit": NSSL2_WRF_REFERENCE_COMMIT,
        },
        "resolved_default_mode": dataclasses.asdict(NSSL2_DEFAULT_MODE),
        "resolved_wrf_namelist_defaults": dict(NSSL2_WRF_NAMELIST_DEFAULTS),
        "state_members": [
            *(f"state/{name}" for name in NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}" for name in NSSL2_RESTART_AUXILIARY_STATE),
        ],
        "precipitation_members": [
            f"scratch/{slot}" for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
        ],
        "first_call_authority": "driver.microphysics_updates == 0",
        "clock_authority": "header.elapsed_seconds",
        "continuation_policy": "bitwise",
    }


def _require_nssl2_array(value, shape: tuple[int, ...], label: str) -> None:
    if value is None or not _is_array_like(value):
        raise RestartManifestError(
            f"MP18 restart requires array {label!r}")
    if tuple(value.shape) != shape:
        raise RestartManifestError(
            f"MP18 restart {label!r} has shape {tuple(value.shape)}, "
            f"expected {shape}")
    if np.dtype(value.dtype) != np.dtype(np.float32):
        raise RestartManifestError(
            f"MP18 restart {label!r} has dtype {value.dtype}, expected "
            "float32")


def _validate_thompson_aerosol_live_restart_state(state, cfg) -> None:
    """Fail an mp=28 WRITE that would omit any aerosol state.

    The generic writer loop already picks these up through
    ``STATE_SERIALIZED_ATTRS`` when they exist; this refuses the write when
    they DO NOT.  Without it, a build whose ``DomainState`` stopped
    allocating (say) ``nwfa2d`` would produce a checkpoint that both the
    writer and the reader consider internally consistent -- the reader's
    inventory check compares the file against the *resuming* state, so two
    equally aerosol-less endpoints agree -- and the resumed run would
    integrate with zero surface aerosol emission forever.  Adding the 28
    identity string without this guard is exactly the "restart proceeds
    while silently dropping the aerosol state" outcome that is worse than
    the previous outright refusal.
    """
    if int(cfg.mp_physics) != 28:
        return

    volume_shape = tuple(state.p.shape)
    surface_shape = tuple(state.mup.shape)
    for name, shape in (
            *((name, volume_shape) for name in
              THOMPSON_AEROSOL_RESTART_STATE),
            *((name, surface_shape) for name in
              THOMPSON_AEROSOL_RESTART_SURFACE_STATE)):
        value = getattr(state, name, None)
        if value is None or not _is_array_like(value):
            raise RestartManifestError(
                f"mp_physics=28 restart requires array 'state/{name}' "
                "(Thompson aerosol-aware carries prognostic nc plus the "
                "nwfa/nifa tracers and their nwfa2d/nifa2d surface "
                "emission); refusing to write a checkpoint that would "
                "resume with an aerosol-inert column")
        if tuple(value.shape) != shape:
            raise RestartManifestError(
                f"mp_physics=28 restart 'state/{name}' has shape "
                f"{tuple(value.shape)}, expected {shape}")
        if np.dtype(value.dtype) != np.dtype(np.float32):
            raise RestartManifestError(
                f"mp_physics=28 restart 'state/{name}' has dtype "
                f"{value.dtype}, expected float32")


def _validate_thompson_aerosol_stored_restart_state(
        stored: dict[str, np.ndarray], state, cfg, path) -> None:
    """Reject an mp=28 restart FILE that omits any aerosol state."""
    if int(cfg.mp_physics) != 28:
        return

    required = {
        f"state/{name}" for name in (
            *THOMPSON_AEROSOL_RESTART_STATE,
            *THOMPSON_AEROSOL_RESTART_SURFACE_STATE)
    }
    stored_state = {key for key in stored if key.startswith("state/")}
    missing = sorted(required - stored_state)
    if missing:
        raise RestartMismatchError(
            f"restart file {path} omits canonical mp_physics=28 aerosol "
            f"state {missing}; resuming would silently integrate an "
            "aerosol-inert Thompson column (the terminal clamps at "
            "module_mp_thompson.F:3976-3982 keep it finite and bounded, so "
            "nothing downstream would notice)")
    for key in sorted(required):
        name = key[len("state/"):]
        target = getattr(state, name, None)
        if target is None:
            # The RESUMING model has no slot for a field the file carries.
            # A DomainState built from an mp=28 RunConfig always allocates
            # all five (gpuwm/core/state.py's mp==28 arm), so reaching this
            # means the two ends disagree about what mp=28 is -- report it
            # here rather than raising AttributeError from _check_array.
            raise RestartMismatchError(
                f"restart file {path} carries {key} but this build's "
                f"mp_physics=28 DomainState has no {name!r}")
        _check_array(stored[key], target, key)


def _validate_nssl2_live_restart_state(state, cfg) -> None:
    """Fail a write unless every persistent MP18 value is canonical."""
    if int(cfg.mp_physics) != 18:
        return

    shape = tuple(state.p.shape)
    for name in (*NSSL2_RESTART_PROGNOSTICS,
                 *NSSL2_RESTART_AUXILIARY_STATE):
        _require_nssl2_array(getattr(state, name, None), shape, f"state/{name}")

    driver = getattr(state, "physics", None)
    if driver is None or int(getattr(driver, "mp_physics", -1)) != 18:
        raise RestartManifestError(
            "MP18 restart requires an attached MP18 PhysicsDriver so "
            "precipitation and first-call state cannot be omitted")
    pool = getattr(state, "_scratch", {})
    surface_shape = tuple(state.mup.shape)
    missing = [
        slot for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
        if slot not in pool
    ]
    if missing:
        raise RestartManifestError(
            f"MP18 restart lacks persistent precipitation slots {missing}")
    for slot in NSSL2_RESTART_PRECIPITATION_SLOTS:
        _require_nssl2_array(pool[slot], surface_shape, f"scratch/{slot}")
    unexpected_mp = sorted(
        slot for slot in pool
        if (slot.startswith("mp_")
            and classify_scratch_slot(slot) == "serialize"
            and slot not in NSSL2_RESTART_PRECIPITATION_SLOTS))
    if unexpected_mp:
        raise RestartManifestError(
            "MP18 restart has noncanonical persistent microphysics slots "
            f"{unexpected_mp}")

    updates = getattr(driver, "microphysics_updates", None)
    if (isinstance(updates, bool) or not isinstance(updates, int)
            or updates < 0):
        raise RestartManifestError(
            "MP18 restart first-call authority microphysics_updates must be "
            f"a non-negative integer, got {updates!r}")
    try:
        _admissible_elapsed_seconds(state.elapsed_seconds, "MP18 restart")
    except RestartManifestError:
        raise RestartManifestError(
            "MP18 restart elapsed_seconds must be finite and non-negative") \
            from None


def restart_filename(valid_time: datetime, domain: str = "d01") -> str:
    """WRF ``wrfrst``-style file name for a restart valid time.

    Whole seconds only, refused rather than truncated, for the reason
    ``gpuwm.io.wrfout.wrfout_filename`` gives: this name is the standalone
    checkpoint's whole identity and its publisher replaces, so two legal
    sub-second checkpoints used to collapse onto one file and the earlier
    one ceased to exist.
    """
    if valid_time.microsecond:
        raise ValueError(
            f"restart valid time {valid_time!r} is not on a whole second; "
            "checkpoint filenames carry whole seconds only, so distinct "
            "sub-second instants would alias onto one file and the later "
            "checkpoint would replace the earlier one")
    return valid_time.strftime(f"gpuwmrst_{domain}_%Y-%m-%d_%H_%M_%S.npz")


def _host(value) -> np.ndarray:
    """Return a host ndarray view/copy of a device or host array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _is_array_like(value) -> bool:
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and hasattr(value, "ndim"))


def classify_state_attr(name: str) -> str:
    """Classify one ``DomainState`` attribute name.

    Returns ``"serialize"``, ``"rebuild"``, ``"setup"``, or ``"infra"``;
    raises :class:`RestartManifestError` for anything unclassified so new
    state cannot silently skip the restart stream.
    """
    if name in STATE_SERIALIZED_ATTRS:
        return "serialize"
    if name in STATE_REBUILT_ATTRS:
        return "rebuild"
    if name in STATE_SETUP_ARRAYS or name in STATE_SETUP_SCALARS:
        return "setup"
    if name in STATE_INFRA_ATTRS:
        return "infra"
    raise RestartManifestError(
        f"DomainState attribute {name!r} is not classified in the restart "
        "manifest (gpuwm/io/restart.py): declare it serialized (cross-step "
        "state), rebuilt (overwritten before every read), setup "
        "(fingerprint-validated), or infra")


def classify_scratch_slot(slot: str) -> str:
    """Classify one ``DomainState.scratch`` slot name.

    Returns ``"serialize"`` or ``"rebuild"``; raises
    :class:`RestartManifestError` for unknown slots.
    """
    if slot in SERIALIZED_SCRATCH_SLOTS:
        return "serialize"
    if slot in REBUILT_SCRATCH_SLOTS:
        return "rebuild"
    if any(slot.startswith(prefix) for prefix in REBUILT_SCRATCH_PREFIXES):
        return "rebuild"
    raise RestartManifestError(
        f"scratch slot {slot!r} is not classified in the restart manifest "
        "(gpuwm/io/restart.py): declare it serialized (persistent "
        "accumulator/held state) or rebuilt (per-call work buffer)")


def setup_fingerprint(state) -> str:
    """SHA-256 over the deterministic setup arrays/scalars AND the
    attached lateral-boundary forcing tables.

    A restart written on one setup (base state, coordinates, map factors)
    refuses to restore onto another: silently continuing on different
    reference profiles would not be the same trajectory.  The LBC digest
    (review F3) covers every interval's time bounds and every side's
    value/tendency bytes, so a same-config resume against a modified or
    replaced reference bundle — which passes the config echo — is
    rejected instead of silently integrating different boundary forcing.
    """
    return _shared_setup_fingerprint(
        state, error_type=RestartManifestError)


def setup_core_fingerprint(state) -> str:
    """SHA-256 over immutable setup, excluding a root's LBC inventory."""
    return _shared_setup_core_fingerprint(
        state, error_type=RestartManifestError)


def lateral_boundary_prefix_identity(state):
    """Compact byte identity for the root forcing interval inventory."""
    return _lateral_boundary_prefix_identity(
        state, error_type=RestartManifestError)


def _canonical_json(value) -> str:
    """Stable, strict JSON used by semantic restart fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value, label: str):
    """Return a strict JSON value without silently stringifying objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RestartManifestError(
                f"{label} contains non-finite value {value!r}")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), label)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        normalized = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise RestartManifestError(
                    f"{label} has non-string identity key {key!r}")
            normalized[key] = _json_value(value[key], f"{label}.{key}")
        return normalized
    raise RestartManifestError(
        f"{label} value {value!r} is not strict JSON restart identity; "
        "declare strings/numbers/bools/lists/mappings only")


def _array_setup_identity(value) -> dict:
    """Shape/type/content identity for a resolved setup array."""
    host = np.ascontiguousarray(_host(value))
    digest = hashlib.sha256()
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(host.tobytes(order="C"))
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "sha256": digest.hexdigest(),
    }


def _resolved_object_setup_identity(value, label: str) -> dict:
    """Canonical identity for a resolved host coefficient-table object.

    Private device mirrors are intentionally excluded: they are deterministic
    conversions of these authoritative host arrays and may be lazily absent in
    a freshly prepared process.  Every public dataclass/instance field must be
    a setup array or strict JSON scalar/container.
    """
    if value is None:
        raise RestartManifestError(f"resolved {label} object is missing")
    if dataclasses.is_dataclass(value):
        names = [field.name for field in dataclasses.fields(value)]
    else:
        names = list(getattr(value, "__dict__", {}))
    names = sorted(name for name in names if not name.startswith("_"))
    if not names:
        raise RestartManifestError(
            f"resolved {label} object {_callable_class_name(value)} has no "
            "identifiable public fields")
    arrays = {}
    values = {}
    for name in names:
        item = getattr(value, name)
        if _is_array_like(item):
            arrays[name] = _array_setup_identity(item)
        else:
            values[name] = _json_value(item, f"{label}.{name}")
    payload = {
        "class": _callable_class_name(value),
        "arrays": arrays,
        "values": values,
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _rrtmgp_workspace_identity(workspace) -> dict:
    """Bind the optional workspace code path without hashing scratch bytes."""
    if workspace is None:
        return {"present": False}
    try:
        p_top = float(workspace.p_top)
        if not math.isfinite(p_top) or p_top < 0.0:
            raise ValueError("workspace p_top must be finite and nonnegative")
        identity = {
            "present": True,
            "class": _callable_class_name(workspace),
            "nz": int(workspace.nz),
            "column_chunk": int(workspace.column_chunk),
            "p_top": p_top,
            "nbytes": int(workspace.nbytes),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            "RRTMGP chunk_workspace lacks nz/column_chunk/p_top/nbytes "
            "identity") \
            from exc
    layouts = getattr(workspace, "_phase_layouts", None)
    identity["phase_layouts"] = (
        None if layouts is None
        else _json_value(layouts, "radiation.chunk_workspace.phase_layouts"))
    return identity


def _asset_sha256(path) -> str:
    """Digest the actual packaged bytes consumed by an active scheme."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _active_asset_identity(cfg, driver) -> dict[str, dict]:
    roles = []
    radiation = None if driver is None else driver.radiation_callable
    radiation_class = (None if radiation is None
                       else _callable_class_name(radiation))
    if (radiation_scheme_ids(cfg) == (4, 4)
            and radiation_class == "gpuwm.core.rrtmgp.RRTMGPRadiation"):
        roles.extend(("rrtmgp_gas_lw", "rrtmgp_gas_sw",
                      "rrtmgp_cloud_lw", "rrtmgp_cloud_sw",
                      "rrtmgp_rfmip"))
    if (radiation_scheme_ids(cfg) == (4, 4)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        roles.extend(("wrf_rrtmg_lw_data", "wrf_rrtmg_lw_statics",
                      "wrf_rrtmg_sw_data",
                      "wrf_ozone_data", "wrf_ozone_lat",
                      "wrf_ozone_plev"))
    if radiation_scheme_ids(cfg)[0] == 1:
        roles.append("wrf_rrtm_data")
    land_scheme = int(cfg.sf_surface_physics)
    if land_scheme != 0:
        try:
            _attribute, land_roles = \
                LAND_SURFACE_PARAMETER_SOURCES[land_scheme]
        except KeyError:
            raise RestartManifestError(
                f"land-surface scheme {land_scheme} has no packaged-asset "
                "roles in LAND_SURFACE_PARAMETER_SOURCES "
                "(gpuwm/io/restart.py); its table bytes cannot be bound to "
                "the checkpoint") from None
        roles.extend(land_roles)
    cumulus = None if driver is None else driver.cumulus_callable
    cumulus_class = (None if cumulus is None
                     else _callable_class_name(cumulus))
    if (int(cfg.cu_physics) == 1
            and cumulus_class == "gpuwm.core.kf.KainFritsch"):
        roles.append("kf_lutab")
    identity = {}
    for role in roles:
        relative = PHYSICS_ASSET_PATHS[role]
        path = _PACKAGE_DIR / relative
        try:
            size = path.stat().st_size
            sha256 = _asset_sha256(path)
        except OSError as exc:
            raise RestartManifestError(
                f"active physics asset {role!r} is unreadable at {path}") \
                from exc
        identity[role] = {
            "path": relative.as_posix(),
            "bytes": int(size),
            "sha256": sha256,
        }
    return identity


def _scheme_algorithm(mapping: dict[int, str], scheme_id, label: str) -> str:
    try:
        return mapping[int(scheme_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"cannot identify unsupported {label} scheme {scheme_id!r}") \
            from exc


def _callable_class_name(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _callable_setup_identity(scheme, *, label: str,
                             expected_class: str | None) -> dict:
    """Identify a stock callable, or require a custom declaration.

    A class name alone cannot bind a closure/custom adapter's parameters.
    Non-stock callables therefore opt in with a strict-JSON
    ``restart_identity`` attribute (or zero-argument method).
    """
    if scheme is None:
        raise RestartManifestError(
            f"active {label} scheme has no callable to identify")
    class_name = _callable_class_name(scheme)
    identity = {"class": class_name}
    if expected_class is not None and class_name == expected_class:
        identity["implementation"] = "stock"
        return identity
    declared = getattr(scheme, "restart_identity", None)
    if callable(declared):
        declared = declared()
    if declared is None:
        raise RestartManifestError(
            f"custom {label} callable {class_name} must declare a strict-JSON "
            "restart_identity so restarts cannot cross incompatible code or "
            "parameters")
    identity["implementation"] = "custom"
    identity["declared_identity"] = _json_value(
        declared, f"{label}.restart_identity")
    return identity


def _float_mapping(value, label: str) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        items = value.items()
    except AttributeError as exc:
        raise RestartManifestError(f"{label} must be a mapping or None") \
            from exc
    result = {}
    for key, raw in sorted(items, key=lambda pair: str(pair[0])):
        if not isinstance(key, str):
            raise RestartManifestError(f"{label} key {key!r} is not a string")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise RestartManifestError(
                f"{label}[{key!r}] is not a numeric mole fraction: "
                f"{raw!r}") from exc
        if not np.isfinite(number):
            raise RestartManifestError(
                f"{label}[{key!r}] is non-finite: {raw!r}")
        result[key] = number
    return result


def _radiation_setup_identity(driver, cfg) -> dict:
    lw_id, sw_id = radiation_scheme_ids(cfg)
    legacy_rrtmg = ((lw_id, sw_id) == (4, 4)
                    and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY)
    if legacy_rrtmg:
        # The legacy port shares WRF scheme id 4 with the RTE+RRTMGP
        # substitution but is a different algorithm; its identity strings
        # are distinct so restarts cannot cross the two implementations.
        lw_algorithm = RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
        sw_algorithm = RRTMG_LEGACY_SW_ALGORITHM_IDENTITY
        lw_policy = sw_policy = RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
    elif lw_id == sw_id and lw_id in RADIATION_ALGORITHM_IDENTITIES:
        # Keep the historical public mapping authoritative for coupled
        # 0/0, 4/4, and 90/90 setups (including audit/test monkeypatches).
        lw_algorithm = sw_algorithm = _scheme_algorithm(
            RADIATION_ALGORITHM_IDENTITIES, lw_id, "radiation")
        lw_policy = sw_policy = _scheme_algorithm(
            RADIATION_ABOVE_ATMOSPHERE_POLICIES, lw_id,
            "radiation above-atmosphere policy")
    else:
        lw_algorithm = _scheme_algorithm(
            LONGWAVE_ALGORITHM_IDENTITIES, lw_id, "longwave radiation")
        sw_algorithm = _scheme_algorithm(
            SHORTWAVE_ALGORITHM_IDENTITIES, sw_id, "shortwave radiation")
        lw_policy = _scheme_algorithm(
            LONGWAVE_ABOVE_ATMOSPHERE_POLICIES, lw_id,
            "longwave above-atmosphere policy")
        sw_policy = _scheme_algorithm(
            SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES, sw_id,
            "shortwave above-atmosphere policy")
    algorithm = (lw_algorithm if lw_algorithm == sw_algorithm else
                 f"lw={lw_algorithm};sw={sw_algorithm}")
    policy = (lw_policy if lw_policy == sw_policy else
              f"lw={lw_policy};sw={sw_policy}")
    identity = {
        "scheme_id": lw_id if lw_id == sw_id else None,
        "scheme_ids": {"lw": lw_id, "sw": sw_id},
        "algorithm": algorithm,
        "algorithms": {"lw": lw_algorithm, "sw": sw_algorithm},
        "above_atmosphere_policy": policy,
        "above_atmosphere_policies": {"lw": lw_policy, "sw": sw_policy},
        "callable": None,
    }
    if not (lw_id or sw_id):
        return identity
    if driver is None:
        raise RestartManifestError(
            "active radiation cannot be restart-identified without an "
            "attached PhysicsDriver")
    scheme = driver.radiation_callable
    expected = {
        (4, 4): "gpuwm.core.rrtmgp.RRTMGPRadiation",
        (90, 90): "gpuwm.core.analytic_radiation.AnalyticClearSkyRadiation",
        (0, 1): "gpuwm.core.dudhia.DudhiaShortwaveRadiation",
    }.get((lw_id, sw_id))
    if legacy_rrtmg:
        # The stock legacy adapter landed with the integration wave; any
        # OTHER callable claiming to serve this selection must still
        # declare its own restart_identity.
        expected = "gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation"
    callable_identity = _callable_setup_identity(
        scheme, label="radiation", expected_class=expected)
    identity["callable"] = callable_identity
    if callable_identity["implementation"] == "custom":
        declared = callable_identity["declared_identity"]
        if not isinstance(declared, Mapping):
            raise RestartManifestError(
                "custom radiation restart_identity must be a mapping with "
                "explicit algorithm and above_atmosphere_policy entries")
        missing = sorted(
            {"algorithm", "above_atmosphere_policy"} - set(declared))
        if missing:
            raise RestartManifestError(
                f"custom radiation restart_identity is missing {missing}")
        for name in ("algorithm", "above_atmosphere_policy"):
            if not isinstance(declared[name], str) or not declared[name]:
                raise RestartManifestError(
                    f"custom radiation restart_identity[{name!r}] must be "
                    "a non-empty string")
        identity["configured_slot_algorithm"] = algorithm
        identity["algorithm"] = declared["algorithm"]
        identity["above_atmosphere_policy"] = \
            declared["above_atmosphere_policy"]
        return identity
    try:
        start_time = scheme.start_time
        latitude = scheme.latitude_deg
        longitude = scheme.longitude_deg
    except AttributeError as exc:
        raise RestartManifestError(
            "active radiation callable is missing start_time/latitude_deg/"
            "longitude_deg setup required by restart identity") from exc
    if not isinstance(start_time, datetime):
        raise RestartManifestError(
            "radiation start_time is not a datetime restart identity")
    identity.update({
        "start_time": start_time.isoformat(),
        "latitude": _array_setup_identity(latitude),
        "longitude": _array_setup_identity(longitude),
    })
    if (lw_id, sw_id) == (4, 4) and not legacy_rrtmg:
        try:
            identity.update({
                "column_chunk": int(scheme.column_chunk),
                "validation_mode": str(scheme.validation_mode),
                "trace_gas_policy": RRTMGP_TRACE_GAS_POLICY_IDENTITY,
                "trace_gas_overrides": _float_mapping(
                    scheme.trace_gas_overrides,
                    "radiation.trace_gas_overrides"),
                "trace_vmr": _float_mapping(
                    scheme.trace_vmr, "radiation.trace_vmr"),
                "ozone_log_pressure": _array_setup_identity(
                    scheme._ozone_logp),
                "ozone_vmr": _array_setup_identity(scheme._ozone_vmr),
                "coefficient_tables": {
                    "gas_lw": _resolved_object_setup_identity(
                        scheme.lw_tables, "RRTMGP LW gas table"),
                    "gas_sw": _resolved_object_setup_identity(
                        scheme.sw_tables, "RRTMGP SW gas table"),
                    "cloud_lw": _resolved_object_setup_identity(
                        scheme.lw_cloud_tables, "RRTMGP LW cloud table"),
                    "cloud_sw": _resolved_object_setup_identity(
                        scheme.sw_cloud_tables, "RRTMGP SW cloud table"),
                },
                "chunk_workspace": _rrtmgp_workspace_identity(
                    getattr(scheme, "chunk_workspace", None)),
            })
        except AttributeError as exc:
            raise RestartManifestError(
                "RRTMGP radiation callable is missing resolved gas/ozone/"
                "execution setup required by restart identity") from exc
    elif (lw_id, sw_id) == (90, 90):
        # These are module constants used directly by the analytic proxy.
        # Pin their resolved numerical values as well as its algorithm tag.
        from gpuwm.core.analytic_radiation import (
            CLEAR_SKY_TRANSMISSIVITY, SOLAR_CONSTANT_WM2,
            STEFAN_BOLTZMANN)
        identity["constants"] = {
            "clear_sky_transmissivity": float(CLEAR_SKY_TRANSMISSIVITY),
            "solar_constant_wm2": float(SOLAR_CONSTANT_WM2),
            "stefan_boltzmann": float(STEFAN_BOLTZMANN),
        }
    elif (lw_id, sw_id) == (0, 1):
        try:
            identity["dudhia"] = {
                "swrad_scat": float(scheme.swrad_scat),
                "icloud": int(scheme.icloud),
                "supported_path": "no-chem,no-eclipse,no-slope",
                "oracle": "WRF-v4.6.1 phys/module_ra_sw.F:SWRAD/SWPARA",
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise RestartManifestError(
                "Dudhia radiation callable is missing resolved setup "
                "required by restart identity") from exc
    return identity


def _land_surface_parameters_identity(cfg, driver) -> dict:
    """Identify the ACTIVE land-surface scheme's own parameter bundle.

    Dispatches on the scheme value through
    :data:`LAND_SURFACE_PARAMETER_SOURCES`; an unregistered scheme fails
    instead of falling through to Noah's bundle, which would write a
    checkpoint claiming Noah's tables for a run that never used them.
    """
    scheme = int(cfg.sf_surface_physics)
    try:
        attribute, _roles = LAND_SURFACE_PARAMETER_SOURCES[scheme]
    except KeyError:
        raise RestartManifestError(
            f"land-surface scheme {scheme} has no parameter-bundle row in "
            "LAND_SURFACE_PARAMETER_SOURCES (gpuwm/io/restart.py); a "
            "checkpoint cannot identify the tables it ran with") from None
    params = getattr(driver, attribute, None)
    label = LAND_SURFACE_ALGORITHM_IDENTITIES[scheme]
    payload = _packed_parameters_identity(params, label=label,
                                          attribute=attribute)
    geometry = getattr(driver, "noahmp_geometry", None)
    if geometry is not None:
        # Noah-MP reads COSZ, XLAT, JULIAN and YR, none of which any other
        # part of the checkpoint records.  Resuming with a different start
        # time or latitude grid would silently continue a different
        # trajectory, so the geometry is bound here and the digest is
        # recomputed over the merged payload.
        declared = getattr(geometry, "restart_identity", None)
        if declared is None:
            raise RestartManifestError(
                "Noah-MP solar geometry must declare a strict-JSON "
                "restart_identity")
        payload.pop("sha256", None)
        payload["solar_geometry"] = _json_value(
            declared, f"{attribute}.solar_geometry")
        payload["sha256"] = _json_sha256(payload)
    return payload


def _packed_parameters_identity(params, *, label: str,
                                attribute: str = "noah_params") -> dict:
    if params is None:
        raise RestartManifestError(
            f"active land surface scheme {label!r} has no resolved "
            f"parameters on PhysicsDriver.{attribute}")
    class_name = _callable_class_name(params)
    if not dataclasses.is_dataclass(params):
        declared = getattr(params, "restart_identity", None)
        if callable(declared):
            declared = declared()
        if declared is None:
            raise RestartManifestError(
                f"custom land-surface parameters {class_name} must declare "
                "a strict-JSON restart_identity")
        payload = {
            "class": class_name,
            "declared_identity": _json_value(
                declared, f"{attribute}.restart_identity"),
        }
        payload["sha256"] = _json_sha256(payload)
        return payload
    arrays = {}
    values = {}
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if _is_array_like(value):
            arrays[field.name] = _array_setup_identity(value)
        else:
            values[field.name] = _json_value(
                value, f"{attribute}.{field.name}")
    payload = {"class": class_name, "arrays": arrays, "values": values}
    payload["sha256"] = _json_sha256(payload)
    return payload


def _configuration_fingerprint(cfg) -> str:
    values = {key: value for key, value in dataclasses.asdict(cfg).items()
              if key not in CONFIG_RUN_LENGTH_FIELDS
              and key not in CONFIG_DIAGNOSTIC_FIELDS}
    return _json_sha256(_json_value(values, "RunConfig"))


def _thompson_table_identity(path) -> dict:
    """Validate and identify every external classic-Thompson table byte.

    The runtime table owner performs the stronger parse/upload/round-trip
    gate.  Restart identity deliberately validates the source assets again:
    it must reject a same-path byte replacement before mutating live state,
    without allocating the roughly 380 MiB host table payload merely to
    construct the header.
    """
    from gpuwm.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS,
        TABLE_SET_ID,
        WRF_REFERENCE_COMMIT,
        WRF_REFERENCE_VERSION,
        validate_table_assets,
    )

    assets = validate_table_assets(path)
    if assets != CLASSIC_TABLE_ASSETS:
        raise RestartManifestError(
            "validated Thompson table assets are not the canonical set")
    return {
        "schema": 1,
        "table_set": TABLE_SET_ID,
        "wrf_version": WRF_REFERENCE_VERSION,
        "wrf_commit": WRF_REFERENCE_COMMIT,
        "assets": [
            {"filename": item.filename, "bytes": int(item.bytes),
             "sha256": item.sha256}
            for item in assets
        ],
    }


def _thompson_setup_identity() -> dict:
    """Resolved implementation/table identity for ``mp_physics=8``.

    The table root resolves exactly as the forecast adapter resolves it
    (:func:`gpuwm.physics_compat.thompson_table_root`: the packaged
    ``gpuwm/data/thompson/tables`` directory unless
    ``GPUWM_THOMPSON_TABLE_ROOT`` overrides it), so the identity written
    into a checkpoint names the bytes the trajectory actually loaded.
    The ``admission`` token replaces the retired
    ``GPUWM_EXPERIMENTAL_THOMPSON_MP8=1`` ``implementation_guard`` entry
    (mp8 promotion to first-class, product/v1 packaging lane 2026-07-28):
    checkpoints written under the guarded runtime fail the physics-setup
    equality check rather than silently continuing across the promotion.
    """
    from gpuwm.physics_compat import thompson_table_root

    root = thompson_table_root()
    try:
        tables = _thompson_table_identity(root)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"active Thompson table identity is invalid at {root}") from exc
    return {
        "admission": "first-class-mp8-packaged-tables-v1",
        "tables": _json_value(tables, "Thompson table identity"),
        "graupel_number_policy": (
            "wrf-private-classic-ng-reconstructed-and-transported-per-call-v1"),
        "reflectivity_policy": (
            "wrf-v4.6.1-calc-refl10cm-post-fallout-output-only-v1"),
        "snow_rime_conversion_policy": (
            "wrf-v4.6.1-prs-scw-prg-scw-png-scw-held-number-v1"),
        "snow_fall_speed_policy": (
            "wrf-v4.6.1-deposition-conditioned-vts-boost-same-call-v1"),
    }


def _thompson_aerosol_setup_identity() -> dict:
    """Resolved implementation/table identity for ``mp_physics=28``.

    mp=8 has carried a ``thompson`` sub-record since it landed, and without
    the parallel record here an mp=28 checkpoint would bind NO table bytes
    at all: it would resume against a different ``freezeH2O.dat`` or a
    different ``CCN_ACTIVATE.BIN`` with every identity check passing.  The
    scheme is table-driven to an unusual degree -- the activation fraction
    that sets droplet number is READ from ``tnccn_act``, not computed
    (module_mp_thompson.F:5229-5230 index it at fixed radius/kappa) -- so a
    silent table substitution is a silent trajectory substitution.

    Two inventories, deliberately kept apart.  ``classic_tables`` is the
    SAME four-asset set mp=8 pins, because mp=28 genuinely loads them (its
    adapter calls ``load_classic_device_tables`` and reuses the frozen mp=8
    sedimentation and classic-graupel launchers unchanged).
    ``aerosol_tables`` is the one asset only mp=28 reads, addressed by the
    path the run resolved rather than by ``root / filename``: gpuwm ships the
    blob since 2026-08-01 but the file and root overrides let a run bind to a
    copy in a WRF ``run/`` directory instead, so its LOCATION is not part of
    the identity but its BYTES are.  Neither is the fact that it ships --
    see the note beside the record; a packaging fact in a trajectory identity
    only buys refused resumes.

    The ``aerosol_source`` token records that this build's aerosol initial
    condition is ``thompson_init``'s synthetic CCN/IN profile
    (module_mp_thompson.F:493-551) at ``aer_init_opt = wif_input_opt = 0``.
    A future WIF metgrid ingest is a different initial condition and must
    move this token rather than resume onto a checkpoint written without it.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_SET_ID,
        resolve_aerosol_table_root,
        resolve_ccn_activation_path,
        validate_ccn_activation_asset,
    )
    from gpuwm.physics_compat import thompson_table_root

    root = thompson_table_root()
    try:
        classic = _thompson_table_identity(root)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"active Thompson table identity is invalid at {root}") from exc
    try:
        # Resolve from the SAME root the adapter used
        # (gpuwm/core/microphysics_aerosol.py:182 takes
        # ``physics_compat.thompson_table_root()`` and hands it to BOTH table
        # owners, so tnccn_act and the four classic caches cannot come from
        # different WRF builds).  Passing ``root`` explicitly rather than
        # letting the aerosol contract re-resolve it keeps that guarantee
        # even if the two resolution orders ever diverge.  The
        # ``GPUWM_THOMPSON_CCN_ACTIVATE`` file override is still honoured,
        # because that is the path the adapter would have loaded too.
        ccn_path = resolve_ccn_activation_path(
            None, resolve_aerosol_table_root(root))
        ccn_asset = validate_ccn_activation_asset(ccn_path)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            "mp_physics=28 restart identity requires the resolved "
            f"CCN activation table: {exc}") from exc
    return {
        "admission": "component-override-mp28-unvendored-ccn-table-v1",
        "classic_tables": _json_value(
            classic, "Thompson table identity"),
        "aerosol_tables": {
            "schema": 1,
            "table_set": AEROSOL_TABLE_SET_ID,
            # DELIBERATELY ABSENT: ``AEROSOL_ASSET_REDISTRIBUTED``.  This
            # dict is hashed into ``physics_setup_fingerprint``, so anything
            # placed here becomes part of the trajectory identity and a
            # change to it refuses every earlier checkpoint.  Whether the
            # blob arrives inside the wheel or from a WRF ``run/`` directory
            # is a PACKAGING fact: it cannot move a single float.  It was
            # briefly written here on 2026-08-01 and removed the same day,
            # because flipping the constant False -> True would have broken
            # resume for every mp=28 checkpoint in existence while the bound
            # bytes stayed identical.  The ``sha256`` below is what actually
            # determines the trajectory and it is unaffected by delivery.
            # Do not re-add it; the constant is published on the registry row
            # (``redistributed_by_gpuwm``), which is where a packaging fact
            # belongs.
            "assets": [{
                "filename": ccn_asset.filename,
                "bytes": int(ccn_asset.bytes),
                "sha256": ccn_asset.sha256,
            }],
        },
        "aerosol_source": (
            "wrf-v4.6.1-thompson-init-synthetic-ccn-in-profile-"
            "aer-init-opt-0-wif-input-opt-0-v1"),
        "graupel_number_policy": (
            "wrf-private-classic-ng-reconstructed-and-transported-per-call-v1"),
        "reflectivity_policy": (
            "wrf-v4.6.1-calc-refl10cm-post-fallout-output-only-v1"),
        "aerosol_tendency_policy": (
            "wrf-v4.6.1-single-terminal-ncten-nwfaten-nifaten-apply-and-"
            "clamp-then-unclamped-surface-emission-v1"),
    }


def physics_setup_identity(state, cfg) -> dict:
    """Return the complete JSON-able trajectory-defining physics setup.

    The ordinary config echo pins all configured knobs.  This resolves the
    remaining runtime inputs that config alone cannot prove: callable
    implementation/policy, radiation calendar/grid/gases/ozone, packed Noah
    parameters, selected Morrison constants, resolved driver cadence, and
    the byte digests of every packaged table active on this trajectory.
    """
    driver = getattr(state, "physics", None)
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ((ra_lw_physics, ra_sw_physics) == (4, 4)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        # The legacy port shares WRF scheme id 4 with the RTE+RRTMGP
        # substitution but is a different algorithm: the SUMMARY
        # identities must be as distinct as the detailed ones, or a
        # legacy restart would advertise itself as rte-rrtmgp-v1 at the
        # top level (integration finding, 2026-07-27).
        lw_algorithm = RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
        sw_algorithm = RRTMG_LEGACY_SW_ALGORITHM_IDENTITY
    elif (ra_lw_physics == ra_sw_physics
            and ra_lw_physics in RADIATION_ALGORITHM_IDENTITIES):
        lw_algorithm = sw_algorithm = _scheme_algorithm(
            RADIATION_ALGORITHM_IDENTITIES, ra_lw_physics, "radiation")
    else:
        lw_algorithm = _scheme_algorithm(
            LONGWAVE_ALGORITHM_IDENTITIES, ra_lw_physics,
            "longwave radiation")
        sw_algorithm = _scheme_algorithm(
            SHORTWAVE_ALGORITHM_IDENTITIES, ra_sw_physics,
            "shortwave radiation")
    algorithms = {
        "physics_driver": PHYSICS_DRIVER_ALGORITHM_IDENTITY,
        "microphysics": _scheme_algorithm(
            MICROPHYSICS_ALGORITHM_IDENTITIES, cfg.mp_physics,
            "microphysics"),
        "surface_layer": _scheme_algorithm(
            SURFACE_LAYER_ALGORITHM_IDENTITIES, cfg.sf_sfclay_physics,
            "surface layer"),
        "land_surface": _scheme_algorithm(
            LAND_SURFACE_ALGORITHM_IDENTITIES, cfg.sf_surface_physics,
            "land surface"),
        "pbl": _scheme_algorithm(
            PBL_ALGORITHM_IDENTITIES, cfg.bl_pbl_physics, "PBL"),
        "radiation": (lw_algorithm if lw_algorithm == sw_algorithm else
                      f"lw={lw_algorithm};sw={sw_algorithm}"),
        "radiation_lw": lw_algorithm,
        "radiation_sw": sw_algorithm,
        "cumulus": _scheme_algorithm(
            CUMULUS_ALGORITHM_IDENTITIES, cfg.cu_physics, "cumulus"),
    }
    microphysics = {"scheme_id": int(cfg.mp_physics)}
    if int(cfg.mp_physics) == 6:
        from gpuwm.core.wsm6_constants import rimed_ice_constants
        selection = int(cfg.wsm6_hail_opt)
        constants = rimed_ice_constants(selection)
        microphysics["wsm6_rimed_ice"] = {
            "selection": selection,
            "n0g": float(constants.n0g),
            "deng": float(constants.deng),
            "avtg": float(constants.avtg),
            "bvtg": float(constants.bvtg),
            "lamdagmax": float(constants.lamdagmax),
        }
    if int(cfg.mp_physics) == 10:
        from gpuwm.core.morrison_constants import rimed_ice_constants
        selection = int(cfg.morr_rimed_ice)
        constants = rimed_ice_constants(selection)
        microphysics["morrison_rimed_ice"] = {
            "selection": selection,
            "ag": float(constants.ag),
            "bg": float(constants.bg),
            "rhog": float(constants.rhog),
            "cg": float(constants.cg),
        }
    if int(cfg.mp_physics) == 8:
        microphysics["thompson"] = _thompson_setup_identity()
    if int(cfg.mp_physics) == 28:
        microphysics["thompson_aerosol"] = _thompson_aerosol_setup_identity()
    if int(cfg.mp_physics) == 18:
        microphysics["restart_contract"] = \
            _nssl2_restart_contract_identity()

    land_surface = {
        "scheme_id": int(cfg.sf_surface_physics),
        "parameters": None,
    }
    if int(cfg.sf_surface_physics) != 0:
        if driver is None:
            raise RestartManifestError(
                "active land-surface scheme cannot be restart-identified "
                "without an attached PhysicsDriver")
        land_surface["parameters"] = _land_surface_parameters_identity(
            cfg, driver)

    cumulus = {
        "scheme_id": int(cfg.cu_physics),
        "callable": None,
        "coefficient_table": None,
    }
    if int(cfg.cu_physics) != 0:
        if driver is None:
            raise RestartManifestError(
                "active cumulus cannot be restart-identified without an "
                "attached PhysicsDriver")
        expected = ("gpuwm.core.kf.KainFritsch"
                    if int(cfg.cu_physics) == 1 else None)
        callable_identity = _callable_setup_identity(
            driver.cumulus_callable, label="cumulus",
            expected_class=expected)
        cumulus["callable"] = callable_identity
        if callable_identity["implementation"] == "stock":
            from gpuwm.core.kf import load_kf_table
            cumulus["coefficient_table"] = \
                _resolved_object_setup_identity(
                    load_kf_table(), "Kain-Fritsch lookup table")

    driver_identity = {
        "attached": driver is not None,
        "class": None if driver is None else _callable_class_name(driver),
        "resolved_schemes": None,
        "cadence": None,
    }
    if driver is not None:
        driver_identity["resolved_schemes"] = {
            "mp_physics": int(driver.mp_physics),
            "ra_physics": int(driver.ra_physics),
            "ra_lw_physics": int(driver.ra_lw_physics),
            "ra_sw_physics": int(driver.ra_sw_physics),
            "radiation_active": bool(driver.radiation_active),
            "cu_physics": int(driver.cu_physics),
            "surface_enabled": bool(driver.surface_enabled),
        }
        cadence = {}
        for name in ("bldt_seconds", "stepbl", "radt_minutes",
                     "radt_seconds", "stepra", "cudt_minutes",
                     "cudt_seconds", "stepcu"):
            cadence[name] = _json_value(
                getattr(driver, name), f"PhysicsDriver.{name}")
        driver_identity["cadence"] = cadence

    return {
        "schema_version": PHYSICS_SETUP_SCHEMA_VERSION,
        "configuration_sha256": _configuration_fingerprint(cfg),
        "algorithms": algorithms,
        "driver": driver_identity,
        "microphysics": microphysics,
        "radiation": _radiation_setup_identity(driver, cfg),
        "land_surface": land_surface,
        "cumulus": cumulus,
        "assets": _active_asset_identity(cfg, driver),
    }


def physics_setup_fingerprint(state, cfg) -> str:
    """SHA-256 of :func:`physics_setup_identity`."""
    return _json_sha256(physics_setup_identity(state, cfg))


def _callable_state_check(scheme, allowed_arrays: frozenset,
                          allowed_containers: frozenset,
                          label: str) -> None:
    """Enforce manifest coverage for arrays on a scheme callable.

    Covers direct array attributes, dict-valued attributes (every value),
    and one level of object containers, so an adapter stashing per-call
    state in a dict or sub-object (the ``driver.cu_rates`` pattern) is
    caught instead of silently skipping the stream.  Containers that
    legitimately carry arrays must be classified in the ``*_CONTAINERS``
    allowlists above.
    """
    for name, value in getattr(scheme, "__dict__", {}).items():
        if _is_array_like(value):
            if name not in allowed_arrays:
                raise RestartManifestError(
                    f"{label} callable attribute {name!r} is an "
                    "unclassified array: add it to the restart manifest "
                    "(gpuwm/io/restart.py) as serialized or setup state")
            continue
        if name in allowed_containers:
            continue
        if isinstance(value, dict):
            arrays = sorted(str(key) for key, item in value.items()
                            if _is_array_like(item))
            if arrays:
                raise RestartManifestError(
                    f"{label} callable dict attribute {name!r} carries "
                    f"unclassified arrays {arrays}: classify the container "
                    "in gpuwm/io/restart.py or serialize its state")
            continue
        nested = getattr(value, "__dict__", None)
        if nested and any(_is_array_like(item) for item in nested.values()):
            raise RestartManifestError(
                f"{label} callable attribute {name!r} is an object "
                "container carrying unclassified arrays: classify it in "
                "gpuwm/io/restart.py (rebuild-on-load) or serialize its "
                "state")


def _require_dataclass_components(container, expected, label: str) -> None:
    """Pin a serialized dataclass's fields to its component manifest.

    A field added to :class:`PhysicsTendencies` or
    :class:`MicrophysicsDiagnostics` without a manifest update would
    silently serialize nothing; this makes every write (and the CPU
    manifest tests) fail instead.
    """
    names = {field.name for field in dataclasses.fields(container)}
    expected = set(expected)
    if names != expected:
        raise RestartManifestError(
            f"{label} dataclass fields do not match the restart component "
            f"manifest (gpuwm/io/restart.py): unclassified "
            f"{sorted(names - expected)}, stale {sorted(expected - names)}")


def state_manifest(state) -> dict[str, object]:
    """Serialized ``state/<name>`` arrays for this state's configuration.

    Walks EVERY instance attribute through :func:`classify_state_attr`, so
    an unclassified attribute raises here (and therefore in every
    ``write_restart`` call and in the manifest tests).
    """
    manifest = {}
    for name in sorted(vars(state)):
        kind = classify_state_attr(name)
        value = getattr(state, name)
        if kind == "serialize" and value is not None:
            manifest[f"state/{name}"] = value
    return manifest


def _scratch_manifest(state) -> dict[str, object]:
    pool = getattr(state, "_scratch", {})
    manifest = {}
    for slot in sorted(pool):
        if classify_scratch_slot(slot) == "serialize":
            manifest[f"scratch/{slot}"] = pool[slot]
    return manifest


def _driver_manifest(driver) -> dict[str, object]:
    """Serialized driver arrays; enforces driver attribute coverage."""
    for name in sorted(vars(driver)):
        if (name not in DRIVER_SERIALIZED_ATTRS
                and name not in DRIVER_REBUILT_ATTRS):
            raise RestartManifestError(
                f"PhysicsDriver attribute {name!r} is not classified in "
                "the restart manifest (gpuwm/io/restart.py): declare it "
                "serialized or rebuilt")
    # Normal compute() finalizes the transient KF expiry mask immediately
    # after composing this step's RK target and before Morrison.  A surviving
    # mask therefore denotes an incomplete synthetic/direct-driver transition;
    # reject it instead of serializing a state whose persistent rates and held
    # coupled tendencies have not yet received their required simultaneous
    # clear.  ``cu_expiring`` itself remains rebuild-only scratch.
    cu_expiring = getattr(driver, "cu_expiring", None)
    if (bool(getattr(driver, "_cu_expiry_pending", False))
            or (cu_expiring is not None and bool(cu_expiring.any()))):
        raise RestartManifestError(
            "cannot write restart while KF expiry finalization is pending; "
            "call PhysicsDriver.finish_step() before checkpointing")
    manifest = {
        "driver/rthratenlw": driver.rthratenlw,
        "driver/rthratensw": driver.rthratensw,
        "driver/pending_rainbl": driver._pending_rainbl,
    }
    for tend_name in DRIVER_TENDENCY_ATTRS:
        tend = getattr(driver, tend_name)
        _require_dataclass_components(
            tend, TENDENCY_COMPONENTS, f"{tend_name} (PhysicsTendencies)")
        for comp in TENDENCY_COMPONENTS:
            value = getattr(tend, comp)
            if value is not None:
                manifest[f"driver/{tend_name}/{comp}"] = value
    _require_dataclass_components(
        driver.microphysics, MICROPHYSICS_COMPONENTS,
        "MicrophysicsDiagnostics")
    from gpuwm.core.physics import microphysics_scratch_slots
    micro_slots = dict(microphysics_scratch_slots(driver.mp_physics))
    for comp, slot in micro_slots.items():
        scratch = getattr(driver.state, "_scratch", {}).get(slot)
        if scratch is None or getattr(driver.microphysics, comp) is not scratch:
            raise RestartManifestError(
                f"driver.microphysics.{comp} does not alias canonical scratch "
                f"slot {slot!r}: restart v{RESTART_FORMAT_VERSION} writes "
                "exactly one microphysics accumulator set")
    for comp in set(MICROPHYSICS_COMPONENTS) - set(micro_slots):
        if driver.mp_physics and getattr(driver.microphysics, comp) is not None:
            raise RestartManifestError(
                f"driver.microphysics.{comp} is populated but has no canonical "
                f"scratch slot for mp_physics={driver.mp_physics}")
    for name in sorted(driver.fields):
        manifest[f"fields/{name}"] = driver.fields[name]
    # The in-place fields restore relies on SFClayResult aliasing the
    # fields-dict arrays; verify the alias contract (and that every
    # SFClayResult field IS a fields entry) at every write.
    for field in (() if driver.sfclay_result is None
                  else dataclasses.fields(driver.sfclay_result)):
        if driver.fields.get(field.name) is not getattr(
                driver.sfclay_result, field.name):
            raise RestartManifestError(
                f"sfclay_result.{field.name} does not alias "
                f"fields[{field.name!r}]: the in-place surface-field "
                "restore depends on that aliasing")
    for attribute, suffix in (
            ("mynn_sfclay_result", ""),
            ("mynn_sfclay_sea_result", "_sea")):
        result = getattr(driver, attribute)
        for field in (() if result is None else dataclasses.fields(result)):
            key = f"{field.name}{suffix}"
            if driver.fields.get(key) is not getattr(result, field.name):
                raise RestartManifestError(
                    f"{attribute}.{field.name} does not alias fields[{key!r}]: "
                    "the in-place MYNN surface-field restore depends on that "
                    "aliasing")
    _callable_state_check(driver.radiation_callable,
                          RADIATION_CALLABLE_ARRAYS,
                          RADIATION_CALLABLE_CONTAINERS, "radiation")
    _callable_state_check(driver.cumulus_callable,
                          CUMULUS_CALLABLE_ARRAYS,
                          CUMULUS_CALLABLE_CONTAINERS, "cumulus")
    w0avg = getattr(driver.cumulus_callable, "w0avg", None)
    if w0avg is not None:
        manifest["cumulus/w0avg"] = w0avg
    o33d_grid = getattr(driver.radiation_callable, "_o33d_grid", None)
    if o33d_grid is not None:
        # Legacy-RRTMG root o33d field (WRF's restart-carried O3RAD
        # analogue): serialized so child-domain ozone routing resumes
        # bit-identically (see RADIATION_CALLABLE_ARRAYS note).
        manifest["radiation/o33d_grid"] = o33d_grid
    return manifest


def root_external_lbc_clock_identity(state, cfg) -> str | None:
    """The root external-LBC clock semantic active on this state.

    ``None`` for domains without an external Davies consumer (nested
    children on rolling mirrors, periodic/open cases).  For specified
    domains: :data:`ROOT_EXTERNAL_LBC_CLOCK_IDENTITY` when the attached
    external mirror is bound to a DomainClock (production tree build /
    N5S builder), :data:`ROOT_EXTERNAL_LBC_CLOCK_LEGACY` otherwise
    (legacy direct paths, or pre-attachment shim states).
    """
    if not getattr(cfg, "specified", False):
        return None
    resident = getattr(state, "_lateral_boundary_device", None)
    bound = (resident is not None and not getattr(resident, "rolling", False)
             and getattr(resident, "clock", None) is not None)
    return (ROOT_EXTERNAL_LBC_CLOCK_IDENTITY if bound
            else ROOT_EXTERNAL_LBC_CLOCK_LEGACY)


def write_restart(path, state, cfg, *, run_trackers=None,
                  tree_header: dict | None = None,
                  sealed_forcing_extension: bool = False) -> Path:
    """Serialize the complete cross-step model state to ``path``.

    ``run_trackers`` (optional JSON-able dict) carries the caller's
    run-summary bookkeeping (w-max trackers, SWDOWN peak, nan flag) so a
    resumed run reports the same summary as an uninterrupted one — model
    evolution itself never reads them.
    """
    path = Path(path)
    _validate_nssl2_live_restart_state(state, cfg)
    _validate_thompson_aerosol_live_restart_state(state, cfg)
    if sealed_forcing_extension:
        _require_sealable_forcing_prefix(
            state, cfg, path=path,
            elapsed=_admissible_elapsed_seconds(
                state.elapsed_seconds, "sealed restart write"))
    manifest: dict[str, object] = {}
    manifest.update(state_manifest(state))
    manifest.update(_scratch_manifest(state))
    driver = getattr(state, "physics", None)
    driver_header = None
    if driver is not None:
        manifest.update(_driver_manifest(driver))
        driver_header = {
            "call_counts": {key: int(value)
                            for key, value in driver.call_counts.items()},
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "microphysics_updates": int(driver.microphysics_updates),
        }
    physics_setup = physics_setup_identity(state, cfg)
    physics_setup_sha256 = _json_sha256(physics_setup)

    arrays = {}
    array_manifest = {}
    for key in sorted(manifest):
        host = _host(manifest[key])
        arrays[key] = host
        array_manifest[key] = {"shape": list(host.shape),
                               "dtype": str(host.dtype)}
    header = {
        "format_version": RESTART_FORMAT_VERSION,
        "case": cfg.case,
        "created": datetime.now(timezone.utc).isoformat(),
        # The producer's own identity, so a checkpoint separated from its
        # logs can still say which build wrote it.  ``gpuwm.__version__``
        # comes from installed distribution metadata, so this is the release
        # that is speaking rather than a hand-maintained constant.
        "producer": producer_identity(),
        "elapsed_seconds": _admissible_elapsed_seconds(
            state.elapsed_seconds, "restart write"),
        "config": dataclasses.asdict(cfg),
        "setup_fingerprint": setup_fingerprint(state),
        "physics_setup": physics_setup,
        "physics_setup_fingerprint": physics_setup_sha256,
        "driver": driver_header,
        "run_trackers": (None if run_trackers is None
                         else dict(run_trackers)),
        "array_manifest": array_manifest,
    }
    if sealed_forcing_extension:
        header.update({
            "forcing_extension_mode": SEALED_FORCING_EXTENSION_MODE,
            "setup_core_fingerprint": setup_core_fingerprint(state),
            "lateral_boundary_prefix": lateral_boundary_prefix_identity(state),
        })
    lbc_clock_identity = root_external_lbc_clock_identity(state, cfg)
    if lbc_clock_identity is not None:
        header["root_external_lbc_clock"] = lbc_clock_identity
    if tree_header is not None:
        overlap = set(header) & set(tree_header)
        if overlap:
            raise ValueError(
                f"tree restart header may not replace base keys "
                f"{sorted(overlap)}")
        header.update(dict(tree_header))
    # ``allow_nan=False``: Python's json emits bare ``NaN``/``Infinity``
    # tokens by default, which are not JSON, and the reader accepts them
    # back.  A header that cannot express a non-finite clock is a header
    # whose clock cannot poison resume arithmetic after every identity
    # check has already passed.
    payload = {_HEADER_KEY: np.frombuffer(
        json.dumps(header, allow_nan=False).encode("utf-8"),
        dtype=np.uint8)}
    payload.update(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: a crash mid-write must not leave a truncated file
    # under the valid gpuwmrst name (review F4).
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.savez(stream, **payload)
        # Atomic visibility was already sound; durability was not.  Closing
        # a file leaves its bytes in the page cache, so a machine or volume
        # crash could expose the published name with content that never
        # reached the disk.  The wrfout writer already fsyncs before it
        # replaces; the checkpoint did not, and a checkpoint is the thing a
        # crash is supposed to leave behind.
        fsync_file(temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def _decode_header(data, path) -> dict:
    if _HEADER_KEY not in data.files:
        raise RestartMismatchError(
            f"{path} is not a gpuwm restart file (missing header)")
    return json.loads(bytes(bytearray(data[_HEADER_KEY])).decode("utf-8"))


def _load_restart(path, *, with_arrays: bool):
    """Load header (and optionally arrays), wrapping corruption loudly."""
    try:
        with np.load(path, allow_pickle=False) as data:
            header = _decode_header(data, path)
            stored = ({key: data[key] for key in data.files
                       if key != _HEADER_KEY} if with_arrays else None)
    except RestartMismatchError:
        raise
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
        raise RestartMismatchError(
            f"gpuwm restart file {path} is unreadable (truncated or "
            "corrupt archive — likely an interrupted copy or a crash "
            "mid-write; gpuwm itself publishes restart files atomically "
            "via a .tmp rename)") from exc
    return header, stored


def read_restart_header(path) -> dict:
    """Return the JSON header (config echo, clock, manifest, trackers)."""
    return _load_restart(Path(path), with_arrays=False)[0]


def _require_config_match(stored_config: dict, cfg, path) -> None:
    live_config = dataclasses.asdict(cfg)
    absent = object()
    differences = []
    for key in sorted(set(stored_config) | set(live_config)):
        if key in CONFIG_RUN_LENGTH_FIELDS:
            continue
        if key in CONFIG_DIAGNOSTIC_FIELDS:
            continue
        stored = stored_config.get(key, absent)
        live = live_config.get(key, absent)
        if key == "ra_rrtmg_variant" and stored is absent:
            # Migration rule (2026-07-27 assembly dossier): a v5 header
            # written before the radiation-variant field existed could
            # only have run the RTE+RRTMGP substitution, so restore it
            # under that value.  An explicitly STORED variant that
            # mismatches the live config stays fail-closed below --
            # never infer legacy, never widen.
            stored = RRTMG_VARIANT_RTE_RRTMGP
        if stored is not live and stored != live:
            differences.append(
                f"{key}: restart={stored!r} run={live!r}")
    if differences:
        raise RestartMismatchError(
            f"restart file {path} was written under a different "
            "configuration; refusing to continue a different model:\n  "
            + "\n  ".join(differences))


def _require_physics_setup_match(header: dict, state, cfg, path) -> None:
    """Validate stored self-consistency and live physics identity."""
    stored = header["physics_setup"]
    stored_fingerprint = header["physics_setup_fingerprint"]
    if not isinstance(stored, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed physics setup identity")
    try:
        computed_stored_fingerprint = _json_sha256(stored)
    except (TypeError, ValueError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has a malformed physics setup identity") \
            from exc
    if (not isinstance(stored_fingerprint, str)
            or stored_fingerprint != computed_stored_fingerprint):
        raise RestartMismatchError(
            f"restart file {path} physics setup fingerprint does not match "
            "its own stored identity")
    if stored.get("schema_version") != PHYSICS_SETUP_SCHEMA_VERSION:
        raise RestartMismatchError(
            f"restart file {path} physics setup schema "
            f"{stored.get('schema_version')!r} is not supported by this "
            f"build (expected {PHYSICS_SETUP_SCHEMA_VERSION})")
    try:
        live = physics_setup_identity(state, cfg)
        live_fingerprint = _json_sha256(live)
    except RestartManifestError as exc:
        raise RestartMismatchError(
            f"resuming model has no complete physics setup identity: {exc}") \
            from exc
    if stored_fingerprint != live_fingerprint or stored != live:
        raise RestartMismatchError(
            f"restart file {path} was written under a different physics "
            "setup (algorithm/policy, resolved radiation/gas/Noah values, "
            "driver cadence, or active asset digest mismatch); rebuild the "
            "identical physics preparation before restoring")


def _require_nssl2_restart_contract(header: dict, cfg, path) -> None:
    """Reject an absent, stale, or extended MP18 nested schema."""
    if int(cfg.mp_physics) != 18:
        return
    setup = header.get("physics_setup")
    try:
        actual = setup["microphysics"]["restart_contract"]
    except (KeyError, TypeError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has no versioned MP18 restart contract") \
            from exc
    if not isinstance(actual, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed MP18 restart contract")
    version = actual.get("schema_version")
    if version != NSSL2_RESTART_CONTRACT_VERSION:
        raise RestartMismatchError(
            f"restart file {path} has MP18 restart contract version "
            f"{version!r}; expected {NSSL2_RESTART_CONTRACT_VERSION}")
    if actual != _nssl2_restart_contract_identity():
        raise RestartMismatchError(
            f"restart file {path} MP18 restart contract does not exactly "
            "match the canonical state/timing inventory")


def _check_array(stored: np.ndarray, target, key: str) -> None:
    if tuple(stored.shape) != tuple(target.shape):
        raise RestartMismatchError(
            f"{key}: restart shape {tuple(stored.shape)} does not match "
            f"state shape {tuple(target.shape)}")
    if stored.dtype != target.dtype:
        raise RestartMismatchError(
            f"{key}: restart dtype {stored.dtype} does not match state "
            f"dtype {target.dtype}")


def _validate_nssl2_stored_restart_state(
        header: dict, stored: dict[str, np.ndarray], state, cfg,
        path, elapsed: float) -> None:
    """Hoist every MP18 inventory/timing refusal before restore mutation."""
    if int(cfg.mp_physics) != 18:
        return

    aliases = sorted(set(stored) & NSSL2_LEGACY_RESTART_ALIASES)
    if aliases:
        raise RestartMismatchError(
            f"restart file {path} uses legacy MP18 aliases {aliases}; only "
            "canonical Registry and scratch names are accepted")

    expected_state = {
        f"state/{name}" for name in STATE_SERIALIZED_ATTRS
        if getattr(state, name, None) is not None
    }
    stored_state = {key for key in stored if key.startswith("state/")}
    missing_state = sorted(expected_state - stored_state)
    extra_state = sorted(stored_state - expected_state)
    if missing_state or extra_state:
        raise RestartMismatchError(
            f"restart file {path} MP18 state inventory mismatch "
            f"(missing {missing_state}, extra {extra_state})")

    required_state = {
        *(f"state/{name}" for name in NSSL2_RESTART_PROGNOSTICS),
        *(f"state/{name}" for name in NSSL2_RESTART_AUXILIARY_STATE),
    }
    missing_required = sorted(required_state - stored_state)
    if missing_required:
        raise RestartMismatchError(
            f"restart file {path} omits canonical MP18 state "
            f"{missing_required}")

    pool = getattr(state, "_scratch", {})
    expected_scratch = {
        f"scratch/{slot}" for slot in pool
        if classify_scratch_slot(slot) == "serialize"
    }
    stored_scratch = {key for key in stored if key.startswith("scratch/")}
    missing_scratch = sorted(expected_scratch - stored_scratch)
    extra_scratch = sorted(stored_scratch - expected_scratch)
    if missing_scratch or extra_scratch:
        raise RestartMismatchError(
            f"restart file {path} MP18 scratch inventory mismatch "
            f"(missing {missing_scratch}, extra {extra_scratch})")
    required_precipitation = {
        f"scratch/{slot}" for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
    }
    missing_precipitation = sorted(
        required_precipitation - stored_scratch)
    if missing_precipitation:
        raise RestartMismatchError(
            f"restart file {path} omits MP18 precipitation state "
            f"{missing_precipitation}")

    for key in sorted(required_state):
        _check_array(stored[key], getattr(state, key[len("state/"):]), key)
    for key in sorted(required_precipitation):
        _check_array(stored[key], pool[key[len("scratch/"):]], key)

    driver = getattr(state, "physics", None)
    if driver is None or int(getattr(driver, "mp_physics", -1)) != 18:
        raise RestartMismatchError(
            "MP18 restart requires a prepared MP18 PhysicsDriver")
    driver_header = header.get("driver")
    updates = (None if not isinstance(driver_header, dict)
               else driver_header.get("microphysics_updates"))
    if (isinstance(updates, bool) or not isinstance(updates, int)
            or updates < 0):
        raise RestartMismatchError(
            "MP18 restart first-call authority microphysics_updates must be "
            f"a non-negative integer, got {updates!r}")
    if (isinstance(header.get("elapsed_seconds"), bool)
            or not math.isfinite(elapsed) or elapsed < 0.0):
        raise RestartMismatchError(
            "MP18 restart elapsed_seconds must be finite and non-negative")


def _asarray_like(state):
    """Return host->model-array converter for the state's array module."""
    if type(state.u).__module__.partition(".")[0] == "cupy":
        import cupy
        return cupy.asarray
    return lambda host: np.array(host, copy=True)


@dataclasses.dataclass(frozen=True)
class RestartInfo:
    """Restore result: the restored clock and the writer's run trackers."""

    elapsed_seconds: float
    run_trackers: dict | None
    header: dict


@dataclasses.dataclass(frozen=True)
class TreeRestartInfo:
    """Validated all-domain restore result."""

    elapsed_ticks: int
    tick_den: int
    phase: str
    paths_by_grid_id: dict[int, Path]
    headers_by_grid_id: dict[int, dict]


@dataclasses.dataclass(frozen=True)
class _ValidatedRestart:
    """One fully loaded member whose every semantic refusal has passed.

    Tree restore retains these payloads through the complete-set validation
    pass and applies these exact arrays afterward.  Reopening a member between
    validation and mutation would recreate the partial-restore/TOCTOU hole.
    """

    path: Path
    header: dict
    stored: dict[str, np.ndarray]
    format_version: int
    elapsed: float


def require_tree_checkpoint_legal(model) -> tuple[int, int]:
    """Enforce the binding PERIOD_BEGIN tree checkpoint contract."""
    from gpuwm.core.model import PERIOD_BEGIN

    status = getattr(model, "_runtime_status", None)
    if status is None:
        raise RestartMismatchError(
            "tree checkpoint has no model runtime phase state")
    if status.schedule_cursor != PERIOD_BEGIN:
        raise RestartMismatchError(
            "tree checkpoint is legal only at explicit PERIOD_BEGIN; "
            f"schedule_cursor={status.schedule_cursor!r}")
    pending = {
        "FORCE": int(status.pending_force),
        "FEEDBACK": int(status.pending_feedback),
        "D2H": int(status.pending_d2h),
        "mutation": int(bool(status.mutation_in_progress)),
    }
    io_manager = getattr(model, "_io_manager", None)
    if io_manager is not None:
        pending["D2H"] = max(pending["D2H"], int(io_manager.pending))
    active = {name: count for name, count in pending.items() if count}
    if active:
        raise RestartMismatchError(
            f"tree checkpoint has pending work {active}; drain/commit it "
            "before PERIOD_BEGIN publication")
    if not bool(status.prior_feedback_committed):
        raise RestartMismatchError(
            "tree checkpoint requires all prior-period feedback committed")

    nodes = tuple(model.walk_parent_first())
    if not nodes:
        raise RestartMismatchError("tree checkpoint has no domains")
    tick_den = int(nodes[0].clock.tick_den)
    ticks = int(nodes[0].clock.ticks)
    for node in nodes:
        if int(node.clock.tick_den) != tick_den:
            raise RestartMismatchError(
                f"tree checkpoint tick denominator mismatch on "
                f"d{node.cfg.grid_id:02d}: {node.clock.tick_den} != "
                f"{tick_den}")
        if int(node.clock.ticks) != ticks:
            raise RestartMismatchError(
                f"tree checkpoint elapsed tick mismatch on "
                f"d{node.cfg.grid_id:02d}: {node.clock.ticks} != {ticks}")
    if ticks % model.schedule.period_ticks != 0:
        raise RestartMismatchError(
            f"tree checkpoint tick {ticks} is not a PERIOD_BEGIN boundary")
    return ticks, tick_den


def _fingerprint_components(model):
    """The named restart-identity components, when the route publishes them."""

    return getattr(model, "_experiment_fingerprint_components", None)


def tree_fingerprint_mismatch_reason(gid: int, header, model) -> str:
    """Name what actually differs, and what a restart is allowed to change.

    ``gpuwm run --restart`` publishes a tolerance -- only the forecast
    length and the output/restart cadence may differ -- and until 1.4.1
    the tree route enforced something strictly narrower without saying
    so, refusing every one of those three changes as nine words and a
    traceback.  The tolerance is honoured now; this message is what the
    remaining, genuine mismatches say.
    """

    stored = header.get("experiment_fingerprint_components")
    live = _fingerprint_components(model)
    prefix = f"tree restart d{gid:02d} was written for a different run"
    if not isinstance(stored, Mapping) or not isinstance(live, Mapping):
        return (f"{prefix} (experiment fingerprint mismatch); a checkpoint "
                "resumes only into the run that wrote it")
    _absent = object()
    differing = sorted(
        name for name in set(stored) | set(live)
        if stored.get(name, _absent) != live.get(name, _absent))
    if not differing:
        # Equal components, unequal digest: the digest itself moved, which
        # is a format change rather than a configuration change.
        return (f"{prefix} (fingerprint differs but every named component "
                "matches; the checkpoint predates this restart-identity "
                "format and must be rerun from the start)")
    named = ", ".join(differing)
    return (f"{prefix}: {named} differ(s) from the checkpoint.  A restart "
            "may change the forecast length and the output/restart cadence; "
            "everything else -- geometry, timestep, physics, nesting, "
            "prepared inputs -- must be the run that wrote the checkpoint")


def write_tree_restart(directory, model, valid_time: datetime, *,
                       run_trackers_by_grid_id=None,
                       sealed_forcing_extension: bool = False) -> Path:
    """Publish one immutable generation per domain, with d01 last.

    Every generation has UUID-qualified member names.  Publishing the root
    last makes it the set's commit marker: a crash while writing children
    cannot be advertised as a durable d01 checkpoint, and a failed rewrite at
    the same valid time cannot replace members of the prior committed set.  A
    restore still validates the complete sibling set before mutating state.
    """
    from gpuwm.core.model import PERIOD_BEGIN

    ticks, tick_den = require_tree_checkpoint_legal(model)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nodes = tuple(model.walk_parent_first())
    ids = sorted(int(node.cfg.grid_id) for node in nodes)
    trackers = dict(run_trackers_by_grid_id or {})
    paths: dict[int, Path] = {}
    if sealed_forcing_extension:
        # Validate the whole generation before assigning its UUID or writing
        # even a child member.  A malformed root prefix must never leave an
        # orphan child that looks like part of a publish attempt.
        sealed_elapsed = ticks / tick_den
        for node in nodes:
            _require_sealable_forcing_prefix(
                node.state, node.cfg.run,
                path=directory / f"d{int(node.cfg.grid_id):02d}",
                elapsed=sealed_elapsed)
    checkpoint_set_id = uuid.uuid4().hex
    published: list[Path] = []
    try:
        # Children first, root commit marker last.
        for node in reversed(nodes):
            gid = int(node.cfg.grid_id)
            node.state.elapsed_seconds = ticks / tick_den
            bits = int(np.float32(node.clock.dtbc_fp32).view(np.uint32))
            started = bool(getattr(node, "_started", True))
            lifecycle = "STARTED" if started else "NOT_STARTED"
            domain_start_time = (
                model.schedule.clock.start_time
                if node.cfg.start_time is None
                else node.cfg.start_time)
            tree_header = {
                "experiment_fingerprint": model.experiment_fingerprint,
                # The named components the fingerprint is a digest of,
                # when the building route publishes them.  Stored so a
                # mismatch on restore can say WHICH one moved instead of
                # reporting that a hash differs.  Absent for routes that
                # build the fingerprint some other way; the comparison
                # below degrades to the bare-hash message.
                **({} if _fingerprint_components(model) is None else {
                    "experiment_fingerprint_components":
                        _fingerprint_components(model)}),
                "checkpoint_set_id": checkpoint_set_id,
                "grid_id": gid,
                "parent_id": int(node.cfg.parent_id),
                "domain_ids": ids,
                "elapsed_ticks": ticks,
                "tick_den": tick_den,
                "phase": PERIOD_BEGIN,
                "domain_start_time": domain_start_time.isoformat(),
                "domain_start_ticks": int(node.clock.spec.start_ticks),
                "domain_lifecycle": lifecycle,
                "nest_tables": (
                    "REBUILT" if started or node.parent is None
                    else "NOT_STARTED"),
                "dtbc_fp32_bits": bits,
            }
            base = Path(restart_filename(valid_time, f"d{gid:02d}"))
            member = base.with_name(
                f"{base.stem}__{checkpoint_set_id}{base.suffix}")
            path = directory / member
            paths[gid] = write_restart(
                path, node.state, node.cfg.run,
                run_trackers=trackers.get(gid), tree_header=tree_header,
                sealed_forcing_extension=sealed_forcing_extension)
            published.append(paths[gid])
    except BaseException:
        # These names are unique to this uncommitted generation, so cleanup
        # can never remove a member referenced by an older root marker.
        for path in published:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    root_id = int(model.root.cfg.grid_id)
    model._last_checkpoint = paths[root_id]
    return paths[root_id]


_TREE_RESTART_NAME = re.compile(
    r"^gpuwmrst_d(?P<grid_id>[0-9]+)_(?P<instant>.+\.npz)$")


def _tree_restart_paths(path: Path, expected_ids: set[int]
                        ) -> dict[int, Path]:
    match = _TREE_RESTART_NAME.fullmatch(path.name)
    if match is None:
        raise RestartMismatchError(
            f"tree restart path {path} is not gpuwmrst_d0X_<instant>.npz")
    instant = match.group("instant")
    found: dict[int, Path] = {}
    for candidate in path.parent.glob(f"gpuwmrst_d*_{instant}"):
        parsed = _TREE_RESTART_NAME.fullmatch(candidate.name)
        if parsed is not None:
            gid = int(parsed.group("grid_id"))
            if gid in found:
                raise RestartMismatchError(
                    f"duplicate tree restart file for grid_id={gid}")
            found[gid] = candidate
    if set(found) != expected_ids:
        raise RestartMismatchError(
            "tree restart refuses a partial/mismatched domain set: "
            f"expected {sorted(expected_ids)}, found {sorted(found)} for "
            f"instant {instant}")
    return found


def restore_tree_restart(path, model, *,
                         sealed_forcing_extension: bool = False
                         ) -> TreeRestartInfo:
    """Validate the complete current-format set, then restore all domains."""
    from gpuwm.core.model import PERIOD_BEGIN, ModelRuntimeStatus

    path = Path(path)
    nodes = {int(node.cfg.grid_id): node
             for node in model.walk_parent_first()}
    paths = _tree_restart_paths(path, set(nodes))
    # Load every payload and hoist restore_restart's config/setup/manifest/
    # inventory/boundary refusal checks across the COMPLETE member set before
    # touching any live domain.  The validated objects retain the exact host
    # arrays that will be applied, so a later member cannot be swapped or
    # become unreadable after an earlier domain has mutated.
    if sealed_forcing_extension:
        validated = {
            gid: _validate_restart(
                paths[gid], node.state, node.cfg.run,
                sealed_forcing_extension=True)
            for gid, node in nodes.items()
        }
    else:
        # Preserve the historical default call shape as well as its exact
        # semantics.  A few out-of-tree diagnostic wrappers substitute this
        # private validator with the original three-argument signature.
        validated = {
            gid: _validate_restart(paths[gid], node.state, node.cfg.run)
            for gid, node in nodes.items()
        }
    headers = {gid: member.header for gid, member in validated.items()}

    elapsed_pairs = set()
    checkpoint_set_ids = set()
    for gid, node in nodes.items():
        header = headers[gid]
        if header.get("format_version") != RESTART_FORMAT_VERSION:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} must be "
                f"v{RESTART_FORMAT_VERSION}, got "
                f"{header.get('format_version')!r}; v2 is single-domain only")
        if sealed_forcing_extension and header.get(
                "forcing_extension_mode") != SEALED_FORCING_EXTENSION_MODE:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} was not intentionally sealed for "
                "forcing extension")
        if header.get("experiment_fingerprint") != \
                model.experiment_fingerprint:
            raise RestartMismatchError(
                tree_fingerprint_mismatch_reason(gid, header, model))
        checkpoint_set_id = header.get("checkpoint_set_id")
        if (not isinstance(checkpoint_set_id, str)
                or not checkpoint_set_id):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has no checkpoint_set_id")
        checkpoint_set_ids.add(checkpoint_set_id)
        if header.get("grid_id") != gid or header.get("parent_id") != \
                int(node.cfg.parent_id):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} grid_id/parent_id mismatch")
        if header.get("domain_ids") != sorted(nodes):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain set header mismatch")
        if header.get("phase") != PERIOD_BEGIN:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} phase must explicitly be "
                f"{PERIOD_BEGIN}, got {header.get('phase')!r}")
        expected_start_time = (
            model.schedule.clock.start_time
            if node.cfg.start_time is None else node.cfg.start_time)
        stored_start_time = header.get("domain_start_time")
        stored_start_ticks = header.get("domain_start_ticks")
        lifecycle = header.get("domain_lifecycle")
        # Current-format checkpoints written before delayed starts existed
        # can only represent the all-live-at-t0 lifecycle.  Admit that one
        # unambiguous migration; a delayed live config still requires every
        # new field and therefore remains fail-closed.
        if node.clock.spec.start_ticks == 0:
            if stored_start_time is None:
                stored_start_time = expected_start_time.isoformat()
            if stored_start_ticks is None:
                stored_start_ticks = 0
            if lifecycle is None:
                lifecycle = "STARTED"
        if stored_start_time != expected_start_time.isoformat():
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain_start_time mismatch")
        if stored_start_ticks != node.clock.spec.start_ticks:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain_start_ticks mismatch")
        if lifecycle not in ("STARTED", "NOT_STARTED"):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid domain_lifecycle "
                f"{lifecycle!r}")
        ticks = header.get("elapsed_ticks")
        den = header.get("tick_den")
        if (isinstance(ticks, bool) or not isinstance(ticks, int)
                or isinstance(den, bool) or not isinstance(den, int)
                or ticks < 0 or den <= 0):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid exact tick pair "
                f"{(ticks, den)!r}")
        expected_lifecycle = (
            "STARTED" if ticks >= node.clock.spec.start_ticks
            else "NOT_STARTED")
        if lifecycle != expected_lifecycle:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} lifecycle {lifecycle} disagrees "
                f"with elapsed/start ticks ({ticks}, "
                f"{node.clock.spec.start_ticks})")
        expected_tables = (
            "REBUILT" if lifecycle == "STARTED" or node.parent is None
            else "NOT_STARTED")
        if header.get("nest_tables") != expected_tables:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} must classify nest tables "
                f"{expected_tables}")
        stored_seconds = header.get("elapsed_seconds")
        if (isinstance(stored_seconds, bool)
                or not isinstance(stored_seconds, (int, float))
                or float(stored_seconds) != ticks / den):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} elapsed_seconds disagrees with "
                "its exact tick pair")
        bits = header.get("dtbc_fp32_bits")
        if (isinstance(bits, bool) or not isinstance(bits, int)
                or not 0 <= bits <= 0xFFFFFFFF):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid dtbc_fp32_bits")
        elapsed_pairs.add((ticks, den))
    if len(elapsed_pairs) != 1:
        raise RestartMismatchError(
            f"tree restart domain elapsed ticks mismatch: "
            f"{sorted(elapsed_pairs)}")
    if len(checkpoint_set_ids) != 1:
        raise RestartMismatchError(
            "tree restart files come from mismatched checkpoint sets")
    ticks, tick_den = elapsed_pairs.pop()
    if tick_den != model.schedule.clock.tick_den:
        raise RestartMismatchError(
            f"tree restart tick_den {tick_den} != live {model.schedule.clock.tick_den}")
    if ticks % model.schedule.period_ticks != 0:
        raise RestartMismatchError(
            f"tree restart tick {ticks} is not PERIOD_BEGIN")
    if ticks >= model.schedule.clock.run_ticks:
        raise RestartMismatchError(
            f"tree restart at {ticks} ticks has nothing left before stop "
            f"{model.schedule.clock.run_ticks}")

    # All refusal checks over all members precede the first mutation.
    for gid, node in nodes.items():
        _apply_validated_restart(validated[gid], node.state, node.cfg.run)
        clock = node.clock
        clock.ticks = ticks
        clock.step_count = max(
            0, (ticks - clock.spec.start_ticks) // clock.spec.step_ticks)
        bits = headers[gid]["dtbc_fp32_bits"]
        clock.dtbc_fp32 = np.asarray(bits, dtype=np.uint32).view(np.float32)
        node.state.elapsed_seconds = ticks / tick_den
        stored_lifecycle = headers[gid].get("domain_lifecycle")
        node._started = (
            True if stored_lifecycle is None
            and node.clock.spec.start_ticks == 0
            else stored_lifecycle == "STARTED")
        if node.parent is not None:
            node.coupler.invalidate()

    model._runtime_status = ModelRuntimeStatus()
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(
        gid for gid, node in nodes.items() if node.clock.history_due())
    root_id = int(model.root.cfg.grid_id)
    model._last_checkpoint = paths[root_id]
    return TreeRestartInfo(
        elapsed_ticks=ticks, tick_den=tick_den, phase=PERIOD_BEGIN,
        paths_by_grid_id=paths, headers_by_grid_id=headers)


def _validate_scratch_target(state, slot: str, host: np.ndarray,
                             key: str) -> None:
    """Prove a scratch copy can be applied without creating a live slot."""
    target = getattr(state, "_scratch", {}).get(slot)
    if target is not None:
        _check_array(host, target, key)
        return
    arena = getattr(state, "_scratch_arena", None)
    if arena is not None and arena.has_slot(slot):
        try:
            target = arena.view(host.shape, slot, host.dtype)
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartMismatchError(
                f"{key}: restart payload does not fit the live scratch arena"
            ) from exc
        _check_array(host, target, key)
        return
    # A non-arena missing slot will be allocated with the model field dtype.
    dtype = np.dtype(getattr(state.u, "dtype", np.float32))
    if host.dtype != dtype:
        raise RestartMismatchError(
            f"{key}: restart dtype {host.dtype} does not match state "
            f"scratch dtype {dtype}")


def _validate_driver_payload(stored, header, state, driver, elapsed,
                             format_version: int) -> None:
    """Hoist every PhysicsDriver refusal without mutating the driver."""
    stored_fields = {key[len("fields/"):]: value
                     for key, value in stored.items()
                     if key.startswith("fields/")}
    if set(stored_fields) != set(driver.fields):
        raise RestartMismatchError(
            "restart surface-field inventory does not match the resuming "
            f"driver (missing {sorted(set(driver.fields) - set(stored_fields))}, "
            f"extra {sorted(set(stored_fields) - set(driver.fields))})")
    for name, host in stored_fields.items():
        _check_array(host, driver.fields[name], f"fields/{name}")

    for name, target in (("rthratenlw", driver.rthratenlw),
                         ("rthratensw", driver.rthratensw),
                         ("pending_rainbl", driver._pending_rainbl)):
        key = f"driver/{name}"
        if key not in stored:
            raise RestartMismatchError(f"restart is missing {key}")
        _check_array(stored[key], target, key)

    for tend_name in DRIVER_TENDENCY_ATTRS:
        missing = [comp for comp in TENDENCY_REQUIRED_COMPONENTS
                   if f"driver/{tend_name}/{comp}" not in stored]
        if missing:
            raise RestartMismatchError(
                f"restart tendency {tend_name} is missing {missing}")
        for comp in TENDENCY_COMPONENTS:
            key = f"driver/{tend_name}/{comp}"
            host = stored.get(key)
            if format_version == RESTART_FORMAT_VERSION:
                live = getattr(getattr(driver, tend_name), comp)
                if (host is not None) != (live is not None):
                    disposition = ("missing" if host is None
                                   else "unexpected")
                    raise RestartMismatchError(
                        f"restart tendency inventory has {disposition} "
                        f"canonical member {key}")
            if host is not None:
                target = (state.u if comp == "ru" else
                          state.v if comp == "rv" else state.p)
                _check_array(host, target, key)

    from gpuwm.core.physics import microphysics_scratch_slots
    slot_map = dict(microphysics_scratch_slots(driver.mp_physics))
    if not slot_map and format_version == 2:
        missing = [comp for comp in MICROPHYSICS_REQUIRED_COMPONENTS
                   if f"driver/microphysics/{comp}" not in stored]
        if missing:
            raise RestartMismatchError(
                f"v2 restart microphysics diagnostics are missing {missing}")
    for comp, slot in slot_map.items():
        scratch_key = f"scratch/{slot}"
        legacy_key = f"driver/microphysics/{comp}"
        scratch_host = stored.get(scratch_key)
        legacy_host = stored.get(legacy_key)
        if format_version == RESTART_FORMAT_VERSION:
            if legacy_host is not None:
                raise RestartMismatchError(
                    f"v{RESTART_FORMAT_VERSION} restart unexpectedly "
                    f"contains removed member {legacy_key}")
            if scratch_host is None:
                raise RestartMismatchError(
                    f"v{RESTART_FORMAT_VERSION} restart is missing "
                    f"canonical microphysics member {scratch_key}")
        elif scratch_host is not None and legacy_host is not None:
            _validate_scratch_target(
                state, slot, legacy_host, legacy_key)
            if scratch_host.tobytes() != legacy_host.tobytes():
                raise RestartMismatchError(
                    f"v2 restart's duplicate microphysics members "
                    f"{scratch_key} and {legacy_key} differ byte-for-byte; "
                    "they cannot be rebuilt as one alias without changing "
                    "the trajectory")
        source = scratch_host if scratch_host is not None else legacy_host
        if source is None:
            if comp in MICROPHYSICS_REQUIRED_COMPONENTS:
                raise RestartMismatchError(
                    f"v2 restart microphysics diagnostics are missing "
                    f"both {scratch_key} and {legacy_key}")
            empty = np.empty(state.mup.shape, dtype=state.u.dtype)
            _validate_scratch_target(state, slot, empty, scratch_key)
        else:
            _validate_scratch_target(state, slot, source, scratch_key)

    driver_header = header.get("driver")
    if not isinstance(driver_header, dict):
        raise RestartMismatchError(
            "restart has no physics-driver header but the resuming state "
            "has a PhysicsDriver attached")
    try:
        call_counts = driver_header["call_counts"]
        int(driver_header["ysu_nan_guard_fires"])
        int(driver_header["microphysics_updates"])
        if not isinstance(call_counts, dict):
            raise TypeError("call_counts is not a mapping")
        for value in call_counts.values():
            int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RestartMismatchError(
            "restart physics-driver header is incomplete or malformed") from exc

    if "cumulus/w0avg" in stored:
        adapter = driver.cumulus_callable
        if adapter is None or not hasattr(adapter, "w0avg"):
            raise RestartMismatchError(
                "restart carries cumulus W0AVG but the resuming driver has "
                "no trigger-history cumulus adapter")
        _check_array(stored["cumulus/w0avg"], state.p, "cumulus/w0avg")
    elif getattr(driver.cumulus_callable, "w0avg", None) is not None:
        raise RestartMismatchError(
            "restart carries no cumulus W0AVG history but the resuming "
            "driver already has live W0AVG state; prepare a fresh adapter")
    if "radiation/o33d_grid" in stored:
        if getattr(driver.radiation_callable, "_o33d_grid", "absent") \
                == "absent":
            raise RestartMismatchError(
                "restart carries the legacy-RRTMG o33d field but the "
                "resuming radiation callable has no _o33d_grid slot "
                "(variant mismatch should have refused earlier)")
        _check_array(stored["radiation/o33d_grid"], state.p,
                     "radiation/o33d_grid")


def _validated_forcing_prefix(value, *, label: str, path: Path):
    """Validate and normalize one append-only LBC identity document."""
    if not isinstance(value, dict):
        raise RestartMismatchError(
            f"restart file {path} has no {label} forcing inventory")
    expected_keys = {
        "schema", "spec_bdy_width", "spec_zone", "relax_zone", "intervals",
    }
    if set(value) != expected_keys:
        missing = sorted(expected_keys - set(value))
        extra = sorted(set(value) - expected_keys)
        raise RestartMismatchError(
            f"restart file {path} has malformed {label} forcing document "
            f"(missing {missing}, extra {extra})")
    if value.get("schema") != LATERAL_BOUNDARY_PREFIX_SCHEMA:
        raise RestartMismatchError(
            f"restart file {path} has unknown {label} forcing schema "
            f"{value.get('schema')!r}")
    controls = []
    for key in ("spec_bdy_width", "spec_zone", "relax_zone"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RestartMismatchError(
                f"restart file {path} has invalid {label} forcing "
                f"control {key}={item!r}")
        controls.append(item)
    raw_intervals = value.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise RestartMismatchError(
            f"restart file {path} has an empty/malformed {label} forcing "
            "inventory")
    intervals = []
    field_inventory = None
    for index, row in enumerate(raw_intervals):
        if not isinstance(row, dict) or set(row) != {
                "start_seconds", "end_seconds", "fields", "sha256",
                "start_frame_sha256", "end_frame_sha256"}:
            raise RestartMismatchError(
                f"restart file {path} has malformed {label} forcing "
                f"interval {index}")
        start = row["start_seconds"]
        end = row["end_seconds"]
        if (isinstance(start, bool) or not isinstance(start, (int, float))
                or isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < 0.0 or float(end) <= float(start)):
            raise RestartMismatchError(
                f"restart file {path} has invalid {label} forcing bounds "
                f"at interval {index}: {(start, end)!r}")
        if intervals and float(start) != float(intervals[-1]["end_seconds"]):
            raise RestartMismatchError(
                f"restart file {path} has non-contiguous {label} forcing "
                f"inventory at interval {index}")
        fields = row["fields"]
        if (not isinstance(fields, list) or not fields
                or any(not isinstance(name, str) or not name for name in fields)
                or fields != sorted(set(fields))):
            raise RestartMismatchError(
                f"restart file {path} has malformed {label} forcing field "
                f"inventory at interval {index}")
        if field_inventory is None:
            field_inventory = fields
        elif fields != field_inventory:
            raise RestartMismatchError(
                f"restart file {path} changes {label} forcing fields at "
                f"interval {index}")
        for digest_key in (
                "sha256", "start_frame_sha256", "end_frame_sha256"):
            sha = row[digest_key]
            if (not isinstance(sha, str) or len(sha) != 64
                    or any(char not in "0123456789abcdef" for char in sha)):
                raise RestartMismatchError(
                    f"restart file {path} has invalid {label} forcing "
                    f"{digest_key} at interval {index}")
        if intervals and intervals[-1]["end_frame_sha256"] != \
                row["start_frame_sha256"]:
            raise RestartMismatchError(
                f"restart file {path} has a discontinuous {label} forcing "
                f"frame at interval {index}")
        intervals.append({
            "start_seconds": start,
            "end_seconds": end,
            "fields": fields,
            "sha256": row["sha256"],
            "start_frame_sha256": row["start_frame_sha256"],
            "end_frame_sha256": row["end_frame_sha256"],
        })
    if float(intervals[0]["start_seconds"]) != 0.0:
        raise RestartMismatchError(
            f"restart file {path} {label} forcing does not begin at zero")
    return tuple(controls), intervals


def _require_sealable_forcing_prefix(state, cfg, *, path: Path,
                                     elapsed: float) -> None:
    """Refuse an invalid sealed checkpoint before any bytes are published."""
    prefix = lateral_boundary_prefix_identity(state)
    if getattr(cfg, "specified", False):
        _, intervals = _validated_forcing_prefix(
            prefix, label="live", path=path)
        if float(intervals[-1]["end_seconds"]) != elapsed:
            raise RestartMismatchError(
                f"restart file {path} forcing inventory is not sealed at "
                f"its checkpoint boundary "
                f"({intervals[-1]['end_seconds']!r} != {elapsed!r})")
        return
    if getattr(cfg, "nested", False):
        if prefix is not None:
            raise RestartMismatchError(
                f"restart file {path} nested child unexpectedly carries "
                "an external forcing prefix")
        return
    raise RestartMismatchError(
        f"restart file {path} cannot use sealed forcing extension: the "
        "domain is neither a specified external-boundary root nor a nested "
        "child")


def _require_sealed_forcing_extension(header, state, *, path: Path,
                                      elapsed: float) -> None:
    """Admit only a byte-identical sealed prefix plus contiguous future LBCs."""
    if header.get("forcing_extension_mode") != SEALED_FORCING_EXTENSION_MODE:
        raise RestartMismatchError(
            f"restart file {path} was not intentionally sealed for forcing "
            "extension")
    stored_core = header.get("setup_core_fingerprint")
    live_core = setup_core_fingerprint(state)
    if stored_core != live_core:
        raise RestartMismatchError(
            f"restart file {path} immutable setup changed while extending "
            "forcing (base state / coordinates / map factors mismatch)")
    stored_value = header.get("lateral_boundary_prefix")
    live_value = lateral_boundary_prefix_identity(state)
    stored_controls, stored = _validated_forcing_prefix(
        stored_value, label="sealed", path=path)
    live_controls, live = _validated_forcing_prefix(
        live_value, label="live", path=path)
    if stored_controls != live_controls:
        raise RestartMismatchError(
            f"restart file {path} changes specified-boundary controls while "
            "extending forcing")
    if float(stored[-1]["end_seconds"]) != elapsed:
        raise RestartMismatchError(
            f"restart file {path} forcing inventory was not sealed at its "
            f"checkpoint boundary ({stored[-1]['end_seconds']!r} != "
            f"{elapsed!r})")
    if len(live) <= len(stored):
        raise RestartMismatchError(
            f"restart file {path} forcing extension must append at least one "
            "future interval")
    if live[:len(stored)] != stored:
        raise RestartMismatchError(
            f"restart file {path} forcing extension changed a sealed interval "
            "or its byte digest")
    suffix = live[len(stored):]
    if float(suffix[0]["start_seconds"]) != elapsed:
        raise RestartMismatchError(
            f"restart file {path} forcing suffix does not begin at the "
            "checkpoint boundary")
    if any(float(row["start_seconds"]) < elapsed for row in suffix):
        raise RestartMismatchError(
            f"restart file {path} forcing suffix is not strictly future")
    if stored[-1]["end_frame_sha256"] != suffix[0][
            "start_frame_sha256"]:
        raise RestartMismatchError(
            f"restart file {path} forcing suffix changes the shared "
            "checkpoint-boundary frame")


def _validate_restart(path, state, cfg, *,
                      sealed_forcing_extension: bool = False
                      ) -> _ValidatedRestart:
    """Load one archive and perform all refusal checks without mutation."""
    path = Path(path)
    header, stored = _load_restart(path, with_arrays=True)
    required_header = {
        "format_version", "config", "setup_fingerprint",
        "physics_setup", "physics_setup_fingerprint",
        "array_manifest", "elapsed_seconds",
    }
    missing_header = sorted(required_header - set(header))
    if missing_header:
        raise RestartMismatchError(
            f"restart file {path} header is missing {missing_header}")
    format_version = header.get("format_version")
    if format_version not in READABLE_RESTART_FORMAT_VERSIONS:
        raise RestartMismatchError(
            f"restart file {path} has format version "
            f"{format_version!r}; this build reads "
            f"{sorted(READABLE_RESTART_FORMAT_VERSIONS)}")
    _require_config_match(header["config"], cfg, path)
    live_lbc_clock = root_external_lbc_clock_identity(state, cfg)
    if live_lbc_clock is not None:
        stored_lbc_clock = header.get("root_external_lbc_clock",
                                      ROOT_EXTERNAL_LBC_CLOCK_LEGACY)
        if stored_lbc_clock != live_lbc_clock:
            raise RestartMismatchError(
                f"restart file {path} was integrated under the "
                f"root_external_lbc_clock semantic {stored_lbc_clock!r} "
                f"but the resuming state runs {live_lbc_clock!r}: the "
                "root external Davies dtbc consumption differs (WRF "
                "post-increment bind vs legacy elapsed-based), so "
                "resuming would splice two different trajectories.  "
                "Regenerate the checkpoint under the current semantic "
                "(a header without the key is a pre-bind file).")
    _require_nssl2_restart_contract(header, cfg, path)
    _require_physics_setup_match(header, state, cfg, path)
    try:
        elapsed = _admissible_elapsed_seconds(
            header["elapsed_seconds"], f"restart file {path}")
    except (TypeError, ValueError, KeyError, RestartManifestError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has an invalid elapsed_seconds") from exc
    live_setup_fingerprint = setup_fingerprint(state)
    if sealed_forcing_extension:
        if header.get("forcing_extension_mode") != \
                SEALED_FORCING_EXTENSION_MODE:
            raise RestartMismatchError(
                f"restart file {path} was not intentionally sealed for "
                "forcing extension")
        stored_prefix = header.get("lateral_boundary_prefix")
        if getattr(cfg, "specified", False):
            if stored_prefix is None:
                raise RestartMismatchError(
                    f"restart file {path} specified root has no sealed "
                    "forcing inventory")
            _require_sealed_forcing_extension(
                header, state, path=path, elapsed=elapsed)
        elif getattr(cfg, "nested", False):
            if stored_prefix is not None:
                raise RestartMismatchError(
                    f"restart file {path} nested child unexpectedly carries "
                    "a sealed external forcing inventory")
            if (header.get("setup_core_fingerprint")
                    != setup_core_fingerprint(state)
                    or header["setup_fingerprint"]
                    != live_setup_fingerprint):
                raise RestartMismatchError(
                    f"restart file {path} changed immutable child/nest setup "
                    "while extending root forcing")
        else:
            raise RestartMismatchError(
                f"restart file {path} cannot use sealed forcing extension: "
                "the live domain is neither a specified root nor a nested "
                "child")
    elif header["setup_fingerprint"] != live_setup_fingerprint:
        raise RestartMismatchError(
            f"restart file {path} was written on a different model setup "
            "(base state / coordinates / map factors fingerprint "
            "mismatch); rebuild the identical preparation before restoring")
    manifest = header["array_manifest"]
    if not isinstance(manifest, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed array manifest")
    if set(stored) != set(manifest):
        missing = sorted(set(manifest) - set(stored))
        extra = sorted(set(stored) - set(manifest))
        raise RestartMismatchError(
            f"restart file {path} arrays disagree with its own manifest "
            f"(missing {missing}, extra {extra})")
    for key, spec in manifest.items():
        try:
            matches = (list(stored[key].shape) == list(spec["shape"])
                       and str(stored[key].dtype) == spec["dtype"])
        except (KeyError, TypeError) as exc:
            raise RestartMismatchError(
                f"restart member {key} has a malformed manifest entry") from exc
        if not matches:
            raise RestartMismatchError(
                f"restart member {key} does not match its manifest entry")

    _validate_nssl2_stored_restart_state(
        header, stored, state, cfg, path, elapsed)
    _validate_thompson_aerosol_stored_restart_state(stored, state, cfg, path)
    stored_state = {key[len("state/"):]: value
                    for key, value in stored.items()
                    if key.startswith("state/")}
    expected_state = {name for name in STATE_SERIALIZED_ATTRS
                      if getattr(state, name, None) is not None}
    if set(stored_state) != expected_state:
        raise RestartMismatchError(
            f"restart state arrays {sorted(set(stored_state))} do not "
            f"match this configuration's state {sorted(expected_state)}")
    for name, host in stored_state.items():
        _check_array(host, getattr(state, name), f"state/{name}")

    for key, host in stored.items():
        if not key.startswith("scratch/"):
            continue
        slot = key[len("scratch/"):]
        if classify_scratch_slot(slot) != "serialize":
            raise RestartMismatchError(
                f"restart carries non-serializable scratch slot {slot!r}")
        _validate_scratch_target(state, slot, host, key)

    driver = getattr(state, "physics", None)
    driver_keys = [key for key in stored
                   if key.startswith(("driver/", "fields/", "cumulus/"))]
    if driver is None:
        if driver_keys:
            raise RestartMismatchError(
                "restart carries physics-driver state but the resuming "
                "state has no PhysicsDriver attached")
        if header.get("driver") is not None:
            raise RestartMismatchError(
                "restart has a physics-driver header but the resuming state "
                "has no PhysicsDriver attached")
    else:
        _validate_driver_payload(
            stored, header, state, driver, elapsed, format_version)

    if cfg.specified:
        if (state.lateral_boundaries is None
                or getattr(state, "_lateral_boundary_device", None) is None):
            raise RestartMismatchError(
                "cfg.specified requires attach_lateral_boundaries(state, "
                "...) BEFORE restore_restart: the resident LBC device "
                "tables are rebuilt by preparation, not by the restart")
        state.lateral_boundaries.interval_at(elapsed)
    return _ValidatedRestart(
        path=path, header=header, stored=stored,
        format_version=format_version, elapsed=elapsed)


def _apply_validated_restart(validated: _ValidatedRestart,
                             state, cfg) -> RestartInfo:
    """Apply an already complete-set-validated member in place."""
    header = validated.header
    stored = validated.stored
    format_version = validated.format_version
    elapsed = validated.elapsed
    driver = getattr(state, "physics", None)
    asarray = _asarray_like(state)

    stored_state = {key[len("state/"):]: value
                    for key, value in stored.items()
                    if key.startswith("state/")}
    for name, host in stored_state.items():
        getattr(state, name)[...] = asarray(host)
    stored_scratch = set()
    for key, host in stored.items():
        if key.startswith("scratch/"):
            slot = key[len("scratch/"):]
            stored_scratch.add(slot)
            if slot == "up_heli_max" \
                    and state.existing_scratch(slot) is None:
                # The eagerly allocated diagnostic accumulator: a state
                # prepared with nwp_diagnostics=0 does not carry the slot,
                # and restoring it anyway would emit a stale, never-again-
                # updated UP_HELI_MAX into every frame.  Dropping a
                # diagnostic is a note, never a refusal.
                print(f"note: checkpoint {validated.path.name} carries "
                      f"{slot!r} but this run has nwp_diagnostics=0; the "
                      "diagnostic accumulator is dropped")
                continue
            state.scratch(host.shape, slot)[...] = asarray(host)
    # Old-checkpoint tolerance for accumulators that postdate the file: a
    # live serialized slot with no stored counterpart stays at its
    # zero-initialized allocation.  That is the correct diagnostic restore
    # (the first frame after resume simply covers a shortened max window),
    # so it is a NOTE, never a refusal.
    for slot in sorted(SERIALIZED_SCRATCH_SLOTS):
        if slot not in stored_scratch \
                and state.existing_scratch(slot) is not None:
            state.existing_scratch(slot)[...] = 0.0
            print(f"note: checkpoint {validated.path.name} predates the "
                  f"serialized accumulator {slot!r}; restored "
                  "zero-initialized")
    if driver is not None:
        _restore_driver(stored, header, state, driver, elapsed, asarray,
                        format_version)
    # v5 migration normalization: checkpoints written before the
    # spec-zone ring exclusion (same format version) can carry nonzero
    # ring MP accumulators/diagnostics and stale ring h_diabatic from
    # whole-field microphysics; no WRF-valid trajectory can contain them.
    # Idempotent on post-fix checkpoints (see the function's docstring).
    from gpuwm.core.microphysics import normalize_spec_zone_ring_after_restore
    normalize_spec_zone_ring_after_restore(state, cfg)
    state.elapsed_seconds = elapsed
    return RestartInfo(elapsed_seconds=elapsed,
                       run_trackers=header.get("run_trackers"),
                       header=header)


def restore_restart(path, state, cfg) -> RestartInfo:
    """Restore a restart file into a freshly PREPARED state, in place.

    The caller must have completed the normal deterministic setup first
    (base state loaded, physics initialized, lateral boundaries attached):
    restore validates the config echo and the setup fingerprint, then
    overwrites every serialized array in place — surface fields are never
    rebound, preserving the SFCLAY-result aliasing — rebinds the held
    tendency containers with the stored coupled arrays, restores the KF
    W0AVG onto the cumulus adapter (rebinding its history to this state),
    and restores ``elapsed_seconds`` LAST, after
    ``attach_lateral_boundaries`` reset it to zero.
    """
    validated = _validate_restart(path, state, cfg)
    return _apply_validated_restart(validated, state, cfg)


def _restore_driver(stored, header, state, driver, elapsed, asarray,
                    format_version: int) -> None:
    from gpuwm.core.microphysics import MicrophysicsDiagnostics
    from gpuwm.core.physics import PhysicsTendencies

    # Surface/Noah fields: in place only (never rebind — sfclay_result and
    # the Noah launch read these exact device arrays).
    stored_fields = {key[len("fields/"):]: value
                     for key, value in stored.items()
                     if key.startswith("fields/")}
    if set(stored_fields) != set(driver.fields):
        raise RestartMismatchError(
            "restart surface-field inventory does not match the resuming "
            f"driver (missing {sorted(set(driver.fields) - set(stored_fields))}, "
            f"extra {sorted(set(stored_fields) - set(driver.fields))})")
    for name, host in stored_fields.items():
        target = driver.fields[name]
        _check_array(host, target, f"fields/{name}")
        target[...] = asarray(host)

    for name in ("rthratenlw", "rthratensw"):
        key = f"driver/{name}"
        if key not in stored:
            raise RestartMismatchError(f"restart is missing {key}")
        target = getattr(driver, name)
        _check_array(stored[key], target, key)
        target[...] = asarray(stored[key])
    key = "driver/pending_rainbl"
    if key not in stored:
        raise RestartMismatchError(f"restart is missing {key}")
    _check_array(stored[key], driver._pending_rainbl, key)
    driver._pending_rainbl[...] = asarray(stored[key])

    # Held tendencies: rebind with the stored COUPLED arrays (no
    # recoupling — see the manifest argument).  compute() recomposes the
    # working sum from these components before the next consumption.
    for tend_name in DRIVER_TENDENCY_ATTRS:
        components = {}
        live_tendency = getattr(driver, tend_name)
        for comp in TENDENCY_COMPONENTS:
            comp_key = f"driver/{tend_name}/{comp}"
            if comp_key in stored:
                components[comp] = asarray(stored[comp_key])
            elif format_version == 2:
                # Legacy absence meant an identically zero optional held
                # category.  Preserve the constructor's eager canonical
                # buffer so a supported v2 resume cannot grow its inventory
                # at the first scheduled physics call.
                components[comp] = getattr(live_tendency, comp)
            else:
                components[comp] = None
        missing = [comp for comp in TENDENCY_REQUIRED_COMPONENTS
                   if components[comp] is None]
        if missing:
            raise RestartMismatchError(
                f"restart tendency {tend_name} is missing {missing}")
        setattr(driver, tend_name, PhysicsTendencies(**components))
    config = header["config"]
    reuse_pbl = bool(config.get("bl_pbl_physics")
                     and config.get("bldt") == 0.0
                     and (driver.radiation_active or driver.cu_physics))
    if not (driver.radiation_active or driver.cu_physics) or reuse_pbl:
        # Preserve both proven identity paths and release the constructor's
        # superseded initial PBL buffers immediately on restore.
        driver.tendencies = driver.pbl_tendencies

    from gpuwm.core.physics import microphysics_scratch_slots
    slot_map = dict(microphysics_scratch_slots(driver.mp_physics))
    components = {comp: None for comp in MICROPHYSICS_COMPONENTS}
    if not slot_map:
        # v3+ mp=0 diagnostics are deterministic zero placeholders.  Preserve
        # a v2 file's old owned values exactly for compatibility.
        if format_version == 2:
            for comp in MICROPHYSICS_COMPONENTS:
                key = f"driver/microphysics/{comp}"
                if key in stored:
                    components[comp] = asarray(stored[key])
            missing = [comp for comp in MICROPHYSICS_REQUIRED_COMPONENTS
                       if components[comp] is None]
            if missing:
                raise RestartMismatchError(
                    f"v2 restart microphysics diagnostics are missing "
                    f"{missing}")
            driver.microphysics = MicrophysicsDiagnostics(**components)
    else:
        for comp, slot in slot_map.items():
            scratch_key = f"scratch/{slot}"
            legacy_key = f"driver/microphysics/{comp}"
            scratch_host = stored.get(scratch_key)
            legacy_host = stored.get(legacy_key)
            if format_version == RESTART_FORMAT_VERSION:
                if legacy_host is not None:
                    raise RestartMismatchError(
                        f"v{RESTART_FORMAT_VERSION} restart unexpectedly "
                        f"contains removed member {legacy_key}")
                if scratch_host is None:
                    raise RestartMismatchError(
                        f"v{RESTART_FORMAT_VERSION} restart is missing "
                        f"canonical microphysics member {scratch_key}")
            elif scratch_host is not None and legacy_host is not None:
                legacy_target = state.scratch(legacy_host.shape, slot)
                _check_array(legacy_host, legacy_target, legacy_key)
                if scratch_host.tobytes() != legacy_host.tobytes():
                    raise RestartMismatchError(
                        f"v2 restart's duplicate microphysics members "
                        f"{scratch_key} and {legacy_key} differ byte-for-byte; "
                        "they cannot be rebuilt as one alias without changing "
                        "the trajectory")
            source = scratch_host if scratch_host is not None else legacy_host
            if source is None:
                if comp in MICROPHYSICS_REQUIRED_COMPONENTS:
                    raise RestartMismatchError(
                        f"v2 restart microphysics diagnostics are missing "
                        f"both {scratch_key} and {legacy_key}")
                # An old pre-first-call optional can legitimately be absent;
                # the new canonical scratch slot is already zero-filled.
                target = state.scratch(state.mup.shape, slot)
            else:
                target = state.scratch(source.shape, slot)
                _check_array(source, target, scratch_key)
                if scratch_host is None:
                    target[...] = asarray(source)
            components[comp] = target
        driver.microphysics = MicrophysicsDiagnostics(**components)

    driver_header = header["driver"]
    driver.call_counts.update(
        {key: int(value)
         for key, value in driver_header["call_counts"].items()})
    driver.ysu_nan_guard_fires = int(driver_header["ysu_nan_guard_fires"])
    driver.microphysics_updates = int(
        driver_header["microphysics_updates"])

    if "cumulus/w0avg" in stored:
        adapter = driver.cumulus_callable
        if adapter is None or not hasattr(adapter, "w0avg"):
            raise RestartMismatchError(
                "restart carries cumulus W0AVG but the resuming driver has "
                "no trigger-history cumulus adapter")
        restored_w0avg = asarray(stored["cumulus/w0avg"])
        if (adapter.w0avg is not None
                and adapter.w0avg.shape == restored_w0avg.shape
                and adapter.w0avg.dtype == restored_w0avg.dtype):
            adapter.w0avg[...] = restored_w0avg
        else:
            adapter.w0avg = restored_w0avg
        # Rebind the history adapter to THIS state object so the next
        # update does not re-zero W0AVG (the object-identity trap: the
        # adapter invalidates on `self._history_state is not state`).
        adapter._history_state = state
        adapter._history_time = elapsed

    if "radiation/o33d_grid" in stored:
        # Legacy-RRTMG root o33d (WRF's restart-carried O3RAD analogue):
        # reattach as HOST float32 exactly as the adapter retains it, so
        # a child domain's first post-restore radiation call interpolates
        # the identical field the uninterrupted run would have used.
        radiation = driver.radiation_callable
        restored_o33d = np.asarray(
            stored["radiation/o33d_grid"], dtype=np.float32)
        radiation._o33d_grid = np.ascontiguousarray(restored_o33d)


__all__ = [
    "CONFIG_RUN_LENGTH_FIELDS", "DRIVER_REBUILT_ATTRS",
    "DRIVER_SERIALIZED_ATTRS", "DRIVER_TENDENCY_ATTRS",
    "CUMULUS_ALGORITHM_IDENTITIES", "LAND_SURFACE_ALGORITHM_IDENTITIES",
    "LAND_SURFACE_PARAMETER_SOURCES",
    "MICROPHYSICS_COMPONENTS", "REBUILT_SCRATCH_PREFIXES",
    "MICROPHYSICS_ALGORITHM_IDENTITIES", "PBL_ALGORITHM_IDENTITIES",
    "NSSL2_LEGACY_RESTART_ALIASES", "NSSL2_RESTART_AUXILIARY_STATE",
    "NSSL2_RESTART_CONTRACT_VERSION", "NSSL2_RESTART_PRECIPITATION_SLOTS",
    "NSSL2_RESTART_PROGNOSTICS",
    "PHYSICS_ASSET_PATHS", "PHYSICS_SETUP_SCHEMA_VERSION",
    "RADIATION_ABOVE_ATMOSPHERE_POLICIES",
    "RADIATION_ALGORITHM_IDENTITIES",
    "SURFACE_LAYER_ALGORITHM_IDENTITIES",
    "READABLE_RESTART_FORMAT_VERSIONS", "REBUILT_SCRATCH_SLOTS",
    "RESTART_FORMAT_VERSION", "ROOT_EXTERNAL_LBC_CLOCK_IDENTITY",
    "ROOT_EXTERNAL_LBC_CLOCK_LEGACY", "RestartInfo", "TreeRestartInfo",
    "SEALED_FORCING_EXTENSION_MODE",
    "RestartManifestError", "RestartMismatchError",
    "SERIALIZED_SCRATCH_SLOTS", "STATE_REBUILT_ATTRS",
    "STATE_SERIALIZED_ATTRS", "STATE_SETUP_ARRAYS", "STATE_SETUP_SCALARS",
    "THOMPSON_AEROSOL_RESTART_STATE",
    "THOMPSON_AEROSOL_RESTART_SURFACE_STATE",
    "TENDENCY_COMPONENTS", "classify_scratch_slot", "classify_state_attr",
    "physics_setup_fingerprint", "physics_setup_identity",
    "root_external_lbc_clock_identity",
    "read_restart_header", "require_tree_checkpoint_legal",
    "restart_filename", "restore_restart", "restore_tree_restart",
    "lateral_boundary_prefix_identity", "setup_core_fingerprint",
    "setup_fingerprint", "state_manifest", "write_restart",
    "write_tree_restart",
]
