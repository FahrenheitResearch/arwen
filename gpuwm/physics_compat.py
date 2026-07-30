"""Fail-closed status for WRF physics suites that are being ported.

This module is intentionally small and declarative.  A WRF scheme number is
not an implementation: the namelist importer may only admit a scheme after
its state, setup, driver, CUDA implementation, restart/output contract, and
validation gates have all landed.  Until then, a request fails once with a
complete list of the missing coupled components instead of failing on the
first number or, worse, silently choosing a nearby scheme.

The source of truth for the requested bindings is WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  Source anchors and the staged
acceptance gates live in ``docs/wrf_thompson_mynn_ruc_port.md`` and
``docs/wrf_nssl2_mp18_port.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType


# Exact token emitted into imported gpuwm configurations when WRF's legacy
# RRTMG 4/4 pair is intentionally routed to gpuwm's modern RTE+RRTMGP
# implementation.  It is trajectory-bound through RunConfig/restart identity.
#
# TOKEN FAMILIES: the substitution family (wrf-rrtmg-4-4-to-rte-rrtmgp-*)
# versions the RTE+RRTMGP adapter's WRF-matching behavior; the legacy
# family (wrf-rrtmg-4-4-legacy-*) is a DIFFERENT algorithm (the exact
# port) and carries WRF's snow discount verbatim from birth, so it has
# no -v2.  Assembly resolution (2026-07-28, per the radiation branch's
# own merge note): WRF_RRTMG_TO_RTE_RRTMGP rebinds to -v2, -v1 stays
# accepted for historical receipts, and the substitution tuple carries
# both -- the pairing rule in gpuwm.config (every substitution-family
# token requires the rte-rrtmgp variant; the legacy token requires
# rrtmg_legacy) generalizes unchanged.
#
# Substitution-family version history (the token is a receipt: bumping
# it relabels NO old run):
#   -v1: snow entered the radiative ice path at full mass with its native
#        effective radius (adapter-native coupling).
#   -v2: WRF-matching option-4 explicit-snow-radius coupling -- ice path from
#        cloud ice only, snow mass discounted by MIN(0.99, (130/re_s)^2) with
#        re_s capped at 130 microns (module_ra_rrtmg_lw.F:12500-12532,
#        module_ra_rrtmg_sw.F:11040-11067).  Current importer default.
# Configurations carrying the -v1 token keep the -v1 behavior; unknown token
# values fail closed in config validation.
WRF_RRTMG_TO_RTE_RRTMGP = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
WRF_RRTMG_TO_RTE_RRTMGP_V1 = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"

# Peer token for the exact port of WRF v4.6.1's bundled legacy RRTMG
# (ra_lw_physics = ra_sw_physics = 4 running WRF's own algorithm rather
# than the RTE+RRTMGP substitution above).  Selected through
# RunConfig.ra_rrtmg_variant = "rrtmg_legacy"; the pairing rules live in
# gpuwm.config.validate_run_config.
WRF_RRTMG_LEGACY = "wrf-rrtmg-4-4-legacy-v1"

#: Every wrf_rrtmg_compatibility value this lineage's code can honor,
#: beside "none".  Substitution-family tokens pair with the rte-rrtmgp
#: variant; the legacy token pairs with rrtmg_legacy (enforced in
#: gpuwm.config.validate_run_config).
WRF_RRTMG_SUBSTITUTION_TOKENS = (
    WRF_RRTMG_TO_RTE_RRTMGP_V1,
    WRF_RRTMG_TO_RTE_RRTMGP,
)
WRF_RRTMG_COMPATIBILITY_TOKENS = (
    *WRF_RRTMG_SUBSTITUTION_TOKENS,
    WRF_RRTMG_LEGACY,
)

#: RunConfig.ra_rrtmg_variant values: which implementation serves a
#: resolved 4/4 RRTMG request.
RRTMG_VARIANT_RTE_RRTMGP = "rte-rrtmgp"
RRTMG_VARIANT_LEGACY = "rrtmg_legacy"


def rrtmg_variant(cfg) -> str:
    """The 4/4 implementation selector, tolerant of pre-field configs."""

    return str(getattr(cfg, "ra_rrtmg_variant", RRTMG_VARIANT_RTE_RRTMGP))


#: Import surface of the legacy RRTMG port: every compute/prep/ingest
#: module a legacy-selected forecast executes.
_RRTMG_LEGACY_MODULES = (
    "gpuwm.ingest.rrtmg_coeffs",
    "gpuwm.ingest.wrf_ozone",
    "gpuwm.core.rrtmg_mcica",
    "gpuwm.core.rrtmg_lw",
    "gpuwm.core.rrtmg_sw",
    "gpuwm.core.rrtmg_legacy_prep",
    "gpuwm.core.rrtmg_legacy",
)

#: Packaged data assets and CUDA kernel sources the legacy port reads,
#: relative to the gpuwm package directory.
_RRTMG_LEGACY_ASSETS = (
    "data/wrf_radiation/RRTMG_LW_DATA",
    "data/wrf_radiation/RRTMG_SW_DATA",
    "data/wrf_radiation/ozone.formatted",
    "data/wrf_radiation/ozone_lat.formatted",
    "data/wrf_radiation/ozone_plev.formatted",
    "core/kernels/rrtmg_lw.cu",
    "core/kernels/rrtmg_lw_chain.cu",
    "core/kernels/rrtmg_lw_taugb02_10_11_12.cu",
    "core/kernels/rrtmg_lw_taugb03_05.cu",
    "core/kernels/rrtmg_lw_taugb06_09.cu",
    "core/kernels/rrtmg_lw_taugb13_16.cu",
    "core/kernels/rrtmg_mcica_wrf.cu",
    "core/kernels/rrtmg_sw.cu",
)


def require_rrtmg_legacy_ready() -> None:
    """Fail closed unless the legacy RRTMG port is genuinely present.

    Import-checks every compute module of the port and verifies the
    packaged coefficient/ozone data files and CUDA kernel sources exist,
    raising ONE receipt listing everything missing.  This replaced the
    pre-integration ``require_rrtmg_legacy_executable`` stub (which
    raised unconditionally); construction of
    ``gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation`` performs the
    deeper readiness proof (SHA-pinned loads, kernel compilation,
    live-device preflights) and fails closed itself.  Selecting the
    legacy port must never silently fall back to RTE+RRTMGP.
    """

    import importlib
    from pathlib import Path

    missing: list[str] = []
    for name in _RRTMG_LEGACY_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - receipt, then raise
            missing.append(f"module {name}: {exc}")
    package_dir = Path(__file__).resolve().parent
    for relative in _RRTMG_LEGACY_ASSETS:
        if not (package_dir / relative).is_file():
            missing.append(f"asset gpuwm/{relative}: file not found")
    if missing:
        raise NotImplementedError(
            "ra_rrtmg_variant='rrtmg_legacy' (exact port of WRF v4.6.1's "
            "bundled RRTMG, ra_lw/sw_physics = 4/4) is selected but not "
            "executable on this installation; no silent fallback to "
            "RTE+RRTMGP is applied.  Missing:\n  - "
            + "\n  - ".join(missing))


#: Backwards-compatible name: earlier construction sites and external
#: callers guarded legacy selection through this symbol while the compute
#: lanes were still landing.  It now performs the readiness check.
require_rrtmg_legacy_executable = require_rrtmg_legacy_ready

# Thompson 8 completed the audited Node-3 CUDA/WRF-oracle forecast lane and
# the matched four-domain verification rerun of 2026-07-28
# (docs/thompson-rematch-20260728.md), and the canonical WRF v4.6.1 classic
# tables now ship as package data (gpuwm/data/thompson/tables).  mp_physics=8
# is therefore selectable first-class: the process-environment enable guard
# is retired for selection (product decision, product/v1 packaging lane
# 2026-07-28), and the table root defaults to the packaged directory.
# GPUWM_THOMPSON_TABLE_ROOT remains honored as an override naming a
# directory with the same byte-validated table set; every load still
# re-validates exact sizes and SHA-256 before GPU setup.  The retired
# enable-guard name is kept because the guarded evidence/benchmark runners
# under tools/ still enforce it for their own launch contracts.
EXPERIMENTAL_THOMPSON_ENV = "GPUWM_EXPERIMENTAL_THOMPSON_MP8"
THOMPSON_TABLE_ROOT_ENV = "GPUWM_THOMPSON_TABLE_ROOT"


def packaged_thompson_table_root() -> "Path":
    """The in-package canonical classic-table directory.

    The four assets and their MANIFEST.sha256 are committed under
    ``gpuwm/data/thompson/tables``; byte identity against
    ``gpuwm.core.thompson_contract.CLASSIC_TABLE_ASSETS`` is enforced at
    load time, not assumed here.
    """

    from pathlib import Path

    return Path(__file__).resolve().parent / "data" / "thompson" / "tables"


def thompson_table_root() -> str:
    """Resolved mp8 table root: env override first, packaged default second.

    The override exists for byte-identical mirrors (fast local disks,
    cluster scratch); a root with different bytes fails closed in
    ``validate_table_assets`` exactly as the packaged one would.
    """

    override = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    return override if override else str(packaged_thompson_table_root())

# Noah-MP's whole column runs on the DEVICE: the slab orchestration in
# gpuwm/core/noahmp_column_slab.py answers every land column with no Python
# per column, in chunks of gpuwm.core.noahmp_runtime.SLAB_COLUMN_CHUNK
# (65,536), bitwise (max ULP 0) against the scalar column authority.
# Measured 2026-07-27, twice, on one RTX 5090, end to end through
# noahmp_lsm_step:
#
#     one 360,000-land-column call      0.202 - 0.227 s   (slab path)
#     the same call, per-column staged  166   - 206   s   (paired 2nd impl)
#
# At dt=1.667 s and bldt=0 that is 7.3 - 8.2 wall seconds per simulated
# minute of land surface, against 4,982 - 5,977 through the retired
# per-column CPython solver whose flat 7.18 ms/column this block used to
# tabulate.  Absolute seconds are a property of the machine; this box varies
# up to 30% between harnesses and hours.
#
# The ceiling below is still the LARGEST MEASURED configuration and nothing
# wider.  It is not a performance target and raising it makes nothing
# faster: above it a Noah-MP request is refused unless the caller states the
# column budget they accept, which is the same shape as the retired Thompson
# enable gate once had -- an unmeasured width is refused before the run
# starts, not discovered mid-forecast.  Until 2026-07-27 the ceiling was
# 352, the widest the host-era solver was ever measured at; the slab
# measurement at d04's full 360,000 columns is what moved it.
NOAHMP_EXPERT_COLUMN_BUDGET_ENV = "GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET"
#: Largest column count at which Noah-MP throughput has been measured:
#: d04 of the four-domain production tree, slab path, 2026-07-27.
NOAHMP_MEASURED_COLUMN_CEILING = 360_000
#: Seconds per land-surface call at exactly that ceiling, the (low, high)
#: of two paired timing runs on one RTX 5090, 2026-07-27.  Replaces the
#: retired host-era ``NOAHMP_MEASURED_MS_PER_COLUMN`` (a flat 7.18
#: ms/column), which described the per-column CPython solver.
NOAHMP_MEASURED_SLAB_CALL_SECONDS = (0.202, 0.227)

# Public forecast-runner profile identifiers.  These strings are shared with
# the Rust Studio runtime manifest and are deliberately more specific than a
# bare WRF selector: selecting ``mp_physics=8`` must not silently inherit the
# fixed WSM6 runner configuration.
WSM6_PROFILE_ID = "wsm6-ysu-mm5-noah-no-radiation-v1"
THOMPSON_PROFILE_ID = "thompson-mp8-ysu-mm5-noah-validation-v1"
MORRISON_PROFILE_ID = (
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
)
NSSL2_PROFILE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1"
)
#: The MYNN 5/5 suite, which differs from :data:`WSM6_PROFILE_ID` in exactly
#: two selectors (``bl_pbl_physics`` 1 -> 5 and ``sf_sfclay_physics`` 91 -> 5).
#: It is a peer profile rather than an expert one because it RUNS at production
#: width: ``gpuwm/core/physics.py`` ``initialize_physics`` allocates every MYNN
#: array itself, so no runner needs extra wiring, and the standing runtime gate
#: forecasts 300 coupled steps with a bitwise restart
#: (``tests/test_mynn_pbl_runtime.py``).
MYNN_PROFILE_ID = "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"
#: The RUC land-surface suite.  It differs from :data:`WSM6_PROFILE_ID` in
#: exactly one selector, ``sf_surface_physics`` 2 -> 3, with
#: ``num_soil_layers`` 4 -> 9 following from it, so a side-by-side run
#: isolates the land-surface change from everything else.  It is a peer
#: profile rather than an expert one because its column runs on the CARD:
#: 16.8 wall seconds per simulated minute at d04's 360,000 columns snow-free
#: and 65.7 fully snow-covered, measured at that width, not extrapolated.
RUC_PROFILE_ID = "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
#: The profiles the HRRR single-domain path offers.  Neither
#: :data:`MYNN_PROFILE_ID` nor :data:`RUC_PROFILE_ID` is a member, although
#: both declare complete runtime switches below and both are offered by
#: ``tools/prepared_single_domain_forecast.py``: every consumer of THIS tuple
#: carries its own per-profile table -- ``_HRRR_SOURCE_ABSENT_STATE_DEFAULTS``
#: and ``_HRRR_SOURCE_ABSENT_WRF_FIELDS`` in
#: ``tools/hrrr_single_domain_benchmark.py``, and the pair map in
#: ``tools/prepare_hrrr_wrf.py`` -- and none of them has a row for either.
#: Membership would put the profile in ``gpuwm/source_cli.py``'s
#: ``--physics-profile`` choices and then ``KeyError`` on the first
#: source-absent field, which is selectable and not usable.  Adding one is a
#: change to those three tables, made with an HRRR case in hand.
SINGLE_DOMAIN_PHYSICS_PROFILES = (
    WSM6_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
)

# Complete runtime products shared by every source-specific single-domain
# launcher.  These are intentionally not bare microphysics selectors: each
# profile fixes the surrounding surface/PBL/cumulus/radiation/diffusion
# switches that were validated by the native HRRR runner.  Source adapters
# may materialize these switches into a case-specific experiment descriptor,
# but they must not reinterpret or partially apply them.
_SINGLE_DOMAIN_RUNTIME_SWITCHES = MappingProxyType({
    # Byte-for-byte the WSM6 row apart from the land surface:
    # sf_surface_physics 2 -> 3 and the nine soil layers RUC's own level
    # geometry needs.  Everything RUC pins that has no namelist field --
    # XICE_THRESHOLD, seaice_albedo_default, isncovr_opt, c1sn, c2sn, myj,
    # rdlai2d, FRACTIONAL_SEAICE -- is NOT restated here; it is published as
    # data in gpuwm.core.ruc_runtime.RUC_RUNTIME_RESTRICTIONS, and a profile
    # that repeated it could drift from it.
    RUC_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 91, "sf_surface_physics": 3,
        "bl_pbl_physics": 1, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 9, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
    WSM6_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
    THOMPSON_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 8,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
    MORRISON_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 10,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 1, "cudt_minutes": 5.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
    }),
    NSSL2_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 18,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 1, "cudt_minutes": 5.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
    }),
    # Byte-for-byte the WSM6 row apart from the two MYNN selectors.  Keeping it
    # that way is the point: a side-by-side run isolates the PBL/surface-layer
    # change from the microphysics, radiation, cumulus and diffusion settings.
    # The MYNN option identity (bl_mynn_*, icloud_bl, iz0tlnd) is NOT repeated
    # here -- those are RunConfig defaults that validate_run_config pins
    # unconditionally, so a profile that restated them could drift from them.
    MYNN_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 5, "sf_surface_physics": 2,
        "bl_pbl_physics": 5, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
})


def single_domain_runtime_switches(profile: str) -> dict[str, object]:
    """Return one complete canonical single-domain runtime product."""

    try:
        return dict(_SINGLE_DOMAIN_RUNTIME_SWITCHES[profile])
    except KeyError:
        raise ValueError(
            f"unsupported single-domain physics profile {profile!r}"
        ) from None


def identify_single_domain_profile(run_config) -> str | None:
    """Which shipped profile a run config's switches ARE, or ``None``.

    The inverse of :func:`single_domain_runtime_switches`, and the reason
    it lives beside it: a printed next-command has to name the
    ``--physics-profile`` the prepared-forecast runner will accept, and
    the only honest way to know is to ask the same table the runner's
    guard asks.  A second copy of this comparison somewhere else is how
    a printed command drifts into being wrong.

    Returns ``None`` when the config matches no profile exactly -- a
    hand-authored suite, or a profile this ArWen does not ship -- which
    the caller must report rather than guess around.  Ambiguity cannot
    arise (no two profiles share a switch set), but if it ever did,
    ``None`` is the answer: two names for one config is not an answer.
    """

    matched = [
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if all(getattr(run_config, name, _MISSING) == value
               for name, value in
               single_domain_runtime_switches(profile).items())
    ]
    return matched[0] if len(matched) == 1 else None


#: Sentinel for :func:`identify_single_domain_profile`: a config that
#: lacks a switch entirely never matches a profile that pins it.
_MISSING = object()


def thompson_runtime_requirements() -> dict[str, object]:
    """Describe the guarded evidence-runner MP8 contract (env untouched).

    This receipt is consumed by the guarded benchmark/evidence runners under
    ``tools/`` (hrrr_single_domain_benchmark, prepared_single_domain_forecast),
    which keep their own explicit environment gates.  The PRODUCT selection
    path (config validation, namelist import, forecast dispatch, restart
    identity) admits mp_physics=8 first-class with the packaged table root
    (:func:`thompson_table_root`); it does not read this dict.
    """

    from gpuwm.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS,
        TABLE_SET_ID,
        WRF_REFERENCE_COMMIT,
        WRF_REFERENCE_VERSION,
    )

    return {
        "readiness": "MODEL_VALIDATED_EXPERIMENTAL_RUNTIME",
        "explicit_expert_consent_required": False,
        "runtime_guard": {
            "environment": EXPERIMENTAL_THOMPSON_ENV,
            "required_value": "1",
        },
        "table_root": {
            "environment": THOMPSON_TABLE_ROOT_ENV,
            "must_name_directory": True,
            "validation": "exact-size-and-sha256-before-GPU-setup",
        },
        "table_authority": {
            "table_set": TABLE_SET_ID,
            "wrf_version": WRF_REFERENCE_VERSION,
            "wrf_commit": WRF_REFERENCE_COMMIT,
            "payload_bytes": sum(asset.bytes for asset in CLASSIC_TABLE_ASSETS),
            "assets": [
                {
                    "filename": asset.filename,
                    "bytes": asset.bytes,
                    "sha256": asset.sha256,
                }
                for asset in CLASSIC_TABLE_ASSETS
            ],
        },
        "capability_probe_validates_environment_or_table_bytes": False,
    }


def noahmp_expert_column_budget() -> int:
    """Columns the caller has explicitly accepted for a Noah-MP run.

    Zero when unset or unparseable, so a malformed budget is worth exactly as
    much as no budget at all.  A budget is never read as a ceiling on its own:
    :func:`pending_wrf_physics_components` takes the larger of it and the
    measured ceiling, so the environment can only widen what the caller has
    said they accept, never narrow the measured evidence.
    """

    raw = os.environ.get(NOAHMP_EXPERT_COLUMN_BUDGET_ENV)
    if raw is None:
        return 0
    try:
        budget = int(raw.strip())
    except ValueError:
        return 0
    return budget if budget > 0 else 0


def noahmp_projected_call_seconds(columns: int) -> float:
    """Projected wall clock of one land-surface call at ``columns`` columns.

    Linear from the top of the measured range at the 360,000-column ceiling
    (slab path, 2026-07-27), and stated as a PROJECTION: widths beyond the
    ceiling are exactly the ones nothing has measured, which is why the
    budget rail quotes this number instead of admitting them silently.
    """

    return (columns * NOAHMP_MEASURED_SLAB_CALL_SECONDS[1]
            / NOAHMP_MEASURED_COLUMN_CEILING)


@dataclass(frozen=True)
class PhysicsPortBlocker:
    """One coupled physics component requested before it is executable."""

    component: str
    selectors: tuple[tuple[str, int], ...]
    missing: tuple[str, ...]

    def format(self) -> str:
        selected = ", ".join(f"{key}={value}" for key, value in self.selectors)
        return f"{self.component} ({selected}): " + "; ".join(self.missing)


class UnsupportedPhysicsSuiteError(ValueError):
    """A requested WRF suite contains one or more unfinished components."""

    def __init__(self, blockers: tuple[PhysicsPortBlocker, ...]):
        if not blockers:
            raise ValueError("UnsupportedPhysicsSuiteError needs blockers")
        self.blockers = blockers
        details = "\n".join(f"  - {item.format()}" for item in blockers)
        super().__init__(
            "requested WRF physics suite is not executable in gpuwm yet; "
            "no substitutions were applied:\n" + details)


def pending_wrf_physics_components(
        *, mp_physics: int, sf_sfclay_physics: int,
        bl_pbl_physics: int, sf_surface_physics: int,
        num_soil_layers: int,
        columns: int | None = None,
        ) -> tuple[PhysicsPortBlocker, ...]:
    """Return unfinished components selected by a WRF physics request.

    MYNN surface layer and PBL are one coupled port.  Reporting them together
    prevents either half from being mistaken for a scientifically meaningful
    stand-alone implementation.  RUC's layer count is included in its blocker
    even though WRF also supports a six-layer RUC configuration; the target
    suite explicitly requests nine.

    This function is the single readiness authority.  ``gpuwm/config.py``
    deliberately accepts every selector value in its schema tables and lets
    this receipt do the refusing, so admitting a scheme is an edit here and
    a dispatch row in ``gpuwm/core/physics.py`` -- never a silent widening of
    a numeric range check.
    """
    blockers: list[PhysicsPortBlocker] = []
    # mp_physics == 8 no longer appends a blocker: Thompson was promoted to
    # a first-class selection when the canonical classic tables became
    # package data (see the EXPERIMENTAL_THOMPSON_ENV comment block above).
    # Byte validation of the resolved table root still fails closed at
    # setup, which is the guard that was ever load-bearing at run time.
    if (sf_sfclay_physics == 5) != (bl_pbl_physics == 5):
        # MYNN is admitted only as the coupled 5/5 suite.  Its surface layer
        # diagnoses T2/Q2/TH2 itself and its PBL driver consumes UST/HFX/QFX/
        # WSPD/RMOL from that same surface layer, so half a suite is not a
        # smaller configuration -- it is a different, unvalidated one.
        blockers.append(PhysicsPortBlocker(
            component="MYNN half-suite",
            selectors=(("sf_sfclay_physics", sf_sfclay_physics),
                       ("bl_pbl_physics", bl_pbl_physics)),
            missing=(
                "MYNN is admitted only as the coupled pair "
                "sf_sfclay_physics=5 with bl_pbl_physics=5",
                "the surface layer supplies UST/HFX/QFX/WSPD/RMOL to the PBL "
                "driver and diagnoses T2/Q2/TH2 in place of SFCDIAGS",
            )))
    if sf_surface_physics == 3 and num_soil_layers != 9:
        # RUC itself is no longer refused: it has a dispatch row, a seam, a
        # cold start, restart identity and output.  What is still refused is
        # the six-layer geometry WRF also defines.  This is a narrower
        # blocker than the old one, not a widened gate -- the old row refused
        # every RUC request.
        blockers.append(PhysicsPortBlocker(
            component="RUC soil geometry",
            selectors=(("sf_surface_physics", 3),
                       ("num_soil_layers", num_soil_layers)),
            missing=(
                "RUC is admitted at num_soil_layers=9 only",
                "share/module_soil_pre.F:init_soil_depth_3 also tabulates a "
                "six-level RUC grid, but every RUC oracle fixture in the "
                "tree is nine-level and the CUDA leaves index a "
                "__constant__ real ruc_soil_layer_depth[9]",
            )))
    if sf_surface_physics == 3 and sf_sfclay_physics == 5:
        # The same shape as the Noah-MP/MYNN refusal below, and for the same
        # reason: MYNN's surface layer diagnoses T2/Q2/TH2 itself, and RUC
        # brings its OWN 2-m diagnostic (SFCDIAGS_RUCLSM) which would
        # overwrite MYNN's with a differently-derived one.  Nobody has
        # measured that pair.
        blockers.append(PhysicsPortBlocker(
            component="MYNN surface layer with RUC",
            selectors=(("sf_sfclay_physics", 5),
                       ("sf_surface_physics", 3)),
            missing=(
                "RUC is admitted with the MM5 surface layer only "
                "(sf_sfclay_physics 1 or 91)",
                "RUC runs SFCDIAGS_RUCLSM after the LSM and MYNN diagnoses "
                "T2/Q2/TH2 itself, so the pair has two 2-m diagnostics and "
                "the second silently wins",
            )))
    if sf_surface_physics == 4 and sf_sfclay_physics == 5:
        # The registry blocks this pair at plan time through
        # requires_components; this is the runtime half of the same refusal,
        # so the two authorities cannot disagree.  It is not a "MYNN is
        # broken" statement: MYNN's surface layer diagnoses T2/Q2/TH2 itself
        # and supplies UST/HFX/QFX, all of which Noah-MP also writes, so the
        # pair is a coupling nobody has measured.
        blockers.append(PhysicsPortBlocker(
            component="MYNN surface layer with Noah-MP",
            selectors=(("sf_sfclay_physics", 5),
                       ("sf_surface_physics", 4)),
            missing=(
                "Noah-MP is admitted with the MM5 surface layer only "
                "(sf_sfclay_physics 1 or 91)",
                "MYNN diagnoses T2/Q2/TH2 in place of SFCDIAGS and supplies "
                "UST/HFX/QFX, which Noah-MP's own write-back overwrites",
            )))
    if sf_surface_physics == 4 and columns is not None:
        # The one blocker here that is about GRID WIDTH rather than a missing
        # branch.  Noah-MP is fully ported and bitwise, and since the slab
        # orchestration it also FINISHES at production width; what stays
        # refused is a width nothing has measured.  ``columns`` is the whole
        # grid, i.e. the worst case, because the land fraction is not known
        # until the landmask is ingested and a fail-closed rail must not
        # assume the friendlier number.
        budget = max(NOAHMP_MEASURED_COLUMN_CEILING,
                     noahmp_expert_column_budget())
        if columns > budget:
            blockers.append(PhysicsPortBlocker(
                component="Noah-MP column budget",
                selectors=(("sf_surface_physics", 4), ("columns", columns)),
                missing=(
                    f"Noah-MP's slab path is measured to "
                    f"{NOAHMP_MEASURED_COLUMN_CEILING} columns "
                    f"({NOAHMP_MEASURED_SLAB_CALL_SECONDS[0]}-"
                    f"{NOAHMP_MEASURED_SLAB_CALL_SECONDS[1]} s per "
                    f"land-surface call on one RTX 5090, 2026-07-27); "
                    f"{columns} columns project linearly to "
                    f"{noahmp_projected_call_seconds(columns):.2f} s per "
                    f"call, and no run has measured that width",
                    f"set {NOAHMP_EXPERT_COLUMN_BUDGET_ENV} to the column "
                    f"count you accept to run beyond the measured width",
                    "raising the budget makes nothing faster -- it records "
                    "that the projection above was read and accepted",
                )))
    if sf_surface_physics == 4 and num_soil_layers != 4:
        # Noah-MP itself is no longer refused: it has a dispatch row, a
        # driver, a cold start, restart identity and output.  What is still
        # refused is a soil geometry no Noah-MP fixture covers.  This is a
        # narrower blocker than the old one, not a widened gate: the old row
        # refused every Noah-MP request.
        blockers.append(PhysicsPortBlocker(
            component="Noah-MP soil geometry",
            selectors=(("sf_surface_physics", 4),
                       ("num_soil_layers", num_soil_layers)),
            missing=(
                "Noah-MP is admitted at num_soil_layers=4 only",
                "every Noah-MP oracle fixture in the tree is four-layer, so "
                "another count would extrapolate TRANSFER_MP_PARAMETERS to "
                "layers nothing has measured",
            )))
    return tuple(blockers)


def require_ready_wrf_physics(**selection: int) -> None:
    """Raise a complete fail-closed receipt for any pending component."""
    blockers = pending_wrf_physics_components(**selection)
    if blockers:
        raise UnsupportedPhysicsSuiteError(blockers)


__all__ = [
    "EXPERIMENTAL_THOMPSON_ENV",
    "MYNN_PROFILE_ID",
    "NOAHMP_EXPERT_COLUMN_BUDGET_ENV",
    "NOAHMP_MEASURED_COLUMN_CEILING",
    "NOAHMP_MEASURED_SLAB_CALL_SECONDS",
    "SINGLE_DOMAIN_PHYSICS_PROFILES",
    "MORRISON_PROFILE_ID",
    "noahmp_expert_column_budget",
    "noahmp_projected_call_seconds",
    "NSSL2_PROFILE_ID",
    "RUC_PROFILE_ID",
    "PhysicsPortBlocker",
    "RRTMG_VARIANT_LEGACY",
    "RRTMG_VARIANT_RTE_RRTMGP",
    "THOMPSON_PROFILE_ID",
    "THOMPSON_TABLE_ROOT_ENV",
    "UnsupportedPhysicsSuiteError",
    "WRF_RRTMG_COMPATIBILITY_TOKENS",
    "WRF_RRTMG_LEGACY",
    "WSM6_PROFILE_ID",
    "WRF_RRTMG_TO_RTE_RRTMGP",
    "WRF_RRTMG_TO_RTE_RRTMGP_V1",
    "identify_single_domain_profile",
    "packaged_thompson_table_root",
    "pending_wrf_physics_components",
    "thompson_table_root",
    "require_ready_wrf_physics",
    "require_rrtmg_legacy_executable",
    "require_rrtmg_legacy_ready",
    "rrtmg_variant",
    "single_domain_runtime_switches",
    "thompson_runtime_requirements",
]
