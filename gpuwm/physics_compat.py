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
from typing import Mapping

from gpuwm.explain import layered, warn
from gpuwm.wrf461_compatibility import (
    PBL_OPTIONS,
    SURFACE_LAYER_OPTIONS,
    WRFVerdict,
    pbl_surface_layer_verdict,
)


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

    if isinstance(cfg, Mapping):
        return str(cfg.get(
            "ra_rrtmg_variant", RRTMG_VARIANT_RTE_RRTMGP))
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

#: The registry option id aerosol-aware Thompson (``mp_physics=28``) resolves
#: to.  Named once, here, because three different questions have to agree on
#: it: the vertical-bounds dispatch below, the registry document's
#: ``components.microphysics.options`` key, and any test that asserts what a
#: selector resolves to.  A literal repeated in those three places is how a
#: renamed option silently stops being bounds-checked.
MP28_REGISTRY_OPTION_ID = "thompson-aerosol-mp28"


def packaged_thompson_table_root() -> "Path":
    """The in-package canonical classic-table directory.

    The four assets and their MANIFEST.sha256 are committed under
    ``gpuwm/data/thompson/tables``; byte identity against
    ``gpuwm.core.thompson_contract.CLASSIC_TABLE_ASSETS`` is enforced at
    load time, not assumed here.
    """

    from pathlib import Path

    return Path(__file__).resolve().parent / "data" / "thompson" / "tables"


def user_thompson_table_root() -> "Path":
    """User-level staging directory: ``~/.gpuwm/tables/thompson``.

    The same place ``~/.gpuwm/bridges`` occupies for the built bridges,
    and for the same reason: ``gpuwm fetch-tables`` used to stage its
    two downloads *inside site-packages*, where the next wheel upgrade
    or venv rebuild deletes them without saying so.  A user who had
    already paid for a 315 MiB download then met a bare
    ``FileNotFoundError`` in the middle of a forecast.  Under the home
    directory the staged set outlives the install that fetched it, and
    every checkout and venv on the machine reads the same bytes.

    Named per table set rather than a flat ``tables/`` because a root is
    validated as a *complete set*: mixing two schemes' assets in one
    directory would make "complete" unanswerable.
    """

    from pathlib import Path

    return Path.home() / ".gpuwm" / "tables" / "thompson"


def _table_root_is_complete(root: "Path") -> bool:
    """Every classic asset filename present as a file in ``root``.

    Presence only.  The SHA-256 pins are re-checked at load by
    ``validate_table_assets``; resolution asks the cheap question
    (which root can this run read?) and answers it with stats, not with
    362 MiB of hashing on every import.
    """

    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS

    try:
        return all((root / asset.filename).is_file()
                   for asset in CLASSIC_TABLE_ASSETS)
    except OSError:  # pragma: no cover - unreadable home/site-packages
        return False


def thompson_table_root() -> str:
    """Resolved mp8 table root: env override, then staged, then packaged.

    The override exists for byte-identical mirrors (fast local disks,
    cluster scratch); a root with different bytes fails closed in
    ``validate_table_assets`` exactly as the packaged one would.

    Below it, a *complete* packaged directory still answers first --
    that is the read fallback that keeps every install which already
    staged into site-packages working, and it makes a git clone (whose
    packaged root ships the whole set) resolve exactly where it always
    did, the same way a checkout's own bridge build outranks a staged
    one in :func:`gpuwm.bridges.artifact_candidates`.  When the packaged
    directory is short an asset -- every fresh wheel, and every wheel an
    upgrade has just emptied -- the complete staged set under
    :func:`user_thompson_table_root` answers instead.  Both roots are
    pinned to the same bytes, so this order chooses a location, never a
    numerical setup.
    """

    override = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    if override:
        return override
    packaged = packaged_thompson_table_root()
    if _table_root_is_complete(packaged):
        return str(packaged)
    try:
        staged = user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return str(packaged)
    if _table_root_is_complete(staged):
        return str(staged)
    return str(packaged)


def thompson_guard_exports() -> tuple[str, str]:
    """The two exports the guarded mp8 runners demand, ready to paste.

    Selection through the library is first-class and needs neither
    variable (see the block above).  The evidence and benchmark runners
    under ``tools/`` kept the launch contract on purpose, and a field run
    of the shipped 1.5.0 wheel met it as two consecutive one-line
    RuntimeErrors with no value in either: the reader had to find the
    table root themselves, having already downloaded it.

    So the pair is composed ONCE, here, beside the names and the
    resolver -- the generated command chain prints it before the stage
    that needs it, and the refusal that fires when it is missing prints
    the same two lines.  The root is this install's own resolution, so
    the printed line is the root the loader would have used, not a
    guess; ``validate_table_assets`` still re-checks every byte, so a
    root that has been pasted from somewhere else fails closed exactly
    as it did before.

    The shell form follows the platform, because a POSIX ``export`` line
    pasted into PowerShell is a syntax error and the reader is then
    debugging the instructions instead of the run.
    """

    import shlex

    from gpuwm.bridges import WINDOWS_SHELL

    root = thompson_table_root()
    if WINDOWS_SHELL:
        return (f'$env:{EXPERIMENTAL_THOMPSON_ENV} = "1"',
                f'$env:{THOMPSON_TABLE_ROOT_ENV} = "{root}"')
    return (f"export {EXPERIMENTAL_THOMPSON_ENV}=1",
            f"export {THOMPSON_TABLE_ROOT_ENV}={shlex.quote(str(root))}")

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
#: The native-HRRR warm-rain profile.  WRF v4.6.1
#: ``Registry/Registry.EM_COMMON:3015`` declares Kessler's qv/qc/qr package.
#: The profile is admitted only with the end-to-end HRRR probe and its
#: source-frozen-species discard receipt.
KESSLER_PROFILE_ID = "kessler-mp1-ysu-mm5-noah-dudhia-v1"
THOMPSON_PROFILE_ID = "thompson-mp8-ysu-mm5-noah-validation-v1"
#: The observation battery's registered composition (lead ruling,
#: obs-battery integration wave 2026-08-04): the Thompson validation
#: suite with the exact WRF v4.6.1 legacy RRTMG in place of no-radiation,
#: cumulus off, transcribed switch for switch from the battery's
#: registered configuration.  wrf-matched-run-candidate in the registry:
#: the composition's first stock-WRF-paired t0/case receipt is the named
#: upgrade payer.
THOMPSON_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
)
#: The Shin-Hong sibling of the row above: the SAME composition with the
#: gray-zone PBL in place of YSU (``bl_pbl_physics`` 1 -> 11), which is
#: the one edge the divergence ledger's L3 entry moves
#: (:mod:`gpuwm.physics_mode`).  It is registered because a fidelity-axis
#: arm that selects L3 resolves to exactly this suite, and an arm whose
#: physics no profile names cannot have a root prepared for it at all.
#: Every other switch is the row above's, transcribed rather than
#: re-derived, so the two rows differ in the PBL and nothing else and a
#: paired run of them isolates the closure.  wrf-matched-run-candidate in
#: the registry on the same terms as its sibling: Shin-Hong's own port is
#: measured bitwise against WRF v4.6.1 on both halves, and the payer that
#: moves the label is the composition's first stock-WRF-paired t0/case
#: receipt.
THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"
)
MORRISON_PROFILE_ID = (
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
)
NSSL2_PROFILE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1"
)
NSSL2_LEGACY_RRTMG_PROFILE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1"
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
#: HRRR's operational suite class: MYNN surface/PBL with the RUC LSM.
MYNN_RUC_PROFILE_ID = (
    "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1"
)
#: The Noah-MP fixed template remains expert-only because its component is
#: implemented but has no GPUWM/WRF forecast-trajectory comparison.  The
#: registry owns the acknowledgement id and warnings; callers must not
#: duplicate them as a profile-name special case.
NOAHMP_PROFILE_ID = (
    "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
)
#: The MYNN 5/5 counterpart of the expert Noah-MP template.
MYNN_NOAHMP_PROFILE_ID = (
    "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1"
)
#: Every fixed single-domain template the front door can validate.  Expert
#: templates remain in this discovery tuple: selection is accepted by the
#: parser, then the registry-owned acknowledgement/capability checks below
#: fail closed before preparation if consent or an implementation is absent.
SINGLE_DOMAIN_PHYSICS_PROFILES = (
    WSM6_PROFILE_ID,
    KESSLER_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    MYNN_PROFILE_ID,
    RUC_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
)

# Complete runtime products shared by every source-specific single-domain
# launcher.  These are intentionally not bare microphysics selectors: each
# profile fixes the surrounding surface/PBL/cumulus/radiation/diffusion
# switches that were validated by the native HRRR runner.  Source adapters
# may materialize these switches into a case-specific experiment descriptor,
# but they must not reinterpret or partially apply them.
_SINGLE_DOMAIN_RUNTIME_SWITCHES = MappingProxyType({
    KESSLER_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 1,
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
    MYNN_NOAHMP_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 5, "sf_surface_physics": 4,
        "bl_pbl_physics": 5, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
    NOAHMP_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 91, "sf_surface_physics": 4,
        "bl_pbl_physics": 1, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.08,
        "diff_6th_slopeopt": 1,
    }),
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
    MYNN_RUC_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": False, "mp_physics": 6,
        "top_lid": True, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 0, "ra_sw_physics": 1, "radt": 1.0,
        "wrf_rrtmg_compatibility": "none",
        "sf_sfclay_physics": 5, "sf_surface_physics": 3,
        "bl_pbl_physics": 5, "cu_physics": 0, "cudt_minutes": 0.0,
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
    # The Thompson row with the exact legacy RRTMG in place of
    # no-radiation, every value transcribed from
    # configs/battery/shape_3km_thompson_rrtmg_legacy.toml as registered
    # (radt 12.0 and diff_6th_factor 0.12 are that config's values, not
    # the validation row's 1.0/0.08).
    THOMPSON_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 8,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_LEGACY,
        "ra_rrtmg_variant": RRTMG_VARIANT_LEGACY,
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
    }),
    # The row above with ONE switch moved: bl_pbl_physics 1 -> 11, the
    # divergence ledger's L3 edge (gpuwm/physics_mode.py).  Every other
    # value is transcribed from that row rather than re-derived, because
    # the pair's whole purpose is that a paired run isolates the closure;
    # a second value moving here would make the comparison a composition
    # comparison instead.  sf_sfclay_physics stays 91: WRF v4.6.1's
    # SHINHONGSCHEME arm (phys/module_physics_init.F:3702-3704) requires
    # isfc=1 exactly as YSU does, which the classic MM5 surface layer is.
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 8,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_LEGACY,
        "ra_rrtmg_variant": RRTMG_VARIANT_LEGACY,
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 11, "cu_physics": 0, "cudt_minutes": 0.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
    }),
    MORRISON_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 10,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
        "ra_rrtmg_variant": RRTMG_VARIANT_RTE_RRTMGP,
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
        "ra_rrtmg_variant": RRTMG_VARIANT_RTE_RRTMGP,
        "sf_sfclay_physics": 91, "sf_surface_physics": 2,
        "bl_pbl_physics": 1, "cu_physics": 1, "cudt_minutes": 5.0,
        "num_soil_layers": 4, "terrain_opt": 1,
        "km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
    }),
    NSSL2_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "moist": True, "moist_cq": True, "mp_physics": 18,
        "top_lid": False, "epssm": 0.5, "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0, "ra_physics": 0,
        "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_LEGACY,
        "ra_rrtmg_variant": RRTMG_VARIANT_LEGACY,
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


#: Runtime switches a WRF namelist cannot state honestly, so somebody has
#: to decide them for it.  ``moist_cq`` has no WRF namelist key at all
#: (WRF derives calc_cq internally from the moist species that exist),
#: and gpuwm's ``top_lid`` default is deliberately NOT WRF's Registry
#: default (gpuwm/config.py records the falsification bar for flipping
#: it).  Both are part of every prepared-cache domain identity.
#:
#: They had three answers.  The shipped profiles above state them; the
#: domain wizard reads those same profiles; the WRF importer invented its
#: own -- ``moist_cq = mp_physics > 0`` and WRF's open-top default -- and
#: so a public root prepared from a profile and a public hierarchy
#: imported from the SAME namelist could not produce matching d01
#: identities for WSM6, Kessler, MYNN, RUC or Noah-MP.  This function is
#: the single answer all three now ask for.
IMPLICIT_RUNTIME_SWITCHES = ("moist_cq", "top_lid")

#: The switches that pick a shipped profile's row.  Every profile above
#: is unique on this tuple except the NSSL pair, which differs only in
#: its RRTMG variant and agrees on both implicit switches -- so a tie is
#: an answer here, not an ambiguity.
_PROFILE_SELECTOR_KEYS = (
    "mp_physics", "sf_sfclay_physics", "sf_surface_physics",
    "bl_pbl_physics", "cu_physics", "num_soil_layers",
    "ra_lw_physics", "ra_sw_physics",
)


def implicit_runtime_switches(**selection) -> dict[str, object]:
    """Certified ``moist_cq``/``top_lid`` for one WRF physics selection.

    ``selection`` is the WRF switch set an importer already resolved
    (:data:`_PROFILE_SELECTOR_KEYS`; unknown keys are ignored so callers
    may pass their whole suite).  When every shipped profile matching it
    agrees, those are the values -- byte-for-byte the ones
    :func:`single_domain_runtime_switches` hands the root preparer and
    the domain wizard.  When nothing matches, or matches disagree, the
    answer is gpuwm's own ``RunConfig`` default, which is what an
    unstated switch has always resolved to.

    Returns ``{"moist_cq": ..., "top_lid": ..., "source": ...,
    "profiles": (...)}``: the source string is a receipt line, because a
    value decided FOR the user has to be able to say who decided it.
    """

    from gpuwm.config import RunConfig

    requested = {key: selection[key] for key in _PROFILE_SELECTOR_KEYS
                 if key in selection}
    matched = []
    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        row = _SINGLE_DOMAIN_RUNTIME_SWITCHES[profile]
        if requested and all(row.get(key) == value
                             for key, value in requested.items()):
            matched.append(profile)
    answers = {
        tuple(_SINGLE_DOMAIN_RUNTIME_SWITCHES[profile][name]
              for name in IMPLICIT_RUNTIME_SWITCHES)
        for profile in matched
    }
    if len(answers) == 1:
        row = _SINGLE_DOMAIN_RUNTIME_SWITCHES[matched[0]]
        return {
            **{name: row[name] for name in IMPLICIT_RUNTIME_SWITCHES},
            "source": (
                "the shipped single-domain physics profile this suite IS "
                if len(matched) == 1 else
                "the shipped single-domain physics profiles this suite IS, "
                "which agree here: ")
            + ", ".join(matched),
            "profiles": tuple(matched),
        }
    defaults = RunConfig(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, ztop=1.0,
                         dt=1.0, run_seconds=1.0)
    return {
        **{name: getattr(defaults, name)
           for name in IMPLICIT_RUNTIME_SWITCHES},
        "source": (
            "gpuwm's RunConfig defaults: this suite is not one of the "
            "shipped single-domain physics profiles"
            if not matched else
            "gpuwm's RunConfig defaults: this suite matches shipped "
            "profiles that disagree about these switches "
            f"({', '.join(sorted(matched))})"),
        "profiles": tuple(matched),
    }


class PhysicsCapabilityError(ValueError):
    """A selector combination has no executable registry capability."""


ACK_FLAG_SOURCE = "--ack"
ACK_TOML_SOURCE = "[experiment].acknowledgements"


def acknowledgement_delivery(
        *,
        flag: tuple[str, ...] = (),
        toml: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """Merge acknowledgement delivery and retain its exact provenance."""

    delivered: dict[str, set[str]] = {}
    for source, values in (
        (ACK_FLAG_SOURCE, flag),
        (ACK_TOML_SOURCE, toml),
    ):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{source} acknowledgement IDs must be non-empty strings")
            delivered.setdefault(value, set()).add(source)
    return (
        tuple(sorted(delivered)),
        {
            value: sorted(sources)
            for value, sources in sorted(delivered.items())
        },
    )


def _acknowledgement_receipt(
        acknowledged: set[str],
        required: set[str],
        provenance: Mapping[str, object] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    used = sorted(acknowledged & required)
    sources: dict[str, list[str]] = {}
    for acknowledgement in used:
        raw = None if provenance is None else provenance.get(acknowledgement)
        if isinstance(raw, (list, tuple)) and all(
                isinstance(value, str) for value in raw):
            sources[acknowledgement] = sorted(set(raw))
        else:
            sources[acknowledgement] = ["api"]
    return used, sources


def _ack_instruction(acknowledgement: str) -> str:
    return (
        f"--ack {acknowledgement} or "
        f'acknowledgements = ["{acknowledgement}"]'
    )


#: Sentinel for "this caller did not mention the selector at all", which is
#: a different statement from "it named None".
_ABSENT = object()


def _selection_value(settings: Mapping[str, object] | object, name: str):
    value = _selection_value_or_absent(settings, name)
    return None if value is _ABSENT else value


def _selection_value_or_absent(settings: Mapping[str, object] | object,
                               name: str):
    if isinstance(settings, Mapping):
        return settings.get(name, _ABSENT)
    return getattr(settings, name, _ABSENT)


def _registry_pointer(component_id: str, option_id: str | None = None) -> str:
    pointer = (
        "gpuwm/physics_registry_v2.json#/components/"
        f"{component_id}"
    )
    if option_id is not None:
        pointer += f"/options/{option_id}"
    return pointer


def _resolve_physics_component_options(
        settings: Mapping[str, object] | object,
) -> tuple[dict[str, str], dict[str, Mapping[str, object]]]:
    """Resolve implemented component options from selectors only."""
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    resolved: dict[str, str] = {}
    options_by_component: dict[str, Mapping[str, object]] = {}
    for component_id, raw_component in registry["components"].items():
        component = raw_component
        selector_keys = tuple(component.get("selector_keys", ()))
        if not selector_keys:
            continue
        raw_selected = {
            key: _selection_value_or_absent(settings, key)
            for key in selector_keys
        }
        if all(value is _ABSENT for value in raw_selected.values()):
            # A caller that never mentions a component's selectors is not
            # selecting that component, and resolving one it did not ask
            # about cannot be right.  This matters because a selector can
            # MOVE here: km_opt was a registry parameter with a default
            # until it became components/turbulence's selector key, and
            # every caller that passes an explicit settings mapping --
            # gpuwm.physics_compat.validate_resolved_physics_vertical_levels
            # is the public one -- was written before the component
            # existed and omits it.  Refusing all of them because one key
            # is unmentioned would be a silent contract change on a
            # public API, and it reported itself as "no implemented
            # option for selectors {'km_opt': None}", which names a
            # value nobody wrote.  An object (a RunConfig) always carries
            # its own default, so nothing on the run path is skipped.
            continue
        selected = {
            key: (None if value is _ABSENT else value)
            for key, value in raw_selected.items()
        }
        candidates = []
        for option_id, raw_option in component["options"].items():
            option = raw_option
            selectors = option.get("selectors", {})
            if (isinstance(selectors, Mapping)
                    and set(selectors) == set(selector_keys)
                    and all(selected[key] == selectors[key]
                            for key in selector_keys)):
                candidates.append((option_id, option))
        if len(candidates) != 1:
            raise PhysicsCapabilityError(
                f"{_registry_pointer(component_id)} has no implemented "
                f"option for selectors {selected}; no source/profile "
                "substitution is allowed")
        option_id, option = candidates[0]
        if option.get("implemented") is not True:
            declaration = option.get("reachability", {})
            blocker = (
                declaration.get("blocker")
                if isinstance(declaration, Mapping) else None
            )
            raise PhysicsCapabilityError(
                f"{_registry_pointer(component_id, option_id)} blocks "
                f"selectors {selected}: "
                f"{blocker or 'component is declared unimplemented'}")
        resolved[component_id] = option_id
        options_by_component[component_id] = option
    return resolved, options_by_component


def validate_physics_capabilities(
        settings: Mapping[str, object] | object,
) -> dict[str, str]:
    """Resolve selectors to implemented registry components, fail closed.

    This check deliberately knows no source id and no profile id.  It answers
    only whether the selected component implementations and their couplings
    exist.  The returned mapping is component id -> option id and is suitable
    for comparing a named template after capability has been established.
    """

    resolved, options_by_component = _resolve_physics_component_options(
        settings)

    for component_id, option in options_by_component.items():
        constraints = option.get("constraints", {})
        if not isinstance(constraints, Mapping):
            continue
        required_settings = constraints.get("required_settings", {})
        if isinstance(required_settings, Mapping):
            drift = {
                name: {
                    "selected": _selection_value(settings, name),
                    "required": required,
                }
                for name, required in required_settings.items()
                if _selection_value(settings, name) is not None
                and _selection_value(settings, name) != required
            }
            if drift:
                option_id = resolved[component_id]
                raise PhysicsCapabilityError(
                    f"{_registry_pointer(component_id, option_id)} requires "
                    f"settings {drift}")
        requirements = constraints.get("requires_components", {})
        if isinstance(requirements, Mapping):
            for required_component, allowed in requirements.items():
                selected_option = resolved.get(required_component)
                if (not isinstance(allowed, list)
                        or selected_option not in allowed):
                    option_id = resolved[component_id]
                    raise PhysicsCapabilityError(
                        f"{_registry_pointer(component_id, option_id)} "
                        f"requires {required_component} in {allowed}, got "
                        f"{selected_option!r}")

    # Runtime-only coupled restrictions and measured-width rails remain the
    # executable authority.  They cite the exact WRF/CUDA reason in their
    # blocker receipts and are intentionally applied after registry
    # implementation resolution.
    runtime_selection = {
        name: int(_selection_value(settings, name))
        for name in (
            "mp_physics", "sf_sfclay_physics", "bl_pbl_physics",
            "sf_surface_physics", "num_soil_layers",
        )
    }
    nx = _selection_value(settings, "nx")
    ny = _selection_value(settings, "ny")
    if (isinstance(nx, int) and not isinstance(nx, bool)
            and isinstance(ny, int) and not isinstance(ny, bool)):
        runtime_selection["columns"] = nx * ny
    require_ready_wrf_physics(**runtime_selection)
    return resolved


class PhysicsVerticalPreflightError(ValueError):
    """Resolved physics cannot execute on the requested vertical grid."""


def validate_resolved_physics_vertical_levels(
        settings: Mapping[str, object] | object, *,
        p_top: float | None = None,
) -> dict[str, object]:
    """Aggregate component-owned first-call vertical bounds.

    ``p_top=None`` performs the RunConfig-only checks.  Experiment
    preparation calls again with its authoritative model-top pressure so the
    radiation adapters can include their WRF above-model cap layers.
    """

    resolved, _ = _resolve_physics_component_options(settings)
    raw_nz = _selection_value(settings, "nz")
    if isinstance(raw_nz, bool):
        raise PhysicsVerticalPreflightError("nz must be an integer")
    try:
        nz = int(raw_nz)
    except (TypeError, ValueError):
        raise PhysicsVerticalPreflightError(
            f"nz must be an integer, got {raw_nz!r}") from None

    checks: list[dict[str, object]] = []
    violations: list[str] = []

    def bounded(
            label: str,
            bounds: tuple[int | None, int | None],
    ) -> None:
        minimum, maximum = bounds
        checks.append({
            "component": label,
            "model_levels": nz,
            "minimum": minimum,
            "maximum": maximum,
        })
        below = minimum is not None and nz < minimum
        above = maximum is not None and nz > maximum
        if below or above:
            if minimum is None:
                wording = f"nz <= {maximum}"
            elif maximum is None:
                wording = f"nz >= {minimum}"
            else:
                wording = f"{minimum} <= nz <= {maximum}"
            violations.append(f"{label} requires {wording}, got nz={nz}")

    from gpuwm.physics_vertical_contract import (
        KESSLER_VERTICAL_LEVEL_BOUNDS,
        GF_VERTICAL_LEVEL_BOUNDS,
        KF_VERTICAL_LEVEL_BOUNDS,
        MAX_LEGACY_LONGWAVE_LAYERS,
        MAX_LEGACY_SHORTWAVE_LAYERS,
        MAX_RRTMGP_LAYERS,
        MORRISON_VERTICAL_LEVEL_BOUNDS,
        MYNN_VERTICAL_LEVEL_BOUNDS,
        NSSL2_VERTICAL_LEVEL_BOUNDS,
        THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS,
        THOMPSON_VERTICAL_LEVEL_BOUNDS,
        WSM6_VERTICAL_LEVEL_BOUNDS,
        legacy_radiation_layer_counts,
        rrtmgp_above_model_layer_counts,
    )

    if resolved.get("pbl") == "mynn":
        bounded("MYNN PBL", MYNN_VERTICAL_LEVEL_BOUNDS)
    if resolved.get("cumulus") == "kain-fritsch":
        bounded("Kain-Fritsch cumulus", KF_VERTICAL_LEVEL_BOUNDS)
    if resolved.get("cumulus") == "grell-freitas":
        bounded("Grell-Freitas cumulus", GF_VERTICAL_LEVEL_BOUNDS)

    microphysics = resolved.get("microphysics")
    if microphysics == "kessler-mp1":
        bounded("Kessler microphysics", KESSLER_VERTICAL_LEVEL_BOUNDS)
    elif microphysics == "wsm6-mp6":
        bounded("WSM6 microphysics", WSM6_VERTICAL_LEVEL_BOUNDS)
    elif microphysics == "thompson-mp8":
        bounded("Thompson microphysics", THOMPSON_VERTICAL_LEVEL_BOUNDS)
    elif microphysics == MP28_REGISTRY_OPTION_ID:
        bounded("Thompson aerosol-aware microphysics",
                THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS)
    elif microphysics == "morrison-mp10":
        bounded("Morrison microphysics", MORRISON_VERTICAL_LEVEL_BOUNDS)
    elif microphysics == "nssl2-mp18":
        bounded("NSSL-2 microphysics", NSSL2_VERTICAL_LEVEL_BOUNDS)

    if resolved.get("radiation") in {
            "rte-rrtmgp", "rte-rrtmgp-legacy-aggregate"}:
        top_pressure = 0.0 if p_top is None else float(p_top)
        if rrtmg_variant(settings) == RRTMG_VARIANT_LEGACY:
            lw_layers, sw_layers = legacy_radiation_layer_counts(
                nz, top_pressure)
            checks.extend((
                {
                    "component": "legacy RRTMG longwave",
                    "model_levels": nz,
                    "above_model_layers": lw_layers - nz,
                    "total_layers": lw_layers,
                    "maximum": MAX_LEGACY_LONGWAVE_LAYERS,
                },
                {
                    "component": "legacy RRTMG shortwave",
                    "model_levels": nz,
                    "above_model_layers": sw_layers - nz,
                    "total_layers": sw_layers,
                    "maximum": MAX_LEGACY_SHORTWAVE_LAYERS,
                },
            ))
            if lw_layers > MAX_LEGACY_LONGWAVE_LAYERS:
                violations.append(
                    "legacy RRTMG longwave requires model plus cap layers "
                    f"<= {MAX_LEGACY_LONGWAVE_LAYERS}, got {nz}+"
                    f"{lw_layers - nz}={lw_layers}")
            if sw_layers > MAX_LEGACY_SHORTWAVE_LAYERS:
                violations.append(
                    "legacy RRTMG shortwave requires model plus wrapper "
                    f"layers <= {MAX_LEGACY_SHORTWAVE_LAYERS}, got {nz}+"
                    f"{sw_layers - nz}={sw_layers}")
        else:
            lw_upper, sw_upper = rrtmgp_above_model_layer_counts(
                top_pressure)
            for kind, upper in (("longwave", lw_upper),
                                ("shortwave", sw_upper)):
                total = nz + upper
                checks.append({
                    "component": f"RTE+RRTMGP {kind}",
                    "model_levels": nz,
                    "above_model_layers": upper,
                    "total_layers": total,
                    "maximum": MAX_RRTMGP_LAYERS,
                })
                if total > MAX_RRTMGP_LAYERS:
                    violations.append(
                        f"RTE+RRTMGP {kind} requires model plus cap layers "
                        f"<= {MAX_RRTMGP_LAYERS}, got "
                        f"{nz}+{upper}={total}")

    if violations:
        raise PhysicsVerticalPreflightError(
            "resolved physics vertical preflight failed:\n  - "
            + "\n  - ".join(violations))
    return {
        "schema": "gpuwm-resolved-physics-vertical-preflight-v1",
        "model_levels": nz,
        "p_top_pa": p_top,
        "resolved_components": dict(sorted(resolved.items())),
        "checks": checks,
    }


def validate_single_domain_physics_profile(
        profile: str,
        *,
        config: Mapping[str, object] | object | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate one named template through registry/runtime capabilities.

    A named profile supplies the intended immutable product.  The executable
    decision remains selector-based: when ``config`` is present its selectors
    are resolved first, so an unfinished piece reports its cited capability
    blocker rather than being hidden behind a profile-name mismatch.
    """

    from gpuwm.physics_registry import physics_registry, registry_sha256

    registry = physics_registry()
    template = registry["templates"].get(profile)
    if not isinstance(template, Mapping):
        raise ValueError(
            f"physics profile {profile!r} is not a registered fixed template")
    expected = single_domain_runtime_switches(profile)
    selected_settings = expected if config is None else config
    resolved_components = validate_physics_capabilities(selected_settings)
    expected_components = dict(template.get("components", {}))
    if resolved_components != expected_components:
        raise ValueError(
            f"selected physics differs from profile {profile!r}: "
            f"components={resolved_components}, expected={expected_components}")
    if config is not None:
        drift = {
            name: {
                "selected": _selection_value(config, name),
                "expected": value,
            }
            for name, value in expected.items()
            if _selection_value(config, name) != value
        }
        if drift:
            raise ValueError(
                f"selected physics differs from profile {profile!r}: "
                f"settings={drift}")

    required_acknowledgements: dict[str, list[str]] = {}
    for route_id, route in registry["runner_routes"].items():
        expert = route.get("expert_template_ids", {})
        if not isinstance(expert, Mapping):
            continue
        if any(
                isinstance(template_ids, list) and profile in template_ids
                for template_ids in expert.values()):
            acknowledgement = route.get("expert_acknowledgement_id")
            if isinstance(acknowledgement, str):
                required_acknowledgements.setdefault(
                    acknowledgement, []).append(route_id)
    acknowledged = set(expert_acknowledgements)
    missing = sorted(set(required_acknowledgements) - acknowledged)
    if missing:
        selectors = {
            key: _selection_value(selected_settings, key)
            for component in registry["components"].values()
            for key in component.get("selector_keys", ())
        }
        tuple_text = ", ".join(
            f"{key}={selectors[key]!r}" for key in sorted(selectors))
        acknowledgement = missing[0]
        raise PhysicsCapabilityError(
            f"d01 resolved physics tuple ({tuple_text}) selects expert "
            f"profile {profile!r}; add "
            f"{_ack_instruction(acknowledgement)} to proceed")

    selectors = {
        key: _selection_value(selected_settings, key)
        for component in registry["components"].values()
        for key in component.get("selector_keys", ())
    }
    used, provenance = _acknowledgement_receipt(
        acknowledged, set(required_acknowledgements),
        acknowledgement_provenance)
    return {
        "schema": "gpuwm-front-door-physics-selection-v1",
        "profile": profile,
        "registry_sha256": registry_sha256(registry),
        "components": resolved_components,
        "selectors": selectors,
        "resolved": expected,
        "acknowledgements": used,
        "acknowledgement_provenance": provenance,
        "maturity": template.get("maturity"),
    }


#: Verification-status vocabulary for the REPORTED physics metadata.
#:
#: Owner ruling (2026-07-31): the physics-suite choice is the user's.
#: The single-domain profile whitelist is gone as a gate; what remains
#: of it is this status, computed from the same profile data that used
#: to feed the whitelist and STATED in receipts and ``--explain`` --
#: never used to refuse a suite the engine implements.
VERIFICATION_WRF_VERIFIED = "wrf-verified"
VERIFICATION_SUPPORTED = "supported-not-wrf-verified"
#: A suite selecting at least one EXPERIMENTAL option.  Distinct from
#: "supported": that word promises a WRF comparison that is outstanding,
#: and an ArWen-only scheme has none to be outstanding.
VERIFICATION_EXPERIMENTAL = "experimental-not-wrf-verified"

#: The maturity rung whose options are experimental.  Read off the
#: registry ladder rather than spelled twice.
_EXPERIMENTAL_MATURITY = "experimental-runtime"

#: The SECOND, independent trigger for the experimental warning: an
#: option the registry declares has no WRF counterpart at all.
#:
#: The conformance ladder measures distance from WRF, so an
#: ArWen-original scheme cannot climb it and sits at
#: 'implemented-unverified' permanently -- the same rung as a
#: WRF-transcribed port whose forecast comparison is merely outstanding.
#: Keying the warning on the rung alone would therefore make the two
#: indistinguishable and silence the warning for exactly the option that
#: needs it most.  The declaration is what separates them.
_NO_WRF_COUNTERPART = "wrf_counterpart"
VERIFICATION_STATUS_SCHEMA = "gpuwm-physics-verification-status-v1"

#: The registry maturity that constitutes WRF-verification evidence.
#: Everything else the engine implements is honestly "supported".
_WRF_VERIFIED_MATURITY = "wrf-matched-run"


def _experimental_reason(option) -> str | None:
    """Why this option warns, or ``None`` when it does not.

    Two independent triggers, and the reason differs because what the
    reader must not conclude differs.  A table-bound runtime IS a WRF
    scheme whose comparison is outstanding; an ArWen-original closure
    has no comparison to be outstanding, and telling a user it is
    merely "not WRF-verified" invites them to wait for a verification
    that will never arrive.
    """
    counterpart = option.get(_NO_WRF_COUNTERPART)
    if isinstance(counterpart, Mapping) and counterpart.get("exists") is False:
        return "experimental, ArWen-original with no WRF counterpart"
    if option.get("maturity") == _EXPERIMENTAL_MATURITY:
        return "experimental, not WRF-verified"
    return None


def experimental_component_labels(run_config, registry=None):
    """Labels of every EXPERIMENTAL component option this config selects.

    Returns a sorted tuple, empty when the suite is entirely
    non-experimental.  Reads the registry rather than naming any scheme:
    a second experimental option added later is surfaced by registering
    it, not by editing this function.
    """
    return tuple(sorted(
        label for label, _ in _experimental_components(run_config, registry)))


def _experimental_components(run_config, registry=None):
    """(label, reason) for every experimental option this config selects."""
    from gpuwm.physics_registry import physics_registry

    if registry is None:
        registry = physics_registry()
    # Matched PER COMPONENT off its own selectors, deliberately, rather
    # than through a whole-suite capability resolve: that resolve fails
    # as a unit, so an unrelated mismatch elsewhere in the suite would
    # silently swallow this warning -- and a warning that disappears
    # when something else is wrong is worse than no warning.
    found = []
    for component in registry.get("components", {}).values():
        if not isinstance(component, Mapping):
            continue
        for option_id, option in component.get("options", {}).items():
            if not isinstance(option, Mapping):
                continue
            reason = _experimental_reason(option)
            if reason is None:
                continue
            selectors = option.get("selectors") or {}
            if selectors and all(
                    getattr(run_config, key, None) == value
                    for key, value in selectors.items()):
                found.append((str(option.get("label") or option_id), reason))
    return found


def experimental_selection_sentence(run_configs, registry=None) -> str | None:
    """The one warn-not-block sentence for a run's experimental options.

    ``None`` when nothing experimental is selected.  Takes an ITERABLE
    of run configs rather than one, because a domain tree has several
    and the reader is owed the sentence ONCE for the run rather than
    once per nest -- and because the two runners must not drift into
    saying different things about the same closure.  Every product
    surface that prints this sentence gets it from here; the string is
    defined once.
    """

    clauses: dict[str, str] = {}
    for run_config in run_configs:
        for label, reason in _experimental_components(run_config, registry):
            clauses.setdefault(label, reason)
    if not clauses:
        return None
    # An experimental option is not "supported, not yet WRF-verified"
    # -- that phrasing promises a WRF comparison that is merely
    # outstanding.  For a scheme WRF does not have, no such comparison
    # exists or can, and saying otherwise would be the most misleading
    # sentence the product prints.  Warn-not-block: this is the
    # wording, never a refusal.
    return ("physics: "
            + "; ".join(f"{label}: {clauses[label]}"
                        for label in sorted(clauses))
            + " -- the run continues.")


def single_domain_verification_status(run_config) -> dict[str, object]:
    """WRF-verification evidence for one run config, as reported metadata.

    Never a gate.  A switch-exact match against a shipped profile
    carries that profile's registry-template maturity; a suite matching
    no profile may still match a registered template at COMPONENT level
    (the wizard's default suite does), which is named without being
    claimed as switch-level evidence.  The ``sentence`` is the one line
    product surfaces print -- detail stays in this receipt.
    """

    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    matched = identify_single_domain_profile(run_config)
    maturity = None
    if matched is not None:
        template = registry["templates"].get(matched)
        if isinstance(template, Mapping):
            maturity = template.get("maturity")
    component_match: dict[str, object] | None = None
    if matched is None:
        try:
            resolved = validate_physics_capabilities(run_config)
        except (PhysicsCapabilityError, TypeError, ValueError):
            resolved = None
        if resolved is not None:
            for template_id, template in registry["templates"].items():
                if (isinstance(template, Mapping)
                        and dict(template.get("components", {}))
                        == resolved):
                    component_match = {
                        "template": template_id,
                        "maturity": template.get("maturity"),
                        "scope": "components-only-not-switch-level",
                    }
                    break
    verified = matched is not None and maturity == _WRF_VERIFIED_MATURITY
    experimental = experimental_component_labels(run_config, registry)
    if experimental:
        # One definition, in experimental_selection_sentence, so this
        # surface and the domain-tree runner cannot drift apart.
        sentence = experimental_selection_sentence([run_config], registry)
    elif verified:
        sentence = (
            f"physics: {matched} carries WRF-verification evidence "
            f"(registry maturity {maturity!r}).")
    elif matched is not None:
        sentence = (
            f"physics: {matched} is supported, not yet WRF-verified "
            f"(registry maturity {maturity!r}); the run continues.")
    else:
        sentence = (
            "physics: this suite is supported, not yet WRF-verified; "
            "the run continues.")
    return {
        "schema": VERIFICATION_STATUS_SCHEMA,
        "status": (
            VERIFICATION_EXPERIMENTAL if experimental
            else VERIFICATION_WRF_VERIFIED if verified
            else VERIFICATION_SUPPORTED),
        "matched_profile": matched,
        "matched_profile_maturity": maturity,
        "component_matched_template": component_match,
        "experimental_components": list(experimental),
        "sentence": sentence,
    }


def single_domain_physics_selection(
        config,
        *,
        profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The single-domain front-door selection receipt, profile optional.

    Owner ruling (2026-07-31): any physics suite the engine implements
    runs on the prepared single-domain route.  A profile the caller
    NAMED still gates -- a gate asked for is not a gate to drop, and it
    is how "this config IS that shipped suite" stays assertable -- but
    an unnamed config is governed exactly the way the domain-tree route
    has always governed a one-node tree: engine-valid selectors, the
    registry's source-neutral tuple governance, and a recorded blocker
    (not a refusal) where the registry has no spelling for the tuple.

    This is THE one spelling of the single-domain selection decision,
    and its callers are the whole point: the GFS front door
    (:func:`gpuwm.gfs_direct.front_door_physics_selection`), the direct
    exporter's profileless contract
    (:func:`gpuwm.wrf_direct.export_prepared_wrf` with
    ``experiment_config_suite=True``), and the prepared-forecast
    runner's own tuple governance and proof recompute
    (:mod:`gpuwm.prepared_single_domain_forecast`).  All of them compute
    THIS receipt from the hash-bound experiment config, which is what
    keeps "the physics executed is the physics prepared"
    byte-comparable across every seam.
    """

    if profile is not None:
        return validate_single_domain_physics_profile(
            profile, config=config,
            expert_acknowledgements=expert_acknowledgements,
            acknowledgement_provenance=acknowledgement_provenance)
    # The reachability union here spans EVERY implemented route's
    # declared templates, not only the domain-tree route's: the
    # registry declares RUC among this route's own shipped products
    # (ERA5), and a shipped product must not be governed as an outside
    # tuple at its own runner.  Expert templates keep their published
    # acknowledgement from whichever route declares them.
    return multi_domain_physics_selection(
        {1: config},
        expert_acknowledgements=expert_acknowledgements,
        acknowledgement_provenance=acknowledgement_provenance,
        governance_route_modes=("experiment-per-domain", "fixed-template"))


#: Route/source pairs whose declared template list this front door
#: enforces at PREPARATION time.
#:
#: The registry has always published which templates each route offers
#: each source (``runner_routes.<route>.source_template_ids.<source>``),
#: but until now only plan validation and the GUI read it -- nothing
#: consulted it before a preparation ran.  That is how RUC came to be
#: selectable through the GFS front door, preparable in full, and unable
#: to complete its first integration step.  Declaring and enforcing in
#: two places that never met is the whole "selectable but not usable"
#: pattern; this is the missing half.
_ROUTE_FOR_SOURCE = {
    "gfs": "tools.prepared_single_domain_forecast",
}


def offered_land_surfaces(source: str) -> frozenset[str] | None:
    """Land-surface components this source's declared templates reach.

    ``None`` where no route/source declaration is enforced here, which
    means "this function has no opinion" and never "everything is
    allowed": callers keep whatever other gates they already applied.
    """

    from gpuwm.physics_registry import physics_registry

    route_id = _ROUTE_FOR_SOURCE.get(source)
    if route_id is None:
        return None
    registry = physics_registry()
    route = registry["runner_routes"].get(route_id)
    if not isinstance(route, Mapping):
        return None
    declared = []
    for route_key in ("source_template_ids", "expert_template_ids"):
        values = route.get(route_key, {}).get(source)
        if isinstance(values, list):
            declared.extend(values)
    if not declared:
        return None
    offered = set()
    for template_id in declared:
        template = registry["templates"].get(template_id)
        if isinstance(template, Mapping):
            component = template.get("components", {}).get("land_surface")
            if isinstance(component, str):
                offered.add(component)
    return frozenset(offered)


def land_surface_component_for_selector(value) -> str | None:
    """The registry's name for an ``sf_surface_physics`` value.

    Resolved from the one selector rather than from a whole-suite
    capability resolution, because the two answer different questions.
    Resolving the suite can fail for a reason that has nothing to do
    with the land surface -- the committed two-domain descriptor selects
    a radiation spelling the registry has no option for -- and a route
    gate that quietly skips those configurations is a gate with a hole
    in exactly the shape of the configurations nobody has classified.
    """

    from gpuwm.physics_registry import physics_registry

    if value is None or isinstance(value, bool):
        return None
    try:
        selector = int(value)
    except (TypeError, ValueError):
        return None
    options = physics_registry()["components"]["land_surface"]["options"]
    for option_id, option in options.items():
        if not isinstance(option, Mapping):
            continue
        selectors = option.get("selectors", {})
        if (isinstance(selectors, Mapping)
                and selectors.get("sf_surface_physics") == selector):
            return option_id
    return None


def land_surface_route_blocker(component: str, *, source: str) -> str | None:
    """Why ``source`` does not offer ``component``, or ``None``.

    Cites the registry declaration it enforces, says what was observed
    rather than what is suspected, and names the sources this refusal is
    NOT speaking for -- an unexercised route is not a broken one, and
    withdrawing it on inference would be the same error as the
    front-door regression this release fixes, pointing the other way.
    """

    offered = offered_land_surfaces(source)
    if offered is None or component in offered:
        return None
    detail = ""
    if component == "ruc-lsm":
        detail = (
            "  A GFS-initialised RUC forecast prepares cleanly and then "
            "dies on its first surface-temperature call with `mavail must "
            "be finite`, having advanced no model time, so preparing one "
            "spends the run to reach a refusal.  Completing the GFS "
            "route's RUC land/soil initialisation is tracked for v1.2.  "
            "This says nothing about RUC on the ERA5 or HRRR routes, "
            "which still offer it and were not exercised by the run that "
            "found this.")
    return (
        f"the {source.upper()} route does not offer the {component} "
        f"land-surface component: "
        f"gpuwm/physics_registry_v2.json#/runner_routes/"
        f"{_ROUTE_FOR_SOURCE[source]}/source_template_ids/{source} "
        f"declares the templates it does offer, and none of them selects "
        f"it.{detail}")


#: The domain-tree route's front-door physics receipt.  A DISTINCT
#: schema from the single-domain one because it records a different
#: decision: a tree has no single profile, it has one resolved selector
#: set per domain, and giving both shapes one schema id would make a
#: consumer guess which it was handed.
MULTI_DOMAIN_SELECTION_SCHEMA = (
    "gpuwm-front-door-physics-selection-multi-domain-v1")


def multi_domain_physics_selection(
        domain_settings: Mapping[int, Mapping[str, object] | object],
        *,
        profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        acknowledgement_provenance: Mapping[str, object] | None = None,
        governance_route_modes: tuple[str, ...] = ("experiment-per-domain",),
) -> dict[str, object]:
    """Record and govern a domain tree's resolved physics tuple.

    The domain-tree runner never had a profile whitelist and does not
    grow one here, in either of its two possible spellings -- and since
    the 2026-07-31 owner ruling this same governance also admits an
    UNNAMED single-domain configuration (a one-node tree) on the
    prepared single-domain route, via
    :func:`single_domain_physics_selection`.

    Every domain has already passed the executable selector checks.  This
    second gate asks a different, registry-owned governance question about
    the complete resolved component tuple.  The tuple is compared with the
    union of registry-declared tree templates and component overrides across
    every source, so source identity can neither admit nor refuse it.
    Registry-normal tuples are unchanged, expert-template tuples retain their
    published acknowledgement, and a tuple outside the declared union needs
    the authority-level acknowledgement named in the refusal.

    ``profile`` is honoured when the caller named one explicitly, and
    then it does gate -- a gate asked for is not a gate to drop.  It
    binds the ROOT only: the wizard's own nested emission of a shipped
    profile turns cumulus off on the inner domains, so demanding the
    profile of every domain would refuse the very configuration the
    product tells a user to write.
    """

    from gpuwm.physics_registry import physics_registry, registry_sha256

    registry = physics_registry()
    selector_keys = tuple(
        key
        for component in registry["components"].values()
        for key in component.get("selector_keys", ())
    )
    grid_ids = sorted(int(grid_id) for grid_id in domain_settings)
    if not grid_ids:
        raise ValueError("a domain tree has no domains to record")
    domains: dict[str, object] = {}
    acknowledged = set(expert_acknowledgements)
    for grid_id in grid_ids:
        settings = domain_settings[grid_id]
        try:
            components = validate_physics_capabilities(settings)
            blocker = None
        except PhysicsCapabilityError as error:
            components = None
            blocker = str(error)
        governance = {
            "state": "unresolved",
            "required_acknowledgement": None,
            "acknowledged": False,
        }
        if components is not None:
            state, required = _tree_tuple_registry_governance(
                registry, components, route_modes=governance_route_modes)
            governance = {
                "state": state,
                "required_acknowledgement": required,
                "acknowledged": (
                    required is None or required in acknowledged),
            }
            if required is not None and required not in acknowledged:
                # Warn-not-block: an implemented tuple outside the
                # registry's blessed reachability RUNS, with one line
                # saying so.  The governance record above already
                # carries acknowledged=False, so every receipt states
                # the truth; the acknowledgement remains the way to
                # silence the warning.
                tuple_text = ", ".join(
                    f"{key}={_selection_value(settings, key)!r}"
                    for key in sorted(selector_keys))
                warn(f"d{grid_id:02d} physics tuple ({tuple_text}) has "
                     f"registry reachability {state!r} -- running it "
                     f"unblessed; add {_ack_instruction(required)} to "
                     "silence this warning",
                     why="Every component in the tuple is individually "
                         "implemented and verified; the registry has "
                         "simply not blessed this combination end to "
                         "end.  The run record carries "
                         "acknowledged=false either way.")
        domains[str(grid_id)] = {
            "components": components,
            "registry_blocker": blocker,
            "governance": governance,
            "selectors": {
                key: _selection_value(settings, key)
                for key in selector_keys
            },
        }
    if profile is not None:
        # The root carries the named profile exactly as the single-domain
        # door would check it, so an explicitly bound tree is refused for
        # the same reason and in the same words.
        validate_single_domain_physics_profile(
            profile, config=domain_settings[grid_ids[0]],
            expert_acknowledgements=expert_acknowledgements,
            acknowledgement_provenance=acknowledgement_provenance)
    used, provenance = _acknowledgement_receipt(
        acknowledged, acknowledged, acknowledgement_provenance)
    return {
        "schema": MULTI_DOMAIN_SELECTION_SCHEMA,
        "profile": profile,
        "registry_sha256": registry_sha256(registry),
        "domains": domains,
        "acknowledgements": used,
        "acknowledgement_provenance": provenance,
    }


def _tree_tuple_registry_governance(
        registry: Mapping[str, object],
        selected: Mapping[str, str],
        *,
        route_modes: tuple[str, ...] = ("experiment-per-domain",),
) -> tuple[str, str | None]:
    """Resolve a complete tuple without consulting source identity.

    ``route_modes`` names which implemented routes' declared templates
    make up the reachability union.  The domain-tree question keeps its
    historical union (the per-domain routes and their component
    overrides); the single-domain selection widens it to the
    fixed-template routes too, because a template the registry declares
    as one of THIS route's shipped products (RUC on ERA5 is the
    exhibit) must not read as outside the registry's reachability at
    its own runner.
    """

    components = registry.get("components", {})
    templates = registry.get("templates", {})
    routes = registry.get("runner_routes", {})
    if (
        not isinstance(components, Mapping)
        or not isinstance(templates, Mapping)
        or not isinstance(routes, Mapping)
    ):
        raise PhysicsCapabilityError(
            "physics registry lacks tuple reachability declarations")

    def key(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(value.items()))

    normal: set[tuple[tuple[str, str], ...]] = set()
    expert: dict[tuple[tuple[str, str], ...], set[str]] = {}

    for route in routes.values():
        if (
            not isinstance(route, Mapping)
            or route.get("implemented") is not True
            or route.get("mode") not in route_modes
        ):
            continue
        override_components = [
            value for value in route.get(
                "allowed_component_overrides", ())
            if isinstance(value, str)
        ]
        option_sets: dict[str, tuple[str, ...]] = {}
        for component_id in override_components:
            component = components.get(component_id)
            options = (
                component.get("options", {})
                if isinstance(component, Mapping) else {}
            )
            option_sets[component_id] = tuple(sorted(
                option_id
                for option_id, option in options.items()
                if isinstance(option_id, str)
                and isinstance(option, Mapping)
                and option.get("implemented") is True
            ))
        allowed_component_options = route.get(
            "allowed_component_options", {})
        if isinstance(allowed_component_options, Mapping):
            for component_id, option_ids in allowed_component_options.items():
                if not (
                    isinstance(component_id, str)
                    and isinstance(option_ids, (list, tuple))
                ):
                    continue
                component = components.get(component_id)
                options = (
                    component.get("options", {})
                    if isinstance(component, Mapping) else {}
                )
                admitted = {
                    option_id for option_id in option_ids
                    if isinstance(option_id, str)
                    and isinstance(options.get(option_id), Mapping)
                    and options[option_id].get("implemented") is True
                }
                admitted.update(option_sets.get(component_id, ()))
                option_sets[component_id] = tuple(sorted(admitted))

        def variants(template_id: str):
            template = templates.get(template_id)
            base = (
                template.get("components", {})
                if isinstance(template, Mapping) else {}
            )
            if not isinstance(base, Mapping):
                return
            candidates = [dict(base)]
            for component_id, option_ids in option_sets.items():
                expanded = []
                for candidate in candidates:
                    for option_id in option_ids:
                        expanded.append({
                            **candidate, component_id: option_id})
                candidates = expanded
            yield from candidates

        source_template_ids = route.get("source_template_ids", {})
        if isinstance(source_template_ids, Mapping):
            normal_ids = {
                template_id
                for declared in source_template_ids.values()
                if isinstance(declared, list)
                for template_id in declared
                if isinstance(template_id, str)
            }
            for template_id in normal_ids:
                normal.update(key(candidate)
                              for candidate in variants(template_id))

        expert_template_ids = route.get("expert_template_ids", {})
        acknowledgement = route.get("expert_acknowledgement_id")
        if (
            isinstance(expert_template_ids, Mapping)
            and isinstance(acknowledgement, str)
        ):
            expert_ids = {
                template_id
                for declared in expert_template_ids.values()
                if isinstance(declared, list)
                for template_id in declared
                if isinstance(template_id, str)
            }
            for template_id in expert_ids:
                for candidate in variants(template_id):
                    expert.setdefault(key(candidate), set()).add(
                        acknowledgement)

    selected_key = key(selected)
    if selected_key in normal:
        return "registry-reachable", None
    if selected_key in expert:
        acknowledgements = sorted(expert[selected_key])
        if len(acknowledgements) != 1:
            raise PhysicsCapabilityError(
                "registry expert tuple publishes ambiguous acknowledgements "
                f"{acknowledgements}")
        return "registry-expert-template", acknowledgements[0]
    authority = registry.get("authority", {})
    acknowledgement = (
        authority.get(
            "unnamed_tree_outside_reachability_acknowledgement_id")
        if isinstance(authority, Mapping) else None
    )
    if not isinstance(acknowledgement, str) or not acknowledgement:
        raise PhysicsCapabilityError(
            "registry does not publish the unnamed-tree outside-"
            "reachability acknowledgement setting")
    return "outside-registry-declared-reachability", acknowledgement


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
        "readiness": "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME",
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
    """One coupled physics component requested before it is executable.

    ``missing`` has been an ordered tuple since it was written: element
    zero states the RULE -- the admitted configuration, which is also
    the thing to change -- and the elements after it explain the
    mechanism that makes the rule necessary.  ``action_count`` names
    where that boundary falls instead of leaving it to a reader (or a
    formatter) to infer, because one blocker breaks the pattern: the
    Noah-MP column budget's second element is the remedy, an
    environment variable to set, and burying a remedy behind a flag is
    the one thing this layering must never do.
    """

    component: str
    selectors: tuple[tuple[str, int], ...]
    missing: tuple[str, ...]
    #: How many leading ``missing`` elements are the action half.
    action_count: int = 1

    def _selected(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.selectors)

    def format(self) -> str:
        """The whole blocker on one line -- rule and mechanism together.

        Unchanged, and still the text the namelist importer's port
        receipt embeds: this is the full statement, and every consumer
        that wants all of it keeps getting all of it.
        """

        return f"{self.component} ({self._selected()}): " + "; ".join(
            self.missing)

    def action(self) -> str:
        """The rule and any remedy: what a reader has to change."""

        return f"{self.component} ({self._selected()}): " + "; ".join(
            self.missing[:self.action_count])

    def why(self) -> str:
        """The mechanism behind the rule; ``""`` when none was written."""

        return "; ".join(self.missing[self.action_count:])


class UnsupportedPhysicsSuiteError(ValueError):
    """A requested WRF suite contains one or more unfinished components.

    The message is layered (:mod:`gpuwm.explain`): every blocker's rule
    prints by default, and the mechanism paragraphs -- which coupling
    writes over which diagnostic, which oracle fixtures exist -- follow
    ``--explain``.  Both halves live in this one string, so anything
    reading ``str(error)`` still sees the complete receipt; only the CLI
    print boundary chooses.
    """

    def __init__(self, blockers: tuple[PhysicsPortBlocker, ...]):
        if not blockers:
            raise ValueError("UnsupportedPhysicsSuiteError needs blockers")
        self.blockers = blockers
        actions = "\n".join(f"  - {item.action()}" for item in blockers)
        reasons = "\n".join(f"  - {item.component}: {item.why()}"
                            for item in blockers if item.why())
        super().__init__(layered(
            "requested WRF physics suite is not executable in gpuwm yet; "
            "no substitutions were applied:\n" + actions,
            ("why these pairings are refused:\n" + reasons) if reasons
            else ""))


def pending_wrf_physics_components(
        *, mp_physics: int, sf_sfclay_physics: int,
        bl_pbl_physics: int, sf_surface_physics: int,
        num_soil_layers: int,
        columns: int | None = None,
        ) -> tuple[PhysicsPortBlocker, ...]:
    """Return unfinished components selected by a WRF physics request.

    The PBL/surface-layer verdict comes from the complete declarative WRF
    v4.6.1 table in :mod:`gpuwm.wrf461_compatibility`.  RUC's layer count is
    included in its blocker even though WRF also supports a six-layer RUC
    configuration; the target suite explicitly requests nine.

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
    # Namelist preview invokes this readiness layer before its documented
    # WRF-to-ArWen selector mappings (for example ISHMAEL 55 -> Morrison
    # 10).  Shin-Hong 11 is no longer such a mapping: it imports natively
    # since the Shin-Hong port, so bl_pbl_physics=11 is inside PBL_OPTIONS
    # and its four (11, sfclay) cells get their verdict from the WRF matrix
    # below exactly like 0/1/5.  The matrix governs the ported selector set
    # only; an outside raw WRF value continues to the importer's
    # mapping/refusal authority.
    #
    # mp_physics == 28 (aerosol-aware Thompson) appends no blocker either,
    # and that is a decision recorded here rather than an omission.  It has
    # a dispatch row (gpuwm/core/microphysics.py), a complete adapter
    # (gpuwm/core/microphysics_aerosol.py) over eight aerosol CUDA
    # translation units, prognostic nc/nwfa/nifa state with transport,
    # restart and nesting, a reflectivity route, and a wrfout inventory.
    # What is NOT admitted is refused by NAME somewhere a user can see it,
    # never by a silent numeric gate here:
    #   * the two aerosol-source selectors fail closed in
    #     gpuwm.config.validate_aerosol_source_options -- aer_init_opt and
    #     wif_input_opt are honoured at 0 only, because ArWen has no WIF
    #     metgrid ingest and no nbca species;
    #   * WRF's real.exe FATALs mp_physics=28 at wif_input_opt=0
    #     (dyn_em/module_initialize_real.F:2735-2736) while ArWen runs
    #     thompson_init's synthetic CCN/IN profile.  Same physics, an
    #     initialization WRF's initializer refuses to produce.  That is
    #     published as gpuwm.config.MP28_AEROSOL_SOURCE_DEVIATION and is
    #     carried in the namelist importer's printed receipt;
    #   * the registry decides REACHABILITY.  mp=28 registers no template
    #     and appears in no runner_routes source_template_ids (verified
    #     against the shipped registry), so it is selectable only as a
    #     per-domain component override -- never a default, never the
    #     scheme a user gets by accident.
    # Adding a blocker here instead would be the wrong shape twice over: it
    # would refuse the whole scheme for a limitation that is really about
    # the aerosol SOURCE, and it would hide the WRF citation that makes the
    # limitation checkable.
    if (
        bl_pbl_physics in PBL_OPTIONS
        and sf_sfclay_physics in SURFACE_LAYER_OPTIONS
    ):
        pair_verdict, pair_citation = pbl_surface_layer_verdict(
            bl_pbl_physics, sf_sfclay_physics)
        if pair_verdict is WRFVerdict.FATAL:
            blockers.append(PhysicsPortBlocker(
                component="WRF v4.6.1 PBL/surface-layer compatibility",
                selectors=(("sf_sfclay_physics", sf_sfclay_physics),
                           ("bl_pbl_physics", bl_pbl_physics)),
                missing=(
                    "WRF v4.6.1 refuses this pairing",
                    f"{pair_citation.anchor}: {pair_citation.law}",
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
    if sf_surface_physics == 4 and columns is not None:
        # GRID WIDTH, not a missing branch.  Noah-MP is fully ported and
        # bitwise, and since the slab orchestration it also FINISHES at
        # production width; a wider grid than anything measured is a
        # PERFORMANCE projection, not a correctness gap -- so it is a
        # one-line warning now, never a blocker (warn-not-block ruling).
        # The env override is kept as the way to silence the warning.
        budget = max(NOAHMP_MEASURED_COLUMN_CEILING,
                     noahmp_expert_column_budget())
        if columns > budget:
            warn(
                f"Noah-MP at {columns} columns is beyond the "
                f"{NOAHMP_MEASURED_COLUMN_CEILING}-column measured width; "
                f"the land-surface call projects to about "
                f"{noahmp_projected_call_seconds(columns):.2f} s -- "
                "slow, not wrong; continuing",
                why=f"Measured {NOAHMP_MEASURED_SLAB_CALL_SECONDS[0]}-"
                    f"{NOAHMP_MEASURED_SLAB_CALL_SECONDS[1]} s per call "
                    f"at {NOAHMP_MEASURED_COLUMN_CEILING} columns on one "
                    "RTX 5090 (2026-07-27); the projection is linear in "
                    f"columns.  Set {NOAHMP_EXPERT_COLUMN_BUDGET_ENV} at "
                    "or above your column count to silence this warning.")
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
    "KESSLER_PROFILE_ID",
    "MYNN_NOAHMP_PROFILE_ID",
    "MYNN_PROFILE_ID",
    "MYNN_RUC_PROFILE_ID",
    "NOAHMP_PROFILE_ID",
    "NOAHMP_EXPERT_COLUMN_BUDGET_ENV",
    "NOAHMP_MEASURED_COLUMN_CEILING",
    "NOAHMP_MEASURED_SLAB_CALL_SECONDS",
    "SINGLE_DOMAIN_PHYSICS_PROFILES",
    "MORRISON_PROFILE_ID",
    "MP28_REGISTRY_OPTION_ID",
    "MULTI_DOMAIN_SELECTION_SCHEMA",
    "land_surface_component_for_selector",
    "land_surface_route_blocker",
    "multi_domain_physics_selection",
    "offered_land_surfaces",
    "noahmp_expert_column_budget",
    "noahmp_projected_call_seconds",
    "NSSL2_PROFILE_ID",
    "RUC_PROFILE_ID",
    "PhysicsPortBlocker",
    "PhysicsCapabilityError",
    "PhysicsVerticalPreflightError",
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
    "IMPLICIT_RUNTIME_SWITCHES",
    "identify_single_domain_profile",
    "implicit_runtime_switches",
    "packaged_thompson_table_root",
    "pending_wrf_physics_components",
    "thompson_guard_exports",
    "thompson_table_root",
    "require_ready_wrf_physics",
    "require_rrtmg_legacy_executable",
    "require_rrtmg_legacy_ready",
    "rrtmg_variant",
    "single_domain_physics_selection",
    "single_domain_runtime_switches",
    "thompson_runtime_requirements",
    "user_thompson_table_root",
    "validate_physics_capabilities",
    "validate_resolved_physics_vertical_levels",
    "validate_single_domain_physics_profile",
]
