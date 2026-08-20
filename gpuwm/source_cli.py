"""Model-agnostic native meteorological source -> stock-WRF CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from gpuwm import __version__
from gpuwm.explain import add_explain_flag, explain_enabled
from gpuwm.source_adapters import (
    AdapterStatus,
    get_source_adapter,
    source_capability_manifest,
)
from gpuwm.source_frame import canonical_field_requirements
from gpuwm.mapped_authoring import author_input_manifest, author_mapping
from gpuwm.mapped_engine_bridge import (
    ENGINE_CAPABILITIES as _MAPPED_ENGINE_CAPABILITIES,
    ENGINES as _MAPPED_ENGINES,
    ENGINE_RUST as _MAPPED_ENGINE_RUST,
    MAPPED_ROUTE_SUBCOMMAND as _MAPPED_ROUTE_SUBCOMMAND,
    resolve_engine as _resolve_mapped_engine,
)
from gpuwm.mapped_source import (_load_json_document, _mapped_engine_choice,
                                 read_input_list)
from gpuwm.hrrr_forecast import hrrr_source_window
from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from gpuwm.physics_compat import (
    SINGLE_DOMAIN_PHYSICS_PROFILES,
)


EXIT_USAGE = 64
EXIT_CONFIG = 78
MAX_PIPELINE_WORKERS = 64


def reads_as_a_sentence(detail: str) -> bool:
    """Does this refusal already say what failed, or is it a bare path?

    ``OSError(path)`` stringifies to the path alone, and a path printed
    with no sentence around it tells a reader nothing about WHAT failed
    -- so the refusal boundary labels those with the exception class.
    A refusal that was WRITTEN as a sentence must not be labelled: "the
    manifest already on disk is a different one, here is the remedy"
    does not become clearer with ``FileExistsError:`` in front of it.

    The test is deliberately about SHAPE, not about wording.  An
    absolute path is recognised by its own opening -- a drive letter and
    a separator, or a root separator -- because a path with spaces in it
    (``C:\\Program Files\\...``) is common and a space alone would call
    it prose.
    """

    head = next((line for line in detail.splitlines() if line.strip()), "")
    head = head.strip()
    if not head:
        return False
    if head[0] in "/\\" or head[1:3] in (":\\", ":/"):
        return False
    return " " in head
_SUPPORT_MATRIX = Path(__file__).with_name("native_wrf_support_v1.json")
_HRRR_DOMAIN_VALIDATION_SCHEMA = "gpuwm-hrrr-domain-validation-v1"


def _parser(*, prog: str = "gpuwm-wrf-init", add_help: bool = True,
            include_version: bool = True) -> argparse.ArgumentParser:
    """The preprocessing stage's ONE argument surface.

    ``prog``/``add_help``/``include_version`` exist so ``gpuwm prep``
    can adopt this exact parser as an argparse ``parents=`` donor rather
    than restating it.  A second copy of this vocabulary is the drift
    the whole seam is meant to prevent: the standalone ``rw-wps`` entry
    point and the ``gpuwm prep`` subcommand must accept the same flags,
    spelled once.  ``gpuwm`` already owns ``gpuwm version``, so the
    subcommand donor drops ``--version`` rather than shipping a second
    spelling of it.
    """

    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=add_help,
        description=(
            "Prepare wrfinput/wrfbdy directly from a native meteorological "
            "source without running WPS or real.exe."
        ),
    )
    if include_version:
        parser.add_argument(
            "--version",
            action="version",
            version=f"RW-WPS {__version__}",
        )
    # The same --explain the gpuwm CLI registers on every subcommand.
    # rw-wps is a separate entry point with its own parser, so it needs
    # the flag declared here -- one convention, two front doors.
    add_explain_flag(parser)
    inventory = parser.add_argument_group("inventory")
    inventory.add_argument(
        "--list-sources",
        action="store_true",
        help="print the provenance-bound source capability manifest as JSON",
    )
    inventory.add_argument(
        "--show-source",
        metavar="MODEL",
        help="print one source declaration as JSON",
    )
    inventory.add_argument(
        "--show-support-matrix",
        action="store_true",
        help="print the versioned native WRF compatibility matrix as JSON",
    )
    inventory.add_argument(
        "--show-physics-registry",
        action="store_true",
        help="print the canonical GPUWM-owned physics registry v2 as JSON",
    )
    inventory.add_argument(
        "--validate-physics-plan",
        type=Path,
        metavar="PATH",
        help="validate and resolve a gpuwm-physics-plan-v2 JSON document",
    )
    inventory.add_argument(
        "--validate-hrrr-domain",
        type=Path,
        metavar="PATH",
        help=(
            "validate a strict HRRR target domain and its complete native "
            "interpolation window"
        ),
    )
    inventory.add_argument(
        "--canonical-physics-plan-output",
        type=Path,
        metavar="PATH",
        help=(
            "create an exact canonical UTF-8 copy of the plan validated by "
            "--validate-physics-plan; refuses an existing output"
        ),
    )
    inventory.add_argument(
        "--namelist-support-report",
        action="store_true",
        help=(
            "classify --wps-namelist/--namelist-input and print the exact "
            "stock-WRF versus gpuwm support report as JSON"
        ),
    )
    inventory.add_argument(
        "--source-top-pressure-pa",
        type=float,
        help=(
            "smallest pressure represented by the selected source; used by "
            "--namelist-support-report to reject vertical extrapolation"
        ),
    )
    parser.add_argument("--source", metavar="MODEL", help="native source adapter id")
    mapped = parser.add_argument_group("declarative mapped-source adapter")
    mapped.add_argument(
        "--source-format",
        choices=("grib1", "grib2", "netcdf"),
        help="input format; must agree with the sealed rw-wps.mapping.v1 document",
    )
    mapped.add_argument(
        "--mapping",
        type=Path,
        help="strict rw-wps.mapping.v1 field/coordinate/target contract",
    )
    mapped.add_argument(
        "--descriptor",
        type=Path,
        help=(
            "explicit rw-wps.descriptor.v1 science contract; requires "
            "--author-mapping and, for GRIB, --vtable"
        ),
    )
    mapped.add_argument(
        "--author-mapping",
        type=Path,
        help=(
            "create-only path for a mapping compiled from --descriptor; "
            "the adjacent *.authoring.json receipt binds descriptor/Vtable bytes"
        ),
    )
    mapped.add_argument(
        "--author-input-manifest",
        type=Path,
        help=(
            "create an exact mapped or 20CRv3 input manifest; conflicts with "
            "an existing --source-manifest/--source-manifest-sha256 pair"
        ),
    )
    mapped.add_argument(
        "--author-only",
        action="store_true",
        help=(
            "author the requested create-only mapped contract or 20CRv3 "
            "member manifest and exit; requires --author-input-manifest and "
            "does not need run geometry"
        ),
    )
    mapped.add_argument(
        "--composition",
        type=Path,
        help="strict gpuwm-mapped-composition-v2 product join contract",
    )
    mapped.add_argument(
        "--input",
        dest="mapped_inputs",
        action="append",
        type=Path,
        help="mapped source file; repeat in deterministic time/file order",
    )
    mapped.add_argument(
        "--input-list",
        dest="input_list",
        type=Path,
        help="file naming the mapped source files, one path per line, in "
             "the same deterministic time/file order the repeated --input "
             "flag spells; the spelling that keeps a field-per-file "
             "source's hundreds of inputs inside the Windows 32 KB "
             "command-line limit",
    )
    mapped.add_argument(
        "--supplement",
        action="append",
        metavar="ROLE=PATH",
        help="composition supplement binding; repeat roles for multiple files",
    )
    mapped.add_argument(
        "--provenance",
        action="append",
        metavar="ROLE=PATH",
        help="composition provenance binding",
    )
    mapped.add_argument(
        "--contributing-mapping",
        action="append",
        metavar="ROLE=PATH",
        help=(
            "cross-source composition: a contributing source's own mapping "
            "document under the mapping_role its field_sources binding "
            "declares; bytes must hash to the composition's pinned SHA-256"
        ),
    )
    mapped.add_argument(
        "--grib2-inventory",
        type=Path,
        help=(
            "override the GRIB2 inventory tool; omitted, it resolves "
            "through the shared bridge ladder (GPUWM_GRIB2_INVENTORY, a "
            "checkout build, the wheel's bundled copy, then the staged "
            "~/.gpuwm/bridges)"
        ),
    )
    mapped.add_argument(
        "--grib2-dump",
        type=Path,
        help=(
            "override the GRIB2 dump tool; omitted, it resolves through "
            "the shared bridge ladder exactly as --grib2-inventory does"
        ),
    )
    mapped.add_argument(
        "--mapped-engine",
        choices=_MAPPED_ENGINES,
        help=(
            "which engine decodes mapped source bytes; omitted, the "
            "default engine runs. `python` is a documented WORKAROUND "
            "-- the slower Python decode path, kept reachable so a "
            "decode the Rust engine gets wrong has a way around it "
            "while the defect is fixed -- not a supported mode to "
            "prefer"
        ),
    )
    mapped.add_argument(
        "--hierarchy-workers",
        type=int,
        help="bounded mapped d02..dNN initialization workers (1..32)",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--source-sha256s",
        "--source-manifest",
        dest="source_sha256s",
        type=Path,
        help="SHA-256 file manifest covering every downloaded source file",
    )
    parser.add_argument(
        "--source-sha256s-sha256",
        "--source-manifest-sha256",
        dest="source_sha256s_sha256",
        help="expected SHA-256 of --source-sha256s",
    )
    parser.add_argument("--static-cache", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument(
        "--root-preparation",
        type=Path,
        help=(
            "sealed output of the native HRRR root-preparation command; "
            "enables parallel d01..dNN hierarchy export for max_dom 1..21; "
            "the two namelists remain the topology authority"
        ),
    )
    parser.add_argument(
        "--sealed-prepared-cache", action="store_true",
        help="opt in to a prefix-sealed operational HRRR root preparation",
    )
    parser.add_argument(
        "--extend-root-preparation", type=Path,
        help="sealed HRRR predecessor to extend by exactly one forcing hour",
    )
    parser.add_argument(
        "--geog-root",
        type=Path,
        help=(
            "WPS_GEOG root used to build a domain-specific native static "
            "cache; requires --domain-spec and replaces --static-cache/"
            "--static-receipt"
        ),
    )
    parser.add_argument(
        "--domain-spec",
        type=Path,
        help=(
            "strict gpuwm-hrrr-target-domain-v1 Lambert root-domain JSON; "
            "nested layouts come from --wps-namelist/--namelist-input"
        ),
    )
    parser.add_argument("--namelist-input", type=Path)
    parser.add_argument(
        "--physics-profile",
        help=(
            "optional assertion that the experiment IS this shipped "
            "single-domain suite, refused on any switch drift; omitted, "
            "the config's own physics is prepared as written and its "
            "WRF-verification status is reported (the HRRR route still "
            "requires a shipped profile: its cold-start evidence "
            "contract is profile-keyed)"
        ),
    )
    parser.add_argument(
        "--ack",
        action="append",
        default=[],
        help="registry-owned expert physics acknowledgement id; repeatable",
    )
    parser.add_argument(
        "--stock-wrf-namelist-input",
        type=Path,
        help=(
            "unchanged-stock-WRF namelist matching the native hierarchy "
            "except for the certified LW and moist-theta representation "
            "selections"
        ),
    )
    parser.add_argument(
        "--valid-time",
        help="initial UTC time in WRF form YYYY-MM-DD_HH:MM:SS.  On "
             "--source hrrr this is the CYCLE; model time zero is cycle + "
             "--forecast-start-hour and is derived for every stage",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-seconds", type=int)
    parser.add_argument(
        "--history-interval-seconds",
        type=float,
        help=(
            "positive output cadence used by HRRR preparation and the "
            "prepared-cache forecast identity"
        ),
    )
    parser.add_argument(
        "--forecast-start-hour", type=int,
        help="absolute cycle-relative HRRR lead used for model time zero",
    )
    parser.add_argument(
        "--forecast-end-hour", type=int,
        help="inclusive absolute HRRR source lead",
    )
    parser.add_argument("--pipeline-workers", type=int)
    parser.add_argument("--prepare-workers", type=int)
    parser.add_argument(
        "--child-workers",
        type=int,
        help=("bounded CPU worker budget for parallel d02..dNN initialization (1..32)"),
    )
    preprocessing = parser.add_argument_group(
        "native source-grid/WRF-real preprocessing"
    )
    preprocessing.add_argument(
        "--preprocess-backend",
        choices=("cuda", "cpu", "auto"),
        help="select CUDA or deterministic parallel CPU preprocessing",
    )
    preprocessing.add_argument("--preprocess-workers", type=int)
    preprocessing.add_argument("--cpu-preprocess-bridge", type=Path)
    era5 = parser.add_argument_group("ERA5 combined-GRIB1 adapter")
    era5.add_argument("--grib", type=Path, help="combined ERA5 GRIB1 series")
    era5.add_argument("--vtable", type=Path, help="ERA5 GRIB1 Vtable")
    era5.add_argument(
        "--bridge",
        type=Path,
        help="prebuilt gpuwm all-Rust source-specific GRIB bridge "
             "executable; omitted on the era5/gfs routes it resolves "
             "through the shared bridge ladder (environment override, a "
             "checkout build, staged bridges under ~/.gpuwm/bridges) "
             "exactly as gpuwm go does",
    )
    era5.add_argument(
        "--wps-namelist",
        type=Path,
        help="standard WPS geometry/static-selection namelist",
    )
    era5.add_argument("--static-input", type=Path)
    era5.add_argument("--source-orography", type=Path)
    era5.add_argument("--source-orography-variable")
    era5.add_argument(
        "--domain-source-orography",
        action="append",
        metavar="DNN=PATH",
        help=(
            "ERA5 hierarchy source-orography binding; repeat once for every "
            "domain (d01..dNN). All bindings use "
            "--source-orography-variable"
        ),
    )
    era5.add_argument("--experiment-config", type=Path)
    gfs = parser.add_argument_group("GFS pgrb2.0p25 adapter")
    gfs.add_argument(
        "--gfs-series",
        type=Path,
        help="tab-separated HOUR and GFS GRIB2 path inventory",
    )
    gfs.add_argument("--cycle", help="GFS cycle in YYYY-MM-DD_HH:MM:SS form")
    # store_true with a falsy default, not store_false/default=True:
    # `_active_action_arguments` reads every non-None, non-False namespace
    # entry as a supplied argument, so a flag whose DEFAULT is True makes
    # `--validate-hrrr-domain` and its siblings believe the caller combined
    # them with something.  Same shape as --author-only for the same reason.
    gfs.add_argument(
        "--no-stock-wrf-export",
        action="store_true",
        help="prepare the forecast only, and do not attempt the bonus "
             "unchanged-WRF wrfinput/wrfbdy export of a domain tree",
    )
    gfs.add_argument(
        "--statics-corridor",
        nargs="?",
        const="all",
        default=None,
        metavar="GRID_IDS",
        help="also seal child-resolution statics over each child's whole "
             "parent extent (the moving-nest corridor); bare flag covers "
             "every child domain, or pass comma-separated child grid ids "
             "(e.g. 2,3).  Required before the prepared tree runner will "
             "honor a [relocation] follow source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate route-specific arguments and print the exact internal command",
    )
    return parser


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _write_canonical_json(value: object) -> None:
    """Write canonical JSON with an exact LF on Windows and POSIX."""

    from gpuwm.physics_registry import canonical_json

    payload = canonical_json(value).encode("utf-8") + b"\n"
    binary = getattr(sys.stdout, "buffer", None)
    if binary is not None:
        binary.write(payload)
        binary.flush()
    else:
        # pytest capture and embedders may expose only a text stream.  The
        # explicit string remains byte-equivalent outside newline-translating
        # console wrappers; production uses the binary branch above.
        sys.stdout.write(payload.decode("utf-8"))
        sys.stdout.flush()


def _active_action_arguments(
    args: argparse.Namespace,
    *,
    allowed: frozenset[str],
) -> list[str]:
    """Return non-empty CLI destinations outside one inventory action."""

    active = []
    for name, value in vars(args).items():
        if name in allowed or value is None or value is False:
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        active.append("--" + name.replace("_", "-"))
    return sorted(active)


def _hrrr_domain_validation(path: Path) -> dict[str, object]:
    """Return the stable HRRR coverage receipt consumed by launch preflight."""

    from gpuwm.ingest.hrrr_target import (
        load_hrrr_target_domain,
        required_hrrr_source_window,
    )

    domain_sha256 = None
    try:
        target = load_hrrr_target_domain(path)
        domain_sha256 = target.identity_sha256()
        window = required_hrrr_source_window(target)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": _HRRR_DOMAIN_VALIDATION_SCHEMA,
            "status": "REFUSED",
            "domain_sha256": domain_sha256,
            "window": None,
            "error": str(exc),
        }
    return {
        "schema": _HRRR_DOMAIN_VALIDATION_SCHEMA,
        "status": "PASS",
        "domain_sha256": domain_sha256,
        "window": window.to_dict(),
        "error": None,
    }


def _create_canonical_json(path: Path, value: object) -> None:
    """Create one exact no-newline canonical JSON file without replacing data."""

    from gpuwm.physics_registry import canonical_json

    payload = canonical_json(value).encode("utf-8")
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_decoder(
    requested: Path | None,
    environment_name: str | None,
    label: str,
) -> Path | None:
    """Resolve a decoder and reject installed-runtime path substitution."""

    installed_raw = (
        os.environ.get(environment_name) if environment_name is not None else None
    )
    installed = Path(installed_raw).resolve() if installed_raw else None
    # One validator for the whole schema, at the first read.  This used
    # to accept any document carrying schema+status and then bind
    # decoders out of it, so a manifest that was already invalid in
    # three other fields got as far as launching a bridge.
    from gpuwm.runtime_manifest import manifest_from_environment

    bound = manifest_from_environment()
    if bound is not None:
        manifest, payload = bound
        if installed is None:
            raise ValueError(
                f"installed runtime did not export required {environment_name}"
            )
        bridge_name = {
            "GPUWM_GRIB1_BRIDGE": "grib1_bridge",
            "GPUWM_GRIB2_INVENTORY": "grib2_inventory",
            "GPUWM_GRIB2_DUMP": "grib2_dump",
            "GPUWM_GFS_GRIB2_BRIDGE": "gfs_grib2_bridge",
        }.get(str(environment_name))
        if bridge_name is None:
            raise ValueError(f"unsupported installed decoder role {environment_name}")
        runtime_payload = payload.get("payload")
        if not isinstance(runtime_payload, dict):
            raise ValueError("native distribution manifest lacks its payload inventory")
        candidates = (
            f"libexec/bridges/{bridge_name}",
            f"libexec/bridges/{bridge_name}.exe",
        )
        available = [name for name in candidates if name in runtime_payload]
        if len(available) != 1:
            raise ValueError(
                "native distribution manifest must contain exactly one "
                f"platform decoder for {bridge_name}: {available}"
            )
        relative = available[0]
        expected_path = (manifest.parent / relative).resolve()
        if installed != expected_path:
            raise ValueError(
                f"{environment_name} does not resolve under the installed runtime"
            )
        record = runtime_payload.get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"native distribution manifest lacks {relative}")
        if (
            not installed.is_file()
            or record.get("bytes") != installed.stat().st_size
            or record.get("sha256") != _sha256(installed)
            or record.get("executable") is not True
            or not os.access(installed, os.X_OK)
        ):
            raise ValueError(f"installed decoder bytes differ from manifest: {relative}")
        if requested is not None and requested.resolve() != installed:
            raise ValueError(
                f"{label} differs from the decoder bound by the installed runtime"
            )
        return installed
    return requested if requested is not None else installed


_ROLE_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _role_bindings(
    values: list[str] | tuple[str, ...],
    *,
    multiple: bool,
) -> dict[str, Path | tuple[Path, ...]]:
    """Parse repeatable ROLE=PATH bindings without silent replacement."""

    grouped: dict[str, list[Path]] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not _ROLE_PATTERN.fullmatch(role) or not raw_path:
            raise ValueError(
                "role bindings must use non-empty ROLE=PATH with a portable "
                "role name"
            )
        if not multiple and role in grouped:
            raise ValueError(f"duplicate binding for role {role!r}")
        grouped.setdefault(role, []).append(Path(raw_path))
    return {
        role: tuple(paths) if multiple else paths[0]
        for role, paths in grouped.items()
    }


def _role_binding_errors(
    values: list[str] | None,
    flag: str,
    *,
    unique: bool,
) -> list[str]:
    errors = []
    seen = set()
    for value in values or ():
        role, separator, path = value.partition("=")
        if not separator or not _ROLE_PATTERN.fullmatch(role) or not path:
            errors.append(f"{flag} must use non-empty portable ROLE=PATH bindings")
            continue
        if unique and role in seen:
            errors.append(f"{flag} repeats singleton role {role!r}")
        seen.add(role)
    return errors


def _required_hrrr_args(args: argparse.Namespace) -> list[str]:
    if args.root_preparation is not None:
        required = {
            "--root-preparation": args.root_preparation,
            "--domain-spec": args.domain_spec,
            "--wps-namelist": args.wps_namelist,
            "--namelist-input": args.namelist_input,
            "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
            "--geog-root": args.geog_root,
            "--source-sha256s": args.source_sha256s,
            "--source-sha256s-sha256": args.source_sha256s_sha256,
            "--valid-time": args.valid_time,
            "--output-root": args.output_root,
        }
        errors = [flag for flag, value in required.items() if value is None]
        unused = {
            "--source-root": args.source_root,
            "--static-cache": args.static_cache,
            "--static-receipt": args.static_receipt,
            "--run-seconds": args.run_seconds,
            "--history-interval-seconds": args.history_interval_seconds,
            # --forecast-start-hour is NOT unused here.  The hierarchy's
            # model time zero is cycle + K, and --valid-time on this door
            # is the cycle, so refusing the lead left the nested route
            # reachable only at lead 0 -- and reachable WRONGLY at any
            # other, because the cycle would have been forwarded as the
            # model start.  --forecast-end-hour stays unused: the
            # hierarchy reads its forcing horizon off the sealed root.
            "--forecast-end-hour": args.forecast_end_hour,
            "--pipeline-workers": args.pipeline_workers,
            "--prepare-workers": args.prepare_workers,
            "--grib": args.grib,
            "--vtable": args.vtable,
            "--bridge": args.bridge,
            "--static-input": args.static_input,
            "--source-orography": args.source_orography,
            "--source-orography-variable": args.source_orography_variable,
            "--domain-source-orography": args.domain_source_orography,
            "--experiment-config": args.experiment_config,
            "--gfs-series": args.gfs_series,
            "--cycle": args.cycle,
            "--preprocess-backend": args.preprocess_backend,
            "--preprocess-workers": args.preprocess_workers,
            "--source-format": args.source_format,
            "--physics-profile": args.physics_profile,
            "--ack": args.ack or None,
            "--mapping": args.mapping,
            "--descriptor": args.descriptor,
            "--author-mapping": args.author_mapping,
            "--author-input-manifest": args.author_input_manifest,
            "--author-only": args.author_only or None,
            "--composition": args.composition,
            "--input": args.mapped_inputs,
            "--input-list": args.input_list,
            "--supplement": args.supplement,
            "--contributing-mapping": args.contributing_mapping,
            "--provenance": args.provenance,
            "--grib2-inventory": args.grib2_inventory,
            "--grib2-dump": args.grib2_dump,
            "--hierarchy-workers": args.hierarchy_workers,
            "--mapped-engine": args.mapped_engine,
            "--sealed-prepared-cache": args.sealed_prepared_cache or None,
            "--extend-root-preparation": args.extend_root_preparation,
        }
        errors.extend(
            f"{flag} is not used by HRRR hierarchy export"
            for flag, value in unused.items()
            if value is not None
        )
        if args.child_workers is not None and args.child_workers not in range(1, 33):
            errors.append("--child-workers must be between 1 and 32")
        if (args.forecast_start_hour is not None
                and args.forecast_start_hour < 0):
            errors.append(
                "--forecast-start-hour must be a nonnegative forecast lead")
        if args.valid_time is not None:
            try:
                parsed = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
            except ValueError:
                errors.append("--valid-time must use YYYY-MM-DD_HH:MM:SS")
            else:
                if parsed.minute != 0 or parsed.second != 0:
                    errors.append("--valid-time must be an exact hourly HRRR cycle")
        return errors

    required = {
        "--source-root": args.source_root,
        "--source-sha256s": args.source_sha256s,
        "--source-sha256s-sha256": args.source_sha256s_sha256,
        "--namelist-input": args.namelist_input,
        "--valid-time": args.valid_time,
        "--output-root": args.output_root,
    }
    errors = [flag for flag, value in required.items() if value is None]
    if args.extend_root_preparation is not None \
            and not args.sealed_prepared_cache:
        errors.append(
            "--extend-root-preparation requires --sealed-prepared-cache")
    if args.sealed_prepared_cache \
            and args.forecast_start_hour not in (None, 0):
        errors.append("--sealed-prepared-cache requires --forecast-start-hour 0")
    if args.geog_root is not None:
        if args.domain_spec is None:
            errors.append("--domain-spec (required with --geog-root)")
        if args.static_cache is not None or args.static_receipt is not None:
            errors.append(
                "--geog-root cannot be mixed with --static-cache/--static-receipt"
            )
    else:
        if args.static_cache is None:
            errors.append("--static-cache (or use --geog-root)")
        if args.static_receipt is None:
            errors.append("--static-receipt (or use --geog-root)")
    # --wps-namelist is DELIBERATELY not in this list.  On the
    # single-domain HRRR route it is optional and it means one thing:
    # bind YOUR namelist into the portable authorities instead of the
    # one the route renders from the domain it was already given.  The
    # portable authorities themselves -- proof.json, the role-keyed
    # source manifest, experiment.toml and namelist.wps -- are published
    # on every run, because a prepared tree `gpuwm sim` cannot run is a
    # tree with no front door.
    era5_only = {
        "--grib": args.grib,
        "--vtable": args.vtable,
        "--bridge": args.bridge,
        "--static-input": args.static_input,
        "--experiment-config": args.experiment_config,
        "--gfs-series": args.gfs_series,
        "--cycle": args.cycle,
        "--source-orography-variable": args.source_orography_variable,
        "--domain-source-orography": args.domain_source_orography,
        "--root-preparation": args.root_preparation,
        "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
        "--child-workers": args.child_workers,
        "--source-format": args.source_format,
        "--mapping": args.mapping,
        "--descriptor": args.descriptor,
        "--author-mapping": args.author_mapping,
        "--author-input-manifest": args.author_input_manifest,
        "--author-only": args.author_only or None,
        "--composition": args.composition,
        "--input": args.mapped_inputs,
        "--input-list": args.input_list,
        "--supplement": args.supplement,
        "--contributing-mapping": args.contributing_mapping,
        "--provenance": args.provenance,
        "--grib2-inventory": args.grib2_inventory,
        "--grib2-dump": args.grib2_dump,
        "--hierarchy-workers": args.hierarchy_workers,
        "--mapped-engine": args.mapped_engine,
        "--no-stock-wrf-export": args.no_stock_wrf_export or None,
        "--statics-corridor": args.statics_corridor,
    }
    errors.extend(
        f"{flag} is not used by --source hrrr"
        for flag, value in era5_only.items()
        if value is not None
    )
    if args.valid_time is not None:
        try:
            parsed = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            errors.append("--valid-time must use YYYY-MM-DD_HH:MM:SS")
        else:
            if parsed.minute != 0 or parsed.second != 0:
                errors.append("--valid-time must be an exact hourly HRRR cycle")
    if args.valid_time is not None:
        try:
            cycle = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
            hrrr_source_window(
                cycle=cycle,
                start_hour=(
                    0 if args.forecast_start_hour is None
                    else args.forecast_start_hour
                ),
                run_seconds=(
                    43_200 if args.run_seconds is None else args.run_seconds),
                end_hour=args.forecast_end_hour,
            )
        except (TypeError, ValueError) as error:
            errors.append(f"invalid HRRR source forecast window: {error}")
    if (args.pipeline_workers is not None
            and args.pipeline_workers not in range(1, MAX_PIPELINE_WORKERS + 1)):
        errors.append(
            f"--pipeline-workers must be between 1 and {MAX_PIPELINE_WORKERS}")
    if args.prepare_workers is not None and args.prepare_workers not in range(1, 33):
        errors.append("--prepare-workers must be between 1 and 32")
    if args.history_interval_seconds is not None and (
        not math.isfinite(args.history_interval_seconds)
        or args.history_interval_seconds <= 0.0
    ):
        errors.append("--history-interval-seconds must be positive and finite")
    if (args.physics_profile is not None
            and args.physics_profile not in SINGLE_DOMAIN_PHYSICS_PROFILES):
        # HRRR-route-owned refusal, not a suite whitelist: the HRRR
        # cold-start evidence contract (tools/prepare_hrrr_wrf.py) is
        # keyed by shipped profile id and has no entry to consult for
        # any other name.
        errors.append(
            f"--physics-profile {args.physics_profile!r} has no HRRR "
            f"cold-start evidence contract; this route offers "
            f"{list(SINGLE_DOMAIN_PHYSICS_PROFILES)!r}")
    elif args.root_preparation is None:
        from gpuwm.physics_compat import (
            validate_single_domain_physics_profile,
        )
        try:
            validate_single_domain_physics_profile(
                ROUTE_DEFAULT_PHYSICS_PROFILE
                if args.physics_profile is None else args.physics_profile,
                expert_acknowledgements=tuple(
                    args.ack))
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _config_declares_geog_root(experiment_config) -> bool:
    """Whether the experiment config's ``[case_data]`` names a geog_root.

    A courtesy peek for argument validation only: the adapter re-reads
    the config through its own authority-bound loader and owns the real
    refusal, so any read or parse problem here answers ``False`` and
    leaves the full diagnostic to the front door.
    """
    if experiment_config is None:
        return False
    try:
        import io
        import tomllib

        from gpuwm.config_authority import read_config_authority

        raw = tomllib.load(
            io.BytesIO(read_config_authority(experiment_config).payload))
    except Exception:
        return False
    table = raw.get("case_data")
    return isinstance(table, dict) and bool(table.get("geog_root"))


def _experiment_acknowledgements(
        experiment_config) -> tuple[tuple[str, ...], str | None]:
    """The config's ``[experiment].acknowledgements``, or why not.

    Returns the declared ids and ``None``, or ``()`` and one sentence
    naming a TOML decode fault.

    Only a DECODE fault is named here.  The direct front door owns the
    full experiment diagnostic and this door must not grow a second,
    poorer copy of it, so a config that decodes but is not a valid
    experiment still answers ``((), None)`` and is left to the door that
    owns it.  What this must not do either is stay silent: a config that
    does not decode has no ``[experiment].acknowledgements`` to read at
    all, and swallowing that dropped the whole TOML delivery channel
    without a word.  The refusal that followed then told the caller to
    declare an acknowledgement their own config already carried -- and a
    duplicate ``acknowledgements =`` key is exactly how a hand-edited
    wizard config gets there, because the wizard emits one of its own for
    a longwave-OFF suite.  The decoder's message carries the line and
    column, which is what points at the duplicate key.
    """
    if experiment_config is None:
        return (), None
    path = Path(experiment_config)
    if not path.is_file():
        return (), None
    try:
        from gpuwm.experiment import load_experiment

        return tuple(load_experiment(path).acknowledgements), None
    except (OSError, ValueError):
        pass
    # It did not load.  Decide WHY in the one vocabulary this door is
    # entitled to have an opinion about, through the same authority-bound
    # reader every other config read on this route uses.
    import io
    import tomllib

    from gpuwm.config_authority import read_config_authority

    try:
        tomllib.load(io.BytesIO(read_config_authority(path).payload))
    except tomllib.TOMLDecodeError as error:
        # Caught before ValueError below, which it subclasses.
        return (), (
            f"--experiment-config {path} is not decodable TOML ({error}), "
            f"so its [experiment].acknowledgements cannot be read")
    except (OSError, RuntimeError, TypeError, ValueError):
        # Every richer fault stays the direct front door's to report.
        return (), None
    return (), None


def _required_era5_args(args: argparse.Namespace) -> list[str]:
    # ``--bridge`` is deliberately NOT required: omitted, it resolves
    # through the staged-bridge ladder after this validation (see the
    # dispatch), exactly as FIRST-LIGHT documents.  Demanding it here
    # was the staged-tool papercut one format earlier (UX finding N12).
    required = {
        "--grib": args.grib,
        "--vtable": args.vtable,
        "--wps-namelist": args.wps_namelist,
        "--experiment-config": args.experiment_config,
        "--source-sha256s": args.source_sha256s,
        "--source-sha256s-sha256": args.source_sha256s_sha256,
        "--output-root": args.output_root,
    }
    errors = [flag for flag, value in required.items() if value is None]
    if (args.static_input is None) != (args.static_receipt is None):
        errors.append(
            "--static-input and --static-receipt must be supplied together")
    if (args.static_input is None and args.geog_root is None
            and not _config_declares_geog_root(args.experiment_config)):
        # The one-file config `gpuwm domain --source era5` writes declares
        # geog_root in [case_data]; demanding the flag anyway made the
        # wizard's own emission unrunnable through this front door (#204).
        errors.append(
            "--static-input/--static-receipt or --geog-root is required "
            "(a geog_root declared in the experiment config's [case_data] "
            "table also satisfies this, and this config declares none)")
    incompatible = {
        "--source-root": args.source_root,
        "--physics-profile": args.physics_profile,
        "--ack": args.ack or None,
        "--forecast-start-hour": args.forecast_start_hour,
        "--forecast-end-hour": args.forecast_end_hour,
        "--static-cache": args.static_cache,
        "--domain-spec": args.domain_spec,
        "--namelist-input": args.namelist_input,
        "--valid-time": args.valid_time,
        "--prepare-workers": args.prepare_workers,
        "--gfs-series": args.gfs_series,
        "--cycle": args.cycle,
        "--run-seconds": args.run_seconds,
        "--history-interval-seconds": args.history_interval_seconds,
        "--pipeline-workers": args.pipeline_workers,
        "--root-preparation": args.root_preparation,
        "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
        "--child-workers": args.child_workers,
        "--source-format": args.source_format,
        "--mapping": args.mapping,
        "--descriptor": args.descriptor,
        "--author-mapping": args.author_mapping,
        "--author-input-manifest": args.author_input_manifest,
        "--author-only": args.author_only or None,
        "--composition": args.composition,
        "--input": args.mapped_inputs,
        "--input-list": args.input_list,
        "--supplement": args.supplement,
        "--contributing-mapping": args.contributing_mapping,
        "--provenance": args.provenance,
        "--grib2-inventory": args.grib2_inventory,
        "--grib2-dump": args.grib2_dump,
        "--no-stock-wrf-export": args.no_stock_wrf_export or None,
        "--statics-corridor": args.statics_corridor,
    }
    errors.extend(
        f"{flag} is not used by --source era5"
        for flag, value in incompatible.items()
        if value is not None
    )
    if args.domain_source_orography:
        if args.geog_root is None:
            errors.append("--geog-root is required for an ERA5 hierarchy")
        if args.source_orography is None:
            errors.append(
                "--source-orography is required with explicit per-domain "
                "source-orography bindings")
    if args.hierarchy_workers is not None and args.hierarchy_workers not in range(
        1, 33
    ):
        errors.append("--hierarchy-workers must be between 1 and 32")
    errors.extend(
        _role_binding_errors(
            args.domain_source_orography,
            "--domain-source-orography",
            unique=True,
        )
    )
    return errors


def _required_gfs_args(args: argparse.Namespace) -> list[str]:
    # ``--bridge`` is deliberately NOT required -- see _required_era5_args.
    #
    # Neither is the ``--source-manifest`` pair: omitted TOGETHER, the
    # dispatch authors and digest-binds the front-door input manifest
    # itself from the fetched directory the series lives in (UX finding
    # N11 -- demanding it here forced a second `gpuwm fetch
    # --author-front-door-manifest` call between fetch and prep).  Half
    # a pair is still refused: a manifest without its digest binds
    # nothing.
    required = {
        "--gfs-series": args.gfs_series,
        "--cycle": args.cycle,
        "--wps-namelist": args.wps_namelist,
        "--experiment-config": args.experiment_config,
        "--output-root": args.output_root,
    }
    errors = [flag for flag, value in required.items() if value is None]
    if (args.source_sha256s is None) != (args.source_sha256s_sha256 is None):
        errors.append(
            "--source-manifest and --source-manifest-sha256 are an "
            "atomic pair")
    if (args.static_input is None) != (args.static_receipt is None):
        errors.append(
            "--static-input and --static-receipt must be supplied together")
    if args.static_input is None and args.geog_root is None:
        errors.append(
            "--static-input/--static-receipt or --geog-root is required")
    incompatible = {
        "--source-root": args.source_root,
        "--forecast-start-hour": args.forecast_start_hour,
        "--forecast-end-hour": args.forecast_end_hour,
        "--static-cache": args.static_cache,
        "--domain-spec": args.domain_spec,
        "--namelist-input": args.namelist_input,
        "--valid-time": args.valid_time,
        "--prepare-workers": args.prepare_workers,
        "--grib": args.grib,
        "--vtable": args.vtable,
        "--source-orography": args.source_orography,
        "--source-orography-variable": args.source_orography_variable,
        "--domain-source-orography": args.domain_source_orography,
        "--run-seconds": args.run_seconds,
        "--history-interval-seconds": args.history_interval_seconds,
        "--pipeline-workers": args.pipeline_workers,
        "--root-preparation": args.root_preparation,
        "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
        "--child-workers": args.child_workers,
        "--source-format": args.source_format,
        "--mapping": args.mapping,
        "--descriptor": args.descriptor,
        "--author-mapping": args.author_mapping,
        "--author-input-manifest": args.author_input_manifest,
        "--author-only": args.author_only or None,
        "--composition": args.composition,
        "--input": args.mapped_inputs,
        "--input-list": args.input_list,
        "--supplement": args.supplement,
        "--contributing-mapping": args.contributing_mapping,
        "--provenance": args.provenance,
        "--grib2-inventory": args.grib2_inventory,
        "--grib2-dump": args.grib2_dump,
    }
    errors.extend(
        f"{flag} is not used by --source gfs"
        for flag, value in incompatible.items()
        if value is not None
    )
    if args.hierarchy_workers is not None and args.hierarchy_workers not in range(
        1, 33
    ):
        errors.append("--hierarchy-workers must be between 1 and 32")
    if args.hierarchy_workers is not None and args.geog_root is None:
        errors.append("--hierarchy-workers requires --geog-root")
    if args.statics_corridor is not None:
        if args.geog_root is None:
            errors.append(
                "--statics-corridor builds child-resolution statics from "
                "the geography source and requires --geog-root")
        if args.statics_corridor != "all":
            parts = [part for part in args.statics_corridor.split(",")
                     if part]
            if not parts or any(not part.strip().isdigit()
                                for part in parts):
                errors.append(
                    "--statics-corridor accepts 'all' or comma-separated "
                    f"child grid ids, got {args.statics_corridor!r}")
    if args.cycle is not None:
        try:
            parsed = datetime.strptime(args.cycle, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            errors.append("--cycle must use YYYY-MM-DD_HH:MM:SS")
        else:
            if (
                parsed.minute != 0
                or parsed.second != 0
                or parsed.hour not in {0, 6, 12, 18}
            ):
                errors.append("--cycle must be an exact 00/06/12/18 UTC GFS cycle")
    from gpuwm.physics_compat import (
        acknowledgement_delivery,
        validate_single_domain_physics_profile,
    )
    toml_acknowledgements, undecodable = _experiment_acknowledgements(
        args.experiment_config)
    if undecodable is not None:
        # Refusing here and returning is deliberate: the acknowledgement
        # channel is unreadable, so every acknowledgement-dependent
        # verdict below would be computed from a config this door could
        # not see, and the first of them would contradict the file.
        errors.append(undecodable)
        return errors
    acknowledgements, _ = acknowledgement_delivery(
        flag=tuple(args.ack), toml=toml_acknowledgements)
    # Validated only when the caller NAMED a profile.  The old WSM6
    # substitution made "no profile given" indistinguishable from "WSM6
    # requested" one process later; an unnamed config's own suite is
    # prepared as written and its verification status reported (owner
    # ruling 2026-07-31).
    if args.physics_profile is not None:
        try:
            validate_single_domain_physics_profile(
                args.physics_profile,
                expert_acknowledgements=acknowledgements)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _required_twentycr_args(args: argparse.Namespace) -> list[str]:
    if args.author_only:
        required = {
            "--source-root": args.source_root,
            "--author-input-manifest": args.author_input_manifest,
        }
        errors = [flag for flag, value in required.items() if value is None]
        incompatible = {
            "--source-manifest": args.source_sha256s,
            "--physics-profile": args.physics_profile,
            "--ack": args.ack or None,
            "--forecast-start-hour": args.forecast_start_hour,
            "--forecast-end-hour": args.forecast_end_hour,
            "--source-manifest-sha256": args.source_sha256s_sha256,
            "--wps-namelist": args.wps_namelist,
            "--geog-root": args.geog_root,
            "--experiment-config": args.experiment_config,
            "--output-root": args.output_root,
            "--preprocess-backend": args.preprocess_backend,
            "--preprocess-workers": args.preprocess_workers,
            "--cpu-preprocess-bridge": args.cpu_preprocess_bridge,
            "--hierarchy-workers": args.hierarchy_workers,
            "--mapped-engine": args.mapped_engine,
            "--grib2-inventory": args.grib2_inventory,
            "--grib2-dump": args.grib2_dump,
        }
        errors.extend(
            f"{flag} is not used while authoring a 20CRv3 manifest"
            for flag, value in incompatible.items()
            if value is not None
        )
    else:
        required = {
            "--source-manifest": args.source_sha256s,
            "--source-manifest-sha256": args.source_sha256s_sha256,
            # --grib2-inventory / --grib2-dump are deliberately NOT
            # required: omitted, the dispatch resolves both through the
            # shared bridge ladder, and the flags override it.
            "--wps-namelist": args.wps_namelist,
            "--geog-root": args.geog_root,
            "--experiment-config": args.experiment_config,
            "--output-root": args.output_root,
        }
        errors = [flag for flag, value in required.items() if value is None]
        if args.author_input_manifest is not None:
            errors.append(
                "--author-input-manifest requires --author-only for --source 20crv3"
            )
        if args.source_root is not None:
            errors.append(
                "--source-root is only used with 20CRv3 --author-only; the "
                "run consumes paths bound by --source-manifest"
            )

    incompatible = {
        "--source-format": args.source_format,
        "--physics-profile": args.physics_profile,
        "--ack": args.ack or None,
        "--forecast-start-hour": args.forecast_start_hour,
        "--forecast-end-hour": args.forecast_end_hour,
        # --mapped-engine is deliberately NOT refused here: the member
        # door runs the same composition as the generic composed route
        # now, so the route follows the engine table
        # (`twentycr_on_rust_engine` in `dispatch`) and the flag is
        # forwarded to the child.  The pre-port refusal guarded a route
        # that had no engine to select; that route is gone.
        "--mapping": args.mapping,
        "--descriptor": args.descriptor,
        "--author-mapping": args.author_mapping,
        "--composition": args.composition,
        "--input": args.mapped_inputs,
        "--input-list": args.input_list,
        "--supplement": args.supplement,
        "--contributing-mapping": args.contributing_mapping,
        "--provenance": args.provenance,
        "--bridge": args.bridge,
        "--vtable": args.vtable,
        "--grib": args.grib,
        "--gfs-series": args.gfs_series,
        "--cycle": args.cycle,
        "--static-cache": args.static_cache,
        "--static-receipt": args.static_receipt,
        "--static-input": args.static_input,
        "--domain-spec": args.domain_spec,
        "--namelist-input": args.namelist_input,
        "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
        "--valid-time": args.valid_time,
        "--run-seconds": args.run_seconds,
        "--history-interval-seconds": args.history_interval_seconds,
        "--pipeline-workers": args.pipeline_workers,
        "--prepare-workers": args.prepare_workers,
        "--child-workers": args.child_workers,
        "--root-preparation": args.root_preparation,
        "--source-orography": args.source_orography,
        "--source-orography-variable": args.source_orography_variable,
        "--domain-source-orography": args.domain_source_orography,
        "--no-stock-wrf-export": args.no_stock_wrf_export or None,
        "--statics-corridor": args.statics_corridor,
    }
    errors.extend(
        f"{flag} is not used by --source 20crv3"
        for flag, value in incompatible.items()
        if value is not None
    )
    if args.hierarchy_workers is not None and args.hierarchy_workers not in range(
        1, 33
    ):
        errors.append("--hierarchy-workers must be between 1 and 32")
    return errors


def _apply_packaged_profile(
    args: argparse.Namespace, adapter, program: str,
) -> list[str]:
    """Fill the mapped front door in from a packaged profile.

    A source whose mapping SHIPS is not a different program.  It is the
    same declarative mapped route with three of its arguments already
    decided, so this reads the profile -- format, mapping, composition,
    provenance, and the two composition roles -- and writes them into the
    namespace the generic mapped runner already consumes.  Nothing else
    about the route changes, which is the whole claim: adding a model whose
    mapping can be written costs a table row and three JSON documents.

    What a caller may still pass is what only they know: the input files,
    the invariant supplement, the namelist, the geography, the experiment
    and the output root.  What they may NOT pass is any of the five the
    profile decides -- an override there would let a run claim a packaged
    source's name while decoding through a mapping nobody shipped, and the
    receipt would say 20CRv3 about data that never came from that profile.
    """

    from gpuwm.source_authorities import (packaged_authorities,
                                          packaged_contributing_mappings,
                                          packaged_profile)

    try:
        profile = packaged_profile(str(adapter.packaged_profile))
        authorities = packaged_authorities(str(adapter.packaged_profile))
        contributing = packaged_contributing_mappings(
            str(adapter.packaged_profile))
    except (KeyError, FileNotFoundError, RuntimeError) as error:
        return [f"packaged source profile: {error}"]

    overridden = [
        flag for flag, value in {
            "--mapping": args.mapping,
            "--composition": args.composition,
            "--provenance": args.provenance,
            "--descriptor": args.descriptor,
            "--author-mapping": args.author_mapping,
            # A cross-source profile ships its donor mapping as a pinned
            # authority; a caller-supplied one would let a run claim the
            # packaged name while borrowing through a table nobody
            # shipped.  (A profile with no bindings takes none either --
            # the composed decode would refuse a surplus role anyway, and
            # refusing here names the profile instead.)
            "--contributing-mapping": args.contributing_mapping,
        }.items()
        if value
    ]
    errors = [
        f"{flag} is decided by the packaged {adapter.source_id} profile and "
        f"cannot be supplied"
        for flag in overridden
    ]
    declared_format = str(profile["source_format"])
    if args.source_format is not None and args.source_format != declared_format:
        errors.append(
            f"--source-format {args.source_format!r} differs from the packaged "
            f"{adapter.source_id} profile's {declared_format!r}"
        )
    if errors:
        return errors

    args.source_format = declared_format
    args.mapping = authorities["mapping"]
    args.composition = authorities["composition"]
    args.provenance = [
        f"{profile['provenance_role']}={authorities['provenance']}"
    ]
    args.contributing_mapping = [
        f"{role}={path}" for role, path in sorted(contributing.items())
    ]
    # `--supplement` takes a bare path here: the ROLE is the profile's, and
    # a caller retyping it is a caller who can mistype it.  A binding that
    # already names the profile's own role is accepted unchanged so the
    # printed `--dry-run` command can be pasted back.
    role = str(profile["data_role"])
    bound = []
    for value in args.supplement or ():
        text = str(value)
        if text.startswith(f"{role}="):
            bound.append(text)
        elif "=" in text and not Path(text.split("=", 1)[0]).exists():
            return [
                f"--supplement {text!r} names a role, but the packaged "
                f"{adapter.source_id} profile supplies the role {role!r}; "
                f"pass the supplement's PATH alone"
            ]
        else:
            bound.append(f"{role}={text}")
    args.supplement = bound
    return []


def _decodes_in_the_engine(args: argparse.Namespace, source_format: str) -> bool:
    """Will THIS run decode ``source_format`` inside gpuwm_mapped_engine?

    Two questions, both of which have to be yes: does the engine this
    run resolves to decode the format at all (read off
    ``ENGINE_CAPABILITIES``, the same table the router and the docs
    read, so a build whose engine lacks the format answers no), and did
    the caller leave the engine at its default rather than pinning
    ``--mapped-engine python``?

    Named breakage: a decoder-tool FLAG demanded for a format the engine
    decodes in process is not a harmless extra.  Naming a decoder tool
    is the spelling that routes a run back to the Python engine, so the
    demand made the ported decode unreachable through ``gpuwm prep`` --
    the only GRIB1 prep a user could spell was one that could not use
    the port.
    """

    if _resolve_mapped_engine(
            getattr(args, "mapped_engine", None)) != _MAPPED_ENGINE_RUST:
        return False
    return source_format in (
        _MAPPED_ENGINE_CAPABILITIES.get("decode") or frozenset())


def _required_mapped_args(args: argparse.Namespace) -> list[str]:
    # --contributing-mapping is deliberately NOT required: only a
    # cross-source composition declares bindings, and the decode refuses a
    # missing or surplus contributing mapping against the contract itself.
    required = {
        "--source-format": args.source_format,
        "--composition": args.composition,
        "--supplement": args.supplement,
        "--provenance": args.provenance,
    }
    if not args.author_only:
        required.update(
            {
                "--wps-namelist": args.wps_namelist,
                "--geog-root": args.geog_root,
                "--experiment-config": args.experiment_config,
                "--output-root": args.output_root,
            }
        )
    errors = [flag for flag, value in required.items() if not value]
    if (not args.mapped_inputs) == (args.input_list is None):
        # Two transports for one ordered file set: the repeated flag, or
        # the list file that says the same thing inside the Windows 32 KB
        # command-line limit.  Exactly one, because accepting both would
        # leave their relative order unspecified.
        errors.append("choose exactly one of --input or --input-list")
    existing_mapping = args.mapping is not None
    authored_mapping = args.descriptor is not None or args.author_mapping is not None
    if existing_mapping == authored_mapping:
        errors.append(
            "choose exactly one of --mapping or --descriptor with --author-mapping"
        )
    if authored_mapping:
        if args.descriptor is None:
            errors.append("--descriptor is required with --author-mapping")
        if args.author_mapping is None:
            errors.append("--author-mapping is required with --descriptor")
        if args.source_format in {"grib1", "grib2"} and args.vtable is None:
            errors.append("--vtable is required for a GRIB descriptor")
        if args.source_format == "netcdf" and args.vtable is not None:
            errors.append("--vtable is not used by a NetCDF descriptor")
    elif args.vtable is not None:
        errors.append("--vtable is only used with --descriptor on mapped input")

    existing_manifest = (
        args.source_sha256s is not None or args.source_sha256s_sha256 is not None
    )
    authored_manifest = args.author_input_manifest is not None
    if args.author_only and not authored_manifest:
        errors.append("--author-only requires --author-input-manifest")
    if existing_manifest == authored_manifest:
        errors.append(
            "choose exactly one of an existing --source-manifest plus digest "
            "or --author-input-manifest"
        )
    if existing_manifest and (
        args.source_sha256s is None or args.source_sha256s_sha256 is None
    ):
        errors.append(
            "--source-manifest and --source-manifest-sha256 are an atomic pair"
        )
    incompatible = {
        "--source-root": args.source_root,
        "--physics-profile": args.physics_profile,
        "--ack": args.ack or None,
        "--forecast-start-hour": args.forecast_start_hour,
        "--forecast-end-hour": args.forecast_end_hour,
        "--static-cache": args.static_cache,
        "--static-receipt": args.static_receipt,
        "--domain-spec": args.domain_spec,
        "--namelist-input": args.namelist_input,
        "--stock-wrf-namelist-input": args.stock_wrf_namelist_input,
        "--valid-time": args.valid_time,
        "--run-seconds": args.run_seconds,
        "--history-interval-seconds": args.history_interval_seconds,
        "--pipeline-workers": args.pipeline_workers,
        "--prepare-workers": args.prepare_workers,
        "--child-workers": args.child_workers,
        "--root-preparation": args.root_preparation,
        "--grib": args.grib,
        "--static-input": args.static_input,
        "--source-orography": args.source_orography,
        "--source-orography-variable": args.source_orography_variable,
        "--domain-source-orography": args.domain_source_orography,
        "--gfs-series": args.gfs_series,
        "--cycle": args.cycle,
        "--no-stock-wrf-export": args.no_stock_wrf_export or None,
        "--statics-corridor": args.statics_corridor,
    }
    errors.extend(
        f"{flag} is not used by --source mapped"
        for flag, value in incompatible.items()
        if value is not None
    )
    decoder_values = {
        "--bridge": args.bridge,
        "--grib2-inventory": args.grib2_inventory,
        "--grib2-dump": args.grib2_dump,
    }
    # ``allowed`` is not ``required``: the two GRIB2 tool flags are
    # OVERRIDES of the shared bridge ladder the dispatch resolves
    # through when they are omitted, so demanding them here was the
    # staged-tool papercut -- a refusal asking for paths to executables
    # ``gpuwm fetch-bridges`` had already staged.
    #
    # ``--bridge`` was the same papercut one format later: it stayed
    # required for GRIB1 on the reasoning that "that route has no ladder
    # default yet", which stopped being true when the engine gained its
    # own GRIB1 decode.  The requirement now follows the capability
    # table -- a format the engine decodes in process needs no decoder
    # flag, and one it does not still names what it needs at the door
    # rather than failing deep in the run.
    allowed_decoders = {
        "grib1": {"--bridge"},
        "grib2": {"--grib2-inventory", "--grib2-dump"},
        "netcdf": set(),
    }.get(args.source_format)
    required_decoders = {"grib1": {"--bridge"}}.get(args.source_format, set())
    if _decodes_in_the_engine(args, args.source_format):
        required_decoders = set()
    if allowed_decoders is not None:
        present = {flag for flag, value in decoder_values.items() if value is not None}
        errors.extend(
            f"{flag} is required for mapped {args.source_format}"
            for flag in sorted(required_decoders - present)
        )
        errors.extend(
            f"{flag} is not used by mapped {args.source_format}"
            for flag in sorted(present - allowed_decoders)
        )
    if args.hierarchy_workers is not None and args.hierarchy_workers not in range(
        1, 33
    ):
        errors.append("--hierarchy-workers must be between 1 and 32")
    errors.extend(
        _role_binding_errors(
            args.supplement,
            "--supplement",
            unique=False,
        )
    )
    errors.extend(
        _role_binding_errors(
            args.provenance,
            "--provenance",
            unique=True,
        )
    )
    return errors


def _hrrr_command(args: argparse.Namespace) -> list[str]:
    if args.root_preparation is not None:
        command = [
            sys.executable,
            "-m",
            "gpuwm.hrrr_hierarchy_direct",
            "--root-preparation",
            str(args.root_preparation),
            "--root-domain-spec",
            str(args.domain_spec),
            "--wps-namelist",
            str(args.wps_namelist),
            "--namelist-input",
            str(args.namelist_input),
            "--stock-wrf-namelist-input",
            str(args.stock_wrf_namelist_input),
            "--geog-root",
            str(args.geog_root),
            "--source-manifest",
            str(args.source_sha256s),
            "--source-manifest-sha256",
            str(args.source_sha256s_sha256),
            # --valid-time on THIS door is the cycle (it is validated as
            # "an exact hourly HRRR cycle" above and passed to
            # hrrr_source_window as one).  The hierarchy's own
            # --valid-time was model time zero, so forwarding this string
            # under that name handed it a time K hours early at any
            # nonzero lead.  Both values go through, spelled for what
            # they are, and the hierarchy derives its own clock.
            "--cycle",
            str(args.valid_time),
            "--forecast-start-hour",
            str(0 if args.forecast_start_hour is None
                else args.forecast_start_hour),
            "--output-root",
            str(args.output_root),
            "--workers",
            str(8 if args.child_workers is None else args.child_workers),
        ]
        if args.cpu_preprocess_bridge is not None:
            command.extend(("--cpu-preprocess-bridge", str(args.cpu_preprocess_bridge)))
        if args.statics_corridor is not None:
            # Forwarded, not dropped.  The hierarchy takes this flag
            # (``--statics-corridor [GRID_IDS]``) and it is what seals
            # child-resolution statics for a moving nest; the front door
            # accepted it, composed a command without it, and exited 0,
            # so a corridor-declaring tree prepared through `gpuwm prep`
            # reached the tree runner with no corridor set and was
            # refused there -- one stage after the flag that would have
            # prevented it was typed.
            if args.statics_corridor == "all":
                command.append("--statics-corridor")
            else:
                command.extend(("--statics-corridor", args.statics_corridor))
        return command

    tools = Path(__file__).resolve().parent.parent / "tools"
    command = [
        sys.executable,
        str(tools / "prepare_hrrr_wrf.py"),
        "--source-root", str(args.source_root),
        "--source-manifest", str(args.source_sha256s),
        "--source-manifest-sha256", str(args.source_sha256s_sha256),
        "--namelist-input", str(args.namelist_input),
        "--physics-profile", (
            ROUTE_DEFAULT_PHYSICS_PROFILE
            if args.physics_profile is None else args.physics_profile),
        "--cycle", str(args.valid_time),
        "--output-root", str(args.output_root),
        "--run-seconds", str(43_200 if args.run_seconds is None else args.run_seconds),
        "--forecast-start-hour", str(
            0 if args.forecast_start_hour is None
            else args.forecast_start_hour
        ),
        "--pipeline-workers", str(8 if args.pipeline_workers is None else args.pipeline_workers),
    ]
    if args.forecast_end_hour is not None:
        command.extend(("--forecast-end-hour", str(args.forecast_end_hour)))
    if args.history_interval_seconds is not None:
        command.extend(
            (
                "--history-interval-seconds",
                str(args.history_interval_seconds),
            )
        )
    if args.geog_root is not None:
        command.extend(("--geog-root", str(args.geog_root)))
    else:
        command.extend(("--static-cache", str(args.static_cache)))
        command.extend(("--static-receipt", str(args.static_receipt)))
    if args.domain_spec is not None:
        command.extend(("--domain-spec", str(args.domain_spec)))
    if args.wps_namelist is not None:
        command.extend(("--wps-namelist", str(args.wps_namelist)))
    if args.prepare_workers is not None:
        command.extend(("--prepare-workers", str(args.prepare_workers)))
    if args.preprocess_backend is not None:
        command.extend(("--preprocess-backend", args.preprocess_backend))
    if args.preprocess_workers is not None:
        command.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        command.extend((
            "--cpu-preprocess-bridge", str(args.cpu_preprocess_bridge)))
    if args.sealed_prepared_cache:
        command.append("--sealed-prepared-cache")
    if args.extend_root_preparation is not None:
        command.extend((
            "--extend-root-preparation",
            str(args.extend_root_preparation)))
    for acknowledgement in args.ack:
        command.extend(("--ack", acknowledgement))
    return command


def _era5_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gpuwm.era5_direct",
        "--grib",
        str(args.grib),
        "--vtable",
        str(args.vtable),
        "--bridge",
        str(args.bridge),
        "--wps-namelist",
        str(args.wps_namelist),
        "--experiment-config",
        str(args.experiment_config),
        "--input-manifest",
        str(args.source_sha256s),
        "--input-manifest-sha256",
        str(args.source_sha256s_sha256),
        "--output-root",
        str(args.output_root),
    ]
    if args.source_orography is not None:
        command.extend(("--source-orography", str(args.source_orography)))
        command.extend((
            "--source-orography-variable",
            str(args.source_orography_variable or "SOILHGT"),
        ))
    if args.static_input is not None:
        command.extend(("--static-input", str(args.static_input)))
        command.extend(("--static-receipt", str(args.static_receipt)))
    _append_preprocess_options(command, args)
    if args.geog_root is not None:
        command.extend(("--geog-root", str(args.geog_root)))
    for binding in args.domain_source_orography or ():
        command.extend(("--domain-source-orography", binding))
    if args.hierarchy_workers is not None:
        command.extend(("--hierarchy-workers", str(args.hierarchy_workers)))
    return command


def _gfs_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gpuwm.gfs_direct",
        "--series",
        str(args.gfs_series),
        "--cycle",
        str(args.cycle),
        "--bridge",
        str(args.bridge),
        "--wps-namelist",
        str(args.wps_namelist),
        "--experiment-config",
        str(args.experiment_config),
        "--input-manifest",
        str(args.source_sha256s),
        "--input-manifest-sha256",
        str(args.source_sha256s_sha256),
        "--output-root",
        str(args.output_root),
    ]
    if args.physics_profile is not None:
        # Passed only when the caller NAMED one.  Substituting the WSM6
        # default here made "no profile given" indistinguishable from
        # "WSM6 requested" one process later, which is how a domain tree
        # -- a route with no profile whitelist at all -- came to be
        # measured against a single-domain profile it never asked for.
        command.extend(("--physics-profile", args.physics_profile))
    if args.no_stock_wrf_export:
        command.append("--no-stock-wrf-export")
    if args.statics_corridor is not None:
        if args.statics_corridor == "all":
            command.append("--statics-corridor")
        else:
            command.extend(("--statics-corridor", args.statics_corridor))
    if args.static_input is not None:
        command.extend(("--static-input", str(args.static_input)))
        command.extend(("--static-receipt", str(args.static_receipt)))
    _append_preprocess_options(command, args)
    if args.geog_root is not None:
        command.extend(("--geog-root", str(args.geog_root)))
    if args.hierarchy_workers is not None:
        command.extend(("--hierarchy-workers", str(args.hierarchy_workers)))
    for acknowledgement in args.ack:
        command.extend(("--ack", acknowledgement))
    return command


def _twentycr_command(args: argparse.Namespace) -> list[str]:
    from gpuwm.source_authorities import twentycrv3_authorities

    authorities = twentycrv3_authorities()
    command = [
        sys.executable,
        "-m",
        "gpuwm.twentycrv3_wrf",
        "--mapping",
        str(authorities["mapping"]),
        "--composition",
        str(authorities["composition"]),
        "--provenance",
        str(authorities["provenance"]),
        "--manifest",
        str(args.source_sha256s),
        "--manifest-sha256",
        str(args.source_sha256s_sha256),
        "--wps-namelist",
        str(args.wps_namelist),
        "--geog-root",
        str(args.geog_root),
        "--experiment-config",
        str(args.experiment_config),
        "--output-root",
        str(args.output_root),
    ]
    # Forwarded only when RESOLVED: on the bare default the engine
    # composes and answers the record-inventory question in process, and
    # a forwarded tool path is the spelling the route reads as an
    # explicit Python-engine pin.
    if args.grib2_inventory is not None:
        command.extend(("--grib2-inventory", str(args.grib2_inventory)))
    if args.grib2_dump is not None:
        command.extend(("--grib2-dump", str(args.grib2_dump)))
    if args.mapped_engine is not None:
        command.extend(("--mapped-engine", args.mapped_engine))
    _append_preprocess_options(command, args)
    if args.hierarchy_workers is not None:
        command.extend(("--hierarchy-workers", str(args.hierarchy_workers)))
    return command


def _mapped_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gpuwm.mapped_direct",
        "--source-format",
        str(args.source_format),
        "--composition",
        str(args.composition),
        "--mapping",
        str(args.mapping),
        "--input-manifest",
        str(args.source_sha256s),
        "--input-manifest-sha256",
        str(args.source_sha256s_sha256),
        "--wps-namelist",
        str(args.wps_namelist),
        "--geog-root",
        str(args.geog_root),
        "--experiment-config",
        str(args.experiment_config),
        "--output-root",
        str(args.output_root),
    ]
    if args.input_list is not None:
        # The caller's compact spelling is forwarded as itself: expanding
        # it here would rebuild the exact command line Windows refuses.
        command.extend(("--input-list", str(args.input_list)))
    else:
        for path in args.mapped_inputs:
            command.extend(("--input", str(path)))
    for binding in args.supplement:
        command.extend(("--supplement", binding))
    for binding in args.provenance:
        command.extend(("--provenance", binding))
    for binding in args.contributing_mapping or ():
        command.extend(("--contributing-mapping", binding))
    if args.mapped_engine is not None:
        command.extend(("--mapped-engine", args.mapped_engine))
    if args.source_format == "grib1" and args.bridge is not None:
        # Forwarded only when there is something to forward, for the
        # reason spelled beside the GRIB2 pair below: an omitted bridge
        # means the engine decodes GRIB1 in process, and forwarding a
        # path anyway would pin the Python engine -- a default that
        # silently un-defaults itself.
        command.extend(("--grib1-bridge", str(args.bridge)))
    elif args.source_format == "grib2" and args.grib2_inventory is not None:
        # Forwarded only when there is something to forward.  Omitted,
        # the mapped route decodes on the default engine; a resolved
        # pair is forwarded exactly as before so the Python engine keeps
        # the staged-tool default it already had.
        command.extend(
            (
                "--grib2-inventory",
                str(args.grib2_inventory),
                "--grib2-dump",
                str(args.grib2_dump),
            )
        )
    _append_preprocess_options(command, args)
    if args.hierarchy_workers is not None:
        command.extend(
            (
                "--hierarchy-workers",
                str(args.hierarchy_workers),
            )
        )
    # The one thing the generic runner cannot know: which registered
    # source id this preparation is FOR.  It is what the prepared
    # forecast binds to, so without it the mapped route could finish and
    # print no run command at all -- while the GFS route printed a
    # complete hash-bound one -- and a reader hunting the digests by
    # hand lands on proof.json's `proof_content_sha256`, which is the
    # one value --proof-sha256 never accepts.
    command.extend(("--prepared-forecast-source", str(args.source)))
    return command


def _append_preprocess_options(command: list[str], args: argparse.Namespace):
    if args.preprocess_backend is not None:
        command.extend(("--preprocess-backend", args.preprocess_backend))
    if args.preprocess_workers is not None:
        command.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        command.extend(("--cpu-preprocess-bridge", str(args.cpu_preprocess_bridge)))


def _author_mapped_contract(args: argparse.Namespace) -> dict[str, object]:
    """Materialize explicitly requested create-only mapped authorities."""

    result: dict[str, object] = {
        "schema": "rw-wps.contract-authoring.v1",
        "status": "VALIDATED_NOT_STOCK_WRF_CERTIFIED",
    }
    created_mapping: tuple[Path, Path, str] | None = None
    if args.descriptor is not None:
        receipt = author_mapping(
            args.descriptor,
            args.author_mapping,
            vtable_path=args.vtable,
            expected_format=args.source_format,
        )
        args.mapping = args.author_mapping
        created_mapping = (
            Path(args.mapping).resolve(),
            Path(args.mapping).resolve().with_name(
                f"{Path(args.mapping).resolve().stem}.authoring.json"
            ),
            str(receipt["mapping"]["sha256"]),
        )
        result["mapping"] = receipt
    if args.author_input_manifest is not None:
        supplements = _role_bindings(
            args.supplement or (),
            multiple=True,
        )
        provenance = _role_bindings(
            args.provenance or (),
            multiple=False,
        )
        try:
            receipt = author_input_manifest(
                args.author_input_manifest,
                mapping_path=args.mapping,
                composition_path=args.composition,
                primary_files=args.mapped_inputs,
                supplement_files=supplements,
                provenance_files=provenance,
                grib1_bridge=args.bridge,
                grib2_inventory=args.grib2_inventory,
                grib2_dump=args.grib2_dump,
                expected_format=args.source_format,
            )
        except BaseException:
            if created_mapping is not None:
                mapping_path, authoring_path, expected_digest = created_mapping
                if (
                    mapping_path.is_file()
                    and _sha256(mapping_path) == expected_digest
                    and authoring_path.is_file()
                ):
                    try:
                        authored = json.loads(authoring_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        authored = None
                    if (
                        isinstance(authored, dict)
                        and isinstance(authored.get("mapping"), dict)
                        and authored["mapping"].get("sha256") == expected_digest
                    ):
                        authoring_path.unlink()
                        mapping_path.unlink()
            raise
        args.source_sha256s = args.author_input_manifest
        args.source_sha256s_sha256 = receipt["manifest"]["sha256"]
        # A second paste of the printed prep command re-authors nothing
        # -- the manifest already on disk is byte-identical to what this
        # call composed -- and says so rather than claiming a write it
        # did not make (UX finding N13).
        verb = "AUTHORED" if receipt.get("reauthored", True) else "MATCHED"
        print(
            f"{verb} input_manifest="
            f"{args.source_sha256s} sha256={args.source_sha256s_sha256}"
            + ("" if receipt.get("reauthored", True)
               else "  # already on disk, byte-identical; nothing rewritten"),
            file=sys.stderr,
        )
        result["input_manifest"] = receipt
    if created_mapping is not None:
        print(
            f"AUTHORED mapping={args.mapping} "
            f"sha256={created_mapping[2]}",
            file=sys.stderr,
        )
    return result


def _author_twentycr_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Author the 20CRv3 input manifest, and say what to do with it.

    The parity gap this closes: the GFS route's authoring step ends by
    printing the whole front-door command with its digest filled in, and
    every mapped authoring step prints an ``AUTHORED`` line.  20CRv3's
    printed nothing -- a user who had just watched a manifest be written
    still had to find its path and compute its SHA-256 by hand before
    they could run anything.

    It cannot print the WHOLE command, and does not pretend to.  20CRv3
    authoring deliberately REFUSES ``--wps-namelist``, ``--geog-root``,
    ``--experiment-config``, ``--output-root`` and the two GRIB2 tool
    paths, so those values do not exist in this process.  What it prints
    is the half it knows -- bound, exact, pasteable -- and a comment
    naming the half it does not, rather than a command with placeholders
    in it that fails when pasted.
    """

    from gpuwm.twentycrv3_direct import write_20crv3_manifest

    output = Path(args.author_input_manifest).resolve()
    source = write_20crv3_manifest(args.source_root, output)
    digest = _sha256(output)
    print(f"AUTHORED input_manifest={output} sha256={digest}",
          file=sys.stderr)
    print("20crv3: next: feed the 20CRv3 front door, manifest already "
          "bound:", file=sys.stderr)
    print(f"  --source-manifest {output} "
          f"--source-manifest-sha256 {digest}", file=sys.stderr)
    print("  # authoring refuses the rest of the run's flags, so it "
          "cannot bind them\n"
          "  # for you: --grib2-inventory, --grib2-dump, "
          "--wps-namelist, --geog-root,\n"
          "  # --experiment-config, --output-root.",
          file=sys.stderr)
    return {
        "schema": "rw-wps.20crv3-manifest-authoring.v1",
        "status": "PASS",
        "manifest": {
            "path": str(output),
            "sha256": digest,
            "content_sha256": source["content_sha256"],
            "member": source["member"],
            "file_count": source["file_count"],
        },
    }


def _quote_command(command: list[str]) -> str:
    # POSIX display form because the certified runtime is Linux/CUDA.  The
    # command is passed as argv, never through a shell.
    import shlex

    return shlex.join(value.replace("\\", "/") for value in command)


def main(argv: list[str] | None = None) -> int:
    from gpuwm.progress import line_buffer_stdout

    # The standalone console-script door gets the same flush discipline
    # `gpuwm` sets for its subcommands: a redirected preprocessing run
    # streams its stage lines instead of delivering them all at exit
    # (UX finding N10).  `gpuwm prep` comes through `gpuwm.cli.main`,
    # which has already set it; calling it twice costs nothing.
    line_buffer_stdout()
    parser = _parser()
    return dispatch(parser.parse_args(argv), parser=parser)


def dispatch(args: argparse.Namespace, *,
             parser: argparse.ArgumentParser,
             program: str = "rw-wps") -> int:
    """Run the preprocessing stage from an already-parsed namespace.

    Split out of :func:`main` so the stage has ONE body behind TWO front
    doors: the standalone ``rw-wps``/``gpuwm-wrf-init`` console script,
    and the ``gpuwm prep`` subcommand that adopts this module's parser
    through argparse ``parents=``.  Nothing here changed when it was
    split -- ``main`` is the same parse followed by the same body -- and
    that is the point: a second implementation of preprocessing is
    exactly the "one route wearing two names" this seam exists to
    prevent.

    ``parser`` is whichever parser produced ``args``, because the body
    calls ``parser.error`` and the usage line a reader sees has to be
    the usage line of the command they typed.
    """

    # Which tree is executing, before any source bytes are read; and a
    # refusal when this install's version claims contradict each other.
    from gpuwm.provenance_gate import announce_for_main

    refusal = announce_for_main(
        program, explain=bool(getattr(args, "explain", False)))
    if refusal is not None:
        print(f"{program}: {refusal}", file=sys.stderr)
        return 2
    if args.source_top_pressure_pa is not None and not args.namelist_support_report:
        parser.error(
            "--source-top-pressure-pa is only valid with "
            "--namelist-support-report"
        )
    if (
        args.canonical_physics_plan_output is not None
        and args.validate_physics_plan is None
    ):
        parser.error(
            "--canonical-physics-plan-output is only valid with "
            "--validate-physics-plan"
        )
    if args.dry_run and (
        args.descriptor is not None
        or args.author_mapping is not None
        or args.author_input_manifest is not None
    ):
        print(
            "--dry-run is side-effect free and cannot author files; use "
            "--author-only to create mapped contracts without starting a run",
            file=sys.stderr,
        )
        return EXIT_USAGE

    inventory_count = sum(
        (
            bool(args.list_sources),
            args.show_source is not None,
            bool(args.show_support_matrix),
            bool(args.show_physics_registry),
            args.validate_physics_plan is not None,
            args.validate_hrrr_domain is not None,
            bool(args.namelist_support_report),
        )
    )
    if inventory_count > 1:
        parser.error("choose exactly one inventory option")

    if args.list_sources:
        manifest = source_capability_manifest()
        manifest["canonical_source_frame"] = {
            "schema": "gpuwm-canonical-source-frame-v1",
            "field_requirements": canonical_field_requirements(),
        }
        print(_json(manifest))
        return 0

    if args.show_support_matrix:
        support = json.loads(_SUPPORT_MATRIX.read_text(encoding="utf-8"))
        if support.get("schema") != "gpuwm-native-wrf-support-matrix-v1":
            raise RuntimeError("bundled native WRF support matrix schema drift")
        print(_json(support))
        return 0

    if args.show_physics_registry:
        from gpuwm.physics_registry import physics_registry

        _write_canonical_json(physics_registry())
        return 0

    if args.validate_hrrr_domain is not None:
        unrelated = _active_action_arguments(
            args,
            allowed=frozenset({"validate_hrrr_domain"}),
        )
        if unrelated:
            parser.error(
                "--validate-hrrr-domain cannot be combined with other "
                "action arguments: " + ", ".join(unrelated)
            )
        report = _hrrr_domain_validation(args.validate_hrrr_domain)
        _write_canonical_json(report)
        return 0 if report["status"] == "PASS" else EXIT_CONFIG

    if args.validate_physics_plan is not None:
        from gpuwm.physics_registry import (
            VALIDATION_SCHEMA,
            load_physics_plan,
            physics_registry,
            registry_sha256,
            validate_physics_plan,
        )

        try:
            physics_plan = load_physics_plan(args.validate_physics_plan)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                "schema": VALIDATION_SCHEMA,
                "launchable": False,
                "errors": [{
                    "code": "plan-read",
                    "path": str(args.validate_physics_plan),
                    "message": str(exc),
                }],
                "warnings": [],
                "registry_sha256": registry_sha256(physics_registry()),
                "plan_sha256": None,
                "plan_id": None,
                "context": None,
                "resolved_domains": [],
                "asset_requirements": [],
            }
        else:
            report = validate_physics_plan(physics_plan)
            if args.canonical_physics_plan_output is not None:
                try:
                    _create_canonical_json(
                        args.canonical_physics_plan_output,
                        physics_plan,
                    )
                except (OSError, TypeError, ValueError) as exc:
                    report["launchable"] = False
                    report["errors"].append({
                        "code": "canonical-plan-write",
                        "path": str(args.canonical_physics_plan_output),
                        "message": str(exc),
                    })
        _write_canonical_json(report)
        return 0 if report["launchable"] else EXIT_CONFIG

    if args.show_source:
        try:
            adapter = get_source_adapter(args.show_source)
        except ValueError as exc:
            parser.error(str(exc))
        print(_json(adapter.to_dict()))
        return 0

    if args.namelist_support_report:
        if args.wps_namelist is None or args.namelist_input is None:
            parser.error(
                "--namelist-support-report requires --wps-namelist and "
                "--namelist-input"
            )
        from gpuwm.namelist_compat import analyze_namelists

        # This is step one of docs/migrating-from-wps.md, so it is the
        # first thing a person migrating an existing WRF setup runs --
        # and the commonest way to get it wrong is to name a file that
        # is not there yet.  `gpuwm import-namelist` answers that in one
        # sentence; this surface used to answer it with a five-frame
        # traceback ending in pathlib.  Same condition, same sentence.
        try:
            report = analyze_namelists(
                args.wps_namelist,
                args.namelist_input,
                source_top_pressure_pa=args.source_top_pressure_pa,
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            print(f"--namelist-support-report: {error}", file=sys.stderr)
            return EXIT_CONFIG
        print(_json(report))
        return 0 if report["verdict"] == "PASS" else EXIT_CONFIG

    if not args.source:
        parser.error("--source is required unless an inventory option is used")
    try:
        adapter = get_source_adapter(args.source)
    except ValueError as exc:
        parser.error(str(exc))

    if adapter.runner != "hrrr_f00_f12_v1" and (
            args.sealed_prepared_cache
            or args.extend_root_preparation is not None):
        print(
            "invalid or missing run arguments: --sealed-prepared-cache and "
            "--extend-root-preparation are only used by --source hrrr",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if not adapter.runnable:
        # A reason, or nothing.  The status is already on the line, so
        # falling back to it produced `status=adapter_mapping_required:
        # adapter_mapping_required` -- an echo that reads as a truncated
        # message and is still what `rap` and `nam` print.  Source-
        # agnostic on purpose: the adapters that say something useful
        # (gdas grew `notes`, the composition family has
        # `composition_requirement`) are unchanged, and any adapter that
        # has nothing to add stops pretending it does.
        reason = adapter.composition_requirement or adapter.notes
        print(
            f"REFUSED source={adapter.source_id} "
            f"status={adapter.status.value}"
            + (f": {reason}" if reason else ""),
            file=sys.stderr,
        )
        # The mechanism paragraph, on the project's one layering
        # convention: the line above already named the source, the
        # status and the adapter's own reason, which is what a reader
        # acts on.  This says why the bar is where it is, and waits to
        # be asked.
        if explain_enabled(args):
            print(
                "A readable GRIB/NetCDF product is not treated as a complete WRF state. "
                "This adapter must declare field, level, cadence, and missing-state "
                "policies before it can run; unchanged stock-wrf evidence is a "
                "separate certification gate.",
                file=sys.stderr,
            )
        else:
            print(f"  (run {program} --explain for why this bar exists)",
                  file=sys.stderr)
        return EXIT_CONFIG

    runners = {
        "hrrr_f00_f12_v1": (_required_hrrr_args, _hrrr_command),
        "era5_combined_grib1_v1": (_required_era5_args, _era5_command),
        "gfs_pgrb2_0p25_v1": (_required_gfs_args, _gfs_command),
        "twentycrv3_member_grib2_v1": (
            _required_twentycr_args,
            _twentycr_command,
        ),
        "mapped_composition_v1": (_required_mapped_args, _mapped_command),
    }
    if (
        adapter.status
        not in {AdapterStatus.CERTIFIED, AdapterStatus.RUNNABLE_NOT_CERTIFIED}
        or adapter.runner not in runners
    ):
        print("REFUSED: inconsistent runnable adapter declaration", file=sys.stderr)
        return EXIT_CONFIG

    # A packaged profile decides the mapped route's declarative arguments
    # BEFORE they are validated, so the caller is checked against the
    # arguments they actually have to supply rather than against five the
    # distribution already answered.  It also runs before the decoder
    # binding below, because the profile is what decides SOURCE FORMAT:
    # until it has, a packaged GRIB2 source looks formatless and the
    # GRIB2 tool binding would silently not apply to it.
    if (adapter.packaged_profile is not None
            and adapter.runner == "mapped_composition_v1"):
        profile_errors = _apply_packaged_profile(args, adapter, program)
        if profile_errors:
            print(
                "invalid or missing run arguments: " + ", ".join(profile_errors),
                file=sys.stderr,
            )
            return EXIT_USAGE

    # The mapped GRIB1 bridge is bound on the same terms as the mapped
    # GRIB2 tools below: only when the PYTHON engine will decode.  A
    # bare run on the default engine decodes GRIB1 in process, so
    # consulting the distribution manifest for a bridge would refuse on
    # a box that never staged one -- for a subprocess the run is not
    # going to launch.
    mapped_grib1_in_engine = (
        adapter.runner == "mapped_composition_v1"
        and args.source_format == "grib1"
        and args.bridge is None
        and _decodes_in_the_engine(args, "grib1")
    )
    bridge_variable = {
        "era5_combined_grib1_v1": "GPUWM_GRIB1_BRIDGE",
        "gfs_pgrb2_0p25_v1": "GPUWM_GFS_GRIB2_BRIDGE",
        "mapped_composition_v1": (
            "GPUWM_GRIB1_BRIDGE"
            if args.source_format == "grib1" and not mapped_grib1_in_engine
            else None
        ),
    }.get(adapter.runner)
    authoring_twentycr = (
        adapter.runner == "twentycrv3_member_grib2_v1" and args.author_only
    )
    try:
        if not authoring_twentycr and bridge_variable is not None:
            args.bridge = _distribution_decoder(
                args.bridge,
                bridge_variable,
                "--bridge",
            )
        # The mapped route's GRIB2 tool paths are only wanted when the
        # PYTHON engine will do the work.  On the Rust engine the work
        # is in process, and resolving the two executables anyway would
        # both fail on a machine that never staged them and -- worse --
        # forward them as explicit tool pins, which is precisely the
        # spelling that routes a call back to the Python engine.  A
        # default that silently un-defaults itself is one defect this
        # avoids.
        #
        # The question is asked of the capability table for the
        # SUBCOMMAND this route runs, not of the engine default.  Asking
        # the default was the other defect, and it was the worse one:
        # the engine decodes GRIB2 in process but composes nothing, and
        # `gpuwm.mapped_direct` composes on every call
        # (MAPPED_ROUTE_SUBCOMMAND), so a door that read "the default is
        # Rust" forwarded no tools while the route routed itself to the
        # Python engine and died in its decoder contract.  Measured on
        # every one of the twelve registered composition sources.
        # `_mapped_engine_choice` owns the whole rule -- explicit tool
        # pins, an explicit engine request, then the table -- so when
        # the port lane teaches the engine to compose, this door follows
        # the table with no edit here.
        mapped_on_rust_engine = (
            adapter.runner == "mapped_composition_v1"
            and _mapped_engine_choice(
                # The grib2 ladder question is only asked of
                # `--source-format grib2`, and the grib1 bridge for the
                # grib1 spelling was resolved above; passing it here
                # would answer "python" for a reason already decided.
                grib1_bridge=None,
                grib2_inventory=args.grib2_inventory,
                grib2_dump=args.grib2_dump,
                subcommand=_MAPPED_ROUTE_SUBCOMMAND,
                source_format=args.source_format,
                explicit=getattr(args, "mapped_engine", None),
            ) == _MAPPED_ENGINE_RUST
        )
        # The member route asks the SAME question the generic composed
        # route asks, because it now runs the same composition: on the
        # bare default the engine composes AND answers the raw
        # record-inventory question in process, so resolving the
        # subprocess pair would forward explicit tool pins -- the exact
        # spelling that un-defaults the Rust engine.
        twentycr_on_rust_engine = (
            adapter.runner == "twentycrv3_member_grib2_v1"
            and _mapped_engine_choice(
                grib1_bridge=None,
                grib2_inventory=args.grib2_inventory,
                grib2_dump=args.grib2_dump,
                subcommand=_MAPPED_ROUTE_SUBCOMMAND,
                source_format="grib2",
                explicit=getattr(args, "mapped_engine", None),
            ) == _MAPPED_ENGINE_RUST
        )
        uses_generic_grib2 = (
            (
                adapter.runner == "twentycrv3_member_grib2_v1"
                and not twentycr_on_rust_engine
            )
            or (
                adapter.runner == "mapped_composition_v1"
                and args.source_format == "grib2"
                and not mapped_on_rust_engine
            )
        )
        if not authoring_twentycr and uses_generic_grib2:
            args.grib2_inventory = _distribution_decoder(
                args.grib2_inventory,
                "GPUWM_GRIB2_INVENTORY",
                "--grib2-inventory",
            )
            args.grib2_dump = _distribution_decoder(
                args.grib2_dump,
                "GPUWM_GRIB2_DUMP",
                "--grib2-dump",
            )
    except (OSError, TypeError, ValueError) as error:
        print(f"native decoder authority failed: {error}", file=sys.stderr)
        return EXIT_CONFIG

    required_args, build_command = runners[adapter.runner]
    configuration_errors = required_args(args)
    if configuration_errors:
        print(
            "invalid or missing run arguments: " + ", ".join(configuration_errors),
            file=sys.stderr,
        )
        return EXIT_USAGE
    if adapter.runner == "mapped_composition_v1" and args.input_list is not None:
        # Materialized here -- after the route's own argument validation,
        # before authoring or command composition -- so a bad list file is
        # refused at this door in the same sentence shape as any other
        # argument fault, and everything downstream (manifest authoring,
        # the composed command) sees the one ordered file set both
        # spellings describe.
        try:
            args.mapped_inputs = read_input_list(args.input_list)
        except ValueError as error:
            print(
                f"invalid or missing run arguments: {error}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    # The staged-tool default: omitted, the two GRIB2 tool paths resolve
    # through the same ladder every other bridge uses (environment
    # override, checkout build, staged bridges, wheel) -- through the
    # engine's own resolver, so this pre-flight and the route it launches
    # cannot give different answers.  The flags override it per tool,
    # exactly like the per-tool environment variables.  After the usage
    # validation above, deliberately: an argument-vocabulary mistake is
    # the caller's to fix once, before the estate is consulted.
    if not authoring_twentycr and uses_generic_grib2 and (
            args.grib2_inventory is None or args.grib2_dump is None):
        from gpuwm import mapped_source

        try:
            inventory, dump = (
                mapped_source._build_grib2_tools())  # noqa: SLF001 - the resolver of record
        except (OSError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return EXIT_CONFIG
        if args.grib2_inventory is None:
            args.grib2_inventory = inventory
        if args.grib2_dump is None:
            args.grib2_dump = dump

    # The staged-bridge default for the era5/gfs direct routes: an
    # omitted --bridge resolves through the same ladder every other
    # bridge uses (environment override, checkout build, libexec,
    # staged ~/.gpuwm/bridges), through the resolver `gpuwm go` and
    # `gpuwm doctor` already share, so this door and those cannot give
    # different answers.  FIRST-LIGHT documented this sentence before
    # it was true; the measured door demanded the flag and exited 64
    # (UX finding N12).  The flag stays an override, exactly like the
    # GRIB2 tool flags above, and the refusal below fires only when the
    # ladder genuinely resolves nothing -- naming what it searched and
    # the install-aware remedy, never a bare demand for a path the user
    # does not have.  After the usage validation, deliberately: an
    # argument-vocabulary mistake is the caller's to fix once, before
    # the estate is consulted.
    ladder_bridge_source = {
        "era5_combined_grib1_v1": "era5",
        "gfs_pgrb2_0p25_v1": "gfs",
    }.get(adapter.runner)
    if args.bridge is None and ladder_bridge_source is not None:
        from gpuwm import bridges

        try:
            args.bridge = bridges.resolve_source_decoder(
                ladder_bridge_source)
        except (bridges.DecoderContractError, FileNotFoundError) as error:
            print(str(error), file=sys.stderr)
            return EXIT_CONFIG

    # The front-door manifest default for the GFS direct route: with the
    # ``--source-manifest`` pair omitted, this door authors and
    # digest-binds the input manifest itself, from the fetched directory
    # the series lives in -- the same document, through the same
    # authoring function, that `gpuwm fetch --author-front-door-manifest`
    # writes.  That second fetch invocation was the whole handoff gap of
    # UX finding N11: the fetch leaves the four-file front door and prep
    # follows it directly now.  The explicit pair stays an override and
    # pins an existing manifest verbatim.
    if (adapter.runner == "gfs_pgrb2_0p25_v1"
            and args.source_sha256s is None
            and args.source_sha256s_sha256 is None):
        from gpuwm import fetch as fetch_module

        try:
            manifest_path, manifest_digest = (
                fetch_module.author_gfs_front_door_manifest(
                    out=Path(args.gfs_series).parent,
                    bridge=Path(args.bridge),
                    wps_namelist=Path(args.wps_namelist),
                    experiment_config=Path(args.experiment_config),
                    static_input=args.static_input,
                    static_receipt=args.static_receipt,
                    progress=lambda line: None))
        except (OSError, ValueError) as error:
            print(f"gfs front-door manifest: {error}", file=sys.stderr)
            print("  # pass --source-manifest FILE "
                  "--source-manifest-sha256 DIGEST to bind an existing "
                  "manifest instead", file=sys.stderr)
            return EXIT_CONFIG
        args.source_sha256s = manifest_path
        args.source_sha256s_sha256 = manifest_digest
        print(f"prep --source gfs: authored and digest-bound the "
              f"front-door input manifest itself: {manifest_path} "
              f"(sha256 {manifest_digest}); the --source-manifest pair "
              f"pins an existing one instead", file=sys.stderr)

    if authoring_twentycr:
        try:
            receipt = _author_twentycr_manifest(args)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(f"20CRv3 manifest authoring failed: {error}", file=sys.stderr)
            return EXIT_CONFIG
        print(_json(receipt))
        return 0
    if adapter.runner == "hrrr_f00_f12_v1":
        if (args.run_seconds is not None and args.run_seconds <= 0) or (
            args.pipeline_workers is not None and args.pipeline_workers <= 0
        ):
            print("run-seconds and pipeline-workers must be positive", file=sys.stderr)
            return EXIT_USAGE
        if args.prepare_workers is not None and args.prepare_workers <= 0:
            print("prepare-workers must be positive", file=sys.stderr)
            return EXIT_USAGE
        selected = args.preprocess_backend or "cuda"
        if args.root_preparation is None and (
                selected != "cpu" and args.cpu_preprocess_bridge is not None):
            print(
                "cpu-preprocess-bridge requires --preprocess-backend cpu",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if (args.root_preparation is None and selected == "cuda"
                and args.preprocess_workers is not None):
            print(
                "preprocess-workers requires --preprocess-backend cpu or auto",
                file=sys.stderr,
            )
            return EXIT_USAGE
    else:
        if args.preprocess_workers is not None and args.preprocess_workers <= 0:
            print("preprocess-workers must be positive", file=sys.stderr)
            return EXIT_USAGE
        selected = args.preprocess_backend or "cuda"
        if selected != "cpu" and args.cpu_preprocess_bridge is not None:
            print(
                "cpu-preprocess-bridge requires --preprocess-backend cpu",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if selected == "cuda" and args.preprocess_workers is not None:
            print(
                "preprocess-workers requires --preprocess-backend cpu or auto",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if (args.hierarchy_workers is not None
                and args.hierarchy_workers > 1 and selected != "cpu"):
            print(
                "hierarchy-workers greater than 1 requires the explicit "
                "--preprocess-backend cpu",
                file=sys.stderr,
            )
            return EXIT_USAGE

    if adapter.runner == "mapped_composition_v1" and (
        args.descriptor is not None or args.author_input_manifest is not None
    ):
        try:
            authoring_receipt = _author_mapped_contract(args)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            # ``FileNotFoundError(path)`` stringifies to the bare path,
            # so without the class name this line printed a path with
            # no sentence -- a user could not tell WHAT failed about it.
            # A refusal that already IS a sentence keeps its own words:
            # the manifest-overwrite refusal names the breakage and a
            # remedy, and "FileExistsError: refusing to overwrite ..."
            # labels a sentence twice.
            detail = str(error)
            if (isinstance(error, OSError) and not error.strerror
                    and not reads_as_a_sentence(detail)):
                detail = f"{type(error).__name__}: {detail}"
                if isinstance(error, FileNotFoundError):
                    detail += " (a file this step was told to read does not exist)"
            print(f"mapped contract authoring failed: {detail}",
                  file=sys.stderr)
            return EXIT_CONFIG
        if args.author_only:
            print(_json(authoring_receipt))
            return 0

    command = build_command(args)
    if args.dry_run:
        print(_quote_command(command))
        return 0
    return _run_native_adapter(command)


#: CreateProcess refuses a command line longer than 32,767 characters and
#: reports the excess as ERROR_FILENAME_EXCED_RANGE.
_WINDOWS_ARGV_LIMIT_WINERROR = 206


def _is_argv_limit_error(error: OSError) -> bool:
    """Did the PLATFORM refuse the command line's length?

    Windows says WinError 206 out of CreateProcess; POSIX execve says
    ``E2BIG`` past ``ARG_MAX``.  Nothing else is this error: a missing
    interpreter or a permission fault must keep today's report.
    """

    return (
        getattr(error, "winerror", None) == _WINDOWS_ARGV_LIMIT_WINERROR
        or error.errno == errno.E2BIG
    )


def _compact_input_argv(command: list[str]) -> tuple[Path, list[str]] | None:
    """The same command with its ``--input`` pairs carried by a list file.

    ``None`` when there is nothing to compact -- no per-file pairs, or
    the command already spells ``--input-list`` -- in which case the
    launch failure is reported exactly as before.
    """

    paths: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == "--input" and index + 1 < len(command):
            paths.append(command[index + 1])
            index += 2
            continue
        rest.append(command[index])
        index += 1
    if not paths or "--input-list" in rest:
        return None
    handle, name = tempfile.mkstemp(prefix="rw-wps-input-list-",
                                    suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(paths) + "\n")
    return Path(name), rest + ["--input-list", name]


def _run_native_adapter(command: list[str]) -> int:
    """Launch the composed adapter command; exit codes pass through.

    The relaunch is the architecture -- the printed ``--dry-run`` command
    and the executed one are the same object, and the adapters stay
    separate programs.  What it must not be is the reason a source with
    hundreds of per-field input files cannot run on Windows: when the
    PLATFORM refuses the command line's length (and only then -- a
    command it accepts is never rewritten), the per-file ``--input``
    pairs move into a temporary list file the adapter reads, and the
    launch is retried once.  CreateProcess fails before the child
    exists, so the retry re-runs nothing.
    """

    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        compacted = _compact_input_argv(command) \
            if _is_argv_limit_error(exc) else None
        if compacted is None:
            print(f"failed to launch native adapter: {exc}", file=sys.stderr)
            return 70
        list_file, retry_command = compacted
        try:
            return subprocess.run(retry_command, check=False).returncode
        except OSError as second:
            print(f"failed to launch native adapter: {second}",
                  file=sys.stderr)
            return 70
        finally:
            try:
                list_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
