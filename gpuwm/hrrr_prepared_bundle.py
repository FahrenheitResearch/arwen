"""Publish the portable authorities a prepared HRRR bundle needs.

The native HRRR preparation has always produced a complete, certified
prepared case: a prepared cache in the shared format, a native static
cache, a geometry receipt, a sealed decoder bridge manifest, and a
stock-WRF export the project has run WRF from.  What it never produced
is the small set of PORTABLE authorities
:func:`gpuwm.prepared_single_domain_forecast.preflight_prepared_forecast`
re-derives at its front door -- a ``proof.json``, a role-keyed source
manifest, and the experiment config those two are bound to.  Without
them an HRRR case is runnable by the native benchmark and by nothing
else, which is why the cycling DA driver could not start from one.

This module writes exactly those three, from the values the preparation
already computed, and nothing else.  It renames no existing artifact and
recomputes no science: every digest it records is taken from a file the
preparation wrote, and the experiment config it publishes is rendered
from the tables the preparation itself handed ``build_experiment`` and
verified to reload to the same prepared-domain identity
(:mod:`gpuwm.experiment_document`).

The reader for these documents lives in
:mod:`gpuwm.prepared_single_domain_forecast`; the two are held together
by tests that publish with this writer and admit with that reader,
rather than by two hand-matched spellings of one schema.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

#: Schemas this writer emits.  Both are read by
#: ``gpuwm.prepared_single_domain_forecast``'s ``hrrr`` entries; changing
#: one here without changing it there is a refusal at the front door,
#: which is the intended failure mode.
SOURCE_MANIFEST_SCHEMA = "gpuwm-hrrr-native-input-manifest-v1"
PROOF_SCHEMA = "gpuwm-hrrr-native-direct-wrf-proof-v1"
EXPORT_SCHEMA = "gpuwm-native-direct-wrf-export-v3"

#: File names this writer publishes into the bundle root.
EXPERIMENT_CONFIG_NAME = "experiment.toml"
WPS_NAMELIST_NAME = "namelist.wps"
WRF_NAMELIST_NAME = "namelist.input"
SOURCE_MANIFEST_NAME = "source-input-manifest.json"
PROOF_NAME = "proof.json"


class HrrrBundleError(ValueError):
    """A bundle that cannot be published as a portable prepared case."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _write_json(path: Path, payload) -> None:
    # Create-only.  A published authority is bound by digest by the
    # stage after it; replacing one in place would silently invalidate a
    # receipt that already names the old bytes.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(path: Path, relative: str) -> dict[str, object]:
    resolved = Path(path)
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _copy_authority(source: Path, target: Path) -> Path:
    source = Path(source)
    if not source.is_file():
        raise HrrrBundleError(f"missing bundle authority: {source}")
    if target.exists():
        raise HrrrBundleError(f"refusing to replace {target}")
    shutil.copyfile(source, target)
    return target


def _cache_header(prepared_cache: Path) -> Mapping[str, object]:
    header_path = Path(prepared_cache) / "header.json"
    if not header_path.is_file():
        raise HrrrBundleError(
            f"prepared cache has no header.json: {prepared_cache}")
    header = json.loads(header_path.read_text(encoding="utf-8"))
    if not isinstance(header, dict):
        raise HrrrBundleError("prepared cache header is not an object")
    return header


def _relative_to(root: Path, path: Path, label: str) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        raise HrrrBundleError(
            f"{label} {path} is outside the bundle root {root}") from None


def publish_hrrr_prepared_bundle(
        *,
        output_root: Path,
        prepared_cache: Path,
        static_cache: Path,
        geometry_receipt: Path,
        bridge_manifest: Path,
        namelist_input: Path,
        wps_namelist: Path,
        source_manifest: Path,
        experiment_config: Path,
        source_cycle: datetime,
        source_forecast_hours: Sequence[int],
        model_forcing_hours: Sequence[int],
        preprocessing: Mapping[str, object],
        source_identity: Mapping[str, object],
        physics_profile: str | None,
        expert_acknowledgements: Sequence[str] = (),
        static_receipt: Path | None = None,
        domain_spec: Path | None = None,
        namelist_extension_invariant: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write the portable authorities and return the pins to bind them by.

    The return value is the whole handoff: three digests
    (``proof_sha256``, ``source_manifest_sha256``,
    ``prepared_content_sha256``) and the two paths a forecast stage has
    to pass beside them.  Those are exactly
    ``preflight_prepared_forecast``'s pinned arguments, so a caller
    never has to hash a file by hand to run the case it just prepared.

    ``experiment_config`` must already be inside the bundle and must be
    the document the preparation published for itself
    (:func:`gpuwm.experiment_document.publish_experiment_document`).
    Requiring it rather than rendering it here keeps the rendering with
    the only process that holds the tables -- the preparer -- and keeps
    this writer to the one job of binding files it can hash.
    """

    from gpuwm.experiment import load_experiment
    from gpuwm.prepared_single_domain_forecast import (
        _resolved_wrf_direct_contract_sha256,
        single_domain_physics_selection,
        validate_single_domain_physics_profile,
    )

    root = Path(output_root).resolve()
    if not root.is_dir():
        raise HrrrBundleError(f"bundle root does not exist: {root}")
    cache_path = Path(prepared_cache).resolve()
    header = _cache_header(cache_path)

    config_path = Path(experiment_config).resolve()
    if config_path.parent != root:
        raise HrrrBundleError(
            f"the published experiment config must live in the bundle "
            f"root; got {config_path}")
    experiment = load_experiment(config_path)
    if len(experiment.domains) != 1:
        raise HrrrBundleError(
            "a portable HRRR bundle is single-domain; a prepared tree is "
            "the tree runner's route")

    source_hours = [int(hour) for hour in source_forecast_hours]
    forcing_hours = [int(hour) for hour in model_forcing_hours]
    if forcing_hours != list(range(len(source_hours))):
        raise HrrrBundleError(
            "model forcing offsets must be 0..N-1 of the source lead window")
    if len(forcing_hours) < 2:
        raise HrrrBundleError(
            "a prepared case needs at least two forcing frames")
    start_time = source_cycle + timedelta(hours=source_hours[0])
    if experiment.start_time != start_time:
        raise HrrrBundleError(
            f"experiment start {experiment.start_time.isoformat()} is not "
            f"cycle f{source_hours[0]:03d} ({start_time.isoformat()})")

    # ---- the authorities the front door hashes -------------------------
    wps_path = _copy_authority(wps_namelist, root / WPS_NAMELIST_NAME)
    namelist_path = _copy_authority(namelist_input, root / WRF_NAMELIST_NAME)

    files = {
        "bridge": {"name": Path(bridge_manifest).name,
                   "sha256": _sha256(bridge_manifest)},
        "source_manifest": {"name": Path(source_manifest).name,
                            "sha256": _sha256(source_manifest)},
        "experiment_config": {"name": config_path.name,
                              "sha256": _sha256(config_path)},
        "wps_namelist": {"name": wps_path.name, "sha256": _sha256(wps_path)},
        "namelist_input": {"name": namelist_path.name,
                           "sha256": _sha256(namelist_path)},
    }
    if static_receipt is not None:
        files["static_input"] = {"name": Path(static_cache).name,
                                 "sha256": _sha256(static_cache)}
        files["static_receipt"] = {"name": Path(static_receipt).name,
                                   "sha256": _sha256(static_receipt)}
    if domain_spec is not None:
        files["domain_spec"] = {"name": Path(domain_spec).name,
                                "sha256": _sha256(domain_spec)}

    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source": {
            "model": "HRRR",
            "product": "conus/wrfnat+wrfprs",
            "cycle": source_cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_hours": source_hours,
        },
        "files": files,
    }
    manifest_path = root / SOURCE_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    manifest_digest = _sha256(manifest_path)

    # ---- the physics receipt the front door recomputes -----------------
    cfg = experiment.root.run
    acknowledgements = tuple(expert_acknowledgements)
    if physics_profile is None:
        physics = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements)
    else:
        physics = validate_single_domain_physics_profile(
            physics_profile, config=cfg,
            expert_acknowledgements=acknowledgements)

    boundary_interval_seconds = 3600
    cache_relative = _relative_to(root, cache_path, "prepared cache")
    static_relative = _relative_to(root, Path(static_cache), "static cache")
    geometry_relative = _relative_to(
        root, Path(geometry_receipt), "geometry receipt")
    cache_receipt = {
        "schema": header.get("schema"),
        "status": "BUILT",
        "path": cache_relative,
        "content_sha256": header.get("content_sha256"),
        "array_count": len(header.get("arrays") or {}),
        "payload_bytes": header.get("payload_bytes"),
    }
    export_source = {
        "contract_sha256": _sha256(
            Path(__file__).resolve().parent / "wrf_direct_v461_contract.json"),
        "geometry_receipt_sha256": _sha256(geometry_receipt),
        "prepared_content_sha256": header.get("content_sha256"),
        "prepared_header_sha256": _sha256(cache_path / "header.json"),
        "resolved_physics_contract_sha256": (
            _resolved_wrf_direct_contract_sha256(int(cfg.mp_physics))),
        "static_cache_sha256": _sha256(static_cache),
    }
    proof = {
        "schema": PROOF_SCHEMA,
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        # HRRR's two lead vocabularies, both named, never conflated:
        # absolute NOAA leads of one cycle, and the model-relative
        # forcing offsets the cache and exporter use.
        "source_cycle": source_cycle.isoformat(),
        "source_forecast_hours": source_hours,
        "model_start_time": start_time.isoformat(),
        "model_forcing_hours": forcing_hours,
        "forcing_hours": forcing_hours,
        "forcing_times": [
            (start_time + timedelta(hours=hour)).isoformat()
            for hour in forcing_hours],
        "boundary_interval_seconds": boundary_interval_seconds,
        "input_manifest_sha256": manifest_digest,
        "decoder_sha256": files["bridge"]["sha256"],
        "preprocessing": dict(preprocessing),
        "preprocessing_receipt_sha256": hashlib.sha256(
            _canonical(dict(preprocessing)).encode("utf-8")).hexdigest(),
        # The ingest digests the prepared cache's own identity carries.
        # Recorded here so the reader can refuse a cache decoded by a
        # different ingest than the proof names.
        "source_sha256": dict(source_identity.get("source_sha256") or {}),
        "source_inputs": {
            "manifest_schema": SOURCE_MANIFEST_SCHEMA,
            "manifest_sha256": manifest_digest,
            "files": files,
        },
        "initialization_artifacts": {
            "source_manifest": _artifact(manifest_path, SOURCE_MANIFEST_NAME),
            "static_cache": _artifact(Path(static_cache), static_relative),
            "geometry_receipt": _artifact(
                Path(geometry_receipt), geometry_relative),
            "prepared_cache": {
                "path": cache_relative,
                "content_sha256": header.get("content_sha256"),
                "payload_bytes": header.get("payload_bytes"),
            },
            "wrf_files": {},
        },
        "prepared_cache": cache_receipt,
        "physics": physics,
        "export": {
            "schema": EXPORT_SCHEMA,
            "status": "READY",
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_interval_seconds,
            "dimensions": {"nx": int(cfg.nx), "ny": int(cfg.ny),
                           "nz": int(cfg.nz)},
            "valid_time": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
            "source": export_source,
            "physics": physics,
        },
    }
    if namelist_extension_invariant is not None:
        proof["namelist_extension_invariant"] = dict(
            namelist_extension_invariant)
    proof_path = root / PROOF_NAME
    _write_json(proof_path, proof)

    return {
        "schema": "gpuwm-hrrr-portable-bundle-handoff-v1",
        "prepared_root": str(root),
        "proof": str(proof_path),
        "proof_sha256": _sha256(proof_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_digest,
        "prepared_content_sha256": header.get("content_sha256"),
        "experiment_config": str(config_path),
        "wps_namelist": str(wps_path),
        "namelist_input": str(namelist_path),
        "physics_profile": physics_profile,
    }


__all__ = [
    "EXPERIMENT_CONFIG_NAME",
    "EXPORT_SCHEMA",
    "HrrrBundleError",
    "PROOF_NAME",
    "PROOF_SCHEMA",
    "SOURCE_MANIFEST_NAME",
    "SOURCE_MANIFEST_SCHEMA",
    "WPS_NAMELIST_NAME",
    "WRF_NAMELIST_NAME",
    "publish_hrrr_prepared_bundle",
]
