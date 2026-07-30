#!/usr/bin/env python3
"""Fail-closed launcher for the 12 h real74 NSSL-2 d01-d04 run.

The checked-in TOML is a portable launch template.  This controller verifies
an independently supplied SHA-256 input manifest against the registered CPU
500 m authority, materializes an effective config, runs gpuwm's CPU and
measured GPU-allocation preflights, and then launches one supervised trajectory
with restart recovery disabled.  A successful run is not accepted until its
exact 64-frame calendar and NSSL history fields pass on-disk checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = REPOSITORY_ROOT / "configs" / "real74_nssl2_500m.toml"

INPUT_MANIFEST_SCHEMA = "gpuwm.real74-nssl2-500m-input-manifest/v1"
LAUNCH_MANIFEST_SCHEMA = "gpuwm.real74-nssl2-500m-launch/v1"
PREFLIGHT_SCHEMA = "gpuwm.real74-nssl2-500m-preflight/v1"
COMPLETION_SCHEMA = "gpuwm.real74-nssl2-500m-completion/v1"
SUPERVISOR_SERVICE = "gpuwm-real74-nssl2-500m"
SUPERVISOR_CONFIG = Path(
    f"/etc/supervisor/conf.d/{SUPERVISOR_SERVICE}.conf")

# The placeholder the committed template carries, replaced by the node's
# real input root at materialization.  Spelled through the portable
# case-data root rather than one machine's absolute path, so the template
# also LOADS unmaterialized wherever that root points.
TEMPLATE_INPUT_ROOT = (
    "${GPUWM_CASE_DATA_ROOT}/WRF_1974_500M_CPU_AUTHORITY")
SOURCE_WPS = "config/namelist.wps"
SOURCE_GEO_D04 = "WPS_run/geo_em.d04.nc"
SOURCE_D04_OROGRAPHY = "real_run/met_em.d04.1974-04-03_12_00_00.nc"
RUNTIME_FORCING_FILES = ("WPS_run/era5_19740403.grb",)
WPS_ONLY_FORCING_FILES = (
    "WPS_run/era5_19740403_msl_z_supplement.grb",
)
SOURCE_FILES = (
    *RUNTIME_FORCING_FILES,
    *WPS_ONLY_FORCING_FILES,
    "WPS_run/Vtable.ERA5_CDO",
    SOURCE_WPS,
    SOURCE_GEO_D04,
    "real_run/met_em.d01.1974-04-03_12_00_00.nc",
    "real_run/met_em.d02.1974-04-03_12_00_00.nc",
    "real_run/met_em.d03.1974-04-03_12_00_00.nc",
    SOURCE_D04_OROGRAPHY,
)
SOURCE_TREES = ("static/WPS_GEOG",)

# Registered from the exact CPU Thompson 500 m authority run
# wrf_cpu_thompson_500m_19740403_20260721_123510 (reference-runs
# archive, outside this repository).  The physics differs; these are
# geometry/forcing/real-data authorities only.
KNOWN_AUTHORITY_SHA256 = {
    SOURCE_WPS:
        "0d8e865e7ac2e331d3cd83788a51658842fc2e7e98bcb60d11762db61ae25f0f",
    SOURCE_GEO_D04:
        "41182978e2bbe78b1d4aca81b357ead21cd358f1037cefd6f66f405bb24e3217",
    "WPS_run/era5_19740403.grb":
        "9501fd6807edaf1f93bc7ee5ab6424d712dde094f9015f733946bc574e6a0340",
    "WPS_run/era5_19740403_msl_z_supplement.grb":
        "ceae040f1be8cc4585250f8d14fe65c9dc574e1742fcf3671f493113610c07b1",
    "WPS_run/Vtable.ERA5_CDO":
        "64282b5b35ac7302e274f764327923080883f164f4e605ef06529d1baef6620e",
    "real_run/met_em.d01.1974-04-03_12_00_00.nc":
        "baeae9cdf9737eb2c947500fdab8be4e03bc22d05ada45450f984fad75b507e9",
    "real_run/met_em.d02.1974-04-03_12_00_00.nc":
        "d90deb78d0e1fa35d618b49e4875f124dfd349d4c23fe3e23c552f5b5ffec03c",
    "real_run/met_em.d03.1974-04-03_12_00_00.nc":
        "18c9909283364e03b311a42433abf628eeacb53353b2dd29f202c98e1debdec5",
    SOURCE_D04_OROGRAPHY:
        "d9c18f6ac449e2662b0c788f28bef25650ed34e45542161fec9b6441fc2e3f2d",
}
KNOWN_AUTHORITY_BYTES = {"WPS_run/Vtable.ERA5_CDO": 4256}

RUN_SECONDS = 12 * 60 * 60
EXPECTED_OUTPUT_BYTES = 51 * 1024**3
PUBLICATION_TEMP_BYTES = 8 * 1024**3
POST_RUN_RESERVE_BYTES = 16 * 1024**3
MINIMUM_FREE_BYTES = (
    EXPECTED_OUTPUT_BYTES + PUBLICATION_TEMP_BYTES + POST_RUN_RESERVE_BYTES
)
START_TIME = datetime(1974, 4, 3, 12, 0, 0)
EXPECTED_TOPOLOGY = (
    # id, parent, i, j, ratio, time ratio, nx, ny, dx, dt, history
    (1, 0, 1, 1, 1, 1, 250, 200, 12000.0, 60.0, 3600.0),
    (2, 1, 63, 51, 4, 4, 500, 400, 3000.0, 15.0, 3600.0),
    (3, 2, 167, 117, 3, 3, 501, 501, 1000.0, 5.0, 3600.0),
    (4, 3, 151, 151, 2, 2, 400, 400, 500.0, 2.5, 1800.0),
)

REQUIRED_HISTORY_VARIABLES = (
    "Times", "U", "V", "W", "T", "P", "PB", "PH", "PHB",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
    "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
    "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "XLAT", "XLONG",
    "T2",
)
NONNEGATIVE_HISTORY_VARIABLES = frozenset({
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
    "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
    "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC",
})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_SUPERVISOR_PATH = re.compile(r"/[A-Za-z0-9_./-]+")
_FATAL_LOG = re.compile(
    r"(?:traceback \(most recent call last\)|cudaerror|cuda error|"
    r"illegal (?:memory )?address|out of memory|\b(?:nan|inf)\b)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"required directory is missing: {path}")
    digest = hashlib.sha256()
    count = 0
    total = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        row = {
            "path": child.relative_to(path).as_posix(),
            "bytes": child.stat().st_size,
            "sha256": sha256_file(child),
        }
        descriptor = json.dumps(
            row, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        count += 1
        total += int(row["bytes"])
    if count == 0:
        raise ValueError(f"required directory is empty: {path}")
    return {
        "path": str(path), "file_count": count, "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _validated_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or value != parsed.as_posix():
        raise ValueError(f"{field} must be normalized and relative: {value!r}")
    return value


def _expected_identity(raw: object, *, field: str) -> tuple[int, str]:
    if not isinstance(raw, dict) or set(raw) != {"bytes", "sha256"}:
        raise ValueError(f"{field} must contain exactly bytes and sha256")
    size, digest = raw["bytes"], raw["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{field}.bytes must be a nonnegative integer")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field}.sha256 must be lowercase SHA-256")
    return size, digest


def verify_input_manifest(input_root: Path, manifest_path: Path
                          ) -> dict[str, object]:
    """Verify the complete required source inventory against external pins."""
    input_root = input_root.resolve()
    manifest_path = manifest_path.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root is missing: {input_root}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("input manifest must be a JSON object")
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError(
            f"input manifest schema must be {INPUT_MANIFEST_SCHEMA!r}")
    files = manifest.get("files")
    trees = manifest.get("trees")
    if not isinstance(files, dict) or not isinstance(trees, dict):
        raise ValueError("input manifest files and trees must be objects")
    normalized_files = {
        _validated_relative_path(key, field="files key"): value
        for key, value in files.items()
    }
    normalized_trees = {
        _validated_relative_path(key, field="trees key"): value
        for key, value in trees.items()
    }
    if set(normalized_files) != set(SOURCE_FILES):
        raise ValueError(
            "input manifest file inventory mismatch: expected "
            f"{sorted(SOURCE_FILES)}, got {sorted(normalized_files)}")
    if set(normalized_trees) != set(SOURCE_TREES):
        raise ValueError(
            "input manifest tree inventory mismatch: expected "
            f"{sorted(SOURCE_TREES)}, got {sorted(normalized_trees)}")

    verified_files = []
    for relative in SOURCE_FILES:
        expected_size, expected_sha = _expected_identity(
            normalized_files[relative], field=f"files[{relative!r}]")
        registered_sha = KNOWN_AUTHORITY_SHA256.get(relative)
        if registered_sha is not None and expected_sha != registered_sha:
            raise ValueError(
                f"manifest SHA-256 for {relative} does not match registered "
                f"CPU 500 m authority pin {registered_sha}")
        registered_bytes = KNOWN_AUTHORITY_BYTES.get(relative)
        if registered_bytes is not None and expected_size != registered_bytes:
            raise ValueError(
                f"manifest byte count for {relative} does not match "
                f"registered CPU 500 m authority size {registered_bytes}")
        actual = file_identity(input_root / relative)
        if (actual["bytes"], actual["sha256"]) != (
                expected_size, expected_sha):
            raise ValueError(
                f"input identity mismatch for {relative}: expected "
                f"bytes={expected_size} sha256={expected_sha}, got "
                f"bytes={actual['bytes']} sha256={actual['sha256']}")
        verified_files.append(actual)

    verified_trees = []
    for relative in SOURCE_TREES:
        raw = normalized_trees[relative]
        if not isinstance(raw, dict) or set(raw) != {
                "file_count", "bytes", "sha256"}:
            raise ValueError(
                f"trees[{relative!r}] must contain exactly file_count, "
                "bytes, and sha256")
        expected_size, expected_sha = _expected_identity(
            {"bytes": raw["bytes"], "sha256": raw["sha256"]},
            field=f"trees[{relative!r}]")
        expected_count = raw["file_count"]
        if (isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or expected_count < 1):
            raise ValueError(
                f"trees[{relative!r}].file_count must be positive")
        actual = tree_identity(input_root / relative)
        if (actual["file_count"], actual["bytes"], actual["sha256"]) != (
                expected_count, expected_size, expected_sha):
            raise ValueError(
                f"input tree identity mismatch for {relative}: expected "
                f"count={expected_count} bytes={expected_size} "
                f"sha256={expected_sha}, got count={actual['file_count']} "
                f"bytes={actual['bytes']} sha256={actual['sha256']}")
        verified_trees.append(actual)
    return {
        "manifest": file_identity(manifest_path),
        "files": verified_files,
        "trees": verified_trees,
    }


def validate_authority_assets(input_root: Path) -> dict[str, object]:
    """Prove the hash-pinned CPU assets also express the requested geometry."""
    import netCDF4
    import numpy as np

    from gpuwm.namelist_import import parse_namelist

    wps = parse_namelist(input_root / SOURCE_WPS)
    share, geogrid = wps.get("share", {}), wps.get("geogrid", {})
    if tuple(int(value) for value in share.get("max_dom", ())) != (4,):
        raise ValueError("authoritative WPS max_dom is not exactly 4")
    expected_vectors = {
        "parent_id": (1, 1, 2, 3),
        "parent_grid_ratio": (1, 4, 3, 2),
        "i_parent_start": (1, 63, 167, 151),
        "j_parent_start": (1, 51, 117, 151),
        "e_we": (251, 501, 502, 401),
        "e_sn": (201, 401, 502, 401),
    }
    observed_vectors = {}
    for key, expected in expected_vectors.items():
        observed = tuple(int(value) for value in geogrid.get(key, ()))
        if observed != expected:
            raise ValueError(
                f"authoritative WPS {key} mismatch: expected {expected}, "
                f"got {observed}")
        observed_vectors[key] = list(observed)
    for key in ("dx", "dy"):
        observed = tuple(float(value) for value in geogrid.get(key, ()))
        if observed != (12000.0,):
            raise ValueError(
                f"authoritative WPS {key} must be (12000.0,), got {observed}")
        observed_vectors[key] = list(observed)

    orography = {}
    expected_shapes = {
        1: (200, 250), 2: (400, 500),
        3: (501, 501), 4: (400, 400),
    }
    for grid_id, expected_shape in expected_shapes.items():
        path = input_root / (
            f"real_run/met_em.d{grid_id:02d}.1974-04-03_12_00_00.nc")
        with netCDF4.Dataset(path) as dataset:
            if "SOILHGT" not in dataset.variables:
                raise ValueError(f"CPU authority has no SOILHGT: {path}")
            field = np.asarray(dataset.variables["SOILHGT"][0])
        if field.shape != expected_shape or not np.isfinite(field).all():
            raise ValueError(
                f"CPU authority d{grid_id:02d} SOILHGT must be finite "
                f"{expected_shape}, got {field.shape}")
        orography[f"d{grid_id:02d}"] = {
            "shape": list(field.shape),
            "minimum_m": float(field.min()),
            "maximum_m": float(field.max()),
        }
    return {
        "kind": "exact-cpu-500m-wps-real-assets",
        "wps": observed_vectors,
        "orography": orography,
        "derived_from_333m": False,
    }


def materialize_config(template: Path, destination: Path,
                       input_root: Path) -> None:
    text = template.read_text(encoding="utf-8")
    if text.count(TEMPLATE_INPUT_ROOT) != 8:
        raise ValueError(
            "NSSL config template input-root inventory changed; expected 8 "
            f"occurrences, got {text.count(TEMPLATE_INPUT_ROOT)}")
    text = text.replace(TEMPLATE_INPUT_ROOT, input_root.resolve().as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def validate_topology(exp) -> list[dict[str, object]]:
    if exp.name != "real74_nssl2_4dom_500m":
        raise ValueError(f"wrong experiment name: {exp.name!r}")
    if exp.start_time != START_TIME or exp.run_seconds != RUN_SECONDS:
        raise ValueError(
            f"run window mismatch: {exp.start_time}, {exp.run_seconds}")
    if exp.restart_interval_s != 0.0:
        raise ValueError("main validation run must have restarts disabled")
    if len(exp.domains) != 4:
        raise ValueError("NSSL validation requires exactly d01 through d04")
    observed = []
    expected_epssm = (0.5, 0.1, 0.1, 0.1)
    for dc, expected, epssm in zip(
            exp.domains, EXPECTED_TOPOLOGY, expected_epssm, strict=True):
        row = (
            dc.grid_id, dc.parent_id, dc.i_parent_start, dc.j_parent_start,
            dc.parent_grid_ratio, dc.parent_time_step_ratio,
            dc.run.nx, dc.run.ny, float(dc.run.dx), float(dc.run.dt),
            float(dc.history_interval_s),
        )
        if row != expected:
            raise ValueError(
                f"d{dc.grid_id:02d} topology mismatch: expected {expected}, "
                f"got {row}")
        if dc.run.mp_physics != 18:
            raise ValueError(
                f"d{dc.grid_id:02d} mp_physics is {dc.run.mp_physics}, not 18")
        if dc.run.epssm != epssm:
            raise ValueError(
                f"d{dc.grid_id:02d} epssm is {dc.run.epssm}, not {epssm}")
        if not dc.run.moist_cq:
            raise ValueError(
                f"d{dc.grid_id:02d} moist_cq must be true for WRF parity")
        observed.append({
            "grid_id": dc.grid_id, "parent_id": dc.parent_id,
            "i_parent_start": dc.i_parent_start,
            "j_parent_start": dc.j_parent_start,
            "parent_grid_ratio": dc.parent_grid_ratio,
            "parent_time_step_ratio": dc.parent_time_step_ratio,
            "mass_shape": [dc.run.ny, dc.run.nx],
            "dx_m": float(dc.run.dx), "dt_s": float(dc.run.dt),
            "history_interval_s": float(dc.history_interval_s),
            "epssm": dc.run.epssm,
            "moist_cq": dc.run.moist_cq,
            "mp_physics": dc.run.mp_physics,
        })
    return observed


def repository_identity(repo: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True,
            text=True,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1")
    if status:
        raise RuntimeError(
            "production source worktree must be clean before binding:\n"
            + status)
    return {"commit": head, "tree": tree, "git_status": "clean"}


def _require_identity(expected: Mapping[str, object], *, label: str) -> None:
    path = Path(str(expected.get("path", "")))
    actual = file_identity(path)
    if actual != dict(expected):
        raise ValueError(
            f"bound {label} identity changed: expected {dict(expected)}, "
            f"got {actual}")


def require_current_binding(effective_config: Path,
                            binding: Mapping[str, object]) -> None:
    """Recheck every launch-time identity after the long trajectory."""
    if repository_identity(REPOSITORY_ROOT) != binding.get("source"):
        raise ValueError("bound git source identity changed")
    for key in ("runner", "config_template", "effective_config"):
        expected = binding.get(key)
        if not isinstance(expected, dict):
            raise ValueError(f"launch binding has no {key} identity")
        _require_identity(expected, label=key)
    if Path(str(binding["effective_config"]["path"])) != (
            effective_config.resolve()):
        raise ValueError("effective config path disagrees with launch binding")

    inputs = binding.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("launch binding has no input identity inventory")
    manifest = inputs.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("launch binding has no input-manifest identity")
    _require_identity(manifest, label="input manifest")
    bound_files = inputs.get("files")
    bound_trees = inputs.get("trees")
    if (not isinstance(bound_files, list)
            or len(bound_files) != len(SOURCE_FILES)
            or not isinstance(bound_trees, list)
            or len(bound_trees) != len(SOURCE_TREES)):
        raise ValueError("launch binding input inventory is incomplete")
    for index, expected in enumerate(bound_files):
        if not isinstance(expected, dict):
            raise ValueError("bound input file identity is malformed")
        _require_identity(expected, label=f"input file {index}")
    for index, expected in enumerate(bound_trees):
        if not isinstance(expected, dict):
            raise ValueError("bound input tree identity is malformed")
        path = Path(str(expected.get("path", "")))
        actual = tree_identity(path)
        if actual != expected:
            raise ValueError(
                f"bound input tree {index} identity changed: expected "
                f"{expected}, got {actual}")

    authority = binding.get("cpu_500m_authority")
    if (not isinstance(authority, dict)
            or authority.get("derived_from_333m") is not False):
        raise ValueError("launch binding lacks exact CPU 500 m authority")
    comparison = binding.get("comparison_preregistration")
    if not isinstance(comparison, dict):
        raise ValueError("launch binding lacks comparison preregistration")
    registration = comparison.get("registration")
    policy = comparison.get("policy")
    if not isinstance(registration, dict) or not isinstance(policy, dict):
        raise ValueError("comparison preregistration identities are malformed")
    actual_comparison = validate_comparison_preregistration(
        Path(str(registration.get("path", ""))),
        Path(str(policy.get("path", ""))), binding["source"])
    if actual_comparison != comparison:
        raise ValueError("comparison preregistration changed after launch")


def require_disk_headroom(path: Path, minimum_free_bytes: int =
                          MINIMUM_FREE_BYTES) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum_free_bytes:
        raise RuntimeError(
            f"disk preflight failed: free={free} bytes is below mandatory "
            f"minimum={minimum_free_bytes} bytes at {path.resolve()} "
            f"(outputs={EXPECTED_OUTPUT_BYTES}, publication_temp="
            f"{PUBLICATION_TEMP_BYTES}, post_run_reserve="
            f"{POST_RUN_RESERVE_BYTES})")
    return free


def require_input_preflight(report) -> None:
    """Require all CPU input checks and the complete 12 h forcing window."""
    report.raise_for_failures()
    if report.run_ceiling_seconds < RUN_SECONDS:
        raise ValueError(
            f"forcing ceiling {report.run_ceiling_seconds} s is shorter than "
            f"required {RUN_SECONDS} s")


def _outside_repository(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(
        f"run directory must be outside the git worktree: {path.resolve()}")


def validate_comparison_preregistration(
        registration_path: Path, policy_path: Path,
        source: Mapping[str, object]) -> dict[str, object]:
    """Bind the pre-GPU CPU inventory, evaluator, and explicit policy."""
    from tools import compare_real74_nssl2_500m as comparator

    policy, policy_identity = comparator.load_policy(policy_path.resolve())
    registration = comparator.load_json(
        registration_path.resolve(), "comparison registration")
    binding = registration.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("comparison registration has no binding object")
    comparator.validate_registration(
        registration, policy_identity=policy_identity,
        source_identity=source, cpu_inventory=binding.get("cpu_outputs", []))
    if registration.get("policy_status") not in {None, policy["status"]}:
        raise ValueError("comparison registration policy status disagrees")
    return {
        "registration": file_identity(registration_path.resolve()),
        "registration_binding_sha256": registration["binding_sha256"],
        "registration_created_at_utc": registration["created_at_utc"],
        "policy": policy_identity,
    }


def prepare_launch(input_root: Path, input_manifest: Path, run_dir: Path,
                   comparison_registration: Path,
                   comparison_policy: Path,
                   ) -> tuple[Path, dict[str, object]]:
    """Bind source, exact CPU 500 m inputs, config, and CPU preflight."""
    from gpuwm.case_data import load_experiment_case
    from gpuwm.ingest.preflight import preflight_report

    input_root = input_root.resolve()
    run_dir = run_dir.resolve()
    _outside_repository(run_dir)
    free_bytes = require_disk_headroom(run_dir)
    source = repository_identity(REPOSITORY_ROOT)
    comparison = validate_comparison_preregistration(
        comparison_registration, comparison_policy, source)
    verified_inputs = verify_input_manifest(input_root, input_manifest)
    authority = validate_authority_assets(input_root)

    metadata = run_dir / "metadata"
    effective_config = metadata / "real74_nssl2_500m.effective.toml"
    materialize_config(CONFIG_TEMPLATE, effective_config, input_root)

    exp, data = load_experiment_case(effective_config)
    topology = validate_topology(exp)
    input_preflight = preflight_report(exp, data)
    require_input_preflight(input_preflight)

    binding = {
        "source": source,
        "runner": file_identity(Path(__file__)),
        "config_template": file_identity(CONFIG_TEMPLATE),
        "effective_config": file_identity(effective_config),
        "inputs": verified_inputs,
        "runtime_forcing_files": list(RUNTIME_FORCING_FILES),
        "wps_only_forcing_files": list(WPS_ONLY_FORCING_FILES),
        "cpu_500m_authority": authority,
        "comparison_preregistration": comparison,
        "input_catalog_sha256": input_preflight.catalog.fingerprint,
        "forcing_valid_times": [
            value.isoformat() for value in input_preflight.catalog.valid_times
        ],
        "forcing_run_ceiling_seconds": input_preflight.run_ceiling_seconds,
        "topology": topology,
        "requested": {
            "start_time": START_TIME.isoformat(),
            "run_seconds": RUN_SECONDS,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "expected_output_bytes": EXPECTED_OUTPUT_BYTES,
            "publication_temp_bytes": PUBLICATION_TEMP_BYTES,
            "post_run_reserve_bytes": POST_RUN_RESERVE_BYTES,
            "restart_interval_s": 0,
            "expected_output_counts": {"d01": 13, "d02": 13,
                                       "d03": 13, "d04": 25},
        },
    }
    launch_path = metadata / "launch-manifest.json"
    if launch_path.exists():
        with launch_path.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
        if prior.get("schema") != LAUNCH_MANIFEST_SCHEMA:
            raise ValueError("existing launch manifest has the wrong schema")
        if prior.get("binding_sha256") != stable_hash(prior.get("binding")):
            raise ValueError("existing launch manifest binding digest is invalid")
        if prior.get("binding") != binding:
            raise ValueError(
                "existing launch manifest disagrees with current source, "
                "CPU 500 m inputs, or configuration")
    else:
        atomic_json(launch_path, {
            "schema": LAUNCH_MANIFEST_SCHEMA,
            "created_at_utc": utc_now(),
            "initial_disk_free_bytes": free_bytes,
            "binding_sha256": stable_hash(binding),
            "binding": binding,
        })
    return effective_config, binding


def gpu_allocation_preflight(effective_config: Path, run_dir: Path,
                             gpu_uuid: str | None) -> dict[str, object]:
    """Run exclusive-device and measured allocation gates in a fresh process."""
    from gpuwm.supervisor import preflight_exclusive_gpu, select_gpu

    gpu = select_gpu(gpu_uuid)
    preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()})
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    command = [
        sys.executable, "-m", "gpuwm.cli", "check", str(effective_config),
        "--alloc", "--json",
    ]
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=env, capture_output=True,
        text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "measured GPU allocation/input preflight failed closed with "
            f"status {completed.returncode}:\n{completed.stderr}\n"
            f"{completed.stdout}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "allocation preflight did not emit valid JSON") from exc
    gates = report.get("gates")
    if (not isinstance(gates, dict) or not gates
            or any(value is not True for value in gates.values())
            or not isinstance(report.get("alloc"), dict)):
        raise RuntimeError(
            f"allocation report lacks a complete passing measurement: {report}")
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "completed_at_utc": utc_now(),
        "effective_config": file_identity(effective_config),
        "gpu": {"uuid": gpu.uuid, "driver_version": gpu.driver_version,
                "name": gpu.name},
        "allocation": report,
        "stderr": completed.stderr,
        "disk_free_bytes": require_disk_headroom(run_dir),
    }
    atomic_json(run_dir / "metadata" / "preflight-receipt.json", receipt)
    return receipt


def expected_output_names(exp) -> dict[int, tuple[str, ...]]:
    from gpuwm.io.wrfout import wrfout_filename

    result = {}
    for dc in exp.domains:
        interval = int(dc.history_interval_s)
        result[dc.grid_id] = tuple(
            wrfout_filename(
                exp.start_time + timedelta(seconds=offset), dc.grid_id)
            for offset in range(0, int(exp.run_seconds) + 1, interval)
        )
    return result


def _decoded_time(variable) -> str:
    import netCDF4
    import numpy as np

    value = netCDF4.chartostring(np.asarray(variable[:]))
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ValueError(f"Times must contain one record, got {flattened.size}")
    return str(flattened[0])


def verify_one_wrfout(path: Path, dc, expected_time: datetime,
                      expected_title: str) -> dict[str, object]:
    import netCDF4
    import numpy as np

    expected_dims = {
        "west_east": dc.run.nx, "south_north": dc.run.ny,
        "bottom_top": dc.run.nz, "west_east_stag": dc.run.nx + 1,
        "south_north_stag": dc.run.ny + 1,
        "bottom_top_stag": dc.run.nz + 1, "DateStrLen": 19,
    }
    extrema: dict[str, dict[str, float]] = {}
    try:
        with netCDF4.Dataset(path) as dataset:
            if int(getattr(dataset, "GPUWM_WRITE_COMPLETE", 0)) != 1:
                raise ValueError("publication completion marker is absent")
            if str(getattr(dataset, "TITLE", "")) != expected_title:
                raise ValueError("TITLE does not match the NSSL validation")
            for name, size in expected_dims.items():
                if name not in dataset.dimensions:
                    raise ValueError(f"missing dimension {name}")
                if len(dataset.dimensions[name]) != size:
                    raise ValueError(
                        f"dimension {name} is {len(dataset.dimensions[name])}, "
                        f"expected {size}")
            missing = [name for name in REQUIRED_HISTORY_VARIABLES
                       if name not in dataset.variables]
            if missing:
                raise ValueError(f"missing required NSSL variables {missing}")
            expected_times = expected_time.strftime("%Y-%m-%d_%H:%M:%S")
            actual_times = _decoded_time(dataset.variables["Times"])
            if actual_times != expected_times:
                raise ValueError(
                    f"Times={actual_times!r}, expected {expected_times!r}")
            for name in REQUIRED_HISTORY_VARIABLES:
                if name == "Times":
                    continue
                value = np.ma.asarray(dataset.variables[name][:])
                if np.ma.is_masked(value) and np.any(np.ma.getmaskarray(value)):
                    raise ValueError(f"{name} contains masked values")
                array = np.asarray(value)
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} contains non-finite values")
                minimum = float(array.min()) if array.size else math.nan
                maximum = float(array.max()) if array.size else math.nan
                if (name in NONNEGATIVE_HISTORY_VARIABLES
                        and minimum < -1.0e-12):
                    raise ValueError(f"{name} has negative minimum {minimum}")
                extrema[name] = {"minimum": minimum, "maximum": maximum}
    except Exception as exc:
        raise ValueError(f"wrfout verification failed for {path}: {exc}") from exc
    identity = file_identity(path)
    identity["extrema"] = extrema
    return identity


def _reject_fatal_logs(run_dir: Path) -> list[dict[str, object]]:
    logs = []
    for path in sorted(run_dir.glob("worker-*.stderr.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = _FATAL_LOG.search(text)
        if match is not None:
            raise ValueError(
                f"fatal/numerical-health marker {match.group(0)!r} in {path}")
        logs.append(file_identity(path))
    if not logs:
        raise ValueError("no supervised worker stderr log was published")
    return logs


def verify_completed_run(effective_config: Path, binding: Mapping[str, object],
                         run_dir: Path) -> dict[str, object]:
    from gpuwm.case_data import load_experiment_case
    from gpuwm.supervisor import HEARTBEAT_SCHEMA

    require_current_binding(effective_config, binding)
    exp, data = load_experiment_case(effective_config)
    validate_topology(exp)
    if (run_dir / "failure-capsule.json").exists():
        raise ValueError("failure-capsule.json exists; run cannot be accepted")
    restart_files = sorted(run_dir.glob("gpuwmrst_*"))
    if restart_files:
        raise ValueError(
            f"no-restart run published checkpoint files: {restart_files}")
    heartbeat_path = run_dir / "run-progress.json"
    with heartbeat_path.open("r", encoding="utf-8") as stream:
        heartbeat = json.load(stream)
    if (heartbeat.get("schema") != HEARTBEAT_SCHEMA
            or heartbeat.get("status") != "complete"
            or heartbeat.get("model_elapsed_seconds") != RUN_SECONDS
            or heartbeat.get("last_checkpoint") is not None
            or heartbeat.get("config_digest") != sha256_file(effective_config)):
        raise ValueError(f"completion heartbeat failed closed: {heartbeat}")
    logs = _reject_fatal_logs(run_dir)

    expected = expected_output_names(exp)
    actual = tuple(sorted(path.name for path in run_dir.glob("wrfout_d*")))
    expected_flat = tuple(sorted(
        name for names in expected.values() for name in names))
    if actual != expected_flat:
        raise ValueError(
            f"wrfout calendar mismatch: expected {expected_flat}, got {actual}")
    outputs = {}
    for dc in exp.domains:
        interval = int(dc.history_interval_s)
        records = []
        for index, name in enumerate(expected[dc.grid_id]):
            records.append(verify_one_wrfout(
                run_dir / name, dc,
                exp.start_time + timedelta(seconds=index * interval),
                data.output_title))
        outputs[f"d{dc.grid_id:02d}"] = records

    final_free_bytes = shutil.disk_usage(run_dir).free
    if final_free_bytes < POST_RUN_RESERVE_BYTES:
        raise ValueError(
            f"post-run disk reserve violated: free={final_free_bytes}, "
            f"required={POST_RUN_RESERVE_BYTES}")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "completed_at_utc": utc_now(),
        "binding_sha256": stable_hash(binding),
        "effective_config": file_identity(effective_config),
        "heartbeat": heartbeat,
        "stderr_logs": logs,
        "output_counts": {key: len(value) for key, value in outputs.items()},
        "outputs": outputs,
        "disk_free_bytes": final_free_bytes,
    }
    atomic_json(run_dir / "completion.json", completion)
    return completion


def _assert_unstarted(run_dir: Path) -> None:
    forbidden = [
        *run_dir.glob("wrfout_d*"), *run_dir.glob("gpuwmrst_*"),
        *run_dir.glob("worker-*.stdout.log"),
        *run_dir.glob("worker-*.stderr.log"),
    ]
    forbidden.extend(path for path in (
        run_dir / "run-progress.json", run_dir / "failure-capsule.json",
        run_dir / "completion.json",
    ) if path.exists())
    if forbidden:
        raise ValueError(
            "run directory already contains trajectory artifacts; use a new "
            f"directory instead of mixing/overwriting: {forbidden}")


def _supervisor_path(path: Path, *, label: str) -> str:
    resolved = str(path.resolve())
    if _SAFE_SUPERVISOR_PATH.fullmatch(resolved) is None:
        raise ValueError(
            f"{label} is not a shell-free Supervisor-safe absolute path: "
            f"{resolved!r}")
    return resolved


def render_supervisor_config(*, input_root: Path, input_manifest: Path,
                             comparison_registration: Path,
                             comparison_policy: Path,
                             run_dir: Path, gpu_uuid: str) -> str:
    """Render one fixed-argv, non-restarting system Supervisor program."""
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid) is None:
        raise ValueError(f"invalid GPU UUID for Supervisor: {gpu_uuid!r}")
    python = _supervisor_path(Path(sys.executable), label="Python executable")
    runner = _supervisor_path(Path(__file__), label="runner")
    repo = _supervisor_path(REPOSITORY_ROOT, label="repository")
    input_root_text = _supervisor_path(input_root, label="input root")
    manifest = _supervisor_path(input_manifest, label="input manifest")
    registration = _supervisor_path(
        comparison_registration, label="comparison registration")
    policy = _supervisor_path(comparison_policy, label="comparison policy")
    run = _supervisor_path(run_dir, label="run directory")
    stdout = _supervisor_path(
        run_dir / "controller.stdout.log", label="controller stdout")
    stderr = _supervisor_path(
        run_dir / "controller.stderr.log", label="controller stderr")
    command = " ".join((
        python, runner, "run", "--run-dir", run,
        "--input-root", input_root_text,
        "--input-manifest", manifest,
        "--comparison-registration", registration,
        "--comparison-policy", policy,
        "--gpu-uuid", gpu_uuid,
    ))
    return (
        f"[program:{SUPERVISOR_SERVICE}]\n"
        f"command={command}\n"
        f"directory={repo}\n"
        f'environment=PYTHONPATH="{repo}",PYTHONUNBUFFERED="1"\n'
        "autostart=false\n"
        "autorestart=false\n"
        "startsecs=1\n"
        "startretries=0\n"
        "stopsignal=TERM\n"
        "stopasgroup=true\n"
        "killasgroup=true\n"
        "stopwaitsecs=30\n"
        f"stdout_logfile={stdout}\n"
        "stdout_logfile_maxbytes=0\n"
        "stdout_logfile_backups=0\n"
        f"stderr_logfile={stderr}\n"
        "stderr_logfile_maxbytes=0\n"
        "stderr_logfile_backups=0\n"
    )


def _supervisorctl(*arguments: str, allow_missing: bool = False
                   ) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["supervisorctl", *arguments], check=False,
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(
            f"system Supervisor is unavailable for {arguments}: {exc}") from exc
    if completed.returncode != 0 and not allow_missing:
        raise RuntimeError(
            f"supervisorctl {' '.join(arguments)} failed with "
            f"{completed.returncode}: {completed.stdout}\n{completed.stderr}")
    return completed


def launch_under_supervisor(*, input_root: Path, input_manifest: Path,
                            comparison_registration: Path,
                            comparison_policy: Path,
                            run_dir: Path, gpu_uuid: str | None,
                            config_path: Path = SUPERVISOR_CONFIG
                            ) -> dict[str, object]:
    """Install and start the long run under system Supervisor."""
    from gpuwm.supervisor import select_gpu

    gpu = select_gpu(gpu_uuid)
    desired = render_supervisor_config(
        input_root=input_root, input_manifest=input_manifest,
        comparison_registration=comparison_registration,
        comparison_policy=comparison_policy,
        run_dir=run_dir, gpu_uuid=gpu.uuid)
    status = _supervisorctl("status", SUPERVISOR_SERVICE, allow_missing=True)
    status_text = f"{status.stdout}\n{status.stderr}"
    if any(token in status_text for token in ("RUNNING", "STARTING", "STOPPING")):
        raise RuntimeError(
            f"Supervisor service {SUPERVISOR_SERVICE} is already active: "
            f"{status_text.strip()}")
    if config_path.exists():
        current = config_path.read_text(encoding="utf-8")
        if current != desired:
            raise RuntimeError(
                f"existing Supervisor config differs; refusing overwrite: "
                f"{config_path}")
    else:
        atomic_text(config_path, desired)
    reread = _supervisorctl("reread")
    update = _supervisorctl("update")
    started = _supervisorctl("start", SUPERVISOR_SERVICE)
    final_status = _supervisorctl("status", SUPERVISOR_SERVICE)
    if "RUNNING" not in final_status.stdout:
        raise RuntimeError(
            f"Supervisor did not report RUNNING: {final_status.stdout}\n"
            f"{final_status.stderr}")
    return {
        "service": SUPERVISOR_SERVICE,
        "config": file_identity(config_path),
        "gpu_uuid": gpu.uuid,
        "reread": reread.stdout.strip(),
        "update": update.stdout.strip(),
        "start": started.stdout.strip(),
        "status": final_status.stdout.strip(),
        "stdout_log": str((run_dir / "controller.stdout.log").resolve()),
        "stderr_log": str((run_dir / "controller.stderr.log").resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "run", "verify", "supervise"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--comparison-registration", type=Path)
    parser.add_argument("--comparison-policy", type=Path)
    parser.add_argument("--gpu-uuid", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.command == "verify":
        launch_path = run_dir / "metadata" / "launch-manifest.json"
        with launch_path.open("r", encoding="utf-8") as stream:
            launch = json.load(stream)
        if launch.get("schema") != LAUNCH_MANIFEST_SCHEMA:
            raise ValueError("launch manifest missing or invalid")
        binding = launch["binding"]
        if launch.get("binding_sha256") != stable_hash(binding):
            raise ValueError("launch manifest binding digest is invalid")
        effective = run_dir / "metadata" / "real74_nssl2_500m.effective.toml"
        completion = verify_completed_run(effective, binding, run_dir)
        print(json.dumps({
            "status": "PASS", "schema": completion["schema"],
            "output_counts": completion["output_counts"],
        }, sort_keys=True))
        return 0

    if (args.input_root is None or args.input_manifest is None
            or args.comparison_registration is None
            or args.comparison_policy is None):
        raise ValueError(
            "preflight/run/supervise require --input-root and "
            "--input-manifest plus --comparison-registration and "
            "--comparison-policy")
    effective, binding = prepare_launch(
        args.input_root, args.input_manifest, run_dir,
        args.comparison_registration, args.comparison_policy)
    if args.command in ("run", "supervise"):
        _assert_unstarted(run_dir)
    if args.command == "supervise":
        launched = launch_under_supervisor(
            input_root=args.input_root.resolve(),
            input_manifest=args.input_manifest.resolve(), run_dir=run_dir,
            comparison_registration=args.comparison_registration.resolve(),
            comparison_policy=args.comparison_policy.resolve(),
            gpu_uuid=args.gpu_uuid)
        print(json.dumps({
            "status": "RUNNING", "service": launched["service"],
            "supervisor_status": launched["status"],
            "stdout_log": launched["stdout_log"],
            "stderr_log": launched["stderr_log"],
        }, sort_keys=True))
        return 0
    receipt = gpu_allocation_preflight(effective, run_dir, args.gpu_uuid)
    if args.command == "preflight":
        print(json.dumps({
            "status": "PASS", "schema": receipt["schema"],
            "effective_config": str(effective),
        }, sort_keys=True))
        return 0

    from gpuwm.supervisor import supervise_experiment

    gpu_uuid = str(receipt["gpu"]["uuid"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    result = supervise_experiment(
        effective, run_dir, restart=None, gpu_uuid=gpu_uuid,
        max_restarts=0, allow_shared_gpu=False)
    if result.attempts != 1 or result.heartbeat.last_checkpoint is not None:
        raise RuntimeError(
            "no-restart trajectory returned an invalid supervisor result")
    completion = verify_completed_run(effective, binding, run_dir)
    print(json.dumps({
        "status": "PASS", "schema": completion["schema"],
        "output_counts": completion["output_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
