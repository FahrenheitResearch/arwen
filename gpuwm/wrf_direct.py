"""Direct gpuwm prepared-cache -> WRF ``wrfinput``/``wrfbdy`` export.

This module is intentionally independent of WPS and WRF executables.  It
consumes the launch-ready native gpuwm preparation cache, the matching native
static cache, and a small geometry receipt.  A frozen WRF-v4.6.1 declaration
contract supplies only NetCDF metadata; all gridded state is derived from the
native inputs.

The supported slice is a single specified projected domain
(lambert/mercator/polar; MAP_PROJ and MAP_PROJ_CHAR derive from the
geometry receipt) with an explicit validated hybrid eta coordinate and a
registry-resolved fixed physics profile.  Stock-WRF acceptance is proven for
Lambert; mercator/polar exports are oracle-gated but not yet stock-WRF-gated.
Unsupported cache/configuration combinations fail closed.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence
import uuid

import netCDF4
import numpy as np

from gpuwm.core import constants as _model_constants
from gpuwm.ingest.prepared_cache import (
    compare_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_identity_refusal,
    undelayed_identity_defaults,
)
from gpuwm.static.lambert import LambertGrid, grids_from_projection_config
from gpuwm.static.projection import (MAP_PROJ_CHARS, WRF_MAP_PROJ_CODES,
                                     projection_class)
from gpuwm.vertical_contract import (
    validate_coordinate_shapes,
    validate_explicit_eta_grid,
)
from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory


_CONTRACT_PATH = Path(__file__).with_name("wrf_direct_v461_contract.json")
# WRF's rvovrd (share/module_model_constants.F:41 = 1.6083624), not the 1.608
# this used to carry.  module_initialize_real.F writes T as
# theta*(1.+rvovrd*qv) - 300 on every column it produces, so an exporter that
# truncates rvovrd hands WRF an initial condition WRF would not have written:
# 2.3e-04 relative, ~1.1e-03 K of theta_m at qv = 0.01, and the same offset
# again through alpha.  No WRF code path in the stack gpuwm mirrors uses 1.608
# (the only 1.608 in the tree is CAM-ZM's own virtual temperature, which is not
# in any gpuwm template).
_RVOVRD = _model_constants.RVOVRD
_RD = 287.0
_RCP = 2.0 / 7.0
_P0 = 100000.0
_THETA_OFFSET_K = 300.0
_HRRR_SOIL_NODE_DEPTHS_M = np.array(
    [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0],
    dtype=np.float64,
)
_NOAH_LAYER_MIDPOINTS_M = np.array(
    [0.05, 0.25, 0.70, 1.50], dtype=np.float64)
_NOAH_LAYER_THICKNESS_M = np.array(
    [0.10, 0.30, 0.60, 1.00], dtype=np.float64)
_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})


class StockWrfExportUnsupported(ValueError):
    """The prepared state is outside what the stock-WRF export represents.

    Every refusal in this module that means "gpuwm can run this, but the
    unchanged-WRF file set cannot represent it" raises this rather than a
    bare ``ValueError``: mixed-domain microphysics, a microphysics
    selector with no WRF Registry inventory, and the physics slice the
    profile-free compatibility branch requires.

    It exists so a caller can tell export-representability apart from
    everything else that can go wrong here.  A caller whose product IS
    the export re-raises it and is unchanged.  A caller that is merely
    PREPARING a forecast catches it, records the refusal, and proceeds --
    which physics a domain tree may run is a decision the registry and
    the acknowledgement gate own, and answering it a second time from a
    downstream file-format contract is how a registry-reachable,
    explicitly acknowledged, profile-bound MYNN tree came to be
    unrunnable from any shipped GFS front door.

    ``unsupported`` carries the named per-selector deltas
    (``{selector: (observed, required)}``) when there are any, so the
    refusal a user reads names what to change.
    """

    def __init__(self, message: str, *,
                 unsupported: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.unsupported = dict(unsupported or {})


HIERARCHY_EXPORT_SCHEMA = "gpuwm-native-direct-wrf-hierarchy-export-v1"

#: The three states a hierarchy export can be in.  ``READY`` is a real
#: manifest with a ``files`` inventory; the other two are documents that
#: occupy the same slot and answer the same question -- did this
#: preparation publish unchanged-WRF inputs, and if not, why -- so no
#: consumer has to distinguish "absent" from "refused" by guessing.
STOCK_WRF_EXPORT_STATUSES = ("READY", "NOT_REQUESTED", "REFUSED")


def stock_wrf_export_not_requested() -> dict[str, object]:
    """The export slot for a preparation that never asked for an export."""

    return {
        "schema": HIERARCHY_EXPORT_SCHEMA,
        "status": "NOT_REQUESTED",
        "reason": "the caller did not request a stock-WRF export",
    }


def stock_wrf_export_refused(
        error: StockWrfExportUnsupported) -> dict[str, object]:
    """The export slot for an export refused on representability.

    The message is the export gate's own, unchanged -- it already names
    the selector deltas -- and ``unsupported`` repeats them as data so a
    receipt reader does not have to parse prose.  Tuples are flattened to
    lists so this document equals its own JSON round trip.
    """

    return {
        "schema": HIERARCHY_EXPORT_SCHEMA,
        "status": "REFUSED",
        "reason": str(error),
        "unsupported": {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in error.unsupported.items()
        },
    }


def _direct_export_soil_geometry(
        sf_surface_physics: int, num_soil_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the WRF ZS/DZS geometry selected by the land-surface scheme."""

    scheme = int(sf_surface_physics)
    count = int(num_soil_layers)
    if scheme in {2, 4}:
        if count != 4:
            raise ValueError(
                f"Noah/Noah-MP direct export requires four soil layers, "
                f"got sf_surface_physics={scheme}, num_soil_layers={count}")
        return _NOAH_LAYER_MIDPOINTS_M, _NOAH_LAYER_THICKNESS_M
    if scheme == 3:
        from gpuwm.ingest.ruc_soil import ruc_soil_depths

        depths, thicknesses = ruc_soil_depths(count)
        return (
            np.asarray(depths, dtype=np.float64),
            np.asarray(thicknesses, dtype=np.float64),
        )
    raise ValueError(
        f"sf_surface_physics={scheme} has no source-driven direct-export "
        "soil geometry; gpuwm/ingest/ruc_soil.py:"
        "preprocess_land_surface_soil implements selectors 2, 3, and 4")


@dataclass(frozen=True)
class PreparedDomainArtifacts:
    """Native preparation artifacts for one WRF domain.

    A hierarchy export accepts one of these records for every domain in the
    namelist-derived :class:`~gpuwm.experiment.ExperimentConfig`.  Paths are
    explicit and digest-verified by the normal prepared/static/geometry
    contracts before any final output directory is published.
    """

    grid_id: int
    prepared_cache: Path
    static_cache: Path
    geometry_receipt: Path

    def __post_init__(self):
        if (isinstance(self.grid_id, bool)
                or not isinstance(self.grid_id, int)
                or self.grid_id < 1
                or self.grid_id > 99):
            raise ValueError(
                f"grid_id must be an integer in [1, 99], got {self.grid_id!r}")
        for name in ("prepared_cache", "static_cache", "geometry_receipt"):
            object.__setattr__(self, name, Path(getattr(self, name)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b";")
    digest.update(_canonical(list(value.shape)).encode("ascii"))
    digest.update(b";")
    raw = memoryview(value).cast("B")
    block_bytes = 8 * 1024 * 1024
    for start in range(0, len(raw), block_bytes):
        digest.update(raw[start:start + block_bytes])
    return digest.hexdigest()


def _load_contract() -> Mapping[str, object]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "gpuwm-wrf-direct-contract-bundle-v1":
        raise RuntimeError("unsupported bundled WRF direct-export contract")
    return payload


_PACKAGE_FIELD_METADATA = {
    "QHAIL": ("Hail mixing ratio", "kg kg-1"),
    "QNDROP": ("Droplet number mixing ratio", "# kg-1"),
    "QNICE": ("Ice Number concentration", "# kg-1"),
    "QNRAIN": ("Rain Number concentration", "# kg(-1)"),
    "QNSNOW": ("Snow Number concentration", "# kg(-1)"),
    "QNGRAUPEL": ("Graupel Number concentration", "# kg(-1)"),
    "QNHAIL": ("Hail Number concentration", "# kg(-1)"),
    "QNCCN": ("CCN Number concentration", "# kg(-1)"),
    "QVGRAUPEL": ("Graupel Particle Volume", "m(3) kg(-1)"),
    "QVHAIL": ("Hail Particle Volume", "m(3) kg(-1)"),
}


def _physics_contract_bundle(
        contract_bundle: Mapping[str, object], mp_physics: int,
) -> dict[str, object]:
    """Extend the frozen WSM6 NetCDF declarations for one WRF package.

    WRF v4.6.1's Registry marks scheme-specific mass and scalar members with
    input and boundary interpolation flags.  The frozen direct contract came
    from a WSM6 file and therefore cannot contain Thompson/Morrison number
    moments or NSSL-2 hail, number, CCN, and volume moments.  Clone the
    corresponding mass-scalar declaration mechanics while replacing the exact
    Registry name, description, and units.  Existing contract objects remain
    immutable and each returned bundle is independently hashable.
    """

    inventory = stock_wrf_physics_inventory(mp_physics)
    result = copy.deepcopy(dict(contract_bundle))
    input_contract = result["wrfinput"]
    bdy_contract = result["wrfbdy"]
    extension_names = [
        field.netcdf_name
        for field in inventory.wrfinput_fields
        if field.netcdf_name in _PACKAGE_FIELD_METADATA
    ]
    input_names = {item["name"] for item in input_contract["variables"]}
    input_prototype = next(
        item for item in input_contract["variables"]
        if item["name"] == "QCLOUD")
    for name in extension_names:
        if name in input_names:
            continue
        description, units = _PACKAGE_FIELD_METADATA[name]
        spec = copy.deepcopy(input_prototype)
        spec["name"] = name
        spec["attributes"]["description"] = description
        spec["attributes"]["units"] = units
        input_contract["variables"].append(spec)

    bdy_names = {item["name"] for item in bdy_contract["variables"]}
    bdy_prototypes = [
        item for item in bdy_contract["variables"]
        if item["name"].startswith("QCLOUD_B")
    ]
    for name in extension_names:
        description, units = _PACKAGE_FIELD_METADATA[name]
        for prototype in bdy_prototypes:
            suffix = prototype["name"].removeprefix("QCLOUD")
            variable_name = name + suffix
            if variable_name in bdy_names:
                continue
            spec = copy.deepcopy(prototype)
            spec["name"] = variable_name
            spec["attributes"]["description"] = description
            spec["attributes"]["units"] = units
            bdy_contract["variables"].append(spec)
    return result


def _contract_payload_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(contract).encode("utf-8")).hexdigest()


def _load_static_geometry_receipt(
        receipt_path: Path, static_path: Path, *,
        expected_geometry: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    """Verify the static receipt/cache binding and optional target geometry."""

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "gpuwm-native-static-direct-v1"
            or payload.get("status") != "PASS"):
        raise ValueError("unrecognized or non-PASS native static receipt")
    static_sha256 = _sha256(static_path)
    expected_cache = {
        "path": static_path.name,
        "bytes": static_path.stat().st_size,
        "sha256": static_sha256,
    }
    if payload.get("cache") != expected_cache:
        raise ValueError("native static receipt does not bind the supplied cache")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("native static receipt lacks a geometry object")
    if expected_geometry is not None and geometry != dict(expected_geometry):
        keys = sorted(set(geometry) | set(expected_geometry))
        drift = {
            name: {
                "receipt": geometry.get(name),
                "namelist": expected_geometry.get(name),
            }
            for name in keys
            if geometry.get(name) != expected_geometry.get(name)
        }
        raise ValueError(
            f"native static receipt geometry differs from namelist: {drift}")
    return geometry, static_sha256


def _acquire_output_lock(output: Path) -> tuple[int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.native-export.lock")
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another native export or unresolved lock owns {output}: "
            f"{lock_path}") from exc
    os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
    return descriptor, lock_path


def _release_output_lock(descriptor: int, lock_path: Path) -> None:
    os.close(descriptor)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _prepare_output_target(output: Path, *, overwrite: bool) -> Path:
    """Recover an interrupted backup and validate overwrite policy."""

    backup = output.with_name(output.name + ".previous-valid")
    if backup.exists() and output.exists():
        raise RuntimeError(
            f"both current and previous-valid exports exist for {output}; "
            "refusing to choose or delete either automatically")
    if backup.exists():
        os.replace(backup, output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    return backup


def _publish_staging(staging: Path, output: Path, backup: Path) -> None:
    """Publish a validated tree while retaining rollback on replacement."""

    if not output.exists():
        os.replace(staging, output)
        return
    if backup.exists():
        raise RuntimeError(f"refusing to replace existing backup {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    # At this point a complete, validated new export is live.  A process crash
    # before cleanup leaves the former valid tree recoverable at `backup`.
    shutil.rmtree(backup)


@contextmanager
def _output_publication(output: Path, *, overwrite: bool):
    descriptor, lock_path = _acquire_output_lock(output)
    try:
        yield _prepare_output_target(output, overwrite=overwrite)
    finally:
        _release_output_lock(descriptor, lock_path)


def _restore_attribute(value, type_spec: Mapping[str, object]):
    kind = type_spec["kind"]
    if kind == "str":
        return str(value)
    if kind == "bytes":
        return str(value).encode("latin1")
    if kind != "numeric":
        raise RuntimeError(f"unsupported frozen attribute kind {kind!r}")
    array = np.asarray(value, dtype=np.dtype(type_spec["dtype"]))
    shape = tuple(type_spec["shape"])
    if shape:
        return array.reshape(shape)
    return array.reshape(())[()]


class PreparedCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.header_path = self.root / "header.json"
        self.header = json.loads(self.header_path.read_text(encoding="utf-8"))
        if self.header.get("schema") != "gpuwm-prepared-real-cache-v1":
            raise ValueError("prepared cache has an unsupported schema")
        if self.header.get("status") != "READY":
            raise ValueError("prepared cache is not READY")
        self._arrays = self.header["arrays"]
        basis = {
            "schema": self.header["schema"],
            "identity": self.header["identity"],
            "metadata": self.header["metadata"],
            "arrays": self._arrays,
            "payload_bytes": self.header["payload_bytes"],
        }
        observed_content = hashlib.sha256(
            _canonical(basis).encode("utf-8")).hexdigest()
        if observed_content != self.header.get("content_sha256"):
            raise ValueError("prepared cache header content digest mismatch")
        self._loaded: dict[str, np.ndarray] = {}
        self._verified_files: set[Path] = set()

    def array(self, name: str) -> np.ndarray:
        if name in self._loaded:
            return self._loaded[name]
        try:
            spec = self._arrays[name]
        except KeyError as exc:
            raise KeyError(f"prepared cache lacks {name!r}") from exc
        path = self.root / spec["file"]
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"prepared cache declaration drift for {name}")
        if path not in self._verified_files:
            if _array_sha256(value) != spec["sha256"]:
                raise ValueError(f"prepared cache digest drift for {name}")
            self._verified_files.add(path)
        self._loaded[name] = value
        return value


def load_domain_artifacts_manifest(path: str | Path) \
        -> tuple[PreparedDomainArtifacts, ...]:
    """Load the strict, relocatable per-domain artifact inventory.

    Relative paths resolve against the manifest directory.  The manifest is
    intentionally small: it names already provenance-bound native artifacts;
    their content hashes remain authoritative and are rechecked during export.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"schema", "domains"}:
        raise ValueError(
            "domain-artifact manifest keys must be exactly schema/domains")
    if payload["schema"] != "gpuwm-native-domain-artifacts-v1":
        raise ValueError("unsupported domain-artifact manifest schema")
    domains = payload["domains"]
    if not isinstance(domains, list) or not domains:
        raise ValueError("domain-artifact manifest requires a non-empty domains list")
    root = manifest_path.parent
    resolved_root = root.resolve()
    result = []
    expected_keys = {
        "grid_id", "prepared_cache", "static_cache", "geometry_receipt",
    }
    for index, item in enumerate(domains):
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError(
                f"domain-artifact entry {index} keys must be exactly "
                f"{sorted(expected_keys)}")

        def resolved(name: str) -> Path:
            value = item[name]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"domain-artifact entry {index} {name} must be a path string")
            candidate = Path(value)
            if candidate.is_absolute():
                raise ValueError(
                    f"domain-artifact entry {index} {name} must be a "
                    "relocatable relative path")
            target = (root / candidate).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"domain-artifact entry {index} {name} escapes the "
                    "manifest directory") from exc
            return target

        result.append(PreparedDomainArtifacts(
            grid_id=item["grid_id"],
            prepared_cache=resolved("prepared_cache"),
            static_cache=resolved("static_cache"),
            geometry_receipt=resolved("geometry_receipt"),
        ))
    identifiers = [item.grid_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            f"domain-artifact manifest has duplicate grid ids {identifiers}")
    return tuple(result)


def write_domain_artifacts_manifest(
        path: str | Path,
        artifacts: Sequence[PreparedDomainArtifacts],
) -> dict[str, object]:
    """Atomically join published per-domain artifacts with relative paths."""

    manifest_path = Path(path)
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite domain-artifact manifest {manifest_path}")
    records = tuple(sorted(artifacts, key=lambda item: item.grid_id))
    identifiers = [item.grid_id for item in records]
    if identifiers != list(range(1, len(records) + 1)):
        raise ValueError(
            "domain artifacts must contain contiguous grid ids beginning at "
            f"d01, got {identifiers}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = manifest_path.parent.resolve()

    def relative(target: Path, *, directory: bool) -> str:
        resolved = Path(target).resolve()
        if directory:
            if not resolved.is_dir() or not (resolved / "header.json").is_file():
                raise FileNotFoundError(
                    f"prepared cache is missing or incomplete: {resolved}")
        elif not resolved.is_file():
            raise FileNotFoundError(f"domain artifact is missing: {resolved}")
        try:
            value = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"domain artifact {resolved} is outside manifest root {root}") \
                from exc
        return value.as_posix()

    payload = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": artifact.grid_id,
            "prepared_cache": relative(
                artifact.prepared_cache, directory=True),
            "static_cache": relative(artifact.static_cache, directory=False),
            "geometry_receipt": relative(
                artifact.geometry_receipt, directory=False),
        } for artifact in records],
    }
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:12]}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        loaded = load_domain_artifacts_manifest(manifest_path)
        expected = tuple(
            (item.grid_id, item.prepared_cache.resolve(),
             item.static_cache.resolve(), item.geometry_receipt.resolve())
            for item in records)
        observed = tuple(
            (item.grid_id, item.prepared_cache.resolve(),
             item.static_cache.resolve(), item.geometry_receipt.resolve())
            for item in loaded)
        if observed != expected:
            raise RuntimeError("domain-artifact manifest round-trip drift")
    except BaseException:
        temporary.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return payload


def _interp_nodes(nodes: np.ndarray) -> np.ndarray:
    output = []
    for depth in _NOAH_LAYER_MIDPOINTS_M:
        lower = int(np.searchsorted(_HRRR_SOIL_NODE_DEPTHS_M, depth) - 1)
        weight = (
            (depth - _HRRR_SOIL_NODE_DEPTHS_M[lower])
            / (_HRRR_SOIL_NODE_DEPTHS_M[lower + 1]
               - _HRRR_SOIL_NODE_DEPTHS_M[lower])
        )
        output.append(
            nodes[lower] + weight * (nodes[lower + 1] - nodes[lower]))
    return np.stack(output)


def _wrf_noah_landuse(raw_lu: np.ndarray, *, iswater: int = 17,
                      islake: int = 21) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_lu, dtype=np.int32)
    lake_mask = raw == int(islake)
    mapped = np.where(lake_mask, int(iswater), raw).astype(np.int32)
    return mapped, lake_mask


def _moist_pressure(cache: PreparedCache) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the native/WRF moist hydrostatic pressure recurrence."""
    qv = np.asarray(cache.array("state/qv"), dtype=np.float64)
    mub = np.asarray(cache.array("base/mub"), dtype=np.float64)
    mup = np.asarray(cache.array("state/mup"), dtype=np.float64)
    c1f = np.asarray(cache.array("coord/c1f"), dtype=np.float64)
    c2f = np.asarray(cache.array("coord/c2f"), dtype=np.float64)
    rdn = np.asarray(cache.array("coord/rdn"), dtype=np.float64)
    rdnw = np.asarray(cache.array("coord/rdnw"), dtype=np.float64)
    perturbation = np.empty_like(qv, dtype=np.float64)
    nz = qv.shape[0]
    cq = 1.0 / (1.0 + qv[-1])
    load = qv[-1] * cq
    perturbation[-1] = (
        -0.5 * (c1f[nz] * mup
                + load * (c1f[nz] * mub + c2f[nz]))
        / rdnw[nz - 1] / cq
    )
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qv[k] + qv[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        perturbation[k] = (
            perturbation[k + 1]
            - (c1f[kw] * mup
               + load * (c1f[kw] * mub + c2f[kw]))
            / cq / rdn[kw]
        )
    total = np.asarray(cache.array("base/pb"), dtype=np.float64) + perturbation
    if not np.isfinite(total).all() or np.any(total <= 0.0):
        raise ValueError("direct exporter diagnosed invalid moist pressure")
    return perturbation, total


def _date_text(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    if value.microsecond:
        raise ValueError("valid time must resolve to a whole second")
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def _dimensions(contract: Mapping[str, object], *, nx: int, ny: int,
                nz: int, num_soil_layers: int = 4
                ) -> dict[str, int | None]:
    overrides = {
        "Time": None,
        "west_east": nx,
        "west_east_stag": nx + 1,
        "south_north": ny,
        "south_north_stag": ny + 1,
        "bottom_top": nz,
        "bottom_top_stag": nz + 1,
        "soil_layers_stag": num_soil_layers,
    }
    return {
        item["name"]: overrides.get(item["name"], int(item["length"]))
        for item in contract["dimensions"]
    }


def _prepared_vertical_contract(
    cache: PreparedCache, *, nx: int, ny: int, nz: int
) -> float:
    """Validate serialized vertical dimensions and return the bound p_top."""

    coord_shapes = {
        name.removeprefix("coord/"): spec["shape"]
        for name, spec in cache._arrays.items()
        if name.startswith("coord/")
    }
    validate_coordinate_shapes(
        coord_shapes, nz=nz, context="direct-WRF prepared cache")

    expected_shapes = {
        "state/u": (nz, ny, nx + 1),
        "state/v": (nz, ny + 1, nx),
        "state/w": (nz + 1, ny, nx),
        "state/php": (nz + 1, ny, nx),
        "state/thp": (nz, ny, nx),
        "state/qv": (nz, ny, nx),
        "state/qc": (nz, ny, nx),
        "state/qr": (nz, ny, nx),
        "state/qi": (nz, ny, nx),
        "state/qs": (nz, ny, nx),
        "state/qg": (nz, ny, nx),
        "state/mup": (ny, nx),
        "base/mub": (ny, nx),
        "base/pb": (nz, ny, nx),
        "base/alb": (nz, ny, nx),
        "base/thb": (nz, ny, nx),
        "base/phb": (nz + 1, ny, nx),
        "base/terrain_z": (ny, nx),
    }
    missing = sorted(set(expected_shapes) - set(cache._arrays))
    if missing:
        raise ValueError(
            f"direct-WRF prepared cache lacks required arrays {missing}")
    drift = {
        name: {
            "actual": tuple(cache._arrays[name]["shape"]),
            "expected": shape,
        }
        for name, shape in expected_shapes.items()
        if tuple(cache._arrays[name]["shape"]) != shape
    }
    if drift:
        raise ValueError(
            f"direct-WRF prepared vertical/state shape drift: {drift}")

    metadata = cache.header.get("metadata", {})
    try:
        base_p_top = float(metadata["base_scalars"]["p_top"])
        coord_p_top = float(metadata["coord_scalars"]["p_top"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "direct-WRF prepared cache lacks numeric bound p_top metadata"
        ) from exc
    if coord_p_top != base_p_top:
        raise ValueError(
            "direct-WRF prepared coordinate/base p_top values differ")
    validate_explicit_eta_grid(
        cache.array("coord/znw"),
        nz=nz,
        p_top=base_p_top,
        context="direct-WRF prepared cache",
    )
    return base_p_top


def _global_updates(*, valid_time: datetime, nx: int, ny: int, nz: int,
                    dx: float, dy: float, dt: float,
                    geometry: Mapping[str, object],
                    physics_selection: Mapping[str, object] | None = None,
                    ) -> dict[str, object]:
    stamp = _date_text(valid_time)
    updates = {
        # WRF's input gate recognizes versioned initial data from the TITLE
        # token itself.  Preserve an honest producer label while retaining
        # the required V4.x-compatible marker.
        "TITLE": " OUTPUT FROM GPUWM NATIVE WRF V4.6.1-COMPAT DIRECT EXPORTER",
        "START_DATE": stamp,
        "SIMULATION_START_DATE": stamp,
        "WEST-EAST_GRID_DIMENSION": nx + 1,
        "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
        "BOTTOM-TOP_GRID_DIMENSION": nz + 1,
        "WEST-EAST_PATCH_END_UNSTAG": nx,
        "WEST-EAST_PATCH_END_STAG": nx + 1,
        "SOUTH-NORTH_PATCH_END_UNSTAG": ny,
        "SOUTH-NORTH_PATCH_END_STAG": ny + 1,
        "BOTTOM-TOP_PATCH_END_UNSTAG": nz,
        "BOTTOM-TOP_PATCH_END_STAG": nz + 1,
        "GRID_ID": 1,
        "PARENT_ID": 0,
        "I_PARENT_START": 1,
        "J_PARENT_START": 1,
        "PARENT_GRID_RATIO": 1,
        "DX": dx,
        "DY": dy,
        "DT": dt,
        "GHG_INPUT": 0,
        "CEN_LAT": geometry["center_lat"],
        "CEN_LON": geometry["center_lon"],
        "TRUELAT1": geometry["truelat1"],
        "TRUELAT2": geometry["truelat2"],
        "MOAD_CEN_LAT": geometry["ref_lat"],
        "STAND_LON": geometry["stand_lon"],
        # Projection identity: WRF convention 1=lambert, 2=polar
        # stereographic, 3=mercator (a receipt without map_proj is a
        # legacy Lambert one).
        "MAP_PROJ": WRF_MAP_PROJ_CODES[
            str(geometry.get("map_proj", "lambert"))],
        "MAP_PROJ_CHAR": MAP_PROJ_CHARS[
            str(geometry.get("map_proj", "lambert"))],
        "GMT": (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0),
        "JULYR": valid_time.year,
        "JULDAY": valid_time.timetuple().tm_yday,
    }
    if physics_selection is not None:
        selectors = physics_selection.get("selectors")
        if not isinstance(selectors, Mapping):
            raise ValueError(
                "front-door physics selection lacks selector provenance")
        global_names = {
            "mp_physics": "MP_PHYSICS",
            "ra_lw_physics": "RA_LW_PHYSICS",
            "ra_sw_physics": "RA_SW_PHYSICS",
            "sf_sfclay_physics": "SF_SFCLAY_PHYSICS",
            "sf_surface_physics": "SF_SURFACE_PHYSICS",
            "bl_pbl_physics": "BL_PBL_PHYSICS",
            "cu_physics": "CU_PHYSICS",
        }
        missing = sorted(set(global_names) - set(selectors))
        if missing:
            raise ValueError(
                f"front-door physics selection lacks selectors {missing}")
        updates.update({
            global_name: selectors[selector]
            for selector, global_name in global_names.items()
        })
    return updates


def _create_dataset(path: Path, contract: Mapping[str, object],
                    dimensions: Mapping[str, int | None],
                    global_updates: Mapping[str, object]) -> netCDF4.Dataset:
    dataset = netCDF4.Dataset(path, "w", format=str(contract["format"]))
    try:
        for name, length in dimensions.items():
            dataset.createDimension(name, length)
        for name, value in contract["global_attributes"].items():
            restored = _restore_attribute(
                global_updates.get(name, value),
                contract["global_attribute_types"][name],
            )
            dataset.setncattr(name, restored)
        for spec in contract["variables"]:
            compression = spec["compression"]
            kwargs = {
                "zlib": bool(compression["zlib"]),
                "shuffle": bool(compression["shuffle"]),
                "complevel": int(compression["complevel"]),
                "fletcher32": bool(compression["fletcher32"]),
                "endian": spec["endian"],
            }
            if isinstance(spec["chunking"], list):
                kwargs["chunksizes"] = tuple(
                    min(int(chunk), int(dimensions[dim]))
                    if dimensions[dim] is not None else int(chunk)
                    for chunk, dim in zip(spec["chunking"], spec["dimensions"])
                )
            if spec["has_fill_value"]:
                kwargs["fill_value"] = spec["fill_value"]
            variable = dataset.createVariable(
                spec["name"], np.dtype(spec["dtype"]),
                tuple(spec["dimensions"]), **kwargs)
            if spec["attributes"]:
                variable.setncatts({
                    name: _restore_attribute(
                        value, spec["attribute_types"][name])
                    for name, value in spec["attributes"].items()
                })
        return dataset
    except BaseException:
        dataset.close()
        raise


def _shape_without_time(variable: netCDF4.Variable) -> tuple[int, ...]:
    return tuple(len(variable.group().dimensions[name])
                 for name in variable.dimensions if name != "Time")


def _write_time_value(variable: netCDF4.Variable, value) -> None:
    _write_time_value_at(variable, value, 0)


def _write_time_value_at(variable: netCDF4.Variable, value,
                         time_index: int) -> None:
    if not variable.dimensions or variable.dimensions[0] != "Time":
        raise ValueError(f"{variable.name} does not begin with Time")
    if time_index < 0:
        raise ValueError("time index must be non-negative")
    target_shape = _shape_without_time(variable)
    array = np.asarray(value)
    if (array.ndim == len(target_shape) + 1
            and array.shape[0] == 1
            and tuple(array.shape[1:]) == target_shape):
        array = array[0]
    if tuple(array.shape) != target_shape:
        raise ValueError(
            f"{variable.name} source shape {array.shape} != {target_shape}")
    variable[(time_index,) + (slice(None),) * len(target_shape)] = array


def _write_timestamp(variable: netCDF4.Variable, text: str) -> None:
    _write_timestamp_at(variable, text, 0)


def _write_timestamp_at(variable: netCDF4.Variable, text: str,
                        time_index: int) -> None:
    encoded = np.frombuffer(text.encode("ascii"), dtype="S1")
    _write_time_value_at(variable, encoded, time_index)


def _resolved_prototype_value(
    variable: netCDF4.Variable, prototype
):
    """Keep matching frozen prototypes; resize only all-zero scaffolds.

    The v4.6.1 metadata contract was captured from the authoritative
    49-level file.  Several diagnostic base-profile placeholders are all
    zero but therefore carry a prototype shape of 49.  Their semantic value
    is shape-independent zero, so derive that shape from the live NetCDF
    dimensions.  A nonzero mismatched prototype remains a hard error.
    """

    target_shape = _shape_without_time(variable)
    array = np.asarray(prototype)
    compatible = (
        tuple(array.shape) == target_shape
        or (array.ndim == len(target_shape) + 1
            and array.shape[0] == 1
            and tuple(array.shape[1:]) == target_shape)
    )
    if compatible:
        # Preserve the exact accepted 49-level write path and conversion.
        return prototype
    if array.size and np.all(array == 0):
        return np.zeros(target_shape, dtype=variable.dtype)
    raise ValueError(
        f"{variable.name} has a nonzero frozen prototype shape "
        f"{array.shape} that cannot serve derived shape {target_shape}")


def _surface_fields(cache: PreparedCache, static: Mapping[str, np.ndarray],
                    month_index: int) -> dict[str, np.ndarray]:
    declared = {
        name.removeprefix("surface/")
        for name in cache._arrays
        if name.startswith("surface/")
    }
    if declared and declared != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared cache contains an incomplete canonical surface: "
            f"expected {sorted(_CANONICAL_SURFACE_FIELDS)}, "
            f"got {sorted(declared)}")
    if declared:
        result = {
            name: np.asarray(cache.array(f"surface/{name}"), dtype=np.float64)
            for name in sorted(_CANONICAL_SURFACE_FIELDS)
        }
        land = result["LANDMASK"] >= 0.5
        green = np.asarray(static["GREENFRAC"], dtype=np.float64)
        albedo = np.asarray(static["ALBEDO12M"], dtype=np.float64)
        lai12 = np.asarray(static["LAI12M"], dtype=np.float64)
        snoalb = np.asarray(static["SNOALB"], dtype=np.float64) / 100.0
        result.update({
            "VEGFRA": 100.0 * green[month_index],
            "SHDMAX": 100.0 * np.max(green, axis=0),
            "SHDMIN": 100.0 * np.min(green, axis=0),
            "SHDAVG": 100.0 * np.mean(green, axis=0),
            "ALBBCK": albedo[month_index] / 100.0,
            "LAI": lai12[month_index],
            "SNOALB": np.where(land, np.maximum(snoalb, 0.08), 0.08),
        })
        return result

    tsk = np.asarray(cache.array("met/SKINTEMP"), dtype=np.float64)
    land = np.asarray(static["LANDMASK"], dtype=np.float64) >= 0.5
    xice = np.asarray(cache.array("met/XICE"), dtype=np.float64).copy()
    xice[land] = 0.0
    xice = np.where((~land) & (xice >= 0.5), 1.0, 0.0)
    effective_land = land | (xice >= 0.5)
    soilt = _interp_nodes(np.asarray(cache.array("met/SOILT"), dtype=np.float64))
    soilw = _interp_nodes(np.asarray(cache.array("met/SOILW"), dtype=np.float64))
    soilt[:, ~land] = tsk[~land]
    soilw[:, ~land] = 1.0
    # This proof case is warm (all soil nodes > 285 K), so WRF's frozen-water
    # initialization is exactly SH2O == SMOIS.  Fail closed for colder cases
    # until the table-driven partition is exported here as well.
    #
    # EXPORT-REPRESENTABILITY, not preparation: gpuwm's own Noah surface
    # layer carries frozen soil through the forecast; only the
    # unchanged-WRF wrfinput file set has no table-driven partition to
    # say SH2O with here.  That is precisely the category
    # StockWrfExportUnsupported's docstring reserves for itself, so a
    # caller whose product is a PREPARED FORECAST can record the refusal
    # and proceed while a caller whose product IS the export still
    # fails.  As a bare ValueError it was uncatchable: a measured
    # mountainous-west domain had 1438 of 65340 columns with a soil node
    # below freezing -- every one of them land, at the DEEPEST node,
    # between 1600 m and 3274 m of terrain -- and lost a complete,
    # verified GPU hierarchy to a missing oracle file.
    if np.any(soilt < 273.15):
        raise StockWrfExportUnsupported(
            "direct WRF export does not yet support frozen-soil SH2O setup")
    tmn = np.asarray(static["TMN"], dtype=np.float64).copy()
    valid_tmn = np.isfinite(tmn) & (tmn >= 170.0) & (tmn <= 400.0)
    tmn = np.where(land & valid_tmn, tmn, tsk)
    green = np.asarray(static["GREENFRAC"], dtype=np.float64)
    albedo = np.asarray(static["ALBEDO12M"], dtype=np.float64)
    lai12 = np.asarray(static["LAI12M"], dtype=np.float64)
    snoalb = np.asarray(static["SNOALB"], dtype=np.float64) / 100.0
    snoalb = np.where(land, np.maximum(snoalb, 0.08), 0.08)
    return {
        "TSK": tsk,
        "TSLB": soilt,
        "SMOIS": soilw,
        "SH2O": soilw.copy(),
        "TMN": tmn,
        "SEAICE": xice,
        "XLAND": np.where(effective_land, 1.0, 2.0),
        "LANDMASK": effective_land.astype(np.float64),
        "VEGFRA": 100.0 * green[month_index],
        "SHDMAX": 100.0 * np.max(green, axis=0),
        "SHDMIN": 100.0 * np.min(green, axis=0),
        "SHDAVG": 100.0 * np.mean(green, axis=0),
        "ALBBCK": albedo[month_index] / 100.0,
        "LAI": lai12[month_index],
        "SNOALB": snoalb,
    }


def _wrfinput_fields(cache: PreparedCache, static: Mapping[str, np.ndarray],
                     geometry: Mapping[str, object],
                     valid_time: datetime, *,
                     p_top: float, mp_physics: int = 6,
                     sf_surface_physics: int = 2,
                     num_soil_layers: int = 4,
                     ) -> dict[str, np.ndarray]:
    u = cache.array("state/u")
    v = cache.array("state/v")
    nz, ny, nx1 = u.shape
    nx = nx1 - 1
    if v.shape != (nz, ny + 1, nx):
        raise ValueError("prepared U/V staggering is inconsistent")
    dx = float(geometry["dx_m"])
    dy = float(geometry["dy_m"])
    grid = projection_class(str(geometry.get("map_proj", "lambert")))(
        geometry["ref_lat"], geometry["ref_lon"], geometry["truelat1"],
        geometry["truelat2"], geometry["stand_lon"], dx, dy,
        nx + 1, ny + 1,
    )
    lat, lon = grid.latlon_mass()
    lat_u, lon_u = grid.latlon_u()
    lat_v, lon_v = grid.latlon_v()

    qv = np.asarray(cache.array("state/qv"), dtype=np.float64)
    theta = (np.asarray(cache.array("base/thb"), dtype=np.float64)
             + np.asarray(cache.array("state/thp"), dtype=np.float64))
    theta_m = theta * (1.0 + _RVOVRD * qv)
    pressure_perturbation, total_pressure = _moist_pressure(cache)
    alpha = (_RD * theta_m * (total_pressure / _P0) ** _RCP
             / total_pressure)
    al_perturbation = (
        alpha - np.asarray(cache.array("base/alb"), dtype=np.float64))

    psfc = np.asarray(cache.array("result/surface_pressure"), dtype=np.float64)
    t2 = np.asarray(cache.array("met/T2"), dtype=np.float64)
    u10_face = np.asarray(cache.array("met/U10"), dtype=np.float64)
    v10_face = np.asarray(cache.array("met/V10"), dtype=np.float64)
    month_index = valid_time.month - 1
    surface = _surface_fields(cache, static, month_index)
    soil_depths, soil_thicknesses = _direct_export_soil_geometry(
        sf_surface_physics, num_soil_layers)
    soil_shape_drift = {
        name: tuple(np.asarray(surface[name]).shape)
        for name in ("TSLB", "SMOIS", "SH2O")
        if tuple(np.asarray(surface[name]).shape)
        != (int(num_soil_layers), ny, nx)
    }
    if soil_shape_drift:
        raise ValueError(
            f"prepared surface soil geometry differs from "
            f"sf_surface_physics={sf_surface_physics}, "
            f"num_soil_layers={num_soil_layers}: {soil_shape_drift}")
    raw_lu = np.asarray(static["LU_INDEX"], dtype=np.int32)
    # WRF real.exe remaps MODIS lake category 21 to the Noah water category
    # when sf_lake_physics=0.  Leaving IVGTYP=21 reaches Noah's fatal
    # "too many input landuse types" guard because VEGPARM has categories
    # 1..20.  Preserve LAKEMASK separately exactly as real.exe does.
    lu, lake_mask = _wrf_noah_landuse(raw_lu)
    soil = np.asarray(static["SCT_DOM"], dtype=np.int32)
    landmask = surface["LANDMASK"]
    snow = np.asarray(
        surface["SNOW"] if "SNOW" in surface else cache.array("met/SNOW"),
        dtype=np.float64)
    snowh = np.asarray(
        surface["SNOWH"] if "SNOWH" in surface else cache.array("met/SNOWH"),
        dtype=np.float64)

    result: dict[str, np.ndarray] = {
        "XLAT": lat,
        "XLONG": lon,
        "XLAT_U": lat_u,
        "XLONG_U": lon_u,
        "XLAT_V": lat_v,
        "XLONG_V": lon_v,
        "CLAT": lat,
        "LU_INDEX": lu,
        "ZNU": cache.array("coord/znu"),
        "ZNW": cache.array("coord/znw"),
        "ZS": soil_depths,
        "DZS": soil_thicknesses,
        "U": u,
        "V": v,
        "W": cache.array("state/w"),
        "PH": cache.array("state/php"),
        "PHB": cache.array("base/phb"),
        "T": theta - _THETA_OFFSET_K,
        "THM": theta_m - _THETA_OFFSET_K,
        # T_INIT is not trajectory-active after input.  The prepared cache
        # retains the final dry-theta state but not real.exe's pre-adjustment
        # diagnostic, so bind it to the final dry theta explicitly.
        "T_INIT": theta - _THETA_OFFSET_K,
        "MU": cache.array("state/mup"),
        "MUB": cache.array("base/mub"),
        "P": pressure_perturbation,
        "AL": al_perturbation,
        "ALB": cache.array("base/alb"),
        "PB": cache.array("base/pb"),
        "P_HYD": total_pressure,
        "FNM": cache.array("coord/fnm"),
        "FNP": cache.array("coord/fnp"),
        "RDNW": cache.array("coord/rdnw"),
        "RDN": cache.array("coord/rdn"),
        "DNW": cache.array("coord/dnw"),
        "DN": cache.array("coord/dn"),
        "C1H": cache.array("coord/c1h"),
        "C2H": cache.array("coord/c2h"),
        "C1F": cache.array("coord/c1f"),
        "C2F": cache.array("coord/c2f"),
        "C3H": cache.array("coord/c3h"),
        "C4H": cache.array("coord/c4h"),
        "C3F": cache.array("coord/c3f"),
        "C4F": cache.array("coord/c4f"),
        "Q2": cache.array("result/surface_qv"),
        "T2": t2,
        "TH2": t2 * (_P0 / psfc) ** _RCP,
        "PSFC": psfc,
        "U10": 0.5 * (u10_face[:, :-1] + u10_face[:, 1:]),
        "V10": 0.5 * (v10_face[:-1, :] + v10_face[1:, :]),
        "RDX": np.asarray(1.0 / dx),
        "RDY": np.asarray(1.0 / dy),
        "GOT_VAR_SSO": np.asarray(0, dtype=np.int32),
        "QVAPOR": cache.array("state/qv"),
        "QCLOUD": cache.array("state/qc"),
        "QRAIN": cache.array("state/qr"),
        "QICE": cache.array("state/qi"),
        "QSNOW": cache.array("state/qs"),
        "QGRAUP": cache.array("state/qg"),
        "SHDMAX": surface["SHDMAX"],
        "SHDMIN": surface["SHDMIN"],
        "SHDAVG": surface["SHDAVG"],
        "SNOALB": surface["SNOALB"],
        "LANDUSEF": static["LANDUSEF"],
        "SOILCTOP": static["SOILCTOP"],
        "SOILCBOT": static["SOILCBOT"],
        "TSLB": surface["TSLB"],
        "SMOIS": surface["SMOIS"],
        "SH2O": surface["SH2O"],
        "SEAICE": surface["SEAICE"],
        "IVGTYP": lu,
        "ISLTYP": soil,
        "VEGFRA": surface["VEGFRA"],
        "SNOW": snow,
        "SNOWH": snowh,
        "LAI": surface["LAI"],
        "MAPFAC_M": static["MAPFAC_M"],
        "MAPFAC_U": static["MAPFAC_U"],
        "MAPFAC_V": static["MAPFAC_V"],
        "MAPFAC_MX": static["MAPFAC_M"],
        "MAPFAC_MY": static["MAPFAC_M"],
        "MAPFAC_UX": static["MAPFAC_U"],
        "MAPFAC_UY": static["MAPFAC_U"],
        "MAPFAC_VX": static["MAPFAC_V"],
        "MAPFAC_VY": static["MAPFAC_V"],
        "MF_VX_INV": 1.0 / np.asarray(static["MAPFAC_V"], dtype=np.float64),
        "F": static["F"],
        "E": static["E"],
        "SINALPHA": static["SINALPHA"],
        "COSALPHA": static["COSALPHA"],
        "HGT": cache.array("base/terrain_z"),
        "TSK": surface["TSK"],
        "P_TOP": np.asarray(p_top),
        "ALBBCK": surface["ALBBCK"],
        "TMN": surface["TMN"],
        "XLAND": surface["XLAND"],
        "SNOWC": (snow > 0.0).astype(np.float64),
        "LANDMASK": landmask,
        "LAKEMASK": lake_mask.astype(np.float64),
        "SST": surface["TSK"],
    }
    # Source analyses generally do not carry scheme-specific extra moments.
    # WRF real.exe's Registry ``i0`` policy initializes absent package
    # members to zero.  Preserve any explicitly prepared scheme state and
    # otherwise materialize zeros so the stock-WRF file is complete and its
    # package choice is auditable.
    inventory = stock_wrf_physics_inventory(mp_physics)
    for field in inventory.wrfinput_fields:
        if field.netcdf_name in result:
            continue
        state_key = f"state/{field.registry_name}"
        result[field.netcdf_name] = (
            cache.array(state_key) if state_key in cache._arrays
            else np.zeros(qv.shape, dtype=np.float32)
        )
    return result


def _lbc_to_wrf(cache: PreparedCache, interval_index: int, logical: str,
                side: str, kind: str) -> np.ndarray:
    value = np.asarray(
        cache.array(f"lbc/{interval_index}/{logical}/{side}/{kind}"))
    if side in {"west", "east"}:
        if logical == "mu":
            return value[0].T
        return np.transpose(value, (2, 0, 1))
    if logical == "mu":
        return value[0]
    return np.transpose(value, (1, 0, 2))


def _wrfbdy_fields(cache: PreparedCache,
                   interval_index: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    wrf_to_logical = {
        "U": "u", "V": "v", "PH": "phi", "T": "theta",
        "MU": "mu", "QVAPOR": "qv",
    }
    suffix_to_side = {"XS": "west", "XE": "east",
                      "YS": "south", "YE": "north"}
    for wrf_name, logical in wrf_to_logical.items():
        for suffix, side in suffix_to_side.items():
            result[f"{wrf_name}_B{suffix}"] = _lbc_to_wrf(
                cache, interval_index, logical, side, "value")
            result[f"{wrf_name}_BT{suffix}"] = _lbc_to_wrf(
                cache, interval_index, logical, side, "tendency")
    return result


def _write_wrfinput(path: Path, contract: Mapping[str, object],
                    dimensions: Mapping[str, int | None],
                    global_updates: Mapping[str, object],
                    fields: Mapping[str, np.ndarray], stamp: str) -> None:
    dataset = _create_dataset(path, contract, dimensions, global_updates)
    try:
        specs = {item["name"]: item for item in contract["variables"]}
        for name, variable in dataset.variables.items():
            if name == "Times":
                _write_timestamp(variable, stamp)
            elif name in fields:
                _write_time_value(variable, fields[name])
            elif "prototype_value" in specs[name]:
                _write_time_value(
                    variable,
                    _resolved_prototype_value(
                        variable, specs[name]["prototype_value"]),
                )
            else:
                shape = _shape_without_time(variable)
                _write_time_value(variable, np.zeros(shape, dtype=variable.dtype))
    finally:
        dataset.close()


def _write_wrfbdy(path: Path, contract: Mapping[str, object],
                  dimensions: Mapping[str, int | None],
                  global_updates: Mapping[str, object],
                  cache: PreparedCache,
                  boundary_times: list[datetime],
                  boundary_interval_seconds: int) -> None:
    dataset = _create_dataset(path, contract, dimensions, global_updates)
    try:
        for time_index, boundary_time in enumerate(boundary_times):
            fields = _wrfbdy_fields(cache, time_index)
            stamp = _date_text(boundary_time)
            next_stamp = _date_text(
                boundary_time
                + timedelta(seconds=boundary_interval_seconds))
            for name, variable in dataset.variables.items():
                if name in {
                    "Times",
                    "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_",
                }:
                    _write_timestamp_at(variable, stamp, time_index)
                elif name == (
                    "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
                ):
                    _write_timestamp_at(variable, next_stamp, time_index)
                elif name in fields:
                    _write_time_value_at(
                        variable, fields[name], time_index)
                else:
                    shape = _shape_without_time(variable)
                    _write_time_value_at(
                        variable, np.zeros(shape, dtype=variable.dtype),
                        time_index)
    finally:
        dataset.close()


def _attribute_matches(actual, expected) -> bool:
    if isinstance(actual, bytes):
        return actual == (expected if isinstance(expected, bytes)
                          else str(expected).encode("latin1"))
    if isinstance(actual, str):
        return actual == str(expected)
    actual_array = np.asarray(actual)
    try:
        expected_array = np.asarray(expected, dtype=actual_array.dtype)
    except (TypeError, ValueError, OverflowError):
        return False
    return (actual_array.shape == expected_array.shape
            and np.array_equal(actual_array, expected_array))


def _domain_global_attributes(updates: Mapping[str, object]) \
        -> dict[str, object]:
    names = [
        "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
        "PARENT_GRID_RATIO", "WEST-EAST_GRID_DIMENSION",
        "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION",
        "DX", "DY", "DT", "CEN_LAT", "CEN_LON", "MOAD_CEN_LAT",
        "TRUELAT1", "TRUELAT2", "STAND_LON",
    ]
    names.extend(
        name for name in (
            "MP_PHYSICS", "RA_LW_PHYSICS", "RA_SW_PHYSICS",
            "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS",
            "BL_PBL_PHYSICS", "CU_PHYSICS",
        )
        if name in updates
    )
    missing = [name for name in names if name not in updates]
    if missing:
        raise ValueError(
            f"domain global-attribute expectation lacks {missing}")
    return {name: updates[name] for name in names}


def _validate_file(path: Path, contract: Mapping[str, object], *,
                   nx: int, ny: int, nz: int,
                   num_soil_layers: int = 4,
                   expected_global_attributes: Mapping[str, object]
                   | None = None) -> dict[str, object]:
    expected_dimensions = _dimensions(
        contract, nx=nx, ny=ny, nz=nz,
        num_soil_layers=num_soil_layers)
    with netCDF4.Dataset(path) as dataset:
        if expected_global_attributes is not None:
            missing_attributes = sorted(
                set(expected_global_attributes) - set(dataset.ncattrs()))
            if missing_attributes:
                raise ValueError(
                    f"{path.name}: missing global attributes "
                    f"{missing_attributes}")
            drift = {
                name: {
                    "actual": dataset.getncattr(name),
                    "expected": expected,
                }
                for name, expected in expected_global_attributes.items()
                if not _attribute_matches(dataset.getncattr(name), expected)
            }
            if drift:
                raise ValueError(
                    f"{path.name}: global attribute drift {drift}")
        if list(dataset.variables) != [item["name"] for item in contract["variables"]]:
            raise ValueError(f"{path.name}: variable inventory/order drift")
        for name, expected in expected_dimensions.items():
            actual = dataset.dimensions[name]
            if expected is not None and len(actual) != expected:
                raise ValueError(f"{path.name}: dimension {name} mismatch")
            if expected is None and not actual.isunlimited():
                raise ValueError(f"{path.name}: dimension {name} is not unlimited")
        nonfinite = []
        for name, variable in dataset.variables.items():
            if np.dtype(variable.dtype).kind == "f":
                # Read one vertical/chunk slab at a time to keep validation
                # memory bounded for wide domains.
                value = variable[:]
                if not np.isfinite(value).all():
                    nonfinite.append(name)
        if nonfinite:
            raise ValueError(f"{path.name}: non-finite variables {nonfinite}")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _forcing_offsets_from_identity(
        identity: Mapping[str, object],
) -> tuple[list[int], str]:
    offsets = identity.get("forcing_offsets_seconds")
    if offsets is not None:
        key = "forcing_offsets_seconds"
        factor = 1
    else:
        offsets = identity.get("forcing_hours")
        key = "forcing_hours"
        factor = 3600
    if (not isinstance(offsets, list) or len(offsets) < 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in offsets)):
        raise ValueError(
            "direct exporter requires at least two integer forcing offsets")
    seconds = [value * factor for value in offsets]
    if seconds[0] != 0 or any(
            later <= earlier
            for earlier, later in zip(seconds, seconds[1:])):
        raise ValueError(
            "direct exporter requires increasing forcing offsets beginning "
            "at zero")
    return seconds, key


def _forcing_interval_indices(cache: PreparedCache,
                              boundary_interval_seconds: int) -> list[int]:
    identity = cache.header["identity"]
    forcing_offsets, forcing_key = _forcing_offsets_from_identity(identity)
    expected_step_seconds = {
        later - earlier
        for earlier, later in zip(forcing_offsets, forcing_offsets[1:])
    }
    if expected_step_seconds != {boundary_interval_seconds}:
        raise ValueError(
            "prepared forcing cadence does not match the requested boundary "
            f"interval: {forcing_key}={identity[forcing_key]}, "
            f"boundary_interval_seconds={boundary_interval_seconds}")

    observed_indices = set()
    for name in cache.header["arrays"]:
        parts = name.split("/")
        if len(parts) >= 2 and parts[0] == "lbc":
            try:
                observed_indices.add(int(parts[1]))
            except ValueError as error:
                raise ValueError(
                    f"invalid prepared LBC interval key {name!r}") from error
    expected_indices = list(range(len(forcing_offsets) - 1))
    if sorted(observed_indices) != expected_indices:
        raise ValueError(
            "prepared LBC interval inventory does not span every forcing pair: "
            f"expected {expected_indices}, got {sorted(observed_indices)}")
    return expected_indices


def _validated_hierarchy(exp, artifacts: Sequence[PreparedDomainArtifacts]):
    domains = tuple(exp.domains)
    if not domains:
        raise ValueError("hierarchy export requires at least one domain")
    if len(domains) > 99:
        raise ValueError("hierarchy export supports at most 99 WRF domains")
    expected_ids = list(range(1, len(domains) + 1))
    actual_ids = [domain.grid_id for domain in domains]
    if actual_ids != expected_ids:
        raise ValueError(
            "WRF hierarchy grid ids must be contiguous and parent-before-child: "
            f"expected {expected_ids}, got {actual_ids}")
    if exp.projection is None or exp.projection.map_proj not in (
            "lambert", "mercator", "polar"):
        raise ValueError(
            "hierarchy direct export requires a lambert, mercator, or "
            "polar [projection] table")
    declared = {}
    for artifact in artifacts:
        if artifact.grid_id in declared:
            raise ValueError(
                f"duplicate artifacts for grid_id={artifact.grid_id}")
        declared[artifact.grid_id] = artifact
    if sorted(declared) != expected_ids:
        raise ValueError(
            "domain-artifact ids must exactly cover the WRF hierarchy: "
            f"expected {expected_ids}, got {sorted(declared)}")
    seen = set()
    pairs = []
    for domain in domains:
        root = domain.grid_id == 1
        if root:
            if (domain.parent_id != 0 or not domain.run.specified
                    or domain.run.nested):
                raise ValueError(
                    "d01 must be specified=true, nested=false, parent_id=0")
        elif (domain.parent_id not in seen or domain.run.specified
              or not domain.run.nested):
            raise ValueError(
                f"d{domain.grid_id:02d} must name an earlier parent and use "
                "specified=false, nested=true")
        seen.add(domain.grid_id)
        pairs.append((domain, declared[domain.grid_id]))
    return tuple(pairs)


def _prepared_domain_context(artifact: PreparedDomainArtifacts, domain,
                             exp, expected_grid, valid_time: datetime):
    cache = PreparedCache(artifact.prepared_cache)
    identity = cache.header["identity"]
    expected_domain = prepared_domain_config_identity(domain)
    # Default-tolerant on fields that postdate the header, strict on
    # everything else: a package upgrade that adds an unused identity
    # field must not make an existing prepared bundle unreadable, and a
    # field carrying a real value must still bind.
    _, differing = compare_prepared_domain_config(
        identity.get("domain_config"), expected_domain,
        not_in_use=undelayed_identity_defaults(exp))
    if differing:
        raise ValueError(prepared_identity_refusal(
            subject=f"d{domain.grid_id:02d} prepared cache",
            header=cache.header, differing=differing))
    cfg = identity["domain_config"]["run"]
    try:
        stock_wrf_physics_inventory(cfg.get("mp_physics"))
    except (TypeError, ValueError) as error:
        raise StockWrfExportUnsupported(
            f"unsupported d{domain.grid_id:02d} direct-export "
            f"microphysics: {error}") from None
    required = {
        "bl_pbl_physics": 1,
        "sf_sfclay_physics": 91,
        "sf_surface_physics": 2,
        "hybrid_opt": 2,
        "hypsometric_opt": 2,
        "specified": domain.grid_id == 1,
        "nested": domain.grid_id != 1,
        "spec_bdy_width": 5,
    }
    mismatch = {
        name: (cfg.get(name), expected)
        for name, expected in required.items()
        if cfg.get(name) != expected
    }
    if mismatch:
        # This IS "the physics slice the profile-free compatibility branch
        # requires" that StockWrfExportUnsupported's docstring names as its
        # own third category, so it must be catchable.  As a bare
        # ValueError it escaped native_hierarchy's stock_wrf_export
        # ="optional" arm and destroyed the whole preparation: a complete,
        # verified three-domain GPU hierarchy whose only defect was that
        # its 250 m LES child runs bl_pbl_physics = 0, which no
        # unchanged-WRF wrfinput set in the v2 slice can represent.  The
        # export still refuses exactly what it refused before; a caller
        # whose product IS the export still fails on it.
        raise StockWrfExportUnsupported(
            f"unsupported d{domain.grid_id:02d} direct-export "
            f"configuration: {mismatch}")
    prepared_valid_time = datetime.fromisoformat(
        cache.header["metadata"]["user"]["initial_valid_time"])
    if _date_text(prepared_valid_time) != _date_text(valid_time):
        raise ValueError(
            f"d{domain.grid_id:02d} requested valid time does not match "
            "the prepared cache")

    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    from gpuwm.native_wrf_contract import native_geometry_contract

    expected_geometry = native_geometry_contract(expected_grid, domain.run)
    geometry, static_sha256 = _load_static_geometry_receipt(
        artifact.geometry_receipt, artifact.static_cache,
        expected_geometry=expected_geometry)
    if list(geometry["mass_shape"]) != [ny, nx] or int(geometry["nz"]) != nz:
        raise ValueError(
            f"d{domain.grid_id:02d} geometry receipt does not match cache")
    for name, geometry_name in (("dx", "dx_m"), ("dy", "dy_m")):
        if not math.isclose(
                float(geometry[geometry_name]), float(cfg[name]),
                rel_tol=1e-12):
            raise ValueError(
                f"d{domain.grid_id:02d} geometry/cache {name} mismatch")
    p_top = _prepared_vertical_contract(cache, nx=nx, ny=ny, nz=nz)
    if p_top != float(exp.vertical.p_top):
        raise ValueError(
            f"d{domain.grid_id:02d} prepared p_top differs from namelist")
    expected_eta = np.asarray(
        exp.vertical.eta_levels, dtype=cache.array("coord/znw").dtype)
    if not np.array_equal(cache.array("coord/znw"), expected_eta):
        raise ValueError(
            f"d{domain.grid_id:02d} prepared eta levels differ from namelist")
    if static_sha256 != identity.get("static_cache_sha256"):
        raise ValueError(
            f"d{domain.grid_id:02d} static cache digest differs from "
            "prepared identity")
    return cache, cfg, geometry, p_top, static_sha256


def _configured_domain_start(exp, domain) -> datetime:
    if hasattr(exp, "domain_start_time"):
        return exp.domain_start_time(domain.grid_id)
    configured = getattr(domain, "start_time", None)
    return exp.start_time if configured is None else configured


def _hierarchy_global_updates(*, valid_time: datetime, cfg,
                              geometry: Mapping[str, object], domain):
    updates = _global_updates(
        valid_time=valid_time,
        nx=int(cfg["nx"]), ny=int(cfg["ny"]), nz=int(cfg["nz"]),
        dx=float(cfg["dx"]), dy=float(cfg["dy"]), dt=float(cfg["dt"]),
        geometry=geometry,
    )
    updates.update({
        "GRID_ID": domain.grid_id,
        "PARENT_ID": domain.parent_id,
        "I_PARENT_START": domain.i_parent_start,
        "J_PARENT_START": domain.j_parent_start,
        "PARENT_GRID_RATIO": domain.parent_grid_ratio,
    })
    return updates


def export_prepared_wrf_hierarchy(
        exp, domain_artifacts: Sequence[PreparedDomainArtifacts], output_dir,
        *, valid_time: datetime | None = None,
        boundary_interval_seconds: int = 3600,
        overwrite: bool = False,
        input_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically export ``wrfinput_d01..dNN`` plus root ``wrfbdy_d01``.

    Every final initial-condition file is generated from its own prepared
    meteorology/static artifact set, allowing those expensive upstream
    preparations to run independently.  Only d01 receives external lateral
    boundaries; children are initialized as nested WRF domains and are forced
    by their declared parents at runtime.
    """

    # Stock WRF v4.6.1 normalizes microphysics to one selector across the
    # hierarchy.  GPUWM's explicit MP8->MP18 edge is executable only in its
    # own one-way coupler and must never be mislabeled as a READY stock-WRF
    # export or silently normalized here.
    from gpuwm.core.microphysics_transition import (
        resolve_microphysics_transition,
    )
    by_id = {domain.grid_id: domain for domain in exp.domains}
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        transition = resolve_microphysics_transition(
            by_id[domain.parent_id].run, domain.run)
        if transition.mixed:
            raise StockWrfExportUnsupported(
                "mixed-domain microphysics is a GPUWM extension and is not "
                "stock-WRF exportable; choose one uniform stock-WRF "
                "mp_physics selector explicitly",
                unsupported={"mp_physics": (
                    int(domain.run.mp_physics),
                    int(by_id[domain.parent_id].run.mp_physics))})

    pairs = _validated_hierarchy(exp, tuple(domain_artifacts))
    expected_grids = tuple(grids_from_projection_config(exp))
    if len(expected_grids) != len(pairs):
        raise ValueError(
            "namelist hierarchy did not resolve one grid per domain")
    if valid_time is None:
        valid_time = exp.start_time
    if _date_text(valid_time) != _date_text(exp.start_time):
        raise ValueError("export valid time differs from the namelist start time")
    if boundary_interval_seconds <= 0:
        raise ValueError("boundary interval must be positive")
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = output.with_name(output.name + f".tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    contract_bundle = _load_contract()
    files = {}
    sources = {}
    root_manifest = None
    try:
        root_domain, root_artifact = pairs[0]
        root_output = staging / ".root-export"
        root_manifest = export_prepared_wrf(
            root_artifact.prepared_cache, root_artifact.static_cache,
            root_artifact.geometry_receipt, root_output,
            valid_time=valid_time,
            boundary_interval_seconds=boundary_interval_seconds,
            overwrite=False,
        )
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            os.replace(root_output / name, staging / name)
            files[name] = root_manifest["files"][name]
        shutil.rmtree(root_output)

        for (domain, artifact), expected_grid in zip(pairs, expected_grids):
            domain_valid_time = _configured_domain_start(exp, domain)
            cache, cfg, geometry, p_top, static_sha256 = \
                _prepared_domain_context(
                    artifact, domain, exp, expected_grid, domain_valid_time)
            physics_inventory = stock_wrf_physics_inventory(
                int(cfg["mp_physics"]))
            domain_contract_bundle = _physics_contract_bundle(
                contract_bundle, physics_inventory.mp_physics)
            input_contract = domain_contract_bundle["wrfinput"]
            name = f"wrfinput_d{domain.grid_id:02d}"
            if domain.grid_id != 1:
                dimensions = _dimensions(
                    input_contract, nx=int(cfg["nx"]), ny=int(cfg["ny"]),
                    nz=int(cfg["nz"]))
                updates = _hierarchy_global_updates(
                    valid_time=domain_valid_time, cfg=cfg, geometry=geometry,
                    domain=domain)
                with np.load(artifact.static_cache, allow_pickle=False) as static:
                    fields = _wrfinput_fields(
                        cache, static, geometry, domain_valid_time, p_top=p_top,
                        mp_physics=physics_inventory.mp_physics)
                    _write_wrfinput(
                        staging / name, input_contract, dimensions,
                        updates, fields, _date_text(domain_valid_time))
                files[name] = _validate_file(
                    staging / name, input_contract,
                    nx=int(cfg["nx"]), ny=int(cfg["ny"]), nz=int(cfg["nz"]),
                    expected_global_attributes=_domain_global_attributes(
                        updates))
            sources[f"d{domain.grid_id:02d}"] = {
                "prepared_header_sha256": _sha256(cache.header_path),
                "prepared_content_sha256": cache.header["content_sha256"],
                "static_cache_sha256": static_sha256,
                "geometry_receipt_sha256": _sha256(
                    artifact.geometry_receipt),
                "mp_physics": physics_inventory.mp_physics,
                "microphysics": physics_inventory.scheme,
                "start_time": _date_text(domain_valid_time),
                "resolved_physics_contract_sha256":
                    _contract_payload_sha256(domain_contract_bundle),
            }

        hierarchy = [{
            "grid_id": domain.grid_id,
            "parent_id": domain.parent_id,
            "i_parent_start": domain.i_parent_start,
            "j_parent_start": domain.j_parent_start,
            "parent_grid_ratio": domain.parent_grid_ratio,
            "parent_time_step_ratio": domain.parent_time_step_ratio,
            "nx": domain.run.nx,
            "ny": domain.run.ny,
            "nz": domain.run.nz,
            "dx_m": domain.run.dx,
            "dy_m": domain.run.dy,
            "dt_s": domain.run.dt,
            "mp_physics": domain.run.mp_physics,
            "start_time": _date_text(_configured_domain_start(exp, domain)),
        } for domain, _artifact in pairs]
        forcing_key = (
            "forcing_offsets_seconds"
            if "forcing_offsets_seconds" in root_manifest
            else "forcing_hours")
        manifest = {
            "schema": HIERARCHY_EXPORT_SCHEMA,
            "status": "READY",
            "valid_time": _date_text(valid_time),
            "boundary_interval_seconds": boundary_interval_seconds,
            "boundary_record_count": root_manifest["boundary_record_count"],
            "boundary_times": root_manifest["boundary_times"],
            "next_boundary_times": root_manifest["next_boundary_times"],
            forcing_key: root_manifest[forcing_key],
            "hierarchy": hierarchy,
            "vertical": {
                "e_vert": len(exp.vertical.eta_levels),
                "p_top_pa": exp.vertical.p_top,
                "hybrid_opt": exp.vertical.hybrid_opt,
                "eta_sha256": _array_sha256(np.asarray(
                    exp.vertical.eta_levels, dtype=np.float64)),
            },
            "source": {
                "domains": sources,
                "contract_sha256": _sha256(_CONTRACT_PATH),
                "input_provenance": dict(input_provenance or {}),
            },
            "files": files,
            "limitations": [
                "static, one-way nested projected hierarchy only (lambert stock-WRF-gated; mercator/polar exports oracle-gated, not yet stock-WRF-gated)",
                "all domains must use one shared explicit eta coordinate",
                "WRF-v4.6.1 Registry-inventoried "
                "WSM6/Thompson/Morrison/NSSL-2 "
                "microphysics with YSU+classic-MM5+Noah only",
                "T_INIT is bound to final dry theta and is diagnostic-only",
                "hydrometeor/W/HT_SHAD/PC external root LBC arrays are zero",
                "root forcing must be uniformly spaced and begin at f00",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        with _output_publication(output, overwrite=overwrite) as backup:
            _publish_staging(staging, output, backup)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def export_prepared_wrf_namelists(
        wps_namelist: str | Path, namelist_input: str | Path,
        domain_artifacts_manifest: str | Path, output_dir, *,
        valid_time: datetime | None = None,
        boundary_interval_seconds: int = 3600,
        overwrite: bool = False,
) -> dict[str, object]:
    """Namelist-driven hierarchy entry point for the native final exporter."""

    import tomllib

    from gpuwm.experiment import build_experiment
    from gpuwm.namelist_import import import_namelists
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts

    wps_path = Path(wps_namelist)
    input_path = Path(namelist_input)
    artifacts_path = Path(domain_artifacts_manifest)
    resolved_text, report = import_namelists(wps_path, input_path)
    exp = build_experiment(
        tomllib.loads(resolved_text),
        source=f"native export of {wps_path.name} + {input_path.name}")
    validate_native_lambert_contracts(
        exp, wps_path, source_name="native hierarchy")
    provenance = {
        "wps_namelist_sha256": _sha256(wps_path),
        "namelist_input_sha256": _sha256(input_path),
        "domain_artifacts_manifest_sha256": _sha256(artifacts_path),
        "resolved_experiment_sha256": hashlib.sha256(
            resolved_text.encode("utf-8")).hexdigest(),
        "translation_report": asdict(report),
    }
    return export_prepared_wrf_hierarchy(
        exp, load_domain_artifacts_manifest(artifacts_path), output_dir,
        valid_time=valid_time,
        boundary_interval_seconds=boundary_interval_seconds,
        overwrite=overwrite,
        input_provenance=provenance,
    )


def export_prepared_wrf(prepared_cache, static_cache, geometry_receipt,
                        output_dir, *, valid_time: datetime,
                        boundary_interval_seconds: int = 3600,
                        overwrite: bool = False,
                        physics_profile: str | None = None,
                        expert_acknowledgements: Sequence[str] = (),
                        acknowledgement_provenance: Mapping[
                            str, object] | None = None,
                        experiment_config_suite: bool = False,
                        ) -> dict[str, object]:
    """Export wrfinput/wrfbdy under one of three physics contracts.

    - ``physics_profile`` named: the exporter re-validates that the
      cache-bound config IS that shipped suite and the v3 manifest
      carries the named-profile selection receipt.
    - ``experiment_config_suite=True`` (owner ruling 2026-07-31, the
      profileless preparation contract): the exporter recomputes the
      same source-neutral per-domain selection receipt the front door
      computed -- from its OWN cache-bound config, never from a caller
      claim -- and the v3 manifest carries it, so the caller's byte
      equality check proves the prepared physics came from this config
      under this registry.  Expert tuples retain their registry-owned
      acknowledgement here too.
    - Neither: the historical v2 contract, pinned to the exact stock
      slice (WSM6+YSU+classic-MM5+Noah); see
      test_the_profile_free_export_gate_keeps_its_exact_v2_stock_slice.
    """

    cache = PreparedCache(Path(prepared_cache))
    static_path = Path(static_cache)
    geometry_path = Path(geometry_receipt)
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    if boundary_interval_seconds <= 0:
        raise ValueError("boundary interval must be positive")

    identity = cache.header["identity"]
    cfg = identity["domain_config"]["run"]
    physics_selection = None
    if physics_profile is not None:
        from gpuwm.physics_compat import (
            validate_single_domain_physics_profile,
        )

        physics_selection = validate_single_domain_physics_profile(
            physics_profile, config=cfg,
            expert_acknowledgements=tuple(expert_acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    elif experiment_config_suite:
        from gpuwm.physics_compat import single_domain_physics_selection

        physics_selection = single_domain_physics_selection(
            cfg,
            expert_acknowledgements=tuple(expert_acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    try:
        physics_inventory = stock_wrf_physics_inventory(
            cfg.get("mp_physics"))
    except (TypeError, ValueError) as error:
        raise StockWrfExportUnsupported(
            f"unsupported direct-export microphysics: {error}",
            unsupported={"mp_physics": (cfg.get("mp_physics"), None)},
        ) from None
    contract_bundle = _physics_contract_bundle(
        _load_contract(), physics_inventory.mp_physics)
    required = {
        "hybrid_opt": 2,
        "hypsometric_opt": 2,
        "specified": True,
        "nested": False,
        "spec_bdy_width": 5,
    }
    if physics_selection is None:
        # Compatibility entry point for historical callers and proof
        # documents.  New front doors always supply a profile and take the
        # selector-capability path above; omitting it retains the exact v2
        # stock slice rather than widening an old API implicitly.
        required.update({
            "bl_pbl_physics": 1,
            "sf_sfclay_physics": 91,
            "sf_surface_physics": 2,
        })
    mismatch = {name: (cfg.get(name), expected)
                for name, expected in required.items()
                if cfg.get(name) != expected}
    if mismatch:
        raise StockWrfExportUnsupported(
            f"unsupported direct-export configuration: {mismatch}",
            unsupported=mismatch)
    interval_indices = _forcing_interval_indices(
        cache, boundary_interval_seconds)
    _forcing_offsets, forcing_key = _forcing_offsets_from_identity(identity)
    prepared_valid_time = datetime.fromisoformat(
        cache.header["metadata"]["user"]["initial_valid_time"])
    if _date_text(prepared_valid_time) != _date_text(valid_time):
        raise ValueError("requested valid time does not match prepared cache")

    geometry, actual_static_sha256 = _load_static_geometry_receipt(
        geometry_path, static_path)
    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    if list(geometry["mass_shape"]) != [ny, nx] or int(geometry["nz"]) != nz:
        raise ValueError("geometry receipt does not match prepared cache")
    if not math.isclose(float(geometry["dx_m"]), float(cfg["dx"]), rel_tol=1e-12):
        raise ValueError("geometry/cache dx mismatch")
    if not math.isclose(float(geometry["dy_m"]), float(cfg["dy"]), rel_tol=1e-12):
        raise ValueError("geometry/cache dy mismatch")
    p_top = _prepared_vertical_contract(
        cache, nx=nx, ny=ny, nz=nz)
    expected_static_sha256 = identity.get("static_cache_sha256")
    if actual_static_sha256 != expected_static_sha256:
        raise ValueError("static cache digest does not match prepared identity")

    staging = output.with_name(output.name + f".tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    stamp = _date_text(valid_time)
    boundary_times = [
        valid_time + timedelta(
            seconds=interval_index * boundary_interval_seconds)
        for interval_index in interval_indices
    ]
    boundary_stamps = [_date_text(value) for value in boundary_times]
    next_boundary_stamps = [
        _date_text(value + timedelta(seconds=boundary_interval_seconds))
        for value in boundary_times
    ]
    # The suite receipt is per-domain; this exporter writes exactly d01,
    # so the global attributes are stamped from that domain's selector
    # record.  A named-profile receipt already carries flat selectors.
    selection_selectors = physics_selection
    if physics_selection is not None and "domains" in physics_selection:
        selection_selectors = physics_selection["domains"]["1"]
    updates = _global_updates(
        valid_time=valid_time, nx=nx, ny=ny, nz=nz,
        dx=float(cfg["dx"]), dy=float(cfg["dy"]), dt=float(cfg["dt"]),
        geometry=geometry,
        physics_selection=selection_selectors,
    )
    num_soil_layers = int(cfg.get("num_soil_layers", 4))
    sf_surface_physics = int(cfg["sf_surface_physics"])
    try:
        with np.load(static_path, allow_pickle=False) as static:
            wrfinput_fields = _wrfinput_fields(
                cache, static, geometry, valid_time, p_top=p_top,
                mp_physics=physics_inventory.mp_physics,
                sf_surface_physics=sf_surface_physics,
                num_soil_layers=num_soil_layers)
            input_contract = contract_bundle["wrfinput"]
            input_dimensions = _dimensions(
                input_contract, nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers)
            _write_wrfinput(
                staging / "wrfinput_d01", input_contract, input_dimensions,
                updates, wrfinput_fields, stamp)
        bdy_contract = contract_bundle["wrfbdy"]
        bdy_dimensions = _dimensions(
            bdy_contract, nx=nx, ny=ny, nz=nz,
            num_soil_layers=num_soil_layers)
        _write_wrfbdy(
            staging / "wrfbdy_d01", bdy_contract, bdy_dimensions,
            updates, cache, boundary_times, boundary_interval_seconds)
        files = {
            "wrfinput_d01": _validate_file(
                staging / "wrfinput_d01", input_contract,
                nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers,
                expected_global_attributes=_domain_global_attributes(
                    updates)),
            "wrfbdy_d01": _validate_file(
                staging / "wrfbdy_d01", bdy_contract,
                nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers,
                expected_global_attributes=_domain_global_attributes(
                    updates)),
        }
        limitations = [
            "single specified projected domain only (lambert stock-WRF-gated; mercator/polar exports oracle-gated, not yet stock-WRF-gated)",
            (
                f"{physics_inventory.scheme}+YSU+classic-MM5+Noah "
                "initialized state contract"
                if physics_selection is None else
                (f"{physics_selection['profile']} initialized state contract"
                 if physics_selection.get("profile") is not None else
                 "hash-bound experiment-config suite initialized state "
                 "contract")
            ),
            "T_INIT is bound to final dry theta and is diagnostic-only",
            "hydrometeor/W/HT_SHAD/PC external LBC arrays are zero",
            "forcing must be uniformly spaced and begin at f00",
        ]
        if not any(
                name.startswith("surface/") for name in cache._arrays):
            limitations.insert(2, "warm-soil SH2O initialization only")
        manifest = {
            "schema": (
                "gpuwm-native-direct-wrf-export-v2"
                if physics_selection is None
                else "gpuwm-native-direct-wrf-export-v3"
            ),
            "status": "READY",
            "valid_time": stamp,
            "next_boundary_time": next_boundary_stamps[0],
            "boundary_interval_seconds": boundary_interval_seconds,
            "boundary_record_count": len(boundary_times),
            "boundary_times": boundary_stamps,
            "next_boundary_times": next_boundary_stamps,
            forcing_key: identity[forcing_key],
            "dimensions": {"nx": nx, "ny": ny, "nz": nz},
            "source": {
                "prepared_header_sha256": _sha256(cache.header_path),
                "prepared_content_sha256": cache.header["content_sha256"],
                "static_cache_sha256": actual_static_sha256,
                "geometry_receipt_sha256": _sha256(geometry_path),
                "contract_sha256": _sha256(_CONTRACT_PATH),
                "resolved_physics_contract_sha256":
                    _contract_payload_sha256(contract_bundle),
            },
            "files": files,
            "limitations": limitations,
        }
        if physics_selection is not None:
            manifest["physics"] = physics_selection
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        with _output_publication(output, overwrite=overwrite) as backup:
            _publish_staging(staging, output, backup)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--static-cache", type=Path)
    parser.add_argument("--geometry-receipt", type=Path)
    parser.add_argument(
        "--domain-artifacts", type=Path,
        help="gpuwm-native-domain-artifacts-v1 JSON for d01..dNN",
    )
    parser.add_argument("--wps-namelist", type=Path)
    parser.add_argument("--namelist-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--valid-time", help=(
            "UTC YYYY-MM-DD_HH:MM:SS; hierarchy mode defaults to and "
            "cross-checks the namelist start time"))
    parser.add_argument("--boundary-interval-seconds", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--physics-profile",
        help="registered fixed physics template carried by the prepared cache")
    parser.add_argument(
        "--experiment-config-suite", action="store_true",
        help=("no named profile: recompute and carry the cache-bound "
              "config's own per-domain physics selection receipt "
              "(v3 manifest), instead of the legacy v2 stock slice"))
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert acknowledgement id; repeat as needed")
    args = parser.parse_args()
    valid_time = (datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
                  if args.valid_time is not None else None)
    if args.domain_artifacts is not None:
        missing = [name for name, value in (
            ("--wps-namelist", args.wps_namelist),
            ("--namelist-input", args.namelist_input),
        ) if value is None]
        incompatible = [name for name, value in (
            ("--prepared-cache", args.prepared_cache),
            ("--static-cache", args.static_cache),
            ("--geometry-receipt", args.geometry_receipt),
            ("--physics-profile", args.physics_profile),
            ("--experiment-config-suite",
             args.experiment_config_suite or None),
            ("--ack", args.ack or None),
        ) if value is not None]
        if missing or incompatible:
            parser.error(
                "hierarchy mode missing/incompatible options: "
                f"missing={missing}, incompatible={incompatible}")
        manifest = export_prepared_wrf_namelists(
            args.wps_namelist, args.namelist_input, args.domain_artifacts,
            args.output, valid_time=valid_time,
            boundary_interval_seconds=args.boundary_interval_seconds,
            overwrite=args.overwrite)
    else:
        missing = [name for name, value in (
            ("--prepared-cache", args.prepared_cache),
            ("--static-cache", args.static_cache),
            ("--geometry-receipt", args.geometry_receipt),
            ("--valid-time", valid_time),
        ) if value is None]
        incompatible = [name for name, value in (
            ("--wps-namelist", args.wps_namelist),
            ("--namelist-input", args.namelist_input),
        ) if value is not None]
        if missing or incompatible:
            parser.error(
                "single-domain mode missing/incompatible options: "
                f"missing={missing}, incompatible={incompatible}")
        if args.physics_profile is not None and args.experiment_config_suite:
            parser.error(
                "--experiment-config-suite means NO named profile; "
                "pass exactly one of it and --physics-profile")
        manifest = export_prepared_wrf(
            args.prepared_cache, args.static_cache, args.geometry_receipt,
            args.output, valid_time=valid_time,
            boundary_interval_seconds=args.boundary_interval_seconds,
            overwrite=args.overwrite,
            physics_profile=args.physics_profile,
            experiment_config_suite=args.experiment_config_suite,
            expert_acknowledgements=tuple(args.ack),
            acknowledgement_provenance={
                value: ["--ack"] for value in args.ack
            })
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
