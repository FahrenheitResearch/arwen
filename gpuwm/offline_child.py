"""Strict contracts for native offline parent-to-child downscaling.

This module is the file-facing half of gpuwm's CUDA-native ``ndown``
replacement.  It deliberately separates cheap metadata validation from the
later GPU interpolation/build transaction: an archived parent series must be
proved complete, geometrically identical, regularly ordered, and sufficiently
frequent before any child state is allocated.

Both gpuwm and stock-WRF history files are accepted.  The reader is closed
world for trajectory fields but tolerant of additional diagnostic variables.
Physics conversion is explicit.  In particular, active condensate may never
be paired with a fabricated zero number moment merely because the parent used
a different microphysics scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import time

import netCDF4
import numpy as np

from gpuwm.core.grid import (BaseState, compute_hybrid_coeffs,
                             finalize_vertical_coord, make_vertical_coord)
from gpuwm.core.nest_interp import register_nest, sint
from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
from gpuwm.ingest.lateral_bc import (
    LateralBoundaries,
    build_lateral_interval_from_sides,
    extract_lateral_side,
)


_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
_REQUIRED_DYNAMICS = frozenset({
    "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "P_TOP", "ZNU", "ZNW", "QVAPOR",
})
_GEOMETRY_DIMS = (
    "west_east", "south_north", "bottom_top", "west_east_stag",
    "south_north_stag", "bottom_top_stag",
)
_GEOMETRY_ATTRS = (
    "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON",
    "MOAD_CEN_LAT", "CEN_LAT", "CEN_LON", "POLE_LAT", "POLE_LON",
    "HYBRID_OPT", "ETAC",
)
_STATIC_GEOMETRY_FIELDS = ("HGT", "XLAT", "XLONG", "MAPFAC_M", "ZNU", "ZNW")

_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
_NSSL_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)
_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNSNOW": "ns", "QNGRAUPEL": "ng",
})
_NSSL_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
})


class OfflineChildContractError(ValueError):
    """The archived parent cannot safely force the requested child."""


class MomentDiagnosisRequired(OfflineChildContractError):
    """A target number moment needs the official target-scheme initializer."""


@dataclass(frozen=True)
class ParentPhysicsBinding:
    """Authoritative parent-scheme identity from a companion setup record."""

    mp_physics: int
    morr_rimed_ice: int | None
    domain_id: int
    evidence_kind: str
    evidence_path: Path
    evidence_sha256: str

    def __post_init__(self) -> None:
        if int(self.mp_physics) not in {6, 8, 10, 18}:
            raise OfflineChildContractError(
                f"unsupported bound parent mp_physics={self.mp_physics}")
        if int(self.domain_id) < 1:
            raise OfflineChildContractError("bound parent domain_id must be >= 1")
        if int(self.mp_physics) == 10 and self.morr_rimed_ice not in {0, 1}:
            raise OfflineChildContractError(
                "bound Morrison parent requires morr_rimed_ice=0/1")
        if (int(self.mp_physics) != 10
                and self.morr_rimed_ice is not None):
            raise OfflineChildContractError(
                "morr_rimed_ice evidence is only valid for mp_physics=10")
        if self.evidence_kind not in {"gpuwm-restart", "wrf-namelist"}:
            raise OfflineChildContractError(
                f"unsupported parent physics evidence {self.evidence_kind!r}")
        if not Path(self.evidence_path).is_file():
            raise OfflineChildContractError(
                f"parent physics evidence does not exist: {self.evidence_path}")
        digest = str(self.evidence_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise OfflineChildContractError(
                "parent physics evidence_sha256 must be one SHA-256 hex digest")

    def receipt(self) -> Mapping[str, object]:
        return MappingProxyType({
            "mp_physics": int(self.mp_physics),
            "morr_rimed_ice": self.morr_rimed_ice,
            "domain_id": int(self.domain_id),
            "evidence_kind": self.evidence_kind,
            "evidence_path": str(Path(self.evidence_path).resolve()),
            "evidence_sha256": str(self.evidence_sha256).lower(),
        })


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_parent_physics_from_gpuwm_restart(
        path: str | Path) -> ParentPhysicsBinding:
    """Bind parent MP identity to a validated gpuwm restart header."""

    from gpuwm.io.restart import read_restart_header
    path = Path(path).resolve()
    header = read_restart_header(path)
    config = header.get("config")
    setup = header.get("physics_setup")
    fingerprint = header.get("physics_setup_fingerprint")
    if not isinstance(config, dict) or not isinstance(setup, dict):
        raise OfflineChildContractError(
            f"{path} lacks complete gpuwm config/physics setup evidence")
    if fingerprint != _canonical_sha256(setup):
        raise OfflineChildContractError(
            f"{path} physics setup fingerprint is invalid")
    microphysics = setup.get("microphysics")
    if not isinstance(microphysics, dict):
        raise OfflineChildContractError(
            f"{path} lacks resolved microphysics setup evidence")
    mp_physics = int(config.get("mp_physics", -1))
    if int(microphysics.get("scheme_id", -2)) != mp_physics:
        raise OfflineChildContractError(
            f"{path} config and resolved microphysics identities disagree")
    domain_id = int(header.get("grid_id", config.get("grid_id", 0)))
    morr = None
    if mp_physics == 10:
        morr = int(config.get("morr_rimed_ice", -1))
        resolved = microphysics.get("morrison_rimed_ice")
        if (not isinstance(resolved, dict)
                or int(resolved.get("selection", -2)) != morr):
            raise OfflineChildContractError(
                f"{path} Morrison rimed-ice config/setup identities disagree")
    evidence = {
        "format_version": header.get("format_version"),
        "grid_id": domain_id,
        "config": config,
        "physics_setup": setup,
        "physics_setup_fingerprint": fingerprint,
    }
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="gpuwm-restart", evidence_path=path,
        evidence_sha256=_canonical_sha256(evidence))


def bind_parent_physics_from_wrf_namelist(
        path: str | Path, *, domain_id: int = 1) -> ParentPhysicsBinding:
    """Bind one WRF domain's MP identity to exact namelist bytes."""

    from gpuwm.namelist_import import parse_namelist
    path = Path(path).resolve()
    domain_id = int(domain_id)
    if domain_id < 1:
        raise OfflineChildContractError("WRF binding domain_id must be >= 1")
    parsed = parse_namelist(path)
    physics = parsed.get("physics", {})

    def domain_value(name: str, *, required: bool, default=None):
        values = physics.get(name)
        if not values:
            if required:
                raise OfflineChildContractError(
                    f"{path} &physics lacks authoritative {name}")
            return default
        return values[min(domain_id - 1, len(values) - 1)]

    mp_physics = int(domain_value("mp_physics", required=True))
    morr = None
    if mp_physics == 10:
        morr = int(domain_value(
            "morr_rimed_ice", required=False, default=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="wrf-namelist", evidence_path=path,
        evidence_sha256=digest)


def _decode_time(variable) -> datetime:
    raw = np.asarray(variable[:])
    if raw.shape[0] != 1:
        raise OfflineChildContractError(
            f"one history file must contain exactly one Time record, got {raw.shape}")
    row = raw[0]
    if row.dtype.kind == "S":
        value = b"".join(row.tolist()).decode("ascii")
    else:
        value = "".join(str(item) for item in row.tolist())
    try:
        return datetime.strptime(value, _TIME_FORMAT)
    except ValueError as exc:
        raise OfflineChildContractError(
            f"invalid WRF Times value {value!r}") from exc


def _hash_array(digest, label: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _infer_mp_physics(inventory: frozenset[str]) -> int | None:
    if {"QNSNOW", "QNGRAUPEL"} <= inventory:
        return 10
    if {"QNRAIN", "QNICE"} <= inventory:
        return 8
    if {"QICE", "QSNOW", "QGRAUP"} <= inventory:
        return 6
    return None


@dataclass(frozen=True)
class ParentHistoryFrame:
    """Metadata-only proof for one archived parent state."""

    path: Path
    valid_time: datetime
    source_kind: str
    source_mp_physics: int | None
    inferred_mp_physics: int | None
    dimensions: Mapping[str, int]
    variables: frozenset[str]
    geometry_sha256: str


@dataclass(frozen=True)
class ParentHistoryContract:
    """Validated ordered parent series ready for native child preparation."""

    frames: tuple[ParentHistoryFrame, ...]
    interval_seconds: float
    geometry_sha256: str
    source_kind: str
    source_mp_physics: int | None
    physics_binding: ParentPhysicsBinding | None
    max_boundary_interval_seconds: float

    @property
    def start_time(self) -> datetime:
        return self.frames[0].valid_time

    @property
    def end_time(self) -> datetime:
        return self.frames[-1].valid_time


def inspect_parent_history_frame(
        path: str | Path, *, source_mp_physics: int | None = None,
) -> ParentHistoryFrame:
    """Inspect one gpuwm/WRF history file without loading 3-D trajectory data."""

    path = Path(path)
    with netCDF4.Dataset(path) as dataset:
        if "Times" not in dataset.variables:
            raise OfflineChildContractError(f"{path} has no WRF Times variable")
        valid_time = _decode_time(dataset.variables["Times"])
        variables = frozenset(dataset.variables)
        missing = sorted(_REQUIRED_DYNAMICS - variables)
        if missing:
            raise OfflineChildContractError(
                f"{path} is missing offline-child trajectory fields {missing}")
        dimensions = {}
        for name in _GEOMETRY_DIMS:
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing WRF geometry dimension {name!r}")
            dimensions[name] = len(dataset.dimensions[name])

        inferred_mp = _infer_mp_physics(variables)
        bound_mp = None
        if source_mp_physics is not None:
            requested = int(source_mp_physics)
            if requested not in {6, 8, 10, 18}:
                raise OfflineChildContractError(
                    f"unsupported declared parent mp_physics={requested}")
            # Inventory-only inference is advisory. WRF streams may retain
            # dormant number variables, and unified NSSL can expose an
            # inventory that looks like another multi-moment scheme. The
            # companion namelist/setup/manifest is authoritative.
            bound_mp = requested

        digest = hashlib.sha256()
        for name in _GEOMETRY_DIMS:
            digest.update(f"dim:{name}={dimensions[name]};".encode("ascii"))
        for name in _GEOMETRY_ATTRS:
            if name in dataset.ncattrs():
                digest.update(f"attr:{name}={dataset.getncattr(name)!r};".encode())
        for name in _STATIC_GEOMETRY_FIELDS:
            if name not in dataset.variables:
                if name in {"XLAT", "XLONG", "MAPFAC_M"}:
                    # Some minimal stock-WRF history streams omit these; the
                    # projection/dimension identity remains enforceable and a
                    # companion native setup archive supplies the arrays.
                    continue
                raise OfflineChildContractError(
                    f"{path} is missing vertical/static geometry field {name}")
            value = dataset.variables[name][:]
            if value.ndim and value.shape[0] == 1:
                value = value[0]
            _hash_array(digest, name, value)

        title = str(getattr(dataset, "TITLE", ""))
        source_kind = "gpuwm" if (
            "gpuwm" in title.lower() or "GPUWM_WRITE_COMPLETE" in dataset.ncattrs()
        ) else "wrf"
    return ParentHistoryFrame(
        path=path.resolve(), valid_time=valid_time, source_kind=source_kind,
        source_mp_physics=bound_mp, inferred_mp_physics=inferred_mp,
        dimensions=MappingProxyType(dimensions), variables=variables,
        geometry_sha256=digest.hexdigest(),
    )


def validate_parent_history(
        paths: Sequence[str | Path], *, max_boundary_interval_seconds: float,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
) -> ParentHistoryContract:
    """Prove an archived parent series before constructing a child.

    ``max_boundary_interval_seconds`` is intentionally mandatory.  Cadence is
    a scientific choice tied to child resolution and expected advection; the
    tool will not silently bless hourly parent history for a 500-m child.
    """

    if physics_binding is not None:
        if source_mp_physics is not None:
            raise OfflineChildContractError(
                "pass physics_binding or source_mp_physics, not both")
        source_mp_physics = int(physics_binding.mp_physics)
    maximum = float(max_boundary_interval_seconds)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise OfflineChildContractError(
            "max_boundary_interval_seconds must be finite and positive")
    frames = tuple(inspect_parent_history_frame(
        path, source_mp_physics=source_mp_physics) for path in paths)
    if len(frames) < 2:
        raise OfflineChildContractError(
            "offline child forcing requires at least two parent history frames")
    if len({frame.geometry_sha256 for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history geometry/static state changes between frames")
    if len({frame.source_kind for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history mixes gpuwm and stock-WRF producers")
    if len({frame.source_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history bound microphysics identity changes between frames")
    if len({frame.inferred_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history advisory moisture inventory changes between frames")
    seconds = np.asarray([
        (frame.valid_time - frames[0].valid_time).total_seconds()
        for frame in frames
    ], dtype=np.float64)
    differences = np.diff(seconds)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise OfflineChildContractError(
            "parent history times must be unique and strictly increasing")
    interval = float(differences[0])
    if not np.all(differences == interval):
        raise OfflineChildContractError(
            f"parent history cadence is irregular: {differences.tolist()} seconds")
    if interval > maximum:
        raise OfflineChildContractError(
            f"parent history cadence {interval:g} s exceeds the declared child "
            f"forcing limit {maximum:g} s; regenerate a denser parent archive")
    return ParentHistoryContract(
        frames=frames, interval_seconds=interval,
        geometry_sha256=frames[0].geometry_sha256,
        source_kind=frames[0].source_kind,
        source_mp_physics=frames[0].source_mp_physics,
        physics_binding=physics_binding,
        max_boundary_interval_seconds=maximum,
    )


def read_parent_microphysics(path: str | Path) -> Mapping[str, np.ndarray]:
    """Read only transported moisture fields from one WRF history record."""

    fields: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(path) as dataset:
        for wrf_name, state_name in _WRF_TO_STATE.items():
            if wrf_name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[wrf_name][:])
            if value.shape[0] != 1:
                raise OfflineChildContractError(
                    f"{path}/{wrf_name} must have exactly one Time record")
            fields[state_name] = np.ascontiguousarray(value[0], dtype=np.float32)
    missing = sorted(set(_MASS_FIELDS) - set(fields))
    if missing:
        raise OfflineChildContractError(
            f"{path} lacks transported parent mass fields {missing}")
    return MappingProxyType(fields)


def _same_shape(fields: Mapping[str, np.ndarray]) -> tuple[int, ...]:
    shapes = {tuple(np.asarray(value).shape) for value in fields.values()}
    if len(shapes) != 1:
        raise OfflineChildContractError(
            f"microphysics fields have inconsistent shapes {sorted(shapes)}")
    return next(iter(shapes))


def map_microphysics_to_nssl18(
        fields: Mapping[str, np.ndarray], *, source_mp_physics: int,
        morr_rimed_ice: int | None = None,
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
        active_mass_threshold: float = 1.0e-8,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    """Map WSM6/Thompson/Morrison transported state into NSSL mp18.

    The optional ``diagnose_missing`` callback is the exact target-scheme
    ``calcnfromq`` implementation.  Until that official initializer is
    supplied, any active mass category lacking its number moment fails closed.
    This prevents a superficially runnable but physically invalid child.
    """

    source_mp = int(source_mp_physics)
    if source_mp not in {6, 8, 10, 18}:
        raise OfflineChildContractError(
            f"NSSL conversion currently accepts source mp 6/8/10/18, got {source_mp}")
    normalized = {name: np.asarray(value, dtype=np.float32)
                  for name, value in fields.items()}
    if source_mp == 18:
        missing = sorted(set(_NSSL_FIELDS) - set(normalized))
        if missing:
            raise OfflineChildContractError(
                f"NSSL passthrough lacks transported fields {missing}")
        result = {}
        shape = _same_shape({name: normalized[name] for name in _NSSL_FIELDS})
        for name in _NSSL_FIELDS:
            value = normalized[name]
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
                raise OfflineChildContractError(
                    f"NSSL passthrough field {name} is non-finite, negative, "
                    "or wrong-shaped")
            result[name] = np.array(value, copy=True, order="C")
        return MappingProxyType(result), MappingProxyType({
            "source_mp_physics": 18,
            "target_mp_physics": 18,
            "category_mapping": "nssl18-passthrough",
            "carried_source_moments": tuple(_NSSL_FIELDS[7:]),
            "diagnosed_target_moments": (),
            "active_mass_threshold": float(active_mass_threshold),
        })
    missing_mass = sorted(set(_MASS_FIELDS) - set(normalized))
    if missing_mass:
        raise OfflineChildContractError(
            f"source microphysics lacks mass fields {missing_mass}")
    shape = _same_shape({name: normalized[name] for name in _MASS_FIELDS})
    for name in _MASS_FIELDS:
        value = normalized[name]
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise OfflineChildContractError(
                f"source microphysics mass field {name} is non-finite or negative")
    result = {name: np.array(normalized[name], copy=True, order="C")
              for name in _MASS_FIELDS}
    result["qh"] = np.zeros(shape, dtype=np.float32)
    category_mapping = "graupel-to-graupel"
    if source_mp == 10:
        if morr_rimed_ice not in {0, 1}:
            raise OfflineChildContractError(
                "Morrison -> NSSL requires explicit morr_rimed_ice=0/1")
        if int(morr_rimed_ice) == 1:
            result["qh"] = result["qg"]
            result["qg"] = np.zeros(shape, dtype=np.float32)
            category_mapping = "morrison-hail-to-nssl-hail"

    aliases = {
        "qndrop": ("qndrop", "nc"), "qnr": ("qnr", "nr"),
        "qni": ("qni", "ni"), "qns": ("qns", "ns"),
        "qng": ("qng", "ng"), "qnh": ("qnh", "nh"),
        "qvolg": ("qvolg", "volg"), "qvolh": ("qvolh", "volh"),
    }
    carried = []
    for target, choices in aliases.items():
        for choice in choices:
            if choice in normalized:
                value = np.asarray(normalized[choice], dtype=np.float32)
                if value.shape != shape:
                    raise OfflineChildContractError(
                        f"source {choice} shape {value.shape} != {shape}")
                if not np.isfinite(value).all() or np.any(value < 0.0):
                    raise OfflineChildContractError(
                        f"source moment {choice} is non-finite or negative")
                result[target] = np.array(value, copy=True, order="C")
                carried.append(target)
                break

    # Morrison hail reclassification carries its graupel moments to hail.
    if source_mp == 10 and morr_rimed_ice == 1:
        if "qng" in result and "qnh" not in result:
            result["qnh"] = result.pop("qng")
        if "qvolg" in result and "qvolh" not in result:
            result["qvolh"] = result.pop("qvolg")

    required_by_mass = {
        "qc": "qndrop", "qr": "qnr", "qi": "qni", "qs": "qns",
        "qg": "qng", "qh": "qnh",
    }
    needs = []
    threshold = float(active_mass_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise OfflineChildContractError(
            "active_mass_threshold must be finite and non-negative")
    for mass_name, number_name in required_by_mass.items():
        active = bool(np.any(result[mass_name] > np.float32(threshold)))
        if number_name not in result:
            if active:
                needs.append(number_name)
            else:
                result[number_name] = np.zeros(shape, dtype=np.float32)
    if needs:
        if diagnose_missing is None:
            raise MomentDiagnosisRequired(
                "active NSSL categories require official calcnfromq diagnosis "
                f"for {sorted(needs)}")
        diagnosed = dict(diagnose_missing(MappingProxyType(result), tuple(needs)))
        for name in needs:
            if name not in diagnosed:
                raise MomentDiagnosisRequired(
                    f"calcnfromq did not return required moment {name}")
            value = np.asarray(diagnosed[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
                raise OfflineChildContractError(
                    f"diagnosed {name} is non-finite, negative, or wrong-shaped")
            result[name] = np.ascontiguousarray(value)

    if "qvolg" not in result:
        result["qvolg"] = np.ascontiguousarray(result["qg"] / np.float32(700.0))
    if "qvolh" not in result:
        result["qvolh"] = np.ascontiguousarray(result["qh"] / np.float32(900.0))
    # WRF NSSL calcnfromq homogeneous background, in # kg-1.
    qnn_background = np.float32(0.5e9 / 1.225)
    result["qnn"] = np.maximum(
        np.float32(0.0), qnn_background - result["qndrop"]).astype(np.float32)

    unknown = set(result) - set(_NSSL_FIELDS)
    missing = set(_NSSL_FIELDS) - set(result)
    if unknown or missing:
        raise OfflineChildContractError(
            f"internal NSSL conversion inventory drift: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}")
    receipt = MappingProxyType({
        "source_mp_physics": source_mp,
        "target_mp_physics": 18,
        "category_mapping": category_mapping,
        "carried_source_moments": tuple(sorted(carried)),
        "diagnosed_target_moments": tuple(sorted(needs)),
        "qnn_background_number_per_kg": float(qnn_background),
        "graupel_init_density_kg_m3": 700.0,
        "hail_init_density_kg_m3": 900.0,
        "active_mass_threshold": threshold,
    })
    return MappingProxyType({name: result[name] for name in _NSSL_FIELDS}), receipt


#: Land/soil identity attributes a child surface source must declare.
#: Defaults here would silently rebind category semantics (water index,
#: ice index) across landuse tables, so they are required evidence.
_SURFACE_IDENTITY_ATTRS = ("MMINLU", "ISWATER", "ISLAKE", "ISICE")

#: Child-grid surface fields.  ``required`` is the minimum honest warm
#: start for a land-surface + surface-layer child; ``optional`` fields
#: are carried when present and receipted either way.  All arrays are
#: read on the EXACT child grid -- like WRF's ``ndown``, gpuwm's offline
#: child takes its static/soil identity from a child-grid initialization
#: file and replaces only the meteorology from the parent archive.
_SURFACE_REQUIRED_FIELDS = (
    "LU_INDEX", "LANDMASK", "ISLTYP", "TSK", "TMN", "VEGFRA",
    "TSLB", "SMOIS", "SNOW",
)
_SURFACE_OPTIONAL_FIELDS = (
    "SH2O", "SNOWH", "SNOWC", "SEAICE", "XICE", "PBLH", "UST",
    "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAT", "XLONG",
)
_SURFACE_CATEGORY_FIELDS = frozenset({"LU_INDEX", "ISLTYP"})
_SURFACE_SOIL_FIELDS = frozenset({"TSLB", "SMOIS", "SH2O"})


@dataclass(frozen=True)
class ChildSurfaceState:
    """Surface/soil warm-start state read from one child-grid file."""

    path: Path
    fields: Mapping[str, np.ndarray]
    identity: Mapping[str, object]
    receipt: Mapping[str, object]


def read_child_surface_state(
        path: str | Path, *, child_ny: int, child_nx: int,
        num_soil_layers: int) -> ChildSurfaceState:
    """Read surface/soil warm-start fields from an exact-child-grid file.

    Accepts a stock-WRF ``wrfinput``/history file or a gpuwm history
    file whose mass grid is exactly the child's.  This is the offline
    analogue of WRF ``ndown``'s requirement that the child's own
    ``wrfinput`` (from ``real.exe``) supplies static and soil state --
    downscaling replaces the meteorology, never the land identity.
    """

    path = Path(path)
    fields: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(path) as dataset:
        for name, expected in (("south_north", int(child_ny)),
                               ("west_east", int(child_nx))):
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing surface-grid dimension {name!r}")
            actual = len(dataset.dimensions[name])
            if actual != expected:
                raise OfflineChildContractError(
                    f"{path} {name}={actual} does not match the child "
                    f"grid {name}={expected}; the surface source must be "
                    "on the EXACT child grid")
        if "soil_layers_stag" in dataset.dimensions:
            soil = len(dataset.dimensions["soil_layers_stag"])
            if soil != int(num_soil_layers):
                raise OfflineChildContractError(
                    f"{path} soil_layers_stag={soil} does not match the "
                    f"child LSM's {num_soil_layers} soil layers")
        elif int(num_soil_layers) > 0:
            raise OfflineChildContractError(
                f"{path} has no soil_layers_stag dimension; the child's "
                "land-surface scheme requires child-grid soil state")
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes "
                f"{missing_attrs}; category semantics cannot be assumed")
        identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        missing = [name for name in _SURFACE_REQUIRED_FIELDS
                   if name not in dataset.variables]
        if missing:
            raise OfflineChildContractError(
                f"{path} lacks required child surface fields {missing}")
        for name in _SURFACE_REQUIRED_FIELDS + _SURFACE_OPTIONAL_FIELDS:
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            if value.ndim and value.shape[0] == 1 and (
                    dataset.variables[name].dimensions[:1] == ("Time",)):
                value = value[0]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise OfflineChildContractError(
                    f"{path}/{name} contains NaN or infinity")
            expected_shape = (
                (int(num_soil_layers), int(child_ny), int(child_nx))
                if name in _SURFACE_SOIL_FIELDS
                else (int(child_ny), int(child_nx)))
            if tuple(value.shape) != expected_shape:
                raise OfflineChildContractError(
                    f"{path}/{name} shape {tuple(value.shape)} != "
                    f"{expected_shape}")
            if name in _SURFACE_CATEGORY_FIELDS and not np.array_equal(
                    value, np.rint(value)):
                raise OfflineChildContractError(
                    f"{path}/{name} carries non-integer categories; a "
                    "smoothed/interpolated category field is not a valid "
                    "surface identity")
            fields[name] = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = MappingProxyType({
        "path": str(path.resolve()),
        "sha256": digest,
        "identity": dict(identity),
        "carried_fields": tuple(sorted(fields)),
        "policy": "child-grid-surface-source; downscaling replaces "
                  "meteorology, land/soil identity comes from the child's "
                  "own initialization (ndown-equivalent contract)",
    })
    return ChildSurfaceState(
        path=path.resolve(), fields=MappingProxyType(fields),
        identity=MappingProxyType(identity), receipt=receipt)


@dataclass(frozen=True)
class OfflineChildPlacement:
    """Fixed child placement inside one archived parent grid.

    Starts retain WRF's one-based namelist semantics.  Construction runs the
    exact SINT stencil coverage gate for mass, x-staggered, and y-staggered
    fields, so an invalid footprint fails before any parent data are read.
    """

    parent_nx: int
    parent_ny: int
    child_nx: int
    child_ny: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int

    def __post_init__(self) -> None:
        values = (
            self.parent_nx, self.parent_ny, self.child_nx, self.child_ny,
            self.parent_grid_ratio, self.i_parent_start, self.j_parent_start,
        )
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise OfflineChildContractError(
                "offline-child placement values must be integers")
        if min(self.parent_nx, self.parent_ny, self.child_nx, self.child_ny) < 1:
            raise OfflineChildContractError(
                "offline-child parent/child extents must be positive")
        if self.parent_grid_ratio < 1:
            raise OfflineChildContractError("parent_grid_ratio must be >= 1")
        # Coverage checks include SINT's +-2 donor stencil.
        for stagger in ("", "x", "y"):
            self.registration(stagger, wrapper="bdy")

    def registration(self, stagger: str, *, wrapper: str):
        return register_nest(
            nri=self.parent_grid_ratio, nrj=self.parent_grid_ratio,
            i_parent_start=self.i_parent_start,
            j_parent_start=self.j_parent_start,
            child_nx=self.child_nx, child_ny=self.child_ny,
            parent_nx=self.parent_nx, parent_ny=self.parent_ny,
            stagger=stagger, wrapper=wrapper,
        )


@dataclass(frozen=True)
class InterpolatedBoundarySnapshot:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class OfflineBoundaryResult:
    boundaries: LateralBoundaries
    frame_receipts: tuple[Mapping[str, object], ...]
    preparation_seconds: float


@dataclass(frozen=True)
class InterpolatedInitialState:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    microphysics: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


def _read_record(dataset, name: str, *, required: bool = True):
    if name not in dataset.variables:
        if required:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())} is missing required field {name}")
        return None
    value = np.asarray(dataset.variables[name][:])
    if value.ndim and dataset.variables[name].dimensions[:1] == ("Time",):
        if value.shape[0] != 1:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())}/{name} needs exactly one Time record")
        value = value[0]
    if value.dtype.kind not in "fiu":
        raise OfflineChildContractError(
            f"{Path(dataset.filepath())}/{name} is not numeric")
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise OfflineChildContractError(
            f"{Path(dataset.filepath())}/{name} contains NaN or infinity")
    return value


def _backend_array(value, backend: str):
    if backend == "cpu":
        return np.ascontiguousarray(value, dtype=np.float32)
    if backend != "cuda":
        raise OfflineChildContractError(
            f"offline-child backend must be cpu or cuda, got {backend!r}")
    import cupy as cp
    return cp.asarray(value, dtype=cp.float32)


def _to_host(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float32)
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(value), dtype=np.float32)


def _transported_source_fields(source_mp_physics: int) -> tuple[str, ...]:
    source_mp = int(source_mp_physics)
    names = ["qv", "qc", "qr"]
    if source_mp in {6, 8, 10}:
        names += ["qi", "qs", "qg"]
    if source_mp == 8:
        names += ["nr", "ni"]
    elif source_mp == 10:
        names += ["nr", "ni", "ns", "ng"]
    elif source_mp == 18:
        names = list(_NSSL_FIELDS)
    elif source_mp != 0:
        raise OfflineChildContractError(
            f"unsupported parent mp_physics={source_mp}")
    return tuple(names)


def _resolve_source_physics(
        source_mp_physics: int | None,
        physics_binding: ParentPhysicsBinding | None,
        morr_rimed_ice: int | None,
) -> tuple[int, int | None]:
    if physics_binding is not None:
        if (source_mp_physics is not None
                and int(source_mp_physics) != int(physics_binding.mp_physics)):
            raise OfflineChildContractError(
                "declared source mp_physics conflicts with companion binding")
        source_mp_physics = int(physics_binding.mp_physics)
        if physics_binding.morr_rimed_ice is not None:
            if (morr_rimed_ice is not None
                    and int(morr_rimed_ice)
                    != int(physics_binding.morr_rimed_ice)):
                raise OfflineChildContractError(
                    "declared morr_rimed_ice conflicts with companion binding")
            morr_rimed_ice = int(physics_binding.morr_rimed_ice)
    if source_mp_physics is None:
        raise OfflineChildContractError(
            "source physics must be bound from a companion setup record")
    source_mp_physics = int(source_mp_physics)
    if source_mp_physics not in {6, 8, 10, 18}:
        raise OfflineChildContractError(
            f"unsupported parent mp_physics={source_mp_physics}")
    return source_mp_physics, morr_rimed_ice


def _raw_parent_state(dataset, source_mp_physics: int):
    raw = {
        name: _read_record(dataset, name)
        for name in (
            "T", "U", "V", "W", "PH", "MU", "MUB", "MAPFAC_M",
            "MAPFAC_U", "MAPFAC_V", "ZNU", "ZNW", "P_TOP",
        )
    }
    moisture = {}
    wrf_mapping = (_NSSL_WRF_TO_STATE if int(source_mp_physics) == 18
                   else _WRF_TO_STATE)
    inverse = {state_name: wrf_name for wrf_name, state_name in wrf_mapping.items()}
    missing = []
    for name in _transported_source_fields(source_mp_physics):
        wrf_name = inverse.get(name)
        if wrf_name is None:
            missing.append(name)
            continue
        moisture[name] = _read_record(dataset, wrf_name)
    if missing:
        raise OfflineChildContractError(
            "history reader has no bound WRF variable mapping for parent "
            f"mp_physics={source_mp_physics} fields {missing}")
    return raw, moisture


def _vertical_coefficients(raw, dataset):
    znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
    znu = np.asarray(raw["ZNU"], dtype=np.float64).reshape(-1)
    if znw.size != znu.size + 1 or not np.all(np.diff(znw) < 0.0):
        raise OfflineChildContractError("parent ZNU/ZNW are not one valid eta grid")
    p_top_values = np.asarray(raw["P_TOP"], dtype=np.float64).reshape(-1)
    if p_top_values.size != 1:
        raise OfflineChildContractError("parent P_TOP must be scalar")
    hybrid_opt = int(getattr(dataset, "HYBRID_OPT", 2))
    etac = float(getattr(dataset, "ETAC", 0.2))
    from gpuwm.core.constants import P0
    coeffs = compute_hybrid_coeffs(
        znw, hybrid_opt, etac, float(P0), float(p_top_values[0]))
    return coeffs, hybrid_opt, etac, float(p_top_values[0])


_INITIAL_FIELD_STAGGER = MappingProxyType({
    "U": "x", "V": "y", "MAPFAC_U": "x", "MAPFAC_V": "y",
    "XLAT_U": "x", "XLONG_U": "x", "XLAT_V": "y", "XLONG_V": "y",
})
_INITIAL_CORE_FIELDS = (
    "T", "U", "V", "W", "PH", "MU", "PHB", "MUB", "HGT", "P", "PB",
    "PSFC", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG",
)


def interpolate_parent_initial_state(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
) -> InterpolatedInitialState:
    """SINT one archived parent state into a standalone child cold start.

    This first executable mode deliberately inherits SINT parent terrain and
    base state.  A later static-geography join may replace/blend those fields,
    but it must run the existing ``blend_terrain``/``adjust_tempqv`` contract;
    this function never claims a high-resolution terrain adjustment happened.
    """

    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    started = time.perf_counter()
    with netCDF4.Dataset(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics))
        for name in _INITIAL_CORE_FIELDS:
            if name not in raw:
                raw[name] = _read_record(dataset, name)
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
    registrations = {
        "": placement.registration("", wrapper="interp"),
        "x": placement.registration("x", wrapper="interp"),
        "y": placement.registration("y", wrapper="interp"),
    }
    constants = {"ZNU", "ZNW", "P_TOP"}
    interpolated = {}
    for name in _INITIAL_CORE_FIELDS:
        value = _backend_array(raw[name], backend)
        interpolated[name] = sint(
            value, registrations[_INITIAL_FIELD_STAGGER.get(name, "")])
    source_mixing = {
        name: sint(_backend_array(value, backend), registrations[""])
        for name, value in moisture.items()
    }
    conversion_receipt = None
    if target_mp == 18:
        mapped, conversion_receipt = map_microphysics_to_nssl18(
            {name: _to_host(value) for name, value in source_mixing.items()},
            source_mp_physics=int(source_mp_physics),
            morr_rimed_ice=morr_rimed_ice,
            diagnose_missing=diagnose_missing,
        )
        source_mixing = mapped
    elif target_mp != int(source_mp_physics):
        raise OfflineChildContractError(
            "cross-physics offline initialization currently targets NSSL mp18")
    fields = {name: _to_host(value) for name, value in interpolated.items()}
    fields.update({name: np.array(raw[name], copy=True, dtype=np.float32)
                   for name in constants})
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        "terrain_policy": "sint-parent-inherited",
        "spinup_policy": (
            "new-standalone-child; source held physics tendencies and "
            "scheduler state are not inherited"),
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedInitialState(
        info.valid_time, MappingProxyType(fields),
        MappingProxyType({name: _to_host(value)
                          for name, value in source_mixing.items()}), receipt)


def _base_from_interpolated_initial(initial: InterpolatedInitialState,
                                    cfg):
    fields = initial.fields
    znw = np.asarray(fields["ZNW"], dtype=np.float64).reshape(-1)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=int(initial.receipt["hybrid_opt"]),
        etac=float(initial.receipt["etac"]), eta_levels=znw)
    p_top = float(np.asarray(fields["P_TOP"]).reshape(-1)[0])
    finalize_vertical_coord(coord, p_top)
    mub = np.asarray(fields["MUB"], dtype=np.float64)
    pb = np.asarray(fields["PB"], dtype=np.float64)
    phb = np.asarray(fields["PHB"], dtype=np.float64)
    if pb.shape != (cfg.nz, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PB shape {pb.shape} does not match child")
    if phb.shape != (cfg.nz + 1, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PHB shape {phb.shape} does not match child")
    delta_phi = phb[1:] - phb[:-1]
    if cfg.hypsometric_opt == 1:
        denominator = (-coord.dnw[:, None, None]
                       * (coord.c1h[:, None, None] * mub[None]
                          + coord.c2h[:, None, None]))
    elif cfg.hypsometric_opt == 2:
        pfu = (coord.c3f[1:, None, None] * mub[None]
               + coord.c4f[1:, None, None] + p_top)
        pfd = (coord.c3f[:-1, None, None] * mub[None]
               + coord.c4f[:-1, None, None] + p_top)
        phm = (coord.c3h[:, None, None] * mub[None]
               + coord.c4h[:, None, None] + p_top)
        denominator = phm * np.log(pfd / pfu)
    else:
        raise OfflineChildContractError(
            f"unsupported hypsometric_opt={cfg.hypsometric_opt}")
    if np.any(denominator <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base has non-positive hydrostatic increments")
    alb = delta_phi / denominator
    if not np.isfinite(alb).all() or np.any(alb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent PHB/PB imply invalid base inverse density")
    from gpuwm.core import constants as c
    thb = alb * pb / (c.RD * (pb / c.P0) ** c.RCP)
    if not np.isfinite(thb).all() or np.any(thb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base implies invalid base potential temperature")
    return coord, BaseState(
        mub=mub, p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
        terrain_z=np.asarray(fields["HGT"], dtype=np.float64))


def build_offline_child_domain_state(
        initial: InterpolatedInitialState, cfg, *, array_module=None):
    """Upload an interpolated parent-only cold start into ``DomainState``.

    The returned state has complete dynamics, moisture, map factors, and RK
    time-t copies. Physics and lateral tables are attached by their existing
    production APIs so callers can select the new child physics explicitly.
    """

    if not cfg.specified or cfg.nested:
        raise OfflineChildContractError(
            "standalone offline child requires specified=True and nested=False")
    if initial.receipt.get("source_physics_binding") is None:
        raise OfflineChildContractError(
            "standalone offline child requires authoritative parent physics "
            "evidence from a setup/namelist/restart companion")
    if not cfg.moist:
        raise OfflineChildContractError(
            "offline child with microphysics requires moist=True")
    if not cfg.terrain_opt:
        raise OfflineChildContractError(
            "parent-inherited offline child base requires terrain_opt != 0")
    if int(cfg.hybrid_opt) != int(initial.receipt["hybrid_opt"]):
        raise OfflineChildContractError(
            f"child hybrid_opt={cfg.hybrid_opt} differs from archived parent "
            f"{initial.receipt['hybrid_opt']}")
    # History files persist ETAC as FP32, while RunConfig retains the user's
    # decimal as a Python float.  Compare at the precision of the archived
    # evidence so an exact FP32 round-trip (for example 0.2) is accepted.
    if not np.isclose(
            float(cfg.etac), float(initial.receipt["etac"]),
            rtol=0.0,
            atol=abs(float(np.spacing(np.float32(initial.receipt["etac"]))))):
        raise OfflineChildContractError(
            f"child etac={cfg.etac} differs from archived parent "
            f"{initial.receipt['etac']}")
    target_mp = int(initial.receipt["target_mp_physics"])
    if int(cfg.mp_physics) != target_mp:
        raise OfflineChildContractError(
            f"child cfg mp_physics={cfg.mp_physics} != prepared target {target_mp}")
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import DomainState
    coord, base = _base_from_interpolated_initial(initial, cfg)
    state = DomainState(cfg, array_module=array_module)
    state.load_base(coord, base)
    xp = np if array_module is np else __import__("cupy")

    def assign(name, value):
        target = getattr(state, name)
        host = np.asarray(value, dtype=np.float32)
        if tuple(target.shape) != tuple(host.shape):
            raise OfflineChildContractError(
                f"initial {name} shape {host.shape} != state {target.shape}")
        target[...] = xp.asarray(host, dtype=xp.float32)

    for state_name, wrf_name in (
            ("u", "U"), ("v", "V"), ("w", "W"),
            ("php", "PH"), ("mup", "MU")):
        assign(state_name, initial.fields[wrf_name])
    total_theta = np.asarray(initial.fields["T"], dtype=np.float32) + np.float32(300.0)
    state.thp[...] = xp.asarray(
        total_theta - np.asarray(base.thb, dtype=np.float32), dtype=xp.float32)
    state.set_map_coriolis(
        initial.fields["MAPFAC_M"], initial.fields["MAPFAC_U"],
        initial.fields["MAPFAC_V"], initial.fields["F"], initial.fields["E"],
        sina=initial.fields["SINALPHA"], cosa=initial.fields["COSALPHA"])
    for name, value in initial.microphysics.items():
        target = getattr(state, name, None)
        if target is None:
            raise OfflineChildContractError(
                f"child DomainState has no target microphysics field {name!r}")
        assign(name, value)
    for current, seed in (
            ("u", "u0"), ("v", "v0"), ("w", "w0"),
            ("thp", "thp0"), ("php", "php0"), ("mup", "mup0"),
            ("qv", "qv0"), ("qc", "qc0"), ("qr", "qr0"),
            ("qi", "qi0"), ("qs", "qs0"), ("qg", "qg0"),
            ("nr", "nr0"), ("ni", "ni0"), ("ns", "ns0"),
            ("ng", "ng0")):
        source = getattr(state, current, None)
        target = getattr(state, seed, None)
        if source is not None and target is not None:
            target[...] = source
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _couple_parent(raw, moisture, coeffs, backend: str):
    arrays = {name: _backend_array(value, backend) for name, value in raw.items()}
    scalars = {name: _backend_array(value, backend)
               for name, value in moisture.items()}
    total_mu = arrays["MUB"] + arrays["MU"]
    mux = mu_at_u_faces(total_mu)
    muy = mu_at_v_faces(total_mu)
    mux[:, 0] = total_mu[:, 0]
    mux[:, -1] = total_mu[:, -1]
    muy[0, :] = total_mu[0, :]
    muy[-1, :] = total_mu[-1, :]
    xp = np if backend == "cpu" else __import__("cupy")
    c1h = xp.asarray(coeffs["c1h"], dtype=xp.float32)[:, None, None]
    c2h = xp.asarray(coeffs["c2h"], dtype=xp.float32)[:, None, None]
    c1f = xp.asarray(coeffs["c1f"], dtype=xp.float32)[:, None, None]
    c2f = xp.asarray(coeffs["c2f"], dtype=xp.float32)[:, None, None]
    chm = c1h * total_mu[None] + c2h
    chf = c1f * total_mu[None] + c2f
    result = {
        "u": (c1h * mux[None] + c2h) * arrays["U"] / arrays["MAPFAC_U"][None],
        "v": (c1h * muy[None] + c2h) * arrays["V"] / arrays["MAPFAC_V"][None],
        "w": chf * arrays["W"] / arrays["MAPFAC_M"][None],
        "theta": chm * arrays["T"],
        "phi": chf * arrays["PH"],
        "mu": arrays["MU"][None],
    }
    result.update({name: chm * value for name, value in scalars.items()})
    return result, arrays, chm


def interpolate_parent_boundary_snapshot(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
) -> InterpolatedBoundarySnapshot:
    """Conservatively SINT one archived parent state onto a child frame.

    Dynamics and source scalars are coupled on the parent before SINT, matching
    online ``bdy_interp1`` spatial semantics.  When target physics differs,
    coupled source scalars are uncoupled on the child, converted there, and
    recoupled in the target inventory.
    """

    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    started = time.perf_counter()
    with netCDF4.Dataset(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics))
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
    coupled, raw_device, _ = _couple_parent(raw, moisture, coeffs, backend)
    registrations = {
        "": placement.registration("", wrapper="bdy"),
        "x": placement.registration("x", wrapper="bdy"),
        "y": placement.registration("y", wrapper="bdy"),
    }
    stagger = {"u": "x", "v": "y"}
    interpolated = {
        name: sint(value, registrations[stagger.get(name, "")])
        for name, value in coupled.items()
    }
    conversion_receipt = None
    if target_mp != int(source_mp_physics):
        if target_mp != 18:
            raise OfflineChildContractError(
                "cross-physics offline forcing currently targets NSSL mp18")
        xp = np if backend == "cpu" else __import__("cupy")
        child_mub = sint(raw_device["MUB"], registrations[""])
        child_mup = interpolated["mu"][0]
        child_mu = child_mub + child_mup
        c1h = xp.asarray(coeffs["c1h"], dtype=xp.float32)[:, None, None]
        c2h = xp.asarray(coeffs["c2h"], dtype=xp.float32)[:, None, None]
        child_chm = c1h * child_mu[None] + c2h
        source_mixing = {
            name: _to_host(interpolated.pop(name) / child_chm)
            for name in tuple(moisture)
        }
        mapped, conversion_receipt = map_microphysics_to_nssl18(
            source_mixing, source_mp_physics=int(source_mp_physics),
            morr_rimed_ice=morr_rimed_ice,
            diagnose_missing=diagnose_missing,
        )
        child_chm_host = _to_host(child_chm)
        interpolated.update({
            name: child_chm_host * value for name, value in mapped.items()
        })
    fields = MappingProxyType({name: _to_host(value)
                               for name, value in interpolated.items()})
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_kind": info.source_kind,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        "field_inventory": tuple(sorted(fields)),
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedBoundarySnapshot(info.valid_time, fields, receipt)


def build_offline_lateral_boundaries(
        contract: ParentHistoryContract, placement: OfflineChildPlacement, *,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
        spec_bdy_width: int = 5, spec_zone: int = 1, relax_zone: int = 4,
) -> OfflineBoundaryResult:
    """Stream parent frames into compact child lateral value/tendency strips."""

    if contract.source_mp_physics is None:
        raise OfflineChildContractError(
            "offline boundary generation requires source mp_physics bound "
            "from a companion setup/namelist/manifest; inventory inference is "
            "advisory only")
    if contract.physics_binding is None:
        raise OfflineChildContractError(
            "offline boundary generation requires authoritative companion "
            "setup/namelist/restart evidence, not a bare scheme integer")
    started = time.perf_counter()
    side_names = ("west", "east", "south", "north")
    previous = None
    intervals = []
    receipts = []
    origin = contract.start_time
    for frame in contract.frames:
        snapshot = interpolate_parent_boundary_snapshot(
            frame.path, placement,
            source_mp_physics=contract.source_mp_physics,
            physics_binding=contract.physics_binding,
            target_mp_physics=target_mp_physics,
            morr_rimed_ice=morr_rimed_ice, backend=backend,
            diagnose_missing=diagnose_missing,
        )
        sides = {
            side: extract_lateral_side(snapshot.fields, side, spec_bdy_width)
            for side in side_names
        }
        if previous is not None:
            previous_time, previous_sides = previous
            intervals.append(build_lateral_interval_from_sides(
                previous_sides, sides,
                start_seconds=(previous_time - origin).total_seconds(),
                end_seconds=(snapshot.valid_time - origin).total_seconds(),
            ))
        previous = (snapshot.valid_time, sides)
        receipts.append(snapshot.receipt)
    boundaries = LateralBoundaries(
        tuple(intervals), int(spec_bdy_width), int(spec_zone), int(relax_zone))
    return OfflineBoundaryResult(
        boundaries=boundaries, frame_receipts=tuple(receipts),
        preparation_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ChildSurfaceState",
    "InterpolatedBoundarySnapshot", "InterpolatedInitialState",
    "OfflineBoundaryResult",
    "OfflineChildPlacement",
    "read_child_surface_state",
    "MomentDiagnosisRequired", "OfflineChildContractError",
    "ParentHistoryContract", "ParentHistoryFrame", "ParentPhysicsBinding",
    "bind_parent_physics_from_gpuwm_restart",
    "bind_parent_physics_from_wrf_namelist",
    "build_offline_child_domain_state", "build_offline_lateral_boundaries",
    "inspect_parent_history_frame", "interpolate_parent_boundary_snapshot",
    "interpolate_parent_initial_state", "map_microphysics_to_nssl18",
    "read_parent_microphysics", "validate_parent_history",
]
