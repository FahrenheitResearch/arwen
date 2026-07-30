"""Runtime ownership for canonical classic-Thompson coefficient tables.

The table parser in :mod:`gpuwm.core.thompson_contract` proves the bytes on
disk.  This module closes the second half of that boundary: it checks the
complete in-memory inventory, uploads every record without a precision or
layout conversion, verifies a device round trip, and retains one process-local
owner per CUDA device.  Merely importing this module does not make
``mp_physics=8`` selectable; the production scheme remains fail-closed until
the simultaneous process driver and coupled gates are complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.core.thompson_contract import (
    AUXILIARY_TABLE_RECORDS,
    CLASSIC_TABLE_ASSETS,
    GENERATED_TABLE_FILES,
    ClassicTableSet,
    TableAsset,
    TableRecord,
    load_validated_classic_tables,
)


RAIN_SNOW_TABLE_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
)
RAIN_GRAUPEL_TABLE_NAMES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
)
RAIN_FREEZING_TABLE_NAMES = (
    "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz",
)


@dataclass(frozen=True)
class DeviceClassicColdSourceTables:
    """Canonical device-table groups consumed by the fused cold source."""

    ice_deposition_partition: object
    ice_to_snow_mass: object
    ice_to_snow_number: object
    rain_snow_tables: tuple[object, ...]
    rain_graupel_tables: tuple[object, ...]
    rain_freezing_tables: tuple[object, ...]
    rain_cloud_efficiency: object
    cloud_freezing_tables: tuple[object, object]
    identity_sha256: str


def _classic_records() -> tuple[TableRecord, ...]:
    records = tuple(
        record
        for group in GENERATED_TABLE_FILES.values()
        for record in group
    ) + AUXILIARY_TABLE_RECORDS
    names = tuple(record.name for record in records)
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate records in the Thompson table contract")
    return records


def _payload_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="F")).hexdigest()


def _synchronize(backend) -> None:
    cuda = getattr(backend, "cuda", None)
    stream = getattr(cuda, "Stream", None)
    null = getattr(stream, "null", None)
    synchronize = getattr(null, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _to_host_fortran(backend, value) -> np.ndarray:
    asnumpy = getattr(backend, "asnumpy", None)
    if asnumpy is None:
        result = np.array(value, dtype=np.float64, order="F", copy=True)
    else:
        try:
            result = asnumpy(value, order="F")
        except TypeError:
            result = asnumpy(value)
        result = np.array(result, dtype=np.float64, order="F", copy=False)
    return result


@dataclass(frozen=True)
class DeviceClassicTableSet:
    """One verified, immutable mapping of canonical tables on a device."""

    root: Path
    arrays: Mapping[str, object]
    payload_bytes: int
    array_sha256: Mapping[str, str]
    identity_json: str
    identity_sha256: str
    device_id: int | None
    upload_seconds: float
    verification_seconds: float
    roundtrip_verified: bool

    def __getattr__(self, name: str):
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def identity(self) -> dict[str, object]:
        """Return a detached JSON-compatible restart identity."""
        return json.loads(self.identity_json)

    @property
    def cold_source_tables(self) -> DeviceClassicColdSourceTables:
        """Return the exact ordered groups for the fused cold-source ABI."""
        return DeviceClassicColdSourceTables(
            ice_deposition_partition=self.arrays["tpi_ide"],
            ice_to_snow_mass=self.arrays["tps_iaus"],
            ice_to_snow_number=self.arrays["tni_iaus"],
            rain_snow_tables=tuple(
                self.arrays[name] for name in RAIN_SNOW_TABLE_NAMES),
            rain_graupel_tables=tuple(
                self.arrays[name] for name in RAIN_GRAUPEL_TABLE_NAMES),
            rain_freezing_tables=tuple(
                self.arrays[name] for name in RAIN_FREEZING_TABLE_NAMES),
            rain_cloud_efficiency=self.arrays["t_Efrw"],
            cloud_freezing_tables=(
                self.arrays["tpi_qcfz"], self.arrays["tni_qcfz"]),
            identity_sha256=self.identity_sha256,
        )


def _upload_table_set(
        table_set: ClassicTableSet, backend, *,
        records: tuple[TableRecord, ...],
        expected_assets: tuple[TableAsset, ...],
        device_id: int | None,
        verify_roundtrip: bool,
        ) -> DeviceClassicTableSet:
    """Upload a table set after enforcing an exact supplied contract.

    ``records`` and ``expected_assets`` are explicit so small CPU unit tests can
    exercise the same validator without allocating the 380 MB production set.
    The public loader below always supplies the canonical WRF contract.
    """
    if table_set.assets != expected_assets:
        raise ValueError("Thompson table-set asset identity is not canonical")

    expected = {record.name: record for record in records}
    actual_names = set(table_set.arrays)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Thompson table inventory mismatch: missing={missing}, extra={extra}")

    host_hashes: dict[str, str] = {}
    payload_bytes = 0
    for name, record in expected.items():
        value = table_set.arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Thompson host table {name} must be a NumPy array")
        if value.shape != record.shape:
            raise ValueError(
                f"Thompson host table {name} has shape {value.shape}; "
                f"expected {record.shape}")
        if value.dtype != np.dtype(np.float64):
            raise TypeError(
                f"Thompson host table {name} has dtype {value.dtype}; "
                "expected float64")
        if not value.flags.f_contiguous:
            raise ValueError(f"Thompson host table {name} is not Fortran-contiguous")
        if value.flags.writeable:
            raise ValueError(f"Thompson host table {name} is not immutable")
        if value.nbytes != record.payload_bytes:
            raise RuntimeError(
                f"Thompson host table {name} has {value.nbytes} bytes; "
                f"expected {record.payload_bytes}")
        payload_bytes += value.nbytes
        host_hashes[name] = _payload_sha256(value)

    upload_start = perf_counter()
    device_arrays = {
        name: backend.asarray(
            table_set.arrays[name], dtype=backend.float64, order="F")
        for name in expected
    }
    _synchronize(backend)
    upload_seconds = perf_counter() - upload_start

    for name, record in expected.items():
        value = device_arrays[name]
        if value.shape != record.shape:
            raise RuntimeError(
                f"Thompson device table {name} has shape {value.shape}; "
                f"expected {record.shape}")
        if np.dtype(value.dtype) != np.dtype(np.float64):
            raise RuntimeError(
                f"Thompson device table {name} has dtype {value.dtype}; "
                "expected float64")
        if not value.flags.f_contiguous:
            raise RuntimeError(
                f"Thompson device table {name} is not Fortran-contiguous")
        if value.nbytes != record.payload_bytes:
            raise RuntimeError(
                f"Thompson device table {name} has {value.nbytes} bytes; "
                f"expected {record.payload_bytes}")

    verification_seconds = 0.0
    if verify_roundtrip:
        verify_start = perf_counter()
        for name in expected:
            returned = _to_host_fortran(backend, device_arrays[name])
            if returned.shape != table_set.arrays[name].shape:
                raise RuntimeError(
                    f"Thompson device round trip changed {name} shape")
            if not returned.flags.f_contiguous:
                raise RuntimeError(
                    f"Thompson device round trip changed {name} layout")
            returned_hash = _payload_sha256(returned)
            if returned_hash != host_hashes[name]:
                raise RuntimeError(
                    f"Thompson device round trip changed {name}: "
                    f"{returned_hash} != {host_hashes[name]}")
        _synchronize(backend)
        verification_seconds = perf_counter() - verify_start

    identity_json = json.dumps(
        table_set.identity, sort_keys=True, separators=(",", ":"))
    identity_sha256 = hashlib.sha256(identity_json.encode("ascii")).hexdigest()
    return DeviceClassicTableSet(
        root=table_set.root,
        arrays=MappingProxyType(device_arrays),
        payload_bytes=payload_bytes,
        array_sha256=MappingProxyType(host_hashes),
        identity_json=identity_json,
        identity_sha256=identity_sha256,
        device_id=device_id,
        upload_seconds=upload_seconds,
        verification_seconds=verification_seconds,
        roundtrip_verified=verify_roundtrip,
    )


_DEVICE_CACHE: dict[tuple[Path, int, int | None], DeviceClassicTableSet] = {}


def load_classic_device_tables(
        path: str | Path, *, backend=None,
        verify_roundtrip: bool = True,
        cache: bool = True,
        ) -> DeviceClassicTableSet:
    """Validate and upload every canonical classic-Thompson WRF table once."""
    if backend is None:
        import cupy as backend

    root = Path(path).resolve()
    cuda = getattr(backend, "cuda", None)
    device = getattr(cuda, "Device", None)
    device_id = int(device().id) if device is not None else None
    key = (root, id(backend), device_id)
    existing = _DEVICE_CACHE.get(key) if cache else None
    if existing is not None and (
            existing.roundtrip_verified or not verify_roundtrip):
        return existing

    # This re-reads from canonical, content-addressed files.  A caller cannot
    # smuggle a manually constructed ClassicTableSet across the trust boundary.
    table_set = load_validated_classic_tables(root)
    result = _upload_table_set(
        table_set, backend,
        records=_classic_records(),
        expected_assets=CLASSIC_TABLE_ASSETS,
        device_id=device_id,
        verify_roundtrip=verify_roundtrip,
    )
    if cache:
        _DEVICE_CACHE[key] = result
    return result


__all__ = [
    "DeviceClassicColdSourceTables",
    "DeviceClassicTableSet",
    "RAIN_FREEZING_TABLE_NAMES",
    "RAIN_GRAUPEL_TABLE_NAMES",
    "RAIN_SNOW_TABLE_NAMES",
    "load_classic_device_tables",
]
