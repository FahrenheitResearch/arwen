"""Runtime ownership for the aerosol-aware Thompson coefficient tables.

:mod:`gpuwm.core.thompson_aerosol_contract` proves the bytes on disk and
derives the gamma moments.  This module closes the second half of that
boundary for ``mp_physics=28``: it uploads every array without a precision or
layout conversion, verifies a device round trip, and retains one process-local
owner per CUDA device.

It is deliberately a sibling of :mod:`gpuwm.core.thompson_runtime` rather than
an extension of it.  The classic uploader owns four assets totalling 380 MB
whose SHA-256 pins gate the model-validated ``mp_physics=8`` trajectory; that
contract must remain untouched by this port.  An mp=28 adapter loads *both*
owners from the same table root, so ``tnccn_act`` and the process tables can
never come from different WRF builds.

The one asset uploaded here, ``CCN_ACTIVATE.BIN``, **ships with gpuwm** as
WRF v4.6.1's own ``run/`` file; it is still located per
:func:`~gpuwm.core.thompson_aerosol_contract.resolve_ccn_activation_path`, so
an override can bind the run to another copy, and its absence -- or a
byte-different table at any of those locations -- is fatal, never defaulted.
``tnc_wev``, by contrast, needs no new asset at all and is not uploaded here:
see :func:`device_drop_evaporation_number_table`.

Merely importing this module does not make ``mp_physics=28`` selectable.
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

from gpuwm.core.thompson_aerosol_contract import (
    AEROSOL_TABLE_ASSETS,
    CCN_ACTIVATION_SHAPE,
    TNC_WEV_RECORD_ORDINAL,
    TNC_WEV_SOURCE_FILE,
    TNC_WEV_TABLE_NAME,
    AerosolTableSet,
    load_validated_aerosol_tables,
    resolve_aerosol_table_root,
    resolve_ccn_activation_path,
    validate_ccn_activation_table,
)


#: The one table read from an external asset.  Everything else in the set is
#: derived at import and is small enough that its device copy is a
#: convenience, not a requirement.
CCN_ACTIVATION_TABLE_NAME = "tnccn_act"

#: Derived arrays uploaded alongside it.  The five ``ccg`` rows and the two
#: reciprocals are what turn ``nu_c`` from a compile-time 12 into a live
#: per-gridpoint index; ``g_ratio`` is the effective-radius diagnostic's
#: exact-integer table and is intentionally not interchangeable with them.
DERIVED_DEVICE_TABLE_NAMES = (
    "cce1", "cce2", "cce3", "cce4", "cce5",
    "ccg1", "ccg2", "ccg3", "ccg4", "ccg5",
    "ocg1", "ocg2", "g_ratio", "t_Nc", "Dc", "dtc", "r_c",
)

AEROSOL_DEVICE_TABLE_NAMES = (
    (CCN_ACTIVATION_TABLE_NAME,) + DERIVED_DEVICE_TABLE_NAMES
)


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
class DeviceAerosolTableSet:
    """One verified, immutable mapping of aerosol tables on a device."""

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
    #: Where CCN_ACTIVATE.BIN was read from on this host.  Diagnostic only --
    #: it is not part of ``identity_json``, because the asset is content
    #: addressed and two byte-identical copies at different paths are the same
    #: numerical setup.
    ccn_path: Path | None = None

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
    def ccn_activation_table(self) -> object:
        """The single table ``activ_ncloud`` indexes."""
        return self.arrays[CCN_ACTIVATION_TABLE_NAME]

    @property
    def gamma_moment_tables(self) -> tuple[object, ...]:
        """``cce1..5, ccg1..5, ocg1, ocg2`` in the fixed launcher order."""
        return tuple(self.arrays[name]
                     for name in DERIVED_DEVICE_TABLE_NAMES[:12])


def _upload_aerosol_table_set(
        table_set: AerosolTableSet, backend, *,
        device_id: int | None,
        verify_roundtrip: bool,
        ) -> DeviceAerosolTableSet:
    """Upload an aerosol table set after enforcing the exact contract."""
    if table_set.assets != AEROSOL_TABLE_ASSETS:
        raise ValueError(
            "aerosol Thompson table-set identity is not canonical")

    actual_names = set(table_set.arrays)
    expected_names = set(AEROSOL_DEVICE_TABLE_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"aerosol Thompson table inventory mismatch: missing={missing}, "
            f"extra={extra}")

    host_hashes: dict[str, str] = {}
    payload_bytes = 0
    for name in AEROSOL_DEVICE_TABLE_NAMES:
        value = table_set.arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"aerosol host table {name} must be a NumPy array")
        if value.dtype != np.dtype(np.float64):
            raise TypeError(
                f"aerosol host table {name} has dtype {value.dtype}; "
                "expected float64")
        if not value.flags.f_contiguous:
            raise ValueError(
                f"aerosol host table {name} is not Fortran-contiguous")
        if value.flags.writeable:
            raise ValueError(f"aerosol host table {name} is not immutable")
        payload_bytes += value.nbytes
        host_hashes[name] = _payload_sha256(value)

    ccn = table_set.arrays[CCN_ACTIVATION_TABLE_NAME]
    validate_ccn_activation_table(ccn, source=table_set.root)

    upload_start = perf_counter()
    device_arrays = {
        name: backend.asarray(
            table_set.arrays[name], dtype=backend.float64, order="F")
        for name in AEROSOL_DEVICE_TABLE_NAMES
    }
    _synchronize(backend)
    upload_seconds = perf_counter() - upload_start

    for name in AEROSOL_DEVICE_TABLE_NAMES:
        host = table_set.arrays[name]
        value = device_arrays[name]
        if value.shape != host.shape:
            raise RuntimeError(
                f"aerosol device table {name} has shape {value.shape}; "
                f"expected {host.shape}")
        if np.dtype(value.dtype) != np.dtype(np.float64):
            raise RuntimeError(
                f"aerosol device table {name} has dtype {value.dtype}; "
                "expected float64")
        if not value.flags.f_contiguous:
            raise RuntimeError(
                f"aerosol device table {name} is not Fortran-contiguous")
        if value.nbytes != host.nbytes:
            raise RuntimeError(
                f"aerosol device table {name} has {value.nbytes} bytes; "
                f"expected {host.nbytes}")
    if device_arrays[CCN_ACTIVATION_TABLE_NAME].shape != CCN_ACTIVATION_SHAPE:
        raise RuntimeError("tnccn_act lost its Fortran shape on upload")

    verification_seconds = 0.0
    if verify_roundtrip:
        verify_start = perf_counter()
        for name in AEROSOL_DEVICE_TABLE_NAMES:
            returned = _to_host_fortran(backend, device_arrays[name])
            if returned.shape != table_set.arrays[name].shape:
                raise RuntimeError(
                    f"aerosol device round trip changed {name} shape")
            if not returned.flags.f_contiguous:
                raise RuntimeError(
                    f"aerosol device round trip changed {name} layout")
            returned_hash = _payload_sha256(returned)
            if returned_hash != host_hashes[name]:
                raise RuntimeError(
                    f"aerosol device round trip changed {name}: "
                    f"{returned_hash} != {host_hashes[name]}")
        _synchronize(backend)
        verification_seconds = perf_counter() - verify_start

    identity_json = json.dumps(
        table_set.identity, sort_keys=True, separators=(",", ":"))
    identity_sha256 = hashlib.sha256(identity_json.encode("ascii")).hexdigest()
    return DeviceAerosolTableSet(
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
        ccn_path=table_set.ccn_path,
    )


_DEVICE_CACHE: dict[
    tuple[Path, Path, int, int | None], DeviceAerosolTableSet] = {}


def load_aerosol_device_tables(
        path: str | Path | None = None, *,
        ccn_path: str | Path | None = None,
        backend=None,
        verify_roundtrip: bool = True,
        cache: bool = True,
        require_classic_assets: bool = True,
        ) -> DeviceAerosolTableSet:
    """Validate and upload the aerosol-aware Thompson tables once.

    ``path`` is the shared Thompson table root (default: the environment
    override, then the packaged directory); ``ccn_path`` separately overrides
    the location of ``CCN_ACTIVATE.BIN``.  A missing blob raises
    ``MissingAerosolTableAsset`` from the contract layer.

    ``require_classic_assets`` is forwarded to the contract loader.  Leave it
    True everywhere except CPU/GPU gates on a tree whose externalized classic
    assets have not been staged.

    The cache key carries the resolved CCN path as well as the root, so
    flipping ``GPUWM_THOMPSON_CCN_ACTIVATE`` inside one process cannot be
    answered from a stale device copy.  Resolution happens before the cache
    lookup precisely so a now-absent override fails closed instead of being
    masked by an earlier successful load.
    """
    if backend is None:
        import cupy as backend

    root = resolve_aerosol_table_root(path).resolve()
    source = resolve_ccn_activation_path(ccn_path, root).resolve()
    cuda = getattr(backend, "cuda", None)
    device = getattr(cuda, "Device", None)
    device_id = int(device().id) if device is not None else None
    key = (root, source, id(backend), device_id)
    existing = _DEVICE_CACHE.get(key) if cache else None
    if existing is not None and (
            existing.roundtrip_verified or not verify_roundtrip):
        return existing

    # Re-reads from the canonical, content-addressed files.  A caller cannot
    # smuggle a manually constructed AerosolTableSet across this boundary.
    table_set = load_validated_aerosol_tables(
        root, ccn_path=source,
        require_classic_assets=require_classic_assets)
    result = _upload_aerosol_table_set(
        table_set, backend,
        device_id=device_id,
        verify_roundtrip=verify_roundtrip,
    )
    if cache:
        _DEVICE_CACHE[key] = result
    return result


def device_drop_evaporation_number_table(classic_device_tables) -> object:
    """Return the already-resident ``tnc_wev`` device array.

    mp=28's aerosol-only droplet evaporation branch (module_mp_thompson.F
    around 3447-3466) is the first consumer of ``tnc_wev`` in this codebase.
    It needs **no new asset**: ``tnc_wev`` is record 7 of
    ``thompson_aux_tables.dat``, is declared in
    ``thompson_contract.AUXILIARY_TABLE_RECORDS``, and is uploaded on every
    ``load_classic_device_tables`` call -- mp=8 has been carrying it, unread,
    since the classic port.

    This accessor exists so the aerosol launchers name that fact instead of
    reaching into ``DeviceClassicTableSet.arrays`` and quietly acquiring a
    dependency on the frozen classic runtime's internals.

    ``classic_device_tables`` is a
    ``thompson_runtime.DeviceClassicTableSet``.
    """
    arrays = getattr(classic_device_tables, "arrays", None)
    if arrays is None or TNC_WEV_TABLE_NAME not in arrays:
        raise KeyError(
            f"{TNC_WEV_TABLE_NAME} is absent from the classic Thompson device "
            f"table set; mp=28 reads it as record {TNC_WEV_RECORD_ORDINAL} of "
            f"{TNC_WEV_SOURCE_FILE} and ships no replacement asset")
    return arrays[TNC_WEV_TABLE_NAME]


__all__ = [
    "AEROSOL_DEVICE_TABLE_NAMES",
    "CCN_ACTIVATION_TABLE_NAME",
    "DERIVED_DEVICE_TABLE_NAMES",
    "DeviceAerosolTableSet",
    "device_drop_evaporation_number_table",
    "load_aerosol_device_tables",
]
