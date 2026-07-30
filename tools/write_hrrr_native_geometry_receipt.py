#!/usr/bin/env python3
"""Convert a verified HRRR static receipt to the common WRF contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gpuwm.ingest.hrrr_target import (
    HrrrTargetDomain,
    load_hrrr_target_domain,
)
from gpuwm.native_wrf_contract import (
    native_geometry_contract,
    write_native_geometry_receipt,
)


_HRRR_STATIC_SCHEMAS = frozenset({
    "gpuwm-native-hrrr-static-500x500-v1",
    "gpuwm-native-hrrr-static-v2",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_cfg(target: HrrrTargetDomain) -> SimpleNamespace:
    # One definition, shared with the static builder that seals the document
    # this module verifies.
    return target.contract_cfg()


def write_hrrr_native_geometry_receipt(
        *, static_cache: Path, hrrr_static_receipt: Path,
        output: Path, domain_spec: Path | None = None,
) -> dict[str, object]:
    """Verify the rich HRRR receipt and emit the common geometry receipt."""

    static_cache = Path(static_cache)
    hrrr_static_receipt = Path(hrrr_static_receipt)
    output = Path(output)
    target = load_hrrr_target_domain(domain_spec)
    payload = json.loads(hrrr_static_receipt.read_text(encoding="utf-8"))
    if (payload.get("schema") not in _HRRR_STATIC_SCHEMAS
            or payload.get("status") != "PASS"):
        raise ValueError("unrecognized or non-PASS HRRR static receipt")
    if payload.get("schema") == "gpuwm-native-hrrr-static-v2":
        if payload.get("target_domain_sha256") != target.identity_sha256():
            raise ValueError("HRRR static receipt target identity mismatch")
        if payload.get("target_domain") != target.to_payload():
            raise ValueError("HRRR static receipt target document mismatch")
    elif target != HrrrTargetDomain.legacy_500x500():
        raise ValueError(
            "legacy 500x500 HRRR static receipt cannot bind another target")

    cache = payload.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("HRRR static receipt lacks a cache binding")
    observed_cache = {
        "bytes": static_cache.stat().st_size,
        "sha256": _sha256(static_cache),
    }
    if (cache.get("bytes") != observed_cache["bytes"]
            or cache.get("sha256") != observed_cache["sha256"]):
        raise ValueError("HRRR static receipt does not bind the static cache")
    receipt_path = Path(str(cache.get("path", "")))
    if receipt_path.is_absolute():
        if receipt_path.resolve() != static_cache.resolve():
            raise ValueError("HRRR static receipt names another static cache")
    elif receipt_path.name != static_cache.name:
        raise ValueError("HRRR static receipt cache file name mismatch")

    grid = target.grid()
    expected_geometry = native_geometry_contract(grid, _target_cfg(target))
    if payload.get("geometry") != expected_geometry:
        raise ValueError("HRRR static receipt geometry differs from its target")
    return write_native_geometry_receipt(
        output, grid, _target_cfg(target), static_cache)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-cache", type=Path, required=True)
    parser.add_argument("--hrrr-static-receipt", type=Path, required=True)
    parser.add_argument("--domain-spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = write_hrrr_native_geometry_receipt(
        static_cache=args.static_cache,
        hrrr_static_receipt=args.hrrr_static_receipt,
        output=args.output,
        domain_spec=args.domain_spec,
    )
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output.resolve()),
        "static_cache_sha256": receipt["cache"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
