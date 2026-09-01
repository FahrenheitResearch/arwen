"""Hash-bound receipts for shadow and applied spectral steps."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .pins import PINS_SHA256, SCHEMA, canonical_hash

RECEIPT_SCHEMA = "gpuwm.regional-spectral-step-receipt/v1"


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    # newline="" pins LF: without it write_text translates every "\n" to
    # os.linesep, so the identical receipt is LF on Linux and CRLF on
    # Windows and the two cannot be byte-compared across a dual run or
    # carried between a node and a desktop.
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="")
    os.replace(temporary, target)
    return target


def make_receipt(*, config_sha256: str, mode: str, step: int, dt_s: float,
                 dx_m: float, dy_m: float, backend: str,
                 metrics: Mapping[str, object], applied: bool,
                 source: Mapping[str, object] | None = None) -> dict[str, object]:
    body = {
        "schema": RECEIPT_SCHEMA,
        "operator_schema": SCHEMA,
        "operator_pins_sha256": PINS_SHA256,
        "config_sha256": config_sha256,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "step": int(step),
        "dt_s": float(dt_s),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "backend": backend,
        "applied": bool(applied),
        "metrics": dict(metrics),
        "source": {} if source is None else dict(source),
    }
    body["receipt_sha256"] = canonical_hash(body)
    return body


def validate_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    value = dict(receipt)
    digest = value.pop("receipt_sha256", None)
    if value.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("spectral-step receipt schema mismatch")
    if value.get("operator_pins_sha256") != PINS_SHA256:
        raise ValueError("spectral-step receipt uses another operator identity")
    if digest != canonical_hash(value):
        raise ValueError("spectral-step receipt hash does not match its contents")
    value["receipt_sha256"] = digest
    return value


__all__ = ["RECEIPT_SCHEMA", "make_receipt", "validate_receipt",
           "write_json_atomic"]
