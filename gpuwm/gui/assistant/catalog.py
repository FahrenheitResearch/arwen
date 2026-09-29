"""The assistant's table: models, server builds, endpoints and decision services.

Everything the assistant can run on is a row of ``catalog.json``.  A new
model, server build or endpoint is a new row, never new code.
"""

from __future__ import annotations

from functools import lru_cache
import json
import platform
from pathlib import Path
from typing import Any

CATALOG = Path(__file__).resolve().parent / "catalog.json"


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def models() -> list[dict[str, Any]]:
    return list(load()["models"])


def model(model_id: str) -> dict[str, Any] | None:
    return next((row for row in models() if row["id"] == model_id), None)


def pick_model(card_gib: float | None) -> dict[str, Any]:
    """The largest model the card reaches; the default row when no card is known."""

    rows = models()
    if card_gib is None:
        return next((row for row in rows if row.get("default")), rows[0])
    fitting = [row for row in rows if row["min_card_gib"] <= card_gib + 0.6]
    if not fitting:
        return min(rows, key=lambda row: row["min_card_gib"])
    return max(fitting, key=lambda row: (row["min_card_gib"], row["params_b"]))


def server_build(os_name: str | None = None, arch: str | None = None) -> dict[str, Any] | None:
    """The llama.cpp build for this computer, or None when the table has none."""

    os_name = (os_name or platform.system()).lower()
    arch = (arch or platform.machine()).lower()
    for row in load()["servers"]:
        if row["os"] == os_name and arch in row["arch"]:
            return row
    return None


def endpoints() -> list[dict[str, Any]]:
    return list(load()["endpoints"])


def brought() -> list[dict[str, Any]]:
    """Models a person runs on their own server; listed with their licences, never downloaded."""

    return list(load().get("brought") or [])


def decision_services() -> list[dict[str, Any]]:
    return list(load()["decision_services"])


def decision_service(service_id: str) -> dict[str, Any] | None:
    return next((row for row in decision_services() if row["id"] == service_id), None)


__all__ = ["brought", "decision_service", "decision_services", "endpoints", "load", "model", "models",
           "pick_model", "server_build"]
