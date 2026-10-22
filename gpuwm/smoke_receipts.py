"""CPU-only receipt text shared by preparation and smoke execution."""
from __future__ import annotations

#: The packaged HRRR v4 recipe's own ``smoke_off_reason`` (2.9.4 ships
#: HRRR-Smoke opt-in and hourly history); receipts carry it verbatim.
DEFAULT_OFF_LINE = ("HRRR-Smoke is off by default in 2.9.4 (opt in with "
                    "--smoke; it currently costs ~11x per step); history is "
                    "hourly (the 15-min history subset stream is next release)")


def message(reason):
    if reason == DEFAULT_OFF_LINE:
        return reason
    return f"smoke off for this run: {reason}; run with --devices 2 to carry it"


def receipt(cfg):
    reason = getattr(cfg, "smoke_off_reason", "")
    return {"smoke_off": message(reason)} if reason else {}


def config_receipt(text):
    import tomllib
    reason = tomllib.loads(text).get("shared", {}).get("smoke_off_reason", "")
    return {"smoke_off": message(reason)} if reason else {}
