"""Transport a measured GPU sizing profile without consulting the local GPU."""
from __future__ import annotations

import time

SCHEMA = "arwen.target-sizing.v1"
HOST_SCHEMA = "arwen.target-host-memory.v1"
PROFILE_FIELDS = ("name", "multiprocessor_count", "max_threads_per_multiprocessor",
                  "default_stack_limit_bytes", "bare_context_bytes")


def _integer(value, name, *, minimum=1, maximum=(1 << 63) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Selected GPU {name} must be an integer between {minimum} and {maximum}")
    return value


def validate_sizing(value):
    """Validate the exact measured inputs that the existing estimator reads."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Selected GPU has no compatible sizing measurement; reconnect its node before fitting")
    if set(value) != {"schema", "measured_unix_ms", "total_bytes", "free_bytes", "profile"}:
        raise ValueError("Selected GPU sizing measurement has missing or unknown fields")
    measured = _integer(value["measured_unix_ms"], "measurement time")
    total = _integer(value["total_bytes"], "total bytes")
    free = _integer(value["free_bytes"], "available bytes", minimum=0, maximum=total)
    profile = value["profile"]
    if not isinstance(profile, dict) or set(profile) != set(PROFILE_FIELDS):
        raise ValueError("Selected GPU sizing measurement lacks its complete device profile")
    name = profile["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 256 or any(ord(c) < 32 for c in name):
        raise ValueError("Selected GPU profile needs a valid device name")
    normalized = {"name": name,
        "multiprocessor_count": _integer(profile["multiprocessor_count"], "multiprocessor count", maximum=65536),
        "max_threads_per_multiprocessor": _integer(profile["max_threads_per_multiprocessor"], "threads per multiprocessor", maximum=65536),
        "default_stack_limit_bytes": _integer(profile["default_stack_limit_bytes"], "default stack limit", maximum=(1 << 32)),
        "bare_context_bytes": (None if profile["bare_context_bytes"] is None else
            _integer(profile["bare_context_bytes"], "bare context bytes", maximum=total))}
    return {"schema": SCHEMA, "measured_unix_ms": measured, "total_bytes": total,
            "free_bytes": free, "profile": normalized}


def sizing_from_probe(probe):
    """Select public numerical metadata from an already completed probe."""
    if not isinstance(probe, dict) or not isinstance(probe.get("profile"), dict):
        return None
    return validate_sizing({"schema": SCHEMA, "measured_unix_ms": int(time.time() * 1000),
        "total_bytes": probe.get("total_bytes"), "free_bytes": probe.get("free_bytes"),
        "profile": {key: probe["profile"].get(key) for key in PROFILE_FIELDS}})


def validate_host_memory(value):
    if not isinstance(value, dict) or set(value) != {"schema", "measured_unix_ms", "total_bytes"} or value.get("schema") != HOST_SCHEMA:
        raise ValueError("Selected target has no measured host-memory snapshot; reconnect it before fitting streamed domains")
    return {"schema": HOST_SCHEMA,
            "measured_unix_ms": _integer(value["measured_unix_ms"], "host-memory measurement time"),
            "total_bytes": _integer(value["total_bytes"], "host-memory total bytes")}


def host_memory_snapshot():
    """The same host/cgroup total used by the native tile planner, on its host."""
    from gpuwm.core.streaming import _host_total_bytes
    total = _host_total_bytes()
    if total is None:
        return None
    return validate_host_memory({"schema": HOST_SCHEMA, "measured_unix_ms": int(time.time() * 1000),
                                 "total_bytes": int(total)})
