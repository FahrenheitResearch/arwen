"""Production canonical frame-state hashing shared by run controllers.

The hash is model evidence, not a verification-only implementation detail.
Keeping it in the production package prevents run/benchmark entry points from
depending on the developer-only ``gpuwm.verify`` tree.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

from gpuwm.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS


CANONICAL_STATE_SCHEMA = "gpuwm-canonical-state-v2"
CANONICAL_INVENTORY_SCHEMA = "gpuwm-canonical-state-inventory-v2"
CANONICAL_LAZY_MEMBER_CLASSES = (
    "nest rolling value/tendency, donor, and SINT tables",
    "lateral-boundary relaxation weights",
    "REFL_10CM frame stash",
    "microphysics carrying accumulators",
    "cumulus carrying accumulators",
    "KF W0AVG trigger history",
)
# The digest's nest-slot membership is derived from the SAME shared
# inventory the coupling machinery reads (COUPLED_SCALAR_STATE_FIELDS),
# never spelled inline: this was the SIXTH hand-copied table on 1.9.1
# D1's route.  A hand-maintained duplicate here lacked mp=9's nc/nh,
# WDM6's nn, P3's rime pair AND mp=28's nc/nwfa/nifa, so a nested run of
# any of those schemes integrated perfectly and then died in the
# end-of-run canonical digest ("scratch member 'nest_nc_btxe' ... not a
# concrete registered member") -- the same false-failure class as D3.
_CANONICAL_NEST_FIELD_KINDS = (
    "u", "v", "w", "t", "ph", "mu",
    *sorted(COUPLED_SCALAR_STATE_FIELDS),
)
_CANONICAL_NEST_SLOTS = frozenset({
    *(f"nest_{kind}_{prefix}{side}"
      for kind in _CANONICAL_NEST_FIELD_KINDS
      for prefix in ("b", "bt")
      for side in ("xs", "xe", "ys", "ye")),
    "nest_parent_field",
    "nest_child_field",
    *(f"nest_sint_{component}_{stagger}"
      for component in ("ci", "ip", "cj", "jp", "xig", "xjg")
      for stagger in ("m", "x", "y")),
})
_CANONICAL_LBC_SLOTS = frozenset({"lbc_weights_0"})
_CANONICAL_REFL_SLOTS = frozenset({"refl_10cm"})
_CANONICAL_EXTRA_PREFIXES = ("nest_", "lbc_weights_", "refl_")
_CANONICAL_EXCLUDED_FRAME_SCRATCH = frozenset({"refl_t"})
CHILD_DUTY_SCRATCH_MEMBERS = (
    "scratch/nest_parent_field",
    "scratch/nest_child_field",
)


def _inventory_sha256(members: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256(b"gpuwm-canonical-state-inventory-v2\0")
    for member in members:
        descriptor = json.dumps(
            [str(member["name"]), str(member["dtype"]),
             [int(size) for size in member["shape"]]],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
    return digest.hexdigest()


def lazy_inventory_member_class(name: str) -> str | None:
    """Return the documented lazy class for a canonical member path."""
    from gpuwm.io import restart as restart_io

    scratch_slot = (
        name.removeprefix("scratch/") if name.startswith("scratch/") else None
    )
    if scratch_slot in _CANONICAL_NEST_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[0]
    if scratch_slot in _CANONICAL_LBC_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[1]
    if scratch_slot in _CANONICAL_REFL_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[2]
    if name.startswith("scratch/"):
        slot = name.removeprefix("scratch/")
        if slot in restart_io.SERIALIZED_SCRATCH_SLOTS:
            if slot.startswith("mp_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[3]
            if slot.startswith("cu_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[4]
    if name == "cumulus/w0avg":
        return CANONICAL_LAZY_MEMBER_CLASSES[5]
    return None


def _canonical_extra_manifest(state) -> dict[str, object]:
    manifest = {}
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        name = f"scratch/{slot}"
        if slot in _CANONICAL_EXCLUDED_FRAME_SCRATCH:
            continue
        if lazy_inventory_member_class(name) in CANONICAL_LAZY_MEMBER_CLASSES[:3]:
            manifest[name] = value
        elif slot.startswith(_CANONICAL_EXTRA_PREFIXES):
            raise RuntimeError(
                f"canonical frame-state scratch member {slot!r} is inside an "
                "audited lazy prefix but is not a concrete registered member"
            )
    return manifest


def _canonical_scalar_bytes(scalars: Mapping[str, object]) -> bytes:
    return json.dumps(
        scalars, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def canonical_state_digest(state, clock, *,
                           scope: str = "trajectory") -> dict[str, object]:
    """Hash complete frame-time trajectory state in restart order."""
    if scope not in ("trajectory", "full"):
        raise ValueError(f"unknown canonical digest scope {scope!r}")
    from gpuwm.io import restart as restart_io

    manifest: dict[str, object] = {}
    manifest.update(restart_io.state_manifest(state))
    manifest.update(restart_io._scratch_manifest(state))
    driver = getattr(state, "physics", None)
    if driver is not None:
        manifest.update(restart_io._driver_manifest(driver))
    manifest.update(_canonical_extra_manifest(state))
    if scope == "trajectory":
        for member in CHILD_DUTY_SCRATCH_MEMBERS:
            manifest.pop(member, None)
    dtbc_fp32_bits = int(
        np.asarray(np.float32(clock.dtbc_fp32)).view(np.uint32).item()
    )
    scalars = {
        "elapsed_seconds": float(state.elapsed_seconds),
        "dtbc_fp32_bits": dtbc_fp32_bits,
        "driver": (None if driver is None else {
            "call_counts": {
                key: int(value)
                for key, value in sorted(driver.call_counts.items())
            },
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "microphysics_updates": int(driver.microphysics_updates),
        }),
    }
    scalar_bytes = _canonical_scalar_bytes(scalars)
    digest = hashlib.sha256(
        b"gpuwm-canonical-state-v2:" + scope.encode("ascii") + b"\0"
        + scalar_bytes
    )
    members = []
    for key in sorted(manifest):
        host = np.ascontiguousarray(restart_io._host(manifest[key]))
        member = {"name": key, "dtype": str(host.dtype),
                  "shape": list(host.shape)}
        descriptor = json.dumps(
            [member["name"], member["dtype"], member["shape"]],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(host.tobytes(order="C"))
        members.append(member)
    inventory = {
        "schema": CANONICAL_INVENTORY_SCHEMA,
        "sha256": _inventory_sha256(members),
        "array_count": len(members),
        "members": members,
    }
    return {
        "schema": CANONICAL_STATE_SCHEMA,
        "sha256": digest.hexdigest(),
        "inventory_sha256": inventory["sha256"],
        "scalar_sha256": hashlib.sha256(scalar_bytes).hexdigest(),
        "array_count": len(members),
        "field_order": [item["name"] for item in members],
        "inventory": inventory,
        "scalars": scalars,
    }


__all__ = [
    "CANONICAL_INVENTORY_SCHEMA",
    "CANONICAL_LAZY_MEMBER_CLASSES",
    "CANONICAL_STATE_SCHEMA",
    "CHILD_DUTY_SCRATCH_MEMBERS",
    "canonical_state_digest",
    "lazy_inventory_member_class",
]
