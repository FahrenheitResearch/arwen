"""One canonical content hash for a member's final state (EXPERIMENTAL).

The manifest's ``final_state_sha256`` must mean the same thing whether it
was taken from a live state object at the end of an integration or from
a checkpoint file written by one, so both go through the same reduction:
:data:`gpuwm.io.restart.STATE_SERIALIZED_ATTRS`, in restart order, each
hashed as ``name\\0dtype\\0shape\\0raw-bytes``.

**What that list is, precisely.**  It is the restart layer's *serialised
state* set, not "the prognostics": it includes the equation-of-state
diagnostics ``p``/``al``/``alt``, the effective radii, and
``h_diabatic``, and it excludes serialised scratch and physics
continuation state -- precipitation accumulators, convective-scheme
timers -- that the restart file also carries.  Two runs agreeing on this
hash agree on the serialised state arrays and on nothing else; calling it
"the prognostic state" oversold it and this docstring is the correction.

**A truncated inventory is reported, never silently normal.**  Both
reductions skip contract attributes that are absent, because partial
states are a real and legitimate input (a synthetic gate checkpoint
carrying four fields is not a broken model state).  What they must not do
is hand back a normal-looking sha for a file holding one array with no
way to tell.  Every reduction therefore also produces an
:func:`state_inventory` -- present, missing, complete -- which the engine
records beside the hash, and ``require_complete=True`` turns an
incomplete inventory into a refusal for callers that know they are
holding a whole state.

Deliberately *not* the sha256 of a wrfout: history files carry creation
metadata, so two byte-identical forecasts do not produce byte-identical
wrfouts and a determinism check against one would be a lie.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

#: Versioned reduction label, hashed in as a domain separator.
STATE_SHA_CONTRACT = "gpuwm-ensemble-state-sha.v1"


def serialized_state_attrs() -> tuple[str, ...]:
    """The restart layer's prognostic attribute list, in its own order."""
    from gpuwm.io import restart as restart_io

    return tuple(restart_io.STATE_SERIALIZED_ATTRS)


def _host(value) -> np.ndarray:
    """A host copy of a device or host array, without importing cupy."""
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def hash_state_arrays(arrays) -> str:
    """Hash an ordered ``(name, array)`` sequence under the contract."""
    digest = hashlib.sha256(STATE_SHA_CONTRACT.encode("ascii") + b"\0")
    for name, value in arrays:
        host = _host(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(host.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(host.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def state_inventory(present) -> dict:
    """Which contract attributes a hashed state actually carried.

    Recorded beside every ``*_state_sha256`` so a hash over one array and
    a hash over the whole serialised state are distinguishable *as
    provenance*, not merely as two different hex strings that both look
    fine.
    """
    contract = serialized_state_attrs()
    have = [name for name in contract if name in set(present)]
    missing = [name for name in contract if name not in set(present)]
    return {
        "contract": STATE_SHA_CONTRACT,
        "contract_attrs": len(contract),
        "present": have,
        "missing": missing,
        "complete": not missing,
        "scope": ("gpuwm.io.restart.STATE_SERIALIZED_ATTRS -- serialised "
                  "state arrays including the p/al/alt diagnostics, "
                  "excluding serialised scratch and physics continuation "
                  "state"),
    }


def _require_complete(inventory: dict, what: str) -> None:
    if inventory["complete"]:
        return
    raise ValueError(
        f"{what} carries {len(inventory['present'])} of "
        f"{inventory['contract_attrs']} restart-serialised state "
        f"attributes; missing {', '.join(inventory['missing'])}. A hash "
        "over a truncated inventory looks exactly like a hash over a whole "
        "state, so this caller refuses rather than recording one.")


def live_state_sha256(state, *, require_complete: bool = False) -> str:
    """Content hash of a live state object's serialised state arrays."""
    return live_state_sha_receipt(
        state, require_complete=require_complete)["sha256"]


def live_state_sha_receipt(state, *, require_complete: bool = False) -> dict:
    """:func:`live_state_sha256` plus the inventory it was taken over."""
    arrays = []
    for name in serialized_state_attrs():
        value = getattr(state, name, None)
        if value is None:
            continue
        arrays.append((name, value))
    inventory = state_inventory([name for name, _ in arrays])
    if not arrays:
        raise ValueError(
            "state object carries none of the restart-serialised state "
            f"attributes ({inventory['contract_attrs']} absent); it is "
            "not a model state")
    if require_complete:
        _require_complete(inventory, "the state object")
    return {"sha256": hash_state_arrays(arrays), "inventory": inventory}


#: Checkpoint keys that can carry the forecast clock, most authoritative
#: first.  ``__gpuwm_restart_header__`` is what ``gpuwm.io.restart`` writes
#: (``_HEADER_KEY``); ``meta/elapsed_seconds`` is the reduced spelling a
#: synthetic gate checkpoint uses.  A file with neither states no clock,
#: which is unverifiable and NOT zero.
_ELAPSED_HEADER_KEY = "__gpuwm_restart_header__"
_ELAPSED_META_KEY = "meta/elapsed_seconds"


def checkpoint_elapsed_seconds(path: str | Path) -> float | None:
    """The elapsed forecast time a checkpoint states, or ``None``.

    This is the clock of the state a restart RESTORES, which is the only
    baseline an unchanged-state guard on a restart leg may use: the
    prepared state's clock describes a state the leg discards before its
    first step, and comparing against it let a no-op integrator report a
    60 s advance over a state whose clock never moved.
    """

    import json

    target = Path(path)
    if not target.is_file():
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            if _ELAPSED_HEADER_KEY in data.files:
                header = json.loads(
                    bytes(bytearray(data[_ELAPSED_HEADER_KEY])
                          ).decode("utf-8"))
                value = header.get("elapsed_seconds")
                return None if value is None else float(value)
            if _ELAPSED_META_KEY in data.files:
                return float(data[_ELAPSED_META_KEY])
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError):
        return None
    return None


def checkpoint_state_sha256(path: str | Path, *,
                            require_complete: bool = False) -> str:
    """Content hash of a ``gpuwmrst`` checkpoint's ``state/*`` arrays.

    Uses the checkpoint's own key order restricted to the restart
    contract, so it equals :func:`live_state_sha256` of the state that
    wrote it whenever the checkpoint carries the full contract.
    """
    return checkpoint_state_sha_receipt(
        path, require_complete=require_complete)["sha256"]


def checkpoint_state_sha_receipt(path: str | Path, *,
                                 require_complete: bool = False) -> dict:
    """:func:`checkpoint_state_sha256` plus the inventory it covered."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"no checkpoint at {target}")
    with np.load(target, allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files
                  if key.startswith("state/")}
    arrays = [(name, stored[f"state/{name}"])
              for name in serialized_state_attrs()
              if f"state/{name}" in stored]
    inventory = state_inventory([name for name, _ in arrays])
    if not arrays:
        raise ValueError(
            f"{target} carries no state/* arrays from the restart "
            "contract; it is not a gpuwm checkpoint")
    if require_complete:
        _require_complete(inventory, str(target))
    return {"sha256": hash_state_arrays(arrays), "inventory": inventory}
