"""Ensemble and DA-cycle manifests (EXPERIMENTAL).

Two versioned schemas, both written with tmp+replace so a manifest is
never observed half-written -- a crash between members must leave the
previous complete manifest in place, not a truncated one.

``gpuwm-ensemble-manifest.v1`` lives at ``ens_root/ensemble-manifest.json``
and records, per member: the derived seed, the perturbation provenance,
the final state sha256, and the status.

``gpuwm-da-cycle-manifest.v1`` lives at ``ens_root/da-cycle-manifest.json``
and records the cycle times, the per-cycle per-member state shas, and a
slot for assimilation provenance that the DA lane fills in.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Mapping

from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY

ENSEMBLE_MANIFEST_SCHEMA = "gpuwm-ensemble-manifest.v1"
CYCLE_MANIFEST_SCHEMA = "gpuwm-da-cycle-manifest.v1"

ENSEMBLE_MANIFEST_NAME = "ensemble-manifest.json"
CYCLE_MANIFEST_NAME = "da-cycle-manifest.json"

#: Every state the engine may record for a member.  ``DONE`` is terminal
#: and is what makes a rerun a refusal; ``RUNNING`` left behind by a
#: crash is resumable; ``FAILED`` is resumable after the operator has
#: dealt with the cause.
MEMBER_STATUSES = ("PENDING", "RUNNING", "DONE", "FAILED")

#: Everything this package publishes is experimental.
STABILITY = "experimental"

#: Member directories are zero-padded to three digits, which orders
#: correctly in a shell for every ensemble size this engine allows.
_MEMBER_DIR_WIDTH = 3

#: How a reader outside Python must take the 64-bit member seeds.
#:
#: ``seed`` is a JSON *number*, and Python round-trips it exactly.  Most
#: other JSON stacks -- JavaScript above all -- parse every number as an
#: IEEE-754 binary64 and cannot represent an integer above ``2^53``.  The
#: derived seeds routinely exceed it (all four of base seed 20260730's do,
#: and a binary64 round trip moves them by hundreds), so a consumer that
#: reads ``seed`` in such a stack silently gets a different seed.
#: ``seed_hex`` is the interoperable spelling of the same 64-bit value and
#: is the one an external reproducer must use.
SEED_ENCODING = (
    "seed is an exact 64-bit unsigned integer; it is emitted as a JSON "
    "number for Python readers and as the 16-digit lowercase hex string "
    "seed_hex for every reader whose JSON numbers are IEEE-754 binary64 "
    "and therefore cannot hold it above 2^53")


def seed_hex(seed: int) -> str:
    """``seed`` as the 16-digit hex string external readers must use."""
    value = int(seed)
    if value < 0 or value >= (1 << 64):
        raise ValueError(
            f"member seed {value} is not a 64-bit unsigned integer")
    return f"{value:016x}"


def member_directory_name(index: int) -> str:
    """``member_000``, ``member_001``, ... for member ``index``."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"member index must be a non-negative int, "
                         f"got {index!r}")
    return f"member_{index:0{_MEMBER_DIR_WIDTH}d}"


def _fsync_directory(directory: Path) -> None:
    """Best-effort durability of a rename, where the OS offers it.

    ``os.replace`` makes the new content visible atomically, but on POSIX
    the *rename* itself is only durable after the containing directory is
    fsynced -- without this a power loss can leave the old name. Windows
    has no directory handle to sync and ``os.open`` on one fails; that is
    a platform limit, not a skipped step, so it is caught and ignored
    rather than pretended away.
    """
    try:
        fd = os.open(directory, getattr(os, "O_DIRECTORY", os.O_RDONLY))
    except (OSError, AttributeError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomically(path: str | Path, payload) -> Path:
    """Serialise ``payload`` and publish it with tmp+replace.

    **Single writer per manifest.**  The tmp name is unique per process
    and per call, so two concurrent writers cannot corrupt each other's
    partial file; they can still interleave whole publications, and the
    later one wins outright.  Nothing here makes concurrent invocations
    against one ensemble root safe, and the engine does not attempt them:
    members run sequentially by design.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    tmp = target.with_name(
        f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync_directory(target.parent)
    return target


def write_manifest_atomically(path: str | Path, manifest: Mapping) -> Path:
    """Publish an ensemble/cycle manifest, refusing an unknown schema."""
    schema = manifest.get("schema")
    if schema not in (ENSEMBLE_MANIFEST_SCHEMA, CYCLE_MANIFEST_SCHEMA):
        raise ValueError(
            f"refusing to write a manifest with schema {schema!r}; this "
            f"module writes {ENSEMBLE_MANIFEST_SCHEMA} and "
            f"{CYCLE_MANIFEST_SCHEMA}")
    return write_json_atomically(path, dict(manifest))


def read_manifest(path: str | Path, *, schema: str) -> dict:
    """Load a manifest and check its schema tag.  Fails closed."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"no manifest at {target}")
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{target} is not valid JSON: {error}. A tmp+replace write "
            "cannot produce this, so the file was edited or truncated by "
            "something else.") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{target} is not a JSON object")
    if loaded.get("schema") != schema:
        raise ValueError(
            f"{target} declares schema {loaded.get('schema')!r}, expected "
            f"{schema!r}")
    return loaded


def new_ensemble_manifest(cfg, *, seeds, ens_root: Path) -> dict:
    """A fresh manifest with every member PENDING.

    ``cfg`` is an :class:`gpuwm.ensemble.config.EnsembleConfig`; ``seeds``
    is the ordered list of derived member seeds.
    """
    from gpuwm.ensemble.seeds import SEED_DERIVATION

    if len(seeds) != cfg.n_members:
        raise ValueError(
            f"{len(seeds)} seeds for {cfg.n_members} members")
    return {
        "schema": ENSEMBLE_MANIFEST_SCHEMA,
        "stability": STABILITY,
        "experimental": True,
        "status": "PENDING",
        "ens_root": str(Path(ens_root)),
        "n_members": cfg.n_members,
        "base_seed": cfg.base_seed,
        "seed_derivation": SEED_DERIVATION,
        "seed_encoding": SEED_ENCODING,
        "base_config": str(cfg.base_config),
        "base_config_sha256": cfg.base_config_sha256,
        # Bound on resume: an overlay edit or an options change between
        # members would make one ensemble out of two configurations.
        "ensemble_source_sha256": cfg.source_sha256,
        "perturbation_options_sha256": cfg.perturbation_options_sha256,
        # Set by the first run that states one; bound on every later run.
        # ``null`` means "the base config's own run_seconds".
        "run_seconds": None,
        "run_seconds_bound": False,
        "ensemble_config": cfg.describe(),
        "perturbation": cfg.perturbation,
        "members": [
            {
                "index": index,
                "member_dir": member_directory_name(index),
                "seed": int(seed),
                "seed_hex": seed_hex(seed),
                "perturbation": {
                    "provenance": cfg.perturbation,
                    "options_sha256": cfg.perturbation_options_sha256,
                    "applied": None,
                },
                "final_state_sha256": None,
                "final_state_inventory": None,
                "status": "PENDING",
                "wall_seconds": None,
                "sim_seconds": None,
                # Written when the member finishes.  Present-and-null from
                # the start so "this member declared no inventory" is a
                # value in the schema rather than an absent key a reader
                # has to guess the meaning of.
                "wrfout_count": None,
                WRFOUT_INVENTORY_KEY: None,
                "error": None,
            }
            for index, seed in enumerate(seeds)
        ],
    }


def member_record(manifest: Mapping, index: int) -> dict:
    """The member record for ``index``, or a closed failure."""
    for record in manifest.get("members", ()):
        if record.get("index") == index:
            return record
    raise ValueError(
        f"manifest declares no member {index} (n_members="
        f"{manifest.get('n_members')})")


def first_incomplete_member(manifest: Mapping) -> int | None:
    """Index of the first member that is not ``DONE``, else ``None``."""
    for record in manifest.get("members", ()):
        if record.get("status") != "DONE":
            return int(record["index"])
    return None


def rollup_status(manifest: Mapping) -> str:
    """Ensemble-level status derived from the member records."""
    statuses = [record.get("status") for record in manifest.get("members", ())]
    if not statuses:
        return "PENDING"
    if any(status == "FAILED" for status in statuses):
        return "FAILED"
    if all(status == "DONE" for status in statuses):
        return "COMPLETE"
    if any(status in ("DONE", "RUNNING") for status in statuses):
        return "RUNNING"
    return "PENDING"


def new_cycle_manifest(cfg, *, ens_root: Path, cycle_seconds: float,
                       n_cycles: int, positivity: str,
                       restart_from_analysis: bool) -> dict:
    """A fresh DA-cycle manifest with no cycles recorded yet.

    Everything a resume has to agree with is recorded here, because
    "resume" reads this file and not the command line: the timeline
    (``n_cycles``, ``cycle_seconds``), the ensemble identity
    (``cycle_binding``), and the two driver policies that change what an
    analysis *is* (``positivity``, ``restart_from_analysis``).  A run that
    changed any of them against an existing manifest was reinterpreting an
    existing timeline, and every number it wrote was true about a
    different experiment.
    """
    return {
        "schema": CYCLE_MANIFEST_SCHEMA,
        "stability": STABILITY,
        "experimental": True,
        "status": "PENDING",
        "ens_root": str(Path(ens_root)),
        "n_members": cfg.n_members,
        "n_cycles": int(n_cycles),
        "cycle_seconds": float(cycle_seconds),
        "positivity": str(positivity),
        "restart_from_analysis": bool(restart_from_analysis),
        "base_config": str(cfg.base_config),
        "base_config_sha256": cfg.base_config_sha256,
        "seed_encoding": SEED_ENCODING,
        "cycle_binding": cycle_binding(cfg, cycle_seconds=cycle_seconds,
                                       n_cycles=n_cycles,
                                       positivity=positivity,
                                       restart_from_analysis=(
                                           restart_from_analysis)),
        "ensemble_config": cfg.describe(),
        # One entry per completed cycle; see gpuwm.ensemble.cycle.
        "cycles": [],
    }


def cycle_binding(cfg, *, cycle_seconds: float, n_cycles: int,
                  positivity: str, restart_from_analysis: bool) -> dict:
    """The facts a cycle resume must match, in one comparable block."""
    return {
        "n_members": int(cfg.n_members),
        "n_cycles": int(n_cycles),
        "cycle_seconds": float(cycle_seconds),
        "base_seed": int(cfg.base_seed),
        "base_config_sha256": cfg.base_config_sha256,
        "ensemble_source_sha256": cfg.source_sha256,
        "perturbation": cfg.perturbation,
        "perturbation_options_sha256": cfg.perturbation_options_sha256,
        "positivity": str(positivity),
        "restart_from_analysis": bool(restart_from_analysis),
    }
