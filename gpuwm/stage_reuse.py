"""Whether a chain stage's existing output can be reused, and why.

A run plan that failed at a late stage leaves every earlier stage's
output on disk, verified.  Re-running the same plan into the same
``output_root`` is the cheapest recovery there is -- the forcing is
already fetched and hash-verified, and the prepared bundle beside it
either still describes this run or does not.  Nothing here decides
that by looking at a model's name: the decision is read off artifacts
a stage published, in the vocabulary those artifacts already use, so a
source added to the fetch tables gets the same behaviour with no code
here to edit.

Three rules, and they are different rules for a reason.

**Fetch reuses by itself.**  ``gpuwm fetch`` already verifies an
existing payload's request identity, envelope, record count and
recorded digest before skipping it, and refuses a directory it cannot
tie to this request.  Nothing in this module touches that; the chain
only has to relay what the fetch receipt says it did.

**Preparation reuses when its identity still holds.**  The question is
"would this run hand the preparer the same instructions, over the same
input bytes, from the same engine?", and all three halves are answered
from artifacts:

* the published prepared-cache identity
  (:func:`gpuwm.ingest.prepared_cache.prepared_cache_identity`) carries
  the source-manifest digest, the namelist digest and the DATA-source
  identity of the bundle -- adapter, input-manifest digest, decoder
  digest -- and that identity is the very thing the forecast runner
  re-derives and compares before it will read a cache at all, so asking
  it before spending the preparation asks exactly the right question.
  What no adapter writes into it is the identity of the CODE, which is
  why the engine is recorded beside the bundle by :func:`write_binding`
  rather than read out of a block that never carried it;
* the binding receipt this module writes beside a finished preparation
  records the preparer's own arguments with every path-valued one
  reduced to the digest of the file it named, so a changed cycle, run
  length, cadence, physics profile, domain spec or namelist all move it
  and a relocated run directory does not.

The source identity pins the engine's own git state, so an engine that
has advanced invalidates every bundle it prepared.  That is not a
defect and it is not expensive -- a rebuild costs the same tens of
seconds the first preparation cost, against a fetch measured in
minutes and gigabytes.  It is also the only answer that stays true:
reusing a bundle across an engine change would hand the forecast a
cache the runner would then refuse, turning a cheap rebuild into a
confusing late refusal.

**A forecast's output directory is never reused.**  Its receipt has to
describe one run -- ``claim_output_directory`` in the prepared runners
says so, and it is right.  A previous attempt's output is moved aside
so the retry gets a clean directory and the earlier attempt's receipts
survive beside it.

Nothing here ever deletes.  Superseded output is renamed beside
itself, the same contract ``gpuwm fetch --force-refetch`` offers for a
data directory, and the byte count is reported so a caller can say what
reclaiming it would return.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

#: What a decision can be.  ``build`` is the ordinary first pass with
#: nothing on disk; the other two are the recovery cases.
BUILD = "build"
REUSE = "reuse"
REBUILD = "rebuild"

DECISION_SCHEMA = "gpuwm.stage-reuse-decision.v1"
BINDING_SCHEMA = "gpuwm.stage-binding.v1"
BINDING_NAME = "stage-binding.json"

#: Marks a directory this module moved aside.  Recognised on sight so a
#: second supersede never nests one inside another, and so a caller can
#: list what is reclaimable without keeping a manifest of it.
SUPERSEDED_MARK = ".superseded-"

#: Identity members a chain can state exactly BEFORE the stage runs.
#: Every one is a member of the canonical prepared-cache identity, so
#: the comparison speaks the artifact's own vocabulary rather than a
#: second spelling of it invented here.  ``domain_config`` is
#: deliberately absent: the preparers derive their domain document from
#: the namelist they are handed, not from the plan's TOML, so comparing
#: the plan's copy reports a difference on every run.  The namelist that
#: document is derived FROM is pinned instead, by digest.
STATEABLE = (
    "source_manifest_sha256",
    "namelist_sha256",
    "bridge_manifest_sha256",
    "static_cache_sha256",
    "forcing_hours",
    "forcing_offsets_seconds",
)

#: The source-identity members that pin the CODE, spelled the way
#: :func:`gpuwm.runtime_manifest.provenance` spells them.  A member this
#: engine can state and the bundle cannot is a DIFFERENCE, exactly as it
#: is for :data:`STATEABLE`: skipping it turned "the bundle cannot say
#: what built it" into agreement, and since no adapter writes one of
#: these into its own ``source_identity`` -- that block is a DATA-source
#: identity: adapter name, input-manifest digest, decoder digest -- the
#: skip fired on every member of every real bundle, and ``decide``
#: answered REUSE with the sentence "by this same engine" having
#: compared no engine at all.
SOURCE_IDENTITY_KEYS = (
    "identity_source",
    "git_commit",
    "git_tree",
    "git_status_short",
    "distribution_manifest_sha256",
    "installed_wheel",
)


def engine_source_identity() -> dict[str, Any]:
    """This engine's identity, by the resolver every receipt uses.

    One resolver, so a bundle prepared by this install and a decision
    taken by this install cannot disagree about what this install is.
    A broken install raises there; here it degrades to an empty mapping,
    because "I cannot tell what code this is" has to produce a REBUILD
    rather than an exception in front of someone who only asked to
    retry.
    """

    from gpuwm.runtime_manifest import provenance

    repo = Path(__file__).resolve().parents[1]
    try:
        identity = provenance(repo)
    except Exception:                       # noqa: BLE001 - see docstring
        return {}
    return {key: identity[key] for key in SOURCE_IDENTITY_KEYS
            if key in identity}


def argument_binding(arguments: Sequence[Any]) -> dict[str, Any]:
    """A stage's arguments, reduced to what makes its output different.

    A path is replaced by the digest of the file it names, so the same
    instructions over the same bytes bind identically from any
    directory and a changed file is a changed binding.  A directory
    argument keeps its name only: hashing a geography root would cost
    more than the stage being decided, and every directory a stage
    reads is pinned by its named manifest arguments. Prepared-input
    bundles additionally receive a complete content binding below.

    An interpreter path is dropped for the same reason a path is: the
    module name is what identifies the work, and the code behind it is
    pinned by the source identity, not by which python found it.
    """

    binding: dict[str, Any] = {}
    tokens = [str(token) for token in arguments]
    index = 0
    positional = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            if token.startswith("--experiment-config="):
                binding["--experiment-config"] = _experiment_argument_binding(
                    token.split("=", 1)[1])
                index += 1
                continue
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is None or following.startswith("--"):
                binding[token] = True
                index += 1
                continue
            binding[token] = (_experiment_argument_binding(following)
                              if token == "--experiment-config"
                              else _argument_value(following))
            index += 2
            continue
        binding[f"[{positional}]"] = _argument_value(token,
                                                     drop_interpreter=True)
        positional += 1
        index += 1
    return binding


def _experiment_argument_binding(token: str) -> Any:
    """Project only the governed forecast controls out of preparation argv.

    The cache reader already excludes PREPARATION_INERT_RUN_FIELDS from
    its domain comparison. Hashing the whole TOML here still rebuilt that
    same cache when a runtime inflow seed or adaptive target changed. Use
    that exact field table, leaving every other table and setting bound,
    including scientific initial perturbations and forcing duration.
    The directory remains bound because relative input paths in a config
    have meaning there. A malformed config keeps the conservative byte pin.
    """
    from gpuwm.config_authority import read_config_authority
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    try:
        authority = read_config_authority(Path(token))
        raw = tomllib.loads(authority.payload.decode("utf-8"))
        build_experiment_from_config_tables(
            raw, source=str(authority.source), base_dir=authority.base_dir)
    except (OSError, ValueError, UnicodeDecodeError):
        return _argument_value(token)
    inert = {path.removeprefix("run.") for path in PREPARATION_INERT_RUN_FIELDS
             if path.startswith("run.")}
    for table in (raw.get("shared", {}), *raw.get("domain", ())):
        for key in inert:
            table.pop(key, None)
    payload = {"config": raw, "base_directory": str(authority.base_dir)}
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=lambda value: value.isoformat(), allow_nan=False).encode()).hexdigest()
    return {"schema": "gpuwm-preparation-config-controls-v1", "sha256": digest}


def _argument_value(token: str, *, drop_interpreter: bool = False) -> Any:
    """One argument, as the thing about it that can differ."""

    try:
        path = Path(token)
        is_file = path.is_file()
        is_dir = path.is_dir()
    except (OSError, ValueError):
        return token
    if is_file:
        if drop_interpreter and path.suffix.lower() in (".exe", ""):
            # sys.executable heading a `-m` invocation.  Which python
            # ran it is not what makes one preparation differ from
            # another; the module named two tokens later is.
            return {"interpreter": True}
        return {"name": path.name, "sha256": _sha256(path)}
    if is_dir:
        return {"name": path.name, "directory": True}
    return token


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _published_files(root: Path, cache_roots: Sequence[Path]) -> dict:
    """All sealed non-cache bytes; cache arrays use their canonical reader.

    Only this stage's binding and live progress file are bookkeeping. Child
    receipts, configurations, optional corridors and exports remain bound.
    Paths are relative, so moving a complete bundle preserves its identity.
    """
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.as_posix() in (BINDING_NAME, "progress.json"):
            continue
        if any(path.is_relative_to(cache) for cache in cache_roots):
            continue
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"published artifact escapes its bundle: {relative}")
            result[relative.as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _published(root: Path, *, strict_single: bool = False):
    """Return root identity, complete snapshot, readers, and refusal reason."""
    from gpuwm.ingest.prepared_cache import PreparedCacheReader

    root = Path(root).resolve()
    if not root.is_dir():
        return None, None, (), f"{root} does not exist"
    headers = sorted(path for path in root.rglob("prepared-cache/header.json")
                     if not any(SUPERSEDED_MARK in part
                                for part in path.relative_to(root).parts))
    if not headers:
        return None, None, (), (
            f"{root} exists but publishes no prepared-cache header, so it "
            "carries no identity this run can be compared against")
    manifests = sorted(path for path in root.rglob("domain-artifacts.json")
                       if not any(SUPERSEDED_MARK in part
                                  for part in path.relative_to(root).parts))
    try:
        if manifests:
            if len(manifests) != 1:
                raise ValueError("preparation publishes multiple domain-artifact manifests")
            from gpuwm.wrf_direct import (
                load_domain_artifacts_manifest, _load_static_geometry_receipt)
            artifacts = sorted(load_domain_artifacts_manifest(manifests[0]),
                               key=lambda item: item.grid_id)
            if [item.grid_id for item in artifacts] != list(range(1, len(artifacts)+1)):
                raise ValueError("domain artifacts must cover contiguous grids from d01")
            declared = {item.prepared_cache / "header.json" for item in artifacts}
            if declared != set(headers):
                raise ValueError("domain-artifact manifest does not cover every "
                                 "published prepared-cache header exactly")
            readers, domains = [], {}
            for item in artifacts:
                header = json.loads((item.prepared_cache / "header.json").read_text(
                    encoding="utf-8"))
                identity = header.get("identity") if isinstance(header, dict) else None
                if not isinstance(identity, dict):
                    raise ValueError(f"d{item.grid_id:02d} lacks its identity")
                domain = identity.get("domain_config")
                if not isinstance(domain, dict) or domain.get("grid_id") != item.grid_id:
                    raise ValueError(f"d{item.grid_id:02d} cache grid identity differs")
                reader = PreparedCacheReader(item.prepared_cache,
                                             expected_identity=identity)
                _geometry, static_sha = _load_static_geometry_receipt(
                    item.geometry_receipt, item.static_cache)
                if identity.get("static_cache_sha256") != static_sha:
                    raise ValueError(f"d{item.grid_id:02d} static cache identity differs")
                readers.append(reader)
                domains[f"d{item.grid_id:02d}"] = {
                    "identity": identity,
                    "content_sha256": reader.content_sha256,
                    "payload_bytes": reader.payload_bytes,
                    "prepared_cache": item.prepared_cache.relative_to(root).as_posix(),
                }
            snapshot = {"domains": domains,
                        "files": _published_files(root, [r.path for r in readers])}
            return domains["d01"]["identity"], snapshot, tuple(readers), None
        if len(headers) > 1:
            return None, None, (), (
                f"{root} publishes {len(headers)} prepared-cache headers (a "
                "domain tree) without one canonical domain-artifact manifest "
                "accounting for all of them")
        header = json.loads(headers[0].read_text(encoding="utf-8"))
        identity = header.get("identity") if isinstance(header, dict) else None
        if not isinstance(identity, Mapping):
            raise ValueError(f"{headers[0]} carries no identity block")
        if header.get("status") != "READY":
            raise ValueError(f"{headers[0]} is {header.get('status')!r} rather than "
                             "READY, so the preparation that wrote it did not finish")
        if not strict_single:
            return dict(identity), None, (), None
        reader = PreparedCacheReader(headers[0].parent, expected_identity=identity)
        snapshot = {"identity": dict(identity),
                    "content_sha256": reader.content_sha256,
                    "payload_bytes": reader.payload_bytes,
                    "files": _published_files(root, [reader.path])}
        return dict(identity), snapshot, (reader,), None
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return None, None, (), f"published preparation is invalid: {error}"


def published_identity(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Root identity of a single cache or a complete canonical domain tree.

    Reuse additionally binds and verifies every domain in ``decide``. Unknown
    legacy trees are refused; merely finding one root never certifies children.
    """
    identity, _snapshot, _readers, reason = _published(root)
    return identity, reason


def _prepared_inputs(arguments: Sequence[Any]):
    """Bind the preparer's explicit prepared-input role, not its basename.

    GEOG roots and output directories are not prepared-input arguments and
    are deliberately not recursively hashed here.
    """
    tokens = [str(token) for token in arguments]
    snapshots, readers = {}, []
    role = "--root-preparation"
    for index, token in enumerate(tokens):
        if token == role:
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
        elif token.startswith(role + "="):
            value = token.partition("=")[2]
        else:
            continue
        if not value or value.startswith("--"):
            raise ValueError("--root-preparation has no input bundle")
        _identity, snapshot, found, reason = _published(
            Path(value), strict_single=True)
        if reason:
            raise ValueError(f"--root-preparation: {reason}")
        snapshots[role] = snapshot
        readers.extend(found)
    return snapshots, tuple(readers)


def write_binding(root: Path, *, arguments: Sequence[Any],
                  stated: Mapping[str, Any] | None = None) -> Path | None:
    """Record what built the output at ``root``, for the next decision.

    Written only after the stage succeeded, so a binding on disk always
    describes a finished bundle.  The stage's own artifacts stay the
    authority on everything they record; this covers the two things they
    do not -- the instructions the stage was given, and the identity of
    the engine that carried them out.

    ``None`` when the stage left no directory to record against.  That
    is not this function's failure to report: the caller's own gate on
    the stage's output says what a missing bundle means, and the only
    consequence here is that the next pass rebuilds rather than reusing
    something that was never written.
    """

    root = Path(root)
    if not root.is_dir():
        return None
    path = root / BINDING_NAME
    payload = {
        "schema": BINDING_SCHEMA,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arguments": argument_binding(arguments),
        "stated": dict(stated or {}),
        # Written by the process that just ran the stage, so it is a
        # reading of the code that built this output rather than a claim
        # about it.  Nothing the preparers publish records the engine --
        # every adapter's ``source_identity`` is a DATA-source block --
        # so without this the engine half of the decision has no
        # artifact to read, and a binding written before this field
        # existed is answered as the ignorance it is: a difference, and
        # a rebuild.
        "engine": engine_source_identity(),
    }
    _identity, snapshot, _readers, reason = _published(root)
    if snapshot is not None:
        payload["publication"] = snapshot
    if reason:
        payload["publication_refusal"] = reason
    try:
        prepared_inputs, _readers = _prepared_inputs(arguments)
        if prepared_inputs:
            payload["prepared_inputs"] = prepared_inputs
    except ValueError as error:
        payload["prepared_input_refusal"] = str(error)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return path


def _read_binding(root: Path) -> dict[str, Any] | None:
    path = Path(root) / BINDING_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if (not isinstance(payload, dict)
            or payload.get("schema") != BINDING_SCHEMA):
        return None
    return payload


def _recorded_engine(identity: Mapping[str, Any],
                     binding: Mapping[str, Any]) -> dict[str, Any]:
    """The engine identity this bundle can show, out of both its records.

    Two artifacts, and the precedence between them is the rule
    :func:`write_binding` already states: the stage's own published
    identity is the authority on everything it records, and the binding
    beside it covers what that identity does not.  Today only the
    binding answers -- no shipped adapter writes a member of
    :data:`SOURCE_IDENTITY_KEYS` into its ``source_identity`` -- but a
    route that starts publishing one is then believed over the binding
    written around it, which is the right way round: the preparer knows
    what it ran, the chain only knows what it called.

    Members not in :data:`SOURCE_IDENTITY_KEYS` are dropped from both,
    so a data-source block's ``adapter`` or ``decoder`` cannot arrive
    here wearing the name of an engine field.
    """

    def members(block: Any) -> dict[str, Any]:
        if not isinstance(block, Mapping):
            return {}
        return {key: block[key] for key in SOURCE_IDENTITY_KEYS
                if key in block}

    return {**members(binding.get("engine")),
            **members(identity.get("source_identity"))}


def decide(root: Path, *, stated: Mapping[str, Any],
           arguments: Sequence[Any]) -> dict[str, Any]:
    """Reuse, rebuild, or build the stage output at ``root``.

    ``stated`` is what THIS run knows before the stage runs, keyed by
    the prepared-cache identity's own member names; only members the
    caller states AND the bundle recorded are compared, so a member a
    chain cannot compute is never turned into a silent pass.
    ``arguments`` is the argv this run would give the stage.

    The answer is a receipt, not a boolean: ``decision`` is what will
    happen, ``reason`` is one sentence saying why in the artifact's own
    vocabulary, and ``differences`` names every field that moved -- so a
    rebuild after an engine update reads as an engine update rather
    than as an unexplained repeat of work.
    """

    root = Path(root)
    if not root.exists():
        return _answer(BUILD, root, "nothing is prepared here yet", [])
    verification_started = perf_counter()
    identity, snapshot, readers, unreadable = _published(root)
    if identity is None:
        return _answer(
            REBUILD, root,
            f"{unreadable}, so it is rebuilt rather than trusted", [])
    binding = _read_binding(root)
    if binding is None:
        return _answer(
            REBUILD, root,
            f"the bundle already here records no {BINDING_NAME}, so the "
            "arguments that built it cannot be shown to be this run's; it "
            "is rebuilt from the forcing already on disk", [])

    differences: list[dict[str, Any]] = []
    compared = [key for key in STATEABLE
                if key in stated and key in identity]
    for key in compared:
        if not _same(identity[key], stated[key]):
            differences.append({
                "field": key,
                "recorded": _brief(identity[key]),
                "requested": _brief(stated[key]),
            })
    for key in STATEABLE:
        if key in stated and key not in identity:
            differences.append({
                "field": key,
                "recorded": None,
                "requested": _brief(stated[key]),
                "note": "the bundle already here records no such member",
            })

    requested_arguments = argument_binding(arguments)
    recorded_arguments = binding.get("arguments")
    recorded_arguments = (dict(recorded_arguments)
                          if isinstance(recorded_arguments, Mapping) else {})
    for key in sorted(set(recorded_arguments) | set(requested_arguments)):
        if not _same(recorded_arguments.get(key), requested_arguments.get(key)):
            differences.append({
                "field": f"arguments{key}" if key.startswith("[")
                         else f"arguments {key}",
                "recorded": _brief(recorded_arguments.get(key)),
                "requested": _brief(requested_arguments.get(key)),
            })

    recorded_source = _recorded_engine(identity, binding)
    current_source = engine_source_identity()
    if not current_source:
        differences.append({
            "field": "source_identity",
            "recorded": _brief(recorded_source.get("git_commit")),
            "requested": None,
            "note": ("this install cannot state its own identity, so the "
                     "code that built the bundle cannot be shown to be the "
                     "code that would read it"),
        })
    else:
        for key in SOURCE_IDENTITY_KEYS:
            if key not in recorded_source:
                if key not in current_source:
                    continue
                differences.append({
                    "field": f"source_identity.{key}",
                    "recorded": None,
                    "requested": _brief(current_source[key]),
                    "note": ("the bundle already here records no such "
                             "member, so the code that built it cannot be "
                             "shown to be the code that would read it"),
                })
                continue
            if not _same(recorded_source[key], current_source.get(key)):
                differences.append({
                    "field": f"source_identity.{key}",
                    "recorded": _brief(recorded_source[key]),
                    "requested": _brief(current_source.get(key)),
                    "note": ("the engine has changed since this bundle was "
                             "prepared"),
                })

    input_readers = ()
    try:
        inputs, input_readers = _prepared_inputs(arguments)
        if not _same(binding.get("prepared_inputs", {}), inputs):
            differences.append({"field": "prepared_inputs",
                                "recorded": _brief(binding.get("prepared_inputs")),
                                "requested": _brief(inputs)})
    except ValueError as error:
        differences.append({"field": "prepared_inputs", "note": str(error)})
    if snapshot is not None or binding.get("publication") is not None:
        old = binding.get("publication")
        if not isinstance(old, Mapping):
            differences.append({"field": "publication",
                                "note": "the binding does not account for every domain"})
        elif not _same(old, snapshot):
            for section in sorted(set(old) | set(snapshot or {})):
                before, now = old.get(section, {}), (snapshot or {}).get(section, {})
                if not isinstance(before, Mapping) or not isinstance(now, Mapping):
                    differences.append({"field": f"publication.{section}",
                                        "recorded": _brief(before),
                                        "requested": _brief(now)})
                    continue
                for name in sorted(set(before) | set(now)):
                    if not _same(before.get(name), now.get(name)):
                        differences.append({"field": f"publication.{section}.{name}",
                                            "recorded": _brief(before.get(name)),
                                            "requested": _brief(now.get(name))})
        if snapshot is not None:
            for label, domain in snapshot["domains"].items():
                local = domain["identity"]
                for key in STATEABLE:
                    if key == "static_cache_sha256" or key not in stated:
                        continue  # statics are bound separately for each domain
                    if not _same(local.get(key), stated[key]):
                        differences.append({"field": f"{label}.{key}",
                                            "recorded": _brief(local.get(key)),
                                            "requested": _brief(stated[key])})
                recorded = _recorded_engine(local, binding)
                for key in SOURCE_IDENTITY_KEYS:
                    if key in current_source or key in recorded:
                        if not _same(recorded.get(key), current_source.get(key)):
                            differences.append({"field": f"{label}.source_identity.{key}",
                                                "recorded": _brief(recorded.get(key)),
                                                "requested": _brief(current_source.get(key))})
    verification = []
    if not differences:
        try:
            for reader in (*readers, *input_readers):
                verification.append(reader.verify_all())
        except (OSError, ValueError) as error:
            differences.append({"field": "prepared_payload",
                                "note": str(error)})
    if not differences:
        answer = _answer(
            REUSE, root,
            "the bundle already here was built from these same arguments "
            "over these same input bytes by this same engine, so the stage "
            "is skipped and its output reused",
            [], compared=compared)
        if snapshot is not None:
            answer["domains"] = sorted(snapshot["domains"])
        if verification:
            answer["verified_prepared_bytes"] = sum(v["payload_bytes"] for v in verification)
            answer["verified_prepared_arrays"] = sum(v["array_count"] for v in verification)
            seals = ([snapshot] if snapshot else []) + list(inputs.values())
            answer["verified_artifact_bytes"] = sum(
                entry["bytes"] for seal in seals for entry in seal["files"].values())
            answer["verification_seconds"] = perf_counter() - verification_started
        return answer
    first = differences[0]["field"]
    return _answer(
        REBUILD, root,
        f"the bundle already here was built with a different {first}, so it "
        "is rebuilt from the forcing already on disk",
        differences, compared=compared)


def _answer(decision: str, root: Path, reason: str,
            differences: list[dict[str, Any]],
            compared: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "schema": DECISION_SCHEMA,
        "decision": decision,
        "root": str(root),
        "reason": reason,
        "differences": differences,
        "compared": list(compared),
    }


def supersede(path: Path) -> dict[str, Any] | None:
    """Move ``path`` aside, keeping every byte, and say where it went.

    ``None`` when there was nothing there.  The new name carries a UTC
    stamp so repeated retries never collide and never overwrite each
    other -- the same "nothing is deleted" contract
    ``gpuwm fetch --force-refetch`` offers for a data directory.
    """

    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}{SUPERSEDED_MARK}{stamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(
            f"{path.name}{SUPERSEDED_MARK}{stamp}-{suffix}")
        suffix += 1
    os.replace(path, target)
    return {
        "path": str(target),
        "bytes": _tree_bytes(target),
        "note": ("nothing was deleted; removing this directory reclaims the "
                 "bytes named here"),
    }


def _tree_bytes(root: Path) -> int:
    """Total payload under ``root``, counting only real files."""

    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _same(left: Any, right: Any) -> bool:
    """JSON equality, so a tuple and a list are not a false difference."""

    try:
        return (json.dumps(left, sort_keys=True, default=str)
                == json.dumps(right, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return left == right


def _brief(value: Any) -> Any:
    """A value small enough to sit in a receipt a person will read.

    A domain document is ~120 fields; printing it whole in a difference
    list buries the one field that moved.  A digest of it says "this
    changed" without pretending to say which knob, which is what the
    field name already says.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)[:120]
    if len(text) <= 160:
        return json.loads(text)
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text),
    }
