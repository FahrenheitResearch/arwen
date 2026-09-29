"""Local input binding for preparation chains without an acquisition route."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _identity(document: Mapping) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


#: The input manifest a folder's own preparation authors beside the bytes
#: it seals: the same name the table routes' handoff gives it.
SOURCE_ROOT_MANIFEST_NAME = "inputs.json"


def _fetch_hint(fetch, folder: Path, *, cycle=None, hours=None,
                cadence=None) -> str:
    """The `gpuwm fetch` line that writes a row's supplement into ``folder``."""
    from gpuwm import fetch_routes

    parts = [f"gpuwm fetch --source {fetch['source']}",
             f"--cycle {cycle.strftime('%Y-%m-%dT%H') if cycle else 'CYCLE'}",
             f"--hours {hours if hours is not None else 'HOURS'}"]
    if cadence is not None:
        parts.append(f"--cadence {cadence}")
    parts.append("--area AREA")
    if fetch_routes.supplement_fetch_retrieves(fetch):
        parts.append("--retrieve")
    parts.append(f"--out {folder}")
    return " ".join(parts)


def bind_source_root(source: str, source_root, *, cycle=None, hours=None,
                     cadence=None) -> dict:
    """Bind a hand-staged folder's files to the roles its source row declares.

    The source's refusal row in the acquisition-route table states the
    folder's layout (:func:`gpuwm.fetch_routes.source_root_layout`):
    which files are the ordered inputs, and which file is each
    supplement role.  Nothing here names a model.  Inputs are every file
    the row's patterns match whose leading bytes carry the declared
    format, in name order; a supplement file is excluded from them unless
    its row says it is an input as well.

    Returns ``{"source", "source_root", "inputs", "supplements"}`` with
    ``supplements`` as ``(role, path)`` pairs.  Refuses, naming the file
    and how to get it, when a declared role has no file or the folder
    holds no input.
    """

    from fnmatch import fnmatchcase

    from gpuwm import fetch_routes
    from gpuwm.source_adapters import get_source_adapter

    source_id = get_source_adapter(source).source_id
    layout = fetch_routes.source_root_layout(source_id)
    if layout is None:
        raise ValueError(
            f"{source_id} declares no folder layout, so the files in a "
            "folder cannot be bound to its preparation roles. Name them "
            "with --input (or --input-list) and --supplement ROLE=PATH.")
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Local source directory {root} does not exist. "
                         "Supply the directory containing the input files.")
    names = sorted(entry.name for entry in root.iterdir() if entry.is_file())

    def matching(patterns):
        return [name for name in names
                if any(fnmatchcase(name, pattern) for pattern in patterns)]

    supplements: list[tuple[str, Path]] = []
    claimed: set[str] = set()
    for row in layout["supplements"]:
        found = matching(row["match"])
        if not found:
            remedy = ""
            if row["fetch"] is not None:
                remedy = (" Write it into this folder with `"
                          + _fetch_hint(row["fetch"], root, cycle=cycle,
                                        hours=hours, cadence=cadence)
                          + "`.")
            raise ValueError(
                f"{root} holds no {row['role']} file "
                f"({', '.join(row['match'])}): {row['why']}{remedy}")
        for name in found:
            declared = fetch_routes.sniff_format(root / name)
            if declared != row["format"]:
                raise ValueError(
                    f"{root / name} is this folder's {row['role']} file by "
                    f"name but its bytes are {declared or 'no format this door reads'}, "
                    f"not {row['format']}: the composition would refuse it "
                    "after the inputs were decoded. Replace it with the "
                    f"{row['format']} file the row names.")
            supplements.append((row["role"], root / name))
            if not row["input"]:
                claimed.add(name)
    wanted = layout["inputs"]
    inputs = [root / name for name in matching(wanted["match"])
              if name not in claimed
              and not any(fnmatchcase(name, pattern)
                          for pattern in wanted["exclude"])
              and fetch_routes.sniff_format(root / name) == wanted["format"]]
    if not inputs:
        raise ValueError(
            f"{root} holds no {wanted['format']} input file for {source_id} "
            f"({', '.join(wanted['match'])}). Expected: {layout['why']}")
    return {"source": source_id, "source_root": str(root),
            "inputs": inputs, "supplements": supplements}


def source_root_handoff(binding: Mapping, *, cycle: str) -> dict:
    """The preparation handoff a bound folder stands for.

    The same document a table route's fetch writes as
    ``prep-arguments.json``: the ordered input list, each supplement
    binding and the manifest the preparation authors.  Its input-list
    token names the list file a published handoff writes; the ordered
    inputs themselves travel beside it.
    """

    from gpuwm import fetch_routes

    root = Path(binding["source_root"])
    argv = ["--source", str(binding["source"]),
            "--input-list", str(root / fetch_routes.INPUT_LIST_NAME)]
    for role, path in binding["supplements"]:
        argv += ["--supplement", f"{role}={path}"]
    argv += ["--author-input-manifest", str(root / SOURCE_ROOT_MANIFEST_NAME)]
    return {"schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
            "source": str(binding["source"]),
            "prep_source": str(binding["source"]), "cycle": cycle,
            "argv": argv, "unbound_supplement_roles": []}


def resolve_source_root(hints, *, data_dir=None, base_dir=None) -> Path:
    """Use the explicit run override, otherwise the config's local root."""
    value = data_dir if data_dir is not None else hints.get("source_root")
    if value is None or not str(value).strip():
        raise ValueError(
            f"{hints.get('source', 'This source')} needs local input bytes. "
            "Supply data_dir, --data-dir DIR, or [fetch].source_root.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Local source directory {path} does not exist. "
                         "Supply the directory containing the input files.")
    return path


def review_local_inputs(hints, *, data_dir=None, base_dir=None, supplements=()):
    """Review the actual local inputs for both check and execution planning."""
    from gpuwm.fetch import parse_cycle, validate_fetch_hints
    validate_fetch_hints(dict(hints), source="local preparation")
    missing = [key for key in ("source", "cycle", "hours") if hints.get(key) is None]
    if missing:
        raise ValueError("[fetch] is missing " + ", ".join(missing) +
                         "; what to do: supply the source, cycle and forecast window for the local inputs.")
    root = resolve_source_root(hints, data_dir=data_dir, base_dir=base_dir)
    return inspect_local_inputs(hints["source"], root,
        cycle=parse_cycle(hints["cycle"], hints["source"]), hours=hints["hours"],
        cadence=hints.get("cadence"), start_hour=hints.get("forecast_start_hour", 0),
        supplements=supplements)


def inspect_local_inputs(source: str, source_root: Path, *, cycle: datetime,
                         hours: float, cadence: int | None = None, start_hour: int = 0,
                         supplements=()) -> dict:
    """Validate a local preparation contract without writing or downloading."""
    from gpuwm import fetch_routes
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_cli import preparation_runners

    adapter = get_source_adapter(source)
    runner = preparation_runners().get(adapter.runner)
    if cadence is None and adapter.forcing_interval_seconds is not None:
        interval = adapter.forcing_interval_seconds
        cadence = int(interval / 3600) if interval % 3600 == 0 else None
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError(f"Local source directory {source_root} does not exist. "
                         "Supply the input directory.")
    if type(cadence) is not int or cadence <= 0:
        raise ValueError("Local boundary cadence must be a positive whole hour. "
                         "Set [fetch].cadence to the input spacing.")
    if isinstance(hours, bool) or not math.isfinite(float(hours)) or float(hours) <= 0:
        raise ValueError("Local forecast hours must be positive and finite. "
                         "Set [fetch].hours to the requested forecast length.")
    if type(start_hour) is not int or start_hour < 0:
        raise ValueError("Local start lead must be a nonnegative whole hour. "
                         "Correct [fetch].forecast_start_hour.")
    start = cycle + timedelta(hours=start_hour)
    required = [start + timedelta(hours=step * cadence)
                for step in range(math.ceil(float(hours) / cadence) + 1)]
    result = {"schema": "gpuwm-local-preparation-v1", "source": adapter.source_id,
              "source_root": str(source_root), "cycle": cycle.strftime("%Y-%m-%dT%H"),
              "valid_times": [value.isoformat() for value in required]}
    if runner is not None and runner.local_kind == "member_manifest":
        if supplements:
            raise ValueError(
                "The member manifest owns its pressure/surface pairing and has no "
                "external supplement slot. Remove supplement overrides or prepare "
                "a composed bundle separately.")
        manifest = runner.local_inventory(source_root)
        available = set(manifest["valid_times"])
        missing = [value for value in result["valid_times"] if value not in available]
        if missing:
            raise ValueError(f"Local input is missing valid times {missing}. "
                             "Supply the complete pressure and surface series.")
        selected = set(result["valid_times"])
        manifest = dict(manifest)
        manifest["valid_times"] = result["valid_times"]
        manifest["files"] = [row for row in manifest["files"]
                             if row["valid_time"] in selected]
        manifest["file_count"] = len(manifest["files"])
        manifest["cadence_seconds"] = cadence * 3600
        manifest.pop("content_sha256", None)
        manifest["content_sha256"] = _identity(manifest)
        result.update(kind=runner.local_kind, manifest=manifest,
                      files=manifest["files"])
    elif runner is not None and runner.local_kind == "prep_handoff":
        # Reuse the established handoff, not a guessed variable-to-role map.
        path = source_root / fetch_routes.PREP_ARGUMENTS_NAME
        bound_inputs = None
        if path.is_file():
            handoff = json.loads(path.read_text(encoding="utf-8"))
        elif fetch_routes.source_root_layout(adapter.source_id) is not None:
            # A hand-staged folder: the source's row says which file is
            # which, so the handoff a fetch would have written is derived
            # from the folder here, in memory.  `gpuwm go` then publishes
            # it into the run (publish_local_handoff) like any other.
            try:
                binding = bind_source_root(
                    adapter.source_id, source_root, cycle=cycle,
                    hours=hours, cadence=cadence)
            except ValueError as error:
                raise ValueError(
                    f"Local input at {source_root} has no "
                    f"{fetch_routes.PREP_ARGUMENTS_NAME}, and its files do "
                    f"not bind by {adapter.source_id}'s folder layout: "
                    f"{error}") from None
            handoff = source_root_handoff(binding, cycle=result["cycle"])
            bound_inputs = list(binding["inputs"])
        else:
            raise ValueError(
                f"Local input at {source_root} has no {fetch_routes.PREP_ARGUMENTS_NAME}. "
                "Bind its ordered inputs and required supplements with the "
                "source's preparation handoff, or supply a prepared bundle.")
        if (handoff.get("schema") != fetch_routes.PREP_ARGUMENTS_SCHEMA
                or handoff.get("source") != adapter.source_id
                or handoff.get("prep_source") != adapter.source_id
                or handoff.get("cycle") != result["cycle"]):
            raise ValueError("The local preparation handoff has a different source or cycle. "
                             "Rebuild it for this configuration.")
        argv = handoff.get("argv")
        allowed = {"--source", "--input-list", "--supplement", "--author-input-manifest"}
        if (not isinstance(argv, list) or len(argv) % 2
                or any(not isinstance(token, str) for token in argv)
                or any(flag not in allowed for flag in argv[::2])
                or argv[::2].count("--source") != 1
                or argv[::2].count("--input-list") != 1
                or argv[::2].count("--author-input-manifest") != 1):
            raise ValueError("The local handoff has an unsupported argument shape. "
                             "Use source, input-list, supplements and manifest authoring only.")
        pairs = list(zip(argv[::2], argv[1::2]))
        if dict(pairs)["--source"] != adapter.source_id:
            raise ValueError("The local handoff's source argument differs. Rebuild the handoff.")
        supplied = {str(value).split("=", 1)[0] for value in supplements}
        missing_roles = set(handoff.get("unbound_supplement_roles") or ()) - supplied
        if missing_roles:
            raise ValueError(f"Local preparation needs supplement roles {sorted(missing_roles)}. "
                             "Supply each with --supplement ROLE=PATH.")
        from gpuwm.source_authorities import packaged_composition
        # The cached, byte-verified reader -- not a second read of the
        # same document.  Plan review asks this of every registry row on
        # every configuration load, which is why it is cached at all.
        composition = packaged_composition(adapter.packaged_profile)
        required_roles = {
            str(row["data_role"])
            for group in ("supplements", "field_sources")
            for row in composition.get(group, {}).values()
            if "data_role" in row}
        bound_roles = supplied | {
            value.partition("=")[0] for flag, value in pairs if flag == "--supplement"}
        if required_roles != bound_roles:
            raise ValueError(
                f"Local supplement roles differ: missing {sorted(required_roles - bound_roles)}, "
                f"unexpected {sorted(bound_roles - required_roles)}. "
                "Bind exactly the roles in the source's composition.")
        from gpuwm.mapped_source import read_input_list
        listing = Path(dict(pairs)["--input-list"])
        if not listing.is_absolute():
            listing = source_root / listing
        paths = (list(bound_inputs) if bound_inputs is not None
                 else list(read_input_list(listing)))
        ordered_inputs = [str(file.resolve()) for file in paths]
        if any(not file.is_absolute() for file in paths):
            raise ValueError("The local handoff's input list contains relative paths. "
                             "Write absolute file paths so review and prep read the same bytes.")
        bound = []
        for flag, value in pairs:
            if flag == "--input-list":
                value = str(listing.resolve())
            elif flag == "--supplement":
                role, sep, raw = value.partition("=")
                if not role or not sep or not raw:
                    raise ValueError("A local supplement binding is malformed. Use ROLE=PATH.")
                file = Path(raw)
                file = file if file.is_absolute() else source_root / file
                paths.append(file.resolve())
                value = f"{role}={file.resolve()}"
            bound.extend((flag, value))
        # Verify external bindings as well before any stage runs.
        for binding in supplements:
            _, sep, raw = str(binding).partition("=")
            if not sep or not raw:
                raise ValueError("A supplement binding is malformed. Use ROLE=PATH.")
            paths.append(Path(raw).resolve())
        if not paths:
            raise ValueError("The local input list is empty. Supply the ordered input files.")
        files = []
        for file in sorted(set(map(Path, paths))):
            if not file.is_file():
                raise ValueError(f"Local input file {file} is missing. Restore the bound file.")
            from gpuwm.output_identity import file_record
            files.append(file_record(file))
        # A prepared input manifest will enforce the actual field and time
        # coverage. This path only claims source/cycle and byte binding.
        result.update(kind=runner.local_kind, argv=bound, files=files,
                      ordered_inputs=ordered_inputs,
                      unbound_supplement_roles=handoff.get("unbound_supplement_roles", []))
    else:
        raise ValueError(f"{source} has no local preparation contract. "
                         "Supply a supported prepared bundle.")
    result["sha256"] = _identity(result)
    return result


def verify_local_snapshot(snapshot: Mapping, *, input_list: Path | None = None) -> None:
    """Keep the reviewed bytes and ordered primary inputs through preparation."""
    content = dict(snapshot)
    claimed = content.pop("sha256", None)
    if claimed != _identity(content):
        raise ValueError("The local input snapshot digest differs. Resolve the plan again.")
    from gpuwm.output_identity import file_record
    for row in snapshot["files"]:
        path = Path(row["path"])
        if not path.is_file() or file_record(path)["sha256"] != row["sha256"]:
            raise ValueError(f"Local input {path} changed after review. Resolve the plan again.")
    if snapshot["kind"] == "prep_handoff":
        ordered = snapshot.get("ordered_inputs")
        known = {row["path"] for row in snapshot["files"]}
        if (not isinstance(ordered, list) or not ordered
                or any(not isinstance(path, str) or path not in known for path in ordered)):
            raise ValueError("The local snapshot lacks its reviewed ordered inputs. Resolve the plan again.")
        if input_list is not None:
            expected = "".join(f"{path}\n" for path in ordered).encode("utf-8")
            if not input_list.is_file() or input_list.read_bytes() != expected:
                raise ValueError("The reviewed local input list changed during preparation. Resolve the plan again.")


def publish_local_handoff(snapshot: Mapping, out: Path) -> Path:
    """Publish the inspected inputs, refusing bytes changed since inspection."""
    from gpuwm import fetch_routes
    from gpuwm.fetch_guard import atomic_write_text, hold

    verify_local_snapshot(snapshot)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with hold("local-preparation", out, progress=lambda _: None):
        if snapshot["kind"] == "member_manifest":
            manifest_path = out / "source-manifest.json"
            manifest_text = json.dumps(snapshot["manifest"], indent=2, sort_keys=True) + "\n"
            atomic_write_text(manifest_path, manifest_text)
            tokens = ["--source", str(snapshot["source"]), "--source-manifest",
                      str(manifest_path), "--source-manifest-sha256", _sha256(manifest_path)]
        else:
            tokens = list(snapshot["argv"])
            # The external list is an authoring input, not a live selector.
            # Publish the exact reviewed order in this generation's own file.
            input_list = out / f"input-list-{snapshot['sha256']}.txt"
            expected = "".join(f"{path}\n" for path in snapshot["ordered_inputs"])
            try:
                with input_list.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(expected)
            except FileExistsError:
                if input_list.read_bytes() != expected.encode("utf-8"):
                    # Preserve the damaged artifact and recover inside this
                    # owned generation without asking for filesystem changes.
                    from uuid import uuid4
                    input_list = out / f"input-list-{snapshot['sha256']}-{uuid4().hex}.txt"
                    with input_list.open("x", encoding="utf-8", newline="\n") as stream:
                        stream.write(expected)
            tokens[tokens.index("--input-list") + 1] = str(input_list)
            tokens[tokens.index("--author-input-manifest") + 1] = str(out / "inputs.json")
        path = fetch_routes.write_prep_arguments(
            out, source=str(snapshot["source"]), prep_source=str(snapshot["source"]),
            cycle=datetime.strptime(str(snapshot["cycle"]), "%Y-%m-%dT%H"),
            tokens=tokens, unbound_roles=snapshot.get("unbound_supplement_roles", ()))
        atomic_write_text(out / "local-inputs.json", json.dumps(snapshot, indent=2) + "\n")
        return path
