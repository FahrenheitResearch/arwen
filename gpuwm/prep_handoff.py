"""Preparation handoff consumption with byte-verified ensemble selection."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import shlex
from pathlib import Path
from typing import Mapping

from gpuwm import member_prep
from gpuwm.member_grammar import load_member_grammar
from gpuwm.source_authorities import packaged_member_grammar, packaged_member_grammar_sha256


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def lead_generation(steps: list[int]) -> str:
    """The folder name one member selection's lead list is staged under.

    A fetch folder is reused for a longer or shorter window, and each
    window selects its own leads.  Staging every lead list in its own
    folder keeps a tree an earlier preparation listed exactly as it was,
    so a changed window neither replaces it nor is refused by it.
    """
    digest = hashlib.sha256(json.dumps(list(steps)).encode("utf-8")).hexdigest()[:8]
    return f"f{steps[0]:03d}-f{steps[-1]:03d}-{digest}"


def preparation_arguments_from_directory(root) -> list[str]:
    """Consume a structured handoff or a verified older text handoff.

    Some native container acquisitions predate structured prep arguments.
    Their command text is already an artifact hashed by the fetch manifest.
    Parse that bound argv without invoking a shell or inventing source flags.
    """
    root = Path(root).resolve()
    structured = root / "prep-arguments.json"
    if structured.is_file():
        return preparation_arguments(json.loads(structured.read_text(encoding="utf-8")))
    manifest = json.loads((root / "fetch-manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("files", []) if row.get("role") == "prep-command"]
    if len(rows) != 1:
        raise ValueError("acquisition lacks a unique bound preparation handoff")
    row = rows[0]
    path = (root / row["name"]).resolve()
    if path.parent != root or not path.is_file() or _digest(path) != row.get("sha256"):
        raise ValueError("text preparation handoff differs from its acquisition manifest")
    text = path.read_text(encoding="utf-8").replace("\\\r\n", " ").replace("\\\n", " ")
    tokens = shlex.split(text, comments=True, posix=True)
    if tokens[:2] != ["gpuwm", "prep"]:
        raise ValueError("text acquisition handoff is not a preparation argument vector")
    argv = tokens[2:]
    if argv.count("--source") != 1 or argv.index("--source")+1 >= len(argv):
        raise ValueError("text preparation handoff lacks its source authority")
    if argv[argv.index("--source")+1] != manifest.get("source"):
        raise ValueError("text preparation handoff names a different acquisition source")
    return preparation_arguments({"argv":argv})


def _posted_handoff(root, trajectory=None):
    """Validate posted authority without opening any planned source payload."""
    from gpuwm.ensemble.recipes import SourceTrajectory
    from gpuwm.fetch_routes import PREP_ARGUMENTS_SCHEMA
    from gpuwm.forcing_member import member_contract
    from gpuwm.ingest.boundary_stream import (
        POSTING_DIRNAME, POSTING_SCHEDULE_NAME, read_replaced_json)
    from gpuwm.source_posting import SCHEDULE_SCHEMA

    root = Path(root).resolve()
    structured = root / "prep-arguments.json"
    if structured.is_file():
        document = json.loads(structured.read_text(encoding="utf-8"))
        if document.get("schema") != PREP_ARGUMENTS_SCHEMA:
            raise ValueError("posted preparation handoff schema differs")
        argv = document.get("argv")
        if (not isinstance(argv, list) or not argv
                or any(not isinstance(value, str) for value in argv)):
            raise ValueError("posted preparation arguments must be nonempty strings")
        argv = list(argv)
        if document.get("as_posted") is not True or not document.get("posting"):
            raise ValueError("posted preparation requires the acquisition's posted handoff")
        posting = Path(document["posting"]).resolve()
    else:
        # Older deterministic native fetchers publish a hash-bound command.
        # That reader performs no source-file access for a deterministic feed.
        argv = preparation_arguments_from_directory(root)
        posting = root / POSTING_DIRNAME
        document = None
    if posting.parent != root:
        raise ValueError("posted handoff names another acquisition's posting directory")
    schedule = read_replaced_json(posting / POSTING_SCHEDULE_NAME)
    if schedule.get("schema") != SCHEDULE_SCHEMA or not schedule.get("leads"):
        raise ValueError("posted handoff lacks the ordinary source posting schedule")

    def moment(value):
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace("_", "T"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    selected = SourceTrajectory(str(schedule["source"]), moment(schedule["cycle"]), schedule.get("member"))
    if trajectory is not None and (not isinstance(trajectory, SourceTrajectory) or trajectory != selected):
        raise ValueError("posted handoff differs from the requested source trajectory")
    if argv.count("--source") != 1 or argv.index("--source") + 1 == len(argv):
        raise ValueError("posted handoff lacks its unique preparation source")
    declared_source = argv[argv.index("--source") + 1]
    if document is not None:
        if (document.get("source") != selected.source or document.get("member") != selected.member
                or moment(document.get("cycle")) != selected.cycle
                or declared_source != document.get("prep_source", selected.source)):
            raise ValueError("posted handoff source, cycle or member differs from its source schedule")
    elif declared_source != selected.source:
        raise ValueError("posted native command source differs from its source schedule")
    if "--cycle" in argv and moment(argv[argv.index("--cycle") + 1]) != selected.cycle:
        raise ValueError("posted native command cycle differs from its source schedule")
    contract = member_contract(selected.source, selected.member)
    if contract is not None:
        adapter, grammar, member = contract
        if document is None or document.get("member_set") != adapter.member_set:
            raise ValueError("posted ensemble handoff lacks its registered member grammar")
        verification = document.get("member_verification")
        if verification is not None and verification != {"set": adapter.member_set, "member": member}:
            raise ValueError("posted ensemble verification names another grammar or member")
        spec = document.get("member_prep")
        if verification is None and spec is None:
            raise ValueError("posted ensemble handoff lacks native member verification or selection")
        if argv.count("--input-list") != 1 or argv.index("--input-list") + 1 == len(argv):
            raise ValueError("posted ensemble handoff needs one planned input list")
        if spec is not None:
            required = {"set", "member", "cycle", "steps", "inputs", "output", "input_list_after"}
            leads = [int(row["lead"]) for row in schedule["leads"]]
            if (not isinstance(spec, dict) or not required <= spec.keys()
                    or spec["set"] != adapter.member_set or spec["member"] != member
                    or moment(spec["cycle"]) != selected.cycle or spec["steps"] != leads):
                raise ValueError("posted native member selection differs from its frozen lead/source plan")
    elif document is not None and any(document.get(key) is not None for key in
                                     ("member_prep", "member_verification", "member_set")):
        raise ValueError("deterministic posted source cannot declare ensemble member selection")
    if "--as-posted" in argv:
        if argv.count("--as-posted") != 1 or Path(argv[argv.index("--as-posted") + 1]).resolve() != posting:
            raise ValueError("posted preparation argv names another posting plan")
    else:
        argv.extend(("--as-posted", str(posting)))
    return document, argv, schedule, selected, contract


def posted_preparation_arguments_from_directory(root, *, trajectory=None) -> list[str]:
    """Read the actual posted handoff without requiring future member files.

    The ordinary mapped producer calls :func:`verify_posted_member_batch`
    before decoding every ready batch. This only defers when native member
    selection and byte verification happen; it never drops that work.
    """
    return _posted_handoff(root, trajectory)[1]


def verify_posted_member_batch(root, *, leads, primary_files, trajectory=None):
    """Run ordinary native member work on exactly the already-ready batch.

    A handoff requiring a member tree still performs ordinary native staging
    for these leads. The mapped decoder may consume the original planned
    paths only after proving their bytes equal that selected staged tree.
    """
    document, argv, schedule, selected, contract = _posted_handoff(root, trajectory)
    requested = tuple(sorted(set(int(value) for value in leads)))
    planned = {int(row["lead"]) for row in schedule["leads"]}
    if not requested or not set(requested) <= planned:
        raise ValueError("posted member batch contains an unplanned lead")
    if contract is None:
        return None
    adapter, grammar, member = contract
    files = tuple(Path(path).resolve() for path in primary_files)
    if not files or len(set(files)) != len(files):
        raise ValueError("posted member batch needs distinct actual primary files")
    input_list = Path(argv[argv.index("--input-list") + 1])
    listed = {Path(line).resolve() for line in input_list.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    if not set(files) <= listed:
        raise ValueError("posted member batch reads a primary absent from its acquisition input plan")
    before = {str(path): _digest(path) for path in files}
    selected_files = None
    if document.get("member_prep") is not None:
        # Reuse the one native staging/verification door with only this ready
        # lead set. Its generation name remains create-only and hash-bound.
        narrowed = dict(document, member_prep={**document["member_prep"], "steps": list(requested)})
        staged_argv = preparation_arguments(narrowed)
        staged_list = Path(staged_argv[staged_argv.index("--input-list") + 1])
        selected_files = tuple(Path(line).resolve() for line in
                               staged_list.read_text(encoding="utf-8").splitlines() if line.strip())
        if sorted(_digest(path) for path in selected_files) != sorted(before.values()):
            raise ValueError("posted primary bytes differ from the ordinary selected member batch")
    else:
        for path in files:
            member_prep.verify_member_file(grammar, member, path)
    if {str(path): _digest(path) for path in files} != before:
        raise ValueError("posted member bytes changed during native verification")
    return {"schema": "gpuwm-posted-member-batch.v1", "source": selected.source,
            "cycle": selected.cycle.isoformat(), "member": member,
            "grammar_sha256": packaged_member_grammar_sha256(adapter.member_set),
            "leads": list(requested), "files": before,
            "selected_files": None if selected_files is None else
                {str(path): _digest(path) for path in selected_files}}


def preparation_arguments(handoff: Mapping[str, object]) -> list[str]:
    """Run any declared member selection before exposing inputs to prep.

    A previous selected tree is reusable only when every expected file is
    still identical to its upstream input and passes the native member
    identity check again. Extra, missing and changed files do not pass.
    Each lead list is staged under its own :func:`lead_generation`
    folder, so reusing the fetch folder for another window stages that
    window beside the earlier one.
    """
    raw_argv = handoff.get("argv")
    if not isinstance(raw_argv, list):
        raise ValueError("Preparation arguments must be a list. Rebuild the handoff.")
    argv = list(raw_argv)
    if any(not isinstance(token, str) for token in argv):
        raise ValueError("Preparation arguments must be strings. Rebuild the handoff.")
    spec = handoff.get("member_prep")
    if spec is None:
        _verify_selected_inputs(handoff, argv)
        return argv
    if not isinstance(spec, dict):
        raise ValueError("Member preparation must be an object. Rebuild the handoff.")
    required = {"set", "member", "cycle", "steps", "inputs", "output", "input_list_after"}
    if not required <= spec.keys():
        raise ValueError(
            f"Member preparation is missing {sorted(required - spec.keys())}. "
            "Rebuild the acquisition handoff.")
    if spec["set"] != handoff.get("member_set") or spec["member"] != handoff.get("member"):
        raise ValueError("Member preparation disagrees with the selected member. Rebuild the handoff.")
    cycle = datetime.strptime(str(spec["cycle"]), "%Y-%m-%dT%H")
    if handoff.get("cycle") != spec["cycle"]:
        raise ValueError("Member preparation disagrees with the cycle. Rebuild the handoff.")
    steps = spec["steps"]
    if (not isinstance(steps, list) or not steps
            or any(type(step) is not int or step < 0 for step in steps)
            or steps != sorted(set(steps))):
        raise ValueError("Member steps must be increasing nonnegative integer leads. Rebuild the handoff.")
    if argv.count("--input-list") != 1:
        raise ValueError("Member preparation requires one --input-list. Rebuild the handoff.")
    input_index = argv.index("--input-list") + 1
    if input_index >= len(argv):
        raise ValueError("The member input-list path is missing. Rebuild the handoff.")
    grammar = load_member_grammar(packaged_member_grammar(str(spec["set"])))
    selected = grammar.member(str(spec["member"]))
    inputs = Path(str(spec["inputs"])).resolve()
    generation = lead_generation(steps)
    output = Path(str(spec["output"])).resolve() / generation
    member_dir = output / grammar.name / (cycle.strftime("%Y%m%dT%H") + "Z") / selected.member_id
    from gpuwm.fetch_guard import hold
    with hold("member-preparation", member_dir, progress=lambda _: None):
        existed = member_dir.exists()
        if not existed:
            member_dir = member_prep.prepare_member(
                grammar_id=str(spec["set"]), member_id=selected.member_id,
                cycle=cycle, steps=steps, inputs_root=inputs, output_root=output)
        receipt_path = member_dir / member_prep.RECEIPT_NAME
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"The selected member receipt at {receipt_path} is unreadable. "
                "Use a clean output directory.") from error
        if not isinstance(receipt, dict):
            raise ValueError("The selected member receipt is not an object. "
                             "Use a clean output directory.")
        if (receipt.get("schema") != member_prep.RECEIPT_SCHEMA
                or receipt.get("member", {}).get("id") != selected.member_id
                or receipt.get("cycle") != cycle.strftime("%Y-%m-%dT%H:00:00Z")
                or receipt.get("member_set", {}).get("registry_id") != spec["set"]
                or receipt.get("member_set", {}).get("sha256")
                != packaged_member_grammar_sha256(str(spec["set"]))):
            raise ValueError("The selected member receipt has a different identity. Use a clean output directory.")
        expected = []
        for product in grammar.products():
            for step in steps:
                relative = grammar.relative_path(selected.member_id, product, cycle, step)
                staged = Path(product) / Path(relative).name
                expected.append((product, step, relative, staged))
        records = receipt.get("files") or []
        if len(records) != len(expected):
            raise ValueError("The selected member receipt has a different file count. Use a clean output directory.")
        by_staged = {row["staged"]: row for row in records}
        if set(by_staged) != {item[3].as_posix() for item in expected}:
            raise ValueError("The selected member inventory differs. Use a clean output directory.")
        ordered = []
        for product, step, relative, staged in sorted(expected, key=lambda item: (item[1], item[0])):
            row = by_staged[staged.as_posix()]
            source_path = inputs / relative
            path = member_dir / staged
            if (row.get("product") != product or row.get("step_hours") != step
                    or row.get("relative_source") != relative
                    or not path.is_file() or not source_path.is_file()
                    or _digest(path) != row.get("sha256")
                    or _digest(source_path) != row.get("sha256")):
                raise ValueError(f"Selected member file {path} differs from its input. Use a clean output directory.")
            if existed:
                member_prep.verify_member_file(grammar, selected.member_id, path)
            ordered.append(path.resolve())
        actual = {path.relative_to(member_dir).as_posix()
                  for path in member_dir.rglob("*") if path.is_file()}
        if actual != set(by_staged) | {member_prep.RECEIPT_NAME}:
            raise ValueError("The selected member tree contains unexpected files. Use a clean output directory.")
        declared = Path(str(spec["input_list_after"])).resolve()
        input_list = declared.with_name(f"{declared.stem}-{generation}{declared.suffix}")
        from gpuwm.fetch_guard import atomic_write_text
        input_list.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(input_list,
                          "".join(f"{path}\n" for path in ordered),
                          tag="member-prep")
        argv[input_index] = str(input_list)
        return argv


def _verify_selected_inputs(handoff: Mapping[str, object], argv: list[str]) -> None:
    """Check every composed input against the declared single member."""
    spec = handoff.get("member_verification")
    if spec is None:
        return
    if (not isinstance(spec, dict)
            or spec.get("set") != handoff.get("member_set")
            or spec.get("member") != handoff.get("member")
            or not spec.get("set") or not spec.get("member")):
        raise ValueError("Member verification disagrees with the handoff. Rebuild the acquisition.")
    if argv.count("--input-list") != 1 or argv.index("--input-list") + 1 >= len(argv):
        raise ValueError("Member verification requires one input list. Rebuild the acquisition.")
    from gpuwm.mapped_source import read_input_list
    grammar = load_member_grammar(packaged_member_grammar(str(spec["set"])))
    grammar.member(str(spec["member"]))
    for path in read_input_list(Path(argv[argv.index("--input-list") + 1])):
        member_prep.verify_member_file(grammar, str(spec["member"]), Path(path))
