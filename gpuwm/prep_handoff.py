"""Preparation handoff consumption with byte-verified ensemble selection."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
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
