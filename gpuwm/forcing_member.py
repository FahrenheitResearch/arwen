"""Bind a selected forcing member to the GRIB messages preparation consumes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def member_contract(source: str, member: str | None = None):
    from gpuwm.fetch_routes import route_for, resolve_member
    from gpuwm.member_grammar import load_member_grammar
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_authorities import packaged_member_grammar

    adapter = get_source_adapter(source)
    if adapter.member_set is None:
        return None
    selection, _ = resolve_member(route_for(adapter.source_id), member)
    grammar = load_member_grammar(packaged_member_grammar(adapter.member_set))
    grammar.member(selection)
    return adapter, grammar, selection


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_handoff(hints: Mapping, handoff: Mapping, *, out: Path) -> dict | None:
    """Inspect actual input bytes, never infer identity from a member filename.

    The receipt binds member, grammar and file contents for preparation reuse
    without adding preparer flags. The verified input list is create-only.
    Native inventory and all grammar checks run on every launch, including a
    preparation cache hit. Deterministic sources do not enter this path.
    """
    from gpuwm.member_grammar import MemberIdentityRefusal
    from gpuwm.member_prep import verify_member_file
    from gpuwm.source_authorities import packaged_member_grammar_sha256
    from gpuwm.fetch import parse_cycle

    contract = member_contract(str(hints["source"]), hints.get("member"))
    if contract is None:
        return None
    adapter, grammar, selection = contract
    if handoff.get("source") != adapter.source_id or handoff.get("member") != selection:
        raise MemberIdentityRefusal("Fetch handoff source/member differs from the selected forcing member")
    if parse_cycle(str(handoff.get("cycle", "")), adapter.source_id) != parse_cycle(str(hints["cycle"]), adapter.source_id):
        raise MemberIdentityRefusal("Fetch handoff cycle differs from the selected forcing cycle")
    arguments = list(handoff.get("argv") or [])
    if arguments.count("--input-list") != 1:
        raise MemberIdentityRefusal("Member preparation needs exactly one bound --input-list")
    index = arguments.index("--input-list") + 1
    if index == len(arguments):
        raise MemberIdentityRefusal("Member preparation input-list path is missing")
    input_list = Path(arguments[index]).resolve(strict=True)
    text = input_list.read_text(encoding="utf-8")
    paths = [Path(line).resolve(strict=True) for line in text.splitlines() if line.strip()]
    if not paths or len(set(paths)) != len(paths):
        raise MemberIdentityRefusal("Member input-list must contain nonempty, unique file paths")
    evidence = []
    for path in paths:
        before = _digest(path)
        verified = verify_member_file(grammar, selection, path)
        if before != _digest(path):
            raise MemberIdentityRefusal(f"Member input changed during verification: {path}")
        evidence.append({"path": str(path), "sha256": before, **verified.to_dict()})
    if input_list.read_text(encoding="utf-8") != text:
        raise MemberIdentityRefusal("Member input-list changed during verification")
    receipt = {"schema": "arwen.forcing-member.v1", "source": adapter.source_id,
               "member": selection, "cycle": str(handoff["cycle"]),
               "member_set": adapter.member_set,
               "grammar_sha256": packaged_member_grammar_sha256(adapter.member_set),
               "files": evidence}
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verified_list = out / f"member-inputs-{key}.txt"
    verified_text = "".join(f"{path}\n" for path in paths)
    # Create once and check on reuse. An existing content-addressed artifact
    # with different bytes is a corruption, not something to overwrite.
    try:
        with verified_list.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(verified_text)
    except FileExistsError:
        if verified_list.read_text(encoding="utf-8") != verified_text:
            raise MemberIdentityRefusal(f"Verified member input-list is corrupt: {verified_list}")
    receipt["input_list"] = str(verified_list.resolve())
    (out / f"member-verification-{key}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def verify_unchanged(receipt: Mapping | None) -> None:
    """Preparation must not race a replacement of verified forcing bytes."""
    from gpuwm.member_grammar import MemberIdentityRefusal

    if receipt is None:
        return
    files = receipt["files"]
    expected_list = "".join(f"{row['path']}\n" for row in files)
    if Path(receipt["input_list"]).read_text(encoding="utf-8") != expected_list:
        raise MemberIdentityRefusal("Verified member input-list changed during preparation")
    for row in files:
        if _digest(Path(row["path"])) != row["sha256"]:
            raise MemberIdentityRefusal(f"Member input changed during preparation: {row['path']}")


def prepare_verified(receipt: Mapping | None, root: Path, prepare, *,
                     notes: dict | None = None):
    """Reuse only a preparation that records this exact verified member input.

    ``notes``, when given, receives ``member_input_superseded`` before
    ``prepare`` runs, so a chain that reports the seal from inside
    ``prepare`` carries the same record the returned receipt does.
    """
    if receipt is None:
        return prepare()
    from gpuwm import stage_reuse

    verify_unchanged(receipt)
    path = Path(root) / "forcing-member.json"
    previous = None
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    superseded = None
    if Path(root).exists() and previous != receipt:
        superseded = stage_reuse.supersede(Path(root))
        if notes is not None:
            notes["member_input_superseded"] = superseded
    result = prepare()
    verify_unchanged(receipt)
    path.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    if superseded is not None:
        result = {**result, "member_input_superseded": superseded}
    return result
