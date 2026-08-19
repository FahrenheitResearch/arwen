"""Author packaged-profile authority documents from a declared spec.

One tool, no model in it.  A source's authority documents -- the
mapping, the composition, the provenance record -- are produced from a
SPEC FILE that carries every source-specific fact as data.  Adding a
model is writing a spec beside the other table data; it is never a new
script, and never a branch in here.

Two shapes of derivation exist in practice and both are declared, not
coded:

* from nothing -- the document is built out of the spec's own templates
  and constants (a source whose grammar was measured from its octets);
* from a parent -- the document starts as an existing JSON authority or
  fixture and a short ordered list of edits states exactly how it
  differs (a donor mapping, or a profile that reuses a proven one).

Both run through the same closed step catalog below.  An unknown step,
an unknown schema or a missing path REFUSES by name: a silent no-op
would author a wrong authority and pin its hash, and the shipped
document would then disagree with the derivation that claims to
produce it.

Spec grammar (``gpuwm-source-profile-authoring-spec-v1``)::

    {
      "schema": "gpuwm-source-profile-authoring-spec-v1",
      "constants": {"NAME": <any JSON>},
      "templates": {"name": <JSON with "$placeholder" strings>},
      "documents": [
        {"output": "<repo-relative path>",
         "format": {"indent": 1},
         "base": {"kind": "empty"} | {"kind": "file", "path": "..."},
         "steps": [ ... ]}
      ]
    }

Steps:

``set``          set ``value`` at ``path`` (creating mappings as needed)
``delete``       remove ``path``
``keep_where``   filter the list at ``path`` on one key's value
``require``      assert ``path`` exists (a guard for parent documents)

Path elements are object keys or list indices.  ``"*"`` means every key
of a mapping or every element of a list; a LIST of keys means each of
those keys.  A path may therefore reach every selector of every field
at once, which is how a template pin is stated as data.

Values are resolved before use.  A one-key mapping is a directive:

``{"$const": "NAME"}``             the spec's constant
``{"$template": "name", "with": {...}}``  the template, placeholders filled
``{"$expand": {"template": "name", "for_each": [...], "as": "x"}}``
                                   a list, one filled template per item
``{"$sha256_of": "<repo-relative>"}``  the digest of an ALREADY-written
                                   document, so a composition can pin
                                   the mapping it contributes

Documents are authored in declared order, so a ``$sha256_of`` can only
name a document this run has already produced -- naming a later one
refuses rather than pinning a stale digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "gpuwm-source-profile-authoring-spec-v1"

REPO_ROOT = Path(__file__).resolve().parents[1]


class SpecError(Exception):
    """A spec that cannot be executed exactly as written."""


# --------------------------------------------------------------------------
# value resolution


def _fill(node: Any, bindings: dict[str, Any]) -> Any:
    """Substitute ``$placeholder`` strings inside a template body."""
    if isinstance(node, str):
        if node.startswith("$"):
            name = node[1:]
            if name not in bindings:
                raise SpecError(
                    f"template placeholder {node!r} has no binding; bound "
                    f"names are {sorted(bindings)}")
            return bindings[name]
        return node
    if isinstance(node, list):
        return [_fill(item, bindings) for item in node]
    if isinstance(node, dict):
        filled = {}
        for key, value in node.items():
            resolved = _fill(value, bindings)
            # A placeholder resolving to the omission marker drops the key,
            # which is how one template serves a shape whose optional
            # member is genuinely absent rather than null.
            if resolved is _OMIT:
                continue
            filled[key] = resolved
        return filled
    return node


class _Omit:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<omit>"


_OMIT = _Omit()


class Author:
    """Executes one spec.  Holds the spec's constants and templates."""

    def __init__(self, spec: dict, out_root: Path, read_root: Path) -> None:
        schema = spec.get("schema")
        if schema != SCHEMA:
            raise SpecError(
                f"spec schema {schema!r} is not {SCHEMA!r}; this tool "
                "executes that grammar and no other")
        self.constants = spec.get("constants", {})
        self.templates = spec.get("templates", {})
        self.documents = spec.get("documents", [])
        self.out_root = out_root
        self.read_root = read_root
        self.written: dict[str, bytes] = {}

    # -- values ------------------------------------------------------------

    def resolve(self, node: Any) -> Any:
        if isinstance(node, dict) and len(node) == 1:
            (key,) = node
            if key == "$const":
                name = node[key]
                if name not in self.constants:
                    raise SpecError(
                        f"constant {name!r} is not declared; declared "
                        f"constants are {sorted(self.constants)}")
                return self.resolve(self.constants[name])
            if key == "$sha256_of":
                return self._digest_of(node[key])
        if isinstance(node, dict) and "$template" in node:
            return self._template(node)
        if isinstance(node, dict) and "$expand" in node:
            return self._expand(node["$expand"])
        if isinstance(node, dict) and "$omit" in node:
            return _OMIT
        if isinstance(node, dict):
            resolved = {}
            for key, value in node.items():
                value = self.resolve(value)
                if value is _OMIT:
                    continue
                resolved[key] = value
            return resolved
        if isinstance(node, list):
            out = []
            for item in node:
                item = self.resolve(item)
                if item is _OMIT:
                    continue
                out.append(item)
            return out
        return node

    def _template(self, node: dict) -> Any:
        name = node["$template"]
        if name not in self.templates:
            raise SpecError(
                f"template {name!r} is not declared; declared templates "
                f"are {sorted(self.templates)}")
        bindings = {key: self.resolve(value)
                    for key, value in node.get("with", {}).items()}
        return self.resolve(_fill(self.templates[name], bindings))

    def _expand(self, directive: dict) -> list:
        name = directive["template"]
        if name not in self.templates:
            raise SpecError(
                f"template {name!r} is not declared; declared templates "
                f"are {sorted(self.templates)}")
        variable = directive.get("as", "item")
        items = self.resolve(directive["for_each"])
        common = {key: self.resolve(value)
                  for key, value in directive.get("with", {}).items()}
        expanded = []
        for item in items:
            bindings = dict(common)
            bindings[variable] = item
            if isinstance(item, dict):
                # A mapping item also binds its own members by name, so a
                # two-part row (a soil layer's top and bottom, say) reads
                # as "$top"/"$bottom" instead of index arithmetic.
                for key, value in item.items():
                    bindings[key] = value
            expanded.append(self.resolve(_fill(self.templates[name],
                                               bindings)))
        return expanded

    def _digest_of(self, relative: str) -> str:
        if relative not in self.written:
            raise SpecError(
                f"$sha256_of names {relative!r}, which this spec has not "
                f"written yet; written so far: {sorted(self.written)}")
        return hashlib.sha256(self.written[relative]).hexdigest()

    # -- paths -------------------------------------------------------------

    @staticmethod
    def _targets(document: Any, path: list,
                 create_last: bool = False) -> list[tuple[Any, Any]]:
        """[(container, key)] for every place ``path`` reaches.

        A missing INTERMEDIATE key refuses: inventing the structure would
        author a document nobody declared.  Only the final element of a
        ``set`` may name a key that does not exist yet, and only when
        ``create_last`` says so.  Prefix a string element with ``?`` to
        mean "if present" -- that branch then contributes no targets
        instead of refusing, which is how one step reaches every selector
        that exists without inventing one where a field declares none.
        """
        nodes: list[Any] = [document]
        places: list[tuple[Any, Any]] = []
        for depth, element in enumerate(path):
            last = depth == len(path) - 1
            next_nodes: list[Any] = []
            places = []
            for node in nodes:
                if element == "*":
                    if isinstance(node, dict):
                        keys: list[Any] = list(node.keys())
                    elif isinstance(node, list):
                        keys = list(range(len(node)))
                    else:
                        raise SpecError(
                            f"path element {depth} is '*' but the value "
                            f"there is a {type(node).__name__}")
                    for key in keys:
                        places.append((node, key))
                        if not last:
                            next_nodes.append(node[key])
                    continue

                if isinstance(element, list):
                    wanted = list(element)
                    optional = False
                else:
                    optional = (isinstance(element, str)
                                and element.startswith("?"))
                    wanted = [element[1:] if optional else element]

                for key in wanted:
                    if isinstance(node, dict):
                        if key not in node:
                            if optional:
                                continue
                            if last and create_last:
                                places.append((node, key))
                                continue
                            raise SpecError(
                                f"path element {depth} names {key!r}, which "
                                f"is absent; present here: {sorted(node)}")
                    elif isinstance(node, list):
                        if not isinstance(key, int):
                            raise SpecError(
                                f"path element {depth} is {key!r}, but the "
                                "value there is a list")
                    else:
                        raise SpecError(
                            f"path element {depth} descends into a "
                            f"{type(node).__name__}")
                    places.append((node, key))
                    if not last:
                        next_nodes.append(node[key])
            nodes = next_nodes
        return places

    # -- steps -------------------------------------------------------------

    def apply(self, document: Any, step: dict) -> Any:
        op = step.get("op")
        if op == "set":
            value = self.resolve(step["value"])
            path = step["path"]
            if not path:
                return value
            for container, key in self._targets(document, path,
                                                create_last=True):
                container[key] = value
            return document
        if op == "delete":
            for container, key in self._targets(document, step["path"]):
                del container[key]
            return document
        if op == "keep_where":
            key = step["key"]
            for container, index in self._targets(document, step["path"]):
                items = container[index]
                if not isinstance(items, list):
                    raise SpecError(
                        f"keep_where needs a list at {step['path']}, found "
                        f"a {type(items).__name__}")
                if "equals" in step:
                    wanted = self.resolve(step["equals"])
                    kept = [i for i in items if i.get(key) == wanted]
                elif "not_equals" in step:
                    unwanted = self.resolve(step["not_equals"])
                    kept = [i for i in items if i.get(key) != unwanted]
                else:
                    raise SpecError(
                        "keep_where needs 'equals' or 'not_equals'")
                container[index] = kept
            return document
        if op == "require":
            path = step["path"]
            node = document
            for depth, element in enumerate(path):
                if isinstance(node, dict) and element in node:
                    node = node[element]
                    continue
                raise SpecError(
                    f"required path {path} is absent at element {depth} "
                    f"({element!r}); the parent document this spec derives "
                    "from is not the one it was written against")
            return document
        raise SpecError(
            f"unknown step op {op!r}; this tool executes "
            "set, delete, keep_where and require, and refuses anything "
            "else rather than skipping it")

    # -- documents ---------------------------------------------------------

    def _base(self, declared: dict) -> Any:
        kind = declared.get("kind")
        if kind == "empty":
            return {}
        if kind == "file":
            path = self.read_root / declared["path"]
            if not path.is_file():
                raise SpecError(
                    f"base document {declared['path']} does not exist; a "
                    "derived authority cannot be authored without its "
                    "parent")
            return json.loads(path.read_text(encoding="utf-8"))
        raise SpecError(
            f"unknown base kind {kind!r}; a document starts either 'empty' "
            "or from a 'file'")

    def run(self) -> list[tuple[str, str]]:
        produced = []
        for declared in self.documents:
            relative = declared["output"]
            document = self._base(declared.get("base", {"kind": "empty"}))
            for step in declared.get("steps", []):
                document = self.apply(document, step)
            fmt = declared.get("format", {})
            text = json.dumps(document, indent=fmt.get("indent", 1))
            if fmt.get("trailing_newline", True):
                text += "\n"
            payload = text.encode("utf-8")
            self.written[relative] = payload
            path = self.out_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            produced.append((relative,
                             hashlib.sha256(payload).hexdigest()))
        return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Author packaged-profile authority documents from a "
                     "declared spec file."))
    parser.add_argument("spec", type=Path,
                        help="a *.profile-spec.json document")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT,
                        help=("root the documents are written under "
                              "(default: the repository root)"))
    parser.add_argument("--read-root", type=Path, default=REPO_ROOT,
                        help=("root that parent documents are read from "
                              "(default: the repository root)"))
    args = parser.parse_args(argv)

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"cannot read spec: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"spec is not JSON: {exc}", file=sys.stderr)
        return 2

    try:
        author = Author(spec, args.out_root.resolve(),
                        args.read_root.resolve())
        produced = author.run()
    except SpecError as exc:
        print(f"{args.spec}: {exc}", file=sys.stderr)
        return 1

    for relative, digest in produced:
        print(f"{digest}  {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
