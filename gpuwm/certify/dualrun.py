"""``gpuwm dual-run``: two capsules, one answer, and the field that broke it.

The 5090 has no ECC.  A dual run is how this project detects the corruption
the card cannot report: run the same configuration twice and compare the two
receipts field for field.  The comparison is total -- there is no ignore list,
because every field a comparison skips is a field a silent flip can hide in --
and it is ordered, so "the first divergent field" is a property of the two
documents rather than of dictionary iteration order.

The same command is what compares the trajectory-digest instrumentation's
on and off arms: two capsules from the same short window, one with the digest
enabled, and a divergence anywhere outside the digest itself is a finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Sentinel for "this document has no value at this path at all".  A key that
#: exists with value ``null`` and a key that does not exist are different
#: claims, and a comparison that conflated them would miss a dropped field.
ABSENT = "<absent>"


@dataclass(frozen=True)
class Divergence:
    """One path at which two capsules disagree."""

    field_path: str
    left: Any
    right: Any

    def describe(self) -> str:
        return (f"{self.field_path}: "
                f"{json.dumps(self.left, sort_keys=True, default=str)} != "
                f"{json.dumps(self.right, sort_keys=True, default=str)}")


@dataclass(frozen=True)
class DualRunComparison:
    """The result of comparing two capsules."""

    identical: bool
    divergences: tuple[Divergence, ...] = field(default=())

    @property
    def first_divergent_field(self) -> str | None:
        return self.divergences[0].field_path if self.divergences else None

    def report(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm.dual-run-comparison/v1",
            "identical": self.identical,
            "first_divergent_field": self.first_divergent_field,
            "divergence_count": len(self.divergences),
            "divergences": [
                {"field": item.field_path, "a": item.left, "b": item.right}
                for item in self.divergences
            ],
        }


#: Path punctuation, and the escapes that let a key contain it.  Kernel
#: manifest keys are dotted module names (``gpuwm.core.kernels:diff6``), so a
#: path built by naive dot-joining would be ambiguous exactly where the
#: comparison matters most.  A key that needs no escaping is written verbatim,
#: which keeps the ordinary path (``output.frames[0].sha256``) readable.
_ESCAPES: tuple[tuple[str, str], ...] = (("~", "~0"), (".", "~1"),
                                         ("[", "~2"), ("]", "~3"))


def encode_key(key: str) -> str:
    """One mapping key, escaped so a path can be split back apart."""
    text = str(key)
    for raw, escaped in _ESCAPES:
        text = text.replace(raw, escaped)
    return text


def decode_key(token: str) -> str:
    """Inverse of :func:`encode_key`."""
    text = token
    for raw, escaped in reversed(_ESCAPES):
        text = text.replace(escaped, raw)
    return text


def split_path(path: str) -> list[str | int]:
    """A field path as the sequence of steps that reaches it."""
    steps: list[str | int] = []
    for part in path.split("."):
        while "[" in part:
            head, _, rest = part.partition("[")
            if head:
                steps.append(decode_key(head))
            index, _, part = rest.partition("]")
            steps.append(int(index))
        if part:
            steps.append(decode_key(part))
    return steps


def leaf_value(document: Any, path: str) -> Any:
    """The value at ``path``.  Raises if the path does not exist."""
    node = document
    for step in split_path(path):
        node = node[step]
    return node


def set_leaf(document: Any, path: str, value: Any) -> None:
    """Assign ``value`` at ``path``, in place."""
    steps = split_path(path)
    node = document
    for step in steps[:-1]:
        node = node[step]
    node[steps[-1]] = value


def delete_leaf(document: Any, path: str) -> None:
    """Remove whatever ``path`` names, in place."""
    steps = split_path(path)
    node = document
    for step in steps[:-1]:
        node = node[step]
    del node[steps[-1]]


def flatten_capsule(document: Mapping[str, Any] | Any,
                    prefix: str = "") -> dict[str, Any]:
    """Every leaf of a capsule, keyed by its dotted/indexed path.

    Containers are not leaves, but an *empty* container is: otherwise a
    section emptied of its contents would flatten to nothing and compare equal
    to a section that was never there.
    """
    flat: dict[str, Any] = {}
    if isinstance(document, Mapping):
        if not document and prefix:
            flat[prefix] = {}
            return flat
        for key in document:
            token = encode_key(key)
            path = f"{prefix}.{token}" if prefix else token
            flat.update(flatten_capsule(document[key], path))
        return flat
    if isinstance(document, (list, tuple)):
        if not document and prefix:
            flat[prefix] = []
            return flat
        for index, item in enumerate(document):
            flat.update(flatten_capsule(item, f"{prefix}[{index}]"))
        return flat
    flat[prefix or "<root>"] = document
    return flat


def compare_capsules(left: Mapping[str, Any],
                     right: Mapping[str, Any]) -> DualRunComparison:
    """Compare two capsules leaf for leaf, in a deterministic path order."""
    flat_left = flatten_capsule(left)
    flat_right = flatten_capsule(right)
    divergences = []
    for path in sorted(set(flat_left) | set(flat_right)):
        a = flat_left.get(path, ABSENT)
        b = flat_right.get(path, ABSENT)
        if a != b or type(a) is not type(b):
            divergences.append(Divergence(path, a, b))
    return DualRunComparison(identical=not divergences,
                             divergences=tuple(divergences))


def compare_capsule_files(path_a: str | Path,
                          path_b: str | Path) -> DualRunComparison:
    """Read two capsules from disk and compare them."""
    return compare_capsules(
        json.loads(Path(path_a).read_text(encoding="utf-8")),
        json.loads(Path(path_b).read_text(encoding="utf-8")))


def capsule_field_paths(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Every mutable leaf path of a capsule, in comparison order.

    This is what the mutation table iterates: one row per capsule field, so a
    field added to the schema later shows up here without anyone editing a
    list.
    """
    return tuple(sorted(flatten_capsule(document)))


__all__ = [
    "ABSENT",
    "Divergence",
    "DualRunComparison",
    "capsule_field_paths",
    "compare_capsule_files",
    "compare_capsules",
    "decode_key",
    "delete_leaf",
    "encode_key",
    "flatten_capsule",
    "leaf_value",
    "set_leaf",
    "split_path",
]
