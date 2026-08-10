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

The one place the comparison is not literal is where a field records WHERE a
run wrote rather than WHAT it computed.  Two runs of one configuration must
write to different directories -- otherwise they overwrite each other -- so a
literal comparison of the absolute output paths made a clean verdict
unreachable for every control pair, which is the opposite of what this
command is for.  Those fields are NORMALIZED to the leaf name and still
compared: a run that wrote the wrong domain, the wrong valid time or the
wrong number of frames still diverges, and nothing is skipped.  Every byte
that carries physics -- the per-frame SHA-256 set, the trajectory digest, the
pins, the kernel manifest, the input bytes -- is compared verbatim.

Because the comparison is total, it is also vacuous on nothing, and that was
a hole: handed two empty capsules the command found no divergence and said
so.  The screen now refuses a comparison with no leaves in it
(:data:`MINIMUM_COMPARED_LEAVES`) and states how many quantities it compared
when it passes, so "identical" is a claim with a size attached to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Sentinel for "this document has no value at this path at all".  A key that
#: exists with value ``null`` and a key that does not exist are different
#: claims, and a comparison that conflated them would miss a dropped field.
ABSENT = "<absent>"

#: How many leaves two documents must offer before "identical" means
#: anything.
#:
#: Handed two empty capsules -- ``{}`` and ``{}`` -- this command used to
#: print ``dual-run: capsules are identical field for field`` and exit 0.
#: It is total by construction, so it found no divergence, because there
#: was no field: the detector for the corruption a card with no ECC cannot
#: report answered its one question with a green on nothing.
#:
#: One, not some larger number.  Any threshold above "something" would be
#: a figure nobody can derive from the artifact, and inventing one is how
#: a gate starts refusing legitimate work.  The other half of the closure
#: is the count in the success line: a run whose capsules carry a hundred
#: leaves says so, and a reader who is told ``1 compared quantity`` can
#: see the screen was hollow without anyone having guessed a floor.
MINIMUM_COMPARED_LEAVES = 1


class DualRunError(ValueError):
    """The two documents cannot be compared at all.

    A ``ValueError``, so ``gpuwm.cli`` prints it as a refusal and exits 2
    the way every other documented refusal in this package does, rather
    than surfacing a traceback.
    """


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
    """The result of comparing two capsules.

    ``compared_count`` is how many leaf paths the comparison actually
    visited -- the union of the two documents' leaves.  It is carried
    rather than derived because "identical" on its own is the same
    sentence whether the screen compared a hundred quantities or none,
    and a reader deciding whether a card is silently corrupting memory
    needs to know which.
    """

    identical: bool
    divergences: tuple[Divergence, ...] = field(default=())
    compared_count: int = 0

    @property
    def first_divergent_field(self) -> str | None:
        return self.divergences[0].field_path if self.divergences else None

    def report(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm.dual-run-comparison/v1",
            "identical": self.identical,
            "compared_count": self.compared_count,
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


#: Leaf paths that record the DIRECTORY a run wrote into rather than any
#: computed value.  ``output.frames[i].path`` is where a history frame
#: landed -- its bytes are compared through ``.sha256`` and ``.bytes`` right
#: beside it -- and ``receipts.<name>.path`` is where a receipt landed.  A
#: capsule field is on this list only when the value is fully determined by
#: ``--outdir``; the config path under ``numerical_stack`` is an INPUT and
#: stays compared verbatim, because two runs of one configuration read the
#: same file and a difference there is a real finding.
OUTPUT_PATH_FIELDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^output\.frames\[\d+\]\.path$"),
    re.compile(r"^receipts\.[^.]+\.path$"),
)

#: Both separators, because a capsule written on Windows is read on Linux
#: and a leaf name split by the reader's own convention would depend on the
#: reader rather than on the document.
_PATH_SEPARATORS = re.compile(r"[\\/]")


def is_output_path_field(path: str) -> bool:
    """Whether ``path`` names a pure output-location field."""
    return any(pattern.fullmatch(path) for pattern in OUTPUT_PATH_FIELDS)


def normalize_output_path(value: Any) -> Any:
    """The leaf name of an output path; anything else passes through.

    Passing a non-string through unchanged is deliberate: a path field that
    arrived as ``null``, or absent, must stay distinguishable from one that
    arrived as a string, or a dropped field would normalize into agreement.
    """
    if not isinstance(value, str):
        return value
    return _PATH_SEPARATORS.split(value)[-1]


def normalized_leaves(document: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a capsule with its output-location fields normalized."""
    flat = flatten_capsule(document)
    for path in flat:
        if is_output_path_field(path):
            flat[path] = normalize_output_path(flat[path])
    return flat


#: Leaves that witness WHICH INPUT BYTES a run read.  A divergence here is
#: always a real finding -- it is what a corrupted or swapped input looks
#: like -- so nothing on this list is ever normalized or skipped.  It exists
#: only so the refusal can say what the finding usually MEANS, because the
#: commonest cause is a procedural one with a one-line remedy.
INPUT_BYTES_FIELDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^numerical_stack\.input_artifact_bytes\.value(\..+)?$"),
    re.compile(r"^input_bytes\..+$"),
)

#: Input artifacts a PREPARATION stage writes.  Their bytes legitimately
#: differ between two independent preparations of one configuration: a
#: preparation receipt records wall-clock timings and the decoder's own
#: stdout, which carries the staging directory it used.  That is not
#: corruption and it is not physics -- it is two different preparations,
#: and two arms of a dual run are supposed to share ONE.
_PREPARATION_ARTIFACTS = ("proof", "preparation_receipt", "artifact_receipt",
                          "artifact_manifest", "run_receipt", "report")


def input_bytes_divergence_note(divergences) -> str | None:
    """What an input-bytes divergence usually means, or ``None``.

    The comparison itself is unchanged and total: this only reads the
    divergences that were already found and adds the sentence a tester
    spent hours not being told.  A dual run whose two arms each ran their
    OWN preparation is comparing two preparations as well as two
    forecasts, and a preparation receipt can never repeat -- it records
    how long the preparation took and where it staged.  The screen's
    subject is the forecast; the preparation is an input to it, like the
    geography tree, and both arms must read the same one.
    """

    culprits = sorted({
        divergence.field_path for divergence in divergences
        if any(pattern.fullmatch(divergence.field_path)
               for pattern in INPUT_BYTES_FIELDS)})
    if not culprits:
        return None
    preparation = [path for path in culprits
                   if path.rsplit(".", 1)[-1] in _PREPARATION_ARTIFACTS]
    note = (
        "the two arms read DIFFERENT INPUT BYTES: "
        + ", ".join(culprits[:6])
        + (f" (+{len(culprits) - 6} more)" if len(culprits) > 6 else "")
        + ".  This is compared verbatim and always will be -- it is what a "
        "swapped or corrupted input looks like.")
    if preparation:
        note += (
            "  The commonest cause is procedural: "
            + ", ".join(sorted(preparation))
            + " names a PREPARATION receipt, and two independent "
            "preparations of one configuration never produce the same "
            "bytes -- a preparation receipt records its own wall-clock "
            "timings and the staging directory it used.  Prepare ONCE and "
            "run the forecast twice from that one prepared root, into two "
            "--outdir directories.  See docs/public/DETERMINISM.md "
            "section 6.")
    return note


def compare_capsules(left: Mapping[str, Any],
                     right: Mapping[str, Any]) -> DualRunComparison:
    """Compare two capsules leaf for leaf, in a deterministic path order.

    Raises :class:`DualRunError` when the two documents between them offer
    fewer than :data:`MINIMUM_COMPARED_LEAVES` leaves.  There is no
    verdict to give on an empty comparison: "no divergence was found" and
    "no divergence could have been found" are different answers, and this
    command reported the second as the first.
    """
    flat_left = normalized_leaves(left)
    flat_right = normalized_leaves(right)
    paths = sorted(set(flat_left) | set(flat_right))
    if len(paths) < MINIMUM_COMPARED_LEAVES:
        raise DualRunError(
            f"there is nothing to compare: capsule A carries "
            f"{len(flat_left)} leaf field(s) and capsule B carries "
            f"{len(flat_right)}, so this screen would compare "
            f"{len(paths)} quantities and report them all identical.  "
            "A dual run is the only detector this project has for the "
            "silent memory corruption a card with no ECC cannot report, "
            "so an empty screen is refused rather than passed.  Pass the "
            "certification-capsule.json each arm wrote (--outdir "
            "ARM/certification-capsule.json), not an empty or placeholder "
            "document")
    divergences = []
    for path in paths:
        a = flat_left.get(path, ABSENT)
        b = flat_right.get(path, ABSENT)
        if a != b or type(a) is not type(b):
            divergences.append(Divergence(path, a, b))
    return DualRunComparison(identical=not divergences,
                             divergences=tuple(divergences),
                             compared_count=len(paths))


def _read_capsule(path: str | Path, arm: str) -> Any:
    """One capsule document, or a refusal naming the file and the arm.

    A zero-byte file is the literal shape this command was reported
    green on, and ``json.loads`` answers it with a ``JSONDecodeError``
    whose message ("Expecting value: line 1 column 1") names neither the
    file nor which arm it was.  A screen that cannot say which of its two
    inputs is unreadable is a screen nobody can act on.
    """
    resolved = Path(path)
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise DualRunError(f"capsule {arm} ({resolved}) cannot be read: "
                           f"{error}") from error
    if not payload.strip():
        raise DualRunError(
            f"capsule {arm} ({resolved}) is empty ({len(payload)} bytes).  "
            "That is not a capsule, and comparing two of them would report "
            "'identical' having compared nothing.  Point --capsule-"
            f"{arm.lower()} at the certification-capsule.json that arm's run "
            "wrote")
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise DualRunError(
            f"capsule {arm} ({resolved}) is not JSON: {error}") from error


def compare_capsule_files(path_a: str | Path,
                          path_b: str | Path) -> DualRunComparison:
    """Read two capsules from disk and compare them."""
    return compare_capsules(_read_capsule(path_a, "A"),
                            _read_capsule(path_b, "B"))


def capsule_field_paths(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Every mutable leaf path of a capsule, in comparison order.

    This is what the mutation table iterates: one row per capsule field, so a
    field added to the schema later shows up here without anyone editing a
    list.
    """
    return tuple(sorted(flatten_capsule(document)))


__all__ = [
    "ABSENT",
    "INPUT_BYTES_FIELDS",
    "MINIMUM_COMPARED_LEAVES",
    "OUTPUT_PATH_FIELDS",
    "input_bytes_divergence_note",
    "Divergence",
    "DualRunComparison",
    "DualRunError",
    "capsule_field_paths",
    "compare_capsule_files",
    "compare_capsules",
    "decode_key",
    "delete_leaf",
    "encode_key",
    "flatten_capsule",
    "is_output_path_field",
    "leaf_value",
    "normalize_output_path",
    "normalized_leaves",
    "set_leaf",
    "split_path",
]
