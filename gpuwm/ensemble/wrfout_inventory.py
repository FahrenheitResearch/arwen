"""The frame inventory that binds a member record to its wrfout bytes.

A member record used to say ``wrfout_count: 1``, and that is the whole of
what any consumer could check.  ``gpuwm enprod --accept-status FAILED``
did check it -- and a stale wrfout from an unrelated earlier attempt,
sitting under the member directory in place of the one the writer
produced, satisfies it exactly.  One file was declared; one file was
found; the product rendered, the graphic was stamped as an override, and
the bytes in the mean were not the bytes the run wrote.

A count is not an identity.  This module defines the one that is:

    path      -- canonical POSIX-spelled path RELATIVE to the member
                 directory, so the inventory survives the ensemble being
                 moved and cannot name anything outside the member;
    domain    -- the ``dNN`` token the filename carries, so a d02 file
                 cannot answer for a d03 one;
    frames    -- every valid time in the file WITH its frame index, in
                 file order, because "which frame is 18:00Z" is the
                 question the product suite actually asks;
    size      -- the byte length, a free first-order check;
    sha256    -- the content hash, which is the part a stale replacement
                 cannot forge.

The contract is versioned (:data:`WRFOUT_INVENTORY_CONTRACT`) and both
halves import it from here: :mod:`gpuwm.ensemble.engine` records it when a
member finishes, :func:`gpuwm.da.enprod.verify_override_inventory` binds
against it before a widened roster draws anything.  Two spellings of one
contract is how the count check drifted into meaninglessness in the first
place.

**Cost, stated rather than hidden.**  Hashing means reading every wrfout
back once after the run wrote it.  At a few GB per member that is seconds,
not minutes, and it buys the only check that distinguishes the file the
run produced from a file of the same size in the same place.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

#: Versioned label recorded in every inventory entry.
WRFOUT_INVENTORY_CONTRACT = "gpuwm-ensemble-wrfout-inventory.v1"

#: Exactly the keys one v1 entry carries -- no more, no fewer.
#:
#: Exact rather than "at least these": an entry carrying a key this build
#: does not know is an entry written against a contract this build is not
#: checking, and the whole point of the version label is that a reader
#: which cannot check a record refuses it instead of binding the half it
#: recognises.  Whoever adds a field bumps
#: :data:`WRFOUT_INVENTORY_CONTRACT` with it.
INVENTORY_ENTRY_KEYS = ("contract", "path", "domain", "frames",
                        "size_bytes", "sha256")

#: What ``hashlib.sha256().hexdigest()`` produces, and nothing else.
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")

#: The ``dNN`` token a wrfout filename carries.
_DOMAIN_PATTERN = re.compile(r"\Ad[0-9]{2}\Z")

#: Manifest key the inventory is recorded under, per member.
WRFOUT_INVENTORY_KEY = "wrfout_inventory"

#: Read size for the content hash.  1 MiB: large enough that the syscall
#: count is irrelevant beside the disk, small enough that a 4 GB history
#: file is never resident.
_HASH_CHUNK_BYTES = 1 << 20


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in chunks.

    Streamed rather than ``read()``: a wrfout is routinely larger than
    the memory a render host has spare, and a provenance helper that can
    exhaust it is a provenance helper nobody leaves enabled.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_relative_path(path: str | Path,
                            member_dir: str | Path) -> str:
    """``path`` relative to ``member_dir``, POSIX-spelled.

    Relative because an ensemble that is copied or archived keeps its
    member layout and loses its absolute prefix; POSIX-spelled because the
    manifest is JSON read on both platforms and ``member_000\\wrfout_d02``
    and ``member_000/wrfout_d02`` must not be two inventories.

    A path outside the member directory is a refusal, not a ``..`` entry:
    the inventory's whole claim is that these are THIS member's files.
    """

    target = Path(path).resolve()
    root = Path(member_dir).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{target} is not inside the member directory {root}; a frame "
            "inventory names the member's own files and nothing else"
        ) from exc
    return relative.as_posix()


def domain_token(path: str | Path) -> str | None:
    """The ``dNN`` token in a wrfout filename, or ``None``.

    Deliberately the filename token and not a global attribute: the
    inventory is checked against what a consumer resolved from the same
    filename, and reading GRID_ID here would compare two different facts.
    """

    for part in Path(path).name.split("_"):
        if (len(part) == 3 and part[0] == "d" and part[1:].isdigit()):
            return part
    return None


def wrfout_valid_times(path: str | Path) -> tuple[str, ...]:
    """Every ``Times`` record in a wrfout, in file order.

    netCDF4 is imported here rather than at module scope so a consumer
    that only VERIFIES an inventory -- which reads its frames through the
    product suite's own reader -- never needs the writer's dependency.
    """

    import netCDF4  # noqa: PLC0415

    with netCDF4.Dataset(str(path), "r") as dataset:
        if "Times" not in dataset.variables:
            raise ValueError(
                f"{path} carries no Times variable; it is not a wrfout and "
                "cannot be inventoried as one")
        raw = dataset.variables["Times"][:]
        return tuple(
            bytes(bytearray(row.tobytes())).decode("ascii").rstrip("\x00 ")
            for row in raw)


def inventory_entry(path: str | Path, *, member_dir: str | Path) -> dict:
    """One wrfout's inventory record."""

    target = Path(path)
    stamps = wrfout_valid_times(target)
    return {
        "contract": WRFOUT_INVENTORY_CONTRACT,
        "path": canonical_relative_path(target, member_dir),
        "domain": domain_token(target),
        "frames": [{"index": index, "valid_time": stamp}
                   for index, stamp in enumerate(stamps)],
        "size_bytes": int(target.stat().st_size),
        "sha256": file_sha256(target),
    }


def member_inventory(paths, *, member_dir: str | Path) -> tuple[dict, ...]:
    """The inventory for every wrfout one member wrote, path-ordered.

    Ordered by the canonical relative path so two runs that produced the
    same files record the same list, whatever order the integrator
    happened to hand them over in.
    """

    entries = [inventory_entry(path, member_dir=member_dir)
               for path in paths]
    return tuple(sorted(entries, key=lambda entry: entry["path"]))


def entry_frames(entry) -> dict:
    """``{valid_time: frame_index}`` from one inventory entry.

    Fails closed on a duplicated valid time: two frames claiming one time
    is the ambiguity the product suite refuses on the disk side, and an
    inventory that declares it would license the thing being refused.
    """

    frames: dict[str, int] = {}
    for record in entry.get("frames") or ():
        stamp = record.get("valid_time")
        index = record.get("index")
        if not isinstance(stamp, str) or not isinstance(index, int) \
                or isinstance(index, bool):
            raise ValueError(
                f"inventory entry {entry.get('path')!r} carries a frame "
                f"record that is not {{index: int, valid_time: str}}: "
                f"{record!r}")
        if stamp in frames:
            raise ValueError(
                f"inventory entry {entry.get('path')!r} declares valid time "
                f"{stamp} twice; an inventory cannot be ambiguous about "
                "which frame is the forecast")
        frames[stamp] = index
    return frames


def read_inventory(record) -> tuple[dict, ...] | None:
    """A member record's declared inventory, or ``None`` if it has none.

    ``None`` is a real state -- a member the writer never finished records
    no inventory -- and it is the caller's job to decide what to do about
    it.  What this function will not do is return an empty tuple for it,
    because "declared nothing" and "declared no files" are different
    claims and only one of them is checkable.
    """

    declared = record.get(WRFOUT_INVENTORY_KEY)
    if declared is None:
        return None
    if not isinstance(declared, list):
        raise ValueError(
            f"{WRFOUT_INVENTORY_KEY} must be a list of inventory entries, "
            f"got {type(declared).__name__}")
    entries = []
    for entry in declared:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{WRFOUT_INVENTORY_KEY} entries must be objects, got "
                f"{type(entry).__name__}")
        for key in ("path", "sha256", "size_bytes"):
            if key not in entry:
                raise ValueError(
                    f"{WRFOUT_INVENTORY_KEY} entry {entry!r} has no "
                    f"{key!r}; an inventory without it binds nothing")
        entries.append(dict(entry))
    return tuple(entries)


def _is_int(value) -> bool:
    """A real integer.  ``True`` is an ``int`` and is not one of these."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_entry_schema(entry) -> list[str]:
    """Problems with the SHAPE of one inventory entry, before any I/O.

    Separate from :func:`verify_entry` because they answer different
    questions: this one asks whether the record is a v1 inventory entry
    at all, and that one asks whether the bytes on disk are the bytes it
    names.  Re-verification found the second was doing all the work while
    the first was not being asked -- ``read_inventory`` required only the
    PRESENCE of ``path``, ``sha256`` and ``size_bytes``, so an entry with
    no ``contract`` and ``domain="d99"`` verified, and
    ``size_bytes="same as whatever is on disk"`` verified *by disabling
    the size check*, because that check ran only when the declared size
    happened to be an integer.  A verifier that stamps
    :data:`WRFOUT_INVENTORY_CONTRACT` on its report has to have checked
    the contract.

    Every value is checked against what it is FOR:

    * ``contract`` is this exact version -- a record written to another
      one is not a record this code knows how to bind;
    * ``path`` is a canonical, POSIX-spelled, member-relative path, in
      the one spelling :func:`canonical_relative_path` produces, so two
      spellings of one file cannot be two entries;
    * ``domain`` agrees with the ``dNN`` token in ``path``, since the
      whole use of the field is that a d02 file cannot answer for a d03
      one -- a ``domain`` disagreeing with its own filename answers for
      neither;
    * ``frames`` is a list of ``{index, valid_time}`` with non-negative,
      unique indices and unique valid times;
    * ``size_bytes`` is a non-negative integer, always, so the size check
      cannot be switched off by declaring a string;
    * ``sha256`` is 64 lowercase hex digits, so a placeholder cannot sit
      in the field the whole binding rests on.
    """

    if not isinstance(entry, dict):
        return [f"inventory entry must be an object, got "
                f"{type(entry).__name__}"]
    problems: list[str] = []
    keys = set(entry)
    expected = set(INVENTORY_ENTRY_KEYS)
    missing = sorted(expected - keys)
    surplus = sorted(keys - expected)
    label = entry.get("path") if isinstance(entry.get("path"), str) \
        else repr(entry)
    if missing or surplus:
        problems.append(
            f"inventory entry {label!r} is not a "
            f"{WRFOUT_INVENTORY_CONTRACT} record: "
            + "; ".join(part for part in (
                f"missing {missing}" if missing else "",
                f"unknown {surplus}" if surplus else "") if part)
            + f". The contract is exactly {list(INVENTORY_ENTRY_KEYS)}")

    contract = entry.get("contract")
    if contract != WRFOUT_INVENTORY_CONTRACT:
        problems.append(
            f"inventory entry {label!r} declares contract {contract!r}; "
            f"this build binds {WRFOUT_INVENTORY_CONTRACT!r} and will not "
            "check a record against a contract it does not know")

    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        problems.append(
            f"inventory entry {entry!r} declares no path")
    elif relative != _canonical_spelling(relative):
        problems.append(
            f"inventory path {relative!r} is not the canonical "
            "member-relative POSIX spelling the writer records; two "
            "spellings of one file are two entries to every check that "
            "keys on the path")

    domain = entry.get("domain")
    if domain is not None and (not isinstance(domain, str)
                               or not _DOMAIN_PATTERN.match(domain)):
        problems.append(
            f"inventory entry {label!r} declares domain {domain!r}, which "
            "is not a dNN token")
    elif isinstance(relative, str) and relative:
        token = domain_token(relative)
        if domain != token:
            problems.append(
                f"inventory entry {label!r} declares domain {domain!r} for "
                f"a file whose own name says {token!r}; a domain that "
                "disagrees with its filename answers for neither domain")

    frames = entry.get("frames")
    if not isinstance(frames, list):
        problems.append(
            f"inventory entry {label!r} declares frames "
            f"{type(frames).__name__}; it must be a list of "
            "{index, valid_time} records")
    else:
        indices: set[int] = set()
        for record in frames:
            if not isinstance(record, dict) \
                    or set(record) != {"index", "valid_time"}:
                problems.append(
                    f"inventory entry {label!r} carries a frame record "
                    f"that is not exactly {{index, valid_time}}: {record!r}")
                continue
            index = record["index"]
            stamp = record["valid_time"]
            if not _is_int(index) or index < 0:
                problems.append(
                    f"inventory entry {label!r} declares frame index "
                    f"{index!r}; a frame index is a non-negative integer "
                    "position in the file")
            elif index in indices:
                problems.append(
                    f"inventory entry {label!r} declares frame index "
                    f"{index} twice; one position in a file is one frame")
            else:
                indices.add(index)
            if not isinstance(stamp, str) or not stamp:
                problems.append(
                    f"inventory entry {label!r} declares valid time "
                    f"{stamp!r}; it is the identity the product suite asks "
                    "the inventory about")
        try:
            entry_frames(entry)
        except ValueError as exc:
            problems.append(str(exc))

    size = entry.get("size_bytes")
    if not _is_int(size) or size < 0:
        problems.append(
            f"inventory entry {label!r} declares size_bytes {size!r}; it "
            "must be a non-negative integer. A non-integer size used to "
            "SKIP the size check rather than fail it, which turned a "
            "malformed record into a weaker one")

    digest = entry.get("sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.match(digest):
        problems.append(
            f"inventory entry {label!r} declares sha256 {digest!r}; it "
            "must be 64 lowercase hex digits. The digest is the only "
            "thing a stale replacement cannot forge, so a placeholder "
            "there is an inventory that binds nothing")
    return problems


def _canonical_spelling(relative: str) -> str | None:
    """``relative`` as :func:`canonical_relative_path` would spell it.

    ``None`` when there is no such spelling -- an absolute path, a
    leading separator, or a ``..``/``.`` segment, none of which name a
    file inside the member directory.
    """
    if relative[0] in "/\\":
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        return None
    parts = candidate.parts
    if not parts or any(part in ("..", ".") for part in parts):
        return None
    return "/".join(parts)


def duplicate_paths(entries) -> list[str]:
    """Paths declared more than once across one member's inventory.

    Every consumer keys on the path -- enprod builds ``{path: record}``
    and a dict silently keeps the last one -- so two records for one file
    are two different claims about the same bytes with only one of them
    ever checked.  Which one is checked depends on list order, and that
    is not a property to leave to chance.
    """
    seen: dict[str, int] = {}
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        if isinstance(path, str):
            seen[path] = seen.get(path, 0) + 1
    return sorted(path for path, count in seen.items() if count > 1)


def verify_entry(entry, *, member_dir: str | Path) -> list[str]:
    """Problems binding one inventory entry to the bytes on disk.

    Returns a list of human-readable problems; empty means the file at
    the declared path is byte-for-byte the file the writer declared, AND
    the record itself is a well-formed v1 entry
    (:func:`validate_entry_schema`).  Both, because a verifier that
    reported the bytes matched a record it never checked the shape of is
    what re-verification found.
    """

    problems: list[str] = list(validate_entry_schema(entry))
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        return problems or [f"inventory entry {entry!r} declares no path"]
    # ``WindowsPath('/etc/passwd').is_absolute()`` is False -- there is no
    # drive -- but joining it onto the member directory still escapes it,
    # so the leading separator is tested directly rather than through
    # is_absolute() alone.
    if relative[0] in "/\\" or Path(relative).is_absolute() \
            or ".." in Path(relative).parts:
        return problems + [
            f"inventory path {relative!r} is not a relative path "
            "inside the member directory"]
    target = Path(member_dir) / relative
    if not target.is_file():
        return problems + [
            f"{relative} is declared in the manifest inventory and is "
            "not present"]
    size = int(target.stat().st_size)
    declared_size = entry.get("size_bytes")
    if _is_int(declared_size) and size != declared_size:
        problems.append(
            f"{relative} is {size} bytes; the manifest recorded "
            f"{declared_size}")
    digest = file_sha256(target)
    if digest != entry.get("sha256"):
        problems.append(
            f"{relative} hashes to {digest}; the manifest recorded "
            f"{entry.get('sha256')}. These are not the same bytes")
    return problems
