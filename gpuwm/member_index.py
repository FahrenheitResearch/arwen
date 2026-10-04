"""Copy one member's indexed GRIB ranges without decoding or changing them.

Field decoding and ensemble-octet verification remain in the Rust inventory.
The index is an acquisition hint only, never proof of member identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.request import Request

from gpuwm import fetch_pool
from gpuwm.nomads_governor import paced_urlopen


def select_member_ranges(index: bytes, declaration: Mapping[str, object],
                         member_ordinal: int) -> tuple[tuple[int, int], ...]:
    """Return nonoverlapping (offset, length) records in publication order."""
    required = {"member_key", "offset_key", "length_key"}
    if set(declaration) - (required | {"filters"}) or required - set(declaration):
        raise ValueError("Member index needs member_key, offset_key and length_key only, with optional filters")
    member_key, offset_key, length_key = (
        str(declaration[name]) for name in ("member_key", "offset_key", "length_key"))
    filters = declaration.get("filters", {})
    if not isinstance(filters, Mapping):
        raise ValueError("Member index filters must be a field-to-value mapping")
    entries = []
    for line, text in enumerate(index.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            offset, length = row[offset_key], row[length_key]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Member index line {line} is not a complete range record") from error
        if (type(offset) is not int or type(length) is not int
                or offset < 0 or length < 20):
            raise ValueError(f"Member index line {line} has invalid offset or GRIB length")
        selected = (str(row.get(member_key, "")) == str(member_ordinal)
                    and all(str(row.get(key, "")) == str(value)
                            for key, value in filters.items()))
        entries.append((offset, length, selected))
    entries.sort()
    previous_end = 0
    for offset, length, _ in entries:
        if offset < previous_end:
            raise ValueError("Member index ranges overlap, which could duplicate fields or mix members")
        previous_end = offset + length
    selected = tuple((offset, length) for offset, length, keep in entries if keep)
    if not selected:
        raise ValueError(f"Member index contains no records for member {member_ordinal}")
    return selected


def _joined_ranges(ranges):
    joined = []
    for offset, length in ranges:
        if joined and joined[-1][0] + joined[-1][1] == offset:
            start, size = joined[-1]
            joined[-1] = (start, size + length)
        else:
            joined.append((offset, length))
    return joined


def download_indexed_member(url: str, index_url: str, dest: Path, *,
                            declaration: Mapping[str, object], member_ordinal: int,
                            source: str, member: str, opener=None,
                            timeout: float = 300.0) -> dict:
    """Range-fetch, verify every GRIB message, then publish one member file.

Servers must return the requested range exactly. A 200 whole-ensemble reply
is refused before its body is read, preventing a multi-gigabyte accidental
download and mixed-member preparation. Object ETags pin all range requests
to one version while an as-posted source may still be publishing.
"""
    from gpuwm.member_grammar import load_member_grammar
    from gpuwm.member_prep import verify_member_file
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_authorities import packaged_member_grammar
    from gpuwm.filesystem_paths import io_path

    options = {"opener": opener} if opener is not None else {}
    with paced_urlopen(Request(index_url), timeout=timeout, **options) as response:
        index = response.read(16 * 1024 * 1024 + 1)
    if len(index) > 16 * 1024 * 1024:
        raise ValueError("Member index exceeds 16 MiB; refusing an unbounded response")
    ranges = select_member_ranges(index, declaration, member_ordinal)
    dest = io_path(Path(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    etag = None
    total_size = None
    written = 0
    try:
        with part.open("wb") as target:
            for offset, length in _joined_ranges(ranges):
                fetch_pool.raise_if_stopped()
                end = offset + length - 1
                headers = {"Range": f"bytes={offset}-{end}"}
                if etag is not None:
                    headers["If-Match"] = etag
                with paced_urlopen(Request(url, headers=headers),
                                   timeout=timeout, **options) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if (response.status != 206 or match is None
                            or tuple(map(int, match.groups()[:2])) != (offset, end)):
                        raise ValueError("Member range reply did not match the exact requested bytes")
                    object_size = int(match.group(3))
                    if object_size <= end:
                        raise ValueError("Member Content-Range extends beyond the declared object size")
                    received_etag = response.headers.get("ETag")
                    if (received_etag is None
                            or re.fullmatch(r'"[\x21\x23-\x7e\x80-\xff]*"', received_etag) is None):
                        raise ValueError("Member range reply needs a quoted strong ETag to pin exact bytes")
                    if etag is not None and (received_etag != etag or total_size != object_size):
                        raise ValueError("Member object changed between index range requests")
                    etag, total_size = received_etag, object_size
                    remaining = length
                    while remaining:
                        fetch_pool.raise_if_stopped()
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Member range reply ended before its declared length")
                        target.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        remaining -= len(chunk)
                    if response.read(1):
                        raise ValueError("Member range reply exceeded its declared length")
        grammar_id = get_source_adapter(source).member_set
        grammar = load_member_grammar(packaged_member_grammar(grammar_id))
        evidence = verify_member_file(grammar, member, part)
        if evidence.messages != len(ranges):
            raise ValueError("Selected index record count differs from the verified GRIB message count")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {
        "name": dest.name, "bytes": written, "sha256": digest.hexdigest(),
        "url": url, "member": member, "member_ordinal": member_ordinal,
        "index_url": index_url, "index_sha256": hashlib.sha256(index).hexdigest(),
        "selected_records": len(ranges), "object_etag": etag,
        "object_bytes": total_size, "member_verification": evidence.to_dict(),
    }
