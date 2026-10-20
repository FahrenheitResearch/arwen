"""Verify unmanaged local objects against their declared public endpoints.

No source names or URL templates belong here. The route supplies the object,
its index and its endpoint ladder. A failed check leaves the file untouched
for the ordinary fetch to quarantine and replace.
"""

from __future__ import annotations

import hashlib
from http.client import HTTPException
import json
import re
import threading
from urllib.parse import urlsplit
from urllib.request import Request

from gpuwm import fetch_endpoints
from gpuwm.nomads_governor import paced_urlopen

# Shared by concurrent plans in this process. Hashing holds no network slot.
_CONNECTIONS = threading.BoundedSemaphore(16)
_CHUNK = 1 << 20
_MAX_INDEX_BYTES = 16 << 20


def _request(url, *, method="GET", opener=None):
    return paced_urlopen(
        Request(url, method=method, headers={"User-Agent": "gpuwm-fetch-adoption",
                                            "Accept-Encoding": "identity"}),
        timeout=30, **({"opener": opener} if opener is not None else {}))


def _identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _index_offsets(text):
    """NCEP .idx message starts, including submessages at the same offset."""
    offsets = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(":", 2)
        if len(fields) < 3 or not re.fullmatch(r"\d+(?:\.\d+)?", fields[0]):
            raise ValueError("index does not declare message numbers and byte offsets")
        offset = int(fields[1])
        if offset < 0 or (offsets and offset < offsets[-1]):
            raise ValueError("index byte offsets are not ordered")
        if not offsets or offset != offsets[-1]:
            offsets.append(offset)
    if not offsets or offsets[0] != 0:
        raise ValueError("index does not begin at byte zero")
    return offsets


def _grib_offsets(path, size):
    """Validate GRIB envelopes and return each physical message start."""
    offsets = []
    offset = 0
    with path.open("rb") as stream:
        while offset < size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB":
                raise ValueError(f"missing GRIB indicator at byte {offset}")
            if header[7] == 2:
                length = int.from_bytes(header[8:16], "big")
                minimum = 20
            elif header[7] == 1:
                length = int.from_bytes(header[4:7], "big")
                minimum = 12
            else:
                raise ValueError(f"unsupported GRIB edition at byte {offset}")
            if length < minimum or offset + length > size:
                raise ValueError(f"invalid GRIB length at byte {offset}")
            stream.seek(offset + length - 4)
            if stream.read(4) != b"7777":
                raise ValueError(f"missing GRIB terminator at byte {offset}")
            offsets.append(offset)
            offset += length
    return offsets


def adopt(path, obj, *, ladder, magic, verify_payload, opener=None, progress=print):
    """Return a verified entry, or None to take the existing download path.

    A single-part S3 ETag verifies every byte by MD5. Multipart and opaque
    ETags require the host's index plus its length: the complete local GRIB
    envelope walk must match the index's message count and byte offsets.
    The index is fetched from the host, never trusted from a local sidecar.
    """
    before = _identity(path)
    try:
        verify_payload(path, magic=magic, label=path.name)
    except ValueError as error:
        progress(f"fetch adoption {obj.relpath}: rejected: {error}")
        return None
    digests = None
    for endpoint in fetch_endpoints.transfer_order(ladder):
        url = endpoint.url(obj.key) if obj.key else obj.url
        try:
            with _CONNECTIONS, _request(url, method="HEAD", opener=opener) as response:
                size = int(response.headers.get("Content-Length", "-1"))
                etag = response.headers.get("ETag", "").strip('"')
            if size != before[2]:
                raise ValueError(f"host size {size} differs from local size {before[2]}")
            if not etag:
                raise ValueError("host HEAD has no ETag")
            host = urlsplit(url).hostname or ""
            s3 = host.endswith((".amazonaws.com", ".amazonaws.com.cn"))
            md5_etag = s3 and re.fullmatch(r"[0-9a-fA-F]{32}", etag)
            evidence = {"content_length": size, "etag": etag}
            if not md5_etag:
                if magic != "GRIB" or not obj.index_key:
                    progress(f"fetch adoption {obj.relpath}: {endpoint.name} has no "
                             "MD5 ETag or usable GRIB index")
                    continue
                index_url = endpoint.url(obj.index_key)
                with _CONNECTIONS, _request(index_url, opener=opener) as response:
                    raw = response.read(_MAX_INDEX_BYTES + 1)
                if len(raw) > _MAX_INDEX_BYTES:
                    raise ValueError("host index exceeds 16 MiB")
                expected = _index_offsets(raw.decode("utf-8"))
                observed = _grib_offsets(path, size)
                if observed != expected:
                    raise ValueError("GRIB message count or byte offsets differ from host index")
                evidence.update(method="size-index", index_url=index_url,
                                index_sha256=hashlib.sha256(raw).hexdigest(),
                                messages=len(observed), last_byte_offset=observed[-1])
            if digests is None:
                sha256 = hashlib.sha256()
                md5 = hashlib.md5(usedforsecurity=False)
                with path.open("rb") as stream:
                    while chunk := stream.read(_CHUNK):
                        sha256.update(chunk)
                        md5.update(chunk)
                digests = sha256.hexdigest(), md5.hexdigest()
            if md5_etag:
                if digests[1] != etag.lower():
                    raise ValueError(f"local MD5 {digests[1]} differs from host ETag {etag}")
                evidence.update(method="size-md5", md5=digests[1])
            if _identity(path) != before:
                raise ValueError("local file changed during verification")
            return {"name": path.name, "bytes": size, "sha256": digests[0],
                    "url": url, "endpoint": endpoint.name, "adoption": evidence}
        except ValueError as error:
            progress(f"fetch adoption {obj.relpath}: {endpoint.name} rejected: {error}")
            return None
        except (OSError, HTTPException) as error:
            progress(f"fetch adoption {obj.relpath}: {endpoint.name} rejected: {error}")
    return None


def route_throughput(out):
    """Relay table-route acquisition counts without changing input authority."""
    from gpuwm.fetch_routes import MANIFEST_NAME, ROUTE_MANIFEST_SCHEMA
    try:
        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema") != ROUTE_MANIFEST_SCHEMA:
            return None
        receipt = json.loads((out / "fetch-pool-receipt.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        total = sum(entry["bytes"] for entry in files)
        downloaded = receipt.get("downloaded_files", 0)
        downloaded_bytes = receipt.get("downloaded_bytes", 0)
        return {"files": len(files), "bytes": total, "seconds": None,
                "files_timed": 0, "downloaded_files": downloaded,
                "downloaded_bytes": downloaded_bytes, "downloaded_seconds": None,
                "bytes_per_second": None,
                "adopted_files": receipt.get("adopted_files", 0),
                "adopted_bytes": receipt.get("adopted_bytes", 0),
                "reused_files": receipt.get("reused_files", 0),
                "reused_bytes": receipt.get("reused_bytes", 0),
                "verified_files": len(files) - downloaded,
                "verified_bytes": total - downloaded_bytes, "concurrency": receipt}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
