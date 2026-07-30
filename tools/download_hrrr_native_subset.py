"""Download only the HRRR GRIB2 messages required by native initialization.

NOAA's ``.idx`` files are used solely as byte-range transport indexes.  The
native Rust bridge remains the scientific authority: it reopens every
downloaded GRIB message and validates numeric discipline/category/parameter,
surface coordinates, cycle, forecast hour, grid, and required cardinality.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours


BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HYBRID_FIELDS = (
    "PRES", "CLMR", "CIMIXR", "RWMR", "SNMR", "GRLE",
    "HGT", "TMP", "SPFH", "UGRD", "VGRD",
)
SURFACE_FIELDS = (
    ("PRES", "surface"),
    ("HGT", "surface"),
    ("TMP", "surface"),
    ("WEASD", "surface"),
    ("SNOD", "surface"),
    ("TMP", "2 m above ground"),
    ("SPFH", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("LAND", "surface"),
    ("ICEC", "surface"),
)
SOIL_DEPTHS = ("0", "0.01", "0.04", "0.1", "0.3", "0.6", "1", "1.6", "3")
HYBRID_LEVEL = re.compile(r"([1-9]|[1-4][0-9]|50) hybrid level")

#: One published wrfnat atmosphere subset carries exactly the 11 hybrid
#: fields on all 50 levels plus the 11 surface/near-surface records (561).
#: Single source for the completeness bar: the index selection below and
#: ``gpuwm.fetch``'s existing-file re-verification both use these counts.
ATMOSPHERE_RECORD_COUNT = 50 * len(HYBRID_FIELDS) + len(SURFACE_FIELDS)
#: One published soil subset carries TSOIL + SOILW at the nine RUC level
#: depths of the wrfprs file (18).
SOIL_RECORD_COUNT = 2 * len(SOIL_DEPTHS)


@dataclass(frozen=True)
class IndexRow:
    sequence: int
    offset: int
    variable: str
    level: str
    raw: str


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    first_row: str
    last_row: str

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ProductRequest:
    url: str
    index_url: str
    index_path: Path
    destination: Path
    kind: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycle(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    if value.minute or value.second:
        raise ValueError("HRRR cycle must be an exact UTC hour")
    return value


def _hours(raw: str, *, cycle: datetime | None = None) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("forecast hours must be comma-separated integers") from error
    return validate_hrrr_source_forecast_hours(values, cycle=cycle)


def _request_bytes(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "rw-wps-hrrr-range/1"})
    with urlopen(request, timeout=120) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def _head(url: str) -> dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "rw-wps-hrrr-range/1"})
    with urlopen(request, timeout=120) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
    if "content-length" not in headers or int(headers["content-length"]) <= 0:
        raise ValueError(f"HRRR source omitted a positive Content-Length: {url}")
    if headers.get("accept-ranges", "").lower() != "bytes":
        raise ValueError(f"HRRR source does not advertise byte ranges: {url}")
    return headers


def _publish_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing authority differs from NOAA response: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parse_index(payload: bytes, object_bytes: int) -> tuple[IndexRow, ...]:
    rows = []
    for line_number, raw in enumerate(payload.decode("ascii").splitlines(), 1):
        fields = raw.split(":", 5)
        if len(fields) != 6:
            raise ValueError(f"malformed NOAA index line {line_number}")
        sequence, offset = int(fields[0]), int(fields[1])
        if sequence != line_number or offset < 0:
            raise ValueError(f"non-canonical NOAA index line {line_number}")
        rows.append(IndexRow(sequence, offset, fields[3], fields[4], raw))
    if not rows or rows[0].offset != 0:
        raise ValueError("NOAA index must begin with record 1 at byte zero")
    if any(right.offset <= left.offset for left, right in zip(rows, rows[1:])):
        raise ValueError("NOAA index offsets are not strictly increasing")
    if rows[-1].offset >= object_bytes:
        raise ValueError("NOAA index exceeds its source object")
    return tuple(rows)


def _atmosphere_selection(rows: tuple[IndexRow, ...]) -> tuple[int, ...]:
    selected = []
    levels = {name: set() for name in HYBRID_FIELDS}
    surfaces = {key: 0 for key in SURFACE_FIELDS}
    for index, row in enumerate(rows):
        match = HYBRID_LEVEL.fullmatch(row.level)
        if row.variable in levels and match is not None:
            level = int(match.group(1))
            levels[row.variable].add(level)
            selected.append(index)
            continue
        key = (row.variable, row.level)
        if key in surfaces and "acc fcst" not in row.raw:
            surfaces[key] += 1
            selected.append(index)
    expected_levels = set(range(1, 51))
    bad_levels = {name: sorted(values) for name, values in levels.items()
                  if values != expected_levels}
    bad_surfaces = {str(key): count for key, count in surfaces.items() if count != 1}
    if bad_levels or bad_surfaces or len(selected) != ATMOSPHERE_RECORD_COUNT:
        raise ValueError(
            "HRRR atmosphere index lacks the exact native bridge inventory: "
            f"levels={bad_levels}, surfaces={bad_surfaces}, count={len(selected)}")
    return tuple(selected)


def _soil_selection(rows: tuple[IndexRow, ...]) -> tuple[int, ...]:
    expected = {
        (variable, f"{depth}-{depth} m below ground")
        for variable in ("TSOIL", "SOILW") for depth in SOIL_DEPTHS
    }
    counts = {key: 0 for key in expected}
    selected = []
    for index, row in enumerate(rows):
        key = (row.variable, row.level)
        if key in counts:
            counts[key] += 1
            selected.append(index)
    bad = {str(key): count for key, count in counts.items() if count != 1}
    if bad or len(selected) != SOIL_RECORD_COUNT:
        raise ValueError(
            "HRRR pressure-file index lacks the exact soil inventory: "
            f"records={bad}, count={len(selected)}")
    return tuple(selected)


def _coalesce(
    rows: tuple[IndexRow, ...], selected: tuple[int, ...], object_bytes: int,
) -> tuple[ByteRange, ...]:
    groups: list[tuple[int, int]] = []
    first = previous = selected[0]
    for index in selected[1:]:
        if index != previous + 1:
            groups.append((first, previous))
            first = index
        previous = index
    groups.append((first, previous))
    result = []
    for first, last in groups:
        end = rows[last + 1].offset - 1 if last + 1 < len(rows) else object_bytes - 1
        result.append(ByteRange(
            rows[first].offset, end, rows[first].raw, rows[last].raw))
    return tuple(result)


def _download_range(
    url: str, byte_range: ByteRange, path: Path, retries: int,
) -> None:
    expected_content_range = f"bytes {byte_range.start}-{byte_range.end}/"
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={
                "User-Agent": "rw-wps-hrrr-range/1",
                "Range": f"bytes={byte_range.start}-{byte_range.end}",
                "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=300) as response, path.open("wb") as output:
                if response.status != 206:
                    raise ValueError(f"range request returned HTTP {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_content_range):
                    raise ValueError(f"unexpected Content-Range {content_range!r}")
                copied = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    copied += len(block)
            if copied != byte_range.size:
                raise ValueError(f"short range response: {copied} != {byte_range.size}")
            return
        except Exception:
            path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(attempt * 2)


def _download_subset(
    *, url: str, index_url: str, index_path: Path, destination: Path,
    kind: str, workers: int, retries: int,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing subset: {destination}")
    started = time.perf_counter()
    headers = _head(url)
    object_bytes = int(headers["content-length"])
    index_payload, index_headers = _request_bytes(index_url)
    _publish_exact(index_path, index_payload)
    rows = _parse_index(index_payload, object_bytes)
    selected = (
        _atmosphere_selection(rows) if kind == "atmosphere"
        else _soil_selection(rows)
    )
    ranges = _coalesce(rows, selected, object_bytes)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.ranges-", dir=destination.parent))
    try:
        chunks = tuple(staging / f"{index:04d}.part" for index in range(len(ranges)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(
                lambda pair: _download_range(url, pair[0], pair[1], retries),
                zip(ranges, chunks),
            ))
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("xb") as output:
            for chunk in chunks:
                with chunk.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        expected_bytes = sum(item.size for item in ranges)
        if partial.stat().st_size != expected_bytes:
            raise ValueError("assembled HRRR subset byte count differs from ranges")
        with partial.open("rb") as stream:
            signature = stream.read(4)
            stream.seek(-4, os.SEEK_END)
            terminator = stream.read(4)
        if signature != b"GRIB" or terminator != b"7777":
            raise ValueError("assembled HRRR subset is not a complete GRIB2 stream")
        os.replace(partial, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "kind": kind,
        "source_url": url,
        "source_bytes": object_bytes,
        "source_etag": headers.get("etag"),
        "source_last_modified": headers.get("last-modified"),
        "index_url": index_url,
        "index_path": index_path.name,
        "index_bytes": len(index_payload),
        "index_sha256": hashlib.sha256(index_payload).hexdigest(),
        "index_etag": index_headers.get("etag"),
        "selected_record_count": len(selected),
        "range_count": len(ranges),
        "subset_path": destination.name,
        "subset_bytes": destination.stat().st_size,
        "subset_sha256": _sha256(destination),
        "wall_seconds": time.perf_counter() - started,
        "ranges": [
            {
                "start": item.start,
                "end": item.end,
                "bytes": item.size,
                "first_index_row": item.first_row,
                "last_index_row": item.last_row,
            }
            for item in ranges
        ],
    }


def _download_product(
    request: ProductRequest, *, workers: int, retries: int,
) -> dict[str, object]:
    return _download_subset(
        url=request.url,
        index_url=request.index_url,
        index_path=request.index_path,
        destination=request.destination,
        kind=request.kind,
        workers=workers,
        retries=retries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--forecast-hours", default="0,1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--file-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    cycle = _cycle(args.cycle)
    hours = _hours(args.forecast_hours, cycle=cycle)
    if args.workers not in range(1, 33) or args.retries not in range(1, 11):
        raise ValueError("workers must be 1..32 and retries must be 1..10")
    if args.file_workers not in range(1, 17):
        raise ValueError("file-workers must be 1..16")
    if args.workers * args.file_workers > 64:
        raise ValueError("workers * file-workers may not exceed 64")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "download-receipt.json"
    manifest_path = output / "SHA256SUMS"
    if receipt_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace an existing HRRR download receipt")
    prefix = f"{args.base_url.rstrip('/')}/hrrr.{cycle:%Y%m%d}/conus"
    requests = []
    for hour in hours:
        forecast = f"{hour:02d}"
        atmosphere_name = f"hrrr.t{cycle:%H}z.wrfnatf{forecast}.grib2"
        pressure_name = f"hrrr.t{cycle:%H}z.wrfprsf{forecast}.grib2"
        soil_name = f"hrrr.t{cycle:%H}z.soilf{forecast}.grib2"
        requests.append(ProductRequest(
            url=f"{prefix}/{atmosphere_name}",
            index_url=f"{prefix}/{atmosphere_name}.idx",
            index_path=output / f"{atmosphere_name}.idx",
            destination=output / atmosphere_name,
            kind="atmosphere",
        ))
        requests.append(ProductRequest(
            url=f"{prefix}/{pressure_name}",
            index_url=f"{prefix}/{pressure_name}.idx",
            index_path=output / f"{pressure_name}.idx",
            destination=output / soil_name,
            kind="soil",
        ))
    download = partial(
        _download_product, workers=args.workers, retries=args.retries)
    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        # Executor.map preserves request order, so the receipt is deterministic
        # even when products complete out of order.
        products = list(pool.map(download, requests))
    manifest_records = sorted(
        (item["subset_path"], item["subset_sha256"]) for item in products)
    manifest_path.write_text("".join(
        f"{digest}  {name}\n" for name, digest in manifest_records),
        encoding="ascii",
    )
    receipt = {
        "schema": "rw-wps.hrrr-native-range-download.v1",
        "status": "PASS_TRANSPORT_SUBSET_RUST_SCIENCE_VALIDATION_REQUIRED",
        "cycle": cycle.isoformat() + "Z",
        "source_forecast_hours": list(hours),
        # Kept for compatibility with existing download consumers.
        "forecast_hours": list(hours),
        "workers": args.workers,
        "file_workers": args.file_workers,
        "products": products,
        "payload_bytes": sum(item["subset_bytes"] for item in products),
        "source_bytes": sum(item["source_bytes"] for item in products),
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "payload_bytes": receipt["payload_bytes"],
        "source_bytes": receipt["source_bytes"],
        "manifest": str(manifest_path),
        "manifest_sha256": receipt["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
