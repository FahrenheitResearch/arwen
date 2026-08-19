"""Copy selected whole GRIB2 envelopes out of a file, bytes untouched.

Fixture tooling, not a data path: a GRIB2 file is a plain concatenation
of self-delimiting ``GRIB .. 7777`` envelopes, so a subset of envelopes
is itself a valid GRIB2 file of unmodified production bytes.  This is
how the committed real-byte fixtures are produced from files too large
to commit whole; every decoded fact about them still comes from the
Rust bridge.

Usage:
    python tools/slice_grib2_envelopes.py INPUT               # list
    python tools/slice_grib2_envelopes.py INPUT OUTPUT N [N.] # slice
"""

from __future__ import annotations

from pathlib import Path
import sys


def envelopes(payload: bytes) -> list[tuple[int, int]]:
    """(offset, length) of every envelope, validated exactly."""

    result: list[tuple[int, int]] = []
    offset = 0
    while offset < len(payload):
        if payload[offset:offset + 4] != b"GRIB":
            raise SystemExit(
                f"no GRIB marker at byte {offset}: not a GRIB2 file")
        if payload[offset + 7] != 2:
            raise SystemExit(
                f"envelope at byte {offset} is edition {payload[offset + 7]}")
        length = int.from_bytes(payload[offset + 8:offset + 16], "big")
        end = offset + length
        if end > len(payload) or payload[end - 4:end] != b"7777":
            raise SystemExit(f"envelope at byte {offset} is truncated")
        result.append((offset, length))
        offset = end
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    payload = Path(argv[1]).read_bytes()
    found = envelopes(payload)
    if len(argv) == 2:
        for index, (offset, length) in enumerate(found):
            print(f"{index}\t{offset}\t{length}")
        return 0
    output = Path(argv[2])
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    indices = [int(value) for value in argv[3:]]
    for index in indices:
        if not 0 <= index < len(found):
            raise SystemExit(
                f"envelope {index} out of range ({len(found)} present)")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for index in indices:
            offset, length = found[index]
            stream.write(payload[offset:offset + length])
    print(f"{output}: {len(indices)} envelope(s), "
          f"{output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
