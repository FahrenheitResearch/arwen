"""Refresh compiled-artifact counts from the release bundle registry.

Run before update_zarr_license_notice.py, which retains the binary notice whole.
Existing licence and attribution text is preserved byte for byte.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from gpuwm.bridge_assets import BUNDLED_ARTIFACTS

ROOT = Path(__file__).resolve().parents[1]
COPIES = (
    "NOTICE",
    "licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt",
    "tools/rustwx/assets/basemap/THIRD-PARTY-LICENSES.txt",
)


def render(text: str, count: int, *, root_notice: bool) -> str:
    if root_notice:
        pattern = (r"(?m)^(?:[A-Za-z-]+(?: [A-Za-z-]+)*|\d+) Rust executables and libraries "
                   r"(?=\(gpuwm\.bridge_assets\n\.BUNDLED_ARTIFACTS\))")
        replacement = f"{count} Rust executables and libraries "
    else:
        pattern = r"(?<=wheels: )\d+(?= Rust executables and\nlibraries,)"
        replacement = str(count)
    result, matches = re.subn(pattern, replacement, text)
    if matches != 1:
        raise ValueError("expected exactly one compiled-artifact count")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for relative in COPIES:
        path = ROOT / relative
        old = path.read_bytes().decode("utf-8")
        new = render(old, len(BUNDLED_ARTIFACTS), root_notice=relative == "NOTICE")
        if new != old:
            stale.append(relative)
            if not args.check:
                path.write_bytes(new.encode("utf-8"))
    if args.check and stale:
        parser.exit(1, "stale artifact counts: " + ", ".join(stale) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
