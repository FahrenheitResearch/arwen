"""Pack the Noah mosaic oracle's raw case directories into one file per case.

``build.sh`` writes what ``oracle_io.F90`` writes: one directory per case
holding a ``MANIFEST.txt`` and one little-endian ``<name>.bin`` per array,
about 256 files a case.  Committing that corpus as-is is nine thousand files
for five megabytes.  This packs each case directory into ``<case>.npz``:

* ``__manifest__`` -- the manifest text, byte for byte, as uint8;
* ``<name>`` -- the ``.bin`` words as a flat little-endian array of the
  manifest's kind (the Fortran-order reshape stays in the loader, in one
  place, as it did for the raw directories).

Every ``.bin`` is checked against ``receipts/fixture-sha256sums.txt`` before it
is packed, so a pack can only hold the words the Fortran oracle wrote.  The zip
members carry a fixed timestamp, so packing the same corpus twice gives the
same bytes.

Usage::

    python tools/noah_mosaic_wrf471_oracle/pack_fixtures.py RAW_ORACLE_DIR OUT_DIR

``RAW_ORACLE_DIR`` is the directory holding the family folders (``base``,
``cats``, ...); ``OUT_DIR`` receives ``<family>/<case>.npz``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHA_LIST = HERE / "receipts" / "fixture-sha256sums.txt"
_KINDS = {"f4": np.dtype("<f4"), "i4": np.dtype("<i4")}
#: The earliest date a zip can carry; fixed so repacks are byte-identical.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _known_hashes(path: Path = SHA_LIST) -> dict[str, str]:
    table = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            table[name.strip()] = digest
    return table


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array),
                              allow_pickle=False)
    return buffer.getvalue()


def pack_case(case_dir: Path, out_file: Path, known: dict[str, str],
              label: str) -> int:
    """Pack one case directory; returns the number of arrays packed."""
    manifest = (case_dir / "MANIFEST.txt").read_bytes()
    members: list[tuple[str, bytes]] = [
        ("__manifest__.npy",
         _npy_bytes(np.frombuffer(manifest, dtype=np.uint8)))]
    seen: set[str] = set()
    for line in manifest.decode("ascii").splitlines():
        parts = line.split()
        if not parts:
            continue
        name, kind = parts[0], parts[1]
        # run_mosaic.F90 writes the optional urban SFCDIF coefficients twice
        # under one name; both manifest lines name the one .bin on disk.
        if name in seen:
            continue
        seen.add(name)
        raw = (case_dir / f"{name}.bin").read_bytes()
        key = f"{label}/{name}.bin"
        digest = hashlib.sha256(raw).hexdigest()
        if known.get(key) != digest:
            raise ValueError(
                f"{key}: sha256 {digest} is not the recorded oracle word "
                f"hash {known.get(key)}; refusing to pack words the Fortran "
                "oracle did not write")
        members.append((f"{name}.npy",
                        _npy_bytes(np.frombuffer(raw, dtype=_KINDS[kind]))))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member, payload in members:
            info = zipfile.ZipInfo(member, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)
    return len(members) - 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    known = _known_hashes()
    packed = 0
    for family in sorted(p for p in args.raw_dir.iterdir() if p.is_dir()):
        for case in sorted(p for p in family.iterdir() if p.is_dir()):
            if not (case / "MANIFEST.txt").is_file():
                continue
            label = f"{family.name}/{case.name}"
            arrays = pack_case(case, args.out_dir / family.name
                               / f"{case.name}.npz", known, label)
            packed += arrays
            print(f"{label}: {arrays} arrays")
    print(f"packed {packed} arrays")


if __name__ == "__main__":
    main()
