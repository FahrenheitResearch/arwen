"""Pack native WRF stream files without altering any binary32 word."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def pack(build, destination, selected=None):
    build, destination = Path(build), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pins = {}
    for line in (build / "fixture-sha256sums.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        pins[relative.strip()] = digest
    cases = {}
    for manifest in sorted((build / "fixtures").rglob("MANIFEST.txt")):
        case = manifest.parent.relative_to(build / "fixtures").as_posix()
        if selected and case not in selected:
            continue
        arrays = {}
        source_hashes = {}
        for line in manifest.read_text().splitlines():
            name, dtype, rank, *shape = line.split()
            path = manifest.parent / (name + ".bin")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(build).as_posix()
            if pins.get(relative) != digest:
                raise ValueError(f"{relative}: binary stream differs from build receipt")
            dimensions = tuple(map(int, shape))
            array = np.fromfile(path, dtype="<" + dtype).reshape(dimensions, order="F")
            if int(rank) == 2:
                array = array.transpose(1, 0)
            elif int(rank) == 3:
                array = array.transpose(1, 2, 0)
            elif int(rank) == 4:
                array = array.transpose(3, 1, 2, 0)
            arrays[name] = np.ascontiguousarray(array) if dimensions else array
            source_hashes[name] = digest
        output = destination / (case + ".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        cases[case] = {
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "native_stream_sha256": source_hashes,
            "arrays": {n: {"shape": list(a.shape), "dtype": a.dtype.str} for n, a in arrays.items()},
        }
    receipt = {
        "source": "WRF v4.7.1",
        "commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
        "tar_sha256": "0826f8878aa787e016586b6e3b43896c8467570b7cca47741b5e16d2f4def0ae",
        "compiler": (build / "compiler.txt").read_text().strip(),
        "libc": (build / "libc.txt").read_text().strip() if (build / "libc.txt").exists() else "unrecorded",
        "flags": (build / "compiler-flags.txt").read_text().strip(),
        "source_sha256sums": (build / "source-sha256sums.txt").read_text(),
        "oracle_sha256sums": "\n".join(digest + "  " + Path(name).name for digest, name in
                    (line.split(maxsplit=1) for line in (build / "oracle-sha256sums.txt").read_text().splitlines())) + "\n",
        "reference_kind": "explicitly corrected TG source" if (build / "corrected-atm-sha256.txt").exists() else "byte-unmodified original WRF",
        "layout": "2D (j,i); 3D (k,j,i); little-endian f4/i4; scalar rank zero",
        "cases": cases,
    }
    if (build / "corrected-atm-sha256.txt").exists():
        receipt["corrected_atm_sha256"] = (build / "corrected-atm-sha256.txt").read_text().split()[0]
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "words": sum(a.size for c in cases for a in np.load(destination / (c + '.npz')).values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build")
    parser.add_argument("destination")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    pack(args.build, args.destination, args.case)
