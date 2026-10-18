"""Compare two Noah mosaic oracle builds word for word.

Usage::

    python tools/noah_mosaic_wrf461_oracle/compare_raw.py BUILD_A BUILD_B OUT.json [HASH_LIST]

``BUILD_A`` and ``BUILD_B`` are ``build.sh`` output directories (their
``fixtures/`` trees are compared case by case, array by array).  With
``HASH_LIST`` (``tools/noah_mosaic_wrf471_oracle/receipts/fixture-sha256sums.txt``)
every array of ``BUILD_A`` is also checked against the committed corpus's
recorded SHA-256, which proves the committed fixtures are the words this
build writes.  Every differing array reports its count of differing words
and its max ULP.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _arrays(build: Path) -> dict[str, Path]:
    out = {}
    fixtures = build / "fixtures"
    for path in sorted(fixtures.rglob("*.bin")):
        rel = path.relative_to(fixtures)
        parts = rel.parts
        # fixtures/<driver>/<family>/<case>/<name>.bin -> <family>/<case>/<name>.bin
        key = "/".join(parts[1:]) if parts[0] in ("mosaic", "mosaic_ucm") else "/".join(parts)
        out[key] = path
    return out


def _kind(path: Path) -> np.dtype:
    for line in (path.parent / "MANIFEST.txt").read_text(encoding="ascii").splitlines():
        parts = line.split()
        if parts and parts[0] == path.stem:
            return np.dtype("<f4") if parts[1] == "f4" else np.dtype("<i4")
    raise KeyError(path)


def _ulp(a: np.ndarray, b: np.ndarray) -> int:
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ka = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    kb = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    nan = np.isnan(a) | np.isnan(b)
    d = np.abs(ka - kb)
    d = np.where(nan & (ia != ib), 2**32, np.where(nan, 0, d))
    return int(d.max()) if d.size else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a_dir, b_dir, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    a, b = _arrays(a_dir), _arrays(b_dir)
    known = {}
    if len(argv) > 3:
        for line in Path(argv[3]).read_text(encoding="ascii").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                known[name.strip()] = digest
    report = {"a": str(a_dir), "b": str(b_dir), "arrays_a": len(a), "arrays_b": len(b),
              "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)),
              "identical": 0, "differing": {}, "families": {}}
    for key in sorted(set(a) & set(b)):
        family = key.split("/", 1)[0]
        fam = report["families"].setdefault(family, dict(arrays=0, differing_arrays=0,
                                                         words=0, differing_words=0, max_ulp=0))
        ra, rb = a[key].read_bytes(), b[key].read_bytes()
        kind = _kind(a[key])
        wa, wb = np.frombuffer(ra, kind), np.frombuffer(rb, kind)
        fam["arrays"] += 1
        fam["words"] += wa.size
        if ra == rb:
            report["identical"] += 1
            continue
        fam["differing_arrays"] += 1
        bad = int(np.count_nonzero(wa.view(np.uint32) != wb.view(np.uint32))) if wa.size == wb.size else wa.size
        ulp = _ulp(wa, wb) if (kind == np.dtype("<f4") and wa.size == wb.size) else -1
        fam["differing_words"] += bad
        fam["max_ulp"] = max(fam["max_ulp"], ulp)
        report["differing"][key] = dict(words=int(wa.size), differing_words=bad, max_ulp=ulp)
    if known:
        hits = misses = 0
        missing = []
        for key, path in a.items():
            if key not in known:
                missing.append(key)
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() == known[key]:
                hits += 1
            else:
                misses += 1
        report["hash_list"] = dict(path=argv[3], matching=hits, differing=misses,
                                   not_listed=len(missing),
                                   not_listed_families=sorted({m.split("/", 1)[0] for m in missing}))
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("differing",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
