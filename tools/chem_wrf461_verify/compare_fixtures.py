"""Compare rebuilt chem oracle fixtures with the published ones, word for word.

    python3 compare_fixtures.py OUT PUBLISHED_ROOT > compare.json

OUT is a rebuild_oracles.sh (or build_optics461.sh) output folder; PUBLISHED_ROOT
is tests/data/oracles/chem.  For every MANIFEST.txt and .bin under each
published component, the rebuilt file of the same relative path is compared
byte for byte.  A differing float32 array also reports its largest ULP
distance (sign-magnitude ordering, NaN against number counts as 2**32).
Uses numpy only, so it runs without the gpuwm tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# rebuilt fixture folder (relative to OUT) -> published component (relative to PUBLISHED_ROOT)
COMPONENTS = {
    "fixtures-gocart/afwa": "gocart/afwa",
    "fixtures-gocart/ageing": "gocart/ageing",
    "fixtures-gocart/drydep": "gocart/drydep",
    "fixtures-gocart/dust": "gocart/dust",
    "fixtures-gocart/inventory": "gocart/inventory",
    "fixtures-gocart/niwfa": "gocart/niwfa",
    "fixtures-gocart/seasalt": "gocart/seasalt",
    "fixtures-gocart/settling": "gocart/settling",
    "fixtures-gocart/sulfur": "gocart/sulfur",
    "lane-mono/build/fixtures/advect_mono": "core/advect_mono",
    "lane-vertmx/build/fixtures/chem_prep": "core/chem_prep",
    "lane-bdy/build/fixtures/flowdep_bdy": "core/flowdep_bdy",
    "lane-pm/build/fixtures/sum_pm": "core/sum_pm",
    "lane-vertmx/build/fixtures/vertmx": "core/vertmx",
    "lane-smoke/build/fixtures_observed/plumerise_wrfchem": "smoke/plumerise_wrfchem",
    "lane-smoke/build/fixtures/wetdep_ls": "smoke/wetdep_ls",
    "wesely-build/fixtures": "cams/wesely",
    "optics461/fixtures": "gocart/optics",
}


def _ordered(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def max_ulp(a: bytes, b: bytes) -> int | None:
    if len(a) != len(b) or len(a) % 4:
        return None
    x = np.frombuffer(a, "<i4")
    y = np.frombuffer(b, "<i4")
    fx = x.view("<f4")
    fy = y.view("<f4")
    nan_x, nan_y = np.isnan(fx), np.isnan(fy)
    d = np.abs(_ordered(x) - _ordered(y))
    d[nan_x & nan_y] = 0
    d[nan_x ^ nan_y] = 2**32
    return int(d.max()) if d.size else 0


def compare(out: Path, published: Path) -> dict:
    result = {"components": {}, "total": {}}
    for rebuilt_rel, pub_rel in COMPONENTS.items():
        rebuilt = out / rebuilt_rel
        pub = published / pub_rel
        if not rebuilt.is_dir():
            continue
        entry = {"files": 0, "identical": 0, "differ": 0, "missing": 0,
                 "max_ulp": 0, "differing": [], "cases": 0}
        for path in sorted(pub.rglob("*")):
            if not path.is_file() or not (path.suffix == ".bin" or path.name.lower() == "manifest.txt"):
                continue
            rel = path.relative_to(pub)
            entry["files"] += 1
            entry["cases"] += path.name.lower() == "manifest.txt"
            other = rebuilt / rel
            if not other.is_file():
                entry["missing"] += 1
                entry["differing"].append({"file": str(rel), "why": "missing"})
                continue
            a, b = path.read_bytes(), other.read_bytes()
            if a == b:
                entry["identical"] += 1
                continue
            entry["differ"] += 1
            u = max_ulp(a, b) if path.suffix == ".bin" else None
            if u is not None:
                entry["max_ulp"] = max(entry["max_ulp"], u)
            entry["differing"].append({"file": str(rel), "max_ulp": u})
        result["components"][pub_rel] = entry
    comps = result["components"].values()
    result["total"] = {k: sum(c[k] for c in comps)
                       for k in ("files", "identical", "differ", "missing", "cases")}
    result["total"]["max_ulp"] = max((c["max_ulp"] for c in comps), default=0)
    return result


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
