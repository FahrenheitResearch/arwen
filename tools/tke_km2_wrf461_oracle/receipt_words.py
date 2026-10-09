"""Word hashes of the smag2d-pinned diffusion receipt families, for one tree.

Runs the existing receipt producers (tools/wrf_diffusion_oracle) on the
retained WRF v4.7.1 fixtures of the tree on PYTHONPATH and records the
SHA-256 of every output array, so two trees measured on the SAME card and
NVRTC can be diffed word-for-word: which receipt words a source change moved.

Usage: python receipt_words.py OUT_JSON   (PYTHONPATH selects the tree)
       python receipt_words.py --diff A.json B.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _flatten(prefix: str, value, out: dict) -> None:
    """Record every nested ``gpu_sha256`` under its path of keys."""
    if isinstance(value, dict) and "gpu_sha256" in value:
        out[prefix] = value["gpu_sha256"]
    elif isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}/{key}", inner, out)


def measure(root: Path) -> dict:
    tools = root / "tools/wrf_diffusion_oracle"
    data = root / "tests/data/wrf471_diffusion"
    sys.path.insert(0, str(tools))
    import deformation_compare
    import deformation_mutations
    import horizontal_driver
    import vertical_driver
    out: dict = {}
    for path in sorted(data.glob("deformation-*.npz")):
        if path.name.startswith(("deformation-gpu", "deformation-reference")):
            continue
        _flatten(f"deformation/{path.name}", deformation_compare.measure_fixture(path), out)
    manifest = json.loads((data / "km-mutations/manifest.json").read_text())
    for case in manifest["cases"]:
        _flatten(f"km-mutations/{case['file']}",
                 deformation_mutations.compare_case(data / case["file"],
                                                    data / "km-mutations" / case["file"]), out)
    for folder, module in (("horizontal-driver", horizontal_driver),
                           ("vertical-driver", vertical_driver)):
        manifest = json.loads((data / folder / "manifest.json").read_text())
        for case in manifest["cases"]:
            _flatten(f"{folder}/{case['file']}",
                     module.compare_case(data / folder / case["file"]), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--diff", nargs=2)
    args = ap.parse_args()
    if args.diff:
        a, b = (json.loads(Path(p).read_text()) for p in args.diff)
        moved = sorted(k for k in a if a[k] != b.get(k))
        families: dict = {}
        for key in moved:
            fam, _case, *rest = key.split("/")
            label = "/".join([fam] + rest[:-1])
            families.setdefault(label, set()).add(rest[-1])
        print(json.dumps({"arrays": len(a), "moved": len(moved),
                          "moved_by_family_arm": {k: sorted(v) for k, v in sorted(families.items())}},
                         indent=1))
        return
    import gpuwm
    root = Path(gpuwm.__file__).resolve().parents[1]
    Path(args.out).write_text(json.dumps(measure(root), indent=0) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
