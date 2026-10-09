"""Write the maintained F5 registry maturity-surface receipt."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))


def build(registry: dict) -> dict:
    from gpuwm.physics_registry import iter_maturity_surfaces

    surfaces = sorted(iter_maturity_surfaces(registry))
    counts: dict[str, int] = {}
    for path, _maturity in surfaces:
        kind = ("component-option" if path.startswith("components.") else
                "template" if path.startswith("templates.") else
                "transition-cross-option" if path.startswith("transitions.") else "other")
        counts[kind] = counts.get(kind, 0) + 1
    options = [option for component in registry["components"].values()
               for option in component["options"].values()]
    return {
        "schema": "gpuwm-f5-maturity-surface-walk-v1",
        "maturity_bearing_locations": len(surfaces),
        "by_surface_class": dict(sorted(counts.items())),
        "maturity_values_in_use": sorted({maturity for _, maturity in surfaces}),
        "rungs": sorted(registry["maturity_ladder"]["rungs"]),
        "scientific_evidence_in_use": sorted({option.get("scientific_evidence")
                                              for option in options
                                              if option.get("implemented") is True}),
        "implemented_option_count": sum(option.get("implemented") is True
                                        for option in options),
        "component_option_count": len(options),
    }


def main(argv=None) -> int:
    from gpuwm.physics_registry import canonical_json, physics_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=MODEL / "docs/public/receipts/F5-maturity-surface-walk.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = (canonical_json(build(physics_registry())) + "\n").encode("utf-8")
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != data:
            print(f"maturity walk receipt is stale: {args.out}")
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
