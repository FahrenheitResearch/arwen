"""State-payload lower bounds, not complete ensemble admission or pricing."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    from gpuwm.config import RunConfig
    from gpuwm.core import device_inventory

    # This optimistic lower bound assumes these prepared references agree byte
    # for byte. Real ensemble-source admission must verify that condition.
    shared = frozenset(("ht", "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f",
                        "c4f", "dc3f", "dc4f", "msft", "msfu", "msfv", "f", "e",
                        "sina", "cosa", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
                        "znu", "znw", "thb", "pb", "alb", "phb", "dphb_resid", "mub2d"))
    rows = []
    for nx, ny, nz, spacing in ((400, 400, 50, 3000.0), (200, 200, 50, 1000.0)):
        for mp, moist in ((0, False), (8, True), (10, True)):
            cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=spacing, dy=spacing,
                            ztop=20000.0, dt=10.0, run_seconds=120.0,
                            terrain_opt=1, moist=moist, mp_physics=mp)
            shapes = device_inventory.state_array_shapes(cfg)
            payload = {name: prod(shape) * 4 for name, shape in shapes.items()}
            fixed = sum(size for name, size in payload.items() if name in shared)
            member = sum(size for name, size in payload.items() if name not in shared)
            rows.append({"spatial_shape": [nz, ny, nx], "spacing_m": spacing,
                         "microphysics_state_selector": mp, "shared_base_assumed_identical": True,
                         "shared_payload_bytes": fixed, "per_member_payload_bytes": member,
                         "state_only_payload_bytes": {n: fixed + n * member for n in (1, 4, 10, 20, 40)},
                         "inventory": {name: {"shape": shapes[name], "payload_bytes": payload[name],
                                              "ownership": "shared" if name in shared else "member"}
                                       for name in sorted(shapes)}})
    receipt = {"schema": "gpuwm/ensemble-state-lower-bound/v1",
               "scope": "state array payload only; not forecast admission",
               "excluded": ["scratch", "physics driver and tables", "boundaries", "products and output",
                            "pool rounding and retention", "context", "kernel local memory"],
               "inventory_source_sha256": hashlib.sha256(Path(device_inventory.__file__).read_bytes()).hexdigest(),
               "rows": rows}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "rows": len(rows)}))


if __name__ == "__main__":
    main()
