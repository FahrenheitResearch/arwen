#!/usr/bin/env python3
"""Write a fixture manifest that reuses the WRF 4.7.1 advection inputs.

The HRRR-fork fixture runs the same eight real-state cases (the same input
archives, referenced by relative path so no input byte is duplicated) with
the vertical advection orders operational HRRR sets: ``v_sca_adv_order``
and ``v_mom_adv_order`` from the command line (5 for the clone) on every
case.  Every other metadata field is copied unchanged, so a case means the
same state, boundary treatment and map factors in both fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="tests/data/wrf471_advect")
    parser.add_argument("output", type=Path, help="tests/data/wrf_legacy_advect")
    parser.add_argument("--v-sca-adv-order", type=int, default=5)
    parser.add_argument("--v-mom-adv-order", type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads((args.source / "cases.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    relative = Path("..") / args.source.resolve().name
    rows = []
    for row in manifest["cases"]:
        path = args.source / row["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["input_sha256"]:
            raise ValueError(f"{path} does not match its 4.7.1 manifest digest")
        metadata = dict(row["metadata"])
        metadata["v_sca_adv_order"] = args.v_sca_adv_order
        metadata["v_mom_adv_order"] = args.v_mom_adv_order
        metadata["h_sca_adv_order"] = 5
        metadata["h_mom_adv_order"] = 5
        rows.append({
            "name": row["name"],
            "file": (relative / row["file"]).as_posix(),
            "metadata": metadata,
            "input_sha256": row["input_sha256"],
            "array_sha256": row["array_sha256"],
        })
    out = {
        "schema_version": 1,
        "source_file": manifest["source_file"],
        "source_sha256": manifest["source_sha256"],
        "source_size_bytes": manifest["source_size_bytes"],
        "extractor_source_sha256": manifest["extractor_source_sha256"],
        "array_order": manifest["array_order"],
        "inputs_from": "tests/data/wrf471_advect (the same archives, by relative path)",
        "advection_orders": {"h_mom_adv_order": 5, "v_mom_adv_order": args.v_mom_adv_order,
                             "h_sca_adv_order": 5, "v_sca_adv_order": args.v_sca_adv_order},
        "reference": "NOAA-EMC/HRRR 40ee6058c WRFV3.9 module_advect_em.F (tools/advect_wrf_legacy_oracle)",
        "cross_check": "WRF 4.7.1 module_advect_em.F at the same orders (tools/advect_wrf471_oracle), files <case>-wrf471.npz",
        "cases": rows,
    }
    (args.output / "cases.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("ascii"))
    print(f"{len(rows)} cases -> {args.output / 'cases.json'}")


if __name__ == "__main__":
    main()
