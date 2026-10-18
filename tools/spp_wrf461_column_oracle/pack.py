"""Pack an SPP column-oracle build directory into the test fixture.

    python tools/spp_wrf461_column_oracle/pack.py --oracle <build dir> \
        [--output tests/data/oracles/spp/columns]

The GF columns are folded into the order of the GF fixture's column key
(``gpuwm.verify.gf_oracle.load_gf_oracle``), and the prepared columns the
harness echoes are checked word for word against that fixture first: the
SPP capture and the GF inputs WOOF is launched with must be the same
columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import spp_column_fixtures as columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=columns.FIXTURE_DIR)
    args = parser.parse_args()
    from gpuwm.verify.gf_oracle import load_gf_oracle
    receipt = json.loads((args.oracle / "receipt.json").read_text())
    for name, entry in receipt["files"].items():
        if columns.sha256(args.oracle / name) != entry["sha256"]:
            raise SystemExit(f"{name} does not match its receipt")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for mixlength in (1, 2):
        columns.pack_mynn(args.oracle / f"mynn-spp-ml{mixlength}.csv",
                          out / f"mynn-spp-ml{mixlength}.npz")
    fixture = load_gf_oracle()
    for offset in receipt["gf_offsets"]:
        target = out / f"gf-spp-o{offset}.npz"
        columns.pack_gf(args.oracle / f"gf-spp-o{offset}-levels.csv",
                        args.oracle / f"gf-spp-o{offset}-surface.csv",
                        fixture.key, target)
        packed = columns.load_gf(target)
        for name in columns.GF_ECHO_LEVELS:
            want = fixture.stage_levels[name].astype(np.float32)
            if not np.array_equal(packed[name].view(np.uint32), want.view(np.uint32)):
                raise SystemExit(f"GF SPP capture's prepared {name} is not the GF fixture's")
    shutil.copy2(args.oracle / "receipt.json", out / "receipt.json")
    manifest = {path.name: columns.sha256(path) for path in sorted(out.glob("*.npz"))}
    (out / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
