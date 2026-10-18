"""Pack the column oracle's inputs and WRF outputs into the test fixture.

usage: python make_fixture.py ORACLE_DIR FIXTURE_DIR

ORACLE_DIR holds cols/ (make_columns.py), build/<arm>/ and wrf/<arm>/
(build.sh and rad_column_oracle runs).  FIXTURE_DIR receives:

* columns_<grid>.npz, regimes.json  the exact float32 inputs
* wrf461.npz   "<regime>/<field>" -> (ncol, nk) or (ncol,) float32, the
  fields tests/test_rrtmg_legacy_column_oracle.py grades.  aer_opt = 0
  regimes come from the stock arm; thompson28_aer3 from the fork arm (WRF
  v4.6.1 RRTMG with the operational HRRR fork's aerosol optics, the
  generation WOOF's aer_opt = 3 implements).
* provenance.json  build receipts, source hashes and the arm per regime.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import PAIRS, PUBLISHED, read_records, to_cols  # noqa: E402


def arm_for(regime: dict) -> str:
    return "fork" if regime["aer_opt"] == 3 else "stock"


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "cols" / "regimes.json").read_text())
    for name in ["regimes.json"] + sorted(p.name for p in (src / "cols").glob("columns_*.npz")):
        shutil.copyfile(src / "cols" / name, dst / name)
    ncol = meta["ncol"]
    out = {}
    arms = {}
    for name, regime in meta["regimes"].items():
        arm = arm_for(regime)
        arms[name] = arm
        rec = read_records(src / "wrf" / arm / f"out_{name}.bin")
        for field in list(PAIRS) + list(PUBLISHED):
            if field in rec:
                out[f"{name}/{field}"] = np.ascontiguousarray(
                    to_cols(rec[field], ncol), dtype=np.float32)
    np.savez_compressed(dst / "wrf461.npz", **out)
    prov = {"arms": arms, "builds": {}}
    for arm in sorted(set(arms.values())):
        d = src / "build" / arm
        prov["builds"][arm] = {
            "build_receipt": (d / "build-receipt.txt").read_text().splitlines(),
            "units_receipt": json.loads((d / "units-receipt.json").read_text())}
    prov["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(dst.glob("*.np*")) + [dst / "regimes.json"]}
    (dst / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"fixture: {len(out)} arrays -> {dst}")


if __name__ == "__main__":
    main()
