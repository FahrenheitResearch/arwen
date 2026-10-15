"""Pack the oracle's outputs into the committed fixture.

    python make_fixture.py INPUT.bin LABELS.json WRF_O0.bin WRF_O2VEC.bin \
        ORACLE_DIR OUT_DIR

Writes, into OUT_DIR (tests/data/oracles/rrtm_lw/):

* ``columns.bin`` and ``labels.json`` -- the exact float32 inputs, copied;
* ``wrf461-O0.npz`` -- the -O0 reference: every output RRTMLWRAD returns,
  every stage intermediate except the three (ncol, nlayers, 140) g-point
  arrays (TAUG, PFRAC, ITR, ~8 MB; they stay box-side with the raw dump and
  are graded by ``compare.py --stages``), and the tables rrtm_lookuptable
  builds;
* ``wrf461-O2vec-final.npz`` -- WRF's own -O2 -ftree-vectorize build, the
  four outputs plus TAUCLOUD (the only field it moves on this fixture);
* ``receipts.json`` -- sha256 of every input and raw dump, the compiler, the
  libm report.

Keys are ``"<set>/<field>"`` and ``"tables/<name>"``; arrays keep the dtype
the Fortran wrote, so the npz holds the exact bits.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import read_outputs  # noqa: E402

GPOINT_FIELDS = ("taug", "pfrac", "itr")
O2VEC_FIELDS = ("rthraten", "rthratenc", "glw", "olr", "taucloud")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv):
    if len(argv) != 7:
        print(__doc__)
        return 2
    inp, labels, o0, o2, oracle_dir, out = (Path(a) for a in argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inp, out / "columns.bin")
    shutil.copyfile(labels, out / "labels.json")
    names = list(json.loads(labels.read_text()))

    tables, sets = read_outputs(o0)
    packed = {f"tables/{k}": v for k, v in tables.items()}
    for name, rep in zip(names, sets):
        for field, value in rep.items():
            if field in GPOINT_FIELDS:
                continue
            packed[f"{name}/{field}"] = value
    np.savez_compressed(out / "wrf461-O0.npz", **packed)

    _, sets2 = read_outputs(o2)
    packed2 = {}
    for name, rep in zip(names, sets2):
        for field in O2VEC_FIELDS:
            packed2[f"{name}/{field}"] = rep[field]
    np.savez_compressed(out / "wrf461-O2vec-final.npz", **packed2)

    receipts = {
        "wrf": "WRF v4.6.1 d66e442fccc04111067e29274c9f9eaccc3cef28, "
               "phys/module_ra_rrtm.F unmodified",
        "compiler": (oracle_dir / "compiler.txt").read_text().strip(),
        "libm_report": (oracle_dir / "libmvec-report.txt").read_text(),
        "sha256": {
            "columns.bin": _sha(inp),
            "wrf-out-O0.bin": _sha(o0),
            "wrf-out-O2vec.bin": _sha(o2),
        },
        # file names only: the box's directory layout is not a fact about
        # the oracle.
        "oracle_sha256sums": "".join(
            f"{line.split()[0]}  {Path(line.split()[1]).name}\n"
            for line in (oracle_dir / "oracle-sha256sums.txt")
            .read_text().splitlines() if line.strip()),
    }
    (out / "receipts.json").write_text(json.dumps(receipts, indent=1) + "\n")
    print(f"fixture written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
