"""Write a small column-oracle case and its WRF answers as hex CSVs for
tests/test_ntiedtke_column_oracle.py.

    python export_fixture.py CASE OUTDIR TAG=WRF_RUN_DIR [TAG=WRF_RUN_DIR ...]

Every float is its raw IEEE-754 word in hex, the convention of the stage
oracle in tests/data/oracles/ntiedtke, because the bar is 0 ULP and a
decimal rendering is a lossy view of the thing being pinned.  The case is
in WRF order (k = 1 is the surface), as the adapter receives it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_columns import IFACE, LEV, SFC  # noqa: E402
from make_subset import load  # noqa: E402

OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
           "rvcuten")
OUT_SFC = ("raincv", "pratec")


def hx(a):
    return [f"{w:08x}" for w in np.asarray(a, dtype="<f4").view("<u4")]


def main() -> int:
    case, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    meta, lev, iface, sfc = load(case)
    ny, nx, nz = int(meta["ny"]), int(meta["nx"]), int(meta["nz"])
    assert ny == 1, "export a subset case (ny = 1)"
    nl = "\n"
    with open(out / "nt-columns-input-levels.csv", "w", newline=nl) as f:
        f.write("col,k," + ",".join(LEV) + nl)
        for k in range(nz):
            for c in range(nx):
                f.write(f"{c + 1},{k + 1}," + ",".join(
                    hx(lev[n][0, k, c:c + 1])[0] for n in LEV) + nl)
    with open(out / "nt-columns-input-iface.csv", "w", newline=nl) as f:
        f.write("col,k," + ",".join(IFACE) + nl)
        for k in range(nz + 1):
            for c in range(nx):
                f.write(f"{c + 1},{k + 1}," + ",".join(
                    hx(iface[n][0, k, c:c + 1])[0] for n in IFACE) + nl)
    with open(out / "nt-columns-input-surface.csv", "w", newline=nl) as f:
        f.write("col," + ",".join(SFC) + nl)
        for c in range(nx):
            f.write(f"{c + 1}," + ",".join(
                hx(sfc[n][0, c:c + 1])[0] for n in SFC) + nl)
    with open(out / "nt-columns-meta.txt", "w", newline=nl) as f:
        for k in ("nz", "dt", "dx_native", "source", "pressure",
                  "radiation_lanes", "subset_of"):
            if k in meta:
                f.write(f"{k} {meta[k]}{nl}")
        f.write(f"ncol {nx}{nl}")
    for spec in sys.argv[3:]:
        tag, run = spec.split("=", 1)
        run = pathlib.Path(run)
        with open(out / f"nt-columns-wrf-{tag}-levels.csv", "w",
                  newline=nl) as f:
            f.write("col,k," + ",".join(OUT_LEV) + nl)
            arrs = {n: np.fromfile(run / f"{n}.f32", "<f4").reshape(nz, nx)
                    for n in OUT_LEV}
            for k in range(nz):
                for c in range(nx):
                    f.write(f"{c + 1},{k + 1}," + ",".join(
                        hx(arrs[n][k, c:c + 1])[0] for n in OUT_LEV) + nl)
        with open(out / f"nt-columns-wrf-{tag}-surface.csv", "w",
                  newline=nl) as f:
            f.write("col," + ",".join(OUT_SFC) + nl)
            arrs = {n: np.fromfile(run / f"{n}.f32", "<f4") for n in OUT_SFC}
            for c in range(nx):
                f.write(f"{c + 1}," + ",".join(
                    hx(arrs[n][c:c + 1])[0] for n in OUT_SFC) + nl)
    print(f"exported {nx} columns x {nz} levels to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
