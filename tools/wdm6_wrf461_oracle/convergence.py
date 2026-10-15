"""Judge the two rain sub-step counts against a converged reference.

usage: python3 convergence.py FIXTURE_DIR RUNS_DIR [REFERENCE_ARM]

Reads the Fortran arms under RUNS_DIR (stock, only5t, only5, fine8 and the
reference, default fine32; see measure_split.sh) and prints, per
configuration, how far each arm's last-step surface rain, rain mass and
theta lie from the reference.  Every arm differs from the reference only in
the rain sedimentation loop, so the distance is that loop's time error:

  stock   WRF's loop (lossy transfer, WRF's count)
  only5t  conservative transfer, WRF's count
  only5   conservative transfer, WOOF's count (what the kernel runs)
  fine8   conservative transfer, 8x WOOF's count

Single-call configurations isolate the sedimentation error; over many calls
the column physics feeds back and the arms' trajectories separate for that
reason as well, so read the multi-call rows with that in mind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import read_input, read_output  # noqa: E402

ARMS = ("stock", "only5t", "only5", "fine8")


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    fixture, runs = Path(argv[1]), Path(argv[2])
    ref_arm = argv[3] if len(argv) == 4 else "fine32"
    names = [line.split()[1] for line in
             (fixture / "columns.txt").read_text().splitlines()]
    for inp in sorted(fixture.glob("*.in")):
        case = read_input(inp)
        def last(arm):
            steps = read_output(runs / arm / f"{inp.stem}.out", case.ncol,
                                case.nz, case.nsteps)
            return steps[-1]
        ref_lv, ref_sf = last(ref_arm)
        print(f"== {inp.stem} ({case.nsteps} call(s)), distance to {ref_arm}")
        for arm in ARMS:
            if arm == ref_arm:
                continue
            lv, sf = last(arm)
            dr = np.abs(sf["rain"].astype(np.float64) - ref_sf["rain"])
            dq = np.abs(lv["qr"].astype(np.float64) - ref_lv["qr"])
            dth = np.abs(lv["th"].astype(np.float64) - ref_lv["th"])
            print(f"  {arm:7s} rain max {dr.max():.4g} mm "
                  f"({names[int(dr.argmax())]}) mean {dr.mean():.3g} mm; "
                  f"qr max {dq.max():.3g} rms {np.sqrt((dq ** 2).mean()):.3g}; "
                  f"theta max {dth.max():.3g} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
