"""The v1.2 end-to-end composition gate, in one command (EXPERIMENTAL).

    python -m tools.da_synthetic_cycle --out DIR [--members 10] [--no-products]

Runs the whole DA chain on a synthetic domain -- perturbations, a forecast
leg through the real cycle driver, synthetic radar observations packed as a
real ``gpuwm-obs.radar-grid.v1`` file, a LETKF analysis, the caller-side
positivity policy, a second leg restarted from the analysis, and ensemble
products -- and prints a JSON report.

Exit 0 only if the analysis ensemble mean is closer to the truth than the
prior was.  See :mod:`gpuwm.da.synthetic_cycle` for what is real here and
what is stood in for; the short version is that every seam is shipped code
and the dycore is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.da.synthetic_cycle import GateConfig, run_gate  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="da_synthetic_cycle", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the run's artifacts")
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--obs-error-ms", type=float, default=1.0)
    parser.add_argument("--localization-km", type=float, default=18.0)
    parser.add_argument("--no-products", action="store_true",
                        help="skip the enprod stage (no matplotlib/wrf)")
    args = parser.parse_args(argv)

    cfg = GateConfig(
        members=args.members, nx=args.nx, ny=args.ny, nz=args.nz,
        base_seed=args.seed, obs_error_ms=args.obs_error_ms,
        horizontal_localization_m=args.localization_km * 1000.0,
        products=not args.no_products)
    report = run_gate(args.out, cfg)
    print(json.dumps(report, indent=2, sort_keys=True))

    improvement = report["rmse"]["improvement_pct"]["total"]
    if improvement <= 0.0:
        print(f"GATE FAILED: the analysis is not closer to the truth than "
              f"the prior ({improvement:+.2f}%)", file=sys.stderr)
        return 1
    print(f"GATE PASSED: analysis-mean RMSE {improvement:+.2f}% against the "
          f"prior", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
