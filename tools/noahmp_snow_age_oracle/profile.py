"""Small full-column night/dawn check using the delivered WRF driver harness."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noahmp_wrf461_oracle/column"))
import gen_columns as g
from run_woof_column import run

g.CONFIGS["night_dawn"] = dict(start=datetime(2026, 1, 1, 6), dt=300.0,
                               nstep=6, dx=12000.0, seed=4610)
original_columns, original_forcing = g.build_columns, g.forcing_at


def columns(seed, dt):
    return [c for c in original_columns(seed, dt)
            if c["regime"].startswith(("snow_", "glacier_", "stable_", "desert"))]


def forcing(column, index, step, nstep):
    out = original_forcing(column, index, step, nstep)
    out["coszen"] = (-0.3, -0.0, 0.0, 0.0002, 0.1, 0.6)[step]
    out["swdown"] = (0, 0, 0, 2, 40, 200)[step]
    return out


g.build_columns, g.forcing_at = columns, forcing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("gen", "run"))
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--generation", choices=("wrf_461", "wrf_470"))
    args = parser.parse_args()
    if args.action == "gen":
        g.write(args.input, "night_dawn")
    else:
        result = run(args.input, args.output, noahmp_snow_age=args.generation)
        from gpuwm.certify.kernel_manifest import kernel_manifest
        result["kernel_manifest"] = kernel_manifest()
        Path(args.output + ".json").write_text(json.dumps(result, default=str, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "kernel_manifest"}, default=str))
