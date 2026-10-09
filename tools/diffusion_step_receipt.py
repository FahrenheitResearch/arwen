"""Record step times and complete field hashes from a real CLI forecast.

NetCDF decoding uses the engine's Rust bridge. This script keeps no arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("--commit", required=True, help="commit used by the recorded run")
    a = p.parse_args()
    from gpuwm import netcdf_bridge
    progress = list(a.run.rglob("progress.jsonl"))
    steps = [row for path in progress for line in path.read_text().splitlines()
             if (row := json.loads(line)).get("event") == "step"]
    times = [row["step_wall_seconds"] for row in steps]
    frames = {}
    for path in sorted(a.run.rglob("wrfout_d0*")):
        if not path.is_file() or path.suffix == ".json":
            continue
        fields = {}
        with netcdf_bridge.open_dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if name == "Times":
                    continue
                values = np.ascontiguousarray(variable[:])
                fields[name] = {"shape": list(values.shape), "dtype": str(values.dtype),
                                "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
                if np.issubdtype(values.dtype, np.floating):
                    fields[name]["nonfinite_words"] = int(np.count_nonzero(~np.isfinite(values)))
        frames[path.relative_to(a.run).as_posix()] = fields
    result = {"commit": a.commit,
              "observer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "run": str(a.run), "steps": len(times), "step_seconds": times,
              "median_after_first": statistics.median(times[1:]) if len(times) > 1 else None,
              "mean_after_first": statistics.mean(times[1:]) if len(times) > 1 else None,
              "step_sum_seconds": sum(times), "frames": frames}
    a.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("frames", "step_seconds")}))
    return int(not times or not frames)


if __name__ == "__main__":
    raise SystemExit(main())
