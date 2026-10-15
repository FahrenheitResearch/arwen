"""Small full-field hash receipts; no simulation arrays leave the box."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noahmp_wrf461_oracle/column"))
from compare_column import snapshots, GLACIER_UNDEFINED
from gen_columns import read_records


def receipt(input_path, output_path):
    inp = read_records(input_path)
    n = int(inp["nx"][0][0])
    glacier = inp["ivgtyp"][0] == 15
    rows, excluded, diagnostics = [], 0, []
    for snapshot in snapshots(output_path):
        diagnostics.append({"step": snapshot["step"],
                            "tauss_min": float(snapshot["taussxy"].min()),
                            "tauss_max": float(snapshot["taussxy"].max()),
                            "shortwave_absorbed_max": float(np.abs(snapshot["fsaxy"]).max()),
                            "albedo_min": float(snapshot["albedo"].min()),
                            "albedo_max": float(snapshot["albedo"].max())})
        for field, array in snapshot.items():
            if field == "step":
                continue
            values = array.reshape(-1, n)
            if field in GLACIER_UNDEFINED:
                excluded += values[:, glacier].size
                values = values[:, ~glacier]
            rows.append({"step": snapshot["step"], "field": field,
                         "words": int(values.size),
                         "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    return {"input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            "columns": n, "rows": rows, "excluded_undefined_words": excluded,
            "words": sum(row["words"] for row in rows), "diagnostics": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("receipt")
    parser.add_argument("--expected")
    args = parser.parse_args()
    result = receipt(args.input, args.output)
    if args.expected:
        expected = json.loads(Path(args.expected).read_text())
        if result["input_sha256"] != expected["input_sha256"]:
            raise SystemExit("input hashes differ; the source comparison is invalid")
        index = {(r["step"], r["field"]): r for r in expected["rows"]}
        result["mismatches"] = [row for row in result["rows"]
                                if row != index.get((row["step"], row["field"]))]
        result["expected_hashes"] = len(index)
        result["passed"] = not result["mismatches"] and len(result["rows"]) == len(index)
    Path(args.receipt).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "mismatches")}, indent=2))
    if args.expected:
        print("mismatching field hashes:", len(result["mismatches"]))
        raise SystemExit(not result["passed"])
