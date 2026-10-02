"""Measure production bookkeeping launch outputs against frozen WRF words."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.smallstep_bookkeeping_oracle import (
    BOOKKEEPING_DIR, PORT_RUNNERS, ROUNDING_TRACES, load_bookkeeping)
from gpuwm.verify.smallstep_oracle import word_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=BOOKKEEPING_DIR)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    manifest = json.loads((a.fixtures / "manifest.json").read_text())
    measurements = {"schema": "wrf471-smallstep-bookkeeping-measurement-v1",
                    "files": {}}
    for filename, metadata in manifest["files"].items():
        fixture = load_bookkeeping(a.fixtures / filename)
        actual = PORT_RUNNERS[metadata["routine"]](fixture)
        row = {"routine": metadata["routine"], "native": {}, "isolation": {}}
        for key, value in fixture.items():
            if key.startswith("ref_") and key[4:] in actual:
                row["native"][key[4:]] = word_metrics(actual[key[4:]], value)
            if key.startswith("isolation_") and key[10:] in actual:
                row["isolation"][key[10:]] = word_metrics(actual[key[10:]], value)
        row["causal_trace"] = {name: word_metrics(actual[name], value)
                               for name, value in ROUNDING_TRACES[metadata["routine"]](fixture).items()}
        measurements["files"][filename] = row
    a.output.write_text(json.dumps(measurements, indent=2) + "\n")
    aggregates = {}
    for row in measurements["files"].values():
        for output, metric in row["native"].items():
            target = aggregates.setdefault(row["routine"] + "." + output,
                                           {"cases": 0, "different_words": 0, "max_ulp": 0, "max_abs": 0.})
            target["cases"] += 1
            target["different_words"] += metric["different_words"]
            target["max_ulp"] = max(target["max_ulp"], metric["max_ulp"])
            target["max_abs"] = max(target["max_abs"], metric["max_abs"])
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
