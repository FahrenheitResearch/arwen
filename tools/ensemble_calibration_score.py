"""Run the native matched-pair scorer and record its input provenance.

This tool does not generate, interpolate, or calibrate forecast fields. Prepare
matched TSV rows through the native observation reader and remapper first. Each
input table represents one case, recipe, quantity, unit, and lead window. Missing
values are explicit NaN; missing observations are not zero precipitation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(*, binary: Path, matched: Path, thresholds: str,
          provenance: Path) -> dict:
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    required = ("case_id", "recipe", "quantity", "units", "member_ids",
                "source_receipts", "observation_receipts", "match_receipt",
                "valid_start", "valid_end", "spinup_seconds")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"provenance is missing {missing}")
    for key in ("source_receipts", "observation_receipts"):
        if not metadata[key]:
            raise ValueError(f"{key} must identify the actual source artifacts")
    identities = metadata["member_ids"]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("member identities must be nonempty and unique")
    with matched.open(encoding="utf-8") as handle:
        labels = handle.readline().rstrip("\r\n").split("\t")[3:]
    if labels != identities:
        raise ValueError("TSV member order differs from the source identities")
    result = subprocess.run([str(binary.resolve()), str(matched.resolve()),
                             thresholds], check=True, capture_output=True,
                            text=True)
    scores = json.loads(result.stdout)
    if scores.get("schema") != "gpuwm-ensemble-calibration.scores.v1":
        raise ValueError("native scorer returned an unknown schema")
    if scores.get("members") != len(identities):
        raise ValueError("native scorer member count differs from provenance")
    return {"schema": "gpuwm-ensemble-calibration.receipt.v1",
            "status": "matched-pair-scores-only",
            "provenance": metadata,
            "input_sha256": sha256(matched),
            "provenance_sha256": sha256(provenance),
            "scorer_sha256": sha256(binary),
            "scorer_source_sha256": sha256(Path(__file__).with_suffix(".rs")),
            "scores": scores}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--matched", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--thresholds", default="-")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = score(binary=args.binary, matched=args.matched,
                    thresholds=args.thresholds, provenance=args.provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
