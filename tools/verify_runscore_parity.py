"""Score one run pair with both instruments and compare metric for metric.

The Rust paired-run scorer (`tools/rustwx/crates/rw-fieldcmp`, binary
`rw_runscore`) exists to replace the Python one on a growing campaign, which
only holds if the two produce the same numbers on real output.  This runs
both on the same directories and reports every metric that differs, with the
gap measured in representable doubles rather than in a relative epsilon that
hides a wrong answer at small magnitudes.

It also times both, twice each after an untimed warming pass, so the speed
claim and the parity claim come from the same run on the same bytes.

    python tools/verify_runscore_parity.py LEFT RIGHT \\
        --registration LEFT/n5s-preregistration.json \\
        --binary .../rw_runscore.exe \\
        --domain d01=12000 --domain d02=3000 \\
        --neighborhood-domain d04 --json evidence/.../parity.json

Exits 0 when every metric matches to the last bit, 1 when the two disagree
about which metrics exist, and 2 when any value differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
from typing import Mapping, Sequence


def ulp_gap(left: float, right: float) -> int:
    """Representable doubles between two values, sign-magnitude ordered."""

    def ordered(value: float) -> int:
        bits = struct.unpack("<q", struct.pack("<d", value))[0]
        return bits if bits >= 0 else (1 << 63) - bits

    return abs(ordered(left) - ordered(right))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="the rw_runscore executable")
    parser.add_argument("--domain", action="append", default=[], required=True,
                        metavar="LABEL=DX_M",
                        help="a domain and its grid spacing; repeatable")
    parser.add_argument("--neighborhood-domain", action="append", default=[],
                        metavar="LABEL",
                        help="domains carrying the neighbourhood row")
    parser.add_argument("--key-neighborhood", default=None,
                        help="metric-key category for the neighbourhood row")
    parser.add_argument("--key-neighborhood-subject", default=None,
                        help="third field of the neighbourhood key")
    parser.add_argument("--runs", type=int, default=2,
                        help="timed runs per instrument after a warming pass")
    parser.add_argument("--note", default=None,
                        help="how this pair was staged, recorded in the receipt")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full comparison here")
    return parser


def _candidate_command(args: argparse.Namespace,
                       registration: Mapping[str, object]) -> list[str]:
    command = [
        str(args.binary), str(args.left), str(args.right),
        "--start", str(registration["start_time"]),
        "--run-seconds", str(int(registration["run_duration_seconds"])),
        "--cadence-seconds", str(int(registration["cadence_seconds"])),
        "--quiet",
    ]
    for domain in args.domain:
        command += ["--domain", domain]
    for domain in args.neighborhood_domain:
        command += ["--neighborhood-domain", domain]
    if args.key_neighborhood:
        command += ["--key-neighborhood", args.key_neighborhood]
    if args.key_neighborhood_subject:
        command += ["--key-neighborhood-subject", args.key_neighborhood_subject]
    return command


def _time(call, runs: int) -> tuple[object, list[float]]:
    """Warm once untimed, then time `runs` passes.  Returns the last result."""
    result = call()
    elapsed = []
    for _ in range(runs):
        began = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - began)
    return result, elapsed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from gpuwm.verify import n5s_metrics

    registration = n5s_metrics.load_registration(args.registration)
    left = args.left.resolve()
    right = args.right.resolve()

    def reference():
        # The frame re-hash is a separate leg of the campaign door and is not
        # what this compares, so it is declared done rather than timed here.
        return n5s_metrics.score_run_pair(
            left, right, registration, registration,
            _verified_run_directories={left, right})

    command = _candidate_command(args, registration)
    # The candidate's own record sits beside the comparison and is named for
    # it, so a directory of receipts stays legible when it holds several.
    output = (args.json.with_name(args.json.stem + "-candidate.json")
              if args.json else Path("rw_runscore-candidate.json"))
    output.parent.mkdir(parents=True, exist_ok=True)

    def candidate():
        subprocess.run(command + ["--json", str(output)], check=True)
        with output.open("r", encoding="utf-8") as stream:
            return json.load(stream)["score"]["distances"]

    reference_scores, reference_times = _time(reference, args.runs)
    candidate_scores, candidate_times = _time(candidate, args.runs)

    if set(reference_scores) != set(candidate_scores):
        print("METRIC INVENTORIES DIFFER")
        print(" reference only:",
              sorted(set(reference_scores) - set(candidate_scores)))
        print(" candidate only:",
              sorted(set(candidate_scores) - set(reference_scores)))
        return 1

    differing = []
    for metric in sorted(reference_scores):
        expected = float(reference_scores[metric])
        actual = float(candidate_scores[metric])
        if expected != actual:
            differing.append({
                "metric": metric, "reference": expected, "candidate": actual,
                "ulps": ulp_gap(expected, actual),
                "relative": abs(expected - actual) / abs(expected)
                if expected else float("inf"),
            })

    reference_median = statistics.median(reference_times)
    candidate_median = statistics.median(candidate_times)
    payload = {
        "left": str(left), "right": str(right),
        "note": args.note,
        "command": command,
        "candidate_record": str(output),
        "metrics": len(reference_scores),
        "bit_identical": len(reference_scores) - len(differing),
        "differing": differing,
        "reference_seconds": reference_times,
        "candidate_seconds": candidate_times,
        "reference_median_seconds": reference_median,
        "candidate_median_seconds": candidate_median,
        "speedup": reference_median / candidate_median,
        "distances": reference_scores,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    print(f"{payload['metrics']} metrics: {payload['bit_identical']} "
          f"bit-identical, {len(differing)} not")
    for row in sorted(differing, key=lambda item: -item["relative"]):
        print(f"  {row['metric']:<58} reference={row['reference']!r} "
              f"candidate={row['candidate']!r} ulps={row['ulps']}")
    print(f"reference {reference_median:.2f} s, candidate "
          f"{candidate_median:.2f} s, {payload['speedup']:.2f}x")
    return 2 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
