#!/usr/bin/env python3
"""Derive candidate gate fragments from known-good spectral receipts.

This tool never scores a candidate forecast.  Feed it only receipts from the
pre-declared equivalence population (for example, tiny-perturbation chaos
members or repeated known-good WRF/Arwen controls).  It records every source
receipt hash in the generated TOML.  Using the candidate being judged as a
calibration member is circular and invalidates the gate.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpuwm.verify import spectral_receipt  # noqa: E402

DEFAULT_METRICS = (
    "power_ratio",
    "spectral_correlation",
    "coherence_squared",
    "weighted_mean_absolute_phase_error_degrees",
    "normalized_error_power",
    "divergent_energy_fraction_difference",
)


def nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("calibration sample is empty or non-finite")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must lie in (0, 100]")
    return ordered[math.ceil(percentile / 100.0 * len(ordered)) - 1]


def _quoted(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _gate(group: tuple[str, str, str], metric: str, values: list[float],
          percentile: float) -> dict[str, object]:
    field, component, band = group
    base: dict[str, object] = {
        "id": f"calibrated:{field}:{component}:{band}:{metric}",
        "pair": "*", "field": field, "component": component,
        "band": band, "metric": metric, "transform": "identity",
        "samples": len(values),
    }
    if metric in ("power_ratio", "amplitude_ratio"):
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{metric} calibration requires positive samples")
        envelope = nearest_rank(
            (abs(math.log10(value)) for value in values), percentile)
        base["minimum"] = 10.0 ** (-envelope)
        base["maximum"] = 10.0 ** envelope
        base["calibration_statistic"] = (
            f"symmetric log10 nearest-rank E{percentile:g}")
    elif metric in ("spectral_correlation", "coherence_squared"):
        lower_percentile = max(100.0 - percentile, 1e-12)
        base["minimum"] = nearest_rank(values, lower_percentile)
        base["calibration_statistic"] = (
            f"nearest-rank lower {lower_percentile:g}th percentile")
    elif metric == "divergent_energy_fraction_difference":
        base["transform"] = "absolute"
        base["maximum"] = nearest_rank((abs(value) for value in values), percentile)
        base["calibration_statistic"] = (
            f"nearest-rank E{percentile:g} of absolute difference")
    else:
        base["maximum"] = nearest_rank(values, percentile)
        base["calibration_statistic"] = f"nearest-rank E{percentile:g}"
    return base


def calibrate(receipts: Sequence[Mapping[str, object]], *,
              percentile: float = 95.0,
              metrics: Sequence[str] = DEFAULT_METRICS,
              minimum_samples: int = 3,
              group_by_pair: bool = False) -> tuple[list[dict[str, object]], list[str]]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    rows: dict[tuple[str, str, str, str, str], list[float]] = {}
    hashes: list[str] = []
    for receipt in receipts:
        valid = spectral_receipt.validate_receipt(receipt)
        if valid["verdict"] in ("fail", "incomplete"):
            raise ValueError(
                "a failed/incomplete receipt cannot define a known-good envelope")
        hashes.append(str(valid["receipt_sha256"]))
        for row in spectral_receipt.receipt_metric_rows(valid):
            for metric in metrics:
                value = row.get(metric)
                if row.get("status") != "ok" or value is None:
                    continue
                pair = str(row["pair"]) if group_by_pair else "*"
                key = (pair, str(row["field"]), str(row["component"]),
                       str(row["band"]), metric)
                rows.setdefault(key, []).append(float(value))
    gates: list[dict[str, object]] = []
    for (pair, field, component, band, metric), values in sorted(rows.items()):
        if len(values) < minimum_samples:
            continue
        gate = _gate((field, component, band), metric, values, percentile)
        gate["pair"] = pair
        gate["id"] = (
            f"calibrated:{pair}:{field}:{component}:{band}:{metric}")
        gates.append(gate)
    if not gates:
        raise ValueError(
            "calibration produced no gate; add known-good samples or lower "
            "--minimum-samples explicitly")
    return gates, sorted(set(hashes))


def render_toml(gates: Sequence[Mapping[str, object]], *,
                receipt_hashes: Sequence[str], percentile: float) -> str:
    lines = [
        "# gpuwm spectral gate calibration fragment",
        "# WARNING: valid only when every source receipt belongs to the",
        "# pre-declared known-good/equivalence population. Never include the",
        "# candidate forecast being judged.",
        f"# nearest-rank coverage percentile: {percentile:g}",
    ]
    for digest in receipt_hashes:
        lines.append(f"# source_receipt_sha256: {digest}")
    for gate in gates:
        lines.extend(("", "[[gates]]"))
        for key in ("id", "pair", "field", "component", "band", "metric",
                    "transform"):
            lines.append(f"{key} = {_quoted(gate[key])}")
        for key in ("minimum", "maximum"):
            if key in gate:
                lines.append(f"{key} = {float(gate[key]):.17g}")
        lines.append(
            "note = " + _quoted(
                f"{gate['calibration_statistic']}; n={gate['samples']}; "
                "source hashes are in this fragment header"))
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--metric", action="append", choices=DEFAULT_METRICS,
                        dest="metrics")
    parser.add_argument("--group-by-pair", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipts = [spectral_receipt.read_json(path) for path in args.receipt]
    gates, hashes = calibrate(
        receipts, percentile=args.percentile,
        metrics=tuple(args.metrics or DEFAULT_METRICS),
        minimum_samples=args.minimum_samples,
        group_by_pair=args.group_by_pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_toml(gates, receipt_hashes=hashes,
                    percentile=args.percentile),
        encoding="utf-8", newline="\n")
    print(f"gates {len(gates)}")
    print(f"output {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
