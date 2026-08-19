"""Receipt-only plots for spectral verification evidence.

The plotter consumes no model output and performs no scoring.  It renders the
numbers already bound into a validated receipt, then writes a manifest with a
SHA-256 for every PNG so a report can cite the exact images it displayed.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Mapping

from gpuwm.verify import spectral_io, spectral_receipt


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return text or "unnamed"


def _rows(comparison: Mapping[str, object], component: str
          ) -> list[dict[str, object]]:
    result = comparison["result"]
    if result["kind"] == "scalar":
        if component != "scalar":
            return []
        return [dict(item) for item in result["bands"]]
    return [
        {"name": band["name"], **band["components"][component]}
        for band in result["bands"]
        if component in band.get("components", {})
    ]


def _finite_series(rows: list[dict[str, object]], key: str
                  ) -> tuple[list[str], list[float | None]]:
    labels = [str(row["name"]) for row in rows]
    values: list[float | None] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            values.append(None)
            continue
        number = float(value)
        values.append(number if math.isfinite(number) else None)
    return labels, values


def _plot_one(*, labels: list[str], series: list[tuple[str, list[float | None]]],
              title: str, ylabel: str, output: Path,
              reference_line: float | None = None, log_y: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    x = list(range(len(labels)))
    for label, values in series:
        axis.plot(x, [float("nan") if item is None else item for item in values],
                  marker="o", label=label)
    if reference_line is not None:
        axis.axhline(reference_line, linestyle="--", linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_xlabel("Pre-registered wavelength band")
    axis.set_ylabel(ylabel)
    if log_y and any(
            item is not None and item > 0.0
            for _label, values in series for item in values):
        axis.set_yscale("log")
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_receipt(value: Mapping[str, object], output_directory: str | Path
                 ) -> dict[str, object]:
    receipt = spectral_receipt.validate_receipt(value)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []

    for comparison in receipt["comparisons"]:
        components = (["scalar"] if comparison["kind"] == "scalar"
                      else ["total", "rotational", "divergent"])
        for component in components:
            rows = _rows(comparison, component)
            if not rows:
                continue
            labels, left_power = _finite_series(rows, "left_power")
            _, reference_power = _finite_series(rows, "reference_power")
            _, power_ratio = _finite_series(rows, "power_ratio")
            _, correlation = _finite_series(rows, "spectral_correlation")
            _, normalized_error = _finite_series(rows, "normalized_error_power")
            prefix = "__".join((
                _slug(comparison["pair"]), _slug(comparison["field"]),
                _slug(component)))
            title_prefix = (f"{comparison['pair']} / {comparison['field']} / "
                            f"{component}")
            jobs = [
                (f"{prefix}__power.png", [(receipt["left_label"], left_power),
                                           (receipt["reference_label"],
                                            reference_power)],
                 f"{title_prefix}: retained power", "Power / mean-square contribution",
                 None, True),
                (f"{prefix}__power-ratio.png", [("power ratio", power_ratio)],
                 f"{title_prefix}: power ratio", "Candidate / reference power",
                 1.0, True),
                (f"{prefix}__spectral-correlation.png",
                 [("signed spectral correlation", correlation)],
                 f"{title_prefix}: phase/location agreement",
                 "Signed spectral correlation", 1.0, False),
                (f"{prefix}__normalized-error.png",
                 [("normalized error power", normalized_error)],
                 f"{title_prefix}: normalized error power",
                 "Error power / reference power", 0.0, False),
            ]
            for filename, series, title, ylabel, reference_line, log_y in jobs:
                path = root / filename
                _plot_one(
                    labels=labels, series=series, title=title, ylabel=ylabel,
                    output=path, reference_line=reference_line, log_y=log_y)
                outputs.append({
                    "path": str(path.resolve()),
                    "sha256": spectral_io.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "pair": comparison["pair"],
                    "field": comparison["field"],
                    "component": component,
                    "metric": filename.rsplit("__", 1)[-1].removesuffix(".png"),
                })

    manifest: dict[str, object] = {
        "schema": "gpuwm.spectral-plot-manifest/v1",
        "receipt_sha256": receipt["receipt_sha256"],
        "files": outputs,
    }
    manifest["manifest_sha256"] = spectral_receipt.canonical_hash(manifest)
    spectral_receipt.write_json_atomic(root / "manifest.json", manifest)
    return manifest


def plot_file(receipt_path: str | Path, output_directory: str | Path
             ) -> dict[str, object]:
    return plot_receipt(spectral_receipt.read_json(receipt_path), output_directory)


__all__ = ["plot_file", "plot_receipt"]
