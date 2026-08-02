"""Fixture builders shared by the certification tests.

Not a test module (pytest does not collect it).  Everything a certify or
dual-run test needs to construct a *matched* set -- a capsule, a metrics CSV,
a WRF reference manifest -- lives here, so the two suites agree on what
"matched" means instead of each inventing it.

The metrics rows are built from each interval's own endpoints.  Nothing here
transcribes a published or measured number, so no fixture can pre-ordain a
verdict: an in-band row is the midpoint of whatever interval the band carries,
and an out-of-band row is one interval width past its upper endpoint.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from gpuwm.certify.band import (CLASSIFICATION_BANDED, ROW_KEY_COLUMNS,
                                load_band, write_certification_json)
from gpuwm.certify.capsule import (CAPSULE_SCHEMA_ID,
                                   GEOGRAPHY_CONTENT_ALGORITHM)
from gpuwm.certify.pins import PINS

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The one band this repository ships today.  Tests resolve it by scanning the
#: band directory rather than naming a configuration, so nothing here has to
#: be edited when a second configuration is banded.
BAND_DIR = REPO_ROOT / "gpuwm" / "data" / "certification" / "bands"

WRF_REFERENCE_DIR = REPO_ROOT / "docs" / "public" / "wrf-reference"


def shipped_band_paths() -> list[Path]:
    """Every committed band, by path, sorted."""
    return sorted(path for path in BAND_DIR.glob("*.json")
                  if not path.name.endswith(".coverage.json"))


def shipped_band() -> dict[str, Any]:
    """The committed band this repository ships."""
    paths = shipped_band_paths()
    assert paths, f"no acceptance band is committed under {BAND_DIR}"
    return load_band(paths[0])


def config_path_for(config_sha256: str) -> Path:
    """The committed config whose bytes hash to ``config_sha256``.

    Discovery by identity: the band records a digest, never a path, so the
    test finds the file the same way an independent verifier would.
    """
    import hashlib

    for candidate in sorted((REPO_ROOT / "configs").glob("*.toml")):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == config_sha256:
            return candidate
    raise AssertionError(
        f"no config under configs/ hashes to {config_sha256}")


def metric_columns() -> tuple[str, ...]:
    """The comparator's metric columns, from the comparator itself."""
    import ast

    source = (REPO_ROOT / "tools"
              / "matched_wrfout_stream_compare.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_METRIC_COLUMNS"):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(
        "tools/matched_wrfout_stream_compare.py declares no _METRIC_COLUMNS")


# --------------------------------------------------------------------------
# Capsule
# --------------------------------------------------------------------------

def matched_capsule(config_sha256: str, *,
                    emission_site: str = "runtime.run_experiment:domain-tree"
                    ) -> dict[str, Any]:
    """A capsule certify has no reason to refuse."""
    stack = {}
    for pin in PINS:
        stack[pin.key] = {"value": f"fixture:{pin.key}", "source": pin.source,
                          "status": "resolved"}
    stack["config_bytes"]["value"] = {"path": "configs/fixture.toml",
                                      "sha256": config_sha256}
    stack["cupy_accelerators"]["value"] = {"present": False, "value": None}
    return {
        "schema": CAPSULE_SCHEMA_ID,
        "emission_site": emission_site,
        "code": {
            "gpuwm_version": "0+fixture",
            "git_commit": "0" * 40,
            "tree_sha256": "1" * 64,
            "worktree_clean": True,
            "runtime_source_sha256": "2" * 64,
        },
        "registry": {
            "registry_sha256": "3" * 64,
            "template_id": "fixture-template",
            "emitted_physics_tuple": {"ra_rrtmg_variant": "rrtmg_legacy"},
        },
        "input_bytes": {
            "entries": {
                "geography_root": {
                    "algorithm": GEOGRAPHY_CONTENT_ALGORITHM,
                    "sha256": "4" * 64,
                },
                "forcing_catalog": {"sha256": "5" * 64},
            },
        },
        "numerical_stack": stack,
        "kernel_manifest": {
            "gpuwm.core.kernels:diff6": {
                "source_sha256": "6" * 64,
                "options": ["-std=c++17"],
                "compiled_image": {"status": "unavailable",
                                   "reason": "fixture"},
            },
        },
        "arithmetic_contract": {"ftz_receipt_sha256": "7" * 64},
        "run_shape": {
            "route": emission_site,
            "domain_count": 2,
            "run_seconds": 3600.0,
        },
        "output": {
            "frames": [
                {"path": "out/fixture/wrfout_d01_0000", "bytes": 1024,
                 "sha256": "8" * 64},
                {"path": "out/fixture/wrfout_d02_0000", "bytes": 2048,
                 "sha256": "9" * 64},
            ],
            "trajectory_digest": {"d01": "a" * 64, "d02": "b" * 64},
        },
        "receipts": {"run_receipt": {"path": "out/fixture/run-receipt.json"}},
    }


# --------------------------------------------------------------------------
# Metrics CSV, built from the band's own endpoints
# --------------------------------------------------------------------------

def _in_band_value(interval: Mapping[str, Any]) -> str:
    if interval["nan_expected"]:
        return "nan"
    return repr((float(interval["lower"]) + float(interval["upper"])) / 2.0)


def _out_of_band_value(interval: Mapping[str, Any]) -> str:
    if interval["nan_expected"]:
        # A number where the band expects a non-number is outside it.
        return "0.0"
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    width = upper - lower
    return repr(upper + (width if width > 0.0 else 1.0))


def metrics_rows(band: Mapping[str, Any], *, out_of_band: bool = False,
                 limit_rows: int | None = None,
                 extra_columns: Mapping[str, str] | None = None
                 ) -> tuple[list[str], list[dict[str, str]]]:
    """Comparator-shaped rows derived from each interval's own endpoints."""
    coverage = band["metric_coverage"]
    columns = list(ROW_KEY_COLUMNS) + list(metric_columns())
    if extra_columns:
        columns += list(extra_columns)
    rows: list[dict[str, str]] = []
    for domain in sorted(band["intervals"]):
        for lead in sorted(band["intervals"][domain],
                           key=lambda text: float(text)):
            cells = band["intervals"][domain][lead]
            row = {"domain": domain,
                   "valid_time": f"1970-01-01_{int(float(lead)):02d}_00_00",
                   "forecast_hour": lead}
            for column in metric_columns():
                if coverage[column]["classification"] != CLASSIFICATION_BANDED:
                    row[column] = "0.5"
                    continue
                interval = cells[column]
                row[column] = (_out_of_band_value(interval) if out_of_band
                               else _in_band_value(interval))
            if extra_columns:
                row.update(extra_columns)
            rows.append(row)
            if limit_rows is not None and len(rows) >= limit_rows:
                return columns, rows
    return columns, rows


def write_metrics_csv(path: Path, columns: list[str],
                      rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# --------------------------------------------------------------------------
# WRF reference manifest
# --------------------------------------------------------------------------

def complete_wrf_reference(config_sha256: str) -> dict[str, Any]:
    """A manifest carrying all four hash groups.

    Synthetic digests: this fixture exists to exercise the refusal logic in
    both directions, and inventing a plausible-looking digest for the real
    reference binary would be inventing a measurement.  The committed manifest
    under docs/public/wrf-reference/ carries what was actually measured on the
    node that holds the reference bytes; this one carries obvious fakes so a
    reader can never mistake the two.
    """
    return {
        "schema": "gpuwm.wrf-reference-manifest/v1",
        "wrf_version": "0.0.0-fixture",
        "wrf_commit": None,
        "config_sha256": config_sha256,
        "wrf_exe_sha256": "c" * 64,
        "build_recipe_sha256": "d" * 64,
        "namelist_sha256": {"namelist.wps": "e" * 64,
                            "namelist.input": "f" * 64},
        "reference_wrfout_sha256": {"wrfout_d01_0000": "0" * 64},
    }


def committed_wrf_reference_paths() -> list[Path]:
    return sorted(WRF_REFERENCE_DIR.glob("*.manifest.json"))


# --------------------------------------------------------------------------
# One matched set on disk
# --------------------------------------------------------------------------

def matched_set(tmp_path: Path, *, out_of_band: bool = False,
                extra_columns: Mapping[str, str] | None = None,
                capsule_edit=None, band_edit=None, reference_edit=None
                ) -> dict[str, Path]:
    """Write a capsule, metrics CSV, band and manifest that certify accepts.

    Each ``*_edit`` callable receives the document before it is written, so a
    refusal test breaks exactly one thing and leaves the rest matched.
    """
    band = shipped_band()
    config_sha256 = band["config_sha256"]
    capsule = matched_capsule(config_sha256)
    reference = complete_wrf_reference(config_sha256)
    if capsule_edit is not None:
        capsule_edit(capsule)
    if band_edit is not None:
        band_edit(band)
    if reference_edit is not None:
        reference_edit(reference)

    capsule_path = write_certification_json(
        tmp_path / "certification-capsule.json", capsule)
    band_path = write_certification_json(tmp_path / "band.json", band)
    reference_path = write_certification_json(
        tmp_path / "wrf-reference.manifest.json", reference)
    columns, rows = metrics_rows(band, out_of_band=out_of_band,
                                 extra_columns=extra_columns)
    metrics_path = write_metrics_csv(tmp_path / "metrics.csv", columns, rows)
    return {"capsule": capsule_path, "band": band_path,
            "wrf_reference": reference_path, "metrics": metrics_path}


__all__ = [
    "BAND_DIR",
    "REPO_ROOT",
    "WRF_REFERENCE_DIR",
    "committed_wrf_reference_paths",
    "complete_wrf_reference",
    "config_path_for",
    "matched_capsule",
    "matched_set",
    "metric_columns",
    "metrics_rows",
    "shipped_band",
    "shipped_band_paths",
    "write_metrics_csv",
]
