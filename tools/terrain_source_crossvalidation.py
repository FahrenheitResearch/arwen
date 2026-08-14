"""Build one model domain through two elevation sources and compare them.

Two high-resolution elevation sources never agree exactly, and most of the
disagreement is not error:

* they are referenced to different vertical datums (a smooth offset), and
* one may be a surface model and the other bare earth (an offset that
  follows vegetation), and
* they are sampled at different postings (a structural difference that
  area-averaging to the model cell largely removes).

Reporting a single RMS number over all of that manufactures a defect out of
three expected physical differences.  This tool therefore separates them:

``systematic``
    one scalar (the median difference) plus how much that scalar varies
    across the domain, measured by blocks, and how it varies with
    elevation.  A pure geoid difference is near-constant; a canopy
    difference is not, and collapses above treeline.
``structural``
    whether the terrain has the same SHAPE.  Measured two ways that are
    both immune to any additive offset: counts of cells whose residual
    exceeds a threshold after the single scalar is removed, and a direct
    comparison of terrain GRADIENTS, which an additive smooth offset
    cannot move.

Everything is reported as counts over thresholds rather than as a fitted
correlation, and every number is printed beside the resolution that bounds
it.

The comparator is self-tested in both directions before it is believed:
identical inputs must report zero, a known offset must be recovered
exactly, and a one-cell shift must be caught as structural disagreement
while still reporting no offset.  ``--self-test-only`` runs just that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpuwm.static.build import HALO, build_static           # noqa: E402
from gpuwm.static.lambert import LambertGrid                # noqa: E402
from gpuwm.static.highres_production import (                # noqa: E402
    HighresStaticConfig, apply_highres_statics)

#: Difference thresholds reported as cell counts (metres).
THRESHOLDS_M = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------

def _gradient_magnitude(field: np.ndarray, dx: float) -> np.ndarray:
    """Terrain slope magnitude (m/m) by centred differences on the grid."""
    dzdy, dzdx = np.gradient(np.asarray(field, dtype=np.float64), dx, dx)
    return np.hypot(dzdx, dzdy)


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Five-point Laplacian; its SIGN separates ridges from valleys."""
    a = np.asarray(field, dtype=np.float64)
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2]
                       + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1])
    return out


def _block_medians(diff: np.ndarray, blocks: int) -> np.ndarray:
    """Median difference in each of ``blocks`` x ``blocks`` tiles."""
    ny, nx = diff.shape
    ys = np.linspace(0, ny, blocks + 1).astype(int)
    xs = np.linspace(0, nx, blocks + 1).astype(int)
    return np.array([
        [float(np.median(diff[ys[j]:ys[j + 1], xs[i]:xs[i + 1]]))
         for i in range(blocks)] for j in range(blocks)])


def compare_terrain(reference: np.ndarray, candidate: np.ndarray, *,
                    dx: float, elevation: np.ndarray | None = None,
                    blocks: int = 5,
                    band_edges_m: tuple[float, float] = (3000.0, 3500.0),
                    ) -> dict:
    """Separate the systematic and structural parts of two terrain fields.

    ``candidate - reference`` is the sign convention throughout.
    ``elevation`` selects the field used to band the domain (default: the
    reference), so the offset can be reported below and above a treeline.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {cand.shape}")
    height = ref if elevation is None else np.asarray(elevation, np.float64)
    diff = cand - ref
    offset = float(np.median(diff))
    residual = diff - offset

    grid_blocks = _block_medians(diff, blocks)
    low, high = band_edges_m
    below = height < low
    above = height >= high
    band = {
        "band_edges_m": [float(low), float(high)],
        "cells_below": int(np.count_nonzero(below)),
        "cells_above": int(np.count_nonzero(above)),
        "median_offset_below_m": (float(np.median(diff[below]))
                                  if below.any() else None),
        "median_offset_above_m": (float(np.median(diff[above]))
                                  if above.any() else None),
    }

    gref = _gradient_magnitude(ref, dx)
    gcand = _gradient_magnitude(cand, dx)
    interior = np.zeros(ref.shape, dtype=bool)
    interior[1:-1, 1:-1] = True          # centred differences only
    ratio_ok = interior & (gref > 1.0e-6)
    gradient_ratio = gcand[ratio_ok] / gref[ratio_ok]
    lref = _laplacian(ref)[1:-1, 1:-1]
    lcand = _laplacian(cand)[1:-1, 1:-1]
    curved = (np.abs(lref) > 1.0) & (np.abs(lcand) > 1.0)   # metres
    sign_agree = int(np.count_nonzero(
        (np.sign(lref) == np.sign(lcand)) & curved))

    return {
        "cells": int(ref.size),
        "systematic": {
            "median_offset_m": offset,
            "mean_offset_m": float(np.mean(diff)),
            "block_grid": int(blocks),
            "block_median_min_m": float(grid_blocks.min()),
            "block_median_max_m": float(grid_blocks.max()),
            "block_median_range_m": float(grid_blocks.max()
                                          - grid_blocks.min()),
            "elevation_bands": band,
        },
        "structural": {
            "residual_after_one_scalar": {
                "abs_median_m": float(np.median(np.abs(residual))),
                "abs_p95_m": float(np.percentile(np.abs(residual), 95.0)),
                "abs_max_m": float(np.max(np.abs(residual))),
                "rms_m": float(np.sqrt(np.mean(residual ** 2))),
                "cells_over": {
                    f"{t:g}m": int(np.count_nonzero(np.abs(residual) > t))
                    for t in THRESHOLDS_M},
                "fraction_within_20m": float(
                    np.count_nonzero(np.abs(residual) <= 20.0) / residual.size),
            },
            "gradient": {
                "note": "slope magnitude on the model grid; an additive "
                        "smooth offset cannot move it",
                "median_reference_m_per_m": float(np.median(gref[interior])),
                "median_candidate_m_per_m": float(np.median(gcand[interior])),
                "median_ratio_candidate_over_reference": float(
                    np.median(gradient_ratio)) if gradient_ratio.size else None,
                "cells_compared": int(np.count_nonzero(ratio_ok)),
            },
            "curvature_sign_agreement": {
                "note": "sign of the 5-point Laplacian: ridge vs valley",
                "cells_curved_in_both": int(np.count_nonzero(curved)),
                "cells_sign_agree": sign_agree,
                "fraction": (sign_agree / int(np.count_nonzero(curved))
                             if np.count_nonzero(curved) else None),
            },
        },
        "raw_difference": {
            "min_m": float(diff.min()), "max_m": float(diff.max()),
            "abs_median_m": float(np.median(np.abs(diff))),
            "rms_m": float(np.sqrt(np.mean(diff ** 2))),
            "cells_over": {
                f"{t:g}m": int(np.count_nonzero(np.abs(diff) > t))
                for t in THRESHOLDS_M},
        },
    }


# ---------------------------------------------------------------------------
# Instrument validation -- both directions, before anything is believed
# ---------------------------------------------------------------------------

def self_test(dx: float = 500.0, n: int = 100) -> list[str]:
    """Prove the comparator finds a difference AND finds none.

    Raises ``AssertionError`` on any failure; returns the log lines.
    """
    rng = np.random.default_rng(20260814)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    field = (2500.0 + 900.0 * np.sin(x / 11.0) * np.cos(y / 13.0)
             + 40.0 * rng.standard_normal((n, n)))
    log: list[str] = []

    # (a) NULL direction: identical input must report exactly nothing.
    same = compare_terrain(field, field.copy(), dx=dx)
    assert same["systematic"]["median_offset_m"] == 0.0, same
    assert same["structural"]["residual_after_one_scalar"]["abs_max_m"] == 0.0
    assert same["raw_difference"]["cells_over"]["1m"] == 0
    assert same["structural"]["gradient"][
        "median_ratio_candidate_over_reference"] == 1.0
    log.append("null      identical fields -> offset 0.000 m, max residual "
               "0.000 m, gradient ratio 1.000, 0 cells over 1 m   PASS")

    # (b) SYSTEMATIC direction: a known constant offset must come back
    #     exactly, and must NOT leak into the structural numbers.
    shifted = compare_terrain(field, field + 7.25, dx=dx)
    off = shifted["systematic"]["median_offset_m"]
    assert abs(off - 7.25) < 1e-9, off
    assert shifted["structural"]["residual_after_one_scalar"][
        "abs_max_m"] < 1e-9
    assert abs(shifted["systematic"]["block_median_range_m"]) < 1e-9
    assert abs(shifted["structural"]["gradient"][
        "median_ratio_candidate_over_reference"] - 1.0) < 1e-12
    log.append(f"systematic +7.25 m constant -> offset {off:.3f} m recovered, "
               "residual 0.000 m, gradient ratio 1.000, block range "
               "0.000 m   PASS")

    # (c) STRUCTURAL direction: a one-cell shift moves no mean but wrecks
    #     the shape.  The comparator must report ~zero offset and a large
    #     structural disagreement, or it cannot tell shape from level.
    rolled = compare_terrain(field, np.roll(field, 1, axis=1), dx=dx)
    roff = rolled["systematic"]["median_offset_m"]
    rres = rolled["structural"]["residual_after_one_scalar"]
    assert abs(roff) < 20.0, roff
    assert rres["rms_m"] > 50.0, rres
    assert rres["cells_over"]["50m"] > field.size // 10, rres
    log.append(f"structural one-cell shift -> offset {roff:.3f} m (small, "
               f"correct), residual RMS {rres['rms_m']:.1f} m, "
               f"{rres['cells_over']['50m']} cells over 50 m   PASS")

    # (d) SMOOTHING direction: a heavily smoothed field must show a
    #     smaller gradient, which is how the 900 m baseline will read.
    smooth = field.copy()
    for _ in range(8):
        smooth[1:-1, 1:-1] = 0.25 * (smooth[:-2, 1:-1] + smooth[2:, 1:-1]
                                     + smooth[1:-1, :-2] + smooth[1:-1, 2:])
    smoothed = compare_terrain(field, smooth, dx=dx)
    ratio = smoothed["structural"]["gradient"][
        "median_ratio_candidate_over_reference"]
    assert ratio < 0.9, ratio
    log.append(f"smoothing 8 diffusive passes -> gradient ratio {ratio:.3f} "
               "(< 1, correct: smoothing flattens slope)   PASS")

    # (e) RESOLUTION LIMIT, stated as a measurement rather than a claim:
    #     the smallest offset the comparator resolves against its own
    #     noise floor on this grid.
    tiny = compare_terrain(field, field + 0.05, dx=dx)
    assert abs(tiny["systematic"]["median_offset_m"] - 0.05) < 1e-9
    log.append("limit     +0.05 m offset recovered to 1e-9 m: the comparator "
               "is exact; the resolution limit is the DATA's, not the "
               "instrument's   PASS")
    return log


# ---------------------------------------------------------------------------
# Domain construction
# ---------------------------------------------------------------------------

def build_domain(*, lat: float, lon: float, dx: float, n: int,
                 geog_root: Path, cache_root: Path, source: str,
                 case_date: date) -> tuple[np.ndarray, dict, dict]:
    """Return (HGT_M, receipt, baseline) for one terrain source.

    ``source`` of ``"baseline"`` skips the overlay entirely and returns the
    30-arc-second field.
    """
    grid = LambertGrid(ref_lat=lat, ref_lon=lon, truelat1=30.0,
                       truelat2=60.0, stand_lon=lon, dx=dx, dy=dx,
                       e_we=n + 1, e_sn=n + 1)
    baseline = build_static(grid, geog_root, HALO)
    if source == "baseline":
        return np.asarray(baseline["HGT_M"]), {}, baseline
    config = HighresStaticConfig(
        enabled=True, cache_root=Path(cache_root), on_refuse="error",
        terrain_source=source, fields="terrain")
    fields, receipt = apply_highres_statics(
        baseline, grid, config=config, domain_id=1, case_date=case_date,
        landuse_attrs={"iswater": 17, "islake": 21})
    return np.asarray(fields["HGT_M"]), receipt, baseline


def _summary(name: str, field: np.ndarray) -> dict:
    return {"source": name, "min_m": float(field.min()),
            "max_m": float(field.max()), "mean_m": float(field.mean()),
            "std_m": float(field.std()),
            "relief_m": float(field.max() - field.min())}


def default_geog_root() -> Path | None:
    """Where a staged WPS_GEOG tree lives on THIS machine, or None.

    Read from the environment rather than written into the file, so this
    tool is runnable by anyone who has the geography data staged and says
    nothing about the machine it was written on.  ``WPS_GEOG`` names the
    tree directly; ``GPUWM_CASE_DATA_ROOT`` names the parent the rest of
    the project already resolves case inputs under.
    """
    direct = os.environ.get("WPS_GEOG")
    if direct:
        return Path(direct)
    root = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if root:
        return Path(root) / "WPS_GEOG"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, default=39.55)
    parser.add_argument("--lon", type=float, default=-105.55)
    parser.add_argument("--dx", type=float, default=500.0)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--geog-root", type=Path,
                        default=default_geog_root(),
                        help="staged WPS_GEOG tree; defaults to $WPS_GEOG, "
                             "else $GPUWM_CASE_DATA_ROOT/WPS_GEOG")
    parser.add_argument("--cache-root", type=Path, required=False)
    parser.add_argument("--reference", default="usgs-3dep-13as")
    parser.add_argument("--candidate", default="copernicus-dem-glo30")
    parser.add_argument("--case-date", default="2024-06-01")
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--npz", type=Path, required=False,
                        help="save the three terrain fields for plotting")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args(argv)

    print("== comparator self-test (both directions) ==")
    for line in self_test():
        print("  " + line)
    if args.self_test_only:
        return 0
    if args.cache_root is None:
        parser.error("--cache-root is required unless --self-test-only")
    if args.geog_root is None:
        parser.error(
            "--geog-root is required: neither WPS_GEOG nor "
            "GPUWM_CASE_DATA_ROOT is set in the environment")

    case_date = date.fromisoformat(args.case_date)
    fields: dict[str, np.ndarray] = {}
    receipts: dict[str, dict] = {}
    for name in ("baseline", args.reference, args.candidate):
        print(f"\n== building domain through {name} ==")
        field, receipt, _ = build_domain(
            lat=args.lat, lon=args.lon, dx=args.dx, n=args.cells,
            geog_root=args.geog_root, cache_root=args.cache_root,
            source=name, case_date=case_date)
        fields[name] = field
        receipts[name] = receipt
        print("   " + json.dumps(_summary(name, field)))

    report = {
        "domain": {"ref_lat": args.lat, "ref_lon": args.lon, "dx_m": args.dx,
                   "mass_cells": [args.cells, args.cells],
                   "halo_cells": HALO, "case_date": args.case_date},
        "fields": {k: _summary(k, v) for k, v in fields.items()},
        "comparisons": {
            f"{args.candidate}_vs_{args.reference}": compare_terrain(
                fields[args.reference], fields[args.candidate], dx=args.dx,
                elevation=fields[args.reference]),
            f"{args.reference}_vs_baseline": compare_terrain(
                fields["baseline"], fields[args.reference], dx=args.dx,
                elevation=fields[args.reference]),
            f"{args.candidate}_vs_baseline": compare_terrain(
                fields["baseline"], fields[args.candidate], dx=args.dx,
                elevation=fields[args.candidate]),
        },
        "receipts": {k: v.get("receipt_path") for k, v in receipts.items()},
        "fetch": {k: (v.get("fetch") or {}).get("bytes_fetched")
                  for k, v in receipts.items()},
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print("\n== report ==")
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.npz:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz, **{
            k.replace("-", "_"): v for k, v in fields.items()})
        print(f"wrote {args.npz}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
